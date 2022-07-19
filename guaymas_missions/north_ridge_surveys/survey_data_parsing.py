"""Creates basic data products for each survey dive."""

import os
import pandas as pd
import numpy as np
import scipy as sp
import datetime
from scipy.io import loadmat
import datetime as datetime
from gasex import sol

# Imports from utility files
from guaymas_data_analysis.data_processing.nopp.laboratory_calibration import \
    fundamental_conversion, ringdown_conversion, TimeLagCorrection, FUND_TAU, RING_TAU
from guaymas_data_analysis.utils.data_utils import \
    calculate_northing_easting_from_lat_lon, \
    convert_oceanographic_measurements, dens
from guaymas_data_analysis.utils.metadata_utils import calibrate_nopp, \
    sentry_filename_by_dive, nopp_filename_by_dive, hcf_filename_by_dive, \
    sentry_site_by_dive, extent_by_site, sentry_phase_by_dive_and_time
from guaymas_data_analysis.utils import EAST_REFERENCE, NORTH_REFERENCE

############
# General Parameters
############
DIVES = ["sentry607", "sentry608", "sentry610", "sentry611"]
PYTHIA_TIME_CORRECTION_WINDOW = 5*60  # seconds

#############
# Perform the work
#############
if __name__ == '__main__':
    for DIVE_NAME in DIVES:
        # Get the site location and extent
        SITE = sentry_site_by_dive(DIVE_NAME)
        EXTENT = extent_by_site(SITE)

        # Science Sensors
        INPUT_SCC, INPUT_RENAV = sentry_filename_by_dive(DIVE_NAME)

        RENAV_KEYS = ["renav_orp", "renav_optode"]
        RENAV_COL_TARGETS = [{"dorpdt": "dorpdt"},
                             {"concentration": "O2_conc"}]
        SCC_DROP_KEYS = ["eh", "dehdt"]

        # Experimental Sensors
        INPUT_PYTHIA = nopp_filename_by_dive(DIVE_NAME)
        INPUT_SAGE = hcf_filename_by_dive(DIVE_NAME)

        # Output Name
        OUTPUT_NAME = f"{DIVE_NAME}_processed.csv"

        #############
        # Run through the files
        #############
        # create sentry science data object
        """For the _scc filetype, the hierarchy is sci/column_names.
        This filetype provides a 1Hz signal for the mission on all sensors
        (performed via interpolation of mission data) and uses the renav
        lat/lon points). Relevant fields in this file type are:
        * y, m, d, h, mm, s
        * lat, lon, heading, depth, height
        * ctd_temp, ctd_cond, ctd_pres, ctd_sal
        * O2
        * obs
        * orp
        """
        # Read in science data
        print('Creating dataframe for science data')
        scc_mat = loadmat(INPUT_SCC)
        scc_mdata = scc_mat["sci"]
        scc_mdtype = scc_mdata.dtype
        scc_ndata = {n: scc_mdata[n][0, 0].flatten() for n in scc_mdtype.names}
        scc_df = pd.DataFrame(scc_ndata)
        scc_df = scc_df.drop(labels=SCC_DROP_KEYS, axis=1)

        # Convert seconds to timestamps
        scc_df.loc[:, "timestamp"] = pd.to_datetime(scc_df["t"], unit="s")

        if INPUT_RENAV is not None:
            """Sometimes data could be missing from the main scc file.
            This allows us to inject missing data (most likely to be orp and o2)"""
            for idx, merge_file in enumerate(INPUT_RENAV):
                other_df = loadmat(merge_file)
                other_mdata = other_df[RENAV_KEYS[idx]]
                keys_of_interest = ["t"]
                for k in RENAV_COL_TARGETS[idx].keys():
                    keys_of_interest.append(k)
                other_ndata = {n: other_mdata[n][0, 0].flatten()
                               for n in keys_of_interest}

                other_df_subset = pd.DataFrame(other_ndata)
                other_df_subset = other_df_subset.rename(
                    columns=RENAV_COL_TARGETS[idx])
                other_df_subset["t"] = other_df_subset["t"].astype(np.int32)
                other_df_subset = other_df_subset.drop_duplicates(subset="t")

                scc_df = scc_df.merge(
                    other_df_subset[keys_of_interest], how="left", on="t")

                if idx == "renav_optode":
                    # Need to convert concentration to a useful measurement
                    scc_df.O2_conc = scc_df.O2_conc.interpolate()  # uM
                    scc_df.O2_conc = scc_df.O2_conc / \
                        (dens(scc_df["ctd_sal"], scc_df["ctd_temp"],
                         scc_df["ctd_pres"])/1000)  # convert to umol/kg

        # calculates absolute and practical salinity, potential temp, and potential density using gsw
        convert_oceanographic_measurements(scc_df)

        if INPUT_PYTHIA is not None:
            """Allow injection of Pythia data into the SCC frame"""
            pythia_df = pd.read_table(
                INPUT_PYTHIA, delimiter=",",
                names=['insttime', 'lasertemp', 'ringdown', 'fundamental', 'timeaux', 'celltemp', 'cellpress', 'housetemp', 'housepress', 'househum', 'states'])
            pythia_df = pythia_df.dropna()

            # Convert to Unix time seconds
            pythia_df.loc[:, 'timestamp'] = pythia_df.apply(lambda x: datetime.datetime.strptime(
                str(x['insttime']), "%Y%m%dT%H%M%S.%f", ), axis=1)
            pythia_df.loc[:, "t"] = (pythia_df["timestamp"] -
                                     pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
            pythia_df['t'] = pythia_df['t'].astype(np.int32)
            pythia_df = pythia_df.drop_duplicates(subset="t")

            # Convert fundamental with new calibration standards, and normalize
            temp_sig_smoothed_fund = pythia_df.fundamental.rolling(
                PYTHIA_TIME_CORRECTION_WINDOW).mean(centered=True)
            pythia_df["fundamental"] = TimeLagCorrection(fundamental_conversion(
                temp_sig_smoothed_fund), pythia_df["timestamp"], FUND_TAU, FUND_TAU/4)
            temp_sig_smoothed_ring = pythia_df.ringdown.rolling(
                PYTHIA_TIME_CORRECTION_WINDOW).mean(centered=True)
            pythia_df["ringdown"] = TimeLagCorrection(ringdown_conversion(
                temp_sig_smoothed_ring), pythia_df["timestamp"], RING_TAU, RING_TAU/4)

            # perform merge, only keeping signal
            scc_df = scc_df.merge(
                pythia_df[["t", "ringdown", "fundamental"]], how="left", on="t")

        if INPUT_SAGE is not None:
            """Allow injection of HCF data into the SCC frame"""
            sage_df = pd.read_table(
                INPUT_SAGE, delimiter=",", names=["insttime", "methane"])
            sage_df = sage_df.dropna()

            # Convert to Unix time seconds
            sage_df.loc[:, 'timestamp'] = pd.to_datetime(
                sage_df["insttime"], format="%Y%m%dT%H%M%S")
            sage_df.loc[:, "t"] = (sage_df["timestamp"] -
                                   pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
            sage_df['t'] = sage_df['t'].astype(np.int32)
            sage_df = sage_df.drop_duplicates(subset="t")

            if len(sage_df) < len(scc_df):
                # less values than needed! need to interpolate
                # instead of subsample as we are here
                assert(False)

            # Merge the data
            scc_df = scc_df.merge(
                sage_df[["t", "methane"]], how="left", on="t")

            # Compute solubility scaled SAGE data
            scc_df.loc[:, 'methane'] = sage_df.apply(lambda x: sol.sol_SP_pt(
                x["practical_salinity"], x["potential_temp"], gas='CH4', p_dry=x["methane"] * 1e-6, units='mM') * 1e6, axis=1)

        # Set index
        if "timestamp_x" in scc_df.columns:
            # Rename column names changed by join
            scc_df.rename({"timestamp_x": "timestamp"}, axis=1, inplace=True)
        scc_df.set_index("timestamp", inplace=True)

        # calculate northing and easting using utm
        calculate_northing_easting_from_lat_lon(
            scc_df, refeasting=EAST_REFERENCE, refnorthing=NORTH_REFERENCE)

        # compute phases, relevant for AI settings
        try:
            scc_df.loc[:, "phase"] = scc_df.apply(
                lambda x: sentry_phase_by_dive_and_time(x['timestamp'], DIVE_NAME), axis=1)
        except ValueError:
            scc_df.loc[:, "phase"] = np.zeros_like(scc_df.index)

        print('Saving dataframe...')
        output_file = os.path.join(
            os.getenv("SENTRY_DATA"), f"sentry/proc/RR2107_{OUTPUT_NAME}")
        scc_df.to_csv(output_file)
        print('Dataframe saved')
