"""Parse data from sentry dive and output to processed data files."""
import os
import argparse
import os
import sys
import utm
import pandas as pd
import numpy as np
import scipy as sp
import datetime
from scipy.io import loadmat
import datetime as datetime
import pdb


# Imports from utility files
from guaymas_data_analysis.utils.data_utils import \
    calculate_northing_easting_from_lat_lon, \
    convert_oceanographic_measurements, detect_ascent_descent
from guaymas_data_analysis.utils.viz_utils import get_bathy, \
    plot_bathy_underlay, plot_sites_overlay, plot_and_save_science_data
from guaymas_data_analysis.utils.metadata_utils import calibrate_nopp, \
    sentry_filename_by_dive, \
    nopp_filename_by_dive, hcf_filename_by_dive, sentry_site_by_dive, \
    extent_by_site, calibrate_nopp
from guaymas_data_analysis.utils import REFERENCE, EAST_REFERENCE, \
    NORTH_REFERENCE, ZONE_LETT, ZONE_NUM, Extent

############
# CHANGE THESE PARAMETERS
############
DIVE_NAME = "sentry608"

# Dive name
if len(sys.argv) > 1:
    DIVE_NAME = sys.argv[1]
print(f"Generating dataframes for dive {DIVE_NAME}")

# Get the site location
SITE = sentry_site_by_dive(DIVE_NAME)

# Extent information by site
EXTENT = extent_by_site(SITE)

# Science Sensors
INPUT_SCC, INPUT_RENAV = sentry_filename_by_dive(DIVE_NAME)

RENAV_KEYS = ["renav_orp"]
RENAV_COL_TARGETS = [{"dorpdt": "dorpdt"}]
SCC_DROP_KEYS = ["eh", "dehdt"]

# Experimental Sensors
INPUT_NOPP = nopp_filename_by_dive(DIVE_NAME)
INPUT_HCF = hcf_filename_by_dive(DIVE_NAME)
# INPUT_NOPP = None
# INPUT_HCF = None

# Output Name
OUTPUT_NAME = f"{DIVE_NAME}_processed.csv"

# If Timing
START_TIME = None  # datetime.datetime()
END_TIME = None  # datetime.datetime()

# If you want to visualize plots
VISUALIZE = True
VIZ_2D = True
VIZ_3D = True
VIZ_BATHY = True
SAVE_BATHY = False  # include a separate plot of bathymetry
SAVE_2D = True
SAVE_3D = True

# ANALYSIS_VARIABLES = ["orp"]
# VAR_LABELS = ["ORP"]
# PLOT_LOG = [True]
# VAR_LABELS = ["ORP", "dORPdt", "OBS", "O2", "Temperature", "Salinity"]
VAR_LABELS = ["dORPdt", "OBS", "O2", "Temperature", "Salinity"]
ANALYSIS_VARIABLES = ["dorpdt", "obs", "O2", "ctd_temp", "ctd_sal"]
PLOT_LOG = [True, False, False, False, False]
if INPUT_NOPP is not None:
    ANALYSIS_VARIABLES.append("fundamental")
    VAR_LABELS.append("Fundamental")
    PLOT_LOG.append(False)
if INPUT_HCF is not None:
    ANALYSIS_VARIABLES.append("methane")
    VAR_LABELS.append("Methane (ppm)")
    PLOT_LOG.append(False)

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
* mag_vz
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
        if len(other_df_subset) < len(scc_df):
            # less values than needed! need to interpolate
            # instead of subsample as we are here
            assert(False)

        other_df_subset = other_df_subset.rename(
            columns=RENAV_COL_TARGETS[idx])
        other_df_subset["t"] = other_df_subset["t"].astype(np.int32)
        other_df_subset = other_df_subset.drop_duplicates(subset="t")

        scc_df = scc_df.merge(
            other_df_subset[["t", "dorpdt"]], how="left", on="t")

    # scc_df = scc_df.dropna()

if INPUT_NOPP is not None:
    """Allow injection of NOPP data into the SCC frame"""
    # 20211115T181255.344,21.1,0.8118,0.1562,20211115T181255.353,28.96,2.00,31.19,14.77,31.75,10000
    # timeLaser,laserTemp,Ringdown,FUndamental,timeAux,cellTemp,cellPressure,housingTemp,housingPressure,housingRelativeHumidity,states
    nopp_df = pd.read_table(
        INPUT_NOPP, delimiter=",",
        names=['insttime', 'lasertemp', 'ringdown', 'fundamental', 'timeaux', 'celltemp', 'cellpress', 'housetemp', 'housepress', 'househum', 'states'])
    # for now only take three columns
    nopp_df = nopp_df.dropna()
    nopp_df = nopp_df.drop(labels=['lasertemp', 'timeaux', 'celltemp',
                                   'cellpress', 'housetemp', 'housepress', 'househum', 'states'], axis=1)
    nopp_df.loc[:, 'timestamp'] = nopp_df.apply(lambda x: datetime.datetime.strptime(
        str(x['insttime']), "%Y%m%dT%H%M%S.%f", ), axis=1)
    nopp_df["cal_fundamental"] = calibrate_nopp(
        nopp_df["fundamental"])
    nopp_df["nfundamental"] = 1.0 - nopp_df["fundamental"]

    # Convert to Unix time seconds
    nopp_df.loc[:, "t"] = (nopp_df["timestamp"] -
                           pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
    nopp_df['t'] = nopp_df['t'].astype(np.int32)
    nopp_df = nopp_df.drop_duplicates(subset="t")

    scc_df = scc_df.merge(
        nopp_df[["t", "ringdown", "fundamental"]], how="left", on="t")

if INPUT_HCF is not None:
    """Allow injection of HCF data into the SCC frame"""
    # 20210903T020231,8.209949
    # time,value
    hcf_df = pd.read_table(
        INPUT_HCF, delimiter=",", names=["insttime", "methane"])

    # for now only take three columns
    hcf_df = hcf_df.dropna()
    hcf_df.loc[:, 'timestamp'] = pd.to_datetime(
        hcf_df["insttime"], format="%Y%m%dT%H%M%S")
    # Time shift the sensor response by 5 mintues
    hcf_df.loc[:, 'timestamp'] = [
        x + datetime.timedelta(minutes=5) for x in hcf_df["timestamp"]]

    # Convert to Unix time seconds
    hcf_df.loc[:, "t"] = (hcf_df["timestamp"] -
                          pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
    hcf_df['t'] = hcf_df['t'].astype(np.int32)
    hcf_df = hcf_df.drop_duplicates(subset="t")

    if len(hcf_df) < len(scc_df):
        # less values than needed! need to interpolate
        # instead of subsample as we are here
        assert(False)
    scc_df = scc_df.merge(hcf_df[["t", "methane"]], how="left", on="t")

# Add log-versions of some sensors for later processing
# [TODO] removing these for the final data files
# scc_df.loc[:, "dorpdt_log"] = np.log(np.fabs(np.nanmin(scc_df['dorpdt'])) + scc_df['dorpdt'] + 1.0)
# scc_df.loc[:, "obs_log"] = np.log(scc_df['obs'])

# Compute gradients of some sensors
# delta = scc_df.index[1] - scc_df.index[0]
# scc_df.loc[:, "do2dt"] = np.gradient(scc_df["O2"], delta)
# scc_df.loc[:, "do2dt_log"] = np.log(np.fabs(np.nanmin(scc_df["do2dt"])) + scc_df["do2dt"] + 1.0)

# Detrend some sensors that have strong depth-change correlation
# spl = sp.interpolate.UnivariateSpline(scc_df["t"][::10], scc_df["O2"][::10], s=3000)
# scc_df.loc[:, "O2_detrend"] = spl(scc_df["t"]) - scc_df["O2"]

# Set index
if "timestamp_x" in scc_df.columns:
    # Rename column names changed by join
    scc_df.rename({"timestamp_x": "timestamp"}, axis=1, inplace=True)
scc_df.set_index("timestamp", inplace=True)

if VISUALIZE:
    plot_and_save_science_data(
        scc_df, ANALYSIS_VARIABLES, VAR_LABELS, PLOT_LOG, EXTENT, SITE,
        DIVE_NAME, VIZ_2D, VIZ_3D, VIZ_BATHY, SAVE_2D, SAVE_3D,
        SAVE_BATHY)

# calculate northing and easting using utm
calculate_northing_easting_from_lat_lon(
    scc_df, refeasting=EAST_REFERENCE, refnorthing=NORTH_REFERENCE)

# calculates absolute and practical salinity, potential temp, and potential density using gsw
convert_oceanographic_measurements(scc_df)

# get the acent and descent lengths, if present
end_descent_id, start_ascent_id = detect_ascent_descent(scc_df)

# If a path for current is provided, add it to the dataset
dfs = [scc_df]
keys = ['scc']

if len(dfs) > 1:
    df_all = pd.concat(dfs, axis=1, keys=keys)
    df_all = df_all.interpolate(method='linear')
    df_index = scc_df.index
    mission_df = df_all.loc[df_index]
else:
    mission_df = scc_df

# Truncate if desired.
if START_TIME is not None:
    if END_TIME is not None:
        mission_df.loc[START_TIME:END_TIME]
    else:
        mission_df.loc[START_TIME:len(mission_df)]
else:
    if END_TIME is not None:
        mission_df.loc[0:END_TIME]
    else:
        pass

print('Saving dataframe...')
output_file = os.path.join(
    os.getenv("SENTRY_DATA"), f"sentry/proc/RR2107_{OUTPUT_NAME}")
mission_df.to_csv(output_file)
print('Dataframe saved')

# Shunt profiles to file
descent = mission_df[0:end_descent_id]
ascent = mission_df[start_ascent_id:len(mission_df)]
print('Saving profiles...')
if len(descent) > 0:
    output_descent = os.path.join(
        os.getenv("SENTRY_DATA"), f"sentry/proc/descent_profiles/{OUTPUT_NAME}")
    descent.to_csv(output_descent)
if len(ascent) > 0:
    output_ascent = os.path.join(
        os.getenv("SENTRY_DATA"), f"sentry/proc/ascent_profiles/{OUTPUT_NAME}")
    ascent.to_csv(output_ascent)
