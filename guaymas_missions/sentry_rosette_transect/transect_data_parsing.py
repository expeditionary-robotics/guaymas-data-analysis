"""Creates advanced data products for transect analysis."""

import pandas as pd
import numpy as np
from gasex import sol
from scipy.interpolate import interp1d
from transect_utils import get_transect_input_paths, \
    get_transect_sentry_pythia_path, get_transect_bottles_path, \
    get_transect_rosette_sage_path, RIDGE
from guaymas_data_analysis.utils.data_utils import distance

# Import data targets for the transect
INPUT_PYTHIA, INPUT_SAGE_LEG1, INPUT_SAGE_LEG2, INPUT_ROSETTE, INPUT_GGA, \
    INPUT_NH4, INPUT_BOTTLES, INPUT_SENTRY = get_transect_input_paths()
OUTPUT_SENTRY_PYTHIA = get_transect_sentry_pythia_path()
OUTPUT_BOTTLES = get_transect_bottles_path()
OUTPUT_ROSETTE_SAGE = get_transect_rosette_sage_path()


if __name__ == "__main__":
    """Get the Rosette data for each transect leg.
    Strip the ascent and descent for the Rosette.
    Strip where the SAGE stops functioning."""
    ctd_df = pd.read_csv(INPUT_ROSETTE)
    ctd_df.loc[:, "t"] = ctd_df["sys_time"]
    ctd_df['t'] = ctd_df['t'].astype(np.int32)
    ctd_df["datetime"] = pd.to_datetime(
        ctd_df["datetime"], format="%Y-%m-%d %H:%M:%S")
    ctd_df.sort_index(ascending=False)

    # Leg 1
    ctd1_df = ctd_df[ctd_df["cast_name"] == "CTD-10"]
    ctd1_df = ctd1_df[(ctd1_df["t"] > 1638227762) &
                      (ctd1_df["t"] < 1638253808)]
    # Leg 2
    ctd2_df = ctd_df[ctd_df["cast_name"] == "CTD-11"]
    ctd2_df = ctd2_df[(ctd2_df["t"] > 1638263036) & (
        ctd2_df["t"] < 1638270664)]

    """Prepare the SAGE dataframes to apply the correct
        timestamps based on logbook."""
    # Leg 1
    sage1_df = pd.read_table(INPUT_SAGE_LEG1, delimiter=",", usecols=[
        0, 2], names=["insttime", "sage_methane"])
    sage1_df = sage1_df.dropna()
    sage1_df = sage1_df[sage1_df["sage_methane"] >= 0.0]
    sage1_df.loc[:, "sage_methane_ppm"] = sage1_df["sage_methane"]
    sage1_df.loc[:, 'timestamp'] = pd.to_datetime(
        sage1_df["insttime"], format="%Y%m%dT%H%M%S")
    # time on 11.29.2021 22:42:42
    # time off 11.30.2021 07:00:04
    t1_start = pd.Timestamp("2021-11-29 22:42:42")
    t1_delt = t1_start - sage1_df["timestamp"].values[0]
    sage1_df.loc[:, 'timestamp'] = [x + t1_delt for x in sage1_df["timestamp"]]
    sage1_df.loc[:, "t"] = (sage1_df["timestamp"] -
                            pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
    sage1_df['t'] = sage1_df['t'].astype(np.int32)
    sage1_df = sage1_df.drop_duplicates(subset="t")

    # Leg 2
    sage2_df = pd.read_table(INPUT_SAGE_LEG2, delimiter=",", usecols=[
        0, 2], names=["insttime", "sage_methane"])
    sage2_df = sage2_df.dropna()
    sage2_df = sage2_df[(sage2_df["sage_methane"] >= 0.0025)]
    sage2_df.loc[:, "sage_methane_ppm"] = sage2_df["sage_methane"]
    sage2_df.loc[:, 'timestamp'] = pd.to_datetime(
        sage2_df["insttime"], format="%Y%m%dT%H%M%S")
    # time on 11.30.2021 08:32:03
    # time off 11.30.2021 12:49:45
    t2_start = pd.Timestamp("2021-11-30 08:32:03")
    t2_delt = t2_start - sage2_df["timestamp"].values[0]
    sage2_df.loc[:, 'timestamp'] = [x + t2_delt for x in sage2_df["timestamp"]]
    sage2_df.loc[:, "t"] = (sage2_df["timestamp"] -
                            pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
    sage2_df['t'] = sage2_df['t'].astype(np.int32)
    sage2_df = sage2_df.drop_duplicates(subset="t")

    """Merge CTD and SAGE legs"""
    if len(sage1_df) < len(ctd1_df):
        interpy = interp1d(
            sage1_df['t'], sage1_df['sage_methane_ppm'], bounds_error=False)
        ctd1_df.loc[:, "sage_methane_ppm"] = interpy(ctd1_df["t"])
    else:
        ctd1_df = ctd1_df.merge(
            sage1_df[["t", "sage_methane_ppm"]], how="left", on="t")
    ctd1_df.set_index("datetime", inplace=True)

    if len(sage2_df) < len(ctd2_df):
        interpy = interp1d(
            sage2_df['t'], sage2_df['sage_methane_ppm'], bounds_error=False)
        ctd2_df.loc[:, "sage_methane_ppm"] = interpy(ctd2_df["t"])
    else:
        ctd2_df = ctd2_df.merge(
            sage2_df[["t", "sage_methane_ppm"]], how="left", on="t")
    ctd2_df.set_index("datetime", inplace=True)

    """Combine everything back into one dataframe"""
    ctd_df = ctd1_df.append(ctd2_df)
    ctd_df.sort_index(ascending=False)
    # Add distance metric
    ctd_df["ridge_distance"] = ctd_df.apply(lambda x: distance(
        float(RIDGE[0]), float(RIDGE[1]), float(x["usbl_lat"]), float(x["usbl_lon"]))*1000., axis=1)

    """Save the dataframe"""
    print(f"Saving Rosette dataframe to {OUTPUT_ROSETTE_SAGE}")
    ctd_df.to_csv(OUTPUT_ROSETTE_SAGE)

    """Create a GGA and NH4 frame of reference"""
    bott_df = pd.read_table(INPUT_BOTTLES, delimiter=",")
    bott_df["bottle_pos"] = bott_df["bottle_pos"].astype(np.int32)
    gga_df = pd.read_table(INPUT_GGA, delimiter=",")
    gga_df["CTD Bottle #"] = gga_df["CTD Bottle #"].astype(np.int32)
    
    # Perform correction based on ChemYak procedure 
    # ppm > uatm by applying extraction efficiency and internal instrument pressure
    # uatm > nM via solubility computation
    gga_df.loc[:, 'ch4_ppm_corr_05'] = gga_df["GGA Methane"] / 0.05
    gga_df.loc[:, 'ch4_ppm_corr_15'] = gga_df["GGA Methane"] / 0.15
    gga_df.loc[:, 'ch4_ppm_corr_12'] = gga_df["GGA Methane"] / 0.12
    gga_df.loc[:, 'ch4_ppm_corr_064'] = gga_df["GGA Methane"] / 0.064
    gga_df.loc[:, 'ch4_uatm_corr_12'] =  gga_df.apply(lambda x: (((x['GGA Methane'] - 1.86) / 0.12 + 1.86) * (495. / 1000.)), axis=1)
    gga_df.loc[:, 'ch4_uatm_corr_064'] =  gga_df.apply(lambda x: (((x['GGA Methane'] - 1.86) / 0.064 + 1.86) * (495. / 1000.)), axis=1)
    # find the right bottles
    bott_df = bott_df[bott_df["cast_name"] == "CTD-11"]
    # bott_df = bott_df.merge(gga_df, left_on="bottle_pos",
    #                         right_on="CTD Bottle #", how="left")
    bott_df = pd.merge(bott_df, gga_df, left_on="bottle_pos", right_on="CTD Bottle #", how="left")
    print(len(bott_df))
    # bott_df = bott_df.dropna(subset=["CTD Salinity"])
    nh4_df = pd.read_table(INPUT_NH4, delimiter=",")
    nh4_df = nh4_df[nh4_df["Cast"] == 11]
    nh4_df["Bottle #"] = nh4_df["Bottle #"].astype(np.int32)
    # bott_df = bott_df.merge(nh4_df, left_on="bottle_pos",
    #                         right_on="Bottle #", how="left")
    bott_df = pd.merge(bott_df, nh4_df, left_on="bottle_pos",
                            right_on="Bottle #", how="left")
    print(len(bott_df))
    # bott_df = bott_df.dropna(subset=["CTD Salinity"])
    bott_df["datetime"] = pd.to_datetime(bott_df["datetime"])
    bott_df["ridge_distance"] = bott_df.apply(lambda x: distance(
        float(RIDGE[0]), float(RIDGE[1]), float(x["lat"]), float(x["lon"]))*1000., axis=1)
    bott_df["ch4_nM_corr_12"] = bott_df.apply(lambda x: sol.sol_SP_pt(x["prac_salinity"], x["pot_temp_C_its90"], gas='CH4', p_dry=x["ch4_uatm_corr_12"]*0.000001, units='mM')/0.000001, axis=1)
    bott_df["ch4_nM_corr_064"] = bott_df.apply(lambda x: sol.sol_SP_pt(x["prac_salinity"], x["pot_temp_C_its90"], gas='CH4', p_dry=x["ch4_uatm_corr_064"]*0.000001, units='mM')/0.000001, axis=1)
    print(f"Saving GGA, NH4, and Bottle dataframe to {OUTPUT_BOTTLES}")
    bott_df.to_csv(OUTPUT_BOTTLES)

    """Add a distance measure to the Sentry dive"""
    scc_df = pd.read_csv(INPUT_SENTRY)
    scc_df["timestamp"] = pd.to_datetime(scc_df["timestamp"])
    scc_df.set_index("timestamp", inplace=True)
    scc_df.sort_index(ascending=False)
    scc_df = scc_df.dropna()
    scc_df["ridge_distance"] = scc_df.apply(lambda x: distance(
        float(RIDGE[0]), float(RIDGE[1]), float(x["lat"]), float(x["lon"]))*1000., axis=1)
    print(f"Saving Sentry and Pythia dataframe to {OUTPUT_SENTRY_PYTHIA}")
    scc_df.to_csv(OUTPUT_SENTRY_PYTHIA)
