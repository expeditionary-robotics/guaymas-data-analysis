"""Takes in data from CTD casts and creates background objects and plots.

Saves practical salinity, potential temperature computed with formula 90,
and depth to files.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from transect_utils import get_transect_input_paths, RIDGE
from guaymas_data_analysis.utils.data_utils import distance

# Import data targets for the transect
INPUT_NOPP, INPUT_SAGE_LEG1, INPUT_SAGE_LEG2, INPUT_ROSETTE, INPUT_GGA, \
    INPUT_NH4, INPUT_BOTTLES, INPUT_SENTRY = get_transect_input_paths()
OUTPUT_ROSETTE_SAGE = os.path.join(
    os.getenv("SENTRY_OUTPUT"), f"transect/ctd_casts.csv")

# Whether to create file
CREATE_FILE = True

# Whether to read from file
READ_FILE = False

# Whether to visualize
VISUALIZE = True
VIZ_TARG = ["beam_attenuation", "o2_umol_kg",
            "sage_methane_ppm", "pot_temp_C_its90", "prac_salinity"]
VIZ_LABEL = ["Beam Attentuation", "O2 (umol/kg)",
             "SAGE Methane (ppm)", "Potential Temperature",
             "Practical Salinity"]
Y_TARGET = "depth_m"  # "digiquartz_pressure_db"
WINDOW = 120
TOP_DEPTH = 0
BOTTOM_DEPTH = 1750


if __name__ == "__main__":
    if CREATE_FILE is True:
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
        # Leg 2
        ctd2_df = ctd_df[ctd_df["cast_name"] == "CTD-11"]

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
        sage1_df.loc[:, 'timestamp'] = [
            x + t1_delt for x in sage1_df["timestamp"]]
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
        sage2_df.loc[:, 'timestamp'] = [
            x + t2_delt for x in sage2_df["timestamp"]]
        sage2_df.loc[:, "t"] = (sage2_df["timestamp"] -
                                pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
        sage2_df['t'] = sage2_df['t'].astype(np.int32)
        sage2_df = sage2_df.drop_duplicates(subset="t")

        """Merge CTD and HCF legs"""
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

        """Truncate to only the up and down casts"""
        ctd1_df = ctd1_df[(ctd1_df["t"] < 1638227762) |
                          (ctd1_df["t"] > 1638253808)]
        ctd2_df = ctd2_df[(ctd2_df["t"] < 1638263036) |
                          (ctd2_df["t"] > 1638270664)]

        """Label cast legs and up-down status"""
        ctd1_df.loc[:, "Leg"] = [1 for i in range(len(ctd1_df))]
        ctd2_df.loc[:, "Leg"] = [2 for i in range(len(ctd2_df))]
        ctd1_df.loc[:, "Cast"] = ctd1_df.apply(
            lambda x: -1 if x['t'] < 1638227762 else 1, axis=1)
        ctd2_df.loc[:, "Cast"] = ctd2_df.apply(
            lambda x: -1 if x['t'] < 1638263036 else 1, axis=1)

        """Combine everything back into one dataframe"""
        ctd_df = ctd1_df.append(ctd2_df)
        ctd_df.sort_index(ascending=False)
        # Add distance metric
        ctd_df["ridge_distance"] = ctd_df.apply(lambda x: distance(
            float(RIDGE[0]), float(RIDGE[1]), float(x["usbl_lat"]), float(x["usbl_lon"]))*1000., axis=1)

        """Save the dataframe"""
        print(f"Saving Rosette dataframe to {OUTPUT_ROSETTE_SAGE}")
        ctd_df.to_csv(OUTPUT_ROSETTE_SAGE)

    if READ_FILE is True:
        ctd_df = pd.read_csv(OUTPUT_ROSETTE_SAGE)

    """Simple vertical profile visualization"""
    if VISUALIZE is True:
        down_leg1 = ctd_df[(ctd_df["Cast"] == -1) &
                           (ctd_df["Leg"] == 1)].sort_values(by=Y_TARGET)
        up_leg1 = ctd_df[(ctd_df["Cast"] == 1) & (
            ctd_df["Leg"] == 1)].sort_values(by=Y_TARGET)
        down_leg2 = ctd_df[(ctd_df["Cast"] == -1) &
                           (ctd_df["Leg"] == 2)].sort_values(by=Y_TARGET)
        up_leg2 = ctd_df[(ctd_df["Cast"] == 1) & (
            ctd_df["Leg"] == 2)].sort_values(by=Y_TARGET)

        for viz_targ, viz_label in zip(VIZ_TARG, VIZ_LABEL):
            fig, ax = plt.subplots(1, 1, figsize=(5, 15))
            ax.scatter(down_leg1[viz_targ], down_leg1[Y_TARGET],
                       s=1, alpha=0.2)
            ax.plot(down_leg1[viz_targ].rolling(
                WINDOW, center=True).mean(), down_leg1[Y_TARGET], label="Downcast, Leg1")
            ax.scatter(up_leg1[viz_targ], up_leg1[Y_TARGET],
                       s=1, alpha=0.2)
            ax.plot(up_leg1[viz_targ].rolling(
                WINDOW, center=True).mean(), up_leg1[Y_TARGET], label="Upcast, Leg1")
            ax.scatter(down_leg2[viz_targ], down_leg2[Y_TARGET],
                       s=1, alpha=0.2)
            ax.plot(down_leg2[viz_targ].rolling(
                WINDOW, center=True).mean(), down_leg2[Y_TARGET], label="Downcast, Leg2")
            if "sage" not in viz_targ:
                ax.scatter(up_leg2[viz_targ], up_leg2[Y_TARGET],
                           s=1, alpha=0.2)
                ax.plot(up_leg2[viz_targ].rolling(
                    WINDOW, center=True).mean(), up_leg2["depth_m"], label="Upcast, Leg2")
            ax.set_ylim([TOP_DEPTH, BOTTOM_DEPTH])
            ax.invert_yaxis()
            ax.set_ylabel(Y_TARGET)
            ax.set_xlabel(viz_label)
            plt.legend()
            plt.savefig(os.path.join(os.getenv("SENTRY_OUTPUT"),
                        f"transect/figures/cast_{Y_TARGET}_{viz_targ}.png"))
            plt.close()
