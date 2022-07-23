"""Takes in data from CTD casts and creates background objects and plots.

Saves practical salinity, potential temperature computed with formula 90,
and depth to files.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from guaymas_data_analysis.utils.viz_utils import get_bathy, mesh_obj, plot_and_save_ctd_data, extent_by_site
from guaymas_data_analysis.utils.metadata_utils import ctd_site_by_number, ctd_filename_by_cast
from guaymas_data_analysis.utils import REFERENCE

from ctd_utils import detect_bottle_fire

# Select casts to analyze, from options:
# 01, 02, 03, 04, 05, 06-HCF, 07, 08, 09, 10-HCF-Transect, 11-HCF-Transect
CTD_NUMBERS = ["01", "02", "03"]
CTD_DATA_NAMES = [ctd_filename_by_cast(n) for n in CTD_NUMBERS]

# Choose variables to visualize for CTD cast visualization
CREATE_CTD_VISUAL = True  # if True, produce CTD cast visualization
VARS_TO_VIS = ["beam_attenuation", "o2umol_kg",
               "potential_temperature_90", "practical_salinity"]
VAR_NAMES = ["Beam Attenuation", "O2 uMol/Kg", "Po. Temp", "Salinity"]

# Names for CTD output files
TEMP_FILENAME = "proc_temp_profile.csv"
TEMP_FILE = os.path.join(os.getenv("SENTRY_DATA"),
                         f"ctd/proc/{TEMP_FILENAME}")
SALT_FILENAME = "proc_salt_profile.csv"
SALT_FILE = os.path.join(os.getenv("SENTRY_DATA"),
                         f"ctd/proc/{SALT_FILENAME}")

# Get the site location; for now, just use the site for the first cast
SITE = ctd_site_by_number(CTD_NUMBERS[0])

# Extent information by site
EXTENT = extent_by_site(SITE)

#################
# Run the processes
#################
ctd_col_names = ["density00",
                 "depth",
                 "latitude",
                 "longitude",
                 "practical_salinity",
                 "beam_attenuation",
                 "beam_transmission",
                 "system_time",
                 "num_bott_fired",
                 "bottle_pos",
                 "o2umol_l",
                 "o2umol_kg",
                 "o2_saturation",
                 "altitude",
                 "conductivity",
                 "flourescence",
                 "potential_temperature_90",
                 "potential_temperature_68",
                 "pressure",
                 "temperature_90",
                 "temperature_68",
                 "flag"]
ctd_dfs = []
all_bots = []
all_locs = []
all_times = []
salinity = []
temperature = []
depth = []
for name in CTD_DATA_NAMES:
    temp = pd.read_table(name)
    end_index = temp[temp[temp.columns[0]] == "*END*"].index[0]
    df = pd.read_table(name, skiprows=end_index+2,
                       names=ctd_col_names, delim_whitespace=True)

    df = df[(df['depth'] > 100) & (df['depth'] < 2000)]

    ctd_dfs.append(df)
    botts, locs, times = detect_bottle_fire(df)
    all_bots.append(botts)
    all_locs.append(locs)
    all_times.append(times)
    for s in df["practical_salinity"]:
        salinity.append(s)
    for t in df["potential_temperature_90"]:
        temperature.append(t)
    for d in df["depth"]:
        depth.append(d)

depth, salinity, temperature = np.asarray(depth).flatten().T, np.asarray(
    salinity).flatten().T, np.asarray(temperature).flatten().T
df_join = pd.DataFrame(np.asarray([depth, salinity, temperature]).T, columns=[
    "depth", "salinity", "temperature"])
df_join = df_join.set_index("depth")

if CREATE_CTD_VISUAL:
    plot_and_save_ctd_data(ctd_dfs, CTD_NUMBERS, VARS_TO_VIS, VAR_NAMES,
                           EXTENT, SITE, all_bots, all_locs, all_times)

with open(SALT_FILE, "w+") as fh:
    df_join["salinity"].to_csv(fh)
with open(TEMP_FILE, "w+") as fh:
    df_join["temperature"].to_csv(fh)
