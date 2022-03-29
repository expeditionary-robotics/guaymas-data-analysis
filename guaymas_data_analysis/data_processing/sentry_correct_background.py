"""Correct for depth-varying background signal using data from either a
Sentry ascent or descent, or from a CTD cast."""
import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pdb

import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Local utils file
from sentry_data_analysis.utils.metadata_utils import sentry_output_by_dive, \
    sentry_site_by_dive, extent_by_site, ctd_output, \
    nopp_filename_by_dive, hcf_filename_by_dive
from sentry_data_analysis.utils.viz_utils import plot_and_save_science_data

"""User defined arguments. These are intended to be set from the command line."""
# Sentry dive name
DIVE_NAME = "sentry608"
# Type of background correction to apply.
# Ascent uses the sentry613 dive as the background profile
# CTD uses a specified CTD cast as the background profiles
# Trend fits a linear trend to the sentry dive data
# None performs no correction.
BKGND_CORRECT = "trend"  # "ascent", "ctd", "trend", "none"
# If using ctd to correct dive, cast name to use.
CAST_NAME = None

# Dive name
if len(sys.argv) > 1:
    DIVE_NAME = sys.argv[1]
print(f"Visualizing dive {DIVE_NAME}.")

if len(sys.argv) > 2:
    BKGND_CORRECT = sys.argv[2]
print(f"Performing background correction? {BKGND_CORRECT}")

# Cast name
if len(sys.argv) > 3 and BKGND_CORRECT == "ctd":
    CAST_NAME = sys.argv[3]

    print(f"Correcting with CTD cast {CAST_NAME}")


# ANALYSIS_VARIABLES = ["dorpdt", "obs", "O2", "ctd_temp", "ctd_sal"]
# VAR_LABELS = ["dORPdt", "OBS", "O2", "Temperature", "Salinity"]
ANALYSIS_VARIABLES = ['ctd_temp', 'ctd_sal', 'O2', 'practical_salinity',
                      'absolute_salinity', 'potential_temp', 'potential_density', "dorpdt", "obs"]

PLOT_LOG = [False, False, False, False, False, False, False, True, False]
VAR_LABELS = ANALYSIS_VARIABLES.copy()

INPUT_NOPP = nopp_filename_by_dive(DIVE_NAME)
INPUT_HCF = hcf_filename_by_dive(DIVE_NAME)
if INPUT_NOPP is not None:
    ANALYSIS_VARIABLES.append("fundamental")
    VAR_LABELS.append("fundamental")
    PLOT_LOG.append(False)
if INPUT_HCF is not None:
    ANALYSIS_VARIABLES.append("methane")
    VAR_LABELS.append("methane")
    PLOT_LOG.append(False)

# Output Name
OUTPUT_NAME = f"{DIVE_NAME}_processed_anom.csv"

# If you want to visualize plots
VISUALIZE = True
VIZ_2D = False
VIZ_3D = True
VIZ_BATHY = False
SAVE_BATHY = False  # include a separate plot of bathymetry
SAVE_2D = True
SAVE_3D = True

###############################################################################
# END INPUT
###############################################################################

# Get the site location
SITE = sentry_site_by_dive(DIVE_NAME)

# Extent information by site
EXTENT = extent_by_site(SITE)

# Science Sensors
INPUT_DIVE = sentry_output_by_dive(DIVE_NAME)

# Read in science data
df = pd.read_table(INPUT_DIVE, delimiter=",")
df.fillna(method="ffill", inplace=True)
print(df.columns)

if CAST_NAME is not None:
    BKGND = ctd_output()
    # Read in the CTD background data
    df_bkgd = pd.read_table(BKGND, delimiter=",")
    df_bkgd = df_bkgd.rename({"depth_m": "depth"})
    df_bkgd = df_bkgd[df_bkgd['cast_name'] ==
                      f"CTD-{CAST_NAME}"]
elif BKGND_CORRECT == "ascent":
    # Correct with final Sentry dive
    BKGND = sentry_output_by_dive("sentry613_ascent")
    # Read in sentry background data
    df_bkgd = pd.read_table(BKGND, delimiter=",")

if BKGND_CORRECT == "ascent" or BKGND_CORRECT == "ctd":
    # Merge background data from CTD or Sentry ascent and the sentry dive
    df_merge = pd.merge_asof(
        df.sort_values(by="depth"),
        df_bkgd.sort_values(by="depth"),
        on="depth",
        suffixes=(None, "_bkgnd"),
        direction="nearest")
    df_merge = df_merge.set_index("timestamp").sort_index()
    for var in ANALYSIS_VARIABLES:
        df_merge[f"{var}_anom"] = df_merge[var] - df_merge[f"{var}_bkgnd"]
elif BKGND_CORRECT == "trend":
    # Fit a trend between depth and variables based on current dive data
    df_merge = df.copy()
    SKIP = 1  # fit trend based on a subset of the data, spread by SKIP
    poly_fit = []
    for var in ANALYSIS_VARIABLES:
        z = np.polyfit(df['depth'][::SKIP], df[var][::SKIP], 1)
        p = np.poly1d(z)
        poly_fit.append(p)
        df_merge[f"{var}_bkgnd"] = p(df['depth'])
        df_merge[f"{var}_anom"] = df_merge[var] - df_merge[f"{var}_bkgnd"]

    if VISUALIZE:
        # Visualize the determined trend lines
        fig = make_subplots(rows=len(ANALYSIS_VARIABLES), cols=1,
                            subplot_titles=tuple(ANALYSIS_VARIABLES))

        SKIP = 50
        for i, var in enumerate(ANALYSIS_VARIABLES):
            fig.add_trace(go.Scatter(
                x=df['depth'][::SKIP], y=df[var][::SKIP], mode='markers'), row=i+1, col=1)
            fig.add_trace(go.Scatter(
                x=df['depth'][::SKIP], y=poly_fit[i](df['depth'][::SKIP])), row=i+1, col=1)
        fig.show()
        fname = f"background_trend_{DIVE_NAME}.html"
        fname = os.path.join(
            os.getenv("SENTRY_OUTPUT"), f"sentry/{fname}")
        fig.write_html(fname)
        print("Figure", fname, "saved.")
else:
    # Do nothing
    df_merge = df.copy()
    for var in ANALYSIS_VARIABLES:
        df_merge[f"{var}_anom"] = df_merge[var]


if BKGND_CORRECT != "none":
    ANALYSIS_VARIABLES = [f"{v}_anom" for v in ANALYSIS_VARIABLES]
    VAR_LABELS = [f"{n} Anom." for n in VAR_LABELS]
    DIVE_NAME = f"{DIVE_NAME}_anom_{BKGND_CORRECT}"

if VISUALIZE:
    plot_and_save_science_data(
        df_merge, ANALYSIS_VARIABLES, VAR_LABELS, PLOT_LOG, EXTENT, SITE,
        DIVE_NAME, VIZ_2D, VIZ_3D, VIZ_BATHY, SAVE_2D, SAVE_3D,
        SAVE_BATHY)

print('Saving dataframe...')
output_file = os.path.join(
    os.getenv("SENTRY_DATA"), f"sentry/proc/RR2107_{OUTPUT_NAME}")
cols_to_save = df.columns.tolist() + ANALYSIS_VARIABLES
df_merge[cols_to_save].to_csv(output_file)
print('Dataframe saved')
