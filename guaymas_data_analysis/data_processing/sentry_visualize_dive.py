"""Visualize processed sentry data files.

Takes in a given sentry dive name (optional) and dive phase (optional and only
supported for dives sentry610 and sentry611) and produces visualizations of
key scientific sensor data. The specific data and visuals produces are
configurable via variables set at the start of the script.

The input sentry dive should be one of: sentry608 - sentry612

Example usage:
    # Visualize data for dive sentry608 (default), all phases
    python sentry_visualize_dive.py

    # Visualize data for dive sentry611, all phases
    python sentry_visualize_dive.py sentry611

    # Visualize data for dive sentry611, phase 3
    python sentry_visualize_dive.py sentry611 3
"""
import os
import sys

import pandas as pd
from functools import partial
from datetime import timezone

import pdb

# Imports for current data processing
from guaymas_data_analysis.utils.viz_utils import plot_and_save_science_data, \
    plot_and_save_advection_video
from guaymas_data_analysis.utils.current_utils import CurrMag, CurrHead, \
    TimeSeriesInterpolate
from guaymas_data_analysis.utils.metadata_utils import \
    sentry_output_by_dive, sentry_site_by_dive, extent_by_site, \
    sentry_anom_by_dive, sentry_phase_by_dive_and_time, \
    sentry_detections_by_dive, tiltmeter_output_by_deployment, \
    tiltmeter_trainset_by_deployment

############
# CHANGE THESE PARAMETERS
############
# If you want to visualize plots
VISUALIZE = True

# 2D visualization
VIZ_2D = False
SAVE_2D = True

# 3D visualization
VIZ_3D = True
SAVE_3D = True

# Bathy visualization
VIZ_BATHY = False
SAVE_BATHY = False  # include a separate plot of bathymetry

# Current-based advection visualization
VIZ_CURRENTS = True
SAVE_VIDEO = True
CURRENT_METHOD = "interp"  # one of "interp", "gp" for interpolation- or gp-based
ANIM_VAR = "dorpdt_anom"  # variable to animate
ANIM_LAB = "dORP/dt"  # label


# ANALYSIS_VARIABLES = ["dorpdt", "obs", "O2", "ctd_temp", "ctd_sal"]
ANALYSIS_VARIABLES = ["dorpdt_anom", "obs_anom", "O2_anom",
                      "potential_temp_anom", "practical_salinity_anom"]
VAR_LABELS = ["dORPdt", "OBS", "O2", "Temperature", "Salinity"]
# ANALYSIS_VARIABLES = ["dorpdt_anom"]
# ANALYSIS_VARIABLES = ["detections"]
# ANALYSIS_VARIABLES = ["northing", "easting"]
PLOT_LOG = [True, False, False, False, False]
############
# End adjustable parameters
############

# Dive name, set by command line input
DIVE_NAME = "sentry608"
# Dive name
if len(sys.argv) > 1:
    DIVE_NAME = sys.argv[1]
print(f"Generating visualizations for dive {DIVE_NAME}")

# Phase name, set by command line input
PHASE_NAME = None  # by default, include all phases
# Dive name
if len(sys.argv) > 2:
    PHASE_NAME = int(sys.argv[2])
print(f"Generating visualizations for phase {PHASE_NAME}")

# Get the site location
SITE = sentry_site_by_dive(DIVE_NAME)

# Extent information by site
EXTENT = extent_by_site(SITE)

# Science Sensors
INPUT_CSV = sentry_detections_by_dive(DIVE_NAME)
if not os.path.exists(INPUT_CSV):  # try to get processed detection dataframe
    INPUT_CSV = sentry_anom_by_dive(DIVE_NAME)
    if not os.path.exists(INPUT_CSV):  # try to get processed anomaly dataframe
        INPUT_CSV = sentry_output_by_dive(DIVE_NAME)
        if not os.path.exists(INPUT_CSV):  # try to get processed dataframe
            raise ValueError("No processed data file to view.")

# Get the long data deployment
INPUT_CURRENT = tiltmeter_output_by_deployment(deployment="B1")
TRAIN_CURRENT = tiltmeter_trainset_by_deployment(deployment="B1")

# Read in science data
df = pd.read_table(INPUT_CSV, delimiter=",").set_index(
    "timestamp")
df.index = pd.to_datetime(df.index, utc=True)
df.sort_index(inplace=True)
print(df.head())

if VIZ_CURRENTS:
    if CURRENT_METHOD == "gp":
        # Read in training data and train a GP model for current and heading
        df_train = pd.read_table(TRAIN_CURRENT, delimiter=",")
        currmag = CurrMag(df_train["hours"], df_train["mag_mps"],
                          learning_rate=0.5, training_iter=100)
        currhead = CurrHead(df_train["hours"], df_train["head_rad"],
                            learning_rate=0.1, training_iter=200)
    elif CURRENT_METHOD == "interp":
        df_train = pd.read_table(INPUT_CURRENT, delimiter=",")
        df_train["timestamp"] = pd.to_datetime(df_train["timestamp"])
        currmag = TimeSeriesInterpolate(
            df_train["timestamp"], df_train["mag_mps"])
        currhead = TimeSeriesInterpolate(
            df_train["timestamp"], df_train["head_rad"])
    else:
        raise ValueError(
            f"Unrecognized current modeling methods {CURRENT_METHOD}.")

    # Add seconds since midnight UTC from timestamp for current lookup
    df["seconds"] = df.index.hour * 3600 + \
        df.index.minute * 60 + df.index.second

if PHASE_NAME is not None:
    # Get the phase for each timestep
    fun = partial(sentry_phase_by_dive_and_time, dive_name=DIVE_NAME)
    df["phase"] = pd.to_datetime(df.index).map(fun)

    # Subset the data to the relevent phase
    df = df[df["phase"] == PHASE_NAME]

if VISUALIZE:
    if VIZ_CURRENTS:
        SKIP = 100  # time stepping discretization
        TRUNCATE = -1  # truncate simulation at this index

        # Plot advection videos
        plot_and_save_advection_video(
            df[:TRUNCATE], ANIM_VAR, ANIM_LAB, currmag, currhead,
            SKIP, EXTENT, SITE, f"{DIVE_NAME}_{VAR}", SAVE_VIDEO)

    # Plot 3D and 2D plots
    plot_and_save_science_data(
        df, ANALYSIS_VARIABLES, VAR_LABELS, PLOT_LOG, EXTENT, SITE,
        DIVE_NAME, VIZ_2D, VIZ_3D, VIZ_BATHY, SAVE_2D, SAVE_3D,
        SAVE_BATHY)
