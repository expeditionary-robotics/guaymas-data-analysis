"""Generate binary detections from input processed dataframe."""
import os
import sys

import pandas as pd
import numpy as np

import pdb

from sentry_data_analysis.utils.pseudosensor_utils import BinarySensor, SENTRY_DIVE_ALL_KEYS
from sentry_data_analysis.utils.viz_utils import get_bathy, mesh_obj, plot_and_save_detections
from sentry_data_analysis.utils import Extent
from sentry_data_analysis.utils.metadata_utils import sentry_anom_by_dive, \
    sentry_detections_by_dive

# If you want to visualize plots
VISUALIZE = True  # global visualization toggle
VIZ_2D = False  # visualize data as 2D plots
VIZ_3D = True  # visualize data as 3D plots
VIZ_BATHY = True  # visualize the bathymetry data separately
SAVE_BATHY = False  # save the bathymetery plot
SAVE_2D = True  # save the 2D plots
SAVE_3D = True  # save teh 3D plots

# Name of the sentry dive to plot; should be set by command line argument
DIVE_NAME = "sentry608"

# Dive name
if len(sys.argv) > 1:
    DIVE_NAME = sys.argv[1]

# Output Name
OUTPUT_NAME = f"{DIVE_NAME}_processed_detections.csv"

print(f"Generating binary plume observations for dive {DIVE_NAME}")

target_file = sentry_anom_by_dive(DIVE_NAME)
df = pd.read_table(target_file, delimiter=',')
print(df.columns)

SENSOR_KEYS = ["dorpdt_anom",  # from SCC
               "potential_temp_anom",  # from SCC
               "practical_salinity_anom",  # from SCC
               "obs_anom",   # from SCC
               "O2_anom"]  # from SCC
if "fundamental" in df.columns:
    SENSOR_KEYS.append("fundamental_anom")
if "methane" in df.columns:
    SENSOR_KEYS.append("methane_anom")
SENSOR_WEIGHTS = [4,  # dorpdt
                  3,  # temp
                  1,  # salinity
                  3,  # obs
                  1,  # o2
                  1]  # fundamental or methane


# Get detections, and write observations to disk
print("Creating detections dataset...")
bs = BinarySensor(sensor_keys=SENSOR_KEYS,
                  num_sensors_consensus=5,
                  sensor_weights=SENSOR_WEIGHTS,
                  window_width=200,
                  filter_ascent_descent=False,
                  mode="meanstd",
                  num_stddev=2,
                  percentile_lower=1,
                  percentile_upper=99)

bs.calibrate_from_df_csv(df)
detection_out = sentry_detections_by_dive(DIVE_NAME)

print('Saving dataframe...')
output_file = os.path.join(
    os.getenv("SENTRY_DATA"), f"sentry/proc/RR2107_{OUTPUT_NAME}")
df_ret = bs.classify_samples(df=df,
                             write_to_disk=True,  # for now, let's write manually
                             output_file=output_file)
print('Dataframe saved')

if VISUALIZE:
    plot_and_save_detections(df_ret, dive_name=DIVE_NAME)
