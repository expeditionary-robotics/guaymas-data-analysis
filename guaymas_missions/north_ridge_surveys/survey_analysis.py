"""Basic analysis on survey data, including detrending and smoothing."""

import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import scipy.signal
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from itertools import combinations


def extract_trends(df, x, y_list, labels, fit="polyfit", inplace=True, plot=False):
    """Find the trends relationship between inputs and remove."""
    bins = np.linspace(np.nanmin(df[x]), np.nanmax(df[x]), 20)
    if fit is "polyfit" and plot is True:
        fig, ax = plt.subplots(len(y_list), sharex=True)
        for i, y in enumerate(y_list):
            groups = df.groupby(pd.cut(df[x], bins))
            ax[i].scatter(df[x], df[y], label="Original Data", s=1)
            z = np.polyfit(groups.mean()[x], groups.mean()[y], 1)
            p = np.poly1d(z)
            plot_range = bins
            ax[i].plot(plot_range, p(plot_range),
                       color="orange", label="Line of Fit")
            ax[i].set_ylabel(labels[i])
            df[y] = df[y].values - p(df[x].values)
        plt.legend()
        plt.show()
        return df
    elif fit is "polyfit":
        for i, y in enumerate(y_list):
            groups = df.groupby(pd.cut(df[x], bins))
            z = np.polyfit(groups.mean()[x], groups.mean()[y], 1)
            p = np.poly1d(z)
            df[y] = df[y].values - p(df[x].values)
        return df
    else:
        print("Currently only supporting polyfit removal.")
        return df


def smooth_data(df, target_vars, smooth_option="rolling_average", smooth_window=15):
    """Smooth data in df[target_vars] using smooth method."""
    if smooth_option is "rolling_average":
        r_window_size = int(60 * smooth_window)  # seconds
        for col in target_vars:
            df[col] = df[col].rolling(
                r_window_size, center=True).mean()
    elif smooth_option is "butter":
        b, a = scipy.signal.butter(2, 0.01, fs=1)
        for col in target_vars:
            df[col] = scipy.signal.filtfilt(
                b, a, df[col].values, padlen=150)
    else:
        print("Currently only supporting rolling_average and butter filters")
        pass


def norm(targ):
    """Return the 0-1 normalized value of a target data stream."""
    return (targ - np.nanmin(targ))/(np.nanmax(targ) - np.nanmin(targ))


##########
# Globals
##########
DIVES = ["sentry607", "sentry608", "sentry610", "sentry611"]
METHANE_COL = ["fundamental", "fundamental", "fundamental", "methane"]
METHANE_LABEL = ["Pythia Fundamental", "Pythia Fundamental",
                 "Pythia Fundamental", "SAGE Methane"]
PLOT_VARS = ["O2_conc", "potential_temp",
             "practical_salinity", "obs", "dorpdt"]
PLOT_LABELS = [r"Oxygen Concentration (\mu mol kg^{-1})", "Potential Temperature (C)",
               "Practical Salinity (PSU)", "Optical Backscatter (%)", "dORPdt"]

DETREND = True  # whether to remove altitude effects
DETREND_VARS = ["O2_conc", "potential_temp", "practical_salinity"]
DETREND_LABELS = ["O2", "Potential Temperature", "Practical Salinity"]

SMOOTH = False  # whether to smooth the data
SMOOTH_OPTION = "rolling_average"  # rolling_averge or butter
SMOOTH_WINDOW = 15  # sets the rolling average window, minutes
SMOOTH_VARS = ["O2_conc", "obs", "potential_temp", "practical_salinity"]

NORMALIZE = True  # whether to normalize the data
NORM_VARS = []

DEPTH_PLOT = True  # whether to plot detrended data


if __name__ == '__main__':
    for idx, DIVE_NAME in enumerate(DIVES):
        #######
        # Data Processing
        #######
        # Get the data
        input_name = f"{DIVE_NAME}_processed.csv"
        input_file = os.path.join(
            os.getenv("SENTRY_DATA"), f"sentry/proc/RR2107_{input_name}")
        scc_df = pd.read_csv(input_file)
        scc_df["timestamp"] = pd.to_datetime(scc_df['timestamp'])

        # Add the right methane target
        SMOOTH_VARS.append(METHANE_COL[idx])
        NORM_VARS.append(METHANE_COL[idx])
        PLOT_VARS.append(METHANE_COL[idx])
        PLOT_LABELS.append(METHANE_LABEL[idx])

        # Perform base conversions (smoothing, detrending, normalization)
        if DETREND is True:
            scc_df = extract_trends(
                scc_df, 'depth', DETREND_VARS, DETREND_LABELS, plot=DEPTH_PLOT)

        if SMOOTH is True:
            smooth_data(scc_df, SMOOTH_VARS,
                        smooth_option=SMOOTH_OPTION, smooth_window=SMOOTH_WINDOW)

        if NORMALIZE is True:
            scc_df[METHANE_COL[idx]] = norm(scc_df[METHANE_COL[idx]])

        ############
        # Apply binary pseudosensor(s)
        ############
        # TODO

        ###########
        # Simple visualizations of the data
        ###########
        for target, label in zip(PLOT_VARS, PLOT_LABELS):
            # TODO
            pass
            # top-down overview
            # time series
            # time-depth-value slice
            # radius spoke (signed distance by value colored by time?)

        ############
        # Save processed data product and meta data
        ############
        # TODO
