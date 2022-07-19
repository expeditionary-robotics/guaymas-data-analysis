"""Calibration of Pythia instrument from laboratory data."""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from gasex import sol
from scipy.stats import stats
from scipy.io import loadmat
from scipy import optimize
from itertools import combinations


def TimeLagCorrection(sig, sig_time, tau, N):
    """Corrects for time lag in a signal.
    Input:
        sig: smoothed signal
        tau: sensor response time
        N: factor by which to down-sample the original signal
    """

    index_signal = sig_time
    index_sample = sig_time[::N]
    smooth_signal = sig
    signal_subset = smooth_signal[::N]
    tau_N = float(tau)/N
    X = np.exp(-1./tau_N)
    sig1 = signal_subset[0:-1]
    sig2 = signal_subset[1:]
    sigc = (sig2 - sig1*X)/(1.-X)
    return np.interp(index_signal, index_sample[1:], sigc)


# Get the data files for calibration
PYTHIA_CALIBRATION = os.path.join(
    os.getenv("SENTRY_DATA"), f"nopp/calibration.mat")
PYTHIA_STEP = os.path.join(os.getenv("SENTRY_DATA"),
                           f"nopp/T3_TimeResponse.txt")
OUTPUT = os.getenv("SENTRY_OUTPUT")

# Set processing parameters
STEP_SMOOTH_WINDOW = 120  # number of samples to smooth over
FUND_TAU = 35*60
RING_TAU = 40*60

# What to run
CALIB_TARGETS = ["fundamental", "ringdown"]
WITH_SMOOTH = False
WITH_TIME_CORRECTION = True

######################################################
# compute the functionals we want to be able to export
######################################################
mat = loadmat(PYTHIA_CALIBRATION)
cal_df = pd.DataFrame(mat['T3_Concentration_Signal'], columns=[
    "conc_ppm", "index", "ringdown", "fundamental"])

step_df = pd.read_table(PYTHIA_STEP, delimiter=",", skiprows=1, names=[
                        "datetime", "T", "ringdown", "fundamental"])
step_df["datetime"] = pd.to_datetime(step_df["datetime"])

# Smooth targets
step_df.dropna(inplace=True)
step_df.drop_duplicates(subset=["datetime"], keep="last", inplace=True)
step_df["fundamental_smooth"] = step_df.fundamental.rolling(
    STEP_SMOOTH_WINDOW).mean(centered=True)
step_df["ringdown_smooth"] = step_df.ringdown.rolling(
    STEP_SMOOTH_WINDOW).mean(centered=True)

# Convert calibration to nM
cal_df.loc[:, "conc_nM"] = cal_df.apply(lambda x: sol.sol_SP_pt(
    0, 3, gas='CH4', p_dry=x.conc_ppm*0.14*1e-6, units="mM")/1e-6, axis=1)

# Compute calibration functions for each signal
def ff(x):
    """Calibration curve for fundamental signal."""
    return np.interp(x, cal_df.conc_nM.values, cal_df['fundamental'].values)


def fundamental_conversion(x):
    """Takes raw fundamental data and performs lab conversion."""
    return np.interp(-x, -ff(np.linspace(0, np.nanmax(cal_df.conc_nM.values), 1000)), np.linspace(0, np.nanmax(cal_df.conc_nM.values), 1000))


def fr(x):
    """Calibration curve for ringdown signal."""
    return np.interp(x, cal_df.conc_nM.values, cal_df['ringdown'].values)


def ringdown_conversion(x):
    """Takes raw ringdown data and performs lab conversion."""
    return np.interp(x, fr(np.linspace(0, np.nanmax(cal_df.conc_nM.values), 1000)), np.linspace(0, np.nanmax(cal_df.conc_nM.values), 1000))


# Example of how to use calibration functions
if __name__ == "__main__":
    # Convert step and transect data into nM, use smoothed data
    if WITH_SMOOTH is True:
        step_df.loc[:, "fundamental_nM"] = step_df.apply(
            lambda x: fundamental_conversion(x["fundamental_smooth"]), axis=1)
        step_df.loc[:, "ringdown_nM"] = step_df.apply(
            lambda x: ringdown_conversion(x["ringdown_smooth"]), axis=1)

    # Convert step and transect data into nM, with time correction
    elif WITH_TIME_CORRECTION is True:
        step_df.loc[:, "fundamental_nM"] = TimeLagCorrection(
            fundamental_conversion(step_df["fundamental_smooth"]), step_df["datetime"], tau=FUND_TAU, N=int(FUND_TAU/4))
        step_df.loc[:, "ringdown_nM"] = TimeLagCorrection(
            ringdown_conversion(step_df["ringdown_smooth"]), step_df["datetime"], tau=RING_TAU, N=int(RING_TAU/4))

    else:
        step_df.loc[:, "fundamental_nM"] = fundamental_conversion(
            step_df["fundamental"])
        step_df.loc[:, "ringdown_nM"] = ringdown_conversion(
            step_df["ringdown"])
