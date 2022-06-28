"""NOPP data calibration file."""

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
from plotly.subplots import make_subplots
from transect_utils import get_transect_sentry_nopp_path


def TimeLagCorrection(sig, sig_time, tau, S, N):
    """Corrects for time lag in a signal.

    Input:
        sig: original signal
        tau: sensor response time
        S: moving average applied to original signal before correction
        N: factor by which to down-sample the original signal
    """

    index_signal = sig_time#range(0, len(sig))
    index_sample = sig_time[::N]#range(0, len(sig), N)
    smooth_signal = sig#np.convolve(sig, np.ones(S) / S, mode='full')
    signal_subset = smooth_signal[::N]
    tau_N = float(tau)/N
    X = np.exp(-1./tau_N)
    sig1 = signal_subset[0:-1]
    sig2 = signal_subset[1:]
    sigc = (sig2 - sig1*X)/(1.-X)

    return np.interp(index_signal, index_sample[1:], sigc)

def o2styleTimeLagCorrection(sig, sig_times, tau):
    """Corrects for time lag in signal."""
    sig_times_s = [t.timestamp() for t in sig_times]
    
    new = []
    new_times = []
    for t1, t2, c1, c2 in zip(sig_times_s[0:-1], sig_times_s[1:], sig[0:-1], sig[1:]):
        print(t1, t2, c1, c2)
        new_times.append((t1 + t2)/2)
        b = (1. + 2. * tau/(t2-t1))**-1
        a = 1. - 2.*b
        new.append(1/(2*b)*(c2 - a*c1))
    
    return np.interp(sig_times_s, new_times, new)



# Get the data files for calibration
SENTRY_NOPP = get_transect_sentry_nopp_path()
NOPP_CALIBRATION = os.path.join(
    os.getenv("SENTRY_DATA"), f"nopp/calibration.mat")
NOPP_STEP = os.path.join(os.getenv("SENTRY_DATA"),
                         f"nopp/T3_TimeResponse.txt")
OUTPUT = os.getenv("SENTRY_OUTPUT")

# Set processing parameters
STEP_SMOOTH_WINDOW = 120  # number of samples to smooth over
TRANSECT_SMOOTH_WINDOW = 5  # number of minutes to smooth over

# What to run
CALIB_TARGETS = ["fundamental", "ringdown"]
WITH_SMOOTH = False
WITH_TIME_CORRECTION = True
SAVE_TO_FILE = True  # whether to save this to transect file


if __name__ == "__main__":
    # Load in data
    mat = loadmat(NOPP_CALIBRATION)
    cal_df = pd.DataFrame(mat['T3_Concentration_Signal'], columns=[
        "conc_ppm", "index", "ringdown", "fundamental"])

    step_df = pd.read_table(NOPP_STEP, delimiter=",", skiprows=1, names=[
                            "datetime", "T", "ringdown", "fundamental"])
    step_df["datetime"] = pd.to_datetime(step_df["datetime"])

    scc_df = pd.read_csv(SENTRY_NOPP)
    scc_df['timestamp'] = pd.to_datetime(scc_df['timestamp'])

    # Smooth targets
    step_df.dropna(inplace=True)
    step_df.drop_duplicates(subset=["datetime"], keep="last", inplace=True)
    step_df["fundamental_smooth"] = step_df.fundamental.rolling(
        STEP_SMOOTH_WINDOW).mean(centered=True)
    step_df["ringdown_smooth"] = step_df.ringdown.rolling(
        STEP_SMOOTH_WINDOW).mean(centered=True)
    scc_df["fundamental_smooth"] = scc_df.nopp_fundamental.rolling(
        TRANSECT_SMOOTH_WINDOW*60).mean(centered=True)
    scc_df["ringdown_smooth"] = scc_df.nopp_ringdown.rolling(
        TRANSECT_SMOOTH_WINDOW*60).mean(centered=True)
    

    # Convert calibration to nM
    # cal_df.loc[:, "conc_nM"] = cal_df.conc_ppm
    cal_df.loc[:, "conc_nM"] = cal_df.apply(lambda x: sol.sol_SP_pt(
        0, 3, gas='CH4', p_dry=x.conc_ppm*0.14*1e-6, units="mM")/1e-6, axis=1)

    for target in CALIB_TARGETS:
        # Interpolative fit
        def f(x): return np.interp(
            x, cal_df.conc_nM.values, cal_df[target].values)

        # Check interpolation
        m = f(np.linspace(0, 1000, 100))
        plt.scatter(cal_df.conc_nM.values,
                    cal_df[target].values, c="orange", label="Lab Data")
        plt.plot(np.linspace(0, 1000, 100), m, label="Interpolated")
        plt.xlabel("Methane Concentration (nM)")
        plt.ylabel(f"{target} value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT, f"nopp/nopp_{target}_calib.png"))
        plt.close()

        # Invert the interpolation
        if target is "fundamental":
            def fp(x): return np.interp(-x, -f(np.linspace(0, np.nanmax(cal_df.conc_nM.values),
                                                           1000)), np.linspace(0, np.nanmax(cal_df.conc_nM.values), 1000))
            check_num = 0.4
        elif target is "ringdown":
            def fp(x): return np.interp(x, f(np.linspace(0, np.nanmax(cal_df.conc_nM.values),
                                                         1000)), np.linspace(0, np.nanmax(cal_df.conc_nM.values), 1000))
            check_num = 1.2

        # Check the inverted interpolation
        m = fp(np.linspace(0, check_num, 1000))
        plt.scatter(cal_df[target].values,
                    cal_df.conc_nM.values, c="orange", label="Lab Data")
        plt.plot(np.linspace(0, check_num, 1000), m, label="Interpolated")
        plt.ylabel("Methane Concentration (nM)")
        plt.xlabel(f"{target} value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(
            OUTPUT, f"nopp/nopp_{target}_calib_inverted.png"))
        plt.close()

        # Convert step and transect data into nM, use smoothed data
        if WITH_SMOOTH is True:
            step_df.loc[:, f"{target}_nM"] = step_df.apply(
                lambda x: fp(x[f"{target}_smooth"]), axis=1)
            plt.plot(fp(step_df[target]), label="Step, Converted")
            plt.plot(step_df[f"{target}_nM"], label="Step, Smoothed")
            plt.xlabel("Sample Number")
            plt.ylabel("Methane (nM)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(
                OUTPUT, f"nopp/nopp_{target}_step_smoothed.png"))
            plt.close()

            scc_df.loc[:, f"{target}_nM"] = scc_df.apply(
                lambda x: fp(x[f"{target}_smooth"]), axis=1)
            plt.plot(fp(scc_df[f"nopp_{target}"]), label="Transect, Converted")
            plt.plot(scc_df[f"{target}_nM"], label="Transect, Smoothed")
            plt.xlabel("Sample Number")
            plt.ylabel("Methane (nM)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(
                OUTPUT, f"nopp/nopp_{target}_transect_smoothed.png"))
            plt.close()

            # Visualize the transect
            fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
            ax[0].scatter(scc_df.timestamp, scc_df[f"nopp_{target}"])
            ax[1].scatter(scc_df.timestamp, scc_df[f"{target}_nM"])
            ax[0].set_ylabel(f"{target} signal")
            ax[1].set_ylabel("Methane, nM -- Smoothed")
            ax[1].set_xlabel("Datetime")
            fig.tight_layout()
            fig.savefig(os.path.join(
                OUTPUT, f"nopp/nopp_{target}_transect_smoothed_all.png"))
            plt.close()

        # Convert step and transect data into nM, with time correction
        elif WITH_TIME_CORRECTION is True:
            if target is "fundamental":
                tau = 35*60
            elif target is "ringdown":
                tau = 40*60
            step_df.loc[:, f"{target}_nM"] = TimeLagCorrection(
                fp(step_df[f"{target}_smooth"]), step_df["datetime"], tau=tau, S=1, N=int(tau/4))
            plt.plot(step_df["datetime"], fp(step_df[target]), label="Step, Converted")
            plt.plot(step_df["datetime"], fp(step_df[f"{target}_smooth"]), label="Step, Smoothed")
            plt.plot(step_df["datetime"], step_df[f"{target}_nM"], label="Step, Time Correction")
            plt.xlabel("Sample Number")
            plt.ylabel("Methane (nM)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(
                OUTPUT, f"nopp/nopp_{target}_step_corrected.png"))
            plt.close()

            scc_df.loc[:, f"{target}_nM"] = TimeLagCorrection(
                fp(scc_df[f"{target}_smooth"]), scc_df["timestamp"], tau=tau, S=1, N=int(tau/4))
            plt.plot(scc_df["timestamp"], fp(scc_df[f"nopp_{target}"]), label="Transect, Converted")
            plt.plot(scc_df["timestamp"], fp(scc_df[f"{target}_smooth"]), label="Transect, Smoothed")
            plt.plot(scc_df["timestamp"], scc_df[f"{target}_nM"], label="Transect, Time Correction")
            plt.xlabel("Sample Number")
            plt.ylabel("Methane (nM)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(
                OUTPUT, f"nopp/nopp_{target}_transect_corrected.png"))
            plt.close()

            # Visualize the transect
            fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
            ax[0].scatter(scc_df.timestamp, scc_df[f"nopp_{target}"])
            ax[1].scatter(scc_df.timestamp, scc_df[f"{target}_nM"])
            ax[0].set_ylabel(f"{target} signal")
            ax[1].set_ylabel("Methane, nM -- Time Corrected")
            ax[1].set_xlabel("Datetime")
            fig.tight_layout()
            fig.savefig(os.path.join(
                OUTPUT, f"nopp/nopp_{target}_transect_corrected_all.png"))
            plt.close()
        
        else:
            step_df.loc[:, f"{target}_nM"] = fp(step_df[target])
            scc_df.loc[:, f"{target}_nM"] = fp(scc_df[f"nopp_{target}"])


    if SAVE_TO_FILE is True:
        scc_df.to_csv(SENTRY_NOPP)
