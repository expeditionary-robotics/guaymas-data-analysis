"""Utilities file for parsing transect data."""
import os
import utm
import scipy
import numpy as np

# Input files relevant to transect analysis
INPUT_PYTHIA = os.path.join(os.getenv("SENTRY_DATA"),
                          "nopp/raw/N1-sentry613.txt")
INPUT_SAGE_LEG1 = os.path.join(os.getenv(
    "SENTRY_DATA"), "hcf/raw/HCF_205_Cast10_TransectExplorationLeg1.csv")
INPUT_SAGE_LEG2 = os.path.join(os.getenv(
    "SENTRY_DATA"), "hcf/raw/HCF_207_Cast11_TransectExplorationLeg2.csv")
INPUT_ROSETTE = os.path.join(
    os.getenv("SENTRY_DATA"), "ctd/proc/proc_transect.csv")
INPUT_GGA = os.path.join(os.getenv("SENTRY_DATA"),
                         "bottles/proc/gga_transect.csv")
INPUT_NH4 = os.path.join(os.getenv("SENTRY_DATA"),
                         "bottles/proc/nh4_transect.csv")
INPUT_BOTTLES = os.path.join(
    os.getenv("SENTRY_DATA"), "ctd/proc/proc_bottles.csv")
INPUT_SENTRY = os.path.join(
    os.getenv("SENTRY_DATA"), "sentry/proc/RR2107_sentry613_processed.csv")

# Parameters relevant to transect data processing
CHIMA = (27.407489, -111.389893)
CHIMB = (27.412645, -111.386915)
AX, AY, ZN, ZL = utm.from_latlon(CHIMA[0], CHIMA[1])
BX, BY, _, _ = utm.from_latlon(CHIMB[0], CHIMB[1])
RIDGE = (float(AX), float(AY))  # utm.to_latlon((AX + BX) / 2., (AY + BY) / 2., ZN, ZL)

# Processed data file targets for transect analysis
SENTRY_PYTHIA = os.path.join(os.getenv("SENTRY_DATA"),
                           "missions/transect/sentry_nopp.csv")
BOTTLES = os.path.join(os.getenv("SENTRY_DATA"),
                       "missions/transect/bottle_gga_nh4.csv")
ROSETTE_SAGE = os.path.join(os.getenv("SENTRY_DATA"),
                            "missions/transect/rosette_sage_proc.csv")

def get_transect_input_paths():
    """Returns the NOPP, SAGE1, SAGE2, Rosette, GGA, NH4, Bottles, and Sentry filepaths."""
    return INPUT_PYTHIA, INPUT_SAGE_LEG1, INPUT_SAGE_LEG2, INPUT_ROSETTE, INPUT_GGA, INPUT_NH4, INPUT_BOTTLES, INPUT_SENTRY

def get_transect_sentry_pythia_path():
    """Returns the Sentry and Pythia merged filepath."""
    return SENTRY_PYTHIA

def get_transect_bottles_path():
    """Returns the transect bottle merged filepath."""
    return BOTTLES

def get_transect_rosette_sage_path():
    """Returns the transect rosette and Sage merged filepath."""
    return ROSETTE_SAGE

def extract_trends(df, x, y, fit="polyfit", inplace=True):
    """Find the trends relationship between inputs and remove."""
    if fit is "polyfit":
        z = np.polyfit(df[x].values, df[y].values, 1)
        p = np.poly1d(z)
        df[f"{y}_bkgnd_{x}"] = p(df[x].values)
        df[f"{y}_anom_{x}"] = df[y].values - df[f"{y}_bkgnd_{x}"].values
        return df
    else:
        print("Currently only supporting polyfit removal.")
        return df

def smooth_data(df, target_vars, smooth_option="rolling_average", smooth_window=15):
    """Smooth data in df[target_vars] using smooth method."""
    if smooth_option is "rolling_average":
        r_window_size = int(60 * smooth_window)  # seconds
        for col in target_vars:
            df[f"{col}_{smooth_option}"] = df[col].rolling(
                r_window_size, center=True).mean()
    elif smooth_option is "butter":
        b, a = scipy.signal.butter(2, 0.01, fs=1)
        for col in target_vars:
            df[f"{col}_{smooth_option}"] = scipy.signal.filtfilt(
                b, a, df[col].values, padlen=150)
    else:
        print("Currently only supporting rolling_average and butter filters")
        pass
