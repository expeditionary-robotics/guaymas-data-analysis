"""Reads in transect data and performs several analyses with visualization."""

import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from itertools import combinations
from transect_utils import get_transect_bottles_path, \
    get_transect_rosette_sage_path, get_transect_sentry_nopp_path


def compute_global_correlation(df, df_vars, df_labels, fname):
    """Computes global correlation factor and saves figure."""
    zr = np.zeros((len(df_vars), len(df_vars)))
    zp = np.zeros((len(df_vars), len(df_vars)))
    for i, v in enumerate(df_vars):
        for j, s in enumerate(df_vars):
            r, p = stats.pearsonr(df[v].values, df[s].values)
            zr[i, j] = r
            zp[i, j] = p
            # print(f"Rvalue: {r}, Pvalue: {p}, for {v} compared to {s}")
    fig, ax = plt.subplots()
    im = ax.imshow(zr, cmap="coolwarm")
    ax.set_xticks(np.arange(len(df_labels)))
    ax.set_xticklabels(df_labels)
    ax.set_yticks(np.arange(len(df_labels)))
    ax.set_yticklabels(df_labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    for i in range(len(df_labels)):
        for j in range(len(df_labels)):
            text = ax.text(j, i, np.round(zr[i, j], 1),
                           ha="center", va="center", color="w")
    fig.tight_layout()
    plt.savefig(os.path.join(os.getenv("SENTRY_OUTPUT"), fname))
    plt.close()


def compute_local_correlation(df, df_idx, df_vars, df_labels, window, fname):
    """Computes local correlation and saves figures.

    Input:
        df (dataframe): a 1Hz sampled dataframe of data
        df_idx (string): timestamp or datastring column name
        df_vars (list of strings): column names of targets
        df_labels (list of string): common names of targets
        window (int): number of seconds in the window
        fname (string): what to name the output image
    """
    # First check that the dataframe is 1Hz sampled
    sampled_at_1hz = all(df[df_idx].diff()[1:500] == np.timedelta64(1, 's')) == True
    if not sampled_at_1hz:
        print("ERROR: data is not sampled at 1Hz for averaging purposes")
        return
    # If properly sampled, continued with window correlation check
    pairs = [comb for comb in combinations(df_vars, 2)]
    fig = make_subplots(rows=len(pairs), cols=1,
                        shared_xaxes="all", shared_yaxes="all")
    zim = np.zeros((len(pairs), len(df[df_idx])))
    for i, pair in enumerate(pairs):
        rolling_r = df[pair[0]].rolling(
            window=window, center=True).corr(df[pair[1]])
        zim[i, :] = rolling_r
        fig.add_trace(go.Scatter(x=df[df_idx],
                                 y=rolling_r,
                                 mode="markers",
                                 name=f"{pair[0]} vs. {pair[1]}"),
                      row=i+1,
                      col=1)
        fig.add_shape(type="line",
                      x0=df[df_idx].values[0],
                      y0=0,
                      x1=df[df_idx].values[-1],
                      y1=0,
                      line=dict(color="gray"),
                      xref='x',
                      yref='y',
                      row=i+1,
                      col=1)
    imfig = go.Figure(data=go.Heatmap(z=zim,
                                      x=df[df_idx],
                                      y=[f"{p[0]} vs. {p[1]}" for p in pairs],
                                      colorscale="RdBu_r"))
    lname = os.path.join(os.getenv("SENTRY_OUTPUT"),
                         f"transect/figures/{fname}_rolling_correlation.html")
    fig.write_html(lname)
    hname = os.path.join(os.getenv("SENTRY_OUTPUT"),
                         f"transect/figures/{fname}_heat_rolling_correlation.html")
    imfig.write_html(hname)


# Datasets
SENTRY_NOPP = get_transect_sentry_nopp_path()
BOTTLES = get_transect_bottles_path()
ROSETTE_SAGE = get_transect_rosette_sage_path()

# What variables to compare
SENTRY_NOPP_VARS = ["O2", "obs", "nopp_fundamental", "dorpdt",
                    "potential_temp", "practical_salinity", "depth"]
SENTRY_NOPP_LABELS = ["O2", "OBS", "NOPP Inverse Fundamental", "dORPdt",
                      "Potential Temperature", "Practical Salinity", "Depth"]
ROSETTE_SAGE_VARS = ["beam_attenuation", "o2_umol_kg",
                     "sage_methane_ppm", "pot_temp_C_its90", "prac_salinity", "depth_m"]
ROSETTE_SAGE_LABELS = ["Beam Attentuation", "O2 (umol/kg)",
                       "SAGE Methane (ppm)", "Potential Temperature",
                       "Practical Salinity", "Depth"]

# Analyses
GENERATE_ST_PLOTS = False  # generates salinity-temperature plots
GLOBAL_CORRELATION = False  # generates a global correlation matrix
LOCAL_CORRELATION = True  # generates line and heatmaps of rolling correlations
CORR_WINDOW = 15  # sets the rolling correlation window, minutes

if __name__ == '__main__':
    # Get all of the data
    scc_df = pd.read_csv(SENTRY_NOPP)
    scc_df['timestamp'] = pd.to_datetime(scc_df['timestamp'])
    bott_df = pd.read_csv(BOTTLES)
    bott_df['datetime'] = pd.to_datetime(bott_df['datetime'])
    ros_df = pd.read_csv(ROSETTE_SAGE)
    ros_df['datetime'] = pd.to_datetime(ros_df['datetime'])

    if GENERATE_ST_PLOTS is True:
        plt.scatter(scc_df["ctd_sal"], scc_df["ctd_temp"],
                    c=scc_df["nopp_fundamental"], cmap="inferno_r")
        plt.savefig(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                 f"transect/figures/sentry_st_methane.png"))
        plt.close()
        plt.scatter(ros_df["prac_salinity"], ros_df["pot_temp_C_its90"],
                    c=ros_df["sage_methane_ppm"], cmap="inferno")
        plt.savefig(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                 f"transect/figures/rosette_st_methane.png"))
        plt.close()

    if GLOBAL_CORRELATION is True:
        # Global Pearson Correlation
        ros_df = ros_df.dropna(subset=ROSETTE_SAGE_VARS)
        compute_global_correlation(scc_df, SENTRY_NOPP_VARS, SENTRY_NOPP_LABELS,
                                   "transect/figures/sentry_nopp_global_corr.png")
        compute_global_correlation(ros_df, ROSETTE_SAGE_VARS, ROSETTE_SAGE_LABELS,
                                   "transect/figures/rosette_sage_global_corr.png")

    if LOCAL_CORRELATION is True:
        r_window_size = 60 * CORR_WINDOW  # seconds
        ros_df = ros_df.dropna(subset=ROSETTE_SAGE_VARS)
        compute_local_correlation(scc_df, "timestamp", SENTRY_NOPP_VARS,
                                  SENTRY_NOPP_LABELS, r_window_size, "sentry_nopp")
        compute_local_correlation(ros_df, "datetime", ROSETTE_SAGE_VARS,
                                  ROSETTE_SAGE_LABELS, r_window_size, "rosette_sage")
