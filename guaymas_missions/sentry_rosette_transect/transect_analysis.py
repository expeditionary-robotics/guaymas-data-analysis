"""Reads in transect data and performs several analyses with visualization."""

import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from itertools import combinations


# Datasets
SENTRY_NOPP = os.path.join(os.getenv("SENTRY_OUTPUT"),
                           "transect/sentry_nopp.csv")
BOTTLES = os.path.join(os.getenv("SENTRY_OUTPUT"),
                       "transect/bottle_gga_nh4.csv")
ROSETTE_SAGE = os.path.join(os.getenv("SENTRY_OUTPUT"),
                            "transect/rosette_sage_proc.csv")

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
GENERATE_ST_PLOTS = False
GLOBAL_CORRELATION = False
LOCAL_CORRELATION = True
CORR_WINDOW = 15  # minutes

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
        zr = np.zeros((len(SENTRY_NOPP_VARS), len(SENTRY_NOPP_VARS)))
        zp = np.zeros((len(SENTRY_NOPP_VARS), len(SENTRY_NOPP_VARS)))
        for i, v in enumerate(SENTRY_NOPP_VARS):
            for j, s in enumerate(SENTRY_NOPP_VARS):
                r, p = stats.pearsonr(scc_df[v].values, scc_df[s].values)
                zr[i, j] = r
                zp[i, j] = p
                # print(f"Rvalue: {r}, Pvalue: {p}, for {v} compared to {s}")
        fig, ax = plt.subplots()
        im = ax.imshow(zr, cmap="coolwarm")
        ax.set_xticks(np.arange(len(SENTRY_NOPP_LABELS)))
        ax.set_xticklabels(SENTRY_NOPP_LABELS)
        ax.set_yticks(np.arange(len(SENTRY_NOPP_LABELS)))
        ax.set_yticklabels(SENTRY_NOPP_LABELS)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        # Loop over data dimensions and create text annotations.
        for i in range(len(SENTRY_NOPP_LABELS)):
            for j in range(len(SENTRY_NOPP_LABELS)):
                text = ax.text(j, i, np.round(zr[i, j], 1),
                               ha="center", va="center", color="w")
        fig.tight_layout()
        plt.savefig(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                 f"transect/figures/sentry_nopp_global_corr.png"))
        plt.close()

        # Global Pearson Correlation
        ros_df = ros_df.dropna(subset=ROSETTE_SAGE_VARS)
        zr = np.zeros((len(ROSETTE_SAGE_VARS), len(ROSETTE_SAGE_VARS)))
        zp = np.zeros((len(ROSETTE_SAGE_VARS), len(ROSETTE_SAGE_VARS)))
        for i, v in enumerate(ROSETTE_SAGE_VARS):
            for j, s in enumerate(ROSETTE_SAGE_VARS):
                r, p = stats.pearsonr(ros_df[v].values, ros_df[s].values)
                zr[i, j] = r
                zp[i, j] = p
                # print(f"Rvalue: {r}, Pvalue: {p}, for {v} compared to {s}")
        fig, ax = plt.subplots()
        im = ax.imshow(zr, cmap="coolwarm")
        ax.set_xticks(np.arange(len(ROSETTE_SAGE_LABELS)))
        ax.set_xticklabels(ROSETTE_SAGE_LABELS)
        ax.set_yticks(np.arange(len(ROSETTE_SAGE_LABELS)))
        ax.set_yticklabels(ROSETTE_SAGE_LABELS)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        # Loop over data dimensions and create text annotations.
        for i in range(len(ROSETTE_SAGE_LABELS)):
            for j in range(len(ROSETTE_SAGE_LABELS)):
                text = ax.text(j, i, np.round(zr[i, j], 1),
                               ha="center", va="center", color="w")
        fig.tight_layout()
        plt.savefig(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                 f"transect/figures/rosette_sage_global_corr.png"))
        plt.close()

    if LOCAL_CORRELATION is True:
        r_window_size = 60 * CORR_WINDOW  # seconds

        # Sentry and NOPP
        pairs = [comb for comb in combinations(SENTRY_NOPP_VARS, 2)]
        fig = make_subplots(rows=len(pairs), cols=1,
                            shared_xaxes="all", shared_yaxes="all")
        zim = np.zeros((len(pairs), len(scc_df["timestamp"])))
        for i, pair in enumerate(pairs):
            rolling_r = scc_df[pair[0]].rolling(
                window=r_window_size, center=True).corr(scc_df[pair[1]])
            zim[i, :] = rolling_r
            fig.add_trace(go.Scatter(x=scc_df["timestamp"],
                                     y=rolling_r,
                                     mode="markers",
                                     name=f"{pair[0]} vs. {pair[1]}"),
                          row=i+1,
                          col=1)
            fig.add_shape(type="line",
                          x0=scc_df["timestamp"].values[0],
                          y0=0,
                          x1=scc_df["timestamp"].values[-1],
                          y1=0,
                          line=dict(color="gray"),
                          xref='x',
                          yref='y',
                          row=i+1,
                          col=1)
        imfig = go.Figure(data=go.Heatmap(z=zim,
                                          x=scc_df["timestamp"],
                                          y=[f"{p[0]} vs. {p[1]}" for p in pairs],
                                          colorscale="RdBu_r"))
        fname = os.path.join(os.getenv("SENTRY_OUTPUT"),
                             f"transect/figures/sentry_nopp_rolling_correlation.html")
        fig.write_html(fname)
        fname = os.path.join(os.getenv("SENTRY_OUTPUT"),
                             f"transect/figures/sentry_nopp_heat_rolling_correlation.html")
        imfig.write_html(fname)

        # Rosette and SAGE
        ros_df = ros_df.dropna(subset=ROSETTE_SAGE_VARS)
        pairs = [comb for comb in combinations(ROSETTE_SAGE_VARS, 2)]
        fig = make_subplots(rows=len(pairs), cols=1,
                            shared_xaxes="all", shared_yaxes="all")
        zim = np.zeros((len(pairs), len(ros_df["datetime"])))
        for i, pair in enumerate(pairs):
            rolling_r = ros_df[pair[0]].rolling(
                window=r_window_size, center=True).corr(ros_df[pair[1]])
            zim[i, :] = rolling_r
            fig.add_trace(go.Scatter(x=ros_df["datetime"],
                                     y=rolling_r,
                                     mode="markers",
                                     name=f"{pair[0]} vs. {pair[1]}"),
                          row=i+1,
                          col=1)
            fig.add_shape(type="line",
                          x0=ros_df["datetime"].values[0],
                          y0=0,
                          x1=ros_df["datetime"].values[-1],
                          y1=0,
                          line=dict(color="gray"),
                          xref='x',
                          yref='y',
                          row=i+1,
                          col=1)
        imfig = go.Figure(data=go.Heatmap(z=zim,
                                          x=ros_df["datetime"],
                                          y=[f"{p[0]} vs. {p[1]}" for p in pairs],
                                          colorscale="RdBu_r"))
        fname = os.path.join(os.getenv("SENTRY_OUTPUT"),
                             f"transect/figures/rosette_sage_rolling_correlation.html")
        fig.write_html(fname)
        fname = os.path.join(os.getenv("SENTRY_OUTPUT"),
                             f"transect/figures/rosette_sage_heat_rolling_correlation.html")
        imfig.write_html(fname)
