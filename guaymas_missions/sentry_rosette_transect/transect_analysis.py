"""Reads in transect data and performs several analyses with visualization."""

import os
import utm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots

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
GENERATE_ST_PLOTS = True
GLOBAL_CORRELATION = True
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
        plt.scatter(ros_df["prac_salinity"], ros_df["pot_temp_C_its90"],
                    c=ros_df["sage_methane_ppm"], cmap="inferno")
        plt.savefig(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                 f"transect/figures/rosette_st_methane.png"))

    if GLOBAL_CORRELATION is True:
        # Global Pearson Correlation
        zr = np.zeros((len(SENTRY_VARS), len(SENTRY_VARS)))
        zp = np.zeros((len(SENTRY_VARS), len(SENTRY_VARS)))
        for i, v in enumerate(SENTRY_VARS):
            for j, s in enumerate(SENTRY_VARS):
                r, p = stats.pearsonr(scc_df[v].values, scc_df[s].values)
                zr[i, j] = r
                zp[i, j] = p
                # print(f"Rvalue: {r}, Pvalue: {p}, for {v} compared to {s}")
        fig, ax = plt.subplots()
        im = ax.imshow(zr, cmap="coolwarm")
        ax.set_xticks(np.arange(len(SENTRY_LABELS)))
        ax.set_xticklabels(SENTRY_LABELS)
        ax.set_yticks(np.arange(len(SENTRY_LABELS)))
        ax.set_yticklabels(SENTRY_LABELS)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        # Loop over data dimensions and create text annotations.
        for i in range(len(SENTRY_LABELS)):
            for j in range(len(SENTRY_LABELS)):
                text = ax.text(j, i, np.round(zr[i, j], 1),
                               ha="center", va="center", color="w")
        fig.tight_layout()
        plt.show()

        # Global Pearson Correlation
        ctd_df = ctd_df.dropna(subset=ROSETTE_VARS)
        zr = np.zeros((len(ROSETTE_VARS), len(ROSETTE_VARS)))
        zp = np.zeros((len(ROSETTE_VARS), len(ROSETTE_VARS)))
        for i, v in enumerate(ROSETTE_VARS):
            for j, s in enumerate(ROSETTE_VARS):
                r, p = stats.pearsonr(ctd_df[v].values, ctd_df[s].values)
                zr[i, j] = r
                zp[i, j] = p
                # print(f"Rvalue: {r}, Pvalue: {p}, for {v} compared to {s}")
        fig, ax = plt.subplots()
        im = ax.imshow(zr, cmap="coolwarm")
        ax.set_xticks(np.arange(len(ROSETTE_LABELS)))
        ax.set_xticklabels(ROSETTE_LABELS)
        ax.set_yticks(np.arange(len(ROSETTE_LABELS)))
        ax.set_yticklabels(ROSETTE_LABELS)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        # Loop over data dimensions and create text annotations.
        for i in range(len(ROSETTE_LABELS)):
            for j in range(len(ROSETTE_LABELS)):
                text = ax.text(j, i, np.round(zr[i, j], 1),
                               ha="center", va="center", color="w")
        fig.tight_layout()
        plt.show()

    # if LOCAL_CORRELATION is True:
    #     pairs = [comb for comb in combinations(ROSETTE_VARS, 2)]
    #     fig = make_subplots(rows=len(pairs), cols=1,
    #                         shared_xaxes="all", shared_yaxes="all")
    #     zim = np.zeros((len(pairs), len(ctd_df.index)))
    #     for i, pair in enumerate(pairs):
    #         rolling_r = ctd_df[pair[0]].rolling(
    #             window=900, center=True).corr(ctd_df[pair[1]])
    #         zim[i, :] = rolling_r
    #         fig.add_trace(go.Scatter(x=ctd_df.index, y=rolling_r, mode="markers",
    #                       name=f"{pair[0]} vs. {pair[1]}"), row=i+1, col=1)
    #         fig.add_shape(type="line", x0=ctd_df.index[0], y0=0, x1=ctd_df.index[-1], y1=0, line=dict(
    #             color="gray"), xref='x', yref='y', row=i+1, col=1)
    #     imfig = go.Figure(data=go.Heatmap(z=zim, x=ctd_df.index, y=[
    #                       f"{p[0]} vs. {p[1]}" for p in pairs], colorscale="RdBu_r"))
    #     fig.show()
    #     fname = "./rosette_lines_rolling_correlation"
    #     fig.write_html(fname)
    #     print("Figure", fname, "saved.")
    #     imfig.show()
    #     fname = "./rosette_heat_rolling_correlation"
    #     imfig.write_html(fname)
    #     print("Figure", fname, "saved.")

    # r_window_size = 900  # 15 minute interval
    #     pairs = [comb for comb in combinations(SENTRY_VARS, 2)]
    #     fig = make_subplots(rows=len(pairs), cols=1,
    #                         shared_xaxes="all", shared_yaxes="all")
    #     zim = np.zeros((len(pairs), len(scc_df.index)))
    #     for i, pair in enumerate(pairs):
    #         rolling_r = scc_df[pair[0]].rolling(
    #             window=900, center=True).corr(scc_df[pair[1]])
    #         zim[i, :] = rolling_r
    #         fig.add_trace(go.Scatter(x=scc_df.index, y=rolling_r, mode="markers",
    #                       name=f"{pair[0]} vs. {pair[1]}"), row=i+1, col=1)
    #         fig.add_shape(type="line", x0=scc_df.index[0], y0=0, x1=scc_df.index[-1], y1=0, line=dict(
    #             color="gray"), xref='x', yref='y', row=i+1, col=1)
    #     imfig = go.Figure(data=go.Heatmap(z=zim, x=scc_df.index, y=[
    #                       f"{p[0]} vs. {p[1]}" for p in pairs], colorscale="RdBu_r"))
    #     fig.show()
    #     fname = "./sentry_lines_rolling_correlation"
    #     fig.write_html(fname)
    #     print("Figure", fname, "saved.")
    #     imfig.show()
    #     fname = "./sentry_heat_rolling_correlation"
    #     imfig.write_html(fname)
    #     print("Figure", fname, "saved.")