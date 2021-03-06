"""Reads in transect data and performs several analyses with visualization."""

import os
import stumpy
import pandas as pd
import numpy as np
import ruptures as rpt
import scipy.stats as stats
import scipy.signal
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from itertools import combinations
from transect_utils import get_transect_bottles_path, \
    get_transect_rosette_sage_path, get_transect_sentry_nopp_path, \
        extract_trends, smooth_data


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


def compute_local_correlation(df, df_idx, df_vars, df_labels, window, fname, fname_add):
    """Computes local correlation and saves figures.

    Input:
        df (dataframe): a 1Hz sampled dataframe of data
        df_idx (string): timestamp or datastring column name
        df_vars (list of strings): column names of targets
        df_labels (list of string): common names of targets
        window (int): number of seconds in the window
        fname (string): what to name the output image
        fname_add (string): indicator for additional meta data
    """
    # First check that the dataframe is 1Hz sampled
    sampled_at_1hz = all(df[df_idx].diff()[1:500] ==
                         np.timedelta64(1, 's')) == True
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
                         f"transect/figures/{fname}_rolling_correlation{fname_add}.html")
    fig.write_html(lname)
    hname = os.path.join(os.getenv("SENTRY_OUTPUT"),
                         f"transect/figures/{fname}_heat_rolling_correlation{fname_add}.html")
    imfig.write_html(hname)


def compute_anoms_and_regimes(df, df_idx, df_vars, df_labels,  window, fname, fname_add):
    """Computes anomalies and regime changes for a given df."""
    # compute the matrix profile, discord id, and regime changes
    for i, col in enumerate(df_vars):
        targ = df[col]
        mp = stumpy.stump(targ, m=window)
        discord_idx = np.argsort(mp[:, 0])[-1]
        cac, regime_locations = stumpy.fluss(
            mp[:, 1], L=window, n_regimes=3, excl_factor=1)

        # create plots
        fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
        plt.suptitle(
            f'Discord (Anomaly/Novelty) Discovery in {df_labels[i]}')
        axs[0].plot(df[df_idx], targ)
        axs[0].set_ylabel(df_labels[i])
        axs[0].axvline(x=df[df_idx].values[discord_idx], linestyle="dashed")
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Matrix Profile')
        axs[1].axvline(x=df[df_idx].values[discord_idx], linestyle="dashed")
        axs[1].plot(df[df_idx].values[:-(window-1)], mp[:, 0])
        plt.savefig(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                 f"transect/figures/{fname}_anomaly_{col}{fname_add}.png"))
        plt.close()

        fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
        plt.suptitle(
            f'Regime Change Identification in {df_labels[i]}')
        axs[0].plot(df[df_idx], targ)
        axs[0].axvline(
            x=df[df_idx].values[regime_locations[0]], linestyle="dashed")
        axs[0].axvline(
            x=df[df_idx].values[regime_locations[1]], linestyle="dashed")
        axs[1].plot(df[df_idx].values[:-(window-1)], cac, color='C1')
        axs[1].axvline(
            x=df[df_idx].values[regime_locations[0]], linestyle="dashed")
        axs[1].axvline(
            x=df[df_idx].values[regime_locations[1]], linestyle="dashed")
        plt.savefig(os.path.join(os.getenv("SENTRY_OUTPUT"),
                    f"transect/figures/{fname}_regimes_{col}{fname_add}.png"))
        plt.close()

        print(f"{fname}")
        print(
            f"{df_labels[i]} Anomaly Detected at {df[df_idx].values[discord_idx]}")
        print(
            f"{df_labels[i]} Regimes Detected at {df[df_idx].values[regime_locations[0]]}, {df[df_idx].values[regime_locations[1]]}")
        print("------------")


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
REMOVE_DEPTH = True  # compute depth correction
SENTRY_DEPTH_TARGET_VARS = ["O2", "potential_temp", "practical_salinity"]
SENTRY_DEPTH_TARGET_LABELS = [
    "O2", "Potential Temperature", "Practical Salinity"]
ROSETTE_DEPTH_TARGET_VARS = ["o2_umol_kg", "pot_temp_C_its90", "prac_salinity"]
ROSETTE_DEPTH_TARGET_LABELS = [
    "O2 (umol/kg)", "Potential Temperature", "Practical Salinity"]

COMPUTE_WITH_SMOOTH = True  # smooth data before analysis
SMOOTH_OPTION = "rolling_average"  # rolling_averge of butter
SMOOTH_WINDOW = 15  # sets the rolling average window, minutes
SENTRY_SMOOTH_TARGET_VARS = ["O2", "obs", "potential_temp",
                             "practical_salinity", "nopp_fundamental", "dorpdt"]
SENTRY_SMOOTH_TARGET_LABELS = ["O2", "OBS", "Potential Temperature",
                               "Practical Salinity", "NOPP Inverse Fundamental", "dORPdt"]
ROSETTE_SMOOTH_TARGET_VARS = ["beam_attenuation", "o2_umol_kg",
                              "sage_methane_ppm", "pot_temp_C_its90", "prac_salinity"]
ROSETTE_SMOOTH_TARGET_LABELS = ["Beam Attentuation", "O2 (umol/kg)",
                                "SAGE Methane (ppm)", "Potential Temperature",
                                "Practical Salinity"]

GENERATE_ST_PLOTS = False  # generates salinity-temperature plots
GLOBAL_CORRELATION = False  # generates a global correlation matrix
LOCAL_CORRELATION = False  # generates line and heatmaps of rolling correlations
CORR_WINDOW = 15  # sets the rolling correlation window, minutes
STUMPY_FRONT_ANALYSIS = False  # whether to attempt front ID with stumpy package
FRONT_WINDOW = 15  # sets the window for front detection, minutes
RUPTURES_FRONT_ANALYSIS = True # whether to attempt front ID with ruptures package
CREATE_STATS_PLOTS = False  # whether to generate summary stats plots
FIGURE_NAME_ADDITION = ""

if __name__ == '__main__':
    # Get all of the data
    scc_df = pd.read_csv(SENTRY_NOPP)
    scc_df['timestamp'] = pd.to_datetime(scc_df['timestamp'])
    scc_df.dropna(subset=SENTRY_NOPP_VARS, inplace=True)
    bott_df = pd.read_csv(BOTTLES)
    bott_df['datetime'] = pd.to_datetime(bott_df['datetime'])
    ros_df = pd.read_csv(ROSETTE_SAGE)
    ros_df['datetime'] = pd.to_datetime(ros_df['datetime'])
    ros_df = ros_df.dropna(subset=ROSETTE_SAGE_VARS)

    if REMOVE_DEPTH is True:
        for v in SENTRY_DEPTH_TARGET_VARS:
            scc_df = extract_trends(scc_df, 'depth', v)
        for targ in SENTRY_DEPTH_TARGET_VARS:
            SENTRY_NOPP_VARS.remove(targ)
            SENTRY_NOPP_VARS.append(f"{targ}_anom_depth")
            SENTRY_SMOOTH_TARGET_VARS.remove(targ)
            SENTRY_SMOOTH_TARGET_VARS.append(f"{targ}_anom_depth")
        for targ in SENTRY_DEPTH_TARGET_LABELS:
            SENTRY_NOPP_LABELS.remove(targ)
            SENTRY_NOPP_LABELS.append(f"{targ} Depth Corrected")
            SENTRY_SMOOTH_TARGET_LABELS.remove(targ)
            SENTRY_SMOOTH_TARGET_LABELS.append(f"{targ} Depth Corrected")

        for v in ROSETTE_DEPTH_TARGET_VARS:
            ros_df = extract_trends(ros_df, 'depth_m', v)
        for targ in ROSETTE_DEPTH_TARGET_VARS:
            ROSETTE_SAGE_VARS.remove(targ)
            ROSETTE_SAGE_VARS.append(f"{targ}_anom_depth_m")
            ROSETTE_SMOOTH_TARGET_VARS.remove(targ)
            ROSETTE_SMOOTH_TARGET_VARS.append(f"{targ}_anom_depth_m")
        for targ in ROSETTE_DEPTH_TARGET_LABELS:
            ROSETTE_SAGE_LABELS.remove(targ)
            ROSETTE_SAGE_LABELS.append(f"{targ} Depth Corrected")
            ROSETTE_SMOOTH_TARGET_LABELS.remove(targ)
            ROSETTE_SMOOTH_TARGET_LABELS.append(f"{targ} Depth Corrected")

        FIGURE_NAME_ADDITION = FIGURE_NAME_ADDITION + "_depthcorr"

    if COMPUTE_WITH_SMOOTH is True:
        """Smooth all of the data targets"""
        smooth_data(scc_df, SENTRY_SMOOTH_TARGET_VARS, smooth_option=SMOOTH_OPTION, smooth_window=SMOOTH_WINDOW)
        ros_df_1 = ros_df[ros_df.datetime <=
                          pd.Timestamp("2021-11-30 07:00:04")]
        ros_df_2 = ros_df[ros_df.datetime >
                          pd.Timestamp("2021-11-30 07:00:04")]
        smooth_data(ros_df_1, ROSETTE_SMOOTH_TARGET_VARS, smooth_option=SMOOTH_OPTION, smooth_window=SMOOTH_WINDOW)
        smooth_data(ros_df_2, ROSETTE_SMOOTH_TARGET_VARS, smooth_option=SMOOTH_OPTION, smooth_window=SMOOTH_WINDOW)
        ros_df = ros_df_1.append(ros_df_2)

        for targ in SENTRY_SMOOTH_TARGET_VARS:
            SENTRY_NOPP_VARS.remove(targ)
            SENTRY_NOPP_VARS.append(f"{targ}_{SMOOTH_OPTION}")
        for targ in SENTRY_SMOOTH_TARGET_LABELS:
            SENTRY_NOPP_LABELS.remove(targ)
            SENTRY_NOPP_LABELS.append(f"{targ} Smoothed")
        for targ in ROSETTE_SMOOTH_TARGET_VARS:
            ROSETTE_SAGE_VARS.remove(targ)
            ROSETTE_SAGE_VARS.append(f"{targ}_{SMOOTH_OPTION}")
        for targ in ROSETTE_SMOOTH_TARGET_LABELS:
            ROSETTE_SAGE_LABELS.remove(targ)
            ROSETTE_SAGE_LABELS.append(f"{targ} Smoothed")
        
        FIGURE_NAME_ADDITION = FIGURE_NAME_ADDITION + "_smoothed"
    
    if GENERATE_ST_PLOTS is True:
        c = plt.scatter(scc_df["ctd_sal"], scc_df["ctd_temp"],
                    c=scc_df["nopp_fundamental"], cmap="inferno_r")
        plt.colorbar(c)
        plt.xlabel("Salinity")
        plt.ylabel("Temperature")
        plt.savefig(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                 f"transect/figures/sentry_st_methane{FIGURE_NAME_ADDITION}.png"))
        plt.show()
        plt.close()
        
        c = plt.scatter(ros_df["prac_salinity"], ros_df["pot_temp_C_its90"],
                    c=ros_df["sage_methane_ppm"], cmap="inferno")
        plt.colorbar(c)
        plt.xlabel("Salinity")
        plt.ylabel("Temperature")
        plt.savefig(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                 f"transect/figures/rosette_st_methane{FIGURE_NAME_ADDITION}.png"))
        plt.close()

    if GLOBAL_CORRELATION is True:
        # Global Pearson Correlation
        ros_df = ros_df.dropna(subset=ROSETTE_SAGE_VARS)
        scc_df = scc_df.dropna(subset=SENTRY_NOPP_VARS)
        compute_global_correlation(scc_df, SENTRY_NOPP_VARS, SENTRY_NOPP_LABELS,
                                   f"transect/figures/sentry_nopp_global_corr{FIGURE_NAME_ADDITION}.png")
        compute_global_correlation(ros_df, ROSETTE_SAGE_VARS, ROSETTE_SAGE_LABELS,
                                   f"transect/figures/rosette_sage_global_corr{FIGURE_NAME_ADDITION}.png")

    if LOCAL_CORRELATION is True:
        r_window_size = int(60 * CORR_WINDOW)  # seconds
        ros_df = ros_df.dropna(subset=ROSETTE_SAGE_VARS)
        compute_local_correlation(scc_df, "timestamp", SENTRY_NOPP_VARS,
                                  SENTRY_NOPP_LABELS, r_window_size, "sentry_nopp", FIGURE_NAME_ADDITION)
        compute_local_correlation(ros_df, "datetime", ROSETTE_SAGE_VARS,
                                  ROSETTE_SAGE_LABELS, r_window_size, "rosette_sage", FIGURE_NAME_ADDITION)

    if STUMPY_FRONT_ANALYSIS is True:
        scc_df.dropna(inplace=True)
        scc_df = scc_df[scc_df.timestamp > pd.Timestamp("2021-11-30 06:00:00")]
        compute_anoms_and_regimes(scc_df, "timestamp", SENTRY_NOPP_VARS,
                                  SENTRY_NOPP_LABELS, FRONT_WINDOW, "sentry_nopp", FIGURE_NAME_ADDITION)

        # ros_df leg 1
        ros_df_1 = ros_df[ros_df.datetime <=
                          pd.Timestamp("2021-11-30 07:00:04")]
        ros_df_1.dropna(inplace=True, subset=ROSETTE_SAGE_VARS)
        compute_anoms_and_regimes(ros_df_1, "datetime", ROSETTE_SAGE_VARS,
                                  ROSETTE_SAGE_LABELS, FRONT_WINDOW, "rosette_sage_leg1", FIGURE_NAME_ADDITION)

        # ros_df leg 2
        ros_df_2 = ros_df[ros_df.datetime >
                          pd.Timestamp("2021-11-30 07:00:04")]
        ros_df_2.dropna(inplace=True, subset=ROSETTE_SAGE_VARS)
        compute_anoms_and_regimes(ros_df_2, "datetime", ROSETTE_SAGE_VARS,
                                  ROSETTE_SAGE_LABELS, FRONT_WINDOW, "rosette_sage_leg2", FIGURE_NAME_ADDITION)

    if RUPTURES_FRONT_ANALYSIS is True:
        # based on https://techrando.com/2019/08/14/a-brief-introduction-to-change-point-detection-using-python/
        scc_df.dropna(inplace=True)
        scc_df = scc_df[scc_df.timestamp > pd.Timestamp("2021-11-30 06:00:00")]
        model = "rbf"

        for i, col in enumerate(SENTRY_NOPP_VARS):
            algo = rpt.Pelt(model=model).fit(scc_df[col].values[::10])
            result = algo.predict(pen=10)
            rpt.display(scc_df[col].values[::10], result, figsize=(10, 6))
            plt.title(f'Change Point Detection: {SENTRY_NOPP_LABELS[i]}')
            plt.savefig(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                     f"transect/figures/sentry_nopp_ruptures_{col}{FIGURE_NAME_ADDITION}.png"))
            plt.close()

        ros_df_1 = ros_df[ros_df.datetime <=
                          pd.Timestamp("2021-11-30 07:00:04")]
        ros_df_1.dropna(inplace=True, subset=ROSETTE_SAGE_VARS)
        ros_df_2 = ros_df[ros_df.datetime >
                          pd.Timestamp("2021-11-30 07:00:04")]
        ros_df_2.dropna(inplace=True, subset=ROSETTE_SAGE_VARS)

        for i, col in enumerate(ROSETTE_SAGE_VARS):
            algo = rpt.Pelt(model=model).fit(ros_df_1[col].values[::10])
            result = algo.predict(pen=10)
            rpt.display(ros_df_1[col].values[::10], result, figsize=(10, 6))
            plt.title(f'Change Point Detection: {ROSETTE_SAGE_LABELS[i]}')
            plt.savefig(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                     f"transect/figures/rosette_sage_leg1_ruptures_{col}{FIGURE_NAME_ADDITION}.png"))
            plt.close()

            algo = rpt.Pelt(model=model).fit(ros_df_2[col].values)
            result = algo.predict(pen=10)
            rpt.display(ros_df_2[col].values, result, figsize=(10, 6))
            plt.title(f'Change Point Detection: {ROSETTE_SAGE_LABELS[i]}')
            plt.savefig(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                     f"transect/figures/rosette_sage_leg2_ruptures_{col}{FIGURE_NAME_ADDITION}.png"))
            plt.close()

    if CREATE_STATS_PLOTS is True:
        scc_df.dropna(subset=SENTRY_NOPP_VARS, inplace=True)
        scc_df = scc_df[scc_df.timestamp > pd.Timestamp("2021-11-30 06:00:00")]
        for i, col in enumerate(SENTRY_NOPP_VARS):
            # Histograms
            plt.hist(scc_df[col].values, bins=100, density=True)
            plt.title(SENTRY_NOPP_LABELS[i])
            plt.savefig(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                     f"transect/figures/sentry_nopp_histo_{col}{FIGURE_NAME_ADDITION}.png"))
            plt.close()

            # Boxplots
            plt.boxplot(scc_df[col].values, labels=[
                        SENTRY_NOPP_LABELS[i]], meanline=True, showmeans=True)
            plt.savefig(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                     f"transect/figures/sentry_nopp_box_{col}{FIGURE_NAME_ADDITION}.png"))
            plt.close()

        ros_df_1 = ros_df[ros_df.datetime <=
                          pd.Timestamp("2021-11-30 07:00:04")]
        ros_df_1.dropna(inplace=True, subset=ROSETTE_SAGE_VARS)
        ros_df_2 = ros_df[ros_df.datetime >
                          pd.Timestamp("2021-11-30 07:00:04")]
        ros_df_2.dropna(inplace=True, subset=ROSETTE_SAGE_VARS)
        for i, col in enumerate(ROSETTE_SAGE_VARS):
            # Histograms
            fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
            fig.suptitle(ROSETTE_SAGE_LABELS[i])
            ax[0].hist(ros_df_1[col].values, bins=50, density=True)
            ax[0].set_title("Leg 1")
            ax[1].hist(ros_df_2[col].values, bins=50, density=True)
            ax[1].set_title("Leg 2")
            plt.savefig(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                     f"transect/figures/rosette_sage_histo_{col}{FIGURE_NAME_ADDITION}.png"))
            plt.close()

            # Boxplots
            plt.boxplot([ros_df_1[col].values, ros_df_2[col].values], labels=[
                        "Leg 1", "Leg 2"], meanline=True, showmeans=True)
            plt.title(ROSETTE_SAGE_VARS[i])
            plt.savefig(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                     f"transect/figures/rosette_sage_box_{col}{FIGURE_NAME_ADDITION}.png"))
            plt.close()
