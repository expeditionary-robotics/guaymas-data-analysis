"""File that generates the figures included in a manuscript."""

import os
import utm
import scipy.signal
import pandas as pd
import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
from gasex import sol
from scipy.stats import stats
from itertools import combinations
from plotly.subplots import make_subplots
from transect_utils import get_transect_rosette_sage_path, \
    get_transect_sentry_nopp_path, get_transect_bottles_path, CHIMA


def extract_trends(df, x, y, fit="polyfit", inplace=True):
    """Find the trends relationship between inputs and remove."""
    if fit is "polyfit":
        z = np.polyfit(df[x].values, df[y].values, 1)
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


def compute_oriented_distance(ref_coord, traj_coord):
    """Computes an oriented distance from a reference coordinate, where orientation is aligned W-E."""
    # convert to UTM coordinates
    RX, RY, ZN, ZL = utm.from_latlon(ref_coord[0], ref_coord[1])
    TX, TY, _, _ = utm.from_latlon(
        traj_coord[0], traj_coord[1], force_zone_number=ZN, force_zone_letter=ZL)

    # determine sign
    orientation = np.sign(RX-TX)

    # compute oriented euclidean distance
    return orientation * np.sqrt((RX-TX)**2 + (RY-TY)**2)


def get_bathy(rsamp=0.5):
    """Get and window the bathy file."""
    bathy_file = os.path.join(os.getenv("SENTRY_DATA"),
                              "bathy/proc/ridge.txt")
    bathy = pd.read_table(bathy_file, names=["long", "lat", "depth"]).dropna()
    return bathy.sample(frac=rsamp, random_state=1)


def compute_global_correlation(df, df_vars, df_labels):
    """Computes global correlation factor."""
    zr = np.zeros((len(df_vars), len(df_vars)))
    zp = np.zeros((len(df_vars), len(df_vars)))
    for i, v in enumerate(df_vars):
        for j, s in enumerate(df_vars):
            r, p = stats.pearsonr(df[v].values, df[s].values)
            zr[i, j] = r
            zp[i, j] = p
            # print(f"Rvalue: {r}, Pvalue: {p}, for {v} compared to {s}")
    fig, ax = plt.subplots()
    im = ax.imshow(zr, cmap="coolwarm", vmin=-1, vmax=1)
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
    return fig, ax


def compute_local_correlation(df, df_idx, df_vars, df_labels, window):
    """Computes local correlation and saves figures.

    Input:
        df (dataframe): a 1Hz sampled dataframe of data
        df_idx (string): timestamp or datastring column name
        df_vars (list of strings): column names of targets
        df_labels (list of string): common names of targets
        window (int): number of seconds in the window
    """
    # First check that the dataframe is 1Hz sampled
    sampled_at_1hz = all(df[df_idx].diff()[1:50] ==
                         np.timedelta64(1, 's')) == True
    if not sampled_at_1hz:
        print("WARNING: data is not sampled at 1Hz for averaging purposes")
    # If properly sampled, continued with window correlation check
    pairs = [comb for comb in combinations(df_vars, 2)]

    zim = np.zeros((len(pairs), len(df[df_idx])))
    for i, pair in enumerate(pairs):
        rolling_r = df[pair[0]].rolling(
            window=window, center=True).corr(df[pair[1]])
        zim[i, :] = rolling_r
    fig = go.Heatmap(z=zim,
                     x=df[df_idx],
                     y=[f"{p[0]} vs. {p[1]}" for p in pairs],
                     colorscale="RdBu_r",
                     zmin=-1,
                     zmax=1)
    return fig


def norm(targ):
    """Return the 0-1 normalized value of a target data stream"""
    return (targ - np.nanmin(targ))/(np.nanmax(targ) - np.nanmin(targ))


# Data references
SENTRY_NOPP = get_transect_sentry_nopp_path()
BOTTLES = get_transect_bottles_path()
ROSETTE_SAGE = get_transect_rosette_sage_path()

# Data treatment
SENTRY_DEPTH_TARGETS = ["O2", "potential_temp", "practical_salinity"]
SENTRY_SMOOTH_TARGETS = ["O2", "potential_temp",
                         "obs", "fundamental_nM", "practical_salinity", ]
SENTRY_SMOOTH_LABELS = [r"Oxygen ($\mu mol \cdot kg^{-1}$)", "Potential Temperature (C)",
                        "Optical Backscatter (%)", "Methane Fundamental (nM)", "Practical Salinity (PSU)"]
SENTRY_CORR_TARGETS = ["O2", "potential_temp", "obs",
                       "fundamental_nM"]  # , "practical_salinity"]
SENTRY_CORR_LABELS = [r"Oxygen ($\mu mol \cdot kg^{-1}$)", "Potential Temperature (C)",
                      "Optical Backscatter (%)", "Methane Fundamental (nM)"]  # , "Practical Salinity (PSU)"]
ROSETTE_DEPTH_TARGETS = ["o2_umol_kg", "pot_temp_C_its90", "prac_salinity"]
ROSETTE_SMOOTH_TARGETS = ["o2_umol_kg", "pot_temp_C_its90",
                          "beam_attenuation", "sage_nM", "prac_salinity"]
ROSETTE_SMOOTH_LABELS = [r"Oxygen ($\mu mol kg^{-1}$)", "Potential Temperature (C)",
                         "Beam Attenuation (%)", "Methane (nM)", "Practical Salinity (PSU)"]
ROSETTE_CORR_TARGETS = ["o2_umol_kg", "pot_temp_C_its90",
                        "beam_attenuation", "sage_nM"]  # , "prac_salinity"]
ROSETTE_CORR_LABELS = [r"Oxygen ($\mu mol kg^{-1}$)", "Potential Temperature (C)",
                       "Beam Attenuation (%)", "Methane (nM)"]  # , "Practical Salinity (PSU)"]
SMOOTH_WINDOW = 5  # minutes
CORR_WINDOW = 30  # minutes
REG_WINDOW = 30  # minutes

# What point to use as a reference for distance
RIDGE_REFERENCE = CHIMA

# What plots to produce when running this script
BOTTLE_PLOT = False  # SAGE, CH4, and NH4 from Rosette Leg 2
TURBIDITY_PLOT = False  # Turbidity over dist, all platforms
TEMPERATURE_PLOT = False  # Temp anom over dist, all platforms
SALINITY_PLOT = False  # Salt anom over dist, all platforms
O2_PLOT = False  # O2 anom over dist, all platforms
O2_CH4_PLOT = False  # ratio of CH4 and O2 anom
CH4_PLOT = False  # CH4 over dist, all platforms
ORP_PLOT = False  # ORP over dist, Sentry
GLOBAL_CORR_PLOT = False  # global correlations, all platforms
LOCAL_CORR_PLOT = False  # local correlations, all platforms
REGIME_PLOT = True  # regime identification

if __name__ == "__main__":
    # Get all of the data
    scc_df = pd.read_csv(SENTRY_NOPP)
    scc_df['timestamp'] = pd.to_datetime(scc_df['timestamp'])
    bott_df = pd.read_csv(BOTTLES)
    bott_df['datetime'] = pd.to_datetime(bott_df['datetime'])
    ros_df = pd.read_csv(ROSETTE_SAGE)
    ros_df['datetime'] = pd.to_datetime(ros_df['datetime'])

    # Compute an oriented distance measure from ridge reference
    scc_df.loc[:, 'reference_distance'] = scc_df.apply(lambda x: compute_oriented_distance(
        RIDGE_REFERENCE, (float(x['lat']), float(x['lon'])))/1000., axis=1)
    bott_df.loc[:, 'reference_distance'] = bott_df.apply(lambda x: compute_oriented_distance(
        RIDGE_REFERENCE, (float(x['lat']), float(x['lon'])))/1000., axis=1)
    ros_df.loc[:, 'reference_distance'] = ros_df.apply(lambda x: compute_oriented_distance(
        RIDGE_REFERENCE, (float(x['usbl_lat']), float(x['usbl_lon'])))/1000., axis=1)

    # Compute nM data targets
    bott_df.loc[:, 'ch4_uatm_corr_023'] = bott_df.apply(lambda x: (
        ((x['GGA Methane'] - 1.86) / 0.023 + 1.86) * (495. / 1000.)) * 1e-6, axis=1)
    bott_df.loc[:, 'ch4_uatm_corr_033'] = bott_df.apply(lambda x: (
        ((x['GGA Methane'] - 1.86) / 0.033 + 1.86) * (495. / 1000.)) * 1e-6, axis=1)
    bott_df["ch4_nM_corr_023"] = bott_df.apply(lambda x: sol.sol_SP_pt(
        x["prac_salinity"], x["pot_temp_C_its90"], gas='CH4', p_dry=x["ch4_uatm_corr_023"], units='mM') * 1e6, axis=1)
    bott_df["ch4_nM_corr_033"] = bott_df.apply(lambda x: sol.sol_SP_pt(
        x["prac_salinity"], x["pot_temp_C_its90"], gas='CH4', p_dry=x["ch4_uatm_corr_033"], units='mM') * 1e6, axis=1)

    ros_df.loc[:, 'sage_nM'] = ros_df.apply(lambda x: sol.sol_SP_pt(
        x["prac_salinity"], x["pot_temp_C_its90"], gas='CH4', p_dry=x["sage_methane_ppm"] * 1e-6, units='mM') * 1e6, axis=1)

    # Drop bad values
    ros_df.dropna(subset=ROSETTE_SMOOTH_TARGETS, inplace=True)
    scc_df.dropna(subset=SENTRY_SMOOTH_TARGETS, inplace=True)

    # Depth-Correct Data
    for depth_target in SENTRY_DEPTH_TARGETS:
        scc_df = extract_trends(scc_df, 'depth', depth_target)
    for depth_target in ROSETTE_DEPTH_TARGETS:
        ros_df = extract_trends(ros_df, 'depth_m', depth_target)
    ros_df_1 = ros_df[ros_df.datetime <=
                      pd.Timestamp("2021-11-30 07:00:04")]
    ros_df_2 = ros_df[ros_df.datetime >
                      pd.Timestamp("2021-11-30 07:00:04")]

    # Create global correlation plots
    if GLOBAL_CORR_PLOT is True:
        # Global Pearson Correlation
        scc_df.dropna(subset=SENTRY_CORR_TARGETS, inplace=True)
        trunc_scc_df = scc_df[scc_df.timestamp >=
                              pd.Timestamp("2021-11-30 05:37:00")]
        scc_fig, scc_ax = compute_global_correlation(
            trunc_scc_df, SENTRY_CORR_TARGETS, SENTRY_CORR_LABELS)
        scc_fig.savefig(os.path.join(os.getenv("SENTRY_OUTPUT"),
                        f"transect/figures/paper/sentry_global_corr_raw.svg"))
        plt.close()
        ros_df.dropna(subset=ROSETTE_CORR_TARGETS, inplace=True)
        ros_fig, ros_ax = compute_global_correlation(
            ros_df, ROSETTE_CORR_TARGETS, ROSETTE_CORR_LABELS)
        ros_fig.savefig(os.path.join(os.getenv("SENTRY_OUTPUT"),
                        f"transect/figures/paper/rosette_global_corr_raw.svg"))
        plt.close()
        ros_df_1.dropna(subset=ROSETTE_CORR_TARGETS, inplace=True)
        ros_fig, ros_ax = compute_global_correlation(
            ros_df_1, ROSETTE_CORR_TARGETS, ROSETTE_CORR_LABELS)
        ros_fig.savefig(os.path.join(os.getenv("SENTRY_OUTPUT"),
                        f"transect/figures/paper/rosette1_global_corr_raw.svg"))
        plt.close()
        ros_df_2.dropna(subset=ROSETTE_CORR_TARGETS, inplace=True)
        ros_fig, ros_ax = compute_global_correlation(
            ros_df_2, ROSETTE_CORR_TARGETS, ROSETTE_CORR_LABELS)
        ros_fig.savefig(os.path.join(os.getenv("SENTRY_OUTPUT"),
                        f"transect/figures/paper/rosette2_global_corr_raw.svg"))
        plt.close()

    # Create local correlation plots
    if LOCAL_CORR_PLOT is True:
        r_window_size = int(60 * CORR_WINDOW)  # seconds
        scc_fig = compute_local_correlation(scc_df, "timestamp", SENTRY_CORR_TARGETS,
                                                    SENTRY_CORR_LABELS, r_window_size)
        ros_fig = compute_local_correlation(ros_df, "datetime", ROSETTE_CORR_TARGETS,
                                            ROSETTE_CORR_LABELS, r_window_size)
        im = go.Figure(data=[scc_fig])
        im.update_layout(template="plotly",
                         xaxis=dict(tickfont=dict(size=20),
                                    title="Time of Measurement",
                                    titlefont=dict(size=25)),
                         yaxis=dict(tickfont=dict(size=20)),
                         showlegend=False)
        im.write_image(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                    f"transect/figures/paper/sentry_local_corr_raw.svg"), width=1500, height=750)

        im2 = go.Figure(data=[ros_fig])
        im2.update_layout(template="plotly",
                          xaxis=dict(tickfont=dict(size=20),
                                     title="Time of Measurement",
                                     titlefont=dict(size=25)),
                          yaxis=dict(tickfont=dict(size=20)),
                          showlegend=False)
        im2.write_image(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                     f"transect/figures/paper/rosette_local_corr_raw.svg"), width=1500)

    # Smooth Data
    smooth_data(scc_df, SENTRY_SMOOTH_TARGETS, smooth_window=SMOOTH_WINDOW)
    smooth_data(ros_df_1, ROSETTE_SMOOTH_TARGETS, smooth_window=SMOOTH_WINDOW)
    smooth_data(ros_df_2, ROSETTE_SMOOTH_TARGETS, smooth_window=SMOOTH_WINDOW)
    ros_df = ros_df_1.append(ros_df_2)

    # Create a bathy plot
    bathy = get_bathy(rsamp=0.02)
    bathy_plot = go.Scatter(x=bathy.long,
                            y=bathy.lat,
                            mode="markers",
                            marker=dict(size=8,
                                        color=bathy.depth,
                                        opacity=0.8,
                                        colorscale="Viridis",
                                        colorbar=dict(thickness=10, x=-0.2)),
                            name="Bathy")

    # Create location star
    reference_plot = go.Scatter(x=[RIDGE_REFERENCE[1]],
                                y=[RIDGE_REFERENCE[0]],
                                mode="markers",
                                marker=dict(size=10,
                                            color="red",
                                            symbol="star"),
                                name="Reference Point")

    # Create bottle plot
    if BOTTLE_PLOT is True:
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax2 = ax1.twinx()
        sage = ax1.scatter(ros_df_2['reference_distance'],
                           ros_df_2['sage_nM'], label="SAGE")
        sage_fill = ax1.fill_between(
            ros_df_2['reference_distance'], y1=ros_df_2['sage_nM']*0.5, y2=ros_df_2['sage_nM']*1.5, opacity=0.1)
        gga = ax1.vlines(bott_df['reference_distance'], ymin=bott_df['ch4_nM_corr_033'],
                         ymax=bott_df['ch4_nM_corr_023'], label="Bottle Methane", color="gray")
        amm = ax2.scatter(bott_df['reference_distance'], bott_df["[NH4] (nM)"],
                          label="Bottle Ammonium", color="orange", s=100)

        ax1.set_xlabel("Distance from Reference (km)", size=20)
        ax1.set_ylabel("Methane (nM)", color="blue", size=20)
        ax1.tick_params(axis='y', labelcolor="blue", labelsize=15)
        ax1.tick_params(axis='x', labelsize=15)
        ax2.set_ylabel("Ammonium (nM)", color="orange", size=20)
        ax2.tick_params(axis='y', labelcolor="orange", labelsize=15)
        lns = [sage, gga, amm]
        labs = [x.get_label() for x in lns]
        ax1.legend(lns, labs, loc=0)
        fig.tight_layout()
        pltname = os.path.join(os.getenv("SENTRY_OUTPUT"),
                               f"transect/figures/paper/sage_ch4_nh4_raw.svg")
        plt.savefig(pltname)
        plt.close()

        # for the ratios, find corresponding SAGE value
        bott_df.loc[:, "sage_reference"] = bott_df.apply(lambda x: ros_df_2.iloc[(
            ros_df_2['usbl_lon']-x['lon']).abs().argsort()]["sage_nM"].values[0], axis=1)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(bott_df["reference_distance"], bott_df["sage_reference"] /
                   bott_df["[NH4] (nM)"], label="SAGE", color="orange", s=100)
        ax.vlines(bott_df["reference_distance"], ymin=bott_df["ch4_nM_corr_033"] /
                  bott_df["[NH4] (nM)"], ymax=bott_df["ch4_nM_corr_023"]/bott_df["[NH4] (nM)"], label="Bottles")
        ax.set_xlabel("Distance from Reference (km)", size=20)
        ax.tick_params(axis='x', labelsize=15)
        ax.set_ylabel(r"Ratio $CH_4$/$NH_4^{+}$", size=20)
        ax.tick_params(axis='y', labelsize=15)
        plt.legend()
        fig.tight_layout()
        pltname = os.path.join(os.getenv("SENTRY_OUTPUT"),
                               f"transect/figures/paper/ch4_nh4_ratio_raw.svg")
        plt.savefig(pltname)
        plt.close()

    # Normalize methane values
    scc_df["fundamental_nM"] = norm(scc_df["fundamental_nM"])
    ros_df["sage_nM"] = norm(ros_df["sage_nM"])

    # Create Turbidity plot
    if TURBIDITY_PLOT is True:
        scc_df.dropna(subset=['obs'], inplace=True)
        ros_df.dropna(subset=["beam_attenuation"], inplace=True)
        scc_df = scc_df[scc_df.timestamp >=
                        pd.Timestamp("2021-11-30 05:37:00")]
        scc_thresh = scc_df[norm(scc_df['obs']) > 0.5]
        ros_thresh = ros_df[norm(ros_df['beam_attenuation']) > 0.5]
        scc_plot = go.Scatter(x=scc_df['reference_distance'],
                              y=-scc_df['depth'],
                              mode="markers",
                              marker=dict(size=5,
                                          color=norm(scc_df['obs']),
                                          colorscale="Inferno",
                                          colorbar=dict(
                                              thickness=20, x=-0.2, tickfont=dict(size=20)),
                                          cmin=0,
                                          cmax=1,),
                              name="Sentry OBS")
        ros_plot = go.Scatter(x=ros_df['reference_distance'],
                              y=-ros_df['depth_m'],
                              mode="markers",
                              marker=dict(size=5,
                                          color=norm(
                                              ros_df['beam_attenuation']),
                                          colorscale="Inferno",
                                          colorbar=dict(
                                              thickness=20, x=-0.2, tickfont=dict(size=20)),
                                          cmin=0,
                                          cmax=1,
                                          showscale=False),
                              name="Rosette Beam Attenuation")
        thresh_plot = go.Scatter(x=[scc_thresh['reference_distance'].values[0], ros_thresh['reference_distance'].values[0]],
                                 y=[-scc_thresh['depth'].values[0], -
                                     ros_thresh['depth_m'].values[0]],
                                 mode="markers",
                                 marker=dict(size=25,
                                             color='rgba(0, 250, 0, 0)',
                                             symbol="circle",
                                             line=dict(color="green", width=2)),
                                 name="Threshold Turbidity")
        fig = go.Figure(data=[scc_plot, ros_plot, thresh_plot])
        fig.update_layout(template="plotly",
                          xaxis=dict(tickmode='linear',
                                     dtick=1,
                                     tickfont=dict(size=20),
                                     title="Distance from Reference (km)",
                                     titlefont=dict(size=25)),
                          yaxis=dict(tickmode='linear',
                                     dtick=50,
                                     tickfont=dict(size=20),
                                     title="Depth (m)",
                                     titlefont=dict(size=25)),
                          showlegend=False)
        fig.write_image(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                     f"transect/figures/paper/turbidity_raw.svg"), width=1500)

        scc_plot = go.Scatter(x=scc_df['reference_distance'],
                              y=scc_df['obs'],
                              mode="markers",
                              marker=dict(color="green"),
                              name="Sentry OBS")
        ros_plot = go.Scatter(x=ros_df['reference_distance'],
                              y=ros_df['beam_attenuation'],
                              mode="markers",
                              marker=dict(color="brown"),
                              name="Rosette Beam Attenuation")
        fig = go.Figure(data=[scc_plot, ros_plot])
        fig.update_layout(template="plotly",
                          xaxis=dict(tickmode='linear',
                                     dtick=1,
                                     tickfont=dict(size=20),
                                     title="Distance from Reference (km)",
                                     titlefont=dict(size=25)),
                          yaxis=dict(tickfont=dict(size=20),
                                     title="Beam Attenuation",
                                     titlefont=dict(size=25)),
                          showlegend=True)
        fig.write_image(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                     f"transect/figures/paper/turbidity_simple_raw.svg"), width=1500)

    # Create Temperature plot
    if TEMPERATURE_PLOT is True:
        scc_plot = go.Scatter(x=scc_df['reference_distance'],
                              y=-scc_df['depth'],
                              mode="markers",
                              marker=dict(size=5,
                                          color=scc_df['potential_temp'],
                                          colorscale="rdbu_r",
                                          colorbar=dict(
                                              thickness=20, x=-0.2, tickfont=dict(size=20)),
                                          cmid=0.0,),
                              name="Sentry Temperature Anomaly")
        ros_plot = go.Scatter(x=ros_df['reference_distance'],
                              y=-ros_df['depth_m'],
                              mode="markers",
                              marker=dict(size=5,
                                          color=ros_df['pot_temp_C_its90'],
                                          colorscale="rdbu_r",
                                          colorbar=dict(
                                              thickness=20, x=-0.2, tickfont=dict(size=20)),
                                          cmid=0.0,
                                          showscale=False),
                              name="Rosette Temperature Anomaly")
        fig = go.Figure(data=[scc_plot, ros_plot])
        fig.update_layout(template="plotly",
                          xaxis=dict(tickmode='linear',
                                     dtick=1,
                                     tickfont=dict(size=20),
                                     title="Distance from Reference (km)",
                                     titlefont=dict(size=25)),
                          yaxis=dict(tickmode='linear',
                                     dtick=50,
                                     tickfont=dict(size=20),
                                     title="Depth (m)",
                                     titlefont=dict(size=25)),
                          showlegend=False)
        fig.write_image(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                     f"transect/figures/paper/temperature_raw.svg"), width=1500)

        scc_plot = go.Scatter(x=scc_df['reference_distance'],
                              y=scc_df['potential_temp'],
                              mode="markers",
                              marker=dict(color="green"),
                              name="Sentry")
        ros_plot = go.Scatter(x=ros_df['reference_distance'],
                              y=ros_df['pot_temp_C_its90'],
                              mode="markers",
                              marker=dict(color="brown"),
                              name="Rosette")
        fig = go.Figure(data=[scc_plot, ros_plot])
        fig.update_layout(template="plotly",
                          xaxis=dict(tickmode='linear',
                                     dtick=1,
                                     tickfont=dict(size=20),
                                     title="Distance from Reference (km)",
                                     titlefont=dict(size=25)),
                          yaxis=dict(tickfont=dict(size=20),
                                     title="Potential Temperature Anomaly (C)",
                                     titlefont=dict(size=25)),
                          showlegend=True)
        fig.write_image(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                     f"transect/figures/paper/temperature_simple_raw.svg"), width=1500)

    # Create Salinity plot
    if SALINITY_PLOT is True:
        scc_plot = go.Scatter(x=scc_df['reference_distance'],
                              y=-scc_df['depth'],
                              mode="markers",
                              marker=dict(size=5,
                                          color=scc_df['practical_salinity'],
                                          colorscale="rdbu_r",
                                          colorbar=dict(
                                              thickness=20, x=-0.2, tickfont=dict(size=20)),
                                          cmid=0.0,),
                              name="Sentry Salinity Anomaly")
        ros_plot = go.Scatter(x=ros_df['reference_distance'],
                              y=-ros_df['depth_m'],
                              mode="markers",
                              marker=dict(size=5,
                                          color=ros_df['prac_salinity'],
                                          colorscale="rdbu_r",
                                          colorbar=dict(
                                              thickness=20, x=-0.2, tickfont=dict(size=20)),
                                          cmid=0.0,
                                          showscale=False),
                              name="Rosette Salinity Anomaly")
        fig = go.Figure(data=[scc_plot, ros_plot])
        fig.update_layout(template="plotly",
                          xaxis=dict(tickmode='linear',
                                     dtick=1,
                                     tickfont=dict(size=20),
                                     title="Distance from Reference (km)",
                                     titlefont=dict(size=25)),
                          yaxis=dict(tickmode='linear',
                                     dtick=50,
                                     tickfont=dict(size=20),
                                     title="Depth (m)",
                                     titlefont=dict(size=25)),
                          showlegend=False)
        fig.write_image(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                     f"transect/figures/paper/salinity_raw.svg"), width=1500)

        scc_plot = go.Scatter(x=scc_df['reference_distance'],
                              y=scc_df['practical_salinity'],
                              mode="markers",
                              marker=dict(color="green"),
                              name="Sentry")
        ros_plot = go.Scatter(x=ros_df['reference_distance'],
                              y=ros_df['prac_salinity'],
                              mode="markers",
                              marker=dict(color="brown"),
                              name="Rosette")
        fig = go.Figure(data=[scc_plot, ros_plot])
        fig.update_layout(template="plotly",
                          xaxis=dict(tickmode='linear',
                                     dtick=1,
                                     tickfont=dict(size=20),
                                     title="Distance from Reference (km)",
                                     titlefont=dict(size=25)),
                          yaxis=dict(tickfont=dict(size=20),
                                     title="Practical Salinity Anomaly (PSU)",
                                     titlefont=dict(size=25)),
                          showlegend=True)
        fig.write_image(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                     f"transect/figures/paper/salinity_simple_raw.svg"), width=1500)

    # Create O2 plot
    if O2_PLOT is True:
        scc_plot = go.Scatter(x=scc_df['reference_distance'],
                              y=-scc_df['depth'],
                              mode="markers",
                              marker=dict(size=5,
                                          color=scc_df['O2'],
                                          colorscale="rdbu_r",
                                          colorbar=dict(
                                              thickness=20, x=-0.2, tickfont=dict(size=20)),
                                          cmid=0.0,),
                              name="Sentry O2 Anomaly")
        ros_plot = go.Scatter(x=ros_df['reference_distance'],
                              y=-ros_df['depth_m'],
                              mode="markers",
                              marker=dict(size=5,
                                          color=ros_df['o2_umol_kg'],
                                          colorscale="rdbu_r",
                                          colorbar=dict(
                                              thickness=20, x=-0.2, tickfont=dict(size=20)),
                                          cmid=0.0,
                                          showscale=False),
                              name="Rosette O2 Anomaly")
        fig = go.Figure(data=[scc_plot, ros_plot])
        fig.update_layout(template="plotly",
                          xaxis=dict(tickmode='linear',
                                     dtick=1,
                                     tickfont=dict(size=20),
                                     title="Distance from Reference (km)",
                                     titlefont=dict(size=25)),
                          yaxis=dict(tickmode='linear',
                                     dtick=50,
                                     tickfont=dict(size=20),
                                     title="Depth (m)",
                                     titlefont=dict(size=25)),
                          showlegend=False)
        fig.write_image(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                     f"transect/figures/paper/o2_raw.svg"), width=1500)

        scc_plot = go.Scatter(x=scc_df['reference_distance'],
                              y=scc_df['O2'],
                              mode="markers",
                              marker=dict(color="green"),
                              name="Sentry")
        ros_plot = go.Scatter(x=ros_df['reference_distance'],
                              y=ros_df['o2_umol_kg'],
                              mode="markers",
                              marker=dict(color="brown"),
                              name="Rosette")
        fig = go.Figure(data=[scc_plot, ros_plot])
        fig.update_layout(template="plotly",
                          xaxis=dict(tickmode='linear',
                                     dtick=1,
                                     tickfont=dict(size=20),
                                     title="Distance from Reference (km)",
                                     titlefont=dict(size=25)),
                          yaxis=dict(tickfont=dict(size=20),
                                     title=r"Oxygen ($\mu mol \cdot kg^{-1}$)",
                                     titlefont=dict(size=25)),
                          showlegend=True)
        fig.write_image(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                     f"transect/figures/paper/o2_simple_raw.svg"), width=1500)

    # Create O2/CH4 plot
    if O2_CH4_PLOT is True:
        scc_df.dropna(subset=["fundamental_nM", 'O2'], inplace=True)
        ros_df.dropna(subset=["sage_nM", "o2_umol_kg"], inplace=True)
        cmin = np.percentile(np.concatenate(
            (np.fabs(scc_df['fundamental_nM'].values/scc_df['O2'].values), np.fabs(ros_df['sage_nM'].values/ros_df['o2_umol_kg'].values)), axis=None), 10)
        cmax = np.percentile(np.concatenate(
            (np.fabs(scc_df['fundamental_nM'].values/scc_df['O2'].values), np.fabs(ros_df['sage_nM'].values/ros_df['o2_umol_kg'].values)), axis=None), 90)
        scc_plot = go.Scatter(x=scc_df['reference_distance'],
                              y=-scc_df['depth'],
                              mode="markers",
                              marker=dict(size=5,
                                          color=np.fabs(
                                              scc_df['fundamental_nM']/scc_df['O2']),
                                          colorscale="Inferno",
                                          colorbar=dict(
                                              thickness=20, x=-0.2, tickfont=dict(size=20)),
                                          cmin=cmin,
                                          cmax=cmax,),
                              name="Sentry CH4/O2 Ratio")
        ros_plot = go.Scatter(x=ros_df['reference_distance'],
                              y=-ros_df['depth_m'],
                              mode="markers",
                              marker=dict(size=5,
                                          color=np.fabs(
                                              ros_df['sage_nM']/ros_df['o2_umol_kg']),
                                          colorscale="Inferno",
                                          colorbar=dict(
                                              thickness=20, x=-0.2, tickfont=dict(size=20)),
                                          cmin=cmin,
                                          cmax=cmax,
                                          showscale=False),
                              name="Rosette CH4/O2 Ratio")
        fig = go.Figure(data=[scc_plot, ros_plot])
        fig.update_layout(template="plotly",
                          xaxis=dict(tickmode='linear',
                                     dtick=1,
                                     tickfont=dict(size=20),
                                     title="Distance from Reference (km)",
                                     titlefont=dict(size=25)),
                          yaxis=dict(tickmode='linear',
                                     dtick=50,
                                     tickfont=dict(size=20),
                                     title="Depth (m)",
                                     titlefont=dict(size=25)),
                          showlegend=False)
        fig.write_image(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                     f"transect/figures/paper/ch4_o2_raw.svg"), width=1500)

    # Create CH4 plot
    if CH4_PLOT is True:
        scc_df.dropna(subset=["fundamental_nM", 'O2'], inplace=True)
        ros_df.dropna(subset=["sage_nM", "o2_umol_kg"], inplace=True)
        scc_base = np.nanmax(scc_df['fundamental_nM'].values)
        ros_base = np.nanmax(ros_df['sage_nM'].values)
        scc_thresh = scc_df[norm(scc_df['fundamental_nM']) > 0.5]
        ros_thresh = ros_df[norm(ros_df['sage_nM']) > 0.5]
        scc_plot = go.Scatter(x=scc_df['reference_distance'],
                              y=-scc_df['depth'],
                              mode="markers",
                              marker=dict(size=5,
                                          color=norm(scc_df['fundamental_nM']),
                                          colorscale="Inferno",
                                          colorbar=dict(
                                              thickness=20, x=-0.2, tickfont=dict(size=20)),
                                          cmin=0,
                                          cmax=1),
                              name="Sentry Methane")
        ros_plot = go.Scatter(x=ros_df['reference_distance'],
                              y=-ros_df['depth_m'],
                              mode="markers",
                              marker=dict(size=5,
                                          color=norm(ros_df['sage_nM']),
                                          colorscale="Inferno",
                                          colorbar=dict(
                                              thickness=20, x=-0.2, tickfont=dict(size=20)),
                                          cmin=0,
                                          cmax=1,
                                          showscale=False),
                              name="Rosette Methane")
        thresh_plot = go.Scatter(x=[scc_thresh['reference_distance'].values[0], ros_thresh['reference_distance'].values[0]],
                                 y=[-scc_thresh['depth'].values[0], -
                                     ros_thresh['depth_m'].values[0]],
                                 mode="markers",
                                 marker=dict(size=25,
                                             color='rgba(0, 250, 0, 0)',
                                             symbol="circle",
                                             line=dict(color="green", width=2)),
                                 name="Threshold Methane")
        fig = go.Figure(data=[scc_plot, ros_plot, thresh_plot])
        fig.update_layout(template="plotly",
                          xaxis_title="Distance from Reference (km)",
                          yaxis_title="Depth (m)",
                          xaxis=dict(tickmode='linear',
                                     dtick=1,
                                     tickfont=dict(size=20),
                                     titlefont=dict(size=25)),
                          yaxis=dict(tickmode='linear',
                                     dtick=50,
                                     tickfont=dict(size=20),
                                     titlefont=dict(size=25)),
                          showlegend=False,
                          coloraxis_colorbar=dict(title="", tickprefix='1.e'))
        fig.write_image(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                     f"transect/figures/paper/methane_fundamental_raw.svg"), width=1500)

    # Create ORP plot
    if ORP_PLOT is True:
        scc_sort = scc_df.sort_values(by="dorpdt", ascending=False)
        scc_plot = go.Scatter(x=scc_sort['reference_distance'],
                              y=-scc_sort['depth'],
                              mode="markers",
                              marker=dict(size=5,
                                          color=scc_sort['dorpdt'],
                                          colorscale="Inferno",
                                          colorbar=dict(
                                              thickness=20, x=-0.2, tickfont=dict(size=20))),
                              name="Sentry dORPdt")
        fig = go.Figure(data=[scc_plot])
        fig.update_layout(template="plotly",
                          xaxis=dict(tickmode='linear',
                                     dtick=1,
                                     tickfont=dict(size=20),
                                     title="Distance from Reference (km)",
                                     titlefont=dict(size=25)),
                          yaxis=dict(tickmode='linear',
                                     dtick=50,
                                     tickfont=dict(size=20),
                                     title="Depth (m)",
                                     titlefont=dict(size=25)),
                          showlegend=False)
        fig.write_image(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                     f"transect/figures/paper/orp_raw.svg"), width=1500)

        scc_plot = go.Scatter(x=scc_df['reference_distance'],
                              y=scc_df['dorpdt'],
                              mode="markers",
                              marker=dict(color="green"),
                              name="Sentry")
        fig = go.Figure(data=[scc_plot])
        fig.update_layout(template="plotly",
                          xaxis=dict(tickmode='linear',
                                     dtick=1,
                                     tickfont=dict(size=20),
                                     title="Distance from Reference (km)",
                                     titlefont=dict(size=25)),
                          yaxis=dict(tickfont=dict(size=20),
                                     title="dORPdt",
                                     titlefont=dict(size=25)),
                          showlegend=False)
        fig.write_image(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                     f"transect/figures/paper/orp_simple_raw.svg"), width=1500)

    # Create regime plots
    if REGIME_PLOT is True:
        SENTRY_REG_TARGETS = SENTRY_SMOOTH_TARGETS + ["dorpdt"]
        ROSETTE_REG_TARGETS = ROSETTE_SMOOTH_TARGETS
        # based on https://techrando.com/2019/08/14/a-brief-introduction-to-change-point-detection-using-python/
        scc_df.dropna(subset=SENTRY_REG_TARGETS, inplace=True)
        ros_df.dropna(subset=ROSETTE_REG_TARGETS, inplace=True)
        ros_df_1.dropna(subset=ROSETTE_REG_TARGETS, inplace=True)
        ros_df_2.dropna(subset=ROSETTE_REG_TARGETS, inplace=True)
        linecolors = px.colors.qualitative.Safe
        model = "rbf"

        fig = make_subplots(rows=len(SENTRY_REG_TARGETS), cols=1, shared_xaxes=True,
                            subplot_titles=tuple(SENTRY_REG_TARGETS))
        for i, col in enumerate(SENTRY_REG_TARGETS):
            tar_df = scc_df
            if col is "obs":
                tar_df = scc_df[scc_df.timestamp >=
                                pd.Timestamp("2021-11-30 05:37:00")]
            algo = rpt.Pelt(model=model, jump=1, min_size=REG_WINDOW *
                            60/60).fit(tar_df[col].values[::60])
            result = [1]
            result = result + algo.predict(pen=10)
            scat = go.Scatter(y=tar_df[col],
                              x=[pd.Timestamp(m)
                                 for m in tar_df['timestamp'].values],
                              mode="markers")
            ref_index = tar_df['timestamp'].values[::60]
            colors = ["red", "blue"]
            fig.add_trace(scat, row=i+1, col=1)
            for j, (a, b), in enumerate(zip(result, result[1:])):
                fig.add_vrect(
                    x0=pd.Timestamp(ref_index[a-1]),
                    x1=pd.Timestamp(ref_index[b-1]),
                    fillcolor=colors[j % 2],
                    opacity=0.05,
                    layer="below",
                    row=i+1, col=1)
        fig.update_layout(showlegend=False)
        fig.write_image(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                     f"transect/figures/paper/sentry_regimes_raw.svg"), width=1500, height=1500)

        fig = make_subplots(rows=len(ROSETTE_REG_TARGETS), cols=1, shared_xaxes='all',
                            shared_yaxes='rows', row_titles=ROSETTE_REG_TARGETS)
        for i, col in enumerate(ROSETTE_REG_TARGETS):
            algo1 = rpt.Pelt(model='rbf', jump=1, min_size=REG_WINDOW *
                             60/60).fit(ros_df[col].values[::60])
            result1 = [1]
            result1 = result1 + algo1.predict(pen=20)
            scat1 = go.Scatter(y=ros_df[col],
                               x=[pd.Timestamp(m)
                                  for m in ros_df['datetime'].values],
                               mode="markers",
                               marker=dict(color=linecolors[i]))
            ref_index1 = ros_df['datetime'].values[::60]
            colors = ["red", "blue"]
            fig.add_trace(scat1, row=i+1, col=1)
            for j, (a, b), in enumerate(zip(result1, result1[1:])):
                fig.add_vrect(
                    x0=pd.Timestamp(ref_index1[a-1]),
                    x1=pd.Timestamp(ref_index1[b-1]),
                    fillcolor=colors[j % 2],
                    opacity=0.05,
                    layer="below",
                    row=i+1, col=1)

        fig.update_layout(showlegend=False)
        fig.show()
        fig.write_image(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                     f"transect/figures/paper/rosette_regimes_raw.svg"), width=1500, height=1500)
