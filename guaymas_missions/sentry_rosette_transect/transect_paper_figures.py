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
from scipy.io import loadmat
from itertools import combinations
from plotly.subplots import make_subplots
from transect_utils import get_transect_rosette_sage_path, \
    get_transect_sentry_pythia_path, get_transect_bottles_path, CHIMA


def extract_trends(df, x, y_list, fit="polyfit", inplace=True, plot=False):
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
            ax[i].set_ylabel(y)
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


def compute_oriented_distance(ref_coord, traj_coord):
    """Computes an oriented distance from a reference coordinate, where orientation is aligned E-W."""
    # convert to UTM coordinates
    RX, RY, ZN, ZL = utm.from_latlon(ref_coord[0], ref_coord[1])
    TX, TY, _, _ = utm.from_latlon(
        traj_coord[0], traj_coord[1], force_zone_number=ZN, force_zone_letter=ZL)

    # determine sign
    orientation = np.sign(RX-TX)

    # compute oriented euclidean distance
    return orientation * np.sqrt((RX-TX)**2 + (RY-TY)**2)


def compute_global_correlation(df, df_vars, df_labels):
    """Computes global correlation factor."""
    zr = np.zeros((len(df_vars), len(df_vars)))
    zp = np.zeros((len(df_vars), len(df_vars)))
    for i, v in enumerate(df_vars):
        for j, s in enumerate(df_vars):
            r, p = stats.pearsonr(df[v].values, df[s].values)
            zr[i, j] = r
            zp[i, j] = p
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
    """Return the 0-1 normalized value of a target data stream."""
    return (targ - np.nanmin(targ))/(np.nanmax(targ) - np.nanmin(targ))


def _dens0(S, T):
    """Density of seawater at zero pressure.
    As implemented in https://github.com/bjornaa/seawater."""

    # --- Define constants ---
    a0 = 999.842594
    a1 = 6.793952e-2
    a2 = -9.095290e-3
    a3 = 1.001685e-4
    a4 = -1.120083e-6
    a5 = 6.536332e-9

    b0 = 8.24493e-1
    b1 = -4.0899e-3
    b2 = 7.6438e-5
    b3 = -8.2467e-7
    b4 = 5.3875e-9

    c0 = -5.72466e-3
    c1 = 1.0227e-4
    c2 = -1.6546e-6

    d0 = 4.8314e-4

    # --- Computations ---
    # Density of pure water
    SMOW = a0 + (a1 + (a2 + (a3 + (a4 + a5*T)*T)*T)*T)*T

    # More temperature polynomials
    RB = b0 + (b1 + (b2 + (b3 + b4*T)*T)*T)*T
    RC = c0 + (c1 + c2*T)*T

    return SMOW + RB*S + RC*(S**1.5) + d0*S*S


def _seck(S, T, P=0):
    """Secant bulk modulus.
    As implemented in https://github.com/bjornaa/seawater."""

    # --- Pure water terms ---

    h0 = 3.239908
    h1 = 1.43713E-3
    h2 = 1.16092E-4
    h3 = -5.77905E-7
    AW = h0 + (h1 + (h2 + h3*T)*T)*T

    k0 = 8.50935E-5
    k1 = -6.12293E-6
    k2 = 5.2787E-8
    BW = k0 + (k1 + k2*T)*T

    e0 = 19652.21
    e1 = 148.4206
    e2 = -2.327105
    e3 = 1.360477E-2
    e4 = -5.155288E-5
    KW = e0 + (e1 + (e2 + (e3 + e4*T)*T)*T)*T

    # --- seawater, P = 0 ---

    SR = S**0.5

    i0 = 2.2838E-3
    i1 = -1.0981E-5
    i2 = -1.6078E-6
    j0 = 1.91075E-4
    A = AW + (i0 + (i1 + i2*T)*T + j0*SR)*S

    f0 = 54.6746
    f1 = -0.603459
    f2 = 1.09987E-2
    f3 = -6.1670E-5
    g0 = 7.944E-2
    g1 = 1.6483E-2
    g2 = -5.3009E-4
    K0 = KW + (f0 + (f1 + (f2 + f3*T)*T)*T
               + (g0 + (g1 + g2*T)*T)*SR)*S

    # --- General expression ---

    m0 = -9.9348E-7
    m1 = 2.0816E-8
    m2 = 9.1697E-10
    B = BW + (m0 + (m1 + m2*T)*T)*S

    K = K0 + (A + B*P)*P

    return K


def dens(S, T, P=0):
    """Compute density of seawater from salinity, temperature, and pressure.
    As implemented in https://github.com/bjornaa/seawater.
    Usage: dens(S, T, [P])
    Input:
        S = Salinity,     [PSS-78]
        T = Temperature,  [C]
        P = Pressure,     [dbar = 10**4 Pa]
    P is optional, with default value zero
    Output:
        Density,          [kg/m**3]
    Algorithm: UNESCO 1983
    """

    P = 0.1*P  # Convert to bar
    return _dens0(S, T)/(1 - P/_seck(S, T, P))


def add_o2(df):
    """Inserts optode data collected by Sentry and unreported by summary Sentry file."""
    other_df = loadmat(os.path.join(os.getenv(
        "SENTRY_DATA"), "missions/transect/sentry613_20211202_2143_optode_renav.mat"))
    other_mdata = other_df["renav_optode"]
    keys_of_interest = ["t", "concentration"]
    other_ndata = {n: other_mdata[n][0, 0].flatten()
                   for n in keys_of_interest}
    other_df_subset = pd.DataFrame(other_ndata)
    other_df_subset["t"] = other_df_subset["t"].astype(np.int32)
    other_df_subset = other_df_subset.drop_duplicates(subset="t")

    df = df.merge(other_df_subset[["t", "concentration"]], how="left", on="t")
    df.concentration = df.concentration.interpolate()  # uM
    df.concentration = df.concentration / \
        (dens(df["ctd_sal"], df["ctd_temp"], df["ctd_pres"])/1000)  # convert to umol/kg
    return df


# Data references
SENTRY_PYTHIA = get_transect_sentry_pythia_path()
BOTTLES = get_transect_bottles_path()
ROSETTE_SAGE = get_transect_rosette_sage_path()

# Data treatment
# Depth targets are to be depth-corrected
# Smooth targets are to be smoothed with rolling-average windows for visualization
# Corr targets are to be considered for cross-correlation and regime analysis
SENTRY_DEPTH_TARGETS = ["O2", "potential_temp", "ctd_sal"]
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
# Plots will save directly to file
DEPTH_PLOT = False  # to produce depth-corrected plots
BOTTLE_PLOT = False  # SAGE, CH4, and NH4 from Rosette Leg 2
TURBIDITY_PLOT = True  # Normalized turbidity over dist, all platforms
TEMPERATURE_PLOT = False  # Temp anom over dist, all platforms
SALINITY_PLOT = False  # Salt anom over dist, all platforms
O2_PLOT = False  # O2 anom over dist, all platforms
CH4_PLOT = False  # normalized CH4 over dist, all platforms
ORP_PLOT = False  # ORP over dist, Sentry
GLOBAL_CORR_PLOT = False  # global correlations, all platforms
LOCAL_CORR_PLOT = False  # local correlations, all platforms
REGIME_PLOT = False  # regime identification, all platforms

if __name__ == "__main__":
    # Get all of the data
    scc_df = pd.read_csv(SENTRY_PYTHIA)
    scc_df['timestamp'] = pd.to_datetime(scc_df['timestamp'])
    scc_df = add_o2(scc_df)  # add the optode signal
    scc_df["O2"] = scc_df["concentration"]  # rename the optode data column
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

    # Compute solubility scaled SAGE data
    ros_df.loc[:, 'sage_nM'] = ros_df.apply(lambda x: sol.sol_SP_pt(
        x["prac_salinity"], x["pot_temp_C_its90"], gas='CH4', p_dry=x["sage_methane_ppm"] * 1e-6, units='mM') * 1e6, axis=1)

    # Drop bad values
    ros_df.dropna(subset=ROSETTE_SMOOTH_TARGETS, inplace=True)
    scc_df.dropna(subset=SENTRY_SMOOTH_TARGETS, inplace=True)

    # Depth-Correct Data
    scc_df = extract_trends(
        scc_df, 'depth', SENTRY_DEPTH_TARGETS, plot=DEPTH_PLOT)
    ros_df = extract_trends(
        ros_df, 'depth_m', ROSETTE_DEPTH_TARGETS, plot=DEPTH_PLOT)
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


    # Create bottle plot
    if BOTTLE_PLOT is True:
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax2 = ax1.twinx()
        ros_df_2.sort_values(["reference_distance"], inplace=True)
        sage = ax1.scatter(ros_df_2['reference_distance'],
                           ros_df_2['sage_nM'], label="SAGE")
        sage_fill = ax1.fill_between(
            ros_df_2['reference_distance'], y1=ros_df_2['sage_nM']*0.5, y2=ros_df_2['sage_nM']*1.5, alpha=0.1, color="blue")
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

        fig, ax1 = plt.subplots(figsize=(10, 5))
        fig.subplots_adjust(right=0.8)
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()
        ax3.spines.right.set_position(("axes", 1.3))
        ros_df_2.sort_values(["reference_distance"], inplace=True)
        ros_df_2["sage_nM"] = norm(ros_df_2["sage_nM"])
        sage = ax1.scatter(ros_df_2['reference_distance'],
                           ros_df_2['sage_nM'], label="SAGE")
        gga = ax2.vlines(bott_df['reference_distance'], ymin=bott_df['ch4_nM_corr_033'],
                         ymax=bott_df['ch4_nM_corr_023'], label="Bottle Methane", color="gray")
        amm = ax3.scatter(bott_df['reference_distance'], bott_df["[NH4] (nM)"],
                          label="Bottle Ammonium", color="orange", s=100)

        ax1.set_xlabel("Distance from Reference (km)", size=20)
        ax1.set_ylabel("SAGE Normalized Methane", color="blue", size=20)
        ax1.tick_params(axis='y', labelcolor="blue", labelsize=15)
        ax1.tick_params(axis='x', labelsize=15)
        ax2.set_ylabel("Methane (nM)", color="gray", size=20)
        ax2.tick_params(axis='y', labelcolor="gray", labelsize=15)
        ax3.set_ylabel("Ammonium (nM)", color="orange", size=20)
        ax3.tick_params(axis='y', labelcolor="orange", labelsize=15)
        lns = [sage, gga, amm]
        labs = [x.get_label() for x in lns]
        ax1.legend(lns, labs, loc=0)
        fig.tight_layout()
        pltname = os.path.join(os.getenv("SENTRY_OUTPUT"),
                               f"transect/figures/paper/sage_ch4_nh4__3scale_raw.svg")
        plt.savefig(pltname)
        plt.close()

    # Normalize methane values
    scc_df["fundamental_nM"] = norm(scc_df["fundamental_nM"])
    ros_df["sage_nM"] = norm(ros_df["sage_nM"])

    # Create Turbidity plot
    if TURBIDITY_PLOT is True:
        scc_df.dropna(subset=['obs'], inplace=True)
        ros_df.dropna(subset=["beam_attenuation"], inplace=True)
        # cut off sensor error on Sentry
        scc_df = scc_df[scc_df.timestamp >=
                        pd.Timestamp("2021-11-30 05:37:00")]
        # set detection threshold
        scc_thresh = scc_df[norm(scc_df['obs']) > 0.5]
        ros_thresh = ros_df[norm(ros_df['beam_attenuation']) > 0.5]
        print([scc_thresh['reference_distance'].values[0],
              ros_thresh['reference_distance'].values[0]])
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

    # Create Temperature plot
    if TEMPERATURE_PLOT is True:
        scc_df.dropna(subset=['potential_temp'], inplace=True)
        ros_df.dropna(subset=["pot_temp_C_its90"], inplace=True)
        cmax = np.percentile(np.concatenate(
            (scc_df['potential_temp'].values, ros_df['pot_temp_C_its90'].values), axis=None), 99)
        cmin = np.percentile(np.concatenate(
            (scc_df['potential_temp'].values, ros_df['pot_temp_C_its90'].values), axis=None), 1)
        scc_plot = go.Scatter(x=scc_df['reference_distance'],
                              y=-scc_df['depth'],
                              mode="markers",
                              marker=dict(size=5,
                                          color=scc_df['potential_temp'],
                                          colorscale="Viridis",
                                          colorbar=dict(
                                              thickness=20, x=-0.2, tickfont=dict(size=20)),
                                          cmin=cmin,
                                          cmax=cmax),
                              name="Sentry Temperature Anomaly")
        ros_plot = go.Scatter(x=ros_df['reference_distance'],
                              y=-ros_df['depth_m'],
                              mode="markers",
                              marker=dict(size=5,
                                          color=ros_df['pot_temp_C_its90'],
                                          colorscale="Viridis",
                                          colorbar=dict(
                                              thickness=20, x=-0.2, tickfont=dict(size=20)),
                                          cmin=cmin,
                                          cmax=cmax,
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

    # Create Salinity plot
    if SALINITY_PLOT is True:
        scc_df.dropna(subset=['ctd_sal'], inplace=True)
        ros_df.dropna(subset=["prac_salinity"], inplace=True)
        cmax = np.percentile(np.concatenate(
            (scc_df['ctd_sal'].values, ros_df['prac_salinity'].values), axis=None), 99)
        cmin = np.percentile(np.concatenate(
            (scc_df['ctd_sal'].values, ros_df['prac_salinity'].values), axis=None), 1)
        scc_plot = go.Scatter(x=scc_df['reference_distance'],
                              y=-scc_df['depth'],
                              mode="markers",
                              marker=dict(size=5,
                                          color=scc_df['ctd_sal'],
                                          colorscale="Viridis",
                                          colorbar=dict(
                                              thickness=20, x=-0.2, tickfont=dict(size=20)),
                                          cmax=cmax,
                                          cmin=cmin),
                              name="Sentry Salinity Anomaly")
        ros_plot = go.Scatter(x=ros_df['reference_distance'],
                              y=-ros_df['depth_m'],
                              mode="markers",
                              marker=dict(size=5,
                                          color=ros_df['prac_salinity'],
                                          colorscale="Viridis",
                                          colorbar=dict(
                                              thickness=20, x=-0.5, tickfont=dict(size=20)),
                                          cmax=cmax,
                                          cmin=cmin,
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

    # Create O2 plot
    if O2_PLOT is True:
        scc_df.dropna(subset=['O2'], inplace=True)
        ros_df.dropna(subset=["o2_umol_kg"], inplace=True)
        cmax = np.percentile(np.concatenate(
            (scc_df['O2'].values, ros_df['o2_umol_kg'].values), axis=None), 99)
        cmin = np.percentile(np.concatenate(
            (scc_df['O2'].values, ros_df['o2_umol_kg'].values), axis=None), 1)
        scc_plot = go.Scatter(x=scc_df['reference_distance'],
                              y=-scc_df['depth'],
                              mode="markers",
                              marker=dict(size=5,
                                          color=scc_df['O2'],
                                          colorscale="Viridis",
                                          colorbar=dict(
                                              thickness=20, x=-0.2, tickfont=dict(size=20)),
                                          cmin=cmin,
                                          cmax=cmax,),
                              name="Sentry O2 Anomaly")
        ros_plot = go.Scatter(x=ros_df['reference_distance'],
                              y=-ros_df['depth_m'],
                              mode="markers",
                              marker=dict(size=5,
                                          color=ros_df['o2_umol_kg'],
                                          colorscale="Viridis",
                                          colorbar=dict(
                                              thickness=20, x=-0.2, tickfont=dict(size=20)),
                                          cmin=cmin,
                                          cmax=cmax,
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

    # Create CH4 plot
    if CH4_PLOT is True:
        scc_df.dropna(subset=["fundamental_nM", 'O2'], inplace=True)
        ros_df.dropna(subset=["sage_nM", "o2_umol_kg"], inplace=True)
        scc_thresh = scc_df[scc_df['fundamental_nM'] > 0.5]
        ros_thresh = ros_df[ros_df['sage_nM'] > 0.5]
        print([scc_thresh['reference_distance'].values[0],
              ros_thresh['reference_distance'].values[0]])
        scc_plot = go.Scatter(x=scc_df['reference_distance'],
                              y=-scc_df['depth'],
                              mode="markers",
                              marker=dict(size=5,
                                          color=scc_df['fundamental_nM'],
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
                                          color=ros_df['sage_nM'],
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

    # Create regime plots
    if REGIME_PLOT is True:
        SENTRY_REG_TARGETS = SENTRY_CORR_TARGETS
        ROSETTE_REG_TARGETS = ROSETTE_CORR_TARGETS
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
                # make sure to remove bad OBS measurements
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
