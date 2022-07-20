"""Basic analysis on survey data, including detrending and smoothing."""

import os
import utm
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


def get_bathy(rsamp=0.5):
    """Get and window the bathy file."""
    bathy_file = os.path.join(os.getenv("SENTRY_DATA"),
                              "bathy/proc/ridge.txt")
    bathy = pd.read_table(bathy_file, names=["long", "lat", "depth"]).dropna()
    return bathy.sample(frac=rsamp, random_state=1)


def compute_distance_and_angle(ref_coord, traj_coord):
    """Computes an oriented distance from a reference coordinate, where orientation is aligned E-W."""
    # convert to UTM coordinates
    RX, RY, ZN, ZL = utm.from_latlon(ref_coord[0], ref_coord[1])
    TX, TY, _, _ = utm.from_latlon(
        traj_coord[0], traj_coord[1], force_zone_number=ZN, force_zone_letter=ZL)

    # determine distance
    dist = np.sqrt((RX-TX)**2 + (RY-TY)**2)

    # determine angle
    ang = np.arctan2((TY-RY), (TX-RX)) * 180/np.pi

    return dist, ang


##########
# Globals
##########
DIVES = ["sentry607", "sentry608", "sentry610", "sentry611"]
RIDGE_REFERENCE = (27.412645, -111.386915)
METHANE_COL = ["fundamental", "fundamental", "fundamental", "methane"]
METHANE_LABEL = ["Pythia Fundamental", "Pythia Fundamental",
                 "Pythia Fundamental", "SAGE Methane"]
PLOT_VARS = ["O2_conc", "potential_temp",
             "practical_salinity", "obs", "dorpdt", "fundamental"]
PLOT_LABELS = ["Oxygen Concentration (umol/kg)", "Potential Temperature (C)",
               "Practical Salinity (PSU)", "Optical Backscatter (%)", "dORPdt", "Pythia Fundamental"]
PLOT_N = 10

# Create a bathy plot
BATHY = get_bathy(rsamp=0.02)
bathy_plot = go.Scatter(x=BATHY.long,
                        y=BATHY.lat,
                        mode="markers",
                        marker=dict(size=5,
                                    color=BATHY.depth,
                                    opacity=1.0,
                                    colorscale="Viridis",
                                    colorbar=dict(thickness=20, x=1.2, tickfont=dict(size=20))),
                        name="Bathy")
bathy_plot_3d = go.Mesh3d(x=BATHY.long, y=BATHY.lat, z=BATHY.depth,
                          intensity=BATHY.depth,
                          colorscale='Viridis',
                          opacity=0.50,
                          name="Bathy")

DETREND = True  # whether to remove altitude effects
DETREND_VARS = ["O2_conc", "potential_temp", "practical_salinity"]
DETREND_LABELS = ["O2", "Potential Temperature", "Practical Salinity"]

SMOOTH = False  # whether to smooth the data
SMOOTH_OPTION = "rolling_average"  # rolling_averge or butter
SMOOTH_WINDOW = 15  # sets the rolling average window, minutes
SMOOTH_VARS = ["O2_conc", "obs", "potential_temp",
               "practical_salinity", "fundamental"]

NORMALIZE = True  # whether to normalize the methane data
NORM_THRESH = 0.3  # detection threshold for methane presence in normalized data

DEPTH_PLOT = True  # whether to plot detrended data


if __name__ == '__main__':
    all_dfs = []
    for idx, DIVE_NAME in enumerate(DIVES):
        #######
        # Data Processing
        #######
        # Get the data
        input_name = f"{DIVE_NAME}_processed.csv"
        input_file = os.path.join(
            os.getenv("SENTRY_DATA"), f"sentry/proc/RR2107_{input_name}")
        temp_df = pd.read_csv(input_file)
        temp_df["timestamp"] = pd.to_datetime(temp_df['timestamp'])
        all_dfs.append(temp_df)
    
    scc_df = pd.concat(all_dfs)

    # Compute an oriented distance measure from ridge reference
    m = scc_df.apply(lambda x: compute_distance_and_angle(
        RIDGE_REFERENCE, (float(x['lat']), float(x['lon']))), axis=1)
    scc_df["distance"] = [x[0] for x in m.values]
    scc_df["angle"] = [x[1] for x in m.values]

    # Add the right methane target
    SMOOTH_VARS[-1] = METHANE_COL[idx]
    PLOT_VARS[-1] = METHANE_COL[idx]
    PLOT_LABELS[-1] = METHANE_LABEL[idx]

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
    name = 'eye = (x:0., y:0., z:2.5)'
    camera = dict(
        eye=dict(x=0., y=0., z=2.5)
    )

    time_plots = make_subplots(rows=len(PLOT_VARS), cols=1, shared_xaxes=True,
                                subplot_titles=tuple(PLOT_LABELS))

    for idp, (target, label) in enumerate(zip(PLOT_VARS, PLOT_LABELS)):
        scc_df.sort_values(by=target, ascending=True, inplace=True)
        cmin, cmax = np.nanmin(scc_df[target]), np.nanmax(scc_df[target])
        # top-down overview, 3D
        # sfig = go.Scatter3d(x=scc_df["lon"][::PLOT_N],
        #                     y=scc_df["lat"][::PLOT_N],
        #                     z=-scc_df["depth"][::PLOT_N],
        #                     mode="markers",
        #                     marker=dict(size=1,
        #                                 color=scc_df[target][::PLOT_N],
        #                                 colorscale="Inferno",
        #                                 colorbar=dict(
        #                                     thickness=20, x=-0.2, tickfont=dict(size=20)),
        #                                 cmin=cmin,
        #                                 cmax=cmax,),
        #                     name=label)
        # fig = go.Figure([bathy_plot_3d, sfig], layout_title_text=label)
        # fig.update_layout(scene_camera=camera, showlegend=False)
        # fig.update_yaxes(scaleanchor="x", scaleratio=1)
        # fig.show()

        # top-down overview, 2D
        # sfig = go.Scatter(x=scc_df["lon"][::PLOT_N],
        #                   y=scc_df["lat"][::PLOT_N],
        #                   mode="markers",
        #                   marker=dict(size=5,
        #                               color=scc_df[target][::PLOT_N],
        #                               colorscale="Inferno",
        #                               colorbar=dict(
        #                                   thickness=20, x=-0.2, tickfont=dict(size=20)),
        #                               cmin=cmin,
        #                               cmax=cmax,),
        #                   name=label)
        # fig = go.Figure([bathy_plot, sfig], layout_title_text=label)
        # fig.update_layout(showlegend=False)
        # fig.update_yaxes(scaleanchor="x", scaleratio=1)
        # fig.show()

        # time series
        time_plots.add_trace(go.Scatter(
            x=scc_df["timestamp"][::PLOT_N], y=scc_df[target][::PLOT_N], mode="markers"), row=idp+1, col=1)

        # time-depth-value slice
        # tdv_plot = go.Scatter(x=scc_df['timestamp'][::PLOT_N],
        #                       y=-scc_df['depth'][::PLOT_N],
        #                       mode="markers",
        #                       marker=dict(size=5,
        #                                   color=norm(scc_df[target][::PLOT_N]),
        #                                   colorscale="Inferno",
        #                                   colorbar=dict(
        #                                       thickness=20, x=-0.2, tickfont=dict(size=20)),
        #                                   cmin=cmin,
        #                                   cmax=cmax,),
        #                       name=label)
        # fig = go.Figure([tdv_plot], layout_title_text=label)
        # fig.update_layout(showlegend=False)
        # fig.show()

        # radius spoke (signed distance by value colored by time?)
        # spoke_plot = go.Scatterpolar(r=scc_df['distance'][::PLOT_N],
        #                              theta=scc_df['angle'][::PLOT_N],
        #                              mode="markers",
        #                              marker=dict(size=5,
        #                                          color=scc_df[target][::PLOT_N],
        #                                          colorscale="Inferno",
        #                                          colorbar=dict(
        #                                              thickness=20, x=-0.2, tickfont=dict(size=20)),
        #                                          cmin=cmin,
        #                                          cmax=cmax,),
        #                              name=label)
        # fig = go.Figure([spoke_plot], layout_title_text=label)
        # fig.update_layout(showlegend=False)
        # fig.show()

        # spoke_plot = go.Scatter(x=scc_df['distance'][::PLOT_N],
        #                         y=scc_df[target][::PLOT_N],
        #                         mode="markers",
        #                         marker=dict(size=5,
        #                                     color=scc_df["phase"][::PLOT_N],
        #                                     colorscale="Inferno",
        #                                     colorbar=dict(
        #                                         thickness=20, x=-0.2, tickfont=dict(size=20)),
        #                                     ),
        #                         name=label)
        # fig = go.Figure([spoke_plot], layout_title_text=label)
        # fig.update_layout(showlegend=False)
        # fig.show()

    time_plots.show()

    ############
    # Save processed data product and meta data
    ############
    # TODO
