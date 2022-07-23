"""Basic analysis on survey data, including detrending and smoothing."""

import os
import utm
import json
import pandas as pd
import numpy as np
import scipy.stats as stats
import scipy.signal
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import timedelta
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from itertools import combinations

from guaymas_data_analysis.utils.pseudosensor_utils import SciencePseudoSensor


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
                r_window_size, center=True).mean(centered=True)
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


def compute_3D_distance(ref_coord, traj_coord):
    """Compute absolute distance between points with depth."""
    RX, RY, ZN, ZL = utm.from_latlon(ref_coord[0], ref_coord[1])
    TX, TY, _, _ = utm.from_latlon(
        traj_coord[0], traj_coord[1], force_zone_number=ZN, force_zone_letter=ZL)

    # determine distance
    dist = np.sqrt((RX-TX)**2 + (RY-TY)**2 + (ref_coord[2]-traj_coord[2])**2)
    return dist


def compute_proportions(data):
    """Compute proportion of dtections in series."""
    total_obs = len(data)
    positive_obs = data.sum()
    prop_positive_obs = positive_obs/total_obs
    return total_obs, positive_obs, prop_positive_obs


##########
# Globals
##########
DIVES = ["sentry607", "sentry608", "sentry610", "sentry611"]
RIDGE_REFERENCE = (27.412645, -111.386915, -1840)
SENSOR = SciencePseudoSensor(sensors=["O2_conc", "potential_temp", "practical_salinity", "obs", "dorpdt", "methane"],
                             treatments=[dict(method="meanstd_window", num_std=3, window=60*60),
                                         dict(
                                             method="percentile", percentile_lower=0, percentile_upper=75),
                                         dict(method="meanstd", num_std=3),
                                         dict(
                                             method="percentile", percentile_lower=0, percentile_upper=75),
                                         dict(method="threshold",
                                              threshold=-0.005, direction="<"),
                                         dict(method="threshold", threshold=0.3, direction=">")],
                             weights=[1., 2., 1., 2., 2., 2.],
                             num_corroboration=4)
DETECTION_RADIUS = 1000  # meters from reference point to compute spatial diversity
DETECTION_SPATIAL_BINS = 50  # meters
DETECTION_TIME_BINS = 60  # minutes
DETECTION_DATA = {}  # storge for per-mission detection data

METHANE_LABEL = ["Pythia Fundamental", "Pythia Fundamental",
                 "Pythia Fundamental", "SAGE Methane"]
PLOT_VARS = ["O2_conc", "potential_temp",
             "practical_salinity", "obs", "dorpdt", "methane"]
PLOT_LABELS = ["Oxygen Concentration (umol/kg)", "Potential Temperature (C)",
               "Practical Salinity (PSU)", "Optical Backscatter (%)", "dORPdt", "Pythia Fundamental"]

# Static plotting objects/targets
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
name = 'eye = (x:0., y:0., z:2.5)'
camera = dict(
    eye=dict(x=0., y=0., z=2.5)
)

# Data treatment
DETREND = True  # whether to remove altitude effects
DETREND_VARS = ["O2_conc", "potential_temp", "practical_salinity"]
DETREND_LABELS = ["O2", "Potential Temperature", "Practical Salinity"]

SMOOTH = True  # whether to smooth the data
SMOOTH_OPTION = "rolling_average"  # rolling_averge or butter
SMOOTH_WINDOW = 5  # sets the rolling average window, minutes
SMOOTH_VARS = ["O2_conc", "obs", "potential_temp",
               "practical_salinity", "methane"]

NORMALIZE = True  # whether to normalize the methane data

# Plotting options
PLOT_N = 5  # subsample plot visualizations
DEPTH_PLOT = False  # whether to plot detrended data
PLOT_BINARY = False  # whether to plot binary detection data
PLOT_3D = False  # create a 3D HTML plot of data
PLOT_2D = False  # create bathy underlay 2D overview
PLOT_TIME = False  # create a time series plot of the mission
PLOT_TDV_SLICE = False  # create a time vs depth plot colored by value
PLOT_POLAR = False  # plot trajectory on ridge-centered plot
PLOT_DVP = False  # plot distance from source by value, colored by phase

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
        temp_df["dive_name"] = [DIVE_NAME for i in range(
            len(temp_df.timestamp.values))]
        if "fundamental" in temp_df.columns:
            temp_df["methane"] = temp_df["fundamental"]
        all_dfs.append(temp_df)
    scc_df = pd.concat(all_dfs)

    # Compute an oriented distance measure from ridge reference
    m = scc_df.apply(lambda x: compute_distance_and_angle(
        RIDGE_REFERENCE, (float(x['lat']), float(x['lon']))), axis=1)
    scc_df.loc[:, "distance"] = [x[0] for x in m.values]
    scc_df.loc[:, "angle"] = [x[1] for x in m.values]

    # Compute absolute distance measure from ridge reference
    scc_df.loc[:, "abs_distance"] = scc_df.apply(lambda x: compute_3D_distance(
        RIDGE_REFERENCE, (float(x['lat']), float(x['lon']), -float(x['depth']))), axis=1)

    # Perform base conversions (smoothing, detrending, normalization)
    if DETREND is True:
        scc_df = extract_trends(
            scc_df, 'depth', DETREND_VARS, DETREND_LABELS, plot=DEPTH_PLOT)

    ############
    # Apply binary pseudosensor(s)
    ############

    # Break everything back out
    for idx, DIVE_NAME in enumerate(DIVES):
        mission_df = scc_df[scc_df["dive_name"] == DIVE_NAME]

        # Normalize data
        if NORMALIZE is True:
            mission_df.loc[:, "methane"] = norm(mission_df["methane"])

        # Add the right methane target
        if METHANE_LABEL[idx] is "Pythia Fundamental":
            # Fundamental signal is already smoothed at time correction
            try:
                SMOOTH_VARS.remove("methane")
            except ValueError:
                pass
        else:
            SMOOTH_VARS.append("methane")
        PLOT_LABELS[-1] = METHANE_LABEL[idx]

        # Smooth data
        if SMOOTH is True:
            smooth_data(mission_df, SMOOTH_VARS,
                        smooth_option=SMOOTH_OPTION, smooth_window=SMOOTH_WINDOW)

        ############
        # Apply binary pseudosensor(s)
        ############
        detections_df = SENSOR.get_detections(mission_df)
        detections_df.to_csv(os.path.join(
            os.getenv("SENTRY_OUTPUT"), f"north_ridge_surveys/detection_{DIVE_NAME}.csv"))
        mission_df.loc[:, "detections"] = detections_df["detections"]

        if PLOT_BINARY is True:
            sensor_detection_plots = make_subplots(rows=len(PLOT_VARS)+1, cols=1, shared_xaxes=True,
                                                   subplot_titles=tuple(PLOT_LABELS+["Detections"]))
            for ids, sensor in enumerate(PLOT_VARS):
                sensor_detection_plots.add_trace(go.Scatter(x=mission_df["timestamp"][::PLOT_N],
                                                            y=detections_df[f"{sensor}_treatment"][::PLOT_N],
                                                            mode="markers"),
                                                 row=ids+1,
                                                 col=1)
            sensor_detection_plots.add_trace(go.Scatter(x=mission_df["timestamp"][::PLOT_N],
                                                        y=mission_df["detections"][::PLOT_N],
                                                        mode="markers"),
                                             row=ids+2,
                                             col=1)
            sensor_detection_plots.write_image(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                                            f"north_ridge_surveys/figures/binary_{DIVE_NAME}.svg"), width=750, height=750)

            sfig = go.Scatter3d(x=mission_df["lon"][::PLOT_N],
                                y=mission_df["lat"][::PLOT_N],
                                z=-mission_df["depth"][::PLOT_N],
                                mode="markers",
                                marker=dict(size=2,
                                            color=mission_df["detections"][::PLOT_N],
                                            colorscale="Sunset",
                                            colorbar=dict(
                                                thickness=20, x=-0.2, tickfont=dict(size=20)),
                                            opacity=0.8,
                                            cmin=0,
                                            cmax=1),
                                name="Binary Detection")
            fig = go.Figure([bathy_plot_3d, sfig],
                            layout_title_text="Binary Detection")
            fig.update_layout(showlegend=False)
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
            fig.write_html(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                        f"north_ridge_surveys/figures/binary_{DIVE_NAME}.html"))

        ###########
        # Compute simple binary metrics
        ###########
        # Proportion in plume
        total_obs, positive_obs, prop_positive_obs = compute_proportions(
            mission_df.detections)

        # Proportion within horizontal radii bins
        last_dist = 0.
        horiz_radii_detections = {}
        while last_dist < DETECTION_RADIUS:
            r1, r2 = last_dist, last_dist + DETECTION_SPATIAL_BINS
            temp_obs = mission_df[(mission_df["distance"]
                                   > r1) & (mission_df["distance"] <= r2)]
            total, positive, prop = compute_proportions(temp_obs.detections)
            horiz_radii_detections[f"{r1}m-{r2}m"] = dict(low_end=r1,
                                                          high_end=r2,
                                                          total_obs=total,
                                                          positive_obs=positive,
                                                          prop_positive_obs=prop)
            last_dist = r2

        # Proportion within an absolute distance
        last_dist = 0.
        abs_radii_detections = {}
        while last_dist < DETECTION_RADIUS:
            r1, r2 = last_dist, last_dist + DETECTION_SPATIAL_BINS
            temp_obs = mission_df[(mission_df["abs_distance"]
                                   > r1) & (mission_df["abs_distance"] <= r2)]
            total, positive, prop = compute_proportions(temp_obs.detections)
            abs_radii_detections[f"{r1}m-{r2}m"] = dict(low_end=r1,
                                                        high_end=r2,
                                                        total_obs=total,
                                                        positive_obs=positive,
                                                        prop_positive_obs=prop)
            last_dist = r2

        # Distribution of hits throughout mission, by phase
        phase_detects = {}
        for phase in np.unique(mission_df.phase):
            detects = mission_df[mission_df.phase == phase]
            ptot, ppos, pprop = compute_proportions(detects.detections)
            phase_detects[phase] = dict(total_obs=ptot,
                                        positive_obs=ppos,
                                        prop_positive_obs=pprop)

        # Distribution of hits throughout mission, time window
        current_time = pd.to_datetime(
            mission_df.timestamp.values[0]).tz_localize("UTC")
        end_time = pd.to_datetime(
            mission_df.timestamp.values[-1]).tz_localize("UTC")
        time_detections = {}
        mission_hour = 0
        while current_time < end_time:
            r1, r2 = current_time, current_time + \
                pd.to_timedelta(DETECTION_TIME_BINS, unit="minutes")
            temp_obs = mission_df[(mission_df["timestamp"]
                                   > r1) & (mission_df["timestamp"] <= r2)]
            total, positive, prop = compute_proportions(temp_obs.detections)
            time_detections[f"{r1}-{r2}"] = dict(low_end=r1,
                                                 high_end=r2,
                                                 mission_hour=mission_hour,
                                                 total_obs=total,
                                                 positive_obs=positive,
                                                 prop_positive_obs=prop)
            mission_hour += 1
            current_time = r2

        DETECTION_DATA[DIVE_NAME] = dict(total_obs=total_obs,
                                         positive_obs=positive_obs,
                                         prop_positive_obs=prop_positive_obs,
                                         horiz_bins=horiz_radii_detections,
                                         abs_bins=abs_radii_detections,
                                         time_bins=time_detections,
                                         phase_bins=phase_detects)

        ###########
        # Simple visualizations of the data
        ###########
        time_plots = make_subplots(rows=len(PLOT_VARS), cols=1, shared_xaxes=True,
                                   subplot_titles=tuple(PLOT_LABELS))

        for idp, (target, label) in enumerate(zip(PLOT_VARS, PLOT_LABELS)):
            mission_df.sort_values(by=target, ascending=True, inplace=True)
            cmin, cmax = np.nanmin(
                mission_df[target]), np.nanmax(mission_df[target])
            # top-down overview, 3D
            if PLOT_3D is True:
                sfig = go.Scatter3d(x=mission_df["lon"][::PLOT_N],
                                    y=mission_df["lat"][::PLOT_N],
                                    z=-mission_df["depth"][::PLOT_N],
                                    mode="markers",
                                    marker=dict(size=1,
                                                color=mission_df[target][::PLOT_N],
                                                colorscale="Inferno",
                                                colorbar=dict(
                                                    thickness=20, x=-0.2, tickfont=dict(size=20)),
                                                cmin=cmin,
                                                cmax=cmax,),
                                    name=label)
                fig = go.Figure([bathy_plot_3d, sfig], layout_title_text=label)
                fig.update_layout(scene_camera=camera, showlegend=False)
                fig.update_yaxes(scaleanchor="x", scaleratio=1)
                fig.write_html(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                            f"north_ridge_surveys/figures/3D_{DIVE_NAME}_{target}.html"))

            # top-down overview, 2D
            if PLOT_2D is True:
                sfig = go.Scatter(x=mission_df["lon"][::PLOT_N],
                                  y=mission_df["lat"][::PLOT_N],
                                  mode="markers",
                                  marker=dict(size=5,
                                              color=mission_df[target][::PLOT_N],
                                              colorscale="Inferno",
                                              colorbar=dict(
                                                  thickness=20, x=-0.2, tickfont=dict(size=20)),
                                              cmin=cmin,
                                              cmax=cmax,),
                                  name=label)
                fig = go.Figure([bathy_plot, sfig], layout_title_text=label)
                fig.update_layout(showlegend=False)
                fig.update_yaxes(scaleanchor="x", scaleratio=1)
                fig.write_image(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                             f"north_ridge_surveys/figures/2D_{DIVE_NAME}_{target}.svg"), width=750, height=750)

            # time series
            if PLOT_TIME is True:
                time_plots.add_trace(go.Scatter(
                    x=mission_df["timestamp"][::PLOT_N], y=mission_df[target][::PLOT_N], mode="markers"), row=idp+1, col=1)

            # time-depth-value slice
            if PLOT_TDV_SLICE is True:
                tdv_plot = go.Scatter(x=mission_df['timestamp'][::PLOT_N],
                                      y=-mission_df['depth'][::PLOT_N],
                                      mode="markers",
                                      marker=dict(size=5,
                                                  color=mission_df[target][::PLOT_N],
                                                  colorscale="Inferno",
                                                  colorbar=dict(
                                                      thickness=20, x=-0.2, tickfont=dict(size=20)),
                                                  cmin=cmin,
                                                  cmax=cmax,),
                                      name=label)
                fig = go.Figure([tdv_plot], layout_title_text=label)
                fig.update_layout(showlegend=False)
                fig.write_image(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                             f"north_ridge_surveys/figures/tdv_{DIVE_NAME}_{target}.svg"), width=1500)

            # Polar map
            if PLOT_POLAR is True:
                spoke_plot = go.Scatterpolar(r=mission_df['distance'][::PLOT_N],
                                             theta=mission_df['angle'][::PLOT_N],
                                             mode="markers",
                                             marker=dict(size=5,
                                                         color=mission_df[target][::PLOT_N],
                                                         colorscale="Inferno",
                                                         colorbar=dict(
                                                             thickness=20, x=-0.2, tickfont=dict(size=20)),
                                                         cmin=cmin,
                                                         cmax=cmax,),
                                             name=label)
                fig = go.Figure([spoke_plot], layout_title_text=label)
                fig.update_layout(showlegend=False)
                fig.write_image(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                             f"north_ridge_surveys/figures/polar_{DIVE_NAME}_{target}.svg"), width=750, height=750)

            # plot distance versus value, colored by phase
            if PLOT_DVP is True:
                distance_plot = go.Scatter(x=scc_df['distance'][::PLOT_N],
                                           y=scc_df[target][::PLOT_N],
                                           mode="markers",
                                           marker=dict(size=5,
                                                       color=scc_df["phase"][::PLOT_N],
                                                       colorscale="Inferno",
                                                       colorbar=dict(
                                                           thickness=20, x=-0.2, tickfont=dict(size=20)),
                                                       ),
                                           name=label)
                fig = go.Figure([distance_plot], layout_title_text=label)
                fig.update_layout(showlegend=False)
                fig.write_image(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                             f"north_ridge_surveys/figures/dvp_{DIVE_NAME}_{target}.svg"), width=750)

        if PLOT_TIME is True:
            time_plots.write_image(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                                f"north_ridge_surveys/figures/time_{DIVE_NAME}.svg"), width=750, height=750)

    hfig = []
    afig = []
    tfig = []
    for k, v in DETECTION_DATA.items():
        tot, pos, prop = v["total_obs"], v["positive_obs"], v["prop_positive_obs"]
        print(f"Positive detections during {k}: {pos} in {tot}, {prop}")
        print(f"Positive detections in {k} during:")
        for kk, vv in v["phase_bins"].items():
            tot, pos, prop = vv["total_obs"], vv["positive_obs"], vv["prop_positive_obs"]
            print(f"    Phase {kk}: {pos} in {tot}, {prop}")

        labs = []
        heights = []
        for kk, vv in v["horiz_bins"].items():
            labs.append(kk)
            heights.append(vv["prop_positive_obs"])
        hfig.append(go.Bar(name=k, x=labs, y=heights))

        labs = []
        heights = []
        for kk, vv in v["abs_bins"].items():
            labs.append(kk)
            heights.append(vv["prop_positive_obs"])
        afig.append(go.Bar(name=k, x=labs, y=heights))

        detects = []
        for kk, vv in v["time_bins"].items():
            detects.append(vv["positive_obs"]/DETECTION_TIME_BINS)
        print(
            f"Avg/Std detection rate for {k}: {np.nanmean(detects)}, {np.nanstd(detects)}")

    fig = go.Figure(data=hfig)
    fig.update_layout(barmode='group')
    fig.write_image(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                                f"north_ridge_surveys/figures/hdist_detections.svg"), width=750, height=750)

    fig = go.Figure(data=afig)
    fig.update_layout(barmode='group')
    fig.write_image(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                                f"north_ridge_surveys/figures/adist_detections.svg"), width=750, height=750)

    with open(os.path.join(os.getenv("SENTRY_OUTPUT"), f"north_ridge_surveys/detection_metrics.json"), "w") as fp:
        json.dump(DETECTION_DATA, fp)
