"""Utility functions for simulation and plotting."""
import pdb
import os
import sys
import yaml
import utm
import numpy as np
import pandas as pd
import pickle

# Import dependencies
import plotly
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from sentry_data_analysis.utils import tic, toc

# Package dependencies
from .data_utils import convert_to_latlon
from .metadata_utils import sentry_site_by_dive, extent_by_site

# Standard color sequence
COLORS = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
          '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']


def plot_and_save_ctd_data(ctd_dfs,
                           ctd_names,
                           analysis_variables,
                           analysis_labels,
                           extent,
                           site,
                           all_bots,
                           all_locs,
                           all_times,
                           save_plot=True,
                           skip_bathy=100):
    """Plot CTD data."""
    for vi, var in enumerate(analysis_variables):
        all_vars = []
        for df in ctd_dfs:
            for v in df[var]:
                all_vars.append(v)

        # Adjust the color scale
        cmin, cmax = np.nanpercentile(
            all_vars, 20), np.nanpercentile(all_vars, 80)

        # Process the bathymetry data
        bathy = get_bathy(extent, site=site)
        pts_bathy = bathy[['long', 'lat', 'depth']].values[::skip_bathy]
        bathy_fig = mesh_obj(pts_bathy, name="Bathy")
        figs = [bathy_fig]

        for i, df in enumerate(ctd_dfs):
            figs.append(go.Scatter3d(x=df['longitude'],
                                     y=df['latitude'],
                                     z=-df['depth'],
                                     mode="markers",
                                     name=f"{analysis_labels[vi]}_{ctd_names[i]}",
                                     marker=dict(size=1,
                                                 color=df[var],
                                                 colorscale="Inferno",
                                                 cmin=cmin,
                                                 cmax=cmax,
                                                 colorbar=dict(thickness=30, x=-0.1))))

        for i, (bott, loc, tim) in enumerate(zip(all_bots, all_locs, all_times)):
            for b, l, t in zip(bott, loc, tim):
                figs.append(go.Scatter3d(x=[l[0]],
                                         y=[l[1]],
                                         z=[-l[2]],
                                         name=f"{b}_{t}_{ctd_names[i]}",
                                         mode="markers",
                                         marker=dict(size=5,
                                                     color="red",)))
        fig = go.Figure(data=figs, layout_title_text=str(
            var), layout_legend_x=0.0)
        fig.show()

        if save_plot:
            fname = f"ctd_cast_{analysis_variables[vi]}.html"
            fname = os.path.join(
                os.getenv("SENTRY_OUTPUT"), f"ctd/{fname}")
            fig.write_html(fname)
            print("Figure", fname, "saved.")


def plot_and_save_detections(df, dive_name):
    plt.scatter(df['easting'], df['northing'],
                c=df["detections"], cmap='coolwarm', s=0.5)
    plt.xlabel("Meters Easting")
    plt.ylabel("Meters Northing")
    plt.colorbar()
    plt.title("Detections from Binary Pseudosensor")

    fname = f"detections_{dive_name}"
    fname = os.path.join(
        os.getenv("SENTRY_OUTPUT"), f"sentry/{fname}")
    plt.savefig(fname)

    # Get the site location
    site = sentry_site_by_dive(dive_name)

    # Extent information by site
    extent = extent_by_site(site)

    bathy = get_bathy(extent)

    x = bathy['long'].values
    y = bathy['lat'].values
    z = bathy['depth'].values
    bathy_data = go.Mesh3d(x=x, y=y, z=z,
                           intensity=-z, colorscale='Viridis',
                           opacity=0.50)

    all_detects_plot = [bathy_data]
    only_positive_plot = [bathy_data]

    only_detects = df[df['detections'] == 1]

    all_detects_plot.append(go.Scatter3d(x=df['lon'], y=df['lat'], z=-df['depth'],
                                         mode="markers",
                                         marker=dict(size=1,
                                                     color=df['detections'],
                                                     opacity=0.5,)))
    only_positive_plot.append(go.Scatter3d(x=only_detects['lon'], y=only_detects['lat'], z=-only_detects['depth'],
                                           mode="markers",
                                           marker=dict(size=1,
                                                       opacity=0.5,)))

    fig = go.Figure(data=only_positive_plot)
    fig.show()
    fname = f"detections_3d_{dive_name}.html"
    fname = os.path.join(
        os.getenv("SENTRY_OUTPUT"), f"sentry/{fname}")
    fig.write_html(fname)
    print("Figure", fname, "saved.")


def to_homogenous(x, y):
    """Convert data point to homogeneous coordinates."""
    return np.vstack([x, y, np.zeros(y.shape), np.ones(y.shape)]).T


def translation(deltax, deltay, deltaz):
    """Perform affine transformation by delta x, y, z."""
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [deltax, deltay, deltaz, 1],
    ])


def plot_and_save_advection_video(
        df,
        analysis_variable,
        analysis_label,
        mag_model,
        head_model,
        time_skip,
        extent,
        site,
        plot_name=None,
        save_video=True):
    """Plot sensor data, moved over time via advection."""
    # Get subampled time range
    time_index = df.index
    time_seconds = df["seconds"]
    times = zip(time_index, time_seconds)
    # time_seconds = df["seconds"][::time_skip]
    time_frames = list(enumerate(times))[::time_skip]

    # Query the heading and magnitude models at subsampled time frames
    if type(mag_model).__name__ == "TimeSeriesInterpolate":
        # If using raw data, use timestamp
        ts = df.index
    else:
        # Otherwise, use seconds from UTC midnight
        ts = df["seconds"]
    mag = mag_model.magnitude(None, ts)
    head = head_model.heading(ts)

    # fig, axs = plt.subplots(4, 1)
    # axs[0].plot(ts, mag)
    # axs[1].plot(ts, head)
    # axs[2].plot(mag_model.t, mag_model.val)
    # axs[3].plot(head_model.t, head_model.val)
    # plt.show()
    # exit()

    # Initialize list of locations
    global locs_all
    global scat
    locs_all = None

    # Instantiate the animation plot
    fig, ax = plt.subplots(1, 1)
    ax.set_xlim([extent.xmin, extent.xmax])
    ax.set_ylim([extent.ymin, extent.ymax])
    ax.axis('equal')
    # ax.axis('off')

    def init():
        global scat
        global locs_all
        scat = ax.scatter(df.iloc[0]["easting"], df.iloc[0]["northing"],
                          s=2, c=df[analysis_variable].mean(),
                          vmin=df[analysis_variable].quantile(0.10),
                          vmax=df[analysis_variable].quantile(0.90),
                          cmap="inferno")
        ax.set_xlim([extent.xmin, extent.xmax])
        ax.set_ylim([extent.ymin, extent.ymax])
        locs_all = None
        return scat,

    def animate(cur_frame):
        global locs_all
        global scat

        cur_index = time_frames[cur_frame][0]
        cur_time = time_frames[cur_frame][1][0]
        if cur_frame < len(time_frames)-1:
            next_index = time_frames[cur_frame+1][0]
            next_time = time_frames[cur_frame+1][1][0]
        else:
            next_index = -1
            next_time = df.index[-1]
        frame_time_step = next_time - cur_time

        mag_frame = mag[cur_index]
        head_frame = head[cur_index]

        delta_x = frame_time_step.seconds * mag_frame * np.cos(head_frame)
        delta_y = frame_time_step.seconds * mag_frame * np.sin(head_frame)
        T = translation(delta_x, delta_y, 0)

        df_sub = df.iloc[cur_index:next_index]
        locs = df_sub[["easting", "northing", analysis_variable]
                      ].to_numpy().reshape(-1, 3)
        locs_hom = np.array([to_homogenous(*x) for x in locs]).reshape(-1, 4)

        if locs_all is None:
            locs_all = locs_hom
        else:
            locs_all = np.vstack([locs_all, locs_hom])

        # Perform translation
        locs_all = locs_all @ T

        # Update the location and color of the plot
        scat.set_array(locs_all[:, 2])
        scat.set_offsets(locs_all[:, :2])

        print("Adding indices", cur_index, next_index)
        print(locs_all.shape)

        # return s0
        return scat,
    tic()
    anim = animation.FuncAnimation(fig=fig, func=animate, init_func=init,
                                   frames=len(time_frames), repeat=False, blit=True)

    fname = f"sample_advection.gif"
    if plot_name is not None:
        fname = f"{plot_name}_{fname}"
    fname = os.path.join(
        os.getenv("SENTRY_OUTPUT"), f"sentry/{fname}")
    anim.save(fname, writer='imagemagick', fps=5)
    plt.show()
    toc()


def plot_and_save_science_data(df,
                               analysis_variables,
                               analysis_labels,
                               plot_log,
                               extent,
                               site,
                               plot_name=None,
                               viz_2d=True,
                               viz_3d=True,
                               viz_bathy=True,
                               save_2d=True,
                               save_3d=True,
                               save_bathy=True):
    """Plot variables of interest"""
    if viz_2d:
        # Visualize bathymetry
        if viz_bathy and extent is not None:
            bathy_data = plot_bathy_underlay(extent, site=site, step=5000)
            sites_data = plot_sites_overlay(extent)
            if save_bathy:
                b = go.Figure(
                    data=[bathy_data], layout_title_text=f"Bathymetry for Site {site}")
                b.show()
                fname = f"{site}-bathy_2d.html"
                fname = os.path.join(
                    os.getenv("SENTRY_OUTPUT"), f"sentry/{fname}")
                f.write_html(fname)
                print("Figure", fname, "saved.")

        #  Iterate through scientific variables
        num_vars = len(analysis_variables)
        fig = make_subplots(rows=num_vars, cols=1,
                            subplot_titles=tuple(analysis_variables))
        fig_scatter = []
        for i, v in enumerate(analysis_variables):
            var = df[v]
            if plot_log[i]:
                var += np.abs(np.nanmin(var)) + 1.0
                var = np.log(var)
            cmin, cmax = np.nanpercentile(
                var, 10), np.nanpercentile(var, 90)
            fig.add_trace(go.Scatter(
                x=df.index, y=var), row=i+1, col=1)
            fig_s = go.Scatter(x=df['lon'], y=df['lat'],
                               mode="markers",
                               marker=dict(size=2,
                                           color=var,
                                           opacity=0.7,
                                           colorscale="Inferno",
                                           cmin=cmin,
                                           cmax=cmax,
                                           colorbar=dict(thickness=30, x=-0.1)),
                               name=v)
            if viz_bathy and extent is not None:
                fig_scatter.append(
                    go.Figure(data=[bathy_data, sites_data, fig_s], layout_title_text=f"{v}"))
            else:
                fig_scatter.append(
                    go.Figure(data=[fig_s], layout_title_text=f"{v}"))

        fig.show()
        # for f in fig_scatter:
        #     f.show()

        if save_2d:
            for i, f in enumerate(fig_scatter):
                f.show()
                fnames = [f"{analysis_labels[i]}_2d-scatter.html",
                          f"{analysis_labels[i]}_2d-scatter.pdf",
                          f"{analysis_labels[i]}_2d-scatter.png"]

                for fname in fnames:
                    if plot_name is not None:
                        fname = f"{plot_name}_{fname}"
                    fname = os.path.join(
                        os.getenv("SENTRY_OUTPUT"), f"sentry/{fname}")
                    f.write_html(fname)
                    print("Figure", fname, "saved.")

    if viz_3d:
        if viz_bathy and extent is not None:
            bathy = get_bathy(extent, site=site, step=1000)
            x = bathy['long'].values
            y = bathy['lat'].values
            z = bathy['depth'].values
            bathy_data = go.Mesh3d(x=x, y=y, z=z,
                                   intensity=z,
                                   colorscale='Viridis',
                                   opacity=0.50,
                                   name="Bathy")
            if save_bathy:
                b = go.Figure(
                    data=[bathy_data], layout_title_text=f"Bathymetry for Site {site}")
                b.show()
                fname = f"{site}-bathy_3d.html"
                fname = os.path.join(
                    os.getenv("SENTRY_OUTPUT"), f"sentry/{fname}")
                f.write_html(fname)
                print("Figure", fname, "saved.")

        num_vars = len(analysis_variables)
        fig = make_subplots(rows=num_vars, cols=1,
                            subplot_titles=tuple(analysis_variables))
        fig_scatter = []
        for i, v in enumerate(analysis_variables):
            FT_A = 30
            FT_T = 18
            layout = dict(title=analysis_labels[i],
                          scene=dict(
                xaxis=dict(
                    title=dict(text="Longitude", font=dict(size=FT_A, color="black"))),
                yaxis=dict(
                    title=dict(text="Latitude", font=dict(size=FT_A, color="black"))),
                zaxis=dict(
                    title=dict(text="Depth", font=dict(size=FT_A, color="black"))),
            ),
                font=dict(size=FT_T))
            var = df[v]
            if plot_log[i]:
                var += np.abs(np.nanmin(var)) + 1.0
                var = np.log(var)
            cmin, cmax = np.nanpercentile(
                var, 10), np.nanpercentile(var, 90)
            fig.add_trace(go.Scatter(
                x=df.index, y=var), row=i+1, col=1)
            fig_s = go.Scatter3d(x=df['lon'], y=df['lat'], z=-df['depth'],
                                 mode="markers",
                                 marker=dict(size=2,
                                             color=var,
                                             opacity=0.7,
                                             colorscale="Inferno",
                                             cmin=cmin,
                                             cmax=cmax,
                                             colorbar=dict(thickness=30, x=-0.1)),
                                 name=v)
            if viz_bathy and extent is not None:
                fig_scatter.append(go.Figure(data=[bathy_data, fig_s]))
            else:
                fig_scatter.append(go.Figure(data=[fig_s]))
            fig_scatter[-1].update_layout(layout)
        fig.show()

        if save_3d:
            for i, f in enumerate(fig_scatter):
                f.show()
                fname = f"{analysis_labels[i]}_3d-scatter.html"
                if plot_name is not None:
                    fname = f"{plot_name}_{fname}"
                fname = os.path.join(
                    os.getenv("SENTRY_OUTPUT"), f"sentry/{fname}")
                f.write_html(fname)
                print("Figure", fname, "saved.")


def translation(deltax, deltay, deltaz):
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [deltax, deltay, deltaz, 1],
    ])


def scaling(cx, cy, cz):
    return np.array([
        [cx, 0, 0, 0],
        [0, cy, 0, 0],
        [0, 0, cz, 0],
        [0, 0, 0, 1],
    ])


def yrotation(theta):
    return np.array([
        [np.cos(theta), 0, np.sin(theta), 0],
        [0, 1, 0, 0],
        [-np.sin(theta), np.cos(theta), 0, 0],
        [0, 0, 0, 1],
    ])


def xrotation(theta):
    return np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta), -np.sin(theta), 0],
        [0, np.sin(theta), np.cos(theta), 0],
        [0, 0, 0, 1],
    ])


def zrotation(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta), np.cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])


def scatter_obj(data, name, color="blue", opacity=1.0, size=5):
    return {
        "x": data[:, 0],
        "y": data[:, 1],
        "z": data[:, 2],
        "mode": "markers",
        "marker": {
            'size': size,
            'opacity': opacity,
            'color': color
        },
        "name": name,
        "type": "scatter3d"
    }


def mesh_obj(data, name, return_obj=False):
    if return_obj:
        return go.Mesh3d(
            x=data[:, 0].flatten(),
            y=data[:, 1].flatten(),
            z=data[:, 2].flatten(),
            intensity=data[:, 2].flatten(),
            colorscale="Viridis",
            colorbar=dict(thickness=30, x=-0.1),
            opacity=0.5,
            name=name,
        )
    return {
        "x": data[:, 0].flatten(),
        "y": data[:, 1].flatten(),
        "z": data[:, 2].flatten(),
        "intensity": data[:, 2].flatten(),
        "colorscale": "Viridis",
        "opacity": 0.3,
        "name": name,
        "type": "mesh3d"
    }


def draw_ellipse(a, b, n=200):
    # Sample points on a circle in the xy plane
    rad = np.linspace(0, 2 * np.pi, n)
    x = b * np.cos(rad)
    y = a * np.sin(rad)
    return x, y


def to_homogenous(x, y, z=None):
    if z is None:
        return np.vstack([x, y, np.zeros(y.shape), np.ones(y.shape)]).T
    return np.vstack([x, y, z, np.ones(y.shape)]).T


def frame_args(duration):
    return {"frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"}}


def _make_and_configure_fig(title, xlim, ylim, zlim):
    # Make a figure
    fig_dict = {
        "data": [],
        "layout": {},
        "frames": [],
        "type": "scatter3d"
    }
    # Layout
    fig_dict["layout"]["title"] = title
    fig_dict["layout"]["width"] = 1500
    fig_dict["layout"]["height"] = 1500
    fig_dict["layout"]["scene"] = dict(
        xaxis=dict(nticks=10, range=xlim,),
        yaxis=dict(nticks=10, range=ylim,),
        zaxis=dict(nticks=10, range=zlim,),)

    fig_dict["layout"]["updatemenus"] = [
        {
            "buttons": [
                {
                    "args": [None, frame_args(50)],
                    "label": "&#9654;",  # play symbol
                    "method": "animate",
                },
                {
                    "args": [[None], frame_args(0)],
                    "label": "&#9724;",  # pause symbol
                    "method": "animate",
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 70},
            "type": "buttons",
                    "x": 0.1,
                    "y": 0,
        }]

    return fig_dict


def _get_sliders(fig_dict):
    # Configure the sliders
    return [{
        "pad": {"b": 10, "t": 60},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [
            {
                "args": [[f["name"]], frame_args(0)],
                "label": str(k),
                "method": "animate",
            }
            for k, f in enumerate(fig_dict["frames"])
        ],
    }]


def plot_window_in_bathy(env, bathy, sites, xlim, ylim, zlim):
    """Plot an x, y, z extent cuboid in the global bathymap."""
    # Convert extent to latitude and longitude
    lim = np.vstack([[xlim[0], ylim[0], zlim[0]], [xlim[1], ylim[1], zlim[1]]])
    lim = convert_to_latlon(lim, env.extent.origin)
    xlim = lim[:, 0]
    ylim = lim[:, 1]
    zlim = lim[:, 2]

    points = np.array([
        [xlim[0], ylim[0], zlim[0]],
        [xlim[0], ylim[1], zlim[0]],
        [xlim[1], ylim[1], zlim[0]],
        [xlim[1], ylim[0], zlim[0]],
        [xlim[0], ylim[0], zlim[1]],
        [xlim[0], ylim[1], zlim[1]],
        [xlim[1], ylim[1], zlim[1]],
        [xlim[1], ylim[0], zlim[1]],
    ])
    bounds_data = go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
                               mode="markers")
    cuboid_data = go.Mesh3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
                            i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                            j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                            k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                            color='#DC143C',
                            opacity=0.6, flatshading=True)

    x = bathy['long'].values
    y = bathy['lat'].values
    z = bathy['depth'].values
    bathy_data = go.Mesh3d(x=x[::100], y=y[::100], z=-z[::100],
                           intensity=-z[::100], colorscale='Viridis',
                           opacity=0.50)

    sites_data = go.Scatter3d(x=sites[:, 0], y=sites[:, 1], z=sites[:, 2],
                              mode="markers",
                              marker=dict(
                                   size=5,
        color="orange",
        opacity=0.9,)
    )

    fig = go.Figure(data=[bathy_data, cuboid_data, sites_data])
    return fig


def scatter_plume_and_traj(times, env, coord_times, coords, bathy=None, sites=None,
                           title="Plume Evolution", xlim=[-900, 900],
                           ylim=[-900, 900], zlim=[0, 100], ref_global=False):
    """Generate 3D scatter plot of plume and trajectory."""
    if bathy is not None and not ref_global:
        raise ValueError("Bathy can only be plotted in the global reference.")
    if sites is not None and not ref_global:
        raise ValueError("Sites can only be plotted in the global reference.")

    if ref_global:
        # Convert extent to latitude and longitude
        lim = np.vstack([[xlim[0], ylim[0], zlim[0]],
                        [xlim[1], ylim[1], zlim[1]]])
        lim = convert_to_latlon(lim, env.extent.origin)
        xlim = lim[:, 0]
        ylim = lim[:, 1]
        zlim = lim[:, 2]

    # zlim = [-2100, -1000]
    # Generate figure skeleton
    fig_dict = _make_and_configure_fig(
        title, xlim, ylim, zlim)

    # Initial simulation frame; final plume time
    pts_plume = env.get_pointcloud(t=times[-1])
    if ref_global:
        pts_plume = convert_to_latlon(pts_plume, env.extent.origin)
    pts_traj = coords[coord_times < times[-1]]

    # Process bathy data
    if ref_global:
        # Window the bathy
        bathy = bathy[(bathy['long'] >= xlim[0]) & (bathy['long'] <= xlim[1])]
        bathy = bathy[(bathy['lat'] >= ylim[0]) & (bathy['lat'] <= ylim[1])]
        pts_bathy = bathy[['long', 'lat', 'depth']].values

    # Create 3D scatter objects
    if ref_global:
        data3 = mesh_obj(pts_bathy, name="Bathy")
        data4 = scatter_obj(sites, name="Sites", color="orange", size=5)
        fig_dict["data"].append(data3)
        fig_dict["data"].append(data4)

    data1 = scatter_obj(pts_plume, name="Plume", color="blue")
    data2 = scatter_obj(pts_traj, name="Trajectory", color="red", opacity=1.0)
    fig_dict["data"].append(data1)
    fig_dict["data"].append(data2)

    # Fill in each frame of the simulation
    # Start from time 1, since time 0 will have no trajectory by definition
    for i, t in enumerate(times[1:]):
        # Initialize frame
        frame = {"data": [], "name": f"t{t}"}

        # Get plume point cloud
        pts_plume = env.get_pointcloud(t=t)
        if ref_global:
            pts_plume = convert_to_latlon(pts_plume, env.extent.origin)
        pts_traj = coords[coord_times < t]

        # Create 3D scatter objects
        if ref_global:
            data3 = mesh_obj(pts_bathy, name="Bathy")
            data4 = scatter_obj(sites, name="Sites", color="orange", size=5)
            frame["data"].append(data3)
            frame["data"].append(data4)
        data1 = scatter_obj(pts_plume, name="Plume", color="blue")
        data2 = scatter_obj(pts_traj, name="Trajectory",
                            color="red", opacity=1.0)
        frame["data"].append(data1)
        frame["data"].append(data2)

        # Add to frame list
        fig_dict["frames"].append(frame)

    fig_dict["layout"]["sliders"] = _get_sliders(fig_dict)

    # Create plotly figure
    fig = go.Figure(fig_dict)
    return fig


def plot_trajectory_map(trajectory, extent, site="ridge"):
    """Plots bathy, sites, and other relevant information with trajectory object."""
    bathy = get_bathy(extent)
    bathy_fig = plot_bathy_underlay(extent, site=site)
    sites_fig = plot_sites_overlay(extent)

    coords = np.asarray(trajectory.uniformly_sample(0.5))
    easting, northing, z1, z2 = utm.from_latlon(
        extent.origin[0], extent.origin[1])
    xcoords = coords.T[1, :] + easting
    ycoords = coords.T[2, :] + northing

    try:
        lat, lon = utm.to_latlon(xcoords, ycoords, z1, z2)
    except:
        lat, lon = utm.to_latlon(np.float32(
            xcoords), np.float32(ycoords), z1, z2)

    traj_fig = go.Scatter(x=lon, y=lat)
    # fig = go.Figure(data=[bathy_fig, sites_fig, traj_fig], layout=dict(width=900, height=900))
    fig = go.Figure(data=[bathy_fig], layout=dict(width=900, height=900))
    fig.show()


def get_bathy(extent, site='ridge', latlon_provided=False, step=100):
    """Get and window the bathy file."""
    if site == 'ridge':
        bathy_file = os.path.join(
            os.getenv("SENTRY_DATA"), f"bathy/proc/ridge.txt")
        bathy = pd.read_table(
            bathy_file, names=["long", "lat", "depth"]).dropna()
    elif site == 'ring':
        bathy_file = os.path.join(
            os.getenv("SENTRY_DATA"), f"bathy/proc/ring.txt")
        bathy = pd.read_table(
            bathy_file, names=["long", "lat", "depth"]).dropna()
    elif site == 'plain':
        bathy_file = os.path.join(
            os.getenv("SENTRY_DATA"), f"bathy/proc/plain.txt")
        bathy = pd.read_table(
            bathy_file, names=["long", "lat", "depth"]).dropna()
    elif site == 'all':
        ridge_file = os.path.join(
            os.getenv("SENTRY_DATA"), f"bathy/proc/ridge.txt")
        ring_file = os.path.join(
            os.getenv("SENTRY_DATA"), f"bathy/proc/ring.txt")
        ridge_bathy = pd.read_table(
            ridge_file, names=["long", "lat", "depth"]).dropna()
        ring_bathy = pd.read_table(
            ring_file, names=["long", "lat", "depth"]).dropna()
        bathy = pd.concat([ridge_bathy, ring_bathy])
    else:
        return None

    if latlon_provided is False:
        xlim, ylim, zlim = extent_to_lat_lon(extent)
    else:
        xlim = extent.xrange
        ylim = extent.yrange
    bathy = bathy[(bathy['long'] >= xlim[0]) & (bathy['long'] <= xlim[1])]
    bathy = bathy[(bathy['lat'] >= ylim[0]) & (bathy['lat'] <= ylim[1])]
    bathy = bathy[::step]
    return bathy


def plot_bathy_underlay(extent, site="ridge", step=10000, latlon_provided=False):
    """Provides a GO figure for bathy underlaying."""
    bathy = get_bathy(extent, site=site,
                      latlon_provided=latlon_provided, step=step)
    print("Shape of bathy:", bathy.shape)
    x = bathy['long'].values
    y = bathy['lat'].values
    z = bathy['depth'].values
    return go.Contour(x=x, y=y, z=z, colorscale='Viridis', ncontours=15, opacity=0.3)


def get_sites(extent, bathy, latlon_provided=False):
    if latlon_provided is False:
        xrange, yrange, zrange = extent_to_lat_lon(extent)
    else:
        xrange = extent.xrange
        yrange = extent.yrange
    sites_file = os.path.join(os.getenv("SENTRY_DATA"),
                              f"metadata/site_data.yaml")
    sites = yaml.load(open(sites_file), Loader=yaml.FullLoader)

    site_locs = []
    for x in sites.values():
        if x['lon'] <= xrange[1] and x['lon'] >= xrange[0]:
            if x['lat'] <= yrange[1] and x['lat'] >= yrange[0]:
                site_locs.append((x['lon'], x['lat'], 0))
    site_locs = np.array(site_locs).reshape(-1, 3)

    # Find site depth from nearest bathy point
    for i in range(site_locs.shape[0]):
        idx = np.abs(bathy[['long', 'lat']] -
                     site_locs[i, 0:2]).sum(axis=1).idxmin()
        d = bathy.loc[idx, "depth"]
        site_locs[i, 2] = -d
    return site_locs


def plot_sites_overlay(extent, latlon_provided=False):
    """Provides a GO figure for sites overlay."""
    xrange, yrange, zrange = extent_to_lat_lon(extent)
    sites_file = os.path.join(os.getenv("SENTRY_DATA"),
                              f"metadata/site_data.yaml")
    sites = yaml.load(open(sites_file), Loader=yaml.FullLoader)
    site_locs = []
    for x in sites.values():
        if x['lon'] <= xrange[1] and x['lon'] >= xrange[0]:
            if x['lat'] <= yrange[1] and x['lat'] >= yrange[0]:
                site_locs.append((x['lon'], x['lat']))
    site_locs = np.array(site_locs).reshape(-1, 2)
    return go.Scatter(x=site_locs[:, 0], y=site_locs[:, 1], mode="markers")


def plot_sites_underlay(extent, latlon_provided=False):
    """Provides a GO figures for sites underlay."""
    site_locs = get_sites(extent, latlon_provided=latlon_provided)
    return go.Scatter(x=site_locs[:, 0], y=site_locs[:, 1], mode="markers")


def extent_to_lat_lon(extent):
    # Convert extent to latitude and longitude
    lim = np.vstack([[extent.xrange[0], extent.yrange[0], extent.zrange[0]],
                     [extent.xrange[1], extent.yrange[1], extent.zrange[1]]])
    lim = convert_to_latlon(lim, extent.origin)
    xlim = lim[:, 0]
    ylim = lim[:, 1]
    zlim = lim[:, 2]
    return xlim, ylim, zlim


def plot_plume(times, model, title="Plume Dynamics"):
    """Generate 3D scatter plot of plume."""
    # Get plot limits in lat-lon coordinates
    xlim, ylim, zlim = extent_to_lat_lon(model.extent)

    # Generate figure skeleton
    fig_dict = _make_and_configure_fig(
        title, xlim, ylim, zlim)

    # Initial simulation frame; final plume time
    # Draw a vertical slice of the plume envelope
    # model.solve(t=times[-1], overwrite=True)
    # z = model.odesys.z_disp(times[-1])
    # le, cl, re = model.odesys.envelope(t=times[-1])
    # plt.plot(*cl, label="Centerline")
    # plt.plot(*le, label="Left Extent")
    # plt.plot(*re, label="Right Extent")
    # plt.title("Plume Envelope")
    # plt.xlabel("X (meters)")
    # plt.ylabel("Z (meters)")
    # plt.legend()
    # plt.show()

    pts_plume = model.odesys.get_pointcloud(t=times[-1])
    pts_plume = convert_to_latlon(pts_plume, model.extent.origin)

    # Get the bathy data
    bathy = get_bathy(model.extent)
    pts_bathy = bathy[['long', 'lat', 'depth']].values

    # Get the sites data
    sites = get_sites(model.extent, bathy)

    # Create 3D scatter objects
    data1 = mesh_obj(pts_bathy, name="Bathy")
    data2 = scatter_obj(sites, name="Sites", color="orange", size=5)
    fig_dict["data"].append(data1)
    fig_dict["data"].append(data2)

    data3 = scatter_obj(pts_plume, name="Plume", color="blue")
    fig_dict["data"].append(data3)

    # Fill in each frame of the simulation
    for i, t in enumerate(times):
        print(f"Plotting time {t}, {i} of {len(times)}")
        # Initialize frame
        frame = {"data": [], "name": f"t{t}"}

        # Get plume point cloud
        pts_plume = model.odesys.get_pointcloud(t=t)
        pts_plume = convert_to_latlon(pts_plume, model.extent.origin)

        # Create 3D scatter objects
        # data1 = mesh_obj(pts_bathy, name="Bathy")
        # data2 = scatter_obj(sites, name="Sites", color="orange", size=5)
        data3 = scatter_obj(pts_plume, name="Plume", color="blue")
        frame["data"].append(data1)
        frame["data"].append(data2)
        frame["data"].append(data3)

        # Add to frame list
        fig_dict["frames"].append(frame)

    fig_dict["layout"]["sliders"] = _get_sliders(fig_dict)

    # Create plotly figure
    fig = go.Figure(fig_dict)
    return fig


def plot_centerline(times, model, title="Plume Dynamics"):
    """Generate time series of plume centerline."""
    for t in times:
        model.solve(t=t, overwrite=True)
        z = model.odesys.z_disp(t)
        le, cl, re = model.odesys.envelope(t=t)
        plt.plot(*cl, label=f"t={t}")
    plt.title("Plume Evolution")
    plt.xlabel("X (meters)")
    plt.ylabel("Z (meters)")
    plt.legend()
    plt.show()


def visualize_and_save_traj(trajectory, extent, traj_name="temp_traj"):
    """Visualize trajectory in global cooordinates and save path to file.

    Args:
        traj (Trajectory): input Trajectory object
        traj_name (str):  name of the output trajectory object
        extent (Extent): an Extent object for mapping
    """
    REFERENCE = (float(os.getenv("LAT")),
                 float(os.getenv("LON")),
                 float(os.getenv("DEP")))

    EAST_REFERENCE, NORTH_REFERENCE, ZONE_NUM, ZONE_LETT = utm.from_latlon(
        REFERENCE[0], REFERENCE[1])

    with open(traj_name, "wb") as fh:
        pickle.dump(trajectory, fh)

    plot_trajectory_map(trajectory, extent)

    # Convert trajectory to sentry file
    print("Converting trajectory to Sentry mission file...")
    path_x = trajectory.path.xy[0] + EAST_REFERENCE
    path_y = trajectory.path.xy[1] + NORTH_REFERENCE

    # convert to lat lon
    map_lat, map_lon = utm.to_latlon(path_x, path_y, ZONE_NUM, ZONE_LETT)
    save_latlon = np.around(np.asarray([map_lat, map_lon]).T, decimals=5)

    # save file with name of depth
    np.savetxt(os.path.join(os.getenv("SENTRY_DATA"), traj_name),
               save_latlon, delimiter=' ', fmt='%1.5f')


if __name__ == "__main__":
    from fumes.environment.extent import Extent
    REFERENCE = (float(os.getenv("LAT")),
                 float(os.getenv("LON")),
                 float(os.getenv("DEP")))
    extent = Extent(xrange=(0, 2500),
                    xres=50,
                    yrange=(500, 2500),
                    yres=50,
                    zrange=(0, 300),
                    zres=100,
                    global_origin=REFERENCE)
    bathy = get_bathy(extent)
    x = bathy['long'].values
    y = bathy['lat'].values
    z = bathy['depth'].values
    bathy_data = go.Mesh3d(x=x, y=y, z=z,
                           intensity=z,
                           colorscale='Viridis',
                           opacity=0.50,
                           name="Bathy")
    b = go.Figure(data=[bathy_data],
                  layout_title_text="Bathymetry for Site Ridge")
    b.show()
