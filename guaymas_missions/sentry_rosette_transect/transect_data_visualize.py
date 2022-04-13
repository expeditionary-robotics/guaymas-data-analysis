"""Reads in transect data and performs simple visual comparisons."""

import os
import utm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from transect_utils import get_transect_rosette_sage_path, \
    get_transect_sentry_nopp_path, get_transect_bottles_path


def get_bathy(rsamp=0.5):
    """Get and window the bathy file."""
    bathy_file = os.path.join(os.getenv("SENTRY_DATA"),
                              "bathy/proc/ridge.txt")
    bathy = pd.read_table(bathy_file, names=["long", "lat", "depth"]).dropna()
    return bathy.sample(frac=rsamp, random_state=1)


def create_2d_plot(x, y, c, cmin=None, cmax=None, cmap="Inferno", s=5, o=0.7, cbar_loc=-0.1, name="data", cbar_name="data"):
    """Wrapper for defining 2D plotly scatter plot."""
    return go.Scatter(x=x,
                      y=y,
                      mode="markers",
                      marker=dict(size=s,
                                  color=c,
                                  opacity=o,
                                  cmin=cmin,
                                  cmax=cmax,
                                  colorscale=cmap,
                                  colorbar=dict(thickness=10, x=cbar_loc, title=cbar_name)),
                      name=name)


def create_3d_plot(x, y, z, c, cmin=None, cmax=None, cmap="Inferno", s=5, o=0.7, cbar_loc=-0.1, name="data", cbar_name="data"):
    """Wrapper for defining 3d plotly scatter plot."""
    return go.Scatter3d(x=x,
                        y=y,
                        z=z,
                        mode="markers",
                        marker=dict(size=s,
                                    color=c,
                                    opacity=o,
                                    cmin=cmin,
                                    cmax=cmax,
                                    colorscale=cmap,
                                    colorbar=dict(thickness=10, x=cbar_loc, title=cbar_name)),
                        name=name)


SENTRY_NOPP = get_transect_sentry_nopp_path()
BOTTLES = get_transect_bottles_path()
ROSETTE_SAGE = get_transect_rosette_sage_path()

# For 2d plots, what dimension to plot on x-axis
OVER_TIME = True
OVER_DISTANCE = True
OVER_LONGITUDE = True
OVER_LATITUDE = True

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
PAIRED_VARS = {"Oxygen": ("O2", "o2_umol_kg"),
               "Turbidity": ("obs", "beam_attenuation"),
               "Methane": ("nopp_fundamental", "sage_methane_ppm"),
               "Temperature": ("potential_temp", "pot_temp_C_its90"),
               "Salinity": ("practical_salinity", "prac_salinity")}

# Whether to visualize the contents of each input file
VISUALIZE_SENTRY_NOPP = False
VISUALIZE_ROSETTE_SAGE = False
VISUALIZE_ALL_PLATFORMS = True
VISUALIZE_ROSETTE_SAGE_AND_BOTTLES = False

if __name__ == "__main__":
    # Get all of the data
    scc_df = pd.read_csv(SENTRY_NOPP)
    scc_df['timestamp'] = pd.to_datetime(scc_df['timestamp'])
    bott_df = pd.read_csv(BOTTLES)
    bott_df['datetime'] = pd.to_datetime(bott_df['datetime'])
    ros_df = pd.read_csv(ROSETTE_SAGE)
    ros_df['datetime'] = pd.to_datetime(ros_df['datetime'])

    # Set up plotting axes meta data
    plot_xlabels_ros = []
    plot_xlabels_bott = []
    plot_xlabels_scc = []
    plot_xlabel_titles = []
    if OVER_TIME is True:
        plot_xlabels_ros.append("datetime")
        plot_xlabels_bott.append("datetime")
        plot_xlabels_scc.append("timestamp")
        plot_xlabel_titles.append("Time")
    if OVER_DISTANCE is True:
        plot_xlabels_ros.append("ridge_distance")
        plot_xlabels_bott.append("ridge_distance")
        plot_xlabels_scc.append("ridge_distance")
        plot_xlabel_titles.append("Distance from Ridge (m)")
    if OVER_LATITUDE is True:
        plot_xlabels_ros.append("usbl_lat")
        plot_xlabels_bott.append("lat")
        plot_xlabels_scc.append("lat")
        plot_xlabel_titles.append("Latitude")
    if OVER_LONGITUDE is True:
        plot_xlabels_ros.append("usbl_lon")
        plot_xlabels_bott.append("lon")
        plot_xlabels_scc.append("lon")
        plot_xlabel_titles.append("Longitude")

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
    x, y, _, _ = utm.from_latlon(bathy.lat.values, bathy.long.values)
    bathy_plot_3d = go.Mesh3d(x=x, y=y, z=bathy.depth,
                              intensity=bathy.depth,
                              colorscale='Viridis',
                              opacity=0.50,
                              name="Bathy")

    if VISUALIZE_ROSETTE_SAGE_AND_BOTTLES is True:
        # Only grab the second leg of the transect
        temp_ros_df = ros_df[ros_df['datetime'] >
                             pd.Timestamp("2021-11-30 08:32:03")]
        # Create 2D plots
        for rx, bx, tx in zip(plot_xlabels_ros, plot_xlabels_bott, plot_xlabel_titles):
            plt.scatter(
                temp_ros_df[rx], temp_ros_df["sage_methane_ppm"], label="SAGE")
            plt.scatter(bott_df[bx], bott_df["GGA Methane"],
                        label="GGA, Raw PPM")
            plt.scatter(
                bott_df[bx], bott_df["ch4_ppm_corr_05"], label="GGA, 0.05 Eff.")
            plt.scatter(
                bott_df[bx], bott_df["ch4_ppm_corr_15"], label="GGA, 0.15 Eff.")
            plt.vlines(bott_df[bx], ymin=bott_df["GGA Methane"],
                       ymax=bott_df["ch4_ppm_corr_05"], color="blue", label="GGA Range")
            plt.xlabel(tx)
            plt.ylabel("Methane, PPM")
            plt.legend()
            pltname = os.path.join(os.getenv("SENTRY_OUTPUT"),
                                   f"transect/figures/gga_sage_over_{rx}.png")
            plt.savefig(pltname)
            plt.close()

            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            ax1.set_xlabel(tx)
            ax1.set_ylabel('Methane, PPM', color="blue")
            ax1.scatter(
                temp_ros_df[rx], temp_ros_df["sage_methane_ppm"], label="HCF", c="cyan")
            ax1.vlines(x=bott_df[bx], ymin=bott_df["GGA Methane"],
                       ymax=bott_df["ch4_ppm_corr_05"], color="blue", label="GGA Range")
            ax1.scatter(x=bott_df[bx], y=bott_df["GGA Methane"],
                        color="blue", label="GGA Range")
            ax1.scatter(
                x=bott_df[bx], y=bott_df["ch4_ppm_corr_05"], color="blue", label="GGA Range")
            ax1.tick_params(axis='y', labelcolor="blue")
            ax2.set_ylabel("NH4, nM", color="orange")
            ax2.scatter(x=bott_df[bx], y=bott_df["[NH4] (nM)"],
                        label="NH4", color="orange")
            ax2.tick_params(axis='y', labelcolor="orange")
            fig.tight_layout()
            pltname = os.path.join(os.getenv("SENTRY_OUTPUT"),
                                   f"transect/figures/nh4_gga_sage_over_{rx}.png")
            plt.savefig(pltname)
            plt.close()

        # Create spatial plot with methane and nh4 comparisons
        sage_plot = create_2d_plot(x=temp_ros_df['usbl_lon'],
                                   y=temp_ros_df['usbl_lat'],
                                   c=temp_ros_df['sage_methane_ppm'],
                                   cbar_loc=-0.15,
                                   name="SAGE Methane (PPM)",
                                   cbar_name="SAGE")
        sx, sy, _, _ = utm.from_latlon(
            temp_ros_df.usbl_lat.values, temp_ros_df.usbl_lon.values)
        sage_plot_3d = create_3d_plot(x=sx,
                                      y=sy,
                                      z=-temp_ros_df['depth_m'],
                                      c=temp_ros_df['sage_methane_ppm'],
                                      cbar_loc=-0.15,
                                      name="SAGE Methane (PPM)",
                                      cbar_name="SAGE")

        gga_plot = create_2d_plot(x=bott_df['lon'],
                                  y=bott_df['lat'],
                                  c=bott_df['GGA Methane'],
                                  s=20,
                                  o=1.0,
                                  name="GGA Raw PPM",
                                  cbar_name="GGA")
        gx, gy, _, _ = utm.from_latlon(bott_df.lat.values, bott_df.lon.values)
        gga_plot_3d = create_3d_plot(x=gx,
                                     y=gy,
                                     z=bott_df['depth'],
                                     c=bott_df['GGA Methane'],
                                     s=20,
                                     o=1.0,
                                     name="GGA Raw PPM",
                                     cbar_name="GGA")

        nh4_plot = create_2d_plot(x=bott_df['lon'],
                                  y=bott_df['lat'],
                                  c=bott_df['[NH4] (nM)'],
                                  s=20,
                                  o=1.0,
                                  name="NH4 (nM)",
                                  cbar_name="NH4")
        nh4_plot_3d = create_3d_plot(x=gx,
                                     y=gy,
                                     z=bott_df['depth'],
                                     c=bott_df['[NH4] (nM)'],
                                     s=20,
                                     o=1.0,
                                     name="NH4 (nM)",
                                     cbar_name="NH4")

        gga_fig = go.Figure(
            data=[bathy_plot, sage_plot, gga_plot], layout_title_text="SAGE and GGA")
        gga_fig.update_yaxes(scaleanchor="x", scaleratio=1)
        gga_fig.write_html(os.path.join(
            os.getenv("SENTRY_OUTPUT"), "transect/figures/gga_sage_bathy.html"))
        gga_fig.write_image(os.path.join(
            os.getenv("SENTRY_OUTPUT"), "transect/figures/gga_sage_bathy.png"), width=1500)

        gga_fig = go.Figure(
            data=[bathy_plot_3d, sage_plot_3d, gga_plot_3d], layout_title_text="SAGE and GGA", layout_scene_aspectmode="data")
        gga_fig.write_html(os.path.join(
            os.getenv("SENTRY_OUTPUT"), "transect/figures/gga_sage_bathy_3d.html"))

        nh4_fig = go.Figure(
            data=[bathy_plot, sage_plot, nh4_plot], layout_title_text="SAGE and NH4")
        nh4_fig.update_yaxes(scaleanchor="x", scaleratio=1)
        nh4_fig.write_html(os.path.join(
            os.getenv("SENTRY_OUTPUT"), "transect/figures/nh4_sage_bathy.html"))
        nh4_fig.write_image(os.path.join(
            os.getenv("SENTRY_OUTPUT"), "transect/figures/nh4_sage_bathy.png"), width=1500)

        nh4_fig = go.Figure(
            data=[bathy_plot_3d, sage_plot_3d, nh4_plot_3d], layout_title_text="SAGE and NH4", layout_scene_aspectmode="data")
        nh4_fig.write_html(os.path.join(
            os.getenv("SENTRY_OUTPUT"), "transect/figures/nh4_sage_bathy_3d.html"))

    if VISUALIZE_ROSETTE_SAGE is True:
        sx, sy, _, _ = utm.from_latlon(
            ros_df.usbl_lat.values, ros_df.usbl_lon.values)

        # create 2D plots
        for rx, tx in zip(plot_xlabels_ros, plot_xlabel_titles):
            fig = make_subplots(rows=len(ROSETTE_SAGE_VARS), cols=1, shared_xaxes=True,
                                subplot_titles=tuple(ROSETTE_SAGE_LABELS))
            for i, v in enumerate(ROSETTE_SAGE_VARS):
                var = ros_df[v]
                cmin, cmax = np.nanpercentile(
                    var, 10), np.nanpercentile(var, 90)
                fig.add_trace(go.Scatter(x=ros_df[rx], y=var), row=i+1, col=1)
            fig.write_image(os.path.join(os.getenv("SENTRY_OUTPUT"),
                            f"transect/figures/rosette_sage_{tx}.png"), width=1500)
            fig.write_html(os.path.join(os.getenv("SENTRY_OUTPUT"),
                           f"transect/figures/rosette_sage_{tx}.html"))

        # create spatial plots
        for i, v in enumerate(ROSETTE_SAGE_VARS):
            var = ros_df[v]
            cmin, cmax = np.nanpercentile(var, 10), np.nanpercentile(var, 90)
            var_plot = create_2d_plot(x=ros_df['usbl_lon'],
                                      y=ros_df['usbl_lat'],
                                      c=var,
                                      cmin=cmin,
                                      cmax=cmax,
                                      name=ROSETTE_SAGE_LABELS[i],
                                      cbar_name=ROSETTE_SAGE_LABELS[i])
            f = go.Figure(data=[bathy_plot, var_plot],
                          layout_title_text=ROSETTE_SAGE_LABELS[i])
            f.update_yaxes(scaleanchor="x", scaleratio=1)
            f.write_image(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                       f"transect/figures/rosette_sage_bathy_{v}.png"), width=1500)

            var_plot_3d = create_3d_plot(x=sx,
                                         y=sy,
                                         z=-ros_df['depth_m'],
                                         c=var,
                                         cmin=cmin,
                                         cmax=cmax,
                                         name=ROSETTE_SAGE_LABELS[i],
                                         cbar_name=ROSETTE_SAGE_LABELS[i])
            f = go.Figure(data=[bathy_plot_3d, var_plot_3d],
                          layout_title_text=ROSETTE_SAGE_LABELS[i],
                          layout_scene_aspectmode="data")
            f.write_html(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                      f"transect/figures/rosette_sage_bathy_{v}.html"))

    if VISUALIZE_SENTRY_NOPP is True:
        sx, sy, _, _ = utm.from_latlon(
            scc_df.lat.values, scc_df.lon.values)

        # create 2D plots
        for rx, tx in zip(plot_xlabels_scc, plot_xlabel_titles):
            fig = make_subplots(rows=len(SENTRY_NOPP_VARS), cols=1, shared_xaxes=True,
                                subplot_titles=tuple(SENTRY_NOPP_LABELS))
            for i, v in enumerate(SENTRY_NOPP_VARS):
                var = scc_df[v]
                if v is not "obs":
                    cmin, cmax = np.nanpercentile(
                        var, 10), np.nanpercentile(var, 90)
                else:
                    cmin, cmax = 0.0, 0.2
                fig.add_trace(go.Scatter(x=scc_df[rx], y=var), row=i+1, col=1)
            fig.write_image(os.path.join(os.getenv("SENTRY_OUTPUT"),
                            f"transect/figures/sentry_nopp_{tx}.png"), width=1500)
            fig.write_html(os.path.join(os.getenv("SENTRY_OUTPUT"),
                           f"transect/figures/sentry_nopp_{tx}.html"))

        # create spatial 2D slice plots, with altimeter
        alt_fig = create_2d_plot(x=scc_df['ridge_distance'],
                                 y=-scc_df['depth'] - scc_df['height'],
                                 c="Gray",
                                 o=0.1,
                                 cmap=None,
                                 cbar_loc=-0.15,
                                 name="Bottom from Altimeter")
        for i, v in enumerate(SENTRY_NOPP_VARS):
            var = scc_df[v]
            if v is not "obs":
                cmin, cmax = np.nanpercentile(
                    var, 10), np.nanpercentile(var, 90)
            else:
                cmin, cmax = 0.0, 0.2
            spat_fig = create_2d_plot(x=scc_df['ridge_distance'],
                                      y=-scc_df['depth'],
                                      c=var,
                                      cmap="Inferno",
                                      cmin=cmin,
                                      cmax=cmax,
                                      name=SENTRY_NOPP_LABELS[i],
                                      cbar_name=SENTRY_NOPP_LABELS[i])
            fig = go.Figure(data=[alt_fig, spat_fig],
                            layout_title_text=SENTRY_NOPP_LABELS[i])
            fig.write_image(os.path.join(os.getenv("SENTRY_OUTPUT"),
                            f"transect/figures/sentry_nopp_alt_slice_{tx}_{v}.png"), width=1500)
            fig.write_html(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                        f"transect/figures/sentry_nopp_alt_slice_{tx}_{v}.html"))

        # create spatial plots
        for i, v in enumerate(SENTRY_NOPP_VARS):
            var = scc_df[v]
            if v is not "obs":
                cmin, cmax = np.nanpercentile(
                    var, 10), np.nanpercentile(var, 90)
            else:
                cmin, cmax = 0.0, 0.2
            var_plot = create_2d_plot(x=scc_df['lon'],
                                      y=scc_df['lat'],
                                      c=var,
                                      cmin=cmin,
                                      cmax=cmax,
                                      name=SENTRY_NOPP_LABELS[i],
                                      cbar_name=SENTRY_NOPP_LABELS[i])
            f = go.Figure(data=[bathy_plot, var_plot],
                          layout_title_text=SENTRY_NOPP_LABELS[i])
            f.update_yaxes(scaleanchor="x", scaleratio=1)
            f.write_image(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                       f"transect/figures/sentry_nopp_bathy_{v}.png"), width=1500)

            var_plot_3d = create_3d_plot(x=sx,
                                         y=sy,
                                         z=-scc_df['depth'],
                                         c=var,
                                         cmin=cmin,
                                         cmax=cmax,
                                         name=SENTRY_NOPP_LABELS[i],
                                         cbar_name=SENTRY_NOPP_LABELS[i])
            f = go.Figure(data=[bathy_plot_3d, var_plot_3d],
                          layout_title_text=SENTRY_NOPP_LABELS[i],
                          layout_scene_aspectmode="data")
            f.write_html(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                      f"transect/figures/sentry_nopp_bathy_{v}.html"))

    if VISUALIZE_ALL_PLATFORMS is True:
        # Create comparison plots
        sx, sy, _, _ = utm.from_latlon(
            scc_df.lat.values, scc_df.lon.values)
        rx, ry, _, _ = utm.from_latlon(
            ros_df.usbl_lat.values, ros_df.usbl_lon.values)

        # create 2D plots
        for sccx, rosx, titlex in zip(plot_xlabels_scc, plot_xlabels_ros, plot_xlabel_titles):
            for k, v in PAIRED_VARS.items():
                scc_var = scc_df[v[0]]
                if v[0] is not "obs":
                    sccmin, sccmax = np.nanpercentile(
                        scc_var, 10), np.nanpercentile(scc_var, 90)
                else:
                    sccmin, sccmax = 0.0, 0.2
                ros_var = ros_df[v[1]]
                rosmin, rosmax = np.nanpercentile(
                    ros_var, 10), np.nanpercentile(ros_var, 90)

                cmap = "Inferno"
                if v[0] is "nopp_fundamental":
                    cmap = "Inferno_r"
                scc_fig = create_2d_plot(x=scc_df[sccx],
                                         y=-scc_df['depth'],
                                         c=scc_var,
                                         cmap=cmap,
                                         cmin=sccmin,
                                         cmax=sccmax,
                                         name=f"Sentry_{k}",
                                         cbar_name=f"Sentry")
                ros_fig = create_2d_plot(x=ros_df[rosx],
                                         y=-ros_df['depth_m'],
                                         c=ros_var,
                                         cmap="Inferno",
                                         cmin=rosmin,
                                         cmax=rosmax,
                                         name=f"Rosette_{k}",
                                         cbar_name=f"Rosette",
                                         cbar_loc=-0.15)
                fig = go.Figure(data=[scc_fig, ros_fig],
                                layout_title_text=f"Sentry and Rosette over {titlex}: {k}")
                fig.write_image(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                             f"transect/figures/all_over_{titlex}_{k}.png"), width=1500)
                fig.write_html(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                            f"transect/figures/all_over_{titlex}_{k}.html"))

        # create spatial plots
        for k, v in PAIRED_VARS.items():
            scc_var = scc_df[v[0]]
            if v[0] is not "obs":
                sccmin, sccmax = np.nanpercentile(
                    scc_var, 10), np.nanpercentile(scc_var, 90)
            else:
                sccmin, sccmax = 0.0, 0.2
            ros_var = ros_df[v[1]]
            rosmin, rosmax = np.nanpercentile(
                ros_var, 10), np.nanpercentile(ros_var, 90)

            cmap = "Inferno"
            if v[0] is "nopp_fundamental":
                cmap = "Inferno_r"
            sccvar_plot = create_2d_plot(x=scc_df['lon'],
                                         y=scc_df['lat'],
                                         c=scc_var,
                                         cmin=sccmin,
                                         cmax=sccmax,
                                         name=f"Sentry_{k}",
                                         cbar_name=f"Sentry",
                                         cbar_loc=-0.15)
            rosvar_plot = create_2d_plot(x=ros_df['usbl_lon'],
                                         y=ros_df['usbl_lat'],
                                         c=ros_var,
                                         cmin=rosmin,
                                         cmax=rosmax,
                                         name=f"Rosette_{k}",
                                         cbar_name=f"Rosette",
                                         cbar_loc=-0.2)
            f = go.Figure(data=[bathy_plot, sccvar_plot, rosvar_plot],
                          layout_title_text=f"Sentry and Rosette: {k}")
            f.update_yaxes(scaleanchor="x", scaleratio=1)
            f.write_image(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                       f"transect/figures/all_bathy_{k}.png"), width=1500)

            sccvar_plot_3d = create_3d_plot(x=sx,
                                            y=sy,
                                            z=-scc_df['depth'],
                                            c=scc_var,
                                            cmin=sccmin,
                                            cmax=sccmax,
                                            name=f"Sentry_{k}",
                                            cbar_name=f"Sentry",
                                            cbar_loc=-0.15)
            rosvar_plot_3d = create_3d_plot(x=rx,
                                            y=ry,
                                            z=-ros_df['depth_m'],
                                            c=ros_var,
                                            cmin=rosmin,
                                            cmax=rosmax,
                                            name=f"Rosette_{k}",
                                            cbar_name=f"Rosette",
                                            cbar_loc=-0.2)
            f = go.Figure(data=[bathy_plot_3d, sccvar_plot_3d, rosvar_plot_3d],
                          layout_title_text=f"Sentry and Rosette: {k}",
                          layout_scene_aspectmode="data")
            f.write_html(os.path.join(os.getenv("SENTRY_OUTPUT"),
                                      f"transect/figures/all_bathy_{k}.html"))
