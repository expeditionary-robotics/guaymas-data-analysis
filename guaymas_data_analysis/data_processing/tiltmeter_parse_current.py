"""Takes in data from tiltmeter to create current objects."""
import os
import sys
import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from guaymas_data_analysis.utils.current_utils import CurrMag, CurrHead, \
    tidalfunction, TimeSeriesInterpolate
from guaymas_data_analysis.utils.metadata_utils import \
    tiltmeter_filename_by_deployment, \
    tiltmeter_time_by_deployment, \
    tiltmeter_time_exclude_by_deployment, \
    tiltmeter_output_by_deployment,  \
    tiltmeter_trainset_by_deployment

DEPLOYMENT = "A1"
VISUALIZE = True

# Dive name
if len(sys.argv) > 1:
    DEPLOYMENT = sys.argv[1]
print(f"Generating dataframes for tiltmeter deployment {DEPLOYMENT}")

DATA_NAMES = tiltmeter_filename_by_deployment(DEPLOYMENT)
START_DATA, END_DATA = tiltmeter_time_by_deployment(DEPLOYMENT)
TRANSIT_START, TRANSIT_END = tiltmeter_time_exclude_by_deployment(DEPLOYMENT)
CUR_FILE = tiltmeter_output_by_deployment(DEPLOYMENT)
CUR_TRAIN_FILE = tiltmeter_trainset_by_deployment(DEPLOYMENT)

dfs = []
for name in DATA_NAMES:
    temp = pd.read_csv(name)
    temp["timestamp"] = pd.to_datetime(temp["ISO 8601 Time"], utc=True)
    dfs.append(temp)

# Get current and temperature dataframes
df1 = dfs[0][["timestamp",
              "Speed (cm/s)", "Heading (degrees)"]].set_index("timestamp")
df2 = dfs[1][["timestamp", "Temperature (C)"]].set_index("timestamp")

start_time = df1.index[0]
df_join = df1.join(df2, on="timestamp", how="outer", lsuffix="1",
                   rsuffix="2").set_index("timestamp").sort_index().dropna(axis=0)
df_join = df_join.rename({"Speed (cm/s)": "mag", "Heading (degrees)": "head",
                          "timestamp": "time", "Temperature (C)": "temperature"}, axis=1)

# Convert current magnitude from cm/s to m/s
df_join.loc[:, "mag_mps"] = df_join["mag"] / 100.

# Heading is in nautical convention (degrees measured from the
# North, clockwise). Convert to radians in standard convention.
# TODO: we removed this and appear to be keeping everying in
# nautical convention? Is this consistant in our code base
df_join.loc[:, "head_rad"] = (90. - df_join["head"]) / 180. * np.pi

# Convert time to hours after midnight to
df_join.loc[:, "hours"] = df_join.index.hour + df_join.index.minute/60.

if START_DATA is not None and END_DATA is not None:
    df_sub = df_join[(df_join.index >= START_DATA)
                     & (df_join.index <= END_DATA)]
else:
    df_sub = df_join

if TRANSIT_START is not None and TRANSIT_END is not None:
    df_sub = df_sub[(df_sub.index <= TRANSIT_START)
                    | (df_sub.index >= TRANSIT_END)]

with open(CUR_FILE, "w+") as fh:
    df_sub[["mag_mps", "head_rad"]].to_csv(fh)

# Group all data points collected in the same hour and minute together,
# relative to the start of the day
df_mean = df_sub.groupby("hours").mean()

# Average and unwrap the heading in radians
df_mean["head_rad"] = df_sub.groupby("hours").apply(
    lambda x: scipy.stats.circmean(x["head_rad"]))
df_mean["head_rad"] = np.unwrap(df_mean["head_rad"])

# Average and unwrap the heading in nautical degrees
df_mean["head_naut"] = df_sub.groupby("hours").apply(
    lambda x: scipy.stats.circmean(x["head"], high=360.))
df_mean["head_naut"] = np.unwrap(df_mean["head_naut"], period=360.)

# Sort index
df_mean.sort_index(inplace=True)

# Write hourly and minute averaged data to file
with open(CUR_TRAIN_FILE, "w+") as fh:
    df_mean[["mag_mps", "head_rad"]].to_csv(fh)

# Let's do some sanity checking
if VISUALIZE:
    fig = make_subplots(rows=4, cols=1)
    fig.add_trace(go.Scatter(x=df_mean.index,
                             y=df_mean["mag_mps"]), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_mean.index,
                  y=df_mean["head_rad"].values), row=2, col=1)
    # Plot the data, pre-averaging
    fig.add_trace(go.Scatter(
        x=df_sub["hours"], y=df_sub["head_rad"]), row=3, col=1)
    if DEPLOYMENT == "A1":
        # We only have defined a tidal function for the A1 deployment times
        fig.add_trace(go.Scatter(
            x=df_sub["hours"], y=tidalfunction(df_sub.index)), row=4, col=1)
    fig.show()

    # Plotly plot
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(df_mean.index, df_mean["mag_mps"])
    ax[0].set_xlabel("Hour and Minutes (UTC)")
    ax[0].set_ylabel("Current Magnitude (mps)")
    ax[1].plot(df_mean.index, df_mean["head_naut"])
    ax[1].set_xlabel("Hour and Minutes (UTC)")
    ax[1].set_ylabel("Current Heading (degrees in nautical convention)")
    plt.show()

    traint = df_mean.index
    trainmag = df_mean["mag_mps"]
    trainhead = df_mean["head_rad"]

    print("Training magnitude GP")
    currmag = CurrMag(traint, trainmag, learning_rate=0.5, training_iter=100)
    plt.plot(traint, currmag.magnitude(None, traint*3600.))
    plt.scatter(traint, trainmag, alpha=0.1, c="g")
    plt.title("Magnitude GP")
    plt.show()

    print("Training magnitude interpolate.")
    int_mag = TimeSeriesInterpolate(traint, trainmag)
    plt.plot(traint, int_mag.magnitude(None, traint*3600.))
    plt.scatter(traint, trainmag, alpha=0.1, c="g")
    plt.title("Magnitude Interpolate")
    plt.show()

    print("Training magnitude spline")
    spl_mag = sp.interpolate.UnivariateSpline(traint, trainmag, s=1)
    plt.scatter(traint, trainmag, alpha=0.1, c="g")
    plt.plot(traint, spl_mag(traint))
    plt.title("Magnitude Spline")
    plt.show()

    print("Training heading GP")
    currhead = CurrHead(traint, trainhead,
                        learning_rate=0.1, training_iter=200)
    plt.plot(traint, currhead.heading(traint*3600.))
    plt.scatter(traint, trainhead % (2.*np.pi), alpha=0.1, c="g")
    plt.title("Heading GP")
    plt.show()

    print("Training heading interpolate.")
    int_mag = TimeSeriesInterpolate(traint, trainhead)
    plt.plot(traint, int_mag.heading(traint*3600.))
    plt.scatter(traint, trainhead % (2.*np.pi), alpha=0.1, c="g")
    plt.title("Heading Interpolate")
    plt.show()

    print("Training heading spline")
    spl_head = sp.interpolate.interp1d(traint, trainhead)
    plt.scatter(traint, trainhead, alpha=0.1, c="g")
    plt.plot(traint, spl_head(traint))
    plt.title("Heading Spline")
    plt.show()

    # query = np.arange(0, 24*3600., step=3600.)
    # print(currmag.magnitude(None, query))
    # TODO: I don't think we need to convert this to degrees
    # print(np.degrees(currhead.heading(query))) % 360.
    # print(currhead.heading(query) % 360.)

    # Define the number of seconds since UTC midnight
    df_sub["seconds"] = df_sub.index.hour * 3600 + \
        df_sub.index.minute * 60 + df_sub.index.second
    fig = make_subplots(rows=5, cols=1)
    fig.add_trace(go.Scatter(
        x=df_sub.index,
        y=df_sub["mag_mps"],
        mode="markers"),
        row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df_sub.index,
        y=currmag.magnitude(
            None, df_sub["seconds"]), mode="markers"),
        row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df_sub.index,
        y=df_sub["head_rad"],
        mode="markers"),
        row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df_sub.index,
        y=currhead.heading(df_sub["seconds"]), mode="markers"),
        row=4, col=1)
    if DEPLOYMENT == "A1":
        # We only have defined a tidal function for the A1 deployment times
        fig.add_trace(go.Scatter(
            x=df_sub.index,
            y=tidalfunction(df_sub.index), mode="markers"),
            row=5, col=1)
    fig.show()

    if DEPLOYMENT == "B1":
        # This long deployment requires additional data subsampling
        VIS_SKIP = 1000
    else:
        VIS_SKIP = 100
    mag = currmag.magnitude(None, df_sub["hours"]*3600.)[::VIS_SKIP]
    head = currhead.heading(df_sub["hours"]*3600.)[::VIS_SKIP]
    quiver = []
    cmin = np.nanmin(df_sub["hours"])
    cmax = np.nanmax(df_sub["hours"])
    for i, hour in enumerate(df_sub["hours"][::VIS_SKIP]):
        quiver.append(
            go.Scatter(x=np.linspace(0, mag[i]*np.cos(head[i]), 500),
                       y=np.linspace(0, mag[i]*np.sin(head[i]), 500),
                       mode="markers",
                       name=str(hour),
                       marker=dict(size=5,
                                   opacity=0.8,
                                   color=hour *
                                   np.ones_like(
                                       np.linspace(0, 1, 500)),
                                   colorscale="Inferno",
                                   cmin=cmin,
                                   cmax=cmax,
                                   colorbar=dict(thickness=30, x=-0.1))))
    fig = go.Figure(data=quiver)
    fig.show()
