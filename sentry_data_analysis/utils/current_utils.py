"""Creates a current class operator to couple with data."""
import numpy as np
import gpytorch as gpy
import torch
import copy
from scipy import interpolate
from datetime import datetime, timezone, timedelta
import pandas as pd

import pdb

from .model_utils import GPSampler, normalize_data, unnormalize_data


class TimeSeriesInterpolate(object):
    """A model object that interpolates a time series object."""

    def __init__(self, t, val):
        """Initialize model.

        Args:
            t (np.array): array of input time points
            val (np.array): array of input values
        """
        self.t = t
        self.t0 = t[0]
        self.val = val
        self.seconds = (t - pd.Timestamp("1970-01-01",
                        tzinfo=(timezone.utc))) // pd.Timedelta("1s")
        self.seconds = self.seconds.astype(np.int32)
        self.cache_model = interpolate.interp1d(
            self.seconds, val, bounds_error=False, fill_value="extrapolate")

    def magnitude(self, z, t):
        """Returns the magnitude function at time t and height z."""
        # To avoid extrapolation, shift the data back by 1 day
        # until the last point falls within the covered range
        while t[-1] > self.t.iloc[-1]:
            t -= pd.Timedelta(days=1)
        seconds = (t - pd.Timestamp("1970-01-01",
                   tzinfo=(timezone.utc))) // pd.Timedelta("1s")
        seconds = seconds.astype(np.int32)
        return self.cache_model(seconds)

    def heading(self, t):
        """Returns the heading function at time t."""
        while t[-1] > self.t.iloc[-1]:
            t -= pd.Timedelta(days=1)
        seconds = (t - pd.Timestamp("1970-01-01",
                   tzinfo=(timezone.utc))) // pd.Timedelta("1s")
        seconds = seconds.astype(np.int32)
        return self.cache_model(seconds) % (2.*np.pi)


class CurrMag(GPSampler):
    """Creates a current magnitude operator which inherits from GPSampler."""

    def magnitude(self, z, t):
        """Returns MLE functional for use in model class."""
        # Convert time to fractional hours from UTC 00:00:00
        t = t / 3600.
        t = t % 24

        return self.cache_model(t)

    def sample_magnitude(self, num_samples):
        # trainx = copy.copy(self.trainx)
        # trainx = trainx.cpu().numpy()
        # return [interpolate.interp1d(self.trainx.cpu().numpy(), self.sample(self.trainx.cpu().numpy(), 1), bounds_error=False, fill_value="extrapolate") for i in range(num_samples)]
        # return [interpolate.interp1d(self.trainx.cpu().numpy(), self.sample(self.trainx.cpu().numpy(), 1), bounds_error=False, fill_value="extrapolate") for i in range(num_samples)]
        try:
            return [interpolate.interp1d(self.trainx.cpu().numpy(), self.sample(self.trainx.cpu().numpy(), 1), bounds_error=False, fill_value="extrapolate") for i in range(num_samples)]
        except:
            return [interpolate.interp1d(self.trainx, self.sample(self.trainx, 1), bounds_error=False, fill_value="extrapolate") for i in range(num_samples)]


class CurrHead(GPSampler):
    """Creates a current heading operator which inherits from GPSampler."""

    def heading(self, t):
        """Returns MLE functional for use in model class."""
        # Convert time to fractional hours from UTC 00:00:00
        t = t / 3600
        t = t % 24
        return self.cache_model(t) % (2.*np.pi)

    def sample_heading(self, num_samples):
        # return [interpolate.interp1d(self.trainx, self.sample(self.trainx, 1), bounds_error=False, fill_value="extrapolate") for i in range(num_samples)]
        try:
            return [interpolate.interp1d(self.trainx.cpu().numpy(), self.sample(self.trainx.cpu().numpy(), 1), bounds_error=False, fill_value="extrapolate") for i in range(num_samples)]
        except:
            return [interpolate.interp1d(self.trainx, self.sample(self.trainx, 1), bounds_error=False, fill_value="extrapolate") for i in range(num_samples)]


def tidalfunction(time):
    """Describes the tides according to CICESE tide chart.

    These values were estimated by hand from the file guy2111.pdf
    """
    # Set the tidal values for known location
    val_18 = [0.5,  # at start time
              0.48,
              0.45,
              0.48,  # 3AM
              0.5,
              0.6,
              0.7,  # 6AM local
              0.72,
              0.73,
              0.72,  # 9AM local
              0.6,
              0.5,
              0.25,  # 12PM local
              0.0,
              -0.2,
              -0.25,  # 3PM local
              -0.23,
              -0.15,
              0.0,  # 6PM local
              0.15,
              0.25,
              0.48,  # 9PM local
              0.55,
              0.60]

    val_19 = [0.58,  # midnight local 19th
              0.56,
              0.54,
              0.53,  # 3
              0.55,
              0.60,
              0.75,  # 6
              0.8,
              0.8,
              0.75,  # 9
              0.65,
              0.6,
              0.45,  # 12
              0.3,
              -0.05,
              -0.2,  # 15
              -0.3,
              -0.25,
              -0.15,  # 18
              -0.05,
              0.1,
              0.4,  # 21
              0.5,
              0.58]

    val_20 = [0.6,  # midnight local 20th
              0.6,
              0.6,
              0.6,  # 3
              0.62,
              0.68,
              0.72,  # 6
              0.75,
              0.77,
              0.75,  # 9
              0.68,
              0.6,
              0.5,  # 12
              0.25,
              0.0,
              -0.2,  # 15
              -0.25,
              -0.25,
              -0.2,  # 18
              -0.1,
              0.0,
              0.2,  # 21
              0.35,
              0.5]

    # compute timedelta from midnight local Nov 18th
    start_date = datetime(2021, 11, 18, 8, 0, 0,
                          tzinfo=(timezone.utc))  # 8AM UTC
    val_18_times = [start_date + timedelta(hours=i)
                    for i, v in enumerate(val_18)]
    val_19_times = [val_18_times[-1] +
                    timedelta(hours=i+1) for i, v in enumerate(val_19)]
    val_20_times = [val_19_times[-1] +
                    timedelta(hours=i+1) for i, v in enumerate(val_20)]

    all_vals = val_18 + val_19 + val_20
    all_times = val_18_times + val_19_times + val_20_times

    all_time_from_start = [(t - start_date).total_seconds() for t in all_times]
    time_interp = interpolate.interp1d(all_time_from_start, all_vals)
    query_time = time
    query_x = (query_time - start_date).total_seconds()
    return time_interp(query_x)
