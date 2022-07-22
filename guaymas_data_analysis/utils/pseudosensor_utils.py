"""Utility file for the binary pseudo-sensor."""
import json
from typing import List

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from utm.conversion import R


# Which scientific data to use in the analysis
SENTRY_DIVE_ALL_KEYS = [
    "practical_salinity",
    "potential_temp",
    "potential_density",
    "O2",  # oxygen sensor
    "obs",  # optical backscatter
    "dorpdt",  # d/dt(redox potential)
]

# TODO: what do these mean?
BINARY_SENSOR_MODES = [
    "meanstd",
    "meanstd_windowed",
    "percentiles",
]


class BinarySensor():
    """Ingests a dataframe of sensor data. Detects anomalies based on either
    deviations from the average or percentiles and consensus among a window of a
    number of sensors. Essentially creates an upper and lower bound of expected
    ranges for a sensor stream.

    Args:
        sensor_keys (List[str]): sensors to use from a datastream for classification. All
            potential keys defined in SENTRY_DIVE_ALL_KEYS
        num_sensors_consensus (int): number of sensors that must be anomalous for a positive id
        sensor_weights (List[flost]): how much to "count" a measurement
        window_width (int): "window_width // 2" samples before and after an anomalous value will also
            be considered anomalous for each sensor stream
        mode (str): one of BINARY_SENSOR_MODES that determines the anomaly detection for a single data stream
        num_stddev (int): if mode is "meanstd", number of stddevs outside of which data is anomalous
        calibration_window (int): is mode is "meanstd_windowed", the size of the chunks to calibrate
        percentile_lower (int): if mode is "percentiles", lower percentile below which data is anomalous
        percentile_upper (int): if mode is "percentiles", upper percentile above which data is anomalous
    """

    def __init__(
        self,
        sensor_keys: List[str] = SENTRY_DIVE_ALL_KEYS,
        num_sensors_consensus: int = len(SENTRY_DIVE_ALL_KEYS) // 2,
        sensor_weights: List[float] = [1 for m in SENTRY_DIVE_ALL_KEYS],
        window_width: int = 10,
        mode: str = "meanstd",
        num_stddev: int = 2,
        calibration_window: int = 1000,
        percentile_lower: int = 1,
        percentile_upper: int = 99
    ):
        self._sensor_keys = sensor_keys
        self._num_sensors_consensus = num_sensors_consensus
        self._sensor_weights = sensor_weights
        self._window_width = window_width

        self._is_calibrated = False
        self._calibrated_vals = {}

        assert mode in BINARY_SENSOR_MODES
        self._mode = mode

        self._calibration_window = calibration_window
        self._numstddev = num_stddev
        self._percentile_lower = percentile_lower
        self._percentile_upper = percentile_upper

        super().__init__()

    def _window_dfs(self, df: pd.DataFrame):
        """Take a dataframe and return a list of frame chunks"""
        return [df[i:i+self._calibration_window] for i in self._calibration_window]

    def get_single_datastream_limits(self, df: pd.DataFrame):
        """ obtain upper and lower bounds for data based off
            the sensor configuration

        Args:
            df (pd.DataFrame): sensor data series

        Returns:
            min, max (int, int): bounds within which data is not anomalous
        """
        if self._mode == "meanstd":
            mean = np.mean(df)
            std = np.std(df)

            min = mean - self._numstddev * std
            max = mean + self._numstddev * std

        elif self._mode == "percentiles":
            min, max = np.percentile(
                df, [self._percentile_lower, self._percentile_upper]
            )

        elif self._mode == "meanstd_windowed":
            window_means = [np.mean(chunk) for chunk in self._window_df(df)]
            window_stds = [np.std(chunk) for chunk in self._window_df(df)]

            min = window_means - self._numstddev * window_stds
            max = window_means + self._numstddev * window_stds

        else:
            assert False  # shouldn't be possible

        return min, max

    def calibrate_from_df_csv(self, df: pd.DataFrame):
        """determine normal bounds for data based on sensor configuration

        Args:
            df (pd.DataFrame): sensor data, assumed to have keys of interest for
                different datastreams
        """
        start_idx, end_idx = 0, len(df)

        for k in self._sensor_keys:
            min, max = self.get_single_datastream_limits(
                df[k][start_idx:end_idx])
            self._calibrated_vals[k] = {
                "min": min,
                "max": max,
            }
        self._is_calibrated = True

    def classify_samples(
        self, df: pd.DataFrame, write_to_disk: bool = False, output_file: str = None
    ):
        """classify samples using a calibrated sensor

        Args:
            df (pd.DataFrame): sensor data, assumed to have keys of interest for
                different datastreams
            write_to_disk (bool): if binary classifications should be written to disk
            output_file (str): output file path as a string
        """
        assert self._is_calibrated
        for k in self._sensor_keys:
            assert k in df.keys()

        # for each individual sensor stream, detect anomalies
        num_anomalous_sensors = np.zeros((len(df)))
        for i, k in enumerate(self._sensor_keys):
            anomaly_idxs = np.where(
                (df[k] > self._calibrated_vals[k]["max"])
                | (df[k] < self._calibrated_vals[k]["min"])
            )[0]

            windowed_anomaly_idxs = np.zeros((len(df)))
            for idx in anomaly_idxs:
                if idx > 10 and idx < len(df) - self._window_width // 2:
                    windowed_anomaly_idxs[
                        idx - self._window_width // 2: idx + self._window_width // 2
                    ] = 1

            num_anomalous_sensors[windowed_anomaly_idxs ==
                                  1] += 1 * self._sensor_weights[i]

        # aggregate different sensors
        detections = np.zeros((len(df)))
        detections[num_anomalous_sensors >= self._num_sensors_consensus] = 1

        # Add a detection column to the dataframe
        df_return = df.copy()
        df_return["detections"] = detections

        # write to disk
        if write_to_disk and output_file is not None:
            json_config_dict = {
                "detection_algo": {
                    "mode": self._mode,
                    "meanstd_num_stds": self._numstddev,
                    "percentile_num_lowerbound": self._percentile_lower,
                    "percentile_num_upperbound": self._percentile_upper,
                },
                "num_sensors_consensus_required": self._num_sensors_consensus,
                "anomaly_window_size": self._window_width,
                "do_ignore_ascent_descent": self._filter_ascent_descent,
                "sensorstream_min_max": self._calibrated_vals,
            }

            json_output_file = f"{output_file.split('.')[0]}.json"
            j_fp = open(json_output_file, 'w')
            json.dump(json_config_dict, j_fp)
            j_fp.close()

            # write_df = df[["timestamp", "lat", "lon", "northing",
            #                "easting", "depth", "height"]].copy()
            # write_df["detections"] = detections.astype(np.uint8)
            df_return.to_csv(output_file)
            print(f"Detections written to {output_file}")
            print(f"JSON config info written to {json_output_file}")

        return df_return


class SciencePseudoSensor(object):
    """Classifies multi-sensor detections as hydrothermal fluid."""

    def __init__(self, sensors, treatments, weights, num_corroboration):
        """Initializes the class.

        Input:
            sensors (list[string]): sensor targets in df
            treatments (list[dict]): how to treat each sensor data stream
            weights (list[floats]): weight used for corroboration
            num_corroboration (int): total number of sensors used for corroboration
        """

        # Inputs
        self.source_df = DataFrame()
        self.sensors = sensors
        self.treatments = treatments
        self.weights = weights
        self.num_corroboration = num_corroboration
        

    def get_detections(self, df):
        """Returns a dataframe with binary signal."""
        self.source_df = df
        self.detection_df = DataFrame()
        self._apply_treatments()
        self._compute_corroboration()
        return self.detection_df

    def _apply_treatments(self):
        """Applies treatments to each datastream."""
        for sensor, treatment in zip(self.sensors, self.treatments):
            data = self.source_df[sensor]
            self.detection_df.loc[:, sensor] = data
            if treatment["method"] is "threshold":
                if treatment["direction"] is "<":
                    self.detection_df.loc[:,
                                          f"{sensor}_treatment"] = [int(x < treatment["threshold"]) for x in data]
                elif treatment["direction"] is ">":
                    self.detection_df.loc[:,
                                          f"{sensor}_treatment"] = [int(x > treatment["threshold"]) for x in data]
                else:
                    raise ValueError(
                        "Need to specify a direction to apply threshold, > or <.")
            elif treatment["method"] is "meanstd" or treatment["method"] is "percentile":
                mini, maxi = self._compute_stream_limits(data, treatment)
                self.detection_df.loc[:, f"{sensor}_treatment"] = [
                    int(x > maxi or x < mini) for x in data]
            elif treatment["method"] is "meanstd_window":
                mini, maxi = self._compute_stream_limits(data, treatment)
                self.detection_df.loc[:, f"{sensor}_treatment"] = [
                    int(x > maxi.values[i] or x < mini.values[i]) for i, x in enumerate(data.values)]
            else:
                raise ValueError(
                    "Not supporting methods that are not threshold, meanstd, meanstd_window, or percentile.")

    def _compute_corroboration(self):
        """Looks at data and computes final binary signal as a smoothed weighted sum."""
        col_targs = []
        for sensor, weight in zip(self.sensors, self.weights):
            self.detection_df.loc[:,
                                  f"{sensor}_weighted"] = self.detection_df[f"{sensor}_treatment"] * weight
            col_targs.append(f"{sensor}_weighted")

        self.detection_df.loc[:, "corroboration_total"] = self.detection_df[col_targs].sum(
            axis=1)
        self.detection_df.loc[:, "detections"] = [int(x >= self.num_corroboration) for x in self.detection_df.corroboration_total]

    def _compute_stream_limits(self, data, specifications):
        """ obtain upper and lower bounds for data based off
            the sensor configuration

        Args:
            data (pd.DataSeries): sensor data series
            specifications (dict): information for computation

        Returns:
            min, max (float, float): bounds within which data is not anomalous
        """
        if specifications["method"] == "meanstd":
            mean = np.nanmean(data)
            std = np.nanstd(data)
            min = mean - specifications["num_std"] * std
            max = mean + specifications["num_std"] * std

        elif specifications["method"] == "meanstd_window":
            mean = data.rolling(specifications["window"], min_periods=1).mean(centered=True)
            std = data.rolling(specifications["window"], min_periods=1).std(centered=True)
            min = mean - specifications["num_std"] * std
            max = mean + specifications["num_std"] * std

        elif specifications["method"] == "percentile":
            min, max = np.nanpercentile(
                data, [specifications["percentile_lower"],
                       specifications["percentile_upper"]]
            )

        else:
            assert False  # shouldn't be possible

        return min, max
