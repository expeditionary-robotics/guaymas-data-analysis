"""Utilities to do with data locations and metadata."""
import copy
import os
import numpy as np
import pandas as pd
import utm
from datetime import datetime, timezone, timedelta
import warnings

from .extent import Extent

REFERENCE = (float(os.getenv("LAT")),
             float(os.getenv("LON")),
             float(os.getenv("DEP")))

EAST_REFERENCE, NORTH_REFERENCE, ZONE_NUM, ZONE_LETT = utm.from_latlon(
    REFERENCE[0], REFERENCE[1])

# This is specifically good for Sentry611
EXTENT_RIDGE_SMALL = Extent(xrange=(1000, 1800),
                            xres=100,
                            yrange=(1000, 1800),
                            yres=100,
                            zrange=(-300, 450),
                            zres=10,
                            global_origin=REFERENCE)

# This is a broader range, good for all Sentry dives
EXTENT_RIDGE = Extent(xrange=(0, 3000),
                      xres=100,
                      yrange=(0, 3000),
                      yres=100,
                      zrange=(-300, 450),
                      zres=10,
                      global_origin=REFERENCE)

EXTENT_RING = Extent(xrange=(-28000, -26000),
                     xres=100,
                     yrange=(11000, 13000),
                     yres=100,
                     zrange=(-100, 400),
                     zres=10,
                     global_origin=REFERENCE)


EXTENT_SOUTH = Extent(
    # 500m east of PAGODA east, 500m west of CATHEDRAL HILL
    xrange=(-1697.5616496511502, -73.82290453586029),
    xres=100,
    # 500m south of ROBIINS ROOST, 500m north of PAGODA
    yrange=(-43540.775659619365, -42129.67402120307),
    yres=100,
    zrange=(0, 1800),
    zres=10,
    global_origin=REFERENCE)


def calibrate_nopp(fundamental):
    """Calibrate th NOPP fundamental."""
    thresh = 0.09682
    m1 = -0.1014
    b1 = 0.2839
    m2 = -0.065
    b2 = 0.21
    state1 = (fundamental < thresh)

    cal_fun = copy.copy(fundamental)
    cal_fun[state1] = 10**((fundamental[state1] - b1) / m1)
    cal_fun[~state1] = 10**((fundamental[~state1] - b2) / m2)
    return cal_fun


def sentry_site_by_dive(dive_name):
    """The sentry sive location by dive name."""
    if dive_name == "sentry607":
        site = "ridge"
    elif dive_name == "sentry608":
        site = "ridge"
    elif dive_name == "sentry609":
        site = "ring"
    elif dive_name == "sentry610":
        site = "ridge"
    elif dive_name == "sentry611":
        site = "ridge"
    elif dive_name == "sentry612":
        site = "south"
    elif dive_name == "sentry613":
        # site = "transect"
        site = "ridge"
    elif dive_name == "sentry613_full":
        # site = "transect"
        site = "ridge"
    elif dive_name == "sentry613_ascent":
        # site = "transect"
        site = "ridge"
    elif dive_name == "sentry613_descent":
        # site = "transect"
        site = "ridge"
    else:
        raise ValueError(f"Unrecognized dive {dive_name}")
    return site


def jason_site_by_dive(dive_name):
    """The sentry sive location by dive name."""
    # [TODO]: check all of these
    if dive_name == "J2-1388":
        site = "ridge"
    elif dive_name == "J2-1389":
        site = "ridge"
    elif dive_name == "J2-1390":
        site = "ring"
    elif dive_name == "J2-1391":
        site = "ridge"
    elif dive_name == "J2-1392":
        site = "ridge"
    elif dive_name == "J2-1393":
        site = "ring"
    elif dive_name == "J2-1394":
        site = "ring"
    elif dive_name == "J2-1395":
        site = "south"
    elif dive_name == "J2-1396":
        site = "south"
    elif dive_name == "J2-1397":
        site = "south"
    elif dive_name == "J2-1398":
        site = "south"
    elif dive_name == "J2-1399":
        site = "south"
    elif dive_name == "J2-1400":
        site = "south"
    else:
        raise ValueError(f"Unrecognized dive {dive_name}")
    return site


def ctd_site_by_number(cast_number):
    """The CTD location by cast number."""
    # [TODO]: check all of these
    if cast_number == "01":
        site = "ridge"
    elif cast_number == "02":
        site = "ridge"  # only up to the oxidation zone
    elif cast_number == "03":
        site = "ridge"
    elif cast_number == "04":
        site = "ring"
    elif cast_number == "05":
        site = "ring"
    elif cast_number == "06-HCF":
        site = "ridge"
    elif cast_number == "07":
        site = "ridge"
    elif cast_number == "08":
        site = "ridge"
    elif cast_number == "09":
        site = "ridge"  # 22km south point between north and south grabens
    elif cast_number == "10-HCF-Transect":
        site = "ridge"  # transect
    elif cast_number == "11-HCF-Transect":
        site = "ridge"  # transect
    else:
        raise ValueError(f"Unrecognized dive {dive_name}")
    return site


def extent_by_site(site):
    if site == "ridge":
        return EXTENT_RIDGE
    elif site == "ridge_small":
        return EXTENT_RIDGE_SMALL
    elif site == "ring":
        return EXTENT_RING
    elif site == "south":
        return EXTENT_SOUTH
    else:
        return None


def sentry_output_by_dive(dive_name):
    """Get the Sentry output file by dive name."""
    if "ascent" in dive_name:
        return os.path.join(os.getenv("SENTRY_DATA"),
                            f"sentry/proc/ascent_profiles/RR2107_{dive_name}_processed.csv")
    elif "descent" in dive_name:
        return os.path.join(os.getenv("SENTRY_DATA"),
                            f"sentry/proc/descent_profiles/RR2107_{dive_name}_processed.csv")
    return os.path.join(os.getenv("SENTRY_DATA"),
                        f"sentry/proc/RR2107_{dive_name}_processed.csv")


def sentry_anom_by_dive(dive_name):
    """Get the Sentry anomalies output file by dive name."""
    return os.path.join(os.getenv("SENTRY_DATA"),
                        f"sentry/proc/RR2107_{dive_name}_processed_anom.csv")


def sentry_detections_by_dive(dive_name):
    """Get the Sentry anomalies output file by dive name."""
    return os.path.join(os.getenv("SENTRY_DATA"),
                        f"sentry/proc/RR2107_{dive_name}_processed_detections.csv")


def ctd_output():
    """Get the CTD output file by dive name."""
    return os.path.join(os.getenv("SENTRY_DATA"),
                        "ctd/proc/RR2107_CTD_proc_profiles_all.csv")


def ctd_filename_by_cast(cast_name):
    # Find the correct filename by cast
    return os.path.join(
        os.getenv("SENTRY_DATA"), f"ctd/raw/RR2107-CTD-{cast_name}.cnv")


def tiltmeter_filename_by_deployment(deployment):
    """Get the tiltmeter file by deployment number."""
    if deployment == "A1":
        # At the black smoker site
        DATA_PATHS = ["2108300_20211117_TA_JD2_(0)_Current.csv",
                      "2108300_20211117_TA_JD2_(0)_Temperature.csv"]
        FOLDER = "tiltmeterA_202111119_recovery1"
    elif deployment == "A2":
        # At black smoker site
        DATA_PATHS = ["2108300_20211117_TA_JD3_(0)_Current.csv",
                      "2108300_20211117_TA_JD3_(0)_Temperature.csv"]
        FOLDER = "tiltmeterA_20211121_recovery2"
    elif deployment == "A3":
        # At ring vent site
        DATA_PATHS = ["2108300_20211121_TA_JD4_(0)_Current.csv",
                      "2108300_20211121_TA_JD4_(0)_Temperature.csv"]
        FOLDER = "tiltmeterA_20211122_recovery3"
    elif deployment == "B1":
        # At black smoker site
        DATA_PATHS = ["2110300_20211117_TB_JD2_(0)_Current.csv",
                      "2110300_20211117_TB_JD2_(0)_Temperature.csv"]
        FOLDER = "tiltmeterB_20211117_recovery1"
    else:
        raise ValueError(f"Unrecognized tiltmeter deployment {deployment}.")
    PATHS = [os.path.join(os.getenv("SENTRY_DATA"),
                          f"tiltmeter/raw/{FOLDER}/{p}") for p in DATA_PATHS]
    return PATHS


def tiltmeter_time_by_deployment(deployment):
    """Get the tiltmeter start and end time by deployment number."""
    if deployment == "A1":
        START_DATA = datetime(2021, 11, 19, 0, 7, 25, tzinfo=timezone.utc)
        END_DATA = datetime(2021, 11, 20, 3, 47, 55, tzinfo=timezone.utc)
    elif deployment == "A2":
        START_DATA = datetime(2021, 11, 21, 4, 35, 24, tzinfo=timezone.utc)
        END_DATA = datetime(2021, 11, 21, 14, 17, 10, tzinfo=timezone.utc)
    elif deployment == "A3":
        raise ValueError(
            f"We don't know start/end time for ring vent: {deployment}.")
        START_DATA = datetime(2021, 11, 21, 4, 35, 24, tzinfo=timezone.utc)
        END_DATA = datetime(2021, 11, 21, 14, 17, 10, tzinfo=timezone.utc)
    elif deployment == "B1":
        START_DATA = datetime(2021, 11, 19, 0, 8, 7, tzinfo=timezone.utc)
        END_DATA = datetime(2021, 11, 25, 15, 9, 7, tzinfo=timezone.utc)
    else:
        raise ValueError(f"Unrecognized tiltmeter deployment {deployment}.")
    return START_DATA, END_DATA


def tiltmeter_time_exclude_by_deployment(deployment):
    """Get the tiltmeter time to exclude by deployment number."""
    if deployment == "A1":
        EXCLUDE_START = datetime(2021, 11, 19, 10, 30, 0, tzinfo=timezone.utc)
        EXCLUDE_END = datetime(2021, 11, 19, 13, 0, 0, tzinfo=timezone.utc)
        return EXCLUDE_START, EXCLUDE_END
    elif deployment == "A2":
        return None, None
    elif deployment == "A3":
        return None, None
    elif deployment == "B1":
        return None, None
    else:
        raise ValueError(f"Unrecognized tiltmeter deployment {deployment}.")


def tiltmeter_output_by_deployment(deployment):
    """Get the tiltmeter processed output file and training file."""
    cur_file = f"proc_current_profile_{deployment}.csv"
    cur_file = os.path.join(os.getenv("SENTRY_DATA"),
                            f"tiltmeter/proc/{cur_file}")
    return cur_file


def tiltmeter_trainset_by_deployment(deployment):
    """Get the tiltmeter processed output file and training file."""
    cur_train_file = f"proc_current_train_profile_{deployment}.csv"
    cur_train_file = os.path.join(
        os.getenv("SENTRY_DATA"), f"tiltmeter/proc/{cur_train_file}")
    return cur_train_file


def sentry_filename_by_dive(dive_name):
    # Find the correct filename by dive
    if dive_name == "sentry607":
        date = "20211118"
        time = "1847"
    elif dive_name == "sentry608":
        date = "20211120"
        time = "0821"
    elif dive_name == "sentry609":
        date = "20211123"
        time = "1623"
    elif dive_name == "sentry610":
        date = "20211125"
        time = "1653"
    elif dive_name == "sentry611":
        date = "20211127"
        time = "0801"
    elif dive_name == "sentry612":
        date = "20211128"
        time = "2327"
    elif dive_name == "sentry613":
        date = "20211130"
        time = "1558"
    elif dive_name == "sentry613_full":
        date = "20211202"
        time = "2143"
        dive_name = "sentry613"
    else:
        raise ValueError(f"Unrecognized dive {dive_name}")
    input_scc = f"sentry/raw/{dive_name}_{date}_{time}_scc.mat"
    input_orp = f"sentry/raw/{dive_name}_{date}_{time}_orp_renav.mat"
    input_o2 = f"sentry/raw/{dive_name}_{date}_{time}_optode_renav.mat"

    input_scc = os.path.join(os.getenv("SENTRY_DATA"), input_scc)
    input_orp = os.path.join(os.getenv("SENTRY_DATA"), input_orp)
    input_o2 = os.path.join(os.getenv("SENTRY_DATA"), input_o2)

    return input_scc, [input_orp, input_o2]


def sentry_phase_by_dive_and_time(time, dive_name):
    if dive_name == "sentry610":
        # Phase 1 Adaptive
        phase1_start = datetime(2021, 11, 24, 16, 13, 36, tzinfo=timezone.utc)
        phase1_end = datetime(2021, 11, 24, 20, 3, 21, tzinfo=timezone.utc)

        # Phase 2 Naive
        phase2_start = datetime(2021, 11, 24, 20, 18, 33, tzinfo=timezone.utc)
        phase2_end = datetime(2021, 11, 25, 0, 33, 42, tzinfo=timezone.utc)

        # Phase 3 Adaptive
        phase3_start = datetime(2021, 11, 25, 0, 46, 45, tzinfo=timezone.utc)
        phase3_end = datetime(2021, 11, 25, 4, 39, 1, tzinfo=timezone.utc)

        # Phase 4 Adaptive
        phase4_start = datetime(2021, 11, 25, 5, 8, 12, tzinfo=timezone.utc)
        phase4_end = datetime(2021, 11, 25, 8, 42, 41, tzinfo=timezone.utc)

        # Phase 5 Naive
        phase5_start = datetime(2021, 11, 25, 9, 13, 31, tzinfo=timezone.utc)
        phase5_end = datetime(2021, 11, 25, 13, 8, 58, tzinfo=timezone.utc)

        # Phase 6 Adaptive
        phase6_start = datetime(2021, 11, 25, 13, 24, 25, tzinfo=timezone.utc)
        phase6_end = datetime(2021, 11, 25, 13, 47, 42, tzinfo=timezone.utc)

        if time >= phase1_start and time <= phase1_end:
            return 1
        elif time >= phase2_start and time <= phase2_end:
            return 2
        elif time >= phase3_start and time <= phase3_end:
            return 3
        elif time >= phase4_start and time <= phase4_end:
            return 4
        elif time >= phase5_start and time <= phase5_end:
            return 5
        elif time >= phase6_start and time <= phase6_end:
            return 6
        else:
            # warnings.warn(f"Warning: provided time {time} is not assigned a phase.")
            return 0
    elif dive_name == "sentry611":
        # Phase 1
        phase1_start = datetime(2021, 11, 26, 20, 24, 57, tzinfo=timezone.utc)
        phase1_end = datetime(2021, 11, 26, 21, 25, 18, tzinfo=timezone.utc)

        # Phase 2
        phase2_start = datetime(2021, 11, 26, 21, 27, 8, tzinfo=timezone.utc)
        phase2_end = datetime(2021, 11, 26, 22, 22, 28, tzinfo=timezone.utc)

        # Phase 3
        phase3_start = datetime(2021, 11, 26, 22, 23, 31, tzinfo=timezone.utc)
        phase3_end = datetime(2021, 11, 26, 23, 22, 53, tzinfo=timezone.utc)

        # Phase 4
        phase4_start = datetime(2021, 11, 26, 23, 24, 45, tzinfo=timezone.utc)
        phase4_end = datetime(2021, 11, 27, 0, 19, 1, tzinfo=timezone.utc)

        # Phase 5
        phase5_start = datetime(2021, 11, 27, 0, 25, 53, tzinfo=timezone.utc)
        phase5_end = datetime(2021, 11, 27, 1, 14, 35, tzinfo=timezone.utc)

        # Phase 6
        phase6_start = datetime(2021, 11, 27, 1, 22, 16, tzinfo=timezone.utc)
        phase6_end = datetime(2021, 11, 27, 2, 11, 40, tzinfo=timezone.utc)

        # Phase 7
        phase7_start = datetime(2021, 11, 27, 2, 12, 36, tzinfo=timezone.utc)
        phase7_end = datetime(2021, 11, 27, 3, 8, 21, tzinfo=timezone.utc)

        # Phase 8
        phase8_start = datetime(2021, 11, 27, 3, 16, 31, tzinfo=timezone.utc)
        phase8_end = datetime(2021, 11, 27, 4, 5, 24, tzinfo=timezone.utc)

        # Phase 9
        phase9_start = datetime(2021, 11, 27, 4, 13, 42, tzinfo=timezone.utc)
        phase9_end = datetime(2021, 11, 27, 5, 2, 44, tzinfo=timezone.utc)

        # Phase 10
        phase10_start = datetime(2021, 11, 27, 5, 10, 24, tzinfo=timezone.utc)
        phase10_end = datetime(2021, 11, 27, 4, 42, 16, tzinfo=timezone.utc)

        if time >= phase1_start and time <= phase1_end:
            return 1
        elif time >= phase2_start and time <= phase2_end:
            return 2
        elif time >= phase3_start and time <= phase3_end:
            return 3
        elif time >= phase4_start and time <= phase4_end:
            return 4
        elif time >= phase5_start and time <= phase5_end:
            return 5
        elif time >= phase6_start and time <= phase6_end:
            return 6
        elif time >= phase7_start and time <= phase7_end:
            return 7
        elif time >= phase8_start and time <= phase8_end:
            return 8
        elif time >= phase9_start and time <= phase9_end:
            return 9
        elif time >= phase10_start and time <= phase10_end:
            return 10
        else:
            # warnings.warn(f"Warning: provided time {time} is not assigned a phase.")
            return 0
    else:
        raise ValueError(
            f"We haven't determined the phases for {dive_name} yet.")


def jason_filename_by_dive(dive_name, var):
    # Find the correct filename by dive
    file = f"jason/raw/{dive_name}/{dive_name}.{var}.raw"
    file = os.path.join(os.getenv("SENTRY_DATA"), file)
    return file


def jason_starttime_by_dive(dive_name):
    """Subset the Jason dive"""
    if dive_name == "J2-1390":
        return None, None
        start = datetime(2021, 11, 19, 11, 0, 0)
        end = datetime(2021, 11, 20, 3, 30, 0)
    elif dive_name == "J2-1393":
        return None, None
        start = datetime(2021, 11, 21, 23, 0, 0)
        # 2021-11-22T12:51:10.352Z <guest>: Instruments --> seabird_pump_: "Methane Seabird Pump off"
        # from jason log
        end = datetime(2021, 11, 22, 12, 0, 2)
    else:
        start = end = None

    return start, end


def nopp_filename_by_dive(dive_name):
    # Find the correct filename by dive
    """ Sentry DIVES """
    if dive_name == "sentry607":
        file = f"nopp/raw/N2-{dive_name}.txt"
    elif dive_name == "sentry608":
        file = f"nopp/raw/N2-{dive_name}.txt"
    elif dive_name == "sentry609":
        return None
    elif dive_name == "sentry610":
        file = f"nopp/raw/N1-{dive_name}.txt"
    elif dive_name == "sentry611":
        return None
    elif dive_name == "sentry612":
        file = f"nopp/raw/N1-{dive_name}.txt"
    elif dive_name == "sentry613":
        file = f"nopp/raw/N1-{dive_name}.txt"
    elif dive_name == "sentry613_full":
        dive_name = "sentry613"
        file = f"nopp/raw/N1-{dive_name}.txt"
    elif dive_name == "J2-1388":
        """ JASON DIVES [TODO] """
        file = f"nopp/raw/N1-{dive_name}.txt"
    elif dive_name == "J2-1389":
        file = f"nopp/raw/N1-{dive_name}.txt"
    elif dive_name == "J2-1390":
        file = f"nopp/raw/N1-{dive_name}.txt"
    elif dive_name == "J2-1391":
        file = f"nopp/raw/N1-{dive_name}.txt"
    elif dive_name == "J2-1392":
        file = [f"nopp/raw/N1-{dive_name}.txt",
                f"nopp/raw/N2-{dive_name}.txt"]
    elif dive_name == "J2-1393":
        file = f"nopp/raw/N2-{dive_name}.txt"
    elif dive_name == "J2-1394":
        file = f"nopp/raw/N2-{dive_name}.txt"
    elif dive_name == "J2-1395":
        file = f"nopp/raw/N2-{dive_name}.txt"
    elif dive_name == "J2-1396":
        file = f"nopp/raw/N2-{dive_name}.txt"
    elif dive_name == "J2-1397":
        file = f"nopp/raw/N2-{dive_name}.txt"
    elif dive_name == "J2-1398":
        file = f"nopp/raw/N2-{dive_name}.txt"
    elif dive_name == "J2-1399":
        file = f"nopp/raw/N2-{dive_name}.txt"
    elif dive_name == "J2-1400":
        file = f"nopp/raw/N2-{dive_name}.txt"
    else:
        raise ValueError(f"Unrecognized dive {dive_name}")

    if type(file) is list:
        file = [os.path.join(os.getenv("SENTRY_DATA"), f) for f in file]
    else:
        file = os.path.join(os.getenv("SENTRY_DATA"), file)
    return file


def hcf_filename_by_dive(dive_name):
    # Find the correct filename by dive
    if dive_name == "sentry607":
        return None
    elif dive_name == "sentry608":
        return None
    elif dive_name == "sentry609":
        file = f"hcf/raw/HCF_Sentry609_RingVentSurvey.txt"
    elif dive_name == "sentry610":
        return None
    elif dive_name == "sentry611":
        file = f"hcf/raw/HCF_Sentry611_GuaymasNorthMap.txt"
    elif dive_name == "sentry612":
        return None
    elif dive_name == "sentry613":
        return None
    elif dive_name == "sentry613_full":
        return None
    elif dive_name == "J2-1388":
        """ JASON DIVES """
        return None
    elif dive_name == "J2-1389":
        file = f"hcf/raw/HCF_J1389_GuaymasNorthDownAndUp.txt"
    elif dive_name == "J2-1390":
        file = f"hcf/raw/HCF_J1390_GuaymasNorthPlumeExplorationAndMapping.txt"
    elif dive_name == "J2-1391":
        return None
    elif dive_name == "J2-1392":
        return None
    elif dive_name == "J2-1393":
        return None
    elif dive_name == "J2-1394":
        file = f"hcf/raw/HCF_J1394_RingVentCircle.txt"
    elif dive_name == "J2-1395":
        file = f"hcf/raw/HCF_J1395_TubeWormSampling.txt"
    elif dive_name == "J2-1396":
        file = f"hcf/raw/HCFJ1396_SlowDescentAndAscent.txt"
    elif dive_name == "J2-1397":
        file = f"hcf/raw/HCF_J1397_InactiveBubbleSiteTransects.txt"
    elif dive_name == "J2-1398":
        return None
    elif dive_name == "J2-1399":
        file = f"hcf/raw/HCF_J1399_GuaymasSouthExploring.txt"
    elif dive_name == "J2-1400":
        return None
    else:
        raise ValueError(f"Unrecognized dive {dive_name}")
    file = os.path.join(os.getenv("SENTRY_DATA"), file)
    return file
