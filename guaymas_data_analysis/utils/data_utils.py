"""Utils to do with data conversion."""
import gsw
import numpy as np
import pandas as pd
import utm


def calculate_northing_easting_from_lat_lon(df: pd.DataFrame, refeasting=None, refnorthing=None):
    """calculates (norm)northing and (norm)easting from the lat
    and lon. directly modifies the passed in dataframe.

    Args:
        df (pd.DataFrame): [description]
    """
    assert ("lat" in df.keys() and "lon" in df.keys())
    x, y, _, _ = utm.from_latlon(df["lat"].values, df["lon"].values)
    # df.loc[:, "easting"] = x
    if refeasting is None:
        refeasting = np.nanmin(x)
    df.loc[:, "easting"] = x - refeasting
    # df.loc[:, "northing"] = y
    if refnorthing is None:
        refnorthing = np.nanmin(y)
    df.loc[:, "northing"] = y - refnorthing


def convert_oceanographic_measurements(df: pd.DataFrame, vehicle="sentry"):
    """calculates absolute and practical salinity, potential temp, and
    potential density. assumes needed keys exist in the dataframe.
    directly modifies the passed in dataframe.

    Args:
        df (pd.DataFrame): [description]
        vehicle (string): which set of instruments
    """
    if vehicle == "sentry":
        required_keys = ["ctd_cond", "ctd_pres", "ctd_temp", "lat", "lon"]
    elif vehicle == "jason":
        required_keys = ["conductivity",
                         "pressure", "temperature", "lat", "lon"]
    else:
        print("Can't convery measurements from this vehicle. Only support sentry and jason inputs.")
        return
    for k in required_keys:
        assert k in df.keys()

    if vehicle == "jason":
        df.rename({"conductivity": "ctd_cond",
                   "pressure": "ctd_pres",
                   "temperature": "ctd_temp"}, axis=1, inplace=True)

    # calculate practical salinity
    df.loc[:, "practical_salinity"] = df.apply(
        lambda x: gsw.SP_from_C(
            C=x["ctd_cond"]*10,
            t=x["ctd_temp"],
            p=x["ctd_pres"],
        ),
        axis=1,
    )

    # calculate absolute salinity
    df.loc[:, "absolute_salinity"] = df.apply(
        lambda x: gsw.SA_from_SP(
            SP=x["practical_salinity"],
            p=x["ctd_pres"],
            lat=x["lat"],
            lon=x["lon"],
        ),
        axis=1,
    )

    # calculate potential_temp
    df.loc[:, "potential_temp"] = df.apply(
        lambda x: gsw.pt0_from_t(
            SA=x["absolute_salinity"],
            t=x["ctd_temp"],
            p=x["ctd_pres"],
        ),
        axis=1,
    )

    # calculate potential_density
    df.loc[:, "potential_density"] = df.apply(
        lambda x: gsw.pot_rho_t_exact(
            SA=x["absolute_salinity"],
            t=x["ctd_temp"],
            p=x["ctd_pres"],
            p_ref=0,
        ),
        axis=1,
    )


def detect_ascent_descent(
    df: pd.DataFrame,
    window_width_in_samples: int = 600,  # 10 minutes for a 1 Hz sensor
    descent_velocity_threshold: int = 0.4,
    ascent_velocity_threshold: int = -0.2,
    derivative_window_size: int = 4
):
    """A somewhat hacky convenience function to detect ascent and descent based
       off the depth gradient. Threshold values were empirically determined
       from the 2016 teske data.

    Args:
        df (pd.DataFrame): full trajectory series that should include "depth" as a key
        window_width_in_samples (int, optional): How long should the gradient have been
            maintained before we are confident the trajectory is in a regime. Defaults to 600.
        descent_velocity_threshold (int, optional): Defaults to -0.4.
        ascent_velocity_threshold (int, optional): Defaults to -0.2.
        derivative_window_size (int): Window size to smooth over before taking the derivative of
            depth. Should lead to a less noisy gradient.

    Returns:
        start_idx, end_idx: indices into the time series between the ascent and descent
    """
    # HACK: detect ascending / descending based off depth gradient
    # threshold values determined empirically from teske 2016 data
    ddt_depth = np.gradient(df["depth"].rolling(
        center=False, window=derivative_window_size).mean())

    state = 0  # 0 = init, 1 = descending, 2 = cruising, 3 = ascending
    start_idx = 0
    end_idx = len(df)
    for idx in range(len(ddt_depth)):
        rolling_avg = np.mean(ddt_depth[idx: idx + window_width_in_samples])
        if state == 0:
            if rolling_avg > descent_velocity_threshold:
                state = 1
        elif state == 1:
            if rolling_avg < descent_velocity_threshold:
                state = 2
                start_idx = idx
        elif state == 2:
            if rolling_avg < ascent_velocity_threshold:
                state = 3
                end_idx = idx
        else:
            pass

    return start_idx, end_idx


def convert_to_latlon(coords, latlon_origin):
    """Allows arbitrary meters-based coordinates to be converted to lat/lon.

    To be used in conjunction with any script that produces navigational
    coordinates in meters (e.g., in the fumes package, the output missions).
    Provides a commandline interface, or can be imported and called as
    a tool independently.
    """
    # get reference frame of the new origin
    easting, northing, zone_number, zone_letter = utm.from_latlon(
        latlon_origin[0], latlon_origin[1])
    # add easting, northing to appropriate coords
    map_ncoords = np.asarray([nc + northing for nc in coords[:, 1]])
    map_ecoords = np.asarray([ec + easting for ec in coords[:, 0]])
    # now convert back to latlon coordinates
    map_lat, map_lon = utm.to_latlon(
        map_ecoords, map_ncoords, zone_number, zone_letter)

    if coords.shape[1] == 3:
        # xyz data, adapt for height
        return np.hstack([map_lon.reshape(-1, 1),
                          map_lat.reshape(-1, 1),
                          (coords[:, 2] + latlon_origin[2]).reshape(-1, 1)])
    else:
        # xy data
        return np.hstack([map_lon.reshape(-1, 1),
                          map_lat.reshape(-1, 1)])


def distance(lat1, lon1, lat2, lon2):
    """Compute distance in km between two lat-lon points."""
    # approximate radius of earth in km
    R = 6373.0

    # convert decimal degrees to radians
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    # compute distances
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # apply law of cosines to compute distance
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


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
