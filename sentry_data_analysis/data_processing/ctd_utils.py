"""Utility file for CTD cast analysis."""


def detect_bottle_fire(df):
    """Returns coordinate, time, and bottle number of fired bottle."""
    last_fire = 0
    botts = []
    times = []
    locs = []
    for i, fire in enumerate(df['num_bott_fired'].values):
        if fire > last_fire:
            last_fire = fire
            locs.append((df['longitude'].values[i],
                        df['latitude'].values[i], df['depth'].values[i]))
            times.append(df['system_time'].values[i])
            botts.append(df['bottle_pos'].values[i])
    return botts, locs, times
