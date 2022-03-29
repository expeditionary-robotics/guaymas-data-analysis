import os
import utm

from .extent import Extent
from .general_utils import tic, toc
from .data_utils import convert_to_latlon

REFERENCE = (float(os.getenv("LAT")),
             float(os.getenv("LON")),
             float(os.getenv("DEP")))

EAST_REFERENCE, NORTH_REFERENCE, ZONE_NUM, ZONE_LETT = utm.from_latlon(
    REFERENCE[0], REFERENCE[1])
