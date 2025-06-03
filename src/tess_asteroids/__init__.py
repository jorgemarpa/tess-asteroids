import logging
from os import path

import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

# Read in straps table
loc = path.abspath(path.dirname(__file__))
straps = pd.read_csv(f"{loc}/data/straps.csv", comment="#")

# TESS zero-point magnitude and error
TESSmag_zero_point = 20.44
TESSmag_zero_point_err = 0.05

__version__ = "1.2.3"
__all__ = ["MovingTPF"]

from .movingtpf import MovingTPF  # noqa: E402
