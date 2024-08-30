import numpy as np
import pandas as pd
import pytest

from tess_asteroids import MovingObjectTPF


def test_from_name():
    """
    Check that static method from_name() gives expected ephemeris for asteroid 1998 YT6.
    """
    tpf, ephem = MovingObjectTPF.from_name("1998 YT6", sector=6)

    assert tpf.sector == 6
    assert tpf.camera == 1
    assert tpf.ccd == 1

    # Bounds taken from tesswcs pointings.csv file for sector 6.
    assert min(ephem["time"]) >= 2458463.5 - 2457000
    assert max(ephem["time"]) <= 2458490.5 - 2457000


def test_get_data_logic():
    """
    Check that get_data() gives expected output in terms of variable shapes and warnings.
    """
    time = np.linspace(2458790.5, 2458795.5, 100) - 2457000

    # Create artificial track
    track = pd.DataFrame(
        {
            "time": time,
            "sector": np.full(len(time), 18),
            "camera": np.full(len(time), 3),
            "ccd": np.full(len(time), 2),
            "column": np.linspace(-10, 100, len(time)),
            "row": np.linspace(-10, 100, len(time)),
        }
    )

    shape = (12, 10)
    tpf = MovingObjectTPF(track)

    # Check function returns expected warnings
    with pytest.warns(
        UserWarning, match="Some of the requested pixels are outside of the FFI bounds"
    ):
        with pytest.warns(
            UserWarning,
            match="Some of the requested pixels are outside of the FFI science array",
        ):
            time, flux, flux_err, quality, corner, ephemeris, difference = tpf.get_data(
                shape=shape
            )

    # Check all lists have length time
    assert (
        len(flux) == len(time)
        and len(flux_err) == len(time)
        and len(quality) == len(time)
        and len(corner) == len(time)
        and len(ephemeris) == len(time)
        and len(difference) == len(time)
    )

    # Check the data has requested shape.
    assert np.shape(flux)[1] == shape[0] and np.shape(flux)[2] == shape[1]
    assert np.shape(flux_err)[1] == shape[0] and np.shape(flux_err)[2] == shape[1]
    assert np.shape(difference)[1] == shape[0] and np.shape(difference)[2] == shape[1]

    # Check ephemeris and corner shape
    assert np.shape(corner)[1] == 2
    assert np.shape(ephemeris)[1] == 2
