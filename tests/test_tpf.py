import logging
import os

import numpy as np
import pandas as pd

from tess_asteroids import MovingTargetTPF


def test_from_name():
    """
    Check that static method from_name() gives expected ephemeris for asteroid 1998 YT6.
    """
    target, track = MovingTargetTPF.from_name("1998 YT6", sector=6)

    assert target.sector == 6
    assert target.camera == 1
    assert target.ccd == 1

    # Bounds taken from tesswcs pointings.csv file for sector 6.
    assert min(track["time"]) >= 2458463.5 - 2457000
    assert max(track["time"]) <= 2458490.5 - 2457000


def test_data_logic(caplog):
    """
    Check that get_data(), reshape_data() and background_correction() give expected output
    in terms of variable shapes and warnings.
    """

    # Create artificial track
    time = np.linspace(2458790.5, 2458795.5, 100) - 2457000
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
    target = MovingTargetTPF("simulated_track", track)

    # Check get_data() returns expected warnings
    with caplog.at_level(logging.WARNING):
        target.get_data(shape=shape)
    assert "Some of the requested pixels are outside of the FFI bounds" in caplog.text
    assert (
        "Some of the requested pixels are outside of the FFI science array"
        in caplog.text
    )

    # Check attributes have the same length as target.time
    assert (
        len(target.all_flux) == len(target.time)
        and len(target.all_flux_err) == len(target.time)
        and len(target.quality) == len(target.time)
        and len(target.cadence_number) == len(target.time)
        and len(target.corner) == len(target.time)
        and len(target.ephemeris) == len(target.time)
        and len(target.target_mask) == len(target.time)
    )

    # Check the ephemeris and corner have expected shape
    assert np.shape(target.corner)[1] == 2
    assert np.shape(target.ephemeris)[1] == 2

    # Check the reshaped flux data has expected shape
    target.reshape_data()
    assert np.shape(target.flux) == (len(target.time), *shape)
    assert np.shape(target.flux_err) == (len(target.time), *shape)

    # Check the background correction was applied correctly
    target.background_correction()
    assert np.array_equal(target.corr_flux, target.flux - target.bg)
    assert np.array_equal(
        target.corr_flux_err, np.sqrt(target.flux_err**2 + target.bg_err**2)
    )
    assert np.shape(target.corr_flux) == (len(target.time), *shape)
    assert np.shape(target.corr_flux_err) == (len(target.time), *shape)


def test_make_tpf():
    """
    Check that make_tpf() saves TPF as expected.
    """

    # Make TPF for asteroid 1998 YT6
    target, _ = MovingTargetTPF.from_name("1998 YT6", sector=6)
    target.make_tpf(save_loc="tests")

    # Check the file exists
    assert os.path.exists("tests/tess-1998YT6-s0006-shape11x11-moving_tp.fits")

    # Delete the file
    os.remove("tests/tess-1998YT6-s0006-shape11x11-moving_tp.fits")
