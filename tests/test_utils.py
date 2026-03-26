import numpy as np

from tess_asteroids import TESSmag_zero_point
from tess_asteroids.utils import calculate_TESSmag


def test_calculate_TESSmag():
    """
    Check expected behaviour of calculate_TESSmag() for some simple test cases.
    """

    # If flux_fraction = 1, magnitude should be equal to zero-point magnitude.
    mag, _, _, _ = calculate_TESSmag(1.0, 0.1, 1.0)
    assert mag == TESSmag_zero_point

    # If flux = NaN, magnitude should be NaN.
    mag, _, _, _ = calculate_TESSmag(np.nan, 0.1, 1.0)
    assert np.isnan(mag)

    # If flux, flux_err and flux_frac are arrays, mag and errors should have the same length.
    flux = np.array([0.1, 0.5, 0.9, 1.5])
    flux_err = np.array([0.01, 0.05, 0.09, 0.15])
    flux_frac = np.array([1.0, 0.5, 0.8, 0.9])
    mag, mag_err, mag_uerr, mag_lerr = calculate_TESSmag(flux, flux_err, flux_frac)
    assert len(mag) == len(flux)
    assert len(mag_err) == len(flux)
    assert len(mag_uerr) == len(flux)
    assert len(mag_lerr) == len(flux)

    # If any value of flux <= 0, corresponding mag and errors should be NaN.
    flux[0] = -0.3
    mag, mag_err, mag_uerr, mag_lerr = calculate_TESSmag(flux, flux_err, flux_frac)
    assert np.isnan(mag[0])
    assert np.isnan(mag_err[0])
    assert np.isnan(mag_uerr[0])
    assert np.isnan(mag_lerr[0])

    # If any value of flux - flux_err <= 0, only mag_uerr should be NaN.
    flux[0] = 0.005
    mag, mag_err, mag_uerr, mag_lerr = calculate_TESSmag(flux, flux_err, flux_frac)
    assert ~np.isnan(mag[0])
    assert ~np.isnan(mag_err[0])
    assert np.isnan(mag_uerr[0])
    assert ~np.isnan(mag_lerr[0])

    # If any value of flux_frac < 0, function should return ValueError.
    flux_frac[2] = -0.3
    try:
        calculate_TESSmag(flux, flux_err, flux_frac)
    except ValueError:
        assert True
    else:
        assert False
