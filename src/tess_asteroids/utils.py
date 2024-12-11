"""
Utility functions
"""

import warnings
from typing import Optional

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import fit_wcs_from_points

def inside_ellipse(x, y, cxx, cyy, cxy, x0=0, y0=0, R=1):
    """
    Returns a boolean mask indicating positions inside a specified ellipse.
    The ellipse is defined by its center, the radius, and the quadratic coefficients
    (cxx, cyy, cxy).
    Pixels with distance <= R^2 from the center (x0, y0) are considered inside the ellipse.

    Parameters
    ----------
    x : array_like
        The x-coordinates of the points to be tested.
    y : array_like
        The y-coordinates of the points to be tested.
    cxx : float
        The coefficient for the x^2 term in the ellipse equation.
    cyy : float
        The coefficient for the y^2 term in the ellipse equation.
    cxy : float
        The coefficient for the xy term in the ellipse equation.
    x0 : float, optional
        The x-coordinate of the center of the ellipse (default is 0).
    y0 : float, optional
        The y-coordinate of the center of the ellipse (default is 0).
    R : float, optional
        The radius of the ellipse (default is 1).

    Returns
    -------
    mask : ndarray
        A boolean array of the same shape as x and y, where True indicates that the
        corresponding point (x, y) is inside the ellipse defined by the provided parameters.
    """
    return cxx * (x - x0) ** 2 + cyy * (y - y0) ** 2 + cxy * (x - x0) * (y - y0) <= R**2


def compute_moments(
    flux: np.ndarray,
    mask: Optional[np.ndarray] = None,
    second_order: bool = True,
    return_err: bool = False,
):
    """
    Computes first and second order moments of a 2d distribution over time
    using a coordinate grid with the same shape as `flux` (nt, nrows, ncols).
    First order moments (X,Y) are the centroid positions. The X,Y centroids are in
    the range [0, 'ncols'), [0, 'nrows'), respectively i.e. they are zero-indexed.
    Second order moments (X2, Y2, XY) represent the spatial spread of the distribution.

    Parameters
    ----------
    flux: ndarray
        3D array with flux values as (nt, nrows, ncols).
    mask: ndarray
        Mask to select pixels used for computing moments. Shape could
        be 3D (nt, nrows, ncols) or 2D (nrows, ncols). If a 2D mask is given,
        it is used for all frames `nt`.
    second_order: bool
        If True, returns first and second order moments, else returns only first
        order moments.
    return_err: bool
        If True, returns error on first order moments.

    Returns
    -------
    X, Y, XERR, YERR, X2, Y2, XY: ndarrays
        First (X, Y) and second (X2, Y2, XY) order moments, plus error on first order moments (XERR, YERR).
        If `second_order` is False, X2/Y2/XY are not returned. If `return_err` is False, XERR/YERR are
        not returned. Each array has shape (nt).
    """
    # check if mask is None
    if mask is None:
        mask = np.ones_like(flux).astype(bool)
    # reshape 2D mask into 3D mask, if necessary
    if len(mask.shape) == 2:
        mask = np.repeat([mask], flux.shape[0], axis=0)

    # mask negative values in flux (possible artefact of bg subtraction)
    mask = np.logical_and(mask, flux >= 0)
    # mask nans in flux
    mask = np.logical_and(mask, np.isfinite(flux))

    X = np.zeros(flux.shape[0], dtype=float)
    Y = np.zeros(flux.shape[0], dtype=float)
    if second_order or return_err:
        X2 = np.zeros(flux.shape[0], dtype=float)
        Y2 = np.zeros(flux.shape[0], dtype=float)
        XY = np.zeros(flux.shape[0], dtype=float)
    if return_err:
        XERR = np.zeros(flux.shape[0], dtype=float)
        YERR = np.zeros(flux.shape[0], dtype=float)

    # compute moments for each frame
    for nt in range(flux.shape[0]):
        # skip frame if no pixels are used or fluxes sum to zero
        if mask[nt].sum() == 0 or flux[nt, mask[nt]].sum() == 0:
            continue
        # dummy pixel grid
        row, col = np.mgrid[0 : flux.shape[1], 0 : flux.shape[2]]

        # first order moments
        Y[nt] = np.average(row[mask[nt]], weights=flux[nt, mask[nt]])
        X[nt] = np.average(col[mask[nt]], weights=flux[nt, mask[nt]])
        if second_order or return_err:
            # second order moments
            Y2[nt] = (
                np.average(row[mask[nt]] ** 2, weights=flux[nt, mask[nt]]) - Y[nt] ** 2
            )
            X2[nt] = (
                np.average(col[mask[nt]] ** 2, weights=flux[nt, mask[nt]]) - X[nt] ** 2
            )
            XY[nt] = (
                np.average(row[mask[nt]] * col[mask[nt]], weights=flux[nt, mask[nt]])
                - X[nt] * Y[nt]
            )
        if return_err:
            # Error on first-order moments (assumes uncertainties on weights are similar).
            # See eqn. 6 in https://seismo.berkeley.edu/~kirchner/Toolkits/Toolkit_12.pdf
            # If only one non-zero pixel exists in mask, errors will be nan. Catch warning.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="invalid value encountered in scalar divide"
                )
                XERR[nt] = np.sqrt(
                    X2[nt]
                    * (np.nansum(flux[nt, mask[nt]] ** 2))
                    / (
                        np.nansum(flux[nt, mask[nt]]) ** 2
                        - np.nansum(flux[nt, mask[nt]] ** 2)
                    )
                )
                YERR[nt] = np.sqrt(
                    Y2[nt]
                    * (np.nansum(flux[nt, mask[nt]] ** 2))
                    / (
                        np.nansum(flux[nt, mask[nt]]) ** 2
                        - np.nansum(flux[nt, mask[nt]] ** 2)
                    )
                )

    if second_order and return_err:
        return X, Y, XERR, YERR, X2, Y2, XY
    elif second_order and not return_err:
        return X, Y, X2, Y2, XY
    elif return_err and not second_order:
        return X, Y, XERR, YERR
    else:
        return X, Y

def make_wcs_header(shape: Tuple[int, int]):
    """
    Make a dummy WCS header for a moving TPF. In reality, there is a WCS per 
    timestamp that needs to be accounted for.

    Parameters
    ----------
    shape : Tuple(int,int)
        Shape of the TPF. Defined as (nrows,ncols) in pixels.

    Returns
    -------
    wcs_header : astropy.io.fits.header.Header
        Dummy WCS header to use in the TPF.
    """

    # TPF corner (row,column)
    corner = (1, 1)

    # Make a dummy WCS where each pixel in TPF is assigned coordinates 1,1
    row, column = np.meshgrid(
        np.arange(corner[0], corner[0] + shape[0]),
        np.arange(corner[1], corner[1] + shape[1]),
    )
    coord = SkyCoord(np.full([len(row.ravel()), 2], (1, 1)), unit="deg")
    wcs = fit_wcs_from_points((column.ravel(), row.ravel()), coord)

    # Turn WCS into header
    wcs_header = wcs.to_header(relax=True)

    # Add the physical WCS keywords
    wcs_header.set("CRVAL1P", corner[1], "value at reference CCD column")
    wcs_header.set("CRVAL2P", corner[0], "value at reference CCD row")

    wcs_header.set(
        "WCSNAMEP", "PHYSICAL", "name of world coordinate system alternate P"
    )
    wcs_header.set("WCSAXESP", 2, "number of WCS physical axes")

    wcs_header.set("CTYPE1P", "RAWX", "physical WCS axis 1 type CCD col")
    wcs_header.set("CUNIT1P", "PIXEL", "physical WCS axis 1 unit")
    wcs_header.set("CRPIX1P", 1, "reference CCD column")
    wcs_header.set("CDELT1P", 1.0, "physical WCS axis 1 step")

    wcs_header.set("CTYPE2P", "RAWY", "physical WCS axis 2 type CCD col")
    wcs_header.set("CUNIT2P", "PIXEL", "physical WCS axis 2 unit")
    wcs_header.set("CRPIX2P", 1, "reference CCD row")
    wcs_header.set("CDELT2P", 1.0, "physical WCS axis 2 step")

    return wcs_header