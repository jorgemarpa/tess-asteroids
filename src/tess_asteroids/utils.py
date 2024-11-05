"""
Utility functions
"""

import numpy as np


def inside_ellipse(x, y, cxx, cyy, cxy, x0=0, y0=0, R=1):
    """
    Returns a boolean mask indicating positions inside a specified ellipse.
    The ellipse is defined by its center, the radius, and the quadratic coefficients
    (cxx, cyy, cxy).
    Pixels with distnace <= R^2 from the center (x0, y0) are considered inside the ellipse.

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


def compute_moments(flux, mask=None):
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
    mask: ndarray, boolean
        Mask to select pixels used for computing moments. Shape could
        be 3D (nt, nrows, ncols) or 2D (nrows, ncols). If a 2D mask is given,
        it is used for all frames `nt`.

    Returns
    -------
    X, Y, X2, Y2, XY: ndarrays
        First (X, Y) and second (X2, Y2, XY) order moments. Each array has
        shape (nt).
    """
    # check if mask is None
    if mask is None:
        mask = np.ones_like(flux).astype(bool)
    # reshape 2D mask into 3D mask if necessary
    if len(mask.shape) == 2:
        mask = np.repeat([mask], flux.shape[0], axis=0)

    X = np.zeros(flux.shape[0], dtype=float)
    Y = np.zeros(flux.shape[0], dtype=float)
    X2 = np.zeros(flux.shape[0], dtype=float)
    Y2 = np.zeros(flux.shape[0], dtype=float)
    XY = np.zeros(flux.shape[0], dtype=float)

    # compute moments for each frame
    for nt in range(flux.shape[0]):
        # skip frame if no pixels are used
        if mask[nt].sum() == 0:
            continue
        # dummy pixel grid
        row, col = np.mgrid[0 : flux.shape[1], 0 : flux.shape[2]]

        # first order moments
        Y[nt] = np.average(row[mask[nt]], weights=flux[nt, mask[nt]])
        X[nt] = np.average(col[mask[nt]], weights=flux[nt, mask[nt]])
        # second order moments
        Y2[nt] = np.average(row[mask[nt]] ** 2, weights=flux[nt, mask[nt]]) - Y[nt] ** 2
        X2[nt] = np.average(col[mask[nt]] ** 2, weights=flux[nt, mask[nt]]) - X[nt] ** 2
        XY[nt] = (
            np.average(row[mask[nt]] * col[mask[nt]], weights=flux[nt, mask[nt]])
            - X[nt] * Y[nt]
        )
    return X, Y, X2, Y2, XY
