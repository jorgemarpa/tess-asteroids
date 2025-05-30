"""
Utility functions
"""

import warnings
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.visualization import simple_norm
from astropy.wcs.utils import fit_wcs_from_points
from matplotlib import animation, colors, patches

from . import TESSmag_zero_point, TESSmag_zero_point_err


def calculate_TESSmag(
    flux: Union[float, np.ndarray],
    flux_err: Union[float, np.ndarray],
    flux_fraction: Union[float, np.ndarray],
):
    """
    Calculate TESS magnitude from a flux and a zero-point magnitude. The equation was taken from the
    TESS Instrument Handbook 2018 (see also Fausnaugh et al. 2021).

    This function assumes that the background flux has been perfectly removed, i.e. the only flux is that
    from the target. It can account for flux outside of the aperture via `flux_fraction`.

    Parameters
    ----------
    flux : float or ndarray
        Target flux, in electrons/second.
    flux_err : float or ndarray
        Error on target flux, in electrons/second.
    flux_fraction : float or ndarray
        Fraction of target flux inside aperture. Must satisfy: 0 < flux_fraction <= 1.

    Returns
    -------
    mag : float or ndarray
        TESS magnitude.
    mag_err : float or ndarray
        Error on TESS magnitude.
    """

    # Check that all values of flux fraction are within allowed range:
    if (np.asarray([flux_fraction]) <= 0).any() or (
        np.asarray([flux_fraction]) > 1
    ).any():
        raise ValueError(
            "All values of flux fraction must satisfy 0 < flux_fraction <= 1."
        )

    # If flux <= 0 (a remnant of BG correction), set it to be NaN for this calculation.
    # This means corresponding mag and mag_err will be NaN.
    # If flux/flux_err are arrays, must make a copy before manpiulating.
    if not isinstance(flux, np.ndarray) and flux <= 0:
        flux = np.nan
    elif isinstance(flux, np.ndarray):
        flux = flux.copy().astype(float)
        flux[flux <= 0] = np.nan
    if isinstance(flux_err, np.ndarray):
        flux_err = flux_err.copy().astype(float)

    # Account for target flux outside aperture.
    flux /= flux_fraction
    flux_err /= flux_fraction

    # Calculate magnitude and error.
    mag = -2.5 * np.log10(flux) + TESSmag_zero_point
    mag_err = np.sqrt(
        (TESSmag_zero_point_err) ** 2 + ((2.5 / np.log(10)) * (flux_err / flux)) ** 2
    )

    return mag, mag_err


def inside_ellipse(
    x: np.ndarray,
    y: np.ndarray,
    cxx: float,
    cyy: float,
    cxy: float,
    x0: float = 0.0,
    y0: float = 0.0,
    R: float = 1.0,
):
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
            # If the denominator is negative, errors will be nan. Catch warning.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="invalid value encountered in scalar divide"
                )
                warnings.filterwarnings(
                    "ignore", message="invalid value encountered in sqrt"
                )
                warnings.filterwarnings(
                    "ignore", message="divide by zero encountered in scalar divide"
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


def plot_img_aperture(
    img: np.ndarray,
    aperture_mask: Optional[np.ndarray] = None,
    cbar: bool = True,
    ax: Optional[plt.Axes] = None,
    corner: Tuple[int, int] = (0, 0),
    marker: Optional[Tuple[float, float]] = None,
    title: str = "",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cnorm: Optional[colors.Normalize] = None,
):
    """
    Plots an image with an optional aperture mask overlay.

    This function displays an image, optionally overlaying an aperture mask, and
    provides several customization options such as color scaling, title, and axis control.

    Parameters:
    -----------
    img : 2D array
        The image data to be plotted, typically a 2D array or matrix representing pixel values.
    aperture_mask : 2D array, default=None
        A binary mask (same shape as `img`) indicating the aperture region to be overlaid on the image.
    cbar : bool, default=True
        Whether to display a color bar alongside the plot.
    ax : matplotlib.axes.Axes, default=None
        The axes object where the plot will be drawn. If not provided, a new axes will be created.
    corner : list of two ints, default=[0, 0]
        The (row, column) coordinates of the lower left corner of the image.
    marker : tuple of float, default=None
        The (row, column) coordinates at which to plot a marker in the figure.
        This can be used to plot the position of the moving object.
    title : str, default=""
        Title of the plot. If None, no title will be shown.
    vmin : float, optional, default=None
        Minimum value for color scale. If None, the 3%-percentile is used.
    vmax : float, optional, default=None
        Maximum value for color scale. If None, the 97%-percentile is used.
    cnorm : optional, default=None
        Color matplotlib normalization object (e.g. astropy.visualization.simple_norm). If provided,
        then `vmax` and `vmin` are not used.

    Returns:
    --------
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
    """

    # Initialise ax
    if ax is None:
        _, ax = plt.subplots()

    # Define the x and y axis tick labels when using plt.imshow() using `corner`
    # and the image shape
    extent = (
        corner[1] - 0.5,
        corner[1] + img.shape[1] - 0.5,
        corner[0] - 0.5,
        corner[0] + img.shape[0] - 0.5,
    )

    # Define vmin and vmax
    if vmin is None and vmax is None and cnorm is None:
        vmin, vmax = np.nanpercentile(img.ravel(), [3, 97])

    # Plot image, colorbar and marker
    im = ax.imshow(
        img,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        norm=cnorm,
        rasterized=True,
        origin="lower",
        extent=extent,
    )
    if cbar:
        plt.colorbar(im, location="right", shrink=0.8, label="Flux [e-/s]")
    if marker is not None:
        ax.scatter(marker[1], marker[0], marker="x", c="deeppink", alpha=1, s=50)

    ax.set_aspect("equal", "box")
    ax.set_title(title)

    # Plot aperture mask
    if aperture_mask is not None:
        row, col = np.mgrid[
            corner[0] : corner[0] + img.shape[0], corner[1] : corner[1] + img.shape[1]
        ]
        for i, pi in enumerate(row[:, 0]):
            for j, pj in enumerate(col[0, :]):
                if aperture_mask[i, j]:
                    # print("here")
                    rect = patches.Rectangle(
                        xy=(pj - 0.5, pi - 0.5),
                        width=1,
                        height=1,
                        color="deeppink",
                        fill=False,
                        # hatch="//",
                        alpha=0.8,
                    )
                    ax.add_patch(rect)

    ax.set_xlabel("Pixel Column")
    ax.set_ylabel("Pixel Row")

    return ax


def animate_cube(
    cube: np.ndarray,
    aperture_mask: Optional[np.ndarray] = None,
    corner: Union[Tuple, np.ndarray] = (0, 0),
    ephemeris: Optional[np.ndarray] = None,
    cadenceno: Optional[np.ndarray] = None,
    time: Optional[np.ndarray] = None,
    interval: int = 200,
    repeat_delay: int = 1000,
    step: int = 1,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cnorm: bool = False,
    suptitle: str = "",
):
    """
    Creates an animated visualization of a 3D image cube, with an optional aperture mask and
    other customization options.

    This function animates the slices of a 3D image cube, optionally overlaying an aperture mask,
    and provides controls for animation speed, title, and tracking information.

    Parameters:
    -----------
    cube : 3D array
        A 3D array representing the image cube (e.g., a stack of 2D images over time).
    aperture_mask : 2D or 3D array, optional
        A binary mask (same shape or a 2D slice of `cube`) to overlay on each frame of the animation.
        If a 2D mask is passed, it will be repeated for all times.
    corner : list of two ints or 2D array, default=[0, 0]
        The (row, column) coordinates of the lower left corner of the image.
    ephemeris : 2D array, optional, default=None
        A 2D array of object positions (row, column) to be displayed on the plot.
        For proper display of object position, if corner is [0, 0] then track needs to be relative to corner.
        If corner is provided, track needs to be absolute.
        If None, no tracking information is shown.
    cadenceno : int, optional, default=None
        The cadence number of the frames, used for information display.
    time : array-like, optional, default=None
        Array of time values corresponding to the slices in the cube.
    interval : int, default=200
        The time interval (in milliseconds) between each frame of the animation.
    repeat_delay : int, default=1000
        The time delay (in milliseconds) before the animation restarts once it finishes.
    step : int
        Spacing between frames, i.e. plot every nth frame.
    vmin : float, optional, default=None
        Minimum value for color scale. If None, the 3%-percentile is used.
    vmax : float, optional, default=None
        Maximum value for color scale. If None, the 97%-percentile is used.
    cnorm : optional, default=False
        Whether to use asinh color normalization (from astropy.visualization.simple_norm).
        This can be useful for cases when the moving object is too faint compared to other
        features in the background. If provided, then `vmax` and `vmin` are not used.
    suptitle : str, optional, default=""
        A string to be used as the super title of the animation.
        It can be used to provide additional context or information about the animated data,
        for example the target name or observing sector/camera/ccd.

    Returns:
    --------
    ani : matplotlib.animation.FuncAnimation
        The animation object that can be displayed or saved.
    """

    # Initialise figure and set title
    fig, ax = plt.subplots()
    fig.suptitle(suptitle)

    if aperture_mask is None:
        aperture_mask = np.repeat([None], len(cube), axis=0)
    # If aperture_mask is 2D, repeat for all times.
    elif aperture_mask.shape == cube.shape[1:]:
        aperture_mask = np.repeat([aperture_mask], len(cube), axis=0)

    if ephemeris is None:
        ephemeris = np.repeat([None], len(cube), axis=0)
    if cadenceno is None:
        cadenceno = np.repeat([None], len(cube), axis=0)
    if time is None:
        time = np.repeat([None], len(cube), axis=0)

    if cnorm:
        norm = simple_norm(cube.ravel(), "asinh", percent=98)
    elif vmin is None and vmax is None:
        vmin, vmax = np.nanpercentile(cube, [3, 97])

    # If corner is list of two ints, repeat for all times.
    if len(corner) == 2:
        corner = np.repeat([corner], len(cube), axis=0)

    # Plot first image in cube.
    nt = 0
    ax = plot_img_aperture(
        cube[nt],
        aperture_mask=aperture_mask[nt],
        cbar=True,
        ax=ax,
        corner=corner[nt],
        marker=ephemeris[nt],
        title=f"CAD {cadenceno[nt]} | BTJD {time[nt]:.4f}",
        vmin=vmin if not cnorm else None,
        vmax=vmax if not cnorm else None,
        cnorm=norm if cnorm else None,
    )

    # Define function for animation
    def animate(nt):
        ax.clear()
        _ = plot_img_aperture(
            cube[nt],
            aperture_mask=aperture_mask[nt],
            cbar=False,
            ax=ax,
            corner=corner[nt],
            marker=ephemeris[nt],
            title=f"CAD {cadenceno[nt]} | BTJD {time[nt]:.4f}",
            vmin=vmin if not cnorm else None,
            vmax=vmax if not cnorm else None,
            cnorm=norm if cnorm else None,
        )

        return ()

    # Prevent second figure from showing up in interactive mode
    plt.close(ax.figure)  # type: ignore

    # Create the animation
    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=range(0, len(cube), step),
        interval=interval,
        blit=True,
        repeat_delay=repeat_delay,
        repeat=True,
    )

    return ani
