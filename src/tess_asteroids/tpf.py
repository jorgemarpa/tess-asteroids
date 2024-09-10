import time
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs.utils import fit_wcs_from_points
from tess_ephem import ephem
from tesscube import TESSCube
from tesscube.fits import get_wcs_header_by_extension
from tesscube.utils import _sync_call, convert_coordinates_to_runs

from . import logger


class MovingTargetTPF:
    """
    Create a TPF for a moving target (e.g. asteroid) from a TESS FFI.

    Includes methods to efficiently retrieve the data, correct the background,
    define an aperture mask and save a TPF in the SPOC format.

    Parameters
    ----------
    target : str
        Target ID. This is only used when saving the TPF.
    ephem : DataFrame
        Target ephemeris with columns ['time','sector','camera','ccd','column','row'].
            'time' : float with units (BJD - 2457000).
            'sector', 'camera', 'ccd' : int
            'column', 'row' : float. These must be one-indexed, where the lower left pixel of the FFI is (1,1).
    """

    def __init__(self, target: str, ephem: pd.DataFrame):
        self.target = target
        self.ephem = ephem

        # Check self.ephem['time'] has correct units
        if min(self.ephem["time"]) >= 2457000:
            raise ValueError("ephem['time'] must have units (BJD - 2457000).")

        # Check self.ephem['sector'] has one unique value
        if self.ephem["sector"].nunique() == 1:
            self.sector = int(self.ephem["sector"][0])
        else:
            raise ValueError("ephem['sector'] must have one unique value.")

        # Check if target is only observed on one camera/ccd during sector
        if self.ephem["camera"].nunique() == 1 and self.ephem["ccd"].nunique() == 1:
            self.camera = int(self.ephem["camera"][0])
            self.ccd = int(self.ephem["ccd"][0])
        else:
            # >>>>> INCLUDE A WAY TO GET MULTIPLE CUBES. <<<<<
            raise NotImplementedError(
                "Target crosses multiple camera/ccd. Not yet implemented."
            )

        # Initialise tesscube
        self.cube = TESSCube(sector=self.sector, camera=self.camera, ccd=self.ccd)

    def make_tpf(
        self,
        shape: Tuple[int, int] = (11, 11),
        bg_method: str = "rolling",
        file_name: Optional[str] = None,
        save_loc: str = "",
        **kwargs,
    ):
        """
        Performs all steps to create and save a SPOC-like TPF for a moving target.

        Parameters
        ----------
        shape : Tuple(int,int)
            Defined as (row,column), in pixels.
            Defines the pixels that will be retrieved, centred on the target, at each timestamp.
        bg_method : str
            Method used for background correction.
            One of `rolling`.
        file_name : str
            Name of saved TPF.
        save_loc : str
            Directory into which the files will be saved.

        Returns
        -------
        """
        # >>>>> ADD REFINE_COORDINATES() WHEN IMPLEMENTED <<<<<
        self.get_data(shape=shape)
        self.reshape_data()
        self.background_correction(method=bg_method, **kwargs)
        # >>>>> ADD GET_APERTURE() WHEN IMPLEMENTED <<<<<
        self.save_data(file_name_tpf=file_name, save_loc=save_loc, **kwargs)

    def refine_coordinates(self):
        """
        Apply correction to target ephemeris.

        Parameters
        ----------

        Returns
        -------
        """
        raise NotImplementedError("refine_coordinates() is not yet implemented.")

    def get_data(self, shape: Tuple[int, int] = (11, 11)):
        """
        Retrieve pixel data for a moving target from a TESS FFI.

        Parameters
        ----------
        shape : Tuple(int,int)
            Defined as (row,column), in pixels.
            Defines the pixels that will be retrieved, centred on the target, at each timestamp.

        Returns
        -------
        """
        # Shape needs to be >=3 in both dimensions, otherwise _make_wcs_header() errors.
        self.shape = shape

        # Use interpolation to get target (row,column) at cube time.
        column_interp = np.interp(
            self.cube.time,
            self.ephem["time"].astype(float),
            self.ephem["column"].astype(float),
            left=np.nan,
            right=np.nan,
        )
        row_interp = np.interp(
            self.cube.time,
            self.ephem["time"].astype(float),
            self.ephem["row"].astype(float),
            left=np.nan,
            right=np.nan,
        )

        # Remove nans from interpolated position
        nan_mask = np.logical_and(~np.isnan(column_interp), ~np.isnan(row_interp))
        row_interp, column_interp = row_interp[nan_mask], column_interp[nan_mask]

        # Coordinates (row,column) of lower left corner of region to retrieve around target
        self.corner = np.ceil(
            np.asarray(
                [row_interp - self.shape[0] / 2, column_interp - self.shape[1] / 2]
            ).T
        ).astype(int)

        # Pixel positions to retrieve around target
        row, column = (
            np.mgrid[: self.shape[0], : self.shape[1]][:, None, :, :]
            + self.corner.T[:, :, None, None]
        )

        # Remove frames that include pixels outside bounds of FFI.
        # >>>>> COULD KEEP FRAMES WITH PIXELS OUTSIDE OF BOUNDS AND FILL WITH NANS INSTEAD? <<<<<
        bound_mask = np.logical_and(
            [r.all() for r in np.logical_and(row[:, :, 0] >= 1, row[:, :, 0] <= 2078)],
            [
                c.all()
                for c in np.logical_and(column[:, 0, :] >= 1, column[:, 0, :] <= 2136)
            ],
        )
        self.time = self.cube.time[nan_mask][
            bound_mask
        ]  # FFI timestamps of each frame in the data cube.
        self.quality = self.cube.quality[nan_mask][
            bound_mask
        ]  # SPOC quality flag of each frame in the data cube.
        self.cadence_number = self.cube.cadence_number[nan_mask][
            bound_mask
        ]  # SPOC cadence number of each frame in the data cube.
        self.ephemeris = np.asarray(
            [row_interp[bound_mask], column_interp[bound_mask]]
        ).T  # Predicted (row,column) of target.
        self.corner = self.corner[bound_mask]
        row, column = row[bound_mask], column[bound_mask]
        pixel_coordinates = np.asarray([row.ravel(), column.ravel()]).T

        # Check there are pixels inside FFI bounds.
        if len(pixel_coordinates) == 0:
            raise RuntimeError(
                "All pixels are outside of FFI bounds (1<=row<=2078, 1<=col<=2136)."
            )
        # Warn user if some of the pixels are outside of FFI bounds.
        elif sum(~bound_mask) > 0:
            logger.warn(
                "Some of the requested pixels are outside of the FFI bounds (1<=row<=2078, 1<=col<=2136) and will not be returned."
            )
        # Warn user if there are pixels outside of FFI science array.
        if (
            sum(
                ~np.logical_and(
                    [r.all() for r in row[:, :, 0] <= 2048],
                    [
                        c.all()
                        for c in np.logical_and(
                            column[:, 0, :] >= 45, column[:, 0, :] <= 2092
                        )
                    ],
                )
            )
            > 0
        ):
            logger.warn(
                "Some of the requested pixels are outside of the FFI science array (1<=row<=2048, 45<=col<=2092), but they will be included in your TPF."
            )

        # Convert pixels to byte runs
        runs = convert_coordinates_to_runs(pixel_coordinates)

        # Retrieve the data
        logger.info("Started data retrieval.")
        start_time = time.time()
        result = _sync_call(self.cube.async_get_data_per_rows, runs)
        logger.info(
            "Finished data retrieval in {0:.2f} sec.".format(time.time() - start_time)
        )

        # Split result into flux, flux_err and reshape into (ntimes, npixels)
        self.all_flux, self.all_flux_err = np.vstack(result).transpose([2, 1, 0])
        # Apply masks to remove rejected frames.
        self.all_flux, self.all_flux_err = (
            self.all_flux[nan_mask][bound_mask],
            self.all_flux_err[nan_mask][bound_mask],
        )

        # Transform unique pixel indices back into (row,column)
        pixels = np.asarray(
            [
                j
                for i in [
                    np.asarray(
                        [
                            np.full(run["ncolumns"], run["row"]),
                            np.arange(
                                run["start_column"],
                                run["start_column"] + run["ncolumns"],
                            ),
                        ]
                    ).T
                    for run in runs
                ]
                for j in i
            ]
        )

        # Pixel mask that tracks moving target
        target_mask = []
        for t in range(len(self.time)):
            target_mask.append(
                np.logical_and(
                    np.isin(pixels.T[0], row[t].ravel()),
                    np.isin(pixels.T[1], column[t].ravel()),
                ),
            )
        self.target_mask = np.asarray(target_mask)

    def reshape_data(self):
        """
        Reshape flux data into cube with shape (len(self.time), self.shape).

        Parameters
        ----------

        Returns
        -------
        """
        if not hasattr(self, "all_flux"):
            raise AttributeError("Must run `get_data()` before reshaping data.")

        self.flux = []
        self.flux_err = []
        # Reshape flux data.
        for t in range(len(self.time)):
            self.flux.append(self.all_flux[t][self.target_mask[t]].reshape(self.shape))
            self.flux_err.append(
                self.all_flux_err[t][self.target_mask[t]].reshape(self.shape)
            )

        self.flux = np.asarray(self.flux)
        self.flux_err = np.asarray(self.flux_err)

    def background_correction(self, method: str = "rolling", **kwargs):
        """
        Apply background correction to reshaped flux data.

        Parameters
        ----------
        method : str
            Method used for background correction. One of `rolling`.

        Returns
        -------
        """
        if not hasattr(self, "all_flux") or not hasattr(self, "flux"):
            raise AttributeError("Must run `get_data()` and `reshape_data()` before computing background.")

        # Get background via chosen method
        if method == "rolling":
            self.bg, self.bg_err = self._bg_rolling_median(**kwargs)

        else:
            raise ValueError(
                "`method` must be one of: `rolling`. Not `{0}`".format(method)
            )

        # Apply background correction
        self.corr_flux = self.flux - self.bg
        self.corr_flux_err = np.sqrt(self.flux_err**2 + self.bg_err**2)

    def _bg_rolling_median(self, nframes: int = 25):
        """
        Calculate the background using a rolling median of nearby frames.

        Parameters
        ----------
        nframes : int
            Number of frames either side of current frame to use in estimate of background.

        Returns
        -------
        bg : ndarray
            Background flux estimate.
            Array with same shape as self.flux.

        bg_err : ndarray
            Error on background flux estimate.
            Array with same shape as self.flux.
        """

        if not hasattr(self, "all_flux"):
            raise AttributeError("Must run `get_data()` before computing background.")

        bg = []
        bg_err = []
        for i in range(len(self.all_flux)):
            # Get flux window.
            flux_window = self.all_flux[
                i - nframes if i >= nframes else 0 : i + nframes + 1
                if i <= len(self.all_flux) - nframes
                else len(self.all_flux)
            ][:, self.target_mask[i]]
            # Compute background flux.
            bg.append(np.nanmedian(flux_window, axis=0).reshape(self.shape))
            # Use the Median Absolute Deviation (MAD) for error on the background flux.
            bg_err.append(
                np.nanmedian(
                    np.abs(flux_window - np.nanmedian(flux_window, axis=0)), axis=0
                ).reshape(self.shape)
            )

        return np.asarray(bg), np.asarray(bg_err)

    def get_aperture(self):
        """
        Get aperture for LC extraction.

        Parameters
        ----------

        Returns
        -------
        """
        raise NotImplementedError("get_aperture() is not yet implemented.")

    def save_data(
        self,
        save_tpf: bool = True,
        save_table: bool = False,
        file_name_tpf: Optional[str] = None,
        save_loc: str = "",
    ):
        """
        Save retrieved pixel data. Can either save all pixels at all times or save a TPF with the same
        format produced by SPOC.

        Parameters
        ----------
        save_tpf : bool
            If True, save a SPOC-like TPF file where each frame is centred on the moving target.
        save_table : bool
            If True, save a table of all retrieved pixel fluxes at all times.
        file_name_tpf : str
            If save_tpf, this is the filename that will be used for the TPF.
            If no filename is given, a default one will be generated.
        save_loc : str
            Directory into which the files will be saved.

        Returns
        -------
        """
        if save_tpf:
            # Save TPF with SPOC format
            self._save_tpf(file_name=file_name_tpf, save_loc=save_loc)

        if save_table:
            # Save table of all flux data
            self._save_table()

    def _save_tpf(self, file_name: Optional[str] = None, save_loc: str = ""):
        """
        Save data and aperture in format that matches SPOC TPFs.

        Parameters
        ----------
        file_name : str
            This is the filename that will be used for the TPF.
            If no filename is given, a default one will be generated.
        save_loc : str
            Directory into which the TPF will be saved.

        Returns
        -------
        """
        if not hasattr(self, "all_flux") or not hasattr(self, "flux") or not hasattr(self, "corr_flux"):
            raise AttributeError("Must run `get_data()`, `reshape_data()` and `background_correction()` before saving TPF.")

        # Create default file name
        if file_name is None:
            file_name = "tess-{0}-s{1:04}-shape{2}x{3}-moving_tp.fits".format(
                str(self.target).replace(" ", ""), self.sector, *self.shape
            )

        if not file_name.endswith(".fits"):
            raise ValueError(
                "`file_name` must be a .fits file. Not `{0}`".format(file_name)
            )
        if len(save_loc) > 0 and not save_loc.endswith("/"):
            save_loc += "/"

        # Compute WCS header
        wcs_header = self._make_wcs_header()

        # Offset between expected target position and center of TPF
        pos_corr1 = self.ephemeris[:, 1] - self.corner[:, 1] - 0.5 * (self.shape[1] - 1)
        pos_corr2 = self.ephemeris[:, 0] - self.corner[:, 0] - 0.5 * (self.shape[0] - 1)

        # Compute TIMECORR
        # >>>>> UPDATE IN FUTURE <<<<<
        time_corr = np.zeros_like(self.time)

        # Define FITS columns
        tform = str(self.corr_flux[0].size) + "E"
        dims = str(self.corr_flux[0].shape[::-1])
        cols = [
            fits.Column(
                name="TIME",
                format="D",
                unit="BJD - 2457000, days",
                disp="D14.7",
                array=self.time,
            ),
            fits.Column(
                name="TIMECORR",
                format="E",
                unit="d",
                disp="E14.7",
                array=time_corr,
            ),
            fits.Column(name="CADENCENO", format="I", array=self.cadence_number),
            # This is included to give the files the same structure as the SPOC files
            fits.Column(
                name="RAW_CNTS",
                format=tform,
                dim=dims,
                unit="e-/s",
                disp="E14.7",
                array=np.zeros_like(self.corr_flux),
            ),
            fits.Column(
                name="FLUX",
                format=tform,
                dim=dims,
                unit="e-/s",
                disp="E14.7",
                array=self.corr_flux,
            ),
            fits.Column(
                name="FLUX_ERR",
                format=tform,
                dim=dims,
                unit="e-/s",
                disp="E14.7",
                array=self.corr_flux_err,
            ),
            fits.Column(
                name="FLUX_BKG",
                format=tform,
                dim=dims,
                unit="e-/s",
                disp="E14.7",
                array=self.bg,
            ),
            fits.Column(
                name="FLUX_BKG_ERR",
                format=tform,
                dim=dims,
                unit="e-/s",
                disp="E14.7",
                array=self.bg_err,
            ),
            fits.Column(
                name="QUALITY",
                format="J",
                disp="B16.16",
                array=self.quality,
            ),
            # >>>>> UPDATE DEFINITION FOR PIXEL QUALITY <<<<<
            fits.Column(
                name="PIXEL_QUALITY",
                format=str(self.corr_flux[0].size) + "J",
                dim=dims,
                disp="B16.16",
                array=np.zeros_like(self.corr_flux),
            ),
            fits.Column(
                name="POS_CORR1",
                format="E",
                unit="pixel",
                disp="E14.7",
                array=pos_corr1,
            ),
            fits.Column(
                name="POS_CORR2",
                format="E",
                unit="pixel",
                disp="E14.7",
                array=pos_corr2,
            ),
            # Original FFI column of lower-left pixel in TPF.
            fits.Column(
                name="CORNER1",
                format="I",
                unit="pixel",
                array=self.corner[:, 1],
            ),
            # Original FFI row of lower-left pixel in TPF.
            fits.Column(
                name="CORNER2",
                format="I",
                unit="pixel",
                array=self.corner[:, 0],
            ),
        ]

        # Create table HDU
        table_hdu = fits.BinTableHDU.from_columns(
            cols,
            header=fits.Header(
                [
                    *self.cube.output_first_header.cards,
                    *get_wcs_header_by_extension(wcs_header, ext=4).cards,
                    *get_wcs_header_by_extension(wcs_header, ext=5).cards,
                    *get_wcs_header_by_extension(wcs_header, ext=6).cards,
                    *get_wcs_header_by_extension(wcs_header, ext=7).cards,
                    *get_wcs_header_by_extension(wcs_header, ext=8).cards,
                ]
            ),
        )
        table_hdu.header["EXTNAME"] = "PIXELS"

        # Create aperture HDU
        # >>>>> UPDATE IN FUTURE WITH REAL APERTURE <<<<<
        aperture_hdu = fits.ImageHDU(
            data=np.ones(self.shape),
            header=fits.Header(
                [*self.cube.output_secondary_header.cards, *wcs_header.cards]
            ),
        )
        aperture_hdu.header["EXTNAME"] = "APERTURE"
        aperture_hdu.header.set("NPIXSAP", None, "Number of pixels in optimal aperture")
        aperture_hdu.header.set(
            "NPIXMISS", None, "Number of op. aperture pixels not collected"
        )

        # Create hdulist and save to file
        self.hdulist = fits.HDUList(
            [self.cube.output_primary_ext, table_hdu, aperture_hdu]
        )
        self.hdulist.writeto(save_loc + file_name, overwrite=True)
        logger.info("Saved TPF to {0}".format(save_loc + file_name))

    def _make_wcs_header(self):
        """
        Make a dummy WCS header for the TPF file. In reality, there is a WCS per timestamp
        that needs to be accounted for.

        Parameters
        ----------

        Returns
        -------
        wcs_header : astropy.io.fits.header.Header
            Dummy WCS header to use in the TPF.
        """
        if not hasattr(self, "shape"):
            raise AttributeError("Must run `get_data()` before making WCS header.")

        # TPF corner (row,column)
        corner = (1, 1)

        # Make a dummy WCS where each pixel in TPF is assigned coordinates 1,1
        row, column = np.meshgrid(
            np.arange(corner[0], corner[0] + self.shape[0]),
            np.arange(corner[1], corner[1] + self.shape[1]),
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

    def _save_table(self):
        """
        Save all data in table.

        Parameters
        ----------

        Returns
        -------
        """
        raise NotImplementedError("_save_table() is not yet implemented.")

    def animate_tpf(self):
        """
        Plot animation of TPF data.

        Parameters
        ----------

        Returns
        -------
        """
        raise NotImplementedError("animate_tpf() is not yet implemented.")

    @staticmethod
    def from_name(target: str, sector: int, time_step: float = 1.0):
        """
        Initialises MovingTargetTPF from target name and TESS sector.

        Parameters
        ----------
        target : str
            JPL/Horizons target ID of e.g. asteroid, comet.
        sector : int
            TESS sector number.
        time_step : float
            Resolution of ephemeris, in days.

        Returns
        -------
        MovingTargetTPF :
            Initialised MovingTargetTPF.
        df_ephem : DataFrame
            Target ephemeris with columns ['time','sector','camera','ccd','column','row'].
                'time' : float with units (JD - 2457000).
                'sector', 'camera', 'ccd' : int
                'column', 'row' : float. These are one-indexed, where the lower left pixel of the FFI is (1,1).
        """

        # Get target ephemeris using tess-ephem
        df_ephem = ephem(target, sector=sector, time_step=time_step)

        # Check whether target was observed in sector.
        if len(df_ephem) == 0:
            raise ValueError("Target {} was not observed in sector {}.".format(target,sector))

        # Add column for time in units (JD - 2457000)
        # >>>>> Note: tess-ephem returns time in JD, not BJD. <<<<<
        df_ephem["time"] = [t.value - 2457000 for t in df_ephem.index.values]
        df_ephem = df_ephem[
            ["time", "sector", "camera", "ccd", "column", "row"]
        ].reset_index(drop=True)

        return MovingTargetTPF(target=target, ephem=df_ephem), df_ephem
