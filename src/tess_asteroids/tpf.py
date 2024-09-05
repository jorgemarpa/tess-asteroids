import time
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
from tess_ephem import ephem
from tesscube import TESSCube
from tesscube.utils import _sync_call, convert_coordinates_to_runs


class MovingObjectTPF:
    """
    Create a TPF for a moving object (e.g. asteroid) from a TESS FFI.

    It includes methods to efficiently retrieve the data, define an aperture mask
    to extract a LC, save a TPF fits file and animate the TPF data.

    Parameters
    ----------
    ephem : DataFrame
            Object ephemeris with columns ['time','sector','camera','ccd','column','row'].
                    'time' : float with units (BJD - 2457000).
                    'sector', 'camera', 'ccd' : int
                    'column', 'row' : float. These must be one-indexed, where the lower left pixel of the FFI is (1,1).
    """

    def __init__(self, ephem: pd.DataFrame):
        self.ephem = ephem

        # Check self.ephem['time'] has correct units
        if min(self.ephem["time"]) >= 2457000:
            raise ValueError("ephem['time'] must have units (BJD - 2457000).")

        # Check self.ephem['sector'] has one unique value
        if self.ephem["sector"].nunique() == 1:
            self.sector = int(self.ephem["sector"][0])
        else:
            raise ValueError("ephem['sector'] must have one unique value.")

        # Check if object is only observed on one camera/ccd during sector
        if self.ephem["camera"].nunique() == 1 and self.ephem["ccd"].nunique() == 1:
            self.camera = int(self.ephem["camera"][0])
            self.ccd = int(self.ephem["ccd"][0])
        else:
            # >>>>> INCLUDE A WAY TO GET MULTIPLE CUBES. <<<<<
            raise NotImplementedError(
                "Object crosses multiple camera/ccd. Not yet implemented."
            )

        # Initialise tesscube
        self.cube = TESSCube(sector=self.sector, camera=self.camera, ccd=self.ccd)

    def refine_coordinates(self):
        """
        Apply correction to object ephemeris.
        """
        raise NotImplementedError("refine_coordinates() is not yet implemented.")

    def get_data(
        self,
        tpf_type: str = "tracking",
        shape: Tuple[int, int] = (11, 11),
        verbose: bool = False,
    ):
        """
        Get TPF data for a moving object from a TESS FFI.

        Parameters
        ----------
        tpf_type : str
                Type of TPF to return, either "tracking" (cutout that tracks object)
                or "static" (all pixels that object crosses, with buffer).
        shape : Tuple(int,int)
                Shape is defined as (row,column) in pixels.
                If tpf_type = "tracking", this is the shape of the cutout.
                If tpf_type = "static", this is the buffer around the object.
        verbose : bool
                If True, print statements.

        Returns
        -------
        time : list
                FFI timestamps of each frame in the data cube.
        flux : list
                Flux cube.
        flux_err : list
                Flux error cube.
        quality : list
                SPOC quality flag of each frame in the data cube.
        corner : list
                Original FFI (row,column) of lower left pixel in TPF.
        ephemeris : list
                Predicted (row,column) of object from ephem.
        """

        self.type = tpf_type

        # Use interpolation to get object (row,column) at cube time.
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

        # Corner coordinates
        self.corner = np.ceil(
            np.asarray([row_interp - shape[0] / 2, column_interp - shape[1] / 2]).T
        ).astype(int)

        # Pixel positions to retrieve around object
        row, column = (
            np.mgrid[: shape[0], : shape[1]][:, None, :, :] + self.corner.T[:, :, None, None]
        )

        # Remove frames that include pixels outide bounds of FFI.
        # >>>>> COULD KEEP FRAMES WITH PIXELS OUTSIDE OF BOUNDS AND FILL WITH NANS INSTEAD? <<<<<
        bound_mask = np.logical_and(
            [r.all() for r in np.logical_and(row[:, :, 0] >= 1, row[:, :, 0] <= 2078)],
            [
                c.all()
                for c in np.logical_and(column[:, 0, :] >= 1, column[:, 0, :] <= 2136)
            ],
        )
        self.time = self.cube.time[nan_mask][bound_mask]
        self.quality = self.cube.quality[nan_mask][bound_mask]
        self.ephemeris = np.asarray([row_interp[bound_mask], column_interp[bound_mask]]).T
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
            warnings.warn(
                "Some of the requested pixels are outside of the FFI bounds (1<=row<=2078, 1<=col<=2136) and will not be returned.",
                UserWarning,
                stacklevel=2,
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
            warnings.warn(
                "Some of the requested pixels are outside of the FFI science array (1<=row<=2048, 45<=col<=2092), but they will be included in your TPF.",
                UserWarning,
                stacklevel=2,
            )

        # Convert pixels to byte runs
        runs = convert_coordinates_to_runs(pixel_coordinates)

        # Retrieve the data
        if verbose:
            print("Started data retrieval.")
            start_time = time.time()
        result = _sync_call(self.cube.async_get_data_per_rows, runs)
        if verbose:
            print(
                "Finished data retrieval in {0:.2f} sec.".format(
                    time.time() - start_time
                )
            )

        # Split result into flux, flux_err and reshape into (ntimes, npixels)
        self.all_flux, self.all_flux_err = np.vstack(result).transpose([2, 1, 0])
        # Apply masks to remove rejected frames.
        self.all_flux, self.all_flux_err = self.all_flux[nan_mask][bound_mask], self.all_flux_err[nan_mask][bound_mask]

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

        # Convert data to TPF that moves with object over time.
        if tpf_type == "tracking":
            self.flux = []
            self.flux_err = []
            self.target_indices = []

            for t in range(len(self.time)):
                self.target_indices.append(np.logical_and(
                    np.isin(pixels.T[0], row[t].ravel()),
                    np.isin(pixels.T[1], column[t].ravel()),
                ))
                self.flux.append(self.all_flux[t][self.target_indices[-1]].reshape(shape))
                self.flux_err.append(self.all_flux_err[t][self.target_indices[-1]].reshape(shape))

            self.flux = np.asarray(self.flux) 
            self.flux_err = np.asarray(self.flux_err)
            self.target_indices = np.asarray(self.target_indices)

        elif tpf_type == "static":
            raise NotImplementedError("Not yet implemented static TPF.")

        else:
            raise ValueError(
                "`tpf_type` must be `tracking` or `static`. Not `{0}`".format(tpf_type)
            )

    def background_correction(self, method='rolling', **kwargs):
        """
        Get background correction for TPF.
        """
        if not hasattr(self, 'all_flux'):
            raise AttributeError("Must run `get_data()` before computing background.")

        if method == 'rolling':
            self.bg, self.bg_err = self._bg_rolling_median(**kwargs)

        else:
           raise ValueError("`method` must be one of: `rolling`. Not `{0}`".format(method)) 
        
        self.corr_flux = self.flux - self.bg
        self.corr_flux_err = np.sqrt(self.flux_err**2 + self.bg_err**2)

    def _bg_rolling_median(self, nframes=25):

        if not hasattr(self, 'all_flux'):
            raise AttributeError("Must run `get_data()` before computing background.")
        
        # >>>>> ERROR ON BG - is using median absolute deviation for error on median appropriate? <<<<<
        bg = []
        bg_err = []
        for i in range(len(self.all_flux)):
            flux_window = self.all_flux[i - nframes if i >= nframes else 0 : i + nframes + 1 if i <= len(self.all_flux) - nframes else len(self.all_flux)][:, self.target_indices[i]]
            bg.append(np.nanmedian(flux_window, axis=0).reshape(np.shape(self.flux[i])))
            bg_err.append(np.nanmedian(np.abs(flux_window-np.nanmedian(flux_window, axis=0)), axis=0).reshape(np.shape(self.flux[i])))

        return np.asarray(bg), np.asarray(bg_err)

    def get_aperture(self):
        """
        Get aperture for LC extraction.
        """
        raise NotImplementedError("get_aperture() is not yet implemented.")

    def save_tpf(self):
        """
        Save data and aperture in format that matches SPOC TPFs.
        """
        raise NotImplementedError("save_tpf() is not yet implemented.")

    def animate_tpf(self):
        """
        Plot animation of TPF data.
        """
        raise NotImplementedError("animate_tpf() is not yet implemented.")

    @staticmethod
    def from_name(target: str, sector: int, time_step: float = 1.0):
        """
        Initialises MovingObjectTPF from object name and TESS sector.

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
        MovingObjectTPF :
                Initiliased MovingObjectTPF.
        df_ephem : DataFrame
                Object ephemeris with columns ['time','sector','camera','ccd','column','row'].
                        'time' : float with units (JD - 2457000).
                        'sector', 'camera', 'ccd' : int
                        'column', 'row' : float. These are one-indexed, where the lower left pixel of the FFI is (1,1).
        """

        # Get object ephemeris using tess-ephem
        df_ephem = ephem(target, sector=sector, time_step=time_step)

        # Add column for time in units (JD - 2457000)
        # >>>>> Note: tess-ephem returns time in JD, not BJD. <<<<<
        df_ephem["time"] = [t.value - 2457000 for t in df_ephem.index.values]
        df_ephem = df_ephem[
            ["time", "sector", "camera", "ccd", "column", "row"]
        ].reset_index(drop=True)

        return MovingObjectTPF(ephem=df_ephem), df_ephem
