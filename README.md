[![pytest](https://github.com/altuson/tess-asteroids/actions/workflows/test.yml/badge.svg)](https://github.com/altuson/tess-asteroids/actions/workflows/test.yml)
[![mypy](https://github.com/altuson/tess-asteroids/actions/workflows/mypy.yml/badge.svg)](https://github.com/altuson/tess-asteroids/actions/workflows/mypy.yml/)
[![ruff](https://github.com/altuson/tess-asteroids/actions/workflows/ruff.yml/badge.svg)](https://github.com/altuson/tess-asteroids/actions/workflows/ruff.yml)

# tess-asteroids

`tess-asteroids` allows you to make TPFs and LCs for any object that moves through the TESS field of view, for example solar system asteroids, comets or minor planets.

## Example use

### Making a TPF

You can create a TPF that tracks a moving object by providing the ephemeris, as follows:

```python
import numpy as np
import pandas as pd
from tess_asteroids import MovingObjectTPF

# Create an artificial ephemeris
time = np.linspace(1790.5, 1795.5, 100)
ephem = pd.DataFrame({
            "time": time,
            "sector": np.full(len(time), 18),
            "camera": np.full(len(time), 3),
            "ccd": np.full(len(time), 2),
            "column": np.linspace(500, 600, len(time)),
            "row": np.linspace(1000, 900, len(time)),
        })

# Initialise TPF
tpf = MovingObjectTPF(ephem)

# Get TPF data - this step queries AWS to get TESS data
time, flux, flux_err, quality, corner, ephemeris, difference = tpf.get_data()

```

This will return a TPF cutout, centred on the moving object, from the full FFI. A few things to note about the format of the ephemeris:
- `time` must have units BTJD = BJD - 2457000.
- `sector`, `camera`, `ccd` must each have one unique value.
- `column`, `row` must be one-indexed, where the lower left pixel of the FFI has value (1,1).

There are a few optional parameters in the `get_data()` function:
- `shape` controls the shape of the TPF cutout. Default : (11,11).
- `difference_image` determines whether or not the function returns a difference image as well as the raw flux. Default : True.
- `verbose` enables you to print the time taken to retrieve the TESS data. Default : False.

These settings can be changed as follows:

```python
# Get TPF data - changed default settings
time, flux, flux_err, quality, corner, ephemeris = tpf.get_data(shape=(20,10), difference_image=False, verbose=True)
```

Instead of inputting an ephemeris, you can also create a TPF using the name of an object from the JPL/Horizons database and the TESS sector. This will use `tess-ephem` to compute the ephemeris for you.

```python
# Initialise TPF for asteroid 1998 YT6 from TESS sector 6.
tpf, ephem = MovingObjectTPF.from_name("1998 YT6", sector=6)

# Get TPF data
time, flux, flux_err, quality, corner, ephemeris, difference = tpf.get_data()
```
