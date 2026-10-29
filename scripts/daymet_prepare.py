"""Native-grid monthly Daymet forcings; Oudin Eq3 with FAO56 astronomy."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import netCDF4
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def oudin(temperature: Array, latitude: Array, day: NDArray[np.int64]) -> Array:
    """mm/day; FAO56 Eqs21–25 Re, Oudin2005 Eq3, lambda=2.45 MJ/kg."""
    phi = np.deg2rad(latitude)
    angle = 2 * np.pi * day / 365
    distance = 1 + 0.033 * np.cos(angle)
    declination = 0.409 * np.sin(angle - 1.39)
    sunset = np.arccos(np.clip(-np.tan(phi) * np.tan(declination), -1, 1))
    radiation = (
        1440
        / np.pi
        * 0.0820
        * distance
        * (sunset * np.sin(phi) * np.sin(declination) + np.cos(phi) * np.cos(declination) * np.sin(sunset))
    )
    return radiation / 2.45 * np.maximum(temperature + 5, 0) / 100


def dates(dataset: netCDF4.Dataset, year: int) -> list[datetime]:
    time = dataset["time"]
    decoded = netCDF4.num2date(
        time[:], time.units, calendar=time.calendar, only_use_cftime_datetimes=False, only_use_python_datetimes=True
    )
    result = [datetime(d.year, d.month, d.day, d.hour, d.minute, d.second) for d in np.asarray(decoded)]
    expected = [datetime(year, 1, 1, 12) + timedelta(days=i) for i in range(365)]
    if result != expected:
        raise ValueError("Unexpected native Daymet dates (365 local noon labels required)")
    bounds = np.asarray(dataset["time_bnds"][:])
    if bounds.shape != (365, 2) or not np.array_equal(bounds, np.asarray(time[:])[:, None] + [-0.5, 0.5]):
        raise ValueError("Unexpected native Daymet daily bounds")
    return result


def values(variable: netCDF4.Variable, index: tuple[slice, ...]) -> Array:
    return np.asarray(np.ma.filled(variable[index].astype(np.float64), np.nan))


def prepare_month(paths: dict[str, Path], year: int, month: int, mask: NDArray[np.bool_], output: Path) -> None:
    """Bound memory with 10-day x 300 x 300 tiles; commit only closed files."""
    from contextlib import ExitStack

    netCDF4.set_chunk_cache(size=16 * 1024 * 1024, nelems=1009, preemption=0.75)
    with ExitStack() as stack:
        sources = {name: stack.enter_context(netCDF4.Dataset(path)) for name, path in paths.items()}
        reference = sources["prcp"]
        decoded = dates(reference, year)
        if not np.all(np.diff(reference["x"][:]) == 1000) or not np.all(np.diff(reference["y"][:]) == -1000):
            raise ValueError("Expected native 1km Daymet grid")
        mapping = "lambert_conformal_conic"
        crs = {attr: reference[mapping].getncattr(attr) for attr in reference[mapping].ncattrs()}
        for name, dataset in sources.items():
            if dates(dataset, year) != decoded:
                raise ValueError("Mismatched source dates")
            expected_unit = "mm/day" if name == "prcp" else "degrees C"
            if dataset[name].units != expected_unit or dataset[name].dimensions != ("time", "y", "x"):
                raise ValueError("Unexpected source variable units/dimensions")
            for axis in ("x", "y", "time", "time_bnds"):
                if not np.array_equal(dataset[axis][:], reference[axis][:]):
                    raise ValueError(f"Mismatched source coordinates: {axis}")
            if dataset["x"].units != "m" or dataset["y"].units != "m":
                raise ValueError("Expected projected coordinates in metres")
            other = {attr: dataset[mapping].getncattr(attr) for attr in dataset[mapping].ncattrs()}
            if other.keys() != crs.keys() or any(not np.array_equal(other[k], crs[k]) for k in crs):
                raise ValueError("Mismatched source projection")
        if mask.shape != (len(reference.dimensions["y"]), len(reference.dimensions["x"])) or not mask.any():
            raise ValueError("Invalid CONUS mask")
        yy, xx = np.nonzero(mask)
        y0, y1, x0, x1 = int(yy.min()), int(yy.max() + 1), int(xx.min()), int(xx.max() + 1)
        indices = [i for i, date in enumerate(decoded) if date.month == month]
        start, stop = indices[0], indices[-1] + 1
        partial = output.with_suffix(output.suffix + ".part")
        with netCDF4.Dataset(partial, "w") as target:
            for dim, size in (("time", stop - start), ("y", y1 - y0), ("x", x1 - x0), ("nv", 2)):
                target.createDimension(dim, size)
            target.Conventions = "CF-1.8"
            target.source = "Daymet V4 R1; NASA C2532426483-ORNL_CLOUD; DOI 10.3334/ORNLDAAC/2129"
            target.pet_method = "Oudin2005 Eq3; daily (tmin+tmax)/2; FAO56 Eqs21-25 extraterrestrial radiation; lambda=2.45 MJ/kg; solar constant=.0820 MJ/m2/min; denominator=365"
            target.references = (
                "https://doi.org/10.1016/j.jhydrol.2004.08.026 https://www.fao.org/4/x0490e/x0490e07.htm"
            )
            target.calendar_note = "Native local-day labels, not UTC alignment; leap Feb29 retained and Dec31 omitted"
            target.mask_note = "Census2024 1:500000 contiguous48+DC, projected cell-center covers; simplified coast; source missing retained"
            target.source_index_window = [y0, y1, x0, x1]
            for name, index in (
                ("x", slice(x0, x1)),
                ("y", slice(y0, y1)),
                ("time", slice(start, stop)),
                ("time_bnds", slice(start, stop)),
            ):
                original = reference[name]
                copied = target.createVariable(name, original.dtype, original.dimensions)
                copied.setncatts({a: original.getncattr(a) for a in original.ncattrs() if a != "_FillValue"})
                copied[:] = original[index]
            target.createVariable(mapping, "i2").setncatts(crs)
            target.createVariable("conus_mask", "u1", ("y", "x"), zlib=True)[:] = mask[y0:y1, x0:x1]
            for coord in ("lat", "lon"):
                v = target.createVariable(coord, "f4", ("y", "x"), zlib=True)
                v.units = "degrees_north" if coord == "lat" else "degrees_east"
            for name, unit in (("P", "mm/day"), ("T", "degrees_Celsius"), ("PET", "mm/day")):
                v = target.createVariable(
                    name,
                    "f4",
                    ("time", "y", "x"),
                    fill_value=np.float32(-9999),
                    zlib=True,
                    complevel=4,
                    shuffle=True,
                    chunksizes=(min(10, stop - start), min(300, y1 - y0), min(300, x1 - x0)),
                )
                v.units = unit
                v.grid_mapping = mapping
                v.coordinates = "lat lon"
            for y in range(y0, y1, 300):
                for x in range(x0, x1, 300):
                    ys, xs = slice(y, min(y + 300, y1)), slice(x, min(x + 300, x1))
                    oy, ox = slice(y - y0, ys.stop - y0), slice(x - x0, xs.stop - x0)
                    inside = mask[ys, xs]
                    latitude = values(reference["lat"], (ys, xs))
                    for coord in ("lat", "lon"):
                        coords = values(reference[coord], (ys, xs))
                        for dataset in sources.values():
                            if not np.array_equal(coords, values(dataset[coord], (ys, xs)), equal_nan=True):
                                raise ValueError("Mismatched geographic coordinates")
                        target[coord][oy, ox] = coords
                    for day in range(start, stop, 10):
                        end = min(day + 10, stop)
                        index = (slice(day, end), ys, xs)
                        precipitation = values(reference["prcp"], index)
                        temperature = (
                            values(sources["tmin"]["tmin"], index) + values(sources["tmax"]["tmax"], index)
                        ) / 2
                        doy = np.array([d.timetuple().tm_yday for d in decoded[day:end]], dtype=np.int64)[:, None, None]
                        pet = oudin(temperature, latitude[None, :, :], doy)
                        for name, data in (("P", precipitation), ("T", temperature), ("PET", pet)):
                            data[:, ~inside] = np.nan
                            target[name][slice(day - start, end - start), oy, ox] = np.ma.masked_invalid(data)
        partial.replace(output)
