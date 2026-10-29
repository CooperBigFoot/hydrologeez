"""Census cartographic CONUS membership at native Daymet cell centres."""

import zipfile
from pathlib import Path

import netCDF4
import numpy as np
import pyproj
import shapefile
import shapely
from numpy.typing import NDArray
from shapely.ops import transform

STATEFP = [
    "01",
    "04",
    "05",
    "06",
    "08",
    "09",
    "10",
    "11",
    "12",
    "13",
    "16",
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "27",
    "28",
    "29",
    "30",
    "31",
    "32",
    "33",
    "34",
    "35",
    "36",
    "37",
    "38",
    "39",
    "40",
    "41",
    "42",
    "44",
    "45",
    "46",
    "47",
    "48",
    "49",
    "50",
    "51",
    "53",
    "54",
    "55",
    "56",
]
BOUNDARY_URL = "https://www2.census.gov/geo/tiger/GENZ2024/shp/cb_2024_us_state_500k.zip"


def conus_mask(source: Path, boundary: Path) -> NDArray[np.bool_]:
    with zipfile.ZipFile(boundary) as archive:
        projection = archive.read("cb_2024_us_state_500k.prj").decode()
    reader = shapefile.Reader(str(boundary))
    geometries = []
    states: list[str] = []
    for record in reader.iterShapeRecords():
        assert record.record is not None and record.shape is not None
        state = str(record.record["STATEFP"])
        if state in STATEFP:
            states.append(state)
            geometries.append(shapely.geometry.shape(record.shape.__geo_interface__))
    if sorted(states) != STATEFP:
        raise ValueError("Boundary must contain all 48 states and DC exactly once")
    geometry = shapely.union_all(geometries)
    with netCDF4.Dataset(source) as dataset:
        variable = dataset["lambert_conformal_conic"]
        crs = pyproj.CRS.from_cf({a: variable.getncattr(a) for a in variable.ncattrs()})
        project = pyproj.Transformer.from_crs(projection, crs, always_xy=True)
        geometry = transform(project.transform, geometry)
        shapely.prepare(geometry)
        x, y = np.asarray(dataset["x"][:]), np.asarray(dataset["y"][:])
        mask = np.zeros((len(y), len(x)), dtype=bool)
        for row in range(0, len(y), 300):
            xx, yy = np.meshgrid(x, y[row : row + 300])
            mask[row : row + 300] = shapely.covers(geometry, shapely.points(xx, yy))
    return mask
