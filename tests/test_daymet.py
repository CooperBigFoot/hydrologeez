from pathlib import Path

import boto3
import pytest
from moto import mock_aws


def test_archive_detects_corruption_and_reuses_verified_object(tmp_path: Path) -> None:
    from scripts.daymet_archive import Archive

    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="hydrologeez-daymet")
        archive = Archive(client)
        source = tmp_path / "source.nc"
        source.write_bytes(b"original provider bytes")
        receipt = archive.put(source, "raw/source.nc")
        assert archive.put(source, "raw/source.nc") == receipt
        client.put_object(Bucket=archive.bucket, Key="raw/source.nc", Body=b"corruption")
        with pytest.raises(ValueError, match="integrity"):
            archive.put(source, "raw/source.nc")


def test_download_recovers_partial_and_checks_provider_hash(tmp_path: Path) -> None:
    import hashlib

    import requests
    import responses

    from scripts.daymet_acquire import Source, download

    payload = b"provider payload"
    source = Source("prcp", 2024, "https://example.org/source.nc", len(payload), hashlib.sha256(payload).hexdigest())
    target = tmp_path / "source.nc"
    target.with_suffix(".nc.part").write_bytes(b"broken download")
    with responses.RequestsMock() as server:
        server.get(source.url, body=payload)
        download(requests.Session(), source, target)
        download(requests.Session(), source, target)
        assert len(server.calls) == 1
    assert target.read_bytes() == payload
    target.write_bytes(b"bad")
    with pytest.raises(ValueError, match="integrity"):
        download(requests.Session(), source, target)


def test_pet_independent_references_and_missing() -> None:
    import numpy as np
    from numpy.testing import assert_allclose

    from scripts.daymet_prepare import oudin

    assert_allclose(
        oudin(
            np.array([20.0, 20.0, 20.0, -5.0, -6.0, -4.0, np.nan, -6.0]),
            np.array([-20.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, np.nan]),
            np.array([246, 172, 355, 172, 172, 172, 172, 172]),
        ),
        [3.28510161991, 4.27271918029, 1.37947267640, 0, 0, 0.170908767212, np.nan, np.nan],
        rtol=1e-9,
    )


def native_sources(tmp_path: Path) -> dict[str, Path]:
    import netCDF4
    import numpy as np

    paths = {}
    for name, value, unit in [("prcp", 0.0, "mm/day"), ("tmin", 10.0, "degrees C"), ("tmax", 20.0, "degrees C")]:
        paths[name] = tmp_path / f"{name}.nc"
        with netCDF4.Dataset(paths[name], "w") as ds:
            for dim, size in [("time", 365), ("y", 2), ("x", 3), ("nv", 2)]:
                ds.createDimension(dim, size)
            time = ds.createVariable("time", "f8", ("time",))
            time.units = "days since 2024-01-01"
            time.calendar = "standard"
            time[:] = np.arange(365) + 0.5
            ds.createVariable("time_bnds", "f8", ("time", "nv"))[:] = np.column_stack(
                (np.arange(365), np.arange(365) + 1)
            )
            for axis, data in [("x", [0, 1000, 2000]), ("y", [1000, 0])]:
                v = ds.createVariable(axis, "f4", (axis,))
                v.units = "m"
                v[:] = data
            for coord, val in [("lat", 40.0), ("lon", -100.0)]:
                ds.createVariable(coord, "f4", ("y", "x"))[:] = val
            crs = ds.createVariable("lambert_conformal_conic", "i2")
            crs.grid_mapping_name = "lambert_conformal_conic"
            crs.standard_parallel = [25.0, 60.0]
            crs.longitude_of_central_meridian = -100.0
            crs.latitude_of_projection_origin = 42.5
            crs.semi_major_axis = 6378137.0
            crs.inverse_flattening = 298.257223563
            v = ds.createVariable(name, "f4", ("time", "y", "x"), fill_value=-9999.0)
            v.units = unit
            v[:] = value
            v[:, 0, 1] = -9999.0
    return paths


def test_monthly_native_grid_missing_dates_and_interruption(tmp_path: Path) -> None:
    import netCDF4
    import numpy as np
    from numpy.testing import assert_allclose

    from scripts.daymet_prepare import prepare_month

    paths = native_sources(tmp_path)
    mask = np.array([[True, True, False], [True, True, True]])
    output = tmp_path / "feb.nc"
    output.with_suffix(".nc.part").write_bytes(b"interrupted")
    prepare_month(paths, 2024, 2, mask, output)
    with netCDF4.Dataset(output) as ds:
        assert ds["P"].shape == (29, 2, 3)
        assert_allclose(ds["P"][:, 1, :], 0)
        assert_allclose(ds["T"][:, 1, :], 15)
        assert ds["PET"][:].mask[:, 0, 1:].all()
        decoded = np.asarray(netCDF4.num2date(ds["time"][:], ds["time"].units, ds["time"].calendar))
        assert (decoded[-1].month, decoded[-1].day) == (2, 29)
        assert_allclose(ds["x"][:], [0, 1000, 2000])
    with netCDF4.Dataset(paths["tmin"], "a") as ds:
        ds["x"][0] = 2
    with pytest.raises(ValueError, match="coordinates"):
        prepare_month(paths, 2024, 3, mask, tmp_path / "march.nc")


def test_end_to_end_restart_after_uploaded_month(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    import numpy as np
    import requests
    import responses

    from scripts import daymet
    from scripts.daymet_acquire import Source
    from scripts.daymet_archive import Archive, checksum

    paths = native_sources(tmp_path)
    sources = [
        Source(name, 2024, f"https://example.org/{name}.nc", path.stat().st_size, checksum(path))
        for name, path in paths.items()
    ]
    snapshot = tmp_path / "inventory.json"
    snapshot.write_text("{}")
    boundary = tmp_path / "boundary.zip"
    boundary.write_bytes(b"fixture boundary")
    monkeypatch.setattr(daymet, "inventory", lambda _: sources)
    monkeypatch.setattr(daymet, "conus_mask", lambda *_: np.ones((2, 3), dtype=bool))
    stage = tmp_path / "stage"
    original = daymet.prepare_month
    months = []

    def interrupted(paths: dict[str, Path], year: int, month: int, mask: np.ndarray, output: Path) -> None:
        months.append(month)
        if month == 2:
            raise RuntimeError("interrupted")
        original(paths, year, month, mask, output)

    monkeypatch.setattr(daymet, "prepare_month", interrupted)
    with mock_aws(), responses.RequestsMock(assert_all_requests_are_fired=False) as server:
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="hydrologeez-daymet")
        archive = Archive(client)
        for source in sources:
            server.get(source.url, body=paths[source.variable].read_bytes())
        with pytest.raises(RuntimeError, match="interrupted"):
            daymet.run(snapshot, [snapshot], boundary, stage, archive, requests.Session())
        assert all((stage / source.name).exists() for source in sources)
        keys = [item["Key"] for item in client.list_objects_v2(Bucket=archive.bucket)["Contents"]]
        assert not any(key.endswith("complete.json") for key in keys)
        monkeypatch.setattr(daymet, "prepare_month", original)
        daymet.run(snapshot, [snapshot], boundary, stage, archive, requests.Session())
        assert len(server.calls) == 3
        assert not any((stage / source.name).exists() for source in sources)
        keys = [item["Key"] for item in client.list_objects_v2(Bucket=archive.bucket)["Contents"]]
        completion = next(key for key in keys if key.endswith("/complete.json") and "/2024/" not in key)
        record = json.loads(client.get_object(Bucket=archive.bucket, Key=completion)["Body"].read())
        assert len(record["objects"]) == 16
        daymet.run(snapshot, [snapshot], boundary, stage, archive, requests.Session())
        assert len(server.calls) == 3


def test_inventory_discovers_exact_variables_and_preserves_provider(tmp_path: Path) -> None:
    import json

    import requests
    import responses

    from scripts.daymet_acquire import inventory
    from scripts.daymet_inventory import discover

    items = []
    for name in ["prcp", "tmin", "tmax", "srad"]:
        filename = f"daymet_v4_daily_na_{name}_1980.nc"
        items.append(
            {
                "meta": {"revision-id": 1},
                "umm": {
                    "GranuleUR": f"Daymet_Daily_V4R1.{filename}",
                    "DataGranule": {
                        "ArchiveAndDistributionInformation": [
                            {"SizeInBytes": 123, "Checksum": {"Algorithm": "SHA-256", "Value": "a" * 64}}
                        ]
                    },
                    "TemporalExtent": {
                        "RangeDateTime": {"BeginningDateTime": "1980-01-01", "EndingDateTime": "1980-12-30"}
                    },
                    "RelatedUrls": [
                        {
                            "Type": "GET DATA",
                            "URL": f"https://data.ornldaac.earthdata.nasa.gov/protected/daymet/Daymet_Daily_V4R1/data/{filename}",
                        }
                    ],
                },
            }
        )
    body = json.dumps({"items": items})
    with responses.RequestsMock() as server:
        server.get("https://cmr.earthdata.nasa.gov/search/granules.umm_json", body=body)
        discover(requests.Session(), tmp_path)
    assert len(inventory(tmp_path / "inventory.json")) == 3
    assert (tmp_path / "cmr-page-0001.json").read_text() == body


def test_download_stream_failure_retries_without_accepting_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    import requests
    import responses

    from scripts.daymet_acquire import Source, download

    payload = b"complete payload"
    source = Source("prcp", 2024, "https://example.org/source.nc", len(payload), hashlib.sha256(payload).hexdigest())
    target = tmp_path / "source.nc"
    original = requests.Response.iter_content

    def broken(self: requests.Response, chunk_size: int = 1, decode_unicode: bool = False):
        yield b"partial"
        raise requests.ConnectionError("interruption")

    with responses.RequestsMock() as server:
        server.get(source.url, body=payload)
        monkeypatch.setattr(requests.Response, "iter_content", broken)
        with pytest.raises(requests.ConnectionError):
            download(requests.Session(), source, target)
        assert not target.exists()
        monkeypatch.setattr(requests.Response, "iter_content", original)
        download(requests.Session(), source, target)
    assert target.read_bytes() == payload


def test_earthdata_does_not_send_credentials_to_cdn() -> None:
    import responses

    from scripts.daymet_acquire import EarthdataSession

    with responses.RequestsMock() as server:
        server.get(
            "https://data.ornldaac.earthdata.nasa.gov/data",
            status=302,
            headers={"Location": "https://urs.earthdata.nasa.gov/login"},
        )
        server.get(
            "https://urs.earthdata.nasa.gov/login", status=302, headers={"Location": "https://cdn.example.org/object"}
        )
        server.get("https://cdn.example.org/object", body=b"file")
        session = EarthdataSession()
        session.auth = ("test-user", "test-secret")
        session.get("https://data.ornldaac.earthdata.nasa.gov/data")
        assert "Authorization" in server.calls[1].request.headers
        assert "Authorization" not in server.calls[2].request.headers
