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
        from scripts import daymet_acquire

        original_download = daymet_acquire.download

        def failed_download(session: requests.Session, source: Source, target: Path) -> None:
            if source.variable == "tmin":
                raise OSError("source transfer failed")
            original_download(session, source, target)

        monkeypatch.setattr(daymet_acquire, "download", failed_download)
        with pytest.raises(OSError, match="source transfer failed"):
            daymet.run(snapshot, [snapshot], boundary, stage, archive, requests.Session)
        keys = [item["Key"] for item in client.list_objects_v2(Bucket=archive.bucket)["Contents"]]
        assert not any(key.endswith("complete.json") or key.startswith("raw/") for key in keys)
        assert months == []
        monkeypatch.setattr(daymet_acquire, "download", original_download)
        with pytest.raises(RuntimeError, match="interrupted"):
            daymet.run(snapshot, [snapshot], boundary, stage, archive, requests.Session)
        assert all((stage / source.name).exists() for source in sources)
        keys = [item["Key"] for item in client.list_objects_v2(Bucket=archive.bucket)["Contents"]]
        assert not any(key.endswith("complete.json") for key in keys)
        monkeypatch.setattr(daymet, "prepare_month", original)
        daymet.run(snapshot, [snapshot], boundary, stage, archive, requests.Session)
        assert len(server.calls) == 3
        assert not any((stage / source.name).exists() for source in sources)
        keys = [item["Key"] for item in client.list_objects_v2(Bucket=archive.bucket)["Contents"]]
        completion = next(key for key in keys if key.endswith("/complete.json") and "/2024/" not in key)
        record = json.loads(client.get_object(Bucket=archive.bucket, Key=completion)["Body"].read())
        assert len(record["objects"]) == 16
        daymet.run(snapshot, [snapshot], boundary, stage, archive, requests.Session)
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


def test_empty_receipt_cannot_complete_delivery() -> None:
    from scripts.daymet import verified_receipt
    from scripts.daymet_archive import Archive

    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="hydrologeez-daymet")
        client.put_object(Bucket="hydrologeez-daymet", Key="complete.json", Body=b'{"objects": []}')
        with pytest.raises(ValueError, match="coverage"):
            verified_receipt(Archive(client), "complete.json", {"required.nc": None})
        client.put_object(
            Bucket="hydrologeez-daymet",
            Key="complete.json",
            Body=b'{"objects": [{"key":"required.nc","bytes":1,"sha256":"bad"}]}',
        )
        with pytest.raises(ValueError, match="provider"):
            verified_receipt(Archive(client), "complete.json", {"required.nc": (2, "expected")})


def test_earthdata_rejects_http_downgrade() -> None:
    import requests

    from scripts.daymet_acquire import EarthdataSession

    session = EarthdataSession()
    session.auth = ("user", "secret")
    request = requests.Request("GET", "http://urs.earthdata.nasa.gov/login").prepare()
    request.headers["Authorization"] = "Basic secret"
    session.rebuild_auth(request, requests.Response())
    assert "Authorization" not in request.headers


@pytest.mark.parametrize("keys", [[], ["first.nc"], ["first.nc", "wrong.nc"], ["first.nc", "first.nc"]])
def test_receipt_requires_exact_coverage(keys: list[str]) -> None:
    import json

    from scripts.daymet import verified_receipt
    from scripts.daymet_archive import Archive

    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="hydrologeez-daymet")
        client.put_object(
            Bucket="hydrologeez-daymet",
            Key="receipt.json",
            Body=json.dumps({"objects": [{"key": key, "bytes": 1, "sha256": "bad"} for key in keys]}),
        )
        with pytest.raises(ValueError, match="coverage"):
            verified_receipt(Archive(client), "receipt.json", {"first.nc": None, "second.nc": None})


def test_dependency_changes_isolate_and_archive_processing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import requests

    from scripts import daymet
    from scripts.daymet_archive import Archive

    scripts = tmp_path / "scripts"
    scripts.mkdir()
    monkeypatch.setattr(daymet, "__file__", str(scripts / "daymet.py"))
    # Isolate provenance from the already-covered annual acquisition loop.
    monkeypatch.setattr(daymet, "inventory", lambda _: [])
    snapshot = tmp_path / "inventory.json"
    snapshot.write_text("{}")
    boundary = tmp_path / "boundary.zip"
    boundary.write_bytes(b"boundary")
    lock = tmp_path / "uv.lock"
    lock.write_text("version=1")
    config = tmp_path / "pyproject.toml"
    config.write_text("[project]")
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="hydrologeez-daymet")
        archive = Archive(client)
        for change in (None, lock, config):
            if change is not None:
                change.write_text(change.read_text() + "\n# changed")
            daymet.run(snapshot, [snapshot], boundary, tmp_path / "stage", archive, requests.Session)
        keys = [item["Key"] for item in client.list_objects_v2(Bucket=archive.bucket)["Contents"]]
        for name in ("uv.lock", "pyproject.toml"):
            assert len([key for key in keys if key.endswith("/provenance/" + name)]) == 3


@pytest.mark.parametrize("fail", [False, True])
def test_annual_downloads_overlap_with_private_sessions_and_join_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    from threading import Barrier, Lock

    import requests

    from scripts import daymet_acquire
    from scripts.daymet_acquire import Source, download_annual

    barrier = Barrier(3, timeout=5)
    lock = Lock()
    sessions: list[requests.Session] = []
    active = 0
    peak = 0
    finished = []

    def transfer(session: requests.Session, source: Source, target: Path) -> None:
        nonlocal active, peak
        with lock:
            sessions.append(session)
            active += 1
            peak = max(peak, active)
        try:
            assert not session.cookies
            session.cookies.set("variable", source.variable)
            barrier.wait()
            if fail and source.variable == "tmin":
                raise OSError("disk failure")
            target.write_bytes(source.variable.encode())
        finally:
            with lock:
                active -= 1
                finished.append(source.variable)

    monkeypatch.setattr(daymet_acquire, "download", transfer)
    sources = [Source(name, 2024, f"https://example.org/{name}.nc", 1, "unused") for name in ("prcp", "tmin", "tmax")]
    if fail:
        with pytest.raises(OSError, match="disk failure"):
            download_annual(sources, tmp_path, requests.Session)
    else:
        download_annual(sources, tmp_path, requests.Session)
    assert peak == 3 and active == 0
    assert len({id(session) for session in sessions}) == 3
    assert sorted(finished) == ["prcp", "tmax", "tmin"]


def test_resume_rejects_unrecognized_provenance(tmp_path: Path) -> None:
    import json

    from scripts.daymet import resume_processing
    from scripts.daymet_archive import Archive

    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="hydrologeez-daymet")
        client.put_object(
            Bucket="hydrologeez-daymet",
            Key="prepared/" + "a" * 64 + "/provenance/processing.json",
            Body=json.dumps({"code_sha256": "wrong"}),
        )
        with pytest.raises(ValueError, match="provenance"):
            resume_processing(Archive(client), "a" * 64, {}, tmp_path)


def test_resume_authenticates_original_science_and_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hashlib
    import json

    from scripts import daymet
    from scripts.daymet_archive import Archive

    scripts = tmp_path / "scripts"
    scripts.mkdir()
    names = (
        "daymet.py",
        "daymet_acquire.py",
        "daymet_archive.py",
        "daymet_inventory.py",
        "daymet_mask.py",
        "daymet_prepare.py",
    )
    digest = hashlib.sha256()
    for name in names:
        (scripts / name).write_bytes(name.encode())
        digest.update(name.encode())
        digest.update(name.encode())
    monkeypatch.setattr(daymet, "RESUMABLE_CODE_SHA256", digest.hexdigest())
    original = {
        "code_sha256": digest.hexdigest(),
        "dependencies": {"uv.lock": hashlib.sha256(b"lock").hexdigest()},
        "inventory_sha256": "input",
    }
    identity = hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()
    current = dict(original, code_sha256="new acquisition")
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="hydrologeez-daymet")
        prefix = f"prepared/{identity}/provenance"
        for key, body in [("processing.json", json.dumps(original).encode()), ("uv.lock", b"lock")] + [
            (f"scripts/{n}", n.encode()) for n in names
        ]:
            client.put_object(Bucket="hydrologeez-daymet", Key=f"{prefix}/{key}", Body=body)
        archive = Archive(client)
        assert daymet.resume_processing(archive, identity, current, tmp_path) == original
        with pytest.raises(ValueError, match="inventory"):
            daymet.resume_processing(archive, identity, dict(current, inventory_sha256="different"), tmp_path)
        (scripts / "daymet_prepare.py").write_text("changed science")
        with pytest.raises(ValueError, match="Scientific"):
            daymet.resume_processing(archive, identity, current, tmp_path)


@pytest.mark.parametrize("fail", [False, True])
def test_prefetch_overlaps_only_next_year_and_joins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    from threading import Event

    import numpy as np
    import requests

    from scripts import daymet
    from scripts.daymet_acquire import Source
    from scripts.daymet_archive import Archive

    started = Event()
    release = Event()
    finished = Event()
    sources = [
        Source(name, year, f"https://example.org/{name}_{year}.nc", 1, "unused")
        for year in (2024, 2025)
        for name in ("prcp", "tmin", "tmax")
    ]
    monkeypatch.setattr(daymet, "inventory", lambda _: sources)
    monkeypatch.setattr(daymet, "conus_mask", lambda *_: np.ones((1, 1), dtype=bool))

    def download(sources: list[Source], stage: Path, factory: object) -> None:
        if sources[0].year == 2025 and not finished.is_set():
            started.set()
            try:
                assert release.wait(5)
                if fail:
                    raise OSError("prefetch failed")
            finally:
                finished.set()
        for source in sources:
            (stage / source.name).write_bytes(b"x")

    def prepare(paths: dict[str, Path], year: int, month: int, mask: object, output: Path) -> None:
        if year == 2024 and month == 1:
            assert started.wait(5)
            release.set()
            assert finished.wait(5)
        output.write_bytes(b"prepared")

    monkeypatch.setattr(daymet, "download_annual", download)
    monkeypatch.setattr(daymet, "prepare_month", prepare)
    snapshot = tmp_path / "inventory.json"
    snapshot.write_text("{}")
    boundary = tmp_path / "boundary.zip"
    boundary.write_bytes(b"boundary")
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="hydrologeez-daymet")
        args = (snapshot, [snapshot], boundary, tmp_path / "stage", Archive(client), requests.Session)
        if fail:
            with pytest.raises(OSError, match="prefetch failed"):
                daymet.run(*args)
            keys = [item["Key"] for item in client.list_objects_v2(Bucket="hydrologeez-daymet")["Contents"]]
            assert not any(key.endswith("complete.json") for key in keys)
        else:
            daymet.run(*args)
        assert finished.is_set()


def test_resume_preserves_original_namespace_and_appends_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    import requests

    from scripts import daymet
    from scripts.daymet_archive import Archive

    scripts = tmp_path / "scripts"
    scripts.mkdir()
    names = (
        "daymet.py",
        "daymet_acquire.py",
        "daymet_archive.py",
        "daymet_inventory.py",
        "daymet_mask.py",
        "daymet_prepare.py",
    )
    digest = hashlib.sha256()
    for name in names:
        (scripts / name).write_text(name)
        digest.update(name.encode())
        digest.update(name.encode())
    monkeypatch.setattr(daymet, "__file__", str(scripts / "daymet.py"))
    monkeypatch.setattr(daymet, "RESUMABLE_CODE_SHA256", digest.hexdigest())
    monkeypatch.setattr(daymet, "inventory", lambda _: [])
    for name in ("uv.lock", "pyproject.toml", "inventory.json", "boundary.zip"):
        (tmp_path / name).write_text(name)
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="hydrologeez-daymet")
        args = (
            tmp_path / "inventory.json",
            [tmp_path / "inventory.json"],
            tmp_path / "boundary.zip",
            tmp_path / "stage",
            Archive(client),
            requests.Session,
        )
        daymet.run(*args)
        keys = [item["Key"] for item in client.list_objects_v2(Bucket="hydrologeez-daymet")["Contents"]]
        key = next(k for k in keys if k.endswith("/provenance/processing.json"))
        identity = key.split("/")[1]
        original = client.get_object(Bucket="hydrologeez-daymet", Key=key)["Body"].read()
        (scripts / "daymet.py").write_text("acquisition-only revision")
        daymet.run(*args, resume=identity)
        assert client.get_object(Bucket="hydrologeez-daymet", Key=key)["Body"].read() == original
        keys = [item["Key"] for item in client.list_objects_v2(Bucket="hydrologeez-daymet")["Contents"]]
        assert {k.split("/")[1] for k in keys} == {identity}
        assert any("/revisions/" in k and k.endswith("/resume.json") for k in keys)
