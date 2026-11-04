from pathlib import Path

import pytest
import requests
from shapely.geometry import box


def test_inventory_selects_conus_cells_and_reports_missing_files() -> None:
    from scripts.dem_inventory import Source, select

    sources = [Source("n39w105", "n39w105/USGS_1_n39w105.tif", 12, '"abc"', "2026-01-01")]
    selected, missing = select(sources, box(-104.9, 38.2, -103.8, 38.9))
    assert selected == sources
    assert missing == {"n39w104": ["tif", "xml"], "n39w105": ["xml"]}


def test_download_pins_identity_and_replaces_interrupted_stage(tmp_path: Path) -> None:
    import hashlib

    import responses

    from scripts.dem_acquire import download
    from scripts.dem_inventory import Source

    data = b"native unchanged provider bytes"
    etag = '"' + hashlib.md5(data).hexdigest() + '"'
    source = Source("n39w105", "n39w105/USGS_1_n39w105.tif", len(data), etag, "date")
    path = tmp_path / "stage"
    path.write_bytes(b"partial")
    with responses.RequestsMock() as server:
        server.get(source.url, body=data, headers={"ETag": etag})
        download(requests.Session(), source, path)
        assert server.calls[0].request.headers["If-Match"] == etag
    assert path.read_bytes() == data
    with responses.RequestsMock() as server:
        server.get(source.url, body=b"changed", headers={"ETag": etag})
        with pytest.raises(ValueError, match="integrity"):
            download(requests.Session(), source, path)


def test_transfer_restart_reconciles_content_and_reports_failures(tmp_path: Path) -> None:
    import hashlib
    import json
    from dataclasses import asdict

    import boto3
    import responses
    from moto import mock_aws

    from scripts.daymet_archive import Archive
    from scripts.dem_acquire import run
    from scripts.dem_inventory import BASE, CURRENT, PREFIX, Source

    payload = b"unchanged raster fixture"
    source = Source(
        "n39w105",
        "n39w105/USGS_1_n39w105.tif",
        len(payload),
        '"' + hashlib.md5(payload).hexdigest() + '"',
        "2026-01-01",
    )
    snapshot = tmp_path / "inventory.json"
    snapshot.write_text(json.dumps({"source_prefix": BASE + CURRENT, "missing": {}, "files": [asdict(source)]}))
    with mock_aws(), responses.RequestsMock() as server:
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="hydrologeez-daymet")
        archive = Archive(client)
        server.get(source.url, body=payload, headers={"ETag": source.etag})
        first = run(snapshot, tmp_path / "stage", archive, requests.Session())
        assert first["complete"]
        assert first["objects"][0]["sha256"] == hashlib.sha256(payload).hexdigest()
        assert run(snapshot, tmp_path / "stage", archive, requests.Session())["complete"]
        assert len(server.calls) == 1
        assert not (tmp_path / "stage/provider.part").exists()
        client.put_object(Bucket=archive.bucket, Key=f"{PREFIX}/native/{source.name}", Body=b"corrupt")
        failed = run(snapshot, tmp_path / "stage", archive, requests.Session())
        assert not failed["complete"]
        assert failed["failed"][0]["name"] == source.name


def test_metadata_retains_dated_release_and_rejects_wrong_datum(tmp_path: Path) -> None:
    from scripts.dem_acquire import metadata

    path = tmp_path / "provider.xml"
    body = """<metadata><title>USGS 1 Arc Second n39w105 20260528</title><pubdate>20260528</pubdate>
    <networkr>https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1/TIFF/historical/n39w105/USGS_1_n39w105_20260528.tif</networkr>
    <horizdn>North American Datum of 1983</horizdn><altdatum>North American Vertical Datum of 1988</altdatum>
    <altunits>meters</altunits><latres>-0.00027777777778</latres><longres>0.00027777777778</longres></metadata>"""
    path.write_text(body)
    assert metadata(path)["pubdate"] == "20260528"
    assert "/historical/" in metadata(path)["networkr"]
    path.write_text(body.replace("1988", "1929"))
    with pytest.raises(ValueError, match="reference system"):
        metadata(path)


def test_restart_after_upload_without_receipt_and_missing_source(tmp_path: Path) -> None:
    import hashlib
    import json
    from dataclasses import asdict

    import boto3
    import responses
    from moto import mock_aws

    from scripts.daymet_archive import Archive
    from scripts.dem_acquire import run
    from scripts.dem_inventory import BASE, CURRENT, PREFIX, Source

    payload = b"previously uploaded"
    source = Source(
        "n39w105",
        "n39w105/USGS_1_n39w105.tif",
        len(payload),
        '"' + hashlib.md5(payload).hexdigest() + '"',
        "2026-01-01",
    )
    missing = Source("n39w104", "n39w104/USGS_1_n39w104.tif", 10, '"' + "a" * 32 + '"', "date")
    snapshot = tmp_path / "inventory.json"
    snapshot.write_text(
        json.dumps(
            {
                "source_prefix": BASE + CURRENT,
                "missing": {"n39w104": ["xml"]},
                "files": [asdict(source), asdict(missing)],
            }
        )
    )
    with mock_aws(), responses.RequestsMock() as server:
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="hydrologeez-daymet")
        client.put_object(Bucket="hydrologeez-daymet", Key=f"{PREFIX}/native/{source.name}", Body=payload)
        server.get(source.url, body=payload, headers={"ETag": source.etag})
        server.get(missing.url, status=404)
        report = run(snapshot, tmp_path / "stage", Archive(client), requests.Session())
        assert not report["complete"]
        assert len(report["objects"]) == 1
        assert report["failed"][0]["name"] == missing.name
        assert report["missing_cells"] == {"n39w104": ["xml"]}
        assert len(server.calls) == 4  # successful file once, missing file bounded to three attempts


def test_selection_excludes_wrong_resolution_files_in_current_directory() -> None:
    from scripts.dem_inventory import Source, select

    native = Source("n30w090", "n30w090/USGS_1_n30w090.tif", 1, '"abc"', "date")
    wrong = Source("n30w090", "n30w090/USGS_13_n30w090.tif", 1, '"abc"', "date")
    aux = Source("n30w090", "n30w090/USGS_1_n30w090.tif.aux.xml", 1, '"abc"', "date")
    selected, missing = select([native, wrong, aux], box(-89.9, 29.1, -89.1, 29.9))
    assert selected == [native, aux]
    assert missing == {"n30w090": ["xml"]}


@pytest.mark.parametrize("failure", ["xml", "upload"])
def test_expected_file_failures_leave_reconciliation_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    import hashlib
    import json
    from dataclasses import asdict

    import boto3
    import responses
    from boto3.exceptions import S3UploadFailedError
    from moto import mock_aws

    from scripts.daymet_archive import Archive
    from scripts.dem_acquire import run
    from scripts.dem_inventory import BASE, CURRENT, Source

    data = b"not valid XML"
    source = Source(
        "n39w105", "n39w105/USGS_1_n39w105.xml", len(data), '"' + hashlib.md5(data).hexdigest() + '"', "date"
    )
    if failure == "upload":
        source = Source(source.tile, source.name.replace(".xml", ".tif"), source.size, source.etag, source.modified)
    snapshot = tmp_path / "inventory.json"
    snapshot.write_text(json.dumps({"source_prefix": BASE + CURRENT, "missing": {}, "files": [asdict(source)]}))
    with mock_aws(), responses.RequestsMock() as server:
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="hydrologeez-daymet")
        original = client.upload_file

        def upload(Filename: str, Bucket: str, Key: str, **kwargs: object) -> None:
            if "/native/" in Key:
                raise S3UploadFailedError("simulated provider outage")
            original(Filename, Bucket, Key)

        if failure == "upload":
            monkeypatch.setattr(client, "upload_file", upload)
        server.get(source.url, body=data, headers={"ETag": source.etag})
        report = run(snapshot, tmp_path / "stage", Archive(client), requests.Session())
        assert not report["complete"]
        assert report["failed"][0]["error"] == ("ParseError" if failure == "xml" else "S3UploadFailedError")
        assert json.loads((tmp_path / "stage/report.json").read_text()) == report


def test_release_link_checks_frozen_tiff_and_reports_legacy() -> None:
    import responses

    from scripts.dem_acquire import release_identity
    from scripts.dem_inventory import Source

    source = Source("n39w105", "n39w105/USGS_1_n39w105.tif", 123, '"' + "a" * 32 + '"', "date")
    url = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1/TIFF/historical/n39w105/USGS_1_n39w105_20260528.tif"
    fields = {"networkr": url, "pubdate": "20260528", "title": "USGS 1 Arc Second n39w105 20260528"}
    with responses.RequestsMock() as server:
        server.head(url, headers={"ETag": source.etag, "Content-Length": "123"})
        assert release_identity(requests.Session(), fields, source)["status"] == "matched"
    with responses.RequestsMock() as server:
        server.head(url, headers={"ETag": '"' + "b" * 32 + '"', "Content-Length": "123"})
        with pytest.raises(ValueError, match="release"):
            release_identity(requests.Session(), fields, source)
    assert (
        release_identity(
            requests.Session(),
            {"networkr": "ftp://legacy/tile.tif", "pubdate": "20191212", "title": "USGS 1 Arc Second n39w105 20191212"},
            source,
        )["status"]
        == "unverifiable"
    )


def test_old_receipt_cannot_bypass_release_check(tmp_path: Path) -> None:
    import hashlib
    import json
    from dataclasses import asdict

    import boto3
    import responses
    from moto import mock_aws

    from scripts.daymet_archive import Archive, checksum
    from scripts.dem_acquire import run
    from scripts.dem_inventory import BASE, CURRENT, PREFIX, Source

    data = b"malformed metadata must not be skipped"
    source = Source(
        "n39w105", "n39w105/USGS_1_n39w105.xml", len(data), '"' + hashlib.md5(data).hexdigest() + '"', "date"
    )
    snapshot = tmp_path / "inventory.json"
    snapshot.write_text(json.dumps({"source_prefix": BASE + CURRENT, "missing": {}, "files": [asdict(source)]}))
    key = f"{PREFIX}/native/{source.name}"
    old_key = f"{PREFIX}/inventories/{checksum(snapshot)}/receipts/{source.name}.json"
    old = json.dumps(
        {"key": key, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "etag": source.etag}
    ).encode()
    with mock_aws(), responses.RequestsMock() as server:
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="hydrologeez-daymet")
        client.put_object(Bucket="hydrologeez-daymet", Key=key, Body=data)
        client.put_object(Bucket="hydrologeez-daymet", Key=old_key, Body=old)
        server.get(source.url, body=data, headers={"ETag": source.etag})
        report = run(snapshot, tmp_path / "stage", Archive(client), requests.Session())
        assert not report["complete"]
        assert report["failed"][0]["error"] == "ParseError"
        assert client.get_object(Bucket="hydrologeez-daymet", Key=old_key)["Body"].read() == old
