from pathlib import Path

import boto3
import pytest
from moto import mock_aws


def test_publishes_matching_native_files_and_restores_bytes(tmp_path: Path) -> None:
    from scripts.usgs_archive import USGSArchive

    files = {
        "cache/usgs_nwis/data.bin": b"native cache",
        "receipts/scope.rrbundle": b"publisher bytes",
        "manifests/inventory.json": b'{"complete": false}',
    }
    source = tmp_path / "source"
    for relative, payload in files.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="hydrologeez-daymet")
        archive = USGSArchive(client)
        published = archive.publish(source, "pilot")
        keys = {entry["Key"] for entry in client.list_objects_v2(Bucket=archive.bucket)["Contents"]}
        assert keys == {
            "streamflow/usgs/cache/pilot/usgs_nwis/data.bin",
            "streamflow/usgs/receipts/pilot/scope.rrbundle",
            "streamflow/usgs/manifests/pilot/inventory.json",
            "streamflow/usgs/manifests/pilot/complete.json",
        }
        restored = tmp_path / "restored"
        assert archive.restore("pilot", restored) == published
        for relative, payload in files.items():
            assert (restored / relative).read_bytes() == payload


def test_corruption_blocks_completion_and_restore(tmp_path: Path) -> None:
    import pytest

    from scripts.usgs_archive import USGSArchive

    source = tmp_path / "source"
    for directory in ("cache", "receipts", "manifests"):
        (source / directory).mkdir(parents=True)
    (source / "cache/data").write_bytes(b"native")
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="hydrologeez-daymet")
        archive = USGSArchive(client)
        key = "streamflow/usgs/cache/pilot/data"
        client.put_object(Bucket=archive.bucket, Key=key, Body=b"corrupt")
        with pytest.raises(ValueError, match="integrity"):
            archive.publish(source, "pilot")
        keys = {entry["Key"] for entry in client.list_objects_v2(Bucket=archive.bucket)["Contents"]}
        assert "streamflow/usgs/manifests/pilot/complete.json" not in keys
        client.delete_object(Bucket=archive.bucket, Key=key)
        archive.publish(source, "pilot")
        client.put_object(Bucket=archive.bucket, Key=key, Body=b"corrupt")
        with pytest.raises(ValueError, match="integrity"):
            archive.restore("pilot", tmp_path / "restore")
        with pytest.raises(ValueError, match="empty"):
            archive.restore("pilot", source)


def test_workspace_lock_excludes_acquisition_and_validation_failure(tmp_path: Path) -> None:
    import pytest

    from scripts.usgs_archive import USGSArchive, workspace_lock

    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="hydrologeez-daymet")
        archive = USGSArchive(client)
        with workspace_lock(tmp_path), pytest.raises(RuntimeError, match="busy"):
            archive.publish(tmp_path, "pilot")

        def invalid(workspace: Path) -> None:
            assert workspace == tmp_path
            raise ValueError("Missing publisher bundle")

        with pytest.raises(ValueError, match="publisher bundle"):
            archive.publish(tmp_path, "pilot", validate=invalid)
        assert client.list_objects_v2(Bucket=archive.bucket)["KeyCount"] == 0


def test_completed_generation_is_immutable(tmp_path: Path) -> None:
    import pytest

    from scripts.usgs_archive import USGSArchive

    for directory in ("cache", "receipts", "manifests"):
        (tmp_path / directory).mkdir()
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="hydrologeez-daymet")
        archive = USGSArchive(client)
        archive.publish(tmp_path, "pilot")
        marker_key = "streamflow/usgs/manifests/pilot/complete.json"
        original = client.get_object(Bucket=archive.bucket, Key=marker_key)["Body"].read()
        (tmp_path / "cache/new-data").write_bytes(b"new")
        with pytest.raises(ValueError, match="already complete"):
            archive.publish(tmp_path, "pilot")
        assert client.list_objects_v2(Bucket=archive.bucket)["KeyCount"] == 1
        assert client.get_object(Bucket=archive.bucket, Key=marker_key)["Body"].read() == original


def test_rejects_symlinked_cache_root(tmp_path: Path) -> None:
    import pytest

    from scripts.usgs_archive import USGSArchive

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (workspace / "cache").symlink_to(external, target_is_directory=True)
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="hydrologeez-daymet")
        with pytest.raises(ValueError, match="Symlink"):
            USGSArchive(client).publish(workspace, "pilot")


def test_cli_restore_verifies_native_cache_and_bundles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from unittest.mock import patch

    from scripts.usgs import main
    from scripts.usgs_acquire import validate_workspace

    monkeypatch.setattr("sys.argv", ["usgs", "restore-files", str(tmp_path), "pilot"])
    with patch("scripts.usgs.boto3.Session"), patch("scripts.usgs.USGSArchive") as archive:
        archive.return_value.restore.return_value = {"generation": "pilot"}
        main()
    archive.return_value.restore.assert_called_once_with("pilot", tmp_path, validate=validate_workspace)
    assert '"native_local_cache_usable": true' in capsys.readouterr().out
