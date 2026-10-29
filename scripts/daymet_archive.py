"""Content-verified immutable objects for the dedicated Daymet archive."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

BLOCK = 8 * 1024 * 1024


def checksum(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Archive:
    bucket = "hydrologeez-daymet"

    def __init__(self, client: S3Client) -> None:
        self.client = client

    def digest(self, key: str) -> tuple[int, str] | None:
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
        except ClientError as error:
            if error.response["Error"]["Code"] == "NoSuchKey":
                return None
            raise
        digest = hashlib.sha256()
        size = 0
        stream = response["Body"]
        try:
            while block := stream.read(BLOCK):
                digest.update(block)
                size += len(block)
        finally:
            stream.close()
        return size, digest.hexdigest()

    def put(self, path: Path, key: str) -> dict[str, str | int]:
        expected = path.stat().st_size, checksum(path)
        existing = self.digest(key)
        if existing is None:
            self.client.upload_file(
                str(path),
                self.bucket,
                key,
                Config=TransferConfig(multipart_chunksize=64 * 1024 * 1024, max_concurrency=2),
            )
            existing = self.digest(key)
        if existing != expected:
            raise ValueError(f"Object integrity mismatch: {key}")
        return {"key": key, "bytes": expected[0], "sha256": expected[1], "verification": "full GET SHA256"}
