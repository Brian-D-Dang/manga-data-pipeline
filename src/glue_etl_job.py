"""AWS Glue Python Shell ETL: MaNGA LOGCUBE FITS -> mask FITS.

Job type in the Glue console must be **Python shell** (not Spark).
FITS cubes are binary multi-extension images; DynamicFrames cannot read them.

Script editor setup
-------------------
- Script path: this file (paste into the editor, or point at this object in S3)
- Python library path / extra-py-files:
    s3://manga-data-038454221846-us-east-1-an/scripts/manga_mask_pipeline.py
- Job parameters:
    --additional-python-modules  astropy==6.1.7,numpy==1.26.4
    --RAW_S3                     s3://manga-data-038454221846-us-east-1-an/raw/
    --PROCESSED_S3               s3://manga-data-038454221846-us-east-1-an/processed/

Extract: list new ``.fits`` objects under raw/
Transform: ``manga_mask_pipeline.run_pipeline``
Load: upload ``*-MASK.fits`` under processed/

A cube is treated as **new** when the destination MASK is missing, or the
raw object is newer than the existing MASK (re-uploaded cube).
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError

from awsglue.utils import getResolvedOptions

from manga_mask_pipeline import DEFAULT_LOGU_LIMIT, DEFAULT_SOLAR_OH, run_pipeline

log = logging.getLogger("glue_etl")

DEFAULT_RAW = "s3://manga-data-038454221846-us-east-1-an/raw/"
DEFAULT_PROCESSED = "s3://manga-data-038454221846-us-east-1-an/processed/"


def _glue_args() -> dict:
    """JOB_NAME is always injected by Glue; the rest are optional job parameters."""
    args = getResolvedOptions(sys.argv, ["JOB_NAME"])
    for key in ("RAW_S3", "PROCESSED_S3", "SOLAR_OH", "LOGU_LIMIT"):
        if f"--{key}" in sys.argv:
            args.update(getResolvedOptions(sys.argv, [key]))
    return args


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"expected s3://bucket/prefix, got {uri!r}")
    prefix = parsed.path.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return parsed.netloc, prefix


def mask_key_for(raw_key: str, dest_prefix: str) -> str:
    name = Path(raw_key).name
    lower = name.lower()
    if lower.endswith("-logcube.fits"):
        stem = name[: len(name) - len("-logcube.fits")]
    elif lower.endswith(".fits"):
        stem = name[: len(name) - len(".fits")]
    else:
        stem = name
    return f"{dest_prefix}{stem}-MASK.fits"


def list_fits(s3, bucket: str, prefix: str) -> list[dict]:
    objects = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/") or obj.get("Size", 0) == 0:
                continue
            if not key.lower().endswith(".fits"):
                continue
            objects.append(obj)
    return objects


def head_or_none(s3, bucket: str, key: str):
    try:
        return s3.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return None
        raise


def is_new(raw_obj: dict, dest_head) -> bool:
    if dest_head is None:
        return True
    return raw_obj["LastModified"] > dest_head["LastModified"]


def process_one(s3, src_bucket: str, raw_key: str, dest_bucket: str, dest_key: str,
                solar_oh: float, logu_limit: float) -> None:
    with tempfile.TemporaryDirectory() as td:
        local_in = Path(td) / Path(raw_key).name
        local_out = Path(td) / Path(dest_key).name
        log.info("download s3://%s/%s", src_bucket, raw_key)
        s3.download_file(src_bucket, raw_key, str(local_in))
        run_pipeline(local_in, local_out, solar_oh=solar_oh, logu_limit=logu_limit)
        log.info("upload s3://%s/%s", dest_bucket, dest_key)
        s3.upload_file(str(local_out), dest_bucket, dest_key)


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=logging.INFO,
    )
    args = _glue_args()
    raw_uri = args.get("RAW_S3", DEFAULT_RAW)
    dest_uri = args.get("PROCESSED_S3", DEFAULT_PROCESSED)
    solar_oh = float(args.get("SOLAR_OH", DEFAULT_SOLAR_OH))
    logu_limit = float(args.get("LOGU_LIMIT", DEFAULT_LOGU_LIMIT))

    raw_bucket, raw_prefix = parse_s3_uri(raw_uri)
    dest_bucket, dest_prefix = parse_s3_uri(dest_uri)

    s3 = boto3.client("s3")
    cubes = list_fits(s3, raw_bucket, raw_prefix)
    log.info("found %d FITS object(s) under s3://%s/%s", len(cubes), raw_bucket, raw_prefix)

    n_new = n_skip = n_fail = 0
    failures: list[str] = []
    for obj in cubes:
        raw_key = obj["Key"]
        dest_key = mask_key_for(raw_key, dest_prefix)
        dest_head = head_or_none(s3, dest_bucket, dest_key)
        if not is_new(obj, dest_head):
            log.info("skip (already processed): %s", raw_key)
            n_skip += 1
            continue
        try:
            process_one(
                s3, raw_bucket, raw_key, dest_bucket, dest_key, solar_oh, logu_limit,
            )
            n_new += 1
        except Exception:
            n_fail += 1
            failures.append(raw_key)
            log.exception("failed: %s", raw_key)

    log.info("done job=%s processed=%d skipped=%d failed=%d",
             args.get("JOB_NAME"), n_new, n_skip, n_fail)
    if n_new == 0 and n_fail == 0 and not cubes:
        raise SystemExit("no FITS files under raw/")
    if n_fail and n_new == 0:
        raise SystemExit("all new files failed: " + ", ".join(failures))
    if n_fail:
        raise SystemExit(f"{n_fail} file(s) failed: " + ", ".join(failures))


if __name__ == "__main__":
    main()
