"""USGS daily mean series inventory with explicit unknown history and geography."""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import pyproj
import requests
import rivretrieve as rr
import shapefile
import shapely
from shapely.geometry import Point, shape
from shapely.geometry.base import BaseGeometry

from scripts.daymet_mask import BOUNDARY_URL, STATEFP

METADATA_URL = "https://api.waterdata.usgs.gov/ogcapi/v0/collections/time-series-metadata/items"
DAILY_URL = "https://api.waterdata.usgs.gov/ogcapi/v1/collections/daily/items"


def reconcile_inventory(
    features: list[dict[str, Any]],
    descriptions: list[dict[str, Any]],
    stations: list[dict[str, Any]],
    boundary: BaseGeometry,
    boundary_crs: str,
) -> list[dict[str, Any]]:
    """Union live Daily/Mean metadata and retained RR catalogue identities.

    Live geometry uses the collection's CRS84 declaration. Catalogue coordinates
    retain their declared CRS. No state-name or bounding-box fallback is used.
    """
    station_index = {s["station_id"]: s for s in stations if s.get("provider_id") == "usgs_nwis"}
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for description in descriptions:
        if description["product_id"] != "discharge_daily_mean":
            continue
        station = description["station_id"]
        series = description["identity"]["published_id"]
        rows[station, series] = {"station_id": station, "time_series_id": series, "catalogue": description}
    for feature in features:
        properties = feature["properties"]
        location = properties["monitoring_location_id"]
        if not location.startswith("USGS-") or (
            properties.get("parameter_code"),
            properties.get("statistic_id"),
            properties.get("computation_period_identifier"),
            properties.get("computation_identifier"),
        ) != ("00060", "00003", "Daily", "Mean"):
            continue
        station, series = location.removeprefix("USGS-"), str(feature["id"])
        row = rows.setdefault((station, series), {"station_id": station, "time_series_id": series})
        row["live"] = feature
    transforms: dict[str, pyproj.Transformer] = {}
    for (station, _), row in rows.items():
        live = row.get("live", {})
        properties = live.get("properties", {})
        row.update(start=properties.get("begin"), end=properties.get("end"), history_status="unknown")
        if row["start"] is not None and row["end"] is not None:
            try:
                start = datetime.fromisoformat(row["start"]).date()
                end = datetime.fromisoformat(row["end"]).date()
                if end < start:
                    raise ValueError("Reversed metadata bounds")
            except (TypeError, ValueError) as error:
                row.update(history_status="failed", history_error=str(error))
            else:
                row["history_status"] = "metadata_bounds"
        row["catalogue_access"] = "available" if "catalogue" in row else "unavailable"
        site = station_index.get(station)
        row["station"] = site
        geometry = live.get("geometry")
        coordinates = geometry.get("coordinates") if geometry else None
        crs = "OGC:CRS84" if coordinates else None
        coordinate_source = "live_metadata" if coordinates else None
        if coordinates is None and site is not None:
            lon, lat = site.get("longitude"), site.get("latitude")
            if lon is not None and lat is not None and site.get("crs"):
                coordinates, crs = [lon, lat], site["crs"]
                coordinate_source = "catalogue"
        row.update(coordinates=coordinates, coordinate_crs=crs, coordinate_source=coordinate_source)
        if coordinates is None or crs is None:
            row["selection"] = "unknown_coordinates"
            continue
        if crs not in transforms:
            transforms[crs] = pyproj.Transformer.from_crs(crs, boundary_crs, always_xy=True)
        x, y = transforms[crs].transform(*coordinates[:2], errcheck=True)
        row["selection"] = "included" if boundary.covers(Point(x, y)) else "outside"
    return [rows[key] for key in sorted(rows)]


def load_boundary(path: Path) -> tuple[BaseGeometry, str]:
    """Census 2024 1:500k state polygons, all contiguous states plus DC."""
    with zipfile.ZipFile(path) as archive:
        projection = archive.read("cb_2024_us_state_500k.prj").decode()
    geometries: list[BaseGeometry] = []
    states: list[str] = []
    reader = shapefile.Reader(str(path))
    with reader:
        for record in reader.iterShapeRecords():
            assert record.record is not None and record.shape is not None
            state = str(record.record["STATEFP"])
            if state in STATEFP:
                states.append(state)
                geometries.append(shape(record.shape.__geo_interface__))
    if sorted(states) != STATEFP:
        raise ValueError("Boundary must contain all 48 states and DC exactly once")
    return shapely.union_all(geometries), projection


def source_page(session: requests.Session, url: str, directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep exact response bytes and request provenance, including failed answers."""
    response = session.get(url, timeout=120)
    digest = hashlib.sha256(response.content).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest}.json"
    path.write_bytes(response.content)
    evidence = {
        "url": response.url,
        "status": response.status_code,
        "sha256": digest,
        "path": str(path),
        "retrieved_at": datetime.now(UTC).isoformat(),
        "response_headers": {
            name.lower(): value
            for name, value in response.headers.items()
            if name.lower()
            in {
                "retry-after",
                "ratelimit",
                "ratelimit-policy",
                "ratelimit-limit",
                "ratelimit-remaining",
                "ratelimit-reset",
                "x-ratelimit-limit",
                "x-ratelimit-remaining",
                "x-ratelimit-reset",
            }
        },
    }
    (directory / f"{digest}.provenance.json").write_text(json.dumps(evidence, indent=2))
    response.raise_for_status()
    return response.json(), evidence


def history_bounds(session: requests.Session, row: dict[str, Any], directory: Path) -> dict[str, Any]:
    """Probe oldest/newest daily source labels; an empty answer is not an error."""
    labels: list[str | None] = []
    evidence: list[dict[str, Any]] = []
    for order in ("time", "-time"):
        request = requests.Request(
            "GET",
            DAILY_URL,
            params={
                "f": "json",
                "limit": 1,
                "monitoring_location_id": "USGS-" + row["station_id"],
                "time_series_id": row["time_series_id"],
                "sortby": order,
            },
        ).prepare()
        assert request.url is not None
        document, provenance = source_page(session, request.url, directory)
        evidence.append(provenance)
        features = document["features"]
        if not features and any(link.get("rel") == "next" for link in document.get("links", [])):
            raise ValueError("Empty history page has unresolved next link")
        if features:
            properties = features[0]["properties"]
            if (properties["monitoring_location_id"], properties["time_series_id"]) != (
                "USGS-" + row["station_id"],
                row["time_series_id"],
            ):
                raise ValueError("History answer has wrong identity")
            label = properties["time"]
            date.fromisoformat(label)
            labels.append(label)
        else:
            labels.append(None)
    if (labels[0] is None) != (labels[1] is None):
        raise ValueError("Inconsistent earliest/latest history answers")
    if labels[0] is not None and labels[1] is not None and labels[0] > labels[1]:
        raise ValueError("Reversed history bounds")
    return {
        "start": labels[0],
        "end": labels[1],
        "history_status": "source_empty" if labels[0] is None else "daily_bounds",
        "history_evidence": evidence,
    }


def discover_inventory(
    workspace: Path,
    boundary: Path,
    catalogue_series: dict[str, Any],
    stations: list[dict[str, Any]],
    session: requests.Session,
) -> dict[str, Any]:
    """Persist paginated discovery and selection; leave unknown history explicit.

    Resolve included unknown bounds with history_bounds separately so discovery
    does not depend on hundreds of history requests succeeding in one run.
    """
    directory = workspace / "manifests/discovery"
    directory.mkdir(parents=True, exist_ok=True)
    request = requests.Request(
        "GET",
        METADATA_URL,
        params={
            "f": "json",
            "limit": 50000,
            "parameter_code": "00060",
            "statistic_id": "00003",
            "computation_period_identifier": "Daily",
        },
    ).prepare()
    url = request.url
    features: list[dict[str, Any]] = []
    pages: list[dict[str, Any]] = []
    visited: set[str] = set()
    while url is not None:
        if url in visited:
            raise ValueError("Discovery pagination repeated a page")
        visited.add(url)
        document, evidence = source_page(session, url, directory / "source")
        pages.append(evidence)
        features.extend(document["features"])
        links = [link["href"] for link in document.get("links", []) if link.get("rel") == "next"]
        if len(links) > 1:
            raise ValueError("Ambiguous discovery pagination")
        url = urljoin(url, links[0]) if links else None
    geometry, crs = load_boundary(boundary)
    rows = reconcile_inventory(features, catalogue_series["descriptions"], stations, geometry, crs)
    result = {
        "series": rows,
        "pages": pages,
        "boundary": {
            "url": BOUNDARY_URL,
            "sha256": hashlib.sha256(boundary.read_bytes()).hexdigest(),
            "crs": crs,
            "states": STATEFP,
        },
        "limitations": [
            "Metadata bounds describe source coverage, not continuity; gaps require observation audit.",
            "Unknown geography/history and catalogue-only identities require explicit resolution.",
        ],
    }
    (directory / "catalogue-series.json").write_text(json.dumps(catalogue_series))
    (directory / "catalogue-stations.json").write_text(json.dumps(stations))
    (directory / boundary.name).write_bytes(boundary.read_bytes())
    (directory / "inventory.json").write_text(json.dumps(result, indent=2))
    return result


def build_inventory(boundary: Path, workspace: Path, *, max_history_probes: int = 0) -> dict[str, Any]:
    """Build/resume inventory; optionally resolve a bounded number of series pairs.

    A zero probe budget performs metadata discovery only. Each history probe uses
    two HTTP requests. Source errors remain unresolved and stop this invocation;
    callers can respect Retry-After before resuming, rather than retrying blindly.
    """
    if max_history_probes < 0:
        raise ValueError("max_history_probes must be nonnegative")
    path = workspace / "manifests/inventory.json"

    def save(result: dict[str, Any]) -> None:
        stage = path.with_suffix(".tmp")
        stage.write_text(json.dumps(result, indent=2))
        stage.replace(path)

    with requests.Session() as session:
        if path.exists():
            result = json.loads(path.read_text())
        else:
            selection = rr.find(provider="usgs_nwis", quantity="discharge", frequency="daily", statistic="mean")
            descriptions = rr.series(selection).to_dicts()
            for description in descriptions:
                description["identity"] = {"published_id": description["published_id"]}
            result = discover_inventory(
                workspace, boundary, {"descriptions": descriptions}, rr.metadata(selection).to_dicts(), session
            )
            save(result)
        attempted = 0
        for row in result["series"]:
            if attempted >= max_history_probes:
                break
            if row["selection"] != "included" or row["history_status"] not in {"unknown", "failed"}:
                continue
            attempted += 1
            try:
                row.update(history_bounds(session, row, workspace / "manifests/discovery/history"))
                row.pop("history_error", None)
            except (requests.RequestException, ValueError, KeyError) as error:
                row.update(history_status="failed", history_error=str(error))
                save(result)
                break
            save(result)
    unresolved = [
        r
        for r in result["series"]
        if r["selection"] == "unknown_coordinates"
        or (
            r["selection"] == "included"
            and (r["history_status"] in {"unknown", "failed"} or r.get("catalogue_access") == "unavailable")
        )
    ]
    blocked = {r["station_id"] for r in unresolved}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in result["series"]:
        if row["selection"] == "included" and row["station_id"] not in blocked:
            grouped.setdefault(row["station_id"], []).append(row)
    scopes: list[dict[str, Any]] = []
    for station, rows in sorted(grouped.items()):
        bounded = [r for r in rows if r["history_status"] in {"metadata_bounds", "daily_bounds"}]
        if bounded:
            scopes.append(
                {
                    "station": station,
                    "start": min(r["start"][:10] for r in bounded),
                    "end": max(r["end"][:10] for r in bounded),
                    "series": [r["time_series_id"] for r in rows],
                }
            )
    result.update(stations=scopes, unresolved=unresolved)
    save(result)
    return result
