"""Selection retains identities and distinguishes unknown geography/history."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import requests
from shapely.geometry import Polygon

from scripts.usgs_inventory import build_inventory, history_bounds, reconcile_inventory, source_page


def feature(station: str, series: str, coordinates: list[float] | None) -> dict[str, Any]:
    return {
        "id": series,
        "properties": {
            "monitoring_location_id": station,
            "parameter_code": "00060",
            "statistic_id": "00003",
            "computation_period_identifier": "Daily",
            "computation_identifier": "Mean",
            "begin": None,
            "end": None,
        },
        "geometry": None if coordinates is None else {"type": "Point", "coordinates": coordinates},
    }


def test_union_retains_unknowns_and_uses_polygon_not_bbox() -> None:
    boundary = Polygon([(0, 0), (2, 0), (0, 2)])
    live = [
        feature("USGS-01", "a", [0.5, 0.5]),
        feature("USGS-01", "b", None),
        feature("USGS-02", "c", [1.5, 1.5]),
        feature("OTHER-03", "d", [0.5, 0.5]),
    ]
    catalogue = [{"station_id": "04", "product_id": "discharge_daily_mean", "identity": {"published_id": "retained"}}]
    rows = reconcile_inventory(live, catalogue, [], boundary, "OGC:CRS84")
    indexed = {row["time_series_id"]: row for row in rows}
    assert set(indexed) == {"a", "b", "c", "retained"}
    assert indexed["a"]["selection"] == "included"
    assert indexed["c"]["selection"] == "outside"
    assert indexed["b"]["selection"] == "unknown_coordinates"
    assert indexed["retained"]["selection"] == "unknown_coordinates"
    assert indexed["retained"]["history_status"] == "unknown"
    assert indexed["a"]["history_status"] == "unknown"


def test_declared_crs_is_transformed() -> None:
    # Web Mercator x=111319.49 is longitude 1 degree, not 111319 degrees.
    catalogue = [{"station_id": "04", "product_id": "discharge_daily_mean", "identity": {"published_id": "retained"}}]
    stations = [
        {
            "provider_id": "usgs_nwis",
            "station_id": "04",
            "longitude": 111319.49079327357,
            "latitude": 111325.1428663851,
            "crs": "EPSG:3857",
        }
    ]
    rows = reconcile_inventory([], catalogue, stations, Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]), "OGC:CRS84")
    assert rows[0]["selection"] == "included"
    assert rows[0]["coordinate_crs"] == "EPSG:3857"


def test_history_rejects_wrong_identity_and_reversed_dates(tmp_path: Path) -> None:
    row = {"station_id": "01", "time_series_id": "a"}
    with (
        patch(
            "scripts.usgs_inventory.source_page",
            return_value=(
                {
                    "features": [
                        {
                            "properties": {
                                "monitoring_location_id": "USGS-01",
                                "time_series_id": "wrong",
                                "time": "2000-01-01",
                            }
                        }
                    ]
                },
                {},
            ),
        ),
        pytest.raises(ValueError, match="identity"),
    ):
        history_bounds(requests.Session(), row, tmp_path)
    documents = [
        ({"features": [{"properties": {"monitoring_location_id": "USGS-01", "time_series_id": "a", "time": day}}]}, {})
        for day in ("2001-01-01", "2000-01-01")
    ]
    with (
        patch("scripts.usgs_inventory.source_page", side_effect=documents),
        pytest.raises(ValueError, match="Reversed"),
    ):
        history_bounds(requests.Session(), row, tmp_path)


def test_history_empty_is_explicit_and_keeps_evidence(tmp_path: Path) -> None:
    with patch("scripts.usgs_inventory.source_page", return_value=({"features": []}, {"status": 200})):
        result = history_bounds(requests.Session(), {"station_id": "01", "time_series_id": "a"}, tmp_path)
    assert result == {
        "start": None,
        "end": None,
        "history_status": "source_empty",
        "history_evidence": [{"status": 200}, {"status": 200}],
    }


def test_station_scopes_do_not_hide_unresolved_series(tmp_path: Path) -> None:
    rows = [
        {
            "station_id": "01",
            "time_series_id": "a",
            "selection": "included",
            "history_status": "metadata_bounds",
            "start": "1900-01-01T00:00:00.000001",
            "end": "2000-01-01T00:00:00.000001",
        },
        {"station_id": "01", "time_series_id": "b", "selection": "included", "history_status": "unknown"},
        {
            "station_id": "02",
            "time_series_id": "c",
            "selection": "included",
            "history_status": "daily_bounds",
            "start": "1861-01-01",
            "end": "2026-10-31",
        },
    ]
    path = tmp_path / "manifests/inventory.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"series": rows}))
    result = build_inventory(tmp_path / "not-needed-on-resume.zip", tmp_path)
    assert result["stations"] == [{"station": "02", "start": "1861-01-01", "end": "2026-10-31", "series": ["c"]}]
    assert [r["time_series_id"] for r in result["unresolved"]] == ["b"]


def test_failed_source_page_preserves_rate_headers_without_secrets(tmp_path: Path) -> None:
    response = requests.Response()
    response.status_code = 429
    response.url = "https://api.waterdata.usgs.gov/ogcapi/v1/collections/daily/items"
    response._content = b'{"error": "rate limit"}'
    response.headers.update(
        {
            "Retry-After": "60",
            "X-RateLimit-Remaining": "0",
            "RateLimit-Reset": "60",
            "Set-Cookie": "secret",
            "Authorization": "secret",
            "X-Unrelated": "not retained",
        }
    )
    with (
        requests.Session() as session,
        patch.object(session, "get", return_value=response),
        pytest.raises(requests.HTTPError),
    ):
        source_page(session, response.url, tmp_path)
    evidence = json.loads(next(tmp_path.glob("*.provenance.json")).read_text())
    assert evidence["response_headers"] == {"retry-after": "60", "x-ratelimit-remaining": "0", "ratelimit-reset": "60"}
    assert "secret" not in json.dumps(evidence)


@pytest.mark.parametrize("start,end", [("bad", "2000-01-01"), ("2001-01-01", "2000-01-01")])
def test_invalid_metadata_bounds_remain_unresolved(start: str, end: str) -> None:
    live = feature("USGS-01", "a", [0.5, 0.5])
    live["properties"].update(begin=start, end=end)
    rows = reconcile_inventory([live], [], [], Polygon([(0, 0), (2, 0), (0, 2)]), "OGC:CRS84")
    assert rows[0]["history_status"] == "failed"
    assert rows[0]["start"] == start


def test_live_identity_missing_from_catalogue_blocks_plan(tmp_path: Path) -> None:
    live = feature("USGS-01", "a", [0.5, 0.5])
    live["properties"].update(begin="1900-01-01", end="2000-01-01")
    rows = reconcile_inventory([live], [], [], Polygon([(0, 0), (2, 0), (0, 2)]), "OGC:CRS84")
    path = tmp_path / "manifests/inventory.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"series": rows}))
    result = build_inventory(tmp_path / "unused.zip", tmp_path)
    assert result["stations"] == []
    assert result["unresolved"][0]["catalogue_access"] == "unavailable"
