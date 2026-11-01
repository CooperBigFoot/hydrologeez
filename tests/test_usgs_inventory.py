"""Selection retains identities and distinguishes unknown geography/history."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import requests
from shapely.geometry import Polygon

from scripts.usgs_acquire import plan_scopes
from scripts.usgs_inventory import apply_locations, build_inventory, history_bounds, reconcile_inventory, source_page


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
    path.write_text(json.dumps({"series": rows, "location_refinement": {"status": "complete"}}))
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
    path.write_text(json.dumps({"series": rows, "location_refinement": {"status": "complete"}}))
    result = build_inventory(tmp_path / "unused.zip", tmp_path)
    assert result["stations"] == []
    assert result["unresolved"][0]["catalogue_access"] == "unavailable"


def test_source_country_excludes_polygon_conflict_and_keeps_us_unknown() -> None:
    rows = [
        {"station_id": "01", "selection": "included"},
        {"station_id": "02", "selection": "unknown_coordinates"},
        {"station_id": "03", "selection": "unknown_coordinates"},
    ]
    locations = [
        {"id": "USGS-" + station, "properties": {"agency_code": "USGS", "country_code": country}, "geometry": None}
        for station, country in [("01", "CA"), ("02", "JP"), ("03", "US")]
    ]
    apply_locations(rows, locations, Polygon([(0, 0), (2, 0), (0, 2)]), "OGC:CRS84")
    assert rows[0]["selection"] == "outside_country"
    assert rows[0]["polygon_selection"] == "included"
    assert rows[0]["country_polygon_conflict"] is True
    assert rows[1]["selection"] == "outside_country"
    assert rows[2]["selection"] == "unknown_coordinates"
    assert rows[0]["monitoring_location"] == locations[0]


def test_location_refinement_refused_after_planning(tmp_path: Path) -> None:
    path = tmp_path / "manifests/plan.json"
    path.parent.mkdir()
    path.write_text("{}")
    with pytest.raises(ValueError, match="planned"):
        build_inventory(tmp_path / "unused.zip", tmp_path, refine_locations=True)


def test_refinement_preserves_history_and_saves_source_evidence(tmp_path: Path) -> None:
    row = {
        "station_id": "01",
        "time_series_id": "a",
        "selection": "included",
        "history_status": "daily_bounds",
        "start": "1900-01-01",
        "end": "2000-01-01",
    }
    path = tmp_path / "manifests/inventory.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"series": [row]}))
    location = {"id": "USGS-01", "properties": {"agency_code": "USGS", "country_code": "CA"}, "geometry": None}
    with (
        patch(
            "scripts.usgs_inventory.source_page",
            return_value=({"features": [location]}, {"sha256": "evidence", "url": "source"}),
        ),
        patch("scripts.usgs_inventory.load_boundary", return_value=(Polygon([(0, 0), (2, 0), (0, 2)]), "OGC:CRS84")),
    ):
        result = build_inventory(tmp_path / "boundary.zip", tmp_path, refine_locations=True)
    assert result["series"][0]["history_status"] == "daily_bounds"
    assert result["stations"] == []
    assert len(result["location_refinement"]["pages"]) == 1
    assert result["location_refinement"]["status"] == "complete"
    assert json.loads(path.read_text()) == result


def test_location_failure_checkpoints_then_resumes_without_repeating_pages(tmp_path: Path) -> None:
    path = tmp_path / "manifests/inventory.json"
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {
                "series": [
                    {
                        "station_id": "01",
                        "time_series_id": "a",
                        "selection": "included",
                        "history_status": "daily_bounds",
                        "start": "1900-01-01",
                        "end": "2000-01-01",
                    }
                ]
            }
        )
    )
    with (
        patch(
            "scripts.usgs_inventory.source_page",
            side_effect=[
                ({"features": [], "links": [{"rel": "next", "href": "https://example.test/next"}]}, {"url": "first"}),
                requests.HTTPError("temporary source error"),
                ({"features": []}, {"url": "https://example.test/next"}),
            ],
        ) as fetch,
        patch("scripts.usgs_inventory.load_boundary", return_value=(Polygon([(0, 0), (2, 0), (0, 2)]), "OGC:CRS84")),
    ):
        first = build_inventory(tmp_path / "boundary.zip", tmp_path, refine_locations=True)
        assert len(first["unresolved"]) == 1
        assert first["location_refinement"]["status"] == "pending"
        assert first["stations"] == []
        second = build_inventory(tmp_path / "boundary.zip", tmp_path, refine_locations=True)
        assert fetch.call_count == 3
        assert second["location_refinement"]["status"] == "complete"
        assert len(second["stations"]) == 1
        assert "country_code=CA" in fetch.call_args_list[0].args[1]
        assert fetch.call_args_list[1].args[1] == "https://example.test/next"


def test_location_failure_stops_history_requests(tmp_path: Path) -> None:
    path = tmp_path / "manifests/inventory.json"
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {
                "series": [
                    {"station_id": "01", "time_series_id": "a", "selection": "included", "history_status": "unknown"}
                ]
            }
        )
    )
    with (
        patch("scripts.usgs_inventory.source_page", side_effect=requests.HTTPError("429")),
        patch("scripts.usgs_inventory.load_boundary", return_value=(Polygon([(0, 0), (2, 0), (0, 2)]), "OGC:CRS84")),
        patch("scripts.usgs_inventory.history_bounds") as history,
    ):
        result = build_inventory(tmp_path / "boundary.zip", tmp_path, refine_locations=True, max_history_probes=10)
    history.assert_not_called()
    assert result["location_refinement"]["status"] == "pending"


def test_approved_oregon_exception_preserves_missing_geometry_on_resume(tmp_path: Path) -> None:
    # USGS monitoring-location classifications; neither station has coordinates.
    rows = reconcile_inventory(
        [feature("USGS-11516100", "oregon", None), feature("USGS-11341370", "california", None)],
        [],
        [],
        Polygon([(0, 0), (2, 0), (0, 2)]),
        "OGC:CRS84",
    )
    for row in rows:
        row.update(catalogue_access="available", history_status="source_empty")
        if row["station_id"] == "11516100":
            row.update(history_status="daily_bounds", start="1960-10-01", end="1965-09-30")
    locations = [
        {
            "id": "USGS-" + station,
            "properties": {"agency_code": "USGS", "country_code": "US", "state_code": state},
            "geometry": None,
        }
        for station, state in [("11516100", "41"), ("11341370", "06")]
    ]
    path = tmp_path / "manifests/inventory.json"
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {
                "series": rows,
                "location_refinement": {
                    "status": "complete",
                    "pending": [],
                    "pages": [{"url": "source", "sha256": "retained"}],
                    "locations": locations,
                },
            }
        )
    )
    with (
        patch("scripts.usgs_inventory.source_page") as fetch,
        patch("scripts.usgs_inventory.load_boundary", return_value=(Polygon([(0, 0), (2, 0), (0, 2)]), "OGC:CRS84")),
    ):
        result = build_inventory(tmp_path / "boundary.zip", tmp_path, refine_locations=True)
    fetch.assert_not_called()
    assert result["stations"] == [
        {"station": "11516100", "start": "1960-10-01", "end": "1965-09-30", "series": ["oregon"]}
    ]
    selected = next(row for row in result["series"] if row["station_id"] == "11516100")
    assert selected["coordinates"] is None
    assert selected["coordinate_crs"] is None
    assert selected["coordinate_source"] is None
    assert selected["polygon_selection"] == "unknown_coordinates"
    assert selected["monitoring_location"] == locations[0]
    assert selected["selection_exception"]
    assert result["location_refinement"]["pages"] == [{"url": "source", "sha256": "retained"}]
    assert result["unresolved"] == []
    assert result["availability_limitations"][0]["station_id"] == "11341370"
    assert result["availability_limitations"][0]["selection"] == "unknown_coordinates"
    assert json.loads(path.read_text()) == result


@pytest.mark.parametrize(
    "country,state,selection",
    [
        ("US", "06", "unknown_coordinates"),
        (None, "41", "unknown_coordinates"),
        ("CA", "41", "unknown_coordinates"),
        ("US", "41", "outside"),
    ],
)
def test_oregon_exception_requires_authoritative_classification_and_missing_coordinates(
    country: str | None,
    state: str,
    selection: str,
) -> None:
    rows = [{"station_id": "11516100", "selection": selection}]
    locations = [
        {
            "id": "USGS-11516100",
            "properties": {"agency_code": "USGS", "country_code": country, "state_code": state},
            "geometry": None,
        }
    ]
    apply_locations(rows, locations, Polygon([(0, 0), (2, 0), (0, 2)]), "OGC:CRS84")
    assert rows[0]["selection"] != "included"
    assert "selection_exception" not in rows[0]


@pytest.mark.parametrize("answer", ["empty", "nonempty", "failure"])
def test_unknown_geography_probes_history_without_claiming_location(tmp_path: Path, answer: str) -> None:
    row = {
        "station_id": "11341370",
        "time_series_id": "a",
        "selection": "unknown_coordinates",
        "history_status": "unknown",
    }
    path = tmp_path / "manifests/inventory.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"series": [row], "location_refinement": {"status": "complete"}}))
    documents = [
        (
            {
                "features": []
                if answer == "empty"
                else [
                    {
                        "properties": {
                            "monitoring_location_id": "USGS-11341370",
                            "time_series_id": "a",
                            "time": day,
                        }
                    }
                ]
            },
            {"status": 200, "url": order},
        )
        for day, order in [("1960-10-01", "earliest"), ("1965-09-30", "latest")]
    ]
    with patch(
        "scripts.usgs_inventory.source_page",
        side_effect=requests.HTTPError("429") if answer == "failure" else documents,
    ) as fetch:
        result = build_inventory(tmp_path / "unused.zip", tmp_path, max_history_probes=1)
    assert fetch.call_count == (1 if answer == "failure" else 2)
    assert result["series"][0]["selection"] == "unknown_coordinates"
    assert result["stations"] == []
    if answer == "empty":
        assert result["unresolved"] == []
        assert result["availability_limitations"] == result["series"]
        assert plan_scopes(result, max_days=365) == []
        assert result["series"][0]["history_status"] == "source_empty"
        assert result["series"][0]["history_evidence"] == [
            {"status": 200, "url": "earliest"},
            {"status": 200, "url": "latest"},
        ]
    else:
        assert result["unresolved"] == result["series"]
        assert result["availability_limitations"] == []
        with pytest.raises(ValueError, match="unresolved"):
            plan_scopes(result, max_days=365)
        assert result["series"][0]["history_status"] == ("failed" if answer == "failure" else "daily_bounds")
    assert json.loads(path.read_text()) == result
