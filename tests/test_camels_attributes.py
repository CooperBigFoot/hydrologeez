"""Static attributes stay labeled and independent of daily records."""

from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from hydrologeez.camels import prepare_attributes, read_attributes


def attribute_sources(raw: Path) -> None:
    raw.mkdir()
    columns = {
        "name": ("huc_02;gauge_name", "01;A River", "02;B River"),
        "topo": ("gauge_lat", "47.23739", "40"),
        "clim": ("p_mean;high_prec_timing", "3.12667898699521;son", "2;mam"),
        "hydro": ("q5", "NA", "0"),
        "soil": ("soil_depth_statsgo", "1.24840792462702", "0"),
        "vege": ("dom_land_cover;root_depth_50", "    Mixed Forests;NA", "Grasslands;0.3"),
        "geol": ("geol_2nd_class;geol_porostiy", "NA;0.1714", "Water bodies;NA"),
    }
    for name, (header, first, second) in columns.items():
        (raw / f"camels_{name}.txt").write_text(f"gauge_id;{header}\n01013500;{first}\n00000002;{second}\n")


def test_attributes_preserve_labels_values_metadata_and_resident_selection(tmp_path: Path) -> None:
    raw, output = tmp_path / "raw", tmp_path / "attributes"
    attribute_sources(raw)
    prepare_attributes(raw, output)
    # No raw inputs or daily store are available to the reader.
    for path in raw.iterdir():
        path.unlink()
    attributes = read_attributes(output, ["00000002", "01013500"])
    table = attributes.table
    assert table["gauge_id"].to_list() == ["00000002", "01013500"]
    assert table.schema["gauge_id"] == pl.String
    assert table.schema["q5"] == pl.Float64
    assert table.schema["geol_2nd_class"] == pl.String
    first = table.row(1, named=True)
    assert first["huc_02"] == "01"
    assert first["gauge_name"] == "A River"
    assert first["p_mean"] == 3.12667898699521
    assert first["dom_land_cover"] == "    Mixed Forests"
    assert first["q5"] is None
    assert first["geol_2nd_class"] is None
    assert table["q5"][0] == 0
    fields = attributes.metadata["fields"]
    assert fields["p_mean"]["unit"] == "mm/day"
    assert fields["soil_depth_statsgo"]["documented_name"] == "soil_depth_statgso"
    assert fields["geol_porostiy"]["source_file"] == "camels_geol.txt"
    assert fields["huc_02"]["dtype"] == "string"
    for path in output.iterdir():
        path.unlink()
    assert table.filter(pl.col("gauge_id") == "00000002")["gauge_name"].item() == "B River"


@pytest.mark.parametrize("ids", [[], ["01013500", "01013500"], ["unknown"]])
def test_attribute_selection_rejects_invalid_ids(tmp_path: Path, ids: list[str]) -> None:
    raw, output = tmp_path / "raw", tmp_path / "attributes"
    attribute_sources(raw)
    prepare_attributes(raw, output)
    with pytest.raises(ValueError):
        read_attributes(output, ids)


@pytest.mark.parametrize("rows", ["01013500;2\n01013500;3\n", "99999999;2\n00000002;3\n"])
def test_attributes_reject_disagreeing_or_duplicate_identities(tmp_path: Path, rows: str) -> None:
    raw = tmp_path / "raw"
    attribute_sources(raw)
    (raw / "camels_clim.txt").write_text("gauge_id;p_mean\n" + rows)
    with pytest.raises(ValueError, match="unique gauge identities"):
        prepare_attributes(raw, tmp_path / "attributes")


def test_attribute_rows_align_with_daily_selection(tmp_path: Path) -> None:
    from test_camels import source_files

    from hydrologeez.camels import Camels, prepare

    raw, areas = source_files(tmp_path)
    daily = tmp_path / "daily"
    prepare(raw, daily, areas)
    selected = Camels(daily).select(["00000002", "00000001"], "2000-02-28", "2000-03-01")
    static_raw = tmp_path / "static_raw"
    attribute_sources(static_raw)
    for path in static_raw.iterdir():
        path.write_text(path.read_text().replace("01013500", "00000001"))
    output = tmp_path / "attributes"
    prepare_attributes(static_raw, output)
    table = read_attributes(output, selected.ids).table
    assert tuple(table["gauge_id"]) == selected.ids
    assert table["gauge_name"].to_list() == ["B River", "A River"]
    resident = read_attributes(output).table
    for path in output.iterdir():
        path.unlink()
    aligned = pl.DataFrame({"gauge_id": selected.ids}).join(resident, on="gauge_id", how="left", maintain_order="left")
    assert_frame_equal(aligned, table)


def test_local_attributes_match_all_source_identities_fields_and_values(tmp_path: Path) -> None:
    import csv

    raw = Path(__file__).resolve().parents[1] / "data/camels-us/raw"
    if not (raw / "camels_topo.txt").exists():
        pytest.skip("Local CAMELS-US attributes are not available")
    output = tmp_path / "attributes"
    prepare_attributes(raw, output)
    attributes = read_attributes(output)
    table = attributes.table
    assert table.shape == (671, 60)
    by_id = {row["gauge_id"]: row for row in table.iter_rows(named=True)}
    strings = {
        "huc_02",
        "gauge_name",
        "high_prec_timing",
        "low_prec_timing",
        "dom_land_cover",
        "geol_1st_class",
        "geol_2nd_class",
    }
    fields: set[str] = set()
    for path in raw.glob("camels_*.txt"):
        with path.open(newline="") as source:
            rows = list(csv.DictReader(source, delimiter=";"))
        assert set(table["gauge_id"]) == {row["gauge_id"] for row in rows}
        for row in rows:
            for field, value in row.items():
                if field == "gauge_id":
                    continue
                fields.add(field)
                actual = by_id[row["gauge_id"]][field]
                if value == "NA":
                    assert actual is None
                else:
                    assert actual == (value if field in strings else float(value))
    assert fields == set(table.columns) - {"gauge_id"}
    assert by_id["01013500"]["huc_02"] == "01"
    assert by_id["01013500"]["dom_land_cover"] == "    Mixed Forests"
    assert by_id["01013500"]["geol_porostiy"] == 0.1714
    assert by_id["03281100"]["q_mean"] is None
    assert attributes.metadata["fields"]["zero_q_freq"]["unit"] == "%"
    assert attributes.metadata["fields"]["geol_permeability"]["unit"] == "m2"
    assert "log10" in attributes.metadata["fields"]["geol_permeability"]["description"]
