"""Static attributes stay labeled and independent of daily records."""

from pathlib import Path

import pandas as pd
import pytest

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
    table = read_attributes(output, ["00000002", "01013500"])
    assert table.index.name == "gauge_id"
    assert table.index.tolist() == ["00000002", "01013500"]
    assert table.loc["01013500", "huc_02"] == "01"
    assert table.loc["01013500", "gauge_name"] == "A River"
    assert table.loc["01013500", "p_mean"] == 3.12667898699521
    assert table.loc["01013500", "dom_land_cover"] == "    Mixed Forests"
    assert pd.isna(table.loc["01013500", "q5"])
    assert pd.isna(table.loc["01013500", "geol_2nd_class"])
    assert table.loc["00000002", "q5"] == 0
    fields = table.attrs["fields"]
    assert fields["p_mean"]["unit"] == "mm/day"
    assert fields["soil_depth_statsgo"]["documented_name"] == "soil_depth_statgso"
    assert fields["geol_porostiy"]["source_file"] == "camels_geol.txt"
    assert fields["huc_02"]["dtype"] == "string"
    for path in output.iterdir():
        path.unlink()
    assert table.loc["00000002", "gauge_name"] == "B River"


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
    table = read_attributes(output, selected.ids)
    assert tuple(table.index) == selected.ids
    assert table["gauge_name"].tolist() == ["B River", "A River"]


def test_local_attributes_match_all_source_identities_fields_and_values(tmp_path: Path) -> None:
    import csv

    raw = Path(__file__).resolve().parents[1] / "data/camels-us/raw"
    if not (raw / "camels_topo.txt").exists():
        pytest.skip("Local CAMELS-US attributes are not available")
    output = tmp_path / "attributes"
    prepare_attributes(raw, output)
    table = read_attributes(output)
    assert table.shape == (671, 59)
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
        assert set(table.index) == {row["gauge_id"] for row in rows}
        for row in rows:
            for field, value in row.items():
                if field == "gauge_id":
                    continue
                fields.add(field)
                actual = table.loc[row["gauge_id"], field]
                if value == "NA":
                    assert pd.isna(actual)
                else:
                    assert actual == (value if field in strings else float(value))
    assert fields == set(table.columns)
    assert table.loc["01013500", "huc_02"] == "01"
    assert table.loc["01013500", "dom_land_cover"] == "    Mixed Forests"
    assert table.loc["01013500", "geol_porostiy"] == 0.1714
    assert pd.isna(table.loc["03281100", "q_mean"])
    assert table.attrs["fields"]["zero_q_freq"]["unit"] == "%"
    assert table.attrs["fields"]["geol_permeability"]["unit"] == "m2"
    assert "log10" in table.attrs["fields"]["geol_permeability"]["description"]
