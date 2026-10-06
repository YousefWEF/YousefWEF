"""Tests for :mod:`wefnexus.io` - basin / scenario (de)serialisation.

Covers: full round trips of every field of ``Basin`` / ``Riparian`` /
``WaterDemand`` / ``Crop`` / ``EnergySystem`` (dict, JSON text, JSON file),
``Sector``-keyed dictionaries written with ``Sector.value`` keys, defaults on
minimal documents, strict-JSON behaviour (no ``NaN`` / ``Infinity``), clear
``ValueError`` messages on malformed documents, scenario round trips for the
whole scenario library, and no-mutation / no-aliasing guarantees.
"""
from __future__ import annotations

import copy
import dataclasses
import doctest
import json
import math
import pathlib
import subprocess
import sys

import pytest

import wefnexus.io as io_module
from wefnexus import io as IO
from wefnexus import scenarios as S
from wefnexus.data import example_basin
from wefnexus.models import (
    DEFAULT_CONSUMPTION_FRACTION,
    DEFAULT_VALUE_USD_PER_M3,
    Basin,
    Crop,
    EnergySystem,
    Riparian,
    Scenario,
    Sector,
    WaterDemand,
)
from wefnexus.nexus import run_nexus

WITHDRAWAL_SECTOR_VALUES = {"municipal", "industrial", "agricultural", "energy"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _walk(obj):
    """Yield every leaf (and every dict key) of a nested JSON-like structure."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _walk(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk(v)
    else:
        yield obj


def _minimal_doc():
    return {
        "name": "Toy",
        "riparians": [
            {
                "name": "Up",
                "population": 1e6,
                "gdp_usd": 1e9,
                "local_inflow_mm3": 500.0,
                "demand": {"municipal": 50.0, "environmental": 100.0},
            },
            {
                "name": "Down",
                "population": 2e6,
                "gdp_usd": 3e9,
                "local_inflow_mm3": 100.0,
                "demand": {"agricultural": 200.0},
            },
        ],
    }


def _toy_basin():
    """A basin with unusual-but-valid field values to exercise every path."""
    up = Riparian(
        name="Up",
        population=1_000_000,
        gdp_usd=1e9,
        local_inflow_mm3=500.0,
        demand=WaterDemand(
            municipal=50.0,
            industrial=10.0,
            agricultural=0.0,
            energy=5.0,
            environmental=100.0,
            consumption_fraction={Sector.MUNICIPAL: 0.3, Sector.INDUSTRIAL: 0.15},
            value_usd_per_m3={Sector.MUNICIPAL: 2.0, Sector.ENVIRONMENT: 0.05},
        ),
        crops=[],
        energy=EnergySystem(demand_gwh=100.0, hydropower_capacity_mw=0.0),
        reservoir_capacity_mm3=0.0,
        reservoir_storage_mm3=0.0,
        treaty_allocation_mm3=None,
    )
    down = Riparian(
        name="Río Abajo",
        population=2_000_000,
        gdp_usd=3e9,
        local_inflow_mm3=100.0,
        demand=WaterDemand(municipal=80.0, agricultural=300.0, environmental=50.0),
        crops=[Crop("maize", area_ha=10_000, kc=0.9, season_days=130, yield_max_t_ha=6.0, ky=1.25, kcal_per_kg=3600.0)],
        treaty_allocation_mm3=250.0,
        groundwater_recharge_mm3=20.0,
        groundwater_abstraction_mm3=30.0,
    )
    return Basin("Toy basin", [up, down], headwater_inflow_mm3=10.0, climate_cv=0.35)


@pytest.fixture
def toy():
    return _toy_basin()


# ---------------------------------------------------------------------------
# module hygiene
# ---------------------------------------------------------------------------
def test_doctests_pass():
    failed, attempted = doctest.testmod(io_module)
    assert attempted > 0
    assert failed == 0


def test_all_exports_exist_with_docstrings():
    for name in io_module.__all__:
        obj = getattr(io_module, name)
        if callable(obj):
            assert obj.__doc__ and obj.__doc__.strip(), name


def test_module_imports_cleanly():
    # Import hygiene is checked in a *fresh* interpreter (as test_scenarios.py
    # and test_sustainability.py do), never via ``importlib.reload`` in the
    # running one: a reload re-binds every function and constant of
    # ``wefnexus.io`` mid-session, so modules that hold references to it
    # (``wefnexus.cli`` imports it as ``_io``) and tests comparing object
    # identities would see different objects depending on test ordering.
    #
    # ``wefnexus.io`` is a leaf module: importing it must not pull in
    # ``wefnexus.nexus`` (ARCHITECTURE.md, "Testing & quality bar") nor any
    # optional / heavy dependency.
    code = (
        "import sys, wefnexus.io; "
        "assert 'wefnexus.nexus' not in sys.modules, 'nexus'; "
        "assert 'pandas' not in sys.modules, 'pandas'; "
        "assert 'matplotlib' not in sys.modules, 'matplotlib'; "
        "assert 'networkx' not in sys.modules, 'networkx'; "
        "assert 'scipy' not in sys.modules, 'scipy'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# ---------------------------------------------------------------------------
# sector keys
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sector", list(Sector))
def test_sector_key_round_trip(sector):
    assert IO.sector_key(sector) == sector.value
    assert IO.parse_sector(IO.sector_key(sector)) is sector
    assert IO.parse_sector(sector.name) is sector
    assert IO.parse_sector(f"  {sector.value.upper()} ") is sector
    assert IO.parse_sector(sector) is sector


@pytest.mark.parametrize("bad", ["water", "", "MUNICIPALITY", 3, None])
def test_parse_sector_rejects_unknown(bad):
    with pytest.raises(ValueError):
        IO.parse_sector(bad)


# ---------------------------------------------------------------------------
# basin_to_dict
# ---------------------------------------------------------------------------
def test_basin_to_dict_structure(basin):
    d = IO.basin_to_dict(basin)
    assert d["format"] == IO.BASIN_FORMAT
    assert d["format_version"] == IO.FORMAT_VERSION
    assert set(d) == {"format", "format_version"} | {f.name for f in dataclasses.fields(Basin)}
    assert d["name"] == basin.name
    assert d["headwater_inflow_mm3"] == 2000.0
    assert d["climate_cv"] == 0.20
    assert [r["name"] for r in d["riparians"]] == ["Highland", "Midland", "Delta"]
    riparian_fields = {f.name for f in dataclasses.fields(Riparian)}
    demand_fields = {f.name for f in dataclasses.fields(WaterDemand)}
    crop_fields = {f.name for f in dataclasses.fields(Crop)}
    energy_fields = {f.name for f in dataclasses.fields(EnergySystem)}
    for i, rd in enumerate(d["riparians"]):
        assert set(rd) == riparian_fields
        assert rd["position"] == i
        assert set(rd["demand"]) == demand_fields
        assert set(rd["energy"]) == energy_fields
        assert isinstance(rd["crops"], list)
        for cd in rd["crops"]:
            assert set(cd) == crop_fields


def test_basin_to_dict_values(basin):
    d = IO.basin_to_dict(basin)
    delta = d["riparians"][2]
    assert delta["treaty_allocation_mm3"] == 16_000.0
    assert delta["crops"][0]["name"] == "rice"
    assert delta["crops"][0]["season_days"] == 150
    assert delta["energy"]["desalination_capacity_mm3"] == 300.0
    assert delta["demand"]["agricultural"] == 13_000.0
    assert delta["demand"]["environmental"] == 1_500.0
    assert d["riparians"][0]["energy"]["hydropower_capacity_mw"] == 2_000.0


def test_sector_dicts_use_sector_values_as_keys(basin):
    d = IO.basin_to_dict(basin)
    for rd in d["riparians"]:
        cf = rd["demand"]["consumption_fraction"]
        vu = rd["demand"]["value_usd_per_m3"]
        assert set(cf) == WITHDRAWAL_SECTOR_VALUES
        assert set(vu) == WITHDRAWAL_SECTOR_VALUES
        assert cf["agricultural"] == DEFAULT_CONSUMPTION_FRACTION[Sector.AGRICULTURAL]
        assert vu["municipal"] == DEFAULT_VALUE_USD_PER_M3[Sector.MUNICIPAL]
    assert not any(isinstance(x, Sector) for x in _walk(d))


def test_basin_to_dict_is_strict_json_serialisable(basin, toy):
    for b in (basin, toy):
        text = json.dumps(IO.basin_to_dict(b), allow_nan=False)
        assert json.loads(text) == IO.basin_to_dict(b)


def test_basin_to_dict_returns_fresh_containers(basin):
    d1 = IO.basin_to_dict(basin)
    d2 = IO.basin_to_dict(basin)
    assert d1 == d2
    assert d1 is not d2 and d1["riparians"] is not d2["riparians"]
    # mutating the dictionary never touches the basin
    d1["riparians"][0]["demand"]["municipal"] = -1.0
    d1["riparians"][0]["crops"].clear()
    d1["riparians"][0]["demand"]["consumption_fraction"]["municipal"] = 0.99
    assert basin.riparians[0].demand.municipal == 600.0
    assert len(basin.riparians[0].crops) == 2
    assert basin.riparians[0].demand.consumption_fraction[Sector.MUNICIPAL] == 0.2


def test_basin_to_dict_does_not_mutate(basin):
    # snapshot with ``asdict`` (deep copies the nested sector dictionaries) so
    # that the guard does not rely on how deep ``Basin.copy()`` happens to be
    before = dataclasses.asdict(basin)
    IO.basin_to_dict(basin)
    IO.basin_to_json(basin)
    assert dataclasses.asdict(basin) == before
    assert basin.riparians[2].demand.value_usd_per_m3[Sector.MUNICIPAL] == 1.5
    assert basin.riparians[0].demand.consumption_fraction[Sector.MUNICIPAL] == 0.2
    assert [r.position for r in basin] == [0, 1, 2]


def test_basin_to_dict_rejects_non_basin():
    with pytest.raises(ValueError, match="Basin"):
        IO.basin_to_dict({"name": "x"})
    with pytest.raises(ValueError):
        IO.basin_to_dict(None)


def test_basin_to_dict_rejects_non_finite():
    # fresh basins each time: Basin.copy() shares the sector dicts of WaterDemand
    b = example_basin()
    b.riparians[1].local_inflow_mm3 = math.inf
    with pytest.raises(ValueError, match="local_inflow_mm3"):
        IO.basin_to_dict(b)
    b = example_basin()
    b.riparians[0].demand.consumption_fraction[Sector.MUNICIPAL] = math.nan
    with pytest.raises(ValueError, match="consumption_fraction"):
        IO.basin_to_dict(b)
    b = example_basin()
    b.riparians[2].crops[0].area_ha = float("-inf")
    with pytest.raises(ValueError, match="area_ha"):
        IO.basin_to_dict(b)


def test_basin_to_dict_rejects_wrong_nested_types(basin):
    b = basin.copy()
    b.riparians[0].crops = [{"name": "wheat"}]
    with pytest.raises(ValueError, match="Crop"):
        IO.basin_to_dict(b)
    b = basin.copy()
    b.riparians[0].energy = None
    with pytest.raises(ValueError, match="EnergySystem"):
        IO.basin_to_dict(b)
    b = basin.copy()
    b.riparians[0].demand = {"municipal": 1}
    with pytest.raises(ValueError, match="WaterDemand"):
        IO.basin_to_dict(b)
    b = basin.copy()
    b.riparians[0].demand.consumption_fraction = {"water": 0.5}
    with pytest.raises(ValueError, match="sector"):
        IO.basin_to_dict(b)


# ---------------------------------------------------------------------------
# basin_to_json
# ---------------------------------------------------------------------------
def test_basin_to_json_text_matches_dict(basin):
    text = IO.basin_to_json(basin)
    assert isinstance(text, str)
    assert json.loads(text) == IO.basin_to_dict(basin)
    one_line = IO.basin_to_json(basin, indent=None)
    assert "\n" not in one_line
    assert json.loads(one_line) == IO.basin_to_dict(basin)


def test_basin_to_json_writes_file(basin, tmp_path):
    path = tmp_path / "basin.json"
    text = IO.basin_to_json(basin, path)
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert content == text + "\n"
    assert json.loads(content) == IO.basin_to_dict(basin)
    # str path works too and overwrites
    text2 = IO.basin_to_json(basin, str(path), indent=None)
    assert path.read_text(encoding="utf-8") == text2 + "\n"


def test_basin_to_json_keeps_unicode(toy, tmp_path):
    text = IO.basin_to_json(toy, tmp_path / "toy.json")
    assert "Río Abajo" in text
    assert IO.load_basin(tmp_path / "toy.json").riparians[1].name == "Río Abajo"


def test_basin_to_json_non_finite_raises_before_writing(basin, tmp_path):
    b = basin.copy()
    b.headwater_inflow_mm3 = math.inf
    path = tmp_path / "never.json"
    with pytest.raises(ValueError):
        IO.basin_to_json(b, path)
    assert not path.exists()


def test_basin_to_json_unwritable_path_raises_oserror(basin, tmp_path):
    with pytest.raises(OSError):
        IO.basin_to_json(basin, tmp_path / "missing_dir" / "basin.json")


# ---------------------------------------------------------------------------
# basin_from_dict / load_basin round trips
# ---------------------------------------------------------------------------
def test_round_trip_example_basin(basin):
    loaded = IO.basin_from_dict(IO.basin_to_dict(basin))
    assert loaded == basin
    assert loaded is not basin
    assert loaded.names() == basin.names()
    assert [r.position for r in loaded] == [0, 1, 2]
    for a, b in zip(loaded.riparians, basin.riparians):
        assert a == b
        assert a.demand == b.demand
        assert a.energy == b.energy
        assert a.crops == b.crops
        assert a.demand.consumption_fraction == b.demand.consumption_fraction
        assert a.demand.value_usd_per_m3 == b.demand.value_usd_per_m3
        assert all(isinstance(k, Sector) for k in a.demand.consumption_fraction)


def test_round_trip_through_json_text_and_file(basin, tmp_path):
    text = IO.basin_to_json(basin)
    assert IO.basin_from_dict(json.loads(text)) == basin
    path = tmp_path / "azura.json"
    IO.basin_to_json(basin, path)
    assert IO.load_basin(path) == basin
    assert IO.load_basin(str(path)) == basin


def test_round_trip_toy_basin(toy, tmp_path):
    assert IO.basin_from_dict(IO.basin_to_dict(toy)) == toy
    IO.basin_to_json(toy, tmp_path / "toy.json")
    loaded = IO.load_basin(tmp_path / "toy.json")
    assert loaded == toy
    up = loaded.riparians[0]
    assert up.treaty_allocation_mm3 is None
    assert up.crops == []
    assert up.demand.consumption_fraction == {Sector.MUNICIPAL: 0.3, Sector.INDUSTRIAL: 0.15}
    assert up.demand.value_usd_per_m3 == {Sector.MUNICIPAL: 2.0, Sector.ENVIRONMENT: 0.05}
    assert loaded.riparians[1].treaty_allocation_mm3 == 250.0
    assert loaded.climate_cv == 0.35


def test_round_trip_empty_basin():
    empty = Basin("Empty", [], headwater_inflow_mm3=5.0)
    d = IO.basin_to_dict(empty)
    assert d["riparians"] == []
    assert IO.basin_from_dict(d) == empty
    assert IO.basin_from_dict(json.loads(IO.basin_to_json(empty))) == empty


def test_round_trip_preserves_number_types(basin, tmp_path):
    loaded = IO.basin_from_dict(json.loads(IO.basin_to_json(basin)))
    crop = loaded.riparians[2].crops[0]
    assert isinstance(crop.season_days, int) and crop.season_days == 150
    assert isinstance(crop.area_ha, int) and crop.area_ha == 200_000
    assert isinstance(loaded.riparians[0].population, int) and loaded.riparians[0].population == 8_000_000
    assert isinstance(loaded.riparians[0].local_inflow_mm3, float)
    assert isinstance(loaded.riparians[0].gdp_usd, float)
    # hence a re-export is byte-identical
    first = tmp_path / "first.json"
    IO.basin_to_json(basin, first)
    second = tmp_path / "second.json"
    IO.basin_to_json(IO.load_basin(first), second)
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")


def test_loaded_basin_runs_identically(basin, tmp_path):
    IO.basin_to_json(basin, tmp_path / "b.json")
    loaded = IO.load_basin(tmp_path / "b.json")
    sc = Scenario(name="t", years=2)
    a = run_nexus(basin, sc).summary()
    b = run_nexus(loaded, sc).summary()
    assert a["BASIN"]["nexus_index"] == pytest.approx(b["BASIN"]["nexus_index"])
    assert a["Delta"]["supply_ratio"] == pytest.approx(b["Delta"]["supply_ratio"])


def test_basin_from_dict_does_not_alias_or_mutate_input():
    doc = _minimal_doc()
    doc["riparians"][0]["crops"] = [
        {"name": "wheat", "area_ha": 10.0, "kc": 0.8, "season_days": 100, "yield_max_t_ha": 3.0, "ky": 1.0, "kcal_per_kg": 3400.0}
    ]
    snapshot = copy.deepcopy(doc)
    b = IO.basin_from_dict(doc)
    assert doc == snapshot
    assert "position" not in doc["riparians"][0]
    # mutate the result: the document is untouched, and vice versa
    b.riparians[0].crops.append(Crop("x", 1.0, 1.0, 1, 1.0, 1.0, 1.0))
    b.riparians[0].demand.consumption_fraction[Sector.MUNICIPAL] = 0.0
    assert doc == snapshot
    doc["riparians"][0]["crops"].clear()
    assert len(b.riparians[0].crops) == 2


def test_minimal_document_applies_defaults():
    b = IO.basin_from_dict(_minimal_doc())
    assert isinstance(b, Basin)
    assert b.headwater_inflow_mm3 == 0.0 and b.climate_cv == 0.20
    up, down = b.riparians
    assert (up.position, down.position) == (0, 1)
    assert up.demand == WaterDemand(municipal=50.0, environmental=100.0)
    assert up.demand.consumption_fraction == DEFAULT_CONSUMPTION_FRACTION
    assert up.demand.value_usd_per_m3 == DEFAULT_VALUE_USD_PER_M3
    assert up.energy == EnergySystem()
    assert up.crops == []
    assert up.treaty_allocation_mm3 is None
    assert up.irrigation_efficiency == 0.50
    assert up.reservoir_capacity_mm3 == 0.0
    assert up.material_power == 0.5
    assert down.demand.agricultural == 200.0 and down.demand.municipal == 0.0


def test_sector_keys_accept_names_and_mixed_case():
    doc = _minimal_doc()
    doc["riparians"][0]["demand"]["consumption_fraction"] = {"MUNICIPAL": 0.4, "Agricultural": 0.7}
    doc["riparians"][0]["demand"]["value_usd_per_m3"] = {"energy": 0.3}
    b = IO.basin_from_dict(doc)
    assert b.riparians[0].demand.consumption_fraction == {Sector.MUNICIPAL: 0.4, Sector.AGRICULTURAL: 0.7}
    assert b.riparians[0].demand.value_usd_per_m3 == {Sector.ENERGY: 0.3}


def test_format_tag_handling():
    doc = _minimal_doc()
    assert isinstance(IO.basin_from_dict(doc), Basin)  # no tag at all
    doc["format"] = IO.BASIN_FORMAT
    doc["format_version"] = IO.FORMAT_VERSION
    assert isinstance(IO.basin_from_dict(doc), Basin)
    doc["format"] = IO.SCENARIO_FORMAT
    with pytest.raises(ValueError, match="format"):
        IO.basin_from_dict(doc)
    doc["format"] = IO.BASIN_FORMAT
    doc["format_version"] = IO.FORMAT_VERSION + 1
    with pytest.raises(ValueError, match="format_version"):
        IO.basin_from_dict(doc)


def test_integral_floats_are_accepted_for_integer_fields():
    doc = _minimal_doc()
    doc["riparians"][0]["crops"] = [
        {"name": "w", "area_ha": 1.0, "kc": 0.8, "season_days": 120.0, "yield_max_t_ha": 3.0, "ky": 1.0, "kcal_per_kg": 3400}
    ]
    crop = IO.basin_from_dict(doc).riparians[0].crops[0]
    assert crop.season_days == 120 and isinstance(crop.season_days, int)
    doc["riparians"][0]["crops"][0]["season_days"] = 120.5
    with pytest.raises(ValueError, match="season_days"):
        IO.basin_from_dict(doc)


def _bad_docs():
    cases = []

    def add(label, mutate, match):
        doc = _minimal_doc()
        mutate(doc)
        cases.append(pytest.param(doc, match, id=label))

    add("missing-name", lambda d: d.pop("name"), "name")
    add("missing-riparians", lambda d: d.pop("riparians"), "riparians")
    add("riparians-not-list", lambda d: d.update(riparians={"a": 1}), "list")
    add("riparian-not-dict", lambda d: d["riparians"].__setitem__(0, "Up"), "object")
    add("riparian-missing-population", lambda d: d["riparians"][0].pop("population"), "population")
    add("riparian-missing-demand", lambda d: d["riparians"][0].pop("demand"), "demand")
    add("unknown-top-key", lambda d: d.update(headwater=1.0), "unknown key")
    add("unknown-riparian-key", lambda d: d["riparians"][0].update(polulation=1.0), "polulation")
    add("unknown-demand-key", lambda d: d["riparians"][0]["demand"].update(domestic=1.0), "domestic")
    add("string-number", lambda d: d["riparians"][0].update(population="many"), "population")
    add("bool-number", lambda d: d["riparians"][0].update(gdp_usd=True), "gdp_usd")
    add("nan-number", lambda d: d["riparians"][0].update(local_inflow_mm3=math.nan), "finite")
    add("inf-number", lambda d: d.update(headwater_inflow_mm3=math.inf), "finite")
    add("name-not-string", lambda d: d["riparians"][0].update(name=3), "name")
    add("treaty-string", lambda d: d["riparians"][0].update(treaty_allocation_mm3="none"), "treaty_allocation_mm3")
    add("crops-not-list", lambda d: d["riparians"][0].update(crops={"wheat": 1}), "crops")
    add("crop-missing-kc", lambda d: d["riparians"][0].update(crops=[{"name": "w", "area_ha": 1, "season_days": 1, "yield_max_t_ha": 1, "ky": 1, "kcal_per_kg": 1}]), "kc")
    add("crop-unknown-key", lambda d: d["riparians"][0].update(crops=[{"name": "w", "area_ha": 1, "kc": 1, "season_days": 1, "yield_max_t_ha": 1, "ky": 1, "kcal_per_kg": 1, "colour": "green"}]), "colour")
    add("energy-not-dict", lambda d: d["riparians"][0].update(energy=[1, 2]), "energy")
    add("energy-unknown-key", lambda d: d["riparians"][0].update(energy={"solar_mw": 1.0}), "solar_mw")
    add("demand-not-dict", lambda d: d["riparians"][0].update(demand=5.0), "demand")
    add("unknown-sector", lambda d: d["riparians"][0]["demand"].update(consumption_fraction={"water": 0.5}), "sector")
    add("sector-dict-not-dict", lambda d: d["riparians"][0]["demand"].update(consumption_fraction=[0.5]), "consumption_fraction")
    add("position-mismatch", lambda d: d["riparians"][0].update(position=1), "position")
    add("duplicate-names", lambda d: d["riparians"][1].update(name="Up"), "unique")
    add("non-string-key", lambda d: d.__setitem__(3, 1), "key")
    return cases


@pytest.mark.parametrize("doc, match", _bad_docs())
def test_basin_from_dict_rejects_malformed_documents(doc, match):
    with pytest.raises(ValueError, match=match):
        IO.basin_from_dict(doc)


@pytest.mark.parametrize("doc", [[], "basin", 3, None])
def test_basin_from_dict_rejects_non_mapping(doc):
    with pytest.raises(ValueError, match="object"):
        IO.basin_from_dict(doc)


def test_load_basin_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        IO.load_basin(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        IO.load_basin(bad)
    inf = tmp_path / "inf.json"
    inf.write_text('{"name": "x", "riparians": [], "headwater_inflow_mm3": Infinity}', encoding="utf-8")
    with pytest.raises(ValueError, match="Infinity"):
        IO.load_basin(inf)
    nan = tmp_path / "nan.json"
    nan.write_text('{"name": "x", "riparians": [], "climate_cv": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="NaN"):
        IO.load_basin(nan)
    lst = tmp_path / "list.json"
    lst.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="object"):
        IO.load_basin(lst)
    sc = tmp_path / "scenario.json"
    IO.scenario_to_json(Scenario(), sc)
    with pytest.raises(ValueError, match="format"):
        IO.load_basin(sc)


def test_load_basin_accepts_pathlib_and_str(basin, tmp_path):
    p = pathlib.Path(tmp_path) / "p.json"
    IO.basin_to_json(basin, p)
    assert IO.load_basin(p) == IO.load_basin(str(p)) == basin


# ---------------------------------------------------------------------------
# scenarios
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(S.SCENARIOS))
def test_scenario_round_trip_library(name):
    sc = S.get_scenario(name, years=3)
    d = IO.scenario_to_dict(sc)
    assert d["format"] == IO.SCENARIO_FORMAT
    assert set(d) == {"format", "format_version"} | {f.name for f in dataclasses.fields(Scenario)}
    assert IO.scenario_from_dict(d) == sc
    assert IO.scenario_from_dict(json.loads(json.dumps(d, allow_nan=False))) == sc


def test_scenario_round_trip_custom(tmp_path):
    sc = Scenario(
        name="custom",
        start_year=2030,
        years=12,
        flow_change_pct_by_end=-25.0,
        population_growth_rate=0.02,
        gdp_growth_rate=0.03,
        demand_growth_rate=0.01,
        energy_demand_growth_rate=0.04,
        irrigated_area_change_pct_by_end=15.0,
        irrigation_efficiency_target=0.75,
        renewable_share_target=None,
        cooperation=False,
        allocation_rule="cea",
        stochastic=True,
        seed=7,
        drought_years=[2, 5, 9],
        drought_severity=0.5,
        description="a test",
    )
    d = IO.scenario_to_dict(sc)
    assert d["drought_years"] == [2, 5, 9] and d["drought_years"] is not sc.drought_years
    assert d["renewable_share_target"] is None
    assert IO.scenario_from_dict(d) == sc
    path = tmp_path / "sc.json"
    text = IO.scenario_to_json(sc, path)
    assert path.read_text(encoding="utf-8") == text + "\n"
    loaded = IO.load_scenario(path)
    assert loaded == sc
    assert loaded.drought_years is not sc.drought_years
    assert loaded.flow_factor(5) == pytest.approx(sc.flow_factor(5))


def test_scenario_to_dict_does_not_mutate():
    sc = Scenario(name="s", drought_years=[1])
    snapshot = dataclasses.replace(sc, drought_years=list(sc.drought_years))
    d = IO.scenario_to_dict(sc)
    d["drought_years"].append(99)
    d["name"] = "changed"
    assert sc == snapshot


def test_scenario_from_dict_defaults_and_partial():
    assert IO.scenario_from_dict({}) == Scenario()
    sc = IO.scenario_from_dict({"years": 4, "drought_years": [2.0], "format": IO.SCENARIO_FORMAT})
    assert sc.years == 4 and sc.drought_years == [2] and isinstance(sc.drought_years[0], int)
    assert sc.name == "baseline" and sc.cooperation is True


def test_scenario_from_dict_does_not_alias_input():
    doc = {"drought_years": [1, 2]}
    sc = IO.scenario_from_dict(doc)
    sc.drought_years.append(3)
    assert doc["drought_years"] == [1, 2]


@pytest.mark.parametrize(
    "doc, match",
    [
        ({"yearz": 3}, "unknown key"),
        ({"years": "5"}, "years"),
        ({"years": 2.5}, "years"),
        ({"cooperation": 1}, "cooperation"),
        ({"stochastic": "yes"}, "stochastic"),
        ({"drought_years": "1"}, "drought_years"),
        ({"drought_years": [1.5]}, "drought_years"),
        ({"drought_years": [True]}, "drought_years"),
        ({"drought_severity": math.nan}, "finite"),
        ({"flow_change_pct_by_end": "-20"}, "flow_change_pct_by_end"),
        ({"irrigation_efficiency_target": "0.7"}, "irrigation_efficiency_target"),
        ({"name": 5}, "name"),
        ({"format": IO.BASIN_FORMAT}, "format"),
        ({"format_version": 99}, "format_version"),
    ],
)
def test_scenario_from_dict_rejects_malformed(doc, match):
    with pytest.raises(ValueError, match=match):
        IO.scenario_from_dict(doc)


@pytest.mark.parametrize("doc", [[], "x", None])
def test_scenario_from_dict_rejects_non_mapping(doc):
    with pytest.raises(ValueError):
        IO.scenario_from_dict(doc)


def test_scenario_to_dict_rejects_non_scenario_and_non_finite():
    with pytest.raises(ValueError, match="Scenario"):
        IO.scenario_to_dict({"name": "x"})
    with pytest.raises(ValueError, match="finite"):
        IO.scenario_to_dict(Scenario(drought_severity=math.inf))
    with pytest.raises(ValueError, match="drought_years"):
        IO.scenario_to_dict(Scenario(drought_years=[1.5]))


def test_load_scenario_errors(tmp_path, basin):
    with pytest.raises(FileNotFoundError):
        IO.load_scenario(tmp_path / "nope.json")
    b = tmp_path / "basin.json"
    IO.basin_to_json(basin, b)
    with pytest.raises(ValueError, match="format"):
        IO.load_scenario(b)
    bad = tmp_path / "bad.json"
    bad.write_text("nope", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        IO.load_scenario(bad)


# ---------------------------------------------------------------------------
# regression guard: no test may reload ``wefnexus.io`` in-process
# ---------------------------------------------------------------------------
# Snapshot of the public objects taken when this module was *collected*.  If
# any test (here or elsewhere) reloaded ``wefnexus.io`` with ``importlib``
# the module's attributes would be re-bound to new objects and the identity
# checks below would fail, so this test is deliberately the last in the file.
_EXPORTS_AT_COLLECTION = {name: getattr(io_module, name) for name in io_module.__all__}


def test_io_objects_are_not_rebound_during_the_session():
    assert io_module is IO
    assert sys.modules["wefnexus.io"] is io_module
    for name, obj in _EXPORTS_AT_COLLECTION.items():
        assert getattr(io_module, name) is obj, f"wefnexus.io.{name} was re-bound (module reloaded?)"
    # the CLI's private alias must point at the very same module object
    from wefnexus import cli

    assert cli._io is io_module
    assert cli._io.basin_to_dict is _EXPORTS_AT_COLLECTION["basin_to_dict"]
