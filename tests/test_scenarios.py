"""Tests for :mod:`wefnexus.scenarios` - scenario library, runner and comparison tables.

Covers the ARCHITECTURE.md section-9 contract: the eight scenario factories
and :data:`SCENARIOS` / :func:`get_scenario`, :func:`run_scenarios` on fresh
basin copies, :func:`comparison_table` rows (one per scenario x riparian
plus a basin row, numeric columns) and :func:`format_table`; closed-form
checks of the Scenario fields each factory sets, property checks (bounds,
mass balance, scenario-direction properties such as "adaptation beats
stress for Delta"), edge cases (short horizons, droughts beyond the
horizon, single scenario, mapping input) and no-mutation guarantees.
"""
from __future__ import annotations

import csv
import dataclasses
import doctest
import math
import subprocess
import sys

import pytest

import wefnexus.scenarios as scenarios_module
from wefnexus import scenarios as S
from wefnexus.data import example_basin
from wefnexus.models import Basin, Scenario
from wefnexus.nexus import ALLOCATION_RULES, BASIN_KEY, NexusModel, NexusResult

NAMES = [
    "baseline",
    "climate_change",
    "growth",
    "efficiency",
    "unilateral",
    "cooperative",
    "combined_stress",
    "combined_adaptation",
]
RIPARIANS = ["Highland", "Midland", "Delta"]
TOL = 1e-9


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def short_results():
    """Three short scenarios (years=4) run on the example basin."""
    return S.run_scenarios(example_basin(), [S.baseline(years=4), S.unilateral(years=4), S.cooperative(years=4)])


@pytest.fixture(scope="module")
def all_results():
    """Every library scenario (years=4) run on the example basin."""
    return S.run_scenarios(example_basin(), S.all_scenarios(years=4))


@pytest.fixture(scope="module")
def all_rows(all_results):
    return S.comparison_table(all_results)


@pytest.fixture(scope="module")
def ten_year():
    """combined_stress vs combined_adaptation over 10 years."""
    return S.run_scenarios(example_basin(), [S.combined_stress(years=10), S.combined_adaptation(years=10)])


def _row(rows, scenario, riparian):
    picked = [r for r in rows if r["scenario"] == scenario and r["riparian"] == riparian]
    assert len(picked) == 1, (scenario, riparian)
    return picked[0]


def _fields_except(sc: Scenario, *skip):
    d = dataclasses.asdict(sc)
    for k in skip:
        d.pop(k)
    return d


# ---------------------------------------------------------------------------
# module surface
# ---------------------------------------------------------------------------
def test_doctests_pass():
    failed, attempted = doctest.testmod(scenarios_module)
    assert attempted > 0
    assert failed == 0


def test_all_names_exist_and_public_functions_have_docstrings():
    for name in scenarios_module.__all__:
        assert hasattr(scenarios_module, name), name
        obj = getattr(scenarios_module, name)
        if callable(obj):
            assert obj.__doc__ and len(obj.__doc__) > 50, name


def test_registry_and_constants_match_contract():
    assert list(S.SCENARIOS) == NAMES
    assert S.scenario_names() == NAMES
    for name in NAMES:
        assert S.SCENARIOS[name] is getattr(S, name)
    assert S.DEFAULT_YEARS == 25
    assert S.DEFAULT_FLOW_CHANGE_PCT == -20
    assert tuple(S.DEFAULT_DROUGHT_YEARS) == (8, 15, 22)
    assert S.DEFAULT_POPULATION_GROWTH == 0.015
    assert S.DEFAULT_DEMAND_GROWTH == 0.02
    assert S.DEFAULT_ENERGY_GROWTH == 0.03
    assert S.DEFAULT_IRRIGATED_AREA_CHANGE_PCT == 20
    assert S.DEFAULT_IRRIGATION_EFFICIENCY_TARGET == 0.7
    assert S.DEFAULT_RENEWABLE_SHARE_TARGET == 0.5
    assert S.DEFAULT_COOPERATIVE_RULE == "talmud"
    # ARCHITECTURE.md section 9: combined_adaptation = climate + growth + efficiency + talmud
    assert S.DEFAULT_ADAPTATION_RULE == "talmud"
    assert S.DEFAULT_ADAPTATION_RULE in ALLOCATION_RULES
    # table column layout
    assert S.TABLE_COLUMNS == S.KEY_COLUMNS + S.INFO_COLUMNS + S.COMPARISON_COLUMNS + S.BASIN_COLUMNS
    assert len(set(S.TABLE_COLUMNS)) == len(S.TABLE_COLUMNS)
    for required in (
        "mean_supply_ratio",
        "min_supply_ratio",
        "mean_nexus_index",
        "mean_water_security",
        "mean_energy_security",
        "mean_food_security",
        "years_env_flow_unmet",
        "total_hydropower_gwh",
        "food_self_sufficiency",
        "final_storage_mm3",
    ):
        assert required in S.COMPARISON_COLUMNS
    assert set(S.SCENARIO_FIELDS) == {f.name for f in dataclasses.fields(Scenario)}


def test_import_does_not_load_optional_dependencies():
    code = (
        "import sys, wefnexus.scenarios; "
        "assert 'pandas' not in sys.modules, 'pandas'; "
        "assert 'matplotlib' not in sys.modules, 'matplotlib'; "
        "assert 'networkx' not in sys.modules, 'networkx'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# ---------------------------------------------------------------------------
# factories build
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", NAMES)
def test_every_factory_builds_with_defaults(name):
    sc = S.SCENARIOS[name]()
    assert isinstance(sc, Scenario)
    assert sc.name == name
    assert sc.years == 25
    assert sc.start_year == 2025
    assert isinstance(sc.description, str) and len(sc.description) > 30
    assert "25 years" in sc.description
    assert sc.allocation_rule in ALLOCATION_RULES
    # get_scenario builds the same object
    assert dataclasses.asdict(S.get_scenario(name)) == dataclasses.asdict(sc)


@pytest.mark.parametrize("name", NAMES)
def test_every_factory_builds_short_horizon(name):
    sc = S.SCENARIOS[name](years=4)
    assert sc.years == 4
    assert "4 years" in sc.description
    if name in ("climate_change", "combined_stress", "combined_adaptation"):
        # droughts beyond the horizon are kept (they never occur) and the text says so
        assert sc.drought_years == [8, 15, 22]
        assert "without drought years inside the horizon" in sc.description
        assert all(sc.flow_factor(i) > 0.8 - TOL for i in range(4))
    assert S.SCENARIOS[name](years=1).years == 1


def test_factories_return_fresh_independent_objects():
    a = S.climate_change(years=3)
    b = S.climate_change(years=3)
    assert a is not b and dataclasses.asdict(a) == dataclasses.asdict(b)
    a.drought_years.append(99)
    assert b.drought_years == [8, 15, 22]
    assert tuple(S.DEFAULT_DROUGHT_YEARS) == (8, 15, 22)
    assert S.climate_change(years=3).drought_years == [8, 15, 22]


# ---------------------------------------------------------------------------
# factories: closed-form field checks
# ---------------------------------------------------------------------------
def test_baseline_is_business_as_usual():
    sc = S.baseline(years=6)
    ref = Scenario(name="baseline", years=6, description=sc.description)
    assert dataclasses.asdict(sc) == dataclasses.asdict(ref)
    assert sc.cooperation is True and sc.allocation_rule == "treaty"
    assert all(sc.flow_factor(i) == 1.0 for i in range(6))
    assert all(sc.population_factor(i) == 1.0 for i in range(6))
    assert sc.irrigation_efficiency(0.5, 5) == 0.5 and sc.renewable_share(0.1, 5) == 0.1


def test_climate_change_closed_form():
    sc = S.climate_change()
    assert sc.flow_change_pct_by_end == -20.0
    assert sc.drought_years == [8, 15, 22]
    assert sc.drought_severity == 0.4
    assert sc.flow_factor(24) == pytest.approx(0.8)
    assert sc.flow_factor(0) == 1.0
    assert sc.flow_factor(8) == pytest.approx((1.0 - 0.2 * 8 / 24) * 0.6)
    # demand drivers untouched
    assert sc.population_growth_rate == 0.0 and sc.demand_growth_rate == 0.0
    assert sc.irrigated_area_change_pct_by_end == 0.0 and sc.irrigation_efficiency_target is None
    assert "-20 %" in sc.description and "[8, 15, 22]" in sc.description
    custom = S.climate_change(years=11, flow_change_pct=-50, droughts=[3, 3, 1], drought_severity=0.25)
    assert custom.drought_years == [1, 3]
    assert custom.flow_factor(10) == pytest.approx(0.5)
    assert custom.flow_factor(1) == pytest.approx((1.0 - 0.05) * 0.75)
    wetter = S.climate_change(years=3, flow_change_pct=10, droughts=())
    assert wetter.flow_factor(2) == pytest.approx(1.1) and wetter.drought_years == []


def test_growth_closed_form():
    sc = S.growth()
    assert sc.population_growth_rate == 0.015
    assert sc.demand_growth_rate == 0.02
    assert sc.energy_demand_growth_rate == 0.03
    assert sc.gdp_growth_rate == 0.02  # defaults to demand growth
    assert sc.irrigated_area_change_pct_by_end == 20.0
    assert sc.population_factor(1) == pytest.approx(1.015)
    assert sc.demand_factor(2) == pytest.approx(1.02 ** 2)
    assert sc.energy_demand_factor(3) == pytest.approx(1.03 ** 3)
    assert sc.irrigated_area_factor(24) == pytest.approx(1.2)
    assert sc.flow_change_pct_by_end == 0.0 and sc.drought_years == []
    custom = S.growth(years=3, population_growth=0.1, demand_growth=0.0, energy_growth=0.05,
                      irrigated_area_change_pct=-50, gdp_growth=0.04)
    assert custom.population_factor(2) == pytest.approx(1.21)
    assert custom.gdp_growth_rate == 0.04
    assert custom.irrigated_area_factor(2) == pytest.approx(0.5)


def test_efficiency_closed_form():
    sc = S.efficiency()
    assert sc.irrigation_efficiency_target == 0.7
    assert sc.renewable_share_target == 0.5
    assert sc.irrigation_efficiency(0.5, 24) == pytest.approx(0.7)
    assert sc.irrigation_efficiency(0.5, 0) == 0.5
    assert sc.irrigation_efficiency(0.5, 12) == pytest.approx(0.6)
    assert sc.renewable_share(0.12, 24) == pytest.approx(0.5)
    # targets below the base are allowed (linear interpolation works both ways)
    assert sc.irrigation_efficiency(0.9, 24) == pytest.approx(0.7)
    none = S.efficiency(irrigation_efficiency_target=None, renewable_share_target=None)
    assert none.irrigation_efficiency_target is None and none.renewable_share_target is None
    assert "no efficiency or renewable targets" in none.description
    only_irr = S.efficiency(renewable_share_target=None)
    assert "renewable" not in only_irr.description.split(";")[0]


def test_unilateral_closed_form():
    sc = S.unilateral(years=3)
    assert sc.cooperation is False
    assert sc.allocation_rule == "upstream_priority"
    assert NexusModel(example_basin(), sc).allocation_rule == "upstream_priority"
    # drivers as in baseline
    assert _fields_except(sc, "name", "description", "cooperation", "allocation_rule") == _fields_except(
        S.baseline(years=3), "name", "description", "cooperation", "allocation_rule"
    )


@pytest.mark.parametrize("rule", ALLOCATION_RULES)
def test_cooperative_accepts_every_rule(rule):
    sc = S.cooperative(years=2, rule=rule)
    assert sc.allocation_rule == rule
    assert sc.cooperation is True
    assert repr(rule) in sc.description or "treaty" in sc.description or "upstream" in sc.description
    model = NexusModel(example_basin(), sc)
    assert model.allocation_rule == rule


@pytest.mark.parametrize(
    "alias, canonical",
    [
        ("Talmud", "talmud"),
        ("Contested-Garment", "talmud"),
        ("aumann maschler", "talmud"),
        ("adjusted_proportional", "ap"),
        ("CEA", "cea"),
        ("constrained_equal_losses", "cel"),
        ("equal_split", "equal"),
        ("upstream", "upstream_priority"),
        (" TREATY ", "treaty"),
    ],
)
def test_cooperative_canonicalises_rule_aliases(alias, canonical):
    assert S.cooperative(rule=alias).allocation_rule == canonical


def test_combined_stress_is_climate_plus_growth():
    sc = S.combined_stress(years=12)
    climate = S.climate_change(years=12)
    grow = S.growth(years=12)
    for fld in ("flow_change_pct_by_end", "drought_years", "drought_severity"):
        assert getattr(sc, fld) == getattr(climate, fld), fld
    for fld in (
        "population_growth_rate",
        "demand_growth_rate",
        "energy_demand_growth_rate",
        "gdp_growth_rate",
        "irrigated_area_change_pct_by_end",
    ):
        assert getattr(sc, fld) == getattr(grow, fld), fld
    assert sc.irrigation_efficiency_target is None and sc.renewable_share_target is None
    assert sc.cooperation is True and sc.allocation_rule == "treaty"
    assert "no efficiency measures" in sc.description
    # component overrides
    custom = S.combined_stress(years=12, rule="CEA", climate_kw={"flow_change_pct": -35, "droughts": [2]},
                               growth_kw={"population_growth": 0.03, "gdp_growth": 0.05})
    assert custom.allocation_rule == "cea"
    assert custom.flow_change_pct_by_end == -35 and custom.drought_years == [2]
    assert custom.population_growth_rate == 0.03 and custom.gdp_growth_rate == 0.05
    assert custom.demand_growth_rate == 0.02  # untouched default


def test_combined_adaptation_is_stress_plus_efficiency():
    adapt = S.combined_adaptation(years=12)
    stress = S.combined_stress(years=12)
    eff = S.efficiency(years=12)
    drivers = ("name", "description", "irrigation_efficiency_target", "renewable_share_target")
    skip = drivers + ("allocation_rule",)
    assert _fields_except(adapt, *skip) == _fields_except(stress, *skip)
    assert adapt.irrigation_efficiency_target == eff.irrigation_efficiency_target == 0.7
    assert adapt.renewable_share_target == eff.renewable_share_target == 0.5
    # ARCHITECTURE.md section 9: climate + growth + efficiency + talmud
    assert adapt.allocation_rule == S.DEFAULT_ADAPTATION_RULE == "talmud"
    assert adapt.allocation_rule == S.cooperative(years=12).allocation_rule
    assert adapt.cooperation is True
    assert "'talmud'" in adapt.description and "talmud" not in stress.description
    # the fixed-treaty variant is one keyword away and is exactly stress + efficiency
    treaty = S.combined_adaptation(years=12, rule="treaty")
    assert treaty.allocation_rule == stress.allocation_rule == "treaty"
    assert _fields_except(treaty, "allocation_rule", "description") == _fields_except(adapt, "allocation_rule", "description")
    assert _fields_except(treaty, *drivers) == _fields_except(stress, *drivers)
    custom = S.combined_adaptation(years=12, efficiency_kw={"irrigation_efficiency_target": 0.85},
                                   climate_kw={"drought_severity": 0.1})
    assert custom.irrigation_efficiency_target == 0.85 and custom.renewable_share_target == 0.5
    assert custom.drought_severity == 0.1 and custom.flow_change_pct_by_end == -20


def test_component_kw_validation():
    with pytest.raises(ValueError, match="climate_kw"):
        S.combined_stress(climate_kw={"flow_change": -10})
    with pytest.raises(ValueError, match="growth_kw"):
        S.combined_adaptation(growth_kw="fast")
    with pytest.raises(ValueError, match="efficiency_kw"):
        S.combined_adaptation(efficiency_kw={"renewables": 0.5})
    with pytest.raises(ValueError):
        S.combined_stress(climate_kw={"flow_change_pct": -200})


# ---------------------------------------------------------------------------
# factories: overrides and validation
# ---------------------------------------------------------------------------
def test_scenario_field_overrides():
    sc = S.baseline(years=3, stochastic=True, seed=9, start_year=2030, drought_years=[1], drought_severity=0.5)
    assert sc.stochastic is True and sc.seed == 9 and sc.start_year == 2030
    assert sc.drought_years == [1] and sc.flow_factor(1) == 0.5
    assert sc.year(2) == 2032
    renamed = S.cooperative(years=2, rule="cea", name="coop_cea", description="custom text")
    assert renamed.name == "coop_cea" and renamed.description == "custom text"
    assert renamed.allocation_rule == "cea"


@pytest.mark.parametrize(
    "call",
    [
        lambda: S.baseline(bogus=1),
        lambda: S.baseline(years=3, years_=3),
        lambda: S.climate_change(flow_change_pct_by_end=-30),  # set by the factory: use flow_change_pct
        lambda: S.climate_change(drought_years=[1]),
        lambda: S.growth(population_growth_rate=0.1),
        lambda: S.efficiency(irrigation_efficiency_target=0.7, renewable_share_target=0.5, cooperation=False),
        lambda: S.cooperative(allocation_rule="cea"),
        lambda: S.unilateral(cooperation=True),
        lambda: S.combined_stress(demand_growth_rate=0.1),
        lambda: S.combined_adaptation(irrigation_efficiency_target=0.9),
    ],
)
def test_overrides_rejected_for_unknown_or_factory_set_fields(call):
    with pytest.raises(ValueError):
        call()


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("bad_years", [0, -1, 1.5, True, "5", None, float("nan"), float("inf")])
def test_factories_reject_bad_years(name, bad_years):
    with pytest.raises(ValueError):
        S.SCENARIOS[name](years=bad_years)


@pytest.mark.parametrize(
    "call",
    [
        lambda: S.climate_change(flow_change_pct=-101),
        lambda: S.climate_change(flow_change_pct=float("inf")),
        lambda: S.climate_change(flow_change_pct="dry"),
        lambda: S.climate_change(droughts="8"),
        lambda: S.climate_change(droughts=[-1]),
        lambda: S.climate_change(droughts=[1.5]),
        lambda: S.climate_change(droughts=[True]),
        lambda: S.climate_change(droughts=5),
        lambda: S.climate_change(drought_severity=1.1),
        lambda: S.climate_change(drought_severity=-0.1),
        lambda: S.growth(population_growth=-1),
        lambda: S.growth(demand_growth=float("nan")),
        lambda: S.growth(energy_growth=True),
        lambda: S.growth(irrigated_area_change_pct=-150),
        lambda: S.growth(gdp_growth=-2),
        lambda: S.efficiency(irrigation_efficiency_target=0),
        lambda: S.efficiency(irrigation_efficiency_target=1.2),
        lambda: S.efficiency(renewable_share_target=-0.1),
        lambda: S.efficiency(renewable_share_target=1.5),
        lambda: S.cooperative(rule=""),
        lambda: S.cooperative(rule=None),
        lambda: S.cooperative(rule="harmony"),
        lambda: S.cooperative(rule=3),
        lambda: S.combined_stress(rule="nash"),
        lambda: S.combined_adaptation(rule=["talmud"]),
        lambda: S.baseline(seed=-1),
        lambda: S.baseline(stochastic="yes"),
        lambda: S.baseline(start_year=2025.5),
        lambda: S.baseline(drought_severity=2),
        lambda: S.baseline(gdp_growth_rate=-1.5),
    ],
)
def test_factories_reject_bad_parameters(call):
    with pytest.raises(ValueError):
        call()


def test_validate_scenario_returns_clean_copy_and_rejects_bad_fields():
    raw = Scenario(name="x", years=3, allocation_rule="Contested Garment", drought_years=[2, 0, 2])
    before = dataclasses.asdict(raw)
    clean = S.validate_scenario(raw)
    assert clean is not raw
    assert clean.allocation_rule == "talmud" and clean.drought_years == [0, 2]
    assert dataclasses.asdict(raw) == before  # input untouched
    assert S.validate_scenario(Scenario(years=2, cooperation=1)).cooperation is True
    for bad in (
        "not a scenario",
        Scenario(name="", years=2),
        Scenario(name="x", years=0),
        Scenario(name="x", years=2, seed=-3),
        Scenario(name="x", years=2, cooperation="yes"),
        Scenario(name="x", years=2, stochastic=2),
        Scenario(name="x", years=2, start_year=2025.5),
        Scenario(name="x", years=2, allocation_rule="nothing"),
        Scenario(name="x", years=2, irrigation_efficiency_target=0.0),
        Scenario(name="x", years=2, renewable_share_target=1.01),
        Scenario(name="x", years=2, population_growth_rate=-1.0),
        Scenario(name="x", years=2, flow_change_pct_by_end=-100.5),
        Scenario(name="x", years=2, irrigated_area_change_pct_by_end=float("nan")),
        Scenario(name="x", years=2, drought_years=[-2]),
        Scenario(name="x", years=2, drought_severity=1.5),
        Scenario(name="x", years=2, description=None),
    ):
        with pytest.raises(ValueError):
            S.validate_scenario(bad)
    # boundary values are accepted
    edge = S.validate_scenario(Scenario(years=1, flow_change_pct_by_end=-100, irrigated_area_change_pct_by_end=-100,
                                        irrigation_efficiency_target=1.0, renewable_share_target=0.0,
                                        drought_severity=1.0, seed=0))
    assert edge.flow_factor(0) == 0.0


def test_get_scenario_names_and_kwargs():
    assert S.get_scenario("Climate-Change", years=3).name == "climate_change"
    assert S.get_scenario("COMBINED ADAPTATION").name == "combined_adaptation"
    assert S.get_scenario("cooperative", years=3, rule="cel").allocation_rule == "cel"
    assert S.get_scenario("baseline", years=2, stochastic=True).stochastic is True
    with pytest.raises(ValueError, match="unknown scenario"):
        S.get_scenario("business_as_usual")
    with pytest.raises(ValueError):
        S.get_scenario("")
    with pytest.raises(ValueError):
        S.get_scenario(None)
    with pytest.raises(ValueError):
        S.get_scenario("baseline", bogus=2)


def test_all_scenarios_builds_every_factory():
    scs = S.all_scenarios(years=3, seed=5)
    assert [s.name for s in scs] == NAMES
    assert all(s.years == 3 and s.seed == 5 for s in scs)
    assert len({s.name for s in scs}) == len(NAMES)
    with pytest.raises(ValueError):
        S.all_scenarios(years=0)
    with pytest.raises(ValueError):
        S.all_scenarios(years=3, rule="cea")  # factory-specific parameter is not a Scenario field


# ---------------------------------------------------------------------------
# run_scenarios
# ---------------------------------------------------------------------------
def test_run_scenarios_keys_order_and_types(short_results):
    assert list(short_results) == ["baseline", "unilateral", "cooperative"]
    for key, res in short_results.items():
        assert isinstance(res, NexusResult)
        assert res.n_years == 4 and res.years == [2025, 2026, 2027, 2028]
        assert res.riparian_names() == RIPARIANS
        assert res.scenario.name == key
        assert len(res.records) == 12 and len(res.balances) == 4
    assert short_results["baseline"].allocation_rule == "treaty"
    assert short_results["unilateral"].allocation_rule == "upstream_priority"
    assert short_results["cooperative"].allocation_rule == "talmud"


def test_run_scenarios_matches_direct_model_run(short_results):
    direct = NexusModel(example_basin(), S.cooperative(years=4)).run()
    assert direct.to_records() == short_results["cooperative"].to_records()
    assert direct.flow_factors == short_results["cooperative"].flow_factors


def test_run_scenarios_results_differ_between_scenarios(short_results):
    base = short_results["baseline"].summary()
    uni = short_results["unilateral"].summary()
    coop = short_results["cooperative"].summary()
    # unilateral: no caps, Delta fully served; cooperative Talmud divides a consumptive estate
    # (natural flow + storage - environmental flows) that the claims fit into in a normal year,
    # so it caps every riparian at its river demand - above the 16 000 Mm3 treaty volume for Delta
    assert uni["Delta"]["supply_ratio"] >= base["Delta"]["supply_ratio"] - TOL
    assert coop["Delta"]["supply_ratio"] >= base["Delta"]["supply_ratio"] - TOL
    assert coop[BASIN_KEY]["supply_ratio"] >= base[BASIN_KEY]["supply_ratio"] - TOL
    assert coop["Delta"]["supply_ratio"] == pytest.approx(uni["Delta"]["supply_ratio"])
    assert all(math.isinf(v) for v in short_results["unilateral"].series("Delta", "entitlement_mm3"))
    assert all(v == 16000.0 for v in short_results["baseline"].series("Delta", "entitlement_mm3"))
    assert all(16000.0 < v < math.inf for v in short_results["cooperative"].series("Delta", "entitlement_mm3"))


def test_run_scenarios_does_not_mutate_inputs_and_runs_are_independent():
    basin = example_basin()
    scs = [S.cooperative(years=3), S.baseline(years=3, stochastic=True, seed=2), S.combined_adaptation(years=3)]
    before_basin = dataclasses.asdict(basin)
    before_scs = [dataclasses.asdict(s) for s in scs]
    res = S.run_scenarios(basin, scs)
    # mutate the results: the inputs must not notice
    res["baseline"].records[0].demands = {}
    res["baseline"].balances[0].reaches[0].withdrawals.clear()
    res["baseline"].scenario.years = 99
    assert dataclasses.asdict(basin) == before_basin
    assert [r.position for r in basin.riparians] == [0, 1, 2]
    assert [dataclasses.asdict(s) for s in scs] == before_scs
    # order independence: a scenario's result does not depend on what ran before it
    reversed_res = S.run_scenarios(basin, list(reversed(scs)))
    assert reversed_res["combined_adaptation"].to_records() == res["combined_adaptation"].to_records()
    alone = S.run_scenarios(basin, [S.cooperative(years=3)])
    assert alone["cooperative"].to_records() == res["cooperative"].to_records()


def test_run_scenarios_accepts_names_mapping_and_single_scenario():
    basin = example_basin()
    by_name = S.run_scenarios(basin, ["baseline", S.unilateral(years=2)])
    assert list(by_name) == ["baseline", "unilateral"]
    assert by_name["baseline"].n_years == 25 and by_name["unilateral"].n_years == 2
    mapping = S.run_scenarios(basin, {"a": S.baseline(years=2), "b": S.baseline(years=2)})
    assert list(mapping) == ["a", "b"]
    assert mapping["a"].to_records() == mapping["b"].to_records()
    single = S.run_scenarios(basin, S.growth(years=2))
    assert list(single) == ["growth"]
    single_name = S.run_scenarios(basin, "efficiency")
    assert list(single_name) == ["efficiency"]


@pytest.mark.parametrize(
    "scenarios",
    [
        [],
        {},
        [S.baseline(years=2), S.baseline(years=3)],  # duplicate names
        [42],
        ["no_such_scenario"],
        [Scenario(name="bad", years=0)],
        [Scenario(name="bad", years=2, allocation_rule="x")],
        7,
        b"baseline",
    ],
)
def test_run_scenarios_rejects_bad_scenarios(scenarios):
    with pytest.raises(ValueError):
        S.run_scenarios(example_basin(), scenarios)


def test_run_scenarios_rejects_bad_basin_and_model_kw():
    with pytest.raises(ValueError, match="Basin"):
        S.run_scenarios("basin", [S.baseline(years=2)])
    with pytest.raises(ValueError, match="NexusModel keyword"):
        S.run_scenarios(example_basin(), [S.baseline(years=2)], refill=0.5)
    with pytest.raises(ValueError):
        S.run_scenarios(example_basin(), [S.baseline(years=2)], reservoir_refill_fraction=2.0)
    with pytest.raises(ValueError):
        S.run_scenarios(example_basin(), [S.baseline(years=2)], flow_factors=[1.0])  # wrong length


def test_run_scenarios_passes_model_kw_to_every_run():
    basin = example_basin()
    scs = [S.baseline(years=3), S.cooperative(years=3)]
    no_refill = S.run_scenarios(basin, scs, reservoir_refill_fraction=0.0)
    for res in no_refill.values():
        for name in RIPARIANS:
            assert all(v == 0.0 for v in res.series(name, "storage_refill_mm3"))
    overridden = S.run_scenarios(basin, scs, allocation_rule="upstream_priority")
    assert all(res.allocation_rule == "upstream_priority" for res in overridden.values())
    fixed = S.run_scenarios(basin, scs, flow_factors=[0.5, 0.5, 0.5])
    assert all(res.flow_factors == [0.5, 0.5, 0.5] for res in fixed.values())


def test_run_scenarios_stochastic_is_seeded_and_deterministic():
    basin = example_basin()
    a = S.run_scenarios(basin, [S.baseline(years=5, stochastic=True, seed=3)])["baseline"]
    b = S.run_scenarios(basin, [S.baseline(years=5, stochastic=True, seed=3)])["baseline"]
    c = S.run_scenarios(basin, [S.baseline(years=5, stochastic=True, seed=4)])["baseline"]
    assert a.to_records() == b.to_records()
    assert a.flow_factors == b.flow_factors
    assert a.flow_factors != c.flow_factors
    assert any(abs(f - 1.0) > 1e-6 for f in a.flow_factors)


def test_all_scenarios_conserve_mass_and_keep_indices_bounded(all_results):
    assert list(all_results) == NAMES
    for res in all_results.values():
        for bal in res.balances:
            assert bal.mass_balance_error() < 1e-6
        for rec in res.records:
            for fld in ("water_security", "energy_security", "food_security", "nexus_index"):
                assert 0.0 <= getattr(rec, fld) <= 1.0 + TOL, fld
            assert 0.0 <= rec.supply_ratio() <= 1.0 + TOL
            assert rec.total_withdrawal() <= rec.total_demand() + TOL


# ---------------------------------------------------------------------------
# comparison_table
# ---------------------------------------------------------------------------
def test_comparison_table_shape_and_order(all_rows, all_results):
    assert len(all_rows) == len(NAMES) * (len(RIPARIANS) + 1)
    for r in all_rows:
        assert list(r.keys()) == list(S.TABLE_COLUMNS)
    for i, name in enumerate(NAMES):
        block = all_rows[4 * i: 4 * i + 4]
        assert [r["scenario"] for r in block] == [name] * 4
        assert [r["riparian"] for r in block] == RIPARIANS + [BASIN_KEY]
        assert all(r["years"] == 4 and isinstance(r["years"], int) for r in block)
        assert all(r["allocation_rule"] == all_results[name].allocation_rule for r in block)


def test_comparison_table_numeric_columns(all_rows):
    for r in all_rows:
        for col in S.COMPARISON_COLUMNS:
            v = r[col]
            assert isinstance(v, (int, float)) and not isinstance(v, bool), (col, v)
        assert isinstance(r["years_env_flow_unmet"], int)
        for col in S.BASIN_COLUMNS:
            if r["riparian"] == BASIN_KEY:
                assert isinstance(r[col], float), col
            else:
                assert r[col] is None, col


def test_comparison_table_bounds_and_internal_consistency(all_rows):
    for r in all_rows:
        for col in (
            "mean_supply_ratio",
            "min_supply_ratio",
            "supply_reliability",
            "mean_nexus_index",
            "min_nexus_index",
            "mean_water_security",
            "mean_energy_security",
            "mean_food_security",
            "env_flow_met_share",
        ):
            assert 0.0 - TOL <= r[col] <= 1.0 + TOL, col
        assert r["min_supply_ratio"] <= r["mean_supply_ratio"] + TOL
        assert r["min_nexus_index"] <= r["mean_nexus_index"] + TOL
        assert 0 <= r["years_env_flow_unmet"] <= r["years"]
        assert r["env_flow_met_share"] == pytest.approx(1.0 - r["years_env_flow_unmet"] / r["years"]) or r["riparian"] == BASIN_KEY
        for col in ("total_hydropower_gwh", "mean_energy_deficit_gwh", "mean_emissions_t", "food_self_sufficiency",
                    "final_storage_mm3", "mean_total_deficit_mm3", "mean_water_stress_sdg642"):
            assert r[col] >= 0.0, col
        if r["riparian"] == BASIN_KEY:
            assert 0.0 <= r["equity_index"] <= 1.0 + TOL
            assert r["mass_balance_error_mm3"] < 1e-6
            assert r["natural_flow_mm3"] > 0 and r["outflow_to_sea_mm3"] >= 0


def test_comparison_table_matches_summary(all_rows, all_results):
    for name, res in all_results.items():
        summary = res.summary()
        for rip in RIPARIANS + [BASIN_KEY]:
            row = _row(all_rows, name, rip)
            s = summary[rip]
            assert row["mean_supply_ratio"] == s["supply_ratio"]
            assert row["min_supply_ratio"] == s["min_supply_ratio"]
            assert row["supply_reliability"] == s["supply_reliability"]
            assert row["mean_nexus_index"] == s["nexus_index"]
            assert row["mean_water_security"] == s["water_security"]
            assert row["mean_energy_security"] == s["energy_security"]
            assert row["mean_food_security"] == s["food_security"]
            assert row["years_env_flow_unmet"] == s["years_env_flow_unmet"]
            assert row["total_hydropower_gwh"] == s["total_hydropower_gwh"]
            assert row["food_self_sufficiency"] == s["food_self_sufficiency"]
            assert row["final_storage_mm3"] == s["final_storage_mm3"]
            assert row["mean_emissions_t"] == s["emissions_t"]
        basin_row = _row(all_rows, name, BASIN_KEY)
        assert basin_row["equity_index"] == summary[BASIN_KEY]["equity_index"]
        # basin aggregates are consistent with the riparian rows
        rip_rows = [_row(all_rows, name, rip) for rip in RIPARIANS]
        assert basin_row["final_storage_mm3"] == pytest.approx(sum(r["final_storage_mm3"] for r in rip_rows))
        assert basin_row["total_hydropower_gwh"] == pytest.approx(sum(r["total_hydropower_gwh"] for r in rip_rows))
        assert basin_row["years_env_flow_unmet"] >= max(r["years_env_flow_unmet"] for r in rip_rows)
        # the mean of yearly ratios of a riparian equals the mean of its supply_ratio series
        for r in rip_rows:
            series = res.series(r["riparian"], "supply_ratio")
            assert r["mean_supply_ratio"] == pytest.approx(sum(series) / len(series))
            assert r["min_supply_ratio"] == pytest.approx(min(series))


def test_comparison_table_accepts_sequence_and_single_result(short_results):
    rows_map = S.comparison_table(short_results)
    rows_seq = S.comparison_table(list(short_results.values()))
    assert rows_map == rows_seq
    single = S.comparison_table(short_results["baseline"])
    assert len(single) == 4 and single[-1]["riparian"] == BASIN_KEY
    assert single == rows_map[:4]


@pytest.mark.parametrize(
    "bad",
    [
        {},
        [],
        {"x": 1},
        [None],
        "baseline",
        42,
    ],
)
def test_comparison_table_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        S.comparison_table(bad)


def test_comparison_table_rejects_duplicate_names(short_results):
    with pytest.raises(ValueError, match="duplicate"):
        S.comparison_table([short_results["baseline"], short_results["baseline"]])


# ---------------------------------------------------------------------------
# scenario narrative properties
# ---------------------------------------------------------------------------
def test_combined_adaptation_beats_combined_stress_for_delta_over_ten_years(ten_year):
    rows = S.comparison_table(ten_year)
    assert len(rows) == 8
    stress = _row(rows, "combined_stress", "Delta")
    adapt = _row(rows, "combined_adaptation", "Delta")
    assert adapt["mean_supply_ratio"] > stress["mean_supply_ratio"] + 0.05
    assert adapt["min_supply_ratio"] > stress["min_supply_ratio"]
    assert adapt["mean_nexus_index"] > stress["mean_nexus_index"]
    assert adapt["mean_water_security"] > stress["mean_water_security"]
    assert adapt["mean_food_security"] > stress["mean_food_security"]
    assert adapt["mean_energy_security"] > stress["mean_energy_security"]
    assert adapt["mean_total_deficit_mm3"] < stress["mean_total_deficit_mm3"]
    # nobody is worse off and the basin as a whole gains
    for rip in RIPARIANS:
        assert _row(rows, "combined_adaptation", rip)["mean_supply_ratio"] >= _row(rows, "combined_stress", rip)["mean_supply_ratio"] - TOL
    basin_adapt = _row(rows, "combined_adaptation", BASIN_KEY)
    basin_stress = _row(rows, "combined_stress", BASIN_KEY)
    assert basin_adapt["mean_supply_ratio"] > basin_stress["mean_supply_ratio"]
    assert basin_adapt["mean_nexus_index"] > basin_stress["mean_nexus_index"]
    assert basin_adapt["equity_index"] >= basin_stress["equity_index"] - TOL
    # the same holds via the result summaries directly (independent of the table)
    assert ten_year["combined_adaptation"].summary()["Delta"]["supply_ratio"] > ten_year["combined_stress"].summary()["Delta"]["supply_ratio"]


def test_talmud_variant_of_adaptation_no_longer_caps_delta_below_treaty():
    """The bankruptcy estate is consumptive and counts carried storage (nexus review fix), so the
    Talmud rule - the contract default of the adaptation package (ARCHITECTURE.md section 9) -
    never rations Delta below its treaty volume on the example basin."""
    res = S.run_scenarios(example_basin(), [S.combined_stress(years=10), S.combined_adaptation(years=10, rule="talmud")])
    stress = res["combined_stress"].summary()["Delta"]
    talmud = res["combined_adaptation"].summary()["Delta"]
    assert min(res["combined_adaptation"].series("Delta", "entitlement_mm3")) > 16000.0
    assert talmud["supply_ratio"] >= stress["supply_ratio"] - TOL
    assert talmud["supply_ratio"] == pytest.approx(1.0)
    # the explicit rule="talmud" is exactly the default composition
    default = S.run_scenarios(example_basin(), [S.combined_adaptation(years=10)])["combined_adaptation"]
    assert default.allocation_rule == "talmud"
    assert default.to_records() == res["combined_adaptation"].to_records()


def test_combined_adaptation_default_rule_follows_contract():
    """Review fix: ARCHITECTURE.md section 9 defines ``combined_adaptation`` as climate + growth +
    efficiency + talmud, so the factory, the registry and every run report the Talmud rule by default;
    the fixed-treaty variant stays one keyword away."""
    assert S.DEFAULT_ADAPTATION_RULE == "talmud"
    for sc in (S.combined_adaptation(), S.get_scenario("combined_adaptation", years=3), S.all_scenarios(years=3)[-1]):
        assert sc.name == "combined_adaptation"
        assert sc.allocation_rule == "talmud" and sc.cooperation is True
        assert "'talmud'" in sc.description
    assert S.combined_adaptation(years=3).allocation_rule == S.cooperative(years=3).allocation_rule
    # the run and the comparison rows report the rule, and the caps are claims-based, not the treaty volumes
    basin = example_basin()
    res = S.run_scenarios(basin, [S.combined_adaptation(years=3)])["combined_adaptation"]
    assert res.allocation_rule == "talmud"
    assert all(r["allocation_rule"] == "talmud" for r in S.comparison_table(res))
    treaty_volumes = {r.name: r.treaty_allocation_mm3 for r in basin.riparians}
    for rip in RIPARIANS:
        ents = res.series(rip, "entitlement_mm3")
        assert all(math.isfinite(v) and v >= 0.0 for v in ents)
        assert all(v != treaty_volumes[rip] for v in ents)
    # the Talmud package is a genuine adaptation on the example basin: nobody is short
    assert res.summary()[BASIN_KEY]["supply_ratio"] == pytest.approx(1.0)
    assert res.summary()[BASIN_KEY]["equity_index"] == pytest.approx(1.0)
    assert all(bal.mass_balance_error() < 1e-6 for bal in res.balances)
    # the fixed-treaty variant differs from the default only by the rule and keeps the treaty caps
    treaty = S.combined_adaptation(years=3, rule="treaty")
    assert treaty.allocation_rule == "treaty"
    assert _fields_except(treaty, "allocation_rule", "description") == _fields_except(S.combined_adaptation(years=3), "allocation_rule", "description")
    treaty_res = S.run_scenarios(basin, [treaty])["combined_adaptation"]
    assert treaty_res.allocation_rule == "treaty"
    for rip in RIPARIANS:
        assert all(v == treaty_volumes[rip] for v in treaty_res.series(rip, "entitlement_mm3"))
    assert treaty_res.summary()["Delta"]["supply_ratio"] <= res.summary()["Delta"]["supply_ratio"] + TOL
    # the inputs are untouched
    assert dataclasses.asdict(basin) == dataclasses.asdict(example_basin())


def test_single_driver_scenarios_move_in_the_expected_direction(all_rows):
    base_delta = _row(all_rows, "baseline", "Delta")
    base_basin = _row(all_rows, "baseline", BASIN_KEY)
    # drier climate -> less hydropower, lower nexus index
    assert _row(all_rows, "climate_change", BASIN_KEY)["total_hydropower_gwh"] < base_basin["total_hydropower_gwh"]
    assert _row(all_rows, "climate_change", BASIN_KEY)["mean_nexus_index"] < base_basin["mean_nexus_index"]
    # growth -> larger deficits downstream
    assert _row(all_rows, "growth", "Delta")["mean_supply_ratio"] < base_delta["mean_supply_ratio"]
    assert _row(all_rows, "growth", "Delta")["mean_total_deficit_mm3"] > base_delta["mean_total_deficit_mm3"]
    # efficiency -> better supply and nexus index
    assert _row(all_rows, "efficiency", "Delta")["mean_supply_ratio"] >= base_delta["mean_supply_ratio"] - TOL
    assert _row(all_rows, "efficiency", BASIN_KEY)["mean_nexus_index"] > base_basin["mean_nexus_index"]
    assert _row(all_rows, "efficiency", BASIN_KEY)["mean_energy_security"] > base_basin["mean_energy_security"]
    # unilateral: no caps at all -> nobody short in a normal year
    assert _row(all_rows, "unilateral", BASIN_KEY)["mean_supply_ratio"] >= base_basin["mean_supply_ratio"] - TOL
    # cooperative (Talmud on the consumptive estate) does not ration in a normal year: the caps sit
    # at the river demands, above the fixed treaty volumes, so supply and equity are no worse
    assert _row(all_rows, "cooperative", BASIN_KEY)["mean_supply_ratio"] >= base_basin["mean_supply_ratio"] - TOL
    assert _row(all_rows, "cooperative", BASIN_KEY)["equity_index"] >= base_basin["equity_index"] - TOL
    # combined stress is no better than either stressor alone
    cs = _row(all_rows, "combined_stress", BASIN_KEY)
    assert cs["mean_nexus_index"] <= min(_row(all_rows, "climate_change", BASIN_KEY)["mean_nexus_index"],
                                         _row(all_rows, "growth", BASIN_KEY)["mean_nexus_index"]) + TOL
    assert cs["mean_supply_ratio"] <= _row(all_rows, "growth", BASIN_KEY)["mean_supply_ratio"] + TOL


# ---------------------------------------------------------------------------
# scenario_differences and rank_scenarios
# ---------------------------------------------------------------------------
def test_scenario_differences_against_baseline(all_rows):
    diff = S.scenario_differences(all_rows)
    assert len(diff) == (len(NAMES) - 1) * 4
    assert all(r["scenario"] != "baseline" for r in diff)
    assert [r["scenario"] for r in diff[:4]] == ["climate_change"] * 4
    growth_delta = _row(diff, "growth", "Delta")
    assert growth_delta["mean_supply_ratio"] < 0
    assert growth_delta["years"] == 4 and growth_delta["allocation_rule"] == "treaty"
    climate_delta = _row(diff, "climate_change", "Delta")
    assert climate_delta["mean_supply_ratio"] == pytest.approx(0.0, abs=1e-9)
    assert _row(diff, "efficiency", "Delta")["mean_nexus_index"] > 0
    assert _row(diff, "climate_change", "Delta")["equity_index"] is None
    assert isinstance(_row(diff, "climate_change", BASIN_KEY)["equity_index"], float)
    # explicit reference and column subset
    d2 = S.scenario_differences(all_rows, reference="combined_stress", columns=["mean_supply_ratio"])
    assert _row(d2, "combined_adaptation", "Delta")["mean_supply_ratio"] > 0
    assert _row(d2, "combined_adaptation", "Delta")["mean_nexus_index"] == _row(all_rows, "combined_adaptation", "Delta")["mean_nexus_index"]
    with pytest.raises(ValueError, match="reference"):
        S.scenario_differences(all_rows, reference="nope")
    with pytest.raises(ValueError):
        S.scenario_differences(all_rows, columns=["allocation_rule"])
    with pytest.raises(ValueError):
        S.scenario_differences([])
    with pytest.raises(ValueError):
        S.scenario_differences(all_rows + [{"scenario": "x", "riparian": "Atlantis", "mean_supply_ratio": 1.0}])
    # input rows untouched
    assert all(set(r) == set(S.TABLE_COLUMNS) for r in all_rows)


def test_rank_scenarios(all_rows):
    ranked = S.rank_scenarios(all_rows, "mean_supply_ratio", "Delta")
    assert [name for name, _ in ranked][0] == "unilateral"
    assert ranked[0][1] == pytest.approx(1.0)
    values = [v for _, v in ranked]
    assert values == sorted(values, reverse=True)
    assert {name for name, _ in ranked} == set(NAMES)
    ascending = S.rank_scenarios(all_rows, "mean_supply_ratio", "Delta", higher_is_better=False)
    assert [v for _, v in ascending] == sorted(values)
    basin = S.rank_scenarios(all_rows)  # default: basin mean nexus index
    assert len(basin) == len(NAMES) and basin[0][1] >= basin[-1][1]
    with pytest.raises(ValueError):
        S.rank_scenarios(all_rows, "mean_supply_ratio", "Atlantis")
    with pytest.raises(ValueError):
        S.rank_scenarios(all_rows, "allocation_rule")
    with pytest.raises(ValueError):
        S.rank_scenarios(all_rows, "no_such_column")
    with pytest.raises(ValueError):
        S.rank_scenarios(all_rows, "equity_index", "Delta")  # None for riparian rows
    with pytest.raises(ValueError):
        S.rank_scenarios(all_rows, "")
    assert S.rank_scenarios([{"scenario": "a", "riparian": "BASIN", "x": 1}, {"scenario": "b", "riparian": "BASIN", "x": 1}], "x") == [("a", 1.0), ("b", 1.0)]


# ---------------------------------------------------------------------------
# format_table and CSV
# ---------------------------------------------------------------------------
def test_format_table_contains_scenario_and_riparian_names(all_rows):
    text = S.format_table(all_rows)
    lines = text.split("\n")
    assert len(lines) == len(all_rows) + 2
    assert set(lines[1]) == {"-"} and len(lines[1]) == len(lines[0])
    for name in NAMES:
        assert name in text
    for rip in RIPARIANS + [BASIN_KEY]:
        assert rip in text
    for col in S.TABLE_COLUMNS:
        assert col in lines[0]
    assert all(len(line) <= len(lines[0]) for line in lines)
    assert not text.endswith("\n")


def test_format_table_column_selection_and_alignment(all_rows):
    cols = ["scenario", "riparian", "mean_supply_ratio", "years_env_flow_unmet", "equity_index"]
    text = S.format_table(all_rows, columns=cols)
    lines = text.split("\n")
    assert lines[0].split() == cols
    assert "mean_nexus_index" not in text
    # text columns left-aligned, numeric right-aligned; None renders as "-"
    first = lines[2]
    assert first.startswith("baseline")
    assert first.rstrip().endswith("-")  # equity_index is None for a riparian row
    basin_line = lines[5]
    assert basin_line.split()[1] == BASIN_KEY
    assert basin_line.rstrip().endswith(("0.996", "0.997", "0.995", "1.000", "0.998", "0.999"))
    width = len(lines[0])
    for line in lines[2:]:
        assert len(line) <= width
    # a column present in only some rows prints "-" where it is missing
    mixed = [{"a": "x", "b": 1.0}, {"a": "y"}]
    out = S.format_table(mixed)
    assert out.split("\n")[3].split() == ["y", "-"]


def test_format_table_cell_rendering():
    rows = [
        {"label": "big", "value": 1234567.0, "flag": True, "count": 7, "ratio": 0.5},
        {"label": "mid", "value": 123.4, "flag": False, "count": None, "ratio": 12.34},
        {"label": "inf", "value": float("inf"), "flag": None, "count": -3, "ratio": float("nan")},
        {"label": "neg", "value": -0.25, "flag": True, "count": 0, "ratio": -1234.5},
    ]
    text = S.format_table(rows)
    lines = text.split("\n")
    assert lines[0].split() == ["label", "value", "flag", "count", "ratio"]
    assert lines[2].split() == ["big", "1.235e+06", "yes", "7", "0.500"]
    assert lines[3].split() == ["mid", "123", "no", "-", "12.3"]
    assert lines[4].split() == ["inf", "inf", "-", "-3", "nan"]
    assert lines[5].split() == ["neg", "-0.250", "yes", "0", "-1234"]
    # numeric columns are right-aligned: the values end at the same position as the header
    header_end = lines[0].index("value") + len("value")
    for line in lines[2:]:
        label, value = line.split()[0], line.split()[1]
        start = line.index(label) + len(label)  # search after the label column ("inf" label vs "inf" value)
        assert line.index(value, start) + len(value) == header_end


@pytest.mark.parametrize(
    "rows, columns",
    [
        ([], None),
        ([{"a": 1}], []),
        ([{"a": 1}], ["b"]),
        ([{"a": 1}, "x"], None),
        ("rows", None),
        ({"a": 1}, None),
        ([{"a": 1}], "a"),
    ],
)
def test_format_table_rejects_bad_input(rows, columns):
    with pytest.raises(ValueError):
        S.format_table(rows, columns)


def test_write_table_csv_round_trip(tmp_path, all_rows):
    path = tmp_path / "comparison.csv"
    written = S.write_table_csv(all_rows, path)
    assert written == str(path)
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == list(S.TABLE_COLUMNS)
        read = list(reader)
    assert len(read) == len(all_rows)
    assert read[0]["scenario"] == "baseline" and read[0]["riparian"] == "Highland"
    assert read[0]["equity_index"] == ""  # None -> empty cell
    assert float(read[3]["equity_index"]) == pytest.approx(all_rows[3]["equity_index"])
    assert float(read[0]["mean_supply_ratio"]) == pytest.approx(all_rows[0]["mean_supply_ratio"])
    subset = S.write_table_csv(all_rows, tmp_path / "subset.csv", columns=["scenario", "mean_nexus_index"])
    with open(subset, newline="", encoding="utf-8") as fh:
        assert next(csv.reader(fh)) == ["scenario", "mean_nexus_index"]
    with pytest.raises(ValueError):
        S.write_table_csv([], tmp_path / "empty.csv")
    with pytest.raises(ValueError):
        S.write_table_csv(all_rows, tmp_path / "x.csv", columns=[])


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------
def test_single_riparian_basin_and_one_year_horizon():
    basin = example_basin()
    solo = Basin(name="solo", riparians=[basin.riparians[2].copy(position=0)], headwater_inflow_mm3=20000.0)
    res = S.run_scenarios(solo, S.all_scenarios(years=1))
    rows = S.comparison_table(res)
    assert len(rows) == len(NAMES) * 2
    for r in rows:
        assert r["years"] == 1
        assert r["mean_supply_ratio"] == r["min_supply_ratio"]
    basin_rows = [r for r in rows if r["riparian"] == BASIN_KEY]
    assert all(r["equity_index"] == pytest.approx(1.0) for r in basin_rows)  # one riparian: perfect equity
    text = S.format_table(rows)
    assert "Delta" in text and "combined_adaptation" in text


def test_zero_flow_scenario_runs_and_reports():
    dry = S.climate_change(years=2, flow_change_pct=-100, droughts=[0], drought_severity=1.0)
    assert dry.flow_factor(0) == 0.0 and dry.flow_factor(1) == 0.0
    res = S.run_scenarios(example_basin(), [dry])["climate_change"]
    rows = S.comparison_table(res)
    assert all(0.0 <= r["mean_supply_ratio"] <= 1.0 for r in rows)
    assert _row(rows, "climate_change", BASIN_KEY)["natural_flow_mm3"] == 0.0
    assert all(bal.mass_balance_error() < 1e-6 for bal in res.balances)
