"""Tests for :mod:`wefnexus.sustainability` - composite WEF indices and reports."""
import copy
import math
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pytest

from wefnexus.allocation import gini
from wefnexus.models import Scenario, Sector
from wefnexus.water import (
    falkenmark_category,
    groundwater_stress,
    per_capita_water,
    route_basin,
    sdg_642_water_stress,
)
from wefnexus.sustainability import (
    COLUMN_LABELS,
    EMISSION_INTENSITY_RANGE,
    ENERGY_COMPONENTS,
    FOOD_COMPONENTS,
    GEOMETRIC_FLOOR,
    GROUNDWATER_STRESS_RANGE,
    INDEX_KEYS,
    NEXUS_PILLARS,
    RIPARIAN_YEAR_FIELDS,
    SDG_KEYS,
    SUMMARY_COLUMNS,
    WATER_COMPONENTS,
    WATER_STRESS_RANGE,
    SustainabilityReport,
    assess,
    assess_records,
    energy_security_components,
    energy_security_index,
    equity_index,
    food_security_components,
    food_security_index,
    normalise,
    sdg_indicators,
    water_security_components,
    water_security_index,
    water_use_efficiency_usd_per_m3,
    weighted_geometric_mean,
    wef_nexus_index,
)
import wefnexus.sustainability as sustainability_module

MUN, AGR = Sector.MUNICIPAL, Sector.AGRICULTURAL
INF = math.inf
BAD_NUMBERS = [None, "x", float("nan"), True, [1.0], {"a": 1}]


# ---------------------------------------------------------------------------
# helpers: a RiparianYear / NexusResult stand-in that follows ARCHITECTURE.md §7
# ---------------------------------------------------------------------------
@dataclass
class FakeRiparianYear:
    name: str
    year: int
    population: float
    gdp_usd: float
    inflow_mm3: float
    upstream_inflow_mm3: float
    outflow_mm3: float
    storage_end_mm3: float
    entitlement_mm3: float
    demands: Dict[Sector, float]
    withdrawals: Dict[Sector, float]
    deficits: Dict[Sector, float]
    environmental_flow_mm3: float
    env_flow_met: bool
    per_capita_water_m3: float
    falkenmark: str
    water_stress_sdg642: float
    groundwater_stress: float
    hydropower_gwh: float
    thermal_gwh: float
    other_renewable_gwh: float
    energy_supply_gwh: float
    energy_demand_gwh: float
    energy_for_water_gwh: float
    energy_deficit_gwh: float
    emissions_t: float
    food_production_t: float
    food_kcal: float
    food_demand_kcal: float
    food_self_sufficiency: float
    et_ratio: float
    irrigation_requirement_mm3: float
    crop_value_usd: float
    water_value_usd: float
    water_security: float
    energy_security: float
    food_security: float
    nexus_index: float


@dataclass
class FakeNexusResult:
    basin_name: str
    scenario: Any
    years: List[int]
    records: List[Any]
    balances: List[Any]


def make_record(name: str, year: int, **kw: Any) -> FakeRiparianYear:
    """A record with simple defaults; override any field by keyword."""
    base: Dict[str, Any] = dict(
        name=name,
        year=year,
        population=1e6,
        gdp_usd=2e9,
        inflow_mm3=1000.0,
        upstream_inflow_mm3=400.0,
        outflow_mm3=800.0,
        storage_end_mm3=100.0,
        entitlement_mm3=INF,
        demands={MUN: 100.0},
        withdrawals={MUN: 100.0},
        deficits={MUN: 0.0},
        environmental_flow_mm3=300.0,
        env_flow_met=True,
        per_capita_water_m3=2000.0,
        falkenmark="no_stress",
        water_stress_sdg642=20.0,
        groundwater_stress=0.4,
        hydropower_gwh=40.0,
        thermal_gwh=20.0,
        other_renewable_gwh=20.0,
        energy_supply_gwh=80.0,
        energy_demand_gwh=100.0,
        energy_for_water_gwh=5.0,
        energy_deficit_gwh=20.0,
        emissions_t=800.0,
        food_production_t=1e5,
        food_kcal=1000.0,
        food_demand_kcal=1000.0,
        food_self_sufficiency=1.0,
        et_ratio=1.0,
        irrigation_requirement_mm3=50.0,
        crop_value_usd=1e7,
        water_value_usd=1.5e8,
        water_security=0.8,
        energy_security=0.5,
        food_security=1.0,
        nexus_index=0.6,
    )
    base.update(kw)
    return FakeRiparianYear(**base)


def tiny_records() -> List[FakeRiparianYear]:
    """Two riparians x three years with hand-computable aggregates."""
    a = [
        make_record("A", 2025, deficits={MUN: 0.0}, withdrawals={MUN: 100.0}, nexus_index=0.6,
                    water_security=0.9, water_stress_sdg642=20.0, per_capita_water_m3=2000.0,
                    food_kcal=900.0, env_flow_met=True),
        make_record("A", 2026, deficits={MUN: 50.0}, withdrawals={MUN: 50.0}, nexus_index=0.4,
                    water_security=0.7, water_stress_sdg642=40.0, per_capita_water_m3=1500.0,
                    food_kcal=900.0, env_flow_met=False),
        make_record("A", 2027, deficits={MUN: 0.0}, withdrawals={MUN: 100.0}, nexus_index=0.8,
                    water_security=0.8, water_stress_sdg642=60.0, per_capita_water_m3=1000.0,
                    food_kcal=1200.0, env_flow_met=True),
    ]
    b = [
        make_record("B", y, population=3e6, demands={AGR: 200.0}, withdrawals={AGR: 200.0},
                    deficits={AGR: 0.0}, nexus_index=0.9, water_security=0.95, energy_security=0.9,
                    food_security=0.85, entitlement_mm3=250.0, per_capita_water_m3=800.0)
        for y in (2025, 2026, 2027)
    ]
    return a + b


@pytest.fixture
def tiny() -> List[FakeRiparianYear]:
    return tiny_records()


@pytest.fixture
def azura(basin):
    """A NexusResult stand-in built from real routing of the example basin."""
    scen = Scenario(name="smoke", years=4)
    years = [scen.year(i) for i in range(scen.years)]
    records, balances, storages = [], [], None
    for i in range(scen.years):
        ff = scen.flow_factor(i) * (0.6 if i == 2 else 1.0)  # a drought in year 3
        bal = route_basin(basin, flow_factor=ff, storages=storages)
        storages = bal.storages_end()
        balances.append(bal)
        for r in basin:
            reach = bal.reach(r.name)
            renewable = reach.inflow_mm3 + r.groundwater_recharge_mm3
            pc = per_capita_water(renewable, r.population)
            stress = sdg_642_water_stress(
                reach.total_withdrawal() + r.groundwater_abstraction_mm3, renewable, reach.environmental_flow_mm3
            )
            gw = groundwater_stress(r.groundwater_abstraction_mm3, r.groundwater_recharge_mm3)
            hydro = 1000.0 * (1 + 0.1 * i)
            thermal = r.energy.thermal_capacity_mw * 0.5 * 8.76
            other = r.energy.demand_gwh * r.energy.renewable_share
            supply, demand = hydro + thermal + other, r.energy.demand_gwh
            em = thermal * r.energy.grid_emission_factor_t_per_gwh
            kd = r.food_demand_kcal()
            kcal = 0.8 * kd * reach.supply_ratio()
            ws = water_security_index(reach.supply_ratio(), stress, 1.0 if reach.env_flow_met else 0.0, gw)
            es = energy_security_index(min(supply / demand, 1.0), (hydro + other) / supply, em / supply)
            fs = food_security_index(kcal / kd, reach.supply_ratio())
            records.append(
                FakeRiparianYear(
                    name=r.name, year=years[i], population=r.population, gdp_usd=r.gdp_usd,
                    inflow_mm3=reach.inflow_mm3, upstream_inflow_mm3=reach.upstream_inflow_mm3,
                    outflow_mm3=reach.outflow_mm3, storage_end_mm3=reach.storage_end_mm3,
                    entitlement_mm3=reach.entitlement_mm3, demands=dict(reach.demands),
                    withdrawals=dict(reach.withdrawals), deficits=dict(reach.deficits),
                    environmental_flow_mm3=reach.environmental_flow_mm3, env_flow_met=reach.env_flow_met,
                    per_capita_water_m3=pc, falkenmark=falkenmark_category(pc), water_stress_sdg642=stress,
                    groundwater_stress=gw, hydropower_gwh=hydro, thermal_gwh=thermal, other_renewable_gwh=other,
                    energy_supply_gwh=supply, energy_demand_gwh=demand, energy_for_water_gwh=50.0,
                    energy_deficit_gwh=max(demand - supply, 0.0), emissions_t=em, food_production_t=1e6,
                    food_kcal=kcal, food_demand_kcal=kd, food_self_sufficiency=kcal / kd,
                    et_ratio=reach.supply_ratio(), irrigation_requirement_mm3=r.demand.agricultural,
                    crop_value_usd=1e8,
                    water_value_usd=sum(v * r.demand.value_usd_per_m3[s] * 1e6 for s, v in reach.withdrawals.items()),
                    water_security=ws, energy_security=es, food_security=fs,
                    nexus_index=wef_nexus_index(ws, es, fs),
                )
            )
    return FakeNexusResult(basin.name, scen, years, records, balances)


def is_non_decreasing(seq):
    return all(b >= a - 1e-12 for a, b in zip(seq, seq[1:]))


def is_non_increasing(seq):
    return all(b <= a + 1e-12 for a, b in zip(seq, seq[1:]))


# ---------------------------------------------------------------------------
# module hygiene
# ---------------------------------------------------------------------------
def test_import_does_not_load_nexus_or_scipy():
    assert "wefnexus.sustainability" in sys.modules
    # ``sys.modules`` is shared by the whole pytest process, and sibling test
    # modules (test_nexus, test_cli, ...) legitimately import ``wefnexus.nexus``
    # before this test runs.  The contract (ARCHITECTURE.md, "Testing & quality
    # bar") is about *import time*: a fresh import of ``wefnexus.sustainability``
    # must not pull in ``wefnexus.nexus`` or SciPy.  Check it in a clean
    # interpreter so the result does not depend on test ordering.
    code = (
        "import sys; import wefnexus.sustainability; "
        "assert 'wefnexus.nexus' not in sys.modules, 'nexus imported at import time'; "
        "assert 'scipy' not in sys.modules, 'scipy imported at import time'; "
        "assert 'pandas' not in sys.modules, 'pandas imported at import time'; "
        "assert 'matplotlib' not in sys.modules, 'matplotlib imported at import time'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
    with open(sustainability_module.__file__, encoding="utf-8") as fh:
        src = fh.read()
    assert "import scipy" not in src and "from scipy" not in src
    assert "import pandas" in src  # only lazily, inside to_dataframe
    top_level_imports = [ln for ln in src.splitlines() if ln.startswith(("import ", "from "))]
    assert not any("pandas" in ln or "matplotlib" in ln or "wefnexus.nexus" in ln for ln in top_level_imports)


def test_all_names_exist_and_constants():
    for name in sustainability_module.__all__:
        assert hasattr(sustainability_module, name), name
    assert GEOMETRIC_FLOOR == 1e-6
    assert WATER_STRESS_RANGE == (100.0, 0.0)
    assert GROUNDWATER_STRESS_RANGE == (1.5, 0.5)
    assert EMISSION_INTENSITY_RANGE[1] == 0.0 and EMISSION_INTENSITY_RANGE[0] > 0
    assert WATER_COMPONENTS == ("supply", "stress", "environment", "groundwater")
    assert ENERGY_COMPONENTS == ("supply", "renewable", "emissions")
    assert FOOD_COMPONENTS == ("self_sufficiency", "et_ratio")
    assert NEXUS_PILLARS == ("water", "energy", "food")
    assert set(SDG_KEYS.values()) == {
        "6.4.1_water_use_efficiency_usd_per_m3",
        "6.4.2_water_stress_pct",
        "6.5.2_transboundary_cooperation",
        "7.2_renewable_share",
        "2.1_food_self_sufficiency",
    }
    assert set(INDEX_KEYS) <= set(RIPARIAN_YEAR_FIELDS)
    assert all(c in COLUMN_LABELS or isinstance(c, str) for c in SUMMARY_COLUMNS)


# ---------------------------------------------------------------------------
# normalise
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value, lo, hi, hib, expected",
    [
        (50.0, 0.0, 100.0, True, 0.5),
        (0.0, 0.0, 100.0, True, 0.0),
        (100.0, 0.0, 100.0, True, 1.0),
        (150.0, 0.0, 100.0, True, 1.0),  # clipped above
        (-5.0, 0.0, 100.0, True, 0.0),  # clipped below
        (25.0, 100.0, 0.0, True, 0.75),  # reversed scale: water stress 25 % -> 0.75
        (0.0, 100.0, 0.0, True, 1.0),
        (100.0, 100.0, 0.0, True, 0.0),
        (200.0, 100.0, 0.0, True, 0.0),
        (25.0, 0.0, 100.0, False, 0.75),  # higher_is_better=False flips
        (1.0, 1.5, 0.5, True, 0.5),  # groundwater stress 1.0 -> 0.5
        (0.0, 1.5, 0.5, True, 1.0),
        (2.0, 1.5, 0.5, True, 0.0),
        (INF, 100.0, 0.0, True, 0.0),  # infinite stress -> worst
        (INF, 0.0, 100.0, True, 1.0),
        (-INF, 0.0, 100.0, True, 0.0),
        (-INF, 100.0, 0.0, True, 1.0),
        (INF, 0.0, 100.0, False, 0.0),
        (7, 2, 12, True, 0.5),  # ints accepted
    ],
)
def test_normalise_closed_form(value, lo, hi, hib, expected):
    out = normalise(value, lo, hi, hib)
    assert isinstance(out, float)
    assert out == pytest.approx(expected)


def test_normalise_bounds_and_monotone():
    xs = np.linspace(-50, 150, 41)
    up = [normalise(x, 0.0, 100.0) for x in xs]
    down = [normalise(x, 100.0, 0.0) for x in xs]
    assert all(0.0 <= v <= 1.0 for v in up + down)
    assert is_non_decreasing(up) and is_non_increasing(down)
    # the two ways of writing "lower is better" agree
    for x in xs:
        assert normalise(x, 0.0, 100.0, False) == pytest.approx(normalise(x, 100.0, 0.0))
        assert normalise(x, 0.0, 100.0) + normalise(x, 0.0, 100.0, False) == pytest.approx(1.0)


@pytest.mark.parametrize("bad", BAD_NUMBERS)
def test_normalise_rejects_bad_value(bad):
    with pytest.raises(ValueError):
        normalise(bad, 0.0, 1.0)


def test_normalise_rejects_bad_bounds():
    with pytest.raises(ValueError, match="differ"):
        normalise(0.5, 1.0, 1.0)
    with pytest.raises(ValueError):
        normalise(0.5, INF, 1.0)
    with pytest.raises(ValueError):
        normalise(0.5, 0.0, float("nan"))
    with pytest.raises(ValueError):
        normalise(0.5, "a", 1.0)


# ---------------------------------------------------------------------------
# weighted_geometric_mean
# ---------------------------------------------------------------------------
def test_geometric_mean_closed_form():
    assert weighted_geometric_mean({"a": 0.25, "b": 1.0}) == pytest.approx(0.5)
    assert weighted_geometric_mean({"a": 0.5, "b": 0.5, "c": 0.5}) == pytest.approx(0.5)
    assert weighted_geometric_mean({"a": 1.0, "b": 1.0}) == 1.0
    assert weighted_geometric_mean({"a": 0.5}) == pytest.approx(0.5)
    # weights: (0.25^1 * 1^3)^(1/4) = 0.7071
    assert weighted_geometric_mean({"a": 0.25, "b": 1.0}, {"a": 1.0, "b": 3.0}) == pytest.approx(0.25 ** 0.25)
    assert weighted_geometric_mean({"a": 0.25, "b": 1.0}, {"a": 1.0, "b": 0.0}) == pytest.approx(0.25)
    assert weighted_geometric_mean({"a": 0.25, "b": 1.0}, {"a": 0.0, "b": 5.0}) == pytest.approx(1.0)
    # weights are scale-invariant
    assert weighted_geometric_mean({"a": 0.3, "b": 0.9}, {"a": 2.0, "b": 6.0}) == pytest.approx(
        weighted_geometric_mean({"a": 0.3, "b": 0.9}, {"a": 1.0, "b": 3.0})
    )


def test_geometric_mean_clipping():
    # above 1 -> 1 ; inf -> 1
    assert weighted_geometric_mean({"a": 5.0, "b": 2.0}) == 1.0
    assert weighted_geometric_mean({"a": INF, "b": 1.0}) == 1.0
    # zero pillar -> floor, not log(0)
    assert weighted_geometric_mean({"a": 0.0, "b": 1.0, "c": 1.0}) == pytest.approx(GEOMETRIC_FLOOR ** (1 / 3))
    assert weighted_geometric_mean({"a": 0.0}) == pytest.approx(GEOMETRIC_FLOOR)
    # negative values clip to the floor as the spec says ("clipped to [1e-6, 1]")
    assert weighted_geometric_mean({"a": -3.0}) == pytest.approx(GEOMETRIC_FLOOR)
    assert weighted_geometric_mean({"a": -INF, "b": 1.0}) == pytest.approx(GEOMETRIC_FLOOR ** 0.5)


def test_geometric_mean_punishes_weak_pillar_and_am_gm():
    rng = np.random.default_rng(1)
    for _ in range(200):
        n = int(rng.integers(1, 6))
        vals = {f"k{i}": float(v) for i, v in enumerate(rng.uniform(0.0, 1.0, n))}
        w = {k: float(x) for k, x in zip(vals, rng.uniform(0.0, 2.0, n))}
        if sum(w.values()) == 0:
            w = None
        g = weighted_geometric_mean(vals, w)
        assert GEOMETRIC_FLOOR <= g <= 1.0
        clipped = {k: min(max(v, GEOMETRIC_FLOOR), 1.0) for k, v in vals.items()}
        if w is None:
            am = sum(clipped.values()) / n
        else:
            am = sum(w[k] * clipped[k] for k in clipped) / sum(w.values())
        assert g <= am + 1e-12
        assert g >= min(clipped.values()) - 1e-12 and g <= max(clipped.values()) + 1e-12
    # one very low pillar dominates
    assert weighted_geometric_mean({"a": 0.01, "b": 1.0, "c": 1.0}) < 0.25
    assert weighted_geometric_mean({"a": 0.01, "b": 1.0, "c": 1.0}) < (0.01 + 1 + 1) / 3


def test_geometric_mean_monotone_in_each_component():
    base = {"a": 0.5, "b": 0.6, "c": 0.7}
    for key in base:
        seq = []
        for v in np.linspace(0.0, 1.2, 25):
            vals = dict(base)
            vals[key] = float(v)
            seq.append(weighted_geometric_mean(vals))
        assert is_non_decreasing(seq)


def test_geometric_mean_validation_and_no_mutation():
    with pytest.raises(ValueError):
        weighted_geometric_mean({})
    with pytest.raises(ValueError):
        weighted_geometric_mean([0.5, 0.5])
    with pytest.raises(ValueError):
        weighted_geometric_mean({"a": float("nan")})
    with pytest.raises(ValueError):
        weighted_geometric_mean({"a": "x"})
    with pytest.raises(ValueError):
        weighted_geometric_mean({"a": True})
    with pytest.raises(ValueError, match="keys"):
        weighted_geometric_mean({"a": 0.5, "b": 0.5}, {"a": 1.0})
    with pytest.raises(ValueError, match="keys"):
        weighted_geometric_mean({"a": 0.5}, {"a": 1.0, "zz": 1.0})
    with pytest.raises(ValueError):
        weighted_geometric_mean({"a": 0.5, "b": 0.5}, {"a": -1.0, "b": 1.0})
    with pytest.raises(ValueError, match="zero"):
        weighted_geometric_mean({"a": 0.5, "b": 0.5}, {"a": 0.0, "b": 0.0})
    with pytest.raises(ValueError):
        weighted_geometric_mean({"a": 0.5}, [1.0])
    vals, w = {"a": 0.3, "b": 2.0}, {"a": 1.0, "b": 2.0}
    snap_v, snap_w = dict(vals), dict(w)
    weighted_geometric_mean(vals, w)
    assert vals == snap_v and w == snap_w


# ---------------------------------------------------------------------------
# water security
# ---------------------------------------------------------------------------
def test_water_security_components_closed_form():
    c = water_security_components(0.9, 25.0, 0.5, 1.0)
    assert set(c) == set(WATER_COMPONENTS)
    assert c["supply"] == pytest.approx(0.9)
    assert c["stress"] == pytest.approx(0.75)
    assert c["environment"] == pytest.approx(0.5)
    assert c["groundwater"] == pytest.approx(0.5)
    # saturation
    c = water_security_components(1.2, 0.0, 1.0, 0.1)
    assert c == {"supply": 1.0, "stress": 1.0, "environment": 1.0, "groundwater": 1.0}
    c = water_security_components(0.0, INF, 0.0, INF)
    assert c == {"supply": 0.0, "stress": 0.0, "environment": 0.0, "groundwater": 0.0}


def test_water_security_index_closed_form():
    assert water_security_index(1.0, 0.0, 1.0, 0.0) == 1.0
    assert water_security_index(1.0, 0.0, 1.0, 0.5) == 1.0
    assert water_security_index(1.0, 50.0, 1.0, 1.0) == pytest.approx(0.25 ** 0.25)
    assert water_security_index(0.5, 50.0, 0.5, 1.0) == pytest.approx(0.5)
    # stress 100 % (or inf) collapses the index to the floor
    assert water_security_index(1.0, 100.0, 1.0, 0.0) == pytest.approx(GEOMETRIC_FLOOR ** 0.25)
    assert water_security_index(1.0, INF, 1.0, 0.0) == pytest.approx(GEOMETRIC_FLOOR ** 0.25)
    assert water_security_index(1.0, 0.0, 1.0, INF) == pytest.approx(GEOMETRIC_FLOOR ** 0.25)
    # weights
    assert water_security_index(1.0, 50.0, 1.0, 1.0, weights={"supply": 1, "stress": 0, "environment": 1, "groundwater": 0}) == 1.0
    assert water_security_index(1.0, 50.0, 1.0, 1.0, weights={"supply": 0, "stress": 1, "environment": 0, "groundwater": 0}) == pytest.approx(0.5)


def test_water_security_index_monotone_and_bounded():
    assert is_non_decreasing([water_security_index(s, 30.0, 0.8, 0.9) for s in np.linspace(0, 1.1, 23)])
    assert is_non_increasing([water_security_index(0.9, st, 0.8, 0.9) for st in np.linspace(0, 150, 31)])
    assert is_non_decreasing([water_security_index(0.9, 30.0, e, 0.9) for e in np.linspace(0, 1, 11)])
    assert is_non_increasing([water_security_index(0.9, 30.0, 0.8, g) for g in np.linspace(0, 3, 31)])
    rng = np.random.default_rng(7)
    for _ in range(100):
        v = water_security_index(rng.uniform(0, 1.2), rng.uniform(0, 200), rng.uniform(0, 1), rng.uniform(0, 3))
        assert 0.0 <= v <= 1.0


def test_water_security_index_validation():
    with pytest.raises(ValueError):
        water_security_index(-0.1, 10.0, 1.0, 0.5)
    with pytest.raises(ValueError):
        water_security_index(1.0, -1.0, 1.0, 0.5)
    with pytest.raises(ValueError):
        water_security_index(1.0, 10.0, 1.5, 0.5)
    with pytest.raises(ValueError):
        water_security_index(1.0, 10.0, -0.1, 0.5)
    with pytest.raises(ValueError):
        water_security_index(1.0, 10.0, 1.0, -0.5)
    for bad in BAD_NUMBERS:
        with pytest.raises(ValueError):
            water_security_index(bad, 10.0, 1.0, 0.5)
    with pytest.raises(ValueError):
        water_security_index(1.0, 10.0, 1.0, 0.5, weights={"supply": 1.0})


# ---------------------------------------------------------------------------
# energy security
# ---------------------------------------------------------------------------
def test_energy_security_components_and_index_closed_form():
    c = energy_security_components(0.8, 0.3, 500.0)
    assert set(c) == set(ENERGY_COMPONENTS)
    assert c == pytest.approx({"supply": 0.8, "renewable": 0.3, "emissions": 0.5})
    assert energy_security_components(1.5, 1.0, 0.0) == {"supply": 1.0, "renewable": 1.0, "emissions": 1.0}
    assert energy_security_components(0.0, 0.0, INF) == {"supply": 0.0, "renewable": 0.0, "emissions": 0.0}
    assert energy_security_index(1.0, 1.0, 0.0) == 1.0
    assert energy_security_index(0.5, 0.5, 500.0) == pytest.approx(0.5)
    assert energy_security_index(1.0, 1.0, EMISSION_INTENSITY_RANGE[0]) == pytest.approx(GEOMETRIC_FLOOR ** (1 / 3))
    assert energy_security_index(1.0, 0.0, 0.0) == pytest.approx(GEOMETRIC_FLOOR ** (1 / 3))
    assert energy_security_index(0.5, 1.0, 0.0, weights={"supply": 0, "renewable": 1, "emissions": 1}) == 1.0
    # an all-coal grid (IPCC 2014 ~820 t/GWh) scores low but non-zero on emissions
    assert 0.0 < energy_security_components(1.0, 0.0, 820.0)["emissions"] < 0.25


def test_energy_security_index_monotone_bounded_validation():
    assert is_non_decreasing([energy_security_index(s, 0.4, 300.0) for s in np.linspace(0, 1.2, 25)])
    assert is_non_decreasing([energy_security_index(0.9, r, 300.0) for r in np.linspace(0, 1, 21)])
    assert is_non_increasing([energy_security_index(0.9, 0.4, e) for e in np.linspace(0, 1500, 31)])
    rng = np.random.default_rng(3)
    for _ in range(100):
        assert 0.0 <= energy_security_index(rng.uniform(0, 1.5), rng.uniform(0, 1), rng.uniform(0, 1500)) <= 1.0
    with pytest.raises(ValueError):
        energy_security_index(-0.1, 0.5, 100.0)
    with pytest.raises(ValueError):
        energy_security_index(1.0, 1.2, 100.0)
    with pytest.raises(ValueError):
        energy_security_index(1.0, 0.5, -1.0)
    for bad in BAD_NUMBERS:
        with pytest.raises(ValueError):
            energy_security_index(1.0, bad, 100.0)


# ---------------------------------------------------------------------------
# food security
# ---------------------------------------------------------------------------
def test_food_security_components_and_index_closed_form():
    assert set(food_security_components(0.5, 0.5)) == set(FOOD_COMPONENTS)
    assert food_security_components(1.3, 0.7) == {"self_sufficiency": 1.0, "et_ratio": 0.7}
    assert food_security_index(1.0, 1.0) == 1.0
    assert food_security_index(0.25, 1.0) == pytest.approx(0.5)
    assert food_security_index(1.0, 0.25) == pytest.approx(0.5)
    assert food_security_index(0.5, 0.5) == pytest.approx(0.5)
    assert food_security_index(2.0, 0.5) == pytest.approx(0.5 ** 0.5)  # surplus capped at 1
    assert food_security_index(0.0, 1.0) == pytest.approx(GEOMETRIC_FLOOR ** 0.5)
    assert food_security_index(0.3, 1.0, weights={"self_sufficiency": 0.0, "et_ratio": 1.0}) == 1.0


def test_food_security_index_monotone_bounded_validation():
    assert is_non_decreasing([food_security_index(s, 0.8) for s in np.linspace(0, 1.5, 31)])
    assert is_non_decreasing([food_security_index(0.8, e) for e in np.linspace(0, 1.2, 25)])
    rng = np.random.default_rng(5)
    for _ in range(100):
        assert 0.0 <= food_security_index(rng.uniform(0, 2), rng.uniform(0, 1.1)) <= 1.0
    with pytest.raises(ValueError):
        food_security_index(-0.1, 1.0)
    with pytest.raises(ValueError):
        food_security_index(1.0, -0.1)
    for bad in BAD_NUMBERS:
        with pytest.raises(ValueError):
            food_security_index(bad, 1.0)
        with pytest.raises(ValueError):
            food_security_index(1.0, bad)


# ---------------------------------------------------------------------------
# WEF nexus index
# ---------------------------------------------------------------------------
def test_wef_nexus_index_closed_form():
    assert wef_nexus_index(1.0, 1.0, 1.0) == 1.0
    assert wef_nexus_index(0.5, 0.5, 0.5) == pytest.approx(0.5)
    assert wef_nexus_index(1.0, 0.5, 0.25) == pytest.approx(0.125 ** (1 / 3))
    assert wef_nexus_index(1.0, 1.0, 0.0) == pytest.approx(GEOMETRIC_FLOOR ** (1 / 3))
    assert wef_nexus_index(1.2, 1.0, 1.0) == 1.0
    assert wef_nexus_index(0.8, 0.2, 0.3, weights={"water": 1.0, "energy": 0.0, "food": 0.0}) == pytest.approx(0.8)
    assert wef_nexus_index(0.8, 0.2, 0.3, weights={"water": 0.0, "energy": 1.0, "food": 1.0}) == pytest.approx(
        (0.2 * 0.3) ** 0.5
    )


def test_wef_nexus_index_symmetric_monotone_bounded():
    rng = np.random.default_rng(11)
    for _ in range(200):
        w, e, f = rng.uniform(0, 1, 3)
        v = wef_nexus_index(w, e, f)
        assert 0.0 <= v <= 1.0
        assert v == pytest.approx(wef_nexus_index(f, w, e)) == pytest.approx(wef_nexus_index(e, f, w))
        assert v <= (w + e + f) / 3 + 1e-12
        assert min(w, e, f) - 1e-12 <= v <= max(w, e, f) + 1e-12
    for pos in range(3):
        seq = []
        for x in np.linspace(0, 1, 21):
            args = [0.6, 0.7, 0.8]
            args[pos] = float(x)
            seq.append(wef_nexus_index(*args))
        assert is_non_decreasing(seq)
    # a collapse of any single pillar is heavily punished
    assert wef_nexus_index(1.0, 1.0, 0.01) < 0.25


def test_wef_nexus_index_validation():
    with pytest.raises(ValueError):
        wef_nexus_index(-0.1, 1.0, 1.0)
    with pytest.raises(ValueError):
        wef_nexus_index(1.0, float("nan"), 1.0)
    with pytest.raises(ValueError, match="keys"):
        wef_nexus_index(1.0, 1.0, 1.0, weights={"water": 1.0})
    with pytest.raises(ValueError):
        wef_nexus_index(1.0, 1.0, 1.0, weights={"water": 0.0, "energy": 0.0, "food": 0.0})
    with pytest.raises(ValueError):
        wef_nexus_index(1.0, 1.0, 1.0, weights={"water": 1.0, "energy": 1.0, "food": 1.0, "x": 1.0})


# ---------------------------------------------------------------------------
# equity
# ---------------------------------------------------------------------------
def test_equity_index_closed_form_and_types():
    assert equity_index({"A": 1.0, "B": 1.0, "C": 1.0}) == 1.0
    assert equity_index([5.0, 5.0]) == 1.0
    assert equity_index({"A": 1.0, "B": 0.0}) == pytest.approx(0.5)
    assert equity_index([0.0, 0.0, 0.0, 1.0]) == pytest.approx(0.25)
    assert equity_index((0.0, 0.0, 0.0, 1.0)) == pytest.approx(0.25)
    assert equity_index(np.array([0.0, 0.0, 0.0, 1.0])) == pytest.approx(0.25)
    assert equity_index({"only": 3.0}) == 1.0
    assert equity_index({"A": 0.0, "B": 0.0}) == 1.0  # nothing to share unequally
    vals = {"A": 0.2, "B": 0.5, "C": 0.9}
    assert equity_index(vals) == pytest.approx(1.0 - gini(vals))
    assert equity_index(list(vals.values())) == pytest.approx(1.0 - gini(list(vals.values())))


def test_equity_index_properties_and_validation():
    rng = np.random.default_rng(13)
    for _ in range(100):
        vals = rng.uniform(0, 10, int(rng.integers(1, 8)))
        e = equity_index(list(vals))
        assert 0.0 <= e <= 1.0
        assert equity_index(list(vals * 3.0)) == pytest.approx(e)  # scale invariant
    # more concentration -> less equity
    assert equity_index([1, 1, 1, 1]) > equity_index([2, 1, 1, 0]) > equity_index([4, 0, 0, 0])
    with pytest.raises(ValueError):
        equity_index({})
    with pytest.raises(ValueError):
        equity_index([])
    with pytest.raises(ValueError):
        equity_index({"A": -1.0, "B": 1.0})
    with pytest.raises(ValueError):
        equity_index([1.0, INF])
    with pytest.raises(ValueError):
        equity_index("abc")
    with pytest.raises(ValueError):
        equity_index(3.0)


# ---------------------------------------------------------------------------
# SDG indicators
# ---------------------------------------------------------------------------
def test_water_use_efficiency():
    assert water_use_efficiency_usd_per_m3(1e9, 100.0) == pytest.approx(10.0)  # 1e9 / 1e8 m3
    assert water_use_efficiency_usd_per_m3(0.0, 100.0) == 0.0
    assert water_use_efficiency_usd_per_m3(10.0, 0.0) == INF
    assert water_use_efficiency_usd_per_m3(0.0, 0.0) == 0.0
    assert water_use_efficiency_usd_per_m3(40e9, 1700.0) == pytest.approx(40e9 / 1.7e9)
    with pytest.raises(ValueError):
        water_use_efficiency_usd_per_m3(-1.0, 1.0)
    with pytest.raises(ValueError):
        water_use_efficiency_usd_per_m3(1.0, -1.0)
    with pytest.raises(ValueError):
        water_use_efficiency_usd_per_m3(INF, 1.0)


def test_sdg_indicators_keys_and_values():
    out = sdg_indicators(1e9, 100.0, 40.0, 0.3, 1.2, 1.0)
    assert set(out) == set(SDG_KEYS.values())
    assert out["6.4.1_water_use_efficiency_usd_per_m3"] == pytest.approx(10.0)
    assert out["6.4.2_water_stress_pct"] == 40.0
    assert out["6.5.2_transboundary_cooperation"] == 1.0
    assert out["7.2_renewable_share"] == 0.3
    assert out["2.1_food_self_sufficiency"] == 1.2  # not capped
    assert all(isinstance(v, float) for v in out.values())
    # defaults and bools
    assert sdg_indicators(1e9, 100.0, 40.0, 0.3, 1.2)["6.5.2_transboundary_cooperation"] == 0.0
    assert sdg_indicators(1e9, 100.0, 40.0, 0.3, 1.2, True)["6.5.2_transboundary_cooperation"] == 1.0
    assert sdg_indicators(1e9, 100.0, 40.0, 0.3, 1.2, False)["6.5.2_transboundary_cooperation"] == 0.0
    assert sdg_indicators(1e9, 100.0, 40.0, 0.3, 1.2, 0.5)["6.5.2_transboundary_cooperation"] == 0.5
    # inf stress and zero withdrawal are legal
    out = sdg_indicators(1e9, 0.0, INF, 0.0, 0.0)
    assert out["6.4.1_water_use_efficiency_usd_per_m3"] == INF
    assert out["6.4.2_water_stress_pct"] == INF


def test_sdg_indicators_validation():
    with pytest.raises(ValueError):
        sdg_indicators(-1.0, 100.0, 40.0, 0.3, 1.2)
    with pytest.raises(ValueError):
        sdg_indicators(1.0, -100.0, 40.0, 0.3, 1.2)
    with pytest.raises(ValueError):
        sdg_indicators(1.0, 100.0, -40.0, 0.3, 1.2)
    with pytest.raises(ValueError):
        sdg_indicators(1.0, 100.0, 40.0, 1.3, 1.2)
    with pytest.raises(ValueError):
        sdg_indicators(1.0, 100.0, 40.0, 0.3, -1.2)
    with pytest.raises(ValueError):
        sdg_indicators(1.0, 100.0, 40.0, 0.3, 1.2, 1.5)
    with pytest.raises(ValueError):
        sdg_indicators(1.0, 100.0, 40.0, 0.3, 1.2, -0.1)
    with pytest.raises(ValueError):
        sdg_indicators(1.0, 100.0, 40.0, 0.3, 1.2, "yes")


# ---------------------------------------------------------------------------
# SustainabilityReport container
# ---------------------------------------------------------------------------
def make_report() -> SustainabilityReport:
    rip = {
        "Up": {"water_security": 0.9, "energy_security": 0.8, "food_security": 0.7, "nexus_index": 0.79,
               "supply_ratio": 1.0, "falkenmark": "no_stress", "6.4.2_water_stress_pct": 12.0},
        "Down": {"water_security": 0.4, "energy_security": 0.5, "food_security": 0.6, "nexus_index": 0.49,
                 "supply_ratio": 0.7, "falkenmark": "scarcity", "6.4.2_water_stress_pct": INF},
    }
    basin = {"water_security": 0.65, "energy_security": 0.65, "food_security": 0.65, "nexus_index": 0.64,
             "supply_ratio": 0.85, "equity_index": 0.9, "outflow_to_sea_mm3": None, "worst_riparian": "Down",
             "6.4.2_water_stress_pct": INF, "env_flow_met_share": 1.0}
    return SustainabilityReport("Test basin", "test", [2025, 2026], rip, basin)


def test_report_access_and_export():
    rep = make_report()
    assert rep.names() == ["Up", "Down"]
    assert rep.riparian("Up")["nexus_index"] == 0.79
    with pytest.raises(KeyError):
        rep.riparian("Nowhere")
    idx = rep.indices()
    assert set(idx) == {"Up", "Down", "BASIN"}
    assert set(idx["Up"]) == set(INDEX_KEYS)
    assert idx["BASIN"]["nexus_index"] == 0.64
    assert "BASIN" not in rep.indices(include_basin=False)
    sdg = rep.sdg()
    assert set(sdg["Up"]) == set(SDG_KEYS.values())
    assert sdg["Down"]["6.4.2_water_stress_pct"] == INF
    assert sdg["Up"]["7.2_renewable_share"] is None  # absent -> None, never KeyError
    d = rep.to_dict()
    assert d["basin_name"] == "Test basin" and d["scenario"] == "test" and d["years"] == [2025, 2026]
    assert d["riparians"]["Up"]["water_security"] == 0.9 and d["basin"]["equity_index"] == 0.9
    d["riparians"]["Up"]["water_security"] = -1.0  # deep copy: report unaffected
    d["years"].append(2030)
    assert rep.riparians["Up"]["water_security"] == 0.9 and rep.years == [2025, 2026]
    rows = rep.to_rows(["nexus_index", "equity_index"])
    assert [r["name"] for r in rows] == ["Up", "Down", "BASIN"]
    assert rows[0]["equity_index"] is None and rows[2]["equity_index"] == 0.9


def test_report_summary_table():
    rep = make_report()
    text = rep.summary(["nexus_index", "supply_ratio", "falkenmark", "6.4.2_water_stress_pct", "outflow_to_sea_mm3", "equity_index"])
    lines = text.splitlines()
    assert lines[0].startswith("Sustainability assessment: Test basin | scenario: test | 2025-2026 (2 years)")
    assert "nexus" in lines[1] and COLUMN_LABELS["6.4.2_water_stress_pct"] in lines[1]
    assert any(ln.startswith("Up") for ln in lines) and any(ln.startswith("Down") for ln in lines)
    assert lines[-1].startswith("BASIN")
    assert "inf" in lines[-1] and "0.900" in lines[-1]  # inf stress and equity
    assert "-" in lines[-1].split()  # None -> "-"
    assert "no_stress" in text and "scarcity" in text
    assert str(rep) == rep.summary()
    # default columns work even when most are missing (rendered as "-")
    default = rep.summary()
    assert "BASIN" in default and len(default.splitlines()) == 2 + 1 + 2 + 1 + 1
    with pytest.raises(ValueError, match="unknown column"):
        rep.summary(["does_not_exist"])
    with pytest.raises(ValueError):
        rep.summary([])


def test_report_validation():
    with pytest.raises(ValueError):
        SustainabilityReport("b", "s", [], riparians=[1, 2], basin={})
    with pytest.raises(ValueError):
        SustainabilityReport("b", "s", [], riparians={"A": 1.0}, basin={})
    with pytest.raises(ValueError):
        SustainabilityReport("b", "s", [], riparians={}, basin=[])
    rep = SustainabilityReport("b", "s", (2025, 2026), riparians={}, basin={})
    assert rep.years == [2025, 2026]


def test_report_to_dataframe():
    pd = pytest.importorskip("pandas")
    rep = make_report()
    df = rep.to_dataframe(["nexus_index", "supply_ratio"])
    assert isinstance(df, pd.DataFrame)
    assert list(df.index) == ["Up", "Down", "BASIN"]
    assert df.loc["Down", "nexus_index"] == pytest.approx(0.49)


# ---------------------------------------------------------------------------
# assess_records / assess on hand-computable synthetic records
# ---------------------------------------------------------------------------
def test_assess_records_riparian_closed_form(tiny):
    rep = assess_records(tiny, basin_name="Tiny", scenario_name="synthetic")
    assert isinstance(rep, SustainabilityReport)
    assert rep.basin_name == "Tiny" and rep.scenario_name == "synthetic"
    assert rep.years == [2025, 2026, 2027]
    assert rep.names() == ["A", "B"]
    a = rep.riparian("A")
    assert a["years"] == 3 and a["first_year"] == 2025 and a["last_year"] == 2027
    assert a["population"] == 1e6 and a["gdp_usd"] == 2e9
    # index means
    assert a["water_security"] == pytest.approx((0.9 + 0.7 + 0.8) / 3)
    assert a["energy_security"] == pytest.approx(0.5)
    assert a["food_security"] == pytest.approx(1.0)
    assert a["nexus_index"] == pytest.approx(0.6)
    assert a["min_nexus_index"] == pytest.approx(0.4)
    assert a["nexus_index_trend"] == pytest.approx(0.2)
    # water supply: ratios 1, 0.5, 1
    assert a["supply_ratio"] == pytest.approx((1 + 0.5 + 1) / 3)
    assert a["min_supply_ratio"] == pytest.approx(0.5)
    assert a["supply_reliability"] == pytest.approx(2 / 3)
    assert a["resilience"] == pytest.approx(1.0)  # the single failure recovers next year
    assert a["vulnerability"] == pytest.approx(0.5)
    assert a["total_demand_mm3"] == pytest.approx(100.0)
    assert a["total_withdrawal_mm3"] == pytest.approx(250.0 / 3)
    assert a["total_deficit_mm3"] == pytest.approx(50.0 / 3)
    assert a["env_flow_met_share"] == pytest.approx(2 / 3)
    assert a["years_env_flow_unmet"] == 1
    assert a["entitlement_mm3"] == INF
    assert a["water_stress_sdg642"] == pytest.approx(40.0)
    assert a["groundwater_stress"] == pytest.approx(0.4)
    assert a["per_capita_water_m3"] == pytest.approx(1500.0)
    assert a["falkenmark"] == "stress"
    assert a["transboundary_cooperation"] == 0.0  # no finite entitlement ever applied
    assert a["final_storage_mm3"] == 100.0 and a["storage_end_mm3"] == 100.0
    assert a["inflow_mm3"] == 1000.0 and a["upstream_inflow_mm3"] == 400.0 and a["outflow_mm3"] == 800.0
    assert a["environmental_flow_mm3"] == 300.0
    # energy
    assert a["energy_supply_gwh"] == 80.0 and a["energy_demand_gwh"] == 100.0
    assert a["energy_self_sufficiency"] == pytest.approx(0.8)
    assert a["renewable_share"] == pytest.approx(0.75)
    assert a["emission_intensity_t_per_gwh"] == pytest.approx(10.0)
    assert a["hydropower_gwh"] == 40.0 and a["thermal_gwh"] == 20.0 and a["other_renewable_gwh"] == 20.0
    assert a["emissions_t"] == 800.0 and a["energy_for_water_gwh"] == 5.0 and a["energy_deficit_gwh"] == 20.0
    # food
    assert a["food_kcal"] == pytest.approx(1000.0)
    assert a["food_demand_kcal"] == 1000.0
    assert a["food_self_sufficiency"] == pytest.approx(1.0)  # 3000 / 3000
    assert a["et_ratio"] == 1.0 and a["food_production_t"] == 1e5
    assert a["irrigation_requirement_mm3"] == 50.0 and a["crop_value_usd"] == 1e7 and a["water_value_usd"] == 1.5e8
    # SDG
    assert a["6.4.1_water_use_efficiency_usd_per_m3"] == pytest.approx(6e9 / 250e6)
    assert a["6.4.2_water_stress_pct"] == pytest.approx(40.0)
    assert a["6.5.2_transboundary_cooperation"] == 0.0
    assert a["7.2_renewable_share"] == pytest.approx(0.75)
    assert a["2.1_food_self_sufficiency"] == pytest.approx(1.0)
    b = rep.riparian("B")
    assert b["supply_ratio"] == 1.0 and b["supply_reliability"] == 1.0 and b["resilience"] == 1.0
    assert b["vulnerability"] == 0.0 and b["nexus_index"] == pytest.approx(0.9)
    assert b["transboundary_cooperation"] == 1.0 and b["6.5.2_transboundary_cooperation"] == 1.0
    assert b["entitlement_mm3"] == 250.0
    assert b["falkenmark"] == "scarcity"  # 800 m3/cap


def test_assess_records_basin_closed_form(tiny):
    rep = assess_records(tiny)
    bs = rep.basin
    assert bs["riparians"] == 2 and bs["years"] == 3
    assert bs["population"] == pytest.approx(4e6) and bs["gdp_usd"] == pytest.approx(4e9)
    assert bs["nexus_index"] == pytest.approx((0.6 + 0.9) / 2)
    assert bs["water_security"] == pytest.approx((0.8 + 0.95) / 2)
    assert bs["energy_security"] == pytest.approx((0.5 + 0.9) / 2)
    assert bs["food_security"] == pytest.approx((1.0 + 0.85) / 2)
    assert bs["nexus_index_population_weighted"] == pytest.approx((0.6 * 1 + 0.9 * 3) / 4)
    assert bs["min_nexus_index"] == pytest.approx(0.6) and bs["worst_riparian"] == "A"
    assert bs["nexus_index_trend"] == pytest.approx(0.1)
    assert bs["equity_index"] == pytest.approx(1.0 - gini({"A": 2.5 / 3, "B": 1.0}))
    assert bs["equity_nexus_index"] == pytest.approx(1.0 - gini([0.6, 0.9]))
    assert bs["equity_per_capita_water"] == pytest.approx(1.0 - gini([1500.0, 800.0]))
    # basin water: demand 300/yr, deficit 50 in 2026
    assert bs["supply_ratio"] == pytest.approx(1 - 50 / 900)
    assert bs["min_supply_ratio"] == pytest.approx(1 - 50 / 300)
    assert bs["supply_reliability"] == pytest.approx(2 / 3)
    assert bs["resilience"] == pytest.approx(1.0)
    assert bs["vulnerability"] == pytest.approx(50 / 300)
    assert bs["total_demand_mm3"] == pytest.approx(300.0)
    assert bs["total_withdrawal_mm3"] == pytest.approx((250 + 600) / 3)
    assert bs["total_deficit_mm3"] == pytest.approx(50 / 3)
    assert bs["env_flow_met_share"] == pytest.approx(5 / 6) and bs["years_env_flow_unmet"] == 1
    assert bs["water_stress_sdg642"] == pytest.approx((40.0 + 20.0) / 2)
    assert bs["groundwater_stress"] == pytest.approx(0.4)
    assert bs["transboundary_cooperation"] == pytest.approx(0.5)
    # energy totals: each riparian 80 supply / 100 demand, 60 renewable, 800 t
    assert bs["energy_supply_gwh"] == pytest.approx(160.0) and bs["energy_demand_gwh"] == pytest.approx(200.0)
    assert bs["energy_self_sufficiency"] == pytest.approx(0.8)
    assert bs["renewable_share"] == pytest.approx(0.75)
    assert bs["emission_intensity_t_per_gwh"] == pytest.approx(10.0)
    assert bs["hydropower_gwh"] == pytest.approx(80.0) and bs["emissions_t"] == pytest.approx(1600.0)
    # food: A 3000 + B 3000 kcal over 6000 demand
    assert bs["food_self_sufficiency"] == pytest.approx(1.0)
    assert bs["food_kcal"] == pytest.approx(2000.0) and bs["food_demand_kcal"] == pytest.approx(2000.0)
    assert bs["water_value_usd"] == pytest.approx(3e8) and bs["crop_value_usd"] == pytest.approx(2e7)
    # SDG
    assert bs["6.4.1_water_use_efficiency_usd_per_m3"] == pytest.approx(12e9 / 850e6)
    assert bs["6.4.2_water_stress_pct"] == pytest.approx(30.0)
    assert bs["6.5.2_transboundary_cooperation"] == pytest.approx(0.5)
    assert bs["7.2_renewable_share"] == pytest.approx(0.75)
    assert bs["2.1_food_self_sufficiency"] == pytest.approx(1.0)
    # no balances given -> routed quantities unavailable
    for key in ("outflow_to_sea_mm3", "natural_flow_mm3", "per_capita_water_m3", "falkenmark", "mass_balance_error_mm3"):
        assert bs[key] is None
    # all four indices (and nexus) are in [0, 1] everywhere
    for d in list(rep.riparians.values()) + [bs]:
        for k in INDEX_KEYS:
            assert 0.0 <= d[k] <= 1.0


def test_assess_records_sorts_years_and_respects_years_argument(tiny):
    shuffled = list(reversed(tiny))
    rep = assess_records(shuffled)
    a = rep.riparian("A")
    assert a["first_year"] == 2025 and a["last_year"] == 2027
    assert a["nexus_index_trend"] == pytest.approx(0.2)
    assert a["resilience"] == pytest.approx(1.0)
    assert rep.years == [2025, 2026, 2027]
    rep2 = assess_records(tiny, years=[2025, 2026, 2027])
    assert rep2.years == [2025, 2026, 2027] and rep2.basin["years"] == 3
    # a failure in the final year never recovers -> resilience 0
    recs = [make_record("Z", 2025), make_record("Z", 2026, deficits={MUN: 20.0}, withdrawals={MUN: 80.0})]
    z = assess_records(recs).riparian("Z")
    assert z["supply_reliability"] == 0.5 and z["resilience"] == 0.0 and z["vulnerability"] == pytest.approx(0.2)


def test_assess_records_cooperation_flag_and_entitlements():
    recs = [make_record("A", 2025, entitlement_mm3=120.0), make_record("A", 2026, entitlement_mm3=INF),
            make_record("A", 2027, entitlement_mm3=None), make_record("A", 2028, entitlement_mm3=130.0)]
    rep = assess_records(recs, cooperation=True)
    assert rep.riparian("A")["transboundary_cooperation"] == pytest.approx(0.5)
    assert rep.riparian("A")["6.5.2_transboundary_cooperation"] == pytest.approx(0.5)
    assert rep.basin["6.5.2_transboundary_cooperation"] == pytest.approx(0.5)
    assert rep.riparian("A")["entitlement_mm3"] == INF  # None counts as "no cap"
    rep = assess_records(recs, cooperation=False)
    assert rep.riparian("A")["transboundary_cooperation"] == 0.0
    assert rep.basin["6.5.2_transboundary_cooperation"] == 0.0


def test_assess_records_zero_demand_and_zero_energy_edge_cases():
    recs = [
        make_record("N", y, demands={}, withdrawals={}, deficits={}, energy_supply_gwh=0.0, energy_demand_gwh=0.0,
                    hydropower_gwh=0.0, thermal_gwh=0.0, other_renewable_gwh=0.0, emissions_t=0.0,
                    food_kcal=0.0, food_demand_kcal=0.0, gdp_usd=0.0, water_stress_sdg642=0.0)
        for y in (2025, 2026)
    ]
    rep = assess_records(recs)
    n = rep.riparian("N")
    assert n["supply_ratio"] == 1.0 and n["supply_reliability"] == 1.0 and n["resilience"] == 1.0
    assert n["vulnerability"] == 0.0 and n["total_demand_mm3"] == 0.0
    assert n["energy_self_sufficiency"] == 1.0 and n["renewable_share"] == 0.0
    assert n["emission_intensity_t_per_gwh"] == 0.0
    assert n["food_self_sufficiency"] == 1.0
    assert n["6.4.1_water_use_efficiency_usd_per_m3"] == 0.0  # 0 / 0 -> 0
    assert rep.basin["supply_ratio"] == 1.0 and rep.basin["equity_index"] == 1.0
    assert rep.basin["6.4.1_water_use_efficiency_usd_per_m3"] == 0.0
    # gdp but no withdrawal -> infinite efficiency
    recs2 = [make_record("G", 2025, demands={}, withdrawals={}, deficits={})]
    assert assess_records(recs2).riparian("G")["6.4.1_water_use_efficiency_usd_per_m3"] == INF
    assert "inf" in assess_records(recs2).summary()


def test_assess_records_infinite_stress_propagates():
    recs = [make_record("S", 2025, water_stress_sdg642=INF, groundwater_stress=INF), make_record("S", 2026, water_stress_sdg642=50.0)]
    rep = assess_records(recs)
    assert rep.riparian("S")["water_stress_sdg642"] == INF
    assert rep.riparian("S")["6.4.2_water_stress_pct"] == INF
    assert rep.riparian("S")["groundwater_stress"] == INF
    assert rep.basin["6.4.2_water_stress_pct"] == INF
    assert "inf" in rep.summary()


def test_assess_records_unpopulated_riparian_per_capita_equity():
    # an unpopulated riparian has no per-capita water (nexus records ``inf``);
    # it must not break the basin assessment and is left out of the per-capita
    # equity, which is taken over the populated riparians only
    empty = [make_record("E", y, population=0.0, food_demand_kcal=0.0, per_capita_water_m3=INF) for y in (2025, 2026)]
    a = [make_record("A", y, per_capita_water_m3=3000.0) for y in (2025, 2026)]
    b = [make_record("B", y, population=4e6, per_capita_water_m3=1000.0) for y in (2025, 2026)]
    rep = assess_records(empty + a + b)
    assert rep.riparian("E")["per_capita_water_m3"] == INF
    assert rep.riparian("E")["falkenmark"] == "no_stress"
    assert rep.basin["equity_per_capita_water"] == pytest.approx(equity_index({"A": 3000.0, "B": 1000.0}))
    assert rep.basin["equity_per_capita_water"] == pytest.approx(1.0 - gini([3000.0, 1000.0]))
    assert 0.0 < rep.basin["equity_per_capita_water"] < 1.0
    # the other equity indices still cover every riparian
    assert rep.basin["equity_index"] == pytest.approx(equity_index({n: rep.riparian(n)["supply_ratio"] for n in "EAB"}))
    assert "inf" in rep.summary(columns=["per_capita_water_m3", "nexus_index"])
    # nobody populated at all: the distribution is trivially equal and the
    # population-weighted nexus index falls back to the plain mean
    rep0 = assess_records(empty)
    assert rep0.basin["equity_per_capita_water"] == 1.0
    assert rep0.basin["nexus_index_population_weighted"] == pytest.approx(rep0.basin["nexus_index"])
    # a single populated riparian next to an empty one is also trivially equal
    rep1 = assess_records(empty + a)
    assert rep1.basin["equity_per_capita_water"] == 1.0


def test_assess_records_does_not_mutate_records(tiny):
    before = copy.deepcopy(tiny)
    rep = assess_records(tiny)
    assert tiny == before
    rep.riparians["A"]["nexus_index"] = 99.0  # the report is independent of the records
    assert tiny == before


def test_assess_records_errors(tiny):
    with pytest.raises(ValueError, match="empty"):
        assess_records([])
    with pytest.raises(ValueError, match="lacks field"):
        assess_records([object()])

    @dataclass
    class Partial:
        name: str
        year: int

    with pytest.raises(ValueError, match="population"):
        assess_records([Partial("A", 2025)])
    bad = copy.deepcopy(tiny)
    bad[0].demands = [1.0, 2.0]
    with pytest.raises(ValueError, match="demands"):
        assess_records(bad)
    bad = copy.deepcopy(tiny)
    bad[0].nexus_index = "high"
    with pytest.raises(ValueError, match="nexus_index"):
        assess_records(bad)
    bad = copy.deepcopy(tiny)
    bad[0].name = ""
    with pytest.raises(ValueError, match="name"):
        assess_records(bad)
    bad = copy.deepcopy(tiny)
    bad[1].year = "later"
    with pytest.raises(ValueError, match="year"):
        assess_records(bad)
    with pytest.raises(ValueError, match="BasinBalance"):
        assess_records(tiny, balances=[1, 2, 3])


# ---------------------------------------------------------------------------
# assess() on a contract-shaped stand-in built from real routing
# ---------------------------------------------------------------------------
def test_assess_stand_in_result(azura, basin):
    records_before = copy.deepcopy(azura.records)
    balances_before = [b.to_dict() for b in azura.balances]
    rep = assess(azura)
    assert azura.records == records_before
    assert [b.to_dict() for b in azura.balances] == balances_before
    assert rep.basin_name == basin.name and rep.scenario_name == "smoke"
    assert rep.years == azura.years and rep.names() == basin.names()
    for name in basin.names():
        d = rep.riparian(name)
        recs = [r for r in azura.records if r.name == name]
        for k in INDEX_KEYS:
            assert d[k] == pytest.approx(sum(getattr(r, k) for r in recs) / len(recs))
            assert 0.0 <= d[k] <= 1.0
        assert d["years"] == len(azura.years)
        assert 0.0 <= d["supply_ratio"] <= 1.0 and 0.0 <= d["supply_reliability"] <= 1.0
        assert 0.0 <= d["resilience"] <= 1.0 and 0.0 <= d["vulnerability"] <= 1.0
        assert d["falkenmark"] == falkenmark_category(d["per_capita_water_m3"])
        assert d["transboundary_cooperation"] == 1.0  # treaty entitlements are finite caps
        assert d["7.2_renewable_share"] == pytest.approx(d["renewable_share"])
        assert d["2.1_food_self_sufficiency"] == pytest.approx(d["food_self_sufficiency"])
    # routed-balance aggregates
    bs = rep.basin
    assert bs["outflow_to_sea_mm3"] == pytest.approx(sum(b.outflow_to_sea_mm3 for b in azura.balances) / 4)
    assert bs["natural_flow_mm3"] == pytest.approx(sum(b.natural_flow_mm3 for b in azura.balances) / 4)
    assert bs["mass_balance_error_mm3"] < 1e-6
    pop = basin.total_population()
    assert bs["per_capita_water_m3"] == pytest.approx(bs["natural_flow_mm3"] * 1e6 / pop)
    assert bs["falkenmark"] == falkenmark_category(bs["per_capita_water_m3"])
    assert bs["worst_riparian"] == "Delta"  # arid downstream state, severe stress
    assert rep.riparian("Highland")["nexus_index"] > rep.riparian("Delta")["nexus_index"]
    assert 0.0 <= bs["equity_index"] <= 1.0 and 0.0 <= bs["equity_nexus_index"] <= 1.0
    assert bs["supply_ratio"] == pytest.approx(
        1 - sum(sum(r.deficits.values()) for r in azura.records) / sum(sum(r.demands.values()) for r in azura.records)
    )
    text = rep.summary()
    assert "Azura" in text and "Highland" in text and "Midland" in text and "Delta" in text and "BASIN" in text
    assert "6.4.2%" in text
    d = rep.to_dict()
    assert set(d) == {"basin_name", "scenario", "years", "riparians", "basin"}


def test_assess_accepts_duck_typed_results_and_rejects_others(tiny):
    rep = assess(FakeNexusResult("B", Scenario(name="coop", cooperation=True), [2025, 2026, 2027], tiny, []))
    assert rep.scenario_name == "coop" and rep.riparian("B")["6.5.2_transboundary_cooperation"] == 1.0
    rep = assess(FakeNexusResult("B", Scenario(name="uni", cooperation=False), [2025, 2026, 2027], tiny, []))
    assert rep.scenario_name == "uni" and rep.riparian("B")["6.5.2_transboundary_cooperation"] == 0.0

    class Minimal:  # only the records attribute
        records = tiny

    rep = assess(Minimal())
    assert rep.basin_name == "" and rep.scenario_name == "" and rep.years == [2025, 2026, 2027]
    for bad in (None, 3, "result", {"records": tiny}, [tiny]):
        with pytest.raises(ValueError, match="NexusResult"):
            assess(bad)
    with pytest.raises(ValueError):
        assess(FakeNexusResult("B", None, [], [], []))


def test_assess_real_nexus_result_if_available(basin, baseline):
    nexus = pytest.importorskip("wefnexus.nexus")
    result = nexus.run_nexus(basin, baseline)
    assert isinstance(result, nexus.NexusResult)
    rep = assess(result)
    assert rep.names() == basin.names()
    assert rep.years == list(result.years)
    assert rep.scenario_name == baseline.name
    for d in list(rep.riparians.values()) + [rep.basin]:
        for k in INDEX_KEYS:
            assert 0.0 <= d[k] <= 1.0
    assert rep.basin["mass_balance_error_mm3"] is None or rep.basin["mass_balance_error_mm3"] < 1e-6
    assert "BASIN" in rep.summary()


# ---------------------------------------------------------------------------
# review fixes: basin years_env_flow_unmet semantics, explicit ``years``
# validation and the SDG 6.4.1 denominator (surface + groundwater)
# ---------------------------------------------------------------------------
def test_basin_years_env_flow_unmet_counts_years_not_riparian_years():
    # 2025: every reach met; 2026: BOTH reaches fail; 2027: only A fails
    recs = [
        make_record("A", 2025, env_flow_met=True),
        make_record("A", 2026, env_flow_met=False),
        make_record("A", 2027, env_flow_met=False),
        make_record("B", 2025, env_flow_met=True),
        make_record("B", 2026, env_flow_met=False),
        make_record("B", 2027, env_flow_met=True),
    ]
    rep = assess_records(recs)
    assert rep.riparian("A")["years_env_flow_unmet"] == 2
    assert rep.riparian("B")["years_env_flow_unmet"] == 1
    bs = rep.basin
    # years with ANY unmet reach (2026, 2027) - not the 3 unmet riparian-years
    assert bs["years_env_flow_unmet"] == 2
    assert bs["years_env_flow_unmet"] <= bs["years"] == 3
    assert bs["years_env_flow_unmet"] >= max(rep.riparian(n)["years_env_flow_unmet"] for n in "AB")
    assert bs["env_flow_met_share"] == pytest.approx(3 / 6)  # still a share of riparian-years
    # the corrected value reaches every export path
    assert rep.to_dict()["basin"]["years_env_flow_unmet"] == 2
    assert rep.to_rows(columns=["years_env_flow_unmet"])[-1] == {"name": "BASIN", "years_env_flow_unmet": 2}
    assert "env_unmet" in rep.summary(columns=["years_env_flow_unmet"])
    # every riparian-year unmet: the basin count saturates at the horizon length
    allbad = [make_record(n, y, env_flow_met=False) for n in "ABC" for y in (2025, 2026)]
    rep2 = assess_records(allbad)
    assert rep2.basin["years_env_flow_unmet"] == 2 and rep2.basin["env_flow_met_share"] == 0.0
    assert all(rep2.riparian(n)["years_env_flow_unmet"] == 2 for n in "ABC")
    # nothing unmet
    rep3 = assess_records([make_record(n, y) for n in "AB" for y in (2025, 2026)])
    assert rep3.basin["years_env_flow_unmet"] == 0 and rep3.basin["env_flow_met_share"] == 1.0


def test_basin_years_env_flow_unmet_matches_nexus_summary(basin):
    nexus = pytest.importorskip("wefnexus.nexus")
    from wefnexus.models import Basin, Riparian, WaterDemand

    # (a) two reaches that both fail every year of a 3-year run: 3 years, not 6 riparian-years
    def rip(name, local, env):
        return Riparian(name=name, population=1e6, gdp_usd=1e9, local_inflow_mm3=local,
                        demand=WaterDemand(municipal=10.0, environmental=env))

    two = Basin("two", [rip("A", 100.0, 500.0), rip("B", 0.0, 500.0)])
    res = nexus.run_nexus(two, Scenario(years=3))
    assert all(not r.env_flow_met for r in res.records)
    s = res.summary()[nexus.BASIN_KEY]
    rep = assess(res)
    assert s["years_env_flow_unmet"] == 3
    assert rep.basin["years_env_flow_unmet"] == s["years_env_flow_unmet"] == 3
    assert rep.basin["years_env_flow_unmet"] <= res.n_years
    assert rep.basin["env_flow_met_share"] == s["env_flow_met_share"] == 0.0
    # (b) the example basin with empty reservoirs and three very dry years:
    #     several riparians miss their flow in the same year
    dry = basin.copy()
    for r in dry.riparians:
        r.reservoir_storage_mm3 = 0.0
    res2 = nexus.NexusModel(dry, Scenario(years=4), flow_factors=[0.05, 0.05, 1.0, 0.05]).run()
    s2 = res2.summary()
    rep2 = assess(res2)
    unmet_records = sum(1 for r in res2.records if not r.env_flow_met)
    assert unmet_records > res2.n_years  # the riparian-year count would exceed the horizon
    assert 0 < s2[nexus.BASIN_KEY]["years_env_flow_unmet"] <= res2.n_years
    assert rep2.basin["years_env_flow_unmet"] == s2[nexus.BASIN_KEY]["years_env_flow_unmet"]
    assert rep2.basin["env_flow_met_share"] == pytest.approx(s2[nexus.BASIN_KEY]["env_flow_met_share"])
    for n in dry.names():
        assert rep2.riparian(n)["years_env_flow_unmet"] == s2[n]["years_env_flow_unmet"]
        assert rep2.riparian(n)["env_flow_met_share"] == pytest.approx(s2[n]["env_flow_met_share"])


def test_assess_records_rejects_years_without_records(tiny):
    with pytest.raises(ValueError, match="years"):
        assess_records(tiny, years=[1999])
    with pytest.raises(ValueError, match="years"):
        assess_records(tiny, years=[2025, 2030])  # a partial overlap is not silently truncated
    with pytest.raises(ValueError, match="years"):
        assess_records(tiny, years=[])
    with pytest.raises(ValueError, match="duplicate"):
        assess_records(tiny, years=[2025, 2025, 2026])
    with pytest.raises(ValueError, match="years"):
        assess_records(tiny, years="2025")
    with pytest.raises(ValueError, match="years"):
        assess_records(tiny, years=2025)
    with pytest.raises(ValueError, match="hashable"):
        assess_records(tiny, years=[[2025]])
    # a subset is fine and applies consistently to the title, the basin and the riparians
    rep = assess_records(tiny, years=[2025, 2026])
    assert rep.years == [2025, 2026] and rep.basin["years"] == 2
    assert "2025-2026 (2 years)" in rep.summary()
    a = rep.riparian("A")
    assert a["years"] == 2 and a["first_year"] == 2025 and a["last_year"] == 2026
    assert a["nexus_index"] == pytest.approx((0.6 + 0.4) / 2)
    assert a["nexus_index_trend"] == pytest.approx(0.4 - 0.6)
    assert a["supply_reliability"] == pytest.approx(0.5) and a["resilience"] == 0.0  # fails in the last year
    assert rep.basin["env_flow_met_share"] == pytest.approx(3 / 4)  # A fails in 2026 only
    assert rep.basin["years_env_flow_unmet"] == 1
    assert rep.basin["supply_ratio"] == pytest.approx(1 - 50 / 600)
    assert rep.basin["6.4.1_water_use_efficiency_usd_per_m3"] == pytest.approx(8e9 / ((150 + 400) * 1e6))
    # a single year is a valid horizon too
    one = assess_records(tiny, years=[2027])
    assert one.years == [2027] and one.basin["years"] == 1 and one.riparian("A")["nexus_index"] == 0.8
    assert "2027-2027 (1 years)" in one.summary()
    # years are stored chronologically whatever order they were given in
    rev = assess_records(tiny, years=[2027, 2025, 2026])
    assert rev.years == [2025, 2026, 2027]
    assert rev.to_dict() == assess_records(tiny).to_dict()
    # numpy integer years are accepted
    npy = assess_records(tiny, years=np.array([2025, 2026, 2027]))
    assert [int(y) for y in npy.years] == [2025, 2026, 2027] and npy.basin["years"] == 3


def test_assess_handles_empty_or_bad_result_years(tiny):
    # an empty horizon on a duck-typed result means "the years of the records"
    rep = assess(FakeNexusResult("B", Scenario(name="x"), [], tiny, []))
    assert rep.years == [2025, 2026, 2027] and rep.basin["years"] == 3
    # a horizon that names years without records is an error, not a fallback
    with pytest.raises(ValueError, match="years"):
        assess(FakeNexusResult("B", Scenario(name="x"), [1999], tiny, []))
    with pytest.raises(ValueError, match="years"):
        assess(FakeNexusResult("B", Scenario(name="x"), 2025, tiny, []))


def test_sdg_641_denominator_includes_groundwater_used_on_records(tiny):
    recs = copy.deepcopy(tiny)
    for r in recs:
        if r.name == "A":
            r.groundwater_used_mm3 = 50.0  # the optional diagnostic carried by nexus RiparianYear
    before = copy.deepcopy(recs)
    rep = assess_records(recs)
    assert recs == before and all(r.groundwater_used_mm3 == 50.0 for r in recs if r.name == "A")
    a = rep.riparian("A")
    assert a["total_withdrawal_mm3"] == pytest.approx(250 / 3)  # surface only, unchanged
    assert a["groundwater_used_mm3"] == pytest.approx(50.0)
    assert a["freshwater_withdrawal_mm3"] == pytest.approx((250 + 150) / 3)
    assert a["6.4.1_water_use_efficiency_usd_per_m3"] == pytest.approx(6e9 / ((250 + 150) * 1e6))
    b = rep.riparian("B")  # no attribute -> groundwater 0, denominator = surface withdrawal
    assert b["groundwater_used_mm3"] == 0.0 and b["freshwater_withdrawal_mm3"] == pytest.approx(200.0)
    assert b["6.4.1_water_use_efficiency_usd_per_m3"] == pytest.approx(6e9 / 600e6)
    bs = rep.basin
    assert bs["groundwater_used_mm3"] == pytest.approx(50.0)
    assert bs["freshwater_withdrawal_mm3"] == pytest.approx((250 + 150 + 600) / 3)
    assert bs["total_withdrawal_mm3"] == pytest.approx((250 + 600) / 3)
    assert bs["6.4.1_water_use_efficiency_usd_per_m3"] == pytest.approx(12e9 / ((850 + 150) * 1e6))
    assert "gw_used" in rep.summary(columns=["groundwater_used_mm3", "freshwater_withdrawal_mm3"])
    # more groundwater pumping -> lower water-use efficiency (monotone)
    for r in recs:
        if r.name == "A":
            r.groundwater_used_mm3 = 500.0
    assert (assess_records(recs).riparian("A")["6.4.1_water_use_efficiency_usd_per_m3"]
            < a["6.4.1_water_use_efficiency_usd_per_m3"])
    # None counts as "not recorded"
    for r in recs:
        r.groundwater_used_mm3 = None
    assert assess_records(recs).riparian("A")["6.4.1_water_use_efficiency_usd_per_m3"] == pytest.approx(6e9 / 250e6)
    # negative, non-finite or non-numeric values are rejected with the field name
    for bad in (-1.0, INF, "x", [1.0]):
        recs[0].groundwater_used_mm3 = bad
        with pytest.raises(ValueError, match="groundwater_used_mm3"):
            assess_records(recs)


def test_sdg_641_denominator_includes_groundwater(basin):
    nexus = pytest.importorskip("wefnexus.nexus")
    res = nexus.run_nexus(basin, Scenario(years=2))
    rep = assess(res)
    s = res.summary()
    for name in basin.names():
        recs = res.for_riparian(name)
        surface = sum(r.total_withdrawal() for r in recs)
        gw = sum(r.groundwater_used_mm3 for r in recs)
        expected = sum(r.gdp_usd for r in recs) / ((surface + gw) * 1e6)
        d = rep.riparian(name)
        assert d["6.4.1_water_use_efficiency_usd_per_m3"] == pytest.approx(expected)
        assert d["groundwater_used_mm3"] == pytest.approx(gw / 2)
        assert d["freshwater_withdrawal_mm3"] == pytest.approx((surface + gw) / 2)
        assert d["total_withdrawal_mm3"] == pytest.approx(surface / 2)  # surface only, as nexus.summary()
        assert d["total_withdrawal_mm3"] == pytest.approx(s[name]["total_withdrawal_mm3"])
    delta = res.for_riparian("Delta")
    assert sum(r.groundwater_used_mm3 for r in delta) > 0  # Delta pumps 2000 Mm3/yr
    total = sum(r.total_withdrawal() + r.groundwater_used_mm3 for r in delta)
    expected = sum(r.gdp_usd for r in delta) / (total * 1e6)
    assert rep.riparian("Delta")["6.4.1_water_use_efficiency_usd_per_m3"] == pytest.approx(expected)
    # groundwater in the denominator lowers the efficiency relative to a surface-only one
    assert expected < sum(r.gdp_usd for r in delta) / (sum(r.total_withdrawal() for r in delta) * 1e6)
    # desalinated water is not a freshwater withdrawal and stays out of the denominator
    assert sum(r.desalinated_mm3 for r in delta) > 0
    # basin: horizon totals of GDP over horizon totals of surface + groundwater withdrawal
    recs = res.records
    exp_b = sum(r.gdp_usd for r in recs) / (sum(r.total_withdrawal() + r.groundwater_used_mm3 for r in recs) * 1e6)
    assert rep.basin["6.4.1_water_use_efficiency_usd_per_m3"] == pytest.approx(exp_b)
    assert rep.basin["freshwater_withdrawal_mm3"] == pytest.approx(
        sum(r.total_withdrawal() + r.groundwater_used_mm3 for r in recs) / res.n_years
    )
