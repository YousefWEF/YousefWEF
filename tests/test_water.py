"""Tests for :mod:`wefnexus.water` - indicators and basin routing."""
import copy
import dataclasses
import math

import numpy as np
import pytest

from wefnexus.data import example_basin
from wefnexus.models import (
    SECTOR_PRIORITY,
    Basin,
    EnergySystem,
    Riparian,
    Scenario,
    Sector,
    WaterDemand,
)
from wefnexus.water import (
    FALKENMARK_THRESHOLDS,
    BasinBalance,
    ReachResult,
    dependency_ratio,
    falkenmark_category,
    groundwater_stress,
    natural_flows,
    per_capita_water,
    resilience,
    route_basin,
    sdg_642_water_stress,
    stochastic_flow_factors,
    supply_reliability,
    vulnerability,
    water_exploitation_index,
)

FLOW_FACTORS = [0.0, 0.5, 1.0, 2.0]
MUN, IND, ENE, AGR = Sector.MUNICIPAL, Sector.INDUSTRIAL, Sector.ENERGY, Sector.AGRICULTURAL


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def make_riparian(
    name,
    local=0.0,
    mun=0.0,
    ind=0.0,
    ag=0.0,
    energy=0.0,
    env=0.0,
    fractions=None,
    capacity=0.0,
    storage=0.0,
    evap=0.0,
    gw=0.0,
    desal=0.0,
    treaty=None,
):
    """Small riparian with explicit, easy-to-reason-about numbers."""
    demand = WaterDemand(municipal=mun, industrial=ind, agricultural=ag, energy=energy, environmental=env)
    if fractions:
        demand.consumption_fraction.update(fractions)
    return Riparian(
        name=name,
        population=1e6,
        gdp_usd=1e9,
        local_inflow_mm3=local,
        demand=demand,
        groundwater_abstraction_mm3=gw,
        energy=EnergySystem(desalination_capacity_mm3=desal),
        reservoir_capacity_mm3=capacity,
        reservoir_storage_mm3=storage,
        reservoir_evaporation_fraction=evap,
        treaty_allocation_mm3=treaty,
    )


def single(headwater, **kw):
    """Basin with one riparian ``A`` fed by ``headwater``."""
    return Basin("one", [make_riparian("A", **kw)], headwater_inflow_mm3=headwater)


def assert_reach_balance(reach: ReachResult, rel=1e-6):
    lhs = reach.inflow_mm3 + reach.storage_start_mm3
    rhs = reach.total_consumption() + reach.outflow_mm3 + reach.storage_end_mm3 + reach.evaporation_mm3
    assert abs(lhs - rhs) <= rel * max(1.0, lhs), f"{reach.name}: {lhs} != {rhs}"


def assert_physical(bal: BasinBalance, basin: Basin, tol=1e-9):
    """Bounds every routing result must satisfy."""
    assert bal.mass_balance_error() <= 1e-6 * max(1.0, bal.natural_flow_mm3 + sum(r.storage_start_mm3 for r in bal.reaches))
    previous_out = None
    for reach, rip in zip(bal.reaches, basin.riparians):
        assert reach.name == rip.name
        assert_reach_balance(reach)
        assert reach.inflow_mm3 >= -tol
        assert reach.outflow_mm3 >= -tol
        assert -tol <= reach.storage_release_mm3 <= reach.storage_start_mm3 + tol
        assert reach.storage_refill_mm3 >= -tol
        assert reach.evaporation_mm3 >= -tol
        assert -tol <= reach.storage_end_mm3 <= max(rip.reservoir_capacity_mm3, reach.storage_start_mm3) + tol
        supply = reach.inflow_mm3 + reach.storage_start_mm3
        assert reach.total_withdrawal() <= supply + tol
        assert reach.total_withdrawal() <= reach.entitlement_mm3 + tol
        assert reach.total_consumption() <= max(supply - reach.environmental_flow_mm3, 0.0) + 1e-6
        f = rip.demand.consumption_fraction
        for s in SECTOR_PRIORITY:
            assert reach.withdrawals[s] >= -tol
            assert reach.deficits[s] >= -tol
            assert reach.consumption[s] == pytest.approx(reach.withdrawals[s] * f.get(s, 0.0))
            assert reach.consumption[s] <= reach.withdrawals[s] + tol
            assert reach.withdrawals[s] + reach.deficits[s] <= reach.demands[s] + tol
        # env flow is met whenever enough water exists in river + reservoir
        if supply >= reach.environmental_flow_mm3 - tol:
            assert reach.env_flow_met
        assert reach.env_flow_met == (reach.outflow_mm3 >= reach.environmental_flow_mm3 - 1e-9)
        if previous_out is not None:
            assert reach.upstream_inflow_mm3 == pytest.approx(previous_out)
        previous_out = reach.outflow_mm3
    assert bal.outflow_to_sea_mm3 == pytest.approx(bal.reaches[-1].outflow_mm3)


# ---------------------------------------------------------------------------
# indicators
# ---------------------------------------------------------------------------
def test_per_capita_water_closed_form():
    assert per_capita_water(1000.0, 1_000_000) == pytest.approx(1000.0)
    assert per_capita_water(0.0, 10) == 0.0
    # example basin: Delta is in absolute scarcity on surface water alone
    delta = example_basin().riparian("Delta")
    pc = per_capita_water(delta.local_inflow_mm3 + delta.groundwater_recharge_mm3, delta.population)
    assert pc == pytest.approx(2500.0 * 1e6 / 45e6)
    assert falkenmark_category(pc) == "absolute_scarcity"


@pytest.mark.parametrize("population", [0, -5])
def test_per_capita_water_rejects_bad_population(population):
    with pytest.raises(ValueError):
        per_capita_water(100.0, population)


def test_per_capita_water_rejects_negative_or_nan():
    with pytest.raises(ValueError):
        per_capita_water(-1.0, 100)
    with pytest.raises(ValueError):
        per_capita_water(float("nan"), 100)


@pytest.mark.parametrize(
    "value, expected",
    [
        (5000.0, "no_stress"),
        (1700.0, "no_stress"),
        (1699.9, "stress"),
        (1000.0, "stress"),
        (999.9, "scarcity"),
        (500.0, "scarcity"),
        (499.9, "absolute_scarcity"),
        (0.0, "absolute_scarcity"),
        (math.inf, "no_stress"),
    ],
)
def test_falkenmark_categories(value, expected):
    assert falkenmark_category(value) == expected


def test_falkenmark_thresholds_and_validation():
    assert FALKENMARK_THRESHOLDS == {"no_stress": 1700.0, "stress": 1000.0, "scarcity": 500.0}
    with pytest.raises(ValueError):
        falkenmark_category(-1.0)


def test_water_exploitation_index():
    assert water_exploitation_index(20.0, 100.0) == pytest.approx(0.2)
    assert water_exploitation_index(0.0, 100.0) == 0.0
    assert water_exploitation_index(5.0, 0.0) == math.inf
    assert water_exploitation_index(0.0, 0.0) == 0.0
    with pytest.raises(ValueError):
        water_exploitation_index(-1.0, 100.0)
    with pytest.raises(ValueError):
        water_exploitation_index(1.0, -100.0)


def test_sdg_642_water_stress():
    # FAO metadata example: 50 withdrawn of 200 renewable with 50 EFR -> 33.3 %
    assert sdg_642_water_stress(50.0, 200.0, 50.0) == pytest.approx(100.0 / 3.0)
    assert sdg_642_water_stress(0.0, 200.0, 50.0) == 0.0
    assert sdg_642_water_stress(10.0, 100.0, 100.0) == math.inf
    assert sdg_642_water_stress(10.0, 50.0, 100.0) == math.inf
    with pytest.raises(ValueError):
        sdg_642_water_stress(-1.0, 100.0, 0.0)


def test_dependency_ratio():
    assert dependency_ratio(30.0, 100.0) == pytest.approx(0.3)
    assert dependency_ratio(0.0, 100.0) == 0.0
    assert dependency_ratio(100.0, 100.0) == 1.0
    assert dependency_ratio(0.0, 0.0) == 0.0
    with pytest.raises(ValueError):
        dependency_ratio(10.0, 0.0)
    with pytest.raises(ValueError):
        dependency_ratio(150.0, 100.0)
    with pytest.raises(ValueError):
        dependency_ratio(-1.0, 100.0)


def test_groundwater_stress():
    assert groundwater_stress(2000.0, 1500.0) == pytest.approx(4.0 / 3.0)
    assert groundwater_stress(0.0, 0.0) == 0.0
    assert groundwater_stress(10.0, 0.0) == math.inf
    assert groundwater_stress(0.0, 100.0) == 0.0
    with pytest.raises(ValueError):
        groundwater_stress(-1.0, 1.0)


def test_hashimoto_criteria_closed_form():
    supplied = [10, 8, 10, 10, 6, 10]
    demanded = [10] * 6
    assert supply_reliability(supplied, demanded) == pytest.approx(4 / 6)
    assert resilience(supplied, demanded) == pytest.approx(1.0)
    assert vulnerability(supplied, demanded) == pytest.approx((0.2 + 0.4) / 2)


def test_hashimoto_no_failures():
    s = [10.0, 10.0, 10.0]
    assert supply_reliability(s, s) == 1.0
    assert resilience(s, s) == 1.0
    assert vulnerability(s, s) == 0.0


def test_resilience_counts_recoveries_per_failure():
    d = [1.0] * 4
    assert resilience([1, 0, 0, 1], d) == pytest.approx(0.5)      # 2 failures, 1 recovery
    assert resilience([1, 1, 1, 0], d) == pytest.approx(0.0)      # never recovers
    assert resilience([0, 0, 0, 0], d) == pytest.approx(0.0)
    assert resilience([0, 1, 0, 1], d) == pytest.approx(1.0)
    assert supply_reliability([0, 0, 0, 0], d) == 0.0
    assert vulnerability([0, 0.5, 1, 1], d) == pytest.approx(0.75)


def test_reliability_tolerance_and_zero_demand():
    assert supply_reliability([0.9995], [1.0]) == 1.0
    assert supply_reliability([0.998], [1.0]) == 0.0
    assert supply_reliability([0.0, 5.0], [0.0, 0.0]) == 1.0      # zero demand is never a failure
    assert vulnerability([0.0], [0.0]) == 0.0
    assert resilience([0.0], [0.0]) == 1.0


@pytest.mark.parametrize("func", [supply_reliability, resilience, vulnerability])
def test_hashimoto_validation(func):
    with pytest.raises(ValueError):
        func([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        func([], [])
    with pytest.raises(ValueError):
        func([-1.0], [1.0])
    with pytest.raises(ValueError):
        func([float("nan")], [1.0])


# ---------------------------------------------------------------------------
# routing: example basin properties
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ff", FLOW_FACTORS)
def test_mass_balance_example_basin(basin, ff):
    bal = route_basin(basin, flow_factor=ff)
    assert len(bal.reaches) == 3 and bal.names() == ["Highland", "Midland", "Delta"]
    assert bal.natural_flow_mm3 == pytest.approx(28_000.0 * ff)
    assert bal.mass_balance_error() <= 1e-6 * max(1.0, 28_000.0 * ff + 10_500.0)
    for name, err in bal.mass_balance_errors().items():
        reach = bal.reach(name)
        assert err <= 1e-6 * max(1.0, reach.inflow_mm3 + reach.storage_start_mm3)
    assert_physical(bal, basin)
    # chaining and inflow composition
    assert bal.reaches[0].upstream_inflow_mm3 == pytest.approx(2_000.0 * ff)
    for reach, rip in zip(bal.reaches, basin.riparians):
        assert reach.inflow_mm3 == pytest.approx(reach.upstream_inflow_mm3 + rip.local_inflow_mm3 * ff)
        assert reach.entitlement_mm3 == rip.treaty_allocation_mm3
        assert reach.environmental_flow_mm3 == rip.demand.environmental
        assert reach.demands == rip.demand.withdrawals()
        assert reach.storage_start_mm3 == rip.reservoir_storage_mm3


@pytest.mark.parametrize("ff", FLOW_FACTORS)
def test_storage_release_never_exceeds_storage(basin, ff):
    bal = route_basin(basin, flow_factor=ff)
    for reach in bal.reaches:
        assert 0.0 <= reach.storage_release_mm3 <= reach.storage_start_mm3 + 1e-9
    if ff == 0.0:
        # with no runoff the reservoirs are the only supply and are drawn down
        assert bal.reach("Delta").storage_release_mm3 == pytest.approx(6_000.0)
        assert bal.reach("Delta").total_withdrawal() > 0.0


def test_default_entitlements_delta_positive_outflow_and_cap_binds(basin):
    bal = route_basin(basin)
    delta = bal.reach("Delta")
    assert bal.outflow_to_sea_mm3 > 0.0
    assert delta.outflow_mm3 > delta.environmental_flow_mm3
    # Delta's river demand (18600 - 2300 non-river) exceeds its 16000 entitlement
    assert delta.entitlement_mm3 == 16_000.0
    assert delta.total_withdrawal() == pytest.approx(16_000.0)
    assert delta.total_deficit() == pytest.approx(18_600.0 - 2_300.0 - 16_000.0)
    assert delta.non_river_supply_mm3 == pytest.approx(2_300.0)
    assert delta.deficits[AGR] == pytest.approx(300.0)           # lowest priority bears the cut
    assert delta.deficits[MUN] == 0.0
    assert delta.supply_ratio() == pytest.approx(1.0 - 300.0 / 18_600.0)
    # upstream riparians are fully served and below their caps
    for name in ("Highland", "Midland"):
        reach = bal.reach(name)
        assert reach.total_deficit() == pytest.approx(0.0)
        assert reach.total_withdrawal() < reach.entitlement_mm3
    assert all(r.env_flow_met for r in bal.reaches)
    assert bal.total_withdrawal() == pytest.approx(1_500.0 + 7_350.0 + 16_000.0)
    assert bal.total_consumption() == pytest.approx(sum(r.total_consumption() for r in bal.reaches))


def test_entitlement_caps_bind(basin):
    capped = route_basin(basin, entitlements={"Highland": 500.0})
    high = capped.reach("Highland")
    assert high.entitlement_mm3 == 500.0
    assert high.total_withdrawal() == pytest.approx(500.0)
    assert high.total_deficit() == pytest.approx(1_500.0 - 500.0)
    # municipal is served first: 600 * (1 - 200/1700) = 529.4 > 500 -> only municipal
    assert high.withdrawals[MUN] == pytest.approx(500.0)
    assert high.withdrawals[AGR] == 0.0
    # removing the cap (None or inf) restores full service
    for no_cap in (None, math.inf):
        free = route_basin(basin, entitlements={"Highland": no_cap})
        assert free.reach("Highland").entitlement_mm3 == math.inf
        assert free.reach("Highland").total_withdrawal() == pytest.approx(1_500.0)
    # zero entitlement means no surface withdrawal at all
    zero = route_basin(basin, entitlements={"Highland": 0.0}).reach("Highland")
    assert zero.total_withdrawal() == 0.0
    assert zero.total_deficit() == pytest.approx(1_500.0)
    assert zero.outflow_mm3 > high.outflow_mm3 > route_basin(basin).reach("Highland").outflow_mm3


def test_treaty_zero_allocation_is_a_real_cap():
    basin = Basin("z", [make_riparian("A", mun=50.0, treaty=0.0)], headwater_inflow_mm3=100.0)
    reach = route_basin(basin).reach("A")
    assert reach.entitlement_mm3 == 0.0
    assert reach.total_withdrawal() == 0.0
    assert reach.outflow_mm3 == pytest.approx(100.0)


def test_env_flow_constraint_binds_consumption():
    basin = single(1000.0, ag=1000.0, env=800.0, fractions={AGR: 0.6})
    reach = route_basin(basin).reach("A")
    # consumption capped at supply - env = 200 -> withdrawal 200 / 0.6
    assert reach.total_consumption() == pytest.approx(200.0)
    assert reach.withdrawals[AGR] == pytest.approx(200.0 / 0.6)
    assert reach.outflow_mm3 == pytest.approx(800.0)
    assert reach.env_flow_met
    assert_reach_balance(reach)
    # the same riparian without the environmental requirement uses all its demand
    free = route_basin(basin, env_flows={"A": 0.0}).reach("A")
    assert free.withdrawals[AGR] == pytest.approx(1000.0)
    assert free.total_consumption() == pytest.approx(600.0)
    assert free.outflow_mm3 == pytest.approx(400.0)
    assert free.environmental_flow_mm3 == 0.0


def test_env_flow_unmet_when_supply_is_short():
    basin = single(100.0, mun=50.0, env=500.0)
    reach = route_basin(basin).reach("A")
    assert not reach.env_flow_met
    assert reach.outflow_mm3 == pytest.approx(100.0)
    assert reach.total_withdrawal() == 0.0      # consumption cap is zero
    assert reach.total_deficit() == pytest.approx(50.0)
    assert_reach_balance(reach)


def test_env_flow_topped_up_from_storage():
    basin = single(100.0, mun=50.0, env=500.0, capacity=2000.0, storage=1000.0, fractions={MUN: 0.2})
    reach = route_basin(basin).reach("A")
    assert reach.env_flow_met
    assert reach.outflow_mm3 == pytest.approx(500.0)
    assert reach.withdrawals[MUN] == pytest.approx(50.0)
    # river after use = 100 - 10 consumption = 90 -> 410 released for the environment
    assert reach.storage_release_mm3 == pytest.approx(410.0)
    assert reach.storage_refill_mm3 == 0.0
    assert reach.storage_end_mm3 == pytest.approx(1000.0 - 410.0)
    assert_reach_balance(reach)


def test_priority_order_municipal_before_agriculture():
    basin = single(100.0, mun=80.0, ag=80.0, fractions={MUN: 0.2, AGR: 0.6})
    reach = route_basin(basin).reach("A")
    assert reach.withdrawals[MUN] == pytest.approx(80.0)
    assert reach.withdrawals[AGR] == pytest.approx(20.0)
    assert reach.deficits[MUN] == 0.0
    assert reach.deficits[AGR] == pytest.approx(60.0)
    assert reach.total_consumption() == pytest.approx(80 * 0.2 + 20 * 0.6)
    assert reach.outflow_mm3 == pytest.approx(100.0 - 28.0)
    capped = route_basin(basin, entitlements={"A": 50.0}).reach("A")
    assert capped.withdrawals[MUN] == pytest.approx(50.0)
    assert capped.withdrawals[AGR] == 0.0


def test_full_priority_order_all_sectors():
    basin = single(100.0, mun=40.0, ind=40.0, energy=40.0, ag=40.0,
                   fractions={MUN: 0.0, IND: 0.0, ENE: 0.0, AGR: 0.0})
    reach = route_basin(basin).reach("A")
    assert reach.withdrawals[MUN] == pytest.approx(40.0)
    assert reach.withdrawals[IND] == pytest.approx(40.0)
    assert reach.withdrawals[ENE] == pytest.approx(20.0)
    assert reach.withdrawals[AGR] == 0.0
    assert reach.total_consumption() == 0.0
    assert reach.outflow_mm3 == pytest.approx(100.0)         # everything returns


def test_non_river_supply_offsets_demand_proportionally(basin):
    bal = route_basin(basin, flow_factor=3.0, entitlements={"Delta": None})
    delta = bal.reach("Delta")
    rip = basin.riparian("Delta")
    non_river = rip.groundwater_abstraction_mm3 + rip.energy.desalination_capacity_mm3
    total = rip.demand.total_withdrawal()
    assert delta.non_river_supply_mm3 == pytest.approx(non_river)
    for s, d in rip.demand.withdrawals().items():
        assert delta.withdrawals[s] == pytest.approx(d * (1.0 - non_river / total))
        assert delta.deficits[s] == pytest.approx(0.0)
    assert delta.total_withdrawal() == pytest.approx(total - non_river)
    assert delta.river_demand() == pytest.approx({s: d * (1 - non_river / total) for s, d in rip.demand.withdrawals().items()})
    assert delta.supply_ratio() == pytest.approx(1.0)


def test_non_river_supply_capped_at_demand():
    basin = single(10.0, mun=60.0, ag=40.0, gw=300.0, desal=200.0)
    reach = route_basin(basin).reach("A")
    assert reach.non_river_supply_mm3 == pytest.approx(100.0)
    assert reach.total_withdrawal() == 0.0
    assert reach.total_deficit() == 0.0
    assert reach.supply_ratio() == 1.0
    assert reach.outflow_mm3 == pytest.approx(10.0)
    # no demand at all -> no non-river supply "applied"
    none = route_basin(single(10.0, gw=300.0)).reach("A")
    assert none.non_river_supply_mm3 == 0.0


def test_reservoir_refill_limited_by_capacity(basin):
    high = route_basin(basin).reach("Highland")
    rip = basin.riparian("Highland")
    room = rip.reservoir_capacity_mm3 - rip.reservoir_storage_mm3
    surplus = high.inflow_mm3 - high.total_consumption() - high.environmental_flow_mm3
    assert 0.25 * surplus > room
    assert high.storage_refill_mm3 == pytest.approx(room)
    assert high.evaporation_mm3 == pytest.approx(rip.reservoir_capacity_mm3 * rip.reservoir_evaporation_fraction)
    assert high.storage_end_mm3 == pytest.approx(rip.reservoir_capacity_mm3 * (1 - rip.reservoir_evaporation_fraction))
    assert high.storage_end_mm3 <= rip.reservoir_capacity_mm3
    # a full reservoir takes nothing
    full = route_basin(basin, storages={"Highland": rip.reservoir_capacity_mm3}).reach("Highland")
    assert full.storage_refill_mm3 == 0.0
    assert full.outflow_mm3 == pytest.approx(full.inflow_mm3 - full.total_consumption())
    # 500 Mm3 of room -> exactly 500 stored
    part = route_basin(basin, storages={"Highland": rip.reservoir_capacity_mm3 - 500.0}).reach("Highland")
    assert part.storage_refill_mm3 == pytest.approx(500.0)


def test_reservoir_refill_fraction_of_surplus():
    basin = single(1000.0, mun=100.0, env=200.0, capacity=10_000.0, storage=0.0, fractions={MUN: 0.5})
    reach = route_basin(basin, reservoir_refill_fraction=0.25).reach("A")
    river = 1000.0 - 50.0
    assert reach.storage_refill_mm3 == pytest.approx(0.25 * (river - 200.0))
    assert reach.outflow_mm3 == pytest.approx(river - reach.storage_refill_mm3)
    assert reach.outflow_mm3 >= 200.0
    none = route_basin(basin, reservoir_refill_fraction=0.0).reach("A")
    assert none.storage_refill_mm3 == 0.0 and none.outflow_mm3 == pytest.approx(river)
    whole = route_basin(basin, reservoir_refill_fraction=1.0).reach("A")
    assert whole.storage_refill_mm3 == pytest.approx(river - 200.0)
    assert whole.outflow_mm3 == pytest.approx(200.0)


def test_missing_reservoir():
    basin = single(500.0, mun=100.0, fractions={MUN: 0.2})
    reach = route_basin(basin).reach("A")
    assert reach.storage_start_mm3 == reach.storage_end_mm3 == 0.0
    assert reach.storage_refill_mm3 == reach.storage_release_mm3 == reach.evaporation_mm3 == 0.0
    assert reach.outflow_mm3 == pytest.approx(500.0 - 20.0)
    assert_reach_balance(reach)


def test_storage_above_capacity_is_tolerated():
    basin = single(100.0, capacity=50.0, storage=80.0, evap=0.1)
    reach = route_basin(basin).reach("A")
    assert reach.storage_refill_mm3 == 0.0
    assert reach.storage_end_mm3 == pytest.approx(80.0 * 0.9)
    assert_reach_balance(reach)


def test_zero_flow_factor_without_storage_gives_zero_everything():
    basin = Basin("dry", [make_riparian("A", local=100.0, mun=10.0), make_riparian("B", local=50.0, ag=20.0)], headwater_inflow_mm3=30.0)
    bal = route_basin(basin, flow_factor=0.0)
    assert bal.natural_flow_mm3 == 0.0
    assert bal.outflow_to_sea_mm3 == 0.0
    assert bal.total_withdrawal() == 0.0
    assert bal.total_deficit() == pytest.approx(30.0)
    assert bal.supply_ratio() == 0.0
    for reach in bal.reaches:
        assert reach.inflow_mm3 == reach.outflow_mm3 == 0.0
        assert reach.supply_ratio() == 0.0
    assert bal.mass_balance_error() == 0.0


def test_zero_demand(basin):
    zero = {name: {s: 0.0 for s in SECTOR_PRIORITY} for name in basin.names()}
    bal = route_basin(basin, demands=zero)
    for reach in bal.reaches:
        assert reach.total_withdrawal() == 0.0
        assert reach.total_consumption() == 0.0
        assert reach.total_deficit() == 0.0
        assert reach.supply_ratio() == 1.0
        assert reach.non_river_supply_mm3 == 0.0
    assert bal.supply_ratio() == 1.0
    assert_physical(bal, basin)
    # an empty dict is also "zero demand"
    empty = route_basin(basin, demands={"Highland": {}})
    assert empty.reach("Highland").total_withdrawal() == 0.0
    assert empty.reach("Highland").demands == {s: 0.0 for s in SECTOR_PRIORITY}


def test_zero_demand_no_reservoir_passes_natural_flow():
    basin = Basin("nat", [make_riparian("A", local=100.0), make_riparian("B", local=50.0)], headwater_inflow_mm3=30.0)
    for ff in FLOW_FACTORS:
        bal = route_basin(basin, flow_factor=ff)
        assert bal.outflow_to_sea_mm3 == pytest.approx(180.0 * ff)
        assert bal.outflow_to_sea_mm3 == pytest.approx(bal.natural_flow_mm3)
        assert bal.reach("B").inflow_mm3 == pytest.approx(natural_flows(basin, ff)["B"])


def test_demands_accept_waterdemand_and_string_keys(basin):
    default = route_basin(basin)
    as_obj = route_basin(basin, demands={name: basin.riparian(name).demand for name in basin.names()})
    as_str = route_basin(basin, demands={name: {s.value: v for s, v in basin.riparian(name).demand.withdrawals().items()} for name in basin.names()})
    for alt in (as_obj, as_str):
        for a, b in zip(default.reaches, alt.reaches):
            assert a.to_dict() == pytest.approx(b.to_dict())


def test_upstream_priority_starves_downstream(basin):
    """No entitlements + demand far above flow: upstream consumes everything."""
    # deep copy so editing the sector dict can never leak into the shared fixture
    greedy = copy.deepcopy(basin)
    for name in ("Highland", "Midland"):
        greedy.riparian(name).demand.consumption_fraction[AGR] = 1.0
        assert basin.riparian(name).demand.consumption_fraction[AGR] == 0.6
    huge = {name: {AGR: 1e6} for name in ("Highland", "Midland")}
    no_caps = {name: None for name in greedy.names()}
    zero = {name: 0.0 for name in greedy.names()}
    unilateral = route_basin(greedy, flow_factor=0.1, entitlements=no_caps, demands=huge, storages=zero, env_flows=zero,
                             reservoir_refill_fraction=0.0)
    high, mid, delta = unilateral.reaches
    assert high.total_withdrawal() == pytest.approx(high.inflow_mm3)
    assert high.outflow_mm3 == pytest.approx(0.0, abs=1e-9)
    assert mid.outflow_mm3 == pytest.approx(0.0, abs=1e-9)
    assert delta.upstream_inflow_mm3 == pytest.approx(0.0, abs=1e-9)
    assert delta.inflow_mm3 == pytest.approx(100.0)
    assert delta.total_deficit() > 0.0
    assert delta.supply_ratio() < 0.2
    # only Delta's own runoff (minus its use and refill) reaches the sea
    assert 0.0 < unilateral.outflow_to_sea_mm3 < delta.inflow_mm3
    assert_physical(unilateral, greedy)
    # with entitlement caps in force the downstream riparian receives water
    caps = {"Highland": 500.0, "Midland": 300.0, "Delta": None}
    cooperative = route_basin(greedy, flow_factor=0.1, entitlements=caps, demands=huge, storages=zero, env_flows=zero,
                              reservoir_refill_fraction=0.0)
    assert cooperative.reach("Highland").total_withdrawal() == pytest.approx(500.0)
    assert cooperative.reach("Highland").outflow_mm3 == pytest.approx(2_000.0 - 500.0)
    assert cooperative.reach("Midland").total_withdrawal() == pytest.approx(300.0)
    assert cooperative.reach("Delta").upstream_inflow_mm3 == pytest.approx(1_500.0 + 700.0 - 300.0)
    assert cooperative.reach("Delta").supply_ratio() > delta.supply_ratio()
    assert cooperative.outflow_to_sea_mm3 > unilateral.outflow_to_sea_mm3


def test_upstream_priority_simple_chain():
    a = make_riparian("A", local=100.0, ag=500.0, fractions={AGR: 1.0})
    b = make_riparian("B", mun=50.0)
    bal = route_basin(Basin("chain", [a, b]))
    assert bal.reach("A").total_withdrawal() == pytest.approx(100.0)
    assert bal.reach("B").inflow_mm3 == 0.0
    assert bal.reach("B").total_deficit() == pytest.approx(50.0)
    assert bal.reach("B").supply_ratio() == 0.0


def test_storages_carry_between_years(basin):
    first = route_basin(basin)
    carried = first.storages_end()
    assert set(carried) == set(basin.names())
    second = route_basin(basin, storages=carried)
    for reach in second.reaches:
        assert reach.storage_start_mm3 == pytest.approx(carried[reach.name])
    assert_physical(second, basin)
    # multi-year run with a dry spell keeps the balance and bounds every year
    storages = None
    for ff in (1.0, 0.6, 0.2, 0.0, 0.0, 1.5):
        bal = route_basin(basin, flow_factor=ff, storages=storages)
        assert_physical(bal, basin)
        storages = bal.storages_end()


def test_local_inflow_factors_and_env_flow_overrides(basin):
    bal = route_basin(basin, local_inflow_factors={"Midland": 0.0})
    assert bal.natural_flow_mm3 == pytest.approx(28_000.0 - 7_000.0)
    mid = bal.reach("Midland")
    assert mid.inflow_mm3 == pytest.approx(mid.upstream_inflow_mm3)
    assert mid.inflow_mm3 == pytest.approx(bal.reach("Highland").outflow_mm3)
    assert natural_flows(basin, local_inflow_factors={"Midland": 0.0})["Delta"] == pytest.approx(21_000.0)
    env = route_basin(basin, env_flows={"Highland": 15_000.0})
    assert env.reach("Highland").environmental_flow_mm3 == 15_000.0
    assert env.reach("Highland").outflow_mm3 >= 15_000.0 - 1e-9
    assert env.reach("Highland").env_flow_met
    assert_physical(env, basin)


def test_monotonic_in_flow_factor(basin):
    grid = np.linspace(0.0, 2.0, 17)
    sea = [route_basin(basin, flow_factor=ff).outflow_to_sea_mm3 for ff in grid]
    withdrawn = [route_basin(basin, flow_factor=ff).total_withdrawal() for ff in grid]
    deficit = [route_basin(basin, flow_factor=ff).total_deficit() for ff in grid]
    assert all(b >= a - 1e-6 for a, b in zip(sea, sea[1:]))
    assert all(b >= a - 1e-6 for a, b in zip(withdrawn, withdrawn[1:]))
    assert all(b <= a + 1e-6 for a, b in zip(deficit, deficit[1:]))
    assert sea[-1] > sea[0]


def test_route_basin_does_not_mutate_inputs(basin):
    before = dataclasses.asdict(basin)
    entitlements = {"Highland": 500.0, "Delta": None}
    demands = {"Midland": {MUN: 10.0, AGR: 20.0}}
    storages = {"Delta": 100.0}
    env_flows = {"Highland": 10.0}
    factors = {"Midland": 0.5}
    args = [dict(entitlements), {k: dict(v) for k, v in demands.items()}, dict(storages), dict(env_flows), dict(factors)]
    bal = route_basin(basin, 0.7, entitlements, demands, storages, env_flows, factors, 0.1)
    # mutating the result must not leak back into the basin
    bal.reach("Midland").demands[MUN] = 999.0
    bal.reach("Midland").withdrawals[MUN] = 999.0
    assert dataclasses.asdict(basin) == before
    assert [entitlements, demands, storages, env_flows, factors] == args
    assert basin.riparian("Midland").demand.municipal == 1_500.0
    natural_flows(basin, 0.3)
    stochastic_flow_factors(Scenario(years=3, stochastic=True), basin)
    assert dataclasses.asdict(basin) == before


def test_route_basin_validation(basin):
    before = dataclasses.asdict(basin)
    with pytest.raises(ValueError):
        route_basin("not a basin")
    with pytest.raises(ValueError, match="flow_factor"):
        route_basin(basin, flow_factor=-0.1)
    with pytest.raises(ValueError, match="flow_factor"):
        route_basin(basin, flow_factor=float("nan"))
    with pytest.raises(ValueError):
        route_basin(basin, reservoir_refill_fraction=1.5)
    with pytest.raises(ValueError):
        route_basin(basin, reservoir_refill_fraction=-0.1)
    for kw in ("entitlements", "demands", "storages", "env_flows", "local_inflow_factors"):
        with pytest.raises(ValueError, match="unknown riparian"):
            route_basin(basin, **{kw: {"Atlantis": 1.0}})
        with pytest.raises(ValueError):
            route_basin(basin, **{kw: [1.0]})
    with pytest.raises(ValueError):
        route_basin(basin, demands={"Delta": {MUN: -1.0}})
    with pytest.raises(ValueError, match="not a withdrawal sector"):
        route_basin(basin, demands={"Delta": {Sector.ENVIRONMENT: 10.0}})
    with pytest.raises(ValueError, match="unknown sector"):
        route_basin(basin, demands={"Delta": {"fishing": 10.0}})
    with pytest.raises(ValueError):
        route_basin(basin, demands={"Delta": 5.0})
    with pytest.raises(ValueError):
        route_basin(basin, storages={"Delta": -1.0})
    with pytest.raises(ValueError):
        route_basin(basin, env_flows={"Delta": -1.0})
    with pytest.raises(ValueError):
        route_basin(basin, entitlements={"Delta": -1.0})
    with pytest.raises(ValueError):
        route_basin(basin, local_inflow_factors={"Delta": -1.0})
    # each riparian-attribute case gets a brand-new basin so that one bad value
    # can never be the (wrong) reason a later case raises; the message is pinned
    bad = example_basin()
    bad.riparian("Delta").demand.consumption_fraction[AGR] = 1.5
    with pytest.raises(ValueError, match=r"consumption_fraction\[agricultural\].*1\.5"):
        route_basin(bad)
    bad = example_basin()
    bad.riparian("Delta").reservoir_evaporation_fraction = -0.1
    with pytest.raises(ValueError, match=r"reservoir_evaporation_fraction.*-0\.1"):
        route_basin(bad)
    bad = example_basin()
    bad.riparian("Highland").local_inflow_mm3 = -5.0
    with pytest.raises(ValueError, match=r"local_inflow_mm3.*-5"):
        route_basin(bad)
    # none of the above touched the shared fixture
    assert dataclasses.asdict(basin) == before
    assert basin.riparian("Delta").demand.consumption_fraction[AGR] == 0.6
    assert basin.riparian("Delta").reservoir_evaporation_fraction == 0.05
    assert basin.riparian("Highland").local_inflow_mm3 == 18_000.0


BAD_FACTORS = [-0.1, float("nan"), float("inf"), -float("inf"), "abc", True, [1.0]]


@pytest.mark.parametrize("ff", BAD_FACTORS + [None])
def test_route_basin_rejects_bad_flow_factor(basin, ff):
    """Negative, NaN, +/-inf and non-numeric flow factors are refused up front.

    ``inf`` must be rejected explicitly: otherwise it would propagate an
    infinite inflow into every reach and ``inf - inf`` NaNs into storage.
    """
    with pytest.raises(ValueError, match="flow_factor"):
        route_basin(basin, flow_factor=ff)
    with pytest.raises(ValueError, match="flow_factor"):
        natural_flows(basin, ff)
    with pytest.raises(ValueError, match="flow_factor"):
        natural_flows(basin, flow_factor=ff)


@pytest.mark.parametrize("ff", BAD_FACTORS)
def test_route_basin_rejects_bad_local_inflow_factor(basin, ff):
    """Per-riparian inflow multipliers get the same screening as flow_factor."""
    with pytest.raises(ValueError, match=r"local_inflow_factors\['Delta'\]"):
        route_basin(basin, local_inflow_factors={"Delta": ff})
    with pytest.raises(ValueError, match=r"local_inflow_factors\['Delta'\]"):
        natural_flows(basin, local_inflow_factors={"Delta": ff})
    # a bad factor on one riparian is reported even when the others are fine
    with pytest.raises(ValueError, match=r"local_inflow_factors\['Midland'\]"):
        route_basin(basin, local_inflow_factors={"Highland": 1.0, "Midland": ff, "Delta": 0.5})


def test_large_finite_flow_factor_stays_finite(basin):
    """A huge but finite factor routes without NaN/inf anywhere (inf is rejected)."""
    bal = route_basin(basin, flow_factor=1e9)
    assert math.isfinite(bal.outflow_to_sea_mm3) and math.isfinite(bal.natural_flow_mm3)
    for reach in bal.reaches:
        for key, value in reach.to_dict().items():
            if isinstance(value, float):
                assert math.isfinite(value) or (key == "entitlement_mm3" and value == math.inf), key
    assert_physical(bal, basin)


def test_empty_basin_routes_headwater_to_sea():
    bal = route_basin(Basin("empty", [], headwater_inflow_mm3=10.0), flow_factor=2.0)
    assert bal.reaches == []
    assert bal.natural_flow_mm3 == bal.outflow_to_sea_mm3 == 20.0
    assert bal.mass_balance_error() == 0.0
    assert bal.env_flow_met_share() == 1.0
    assert bal.supply_ratio() == 1.0
    assert natural_flows(Basin("empty", [])) == {}


def test_result_helpers(basin):
    bal = route_basin(basin)
    with pytest.raises(KeyError):
        bal.reach("Nowhere")
    assert bal.env_flow_met_share() == 1.0
    assert bal.withdrawals_by_riparian() == {r.name: r.total_withdrawal() for r in bal.reaches}
    assert bal.total_demand() == pytest.approx(1_700.0 + 8_250.0 + 18_600.0)
    assert bal.supply_ratio() == pytest.approx(1.0 - bal.total_deficit() / bal.total_demand())
    d = bal.to_dict()
    assert d["outflow_to_sea_mm3"] == bal.outflow_to_sea_mm3
    assert len(d["reaches"]) == 3
    flat = bal.reach("Delta").to_dict()
    for s in SECTOR_PRIORITY:
        for prefix in ("withdrawal", "consumption", "deficit", "demand"):
            assert f"{prefix}_{s.value}" in flat
    assert flat["withdrawal_agricultural"] == bal.reach("Delta").withdrawals[AGR]
    assert flat["supply_ratio"] == bal.reach("Delta").supply_ratio()
    delta = bal.reach("Delta")
    assert delta.return_flow() == pytest.approx(delta.total_withdrawal() - delta.total_consumption())
    assert delta.total_demand() == pytest.approx(18_600.0)
    empty = ReachResult("x", 0, 0, 0, 0, 0, 0, 0, 0, math.inf)
    assert empty.supply_ratio() == 1.0
    assert empty.total_withdrawal() == 0.0 and empty.mass_balance_error() == 0.0


# ---------------------------------------------------------------------------
# natural flows
# ---------------------------------------------------------------------------
def test_natural_flows(basin):
    flows = natural_flows(basin)
    assert list(flows) == ["Highland", "Midland", "Delta"]
    assert flows == pytest.approx({"Highland": 20_000.0, "Midland": 27_000.0, "Delta": 28_000.0})
    assert flows["Delta"] == pytest.approx(basin.total_natural_flow())
    half = natural_flows(basin, 0.5)
    assert half == pytest.approx({k: v * 0.5 for k, v in flows.items()})
    assert natural_flows(basin, 0.0) == {"Highland": 0.0, "Midland": 0.0, "Delta": 0.0}
    with pytest.raises(ValueError, match="flow_factor"):
        natural_flows(basin, -1.0)
    with pytest.raises(ValueError):
        natural_flows(basin, local_inflow_factors={"Nowhere": 1.0})
    with pytest.raises(ValueError):
        natural_flows("basin")


def test_natural_flows_match_routing_without_use(basin):
    zero = {name: {s: 0.0 for s in SECTOR_PRIORITY} for name in basin.names()}
    no_storage = {name: 0.0 for name in basin.names()}
    no_res = basin.copy()
    for r in no_res.riparians:
        r.reservoir_capacity_mm3 = 0.0
    bal = route_basin(no_res, 0.8, demands=zero, storages=no_storage)
    for name, flow in natural_flows(basin, 0.8).items():
        assert bal.reach(name).inflow_mm3 == pytest.approx(flow)


# ---------------------------------------------------------------------------
# stochastic flow factors
# ---------------------------------------------------------------------------
def test_stochastic_flow_factors_deterministic_trend(basin, baseline):
    factors = stochastic_flow_factors(baseline, basin)
    assert factors == [baseline.flow_factor(i) for i in range(baseline.years)]
    trend = Scenario(years=11, flow_change_pct_by_end=-20, drought_years=[5], drought_severity=0.5)
    assert stochastic_flow_factors(trend, basin) == [trend.flow_factor(i) for i in range(11)]
    assert stochastic_flow_factors(trend, basin)[10] == pytest.approx(0.8)
    assert stochastic_flow_factors(trend, basin)[5] == pytest.approx(0.45)
    assert stochastic_flow_factors(Scenario(years=0, stochastic=True), basin) == []


def test_stochastic_flow_factors_seeded(basin):
    s1 = Scenario(years=30, stochastic=True, seed=7)
    a = stochastic_flow_factors(s1, basin)
    b = stochastic_flow_factors(s1, basin)
    assert a == b
    assert len(a) == 30
    assert all(f > 0.0 for f in a)
    assert a != [1.0] * 30
    other = stochastic_flow_factors(Scenario(years=30, stochastic=True, seed=8), basin)
    assert other != a
    # numpy's generator is the reference implementation
    rng = np.random.default_rng(7)
    sigma = math.sqrt(math.log(1 + 0.2 ** 2))
    expected = rng.lognormal(-0.5 * sigma ** 2, sigma, 30)
    assert a == pytest.approx(list(expected))


def test_stochastic_flow_factors_mean_matches_trend(basin):
    n = 20_000
    flat = stochastic_flow_factors(Scenario(years=n, stochastic=True, seed=3), basin)
    arr = np.asarray(flat)
    assert abs(arr.mean() - 1.0) < 0.01
    assert abs(arr.std() / arr.mean() - basin.climate_cv) < 0.02
    # with a trend and droughts the expectation follows scenario.flow_factor(i)
    sc = Scenario(years=n, stochastic=True, seed=11, flow_change_pct_by_end=-30, drought_years=[100, 2000], drought_severity=0.5)
    trend = np.asarray([sc.flow_factor(i) for i in range(n)])
    draws = np.asarray(stochastic_flow_factors(sc, basin))
    ratio = draws / trend
    assert abs(ratio.mean() - 1.0) < 0.01
    # closed form for the drought years: the linear -30 % trend at year i is
    # 1 - 0.3 * i / (n - 1), halved by the drought, times the (i + 1)-th draw of
    # the seeded reference generator used in test_stochastic_flow_factors_seeded
    sigma = math.sqrt(math.log(1.0 + basin.climate_cv ** 2))
    reference = np.random.default_rng(11).lognormal(-0.5 * sigma ** 2, sigma, n)
    for i in (100, 2000):
        assert trend[i] == pytest.approx(0.5 * (1.0 - 0.3 * i / (n - 1)))
        assert draws[i] == pytest.approx(0.5 * (1.0 - 0.3 * i / (n - 1)) * reference[i], rel=1e-12)
        assert ratio[i] == pytest.approx(reference[i], rel=1e-12)
    # a non-drought year carries the full trend times its own draw
    assert draws[101] == pytest.approx((1.0 - 0.3 * 101 / (n - 1)) * reference[101], rel=1e-12)
    # correlation with the trend: late years are drier on average
    assert draws[: n // 2].mean() > draws[n // 2:].mean()


def test_stochastic_flow_factors_zero_cv_and_validation(basin):
    calm = basin.copy()
    calm.climate_cv = 0.0
    sc = Scenario(years=5, stochastic=True)
    assert stochastic_flow_factors(sc, calm) == [sc.flow_factor(i) for i in range(5)]
    bad = basin.copy()
    bad.climate_cv = -0.1
    with pytest.raises(ValueError):
        stochastic_flow_factors(sc, bad)
    with pytest.raises(ValueError):
        stochastic_flow_factors("scenario", basin)
    with pytest.raises(ValueError):
        stochastic_flow_factors(sc, "basin")
    with pytest.raises(ValueError):
        stochastic_flow_factors(Scenario(years=-1), basin)
