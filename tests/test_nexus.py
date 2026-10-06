"""Tests for :mod:`wefnexus.nexus` - the integrated multi-year WEF nexus model.

Covers the ARCHITECTURE.md section-7 contract: ``RiparianYear``,
``NexusResult`` (access, series, export, summary), ``NexusModel``
(validation, drivers, entitlements, run) and ``run_nexus``; property checks
(bounds, mass conservation, storage carry-over, monotone scenario effects),
closed-form checks against the leaf modules, edge cases (zero flow, zero
demand, no reservoir, no crops, zero population, single riparian) and
no-mutation guarantees.  The last section holds regressions for the review
fixes: groundwater pumping energy, the two renewable-share definitions, the
consumptive bankruptcy estate that counts carried storage, and scenario
driver validation.
"""
from __future__ import annotations

import csv
import dataclasses
import doctest
import math

import numpy as np
import pytest

import wefnexus.nexus as nexus_module
from wefnexus import allocation, energy, food, sustainability, water
from wefnexus.models import (
    SECTOR_PRIORITY,
    Basin,
    Crop,
    EnergySystem,
    Riparian,
    Scenario,
    Sector,
    WaterDemand,
)
from wefnexus.nexus import (
    ALLOCATION_RULES,
    BANKRUPTCY_RULES,
    BASIN_KEY,
    RIPARIAN_YEAR_FIELDS,
    SERIES_AGGREGATORS,
    WITHDRAWAL_SECTORS,
    NexusModel,
    NexusResult,
    RiparianYear,
    run_nexus,
)

MUN, IND, ENE, AGR = Sector.MUNICIPAL, Sector.INDUSTRIAL, Sector.ENERGY, Sector.AGRICULTURAL
INF = math.inf
TOL = 1e-9


# ---------------------------------------------------------------------------
# helpers and fixtures
# ---------------------------------------------------------------------------
def _is_finite(v) -> bool:
    return not isinstance(v, float) or math.isfinite(v)


def spec_record(**kw) -> RiparianYear:
    """A RiparianYear built from the section-7 contract fields only."""
    base = dict(
        name="A", year=2025, population=1e6, gdp_usd=2e9,
        inflow_mm3=1000.0, upstream_inflow_mm3=400.0, outflow_mm3=800.0, storage_end_mm3=100.0,
        entitlement_mm3=INF,
        demands={MUN: 100.0, AGR: 50.0}, withdrawals={MUN: 100.0, AGR: 25.0}, deficits={MUN: 0.0, AGR: 25.0},
        environmental_flow_mm3=300.0, env_flow_met=True,
        per_capita_water_m3=2000.0, falkenmark="no_stress", water_stress_sdg642=20.0, groundwater_stress=0.4,
        hydropower_gwh=40.0, thermal_gwh=20.0, other_renewable_gwh=20.0, energy_supply_gwh=80.0,
        energy_demand_gwh=100.0, energy_for_water_gwh=5.0, energy_deficit_gwh=20.0, emissions_t=800.0,
        food_production_t=1e5, food_kcal=1000.0, food_demand_kcal=1000.0, food_self_sufficiency=1.0,
        et_ratio=1.0, irrigation_requirement_mm3=50.0, crop_value_usd=1e7, water_value_usd=1.5e8,
        water_security=0.8, energy_security=0.5, food_security=1.0, nexus_index=0.6,
    )
    base.update(kw)
    return RiparianYear(**base)


def solo_riparian(**changes) -> Riparian:
    """A single riparian without crops, reservoir or groundwater use."""
    r = Riparian(
        name="Solo",
        population=1_000_000,
        gdp_usd=1e9,
        local_inflow_mm3=1000.0,
        demand=WaterDemand(municipal=100.0, industrial=50.0, agricultural=200.0, energy=10.0, environmental=300.0),
        groundwater_recharge_mm3=100.0,
        groundwater_abstraction_mm3=0.0,
        energy=EnergySystem(
            demand_gwh=1000.0,
            hydropower_capacity_mw=50.0,
            hydropower_head_m=100.0,
            turbine_efficiency=0.9,
            turbined_fraction=1.0,
            thermal_capacity_mw=100.0,
            thermal_capacity_factor=0.5,
            renewable_share=0.2,
            pumped_fraction=0.0,
        ),
        reservoir_capacity_mm3=0.0,
        reservoir_storage_mm3=0.0,
    )
    return dataclasses.replace(r, **changes)


def solo_basin(**changes) -> Basin:
    return Basin(name="Solo basin", riparians=[solo_riparian(**changes)], headwater_inflow_mm3=500.0)


def tight_treaty_basin(basin: Basin) -> Basin:
    """Example basin with upstream entitlements below upstream demand."""
    b = basin.copy()
    b.riparian("Highland").treaty_allocation_mm3 = 800.0
    b.riparian("Midland").treaty_allocation_mm3 = 4000.0
    return b


def fractions_of(r: Riparian) -> dict:
    return {s: r.demand.consumption_fraction.get(s, 0.0) for s in WITHDRAWAL_SECTORS}


def river_demand_of(r: Riparian, demand: dict) -> dict:
    """Demand on the river per sector after the proportional non-river offset (route_basin convention)."""
    total = sum(demand.values())
    if total <= 0.0:
        return {s: 0.0 for s in WITHDRAWAL_SECTORS}
    non_river = min(r.groundwater_abstraction_mm3 + r.energy.desalination_capacity_mm3, total)
    return {s: demand.get(s, 0.0) * (1.0 - non_river / total) for s in WITHDRAWAL_SECTORS}


def consumptive_claim_of(r: Riparian, demand: dict) -> float:
    rd = river_demand_of(r, demand)
    return sum(rd[s] * f for s, f in fractions_of(r).items())


def consumption_at_cap(r: Riparian, river_demand: dict, cap: float) -> float:
    """Consumption when sectors are served in priority order under a withdrawal cap (as route_basin does)."""
    remaining, consumed = cap, 0.0
    for s in SECTOR_PRIORITY:
        w = min(river_demand[s], remaining)
        consumed += w * r.demand.consumption_fraction.get(s, 0.0)
        remaining -= w
    return consumed


def example_estate(flow_factor: float, basin: Basin, storages: dict = None) -> float:
    """Bankruptcy estate of the example basin: natural flow + start storage - environmental flows."""
    stored = sum(r.reservoir_storage_mm3 for r in basin) if storages is None else sum(storages.values())
    return flow_factor * 28000.0 + stored - sum(r.demand.environmental for r in basin)


@pytest.fixture
def result(basin, baseline) -> NexusResult:
    return run_nexus(basin, baseline)


# ---------------------------------------------------------------------------
# module hygiene
# ---------------------------------------------------------------------------
def test_module_imports_lazily_and_exports_everything():
    with open(nexus_module.__file__, encoding="utf-8") as fh:
        src = fh.read()
    top = [ln for ln in src.splitlines() if ln.startswith(("import ", "from "))]
    assert not any(("pandas" in ln) or ("matplotlib" in ln) or ("scipy" in ln) for ln in top)
    assert "import pandas" in src  # only inside to_dataframe
    for name in nexus_module.__all__:
        assert hasattr(nexus_module, name), name
    # everything sustainability.assess() reads is a contract field; 'falkenmark' is
    # part of the section-7 record but not read by assess()
    assert set(sustainability.RIPARIAN_YEAR_FIELDS) <= set(RIPARIAN_YEAR_FIELDS)
    assert set(RIPARIAN_YEAR_FIELDS) - set(sustainability.RIPARIAN_YEAR_FIELDS) == {"falkenmark"}
    assert WITHDRAWAL_SECTORS == tuple(SECTOR_PRIORITY)
    assert "treaty" in ALLOCATION_RULES and "upstream_priority" in ALLOCATION_RULES
    assert set(BANKRUPTCY_RULES) == {"proportional", "cea", "cel", "talmud", "ap", "equal"}
    assert set(SERIES_AGGREGATORS) == {"sum", "mean", "min", "max", "any", "all"}
    assert BASIN_KEY == "BASIN"


def test_doctests_pass():
    failed, attempted = doctest.testmod(nexus_module)
    assert attempted > 0 and failed == 0


# ---------------------------------------------------------------------------
# RiparianYear
# ---------------------------------------------------------------------------
def test_riparian_year_contract_fields_and_defaults():
    rec = spec_record()
    for fld in RIPARIAN_YEAR_FIELDS:
        assert hasattr(rec, fld), fld
    # extras have defaults
    assert rec.storage_start_mm3 == 0.0 and rec.consumption == {} and rec.flow_factor == 1.0
    assert rec.total_demand() == 150.0 and rec.total_withdrawal() == 125.0 and rec.total_deficit() == 25.0
    assert rec.supply_ratio() == pytest.approx(1.0 - 25.0 / 150.0)
    assert rec.total_consumption() == 0.0
    assert spec_record(demands={}, deficits={}).supply_ratio() == 1.0


def test_riparian_year_to_dict_is_flat():
    rec = spec_record(consumption={MUN: 20.0})
    d = rec.to_dict()
    for fld in RIPARIAN_YEAR_FIELDS:
        if fld in ("demands", "withdrawals", "deficits"):
            assert fld not in d
        else:
            assert d[fld] == getattr(rec, fld)
    for s in WITHDRAWAL_SECTORS:
        for prefix in ("demand", "withdrawal", "deficit", "consumption"):
            assert f"{prefix}_{s.value}" in d
    assert d["withdrawal_municipal"] == 100.0 and d["withdrawal_agricultural"] == 25.0
    assert d["withdrawal_industrial"] == 0.0  # missing sector -> 0
    assert d["consumption_municipal"] == 20.0
    assert d["total_demand_mm3"] == 150.0 and d["supply_ratio"] == pytest.approx(rec.supply_ratio())
    assert all(not isinstance(v, (dict, list, tuple)) for v in d.values())


def test_riparian_year_validation_and_copies():
    with pytest.raises(ValueError, match="name"):
        spec_record(name="")
    with pytest.raises(ValueError, match="year"):
        spec_record(year="2025")
    with pytest.raises(ValueError, match="year"):
        spec_record(year=True)
    with pytest.raises(ValueError, match="year"):
        spec_record(year=2025.5)
    assert spec_record(year=np.int64(2030)).year == 2030 and isinstance(spec_record(year=np.int64(2030)).year, int)
    with pytest.raises(ValueError, match="unknown sector"):
        spec_record(demands={"steel": 1.0})
    with pytest.raises(ValueError, match="not a withdrawal sector"):
        spec_record(withdrawals={Sector.ENVIRONMENT: 1.0})
    with pytest.raises(ValueError, match="must be a dict"):
        spec_record(deficits=[1.0])
    with pytest.raises(ValueError, match="falkenmark"):
        spec_record(falkenmark=3)
    # string sector keys are accepted and normalised; input dicts are copied
    src = {"municipal": 5.0}
    rec = spec_record(demands=src)
    assert rec.demands == {MUN: 5.0}
    src["municipal"] = 99.0
    assert rec.demands[MUN] == 5.0


# ---------------------------------------------------------------------------
# NexusModel construction and validation
# ---------------------------------------------------------------------------
def test_model_rejects_bad_inputs(basin, baseline):
    with pytest.raises(ValueError, match="Basin"):
        NexusModel("basin", baseline)
    with pytest.raises(ValueError, match="Scenario"):
        NexusModel(basin, {"years": 5})
    with pytest.raises(ValueError, match="years"):
        NexusModel(basin, Scenario(years=0))
    with pytest.raises(ValueError, match="years"):
        NexusModel(basin, Scenario(years=2.5))
    with pytest.raises(ValueError, match="allocation rule"):
        NexusModel(basin, baseline, allocation_rule="lottery")
    with pytest.raises(ValueError, match="allocation rule"):
        NexusModel(basin, baseline, allocation_rule=3)
    with pytest.raises(ValueError, match="reservoir_refill_fraction"):
        NexusModel(basin, baseline, reservoir_refill_fraction=1.5)
    with pytest.raises(ValueError, match="flow_factors"):
        NexusModel(basin, baseline, flow_factors=[1.0, 1.0])
    with pytest.raises(ValueError, match="flow_factors"):
        NexusModel(basin, baseline, flow_factors=[1.0, 1.0, 1.0, -0.1, 1.0])
    with pytest.raises(ValueError, match="flow_factors"):
        NexusModel(basin, baseline, flow_factors="abcde")
    with pytest.raises(ValueError, match="no riparians"):
        NexusModel(Basin(name="empty", riparians=[]), baseline)
    bad = basin.copy()
    bad.riparian("Delta").energy.turbined_fraction = 1.2
    with pytest.raises(ValueError, match="turbined_fraction"):
        NexusModel(bad, baseline)
    bad = basin.copy()
    bad.riparian("Delta").population = -1.0
    with pytest.raises(ValueError, match="population"):
        NexusModel(bad, baseline)
    model = NexusModel(basin, baseline)
    for idx in (-1, 5, 1.0, True, "0"):
        with pytest.raises(ValueError, match="year_index"):
            model.drivers(idx)
        with pytest.raises(ValueError, match="year_index"):
            model.entitlements(idx, 1000.0)
    with pytest.raises(ValueError, match="natural_flow"):
        model.entitlements(0, -5.0)
    with pytest.raises(ValueError, match="flow_factor"):
        model.natural_flow(0, flow_factor=-1.0)


def test_rule_resolution_and_cooperation_switch(basin):
    assert NexusModel(basin, Scenario(years=1)).allocation_rule == "treaty"
    assert NexusModel(basin, Scenario(years=1, allocation_rule="TREATY")).allocation_rule == "treaty"
    assert NexusModel(basin, Scenario(years=1), allocation_rule="Talmud").allocation_rule == "talmud"
    assert NexusModel(basin, Scenario(years=1), allocation_rule="contested-garment").allocation_rule == "talmud"
    assert NexusModel(basin, Scenario(years=1), allocation_rule="adjusted_proportional").allocation_rule == "ap"
    assert NexusModel(basin, Scenario(years=1), allocation_rule="upstream").allocation_rule == "upstream_priority"
    assert NexusModel(basin, Scenario(years=1, allocation_rule=None)).allocation_rule == "upstream_priority"
    # the model argument overrides the scenario rule
    assert NexusModel(basin, Scenario(years=1, allocation_rule="cea"), allocation_rule="cel").allocation_rule == "cel"
    # cooperation False forces upstream priority whatever the rule
    m = NexusModel(basin, Scenario(years=1, cooperation=False, allocation_rule="talmud"))
    assert m.allocation_rule == "upstream_priority" and m.requested_rule == "talmud"
    assert m.years == [2025]
    assert NexusModel(basin, Scenario(years=3, start_year=2030)).years == [2030, 2031, 2032]


# ---------------------------------------------------------------------------
# drivers and demands
# ---------------------------------------------------------------------------
def test_demands_year0_match_crop_requirement_and_base_values(basin, baseline):
    model = NexusModel(basin, baseline)
    d = model.demands(0)
    assert list(d) == basin.names()
    for r in basin.riparians:
        assert d[r.name][MUN] == r.demand.municipal
        assert d[r.name][IND] == r.demand.industrial
        assert d[r.name][ENE] == r.demand.energy
        assert d[r.name][AGR] == pytest.approx(food.riparian_irrigation_requirement_mm3(r))
    assert d["Highland"][AGR] == pytest.approx(798.0)  # wheat 630 + maize 168 (food docstring)
    drv = model.drivers(0)["Highland"]
    assert drv["population"] == 8_000_000 and drv["gdp_usd"] == 40e9
    assert drv["irrigation_requirement_mm3"] == pytest.approx(798.0)
    assert drv["irrigated_area_ha"] == 200_000 and drv["irrigation_efficiency"] == 0.5
    assert drv["energy_demand_gwh"] == 12_000.0 and drv["renewable_share"] == 0.75
    # returned dicts are fresh copies
    d["Highland"][MUN] = -1.0
    assert model.demands(0)["Highland"][MUN] == 600.0


def test_demands_follow_scenario_growth_factors(basin):
    sc = Scenario(
        years=5,
        population_growth_rate=0.02,
        gdp_growth_rate=0.03,
        demand_growth_rate=0.10,
        energy_demand_growth_rate=0.05,
        irrigated_area_change_pct_by_end=50.0,
        irrigation_efficiency_target=0.8,
        renewable_share_target=0.9,
    )
    model = NexusModel(basin, sc)
    r = basin.riparian("Midland")
    d0, d2, d4 = model.demands(0), model.demands(2), model.demands(4)
    assert d2["Midland"][MUN] == pytest.approx(r.demand.municipal * 1.1 ** 2)
    assert d2["Midland"][IND] == pytest.approx(r.demand.industrial * 1.1 ** 2)
    assert d2["Midland"][ENE] == pytest.approx(r.demand.energy * 1.1 ** 2)
    # agricultural: area factor 1.5 at the end, efficiency 0.5 -> 0.8 (lower gross demand per ha)
    assert d4["Midland"][AGR] == pytest.approx(food.riparian_irrigation_requirement_mm3(r, 1.5, 0.8))
    assert d4["Midland"][AGR] < d0["Midland"][AGR] * 1.5  # efficiency gain partly offsets the area growth
    drv0, drv4 = model.drivers(0)["Midland"], model.drivers(4)["Midland"]
    assert drv4["population"] == pytest.approx(r.population * 1.02 ** 4)
    assert drv4["gdp_usd"] == pytest.approx(r.gdp_usd * 1.03 ** 4)
    assert drv4["energy_demand_gwh"] == pytest.approx(r.energy.demand_gwh * 1.05 ** 4)
    assert drv4["renewable_share"] == pytest.approx(0.9) and drv0["renewable_share"] == pytest.approx(0.25)
    assert drv4["irrigation_efficiency"] == pytest.approx(0.8) and drv4["area_factor"] == pytest.approx(1.5)
    assert drv4["irrigated_area_ha"] == pytest.approx(r.irrigated_area_ha() * 1.5)


def test_demands_without_crops_scale_agricultural_demand_by_area_factor():
    b = solo_basin()
    model = NexusModel(b, Scenario(years=3, irrigated_area_change_pct_by_end=-50.0, demand_growth_rate=0.5))
    assert model.demands(0)["Solo"][AGR] == 200.0
    assert model.demands(2)["Solo"][AGR] == pytest.approx(100.0)  # area factor only, no demand growth
    assert model.demands(2)["Solo"][MUN] == pytest.approx(100.0 * 1.5 ** 2)
    assert model.drivers(2)["Solo"]["irrigation_requirement_mm3"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# entitlements
# ---------------------------------------------------------------------------
def test_entitlements_treaty_fixed_volumes_and_none_without_entitlement(basin):
    model = NexusModel(basin, Scenario(years=2))
    ent = model.entitlements(0, 28000.0)
    assert ent == {"Highland": 2500.0, "Midland": 9000.0, "Delta": 16000.0}
    assert model.entitlements(1, 1.0) == ent  # fixed, not scaled with the flow
    b = basin.copy()
    b.riparian("Midland").treaty_allocation_mm3 = None
    ent2 = NexusModel(b, Scenario(years=2)).entitlements(0, 28000.0)
    assert ent2["Midland"] is None and ent2["Highland"] == 2500.0


def test_entitlements_none_without_cooperation_or_under_upstream_priority(basin):
    assert NexusModel(basin, Scenario(years=1, cooperation=False)).entitlements(0, 28000.0) is None
    assert NexusModel(basin, Scenario(years=1, cooperation=False, allocation_rule="talmud")).entitlements(0, 1.0) is None
    assert NexusModel(basin, Scenario(years=1, allocation_rule="upstream_priority")).entitlements(0, 28000.0) is None
    assert NexusModel(basin, Scenario(years=1, allocation_rule=None)).entitlements(0, 28000.0) is None


@pytest.mark.parametrize("rule", BANKRUPTCY_RULES)
def test_entitlements_bankruptcy_rules_are_valid_awards(basin, rule):
    model = NexusModel(basin, Scenario(years=2, allocation_rule=rule))
    natural = model.natural_flow(0)
    assert natural == pytest.approx(28000.0)
    demands = model.demands(0)
    claims = model.claims(0)
    river_demand = {r.name: river_demand_of(r, demands[r.name]) for r in basin}
    # the estate counts the carried storage: natural flow + storage - environmental flows
    estate = example_estate(1.0, basin)
    ent = model.entitlements(0, natural)
    assert list(ent) == basin.names()
    # caps are gross surface withdrawals bounded by the river demand (demand net of non-river supply)
    for name in claims:
        assert -TOL <= ent[name] <= sum(river_demand[name].values()) + TOL
    # the consumption each cap allows equals the award of the allocation module
    direct = allocation.apply_rule(rule, estate, claims)
    for r in basin:
        assert consumption_at_cap(r, river_demand[r.name], ent[r.name]) == pytest.approx(direct[r.name], abs=1e-6)
    # no scarcity in the base year: every claim fits, so nobody is capped below its river demand
    assert sum(claims.values()) < estate
    for name in claims:
        assert ent[name] == pytest.approx(sum(river_demand[name].values()))
    # a year with no water beyond the EFR and empty reservoirs gives zero caps
    env_total = sum(r.demand.environmental for r in basin)
    dry = model.entitlements(0, env_total * 0.5, storages={n: 0.0 for n in basin.names()})
    assert all(v == 0.0 for v in dry.values())
    # a scarce year: the awards exhaust the estate and somebody is capped below its river demand
    scarce_flow = 0.2 * natural
    scarce = model.entitlements(0, scarce_flow)
    scarce_estate = example_estate(0.2, basin)
    assert sum(claims.values()) > scarce_estate
    awards = allocation.apply_rule(rule, scarce_estate, claims)
    assert sum(awards.values()) == pytest.approx(scarce_estate)
    for r in basin:
        assert -TOL <= scarce[r.name] <= sum(river_demand[r.name].values()) + TOL
        assert consumption_at_cap(r, river_demand[r.name], scarce[r.name]) == pytest.approx(awards[r.name], abs=1e-6)
    assert any(scarce[n] < sum(river_demand[n].values()) - 1e-6 for n in claims)
    # explicit demands form the claims (net of the non-river supply); unknown / missing riparians rejected
    custom = {name: {MUN: 5000.0} for name in basin.names()}
    custom_caps = model.entitlements(0, natural, demands=custom)
    for r in basin:
        non_river = min(r.groundwater_abstraction_mm3 + r.energy.desalination_capacity_mm3, 5000.0)
        assert custom_caps[r.name] == pytest.approx(5000.0 - non_river)
    with pytest.raises(ValueError, match="unknown riparian"):
        model.entitlements(0, natural, demands={**custom, "Atlantis": {MUN: 1.0}})
    with pytest.raises(ValueError, match="lacks"):
        model.entitlements(0, natural, demands={"Highland": {MUN: 1.0}})
    with pytest.raises(ValueError, match="demands"):
        model.entitlements(0, natural, demands=[1, 2, 3])


def test_entitlements_talmud_closed_form(basin):
    """Talmud on consumptive claims with E > C/2: c/2 + CEL(E - C/2, c/2) (Aumann & Maschler 1985)."""
    model = NexusModel(basin, Scenario(years=1, allocation_rule="talmud"))
    demands = model.demands(0)
    claims = model.claims(0)
    for r in basin:
        assert claims[r.name] == pytest.approx(consumptive_claim_of(r, demands[r.name]))
    total = sum(claims.values())
    # river demands (1498, 7352.7, 16339.5) weighted by the sector consumption fractions
    assert total == pytest.approx(11491.6, abs=1.0)
    # a scarce year: natural flow 5600 + 10500 of storage - 7000 of environmental flows
    estate = example_estate(0.2, basin)
    assert estate == pytest.approx(9100.0)
    assert total / 2 < estate < total
    half = {n: c / 2 for n, c in claims.items()}
    cel = allocation.constrained_equal_losses(estate - total / 2, half)
    awards = {n: half[n] + cel[n] for n in claims}
    assert awards == pytest.approx(allocation.talmud(estate, claims))
    ent = model.entitlements(0, 0.2 * 28000.0)
    for r in basin:
        rd = river_demand_of(r, demands[r.name])
        assert ent[r.name] == pytest.approx(nexus_module._withdrawal_cap_for_award(rd, fractions_of(r), awards[r.name]))
        assert consumption_at_cap(r, rd, ent[r.name]) == pytest.approx(awards[r.name], abs=1e-6)
    # Highland, the smallest claimant, keeps exactly half its claim: the cap serves municipal,
    # industrial and energy in full and the rest of the award goes to agriculture at f = 0.6
    h = basin.riparian("Highland")
    rd = river_demand_of(h, demands["Highland"])
    assert awards["Highland"] == pytest.approx(claims["Highland"] / 2)
    served_first = rd[MUN] + rd[IND] + rd[ENE]
    consumed_first = rd[MUN] * 0.2 + rd[IND] * 0.1 + rd[ENE] * 0.03
    assert ent["Highland"] == pytest.approx(served_first + (awards["Highland"] - consumed_first) / 0.6)
    assert ent["Highland"] == pytest.approx(1040.9, abs=0.1)


def test_claims_are_consumptive_river_demand(basin):
    model = NexusModel(basin, Scenario(years=2, allocation_rule="proportional"))
    claims = model.claims(0)
    assert list(claims) == basin.names()
    demands = model.demands(0)
    for r in basin:
        rd = river_demand_of(r, demands[r.name])
        assert claims[r.name] == pytest.approx(sum(rd[s] * r.demand.consumption_fraction[s] for s in WITHDRAWAL_SECTORS))
        assert 0.0 < claims[r.name] < sum(rd.values()) < sum(demands[r.name].values())
    # Delta: 2300 Mm3 of groundwater + desalination come off the gross demand before the
    # demand-weighted consumption fraction applies
    d = basin.riparian("Delta")
    gross = sum(demands["Delta"].values())
    f_bar = sum(demands["Delta"][s] * d.demand.consumption_fraction[s] for s in WITHDRAWAL_SECTORS) / gross
    assert claims["Delta"] == pytest.approx((gross - 2300.0) * f_bar)
    # explicit demands; a demand entirely covered by non-river supply claims nothing
    small = {n: {MUN: 10.0} for n in basin.names()}
    assert model.claims(0, demands=small) == {n: 0.0 for n in basin.names()}
    custom = {n: {MUN: 1000.0, AGR: 1000.0} for n in basin.names()}
    c = model.claims(0, demands=custom)
    for r in basin:
        # the non-river supply is capped at the demand (Delta's 2300 Mm3 covers the 2000 entirely)
        share = 1.0 - min(r.groundwater_abstraction_mm3 + r.energy.desalination_capacity_mm3, 2000.0) / 2000.0
        assert c[r.name] == pytest.approx(1000.0 * share * 0.2 + 1000.0 * share * 0.6)
    assert c["Delta"] == 0.0 and c["Highland"] > 0.0
    with pytest.raises(ValueError, match="year_index"):
        model.claims(5)
    with pytest.raises(ValueError, match="unknown riparian"):
        model.claims(0, demands={**custom, "Atlantis": {MUN: 1.0}})
    with pytest.raises(ValueError, match="lacks"):
        model.claims(0, demands={"Delta": {MUN: 1.0}})
    # the claims use the grown demands of the year
    grow = NexusModel(basin, Scenario(years=3, allocation_rule="talmud", demand_growth_rate=0.1))
    assert grow.claims(2)["Delta"] > grow.claims(0)["Delta"]
    # a riparian whose sectors consume nothing has no claim and is never capped
    b = solo_basin(demand=WaterDemand(municipal=100.0, agricultural=200.0, environmental=300.0,
                                      consumption_fraction={s: 0.0 for s in WITHDRAWAL_SECTORS}))
    free = NexusModel(b, Scenario(years=1, allocation_rule="talmud"))
    assert free.claims(0) == {"Solo": 0.0}
    assert free.entitlements(0, 0.0, storages={"Solo": 0.0}) == {"Solo": pytest.approx(300.0)}


def test_withdrawal_cap_for_award_conversion():
    convert = nexus_module._withdrawal_cap_for_award
    rd = {MUN: 100.0, IND: 50.0, ENE: 10.0, AGR: 200.0}
    f = {MUN: 0.2, IND: 0.1, ENE: 0.03, AGR: 0.6}
    claim = 100 * 0.2 + 50 * 0.1 + 10 * 0.03 + 200 * 0.6  # 145.3
    # the full award (or anything larger) gives the full river demand: never binding
    assert convert(rd, f, claim) == pytest.approx(360.0)
    assert convert(rd, f, claim * (1 - 1e-12)) == pytest.approx(360.0)
    assert convert(rd, f, 1e9) == pytest.approx(360.0)
    # zero award -> nothing may be withdrawn when every sector consumes
    assert convert(rd, f, 0.0) == 0.0
    # partial awards: municipal, industrial and energy are served first, agriculture takes the rest
    assert convert(rd, f, 25.3 + 30.0) == pytest.approx(160.0 + 50.0)  # 30 Mm3 consumed by 50 Mm3 of irrigation
    assert convert(rd, f, 10.0) == pytest.approx(50.0)  # inside the municipal block (f = 0.2)
    # monotone in the award, bounded by the river demand
    grid = np.linspace(0.0, claim, 30)
    caps = [convert(rd, f, a) for a in grid]
    assert all(b >= a - 1e-12 for a, b in zip(caps, caps[1:]))
    assert all(0.0 <= c <= 360.0 + 1e-12 for c in caps)
    # the consumption at the cap, serving sectors in priority order, equals the award
    for a, cap in zip(grid, caps):
        remaining, consumed = cap, 0.0
        for s in SECTOR_PRIORITY:
            w = min(rd[s], remaining)
            consumed += w * f[s]
            remaining -= w
        assert consumed == pytest.approx(a, abs=1e-9)
    # a sector that consumes nothing never depletes the award and is always served
    assert convert(rd, dict(f, **{ENE: 0.0}), 0.0) == pytest.approx(10.0)
    # zero river demand -> zero cap
    assert convert({s: 0.0 for s in WITHDRAWAL_SECTORS}, f, 0.0) == 0.0


@pytest.mark.parametrize("rule", BANKRUPTCY_RULES)
def test_bankruptcy_rules_do_not_ration_when_claims_fit_the_estate(basin, rule):
    """Review finding: with claims and estate on the same (consumptive) basis, a cooperative rule
    reproduces the unilateral outcome at flow factor 1 instead of imposing scarcity."""
    uni = run_nexus(basin, Scenario(years=3, cooperation=False))
    coop = run_nexus(basin, Scenario(years=3, allocation_rule=rule))
    model = NexusModel(basin, Scenario(years=3, allocation_rule=rule))
    env_total = sum(r.demand.environmental for r in basin)
    for i, y in enumerate(coop.years):
        bal = coop.balances[i]
        estate = bal.natural_flow_mm3 + sum(r.storage_start_mm3 for r in bal.reaches) - env_total
        assert sum(model.claims(i).values()) <= estate
        # the river demand is physically withdrawable: the unconstrained route serves it in full
        assert all(r.supply_ratio() == pytest.approx(1.0) for r in uni.for_year(y))
        for rec in coop.for_year(y):
            u = uni.record(rec.name, y)
            assert rec.supply_ratio() >= u.supply_ratio() - 1e-12
            assert rec.total_withdrawal() == pytest.approx(u.total_withdrawal())
            assert math.isfinite(rec.entitlement_mm3)
            assert rec.entitlement_mm3 == pytest.approx(rec.total_withdrawal())  # cap = river demand, met
    for name in basin.names():
        for key in ("supply_ratio", "food_self_sufficiency", "nexus_index", "water_security"):
            assert coop.summary()[name][key] == pytest.approx(uni.summary()[name][key])
    assert coop.summary()["Highland"]["food_self_sufficiency"] == pytest.approx(0.427, abs=0.01)


@pytest.mark.parametrize("rule", BANKRUPTCY_RULES)
def test_bankruptcy_rules_ration_only_the_consumptive_shortfall(basin, rule):
    """In a scarce year the awards exhaust the estate, the caps bind and the mass balance closes."""
    res = run_nexus(basin, Scenario(years=1, allocation_rule=rule), flow_factors=[0.2])
    model = NexusModel(basin, Scenario(years=1, allocation_rule=rule))
    claims = model.claims(0)
    estate = example_estate(0.2, basin)
    assert sum(claims.values()) > estate
    awards = allocation.apply_rule(rule, estate, claims)
    assert sum(awards.values()) == pytest.approx(estate)
    demands = model.demands(0)
    capped = 0
    for rec in res.records:
        r = basin.riparian(rec.name)
        rd = river_demand_of(r, demands[rec.name])
        assert rec.total_withdrawal() <= rec.entitlement_mm3 + 1e-9
        assert rec.entitlement_mm3 <= sum(rd.values()) + 1e-9
        # the consumption the cap allows equals the consumptive award ...
        assert consumption_at_cap(r, rd, rec.entitlement_mm3) == pytest.approx(awards[rec.name], abs=1e-6)
        # ... and the routed consumption never exceeds it
        assert rec.total_consumption() <= awards[rec.name] + 1e-6
        capped += rec.entitlement_mm3 < sum(rd.values()) - 1e-6
    assert capped >= 1
    assert res.balances[0].mass_balance_error() < 1e-6


def test_bankruptcy_estate_includes_carried_storage(basin):
    """Review finding: a zero-flow drought with full reservoirs is not a zero estate."""
    sc = dict(years=2, drought_years=[1], drought_severity=1.0)
    talmud = run_nexus(basin, Scenario(allocation_rule="talmud", **sc))
    treaty = run_nexus(basin, Scenario(allocation_rule="treaty", **sc))
    assert talmud.flow_factors == [1.0, 0.0]
    dry = talmud.for_year(2026)
    assert sum(r.storage_start_mm3 for r in dry) > 16_000.0  # the reservoirs are nearly full
    model = NexusModel(basin, Scenario(allocation_rule="talmud", **sc))
    storages = talmud.balances[0].storages_end()
    caps = model.entitlements(1, 0.0, storages=storages)
    for rec in dry:
        assert rec.entitlement_mm3 == pytest.approx(caps[rec.name])
        assert rec.entitlement_mm3 > 0.0 and rec.total_withdrawal() > 0.0
        assert rec.storage_release_mm3 > 0.0  # the reservoirs are drawn down, not locked away
        # municipal demand is served from storage exactly as under the fixed treaty
        t = treaty.record(rec.name, 2026)
        assert rec.withdrawals[MUN] == pytest.approx(t.withdrawals[MUN]) and rec.withdrawals[MUN] > 0.0
        assert rec.supply_ratio() > 0.5
    estate = sum(storages.values()) - sum(r.demand.environmental for r in basin)
    awards = allocation.talmud(estate, model.claims(1))
    assert sum(awards.values()) == pytest.approx(estate)  # the stored water net of the EFR is all awarded
    assert talmud.balances[1].mass_balance_error() < 1e-6
    # without the storage the same year would forbid every withdrawal
    assert all(v == 0.0 for v in model.entitlements(1, 0.0, storages={n: 0.0 for n in basin.names()}).values())
    # entitlements() defaults to the riparians' initial storages, which is what run() uses in year 0
    initial = {r.name: r.reservoir_storage_mm3 for r in basin}
    assert model.entitlements(0, 28000.0) == pytest.approx(model.entitlements(0, 28000.0, storages=initial))
    assert talmud.record("Delta", 2025).entitlement_mm3 == pytest.approx(model.entitlements(0, 28000.0)["Delta"])
    # a milder drought (natural flow 8400): the claims fit the estate, nobody is rationed and the
    # reservoirs are drawn down instead of gaining water behind forbidding caps
    mild = dict(years=2, drought_years=[1], drought_severity=0.7)
    for rule in ("talmud", "cea"):
        res = run_nexus(basin, Scenario(allocation_rule=rule, **mild))
        recs = res.for_year(2026)
        assert res.balances[1].natural_flow_mm3 == pytest.approx(8400.0)
        assert sum(r.storage_end_mm3 for r in recs) < sum(r.storage_start_mm3 for r in recs)
        m = NexusModel(basin, Scenario(allocation_rule=rule, **mild))
        demands = m.demands(1)
        for rec in recs:
            rd = river_demand_of(basin.riparian(rec.name), demands[rec.name])
            assert rec.entitlement_mm3 == pytest.approx(sum(rd.values()))  # non-binding cap
        assert all(r.supply_ratio() == pytest.approx(1.0) for r in recs if r.name != "Delta")
        assert res.record("Delta", 2026).supply_ratio() > 0.7


def test_entitlements_storages_validation_and_monotonicity(basin):
    model = NexusModel(basin, Scenario(years=1, allocation_rule="proportional"))
    with pytest.raises(ValueError, match="unknown riparian"):
        model.entitlements(0, 28000.0, storages={"Atlantis": 1.0})
    with pytest.raises(ValueError, match="storages"):
        model.entitlements(0, 28000.0, storages={"Delta": -1.0})
    with pytest.raises(ValueError, match="storages"):
        model.entitlements(0, 28000.0, storages=[1.0])
    # riparians missing from ``storages`` fall back to their initial storage
    partial = model.entitlements(0, 2000.0, storages={"Delta": 0.0})
    full = model.entitlements(0, 2000.0, storages={"Delta": 0.0, "Highland": 3000.0, "Midland": 1500.0})
    assert partial == pytest.approx(full)
    assert sum(partial.values()) < sum(model.entitlements(0, 2000.0).values())
    # caps are non-decreasing in the natural flow and in the stored water
    prev = None
    for flow in np.linspace(0.0, 28000.0, 15):
        caps = model.entitlements(0, float(flow), storages={n: 0.0 for n in basin.names()})
        if prev is not None:
            assert all(caps[n] >= prev[n] - 1e-9 for n in caps)
        prev = caps
    prev = None
    for stored in np.linspace(0.0, 12000.0, 13):
        caps = model.entitlements(0, 3000.0, storages={n: float(stored) / 3 for n in basin.names()})
        if prev is not None:
            assert all(caps[n] >= prev[n] - 1e-9 for n in caps)
        prev = caps
    # the treaty and upstream-priority rules ignore the storages
    assert NexusModel(basin, Scenario(years=1)).entitlements(0, 1.0, storages={"Delta": 0.0}) == {
        "Highland": 2500.0, "Midland": 9000.0, "Delta": 16000.0}
    assert NexusModel(basin, Scenario(years=1, cooperation=False)).entitlements(0, 1.0, storages={"Delta": 0.0}) is None


# ---------------------------------------------------------------------------
# baseline run: shape, bounds, conservation
# ---------------------------------------------------------------------------
def test_baseline_run_shape_and_order(basin, baseline, result):
    assert isinstance(result, NexusResult)
    assert result.basin_name == basin.name
    assert result.years == [2025, 2026, 2027, 2028, 2029]
    assert result.n_years == 5 and result.n_riparians == 3
    assert len(result.records) == 5 * 3 and len(result.balances) == 5
    assert result.flow_factors == [1.0] * 5 and result.allocation_rule == "treaty"
    assert result.riparian_names() == basin.names()
    # year-major order: the first three records are the three riparians of 2025
    assert [r.name for r in result.records[:3]] == basin.names()
    assert [r.year for r in result.records[::3]] == result.years
    for name in basin.names():
        recs = result.for_riparian(name)
        assert [r.year for r in recs] == result.years and all(r.name == name for r in recs)
    for y in result.years:
        assert [r.name for r in result.for_year(y)] == basin.names()
    assert result.record("Delta", 2027).year == 2027 and result.record("Delta", 2027).name == "Delta"
    assert result.balance(2026) is result.balances[1]
    # the scenario is a copy with the same content
    assert result.scenario is not baseline and result.scenario == baseline
    assert result.scenario.name == baseline.name


def test_baseline_records_finite_and_bounded(result):
    for rec in result.records:
        d = rec.to_dict()
        for k, v in d.items():
            if k == "entitlement_mm3":
                continue
            assert _is_finite(v), (rec.name, rec.year, k, v)
        assert 0.0 <= rec.supply_ratio() <= 1.0
        for k in ("water_security", "energy_security", "food_security", "nexus_index", "et_ratio",
                  "energy_supply_ratio", "env_flow_compliance", "renewable_share"):
            assert 0.0 <= getattr(rec, k) <= 1.0, k
        assert rec.food_self_sufficiency >= 0.0 and rec.emissions_t >= 0.0
        assert rec.water_stress_sdg642 >= 0.0 and rec.groundwater_stress >= 0.0
        assert rec.per_capita_water_m3 > 0.0 and rec.falkenmark in (
            "no_stress", "stress", "scarcity", "absolute_scarcity")
        for s in WITHDRAWAL_SECTORS:
            assert rec.withdrawals[s] >= 0.0 and rec.deficits[s] >= 0.0
            assert rec.withdrawals[s] + rec.deficits[s] <= rec.demands[s] + 1e-9
        assert rec.energy_deficit_gwh >= 0.0 and rec.energy_supply_gwh >= 0.0
        assert rec.storage_end_mm3 >= 0.0 and rec.outflow_mm3 >= 0.0


def test_baseline_mass_balance_closes(result):
    for bal in result.balances:
        assert bal.mass_balance_error() < 1e-6
        for reach in bal.reaches:
            assert reach.mass_balance_error() < 1e-6


def test_storages_carried_between_years(basin, result):
    for i in range(1, len(result.balances)):
        for r in basin.riparians:
            prev, cur = result.balances[i - 1].reach(r.name), result.balances[i].reach(r.name)
            assert cur.storage_start_mm3 == pytest.approx(prev.storage_end_mm3)
            rec = result.record(r.name, result.years[i])
            assert rec.storage_start_mm3 == pytest.approx(prev.storage_end_mm3)
    for r in basin.riparians:
        assert result.balances[0].reach(r.name).storage_start_mm3 == r.reservoir_storage_mm3
    # storage never exceeds capacity
    for rec in result.records:
        assert rec.storage_end_mm3 <= basin.riparian(rec.name).reservoir_capacity_mm3 + 1e-9


def test_delta_less_water_secure_than_highland(result):
    hi = result.series("Highland", "water_security")
    de = result.series("Delta", "water_security")
    assert all(d < h for d, h in zip(de, hi))
    assert sum(de) / len(de) < sum(hi) / len(hi)
    assert result.summary()["Delta"]["water_security"] < result.summary()["Highland"]["water_security"]
    # Delta is also the most water-stressed and most groundwater-stressed riparian
    assert result.summary()["Delta"]["water_stress_sdg642"] > result.summary()["Highland"]["water_stress_sdg642"]
    assert result.record("Delta", 2025).falkenmark == "absolute_scarcity"
    assert result.record("Highland", 2025).falkenmark == "no_stress"


# ---------------------------------------------------------------------------
# closed-form consistency of each block with the leaf modules
# ---------------------------------------------------------------------------
def test_record_matches_routed_reach(basin, result):
    for i, y in enumerate(result.years):
        bal = result.balances[i]
        for r in basin.riparians:
            reach, rec = bal.reach(r.name), result.record(r.name, y)
            assert rec.inflow_mm3 == reach.inflow_mm3
            assert rec.upstream_inflow_mm3 == reach.upstream_inflow_mm3
            assert rec.outflow_mm3 == reach.outflow_mm3
            assert rec.storage_end_mm3 == reach.storage_end_mm3
            assert rec.entitlement_mm3 == reach.entitlement_mm3 == basin.riparian(r.name).treaty_allocation_mm3
            assert rec.demands == reach.demands and rec.withdrawals == reach.withdrawals
            assert rec.deficits == reach.deficits and rec.consumption == reach.consumption
            assert rec.environmental_flow_mm3 == reach.environmental_flow_mm3
            assert rec.env_flow_met == reach.env_flow_met
            assert rec.supply_ratio() == pytest.approx(reach.supply_ratio())
            assert rec.non_river_supply_mm3 == reach.non_river_supply_mm3
            assert rec.storage_release_mm3 == reach.storage_release_mm3
            assert rec.storage_refill_mm3 == reach.storage_refill_mm3
            assert rec.evaporation_mm3 == reach.evaporation_mm3
            # record dicts are copies, not aliases of the reach dicts
            assert rec.withdrawals is not reach.withdrawals
            # agricultural water delivered = demand - deficit (surface + non-river share)
            assert rec.agricultural_water_mm3 == pytest.approx(reach.demands[AGR] - reach.deficits[AGR])
            assert rec.groundwater_used_mm3 + rec.desalinated_mm3 == pytest.approx(reach.non_river_supply_mm3)
            gw, ds = r.groundwater_abstraction_mm3, r.energy.desalination_capacity_mm3
            assert rec.groundwater_used_mm3 == pytest.approx(reach.non_river_supply_mm3 * gw / (gw + ds))


def test_water_indicators_closed_form(basin, result):
    for rec in result.records:
        r = basin.riparian(rec.name)
        renewable = rec.inflow_mm3 + r.groundwater_recharge_mm3
        assert rec.renewable_water_mm3 == pytest.approx(renewable)
        assert rec.per_capita_water_m3 == pytest.approx(renewable * 1e6 / rec.population)
        assert rec.falkenmark == water.falkenmark_category(rec.per_capita_water_m3)
        withdrawal = rec.total_withdrawal() + rec.groundwater_used_mm3
        assert rec.water_stress_sdg642 == pytest.approx(
            water.sdg_642_water_stress(withdrawal, renewable, rec.environmental_flow_mm3))
        assert rec.groundwater_stress == pytest.approx(
            water.groundwater_stress(rec.groundwater_used_mm3, r.groundwater_recharge_mm3))
        assert rec.env_flow_compliance == pytest.approx(min(rec.outflow_mm3 / rec.environmental_flow_mm3, 1.0))
        assert rec.water_value_usd == pytest.approx(
            sum(w * r.demand.value_usd_per_m3[s] * 1e6 for s, w in rec.withdrawals.items()))
        assert rec.population == r.population and rec.gdp_usd == r.gdp_usd  # no growth in the baseline


def test_energy_block_closed_form(basin, result):
    for rec in result.records:
        r = basin.riparian(rec.name)
        es = r.energy
        assert rec.hydropower_gwh == pytest.approx(energy.hydropower_gwh(
            rec.outflow_mm3 * es.turbined_fraction, es.hydropower_head_m, es.turbine_efficiency,
            capacity_mw=es.hydropower_capacity_mw))
        assert rec.hydropower_gwh <= es.hydropower_capacity_mw * 8760.0 / 1000.0 + 1e-9
        assert rec.thermal_gwh == pytest.approx(energy.thermal_generation_gwh(es.thermal_capacity_mw, es.thermal_capacity_factor))
        assert rec.other_renewable_gwh == pytest.approx(es.demand_gwh * es.renewable_share)
        efw = energy.energy_for_water(r, rec.withdrawals, rec.desalinated_mm3)
        gw_pump = energy.pumping_energy_gwh(rec.groundwater_used_mm3, es.pumping_lift_m, es.pumping_efficiency)
        assert rec.groundwater_pumping_gwh == pytest.approx(gw_pump)
        assert rec.energy_for_water_gwh == pytest.approx(efw["total"] + gw_pump)
        assert rec.energy_demand_gwh == pytest.approx(es.demand_gwh + efw["total"] + gw_pump)
        assert rec.other_renewable_share_of_demand == es.renewable_share
        assert rec.energy_supply_gwh == pytest.approx(rec.hydropower_gwh + rec.thermal_gwh + rec.other_renewable_gwh)
        assert rec.energy_deficit_gwh == pytest.approx(max(rec.energy_demand_gwh - rec.energy_supply_gwh, 0.0))
        assert rec.emissions_t == pytest.approx(rec.thermal_gwh * es.grid_emission_factor_t_per_gwh)
        assert rec.renewable_share == pytest.approx((rec.hydropower_gwh + rec.other_renewable_gwh) / rec.energy_supply_gwh)
        assert rec.emission_intensity_t_per_gwh == pytest.approx(rec.emissions_t / rec.energy_supply_gwh)
        assert rec.energy_supply_ratio == pytest.approx(min(rec.energy_supply_gwh / rec.energy_demand_gwh, 1.0))
        assert rec.cooling_water_mm3 == pytest.approx(energy.water_for_energy(r, rec.thermal_gwh))
    # Delta's desalination plant runs flat out (demand far above capacity)
    d = result.record("Delta", 2025)
    assert d.desalinated_mm3 == pytest.approx(300.0)
    assert d.energy_for_water_gwh > energy.desalination_energy_gwh(300.0, 3.5)


def test_food_block_closed_form(basin, result):
    for rec in result.records:
        r = basin.riparian(rec.name)
        fp = food.riparian_food_production(r, rec.agricultural_water_mm3)
        assert rec.food_production_t == pytest.approx(fp["production_t"])
        assert rec.food_kcal == pytest.approx(fp["kcal"])
        assert rec.crop_value_usd == pytest.approx(fp["value_usd"])
        assert rec.et_ratio == pytest.approx(fp["et_ratio"])
        assert rec.irrigation_requirement_mm3 == pytest.approx(fp["requirement_mm3"])
        assert rec.food_demand_kcal == pytest.approx(r.food_demand_kcal())
        assert rec.food_self_sufficiency == pytest.approx(rec.food_kcal / rec.food_demand_kcal)
        assert rec.irrigated_area_ha == pytest.approx(r.irrigated_area_ha())
    # Highland is fully supplied: 900 000 t at ETa/ETm = 1 (food docstring example)
    h = result.record("Highland", 2025)
    assert h.et_ratio == pytest.approx(1.0) and h.food_production_t == pytest.approx(900_000.0)
    assert h.irrigation_requirement_mm3 == pytest.approx(798.0)
    # Delta is capped by its treaty: a small agricultural deficit lowers ETa/ETm below 1
    d = result.record("Delta", 2025)
    assert d.deficits[AGR] > 0.0 and d.et_ratio < 1.0
    # FAO-33 ETa/ETm counts Delta's 20 mm of effective rainfall on top of the net
    # irrigation, so it sits strictly above the gross irrigation supply ratio
    gross_supply_ratio = d.agricultural_water_mm3 / d.irrigation_requirement_mm3
    assert gross_supply_ratio < d.et_ratio < 1.0
    fp = food.riparian_food_production(basin.riparian("Delta"), d.agricultural_water_mm3)
    assert fp["irrigation_supply_ratio"] == pytest.approx(gross_supply_ratio)
    assert d.et_ratio == pytest.approx(fp["eta_mm3"] / fp["etm_mm3"])


def test_indices_match_sustainability_functions(result):
    for rec in result.records:
        ws = sustainability.water_security_index(
            rec.supply_ratio(), rec.water_stress_sdg642, rec.env_flow_compliance, rec.groundwater_stress)
        es = sustainability.energy_security_index(
            rec.energy_supply_ratio, rec.renewable_share, rec.emission_intensity_t_per_gwh)
        fs = sustainability.food_security_index(rec.food_self_sufficiency, rec.et_ratio)
        assert rec.water_security == pytest.approx(ws)
        assert rec.energy_security == pytest.approx(es)
        assert rec.food_security == pytest.approx(fs)
        assert rec.nexus_index == pytest.approx(sustainability.wef_nexus_index(ws, es, fs))
        assert rec.nexus_index <= (ws + es + fs) / 3.0 + 1e-12  # geometric <= arithmetic mean


# ---------------------------------------------------------------------------
# scenario effects
# ---------------------------------------------------------------------------
def test_uniform_minus_40_percent_flow_reduces_delta_supply_ratio(basin, baseline, result):
    dry = run_nexus(basin, baseline, flow_factors=[0.6] * 5)
    assert dry.flow_factors == [0.6] * 5
    base_sr, dry_sr = result.series("Delta", "supply_ratio"), dry.series("Delta", "supply_ratio")
    assert sum(dry_sr) / 5 < sum(base_sr) / 5
    assert all(d <= b + 1e-12 for d, b in zip(dry_sr, base_sr)) and min(dry_sr) < min(base_sr)
    assert all(d < b for d, b in zip(dry.series("Delta", "inflow_mm3"), result.series("Delta", "inflow_mm3")))
    assert all(d.natural_flow_mm3 == pytest.approx(0.6 * b.natural_flow_mm3) for d, b in zip(dry.balances, result.balances))
    assert dry.summary()["Delta"]["supply_ratio"] < result.summary()["Delta"]["supply_ratio"]
    assert dry.summary()["Delta"]["water_security"] < result.summary()["Delta"]["water_security"]
    for bal in dry.balances:
        assert bal.mass_balance_error() < 1e-6


def test_linear_minus_40_percent_trend_dries_the_basin(basin, baseline, result):
    trend = run_nexus(basin, Scenario(name="dry", years=5, flow_change_pct_by_end=-40.0))
    assert trend.flow_factors == pytest.approx([1.0, 0.9, 0.8, 0.7, 0.6])
    assert trend.record("Delta", 2029).inflow_mm3 < result.record("Delta", 2029).inflow_mm3
    assert trend.balances[-1].outflow_to_sea_mm3 < result.balances[-1].outflow_to_sea_mm3
    assert sum(trend.series("Highland", "hydropower_gwh")) < sum(result.series("Highland", "hydropower_gwh"))
    assert trend.summary()["Delta"]["supply_ratio"] <= result.summary()["Delta"]["supply_ratio"] + 1e-12
    assert trend.summary()["Delta"]["per_capita_water_m3"] < result.summary()["Delta"]["per_capita_water_m3"]
    # the gradual trend is buffered by Delta's reservoir, which is drawn down in the dry years
    assert trend.record("Delta", 2029).storage_release_mm3 > result.record("Delta", 2029).storage_release_mm3
    assert trend.record("Delta", 2029).storage_end_mm3 < result.record("Delta", 2029).storage_end_mm3


def test_unilateral_hurts_delta_more_than_treaty_when_upstream_caps_bind(basin):
    b = tight_treaty_basin(basin)
    drought = dict(years=5, drought_years=[1, 2, 3], drought_severity=0.6)
    treaty = run_nexus(b, Scenario(name="treaty", **drought))
    uni = run_nexus(b, Scenario(name="unilateral", cooperation=False, **drought))
    assert treaty.allocation_rule == "treaty" and uni.allocation_rule == "upstream_priority"
    t_sr, u_sr = treaty.series("Delta", "supply_ratio"), uni.series("Delta", "supply_ratio")
    assert sum(u_sr) / 5 < sum(t_sr) / 5
    assert min(u_sr) < min(t_sr)
    # mechanism: without caps Midland withdraws its full demand, leaving less for Delta
    assert all(u > t for u, t in zip(uni.series("Midland", "total_withdrawal_mm3"), treaty.series("Midland", "total_withdrawal_mm3")))
    assert all(w == pytest.approx(4000.0) for w in treaty.series("Midland", "total_withdrawal_mm3"))
    assert all(u < t for u, t in zip(uni.series("Delta", "inflow_mm3"), treaty.series("Delta", "inflow_mm3")))
    assert all(e == INF for e in uni.series("Delta", "entitlement_mm3"))
    assert uni.summary()["Delta"]["water_security"] < treaty.summary()["Delta"]["water_security"]
    assert uni.summary()["Delta"]["vulnerability"] > treaty.summary()["Delta"]["vulnerability"]


def test_unilateral_removes_every_cap(basin, baseline, result):
    uni = run_nexus(basin, Scenario(name="uni", years=5, cooperation=False))
    for rec in uni.records:
        assert rec.entitlement_mm3 == INF
    # in the example basin only Delta's own treaty cap binds, so Delta gains from no caps
    assert all(u >= t for u, t in zip(uni.series("Delta", "supply_ratio"), result.series("Delta", "supply_ratio")))
    assert uni.record("Delta", 2025).supply_ratio() == pytest.approx(1.0)
    assert result.record("Delta", 2025).total_withdrawal() == pytest.approx(16000.0)
    assert uni.record("Delta", 2025).total_withdrawal() > 16000.0


def test_treaty_rule_leaves_untreatied_riparian_uncapped(basin):
    b = basin.copy()
    b.riparian("Delta").treaty_allocation_mm3 = None
    res = run_nexus(b, Scenario(years=2))
    assert res.record("Delta", 2025).entitlement_mm3 == INF
    assert res.record("Delta", 2025).supply_ratio() == pytest.approx(1.0)
    assert res.record("Highland", 2025).entitlement_mm3 == 2500.0


@pytest.mark.parametrize("rule", BANKRUPTCY_RULES)
def test_bankruptcy_rule_runs_cap_withdrawals(basin, rule):
    res = run_nexus(basin, Scenario(years=3, allocation_rule=rule))
    assert res.allocation_rule == rule
    model = NexusModel(basin, Scenario(years=3, allocation_rule=rule))
    for i, y in enumerate(res.years):
        storages = res.balances[i - 1].storages_end() if i else None
        ent = model.entitlements(i, res.balances[i].natural_flow_mm3, storages=storages)
        for rec in res.for_year(y):
            assert rec.entitlement_mm3 == pytest.approx(ent[rec.name])
            assert rec.total_withdrawal() <= rec.entitlement_mm3 + 1e-9
        assert res.balances[i].mass_balance_error() < 1e-6
    # equal split == CEA (allocation contract)
    if rule == "equal":
        cea = run_nexus(basin, Scenario(years=3, allocation_rule="cea"))
        assert cea.series("Delta", "entitlement_mm3") == pytest.approx(res.series("Delta", "entitlement_mm3"))


def test_cea_protects_small_claimants_and_cel_large_ones(basin):
    """In a scarce year CEA fills the smallest claim (Highland) first; CEL makes it bear the loss."""
    cea = run_nexus(basin, Scenario(years=1, allocation_rule="cea"), flow_factors=[0.2])
    cel = run_nexus(basin, Scenario(years=1, allocation_rule="cel"), flow_factors=[0.2])
    claims = NexusModel(basin, Scenario(years=1, allocation_rule="cea")).claims(0)
    assert sum(claims.values()) > example_estate(0.2, basin)  # bankruptcy: claims exceed the estate
    h_cea, h_cel = cea.record("Highland", 2025), cel.record("Highland", 2025)
    assert h_cea.supply_ratio() == pytest.approx(1.0)
    assert h_cea.entitlement_mm3 == pytest.approx(h_cea.total_withdrawal())  # uncapped at its river demand
    assert h_cel.entitlement_mm3 == 0.0 and h_cel.total_withdrawal() == 0.0
    assert h_cel.supply_ratio() < h_cea.supply_ratio()
    assert cel.record("Delta", 2025).entitlement_mm3 > cea.record("Delta", 2025).entitlement_mm3
    assert cel.record("Midland", 2025).entitlement_mm3 < cea.record("Midland", 2025).entitlement_mm3


def test_stochastic_runs_reproducible_and_seed_sensitive(basin):
    s1 = Scenario(name="stoch", years=5, stochastic=True, seed=7)
    a, b = run_nexus(basin, s1), run_nexus(basin, s1)
    assert a.to_records() == b.to_records()
    assert a.flow_factors == b.flow_factors
    assert a.flow_factors == pytest.approx(water.stochastic_flow_factors(s1, basin))
    assert a.flow_factors != [1.0] * 5
    c = run_nexus(basin, Scenario(name="stoch", years=5, stochastic=True, seed=8))
    assert c.flow_factors != a.flow_factors
    assert c.to_records() != a.to_records()
    assert all(f >= 0.0 for f in a.flow_factors + c.flow_factors)
    for bal in a.balances + c.balances:
        assert bal.mass_balance_error() < 1e-6
    # a basin with no climate variability is deterministic even when stochastic
    calm = basin.copy()
    calm.climate_cv = 0.0
    assert run_nexus(calm, s1).flow_factors == [1.0] * 5


def test_drought_years_lower_flow_and_delta_inflow(basin, result):
    res = run_nexus(basin, Scenario(years=5, drought_years=[2], drought_severity=0.5))
    assert res.flow_factors == pytest.approx([1.0, 1.0, 0.5, 1.0, 1.0])
    assert res.balances[2].natural_flow_mm3 == pytest.approx(14000.0)
    assert res.record("Delta", 2027).inflow_mm3 < result.record("Delta", 2027).inflow_mm3
    assert res.record("Delta", 2027).supply_ratio() <= result.record("Delta", 2027).supply_ratio() + 1e-12
    # reservoirs refill after the drought year: Delta's storage drops then recovers
    st = res.series("Delta", "storage_end_mm3")
    assert st[2] < st[1] and st[4] > st[2]


def test_growth_scenario_raises_demand_and_lowers_delta_supply(basin, result):
    sc = Scenario(name="growth", years=5, population_growth_rate=0.02, demand_growth_rate=0.03,
                  irrigated_area_change_pct_by_end=30.0, energy_demand_growth_rate=0.03)
    res = run_nexus(basin, sc)
    for name in basin.names():
        dem = res.series(name, "total_demand_mm3")
        assert all(b > a for a, b in zip(dem, dem[1:]))
        pop = res.series(name, "population")
        assert all(b > a for a, b in zip(pop, pop[1:]))
        fdem = res.series(name, "food_demand_kcal")
        assert all(b > a for a, b in zip(fdem, fdem[1:]))
        edem = res.series(name, "energy_demand_gwh")
        assert edem[-1] > edem[0]
    assert res.record("Delta", 2029).supply_ratio() < result.record("Delta", 2029).supply_ratio()
    assert res.summary()["Delta"]["supply_ratio"] < result.summary()["Delta"]["supply_ratio"]


def test_efficiency_scenario_lowers_requirement_and_raises_renewables(basin, result):
    sc = Scenario(name="eff", years=5, irrigation_efficiency_target=0.8, renewable_share_target=0.6)
    res = run_nexus(basin, sc)
    for name in basin.names():
        req = res.series(name, "irrigation_requirement_mm3")
        assert all(b < a for a, b in zip(req, req[1:]))
        assert req[0] == pytest.approx(result.series(name, "irrigation_requirement_mm3")[0])
        assert res.series(name, "irrigation_efficiency")[-1] == pytest.approx(0.8)
    assert res.record("Delta", 2029).other_renewable_gwh == pytest.approx(0.6 * 160_000.0)
    assert res.record("Delta", 2029).energy_security > result.record("Delta", 2029).energy_security
    assert res.record("Delta", 2029).supply_ratio() >= result.record("Delta", 2029).supply_ratio()
    assert res.record("Delta", 2029).water_stress_sdg642 < result.record("Delta", 2029).water_stress_sdg642


def test_more_water_never_lowers_supply_ratio(basin, baseline, result):
    wet = run_nexus(basin, baseline, flow_factors=[1.3] * 5)
    for name in basin.names():
        for w, b in zip(wet.series(name, "supply_ratio"), result.series(name, "supply_ratio")):
            assert w >= b - 1e-12
        assert sum(wet.series(name, "hydropower_gwh")) >= sum(result.series(name, "hydropower_gwh")) - 1e-9


def test_refill_fraction_changes_storage_but_not_mass_balance(basin, baseline):
    none = run_nexus(basin, baseline, reservoir_refill_fraction=0.0)
    full = run_nexus(basin, baseline, reservoir_refill_fraction=1.0)
    for name in basin.names():
        assert all(f >= n for f, n in zip(full.series(name, "storage_refill_mm3"), none.series(name, "storage_refill_mm3")))
        assert all(v == 0.0 for v in none.series(name, "storage_refill_mm3"))
    for bal in none.balances + full.balances:
        assert bal.mass_balance_error() < 1e-6


# ---------------------------------------------------------------------------
# no mutation
# ---------------------------------------------------------------------------
def test_inputs_are_not_mutated(basin):
    scenarios = [
        Scenario(name="base", years=4),
        Scenario(name="uni", years=4, cooperation=False),
        Scenario(name="talmud", years=4, allocation_rule="talmud", drought_years=[1]),
        Scenario(name="stoch", years=4, stochastic=True, population_growth_rate=0.01,
                 irrigation_efficiency_target=0.9, renewable_share_target=0.5, irrigated_area_change_pct_by_end=20),
    ]
    before_basin = dataclasses.asdict(basin)
    before_scen = [dataclasses.asdict(s) for s in scenarios]
    for s in scenarios:
        res = run_nexus(basin, s)
        model = NexusModel(basin, s)
        model.demands(1)
        model.claims(1)
        model.entitlements(1, 20000.0)
        model.entitlements(1, 20000.0, storages={"Delta": 0.0})
        # mutating the result must not reach back into the inputs
        res.records[0].demands[MUN] = -999.0
        res.balances[0].reaches[0].withdrawals[MUN] = -999.0
        res.scenario.years = 99
    assert dataclasses.asdict(basin) == before_basin
    assert [dataclasses.asdict(s) for s in scenarios] == before_scen
    assert [r.position for r in basin.riparians] == [0, 1, 2]


# ---------------------------------------------------------------------------
# NexusResult API
# ---------------------------------------------------------------------------
def test_result_lookups_raise_keyerror(result):
    with pytest.raises(KeyError):
        result.for_riparian("Atlantis")
    with pytest.raises(KeyError):
        result.for_year(1999)
    with pytest.raises(KeyError):
        result.record("Delta", 1999)
    with pytest.raises(KeyError):
        result.balance(1999)
    with pytest.raises(KeyError):
        result.series("Atlantis", "nexus_index")


def test_series_and_basin_series(basin, result):
    sr = result.series("Delta", "supply_ratio")
    assert sr == [r.supply_ratio() for r in result.for_riparian("Delta")]
    assert result.series("Delta", "withdrawal_agricultural") == [r.withdrawals[AGR] for r in result.for_riparian("Delta")]
    assert result.series("Delta", "env_flow_met") == [True] * 5
    assert result.series("Delta", "falkenmark")[0] == "absolute_scarcity"
    assert isinstance(result.series("Delta", "demands")[0], dict)  # raw attribute access
    hydro_sum = result.basin_series("hydropower_gwh")
    assert len(hydro_sum) == 5
    for i, y in enumerate(result.years):
        assert hydro_sum[i] == pytest.approx(sum(r.hydropower_gwh for r in result.for_year(y)))
    assert result.basin_series("nexus_index", "mean")[0] == pytest.approx(
        sum(r.nexus_index for r in result.for_year(2025)) / 3)
    assert result.basin_series("water_security", "min")[0] == pytest.approx(result.record("Delta", 2025).water_security)
    assert result.basin_series("water_security", "max")[0] == pytest.approx(result.record("Highland", 2025).water_security)
    assert result.basin_series("env_flow_met", "all") == [1.0] * 5
    assert result.basin_series("env_flow_met", "any") == [1.0] * 5
    assert result.basin_series("total_withdrawal_mm3") == pytest.approx([b.total_withdrawal() for b in result.balances])
    with pytest.raises(ValueError, match="unknown field"):
        result.series("Delta", "no_such_field")
    with pytest.raises(ValueError, match="field must be a string"):
        result.series("Delta", 3)
    with pytest.raises(ValueError, match="agg"):
        result.basin_series("hydropower_gwh", "median")
    with pytest.raises(ValueError, match="not numeric"):
        result.basin_series("falkenmark")
    with pytest.raises(ValueError, match="not numeric"):
        result.basin_series("demands")


def test_to_records_to_dataframe_to_csv(basin, baseline, result, tmp_path):
    recs = result.to_records()
    assert len(recs) == 15 and all(set(r) == set(recs[0]) for r in recs)
    assert recs[0]["name"] == "Highland" and recs[0]["year"] == 2025
    for key in ("withdrawal_municipal", "demand_agricultural", "deficit_energy", "supply_ratio", "nexus_index"):
        assert key in recs[0]
    pd = pytest.importorskip("pandas")
    df = result.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == len(result.years) * len(basin.riparians) == 15
    assert set(df["name"]) == set(basin.names()) and sorted(set(df["year"])) == result.years
    assert df["supply_ratio"].between(0.0, 1.0).all()
    path = tmp_path / "nexus.csv"
    out = result.to_csv(path)
    assert out == str(path) and path.exists()
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 15 and list(rows[0]) == list(recs[0])
    assert rows[0]["name"] == "Highland" and rows[0]["year"] == "2025"
    assert float(rows[2]["entitlement_mm3"]) == 16000.0
    assert float(rows[0]["supply_ratio"]) == pytest.approx(recs[0]["supply_ratio"])


def test_summary_structure_and_values(basin, result):
    s = result.summary()
    assert list(s) == basin.names() + [BASIN_KEY]
    for name in basin.names():
        d = s[name]
        assert d["years"] == 5 and d["first_year"] == 2025 and d["last_year"] == 2029
        for k in ("water_security", "energy_security", "food_security", "nexus_index", "supply_ratio"):
            assert d[k] == pytest.approx(sum(result.series(name, k)) / 5)
            assert 0.0 <= d[k] <= 1.0
        assert d["min_supply_ratio"] == pytest.approx(min(result.series(name, "supply_ratio")))
        assert d["min_nexus_index"] == pytest.approx(min(result.series(name, "nexus_index")))
        assert d["final_storage_mm3"] == pytest.approx(result.record(name, 2029).storage_end_mm3)
        assert d["total_hydropower_gwh"] == pytest.approx(sum(result.series(name, "hydropower_gwh")))
        assert d["env_flow_met_share"] == 1.0 and d["years_env_flow_unmet"] == 0
        assert 0.0 <= d["supply_reliability"] <= 1.0 and 0.0 <= d["resilience"] <= 1.0 and 0.0 <= d["vulnerability"] <= 1.0
        assert d["falkenmark"] == water.falkenmark_category(d["per_capita_water_m3"])
        assert d["food_self_sufficiency"] == pytest.approx(
            sum(result.series(name, "food_kcal")) / sum(result.series(name, "food_demand_kcal")))
        assert d["energy_self_sufficiency"] == pytest.approx(
            sum(result.series(name, "energy_supply_gwh")) / sum(result.series(name, "energy_demand_gwh")))
        assert 0.0 <= d["renewable_share"] <= 1.0
        assert d["entitlement_mm3"] == basin.riparian(name).treaty_allocation_mm3
    assert s["Highland"]["supply_reliability"] == 1.0 and s["Highland"]["vulnerability"] == 0.0
    assert s["Delta"]["supply_reliability"] == 0.0  # capped every year -> below 99.9 % of demand
    b = s[BASIN_KEY]
    assert b["riparians"] == 3 and b["years"] == 5
    assert b["natural_flow_mm3"] == pytest.approx(28000.0)
    assert b["outflow_to_sea_mm3"] == pytest.approx(sum(x.outflow_to_sea_mm3 for x in result.balances) / 5)
    assert b["mass_balance_error_mm3"] < 1e-6
    assert b["nexus_index"] == pytest.approx(sum(s[n]["nexus_index"] for n in basin.names()) / 3)
    assert b["worst_riparian"] == "Delta" and b["min_nexus_index"] == pytest.approx(s["Delta"]["nexus_index"])
    assert 0.0 <= b["equity_index"] <= 1.0 and 0.0 <= b["equity_nexus_index"] <= 1.0
    assert b["equity_index"] == pytest.approx(sustainability.equity_index({n: s[n]["supply_ratio"] for n in basin.names()}))
    assert b["final_storage_mm3"] == pytest.approx(sum(s[n]["final_storage_mm3"] for n in basin.names()))
    assert b["total_hydropower_gwh"] == pytest.approx(sum(s[n]["total_hydropower_gwh"] for n in basin.names()))
    assert b["population"] == pytest.approx(basin.total_population())
    assert b["supply_ratio"] == pytest.approx(1.0 - sum(result.basin_series("total_deficit_mm3")) / sum(result.basin_series("total_demand_mm3")))
    assert b["years_env_flow_unmet"] == 0 and b["env_flow_met_share"] == 1.0
    # population-weighted nexus index sits between the min and max riparian values
    vals = [s[n]["nexus_index"] for n in basin.names()]
    assert min(vals) <= b["nexus_index_population_weighted"] <= max(vals)


def test_summary_without_balances_and_result_validation(result):
    bare = NexusResult(result.basin_name, result.scenario, result.years, result.records, [])
    s = bare.summary()[BASIN_KEY]
    assert s["natural_flow_mm3"] is None and s["mass_balance_error_mm3"] is None and s["outflow_to_sea_mm3"] is None
    with pytest.raises(KeyError):
        bare.balance(2025)
    with pytest.raises(ValueError, match="RiparianYear"):
        NexusResult("b", result.scenario, result.years, [{"name": "x"}], [])
    with pytest.raises(ValueError, match="balances"):
        NexusResult("b", result.scenario, result.years, result.records, result.balances[:2])
    with pytest.raises(ValueError, match="flow_factors"):
        NexusResult("b", result.scenario, result.years, result.records, result.balances, flow_factors=[1.0])
    empty = NexusResult("b", result.scenario, [], [], [])
    assert empty.riparian_names() == [] and empty.n_riparians == 0
    with pytest.raises(ValueError, match="without records"):
        empty.summary()


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------
def test_zero_flow_uses_storage_then_fails(basin):
    res = run_nexus(basin, Scenario(years=3), flow_factors=[0.0, 0.0, 0.0])
    assert all(b.natural_flow_mm3 == 0.0 for b in res.balances)
    for bal in res.balances:
        assert bal.mass_balance_error() < 1e-6
    for rec in res.records:
        assert 0.0 <= rec.supply_ratio() <= 1.0
        assert 0.0 <= rec.nexus_index <= 1.0 and 0.0 <= rec.water_security <= 1.0
        assert rec.total_withdrawal() <= rec.storage_start_mm3 + rec.inflow_mm3 + 1e-9
    # Highland's inflow is only its recharge, below its environmental flow -> stress is infinite
    assert res.record("Highland", 2025).water_stress_sdg642 == INF
    delta = res.series("Delta", "supply_ratio")
    assert delta[0] > delta[1] and res.series("Delta", "storage_end_mm3")[-1] == pytest.approx(0.0)
    assert res.series("Delta", "storage_end_mm3")[0] < basin.riparian("Delta").reservoir_storage_mm3
    one = run_nexus(basin, Scenario(years=1, flow_change_pct_by_end=-100.0))
    assert one.flow_factors == [0.0] and one.balances[0].mass_balance_error() < 1e-6


def test_single_riparian_closed_form():
    b = solo_basin()
    r = b.riparians[0]
    res = run_nexus(b, Scenario(years=2))
    rec = res.record("Solo", 2025)
    inflow = 1500.0
    consumption = 100 * 0.2 + 50 * 0.1 + 10 * 0.03 + 200 * 0.6  # 145.3
    assert rec.inflow_mm3 == inflow and rec.upstream_inflow_mm3 == 500.0
    assert rec.total_withdrawal() == pytest.approx(360.0) and rec.supply_ratio() == 1.0
    assert rec.total_consumption() == pytest.approx(consumption)
    assert rec.outflow_mm3 == pytest.approx(inflow - consumption)  # no reservoir, no refill
    assert rec.storage_end_mm3 == 0.0 and rec.storage_start_mm3 == 0.0 and rec.evaporation_mm3 == 0.0
    assert rec.entitlement_mm3 == INF and rec.env_flow_met and rec.env_flow_compliance == 1.0
    assert rec.per_capita_water_m3 == pytest.approx(1600.0) and rec.falkenmark == "stress"
    assert rec.water_stress_sdg642 == pytest.approx(100.0 * 360.0 / (1600.0 - 300.0))
    assert rec.groundwater_stress == 0.0 and rec.groundwater_used_mm3 == 0.0 and rec.desalinated_mm3 == 0.0
    expected_hydro = min(1000.0 * 9.81 * (inflow - consumption) * 1e6 * 100.0 * 0.9 / 3.6e12, 50.0 * 8.76)
    assert rec.hydropower_gwh == pytest.approx(expected_hydro)
    assert rec.thermal_gwh == pytest.approx(438.0) and rec.other_renewable_gwh == pytest.approx(200.0)
    efw = energy.energy_for_water(r, r.demand.withdrawals(), 0.0)
    assert rec.energy_for_water_gwh == pytest.approx(efw["total"]) == pytest.approx(135.0)
    assert rec.energy_demand_gwh == pytest.approx(1135.0)
    assert rec.energy_supply_gwh == pytest.approx(expected_hydro + 438.0 + 200.0)
    assert rec.emissions_t == pytest.approx(438.0 * 400.0)
    assert rec.water_value_usd == pytest.approx(100 * 1.5e6 + 50 * 0.8e6 + 10 * 0.4e6 + 200 * 0.1e6)
    assert rec.food_kcal == 0.0 and rec.food_production_t == 0.0 and rec.crop_value_usd == 0.0
    assert rec.food_demand_kcal == pytest.approx(1e6 * 2500 * 365)
    assert rec.food_self_sufficiency == 0.0 and rec.irrigation_requirement_mm3 == 200.0
    assert rec.et_ratio == 1.0  # no crops: agricultural supply ratio (fully served)
    assert rec.food_security == pytest.approx(sustainability.food_security_index(0.0, 1.0))
    assert rec.irrigated_area_ha == 0.0
    assert res.summary()[BASIN_KEY]["natural_flow_mm3"] == pytest.approx(1500.0)


def test_no_crops_et_ratio_follows_agricultural_supply_ratio():
    # a riparian without crops whose agricultural demand cannot be fully served
    b = solo_basin(demand=WaterDemand(municipal=100.0, agricultural=2000.0, environmental=300.0),
                   local_inflow_mm3=500.0)
    rec = run_nexus(b, Scenario(years=1)).record("Solo", 2025)
    assert rec.deficits[AGR] > 0.0
    assert rec.et_ratio == pytest.approx(1.0 - rec.deficits[AGR] / rec.demands[AGR])
    assert rec.food_kcal == 0.0 and rec.food_security == pytest.approx(sustainability.food_security_index(0.0, rec.et_ratio))


def test_zero_demand_riparian():
    b = solo_basin(demand=WaterDemand(environmental=100.0), energy=EnergySystem())
    res = run_nexus(b, Scenario(years=2))
    for rec in res.records:
        assert rec.supply_ratio() == 1.0 and rec.total_withdrawal() == 0.0 and rec.total_deficit() == 0.0
        assert rec.water_value_usd == 0.0 and rec.energy_for_water_gwh == 0.0
        assert rec.outflow_mm3 == pytest.approx(1500.0) and rec.water_stress_sdg642 == 0.0
        assert rec.energy_supply_gwh == 0.0 and rec.energy_demand_gwh == 0.0 and rec.energy_supply_ratio == 1.0
        assert rec.renewable_share == 0.0 and rec.emission_intensity_t_per_gwh == 0.0
        assert 0.0 <= rec.nexus_index <= 1.0
    assert res.summary()["Solo"]["supply_reliability"] == 1.0


def test_zero_population_riparian():
    rec = run_nexus(solo_basin(population=0.0), Scenario(years=1)).record("Solo", 2025)
    assert rec.per_capita_water_m3 == INF and rec.falkenmark == "no_stress"
    assert rec.food_demand_kcal == 0.0 and rec.food_self_sufficiency == 1.0
    assert 0.0 <= rec.nexus_index <= 1.0 and rec.water_security <= 1.0


def test_zero_population_riparian_can_be_assessed():
    # regression: the ``inf`` per-capita water of an unpopulated riparian used to
    # make sustainability.assess() fail on the basin-level per-capita equity
    res = run_nexus(solo_basin(population=0.0), Scenario(years=2))
    rep = sustainability.assess(res)
    assert rep.riparian("Solo")["per_capita_water_m3"] == INF
    assert rep.basin["equity_per_capita_water"] == 1.0
    assert rep.basin["per_capita_water_m3"] is None  # nobody to divide the flow by
    assert rep.basin["nexus_index_population_weighted"] == pytest.approx(rep.basin["nexus_index"])
    assert BASIN_KEY in rep.summary()
    assert "inf" in rep.summary(columns=["per_capita_water_m3", "nexus_index"])


def test_unpopulated_riparian_is_left_out_of_per_capita_equity(basin):
    b = basin.copy()
    b.riparian("Midland").population = 0.0
    res = run_nexus(b, Scenario(years=2))
    rep = sustainability.assess(res)
    pcs = {n: rep.riparian(n)["per_capita_water_m3"] for n in b.names()}
    assert pcs["Midland"] == INF and all(math.isfinite(pcs[n]) for n in ("Highland", "Delta"))
    populated = {n: pcs[n] for n in ("Highland", "Delta")}
    assert rep.basin["equity_per_capita_water"] == pytest.approx(sustainability.equity_index(populated))
    assert 0.0 < rep.basin["equity_per_capita_water"] < 1.0
    # the basin-level per-capita water still exists: the other riparians have people
    assert rep.basin["per_capita_water_m3"] is not None and math.isfinite(rep.basin["per_capita_water_m3"])
    # the rest of the pipeline is unaffected
    assert res.summary()["Midland"]["population"] == 0.0
    assert all(bal.mass_balance_error() < 1e-6 for bal in res.balances)


def test_crops_but_no_hydropower_or_thermal():
    wheat = Crop("wheat", area_ha=10_000, kc=0.85, season_days=150, yield_max_t_ha=4.0, ky=1.05, kcal_per_kg=3400, price_usd_t=250)
    b = solo_basin(crops=[wheat], et0_mm_day=4.0, effective_rainfall_mm=300.0, irrigation_efficiency=0.5,
                   energy=EnergySystem(demand_gwh=100.0))
    rec = run_nexus(b, Scenario(years=1)).record("Solo", 2025)
    assert rec.irrigation_requirement_mm3 == pytest.approx(42.0)  # (4*0.85*150-300)/0.5 mm over 10 000 ha
    assert rec.demands[AGR] == pytest.approx(42.0)
    assert rec.hydropower_gwh == 0.0 and rec.thermal_gwh == 0.0 and rec.emissions_t == 0.0
    assert rec.energy_supply_gwh == 0.0 and rec.energy_deficit_gwh == pytest.approx(rec.energy_demand_gwh)
    assert rec.food_production_t == pytest.approx(40_000.0) and rec.et_ratio == 1.0
    assert rec.food_kcal == pytest.approx(40_000.0 * 1000 * 3400)
    assert rec.crop_value_usd == pytest.approx(40_000.0 * 250)


def test_run_nexus_kwargs_and_model_equivalence(basin, baseline):
    direct = NexusModel(basin, baseline, allocation_rule="proportional", reservoir_refill_fraction=0.1).run()
    via = run_nexus(basin, baseline, allocation_rule="proportional", reservoir_refill_fraction=0.1)
    assert direct.to_records() == via.to_records() and via.allocation_rule == "proportional"
    assert run_nexus(basin, baseline).to_records() == NexusModel(basin, baseline).run().to_records()
    with pytest.raises(TypeError):
        run_nexus(basin, baseline, bogus=1)
    # a model can be run repeatedly with identical results
    m = NexusModel(basin, baseline)
    assert m.run().to_records() == m.run().to_records()


def test_assess_integration(basin, baseline, result):
    rep = sustainability.assess(result)
    assert rep.names() == basin.names() and rep.years == result.years
    assert rep.scenario_name == baseline.name
    for name in basin.names():
        assert rep.riparian(name)["nexus_index"] == pytest.approx(result.summary()[name]["nexus_index"])
        assert rep.riparian(name)["supply_ratio"] == pytest.approx(result.summary()[name]["supply_ratio"])
    assert rep.basin["mass_balance_error_mm3"] < 1e-6
    assert rep.basin["natural_flow_mm3"] == pytest.approx(28000.0)
    assert BASIN_KEY in rep.summary()


# ---------------------------------------------------------------------------
# review fixes: groundwater pumping energy, renewable-share definitions,
# scenario driver validation
# ---------------------------------------------------------------------------
def test_groundwater_pumping_energy_is_counted(basin, result):
    """Review finding: the groundwater actually abstracted is lifted, on top of the pumped
    share of the surface withdrawals that energy_for_water() covers."""
    d = result.record("Delta", 2025)
    assert d.groundwater_used_mm3 == pytest.approx(2000.0)
    assert d.groundwater_pumping_gwh == pytest.approx(energy.pumping_energy_gwh(2000.0, 40.0, 0.6))
    assert d.groundwater_pumping_gwh == pytest.approx(1000.0 * 9.81 * 2000.0e6 * 40.0 / 0.6 / 3.6e12)  # 363.3 GWh
    assert d.groundwater_pumping_gwh == pytest.approx(363.33, abs=0.01)
    assert result.record("Midland", 2025).groundwater_pumping_gwh == pytest.approx(energy.pumping_energy_gwh(900.0, 35.0, 0.6))
    assert result.record("Midland", 2025).groundwater_pumping_gwh == pytest.approx(143.06, abs=0.01)
    assert result.record("Highland", 2025).groundwater_pumping_gwh == pytest.approx(energy.pumping_energy_gwh(200.0, 20.0, 0.6))
    for rec in result.records:
        r = basin.riparian(rec.name)
        surface = energy.energy_for_water(r, rec.withdrawals, rec.desalinated_mm3)
        assert rec.energy_for_water_gwh == pytest.approx(surface["total"] + rec.groundwater_pumping_gwh)
        assert rec.energy_for_water_gwh > surface["total"]
        assert rec.energy_demand_gwh == pytest.approx(r.energy.demand_gwh + rec.energy_for_water_gwh)
        assert rec.energy_deficit_gwh == pytest.approx(max(rec.energy_demand_gwh - rec.energy_supply_gwh, 0.0))
        assert rec.to_dict()["groundwater_pumping_gwh"] == rec.groundwater_pumping_gwh
    s = result.summary()
    assert s["Delta"]["groundwater_pumping_gwh"] == pytest.approx(sum(result.series("Delta", "groundwater_pumping_gwh")) / 5)
    # without groundwater the term vanishes and energy-for-water is the surface-water account alone
    # (Delta stays treaty-capped at 16 000 Mm3, but its sector split - hence treatment energy - shifts)
    no_gw = basin.copy()
    for r in no_gw.riparians:
        r.groundwater_abstraction_mm3 = 0.0
    rec0 = run_nexus(no_gw, Scenario(years=1)).record("Delta", 2025)
    assert rec0.groundwater_used_mm3 == 0.0 and rec0.groundwater_pumping_gwh == 0.0
    assert rec0.total_withdrawal() == pytest.approx(16000.0) == pytest.approx(d.total_withdrawal())
    surface0 = energy.energy_for_water(no_gw.riparian("Delta"), rec0.withdrawals, rec0.desalinated_mm3)
    assert rec0.energy_for_water_gwh == pytest.approx(surface0["total"])
    assert d.energy_for_water_gwh == pytest.approx(
        energy.energy_for_water(basin.riparian("Delta"), d.withdrawals, d.desalinated_mm3)["total"] + 363.33, abs=0.01)
    # the term scales with the lift and inversely with the pump efficiency
    tall = basin.copy()
    tall.riparian("Delta").energy.pumping_lift_m = 80.0
    assert run_nexus(tall, Scenario(years=1)).record("Delta", 2025).groundwater_pumping_gwh == pytest.approx(2 * d.groundwater_pumping_gwh)
    weak = basin.copy()
    weak.riparian("Delta").energy.pumping_efficiency = 0.3
    assert run_nexus(weak, Scenario(years=1)).record("Delta", 2025).groundwater_pumping_gwh == pytest.approx(2 * d.groundwater_pumping_gwh)
    # only the groundwater actually used is lifted: no demand -> no abstraction -> no pumping
    idle = solo_basin(groundwater_abstraction_mm3=50.0, demand=WaterDemand(environmental=100.0))
    rec = run_nexus(idle, Scenario(years=1)).record("Solo", 2025)
    assert rec.groundwater_used_mm3 == 0.0 and rec.groundwater_pumping_gwh == 0.0 and rec.energy_for_water_gwh == 0.0
    busy = solo_basin(groundwater_abstraction_mm3=50.0)
    rec = run_nexus(busy, Scenario(years=1)).record("Solo", 2025)
    assert rec.groundwater_used_mm3 == pytest.approx(50.0)
    assert rec.groundwater_pumping_gwh == pytest.approx(energy.pumping_energy_gwh(50.0, 30.0, 0.6))


def test_renewable_share_input_and_sdg72_definitions_are_both_reported(basin, result):
    """Review finding: EnergySystem.renewable_share is the NON-hydro share of demand (hydropower is
    modelled separately); the record carries that input and the supply-based SDG 7.2 share."""
    for rec in result.records:
        es = basin.riparian(rec.name).energy
        base_demand = rec.energy_demand_gwh - rec.energy_for_water_gwh
        assert rec.other_renewable_share_of_demand == pytest.approx(es.renewable_share)
        assert rec.other_renewable_gwh == pytest.approx(base_demand * rec.other_renewable_share_of_demand)
        assert rec.renewable_share == pytest.approx((rec.hydropower_gwh + rec.other_renewable_gwh) / rec.energy_supply_gwh)
        assert rec.to_dict()["other_renewable_share_of_demand"] == rec.other_renewable_share_of_demand
        assert 0.0 <= rec.other_renewable_share_of_demand <= 1.0
    # Highland: 75 % non-hydro renewables plus ~3900 GWh of hydropower -> SDG 7.2 share above the input
    h = result.record("Highland", 2025)
    assert h.other_renewable_share_of_demand == 0.75 and h.renewable_share > 0.75
    assert h.renewable_share == pytest.approx(0.907, abs=0.01)
    assert h.hydropower_gwh == pytest.approx(3873.0, abs=5.0)
    s = result.summary()["Highland"]
    assert s["other_renewable_share_of_demand"] == pytest.approx(0.75)
    assert s["renewable_share"] > s["other_renewable_share_of_demand"]
    # a riparian without hydropower: both definitions coincide when supply equals demand
    flat = solo_basin(energy=EnergySystem(demand_gwh=1000.0, thermal_capacity_mw=1000.0 * 0.7 / 8.76,
                                          thermal_capacity_factor=0.7, renewable_share=0.3))
    rec = run_nexus(flat, Scenario(years=1)).record("Solo", 2025)
    assert rec.hydropower_gwh == 0.0 and rec.other_renewable_gwh == pytest.approx(300.0)
    assert rec.other_renewable_share_of_demand == pytest.approx(0.3)
    assert rec.renewable_share == pytest.approx(300.0 / rec.energy_supply_gwh)
    # a scenario target drives the non-hydro share, interpolated linearly
    res = run_nexus(basin, Scenario(years=5, renewable_share_target=0.5))
    assert res.series("Delta", "other_renewable_share_of_demand") == pytest.approx([0.12 + (0.5 - 0.12) * i / 4 for i in range(5)])
    # the definitions are spelled out in the documentation
    assert "non-hydro" in nexus_module.__doc__ and "SDG 7.2" in nexus_module.__doc__
    assert "non-hydro" in NexusModel.__doc__ and "SDG 7.2" in RiparianYear.__doc__


@pytest.mark.parametrize(
    "field, value",
    [
        ("drought_severity", -1.0),
        ("drought_severity", 1.5),
        ("drought_severity", math.nan),
        ("drought_severity", "dry"),
        ("flow_change_pct_by_end", -150.0),
        ("flow_change_pct_by_end", math.inf),
        ("irrigated_area_change_pct_by_end", -101.0),
        ("population_growth_rate", -1.0),
        ("gdp_growth_rate", -2.0),
        ("demand_growth_rate", "fast"),
        ("energy_demand_growth_rate", -1.5),
        ("irrigation_efficiency_target", 0.0),
        ("irrigation_efficiency_target", 1.2),
        ("renewable_share_target", -0.1),
        ("renewable_share_target", 1.5),
        ("seed", -1),
        ("seed", 1.5),
        ("drought_years", [-1]),
        ("drought_years", [1.5]),
        ("drought_years", [True]),
        ("drought_years", "01"),
        ("drought_years", 3),
        ("cooperation", "yes"),
        ("stochastic", None),
        ("start_year", 2025.5),
        ("years", 0),
        ("years", 2.5),
        ("years", True),
    ],
)
def test_model_rejects_bad_scenario_drivers(basin, field, value):
    """Review finding: drivers that Scenario.flow_factor would silently clamp (or that make no
    sense) are rejected at construction with the field named, as validate_scenario does."""
    kw = {"years": 3}
    kw[field] = value
    with pytest.raises(ValueError, match=field):
        NexusModel(basin, Scenario(**kw))
    with pytest.raises(ValueError, match=field):
        run_nexus(basin, Scenario(**kw))


def test_flow_factors_are_never_clamped_for_valid_scenarios(basin):
    # boundary values are legal and give exactly zero flow without any clamping
    assert run_nexus(basin, Scenario(years=3, drought_years=[0], drought_severity=1.0)).flow_factors == [0.0, 1.0, 1.0]
    assert run_nexus(basin, Scenario(years=3, flow_change_pct_by_end=-100.0)).flow_factors == pytest.approx([1.0, 0.5, 0.0])
    # every valid scenario yields factors equal to the unclamped trend, all non-negative
    for sc in (
        Scenario(years=4, flow_change_pct_by_end=-60.0, drought_years=[1, 3], drought_severity=0.9),
        Scenario(years=4, flow_change_pct_by_end=40.0, drought_years=[0], drought_severity=0.0),
        Scenario(years=1, flow_change_pct_by_end=-100.0, drought_years=[0], drought_severity=1.0),
    ):
        model = NexusModel(basin, sc)
        for i, f in enumerate(model.flow_factors()):
            trend = 1.0 + sc.flow_change_pct_by_end / 100.0 * sc.progress(i)
            if i in sc.drought_years:
                trend *= 1.0 - sc.drought_severity
            assert f == pytest.approx(trend) and f >= 0.0
    # the scenarios of the review finding no longer run (a "drought" that doubled the river,
    # a severity above 1, a flow change below -100 %)
    with pytest.raises(ValueError, match="drought_severity"):
        NexusModel(basin, Scenario(years=3, drought_years=[0], drought_severity=-1.0))
    with pytest.raises(ValueError, match="drought_severity"):
        NexusModel(basin, Scenario(years=3, drought_years=[0], drought_severity=1.5))
    with pytest.raises(ValueError, match="flow_change_pct_by_end"):
        NexusModel(basin, Scenario(years=3, flow_change_pct_by_end=-150))
    # drought offsets beyond the horizon are harmless; growth rates above -1, 0/1 flags and
    # integral floats are accepted as in validate_scenario
    ok = Scenario(years=2, drought_years=[7], population_growth_rate=-0.5, cooperation=1, stochastic=0,
                  seed=3.0, start_year=2030.0, drought_severity=0, irrigated_area_change_pct_by_end=-100)
    res = run_nexus(basin, ok)
    assert res.n_years == 2 and res.years == [2030, 2031] and res.allocation_rule == "treaty"
    # the validator agrees with wefnexus.scenarios.validate_scenario on the bad and the good cases
    scenarios = pytest.importorskip("wefnexus.scenarios")
    for kw in (dict(drought_severity=-1.0), dict(flow_change_pct_by_end=-150), dict(seed=-1),
               dict(population_growth_rate=-1.0), dict(drought_years=[-1]), dict(renewable_share_target=2)):
        with pytest.raises(ValueError):
            scenarios.validate_scenario(Scenario(years=3, **kw))
        with pytest.raises(ValueError):
            NexusModel(basin, Scenario(years=3, **kw))
    assert scenarios.validate_scenario(ok).years == 2
