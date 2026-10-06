import copy
import math
from dataclasses import replace

import pytest

from wefnexus.data import example_basin
from wefnexus.models import (
    DEFAULT_CONSUMPTION_FRACTION,
    DEFAULT_VALUE_USD_PER_M3,
    SECTOR_PRIORITY,
    Basin,
    Crop,
    EnergySystem,
    Riparian,
    Scenario,
    Sector,
    WaterDemand,
)


def test_water_demand_totals():
    d = WaterDemand(municipal=10, industrial=5, agricultural=100, energy=2, environmental=50)
    assert d.total_withdrawal() == pytest.approx(117)
    assert Sector.ENVIRONMENT not in d.withdrawals()
    # consumptive demand uses default fractions
    assert d.consumptive_demand() == pytest.approx(10 * 0.2 + 5 * 0.1 + 100 * 0.6 + 2 * 0.03)
    scaled = d.scaled(2.0)
    assert scaled.municipal == 20 and scaled.agricultural == 200
    assert scaled.environmental == 50  # not a withdrawal, untouched
    assert d.municipal == 10  # original untouched


def test_basin_positions_and_lookup(basin):
    assert basin.names() == ["Highland", "Midland", "Delta"]
    assert [r.position for r in basin] == [0, 1, 2]
    assert [r.name for r in basin.upstream_of("Delta")] == ["Highland", "Midland"]
    assert [r.name for r in basin.downstream_of("Highland")] == ["Midland", "Delta"]
    assert basin.total_natural_flow() == pytest.approx(28_000.0)
    with pytest.raises(KeyError):
        basin.riparian("Nowhere")


def test_basin_rejects_duplicate_names():
    r = lambda n: Riparian(name=n, population=1, gdp_usd=1, local_inflow_mm3=1, demand=WaterDemand())
    with pytest.raises(ValueError):
        Basin("dup", [r("A"), r("A")])


def test_basin_copy_is_deep(basin):
    b2 = basin.copy()
    assert b2 == basin and b2 is not basin
    assert [r.position for r in b2] == [0, 1, 2]
    # scalars and crops ...
    b2.riparian("Delta").demand.municipal = 1.0
    b2.riparian("Delta").crops[0].area_ha = 1.0
    # ... and the nested sector dicts / energy system: a shallow
    # ``dataclasses.replace(r.demand)`` would leave these SHARED with the
    # original, so mutating them on the copy must not leak back.
    b2.riparian("Delta").demand.consumption_fraction[Sector.AGRICULTURAL] = 0.99
    b2.riparian("Delta").demand.value_usd_per_m3[Sector.MUNICIPAL] = 99.0
    b2.riparian("Delta").energy.demand_gwh = 1.0
    assert basin.riparian("Delta").demand.municipal == 3_500.0
    assert basin.riparian("Delta").crops[0].area_ha == 200_000
    assert basin.riparian("Delta").demand.consumption_fraction[Sector.AGRICULTURAL] == 0.6
    assert basin.riparian("Delta").demand.value_usd_per_m3[Sector.MUNICIPAL] == 1.5
    assert basin.riparian("Delta").energy.demand_gwh == 160_000.0
    # the edits did land on the copy
    assert b2.riparian("Delta").demand.consumption_fraction[Sector.AGRICULTURAL] == 0.99
    assert b2.riparian("Delta").demand.value_usd_per_m3[Sector.MUNICIPAL] == 99.0
    assert b2.riparian("Delta").energy.demand_gwh == 1.0
    # nested containers are distinct objects, not shared references
    for orig, cp in zip(basin, b2):
        assert cp is not orig
        assert cp.demand is not orig.demand
        assert cp.demand.consumption_fraction is not orig.demand.consumption_fraction
        assert cp.demand.value_usd_per_m3 is not orig.demand.value_usd_per_m3
        assert cp.crops is not orig.crops
        assert all(c1 is not c0 for c0, c1 in zip(orig.crops, cp.crops))
        assert cp.energy is not orig.energy
    assert b2.riparians is not basin.riparians


def test_basin_copy_sector_dicts_are_independent(basin):
    """Finding: editing a copy's consumption/value dicts must not leak into the original."""
    b2 = basin.copy()
    b2.riparian("Highland").demand.consumption_fraction[Sector.AGRICULTURAL] = 1.0
    b2.riparian("Highland").demand.value_usd_per_m3[Sector.MUNICIPAL] = 99.0
    b2.riparian("Midland").demand.consumption_fraction.pop(Sector.ENERGY)
    b2.riparian("Delta").energy.demand_gwh = 1.0
    b2.riparians.append(Riparian("Extra", 1, 1, 1, WaterDemand()))
    b2.riparian("Delta").crops.clear()
    hl = basin.riparian("Highland")
    assert hl.demand.consumption_fraction[Sector.AGRICULTURAL] == DEFAULT_CONSUMPTION_FRACTION[Sector.AGRICULTURAL]
    assert hl.demand.value_usd_per_m3[Sector.MUNICIPAL] == DEFAULT_VALUE_USD_PER_M3[Sector.MUNICIPAL]
    assert Sector.ENERGY in basin.riparian("Midland").demand.consumption_fraction
    assert basin.riparian("Delta").energy.demand_gwh == 160_000.0
    assert basin.names() == ["Highland", "Midland", "Delta"]
    assert len(basin.riparian("Delta").crops) == 4
    # the copy really changed (the edits were not silently dropped)
    assert b2.riparian("Highland").demand.consumption_fraction[Sector.AGRICULTURAL] == 1.0
    assert b2 != basin


def test_basin_copy_matches_deepcopy_and_keeps_original_routing_unchanged():
    """A what-if on a copy must leave the baseline basin (and its routing) untouched."""
    from wefnexus.water import route_basin

    b = example_basin()
    reference = route_basin(b)
    before = copy.deepcopy(b)
    c = b.copy()
    assert c == copy.deepcopy(b)
    c.riparian("Highland").demand.consumption_fraction[Sector.AGRICULTURAL] = 1.0
    c.riparian("Midland").demand.consumption_fraction[Sector.AGRICULTURAL] = 1.0
    assert b == before
    after = route_basin(b)
    assert after.outflow_to_sea_mm3 == pytest.approx(reference.outflow_to_sea_mm3)
    for r0, r1 in zip(reference.reaches, after.reaches):
        assert r1.total_consumption() == pytest.approx(r0.total_consumption())
    # ... while the copy itself consumes more
    greedy = route_basin(c)
    assert greedy.reach("Highland").total_consumption() > reference.reach("Highland").total_consumption()


def test_riparian_copy_is_independent(basin):
    r = basin.riparian("Highland")
    r2 = r.copy()
    assert r2 == r and r2 is not r
    assert r2.position == r.position
    assert r2.demand is not r.demand
    assert r2.demand.consumption_fraction is not r.demand.consumption_fraction
    assert r2.demand.value_usd_per_m3 is not r.demand.value_usd_per_m3
    assert r2.crops is not r.crops and all(a is not b for a, b in zip(r.crops, r2.crops))
    assert r2.energy is not r.energy
    r2.energy.demand_gwh = 5.0
    r2.demand.municipal = 7.0
    r2.demand.consumption_fraction[Sector.MUNICIPAL] = 0.99
    r2.demand.value_usd_per_m3[Sector.AGRICULTURAL] = 42.0
    r2.crops.append(Crop("x", area_ha=1, kc=1, season_days=1, yield_max_t_ha=1, ky=1, kcal_per_kg=1))
    r2.crops[0].area_ha = 1.0
    assert r.energy.demand_gwh == 12_000.0
    assert r.demand.municipal == 600.0
    assert r.demand.consumption_fraction[Sector.MUNICIPAL] == 0.2
    assert r.demand.value_usd_per_m3[Sector.AGRICULTURAL] == 0.10
    assert len(r.crops) == 2 and r.crops[0].area_ha == 150_000
    assert basin.riparian("Highland") is r  # the basin itself untouched


def test_riparian_copy_changes_override_and_are_used_as_passed(basin):
    delta = basin.riparian("Delta")
    stronger = delta.copy(material_power=1.0, position=0)
    assert stronger.material_power == 1.0 and stronger.position == 0
    assert delta.material_power == 0.8 and delta.position == 2
    assert stronger.demand == delta.demand and stronger.demand is not delta.demand
    # an explicitly supplied nested field is used verbatim (not copied again)
    crops = [replace(c, area_ha=c.area_ha * 1.2) for c in delta.crops]
    demand = WaterDemand(municipal=1.0)
    energy = EnergySystem(demand_gwh=3.0)
    custom = delta.copy(crops=crops, demand=demand, energy=energy)
    assert custom.crops is crops and custom.demand is demand and custom.energy is energy
    assert delta.crops[0].area_ha == 200_000 and delta.demand.municipal == 3_500.0
    # unknown fields are rejected with a clear message
    with pytest.raises(ValueError, match="unknown Riparian field"):
        delta.copy(not_a_field=1.0)
    with pytest.raises(ValueError, match="not_a_field"):
        delta.copy(material_power=0.9, not_a_field=1.0)
    assert delta.material_power == 0.8


def test_riparian_copy_without_crops_or_reservoir():
    r = Riparian(name="Solo", population=10.0, gdp_usd=1.0, local_inflow_mm3=5.0, demand=WaterDemand(municipal=1.0))
    r2 = r.copy()
    assert r2 == r and r2.crops == [] and r2.crops is not r.crops
    assert r2.energy == EnergySystem() and r2.energy is not r.energy
    assert r2.reservoir_capacity_mm3 == 0.0 and r2.treaty_allocation_mm3 is None


def test_water_demand_copy():
    d = WaterDemand(municipal=10, industrial=5, agricultural=100, energy=2, environmental=50)
    c = d.copy()
    assert c == d and c is not d
    assert c.consumption_fraction is not d.consumption_fraction
    assert c.value_usd_per_m3 is not d.value_usd_per_m3
    c.consumption_fraction[Sector.MUNICIPAL] = 0.99
    c.value_usd_per_m3[Sector.ENERGY] = 123.0
    c.municipal = 0.0
    assert d.consumption_fraction == DEFAULT_CONSUMPTION_FRACTION
    assert d.value_usd_per_m3 == DEFAULT_VALUE_USD_PER_M3
    assert d.municipal == 10
    # custom (non-default) dicts are copied too
    custom = WaterDemand(municipal=1.0, consumption_fraction={Sector.MUNICIPAL: 0.5}, value_usd_per_m3={})
    cc = custom.copy()
    assert cc.consumption_fraction == {Sector.MUNICIPAL: 0.5} and cc.consumption_fraction is not custom.consumption_fraction
    assert cc.value_usd_per_m3 == {} and cc.value_usd_per_m3 is not custom.value_usd_per_m3


def test_water_demand_default_dicts_are_per_instance():
    a, b = WaterDemand(), WaterDemand()
    assert a.consumption_fraction is not b.consumption_fraction
    assert a.consumption_fraction is not DEFAULT_CONSUMPTION_FRACTION
    a.consumption_fraction[Sector.MUNICIPAL] = 0.5
    assert b.consumption_fraction[Sector.MUNICIPAL] == 0.2
    assert DEFAULT_CONSUMPTION_FRACTION[Sector.MUNICIPAL] == 0.2


def test_scaled_environment_sector_uses_environmental_field():
    """Finding: Sector.ENVIRONMENT ('environment') maps onto the 'environmental' field."""
    d = WaterDemand(municipal=1.0, environmental=5.0)
    s = d.scaled(2.0, [Sector.ENVIRONMENT])
    assert s.environmental == pytest.approx(10.0)
    assert s.municipal == 1.0  # not selected, untouched
    assert d.environmental == 5.0 and d.municipal == 1.0  # original untouched
    # every member of the enum can be selected, by member or by string value
    for sec in Sector:
        by_member = d.scaled(0.0, [sec])
        by_value = d.scaled(0.0, [sec.value])
        assert by_member == by_value
    zero_all = d.scaled(0.0, list(Sector))
    assert zero_all.total_withdrawal() == 0.0 and zero_all.environmental == 0.0


def test_scaled_default_and_explicit_sector_selection():
    d = WaterDemand(municipal=10, industrial=5, agricultural=100, energy=2, environmental=50)
    s = d.scaled(1.5)
    assert s.withdrawals() == pytest.approx({k: v * 1.5 for k, v in d.withdrawals().items()})
    assert s.environmental == 50  # default scales only SECTOR_PRIORITY
    assert d.scaled(1.5, list(SECTOR_PRIORITY)) == s
    assert d.scaled(1.5, None) == s
    only_mun = d.scaled(3.0, ["municipal"])
    assert only_mun.municipal == 30 and only_mun.agricultural == 100
    # an explicit empty selection scales nothing but still returns a copy
    same = d.scaled(3.0, [])
    assert same == d and same is not d
    # scaling by one is the identity; by zero empties the selected sectors
    assert d.scaled(1.0) == d
    assert d.scaled(0.0).total_withdrawal() == 0.0
    # a generator / tuple is accepted as the sector iterable
    assert d.scaled(2.0, (s for s in [Sector.ENERGY])).energy == 4
    assert d.scaled(2.0, (Sector.ENERGY, "industrial")).industrial == 10


def test_scaled_returns_independent_copy():
    d = WaterDemand(municipal=10, environmental=50)
    s = d.scaled(2.0)
    assert s.consumption_fraction == d.consumption_fraction
    assert s.consumption_fraction is not d.consumption_fraction
    assert s.value_usd_per_m3 is not d.value_usd_per_m3
    s.consumption_fraction[Sector.MUNICIPAL] = 0.99
    assert d.consumption_fraction[Sector.MUNICIPAL] == 0.2
    assert d.municipal == 10 and s.municipal == 20


def test_scaled_validates_inputs():
    d = WaterDemand(municipal=10, environmental=50)
    for bad in (-0.5, math.nan, math.inf, -math.inf, "2", None, True, [2.0]):
        with pytest.raises(ValueError, match="factor"):
            d.scaled(bad)
    for bad_sectors in (["bogus"], [3], [None], [Sector.MUNICIPAL, "water"]):
        with pytest.raises(ValueError, match="unknown sector"):
            d.scaled(2.0, bad_sectors)
    # nothing changed on the way out of a failed call
    assert d == WaterDemand(municipal=10, environmental=50)
    # numpy scalars count as real numbers
    np = pytest.importorskip("numpy")
    assert d.scaled(np.float64(2.0)).municipal == pytest.approx(20.0)
    assert d.scaled(np.int64(3)).municipal == pytest.approx(30.0)


def test_scenario_factors():
    s = Scenario(years=11, flow_change_pct_by_end=-20, population_growth_rate=0.01,
                 drought_years=[5], drought_severity=0.5, irrigation_efficiency_target=0.7)
    assert s.flow_factor(0) == pytest.approx(1.0)
    assert s.flow_factor(10) == pytest.approx(0.8)
    assert s.flow_factor(5) == pytest.approx(0.9 * 0.5)
    assert s.population_factor(10) == pytest.approx(1.01 ** 10)
    assert s.irrigation_efficiency(0.5, 0) == pytest.approx(0.5)
    assert s.irrigation_efficiency(0.5, 10) == pytest.approx(0.7)
    assert s.year(3) == 2028
    assert Scenario(years=1).progress(0) == 1.0
