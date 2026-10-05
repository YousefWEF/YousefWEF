import pytest

from wefnexus.models import Basin, Riparian, Scenario, Sector, WaterDemand


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
    b2.riparian("Delta").demand.municipal = 1.0
    b2.riparian("Delta").crops[0].area_ha = 1.0
    assert basin.riparian("Delta").demand.municipal == 3_500.0
    assert basin.riparian("Delta").crops[0].area_ha == 200_000


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
