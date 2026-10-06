"""Tests for :mod:`wefnexus.optimize` - LP allocation and Pareto front."""
import copy
import json
import math

import numpy as np
import pytest

from wefnexus.allocation import gini
from wefnexus.models import (
    SECTOR_PRIORITY,
    Basin,
    EnergySystem,
    Riparian,
    Sector,
    WaterDemand,
)
from wefnexus.optimize import (
    M3_PER_MM3,
    OBJECTIVES,
    OptimizationResult,
    optimize_allocation,
    pareto_front,
    route_allocation,
)
from wefnexus.water import natural_flows, route_basin

MUN, IND, ENE, AGR = Sector.MUNICIPAL, Sector.INDUSTRIAL, Sector.ENERGY, Sector.AGRICULTURAL
FLOW_FACTORS = [0.3, 0.6, 1.0]
#: relative tolerance for LP feasibility checks (HiGHS primal tolerance is 1e-7)
LP_TOL = 1e-6


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
    values=None,
    gw=0.0,
    desal=0.0,
    treaty=None,
):
    """Small riparian with explicit numbers (no reservoir, no crops)."""
    kwargs = {}
    if fractions is not None:
        kwargs["consumption_fraction"] = dict(fractions)
    if values is not None:
        kwargs["value_usd_per_m3"] = dict(values)
    demand = WaterDemand(municipal=mun, industrial=ind, agricultural=ag, energy=energy, environmental=env, **kwargs)
    return Riparian(
        name=name,
        population=1_000_000,
        gdp_usd=1e9,
        local_inflow_mm3=local,
        demand=demand,
        groundwater_abstraction_mm3=gw,
        energy=EnergySystem(desalination_capacity_mm3=desal),
        treaty_allocation_mm3=treaty,
    )


def single(local, env=0.0, **kw):
    return Basin("single", [make_riparian("A", local=local, env=env, **kw)])


def river_demands(r):
    """Demand on the river per sector, as route_basin / optimize define it."""
    gross = r.demand.withdrawals()
    total = sum(gross.values())
    nr = min(r.groundwater_abstraction_mm3 + r.energy.desalination_capacity_mm3, total)
    if total <= 0.0:
        return {s: 0.0 for s in SECTOR_PRIORITY}, 0.0, 0.0
    return {s: gross[s] * (1.0 - nr / total) for s in SECTOR_PRIORITY}, nr, total


def assert_feasible(basin, res, ff, env_hard=True, msr=0.0, caps=None):
    """Check every LP constraint and every reported quantity of ``res``."""
    assert res.success, res.message
    assert res.status == "optimal"
    assert list(res.allocations) == basin.names()
    assert list(res.outflows) == basin.names()
    upstream = basin.headwater_inflow_mm3 * ff
    for r in basin.riparians:
        alloc = res.allocations[r.name]
        assert list(alloc) == SECTOR_PRIORITY
        rd, nr, total = river_demands(r)
        assert res.non_river_supply[r.name] == pytest.approx(nr)
        for s in SECTOR_PRIORITY:
            assert -1e-9 <= alloc[s] <= rd[s] + 1e-9, (r.name, s, alloc[s], rd[s])
            assert res.river_demands[r.name][s] == pytest.approx(rd[s])
        inflow = upstream + r.local_inflow_mm3 * ff
        assert res.inflows[r.name] == pytest.approx(inflow, rel=1e-9, abs=1e-9)
        withdrawal = sum(alloc.values())
        consumption = sum(alloc[s] * r.demand.consumption_fraction.get(s, 0.0) for s in SECTOR_PRIORITY)
        assert withdrawal <= inflow + LP_TOL * max(1.0, inflow)
        for s in SECTOR_PRIORITY:
            assert res.consumption[r.name][s] == pytest.approx(alloc[s] * r.demand.consumption_fraction.get(s, 0.0))
        outflow = inflow - consumption
        assert res.outflows[r.name] == pytest.approx(outflow, rel=1e-9, abs=1e-9)
        assert outflow >= -LP_TOL * max(1.0, inflow)
        env = r.demand.environmental
        if env_hard:
            assert outflow >= env - LP_TOL * max(1.0, env), (r.name, outflow, env)
            assert res.env_flow_met[r.name]
        sr = 1.0 if total <= 0.0 else min(1.0, (withdrawal + nr) / total)
        assert res.supply_ratios[r.name] == pytest.approx(sr, abs=1e-9)
        if msr > 0.0 and total > 0.0:
            assert sr >= msr - LP_TOL
        if caps and caps.get(r.name) is not None:
            assert withdrawal <= caps[r.name] + LP_TOL * max(1.0, caps[r.name])
        benefit = sum(alloc[s] * r.demand.value_usd_per_m3.get(s, 0.0) for s in SECTOR_PRIORITY) * M3_PER_MM3
        assert res.benefits_usd[r.name] == pytest.approx(benefit, rel=1e-9, abs=1e-6)
        upstream = outflow
    assert res.total_benefit_usd == pytest.approx(sum(res.benefits_usd.values()), rel=1e-9)
    assert res.min_supply_ratio == pytest.approx(min(res.supply_ratios.values()))
    assert res.gini == pytest.approx(gini(list(res.supply_ratios.values())))
    assert 0.0 <= res.gini <= 1.0
    assert res.outflow_to_sea_mm3 == pytest.approx(res.outflows[basin.names()[-1]])


# ---------------------------------------------------------------------------
# basics
# ---------------------------------------------------------------------------
class TestBasics:
    def test_objectives_constant(self):
        assert OBJECTIVES == ("benefit", "equity", "weighted")
        assert M3_PER_MM3 == 1e6

    @pytest.mark.parametrize("objective", OBJECTIVES)
    @pytest.mark.parametrize("ff", FLOW_FACTORS)
    def test_solution_is_feasible(self, basin, ff, objective):
        res = optimize_allocation(basin, ff, objective, equity_weight=0.5)
        assert isinstance(res, OptimizationResult)
        assert res.objective_name == objective
        assert res.flow_factor == ff
        assert_feasible(basin, res, ff)

    def test_abundant_water_serves_all_river_demand(self, basin):
        res = optimize_allocation(basin, 1.0, "benefit")
        assert_feasible(basin, res, 1.0)
        expected = 0.0
        for r in basin.riparians:
            rd, _, _ = river_demands(r)
            for s in SECTOR_PRIORITY:
                assert res.allocations[r.name][s] == pytest.approx(rd[s], rel=1e-7)
                expected += rd[s] * r.demand.value_usd_per_m3[s] * M3_PER_MM3
            assert res.supply_ratios[r.name] == pytest.approx(1.0)
        assert res.total_benefit_usd == pytest.approx(expected, rel=1e-7)
        assert res.objective == pytest.approx(expected, rel=1e-7)
        assert res.min_supply_ratio == pytest.approx(1.0)
        assert res.gini == pytest.approx(0.0, abs=1e-9)
        # benefit and equity coincide when nobody is short
        eq = optimize_allocation(basin, 1.0, "equity")
        assert eq.total_benefit_usd == pytest.approx(res.total_benefit_usd, rel=1e-7)

    def test_natural_flows_match_water_module(self, basin):
        for ff in FLOW_FACTORS:
            res = optimize_allocation(basin, ff)
            assert res.natural_flows == pytest.approx(natural_flows(basin, ff))

    def test_routing_identity_links_reaches(self, basin):
        res = optimize_allocation(basin, 0.6)
        names = basin.names()
        for up, down in zip(names[:-1], names[1:]):
            local = basin.riparian(down).local_inflow_mm3 * 0.6
            assert res.inflows[down] == pytest.approx(res.outflows[up] + local)

    def test_no_mutation(self, basin):
        snapshot = copy.deepcopy(basin)
        optimize_allocation(basin, 0.6, "benefit", min_supply_ratio=0.3, entitlements="treaty")
        optimize_allocation(basin, 0.6, "equity")
        res = optimize_allocation(basin, 0.6, "weighted", equity_weight=0.4, values={AGR: 2.0})
        pareto_front(basin, 0.6, points=4)
        route_allocation(basin, res)
        assert basin == snapshot
        for r, s in zip(basin.riparians, snapshot.riparians):
            assert r.demand.value_usd_per_m3 == s.demand.value_usd_per_m3
            assert r.demand.withdrawals() == s.demand.withdrawals()


# ---------------------------------------------------------------------------
# benefit objective
# ---------------------------------------------------------------------------
class TestBenefit:
    @pytest.mark.parametrize("msr", [0.0, 0.3])
    @pytest.mark.parametrize("ff", [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0])
    def test_never_agriculture_before_municipal_when_short(self, basin, ff, msr):
        res = optimize_allocation(basin, ff, "benefit", min_supply_ratio=msr)
        assert res.success, res.message
        for name, alloc in res.allocations.items():
            rd = res.river_demands[name]
            if alloc[AGR] > 1e-6:
                assert alloc[MUN] >= rd[MUN] - 1e-6, (name, alloc, rd)
                # higher-valued sectors are also served before agriculture
                assert alloc[IND] >= rd[IND] - 1e-6
                assert alloc[ENE] >= rd[ENE] - 1e-6

    def test_single_reach_closed_form_withdrawal_bound(self):
        # inflow 100, env 40, municipal 30 (f 0.2, 1.5 $/m3), agriculture 100 (f 0.6, 0.1 $/m3)
        # municipal first (30, consumes 6); agriculture limited by withdrawal cap 100 - 30 = 70
        # (consumption 6 + 42 = 48 <= 100 - 40): outflow 52, benefit (30*1.5 + 70*0.1) M USD
        res = optimize_allocation(single(100.0, env=40.0, mun=30.0, ag=100.0))
        assert res.success
        assert res.allocations["A"][MUN] == pytest.approx(30.0, abs=1e-7)
        assert res.allocations["A"][AGR] == pytest.approx(70.0, abs=1e-7)
        assert res.outflows["A"] == pytest.approx(52.0, abs=1e-7)
        assert res.total_benefit_usd == pytest.approx(52.0e6, rel=1e-9)
        assert res.supply_ratios["A"] == pytest.approx(100.0 / 130.0)
        assert res.env_flow_met["A"]

    def test_single_reach_closed_form_environmental_bound(self):
        # env 60: consumption budget 40 - 6 = 34 -> agriculture 34/0.6 = 56.667; outflow exactly env
        res = optimize_allocation(single(100.0, env=60.0, mun=30.0, ag=100.0))
        assert res.success
        assert res.allocations["A"][MUN] == pytest.approx(30.0, abs=1e-7)
        assert res.allocations["A"][AGR] == pytest.approx(34.0 / 0.6, abs=1e-6)
        assert res.outflows["A"] == pytest.approx(60.0, abs=1e-6)
        assert res.total_benefit_usd == pytest.approx((45.0 + 3.4 / 0.6) * 1e6, rel=1e-8)

    def test_benefit_non_decreasing_in_flow(self, basin):
        benefits = [optimize_allocation(basin, ff).total_benefit_usd for ff in (0.2, 0.3, 0.5, 0.7, 1.0, 1.5)]
        for lo, hi in zip(benefits[:-1], benefits[1:]):
            assert hi >= lo * (1.0 - 1e-9)

    def test_values_change_the_optimum(self):
        # agriculture worth 10x municipal: serve it first; consumption 0.6*100 = 60 = 100 - env(40)
        basin = single(100.0, env=40.0, mun=30.0, ag=100.0)
        res = optimize_allocation(basin, values={AGR: 10.0, MUN: 1.0})
        assert res.allocations["A"][AGR] == pytest.approx(100.0, abs=1e-6)
        assert res.allocations["A"][MUN] == pytest.approx(0.0, abs=1e-6)
        assert res.total_benefit_usd == pytest.approx(1000.0e6, rel=1e-8)
        assert res.values_usd_per_m3["A"][AGR] == 10.0
        assert res.values_usd_per_m3["A"][IND] == 0.8        # untouched default
        # per-riparian partial override, string sector keys accepted
        res2 = optimize_allocation(basin, values={"A": {"agricultural": 10.0, "municipal": 1.0}})
        assert res2.allocations == res.allocations

    def test_benefit_counts_only_value_of_withdrawals(self, basin):
        res = optimize_allocation(basin, 0.6, values={s: 0.0 for s in SECTOR_PRIORITY})
        assert res.success
        assert res.total_benefit_usd == pytest.approx(0.0)
        assert res.objective == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# equity objective
# ---------------------------------------------------------------------------
class TestEquity:
    def test_equity_raises_min_supply_ratio_at_flow_0_6(self, basin):
        ben = optimize_allocation(basin, 0.6, "benefit")
        eq = optimize_allocation(basin, 0.6, "equity")
        assert_feasible(basin, ben, 0.6)
        assert_feasible(basin, eq, 0.6)
        assert ben.min_supply_ratio < 1.0
        assert eq.min_supply_ratio > ben.min_supply_ratio + 0.01
        assert eq.total_benefit_usd <= ben.total_benefit_usd * (1.0 + 1e-9)
        assert eq.gini < ben.gini
        assert eq.gini == pytest.approx(0.0, abs=1e-5)       # all supply ratios equalised
        ratios = list(eq.supply_ratios.values())
        assert max(ratios) - min(ratios) < 1e-5

    def test_equity_objective_is_the_max_min(self, basin):
        eq = optimize_allocation(basin, 0.6, "equity")
        assert eq.objective == pytest.approx(eq.min_supply_ratio, abs=1e-6)
        assert eq.objective >= eq.min_supply_ratio - 1e-12
        assert "max-min" in eq.message

    def test_equity_is_benefit_maximal_among_max_min_optima(self, basin):
        eq = optimize_allocation(basin, 0.6, "equity")
        constrained = optimize_allocation(basin, 0.6, "benefit", min_supply_ratio=eq.min_supply_ratio - 1e-9)
        assert constrained.success
        assert eq.total_benefit_usd == pytest.approx(constrained.total_benefit_usd, rel=1e-6)

    def test_two_riparians_share_equally(self):
        # 100 Mm3 enters at A; both fully consume (f = 1) and claim 100 -> max-min = 0.5 each
        basin = Basin(
            "two",
            [
                make_riparian("A", local=100.0, ag=100.0, fractions={AGR: 1.0}),
                make_riparian("B", local=0.0, ag=100.0, fractions={AGR: 1.0}),
            ],
        )
        eq = optimize_allocation(basin, objective="equity")
        assert eq.success
        assert eq.min_supply_ratio == pytest.approx(0.5, abs=1e-6)
        assert eq.supply_ratios["A"] == pytest.approx(0.5, abs=1e-6)
        assert eq.supply_ratios["B"] == pytest.approx(0.5, abs=1e-6)
        assert eq.gini == pytest.approx(0.0, abs=1e-6)
        assert eq.outflows["B"] == pytest.approx(0.0, abs=1e-6)
        assert eq.total_benefit_usd == pytest.approx(100.0 * 0.1 * 1e6, rel=1e-6)
        # upstream priority would give everything to A: benefit identical (same value), equity worse
        ben = optimize_allocation(basin, objective="benefit")
        assert ben.total_benefit_usd == pytest.approx(eq.total_benefit_usd, rel=1e-6)
        assert ben.min_supply_ratio <= eq.min_supply_ratio + 1e-9

    def test_price_of_fairness_in_unit_interval(self, basin):
        for ff in (0.2, 0.3, 0.6):
            ben = optimize_allocation(basin, ff, "benefit")
            eq = optimize_allocation(basin, ff, "equity")
            if ben.total_benefit_usd > 0:
                pof = 1.0 - eq.total_benefit_usd / ben.total_benefit_usd
                assert -1e-9 <= pof <= 1.0

    def test_zero_demand_riparian_does_not_bind(self):
        basin = Basin(
            "three",
            [
                make_riparian("A", local=100.0, ag=100.0, fractions={AGR: 1.0}),
                make_riparian("None", local=0.0),
                make_riparian("B", local=0.0, ag=100.0, fractions={AGR: 1.0}),
            ],
        )
        eq = optimize_allocation(basin, objective="equity")
        assert eq.supply_ratios["None"] == 1.0
        assert eq.supply_ratios["A"] == pytest.approx(0.5, abs=1e-6)
        assert eq.supply_ratios["B"] == pytest.approx(0.5, abs=1e-6)
        assert eq.min_supply_ratio == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# weighted objective
# ---------------------------------------------------------------------------
class TestWeighted:
    def test_weight_zero_matches_benefit(self, basin):
        ben = optimize_allocation(basin, 0.6, "benefit")
        w0 = optimize_allocation(basin, 0.6, "weighted", equity_weight=0.0)
        assert w0.total_benefit_usd == pytest.approx(ben.total_benefit_usd, rel=1e-7)
        assert w0.objective == pytest.approx(1.0, abs=1e-7)

    def test_weight_one_matches_equity(self, basin):
        eq = optimize_allocation(basin, 0.6, "equity")
        w1 = optimize_allocation(basin, 0.6, "weighted", equity_weight=1.0)
        assert w1.objective_name == "weighted"
        assert w1.min_supply_ratio == pytest.approx(eq.min_supply_ratio, abs=1e-7)
        assert w1.total_benefit_usd == pytest.approx(eq.total_benefit_usd, rel=1e-6)
        assert w1.objective == pytest.approx(eq.objective, abs=1e-7)

    @pytest.mark.parametrize("ew", [0.25, 0.5, 0.75])
    def test_intermediate_weights_lie_between(self, basin, ew):
        ben = optimize_allocation(basin, 0.6, "benefit")
        eq = optimize_allocation(basin, 0.6, "equity")
        res = optimize_allocation(basin, 0.6, "weighted", equity_weight=ew)
        assert_feasible(basin, res, 0.6)
        assert 0.0 <= res.objective <= 1.0 + 1e-9
        # the equity result carries a 1e-7 lexicographic slack, hence the 1e-6 relative tolerance
        assert eq.total_benefit_usd * (1 - 1e-6) <= res.total_benefit_usd <= ben.total_benefit_usd * (1 + 1e-9)
        assert ben.min_supply_ratio - 1e-9 <= res.min_supply_ratio <= eq.min_supply_ratio + 1e-6
        assert res.equity_weight == ew

    def test_weighted_score_formula(self, basin):
        ben = optimize_allocation(basin, 0.6, "benefit")
        res = optimize_allocation(basin, 0.6, "weighted", equity_weight=0.3)
        score = 0.7 * res.total_benefit_usd / ben.total_benefit_usd + 0.3 * res.min_supply_ratio
        assert res.objective == pytest.approx(score, abs=1e-6)

    def test_weighted_with_zero_benefit_scale(self):
        basin = single(100.0, mun=50.0, values={MUN: 0.0})
        res = optimize_allocation(basin, objective="weighted", equity_weight=0.5)
        assert res.success
        assert res.objective == pytest.approx(0.5 * res.min_supply_ratio)
        assert res.min_supply_ratio == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# constraints and options
# ---------------------------------------------------------------------------
class TestConstraints:
    def test_min_supply_ratio_enforced(self, basin):
        free = optimize_allocation(basin, 0.3, "benefit")
        assert free.min_supply_ratio < 0.5
        res = optimize_allocation(basin, 0.3, "benefit", min_supply_ratio=0.5)
        assert_feasible(basin, res, 0.3, msr=0.5)
        assert res.min_supply_ratio >= 0.5 - LP_TOL
        assert res.total_benefit_usd <= free.total_benefit_usd * (1 + 1e-9)
        assert res.min_supply_ratio_constraint == 0.5

    def test_min_supply_ratio_too_high_is_infeasible(self, basin):
        res = optimize_allocation(basin, 0.6, "benefit", min_supply_ratio=1.0)
        assert res.status == "infeasible"
        assert not res.success
        assert "infeasible" in res.message
        assert res.allocations == {} and res.outflows == {}
        assert math.isnan(res.objective) and math.isnan(res.min_supply_ratio) and math.isnan(res.gini)

    def test_entitlements_treaty_and_dict(self, basin):
        res = optimize_allocation(basin, 1.0, entitlements="treaty")
        caps = {r.name: r.treaty_allocation_mm3 for r in basin.riparians}
        assert_feasible(basin, res, 1.0, caps=caps)
        assert res.entitlements == caps
        # Delta's treaty cap (16000) binds below its river demand (16300)
        assert res.withdrawals_by_riparian()["Delta"] == pytest.approx(16000.0, rel=1e-7)
        res2 = optimize_allocation(basin, 1.0, entitlements={"Highland": 500.0})
        assert_feasible(basin, res2, 1.0, caps={"Highland": 500.0})
        assert res2.withdrawals_by_riparian()["Highland"] == pytest.approx(500.0, rel=1e-7)
        assert math.isinf(res2.entitlements["Midland"])
        res3 = optimize_allocation(basin, 1.0, entitlements={"Highland": None, "Delta": math.inf})
        assert res3.total_benefit_usd == pytest.approx(optimize_allocation(basin, 1.0).total_benefit_usd, rel=1e-9)

    def test_entitlement_conflicting_with_min_supply_reported(self, basin):
        res = optimize_allocation(basin, 0.6, entitlements={"Delta": 100.0}, min_supply_ratio=0.5)
        assert res.status == "infeasible"
        assert "entitlement" in res.message and "Delta" in res.message

    def test_env_flows_override(self, basin):
        strict = optimize_allocation(basin, 0.6, env_flows={"Delta": 6000.0})
        loose = optimize_allocation(basin, 0.6)
        assert strict.success and loose.success
        assert strict.env_flows["Delta"] == 6000.0
        assert strict.outflows["Delta"] >= 6000.0 - LP_TOL * 6000.0
        assert strict.total_benefit_usd <= loose.total_benefit_usd
        assert optimize_allocation(basin, 0.6, env_flows={"Delta": 1e6}).status == "infeasible"

    def test_demands_override(self, basin):
        res = optimize_allocation(basin, 1.0, demands={"Delta": {MUN: 100.0, "agricultural": 50.0}})
        assert res.success
        assert res.demands["Delta"] == {MUN: 100.0, IND: 0.0, ENE: 0.0, AGR: 50.0}
        # non-river supply (2300) covers the whole 150 -> nothing taken from the river
        assert res.river_demands["Delta"] == {s: 0.0 for s in SECTOR_PRIORITY}
        assert res.allocations["Delta"] == {s: 0.0 for s in SECTOR_PRIORITY}
        assert res.supply_ratios["Delta"] == pytest.approx(1.0)
        res2 = optimize_allocation(basin, 1.0, demands={"Highland": WaterDemand(municipal=1000.0)})
        assert res2.demands["Highland"][MUN] == 1000.0
        assert res2.allocations["Highland"][MUN] == pytest.approx(800.0, rel=1e-7)   # 1000 - 200 groundwater

    def test_soft_environmental_flow_allows_drying_the_river(self):
        basin = single(100.0, env=90.0, ag=200.0, fractions={AGR: 1.0})
        hard = optimize_allocation(basin, env_flow_hard=True)
        soft = optimize_allocation(basin, env_flow_hard=False)
        assert hard.allocations["A"][AGR] == pytest.approx(10.0, abs=1e-6)
        assert hard.outflows["A"] == pytest.approx(90.0, abs=1e-6)
        assert hard.env_flow_met["A"]
        assert soft.allocations["A"][AGR] == pytest.approx(100.0, abs=1e-6)
        assert soft.outflows["A"] == pytest.approx(0.0, abs=1e-6)
        assert not soft.env_flow_met["A"]
        assert not soft.env_flow_hard
        assert soft.total_benefit_usd > hard.total_benefit_usd

    def test_return_flows_reach_downstream(self):
        # A withdraws 100 with f = 0.3 -> 70 returns; B can withdraw 70 of its 100 demand
        basin = Basin(
            "two",
            [
                make_riparian("A", local=100.0, mun=100.0, fractions={MUN: 0.3}),
                make_riparian("B", local=0.0, mun=100.0, fractions={MUN: 0.3}),
            ],
        )
        res = optimize_allocation(basin)
        assert res.allocations["A"][MUN] == pytest.approx(100.0, abs=1e-6)
        assert res.outflows["A"] == pytest.approx(70.0, abs=1e-6)
        assert res.inflows["B"] == pytest.approx(70.0, abs=1e-6)
        assert res.allocations["B"][MUN] == pytest.approx(70.0, abs=1e-6)
        assert res.outflows["B"] == pytest.approx(49.0, abs=1e-6)


# ---------------------------------------------------------------------------
# infeasibility and edge cases
# ---------------------------------------------------------------------------
class TestInfeasibleAndEdges:
    def test_impossible_environmental_flows_give_status_not_exception(self, basin):
        res = optimize_allocation(basin, 0.05)          # natural flow 1400 < Delta env 1500
        assert res.status == "infeasible"
        assert not res.success
        assert "environmental flow" in res.message and "Delta" in res.message
        assert "exceeds the natural inflow" in res.message
        assert res.allocations == {} and res.outflows == {}
        assert math.isnan(res.objective) and math.isnan(res.total_benefit_usd)
        assert math.isnan(res.outflow_to_sea_mm3) and math.isnan(res.total_withdrawal_mm3())
        assert res.natural_flows == pytest.approx(natural_flows(basin, 0.05))
        assert "infeasible" in res.summary()
        json.dumps(res.to_dict())
        for objective in OBJECTIVES:
            assert optimize_allocation(basin, 0.05, objective, equity_weight=0.5).status == "infeasible"

    def test_zero_flow(self, basin):
        assert optimize_allocation(basin, 0.0).status == "infeasible"
        soft = optimize_allocation(basin, 0.0, env_flow_hard=False)
        assert_feasible(basin, soft, 0.0, env_hard=False)
        assert soft.total_benefit_usd == 0.0
        for r in basin.riparians:
            _, nr, total = river_demands(r)
            assert soft.supply_ratios[r.name] == pytest.approx(nr / total)
            assert soft.outflows[r.name] == pytest.approx(0.0, abs=1e-9)
            assert not soft.env_flow_met[r.name]
        assert optimize_allocation(basin, 0.0, "equity", env_flow_hard=False).min_supply_ratio == pytest.approx(
            min(soft.supply_ratios.values())
        )

    def test_zero_demand_basin(self):
        basin = Basin("dry", [make_riparian("A", local=50.0, env=10.0), make_riparian("B", local=20.0)], headwater_inflow_mm3=30.0)
        for objective in OBJECTIVES:
            res = optimize_allocation(basin, objective=objective, equity_weight=0.5)
            assert_feasible(basin, res, 1.0)
            assert res.total_benefit_usd == 0.0
            assert res.min_supply_ratio == 1.0 and res.gini == 0.0
            assert res.outflows == pytest.approx(natural_flows(basin))
            assert all(v == 0.0 for alloc in res.allocations.values() for v in alloc.values())

    def test_empty_basin(self):
        res = optimize_allocation(Basin("empty", []))
        assert res.success and res.allocations == {} and res.total_benefit_usd == 0.0
        assert res.min_supply_ratio == 1.0 and res.gini == 0.0
        assert math.isnan(res.outflow_to_sea_mm3)
        assert pareto_front(Basin("empty", []), points=3)

    def test_riparian_without_reservoir_or_crops(self):
        r = Riparian(name="plain", population=10.0, gdp_usd=1.0, local_inflow_mm3=10.0, demand=WaterDemand(municipal=4.0, environmental=1.0))
        res = optimize_allocation(Basin("plain", [r]))
        assert res.allocations["plain"][MUN] == pytest.approx(4.0)
        assert res.outflows["plain"] == pytest.approx(10.0 - 0.8)
        assert res.supply_ratios["plain"] == 1.0

    def test_non_river_supply_exceeding_demand(self):
        basin = single(10.0, mun=5.0, gw=4.0, desal=3.0)
        res = optimize_allocation(basin)
        assert res.non_river_supply["A"] == 5.0
        assert res.river_demands["A"][MUN] == 0.0
        assert res.allocations["A"][MUN] == 0.0
        assert res.supply_ratios["A"] == 1.0
        assert res.outflows["A"] == pytest.approx(10.0)

    def test_non_river_supply_offsets_sectors_proportionally(self):
        basin = single(1000.0, mun=100.0, ag=300.0, gw=40.0)
        res = optimize_allocation(basin)
        assert res.river_demands["A"][MUN] == pytest.approx(90.0)
        assert res.river_demands["A"][AGR] == pytest.approx(270.0)
        assert res.allocations["A"][MUN] == pytest.approx(90.0)
        assert res.allocations["A"][AGR] == pytest.approx(270.0)
        assert res.supply_ratios["A"] == pytest.approx(1.0)

    def test_randomised_basins_properties(self):
        rng = np.random.default_rng(7)
        checked = 0
        for k in range(25):
            n = int(rng.integers(1, 5))
            riparians = []
            for i in range(n):
                fractions = {s: float(rng.uniform(0.0, 1.0)) for s in SECTOR_PRIORITY}
                riparians.append(
                    make_riparian(
                        f"R{i}",
                        local=float(rng.uniform(0.0, 600.0)),
                        mun=float(rng.uniform(0.0, 200.0)),
                        ind=float(rng.uniform(0.0, 100.0)),
                        ag=float(rng.uniform(0.0, 800.0)),
                        energy=float(rng.uniform(0.0, 50.0)),
                        env=float(rng.uniform(0.0, 150.0)),
                        fractions=fractions,
                        gw=float(rng.uniform(0.0, 100.0)),
                        desal=float(rng.uniform(0.0, 50.0)),
                    )
                )
            basin = Basin(f"random{k}", riparians, headwater_inflow_mm3=float(rng.uniform(0.0, 300.0)))
            ff = float(rng.uniform(0.2, 1.5))
            ben = optimize_allocation(basin, ff, "benefit")
            eq = optimize_allocation(basin, ff, "equity")
            assert ben.status == eq.status
            if not ben.success:
                assert ben.status == "infeasible"
                assert "environmental flow" in ben.message
                continue
            checked += 1
            assert_feasible(basin, ben, ff)
            assert_feasible(basin, eq, ff)
            assert eq.min_supply_ratio >= ben.min_supply_ratio - 1e-6
            assert eq.total_benefit_usd <= ben.total_benefit_usd * (1 + 1e-7) + 1e-3
            bal = route_allocation(basin, ben)
            assert bal.mass_balance_error() < 1e-6
            for reach in bal.reaches:
                assert reach.outflow_mm3 == pytest.approx(ben.outflows[reach.name], rel=1e-6, abs=1e-6)
        assert checked >= 10


# ---------------------------------------------------------------------------
# Pareto front
# ---------------------------------------------------------------------------
class TestPareto:
    def test_front_shape_and_monotonicity(self, basin):
        points = 7
        front = pareto_front(basin, 0.6, points=points)
        assert len(front) == points
        ben = optimize_allocation(basin, 0.6, "benefit")
        eq = optimize_allocation(basin, 0.6, "equity")
        levels = [p["min_supply_ratio"] for p in front]
        assert levels[0] == 0.0
        assert levels[-1] == pytest.approx(eq.min_supply_ratio)
        assert all(b > a for a, b in zip(levels[:-1], levels[1:]))
        benefits = [p["total_benefit_usd"] for p in front]
        for lo, hi in zip(benefits[:-1], benefits[1:]):
            assert hi <= lo * (1.0 + 1e-9)           # benefit non-increasing as min supply ratio rises
        assert benefits[0] == pytest.approx(ben.total_benefit_usd, rel=1e-9)
        assert benefits[-1] == pytest.approx(eq.total_benefit_usd, rel=1e-6)
        assert benefits[-1] < benefits[0]
        for p in front:
            assert set(p) >= {"min_supply_ratio", "total_benefit_usd", "gini"}
            assert all(isinstance(v, float) for v in p.values())
            assert 0.0 <= p["gini"] <= 1.0
            assert p["achieved_min_supply_ratio"] >= p["min_supply_ratio"] - LP_TOL
            assert p["mean_supply_ratio"] >= p["achieved_min_supply_ratio"] - 1e-9
            assert p["outflow_to_sea_mm3"] >= 1500.0 - LP_TOL * 1500.0
        assert front[-1]["gini"] < front[0]["gini"]

    def test_front_binding_range_with_start(self, basin):
        ben = optimize_allocation(basin, 0.6, "benefit")
        front = pareto_front(basin, 0.6, points=5, start=ben.min_supply_ratio)
        assert len(front) == 5
        assert front[0]["min_supply_ratio"] == pytest.approx(ben.min_supply_ratio)
        benefits = [p["total_benefit_usd"] for p in front]
        assert all(hi < lo for lo, hi in zip(benefits[:-1], benefits[1:]))       # strictly decreasing here
        # a start beyond the feasible maximum collapses onto the equity point
        collapsed = pareto_front(basin, 0.6, points=3, start=1.0)
        assert all(p["min_supply_ratio"] == pytest.approx(front[-1]["min_supply_ratio"]) for p in collapsed)

    def test_front_on_infeasible_problem_is_empty(self, basin):
        assert pareto_front(basin, 0.05) == []
        assert pareto_front(basin, 0.0) == []

    def test_front_with_abundant_water_is_flat(self, basin):
        front = pareto_front(basin, 1.0, points=3)
        assert len(front) == 3
        assert all(p["total_benefit_usd"] == pytest.approx(front[0]["total_benefit_usd"], rel=1e-9) for p in front)
        assert front[-1]["min_supply_ratio"] == pytest.approx(1.0)

    def test_front_forwards_options(self, basin):
        soft = pareto_front(basin, 0.2, points=3, env_flow_hard=False)
        assert len(soft) == 3
        capped = pareto_front(basin, 0.6, points=3, entitlements="treaty", values={AGR: 0.5})
        assert len(capped) == 3

    @pytest.mark.parametrize("points", [0, 1, -3, 2.5, True, "5"])
    def test_points_validation(self, basin, points):
        with pytest.raises(ValueError):
            pareto_front(basin, 0.6, points=points)

    def test_start_validation(self, basin):
        with pytest.raises(ValueError):
            pareto_front(basin, 0.6, start=1.5)


# ---------------------------------------------------------------------------
# cross-check with water.route_basin
# ---------------------------------------------------------------------------
class TestCrossCheck:
    def test_outflows_consistent_with_route_basin(self, basin):
        """LP flows at ff=1, msr=0 must equal route_basin with the same withdrawals,
        no reservoirs (storages 0, refill 0) and no treaty caps."""
        res = optimize_allocation(basin, 1.0, "benefit", min_supply_ratio=0.0)
        # non-river supply is already netted out of the LP demands: route a copy without it
        plain = basin.copy()
        for r in plain.riparians:
            r.groundwater_abstraction_mm3 = 0.0
            r.energy.desalination_capacity_mm3 = 0.0
        bal = route_basin(
            plain,
            flow_factor=1.0,
            entitlements={name: None for name in plain.names()},
            demands={name: dict(alloc) for name, alloc in res.allocations.items()},
            storages={name: 0.0 for name in plain.names()},
            reservoir_refill_fraction=0.0,
        )
        assert bal.mass_balance_error() < 1e-6
        for reach in bal.reaches:
            assert reach.outflow_mm3 == pytest.approx(res.outflows[reach.name], rel=1e-9, abs=1e-6)
            assert reach.inflow_mm3 == pytest.approx(res.inflows[reach.name], rel=1e-9, abs=1e-6)
            for s in SECTOR_PRIORITY:
                assert reach.withdrawals[s] == pytest.approx(res.allocations[reach.name][s], rel=1e-9, abs=1e-6)
                assert reach.consumption[s] == pytest.approx(res.consumption[reach.name][s], rel=1e-9, abs=1e-6)
            assert reach.total_deficit() == pytest.approx(0.0, abs=1e-6)
            assert reach.env_flow_met
        assert bal.outflow_to_sea_mm3 == pytest.approx(res.outflow_to_sea_mm3, rel=1e-9)
        assert bal.natural_flow_mm3 == pytest.approx(res.natural_flows[basin.names()[-1]])

    def test_abundant_water_matches_priority_routing(self, basin):
        """With enough water the LP and the simulation serve identical demands."""
        res = optimize_allocation(basin, 1.0)
        bal = route_basin(
            basin,
            flow_factor=1.0,
            entitlements={name: None for name in basin.names()},
            storages={name: 0.0 for name in basin.names()},
            reservoir_refill_fraction=0.0,
        )
        for reach in bal.reaches:
            for s in SECTOR_PRIORITY:
                assert reach.withdrawals[s] == pytest.approx(res.allocations[reach.name][s], rel=1e-7)
            assert reach.outflow_mm3 == pytest.approx(res.outflows[reach.name], rel=1e-7)
            assert reach.supply_ratio() == pytest.approx(res.supply_ratios[reach.name])

    @pytest.mark.parametrize("objective", OBJECTIVES)
    @pytest.mark.parametrize("ff", FLOW_FACTORS)
    def test_route_allocation_reproduces_lp(self, basin, ff, objective):
        res = optimize_allocation(basin, ff, objective, equity_weight=0.5)
        bal = route_allocation(basin, res)
        assert bal.mass_balance_error() < 1e-6
        assert bal.names() == basin.names()
        for reach in bal.reaches:
            assert reach.storage_start_mm3 == 0.0 and reach.storage_refill_mm3 == 0.0
            assert reach.inflow_mm3 == pytest.approx(res.inflows[reach.name], rel=1e-6, abs=1e-6)
            assert reach.outflow_mm3 == pytest.approx(res.outflows[reach.name], rel=1e-6, abs=1e-6)
            for s in SECTOR_PRIORITY:
                assert reach.withdrawals[s] == pytest.approx(res.allocations[reach.name][s], rel=1e-6, abs=1e-6)
            assert reach.non_river_supply_mm3 == pytest.approx(res.non_river_supply[reach.name], abs=1e-6)
            assert reach.total_deficit() == pytest.approx(0.0, abs=1e-6)
            assert reach.env_flow_met == res.env_flow_met[reach.name]
            assert reach.environmental_flow_mm3 == res.env_flows[reach.name]

    def test_route_allocation_with_entitlements_and_soft_env(self, basin):
        res = optimize_allocation(basin, 0.6, entitlements="treaty")
        bal = route_allocation(basin, res)
        for reach in bal.reaches:
            assert reach.entitlement_mm3 == basin.riparian(reach.name).treaty_allocation_mm3
            assert reach.outflow_mm3 == pytest.approx(res.outflows[reach.name], rel=1e-6)
        soft = optimize_allocation(basin, 0.1, env_flow_hard=False)
        bal = route_allocation(basin, soft)
        assert bal.mass_balance_error() < 1e-6
        for reach in bal.reaches:
            assert reach.environmental_flow_mm3 == 0.0
            assert reach.outflow_mm3 == pytest.approx(soft.outflows[reach.name], rel=1e-6, abs=1e-6)

    def test_route_allocation_rejects_bad_inputs(self, basin):
        failed = optimize_allocation(basin, 0.05)
        with pytest.raises(ValueError, match="non-optimal"):
            route_allocation(basin, failed)
        res = optimize_allocation(basin, 1.0)
        other = Basin("other", [make_riparian("X", local=10.0)])
        with pytest.raises(ValueError, match="do not match"):
            route_allocation(other, res)
        with pytest.raises(ValueError):
            route_allocation("basin", res)
        with pytest.raises(ValueError):
            route_allocation(basin, {"status": "optimal"})


# ---------------------------------------------------------------------------
# result helpers
# ---------------------------------------------------------------------------
class TestResultHelpers:
    def test_to_dict_is_json_serialisable(self, basin):
        res = optimize_allocation(basin, 0.6, "equity")
        d = res.to_dict()
        text = json.dumps(d)
        assert "Highland" in text
        assert d["allocations"]["Delta"]["municipal"] == res.allocations["Delta"][MUN]
        assert d["entitlements"] == {name: None for name in basin.names()}
        assert d["status"] == "optimal" and d["objective_name"] == "equity"
        assert d["outflow_to_sea_mm3"] == res.outflow_to_sea_mm3
        assert d["min_supply_ratio"] == res.min_supply_ratio
        capped = optimize_allocation(basin, 0.6, entitlements={"Highland": 500.0}).to_dict()
        assert capped["entitlements"]["Highland"] == 500.0

    def test_summary_text(self, basin):
        res = optimize_allocation(basin, 0.6, "benefit")
        text = res.summary()
        assert "status=optimal" in text
        for name in basin.names():
            assert name in text
        assert text.count("\n") == 1 + len(basin)

    def test_aggregates(self, basin):
        res = optimize_allocation(basin, 0.6)
        by = res.withdrawals_by_riparian()
        assert by == {name: pytest.approx(sum(alloc.values())) for name, alloc in res.allocations.items()}
        assert res.total_withdrawal_mm3() == pytest.approx(sum(by.values()))
        assert res.total_consumption_mm3() == pytest.approx(sum(sum(c.values()) for c in res.consumption.values()))
        assert res.total_consumption_mm3() <= res.total_withdrawal_mm3()
        assert res.mean_supply_ratio() == pytest.approx(np.mean(list(res.supply_ratios.values())))
        assert res.names() == basin.names()
        # water balance of the whole basin: natural flow = consumption + outflow to sea
        natural = res.natural_flows[basin.names()[-1]]
        assert natural == pytest.approx(res.total_consumption_mm3() + res.outflow_to_sea_mm3, rel=1e-9)

    def test_gini_uses_allocation_module(self, basin):
        res = optimize_allocation(basin, 0.3)
        assert res.gini == pytest.approx(gini([res.supply_ratios[n] for n in basin.names()]))
        assert res.gini > 0.0


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
class TestValidation:
    def test_bad_basin(self):
        with pytest.raises(ValueError, match="Basin"):
            optimize_allocation("basin")

    @pytest.mark.parametrize("ff", [-0.1, math.inf, math.nan, "x", None])
    def test_bad_flow_factor(self, basin, ff):
        with pytest.raises(ValueError):
            optimize_allocation(basin, ff)

    @pytest.mark.parametrize("objective", ["profit", "", None, 3])
    def test_bad_objective(self, basin, objective):
        with pytest.raises(ValueError, match="objective"):
            optimize_allocation(basin, objective=objective)

    def test_objective_case_insensitive(self, basin):
        assert optimize_allocation(basin, 0.6, "Equity ").objective_name == "equity"

    @pytest.mark.parametrize("msr", [-0.1, 1.1, math.nan])
    def test_bad_min_supply_ratio(self, basin, msr):
        with pytest.raises(ValueError, match="min_supply_ratio"):
            optimize_allocation(basin, min_supply_ratio=msr)

    @pytest.mark.parametrize("ew", [-0.1, 1.5])
    def test_bad_equity_weight(self, basin, ew):
        with pytest.raises(ValueError, match="equity_weight"):
            optimize_allocation(basin, objective="weighted", equity_weight=ew)

    def test_bad_values(self, basin):
        with pytest.raises(ValueError):
            optimize_allocation(basin, values={"Atlantis": {MUN: 1.0}})
        with pytest.raises(ValueError):
            optimize_allocation(basin, values={MUN: -1.0})
        with pytest.raises(ValueError):
            optimize_allocation(basin, values={Sector.ENVIRONMENT: 1.0})
        with pytest.raises(ValueError):
            optimize_allocation(basin, values={"Highland": {MUN: 1.0}, AGR: 2.0})
        with pytest.raises(ValueError):
            optimize_allocation(basin, values={"Highland": 2.0})
        with pytest.raises(ValueError):
            optimize_allocation(basin, values=[1.0, 2.0])
        assert optimize_allocation(basin, values={}).success

    def test_bad_per_riparian_dicts(self, basin):
        with pytest.raises(ValueError, match="demands"):
            optimize_allocation(basin, demands={"Atlantis": {MUN: 1.0}})
        with pytest.raises(ValueError, match="environment"):
            optimize_allocation(basin, demands={"Delta": {Sector.ENVIRONMENT: 1.0}})
        with pytest.raises(ValueError):
            optimize_allocation(basin, demands={"Delta": {MUN: -1.0}})
        with pytest.raises(ValueError, match="env_flows"):
            optimize_allocation(basin, env_flows={"Atlantis": 1.0})
        with pytest.raises(ValueError):
            optimize_allocation(basin, env_flows={"Delta": -1.0})
        with pytest.raises(ValueError, match="entitlements"):
            optimize_allocation(basin, entitlements={"Atlantis": 1.0})
        with pytest.raises(ValueError, match="entitlements"):
            optimize_allocation(basin, entitlements="agreement")
        with pytest.raises(ValueError):
            optimize_allocation(basin, entitlements={"Delta": -5.0})

    def test_bad_basin_attributes(self, basin):
        # deepcopy: Basin.copy() shares the consumption_fraction / value dicts
        bad = copy.deepcopy(basin)
        bad.riparians[1].demand.consumption_fraction[AGR] = 1.5
        with pytest.raises(ValueError, match="consumption_fraction"):
            optimize_allocation(bad)
        bad = copy.deepcopy(basin)
        bad.riparians[0].local_inflow_mm3 = -1.0
        with pytest.raises(ValueError, match="local_inflow"):
            optimize_allocation(bad)
        bad = copy.deepcopy(basin)
        bad.riparians[2].demand.municipal = -1.0
        with pytest.raises(ValueError):
            optimize_allocation(bad)
        bad = copy.deepcopy(basin)
        bad.riparians[2].demand.value_usd_per_m3[MUN] = -1.0
        with pytest.raises(ValueError, match="value_usd_per_m3"):
            optimize_allocation(bad)
