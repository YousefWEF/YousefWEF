#!/usr/bin/env python3
"""End-to-end demonstration of ``wefnexus`` on the stylised Azura River basin.

Run from the repository root (or from anywhere once the package is installed
with ``pip install -e .``)::

    python examples/run_example.py                   # 25-year horizon
    WEFNEXUS_FAST=1 python examples/run_example.py   # 3-year horizon (quick smoke run; used by the tests)
    python examples/run_example.py --years 10 --outdir /tmp/wef --no-plots

The script walks through the whole toolkit:

1. builds the three-riparian example basin (Highland -> Midland -> Delta)
   and prints its key figures;
2. routes one normal and one drought year through the basin
   (:func:`wefnexus.water.route_basin`) and checks the water mass balance;
3. simulates the ``baseline``, ``climate_change`` and ``combined_adaptation``
   scenarios over the horizon (:func:`wefnexus.scenarios.run_scenarios`),
   prints the comparison table and the sustainability report of every run
   and writes them to CSV;
4. runs the water-diplomacy toolbox of :mod:`wefnexus.diplomacy`:
   hydro-hegemony, a Talmud-rule negotiation with BATNA and ZOPA and a
   comparison of every sharing rule in a normal, a drought and a severe
   drought year (only the last one, flow factor 0.4, rations the
   consumptive claims, so that is the year in which the rules differ), a
   treaty with its institutional resilience and compliance, a Basins-at-Risk
   event record and the conflict risk index in a normal and in a drought
   year, and the Sadoff & Grey benefit-sharing matrix of the severe drought
   year;
5. traces the efficiency-equity Pareto front of the LP allocation
   (:func:`wefnexus.optimize.pareto_front`);
6. writes PNG figures with :mod:`wefnexus.viz` (skipped when matplotlib is
   not installed).

All files go to ``examples/output/`` by default (git-ignored).  The horizon
is 25 years, or 3 years when the environment variable ``WEFNEXUS_FAST`` is
set to ``1`` (``--years`` overrides both).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

try:  # the package is normally installed (pip install -e .); fall back to the checkout
    import wefnexus  # noqa: F401
except ImportError:  # pragma: no cover - only when running from an uninstalled clone
    sys.path.insert(0, str(REPO_ROOT))
    import wefnexus  # noqa: F401

from wefnexus import diplomacy as D  # noqa: E402
from wefnexus import optimize as O  # noqa: E402
from wefnexus import scenarios as S  # noqa: E402
from wefnexus import sustainability as SU  # noqa: E402
from wefnexus.data import example_basin  # noqa: E402
from wefnexus.models import SECTOR_PRIORITY, Basin  # noqa: E402
from wefnexus.water import natural_flows, route_basin  # noqa: E402

#: ``WEFNEXUS_FAST=1`` shortens the horizon to 3 years (and the Pareto front to 4 points).
FAST = os.environ.get("WEFNEXUS_FAST", "").strip().lower() in {"1", "true", "yes", "on"}
DEFAULT_YEARS = 3 if FAST else 25
DEFAULT_PARETO_POINTS = 4 if FAST else 11
DEFAULT_OUTPUT = HERE / "output"
SCENARIO_NAMES = ("baseline", "climate_change", "combined_adaptation")
DROUGHT_FLOW_FACTOR = 0.6
#: Flow factor at which the consumptive estate falls short of the claims and the sharing rules ration.
SEVERE_DROUGHT_FLOW_FACTOR = 0.4
#: The three negotiation / rule-comparison years, label -> flow factor.
NEGOTIATION_YEARS = (("normal", 1.0), ("drought", DROUGHT_FLOW_FACTOR), ("severe drought", SEVERE_DROUGHT_FLOW_FACTOR))


def slug(label: str) -> str:
    """File-name form of a year label (``"severe drought"`` -> ``"severe_drought"``)."""
    return label.replace(" ", "_")

TABLE_COLUMNS = (
    "scenario",
    "riparian",
    "mean_supply_ratio",
    "min_supply_ratio",
    "supply_reliability",
    "mean_nexus_index",
    "mean_water_security",
    "mean_energy_security",
    "mean_food_security",
    "years_env_flow_unmet",
    "total_hydropower_gwh",
    "food_self_sufficiency",
    "final_storage_mm3",
    "equity_index",
)


# --------------------------------------------------------------------------- #
# small printing helpers
# --------------------------------------------------------------------------- #
def section(title: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n{title}\n{bar}")


def fmt(value: Any, nd: int = 3) -> str:
    """Compact number formatting for the console."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, str):
        return value
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isinf(v):
        return "inf"
    if math.isnan(v):
        return "nan"
    if abs(v) >= 1e6:
        return f"{v:,.0f}"
    if abs(v) >= 100:
        return f"{v:,.1f}"
    return f"{v:.{nd}f}"


def table(rows: Sequence[Dict[str, Any]], columns: Optional[Sequence[str]] = None) -> None:
    print(S.format_table(rows, columns))


# --------------------------------------------------------------------------- #
# steps
# --------------------------------------------------------------------------- #
def describe_basin(basin: Basin) -> None:
    section(f"1. Basin: {basin.name}")
    print(f"Mean natural flow {fmt(basin.total_natural_flow())} Mm3/yr "
          f"(headwater {fmt(basin.headwater_inflow_mm3)} + local runoff), "
          f"renewable water incl. groundwater {fmt(basin.total_renewable_water())} Mm3/yr, "
          f"population {basin.total_population() / 1e6:.0f} million.")
    rows = []
    for r in basin.riparians:
        w = r.demand.withdrawals()
        rows.append({
            "riparian": r.name,
            "position": r.position,
            "population_M": r.population / 1e6,
            "gdp_bn_usd": r.gdp_usd / 1e9,
            "local_inflow_mm3": r.local_inflow_mm3,
            "withdrawal_demand_mm3": sum(w.values()),
            "agricultural_mm3": w[SECTOR_PRIORITY[3]],
            "env_flow_mm3": r.demand.environmental,
            "reservoir_capacity_mm3": r.reservoir_capacity_mm3,
            "irrigated_ha": r.irrigated_area_ha(),
            "hydro_mw": r.energy.hydropower_capacity_mw,
            "treaty_mm3": r.treaty_allocation_mm3,
        })
    table(rows)


def route_one_year(basin: Basin) -> Dict[str, Any]:
    section("2. Single-year basin routing (upstream -> downstream)")
    balances = {"normal year (flow factor 1.0)": route_basin(basin),
                f"drought year (flow factor {DROUGHT_FLOW_FACTOR})": route_basin(basin, DROUGHT_FLOW_FACTOR)}
    for label, bal in balances.items():
        print(f"\n{label}: natural flow {fmt(bal.natural_flow_mm3)} Mm3, outflow to sea "
              f"{fmt(bal.outflow_to_sea_mm3)} Mm3, basin supply ratio {bal.supply_ratio():.3f}, "
              f"max mass-balance error {bal.mass_balance_error():.2e} Mm3")
        rows = []
        for reach in bal.reaches:
            rows.append({
                "riparian": reach.name,
                "inflow_mm3": reach.inflow_mm3,
                "withdrawal_mm3": reach.total_withdrawal(),
                "non_river_mm3": reach.non_river_supply_mm3,
                "deficit_mm3": reach.total_deficit(),
                "supply_ratio": reach.supply_ratio(),
                "outflow_mm3": reach.outflow_mm3,
                "env_flow_mm3": reach.environmental_flow_mm3,
                "env_met": reach.env_flow_met,
                "storage_end_mm3": reach.storage_end_mm3,
            })
        table(rows)
    print("\nNatural (zero-use) flow at each reach, Mm3/yr:",
          ", ".join(f"{k}={fmt(v)}" for k, v in natural_flows(basin).items()))
    return balances


def run_scenarios(basin: Basin, years: int, outdir: Path) -> Dict[str, Any]:
    section(f"3. Multi-year scenarios ({years} years)")
    scenarios = [S.get_scenario(name, years=years) for name in SCENARIO_NAMES]
    for sc in scenarios:
        print(f"- {sc.name}: {sc.description or '(no description)'}")
    t0 = time.time()
    results = S.run_scenarios(basin, scenarios)
    print(f"\nSimulated {len(results)} scenarios x {years} years in {time.time() - t0:.2f} s.")
    rows = S.comparison_table(results)
    print("\nScenario comparison (means over the horizon unless stated; indices in 0..1):\n")
    table(rows, TABLE_COLUMNS)
    path = S.write_table_csv(rows, outdir / "scenario_comparison.csv")
    print(f"\nWrote {path}")
    for name, res in results.items():
        path = res.to_csv(outdir / f"records_{name}.csv")
        print(f"Wrote {path}  ({len(res.records)} riparian-year records)")
        worst = max(b.mass_balance_error() for b in res.balances)
        assert worst < 1e-6, f"mass balance violated in {name}: {worst}"
    print("Water mass balance closes in every scenario-year (max error < 1e-6 Mm3).")
    return results


def sustainability_reports(results: Dict[str, Any], outdir: Path) -> None:
    section("4. Sustainability assessment (composite WEF indices and SDG-style indicators)")
    for name, res in results.items():
        report = SU.assess(res)
        print()
        print(report.summary())
        path = S.write_table_csv(report.to_rows(), outdir / f"sustainability_{name}.csv")
        print(f"Wrote {path}")
    base = SU.assess(results["baseline"]).basin
    print(f"\nBasin-wide, baseline: nexus index {base.get('nexus_index', float('nan')):.3f}, "
          f"equity index (1 - Gini of supply ratios) {base.get('equity_index', float('nan')):.3f}.")


def diplomacy(basin: Basin, balances: Dict[str, Any], results: Dict[str, Any], outdir: Path) -> Dict[str, Any]:
    section("5. Water diplomacy")
    out: Dict[str, Any] = {}

    # -- hydro-hegemony ---------------------------------------------------- #
    print("\n5a. Hydro-hegemony (Zeitoun & Warner 2006), pillars on 0..1:")
    rows = []
    for r in basin.riparians:
        h = D.hydro_hegemony(r, basin)
        rows.append({"riparian": r.name, **{k: h[k] for k in ("geographic", "material", "bargaining", "ideational", "score")}})
    table(rows)
    print(f"Power asymmetry (max - min score): {D.power_asymmetry(basin):.3f}")

    # -- negotiation --------------------------------------------------------- #
    print("\n5b. Negotiation framed as a consumptive claims problem (Talmud rule), normal, drought and severe drought year:")
    negotiations = {}
    for label, ff in NEGOTIATION_YEARS:
        neg = D.negotiate(basin, flow_factor=ff, rule="talmud")
        negotiations[label] = neg
        print(f"{label} year (flow factor {ff:g}): estate (natural flow - terminal environmental flow) "
              f"{fmt(neg['estate'])} Mm3/yr; gross claims (river demands) {fmt(sum(neg['claims'].values()))} Mm3/yr, "
              f"consumptive claims {fmt(sum(neg['consumptive_claims'].values()))} Mm3/yr")
        rows = [{
            "riparian": n,
            "claim_mm3": neg["claims"][n],
            "consumptive_claim_mm3": neg["consumptive_claims"][n],
            "consumptive_award_mm3": neg["consumptive_awards"][n],
            "proposal_cap_mm3": neg["proposal"][n],
            "satisfaction": neg["satisfaction"][n],
            "batna_mm3": neg["batna"][n],
            "acceptable": neg["acceptable"][n],
        } for n in basin.names()]
        table(rows)
        print(f"Zone of possible agreement: {'yes' if neg['zopa'] else 'no'}; Gini of the proposal {neg['gini']:.3f}; "
              f"routed outflow to sea {fmt(neg['outflow_to_sea_mm3'])} Mm3/yr")
        if not neg["zopa"]:
            print("  (the rule rations the consumptive estate and the unilateral upstream-priority BATNA - the full\n"
                  "   river demand of an upstream riparian - beats its capped proposal, so a credible agreement\n"
                  "   needs side payments / benefit sharing, or a rule that protects the small claimants such as CEA)")
        print()
    out["negotiation"] = negotiations["normal"]
    out["negotiation_drought"] = negotiations["drought"]
    out["negotiation_severe_drought"] = negotiations["severe drought"]

    # -- all rules ----------------------------------------------------------- #
    print(f"\n5c. Sharing rules compared (gross caps per riparian), normal, drought (flow factor {DROUGHT_FLOW_FACTOR}) "
          f"and severe drought year (flow factor {SEVERE_DROUGHT_FLOW_FACTOR}):\n"
          "    the consumptive claims fit into the estate down to a flow factor of about 0.46, so every rule honours\n"
          "    every claim in the first two years and the rules only differ in the severe drought year")
    comparisons = {}
    for label, ff in NEGOTIATION_YEARS:
        cmp = D.compare_allocation_rules(basin, ff)
        comparisons[label] = cmp
        rows = []
        for rule, row in cmp.items():
            rows.append({
                "year": label,
                "rule": rule,
                **{f"award_{n}": row["awards"][n] for n in basin.names()},
                "consumptive_total_mm3": row["total_consumptive_award_mm3"],
                "estate_mm3": row["estate"],
                "gini": row["gini"],
                "min_satisfaction": row["min_satisfaction"],
                "routed_supply_ratio": row["supply_ratio"],
                "outflow_to_sea_mm3": row["outflow_to_sea_mm3"],
                "env_flow_met_share": row["env_flow_met_share"],
            })
        table(rows)
        print()
        S.write_table_csv(rows, outdir / f"allocation_rules_{slug(label)}.csv")
    out["comparisons"] = comparisons
    print("Wrote " + ", ".join(str(outdir / f"allocation_rules_{slug(label)}.csv") for label, _ in NEGOTIATION_YEARS))

    # -- treaty, events, conflict risk --------------------------------------- #
    print("\n5d. Treaty, event record and conflict risk (Basins at Risk, Wolf et al. 2003):")
    treaty = D.treaty_from_basin(basin, name="Azura River Water Agreement", data_sharing=True,
                                 joint_institution=True, dispute_resolution=True, review_period_years=10,
                                 year_signed=2010)
    assert treaty is not None
    print(f"Treaty '{treaty.name}' between {', '.join(treaty.parties)}; entitlements "
          f"{ {k: fmt(v) for k, v in treaty.allocations_mm3.items()} }; institutional resilience "
          f"{D.treaty_resilience(treaty):.2f} (no drought provisions, no variable allocation, no benefit sharing)")
    normal_bal = list(balances.values())[0]
    drought_bal = list(balances.values())[1]
    comp = D.treaty_compliance(treaty, drought_bal)
    rows = [{"party": p, **{k: v for k, v in c.items() if k in ("entitlement", "withdrawal", "compliance_ratio",
                                                               "delivery_ratio", "delivered")}}
            for p, c in comp.items()]
    print("Compliance in the drought year:")
    table(rows)
    events = [
        D.BarEvent(2013, ["Highland", "Midland"], -2, "infrastructure", "Midland objects to the Highland dam filling plan"),
        D.BarEvent(2016, ["Highland", "Midland", "Delta"], 3, "data", "Joint hydrological monitoring programme"),
        D.BarEvent(2019, ["Midland", "Delta"], -3, "quantity", "Delta recalls its envoy over dry-season deliveries"),
        D.BarEvent(2022, ["Highland", "Midland", "Delta"], 4, "treaty", "Basin commission and drought protocol agreed"),
    ]
    coop = D.cooperation_index(events)
    conf = D.conflict_intensity(events)
    print(f"Event record: {len(events)} events, cooperation index {coop:.2f}, conflict intensity {conf:.2f}, "
          f"TWINS class '{D.twins_classification(coop, conf)}'")
    risks = {}
    for label, bal in (("normal year, treaty", normal_bal), ("drought year, treaty", drought_bal)):
        risks[label] = D.conflict_risk_index(basin, bal, treaty, events)
    risks["drought year, no treaty"] = D.conflict_risk_index(basin, drought_bal, None, events)
    rows = [{"case": label, "score": r["score"], "category": r["category"], **r["components"]} for label, r in risks.items()]
    table(rows)
    out["risks"] = risks
    S.write_table_csv(rows, outdir / "conflict_risk.csv")
    print(f"Wrote {outdir / 'conflict_risk.csv'}")

    # -- benefit sharing ----------------------------------------------------- #
    print(f"\n5e. Benefit sharing (Sadoff & Grey 2002): unilateral use vs the Talmud proposal routed as entitlements, "
          f"severe drought year (flow factor {SEVERE_DROUGHT_FLOW_FACTOR}, the year in which the proposal caps anybody)")
    severe = negotiations["severe drought"]
    unilateral = route_basin(basin, SEVERE_DROUGHT_FLOW_FACTOR, entitlements={n: None for n in basin.names()})
    cooperative = route_basin(basin, SEVERE_DROUGHT_FLOW_FACTOR, entitlements=severe["proposal"])
    matrix = D.benefit_sharing_matrix(basin, unilateral, cooperative, treaty=treaty, events=events)
    rows = [{"riparian": n, "to_the_river": m["to_the_river"], "from_the_river_bn_usd": m["from_the_river"] / 1e9,
             "because_of_the_river": m["because_of_the_river"], "beyond_the_river_bn_usd": m["beyond_the_river"] / 1e9,
             "hydropower_gwh": m["hydropower_gwh"]} for n, m in matrix.items()]
    table(rows)
    out["treaty"] = treaty
    out["events"] = events
    return out


def pareto(basin: Basin, points: int, outdir: Path) -> List[Dict[str, float]]:
    section("6. Efficiency-equity Pareto front of the LP allocation (epsilon-constraint)")
    fronts = {}
    for label, ff in (("normal", 1.0), ("drought", DROUGHT_FLOW_FACTOR)):
        front = O.pareto_front(basin, ff, points=points)
        fronts[label] = front
        print(f"\n{label} year (flow factor {ff}): {len(front)} points")
        if front:
            rows = [{"min_supply_ratio": p["min_supply_ratio"], "benefit_bn_usd": p["total_benefit_usd"] / 1e9,
                     "gini": p["gini"], "achieved_min_ratio": p["achieved_min_supply_ratio"],
                     "mean_supply_ratio": p["mean_supply_ratio"], "outflow_to_sea_mm3": p["outflow_to_sea_mm3"]}
                    for p in front]
            table(rows)
            first, last = front[0], front[-1]
            if first["total_benefit_usd"] > 0:
                drop = (first["total_benefit_usd"] - last["total_benefit_usd"]) / first["total_benefit_usd"] * 100
                print(f"Price of fairness: guaranteeing every riparian a supply ratio of "
                      f"{last['achieved_min_supply_ratio']:.2f} costs {drop:.1f}% of the maximum benefit.")
            S.write_table_csv(rows, outdir / f"pareto_front_{label}.csv")
    return fronts["drought"] or fronts["normal"]


def make_plots(basin: Basin, results: Dict[str, Any], dip: Dict[str, Any], front: List[Dict[str, float]],
               balances: Dict[str, Any], outdir: Path) -> List[Path]:
    section("7. Figures")
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("matplotlib is not installed - skipping the figures (pip install matplotlib networkx).")
        return []
    from wefnexus import viz as V

    written: List[Path] = []

    def save(fig: Any, name: str) -> None:
        path = outdir / name
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        written.append(path)
        try:
            import matplotlib.pyplot as plt

            plt.close(fig)
        except Exception:  # pragma: no cover - defensive
            pass

    downstream = basin.names()[-1]
    for name, res in results.items():
        save(V.plot_supply_ratio(res), f"supply_ratio_{name}.png")
        save(V.plot_nexus_indices(res), f"nexus_indices_{name}.png")
        save(V.plot_water_balance(res, downstream, basin=basin), f"water_balance_{downstream}_{name}.png")
        save(V.plot_nexus_radar(res.summary()), f"nexus_radar_{name}.png")
    for label, cmp in dip["comparisons"].items():
        flow = dict(NEGOTIATION_YEARS)[label]
        save(V.plot_allocation_rules(cmp, title=f"Allocation rules compared - {label} year (flow factor {flow:g})"),
             f"allocation_rules_{slug(label)}.png")
    if front:
        save(V.plot_pareto(front, title=f"Efficiency-equity trade-off - drought year (flow factor {DROUGHT_FLOW_FACTOR})"),
             "pareto_front.png")
    try:
        import networkx  # noqa: F401
    except ImportError:
        print("networkx is not installed - skipping the nexus graph.")
    else:
        save(V.plot_nexus_graph(basin), "nexus_graph.png")
        save(V.plot_nexus_graph(basin, balance=list(balances.values())[1]), "nexus_graph_drought.png")
    for path in written:
        print(f"Wrote {path}")
    return written


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", type=int, default=DEFAULT_YEARS,
                        help=f"simulation horizon in years (default {DEFAULT_YEARS}; 3 when WEFNEXUS_FAST=1)")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT,
                        help=f"output folder for CSV and PNG files (default {DEFAULT_OUTPUT})")
    parser.add_argument("--pareto-points", type=int, default=DEFAULT_PARETO_POINTS,
                        help=f"points on the Pareto front (default {DEFAULT_PARETO_POINTS})")
    parser.add_argument("--no-plots", action="store_true", help="skip the figures")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.years < 1:
        print("--years must be >= 1", file=sys.stderr)
        return 2
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"wefnexus {wefnexus.__version__} - WEF nexus and water diplomacy demonstration")
    print(f"Horizon: {args.years} years" + (" (WEFNEXUS_FAST=1)" if FAST else "") + f"; outputs in {outdir}")

    basin = example_basin()
    describe_basin(basin)
    balances = route_one_year(basin)
    results = run_scenarios(basin, args.years, outdir)
    sustainability_reports(results, outdir)
    dip = diplomacy(basin, balances, results, outdir)
    front = pareto(basin, args.pareto_points, outdir)
    figures = [] if args.no_plots else make_plots(basin, results, dip, front, balances, outdir)

    section("Done")
    csvs = sorted(p.name for p in outdir.glob("*.csv"))
    print(f"{len(csvs)} CSV files and {len(figures)} figures in {outdir} ({time.time() - t0:.1f} s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
