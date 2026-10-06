# wefnexus

**Water-Energy-Food (WEF) nexus modelling, transboundary water diplomacy and
sustainability indicators for shared river basins, in plain Python.**

`wefnexus` simulates a transboundary river basin year by year, upstream to
downstream, and reports what every riparian state gets out of the river in
terms of water, energy and food. On top of that physical balance it offers the
analytical tools of *water diplomacy*: bankruptcy sharing rules and
cooperative-game solutions for dividing scarce water, hydro-hegemony and
Basins-at-Risk style conflict-risk indices, treaty resilience and compliance
checks, Sadoff & Grey benefit sharing, BATNA/ZOPA negotiation framing and an LP
optimiser that traces the efficiency-equity Pareto front. Every indicator is
anchored in a published method (FAO-56, FAO-33, SDG 6.4.2, Falkenmark,
Hashimoto, Aumann-Maschler, Shapley, Zeitoun & Warner, Wolf et al., ...) and
every public function carries a NumPy-style docstring with units and
references.

The package ships with a stylised three-riparian basin, the **Azura River**
(Highland -> Midland -> Delta), used in the tests, the examples and the
command-line interface. All numbers in it are illustrative; it is not data for
any real river.

---

## Contents

1. [Features](#features)
2. [Installation](#installation)
3. [60-second quick start](#60-second-quick-start)
4. [Command-line interface](#command-line-interface)
5. [Module overview](#module-overview)
6. [The Azura River example basin](#the-azura-river-example-basin)
7. [Methodology summary](#methodology-summary)
8. [Limitations and assumptions](#limitations-and-assumptions)
9. [Adding your own basin](#adding-your-own-basin)
10. [Examples, tests and documentation](#examples-tests-and-documentation)
11. [References](#references)

---

## Features

**Water**
- Annual upstream-to-downstream water-balance routing with sector priorities,
  treaty entitlements, environmental-flow constraints, reservoir release /
  refill / evaporation, groundwater and desalination supply, and an exact
  per-reach mass balance (`wefnexus.water.route_basin`).
- Classic scarcity indicators: Falkenmark per-capita classes, water
  exploitation index, SDG 6.4.2 level of water stress, FAO AQUASTAT dependency
  ratio, groundwater stress, and the Hashimoto reliability / resilience /
  vulnerability criteria.
- Deterministic or seeded stochastic (lognormal) annual flow series.

**Energy**
- Hydropower from turbined outflow (`E = rho g V H eta`, capped by installed
  capacity), thermal generation and its consumptive cooling water, and the
  electricity embedded in water supply (pumping, desalination, drinking-water
  and wastewater treatment).
- Per-riparian energy balance, self-sufficiency, emissions and an energy
  security index.

**Food**
- FAO-56 crop evapotranspiration and net / gross irrigation requirements,
  FAO-33 yield response to water deficit, production in tonnes, kilocalories
  and USD, calorie self-sufficiency, virtual water content and virtual water
  imports.

**Allocation and cooperative game theory**
- Bankruptcy (claims) rules: proportional, constrained equal awards (CEA),
  constrained equal losses (CEL), Talmud (Aumann & Maschler), adjusted
  proportional (Curiel-Maschler-Tijs), equal split and upstream priority.
- Shapley value, core membership and excess tests, nucleolus (sequential LPs),
  O'Neill bankruptcy game, Nash bargaining solution, Gini coefficient,
  satisfaction and envy-freeness.

**Water diplomacy**
- Zeitoun & Warner hydro-hegemony pillars and basin power asymmetry.
- Basins at Risk (BAR) event scale, cooperation index, conflict intensity and
  the Mirumachi & Allan TWINS classification.
- Treaty container with De Stefano et al. institutional-resilience scoring,
  compliance and delivery checks against a routed year.
- Composite conflict-risk index (water stress, dependency, power asymmetry,
  institutional weakness, conflict history, variability, environmental
  shortfall, upstream dam filling).
- Sadoff & Grey benefit typology (to / from / because of / beyond the river).
- Negotiation framing: consumptive claims problem (river demands or treaty
  entitlements times the consumption ratio), rule-based proposal handed back
  as gross withdrawal caps, unilateral BATNA, zone of possible agreement
  (ZOPA).

**Sustainability and scenarios**
- Water, energy and food security indices and an overall WEF nexus index
  (geometric aggregation, HDI-style), an equity index (`1 - Gini`), SDG-style
  indicators (6.4.1, 6.4.2, 6.5.2, 7.2, 2.1) and a text/CSV sustainability
  report.
- Integrated multi-year `NexusModel` with drivers for climate, population,
  demand, energy, irrigated area, efficiency and renewables; a scenario library
  (baseline, climate change, growth, efficiency, unilateral, cooperative,
  combined stress, combined adaptation); comparison tables and rankings.
- LP allocation (SciPy HiGHS) maximising economic benefit, max-min equity or a
  weighted mix, and an epsilon-constraint Pareto front ("price of fairness").

**Integration**
- JSON serialisation of basins and scenarios (`wefnexus.io`), a CLI with seven
  subcommands, optional matplotlib plots and a networkx nexus graph
  (`wefnexus.viz`), pandas export of results.
- Pure NumPy core (SciPy only for the LP solvers); Python 3.9+; inputs are
  never mutated; invalid inputs raise `ValueError` with a clear message.

---

## Installation

Requires Python 3.9 or newer. The core depends only on NumPy and SciPy.

```bash
git clone https://github.com/YousefWEF/YousefWEF.git
cd YousefWEF
pip install -e .
```

Optional extras:

```bash
pip install -e ".[viz]"    # matplotlib + networkx: wefnexus.viz plots and the nexus graph
pip install -e ".[data]"   # pandas: NexusResult.to_dataframe(), SustainabilityReport.to_dataframe()
pip install -e ".[dev]"    # pytest + all of the above, for running the test suite
```

Check the installation:

```bash
python -c "import wefnexus; print(wefnexus.__version__)"
python -m wefnexus --version
```

Both `python -m wefnexus ...` and the console script `wefnexus ...` run the CLI.

---

## 60-second quick start

### 1. Build the example basin, run a scenario, print the summary

```python
from wefnexus.data import example_basin
from wefnexus.scenarios import get_scenario
from wefnexus.nexus import run_nexus
from wefnexus.sustainability import assess

basin = example_basin()                               # Highland -> Midland -> Delta
scenario = get_scenario("climate_change", years=10)   # flow -20 % by the last year, drought in year 8
result = run_nexus(basin, scenario)                   # 10 years x 3 riparians = 30 records

print(assess(result).summary())                       # per-riparian and basin indicator table

delta = result.summary()["Delta"]                     # horizon means for one riparian
print(round(delta["supply_ratio"], 3), round(delta["nexus_index"], 3), delta["falkenmark"])
```

Output:

```text
Sustainability assessment: Azura River (stylised) | scenario: climate_change | 2025-2034 (10 years)
riparian  water  energy   food  nexus  supply  reliab  env_met  6.4.2%    2.1    7.2  6.4.1$/m3
-----------------------------------------------------------------------------------------------
Highland  0.969   0.963  0.654  0.848   1.000   1.000    1.000    11.8  0.427  0.907       23.6
Midland   0.811   0.591  0.533  0.634   1.000   1.000    1.000    41.2  0.284  0.551       18.2
Delta     0.325   0.430  0.616  0.425   0.982   0.000    1.000    97.8  0.389  0.357       16.7
-----------------------------------------------------------------------------------------------
BASIN     0.702   0.661  0.601  0.636   0.988   0.000    1.000    50.3  0.365  0.494       17.5
0.982 0.425 absolute_scarcity
```

`result.records` holds one `RiparianYear` per riparian and year (water, energy,
food and index fields), `result.balances` the routed `BasinBalance` of every
year, `result.to_csv(path)` / `result.to_dataframe()` export everything.

### 2. Negotiate an allocation (claims problem, BATNA, ZOPA)

```python
from wefnexus import diplomacy as D

neg = D.negotiate(basin, flow_factor=0.4, rule="talmud")   # a 60 % drought year
for name in basin.names():
    print(f"{name:9s} claim {neg['claims'][name]:6.0f} (consumptive {neg['consumptive_claims'][name]:5.0f})  "
          f"award {neg['consumptive_awards'][name]:5.0f} -> cap {neg['proposal'][name]:6.0f}  "
          f"BATNA {neg['batna'][name]:6.0f}  accepts {neg['acceptable'][name]}")
print("estate", round(neg["estate"]), "Mm3/yr | ZOPA:", neg["zopa"], "| Gini:", round(neg["gini"], 3))
print("flow factor 0.6: ZOPA", D.negotiate(basin, flow_factor=0.6)["zopa"], "| 1.0: ZOPA", D.negotiate(basin)["zopa"])
```

Output:

```text
Highland  claim   1500 (consumptive   550)  award   275 -> cap    750  BATNA   1500  accepts False
Midland   claim   7350 (consumptive  3322)  award  2576 -> cap   5698  BATNA   7350  accepts False
Delta     claim  16300 (consumptive  7596)  award  6850 -> cap  14698  BATNA   7328  accepts True
estate 9700 Mm3/yr | ZOPA: False | Gini: 0.44
flow factor 0.6: ZOPA True | 1.0: ZOPA True
```

The claims are the riparians' river demands (withdrawal demand net of
groundwater and desalination: 1 500 / 7 350 / 16 300 Mm3/yr; pass
`claim_basis="treaty"` to start from the treaty entitlements instead), the
estate is the natural flow of the year minus the environmental flow that must
still reach the sea at the basin outlet (11 200 - 1 500 Mm3; an upstream
in-stream requirement is not a withdrawal and the water it keeps in the river
stays available to the riparians below, see `D.bankruptcy_estate`). Because
return flows are re-used downstream the division is made on a consumptive
basis: each claim is multiplied by the riparian's demand-weighted consumption
fraction (`D.consumption_ratio`: 0.37 / 0.45 / 0.47), the Talmud rule divides
the estate among those consumptive claims (11 468 Mm3/yr in total) and each
award is converted back into a gross withdrawal cap (`award / ratio`). The
BATNA is what each riparian would withdraw unilaterally from the same year's
flow (upstream priority, no caps, no reservoir draw-down;
`include_storage=True` lets it draw on storage). In a normal year and at a
flow factor of 0.6 the estate covers every consumptive claim, every riparian
is capped at its own demand and a zone of possible agreement exists; at 0.4
the rule rations, and the upstream riparians - whose BATNA is their full
demand - reject the proposal while Delta, which can only withdraw the 7 328
Mm3 that still reach it when everybody acts unilaterally, accepts. Pass
`estate=` to negotiate over a different pie, or compare rules with
`D.compare_allocation_rules(basin, 0.4)` (CEA, which protects the small
upstream claimants in full, does find a ZOPA in that year).

### 3. Conflict risk of a drought year

```python
from wefnexus.water import route_basin
from wefnexus import diplomacy as D

treaty = D.treaty_from_basin(basin, data_sharing=True, joint_institution=True, dispute_resolution=True)
events = [D.BarEvent(2019, ["Midland", "Delta"], scale=-3, issue="quantity"),
          D.BarEvent(2022, ["Highland", "Midland", "Delta"], scale=4, issue="treaty")]

drought = route_basin(basin, flow_factor=0.6)              # one routed year
risk = D.conflict_risk_index(basin, drought, treaty, events)
print(round(risk["score"], 3), risk["category"])
print({k: round(v, 3) for k, v in risk["components"].items()})
print("treaty resilience", round(D.treaty_resilience(treaty), 2),
      "| TWINS", D.twins_classification(D.cooperation_index(events), D.conflict_intensity(events)))
```

Output:

```text
0.359 moderate
{'water_stress': 0.616, 'dependency': 0.575, 'power_asymmetry': 0.112, 'institutional': 0.275, 'conflict_history': 0.214, 'variability': 0.4, 'environmental': 0.0, 'dam_filling': 0.215}
treaty resilience 0.45 | TWINS confrontation/politicised
```

Without any treaty the same drought year scores 0.482 (`D.conflict_risk_index(basin, drought)`).

### More in three lines each

```python
from wefnexus import allocation as A, optimize as O, scenarios as S

# Bankruptcy rules on the Talmud's "marriage contract" example (claims 100, 200, 300)
print(A.talmud(200, [100, 200, 300]))                  # [50.0, 75.0, 75.0]
print(A.compare_rules(200, {"a": 100, "b": 200, "c": 300})["cea"])   # 66.666... each (CEA = equal split)

# Cooperative game of the same claims problem: the nucleolus is the Talmud division
v = A.bankruptcy_game(200, {"a": 100, "b": 200, "c": 300})
print(A.shapley_value(["a", "b", "c"], v), A.nucleolus(["a", "b", "c"], v))

# LP allocation and the efficiency-equity Pareto front in a drought year
best = O.optimize_allocation(basin, flow_factor=0.6, objective="benefit")
fair = O.optimize_allocation(basin, flow_factor=0.6, objective="equity")
print(best.total_benefit_usd / 1e9, best.min_supply_ratio, fair.total_benefit_usd / 1e9, fair.min_supply_ratio)
front = O.pareto_front(basin, flow_factor=0.6, points=5)   # list of dicts: min_supply_ratio, total_benefit_usd, gini, ...

# Scenario comparison
results = S.run_scenarios(basin, [S.baseline(years=5), S.unilateral(years=5), S.cooperative(years=5, rule="talmud")])
rows = S.comparison_table(results)
print(S.format_table(rows, ["scenario", "riparian", "mean_supply_ratio", "mean_nexus_index", "equity_index"]))
print(S.rank_scenarios(rows, "mean_nexus_index"))
```

Plots (need `pip install -e ".[viz]"`):

```python
from wefnexus import viz

fig = viz.plot_supply_ratio(result, save="supply_ratio.png")
fig = viz.plot_nexus_indices(result)
fig = viz.plot_water_balance(result, "Delta", basin=basin)
fig = viz.plot_nexus_radar(result.summary())
fig = viz.plot_allocation_rules(D.compare_allocation_rules(basin, 0.4))   # the rationing year
fig = viz.plot_pareto(O.pareto_front(basin, 0.6, points=5))
fig = viz.plot_nexus_graph(basin, balance=drought)       # networkx DiGraph of river, sectors and sources
```

Every `plot_*` function returns the matplotlib `Figure`, never calls
`plt.show()`, and accepts `save=path`.

---

## Command-line interface

```text
python -m wefnexus <command> [options]      # or:  wefnexus <command> [options]
```

| Command        | What it does                                                                                            |
|----------------|---------------------------------------------------------------------------------------------------------|
| `run`          | simulate one scenario and print the per-riparian summary table (`--csv`, `--json` write full results)   |
| `allocate`     | divide an estate among claims with a bankruptcy rule (explicit `--claims` or a basin's river demands / entitlements on a consumptive basis) |
| `negotiate`    | claims-problem proposal versus each riparian's BATNA; is there a ZOPA?                                  |
| `compare`      | run several library scenarios and print the comparison table                                            |
| `pareto`       | benefit-equity Pareto front of the allocation LP                                                        |
| `report`       | sustainability assessment and yearly conflict-risk index of a run                                       |
| `export-basin` | write the example basin (or any basin file) as JSON                                                     |

Every command that needs a basin takes `--basin example` (default, the Azura
River) or `--basin path.json`. Scenario arguments take a library name
(`baseline`, `climate_change`, `growth`, `efficiency`, `unilateral`,
`cooperative`, `combined_stress`, `combined_adaptation`) or a scenario JSON
file. Exit codes: 0 success, 1 no result (e.g. infeasible LP), 2 usage or
input error. `--help` works on every subcommand.

### `run`

```bash
python -m wefnexus run --scenario climate_change --years 10
python -m wefnexus run --scenario growth --years 25 --rule talmud --csv records.csv --json run.json
python -m wefnexus run --basin my_basin.json --scenario my_scenario.json --refill 0.1
```

```text
Basin: Azura River (stylised) | scenario: climate_change | years: 2025-2034 (10 years) | allocation rule: treaty
  Climate change over 10 years: natural flow changes linearly by -20 % by the final year, with drought years (flow -40 %) at year offsets [8]; ...
flow factors: min 0.493 | mean 0.867 | max 1.000
riparian  supply  min_supply  reliability  water  energy   food  nexus  env_met  stress_pct  hydro_GWh  food_ss  storage_Mm3
----------------------------------------------------------------------------------------------------------------------------
Highland   1.000       1.000        1.000  0.969   0.963  0.654  0.848    1.000        11.8      38197    0.427         5700
Midland    1.000       1.000        1.000  0.811   0.591  0.533  0.634    1.000        41.2      10998    0.284         2850
Delta      0.982       0.982        0.000  0.325   0.430  0.616  0.425    1.000        97.8       2854    0.389         6474
BASIN      0.988       0.988        0.000  0.702   0.661  0.601  0.636    1.000        50.3      52049    0.365        15024
natural flow 24,279.1 Mm3/yr | outflow to sea 11,637.6 Mm3/yr | equity index 0.996 | mass balance error 3.64e-12 Mm3
```

Options: `--basin`, `--scenario`, `--years N`, `--rule RULE` (override:
`treaty`, `upstream_priority`, `proportional`, `cea`, `cel`, `talmud`, `ap`,
`equal`), `--refill F` (reservoir refill fraction, default 0.25), `--csv PATH`,
`--json PATH`.

### `allocate`

```bash
python -m wefnexus allocate --rule talmud --estate 200 --claims a=100,b=200,c=300
python -m wefnexus allocate --rule all --estate 200 --claims 100,200,300
python -m wefnexus allocate --basin example --flow-factor 0.6 --rule all --json rules.json
```

```text
Claims problem (explicit claims): estate 200.0 | total claims 600.0 | shortfall 400.0 | claimants 3
rule: talmud
claimant  claim  award  satisfaction
------------------------------------
a           100   50.0         0.500
b           200   75.0         0.375
c           300   75.0         0.250
total awarded 200.0 | gini 0.083 | gini (satisfaction) 0.148 | min satisfaction 0.250
```

With `--rule all` every rule is tabulated side by side; with `--basin` the
gross claims are the riparians' river demands (`--claims-basis demand`,
default) or treaty entitlements (`--claims-basis treaty`), the rules divide
the consumptive estate - natural flow times `--flow-factor` minus the
in-stream requirement at the basin outlet (`--estate` overrides) - among the
consumptive claims (`c-claim` and `c-award` columns for a single rule, a
`c-total` column with `--rule all`), and each rule's awards are converted
into gross withdrawal caps and routed through the basin (a
`routed_withdrawal` column for a single rule, with the outflow to sea and the
environmental-flow share in the footer line; `outflow_to_sea_mm3` and
`env_flow_met_share` columns with `--rule all`; `--include-storage` lets the
routing draw on the reservoirs).

### `negotiate`

```bash
python -m wefnexus negotiate --flow-factor 0.4 --rule talmud
python -m wefnexus negotiate --claims treaty --include-storage --json negotiation.json
```

```text
Negotiation: Azura River (stylised) | flow factor 0.4 | rule talmud | claims demand | storage excluded
natural flow 11,200.0 Mm3 | environmental flows 7,000.0 Mm3 (reserve at the outlet 1,500.0 Mm3) | estate 9,700.0 Mm3 | total claims 25,150.0 Mm3 (consumptive 11,468.1 Mm3)
riparian  claim  c-claim  c-award  proposal  batna  satisfaction  acceptable  routed
------------------------------------------------------------------------------------
Highland   1500      550      275       750   1500         0.500  no             750
Midland    7350     3322     2576      5698   7350         0.775  no            5698
Delta     16300     7596     6850     14698   7328         0.902  yes           8744
total awarded 21,146.2 Mm3 (consumptive 9,700.0 Mm3) | gini 0.440 | gini (satisfaction) 0.123 | outflow to sea 5,681.5 Mm3 | env flow met share 1.00
ZOPA: no - proposal below the BATNA of Highland, Midland
```

`--claims demand` (default) takes the river demands as gross claims,
`--claims treaty` the treaty entitlements; `--include-storage` lets the
unilateral BATNA and the routed proposal draw on the reservoirs. At flow
factors 1.0 and 0.6 the first command reports `ZOPA: yes` (no rationing);
the second one reports `ZOPA: no - proposal below the BATNA of Delta` even at
mean flow, because Delta's 16 000 Mm3 treaty entitlement is below the 16 300
Mm3 it withdraws unilaterally.

### `compare`

```bash
python -m wefnexus compare --scenarios baseline,climate_change,combined_adaptation --years 10
python -m wefnexus compare --years 25 --columns all --csv comparison.csv      # all eight library scenarios
python -m wefnexus compare --scenarios baseline,unilateral --years 10 --rule cea
```

```text
Scenario comparison: Azura River (stylised) | scenarios: baseline, climate_change, combined_adaptation | years: 10
scenario             riparian  allocation_rule  mean_supply_ratio  min_supply_ratio  supply_reliability  mean_nexus_index  ...  equity_index
baseline             Highland  treaty                       1.000             1.000               1.000             0.851  ...             -
baseline             Midland   treaty                       1.000             1.000               1.000             0.643  ...             -
baseline             Delta     treaty                       0.982             0.982               0.000             0.487  ...             -
baseline             BASIN     treaty                       0.988             0.988               0.000             0.660  ...         0.996
climate_change       ...
combined_adaptation  ...
```

### `pareto`

```bash
python -m wefnexus pareto --flow-factor 0.6 --points 5
python -m wefnexus pareto --flow-factor 0.5 --points 11 --csv front.csv
```

```text
Pareto front (epsilon-constraint on the minimum supply ratio): Azura River (stylised) | flow factor 0.6 | 5 of 5 points feasible
epsilon  min_supply  mean_supply   gini  benefit_MUSD  withdrawal_Mm3  outflow_Mm3
----------------------------------------------------------------------------------
  0.000       0.819        0.940  0.043         10953           21778         7355
  0.216       0.819        0.940  0.043         10953           21778         7355
  0.431       0.819        0.940  0.043         10953           21778         7355
  0.647       0.819        0.940  0.043         10953           21778         7355
  0.863       0.863        0.863  0.000         10899           21232         7683
price of fairness: benefit falls from 10,953.3 to 10,898.7 MUSD (0.5%) as the minimum supply ratio rises from 0.819 to 0.863
```

### `report`

```bash
python -m wefnexus report --scenario combined_stress --years 5
python -m wefnexus report --scenario unilateral --years 25 --json report.json
```

```text
Sustainability assessment: Azura River (stylised) | scenario: combined_stress | 2025-2029 (5 years)
riparian  water  energy   food  nexus  supply  reliab  env_met  6.4.2%    2.1    7.2  6.4.1$/m3
-----------------------------------------------------------------------------------------------
Highland  0.970   0.965  0.675  0.858   1.000   1.000    1.000    11.6  0.456  0.911       22.9
Midland   0.813   0.592  0.551  0.642   1.000   1.000    1.000    41.7  0.304  0.565       17.5
Delta     0.286   0.432  0.566  0.388   0.910   0.000    1.000    92.1  0.365  0.371       17.3
-----------------------------------------------------------------------------------------------
BASIN     0.689   0.663  0.597  0.629   0.939   0.000    1.000    48.5  0.358  0.508       17.7

Conflict risk index (Basins at Risk style, 0 = none .. 1 = extreme) | allocation rule: treaty | agreement: Azura River (stylised) treaty
year  score  category  stress  dependency  asymmetry  institutional  conflict  variability  env_short  dam_fill
---------------------------------------------------------------------------------------------------------------
2025  0.347  moderate   0.462       0.584      0.112          0.500     0.000        0.400      0.000     0.161
...
mean score 0.341 (moderate) | worst year 2029: 0.351 (moderate) | final year 2029: 0.351 (moderate)
```

### `export-basin`

```bash
python -m wefnexus export-basin --out azura.json          # write the example basin as JSON
python -m wefnexus export-basin --indent -1                # single-line JSON to standard output
python -m wefnexus export-basin --basin my_basin.json --out normalised.json
```

---

## Module overview

| Module                  | Kind        | Contents                                                                                                                                                              |
|-------------------------|-------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `wefnexus.models`       | data        | `Sector`, `WaterDemand`, `Crop`, `EnergySystem`, `Riparian`, `Basin`, `Scenario`; `SECTOR_PRIORITY`, default consumption fractions and water values                      |
| `wefnexus.data`         | data        | `example_basin()` - the stylised Azura River                                                                                                                           |
| `wefnexus.water`        | leaf        | `per_capita_water`, `falkenmark_category`, `water_exploitation_index`, `sdg_642_water_stress`, `dependency_ratio`, `groundwater_stress`, `supply_reliability`, `resilience`, `vulnerability`, `route_basin` -> `BasinBalance` / `ReachResult`, `natural_flows`, `stochastic_flow_factors` |
| `wefnexus.energy`       | leaf        | `hydropower_gwh`, `pumping_energy_gwh`, `desalination_energy_gwh`, `treatment_energy_gwh`, `thermal_generation_gwh`, `thermal_cooling_water_mm3`, `energy_for_water`, `water_for_energy`, `energy_balance`, `energy_security_index` |
| `wefnexus.food`         | leaf        | `crop_evapotranspiration_mm`, `net_irrigation_mm`, `gross_irrigation_mm`, `irrigation_requirement_mm3`, `riparian_irrigation_requirement_mm3`, `fao33_yield`, `crop_production`, `riparian_food_production`, `food_self_sufficiency`, `food_security_index`, `virtual_water_content_m3_per_t`, `virtual_water_import_mm3`, `energy_for_agriculture_gwh` |
| `wefnexus.allocation`   | leaf        | `proportional`, `constrained_equal_awards`, `constrained_equal_losses`, `talmud`, `adjusted_proportional`, `equal_split`, `upstream_priority`, `apply_rule`, `compare_rules`, `RULES`; `shapley_value`, `nucleolus`, `is_in_core`, `core_constraints_violations`, `bankruptcy_game`, `nash_bargaining`; `gini`, `satisfaction`, `envy_free` |
| `wefnexus.diplomacy`    | composite   | `Treaty`, `BarEvent`, `BAR_SCALE`, `hydro_hegemony`, `power_asymmetry`, `cooperation_index`, `conflict_intensity`, `twins_classification`, `treaty_from_basin`, `treaty_resilience`, `treaty_compliance`, `water_dependency`, `conflict_risk_index`, `benefit_sharing_matrix`, `bankruptcy_estate`, `consumption_ratio`, `river_demand_mm3`, `negotiate`, `compare_allocation_rules` |
| `wefnexus.sustainability` | composite | `normalise`, `weighted_geometric_mean`, `water_security_index`, `energy_security_index`, `food_security_index`, `wef_nexus_index`, `equity_index`, `sdg_indicators`, `SustainabilityReport`, `assess` |
| `wefnexus.nexus`        | composite   | `NexusModel`, `run_nexus`, `NexusResult`, `RiparianYear` - the integrated multi-year simulation                                                                          |
| `wefnexus.optimize`     | composite   | `optimize_allocation` -> `OptimizationResult`, `pareto_front`, `route_allocation` (SciPy `linprog`, HiGHS)                                                              |
| `wefnexus.scenarios`    | composite   | `baseline`, `climate_change`, `growth`, `efficiency`, `unilateral`, `cooperative`, `combined_stress`, `combined_adaptation`, `get_scenario`, `run_scenarios`, `comparison_table`, `scenario_differences`, `rank_scenarios`, `format_table`, `write_table_csv` |
| `wefnexus.io`           | integration | `basin_to_dict` / `basin_from_dict`, `basin_to_json` / `load_basin`, `scenario_to_dict` / `scenario_from_dict`, `scenario_to_json` / `load_scenario`                     |
| `wefnexus.viz`          | integration | `plot_supply_ratio`, `plot_nexus_indices`, `plot_water_balance`, `plot_allocation_rules`, `plot_pareto`, `plot_nexus_radar`, `nexus_graph`, `plot_nexus_graph` (lazy matplotlib / networkx) |
| `wefnexus.cli`          | integration | `main`, `build_parser`, the `cmd_*` handlers; `python -m wefnexus`                                                                                                       |

Leaf modules depend only on `models` and NumPy; composite modules build on the
leaves; `sustainability.assess` imports `nexus` lazily so that the two never
import each other at module level. Dependency direction:
`models -> water/energy/food/allocation -> diplomacy/sustainability -> nexus -> optimize/scenarios -> viz/io/cli`.

**Units everywhere:** water in Mm3/yr (1 Mm3 = 10^6 m3), energy in GWh/yr,
area in ha, depth in mm, food in t and kcal, money in USD, indices
dimensionless in [0, 1] with 1 = best (except stress and risk measures, where
higher = worse). 1 mm over 1 ha = 10 m3.

---

## The Azura River example basin

`wefnexus.data.example_basin()` returns a fresh `Basin` with three riparians
ordered upstream to downstream, a headwater inflow of 2,000 Mm3/yr above the
first riparian, a mean annual natural flow of 28,000 Mm3/yr and a flow
coefficient of variation of 0.20. Total population is 73 million.

| Riparian     | Role                                                                                    | Population | GDP (USD) | Local inflow (Mm3/yr) | Withdrawal demand (Mm3/yr) municipal / industrial / energy / agricultural | Env. flow (Mm3/yr) | Reservoir capacity / initial storage (Mm3) | Irrigated area (ha) and crops                       | Hydropower            | Thermal (MW) | Groundwater recharge / abstraction (Mm3/yr) | Desalination (Mm3/yr) | Treaty entitlement (Mm3/yr) |
|--------------|-----------------------------------------------------------------------------------------|-----------:|----------:|----------------------:|---------------------------------------------------------------------------|-------------------:|-------------------------------------------:|-----------------------------------------------------|-----------------------|-------------:|--------------------------------------------:|----------------------:|----------------------------:|
| **Highland** | mountainous upstream state: most of the runoff, big hydropower, little irrigation        | 8 M        | 40 bn     | 18,000                | 600 / 200 / 100 / 800 (= 1,700)                                            | 3,000              | 6,000 / 3,000                               | 200,000: wheat 150,000, maize 50,000                | 2,000 MW at 120 m head | 300          | 800 / 200                                   | 0                     | 2,500                       |
| **Midland**  | mid-stream: fast-growing cities, thermal plants, expanding irrigation                   | 20 M       | 150 bn    | 7,000                 | 1,500 / 900 / 300 / 5,550 (= 8,250)                                        | 2,500              | 3,000 / 1,500                               | 480,000: wheat 300,000, cotton 100,000, vegetables 80,000 | 600 MW at 40 m         | 3,000        | 1,200 / 900                                 | 0                     | 9,000                       |
| **Delta**    | arid downstream: largest population and irrigation, groundwater overdraft, desalination | 45 M       | 300 bn    | 1,000                 | 3,500 / 1,500 / 600 / 13,000 (= 18,600)                                    | 1,500              | 10,000 / 6,000                              | 850,000: rice 200,000, wheat 400,000, maize 150,000, vegetables 100,000 | 300 MW at 20 m         | 8,000        | 1,500 / 2,000                               | 300                   | 16,000                      |

Hydro-hegemony pillars (material / bargaining / ideational power, each in
[0, 1]) are 0.40 / 0.65 / 0.40 for Highland, 0.70 / 0.60 / 0.60 for Midland
and 0.80 / 0.50 / 0.70 for Delta; with the geographic pillar (1 upstream, 0
downstream) the hegemony scores are 0.61, 0.60 and 0.50, a power asymmetry of
0.11. Total withdrawal demand (28,550 Mm3/yr) slightly exceeds the mean natural
flow, treaty entitlements sum to 27,500 Mm3/yr and environmental flows to
7,000 Mm3/yr, so the basin is comfortable in a normal year and stressed in a
drought: a normal year routes 9,534 Mm3 to the sea with a Delta supply ratio of
0.984; the consumptive river-water claims (11,468 Mm3/yr) fit into the
negotiation estate down to a flow factor of about 0.46, so at 0.6 every
sharing rule still honours every claim, while at 0.4 the rules ration and the
upstream riparians fall below their unilateral BATNA.

Default sector consumption fractions are 0.20 municipal, 0.10 industrial,
0.03 energy, 0.60 agricultural; default water values are 1.50 / 0.80 / 0.40 /
0.10 USD/m3. Crops carry FAO-56 seasonal `kc`, FAO-33 `ky`, maximum yield,
kcal/kg and a farm-gate price; cotton has `kcal_per_kg = 0` and only
contributes value.

---

## Methodology summary

Full details, with every formula, unit, source and implementing function, are
in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md); the module contract is in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

**Water balance routing** (`water.route_basin`, WEAP-style annual balance,
Yates et al. 2005). For each reach, upstream to downstream:

```text
inflow   = upstream_outflow + local_inflow * flow_factor
supply   = inflow + storage_start
nonriver = min(groundwater_abstraction + desalination, total_demand)      # offsets demand pro rata
serve sectors in priority order (municipal > industrial > energy > agricultural) with
    cumulative withdrawal  <= min(entitlement, supply)
    cumulative consumption <= max(supply - env_flow, 0)
    consumption_s = withdrawal_s * consumption_fraction_s
release  = max(withdrawal - inflow, 0)                                     # from storage (bounded by storage)
river    = inflow + release - consumption  (+ storage top-up to env_flow, also counted in release)
refill   = min((river - env_flow)+ * refill_fraction, capacity - (storage - release))
outflow  = river - refill
evaporation = (storage - release + refill) * evaporation_fraction
storage_end = storage - release + refill - evaporation
```

with the exact per-reach identity
`inflow + storage_start = consumption + outflow + storage_end + evaporation`
(`BasinBalance.mass_balance_error()` is ~1e-12 Mm3 in every scenario).

**Water scarcity.** Falkenmark (1989): `per_capita = renewable * 1e6 /
population` (m3/person/yr); >= 1700 no stress, 1000-1700 stress, 500-1000
scarcity, < 500 absolute scarcity. SDG 6.4.2 (FAO 2018): `stress % = 100 *
withdrawal / (renewable - environmental_flow)`. WEI = withdrawal / renewable.
Dependency ratio = external / total renewable (FAO AQUASTAT). Groundwater
stress = abstraction / recharge (Gleeson et al. 2012). Hashimoto et al. (1982):
reliability = share of years with `supplied >= 0.999 demand`, resilience =
P(recovery | failure), vulnerability = mean relative deficit in failure years.

**Energy.** Hydropower `E [GWh] = rho g V H eta / 3.6e12` with `V` in m3 (1 Mm3
through 100 m at eta = 1 gives 0.2725 GWh), capped at `capacity_mw * 8760 /
1000`. Pumping `E [kWh] = rho g V H / (eta_pump 3.6e6)`. Desalination and
treatment at constant kWh/m3 intensities (Plappally & Lienhard 2012); thermal
cooling water at m3/MWh (Macknick et al. 2012). Emissions = thermal GWh x grid
factor (t/GWh).

**Food.** FAO-56 (Allen et al. 1998): `ETc = ET0 * Kc * season_days`;
`IRn = max(ETc - Pe, 0)`; `IRg = IRn / efficiency`; volume `IRg * area_ha * 10 /
1e6` Mm3. FAO-33 (Doorenbos & Kassam 1979): `Ya = Ym * (1 - ky * (1 -
ETa/ETm))` clipped to `[0, Ym]`, with `ETa/ETm` proxied by the relative gross
irrigation supply. Calories = t x 1000 x kcal/kg; self-sufficiency = produced /
demanded kcal (demand = population x kcal/cap/day x 365). Virtual water import
= deficit kcal / (kcal/kg x 1000) x m3/t / 1e6.

**Bankruptcy rules** (estate `E`, claims `c`, `C = sum c`; awards sum to
`min(E, C)`, `0 <= a_i <= c_i`): proportional `a_i = c_i min(E, C)/C`; CEA
`a_i = min(c_i, lambda)`; CEL `a_i = max(c_i - lambda, 0)`; Talmud (Aumann &
Maschler 1985) `CEA(E, c/2)` if `E <= C/2` else `c/2 + CEL(E - C/2, c/2)`;
adjusted proportional (Curiel, Maschler & Tijs 1987): minimal rights `m_i =
max(E - sum_{j != i} c_j, 0)` then proportional on truncated claims; upstream
priority: serve in order. Claims (100, 200, 300) and `E` = 100 / 200 / 300 give
the Talmud's (33.3, 33.3, 33.3) / (50, 75, 75) / (50, 100, 150).

**Cooperative games.** Shapley (1953) `phi_i = sum_S |S|!(n-|S|-1)!/n! [v(S u
i) - v(S)]`; core: `x(S) >= v(S)` for all `S`, `x(N) = v(N)`; nucleolus
(Schmeidler 1969) by sequential LPs (Maschler, Peleg & Shapley 1979) - for
O'Neill's (1982) bankruptcy game `v(S) = max(E - sum_{j not in S} c_j, 0)` it
coincides with the Talmud rule. Nash (1950) bargaining maximises `prod (u_i -
d_i)` on the simplex. Gini `G = 2 sum_i i x_(i) / (n sum x) - (n+1)/n`.

**Hydro-hegemony** (Zeitoun & Warner 2006; Cascao & Zeitoun 2010): four
pillars in [0, 1] - geographic `1 - position/(n-1)`, material, bargaining,
ideational - averaged into a score; power asymmetry = max - min score.

**BAR scale and TWINS.** Wolf, Yoffe & Giordano (2003) event scale -7 (formal
war) ... 0 (neutral) ... +7 (voluntary unification); cooperation index = mean
scale in a window; conflict intensity = mean |scale| of negative events;
Mirumachi & Allan (2007) TWINS cell `"<cooperation class>/<conflict class>"`
(confrontation / ad_hoc / technical / risk_averting / risk_taking x none /
non-politicised / politicised / securitised / violent).

**Treaty resilience** (De Stefano et al. 2012): weighted presence of variable
allocation 0.20, drought provisions 0.20, data sharing 0.15, joint institution
0.15, dispute resolution 0.15, benefit sharing 0.10, review period 0.05.
Compliance: withdrawal / entitlement (> 1 = over-abstraction) and delivery:
inflow / entitlement.

**Conflict risk index** (Basins-at-Risk style): weighted mean of eight
components in [0, 1] - water stress 0.20, downstream dependency 0.15, power
asymmetry 0.10, institutional weakness `1 - coverage (0.5 + 0.5 resilience)`
0.20, conflict history 0.10, variability `min(CV/0.5, 1)` 0.10, environmental
shortfall 0.05, upstream dam filling 0.10; categories low < 0.25 <= moderate <
0.5 <= high < 0.75 <= very_high.

**Sadoff & Grey (2002) benefits.** *To* the river = environmental-flow
compliance; *from* the river = value of withdrawals + hydropower; *because of*
the river = `1 - conflict risk`; *beyond* the river = gain in "from" under a
cooperative allocation.

**Sustainability indices** (geometric means, HDI-style; components clipped to
[1e-6, 1]): water = GM(supply ratio, normalise(SDG 6.4.2, 100 -> 0 %),
environmental-flow compliance, normalise(groundwater stress, 1.5 -> 0.5));
energy = GM(min(supply/demand, 1), renewable share, normalise(t CO2/GWh,
1000 -> 0)); food = GM(min(self-sufficiency, 1), ETa/ETm); nexus = GM(water,
energy, food); equity = 1 - Gini of riparian supply ratios.

**LP allocation** (`optimize`): variables `w[r, s] >= 0`; routing `out_r = in_r
- sum_s f[r,s] w[r,s]`, `in_r = out_{r-1} + local_r`; constraints `w <= river
demand`, `sum_s w[r,s] <= in_r`, `out_r >= env_r`, minimum supply ratio,
entitlements; objectives: benefit `sum w 1e6 value`, max-min supply ratio
(lexicographic), or weighted; Pareto front by epsilon-constraint on the minimum
supply ratio (Haimes et al. 1971).

---

## Limitations and assumptions

- **Annual, lumped time step.** Seasonality, flood peaks and intra-annual
  reservoir operation are not represented; a drought is a scaled-down year.
- **Single main stem.** Riparians form one upstream-to-downstream chain; local
  runoff enters the main stem within each riparian's reach. Tributary networks
  shared between riparians, deltas with several outlets and inter-basin
  transfers are out of scope.
- **Return flows re-enter the same reach** (`consumption_fraction` per
  sector) and are immediately available downstream; no lag, no quality loss.
  Groundwater abstraction and desalination capacity are constant external
  supplies that offset demand pro rata; aquifer storage is not simulated.
- **Reservoirs** follow a simple rule (release what withdrawals need, top up
  the environmental flow, store a fixed share of the surplus, evaporate a fixed
  share of storage). No hedging, flood-control or hydropower-driven operation.
- **Energy** is a per-riparian annual balance without transmission, trade or
  storage; "other renewables" are a share of demand; thermal cooling water is
  reported but the `energy` sector withdrawal is an input, not derived from it.
- **Crops** use one seasonal `Kc` and the linear seasonal FAO-33 response;
  the relative gross irrigation supply proxies `ETa/ETm` (conservative, ignores
  the rain-fed share); available water is shared pro rata across crops.
- **Economic values, consumption fractions, emission factors, hegemony pillars,
  treaty features and BAR events are user inputs**; the defaults are
  illustrative literature magnitudes, and the composite-index weights are
  transparent constants (`CONFLICT_RISK_WEIGHTS`, `TREATY_RESILIENCE_WEIGHTS`,
  ...) meant to be re-weighted for a real study.
- **Negotiation estate** is natural flow minus the in-stream requirement at the
  basin outlet (`diplomacy.bankruptcy_estate`), a consumptive volume divided
  among consumptive claims (river demand x consumption ratio) and handed back
  as gross withdrawal caps; the BATNA is the unilateral withdrawal from the
  same year's flow. Reservoir storage is left out of the pie by default
  (`include_storage=True` lets the BATNA and the routing draw on it). Return
  flows enter through one demand-weighted consumption ratio per riparian
  whereas the physical routing serves sectors in priority order (municipal,
  industrial, energy, agricultural), so a capped riparian may consume less
  than its award - and never more only when, as with the default fractions,
  every sector served before agriculture has a fraction below the riparian's
  ratio; a high-fraction sector served first lets a riparian consume more
  than its award, and the total routed consumption stays within the estate
  only while the requirement at the outlet is met. With demand claims a ZOPA
  can therefore only fail under physical scarcity; with treaty (or explicit)
  claims a riparian whose claim lies below its river demand - Delta on the
  example basin, 16 000 against a 16 300 Mm3 unilateral withdrawal - rejects
  even in a normal year. `estate=` lets you negotiate a different pie. The
  bankruptcy rules inside `NexusModel` use the same consumptive claims but
  add the carried storage to the estate, subtract every reach's
  environmental flow and invert each award sector by sector in priority
  order (see `docs/METHODOLOGY.md`).
- **LP optimisation** ignores reservoirs (annual steady state) and groundwater
  dynamics, and assumes linear benefits (constant USD/m3 per sector).
- **Stochastic flows** are independent lognormal multipliers with a given CV
  (no persistence or trend beyond the scenario's linear change).
- **The Azura River is fictional.** Nothing here is calibrated to a real basin;
  the package is a transparent, testable framework, not a validated model of
  any river.

---

## Adding your own basin

Basins are plain dataclasses, so you can build one in Python ...

```python
from wefnexus.models import Basin, Crop, EnergySystem, Riparian, WaterDemand
from wefnexus.nexus import run_nexus
from wefnexus.scenarios import baseline

upper = Riparian(
    name="Upper", population=2_000_000, gdp_usd=20e9, local_inflow_mm3=6_000.0,
    demand=WaterDemand(municipal=150.0, industrial=50.0, agricultural=400.0, energy=20.0, environmental=1_000.0),
    crops=[Crop("wheat", area_ha=60_000, kc=0.85, season_days=150, yield_max_t_ha=4.0, ky=1.05, kcal_per_kg=3_400, price_usd_t=250)],
    energy=EnergySystem(demand_gwh=3_000.0, hydropower_capacity_mw=500.0, hydropower_head_m=80.0, renewable_share=0.6),
    reservoir_capacity_mm3=1_500.0, reservoir_storage_mm3=800.0,
    et0_mm_day=4.5, effective_rainfall_mm=200.0, treaty_allocation_mm3=700.0,
)
lower = Riparian(
    name="Lower", population=9_000_000, gdp_usd=60e9, local_inflow_mm3=800.0,
    demand=WaterDemand(municipal=700.0, industrial=300.0, agricultural=3_000.0, energy=100.0, environmental=500.0),
    crops=[Crop("rice", area_ha=120_000, kc=1.10, season_days=150, yield_max_t_ha=6.5, ky=1.20, kcal_per_kg=3_600, price_usd_t=350)],
    energy=EnergySystem(demand_gwh=25_000.0, thermal_capacity_mw=2_000.0, renewable_share=0.15, desalination_capacity_mm3=100.0),
    groundwater_recharge_mm3=400.0, groundwater_abstraction_mm3=500.0,
    et0_mm_day=6.0, effective_rainfall_mm=50.0, treaty_allocation_mm3=4_000.0,
)
my_basin = Basin(name="My River", riparians=[upper, lower], headwater_inflow_mm3=500.0, climate_cv=0.25)

result = run_nexus(my_basin, baseline(years=5))
print(result.summary()["BASIN"]["supply_ratio"])
```

... or describe it in JSON and load it with `wefnexus.io`. The quickest way to
get a complete, correctly structured template is to export the example basin
and edit it:

```bash
python -m wefnexus export-basin --out my_basin.json
# edit my_basin.json, then
python -m wefnexus run --basin my_basin.json --scenario climate_change --years 20
```

Only `name`, `riparians` and, per riparian, `name`, `population`, `gdp_usd`,
`local_inflow_mm3` and `demand` are required; every other field takes its
dataclass default. Riparians must be listed upstream to downstream. A minimal
file:

```json
{
  "format": "wefnexus.basin",
  "name": "My River",
  "headwater_inflow_mm3": 500.0,
  "climate_cv": 0.25,
  "riparians": [
    {
      "name": "Upper",
      "population": 2000000,
      "gdp_usd": 20000000000.0,
      "local_inflow_mm3": 6000.0,
      "demand": {"municipal": 150.0, "industrial": 50.0, "agricultural": 400.0, "energy": 20.0, "environmental": 1000.0},
      "crops": [{"name": "wheat", "area_ha": 60000, "kc": 0.85, "season_days": 150,
                 "yield_max_t_ha": 4.0, "ky": 1.05, "kcal_per_kg": 3400, "price_usd_t": 250}],
      "energy": {"demand_gwh": 3000.0, "hydropower_capacity_mw": 500.0, "hydropower_head_m": 80.0, "renewable_share": 0.6},
      "reservoir_capacity_mm3": 1500.0,
      "reservoir_storage_mm3": 800.0,
      "et0_mm_day": 4.5,
      "effective_rainfall_mm": 200.0,
      "treaty_allocation_mm3": 700.0
    },
    {
      "name": "Lower",
      "population": 9000000,
      "gdp_usd": 60000000000.0,
      "local_inflow_mm3": 800.0,
      "demand": {"municipal": 700.0, "industrial": 300.0, "agricultural": 3000.0, "energy": 100.0, "environmental": 500.0},
      "crops": [{"name": "rice", "area_ha": 120000, "kc": 1.10, "season_days": 150,
                 "yield_max_t_ha": 6.5, "ky": 1.20, "kcal_per_kg": 3600, "price_usd_t": 350}],
      "energy": {"demand_gwh": 25000.0, "thermal_capacity_mw": 2000.0, "renewable_share": 0.15, "desalination_capacity_mm3": 100.0},
      "groundwater_recharge_mm3": 400.0,
      "groundwater_abstraction_mm3": 500.0,
      "et0_mm_day": 6.0,
      "effective_rainfall_mm": 50.0,
      "treaty_allocation_mm3": 4000.0
    }
  ]
}
```

```python
from wefnexus import io
from wefnexus.nexus import run_nexus
from wefnexus.scenarios import get_scenario

my_basin = io.load_basin("my_basin.json")          # ValueError on typos, wrong types or NaN
io.basin_to_json(my_basin, "my_basin_normalised.json")   # writes every field, defaults included
print(run_nexus(my_basin, get_scenario("growth", years=5)).summary()["BASIN"]["nexus_index"])

sc = get_scenario("combined_adaptation", years=20, rule="talmud")
io.scenario_to_json(sc, "my_scenario.json")       # reusable with:  python -m wefnexus run --scenario my_scenario.json
```

Conventions of the JSON format (see `wefnexus/io.py`): sector-keyed
dictionaries use the sector values (`"municipal"`, `"industrial"`,
`"agricultural"`, `"energy"`, `"environment"`) as keys; `null` means `None`
(e.g. no treaty entitlement); numbers must be finite; unknown keys raise
`ValueError` so typos are caught; the optional `"format"` /
`"format_version"` tags protect against loading a scenario file as a basin.
`basin_from_dict(basin_to_dict(b)) == b` for any basin.

---

## Examples, tests and documentation

- `examples/run_example.py` - end-to-end walk-through (routing, scenarios,
  sustainability reports, the whole diplomacy toolbox, Pareto front, figures);
  `python examples/run_example.py --years 10 --no-plots`, or
  `WEFNEXUS_FAST=1 python examples/run_example.py` for a 3-year smoke run.
  Outputs go to `examples/output/`.
- `examples/wef_nexus_demo.ipynb` - the same story as a notebook (Colab-ready).
- `docs/METHODOLOGY.md` - every formula, indicator, unit, source and
  implementing function.
- `docs/ARCHITECTURE.md` - the module contract the package was built against.
- `CONTRIBUTING.md` - how to run the tests and the coding conventions.

Run the tests with `pip install -e ".[dev]"` and `python -m pytest` (one test
file per module under `tests/`; the water mass balance is checked to 1e-6 Mm3
in every scenario, bankruptcy rules against the closed-form Talmud examples,
cooperative solutions against known games, and every function for input
validation and non-mutation).

License: MIT.

---

## References

**Nexus framing and modelling**
- Bazilian, M., Rogner, H., Howells, M., Hermann, S., Arent, D., Gielen, D., Steduto, P., Mueller, A., Komor, P., Tol, R. S. J. & Yumkella, K. K. (2011). Considering the energy, water and food nexus: towards an integrated modelling approach. *Energy Policy* 39(12), 7896-7906.
- Bizikova, L., Roy, D., Swanson, D., Venema, H. D. & McCandless, M. (2013). *The Water-Energy-Food Security Nexus: Towards a Practical Planning and Decision-Support Framework for Landscape Investment and Risk Management.* IISD, Winnipeg.
- Daher, B. T. & Mohtar, R. H. (2015). Water-energy-food (WEF) Nexus Tool 2.0: guiding integrative resource planning and decision-making. *Water International* 40(5-6), 748-771.
- Hoff, H. (2011). *Understanding the Nexus.* Background paper for the Bonn 2011 Conference: The Water, Energy and Food Security Nexus. Stockholm Environment Institute, Stockholm.
- Loucks, D. P. & van Beek, E. (2017). *Water Resource Systems Planning and Management: An Introduction to Methods, Models, and Applications.* Springer, Cham.
- Simpson, G. B., Jewitt, G. P. W., Becker, W., Badenhorst, J., Masia, S., Neves, A. R., Rovira, P. & Pascual, V. (2022). The Water-Energy-Food Nexus Index: a tool to support sustainable development. *Sustainability* 14(1), 45.
- Yates, D., Sieber, J., Purkey, D. & Huber-Lee, A. (2005). WEAP21 - a demand-, priority-, and preference-driven water planning model. Part 1: model characteristics. *Water International* 30(4), 487-500.

**Water indicators**
- Falkenmark, M., Lundqvist, J. & Widstrand, C. (1989). Macro-scale water scarcity requires micro-scale approaches: aspects of vulnerability in semi-arid development. *Natural Resources Forum* 13(4), 258-267.
- FAO (2018). *Progress on Level of Water Stress - Global Baseline for SDG Indicator 6.4.2.* FAO/UN-Water, Rome.
- FAO (2018). *Progress on Water-Use Efficiency - Global Baseline for SDG Indicator 6.4.1.* FAO/UN-Water, Rome.
- Gleeson, T., Wada, Y., Bierkens, M. F. P. & van Beek, L. P. H. (2012). Water balance of global aquifers revealed by groundwater footprint. *Nature* 488, 197-200.
- Hashimoto, T., Stedinger, J. R. & Loucks, D. P. (1982). Reliability, resiliency, and vulnerability criteria for water resource system performance evaluation. *Water Resources Research* 18(1), 14-20.
- Raskin, P., Gleick, P., Kirshen, P., Pontius, G. & Strzepek, K. (1997). *Water Futures: Assessment of Long-range Patterns and Problems.* Stockholm Environment Institute, Stockholm.
- Stedinger, J. R. (1980). Fitting log normal distributions to hydrologic data. *Water Resources Research* 16(3), 481-490.
- UN-Water (2018). *Progress on Transboundary Water Cooperation - Global Baseline for SDG Indicator 6.5.2.* UNECE/UNESCO, Paris.

**Energy**
- Gulliver, J. S. & Arndt, R. E. A. (1991). *Hydropower Engineering Handbook.* McGraw-Hill, New York.
- IPCC (2014). *Climate Change 2014: Mitigation of Climate Change.* Annex III: Technology-specific cost and performance parameters. Cambridge University Press.
- Kruyt, B., van Vuuren, D. P., de Vries, H. J. M. & Groenenberg, H. (2009). Indicators for energy security. *Energy Policy* 37(6), 2166-2181.
- Kumar, A., Schei, T., Ahenkorah, A., Caceres Rodriguez, R., Devernay, J.-M., Freitas, M., Hall, D., Killingtveit, A. & Liu, Z. (2011). Hydropower. In: *IPCC Special Report on Renewable Energy Sources and Climate Change Mitigation*, Ch. 5. Cambridge University Press.
- Macknick, J., Newmark, R., Heath, G. & Hallett, K. C. (2012). Operational water consumption and withdrawal factors for electricity generating technologies: a review of existing literature. *Environmental Research Letters* 7, 045802.
- Plappally, A. K. & Lienhard V, J. H. (2012). Energy requirements for water production, treatment, end use, reclamation, and disposal. *Renewable and Sustainable Energy Reviews* 16(7), 4818-4848.
- Voutchkov, N. (2018). Energy use for membrane seawater desalination - current status and trends. *Desalination* 431, 2-14.

**Food**
- Allan, J. A. (1998). Virtual water: a strategic resource. Global solutions to regional deficits. *Ground Water* 36(4), 545-546.
- Allen, R. G., Pereira, L. S., Raes, D. & Smith, M. (1998). *Crop Evapotranspiration: Guidelines for Computing Crop Water Requirements.* FAO Irrigation and Drainage Paper 56. FAO, Rome.
- Brouwer, C., Prins, K. & Heibloem, M. (1989). *Irrigation Water Management: Irrigation Scheduling.* Training Manual 4. FAO, Rome.
- Doorenbos, J. & Kassam, A. H. (1979). *Yield Response to Water.* FAO Irrigation and Drainage Paper 33. FAO, Rome.
- FAO (2001). *Food Balance Sheets: A Handbook.* FAO, Rome.
- Hoekstra, A. Y. & Chapagain, A. K. (2008). *Globalization of Water: Sharing the Planet's Freshwater Resources.* Blackwell, Oxford.
- Mekonnen, M. M. & Hoekstra, A. Y. (2011). The green, blue and grey water footprint of crops and derived crop products. *Hydrology and Earth System Sciences* 15, 1577-1600.
- Steduto, P., Hsiao, T. C., Fereres, E. & Raes, D. (2012). *Crop Yield Response to Water.* FAO Irrigation and Drainage Paper 66. FAO, Rome.

**Allocation, bankruptcy problems and cooperative games**
- Ansink, E. & Weikard, H.-P. (2012). Sequential sharing rules for river sharing problems. *Social Choice and Welfare* 38, 187-210.
- Aumann, R. J. & Maschler, M. (1985). Game theoretic analysis of a bankruptcy problem from the Talmud. *Journal of Economic Theory* 36(2), 195-213.
- Curiel, I. J., Maschler, M. & Tijs, S. H. (1987). Bankruptcy games. *Zeitschrift fur Operations Research* 31, A143-A159.
- Dinar, A., Ratner, A. & Yaron, D. (1992). Evaluating cooperative game theory in water resources. *Theory and Decision* 32, 1-20.
- Gini, C. (1912). *Variabilita e mutabilita.* Cuppini, Bologna.
- Guajardo, M. & Jornsten, K. (2015). Common mistakes in computing the nucleolus. *European Journal of Operational Research* 241(3), 931-935.
- Madani, K. (2010). Game theory and water resources. *Journal of Hydrology* 381(3-4), 225-238.
- Maschler, M., Peleg, B. & Shapley, L. S. (1979). Geometric properties of the kernel, nucleolus, and related solution concepts. *Mathematics of Operations Research* 4(4), 303-338.
- Mianabadi, H., Mostert, E., Zarghami, M. & van de Giesen, N. (2014). A new bankruptcy method for conflict resolution in water resources allocation. *Journal of Environmental Management* 144, 152-159.
- Nash, J. F. (1950). The bargaining problem. *Econometrica* 18(2), 155-162.
- O'Neill, B. (1982). A problem of rights arbitration from the Talmud. *Mathematical Social Sciences* 2(4), 345-371.
- Schmeidler, D. (1969). The nucleolus of a characteristic function game. *SIAM Journal on Applied Mathematics* 17(6), 1163-1170.
- Shapley, L. S. (1953). A value for n-person games. In: Kuhn, H. W. & Tucker, A. W. (eds), *Contributions to the Theory of Games II*, Annals of Mathematics Studies 28, 307-317. Princeton University Press.
- Thomson, W. (2003). Axiomatic and game-theoretic analysis of bankruptcy and taxation problems: a survey. *Mathematical Social Sciences* 45(3), 249-297.
- Young, H. P. (1994). *Equity: In Theory and Practice.* Princeton University Press, Princeton.

**Water diplomacy and hydro-politics**
- Cascao, A. E. & Zeitoun, M. (2010). Power, hegemony and critical hydropolitics. In: Earle, A., Jagerskog, A. & Ojendal, J. (eds), *Transboundary Water Management: Principles and Practice*, 27-42. Earthscan, London.
- De Stefano, L., Duncan, J., Dinar, S., Stahl, K., Strzepek, K. M. & Wolf, A. T. (2012). Climate change and the institutional resilience of international river basins. *Journal of Peace Research* 49(1), 193-209.
- Dinar, S., Katz, D., De Stefano, L. & Blankespoor, B. (2015). Climate change, conflict, and cooperation: global analysis of the effectiveness of international river treaties in addressing water variability. *Political Geography* 45, 55-66.
- Drieschova, A., Giordano, M. & Fischhendler, I. (2008). Governance mechanisms to address flow variability in water treaties. *Global Environmental Change* 18(2), 285-295.
- Fisher, R. & Ury, W. (1981). *Getting to Yes: Negotiating Agreement Without Giving In.* Houghton Mifflin, Boston.
- Mirumachi, N. & Allan, J. A. (2007). Revisiting transboundary water governance: power, conflict, cooperation and the political economy. *Proceedings of the International Conference on Adaptive and Integrated Water Management (CAIWA)*, Basel.
- Mirumachi, N. (2015). *Transboundary Water Politics in the Developing World.* Routledge, London.
- Sadoff, C. W. & Grey, D. (2002). Beyond the river: the benefits of cooperation on international rivers. *Water Policy* 4(5), 389-403.
- Sadoff, C. W. & Grey, D. (2005). Cooperation on international rivers: a continuum for securing and sharing benefits. *Water International* 30(4), 420-427.
- Warner, J. (2004). Plugging the GAP - working with Buzan: the Ilisu Dam as a security issue. *SOAS Water Issues Study Group Occasional Paper* 67. SOAS/King's College, London.
- Wolf, A. T., Yoffe, S. B. & Giordano, M. (2003). International waters: identifying basins at risk. *Water Policy* 5(1), 29-60.
- Yoffe, S., Wolf, A. T. & Giordano, M. (2003). Conflict and cooperation over international freshwater resources: indicators of basins at risk. *Journal of the American Water Resources Association* 39(5), 1109-1126.
- Zeitoun, M. & Warner, J. (2006). Hydro-hegemony - a framework for analysis of trans-boundary water conflicts. *Water Policy* 8(5), 435-460.

**Composite indicators and optimisation**
- Bertsimas, D., Farias, V. F. & Trichakis, N. (2011). The price of fairness. *Operations Research* 59(1), 17-31.
- Cohon, J. L. (1978). *Multiobjective Programming and Planning.* Academic Press, New York.
- Cullis, J. & van Koppen, B. (2007). *Applying the Gini Coefficient to Measure Inequality of Water Use in the Olifants River Water Management Area, South Africa.* IWMI Research Report 113. International Water Management Institute, Colombo.
- Ebert, U. & Welsch, H. (2004). Meaningful environmental indices: a social choice approach. *Journal of Environmental Economics and Management* 47(2), 270-283.
- Haimes, Y. Y., Lasdon, L. S. & Wismer, D. A. (1971). On a bicriterion formulation of the problems of integrated system identification and system optimization. *IEEE Transactions on Systems, Man, and Cybernetics* 1(3), 296-297.
- Harou, J. J., Pulido-Velazquez, M., Rosenberg, D. E., Medellin-Azuara, J., Lund, J. R. & Howitt, R. E. (2009). Hydro-economic models: concepts, design, applications, and future prospects. *Journal of Hydrology* 375(3-4), 627-643.
- Nardo, M., Saisana, M., Saltelli, A., Tarantola, S., Hoffman, A. & Giovannini, E. (2005). *Handbook on Constructing Composite Indicators: Methodology and User Guide.* OECD Statistics Working Paper 2005/3. OECD, Paris.
- Sullivan, C. (2002). Calculating a water poverty index. *World Development* 30(7), 1195-1210.
- UNDP (2010). *Human Development Report 2010: The Real Wealth of Nations - Pathways to Human Development.* Technical note 1. UNDP, New York.
