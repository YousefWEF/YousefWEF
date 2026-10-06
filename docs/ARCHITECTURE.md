# wefnexus – Architecture and Module Contracts

This document is the contract for the `wefnexus` package. Every module below is a
plain Python module under `wefnexus/`. All functions must:

* work with plain floats / lists / dicts (NumPy allowed internally, SciPy only in
  `optimize` and the nucleolus solver);
* never mutate the `Basin`, `Riparian` or `Scenario` objects they are given
  (use `basin.copy()` / `dataclasses.replace` if you need a modified version);
* validate inputs and raise `ValueError` with a clear message on bad input;
* carry NumPy-style docstrings with units and literature references;
* ship with pytest tests in `tests/test_<module>.py`.

Units: water in **Mm3/yr** (1 Mm3 = 1e6 m3), energy in **GWh/yr** (helpers may
return kWh and must say so), area in **ha**, depth in **mm**, food in **t** and
**kcal**, money in **USD**. 1 mm of water over 1 ha = 10 m3.

Package layout:

```
wefnexus/
  models.py          DONE  – dataclasses (contract below, do not change signatures)
  data/example_basin.py DONE – stylised 3-riparian "Azura River" basin
  water.py           leaf  – indicators + basin routing
  energy.py          leaf  – water-energy links
  food.py            leaf  – water-food links
  allocation.py      leaf  – sharing rules, cooperative games, equity
  diplomacy.py       composite – hydro-politics
  sustainability.py  composite – composite indices
  nexus.py           composite – integrated multi-year simulation
  optimize.py        composite – LP allocation and Pareto front
  scenarios.py       composite – scenario library and runner
  viz.py             integration – optional matplotlib plots
  cli.py             integration – `python -m wefnexus ...`
  __main__.py        integration
```

---

## 0. `models.py` (already written – read it)

`Sector` (MUNICIPAL, INDUSTRIAL, AGRICULTURAL, ENERGY, ENVIRONMENT),
`WaterDemand`, `Crop`, `EnergySystem`, `Riparian`, `Basin`, `Scenario`,
`SECTOR_PRIORITY`, `DEFAULT_CONSUMPTION_FRACTION`, `DEFAULT_VALUE_USD_PER_M3`.

Key facts:

* `Basin.riparians` is ordered upstream → downstream; `Riparian.position` is its
  index. `Basin.headwater_inflow_mm3` enters above riparian 0.
  `Riparian.local_inflow_mm3` is runoff generated inside each riparian.
* `WaterDemand.environmental` is an in-stream flow that must remain in the river
  at the riparian's outlet; it is **not** a withdrawal.
* `WaterDemand.consumption_fraction[s]` is the share of sector `s` withdrawal that
  is consumed; the rest returns to the river within the same reach.
* `Riparian.treaty_allocation_mm3` is the riparian's treaty entitlement to
  **withdraw surface water** (None = no treaty entitlement).
* `Scenario.flow_factor(i)`, `population_factor(i)`, `demand_factor(i)`,
  `energy_demand_factor(i)`, `irrigated_area_factor(i)`,
  `irrigation_efficiency(base, i)`, `renewable_share(base, i)` give year-`i`
  multipliers/values.

---

## 1. `water.py` (leaf)

### Indicators

```python
FALKENMARK_THRESHOLDS = {"no_stress": 1700.0, "stress": 1000.0, "scarcity": 500.0}

def per_capita_water(renewable_mm3: float, population: float) -> float
    # m3 per person per year.  population <= 0 -> ValueError

def falkenmark_category(per_capita_m3: float) -> str
    # >=1700 "no_stress"; 1000-1700 "stress"; 500-1000 "scarcity"; <500 "absolute_scarcity"

def water_exploitation_index(withdrawal_mm3: float, renewable_mm3: float) -> float
    # WEI = withdrawal / renewable (dimensionless, 0.2 moderate, 0.4 severe)

def sdg_642_water_stress(withdrawal_mm3: float, renewable_mm3: float, environmental_flow_mm3: float) -> float
    # SDG 6.4.2 = 100 * withdrawal / (renewable - EFR); if denominator <= 0 return inf

def dependency_ratio(external_inflow_mm3: float, total_renewable_mm3: float) -> float
    # FAO AQUASTAT: share of renewable water originating outside the territory (0..1)

def groundwater_stress(abstraction_mm3: float, recharge_mm3: float) -> float
    # abstraction / recharge; > 1 = overdraft.  recharge <= 0 -> inf if abstraction > 0 else 0

def supply_reliability(supplied: Sequence[float], demanded: Sequence[float]) -> float
    # share of years with supplied >= 0.999 * demanded  (Hashimoto et al. 1982 reliability)

def resilience(supplied, demanded) -> float
    # Hashimoto resilience: P(recover next year | failure this year); 1.0 if no failures

def vulnerability(supplied, demanded) -> float
    # mean relative deficit over failure years (0 if none)
```

### Basin routing

```python
@dataclass
class ReachResult:
    name: str
    inflow_mm3: float            # from upstream (routed) + local inflow this year
    upstream_inflow_mm3: float   # part of inflow that came from upstream
    storage_start_mm3: float
    storage_release_mm3: float   # water taken from the reservoir this year
    storage_refill_mm3: float
    storage_end_mm3: float
    evaporation_mm3: float
    non_river_supply_mm3: float  # groundwater abstraction + desalination applied to demand
    entitlement_mm3: float       # cap on surface withdrawal applied this year (inf if none)
    withdrawals: Dict[Sector, float]
    consumption: Dict[Sector, float]
    deficits: Dict[Sector, float]        # demand - (withdrawal + non-river share)
    demands: Dict[Sector, float]
    environmental_flow_mm3: float        # requirement
    outflow_mm3: float
    env_flow_met: bool

    def total_withdrawal(self) -> float
    def total_consumption(self) -> float
    def total_deficit(self) -> float
    def supply_ratio(self) -> float   # 1 - total_deficit / total_demand (1.0 if demand 0)

@dataclass
class BasinBalance:
    reaches: List[ReachResult]          # upstream -> downstream
    natural_flow_mm3: float             # headwater + sum(local) after flow factor
    outflow_to_sea_mm3: float
    def reach(self, name) -> ReachResult
    def total_withdrawal(self) -> float
    def total_consumption(self) -> float
    def mass_balance_error(self) -> float   # should be ~0, see below

def route_basin(
    basin: Basin,
    flow_factor: float = 1.0,
    entitlements: Optional[Dict[str, float]] = None,
    demands: Optional[Dict[str, Dict[Sector, float]]] = None,
    storages: Optional[Dict[str, float]] = None,
    env_flows: Optional[Dict[str, float]] = None,
    local_inflow_factors: Optional[Dict[str, float]] = None,
    reservoir_refill_fraction: float = 0.25,
) -> BasinBalance
```

Routing algorithm, riparian by riparian from upstream:

```
upstream = headwater_inflow * flow_factor
for r in basin.riparians:
    local  = r.local_inflow * flow_factor * local_inflow_factors.get(r.name, 1)
    inflow = upstream + local
    storage = storages.get(r.name, r.reservoir_storage_mm3)
    demand = demands.get(r.name) or r.demand.withdrawals()   # Dict[Sector, float]
    env    = env_flows.get(r.name, r.demand.environmental)
    cap    = entitlements.get(r.name, r.treaty_allocation_mm3 or inf)   # surface withdrawal cap

    # non-river supply offsets demand proportionally across sectors
    nonriver = r.groundwater_abstraction_mm3 + r.energy.desalination_capacity_mm3
    nonriver = min(nonriver, sum(demand))
    river_demand[s] = demand[s] - nonriver * demand[s]/sum(demand)   (0 if sum(demand)==0)

    # physical limits.  Water physically available for withdrawal this year:
    supply = inflow + storage
    # Serve sectors in SECTOR_PRIORITY order subject to:
    #   cumulative withdrawal  <= min(cap, supply)
    #   cumulative consumption <= max(supply - env, 0)
    #   w_s = min(river_demand[s], remaining_withdrawal_cap, remaining_consumption_cap / f_s if f_s>0 else inf)
    # consumption[s] = w_s * f_s   (f_s = r.demand.consumption_fraction[s])

    remaining = supply - total_consumption              # includes return flows
    release   = max(0, total_withdrawal - inflow)       # how much came out of storage (bounded by storage)
    storage_after_release = storage - release
    # refill: store a share of what is left above the environmental flow, limited by capacity room
    surplus = max(remaining - env, 0)
    refill  = min(surplus * reservoir_refill_fraction, r.reservoir_capacity_mm3 - storage_after_release)
    refill  = max(refill, 0)
    outflow = remaining - refill  ... but at least env if physically possible:
    outflow = max(outflow, min(env, remaining))
    evaporation = (storage_after_release + refill) * r.reservoir_evaporation_fraction
    storage_end = storage_after_release + refill - evaporation
    env_flow_met = outflow >= env - 1e-9
    upstream = outflow
```

Mass balance per reach: `inflow + storage_start = consumption + outflow + storage_end + evaporation`
(within 1e-6 relative). `BasinBalance.mass_balance_error()` returns the max absolute
violation across reaches and must be ~0 in tests. Note that `release` must be
bounded by `storage` (the serving rules guarantee this since withdrawal ≤ supply).

The function must handle `flow_factor == 0`, zero demands, missing reservoirs,
and must not mutate inputs.

Also provide:

```python
def natural_flows(basin, flow_factor=1.0) -> Dict[str, float]   # inflow at each reach with zero use
def stochastic_flow_factors(scenario: Scenario, basin: Basin) -> List[float]
    # scenario.years lognormal multipliers with mean scenario.flow_factor(i) and CV basin.climate_cv,
    # seeded by scenario.seed (numpy default_rng); deterministic trend when scenario.stochastic is False
```

---

## 2. `energy.py` (leaf)

```python
HYDRO_G = 9.81; WATER_DENSITY = 1000.0
def hydropower_gwh(volume_mm3: float, head_m: float, efficiency: float = 0.9,
                   capacity_mw: Optional[float] = None, hours: float = 8760.0) -> float
    # E = rho g Q H eta ; GWh = (volume_m3 * 9.81 * head * eta) / 3.6e9 ... capped at capacity_mw * hours / 1000 if given
def pumping_energy_gwh(volume_mm3: float, lift_m: float, efficiency: float = 0.6) -> float
    # E_kWh = (rho g V H / eta) / 3.6e6
def desalination_energy_gwh(volume_mm3: float, kwh_per_m3: float = 3.5) -> float
def treatment_energy_gwh(volume_mm3: float, kwh_per_m3: float) -> float
def thermal_cooling_water_mm3(generation_gwh: float, intensity_m3_per_mwh: float) -> float
def thermal_generation_gwh(capacity_mw: float, capacity_factor: float) -> float
def energy_for_water(riparian: Riparian, withdrawals: Dict[Sector, float],
                     desalinated_mm3: float, pumped_fraction: Optional[float] = None) -> Dict[str, float]
    # keys: "pumping", "desalination", "treatment", "wastewater", "total"  (GWh)
    # pumping applies to withdrawals[AGRICULTURAL] * pumped_fraction (default riparian.energy.pumped_fraction)
    # treatment applies to municipal + industrial withdrawals; wastewater to (municipal+industrial)*(1-consumption fraction)
def water_for_energy(riparian: Riparian, thermal_generation_gwh: float) -> float   # Mm3 consumptive cooling
def energy_balance(riparian, hydropower_gwh, thermal_gwh, other_renewable_gwh, demand_gwh) -> Dict[str, float]
    # keys: "supply", "demand", "deficit", "self_sufficiency" (supply/demand capped at 1 when reporting? no – raw ratio), "emissions_t"
    # emissions_t = thermal_gwh * grid_emission_factor_t_per_gwh
def energy_security_index(supply_gwh, demand_gwh) -> float   # min(supply/demand, 1); 1 if demand 0
```

---

## 3. `food.py` (leaf)

```python
def crop_evapotranspiration_mm(et0_mm_day, kc, season_days) -> float        # ETc = ET0 * kc * days
def net_irrigation_mm(etc_mm, effective_rainfall_mm) -> float               # max(ETc - Pe, 0)
def gross_irrigation_mm(net_mm, efficiency) -> float                        # net / efficiency (efficiency in (0,1])
def irrigation_requirement_mm3(crop: Crop, et0_mm_day, effective_rainfall_mm, efficiency) -> float
    # gross_mm * area_ha * 10 / 1e6
def riparian_irrigation_requirement_mm3(riparian: Riparian, area_factor=1.0, efficiency=None) -> float
def fao33_yield(yield_max_t_ha, ky, et_ratio) -> float        # Ya = Ym * (1 - ky*(1 - ETa/ETm)), clipped to [0, Ym]
def crop_production(crop: Crop, water_supplied_mm3, water_required_mm3, area_factor=1.0) -> Dict[str, float]
    # keys: "area_ha","et_ratio","yield_t_ha","production_t","kcal","value_usd"
    # et_ratio = min(supplied/required, 1) (1 if required == 0); supplied is gross irrigation water
def riparian_food_production(riparian, agricultural_water_mm3, area_factor=1.0, efficiency=None) -> Dict[str, Any]
    # distributes available agricultural water across crops proportionally to requirement
    # returns {"crops": {name: crop_production dict}, "production_t", "kcal", "value_usd",
    #          "requirement_mm3", "supplied_mm3", "et_ratio"}
def food_self_sufficiency(kcal_produced, kcal_demand) -> float      # ratio, 1 if demand 0, not capped
def food_security_index(kcal_produced, kcal_demand) -> float         # min(ratio, 1)
def virtual_water_content_m3_per_t(water_mm3, production_t) -> float  # inf if production 0
def virtual_water_import_mm3(kcal_deficit, kcal_per_kg=3400.0, m3_per_t=1500.0) -> float
    # water embedded in food needed to close the deficit
def energy_for_agriculture_gwh(area_ha, kwh_per_ha=150.0) -> float
```

---

## 4. `allocation.py` (leaf)

Bankruptcy / claims rules. All take `estate: float`, `claims: Sequence[float]`
(or `Dict[str, float]` – return the same type as given) and return awards that
sum to `min(estate, sum(claims))`, each in `[0, claim_i]`. Raise ValueError on
negative inputs.

```python
def proportional(estate, claims)
def constrained_equal_awards(estate, claims)        # CEA: min(c_i, lambda)
def constrained_equal_losses(estate, claims)        # CEL: max(c_i - lambda, 0)
def talmud(estate, claims)                          # Aumann–Maschler: CEA on half-claims if E <= C/2 else c/2 + CEL(E - C/2, c/2)
def adjusted_proportional(estate, claims)           # Curiel–Maschler–Tijs: minimal rights then proportional on truncated claims
def equal_split(estate, claims)                     # min(c_i, E/n) with redistribution of unused (== CEA)
def sequential_upstream_priority(estate_by_position: Sequence[float]?, ...) -> NO – instead:
def upstream_priority(available: float, claims: Sequence[float]) -> list    # serve in index order
def apply_rule(rule: str, estate, claims)           # dispatch: "proportional","cea","cel","talmud","ap","equal","upstream_priority"
RULES: Dict[str, Callable]
```

Cooperative game theory:

```python
def shapley_value(players: Sequence[str], v: Callable[[frozenset], float]) -> Dict[str, float]
def is_in_core(allocation: Dict[str, float], players, v, tol=1e-9) -> bool
def core_constraints_violations(allocation, players, v) -> Dict[frozenset, float]   # coalition -> shortfall (>0 means violated)
def nucleolus(players, v) -> Dict[str, float]          # sequential LP via scipy.optimize.linprog; n <= 8
def nash_bargaining(utility_fns: Dict[str, Callable[[float], float]], disagreement: Dict[str, float],
                    total: float, grid: int = 2001) -> Dict[str, float]
    # for 2 players: grid search over x in [0,total] maximizing prod(u_i(x_i) - d_i) subject to u_i >= d_i
    # for n players: scipy.optimize.minimize of -sum log(u_i - d_i) with simplex constraint; fall back to grid for 2
def gini(values: Sequence[float]) -> float
def satisfaction(awards: Dict[str, float], claims: Dict[str, float]) -> Dict[str, float]   # award/claim (1 if claim 0)
def envy_free(awards, claims) -> bool   # nobody with a larger award than claim-satisfaction of another? keep simple:
    # envy-free in satisfaction terms: max satisfaction - min satisfaction <= tol
def compare_rules(estate, claims: Dict[str, float], rules=None) -> Dict[str, Dict[str, float]]   # rule -> awards
```

Tests must check: awards sum to min(E, C), bounds, monotonicity in E, Talmud = CEA for E ≤ C/2 and CEL-like above, consistency with closed-form examples from the literature (Talmud contested garment: claims (100, 200, 300), E=100 → (33.3,33.3,33.3); E=200 → (50,75,75); E=300 → (50,100,150)), Shapley efficiency/symmetry on a known 3-player game, nucleolus in the core for a convex game.

---

## 5. `diplomacy.py` (composite; uses water, allocation)

```python
@dataclass
class Treaty:
    name: str
    parties: List[str]
    allocations_mm3: Dict[str, float]          # fixed volumetric entitlements
    variable_allocation: bool = False          # shares scale with actual flow
    drought_provisions: bool = False
    data_sharing: bool = False
    joint_institution: bool = False
    dispute_resolution: bool = False
    benefit_sharing: bool = False
    review_period_years: Optional[int] = None
    year_signed: Optional[int] = None

@dataclass
class BarEvent:  # Basins at Risk (Wolf, Yoffe & Giordano 2003) event
    year: int
    parties: List[str]
    scale: int           # -7 (formal war) .. +7 (voluntary unification); 0 neutral
    issue: str = ""
    description: str = ""

BAR_SCALE: Dict[int, str]   # labels for -7..7

def hydro_hegemony(riparian: Riparian, basin: Basin, weights=None) -> Dict[str, float]
    # Zeitoun & Warner (2006) four pillars: geographic (upstream position -> higher),
    # material, bargaining, ideational.  geographic = 1 - position/(n-1) (1 for single riparian)
    # returns {"geographic","material","bargaining","ideational","score"} with score = weighted mean

def power_asymmetry(basin) -> float   # max - min hegemony score across riparians

def cooperation_index(events: Sequence[BarEvent], window_years: Optional[int] = None, until: Optional[int] = None) -> float
    # mean BAR scale of events in window (0 if none); also
def conflict_intensity(events, ...) -> float    # mean |scale| of negative events (0 if none)
def twins_classification(cooperation: float, conflict: float) -> str
    # Mirumachi & Allan TWINS matrix; conflict intensity classes: "none","non-politicised","politicised","securitised","violent";
    # cooperation: "confrontation","ad_hoc","technical","risk_averting","risk_taking";  return "<coop>/<conflict>"

def treaty_resilience(treaty: Treaty) -> float
    # 0..1 score: weighted presence of variable_allocation, drought_provisions, data_sharing, joint_institution,
    # dispute_resolution, benefit_sharing, review_period  (De Stefano et al. 2012 institutional resilience)

def treaty_compliance(treaty: Treaty, balance: BasinBalance) -> Dict[str, Dict[str, float]]
    # for each party: entitlement, actual withdrawal, compliance ratio (withdrawal/entitlement, >1 = over-abstraction),
    # and whether downstream parties received at least their entitlement as inflow ("delivery_ratio")

def water_dependency(basin: Basin, balance: Optional[BasinBalance] = None) -> Dict[str, float]
    # per riparian: share of inflow originating upstream (external) = upstream_inflow / inflow

def conflict_risk_index(basin, balance, treaty: Optional[Treaty] = None, events=(), climate_cv=None) -> Dict[str, Any]
    # Basins-at-Risk style composite 0..1 (higher = riskier) from:
    #   water stress (mean SDG 6.4.2 normalised), downstream dependency, power asymmetry,
    #   absence/weakness of treaty (1 - treaty_resilience), recent conflict (negative BAR events),
    #   hydrological variability (climate_cv), unmet environmental flows, dam-filling share
    # returns {"score", "components": {...}, "category": "low"/"moderate"/"high"/"very_high"}

def benefit_sharing_matrix(basin, balance, cooperative_balance=None) -> Dict[str, Dict[str, float]]
    # Sadoff & Grey (2002) four benefit types per riparian:
    # "to_the_river" (environmental flow compliance), "from_the_river" (economic value of water + hydropower),
    # "because_of_the_river" (reduced conflict risk ∝ 1 - conflict_risk), "beyond_the_river" (trade/integration proxy: cooperative gain)

def negotiate(basin, flow_factor=1.0, rule="talmud", claims=None, batna=None) -> Dict[str, Any]
    # 1) claims = treaty allocation or total withdrawal demand per riparian
    # 2) estate = basin natural flow * flow_factor minus sum of environmental flows (what can be consumed/withdrawn)
    # 3) proposal = allocation.apply_rule(rule, estate, claims)
    # 4) batna = unilateral outcome: route_basin with upstream_priority (no entitlements)
    # 5) zopa: proposal acceptable to each riparian if proposal >= batna withdrawal (within tol)
    # returns {"estate","claims","proposal","batna","acceptable": {name: bool}, "zopa": bool, "gini", "satisfaction"}

def compare_allocation_rules(basin, flow_factor=1.0, rules=None) -> Dict[str, Dict[str, Any]]
    # for each rule: awards, gini, min satisfaction, basin outflow after routing with those entitlements
```

---

## 6. `sustainability.py` (composite; uses water, energy, food)

```python
def normalise(value, lo, hi, higher_is_better=True) -> float      # min-max to 0..1, clipped
def weighted_geometric_mean(values: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> float
    # values clipped to [1e-6, 1]
def water_security_index(supply_ratio, stress_sdg642, env_flow_met_share, groundwater_stress) -> float
    # 0..1: geometric mean of supply_ratio, normalise(stress, 100, 0), env share, normalise(gw stress, 1.5, 0.5)
def energy_security_index(supply_ratio, renewable_share, emission_intensity_t_per_gwh) -> float
def food_security_index(self_sufficiency, et_ratio) -> float
def wef_nexus_index(water, energy, food, weights=None) -> float   # geometric mean
def equity_index(values_by_riparian: Dict[str, float]) -> float   # 1 - gini
def sdg_indicators(...)-> Dict[str, float]
    # "6.4.1_water_use_efficiency_usd_per_m3" = gdp / total withdrawal m3
    # "6.4.2_water_stress_pct", "6.5.2_transboundary_cooperation" (treaty present & operational: 0/1 or share), "7.2_renewable_share", "2.1_food_self_sufficiency"
@dataclass
class SustainabilityReport: ... per riparian & basin dicts, with .to_dict(), .summary() text table
def assess(result: "NexusResult") -> SustainabilityReport     # imported lazily to avoid circular import
```

---

## 7. `nexus.py` (composite; uses water, energy, food, allocation, sustainability)

```python
@dataclass
class RiparianYear:
    name: str; year: int
    population: float; gdp_usd: float
    inflow_mm3, upstream_inflow_mm3, outflow_mm3, storage_end_mm3
    entitlement_mm3
    demands: Dict[Sector,float]; withdrawals: Dict[Sector,float]; deficits: Dict[Sector,float]
    environmental_flow_mm3: float; env_flow_met: bool
    per_capita_water_m3: float; falkenmark: str; water_stress_sdg642: float; groundwater_stress: float
    hydropower_gwh, thermal_gwh, other_renewable_gwh, energy_supply_gwh, energy_demand_gwh, energy_for_water_gwh, energy_deficit_gwh, emissions_t
    food_production_t, food_kcal, food_demand_kcal, food_self_sufficiency, et_ratio, irrigation_requirement_mm3, crop_value_usd
    water_value_usd           # sum withdrawals * value_usd_per_m3 * 1e6
    water_security, energy_security, food_security, nexus_index
    def to_dict(self) -> Dict[str, Any]   # flat, sector dicts expanded as "withdrawal_municipal" etc.

@dataclass
class NexusResult:
    basin_name: str; scenario: Scenario
    years: List[int]
    records: List[RiparianYear]              # len = years * n_riparians
    balances: List[BasinBalance]
    def for_riparian(name) -> List[RiparianYear]
    def for_year(year) -> List[RiparianYear]
    def series(name, field) -> List[float]
    def basin_series(field, agg="sum") -> List[float]
    def to_records() -> List[Dict]; def to_dataframe() (pandas, lazy import); def to_csv(path)
    def summary() -> Dict[str, Any]   # means over horizon per riparian: supply ratio, nexus index, etc.

class NexusModel:
    def __init__(self, basin: Basin, scenario: Scenario, allocation_rule: Optional[str]=None,
                 reservoir_refill_fraction: float = 0.25)
    def entitlements(self, year_index, natural_flow) -> Optional[Dict[str, float]]
        # rule from scenario.allocation_rule unless cooperation False -> None (upstream priority / no cap)
        # "treaty": riparian.treaty_allocation_mm3 (None -> no cap for that riparian); scaled by flow_factor if
        #           scenario... keep fixed (treaties are usually fixed volumes)
        # "proportional"/"cea"/"cel"/"talmud"/"ap"/"equal": estate = natural_flow - sum(env flows), claims = demand withdrawals
        # "upstream_priority"/None: no caps
    def run(self) -> NexusResult
        # per year i:
        #   ff = flow factors (stochastic if scenario.stochastic)
        #   grow demands: municipal/industrial/energy * demand_factor; agricultural = crop requirement
        #        (riparian_irrigation_requirement with area_factor, efficiency from scenario) if crops else demand.agricultural * area factor
        #   route_basin(...) with carried storages
        #   energy: hydropower_gwh(outflow*turbined_fraction, head, eff, capacity), thermal = thermal_generation,
        #           other renewables = demand * renewable_share (scenario), energy_for_water
        #   food: riparian_food_production(withdrawals[AGRICULTURAL] + non-river share for agriculture ...)
        #   indicators via sustainability functions
def run_nexus(basin, scenario, **kw) -> NexusResult
```

---

## 8. `optimize.py` (composite; scipy.optimize.linprog)

```python
@dataclass
class OptimizationResult: status, objective, allocations: Dict[str, Dict[Sector,float]], outflows: Dict[str,float],
                           total_benefit_usd, min_supply_ratio, gini, message
def optimize_allocation(basin, flow_factor=1.0, objective="benefit", min_supply_ratio=0.0,
                        equity_weight=0.0, env_flow_hard=True, values=None) -> OptimizationResult
    # Decision vars: withdrawal w[r,s] >= 0 for each riparian & sector.  Outflow of reach r:
    #   out_r = in_r - sum_s f_s w[r,s]; in_r = out_{r-1} + local_r*ff (in_0 adds headwater)
    #   constraints: w[r,s] <= demand[r,s]; sum_s w[r,s] <= in_r (expressed linearly via upstream vars);
    #                out_r >= env_r (hard) ; sum_s w[r,s] >= min_supply_ratio * total_demand_r
    #   objective "benefit": maximise sum w[r,s] * value[r,s] * 1e6 ;  "equity": maximise min over r of supply ratio (epigraph var t);
    #   "weighted": (1-equity_weight)*benefit_normalised + equity_weight*t
def pareto_front(basin, flow_factor=1.0, points=11) -> List[Dict[str, float]]
    # epsilon-constraint on min_supply_ratio from 0 to feasible max; each point: {"min_supply_ratio","total_benefit_usd","gini"}
```

---

## 9. `scenarios.py` (composite)

```python
def baseline(years=25) -> Scenario
def climate_change(years=25, flow_change_pct=-20, droughts=(8, 15, 22)) -> Scenario
def growth(years=25, population_growth=0.015, demand_growth=0.02, energy_growth=0.03, irrigated_area_change_pct=20) -> Scenario
def efficiency(years=25, irrigation_efficiency_target=0.7, renewable_share_target=0.5) -> Scenario
def unilateral(years=25) -> Scenario          # cooperation False
def cooperative(years=25, rule="talmud") -> Scenario
def combined_stress(years=25) -> Scenario     # climate + growth, cooperative treaty
def combined_adaptation(years=25) -> Scenario # climate + growth + efficiency + talmud
SCENARIOS: Dict[str, Callable[..., Scenario]]
def get_scenario(name, **kw) -> Scenario
def run_scenarios(basin, scenarios: Sequence[Scenario], **model_kw) -> Dict[str, NexusResult]
def comparison_table(results: Dict[str, NexusResult]) -> List[Dict[str, Any]]
    # one row per (scenario, riparian) + basin row: mean supply ratio, min supply ratio, mean nexus index,
    # mean water/energy/food security, years with env flow unmet, total hydropower, food self-sufficiency, final storage
def format_table(rows, columns=None) -> str   # plain text table
```

---

## 10. `viz.py`, `cli.py`, `__main__.py` (integration)

* `viz.py`: `plot_supply_ratio(result)`, `plot_nexus_indices(result)`, `plot_allocation_rules(comparison)`,
  `plot_pareto(front)`, `plot_nexus_radar(summary)`, `nexus_graph(basin)` (networkx) – all return the
  matplotlib Figure, import matplotlib lazily, `save=path` optional.
* `cli.py` subcommands: `run` (basin example|json path, scenario name, years, --csv, --plot dir),
  `allocate` (--rule --estate --claims a=1,b=2), `negotiate`, `compare` (scenarios), `pareto`, `report`.
  Basin JSON loader `load_basin(path)` / `basin_to_json(basin)` round-trip live in `wefnexus/io.py`.

---

## Testing & quality bar

* `python -m pytest` must pass; no warnings about mutation; deterministic (seeded).
* Water mass balance error < 1e-6 in every scenario.
* Every public function has a docstring with units and at least one test.
* No module may import `nexus` at import time except `scenarios`, `cli`, `viz`, `optimize`;
  `sustainability.assess` imports `NexusResult` lazily inside the function.

---

## Implementation notes (deviations recorded during final validation)

Every signature above is implemented as written.  Two formulas were refined
during review; the code, its tests (`tests/test_diplomacy.py`,
`tests/test_nexus.py`, `tests/test_cli.py`) and the user documentation
(`README.md`, `docs/METHODOLOGY.md`) follow the refined versions:

* **`diplomacy.negotiate` / `compare_allocation_rules` (section 5)** – the
  estate is `max(natural_flow * flow_factor - env_terminal, 0)`
  (`diplomacy.bankruptcy_estate`): only the in-stream requirement at the
  terminal outlet is withheld, not the sum of every reach's requirement,
  because water left at an upstream outlet is not a withdrawal and remains
  available to the riparians below (example basin at mean flow: 26 500
  instead of 21 000 Mm3/yr).
* **`NexusModel.entitlements` (section 7)** – bankruptcy rules work on a
  consumptive basis: `estate = max(natural_flow + sum(start-of-year storages)
  - sum(environmental flows), 0)`, the claims are the consumptive parts of the
  riparians' river-water demands, and each award is converted into a gross
  surface-withdrawal cap, so a rule only rations under physical scarcity and a
  zero-flow year can still draw down full reservoirs.
