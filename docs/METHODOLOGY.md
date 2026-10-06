# wefnexus - Methodology

This document lists every formula and indicator implemented in `wefnexus`,
with its units, the literature it comes from and the function that implements
it. It follows the package layout: data model, water, energy, food,
allocation, diplomacy, sustainability, the integrated simulation, the
optimiser, the scenario library and the integration modules. The module
contract (signatures and test expectations) is in `ARCHITECTURE.md`; the
user-facing overview is in the top-level `README.md`.

Notation: `sum_s` sums over the withdrawal sectors (municipal, industrial,
energy, agricultural), `r` indexes riparians upstream to downstream,
`(x)+ = max(x, 0)`, `GM` is a geometric mean, `clip(x, a, b)` clips to
`[a, b]`.

---

## 0. Conventions and units

| Quantity                 | Unit                      | Notes                                                                 |
|--------------------------|---------------------------|-----------------------------------------------------------------------|
| Water volume / flow      | Mm3/yr (1 Mm3 = 10^6 m3)  | storage in Mm3                                                         |
| Water depth              | mm                        | 1 mm over 1 ha = 10 m3 (`food.M3_PER_HA_MM`)                           |
| Area                     | ha                        |                                                                       |
| Energy                   | GWh/yr                    | helpers that return kWh say so; 1 GWh = 10^6 kWh = 3.6 x 10^12 J       |
| Power                    | MW                        |                                                                       |
| Head / lift              | m                         |                                                                       |
| Food                     | t (metric tonnes), kcal   | yield in t/ha                                                          |
| Money                    | USD                       | water values in USD/m3, benefits in USD/yr                             |
| Emissions                | t CO2/yr                  | emission factors in t CO2/GWh                                          |
| Per-capita water         | m3/person/yr              |                                                                       |
| SDG 6.4.2 water stress   | percent                   |                                                                       |
| Indices                  | dimensionless, [0, 1]     | 1 = best for security / equity indices; higher = worse for stress and risk |

Every public function validates its inputs and raises `ValueError` on
negative, NaN, non-numeric or out-of-range values; none mutates the `Basin`,
`Riparian`, `Scenario` or result objects it receives.

---

## 1. Data model (`wefnexus.models`)

| Object        | Role                                                                                                     |
|---------------|----------------------------------------------------------------------------------------------------------|
| `Sector`      | `MUNICIPAL`, `INDUSTRIAL`, `AGRICULTURAL`, `ENERGY` (withdrawals) and `ENVIRONMENT` (in-stream requirement) |
| `WaterDemand` | per-sector withdrawal demands (Mm3/yr), `environmental` in-stream flow, `consumption_fraction[s]`, `value_usd_per_m3[s]` |
| `Crop`        | `area_ha`, FAO-56 seasonal `kc`, `season_days`, `yield_max_t_ha`, FAO-33 `ky`, `kcal_per_kg`, `price_usd_t` |
| `EnergySystem`| electricity demand, hydropower capacity / head / efficiency / turbined fraction, thermal capacity / capacity factor / water intensity, renewable share, pumping lift / efficiency / pumped fraction, desalination capacity and kWh/m3, treatment and wastewater kWh/m3, grid emission factor |
| `Riparian`    | population, GDP, local inflow, demand, groundwater recharge / abstraction, crops, energy, reservoir capacity / storage / evaporation fraction, irrigation efficiency, ET0, effective rainfall, dietary demand, hegemony pillars, treaty entitlement, position |
| `Basin`       | ordered riparians (upstream -> downstream), headwater inflow, climate CV                                   |
| `Scenario`    | horizon and drivers (below)                                                                               |

Package constants (illustrative literature magnitudes):

- `SECTOR_PRIORITY = [MUNICIPAL, INDUSTRIAL, ENERGY, AGRICULTURAL]` - service
  order under shortage (WEAP-style demand priorities, Yates et al. 2005).
- `DEFAULT_CONSUMPTION_FRACTION = {municipal 0.20, industrial 0.10, energy
  0.03, agricultural 0.60}` - share of a withdrawal that is consumed rather
  than returned to the river.
- `DEFAULT_VALUE_USD_PER_M3 = {municipal 1.50, industrial 0.80, energy 0.40,
  agricultural 0.10}`.

### Scenario driver formulas (`Scenario` methods)

With `i` the 0-based year index and `progress(i) = i / (years - 1)` (1 when
`years <= 1`):

| Driver                                | Formula                                                                                  | Method                        |
|---------------------------------------|------------------------------------------------------------------------------------------|-------------------------------|
| natural-flow multiplier               | `max((1 + flow_change_pct_by_end/100 * progress(i)) * (1 - drought_severity if i in drought_years else 1), 0)` | `flow_factor(i)`              |
| population                            | `(1 + population_growth_rate)^i`                                                          | `population_factor(i)`        |
| GDP                                   | `(1 + gdp_growth_rate)^i`                                                                 | `gdp_factor(i)`               |
| municipal / industrial / energy-sector water demand | `(1 + demand_growth_rate)^i`                                                | `demand_factor(i)`            |
| electricity demand                    | `(1 + energy_demand_growth_rate)^i`                                                       | `energy_demand_factor(i)`     |
| irrigated area                        | `1 + irrigated_area_change_pct_by_end/100 * progress(i)`                                  | `irrigated_area_factor(i)`    |
| irrigation efficiency                 | `base + (irrigation_efficiency_target - base) * progress(i)` (unchanged if target is None) | `irrigation_efficiency(base, i)` |
| renewable electricity share           | `base + (renewable_share_target - base) * progress(i)` (unchanged if target is None)      | `renewable_share(base, i)`    |

---

## 2. Water (`wefnexus.water`)

### 2.1 Scarcity indicators

| Indicator | Formula | Units | Source | Function |
|-----------|---------|-------|--------|----------|
| Per-capita renewable water | `renewable_mm3 * 1e6 / population` | m3/person/yr | Falkenmark, Lundqvist & Widstrand (1989); FAO AQUASTAT | `per_capita_water(renewable_mm3, population)` |
| Falkenmark class | `>= 1700` no_stress; `[1000, 1700)` stress; `[500, 1000)` scarcity; `< 500` absolute_scarcity (`FALKENMARK_THRESHOLDS`; boundaries belong to the less-stressed class) | class | Falkenmark et al. (1989) | `falkenmark_category(per_capita_m3)` |
| Water exploitation index | `WEI = withdrawal / renewable`; `inf` if renewable = 0 and withdrawal > 0, `0` if both are 0; 0.2 moderate, 0.4 severe stress | - | Raskin et al. (1997); EEA CSI 018 | `water_exploitation_index(withdrawal_mm3, renewable_mm3)` |
| SDG 6.4.2 level of water stress | `100 * withdrawal / (renewable - EFR)`; `inf` if `renewable - EFR <= 0`; classes < 25 no stress, 25-50 low, 50-75 medium, 75-100 high, > 100 critical | % | FAO (2018) | `sdg_642_water_stress(withdrawal_mm3, renewable_mm3, environmental_flow_mm3)` |
| Dependency ratio | `external / total_renewable`, in [0, 1]; 0 when total = 0 | - | FAO AQUASTAT glossary | `dependency_ratio(external_inflow_mm3, total_renewable_mm3)` |
| Groundwater stress | `abstraction / recharge`; `> 1` overdraft; `inf` if recharge = 0 and abstraction > 0 | - | Gleeson et al. (2012); Wada et al. (2010) | `groundwater_stress(abstraction_mm3, recharge_mm3)` |

### 2.2 Hashimoto performance criteria

A year is a *failure* when `supplied < RELIABILITY_TOL * demanded` with
`RELIABILITY_TOL = 0.999` (a 0.1 % tolerance for floating-point noise); years
with zero demand are always satisfactory.

| Criterion | Formula | Source | Function |
|-----------|---------|--------|----------|
| Reliability | `1 - (number of failure years) / n` | Hashimoto, Stedinger & Loucks (1982), eq. 2 | `supply_reliability(supplied, demanded)` |
| Resilience | `(number of failure -> success transitions) / (number of failure years)`; `1.0` with no failures; a failure in the last year counts as unrecovered | Hashimoto et al. (1982), eq. 7; Loucks & van Beek (2005) §11.3.3 | `resilience(supplied, demanded)` |
| Vulnerability | `mean over failure years of clip((demanded - supplied) / demanded, 0, 1)`; `0.0` with no failures | Hashimoto et al. (1982), eq. 9 | `vulnerability(supplied, demanded)` |

### 2.3 Basin water-balance routing

`route_basin(basin, flow_factor=1.0, entitlements=None, demands=None,
storages=None, env_flows=None, local_inflow_factors=None,
reservoir_refill_fraction=0.25) -> BasinBalance` is a priority-driven annual
water balance in the tradition of WEAP (Yates et al. 2005) and reservoir-system
simulation (Wurbs 1993; Loucks & van Beek 2005, ch. 12). Reaches are processed
upstream to downstream; the outflow of one reach is the upstream inflow of the
next. For each reach (all volumes in Mm3):

```text
local       = local_inflow_mm3 * flow_factor * local_inflow_factor      (default factor 1)
inflow      = upstream_outflow + local                                   (headwater_inflow * flow_factor for reach 0)
storage     = start-of-year reservoir storage                            (storages[name] or reservoir_storage_mm3)
demand[s]   = withdrawal demand per sector                               (demands[name] or demand.withdrawals())
env         = environmental flow requirement at the outlet               (env_flows[name] or demand.environmental)
cap         = entitlement                                                (entitlements[name], treaty_allocation_mm3, or inf)

# non-river supply (groundwater abstraction + desalination capacity) offsets demand pro rata
nonriver        = min(groundwater_abstraction_mm3 + desalination_capacity_mm3, sum_s demand[s])
river_demand[s] = demand[s] * (1 - nonriver / sum_s demand[s])           (0 when total demand is 0)

# serve sectors in SECTOR_PRIORITY order within two cumulative limits
supply          = inflow + storage
cap_withdrawal  = min(cap, supply)
cap_consumption = (supply - env)+
for s in [municipal, industrial, energy, agricultural]:
    f_s   = consumption_fraction[s]
    w_s   = min(river_demand[s], cap_withdrawal - cum_w, (cap_consumption - cum_c) / f_s)   (last term inf if f_s = 0)
    c_s   = w_s * f_s
deficit[s]  = river_demand[s] - w_s

# reservoir operation
release     = min((sum_s w_s - inflow)+, storage)                        # water taken from storage to meet withdrawals
river       = (inflow + release - sum_s c_s)+                            # water in the river after use (return flows included)
env_release = min((env - river)+, storage - release)                     # top the river up to the environmental flow
release    += env_release;  river += env_release
surplus     = (river - env)+
refill      = clip(surplus * reservoir_refill_fraction, 0, capacity - (storage - release))
outflow     = river - refill
evaporation = ((storage - release) + refill) * reservoir_evaporation_fraction
storage_end = (storage - release) + refill - evaporation
env_flow_met = outflow >= env - 1e-9
```

Every reach satisfies the exact identity

```text
inflow + storage_start = consumption + outflow + storage_end + evaporation
```

checked by `ReachResult.mass_balance_error()` and
`BasinBalance.mass_balance_error()` (largest reach violation, ~1e-12 Mm3 in
practice; the tests require < 1e-6). Derived quantities:

| Quantity | Formula | Function |
|----------|---------|----------|
| reach supply ratio | `clip(1 - total_deficit / total_demand, 0, 1)`; 1 when demand is 0 | `ReachResult.supply_ratio()` |
| basin supply ratio | same with basin totals | `BasinBalance.supply_ratio()` |
| return flow | `total_withdrawal - total_consumption` | `ReachResult.return_flow()` |
| share of reaches meeting their environmental flow | `#met / #reaches` (1 if none) | `BasinBalance.env_flow_met_share()` |
| natural flow | `(headwater + sum of local inflows) * flow_factor` | `BasinBalance.natural_flow_mm3` |
| end-of-year storages (to carry into the next year) | `{name: storage_end}` | `BasinBalance.storages_end()` |

### 2.4 Natural flows and stochastic flow factors

- `natural_flows(basin, flow_factor=1.0, local_inflow_factors=None)` returns
  the zero-use inflow at each reach: `N_r = flow_factor * (headwater + sum_{k <= r} local_k)`.
- `stochastic_flow_factors(scenario, basin)` returns the deterministic trend
  `[scenario.flow_factor(i)]` when `scenario.stochastic` is False or
  `basin.climate_cv` is 0; otherwise each year's trend value is multiplied by
  an independent lognormal variate with mean 1 and coefficient of variation
  `cv = basin.climate_cv`:

  ```text
  sigma = sqrt(ln(1 + cv^2)),  mu = -sigma^2 / 2,  m_i ~ LogNormal(mu, sigma),  factor_i = trend_i * m_i
  ```

  drawn with `numpy.random.default_rng(scenario.seed)` (Stedinger 1980;
  Loucks & van Beek 2005, ch. 7).

---

## 3. Energy (`wefnexus.energy`)

Constants: `HYDRO_G = 9.81` m/s2, `WATER_DENSITY = 1000` kg/m3,
`J_PER_KWH = 3.6e6`, `J_PER_GWH = 3.6e12`, `HOURS_PER_YEAR = 8760`.

| Quantity | Formula | Units | Source | Function |
|----------|---------|-------|--------|----------|
| Specific hydropower yield | `rho g H eta / 3.6e6 = 0.002725 H eta` | kWh/m3 | Gulliver & Arndt (1991) | `specific_hydropower_kwh_per_m3(head_m, efficiency=0.9)` |
| Hydropower generation | `E = rho g (V * 1e6) H eta / 3.6e12`, then `min(E, capacity_mw * hours / 1000)` when a capacity is given (1 Mm3 through 100 m at eta = 1 gives exactly 0.2725 GWh) | GWh | Gulliver & Arndt (1991); Kumar et al. (2011) | `hydropower_gwh(volume_mm3, head_m, efficiency=0.9, capacity_mw=None, hours=8760)` |
| Thermal generation | `capacity_mw * capacity_factor * hours / 1000` | GWh | IPCC (2014) Annex III | `thermal_generation_gwh(capacity_mw, capacity_factor, hours=8760)` |
| Consumptive cooling water | `generation_gwh * 1000 * intensity_m3_per_mwh / 1e6` | Mm3 | Macknick et al. (2012) | `thermal_cooling_water_mm3(generation_gwh, intensity_m3_per_mwh)`; `water_for_energy(riparian, thermal_generation_gwh)` uses `energy.thermal_water_intensity_m3_per_mwh` |
| Specific pumping energy | `rho g H / (eta 3.6e6) = 0.002725 H / eta` | kWh/m3 | Plappally & Lienhard (2012) | `specific_pumping_energy_kwh_per_m3(lift_m, efficiency=0.6)` |
| Pumping energy | `E_kWh = rho g (V * 1e6) H / (eta 3.6e6)`, `E_GWh = E_kWh / 1e6` | GWh | Plappally & Lienhard (2012) | `pumping_energy_gwh(volume_mm3, lift_m, efficiency=0.6)` |
| Desalination energy | `volume_mm3 * kwh_per_m3` (Mm3 x kWh/m3 = GWh); default 3.5 kWh/m3 (seawater RO) | GWh | Voutchkov (2018) | `desalination_energy_gwh(volume_mm3, kwh_per_m3=3.5)` |
| Treatment energy | `volume_mm3 * kwh_per_m3` | GWh | Plappally & Lienhard (2012) | `treatment_energy_gwh(volume_mm3, kwh_per_m3)` |

**Energy embedded in water supply** - `energy_for_water(riparian, withdrawals,
desalinated_mm3, pumped_fraction=None)` returns GWh/yr by component:

```text
pumping      = pumping_energy_gwh(withdrawal[agricultural] * pumped_fraction, pumping_lift_m, pumping_efficiency)
desalination = desalination_energy_gwh(desalinated_mm3, desal_energy_kwh_m3)
treatment    = treatment_energy_gwh(withdrawal[municipal] + withdrawal[industrial], treatment_energy_kwh_m3)
wastewater   = treatment_energy_gwh(withdrawal[municipal] (1 - f_mun) + withdrawal[industrial] (1 - f_ind), wastewater_energy_kwh_m3)
total        = pumping + desalination + treatment + wastewater
```

with `pumped_fraction` defaulting to `energy.pumped_fraction` and `f_s` the
sector consumption fractions (Plappally & Lienhard 2012; Hoff 2011).

**Energy balance** - `energy_balance(riparian, hydropower_gwh, thermal_gwh,
other_renewable_gwh, demand_gwh)`:

```text
supply           = hydropower + thermal + other_renewable                 [GWh]
deficit          = (demand - supply)+                                      [GWh]
self_sufficiency = supply / demand   (1 when demand is 0; not capped)
emissions_t      = thermal * grid_emission_factor_t_per_gwh                [t CO2]  (hydro and other renewables count as 0)
```

**Energy availability index** - `energy_security_index(supply_gwh, demand_gwh)
= min(supply / demand, 1)` (1 when demand is 0), the availability dimension of
Kruyt et al. (2009). The composite energy security index is in section 7.

---

## 4. Food (`wefnexus.food`)

Constants: `M3_PER_HA_MM = 10`, `DAYS_PER_YEAR = 365`, `DEFAULT_KCAL_PER_KG =
3400` (wheat), `DEFAULT_VIRTUAL_WATER_M3_PER_T = 1500` (cereal average),
`DEFAULT_FARM_ENERGY_KWH_PER_HA = 150`.

| Quantity | Formula | Units | Source | Function |
|----------|---------|-------|--------|----------|
| Seasonal crop evapotranspiration | `ETc = ET0 * Kc * season_days` (single seasonal-mean Kc) | mm | Allen et al. (1998), FAO-56 | `crop_evapotranspiration_mm(et0_mm_day, kc, season_days)` |
| Net irrigation requirement | `IRn = (ETc - Pe)+`, `Pe` effective rainfall | mm | FAO-24/56; Brouwer et al. (1989) | `net_irrigation_mm(etc_mm, effective_rainfall_mm)` |
| Gross irrigation requirement | `IRg = IRn / efficiency`, efficiency in (0, 1] | mm | Brouwer et al. (1989) | `gross_irrigation_mm(net_mm, efficiency)` |
| Crop irrigation volume | `IR = IRg * area_ha * 10 / 1e6` | Mm3/yr | Allen et al. (1998) | `irrigation_requirement_mm3(crop, et0_mm_day, effective_rainfall_mm, efficiency)` |
| Riparian irrigation requirement | `sum_i IR_i * area_factor`, efficiency defaulting to `riparian.irrigation_efficiency` | Mm3/yr | - | `riparian_crop_requirements_mm3(riparian, area_factor=1, efficiency=None)` (per crop), `riparian_irrigation_requirement_mm3(...)` (total) |
| Relative yield | `Ya/Ym = clip(1 - ky (1 - min(ETa/ETm, 1)), 0, 1)` | - | Doorenbos & Kassam (1979), FAO-33 | `fao33_relative_yield(ky, et_ratio)` |
| Actual yield | `Ya = Ym * (Ya/Ym)` | t/ha | FAO-33 | `fao33_yield(yield_max_t_ha, ky, et_ratio)` |
| Dietary energy demand | `population * kcal_per_capita_day * days` | kcal/yr | FAO (2001) | `food_demand_kcal(population, kcal_per_capita_day=2500, days=365)`; `Riparian.food_demand_kcal()` |
| Self-sufficiency ratio | `kcal_produced / kcal_demand` (not capped; 1 when demand is 0) | - | FAO (2001) SSR, calorie-based | `food_self_sufficiency(kcal_produced, kcal_demand)` |
| Food availability index | `min(SSR, 1)` | - | FAO (1996, 2001) | `food_security_index(kcal_produced, kcal_demand)` |
| Virtual water content | `water_mm3 * 1e6 / production_t`; `inf` when nothing is produced | m3/t | Allan (1998); Hoekstra & Chapagain (2008) | `virtual_water_content_m3_per_t(water_mm3, production_t)` |
| Crop water productivity | `production_t * 1000 / (water_mm3 * 1e6)` | kg/m3 | Molden et al. (2010) | `water_productivity_kg_per_m3(production_t, water_mm3)` |
| Virtual water import | `max(kcal_deficit, 0) / (kcal_per_kg * 1000) * m3_per_t / 1e6` | Mm3/yr | Allan (1998); Mekonnen & Hoekstra (2011) | `virtual_water_import_mm3(kcal_deficit, kcal_per_kg=3400, m3_per_t=1500)` |
| Direct farm energy | `area_ha * kwh_per_ha / 1e6` (excludes irrigation pumping) | GWh/yr | FAO (2011); Pelletier et al. (2011) | `energy_for_agriculture_gwh(area_ha, kwh_per_ha=150)` |

**Crop production** - `crop_production(crop, water_supplied_mm3,
water_required_mm3, area_factor=1)`:

```text
area        = crop.area_ha * area_factor                                   [ha]
et_ratio    = min(supplied / required, 1)   (1 when required is 0)
yield_t_ha  = fao33_yield(yield_max_t_ha, ky, et_ratio)
production  = yield_t_ha * area                                            [t]
kcal        = production * 1000 * kcal_per_kg                              [kcal]
value_usd   = production * price_usd_t                                     [USD]
```

The relative *gross irrigation supply* stands in for `ETa/ETm`; it ignores the
rain-fed share of crop ET and is therefore conservative.

**Riparian food production** - `riparian_food_production(riparian,
agricultural_water_mm3, area_factor=1, efficiency=None)` shares the available
gross agricultural water among the crops in proportion to their requirement
("equal relative deficit"):

```text
requirement = sum_i IR_i;  supplied = min(available, requirement);  ratio = supplied / requirement  (1 if requirement is 0)
share_i     = IR_i * ratio;  result_i = crop_production(crop_i, share_i, IR_i, area_factor)
```

and returns the per-crop dictionaries plus totals (`production_t`, `kcal`,
`value_usd`, `requirement_mm3`, `supplied_mm3`, `et_ratio`, `available_mm3`,
`deficit_mm3`, `area_ha`). Because FAO-33 is linear, production is
non-decreasing and concave in the water available.

---

## 5. Allocation (`wefnexus.allocation`)

### 5.1 Claims (bankruptcy) rules

A claims problem (O'Neill 1982) divides an estate `E >= 0` among claims
`c_1..c_n >= 0`, `C = sum c_i`. Every rule returns awards in the container
type of `claims` (list, tuple or dict) with `0 <= a_i <= c_i` and
`sum a_i = min(E, C)`; all are homogeneous of degree one, so the unit (Mm3/yr
throughout the package) is immaterial.

| Rule | Formula | Interpretation | Source | Function / `RULES` key |
|------|---------|----------------|--------|------------------------|
| Proportional | `a_i = c_i * min(E, C) / C` | equal satisfaction; "x % of the flow" treaties | O'Neill (1982); Thomson (2003) | `proportional` / `"proportional"` |
| Constrained equal awards | `a_i = min(c_i, lambda)`, `lambda` such that awards exhaust `min(E, C)`; computed exactly by sorting claims | protects small users first (leximin) | Aumann & Maschler (1985) | `constrained_equal_awards` / `"cea"` |
| Constrained equal losses | `a_i = (c_i - lambda)+`; computed by duality `CEL(E, c) = c - CEA(C - E, c)` | equal absolute cuts, protects large users | Aumann & Maschler (1985) | `constrained_equal_losses` / `"cel"` |
| Talmud | `CEA(E, c/2)` if `E <= C/2`, else `c/2 + CEL(E - C/2, c/2)` | contested-garment rule; nucleolus of the bankruptcy game | Aumann & Maschler (1985) | `talmud` / `"talmud"` |
| Minimal rights | `m_i = min((E - sum_{j != i} c_j)+, c_i)` | undisputed part of the estate | Curiel, Maschler & Tijs (1987) | `minimal_rights` |
| Adjusted proportional | award `m`, then divide `E' = E - sum m` proportionally to `c'_i = min(c_i - m_i, E')` | tau-value of the bankruptcy game | Curiel, Maschler & Tijs (1987) | `adjusted_proportional` / `"ap"` |
| Equal split | equal to CEA (equal shares with redistribution of unused shares) | "equal utilisation" read literally | UN (1997) Art. 5 | `equal_split` / `"equal"` |
| Upstream priority | `a_i = min(c_i, E - sum_{j < i} a_j)` in index order | absolute territorial sovereignty (Harmon doctrine); the disagreement point | Ansink & Weikard (2012) | `upstream_priority` / `"upstream_priority"` |

Closed-form checks (claims 100, 200, 300): Talmud gives (33.3, 33.3, 33.3)
at `E = 100`, (50, 75, 75) at `E = 200` and (50, 100, 150) at `E = 300`.
`apply_rule(rule, estate, claims)` dispatches by name (case-insensitive;
aliases in `RULE_ALIASES`, e.g. `"contested_garment"` -> `"talmud"`,
`"harmon"` -> `"upstream_priority"`, or a callable); `compare_rules(estate,
claims, rules=None)` returns `{rule: awards}` for every rule.

### 5.2 Cooperative game solutions

A transferable-utility game is a characteristic function `v(S)` on coalitions
`S` of the player set `N`, with `v(empty) = 0`.

| Concept | Formula | Source | Function |
|---------|---------|--------|----------|
| O'Neill bankruptcy game | `v(S) = (E - sum_{j not in S} c_j)+` (convex; nucleolus = Talmud awards, Shapley value = random-arrival rule, tau-value = adjusted proportional) | O'Neill (1982); Aumann & Maschler (1985) | `bankruptcy_game(estate, claims)` |
| Game from a table | `v` from `{coalition: worth}` | - | `game_from_dict(values, default=None)` |
| Shapley value | `phi_i = sum_{S not containing i} |S|! (n - |S| - 1)! / n! * [v(S + i) - v(S)]`, exact over `2^n` coalitions (`n <= 16` by default) | Shapley (1953); Dinar, Ratner & Yaron (1992) | `shapley_value(players, v)` |
| Coalition excess | `e(S, x) = v(S) - x(S)`; positive = core constraint violated | Gillies (1959); Schmeidler (1969) | `core_constraints_violations(allocation, players, v)` |
| Core membership | `|x(N) - v(N)| <= tol` and `x(S) >= v(S) - tol` for all `S` | Gillies (1959); Madani (2010) | `is_in_core(allocation, players, v, tol=1e-9)` |
| Nucleolus | imputation that lexicographically minimises the non-increasingly ordered excess vector; sequential LPs with `scipy.optimize.linprog` (HiGHS), each level settling only coalitions whose excess cannot be reduced by any optimal solution; worths rescaled to unit magnitude; `n <= 8` by default; `imputation=False` gives the prenucleolus | Schmeidler (1969); Kohlberg (1971); Maschler, Peleg & Shapley (1979); Guajardo & Jornsten (2015) | `nucleolus(players, v, imputation=True, tol=1e-7)` |

### 5.3 Bargaining

`nash_bargaining(utility_fns, disagreement, total, grid=2001, refine=True,
seed=0)` finds the split `x` of `total` (`x_i >= 0`, `sum x_i = total`)
maximising the Nash product `prod_i (u_i(x_i) - d_i)` subject to `u_i(x_i) >=
d_i` (Nash 1950). Two parties: grid search over `x_1 in [0, total]` plus a
golden-section polish; three or more: SLSQP on `-sum log(u_i - d_i)` over the
simplex from seeded Dirichlet starts, polished by exact pairwise transfers.
With linear utilities the surplus over the disagreement payoffs is split
equally.

### 5.4 Equity metrics

| Metric | Formula | Source | Function |
|--------|---------|--------|----------|
| Gini coefficient | `G = 2 sum_i i x_(i) / (n sum x) - (n + 1) / n` on sorted values; 0 for equality, `1 - 1/n` when one party holds all; 0 for empty / all-zero input | Gini (1912); Sen (1973); Cullis & van Koppen (2007) | `gini(values)` |
| Satisfaction | `a_i / c_i` (1 when `c_i = 0`; may exceed 1) | Thomson (2003) | `satisfaction(awards, claims)` |
| Envy-freeness (satisfaction terms) | `max_i a_i/c_i - min_i a_i/c_i <= tol` | Foley (1967); Young (1994) | `envy_free(awards, claims, tol=1e-9)` |

---

## 6. Diplomacy (`wefnexus.diplomacy`)

### 6.1 Hydro-hegemony

`hydro_hegemony(riparian, basin, weights=None)` scores the four pillars of
Zeitoun & Warner (2006) and Cascao & Zeitoun (2010), each in [0, 1]:

```text
geographic = 1 - position / (n - 1)        (1 for the most upstream riparian, 0 for the most downstream; 1 if n = 1)
material   = riparian.material_power
bargaining = riparian.bargaining_power
ideational = riparian.ideational_power
score      = sum_k w_k pillar_k / sum_k w_k   (default weights 1)
```

`power_asymmetry(basin, weights=None) = max_r score_r - min_r score_r`
(0 with fewer than two riparians).

### 6.2 Basins at Risk events, cooperation, conflict and TWINS

`BarEvent(year, parties, scale, issue="", description="")` holds one
interaction on the Basins at Risk water event intensity scale (Yoffe, Wolf &
Giordano 2003, after Azar's COPDAB), `BAR_SCALE`:

| Scale | Label |
|------:|-------|
| -7 | Formal declaration of war |
| -6 | Extensive war acts causing deaths, dislocation or high strategic cost |
| -5 | Small scale military acts |
| -4 | Political-military hostile actions |
| -3 | Diplomatic-economic hostile actions |
| -2 | Strong verbal expressions displaying hostility in interaction |
| -1 | Mild verbal expressions displaying discord in interaction |
| 0 | Neutral or non-significant acts for the inter-nation situation |
| +1 | Minor official exchanges, talks or policy expressions - mild verbal support |
| +2 | Official verbal support of goals, values or regime |
| +3 | Cultural or scientific agreement or support (non-strategic) |
| +4 | Non-military economic, technological or industrial agreement |
| +5 | Military, economic or strategic support |
| +6 | International freshwater treaty; major strategic alliance |
| +7 | Voluntary unification into one nation |

Event windows: `until` defaults to the latest event year and `window_years`
keeps events with `until - window_years < year <= until`; `parties` keeps
events involving at least one named riparian.

| Indicator | Formula | Source | Function |
|-----------|---------|--------|----------|
| Cooperation index | mean BAR scale of the events in the window (0 if none), in [-7, 7] | Wolf, Yoffe & Giordano (2003) | `cooperation_index(events, window_years=None, until=None, parties=None)` |
| Conflict intensity | mean `|scale|` of the *negative* events in the window (0 if none), in [0, 7] | Yoffe et al. (2003); Mirumachi & Allan (2007) | `conflict_intensity(...)` |
| Event summary | counts of cooperative / conflictive / neutral events, shares, extremes, both indices and the TWINS cell | - | `bar_event_summary(...)` |
| TWINS cooperation class | `confrontation` below 1; `ad_hoc >= 1`; `technical >= 3`; `risk_averting >= 5`; `risk_taking >= 6` (`TWINS_COOPERATION_THRESHOLDS` on the mean scale) | Mirumachi & Allan (2007) | `twins_cooperation_class(cooperation)` |
| TWINS conflict class | `none` at 0; `non-politicised > 0`; `politicised >= 2`; `securitised >= 4`; `violent >= 5` (`TWINS_CONFLICT_THRESHOLDS` on the conflict intensity) | Warner (2004); Mirumachi & Allan (2007) | `twins_conflict_class(conflict)` |
| TWINS cell | `"<cooperation class>/<conflict class>"` - conflict and cooperation coexist | Mirumachi & Allan (2007); Mirumachi (2015) | `twins_classification(cooperation, conflict)` |

### 6.3 Treaties

`Treaty(name, parties, allocations_mm3={}, variable_allocation=False,
drought_provisions=False, data_sharing=False, joint_institution=False,
dispute_resolution=False, benefit_sharing=False, review_period_years=None,
year_signed=None, reference_flow_mm3=None)`. `Treaty.entitlements(natural_flow_mm3)`
returns the fixed volumes, scaled by `natural_flow / reference_flow` when
`variable_allocation` is set and a reference flow is given ("shares of actual
flow"). `treaty_from_basin(basin, name=None, **features)` builds a fixed-volume
treaty from the riparians' `treaty_allocation_mm3` (None when no riparian has
one).

**Institutional resilience** (De Stefano et al. 2012; Drieschova et al. 2008;
benefit sharing after Sadoff & Grey 2002) -
`treaty_resilience(treaty, weights=None)`:

```text
resilience = sum_{mechanism present} w_mechanism / sum_all w,   TREATY_RESILIENCE_WEIGHTS =
  variable_allocation 0.20, drought_provisions 0.20, data_sharing 0.15, joint_institution 0.15,
  dispute_resolution 0.15, benefit_sharing 0.10, review_period 0.05   (review_period = review_period_years is not None)
```

A bare fixed-volume treaty scores 0; one with every mechanism scores 1.

**Compliance** - `treaty_compliance(treaty, balance, tol=1e-9)` compares each
party's routed year with its entitlement `e` (Wolf et al. 2003; UN 1997 Arts.
5-7):

```text
compliance_ratio        = withdrawal / e        (> 1 = over-abstraction; 0 for an unlimited entitlement; for e = 0: 1 if nothing withdrawn else inf)
over_abstraction_mm3    = (withdrawal - e)+
compliant               = 1 if withdrawal <= e (1 + tol) else 0
delivery_ratio          = inflow / e            (1 when there is no finite positive entitlement)
delivery_shortfall_mm3  = (e - inflow)+
delivered               = 1 if inflow >= e (1 - tol) else 0
```

`compliance_summary(compliance)` aggregates to `n_parties`,
`share_compliant`, `share_delivered`, the two totals and `operational` (1 when
every party is compliant and delivered - a proxy for an SDG 6.5.2 operational
arrangement).

### 6.4 Upstream dependency

`water_dependency(basin, balance=None)` returns, per riparian,
`dependency_ratio(upstream_inflow, inflow)` = share of the reach inflow that
arrived from upstream (routed flows with a `balance`, natural flows otherwise)
- the surface-water analogue of the AQUASTAT dependency ratio and a classic
Basins-at-Risk vulnerability.

### 6.5 Conflict risk index

`conflict_risk_index(basin, balance, treaty=None, events=(), climate_cv=None,
weights=None, window_years=None, until=None)` is a Basins-at-Risk style
composite (Wolf, Yoffe & Giordano 2003; De Stefano et al. 2012): tension rises
when rapid or extreme change outpaces institutional capacity. Eight components
in [0, 1] (higher = riskier) are combined by a weighted mean with
`CONFLICT_RISK_WEIGHTS`:

| Component | Weight | Formula |
|-----------|-------:|---------|
| `water_stress` | 0.20 | mean over riparians of `min(SDG 6.4.2 / 100, 1)` with withdrawal = surface withdrawal + groundwater used (`non_river * gw_abstraction / (gw_abstraction + desalination)`), renewable = reach inflow + groundwater recharge, EFR = reach environmental flow; `inf` counts as 1 |
| `dependency` | 0.15 | mean of `water_dependency(basin, balance)` |
| `power_asymmetry` | 0.10 | `power_asymmetry(basin)` |
| `institutional` | 0.20 | `1.0` without a treaty; else `1 - coverage * (c + (1 - c) * resilience)` with `coverage` = share of riparians that are parties, `resilience = treaty_resilience(treaty)`, `c = TREATY_PRESENCE_CREDIT = 0.5` (a bare treaty covering everybody scores 0.5, a fully resilient one 0) |
| `conflict_history` | 0.10 | mean over events in the window of `max(-scale, 0) / 7` (0 without events) |
| `variability` | 0.10 | `min(climate_cv / VARIABILITY_CV_REFERENCE, 1)`, reference CV 0.5; `climate_cv` defaults to `basin.climate_cv` |
| `environmental` | 0.05 | mean over reaches of `(env - outflow)+ / env` (0 when `env = 0`) |
| `dam_filling` | 0.10 | `min(sum of storage_refill of reaches that have a downstream neighbour / natural flow, 1)` |

```text
score    = sum_k w_k component_k / sum_k w_k
category = low (score < 0.25), moderate (< 0.50), high (< 0.75), very_high (>= 0.75)     -> risk_category(score)
```

The result also carries the weights used and a `details` dictionary
(per-riparian stress in %, dependencies, hegemony scores, treaty resilience and
coverage, event counts, environmental shortfalls, upstream refill, natural
flow, CV).

### 6.6 Benefit sharing (Sadoff & Grey 2002)

`benefit_sharing_matrix(basin, balance, cooperative_balance=None, treaty=None,
events=(), hydropower_value_usd_per_kwh=0.05)` proxies the four benefit types
per riparian:

```text
to_the_river          = min(outflow / env, 1)                       (1 when env = 0)         [-]
from_the_river        = sum_s withdrawal_s * value_usd_per_m3[s] * 1e6 + hydropower_gwh * 1e6 * price   [USD/yr]
because_of_the_river  = 1 - conflict_risk_index(basin, balance, treaty, events)["score"]      [-]
beyond_the_river      = from_the_river(cooperative_balance) - from_the_river(balance)          [USD/yr] (0 without a cooperative balance)
```

where the hydropower of a reach is `hydropower_gwh(outflow * turbined_fraction,
head, turbine_efficiency, capacity_mw)` (0 when capacity or head is 0). The
matrix also reports `water_value_usd`, `hydropower_gwh` and
`hydropower_value_usd`.

### 6.7 Negotiation

`negotiate(basin, flow_factor=1.0, rule="talmud", claims=None, batna=None,
estate=None, tol=1e-6, claim_basis="demand", include_storage=False)` (Fisher &
Ury 1981; Ansink & Weikard 2012; Mianabadi et al. 2014) divides the water on a
*consumptive* basis - the net-use accounting of river sharing problems, the
same convention as `NexusModel.entitlements` (section 8) - and hands the
awards back as gross withdrawal caps:

```text
1. claims_r   = river_demand_r = (sum_s demand_r[s] - groundwater_r - desalination_r)+   (claim_basis="demand", default)
              = treaty_allocation_mm3_r, or river_demand_r when None                    (claim_basis="treaty")
              `claims` overrides entries (gross withdrawals)
   ratio_r    = sum_s f_r[s] demand_r[s] / sum_s demand_r[s]        (consumption_ratio; 1 when the demand is 0)
   cclaim_r   = claims_r * ratio_r                                   (consumptive claim)
2. estate     = (natural_flow * flow_factor - environmental_flow_terminal)+   (bankruptcy_estate; or `estate` if given)
3. caward     = apply_rule(rule, estate, cclaim)                     (consumptive awards, sum = min(estate, sum cclaim))
   proposal_r = clip(caward_r / ratio_r, 0, claims_r)                (gross withdrawal cap; = claims_r when caward_r = cclaim_r)
4. batna_r    = surface withdrawal of r in route_basin(basin, flow_factor, entitlements={all None},
                                                      storages={all 0}, reservoir_refill_fraction=0)
              (unilateral upstream priority from this year's flow; `include_storage=True` keeps the
              riparians' reservoir storage and the default refill; `batna` overrides)
5. acceptable_r = proposal_r >= batna_r - tol;   zopa = all acceptable
```

`river_demand_mm3(riparian)` is the withdrawal demand net of the non-river
supply, exactly the sum of the `river_demand` that `route_basin` serves (the
pro-rata offset leaves the total unchanged); `consumption_ratio(riparian)` is
the demand-weighted consumption fraction, invariant to that offset, so
`consumptive = gross * ratio` holds for the demand and for the river demand
alike. Claims are demands by default because a claim is what a party asks
for; a treaty entitlement is itself a negotiated outcome and is available as
`claim_basis="treaty"` to renegotiate from the existing settlement.

`bankruptcy_estate(basin, flow_factor=1.0)` defines the estate as the largest
total consumption compatible with every in-stream requirement. With `Q_r` the
cumulative natural flow at the outlet of reach `r` (`natural_flows`) and
`env_r` the requirement there, the constraints `sum_{k <= r} c_k <= Q_r -
env_r` give `estate = (Q_terminal - env_terminal)+`: only the requirement at
the terminal outlet is withheld, because an in-stream flow is not a withdrawal
and the water left at an upstream outlet flows on to the riparians below.
Subtracting every reach's requirement (`natural_flow - sum_r env_r`) would
count the same water once per reach it passes - on the example basin 7 000
instead of 1 500 Mm3/yr withheld at mean flow. The function also returns
`natural_flow_mm3`, `environmental_flow_mm3` (the sum, information only),
`terminal_environmental_flow_mm3`, `environmental_reserve_mm3` (`natural_flow
- estate`), `headroom_mm3` (`{r: Q_r - env_r}`, the cumulative consumption
allowed above each outlet) and `infeasible_reaches` (reaches whose requirement
exceeds the natural flow at their own outlet).

The proposal is also routed as entitlements (same storage setting) to report
`routed_withdrawals`, `routed_consumption`, `outflow_to_sea_mm3` and
`env_flow_met_share`; `gini` (of the gross proposal), `gini_satisfaction` and
`satisfaction` (`proposal / claim = caward / cclaim`) are returned too, with
`claim_basis`, `include_storage`, `consumption_ratio`, `consumptive_claims`,
`consumptive_awards`, `total_consumptive_award_mm3` (`= min(estate, sum
cclaim)`) and `total_awarded_mm3` (sum of the gross caps). The cap is pro rata while the
router fills the sectors in `SECTOR_PRIORITY` order (municipal, industrial,
energy, agricultural), so a capped riparian consumes `sum_s f_s w_s` over the
sectors filled before the cap binds: at most its consumptive award whenever no
prefix of that order has a demand-weighted fraction above the riparian's ratio
- with the default fractions every sector ahead of agriculture (0.20 / 0.10 /
0.03) lies below the example ratios 0.37 / 0.45 / 0.47, so the bound holds
there - but a high-fraction sector served first (e.g. municipal with `f =
0.95`) lets a riparian consume more than its award. The total routed
consumption stays within the estate whenever the terminal requirement is met
and storage is excluded (`natural_flow = consumption + outflow_to_sea` with
`outflow_to_sea >= env_terminal`), whatever the caps; `NexusModel` (section 8)
inverts each award sector by sector in priority order instead and has the
per-riparian guarantee by construction. On the example basin the consumptive
claims (11 468 Mm3/yr) fit into the estate down to a flow factor of about
0.46: at 1.0 and 0.6 every riparian is capped at its river demand and a ZOPA
exists; at 0.4 (estate 9 700) the Talmud rule rations, the upstream riparians
- whose BATNA is their full river demand - reject, and Delta, which can only
withdraw the 7 328 Mm3 that still reach it unilaterally, accepts.

### 6.8 Rule comparison

`compare_allocation_rules(basin, flow_factor=1.0, rules=None, claims=None,
estate=None, claim_basis="demand", include_storage=False)` builds the same
claims problem, applies every rule to the consumptive claims, converts each
award vector into gross caps and routes it, returning per rule: `estate`,
`claims` (gross), `consumption_ratio`, `consumptive_claims`,
`consumptive_awards`, `total_consumptive_award_mm3`, `awards` (gross caps),
`total_awarded_mm3`, `gini`, `gini_satisfaction`, `satisfaction`,
`min_satisfaction`, `withdrawals`, `total_withdrawal_mm3`, `consumption`,
`total_consumption_mm3`, `supply_ratio`, `outflow_to_sea_mm3`,
`env_flow_met_share`.

---

## 7. Sustainability (`wefnexus.sustainability`)

### 7.1 Normalisation and aggregation

| Operation | Formula | Source | Function |
|-----------|---------|--------|----------|
| Min-max normalisation | `t = (value - lo) / (hi - lo)` clipped to [0, 1]; `lo` maps to 0, `hi` to 1 (the scale may run downwards, e.g. `normalise(stress, 100, 0)`); `higher_is_better=False` returns `1 - t`; `+-inf` clip to the matching end | Nardo et al. (2005); OECD/JRC (2008); UNDP (2010) goalposts | `normalise(value, lo, hi, higher_is_better=True)` |
| Weighted geometric mean | `G = exp(sum_i w_i ln v_i / sum_i w_i)` with every `v_i` clipped to `[GEOMETRIC_FLOOR, 1] = [1e-6, 1]` (one zero component among three gives 0.01); partially compensatory, never above the arithmetic mean | Ebert & Welsch (2004); UNDP (2010) HDI; Simpson et al. (2022) | `weighted_geometric_mean(values, weights=None)` |

### 7.2 Pillar indices

| Index | Components (each in [0, 1]) | Aggregation | Source | Function |
|-------|-----------------------------|-------------|--------|----------|
| Water security | `supply = min(supply_ratio, 1)`; `stress = normalise(SDG 6.4.2 %, 100, 0)` (`WATER_STRESS_RANGE`); `environment = env_flow_met_share` (in the nexus model: `min(outflow / env, 1)`); `groundwater = normalise(abstraction / recharge, 1.5, 0.5)` (`GROUNDWATER_STRESS_RANGE`) | GM, default equal weights keyed `WATER_COMPONENTS` | Sullivan (2002); FAO (2018); Gleeson et al. (2012) | `water_security_components(...)`, `water_security_index(supply_ratio, stress_sdg642, env_flow_met_share, groundwater_stress, weights=None)` |
| Energy security | `supply = min(supply / demand, 1)`; `renewable = renewable share of supply`; `emissions = normalise(t CO2 / GWh, 1000, 0)` (`EMISSION_INTENSITY_RANGE`) | GM, keys `ENERGY_COMPONENTS` | Kruyt et al. (2009); Sovacool & Mukherjee (2011); IPCC (2014) | `energy_security_components(...)`, `energy_security_index(supply_ratio, renewable_share, emission_intensity_t_per_gwh, weights=None)` |
| Food security | `self_sufficiency = min(SSR, 1)`; `et_ratio = min(ETa/ETm, 1)` | GM, keys `FOOD_COMPONENTS` | FAO (1996) availability and stability pillars; FAO-33 | `food_security_components(...)`, `food_security_index(self_sufficiency, et_ratio, weights=None)` |
| WEF nexus index | the three pillar indices (capped at 1) | GM, keys `NEXUS_PILLARS` | Hoff (2011); Bizikova et al. (2013); Simpson et al. (2022) | `wef_nexus_index(water, energy, food, weights=None)` |
| Equity index | `1 - gini(values)` of a per-riparian distribution (supply ratios, awards, per-capita water, ...) | - | Gini (1912); Cullis & van Koppen (2007) | `equity_index(values_by_riparian)` |

Worked values: `water_security_index(1, 50, 1, 1) = (1 * 0.5 * 1 * 0.5)^(1/4)
= 0.707`; `energy_security_index(0.5, 0.5, 500) = 0.5`;
`wef_nexus_index(1, 1, 0) = 0.01`.

### 7.3 SDG-style indicators

`sdg_indicators(gdp_usd, withdrawal_mm3, water_stress_pct, renewable_share,
food_self_sufficiency, transboundary_cooperation=0.0)` returns the keys of
`SDG_KEYS`:

| Key | Indicator | Formula | Source |
|-----|-----------|---------|--------|
| `6.4.1_water_use_efficiency_usd_per_m3` | SDG 6.4.1 water-use efficiency | `gdp_usd / (withdrawal_mm3 * 1e6)` USD/m3; `inf` with value added but no withdrawal, 0 when both are 0 (`water_use_efficiency_usd_per_m3`) | FAO (2018b) |
| `6.4.2_water_stress_pct` | SDG 6.4.2 level of water stress | as given (`sdg_642_water_stress`) | FAO (2018) |
| `6.5.2_transboundary_cooperation` | SDG 6.5.2 operational arrangement | 1 / 0 or the share covered, in [0, 1] | UN-Water (2018) |
| `7.2_renewable_share` | SDG 7.2 renewable share of energy supply | as given, in [0, 1] | - |
| `2.1_food_self_sufficiency` | proxy for SDG 2.1 (undernourishment) | calorie SSR, not capped | FAO (2001) |

### 7.4 Sustainability report

`assess(result)` (lazy import of `wefnexus.nexus.NexusResult`) calls
`assess_records(records, basin_name, scenario_name, years, balances,
cooperation)` and returns a `SustainabilityReport` with `riparians[name]` and
`basin` dictionaries, `summary()` (text table), `to_rows()`, `to_dict()`,
`to_dataframe()`, `indices()` and `sdg()`.

Per riparian (over the horizon): means of the four indices (`water_security`,
`energy_security`, `food_security`, `nexus_index`), `min_nexus_index`,
`nexus_index_trend` (last - first), `supply_ratio` (mean of annual
`1 - deficit / demand`), `min_supply_ratio`, Hashimoto `supply_reliability` /
`resilience` / `vulnerability` on supplied vs demanded volumes, means of
demand, withdrawal, deficit, inflow, upstream inflow, outflow, storage,
`final_storage_mm3`, environmental flow, `env_flow_met_share`,
`years_env_flow_unmet`, `entitlement_mm3`, `water_stress_sdg642`,
`groundwater_stress`, `per_capita_water_m3` and its Falkenmark class,
`transboundary_cooperation` (share of years with `scenario.cooperation` and a
finite entitlement - the SDG 6.5.2 input), energy means and ratios of horizon
totals (`energy_self_sufficiency = sum supply / sum demand`, `renewable_share
= sum(hydro + other) / sum supply`, `emission_intensity_t_per_gwh = sum
emissions / sum supply`), food means and `food_self_sufficiency = sum kcal /
sum demand`, and the five SDG keys (6.4.1 from horizon totals of GDP and
freshwater withdrawal - surface withdrawal plus the groundwater used, the FAO
denominator).

Basin: riparian means of the indices, `nexus_index_population_weighted`,
`min_nexus_index`, `worst_riparian`, `equity_index` (1 - Gini of riparian
supply ratios), `equity_nexus_index`, `equity_per_capita_water`, basin supply
ratio and Hashimoto criteria on basin totals, `env_flow_met_share` over all
riparian-years, energy and food totals, SDG keys and, when balances are
supplied, mean `natural_flow_mm3`, mean `outflow_to_sea_mm3`, basin
`per_capita_water_m3` (natural flow / population) and the maximum
`mass_balance_error_mm3`.

---

## 8. Integrated simulation (`wefnexus.nexus`)

`NexusModel(basin, scenario, allocation_rule=None, reservoir_refill_fraction=0.25,
flow_factors=None)` and `run_nexus(basin, scenario, **kw)` simulate every year
`i` of the scenario horizon in six steps (a WEAP-style balance, Yates et al.
2005, extended with the water-energy and water-food links of Daher & Mohtar
2015 and the bankruptcy allocation of Mianabadi et al. 2014).

**Step 1 - drivers** (`NexusModel.drivers(i)`, per riparian):

```text
population      = r.population * population_factor(i)
gdp             = r.gdp_usd * gdp_factor(i)
demand[municipal | industrial | energy] = WaterDemand value * demand_factor(i)
demand[agricultural] = riparian_irrigation_requirement_mm3(r, irrigated_area_factor(i), irrigation_efficiency(r.irrigation_efficiency, i))   if r.crops
                     = WaterDemand.agricultural * irrigated_area_factor(i)                                                                  otherwise
energy_demand   = r.energy.demand_gwh * energy_demand_factor(i)
renewable_share = scenario.renewable_share(r.energy.renewable_share, i)
```

**Step 2 - flow and allocation** (`flow_factors()`, `natural_flow(i)`,
`entitlements(i, natural_flow, demands)`): the year's flow factor is the
explicit `flow_factors` list, else `stochastic_flow_factors(scenario, basin)`.
Caps on surface withdrawal depend on the rule in force (`allocation_rule`,
which is `"upstream_priority"` whenever `scenario.cooperation` is False):

| Rule | Caps |
|------|------|
| `"upstream_priority"` / `None` | none (upstream withdraws first) |
| `"treaty"` | `{r: r.treaty_allocation_mm3}` (None = no cap); fixed volumes, not scaled with flow |
| bankruptcy rule (`proportional`, `cea`, `cel`, `talmud`, `ap`, `equal`) | consumptive-use accounting (`NexusModel.claims`, `NexusModel.entitlements`): `estate = (natural_flow + sum_r storage_start_r - sum_r environmental_r)+`, `claims_r = sum_s river_demand_r[s] * consumption_fraction_r[s]` (gross demand net of the riparian's non-river supply, pro rata as in `route_basin`), `awards = apply_rule(rule, estate, claims)`; the cap of `r` is the gross surface withdrawal at which its consumption, serving sectors in priority order, equals its award (never binding when award = claim, so a rule only rations under physical scarcity). `diplomacy.negotiate` (section 6.7) uses the same consumptive claims (it converts an award back into a cap pro rata, `award / consumption_ratio`, where this method serves the sectors in priority order); the remaining difference is the estate: `NexusModel` adds the carried storage and subtracts every reach's environmental flow (a cautious operating rule for a multi-year run), whereas `diplomacy.bankruptcy_estate` withholds only the terminal reserve and counts no storage (the water a single year's flow can divide) |

**Step 3 - water**: `route_basin(basin, flow_factor, entitlements=caps,
demands=demands, storages=previous year's storages_end(),
reservoir_refill_fraction)`.

**Steps 4-6 - per-riparian record** (`NexusModel._record`, producing a
`RiparianYear`):

```text
# non-river supply split
gw_used          = non_river_supply * gw_abstraction / (gw_abstraction + desalination_capacity)      (0 if none)
desal_used       = non_river_supply - gw_used

# water indicators
renewable        = inflow + groundwater_recharge                                                        [Mm3]
per_capita_water = renewable * 1e6 / population   (inf if population = 0);  falkenmark = falkenmark_category(...)
water_stress     = sdg_642_water_stress(surface_withdrawal + gw_used, renewable, env)                   [%]
groundwater_stress = groundwater_stress(gw_used, groundwater_recharge)
env_flow_compliance = min(outflow / env, 1)   (1 if env = 0)
supply_ratio     = reach.supply_ratio()
water_security   = water_security_index(supply_ratio, water_stress, env_flow_compliance, groundwater_stress)

# energy
hydropower       = hydropower_gwh(outflow * turbined_fraction, hydropower_head_m, turbine_efficiency, capacity_mw=hydropower_capacity_mw)
thermal          = thermal_generation_gwh(thermal_capacity_mw, thermal_capacity_factor)
cooling_water    = water_for_energy(r, thermal)                                                          [Mm3, diagnostic]
other_renewable  = energy_demand * renewable_share                     (non-hydro renewables; the hydropower above is added separately)
groundwater_pumping = pumping_energy_gwh(gw_used, pumping_lift_m, pumping_efficiency)   (energy_for_water lifts only the pumped share of surface withdrawals)
energy_for_water = energy_for_water(r, withdrawals, desal_used)["total"] + groundwater_pumping
energy_demand_total = energy_demand + energy_for_water
balance          = energy_balance(r, hydropower, thermal, other_renewable, energy_demand_total)
energy_supply_ratio = min(supply / demand_total, 1)
renewable_share_of_supply = (hydropower + other_renewable) / supply      (0 if supply = 0)
emission_intensity = emissions_t / supply                                 (0 if supply = 0)
energy_security  = energy_security_index(energy_supply_ratio, renewable_share_of_supply, emission_intensity)

# food
agricultural_water = demand[agricultural] - deficit[agricultural]         (surface withdrawal + non-river share)
fp               = riparian_food_production(r, agricultural_water, area_factor, irrigation_efficiency)
food_demand_kcal = food_demand_kcal(population, food_demand_kcal_per_capita_day)
food_self_sufficiency = fp["kcal"] / food_demand_kcal
et_ratio         = fp["et_ratio"] if r.crops else agricultural supply ratio
food_security    = food_security_index(food_self_sufficiency, et_ratio)

# composite and value
nexus_index      = wef_nexus_index(water_security, energy_security, food_security)
water_value_usd  = sum_s withdrawal_s * value_usd_per_m3[s] * 1e6
```

`NexusResult` holds `records` (year-major), `balances`, `flow_factors`,
`allocation_rule`, and offers `for_riparian`, `for_year`, `record`, `balance`,
`series(name, field)`, `basin_series(field, agg)` (`sum`, `mean`, `min`,
`max`, `any`, `all`), `to_records`, `to_dataframe` (lazy pandas), `to_csv`
and `summary()` (per-riparian horizon means, Hashimoto criteria, totals,
`BASIN` row with equity, outflow to sea, natural flow, mass-balance error).

---

## 9. Optimisation (`wefnexus.optimize`)

`optimize_allocation(basin, flow_factor=1.0, objective="benefit",
min_supply_ratio=0.0, equity_weight=0.0, env_flow_hard=True, values=None,
demands=None, env_flows=None, entitlements=None) -> OptimizationResult`
solves a linear programme with `scipy.optimize.linprog(method="highs")`
(Loucks & van Beek 2005, ch. 4; Harou et al. 2009). Reservoirs are ignored
(annual steady state).

Decision variables `w[r, s] >= 0` (surface withdrawal, Mm3/yr); linear routing
with consumption fractions `f[r, s]` and natural inflows `N_r`:

```text
in_r  = N_r - sum_{k < r} sum_s f[k, s] w[k, s]
out_r = in_r - sum_s f[r, s] w[r, s]
```

Constraints:

```text
0 <= w[r, s] <= river_demand[r, s]                 (gross demand net of the riparian's non-river supply, pro rata as in route_basin)
sum_s w[r, s] <= in_r
out_r >= env_r                                     (when env_flow_hard)
sum_s w[r, s] + non_river_r >= min_supply_ratio * gross_demand_r
sum_s w[r, s] <= entitlement_r                     (when entitlements are given: dict, or "treaty" for treaty_allocation_mm3)
```

Supply ratio of a riparian: `(sum_s w[r, s] + non_river_r) / gross_demand_r`
(1 without demand) - identical to `ReachResult.supply_ratio` for the same
withdrawals. Objectives:

| `objective` | Maximises | Notes |
|-------------|-----------|-------|
| `"benefit"` | `sum_{r,s} w[r, s] * 1e6 * value[r, s]` (USD/yr) | values from `demand.value_usd_per_m3` or `values=` overrides |
| `"equity"` | `t` with `t <= supply_ratio_r` for every riparian with demand (epigraph of the minimum supply ratio, max-min / Rawlsian fairness, Bertsimas, Farias & Trichakis 2011); among max-min optima the benefit-maximal one is selected (second LP with `min_supply_ratio = t* - 1e-7`) | lexicographic |
| `"weighted"` | `(1 - equity_weight) * benefit / max_benefit + equity_weight * t`, `max_benefit` from the benefit LP under the same constraints | `equity_weight = 1` reduces to `"equity"` |

The result reports `status`, `objective`, `allocations`, `outflows`,
`total_benefit_usd`, `min_supply_ratio`, `gini` (of supply ratios),
`message`, plus inflows, natural flows, consumption, demands, supply ratios,
benefits, environmental-flow checks, entitlements and values; infeasibility
never raises (`status == "infeasible"` with a diagnostic message).

`pareto_front(basin, flow_factor=1.0, points=11, start=0.0, ...)` traces the
benefit-equity trade-off by the epsilon-constraint method (Haimes, Lasdon &
Wismer 1971; Cohon 1978): the minimum supply ratio is swept over `points`
levels from `start` to the max-min optimum `t*`, the benefit LP is solved at
each level, and each point records `min_supply_ratio` (the level),
`total_benefit_usd`, `gini`, `achieved_min_supply_ratio`,
`mean_supply_ratio`, `total_withdrawal_mm3`, `outflow_to_sea_mm3`. The
"price of fairness" is the benefit foregone between the first and last point.

`route_allocation(basin, result)` replays an optimal allocation through
`route_basin` (LP withdrawals imposed as river demands, zero storage, no
refill, same entitlements and environmental flows) to obtain a `BasinBalance`
with consumption, return flows and the mass balance.

---

## 10. Scenario library (`wefnexus.scenarios`)

Each factory returns a validated `Scenario` with a descriptive `description`;
extra keyword arguments override other `Scenario` fields (e.g.
`stochastic=True, seed=3`). Defaults are the values of `ARCHITECTURE.md`
section 9 (scenario-analysis practice after Moss et al. 2010; O'Neill et al.
2014; Mahmoud et al. 2009).

| Factory | Drivers set | Defaults |
|---------|-------------|----------|
| `baseline(years=25)` | everything constant, cooperation under the treaty | `allocation_rule="treaty"` |
| `climate_change(years, flow_change_pct=-20, droughts=(8, 15, 22), drought_severity=0.4)` | linear flow change by the final year, drought years | treaty |
| `growth(years, population_growth=0.015, demand_growth=0.02, energy_growth=0.03, irrigated_area_change_pct=20, gdp_growth=None)` | compound growth of population, municipal/industrial/energy-sector water demand, electricity demand; linear irrigated-area change; GDP growth defaults to `demand_growth` | treaty |
| `efficiency(years, irrigation_efficiency_target=0.7, renewable_share_target=0.5)` | linear move of irrigation efficiency and renewable share to targets | treaty |
| `unilateral(years)` | `cooperation=False` | `allocation_rule="upstream_priority"` |
| `cooperative(years, rule="talmud")` | `cooperation=True`, bankruptcy rule | Talmud |
| `combined_stress(years, rule="treaty", climate_kw=None, growth_kw=None)` | climate change + growth | treaty |
| `combined_adaptation(years, rule="talmud", climate_kw=None, growth_kw=None, efficiency_kw=None)` | climate change + growth + efficiency | Talmud (`DEFAULT_ADAPTATION_RULE`, see note) |

Note on `combined_adaptation`: as in `ARCHITECTURE.md`, the adaptation
package is paired with the Talmud rule by default; `rule="treaty"` gives the
fixed-treaty variant. Because `NexusModel` applies a bankruptcy rule on a
consumptive-use basis with the carried reservoir storage in the estate
(section 8), the caps only bind when the basin is physically short; in the
library scenarios on the stylised basin they never do, so the Talmud variant
leaves every riparian at its demand, whereas the fixed treaty holds the
downstream riparian slightly below its demand (16 000 Mm3/yr entitlement
against a 16 300 Mm3/yr river demand).

`run_scenarios(basin, scenarios, **model_kw)` runs each scenario on
`basin.copy()`; `comparison_table(results)` produces one row per (scenario,
riparian) plus a `BASIN` row with the columns `TABLE_COLUMNS`
(`mean_supply_ratio`, `min_supply_ratio`, `supply_reliability`,
`mean_nexus_index`, `min_nexus_index`, `mean_water_security`,
`mean_energy_security`, `mean_food_security`, `years_env_flow_unmet`,
`env_flow_met_share`, `total_hydropower_gwh`, `mean_energy_deficit_gwh`,
`mean_emissions_t`, `food_self_sufficiency`, `final_storage_mm3`,
`mean_total_deficit_mm3`, `mean_water_stress_sdg642`; basin-only
`equity_index`, `natural_flow_mm3`, `outflow_to_sea_mm3`,
`mass_balance_error_mm3`). `scenario_differences(rows, reference="baseline")`
subtracts the reference scenario, `rank_scenarios(rows, column, riparian)`
orders scenarios, `format_table(rows, columns)` renders text and
`write_table_csv(rows, path)` writes CSV. `validate_scenario(scenario)` checks
ranges and canonicalises rule names.

---

## 11. Integration modules

- **`wefnexus.io`** - `basin_to_dict` / `basin_from_dict`, `basin_to_json` /
  `load_basin`, `scenario_to_dict` / `scenario_from_dict`, `scenario_to_json`
  / `load_scenario`. Every dataclass field is written (defaults included);
  sector-keyed dictionaries use the sector values as keys; `None` <-> `null`;
  non-finite numbers and unknown keys raise `ValueError`; documents carry
  `"format"` (`wefnexus.basin` / `wefnexus.scenario`) and `"format_version"`
  tags; `basin_from_dict(basin_to_dict(b)) == b`.
- **`wefnexus.viz`** (lazy matplotlib / networkx) - `plot_supply_ratio`,
  `plot_nexus_indices`, `plot_water_balance`, `plot_allocation_rules`,
  `plot_pareto`, `plot_nexus_radar`, `nexus_graph` (a `networkx.DiGraph` of
  headwater -> riparians -> sea with sector and source nodes, edge weights in
  Mm3/yr) and `plot_nexus_graph`. Every plot returns the `Figure`, never calls
  `plt.show()`, accepts `save=path` and uses one fixed palette (riparian `i`
  always has colour `i`).
- **`wefnexus.cli`** - `python -m wefnexus run | allocate | negotiate |
  compare | pareto | report | export-basin`; `main(argv)` returns the exit
  code (0 success, 1 no result, 2 usage error). See the README for every
  option.

---

## 12. References

- Allan, J. A. (1998). Virtual water: a strategic resource. Global solutions to regional deficits. *Ground Water* 36(4), 545-546.
- Allen, R. G., Pereira, L. S., Raes, D. & Smith, M. (1998). *Crop Evapotranspiration: Guidelines for Computing Crop Water Requirements.* FAO Irrigation and Drainage Paper 56. FAO, Rome.
- Ansink, E. & Weikard, H.-P. (2012). Sequential sharing rules for river sharing problems. *Social Choice and Welfare* 38, 187-210.
- Aumann, R. J. & Maschler, M. (1985). Game theoretic analysis of a bankruptcy problem from the Talmud. *Journal of Economic Theory* 36(2), 195-213.
- Azar, E. E. (1980). The Conflict and Peace Data Bank (COPDAB) project. *Journal of Conflict Resolution* 24(1), 143-152.
- Bazilian, M. et al. (2011). Considering the energy, water and food nexus: towards an integrated modelling approach. *Energy Policy* 39(12), 7896-7906.
- Bertsimas, D., Farias, V. F. & Trichakis, N. (2011). The price of fairness. *Operations Research* 59(1), 17-31.
- Bizikova, L., Roy, D., Swanson, D., Venema, H. D. & McCandless, M. (2013). *The Water-Energy-Food Security Nexus: Towards a Practical Planning and Decision-Support Framework for Landscape Investment and Risk Management.* IISD, Winnipeg.
- Brouwer, C., Prins, K. & Heibloem, M. (1989). *Irrigation Water Management: Irrigation Scheduling.* Training Manual 4. FAO, Rome.
- Cascao, A. E. & Zeitoun, M. (2010). Power, hegemony and critical hydropolitics. In: Earle, A., Jagerskog, A. & Ojendal, J. (eds), *Transboundary Water Management: Principles and Practice*, 27-42. Earthscan, London.
- Cohon, J. L. (1978). *Multiobjective Programming and Planning.* Academic Press, New York.
- Cullis, J. & van Koppen, B. (2007). *Applying the Gini Coefficient to Measure Inequality of Water Use in the Olifants River Water Management Area, South Africa.* IWMI Research Report 113. IWMI, Colombo.
- Curiel, I. J., Maschler, M. & Tijs, S. H. (1987). Bankruptcy games. *Zeitschrift fur Operations Research* 31, A143-A159.
- Daher, B. T. & Mohtar, R. H. (2015). Water-energy-food (WEF) Nexus Tool 2.0: guiding integrative resource planning and decision-making. *Water International* 40(5-6), 748-771.
- De Stefano, L., Duncan, J., Dinar, S., Stahl, K., Strzepek, K. M. & Wolf, A. T. (2012). Climate change and the institutional resilience of international river basins. *Journal of Peace Research* 49(1), 193-209.
- Dinar, A., Ratner, A. & Yaron, D. (1992). Evaluating cooperative game theory in water resources. *Theory and Decision* 32, 1-20.
- Dinar, S., Katz, D., De Stefano, L. & Blankespoor, B. (2015). Climate change, conflict, and cooperation: global analysis of the effectiveness of international river treaties in addressing water variability. *Political Geography* 45, 55-66.
- Doorenbos, J. & Kassam, A. H. (1979). *Yield Response to Water.* FAO Irrigation and Drainage Paper 33. FAO, Rome.
- Drieschova, A., Giordano, M. & Fischhendler, I. (2008). Governance mechanisms to address flow variability in water treaties. *Global Environmental Change* 18(2), 285-295.
- Ebert, U. & Welsch, H. (2004). Meaningful environmental indices: a social choice approach. *Journal of Environmental Economics and Management* 47(2), 270-283.
- Falkenmark, M., Lundqvist, J. & Widstrand, C. (1989). Macro-scale water scarcity requires micro-scale approaches: aspects of vulnerability in semi-arid development. *Natural Resources Forum* 13(4), 258-267.
- FAO (1996). *Rome Declaration on World Food Security and World Food Summit Plan of Action.* FAO, Rome.
- FAO (2001). *Food Balance Sheets: A Handbook.* FAO, Rome.
- FAO (2011). *Energy-Smart Food for People and Climate.* Issue paper. FAO, Rome.
- FAO (2018). *Progress on Level of Water Stress - Global Baseline for SDG Indicator 6.4.2.* FAO/UN-Water, Rome.
- FAO (2018b). *Progress on Water-Use Efficiency - Global Baseline for SDG Indicator 6.4.1.* FAO/UN-Water, Rome.
- Fisher, R. & Ury, W. (1981). *Getting to Yes: Negotiating Agreement Without Giving In.* Houghton Mifflin, Boston.
- Foley, D. K. (1967). Resource allocation and the public sector. *Yale Economic Essays* 7(1), 45-98.
- Gillies, D. B. (1959). Solutions to general non-zero-sum games. In: Tucker, A. W. & Luce, R. D. (eds), *Contributions to the Theory of Games IV*, 47-85. Princeton University Press.
- Gini, C. (1912). *Variabilita e mutabilita.* Cuppini, Bologna.
- Gleeson, T., Wada, Y., Bierkens, M. F. P. & van Beek, L. P. H. (2012). Water balance of global aquifers revealed by groundwater footprint. *Nature* 488, 197-200.
- Guajardo, M. & Jornsten, K. (2015). Common mistakes in computing the nucleolus. *European Journal of Operational Research* 241(3), 931-935.
- Gulliver, J. S. & Arndt, R. E. A. (1991). *Hydropower Engineering Handbook.* McGraw-Hill, New York.
- Haimes, Y. Y., Lasdon, L. S. & Wismer, D. A. (1971). On a bicriterion formulation of the problems of integrated system identification and system optimization. *IEEE Transactions on Systems, Man, and Cybernetics* 1(3), 296-297.
- Harou, J. J., Pulido-Velazquez, M., Rosenberg, D. E., Medellin-Azuara, J., Lund, J. R. & Howitt, R. E. (2009). Hydro-economic models: concepts, design, applications, and future prospects. *Journal of Hydrology* 375(3-4), 627-643.
- Hashimoto, T., Stedinger, J. R. & Loucks, D. P. (1982). Reliability, resiliency, and vulnerability criteria for water resource system performance evaluation. *Water Resources Research* 18(1), 14-20.
- Hoekstra, A. Y. & Chapagain, A. K. (2008). *Globalization of Water: Sharing the Planet's Freshwater Resources.* Blackwell, Oxford.
- Hoff, H. (2011). *Understanding the Nexus.* Background paper for the Bonn 2011 Conference: The Water, Energy and Food Security Nexus. SEI, Stockholm.
- IPCC (2014). *Climate Change 2014: Mitigation of Climate Change.* Annex III: Technology-specific cost and performance parameters. Cambridge University Press.
- Kohlberg, E. (1971). On the nucleolus of a characteristic function game. *SIAM Journal on Applied Mathematics* 20(1), 62-66.
- Kruyt, B., van Vuuren, D. P., de Vries, H. J. M. & Groenenberg, H. (2009). Indicators for energy security. *Energy Policy* 37(6), 2166-2181.
- Kumar, A. et al. (2011). Hydropower. In: *IPCC Special Report on Renewable Energy Sources and Climate Change Mitigation*, Ch. 5. Cambridge University Press.
- Loucks, D. P. & van Beek, E. (2005). *Water Resources Systems Planning and Management: An Introduction to Methods, Models and Applications.* UNESCO, Paris (2nd ed. Springer, 2017).
- Macknick, J., Newmark, R., Heath, G. & Hallett, K. C. (2012). Operational water consumption and withdrawal factors for electricity generating technologies: a review of existing literature. *Environmental Research Letters* 7, 045802.
- Madani, K. (2010). Game theory and water resources. *Journal of Hydrology* 381(3-4), 225-238.
- Mahmoud, M. et al. (2009). A formal framework for scenario development in support of environmental decision-making. *Environmental Modelling & Software* 24(7), 798-808.
- Maschler, M., Peleg, B. & Shapley, L. S. (1979). Geometric properties of the kernel, nucleolus, and related solution concepts. *Mathematics of Operations Research* 4(4), 303-338.
- Mekonnen, M. M. & Hoekstra, A. Y. (2011). The green, blue and grey water footprint of crops and derived crop products. *Hydrology and Earth System Sciences* 15, 1577-1600.
- Mianabadi, H., Mostert, E., Zarghami, M. & van de Giesen, N. (2014). A new bankruptcy method for conflict resolution in water resources allocation. *Journal of Environmental Management* 144, 152-159.
- Mirumachi, N. & Allan, J. A. (2007). Revisiting transboundary water governance: power, conflict, cooperation and the political economy. *Proceedings of the International Conference on Adaptive and Integrated Water Management (CAIWA)*, Basel.
- Mirumachi, N. (2015). *Transboundary Water Politics in the Developing World.* Routledge, London.
- Molden, D. et al. (2010). Improving agricultural water productivity: between optimism and caution. *Agricultural Water Management* 97(4), 528-535.
- Moss, R. H. et al. (2010). The next generation of scenarios for climate change research and assessment. *Nature* 463, 747-756.
- Nardo, M., Saisana, M., Saltelli, A., Tarantola, S., Hoffman, A. & Giovannini, E. (2005). *Handbook on Constructing Composite Indicators: Methodology and User Guide.* OECD Statistics Working Paper 2005/3. OECD, Paris.
- Nash, J. F. (1950). The bargaining problem. *Econometrica* 18(2), 155-162.
- OECD/JRC (2008). *Handbook on Constructing Composite Indicators: Methodology and User Guide.* OECD, Paris.
- O'Neill, B. (1982). A problem of rights arbitration from the Talmud. *Mathematical Social Sciences* 2(4), 345-371.
- O'Neill, B. C. et al. (2014). A new scenario framework for climate change research: the concept of shared socioeconomic pathways. *Climatic Change* 122, 387-400.
- Pelletier, N. et al. (2011). Energy intensity of agriculture and food systems. *Annual Review of Environment and Resources* 36, 223-246.
- Plappally, A. K. & Lienhard V, J. H. (2012). Energy requirements for water production, treatment, end use, reclamation, and disposal. *Renewable and Sustainable Energy Reviews* 16(7), 4818-4848.
- Raskin, P., Gleick, P., Kirshen, P., Pontius, G. & Strzepek, K. (1997). *Water Futures: Assessment of Long-range Patterns and Problems.* Stockholm Environment Institute, Stockholm.
- Sadoff, C. W. & Grey, D. (2002). Beyond the river: the benefits of cooperation on international rivers. *Water Policy* 4(5), 389-403.
- Sadoff, C. W. & Grey, D. (2005). Cooperation on international rivers: a continuum for securing and sharing benefits. *Water International* 30(4), 420-427.
- Schmeidler, D. (1969). The nucleolus of a characteristic function game. *SIAM Journal on Applied Mathematics* 17(6), 1163-1170.
- Sen, A. (1973). *On Economic Inequality.* Clarendon Press, Oxford.
- Shapley, L. S. (1953). A value for n-person games. In: Kuhn, H. W. & Tucker, A. W. (eds), *Contributions to the Theory of Games II*, 307-317. Princeton University Press.
- Shapley, L. S. (1971). Cores of convex games. *International Journal of Game Theory* 1, 11-26.
- Simpson, G. B. et al. (2022). The Water-Energy-Food Nexus Index: a tool to support sustainable development. *Sustainability* 14(1), 45.
- Sovacool, B. K. & Mukherjee, I. (2011). Conceptualizing and measuring energy security: a synthesized approach. *Energy* 36(8), 5343-5355.
- Stedinger, J. R. (1980). Fitting log normal distributions to hydrologic data. *Water Resources Research* 16(3), 481-490.
- Steduto, P., Hsiao, T. C., Fereres, E. & Raes, D. (2012). *Crop Yield Response to Water.* FAO Irrigation and Drainage Paper 66. FAO, Rome.
- Sullivan, C. (2002). Calculating a water poverty index. *World Development* 30(7), 1195-1210.
- Thomson, W. (2003). Axiomatic and game-theoretic analysis of bankruptcy and taxation problems: a survey. *Mathematical Social Sciences* 45(3), 249-297.
- Thomson, W. (2015). Axiomatic and game-theoretic analysis of bankruptcy and taxation problems: an update. *Mathematical Social Sciences* 74, 41-59.
- UN (1997). *Convention on the Law of the Non-navigational Uses of International Watercourses.* United Nations, New York.
- UNDP (2010). *Human Development Report 2010: The Real Wealth of Nations.* Technical note 1. UNDP, New York.
- UN-Water (2018). *Progress on Transboundary Water Cooperation - Global Baseline for SDG Indicator 6.5.2.* UNECE/UNESCO, Paris.
- Voutchkov, N. (2018). Energy use for membrane seawater desalination - current status and trends. *Desalination* 431, 2-14.
- Wada, Y., van Beek, L. P. H., van Kempen, C. M., Reckman, J. W. T. M., Vasak, S. & Bierkens, M. F. P. (2010). Global depletion of groundwater resources. *Geophysical Research Letters* 37, L20402.
- Warner, J. (2004). Plugging the GAP - working with Buzan: the Ilisu Dam as a security issue. *SOAS Water Issues Study Group Occasional Paper* 67.
- Wolf, A. T., Yoffe, S. B. & Giordano, M. (2003). International waters: identifying basins at risk. *Water Policy* 5(1), 29-60.
- Wurbs, R. A. (1993). Reservoir-system simulation and optimization models. *Journal of Water Resources Planning and Management* 119(4), 455-472.
- Yates, D., Sieber, J., Purkey, D. & Huber-Lee, A. (2005). WEAP21 - a demand-, priority-, and preference-driven water planning model. *Water International* 30(4), 487-500.
- Yoffe, S., Wolf, A. T. & Giordano, M. (2003). Conflict and cooperation over international freshwater resources: indicators of basins at risk. *Journal of the American Water Resources Association* 39(5), 1109-1126.
- Young, H. P. (1994). *Equity: In Theory and Practice.* Princeton University Press, Princeton.
- Zeitoun, M. & Warner, J. (2006). Hydro-hegemony - a framework for analysis of trans-boundary water conflicts. *Water Policy* 8(5), 435-460.
