"""Integrated multi-year water-energy-food (WEF) nexus simulation (composite module).

This module ties the leaf modules of :mod:`wefnexus` together into an
annual, upstream-to-downstream simulation of a transboundary river basin
over a :class:`~wefnexus.models.Scenario` horizon.  Every simulated year
follows the same six steps:

1. **Drivers** - population, GDP, municipal / industrial / energy water
   demand, electricity demand, irrigated area, irrigation efficiency and the
   renewable share are scaled by the scenario's year factors.  Agricultural
   water demand is the FAO-56 gross irrigation requirement of the riparian's
   crops (:func:`wefnexus.food.riparian_irrigation_requirement_mm3`, with the
   scenario's irrigated-area factor and irrigation efficiency) when crops are
   defined, otherwise ``WaterDemand.agricultural`` times the area factor.
2. **Allocation** - a sharing arrangement sets each riparian's cap on surface
   withdrawal (:meth:`NexusModel.entitlements`): fixed treaty volumes, or a
   bankruptcy rule of :mod:`wefnexus.allocation` dividing the water the
   basin can consume this year - the natural flow plus the carried reservoir
   storage, net of the environmental flows - among the riparians'
   *consumptive* river-water claims (Mianabadi et al. 2014; consumptive-use
   accounting as in the Colorado River Compact).  Claims and estate are on
   the same net basis, so a rule only rations when the basin is actually
   short, and each consumptive award is converted back into a cap on gross
   surface withdrawal (:meth:`NexusModel.claims`).  Without cooperation, or
   under ``"upstream_priority"``, there are no caps and the upstream riparian
   simply withdraws first (Ansink & Weikard 2012).
3. **Water** - :func:`wefnexus.water.route_basin` routes the year's
   (optionally stochastic) natural flow through the reaches, serving sectors
   in priority order, honouring the caps and the environmental flows and
   operating the reservoirs; end-of-year storages are carried into the next
   year (a WEAP-style annual water balance, Yates et al. 2005).
4. **Energy** - hydropower from the turbined share of the reach outflow
   (capped by installed capacity), thermal generation, *other* (non-hydro)
   renewables as ``EnergySystem.renewable_share`` of demand, and the
   electricity embedded in the water supply (surface-water pumping,
   desalination, treatment and wastewater from
   :func:`wefnexus.energy.energy_for_water`, plus the lifting of the
   groundwater actually abstracted) added to electricity demand.
   ``EnergySystem.renewable_share`` (and a scenario's
   ``renewable_share_target``) is the share of electricity demand met by
   renewables *other than* the hydropower modelled here - wind, solar,
   biomass, geothermal; hydropower is added on top, so a national statistic
   that already includes hydro must not be entered as is.  Each record
   reports both that input definition (``other_renewable_share_of_demand``)
   and the supply-based SDG 7.2 share including hydropower
   (``renewable_share``).
5. **Food** - FAO-33 crop production from the agricultural water actually
   delivered (surface withdrawal plus the agricultural share of groundwater
   and desalination), calorie self-sufficiency against the grown population.
6. **Indicators** - Falkenmark per-capita water, SDG 6.4.2 water stress,
   groundwater stress and the composite water / energy / food security and
   WEF nexus indices of :mod:`wefnexus.sustainability`.

The results are returned as a :class:`NexusResult` holding one
:class:`RiparianYear` record per riparian and year plus the routed
:class:`~wefnexus.water.BasinBalance` of every year, with helpers for time
series, basin aggregates, tabular export and a horizon summary.  Inputs are
never mutated.

Units
-----
Water volumes in **Mm3/yr** (1 Mm3 = 1e6 m3), per-capita water in
**m3/person/yr**, water stress in **percent**, energy in **GWh/yr**,
emissions in **t CO2/yr**, emission intensity in **t CO2/GWh**, food in
**t/yr** and **kcal/yr**, area in **ha**, money in **USD/yr**; indices are
dimensionless in ``[0, 1]``.

References
----------
Allen, R. G., Pereira, L. S., Raes, D. & Smith, M. (1998). *Crop
    Evapotranspiration.* FAO Irrigation and Drainage Paper 56, FAO, Rome.
Ansink, E. & Weikard, H.-P. (2012). Sequential sharing rules for river
    sharing problems. *Social Choice and Welfare* 38, 187-210.
Bazilian, M. et al. (2011). Considering the energy, water and food nexus:
    towards an integrated modelling approach. *Energy Policy* 39, 7896-7906.
Daher, B. T. & Mohtar, R. H. (2015). Water-energy-food (WEF) Nexus Tool 2.0:
    guiding integrative resource planning and decision-making. *Water
    International* 40(5-6), 748-771.
Doorenbos, J. & Kassam, A. H. (1979). *Yield Response to Water.* FAO
    Irrigation and Drainage Paper 33, FAO, Rome.
Falkenmark, M., Lundqvist, J. & Widstrand, C. (1989). Macro-scale water
    scarcity requires micro-scale approaches. *Natural Resources Forum*
    13(4), 258-267.
FAO (2018). *Progress on Level of Water Stress - Global Baseline for SDG
    Indicator 6.4.2.* FAO/UN-Water, Rome.
Hashimoto, T., Stedinger, J. R. & Loucks, D. P. (1982). Reliability,
    resiliency, and vulnerability criteria for water resource system
    performance evaluation. *Water Resources Research* 18(1), 14-20.
Hoff, H. (2011). *Understanding the Nexus.* Background paper for the Bonn
    2011 Conference: The Water, Energy and Food Security Nexus. SEI,
    Stockholm.
Loucks, D. P. & van Beek, E. (2017). *Water Resource Systems Planning and
    Management: An Introduction to Methods, Models, and Applications.*
    Springer, Cham.
Mianabadi, H., Mostert, E., Zarghami, M. & van de Giesen, N. (2014). A new
    bankruptcy method for conflict resolution in water resources allocation.
    *Journal of Environmental Management* 144, 152-159.
Sadoff, C. W. & Grey, D. (2002). Beyond the river: the benefits of
    cooperation on international rivers. *Water Policy* 4(5), 389-403.
Simpson, G. B. et al. (2022). The Water-Energy-Food Nexus Index: a tool to
    support sustainable development. *Sustainability* 14(1), 45.
Wolf, A. T., Yoffe, S. B. & Giordano, M. (2003). International waters:
    identifying basins at risk. *Water Policy* 5(1), 29-60.
Yates, D., Sieber, J., Purkey, D. & Huber-Lee, A. (2005). WEAP21 - a
    demand-, priority-, and preference-driven water planning model.
    *Water International* 30(4), 487-500.
"""
from __future__ import annotations

import csv
import math
import numbers
import os
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from wefnexus import energy as _energy
from wefnexus import food as _food
from wefnexus import sustainability as _sus
from wefnexus.allocation import RULE_ALIASES, RULES, apply_rule
from wefnexus.models import SECTOR_PRIORITY, Basin, Riparian, Scenario, Sector
from wefnexus.water import (
    BasinBalance,
    ReachResult,
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
)

__all__ = [
    "ALLOCATION_RULES",
    "BANKRUPTCY_RULES",
    "BASIN_KEY",
    "WITHDRAWAL_SECTORS",
    "RIPARIAN_YEAR_FIELDS",
    "SERIES_AGGREGATORS",
    "RiparianYear",
    "NexusResult",
    "NexusModel",
    "run_nexus",
]

#: Cubic metres per million cubic metres.
_M3_PER_MM3 = 1.0e6

#: The four withdrawal sectors, in service-priority order (no environment).
WITHDRAWAL_SECTORS: tuple = tuple(SECTOR_PRIORITY)

#: Bankruptcy (claims) rules of :mod:`wefnexus.allocation` that
#: :meth:`NexusModel.entitlements` can apply (``"upstream_priority"`` is a
#: registered rule too, but it means *no caps* here).
BANKRUPTCY_RULES: tuple = tuple(name for name in RULES if name != "upstream_priority")

#: Every allocation rule accepted by :class:`NexusModel` (aliases of
#: :data:`wefnexus.allocation.RULE_ALIASES` are accepted as well).
ALLOCATION_RULES: tuple = ("treaty", "upstream_priority") + BANKRUPTCY_RULES

#: Key of the basin-wide row in :meth:`NexusResult.summary`.
BASIN_KEY = "BASIN"

#: Aggregators accepted by :meth:`NexusResult.basin_series`.
SERIES_AGGREGATORS: Dict[str, Callable[[Sequence[float]], float]] = {
    "sum": lambda v: float(sum(v)),
    "mean": lambda v: float(sum(v) / len(v)),
    "min": lambda v: float(min(v)),
    "max": lambda v: float(max(v)),
    "any": lambda v: 1.0 if any(v) else 0.0,
    "all": lambda v: 1.0 if all(v) else 0.0,
}

#: Fields of :class:`RiparianYear` required by ARCHITECTURE.md section 7 (and
#: read by :func:`wefnexus.sustainability.assess`), in declaration order.
RIPARIAN_YEAR_FIELDS: tuple = (
    "name",
    "year",
    "population",
    "gdp_usd",
    "inflow_mm3",
    "upstream_inflow_mm3",
    "outflow_mm3",
    "storage_end_mm3",
    "entitlement_mm3",
    "demands",
    "withdrawals",
    "deficits",
    "environmental_flow_mm3",
    "env_flow_met",
    "per_capita_water_m3",
    "falkenmark",
    "water_stress_sdg642",
    "groundwater_stress",
    "hydropower_gwh",
    "thermal_gwh",
    "other_renewable_gwh",
    "energy_supply_gwh",
    "energy_demand_gwh",
    "energy_for_water_gwh",
    "energy_deficit_gwh",
    "emissions_t",
    "food_production_t",
    "food_kcal",
    "food_demand_kcal",
    "food_self_sufficiency",
    "et_ratio",
    "irrigation_requirement_mm3",
    "crop_value_usd",
    "water_value_usd",
    "water_security",
    "energy_security",
    "food_security",
    "nexus_index",
)


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
def _as_float(value: Any, label: str, *, allow_inf: bool = False) -> float:
    """Coerce ``value`` to a float; bools, NaN and non-numbers raise ``ValueError``."""
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a number, got {value!r}")
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a number, got {value!r}") from None
    if math.isnan(v):
        raise ValueError(f"{label} must not be NaN")
    if math.isinf(v) and not allow_inf:
        raise ValueError(f"{label} must be finite, got {v}")
    return v


def _nonneg(value: Any, label: str, *, allow_inf: bool = False) -> float:
    v = _as_float(value, label, allow_inf=allow_inf)
    if v < 0.0:
        raise ValueError(f"{label} must be >= 0, got {v}")
    return v


def _fraction(value: Any, label: str) -> float:
    v = _as_float(value, label)
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"{label} must lie in [0, 1], got {v}")
    return v


def _sector_dict(value: Any, label: str) -> Dict[Sector, float]:
    """Fresh ``{Sector: float}`` copy of a per-withdrawal-sector mapping.

    Keys may be :class:`~wefnexus.models.Sector` members or their string
    values; unknown keys and ``Sector.ENVIRONMENT`` (an in-stream flow, not
    a withdrawal) raise ``ValueError``.
    """
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a dict mapping Sector -> value, got {type(value).__name__}")
    out: Dict[Sector, float] = {}
    for key, v in value.items():
        try:
            sector = Sector(key)
        except ValueError:
            raise ValueError(f"{label}: unknown sector {key!r}") from None
        if sector not in WITHDRAWAL_SECTORS:
            raise ValueError(
                f"{label}: {sector.value!r} is not a withdrawal sector "
                "(environmental flow is an in-stream requirement)"
            )
        out[sector] = _as_float(v, f"{label}[{sector.value}]", allow_inf=True)
    return out


def _canonical_rule(rule: Any) -> str:
    """Map an allocation-rule name (or ``None``) to its canonical form.

    ``None`` means no sharing arrangement (upstream priority).  Rule names
    are case-insensitive and may use the aliases of
    :data:`wefnexus.allocation.RULE_ALIASES`.
    """
    if rule is None:
        return "upstream_priority"
    if not isinstance(rule, str):
        raise ValueError(f"allocation rule must be a string or None, got {type(rule).__name__}")
    key = rule.strip().lower().replace("-", "_").replace(" ", "_")
    if key == "treaty":
        return "treaty"
    key = RULE_ALIASES.get(key, key)
    if key not in RULES:
        raise ValueError(
            f"unknown allocation rule {rule!r}; expected one of {list(ALLOCATION_RULES)}"
        )
    return key


def _int_valued(value: Any, label: str) -> int:
    """Coerce an integral number (int or integral float, not bool) to ``int``."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{label} must be an integer, got {value!r}")
    v = float(value)
    if math.isnan(v) or math.isinf(v) or int(v) != v:
        raise ValueError(f"{label} must be an integer, got {value!r}")
    return int(v)


def _bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool) and value not in (0, 1):
        raise ValueError(f"{label} must be a bool, got {value!r}")
    return bool(value)


def _validate_scenario(scenario: Scenario) -> None:
    """Check the drivers of a :class:`~wefnexus.models.Scenario`.

    Mirrors the field checks of :func:`wefnexus.scenarios.validate_scenario`
    (that module imports this one, so it cannot be called from here) so that
    a bad scenario fails at model construction with a message naming the
    field instead of being silently clamped by ``Scenario.flow_factor``:

    * ``years`` a positive integer, ``start_year`` an integer, ``seed`` an
      integer ``>= 0``;
    * ``flow_change_pct_by_end`` and ``irrigated_area_change_pct_by_end``
      finite and ``>= -100`` (percent);
    * the four growth rates finite and ``> -1`` (fraction per year);
    * ``irrigation_efficiency_target`` ``None`` or in ``(0, 1]``,
      ``renewable_share_target`` ``None`` or in ``[0, 1]``;
    * ``drought_years`` an iterable of non-negative integers and
      ``drought_severity`` in ``[0, 1]``;
    * ``cooperation`` and ``stochastic`` booleans.

    Within these bounds ``Scenario.flow_factor(i)`` is non-negative in every
    year, so its ``max(trend, 0)`` guard never changes a value.

    Raises
    ------
    ValueError
        Naming the offending field.
    """
    label = "scenario"
    _int_valued(scenario.start_year, f"{label}.start_year")
    years = _int_valued(scenario.years, f"{label}.years")
    if years < 1:
        raise ValueError(f"{label}.years must be a positive integer, got {scenario.years!r}")
    for fld in ("flow_change_pct_by_end", "irrigated_area_change_pct_by_end"):
        v = _as_float(getattr(scenario, fld), f"{label}.{fld}")
        if v < -100.0:
            raise ValueError(f"{label}.{fld} must be >= -100 percent, got {v}")
    for fld in ("population_growth_rate", "gdp_growth_rate", "demand_growth_rate", "energy_demand_growth_rate"):
        v = _as_float(getattr(scenario, fld), f"{label}.{fld}")
        if v <= -1.0:
            raise ValueError(f"{label}.{fld} must be > -1 (fraction per year), got {v}")
    if scenario.irrigation_efficiency_target is not None:
        v = _as_float(scenario.irrigation_efficiency_target, f"{label}.irrigation_efficiency_target")
        if not 0.0 < v <= 1.0:
            raise ValueError(f"{label}.irrigation_efficiency_target must lie in (0, 1], got {v}")
    if scenario.renewable_share_target is not None:
        _fraction(scenario.renewable_share_target, f"{label}.renewable_share_target")
    _bool(scenario.cooperation, f"{label}.cooperation")
    _bool(scenario.stochastic, f"{label}.stochastic")
    if _int_valued(scenario.seed, f"{label}.seed") < 0:
        raise ValueError(f"{label}.seed must be >= 0, got {scenario.seed!r}")
    droughts = scenario.drought_years
    if droughts is None:
        droughts = []
    if isinstance(droughts, (str, bytes)) or not hasattr(droughts, "__iter__"):
        raise ValueError(
            f"{label}.drought_years must be an iterable of non-negative integers, got {droughts!r}"
        )
    for d in droughts:
        if _int_valued(d, f"{label}.drought_years entry") < 0:
            raise ValueError(f"{label}.drought_years entries must be >= 0, got {d!r}")
    _fraction(scenario.drought_severity, f"{label}.drought_severity")


def _mean(values: Sequence[float]) -> float:
    vals = list(values)
    if not vals:
        raise ValueError("cannot average an empty sequence")
    return float(sum(vals) / len(vals))


def _ratio(numerator: float, denominator: float, when_zero: float) -> float:
    return numerator / denominator if denominator > 0.0 else when_zero


def _supply_ratio(demand: float, deficit: float) -> float:
    if demand <= 0.0:
        return 1.0
    return min(max(1.0 - deficit / demand, 0.0), 1.0)


#: Relative tolerance below which a bankruptcy award counts as the full claim.
_CLAIM_TOL = 1e-9


def _river_demand(r: Riparian, demand: Mapping[Sector, float]) -> Dict[Sector, float]:
    """Demand placed on the river per withdrawal sector, Mm3/yr.

    The non-river supply (groundwater abstraction + desalination capacity,
    capped at the total demand) is deducted proportionally across sectors,
    exactly as :func:`wefnexus.water.route_basin` does; missing sectors count
    as 0.
    """
    label = f"riparian {r.name!r}"
    dem = {s: _nonneg(demand.get(s, 0.0), f"{label} demand[{s.value}]") for s in WITHDRAWAL_SECTORS}
    total = sum(dem.values())
    if total <= 0.0:
        return {s: 0.0 for s in WITHDRAWAL_SECTORS}
    non_river = _nonneg(r.groundwater_abstraction_mm3, f"{label} groundwater_abstraction_mm3") + _nonneg(
        r.energy.desalination_capacity_mm3, f"{label} energy.desalination_capacity_mm3"
    )
    share = 1.0 - min(non_river, total) / total
    return {s: max(v * share, 0.0) for s, v in dem.items()}


def _consumption_fractions(r: Riparian) -> Dict[Sector, float]:
    """Consumption fraction of every withdrawal sector (missing -> 0: fully returned)."""
    cf = r.demand.consumption_fraction
    return {
        s: _fraction(cf.get(s, 0.0), f"riparian {r.name!r} demand.consumption_fraction[{s.value}]")
        for s in WITHDRAWAL_SECTORS
    }


def _consumptive_claim(river_demand: Mapping[Sector, float], fractions: Mapping[Sector, float]) -> float:
    """``sum_s river_demand[s] * f_s``: the river water a riparian would consume, Mm3/yr."""
    return float(sum(river_demand[s] * fractions[s] for s in WITHDRAWAL_SECTORS))


def _withdrawal_cap_for_award(
    river_demand: Mapping[Sector, float], fractions: Mapping[Sector, float], award: float
) -> float:
    """Gross surface-withdrawal cap at which a riparian consumes exactly ``award``.

    Sectors are served in :data:`~wefnexus.models.SECTOR_PRIORITY` order, as
    :func:`wefnexus.water.route_basin` does: each sector withdraws its river
    demand until the consumptive award is used up
    (``w_s = min(rd_s, remaining / f_s)``); a sector with a zero consumption
    fraction never depletes the award and withdraws its full river demand.
    An award equal to the claim (within :data:`_CLAIM_TOL` relative) returns
    the total river demand, so the cap never binds in a year without
    scarcity.  The result lies in ``[0, sum(river_demand)]`` (Mm3/yr).
    """
    total = float(sum(river_demand[s] for s in WITHDRAWAL_SECTORS))
    claim = _consumptive_claim(river_demand, fractions)
    if award >= claim - _CLAIM_TOL * max(claim, 1.0):
        return total
    remaining = max(float(award), 0.0)
    cap = 0.0
    for s in WITHDRAWAL_SECTORS:
        rd = river_demand[s]
        f = fractions[s]
        w = rd if f <= 0.0 else max(min(rd, remaining / f), 0.0)
        cap += w
        remaining = max(remaining - w * f, 0.0)
    return float(min(cap, total))


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class RiparianYear:
    """State and indicators of one riparian in one simulated year.

    The first block of attributes is the contract of ARCHITECTURE.md
    section 7 (:data:`RIPARIAN_YEAR_FIELDS`); the attributes after it are
    additional diagnostics with defaults, so a record can be built from the
    contract fields alone.

    Attributes
    ----------
    name : str
        Riparian name.
    year : int
        Calendar year.
    population : float
        Population in the year (persons).
    gdp_usd : float
        GDP in the year (USD/yr).
    inflow_mm3, upstream_inflow_mm3, outflow_mm3 : float
        Reach inflow (upstream + local), its upstream part, and the outflow
        passed downstream (Mm3/yr), from :class:`~wefnexus.water.ReachResult`.
    storage_end_mm3 : float
        Reservoir storage at the end of the year (Mm3).
    entitlement_mm3 : float
        Cap on surface withdrawal applied this year (Mm3/yr; ``inf`` = none).
    demands, withdrawals, deficits : dict
        Per-sector gross demand, surface withdrawal and unmet demand (Mm3/yr)
        keyed by the four withdrawal :class:`~wefnexus.models.Sector` members.
    environmental_flow_mm3 : float
        In-stream flow requirement at the reach outlet (Mm3/yr).
    env_flow_met : bool
        Whether ``outflow_mm3 >= environmental_flow_mm3``.
    per_capita_water_m3 : float
        Renewable water (inflow + groundwater recharge) per person
        (m3/person/yr; ``inf`` when the population is zero).
    falkenmark : str
        Falkenmark class of ``per_capita_water_m3``.
    water_stress_sdg642 : float
        SDG 6.4.2 water stress, percent: ``100 * (surface withdrawal +
        groundwater used) / (inflow + recharge - environmental flow)``
        (``inf`` when no water is available beyond the environmental flow).
    groundwater_stress : float
        Groundwater used / recharge (``inf`` when there is use but no
        recharge).
    hydropower_gwh, thermal_gwh, other_renewable_gwh : float
        Generation by source (GWh/yr).
    energy_supply_gwh, energy_demand_gwh : float
        Total supply and total demand including ``energy_for_water_gwh``
        (GWh/yr).
    energy_for_water_gwh : float
        Electricity embedded in the water supply (surface-water pumping,
        desalination, treatment, wastewater and the lifting of the
        groundwater abstracted; GWh/yr).
    energy_deficit_gwh : float
        ``max(demand - supply, 0)`` (GWh/yr).
    emissions_t : float
        CO2 emissions of thermal generation (t/yr).
    food_production_t, food_kcal, food_demand_kcal : float
        Crop production (t/yr), its food energy and the population's dietary
        energy demand (kcal/yr).
    food_self_sufficiency : float
        ``food_kcal / food_demand_kcal`` (not capped; 1 when demand is 0).
    et_ratio : float
        Relative evapotranspiration ETa/ETm of the irrigated crops (the
        relative gross irrigation supply); for a riparian without crops the
        agricultural supply ratio.
    irrigation_requirement_mm3 : float
        Gross agricultural water demand of the year (Mm3/yr).
    crop_value_usd : float
        Farm-gate value of crop production (USD/yr).
    water_value_usd : float
        ``sum_s withdrawals[s] * value_usd_per_m3[s] * 1e6`` (USD/yr).
    water_security, energy_security, food_security, nexus_index : float
        Composite indices in ``[0, 1]`` (:mod:`wefnexus.sustainability`).
    storage_start_mm3, storage_release_mm3, storage_refill_mm3, evaporation_mm3 : float
        Reservoir operation of the year (Mm3), extra diagnostics.
    non_river_supply_mm3, groundwater_used_mm3, desalinated_mm3 : float
        Non-river supply applied to demand and its split (Mm3/yr).
    consumption : dict
        Per-sector consumptive use (Mm3/yr).
    flow_factor : float
        Natural-flow multiplier of the year (dimensionless).
    renewable_water_mm3 : float
        ``inflow_mm3 + groundwater recharge`` (Mm3/yr).
    agricultural_water_mm3 : float
        Water delivered to agriculture from all sources (Mm3/yr).
    cooling_water_mm3 : float
        Consumptive cooling water of thermal generation (Mm3/yr).
    renewable_share : float
        Supply-based renewable share *including* hydropower (SDG 7.2):
        ``(hydropower + other renewables) / supply`` (0 when supply is 0).
    emission_intensity_t_per_gwh : float
        ``emissions_t / energy_supply_gwh`` (0 when supply is 0).
    energy_supply_ratio : float
        ``min(supply / demand, 1)`` (1 when demand is 0).
    env_flow_compliance : float
        ``min(outflow / environmental flow, 1)`` (1 when the requirement
        is 0); the "environment" component of ``water_security``.
    irrigated_area_ha : float
        Irrigated area after the scenario's area factor (ha).
    irrigation_efficiency : float
        Irrigation efficiency used in the year (dimensionless).
    groundwater_pumping_gwh : float
        Electricity for lifting the groundwater actually abstracted,
        ``pumping_energy_gwh(groundwater_used_mm3, pumping_lift_m,
        pumping_efficiency)`` (GWh/yr); part of ``energy_for_water_gwh``.
    other_renewable_share_of_demand : float
        The non-hydro renewable share of electricity demand applied in the
        year (``EnergySystem.renewable_share``, or the scenario's
        interpolated ``renewable_share_target``), i.e.
        ``other_renewable_gwh / (energy_demand_gwh - energy_for_water_gwh)``;
        the input definition, as opposed to the supply-based
        ``renewable_share``.
    """

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
    # -- extra diagnostics (not part of the section-7 contract) ---------------
    storage_start_mm3: float = 0.0
    storage_release_mm3: float = 0.0
    storage_refill_mm3: float = 0.0
    evaporation_mm3: float = 0.0
    non_river_supply_mm3: float = 0.0
    groundwater_used_mm3: float = 0.0
    desalinated_mm3: float = 0.0
    consumption: Dict[Sector, float] = field(default_factory=dict)
    flow_factor: float = 1.0
    renewable_water_mm3: float = 0.0
    agricultural_water_mm3: float = 0.0
    cooling_water_mm3: float = 0.0
    renewable_share: float = 0.0
    emission_intensity_t_per_gwh: float = 0.0
    energy_supply_ratio: float = 1.0
    env_flow_compliance: float = 1.0
    irrigated_area_ha: float = 0.0
    irrigation_efficiency: float = 1.0
    groundwater_pumping_gwh: float = 0.0
    other_renewable_share_of_demand: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"name must be a non-empty string, got {self.name!r}")
        if (
            isinstance(self.year, bool)
            or not isinstance(self.year, numbers.Real)
            or math.isnan(float(self.year))
            or math.isinf(float(self.year))
            or int(self.year) != self.year
        ):
            raise ValueError(f"year must be an integer, got {self.year!r}")
        self.year = int(self.year)
        self.demands = _sector_dict(self.demands, "demands")
        self.withdrawals = _sector_dict(self.withdrawals, "withdrawals")
        self.deficits = _sector_dict(self.deficits, "deficits")
        self.consumption = _sector_dict(self.consumption, "consumption")
        self.env_flow_met = bool(self.env_flow_met)
        if not isinstance(self.falkenmark, str):
            raise ValueError(f"falkenmark must be a string, got {self.falkenmark!r}")

    # -- aggregates ---------------------------------------------------------
    def total_demand(self) -> float:
        """Total gross withdrawal demand, Mm3/yr."""
        return float(sum(self.demands.values()))

    def total_withdrawal(self) -> float:
        """Total surface withdrawal, Mm3/yr."""
        return float(sum(self.withdrawals.values()))

    def total_deficit(self) -> float:
        """Total unmet withdrawal demand, Mm3/yr."""
        return float(sum(self.deficits.values()))

    def total_consumption(self) -> float:
        """Total consumptive use, Mm3/yr (0 if not recorded)."""
        return float(sum(self.consumption.values()))

    def supply_ratio(self) -> float:
        """``1 - total_deficit / total_demand`` in ``[0, 1]`` (1 when demand is 0)."""
        return _supply_ratio(self.total_demand(), self.total_deficit())

    def to_dict(self) -> Dict[str, Any]:
        """Flat dictionary of the record.

        Scalar fields keep their names; the sector dictionaries are expanded
        to ``demand_<sector>``, ``withdrawal_<sector>``, ``deficit_<sector>``
        and ``consumption_<sector>`` for the four withdrawal sectors (missing
        sectors appear as 0), and the derived ``total_demand_mm3``,
        ``total_withdrawal_mm3``, ``total_deficit_mm3``,
        ``total_consumption_mm3`` and ``supply_ratio`` are appended.  Keys
        are the same for every record, so a list of these dictionaries is a
        rectangular table.
        """
        out: Dict[str, Any] = {}
        for fld in RIPARIAN_YEAR_FIELDS:
            if fld in ("demands", "withdrawals", "deficits"):
                continue
            out[fld] = getattr(self, fld)
        for prefix, data in (
            ("demand", self.demands),
            ("withdrawal", self.withdrawals),
            ("deficit", self.deficits),
            ("consumption", self.consumption),
        ):
            for s in WITHDRAWAL_SECTORS:
                out[f"{prefix}_{s.value}"] = float(data.get(s, 0.0))
        out["total_demand_mm3"] = self.total_demand()
        out["total_withdrawal_mm3"] = self.total_withdrawal()
        out["total_deficit_mm3"] = self.total_deficit()
        out["total_consumption_mm3"] = self.total_consumption()
        out["supply_ratio"] = self.supply_ratio()
        for fld in (
            "storage_start_mm3",
            "storage_release_mm3",
            "storage_refill_mm3",
            "evaporation_mm3",
            "non_river_supply_mm3",
            "groundwater_used_mm3",
            "desalinated_mm3",
            "flow_factor",
            "renewable_water_mm3",
            "agricultural_water_mm3",
            "cooling_water_mm3",
            "renewable_share",
            "emission_intensity_t_per_gwh",
            "energy_supply_ratio",
            "env_flow_compliance",
            "irrigated_area_ha",
            "irrigation_efficiency",
            "groundwater_pumping_gwh",
            "other_renewable_share_of_demand",
        ):
            out[fld] = getattr(self, fld)
        return out


def _record_value(rec: RiparianYear, fld: str) -> Any:
    """Value of a raw attribute or of a :meth:`RiparianYear.to_dict` key."""
    if not isinstance(fld, str):
        raise ValueError(f"field must be a string, got {type(fld).__name__}")
    if hasattr(rec, fld):
        value = getattr(rec, fld)
        if not callable(value):
            return value
    flat = rec.to_dict()
    if fld in flat:
        return flat[fld]
    raise ValueError(f"unknown field {fld!r}; available fields: {sorted(flat)}")


def _as_number(value: Any, label: str) -> float:
    """Numeric coercion for series aggregation (bools allowed, strings not)."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        v = float(value)
        if math.isnan(v):
            raise ValueError(f"{label} is NaN")
        return v
    raise ValueError(f"{label} is not numeric ({type(value).__name__}); choose a numeric field")


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class NexusResult:
    """Output of :meth:`NexusModel.run` / :func:`run_nexus`.

    Attributes
    ----------
    basin_name : str
        Name of the simulated basin.
    scenario : Scenario
        A copy of the scenario that drove the run.
    years : list of int
        Calendar years of the horizon, in order.
    records : list of RiparianYear
        ``len(years) * n_riparians`` records, year-major (all riparians of
        the first year in basin order, then the second year, ...).
    balances : list of BasinBalance
        The routed water balance of each year (same order as ``years``).
    flow_factors : list of float
        Natural-flow multiplier applied in each year.
    allocation_rule : str
        Canonical name of the sharing rule that was applied
        (``"upstream_priority"`` when there were no caps).
    """

    basin_name: str
    scenario: Scenario
    years: List[int]
    records: List[RiparianYear]
    balances: List[BasinBalance]
    flow_factors: List[float] = field(default_factory=list)
    allocation_rule: str = "upstream_priority"

    def __post_init__(self) -> None:
        self.years = [int(y) for y in self.years]
        self.records = list(self.records)
        self.balances = list(self.balances)
        self.flow_factors = [float(f) for f in self.flow_factors]
        for rec in self.records:
            if not isinstance(rec, RiparianYear):
                raise ValueError(f"records must be RiparianYear objects, got {type(rec).__name__}")
        if self.balances and len(self.balances) != len(self.years):
            raise ValueError(
                f"balances must be empty or hold one BasinBalance per year "
                f"({len(self.balances)} balances for {len(self.years)} years)"
            )
        if self.flow_factors and len(self.flow_factors) != len(self.years):
            raise ValueError("flow_factors must be empty or hold one factor per year")

    # -- access --------------------------------------------------------------
    def riparian_names(self) -> List[str]:
        """Riparian names in first-appearance (upstream -> downstream) order."""
        names: List[str] = []
        for rec in self.records:
            if rec.name not in names:
                names.append(rec.name)
        return names

    @property
    def n_years(self) -> int:
        """Number of simulated years."""
        return len(self.years)

    @property
    def n_riparians(self) -> int:
        """Number of riparians with records."""
        return len(self.riparian_names())

    def for_riparian(self, name: str) -> List[RiparianYear]:
        """Records of riparian ``name`` in year order (``KeyError`` if unknown)."""
        recs = [r for r in self.records if r.name == name]
        if not recs:
            raise KeyError(f"no riparian named {name!r}; result covers {self.riparian_names()}")
        return sorted(recs, key=lambda r: r.year)

    def for_year(self, year: int) -> List[RiparianYear]:
        """Records of calendar ``year`` in basin order (``KeyError`` if unknown)."""
        recs = [r for r in self.records if r.year == year]
        if not recs:
            raise KeyError(f"no records for year {year!r}; result covers {self.years}")
        return recs

    def record(self, name: str, year: int) -> RiparianYear:
        """The single record of riparian ``name`` in ``year`` (``KeyError`` if absent)."""
        for rec in self.for_riparian(name):
            if rec.year == year:
                return rec
        raise KeyError(f"no record for {name!r} in year {year!r}; result covers {self.years}")

    def balance(self, year: int) -> BasinBalance:
        """The :class:`~wefnexus.water.BasinBalance` of calendar ``year``."""
        if not self.balances:
            raise KeyError("this result carries no balances")
        try:
            idx = self.years.index(int(year))
        except ValueError:
            raise KeyError(f"no balance for year {year!r}; result covers {self.years}") from None
        return self.balances[idx]

    # -- series --------------------------------------------------------------
    def series(self, name: str, fld: str) -> List[Any]:
        """Time series of one field for one riparian.

        Parameters
        ----------
        name : str
            Riparian name.
        fld : str
            A :class:`RiparianYear` attribute (e.g. ``"nexus_index"``) or a
            key of :meth:`RiparianYear.to_dict` (e.g. ``"supply_ratio"``,
            ``"withdrawal_agricultural"``, ``"total_deficit_mm3"``).

        Returns
        -------
        list
            One value per year, in year order.

        Raises
        ------
        KeyError
            Unknown riparian.
        ValueError
            Unknown field.
        """
        return [_record_value(r, fld) for r in self.for_riparian(name)]

    def basin_series(self, fld: str, agg: str = "sum") -> List[float]:
        """Per-year aggregate of a numeric field across all riparians.

        Parameters
        ----------
        fld : str
            Field name as in :meth:`series`; must be numeric (bools count as
            0/1, so ``basin_series("env_flow_met", "all")`` is 1 in years
            when every reach met its environmental flow).
        agg : str, optional
            One of :data:`SERIES_AGGREGATORS`: ``"sum"`` (default),
            ``"mean"``, ``"min"``, ``"max"``, ``"any"``, ``"all"``.

        Returns
        -------
        list of float
            One value per year, in ``years`` order.

        Raises
        ------
        ValueError
            Unknown field, non-numeric field or unknown aggregator.
        """
        if not isinstance(agg, str) or agg not in SERIES_AGGREGATORS:
            raise ValueError(f"agg must be one of {sorted(SERIES_AGGREGATORS)}, got {agg!r}")
        fn = SERIES_AGGREGATORS[agg]
        out: List[float] = []
        for year in self.years:
            vals = [_as_number(_record_value(r, fld), f"{r.name}[{fld}]") for r in self.for_year(year)]
            out.append(fn(vals))
        return out

    # -- export --------------------------------------------------------------
    def to_records(self) -> List[Dict[str, Any]]:
        """Flat dictionaries (:meth:`RiparianYear.to_dict`) of all records."""
        return [r.to_dict() for r in self.records]

    def to_dataframe(self):
        """All records as a :class:`pandas.DataFrame` (one row per riparian-year).

        pandas is imported lazily; ``ImportError`` if it is not installed.
        """
        import pandas as pd  # lazy: optional dependency

        return pd.DataFrame(self.to_records())

    def to_csv(self, path: Any) -> str:
        """Write all records to a CSV file (standard library, no pandas).

        Parameters
        ----------
        path : str or path-like
            Destination file; overwritten if it exists.

        Returns
        -------
        str
            The path written.  Columns are the keys of
            :meth:`RiparianYear.to_dict`; sector keys are expanded.
        """
        target = os.fspath(path)
        rows = self.to_records()
        fieldnames = list(rows[0]) if rows else list(RIPARIAN_YEAR_FIELDS)
        with open(target, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return target

    # -- summary -------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        """Horizon summary per riparian plus a basin-wide row.

        Returns
        -------
        dict
            ``{riparian: {...}, "BASIN": {...}}`` (:data:`BASIN_KEY`).  Each
            riparian dictionary holds means over the horizon of the four
            indices (``water_security``, ``energy_security``,
            ``food_security``, ``nexus_index``) and ``min_nexus_index``;
            ``supply_ratio`` (mean), ``min_supply_ratio``,
            ``supply_reliability`` / ``resilience`` / ``vulnerability``
            (Hashimoto et al. 1982); mean ``total_demand_mm3``,
            ``total_withdrawal_mm3``, ``total_deficit_mm3``, ``inflow_mm3``,
            ``outflow_mm3``, ``storage_end_mm3`` and ``final_storage_mm3``;
            ``env_flow_met_share`` and ``years_env_flow_unmet``;
            ``entitlement_mm3`` (mean, ``inf`` if never capped);
            ``water_stress_sdg642``, ``groundwater_stress``,
            ``per_capita_water_m3`` (means) and ``falkenmark`` (class of the
            mean); energy means (``hydropower_gwh``, ``thermal_gwh``,
            ``other_renewable_gwh``, ``energy_supply_gwh``,
            ``energy_demand_gwh``, ``energy_for_water_gwh``,
            ``energy_deficit_gwh``, ``emissions_t``,
            ``groundwater_pumping_gwh``, ``other_renewable_share_of_demand``),
            ``total_hydropower_gwh`` (sum), ``energy_self_sufficiency`` and
            ``renewable_share`` (ratios of horizon totals, hydropower
            included); food means
            (``food_production_t``, ``food_kcal``, ``food_demand_kcal``,
            ``et_ratio``, ``irrigation_requirement_mm3``, ``crop_value_usd``,
            ``water_value_usd``) and ``food_self_sufficiency`` (ratio of
            horizon totals); ``population`` and ``gdp_usd`` (means) and
            ``years``.  The basin row holds the riparian means of the four
            indices, ``nexus_index_population_weighted``, ``equity_index``
            (``1 - Gini`` of the riparian supply ratios), the basin
            ``supply_ratio`` (from total deficit / total demand),
            ``min_supply_ratio`` and ``supply_reliability`` over years, the
            totals above summed over riparians, ``env_flow_met_share`` over
            all riparian-years, ``years_env_flow_unmet`` (years with any
            unmet reach) and, from the balances, mean ``natural_flow_mm3``,
            mean ``outflow_to_sea_mm3`` and the maximum
            ``mass_balance_error_mm3`` (``None`` without balances).

        Raises
        ------
        ValueError
            On an empty result.
        """
        names = self.riparian_names()
        if not names:
            raise ValueError("cannot summarise a result without records")
        out: Dict[str, Any] = {}
        for name in names:
            out[name] = _summarise_riparian(self.for_riparian(name))
        out[BASIN_KEY] = _summarise_basin(self, {n: out[n] for n in names})
        return out


def _summarise_riparian(recs: List[RiparianYear]) -> Dict[str, Any]:
    n = len(recs)
    demand = [r.total_demand() for r in recs]
    deficit = [r.total_deficit() for r in recs]
    withdrawal = [r.total_withdrawal() for r in recs]
    supplied = [max(d - df, 0.0) for d, df in zip(demand, deficit)]
    ratios = [r.supply_ratio() for r in recs]
    nexus = [r.nexus_index for r in recs]
    hydro = [r.hydropower_gwh for r in recs]
    other = [r.other_renewable_gwh for r in recs]
    e_supply = [r.energy_supply_gwh for r in recs]
    e_demand = [r.energy_demand_gwh for r in recs]
    kcal = [r.food_kcal for r in recs]
    kcal_demand = [r.food_demand_kcal for r in recs]
    env_met = [bool(r.env_flow_met) for r in recs]
    mean_pc = _mean([r.per_capita_water_m3 for r in recs])
    total_e_supply = sum(e_supply)

    def mean_of(attr: str) -> float:
        return _mean([float(getattr(r, attr)) for r in recs])

    return {
        "years": n,
        "first_year": recs[0].year,
        "last_year": recs[-1].year,
        "population": mean_of("population"),
        "gdp_usd": mean_of("gdp_usd"),
        # indices
        "water_security": mean_of("water_security"),
        "energy_security": mean_of("energy_security"),
        "food_security": mean_of("food_security"),
        "nexus_index": _mean(nexus),
        "min_nexus_index": min(nexus),
        "nexus_index_trend": nexus[-1] - nexus[0],
        # water
        "supply_ratio": _mean(ratios),
        "min_supply_ratio": min(ratios),
        "supply_reliability": supply_reliability(supplied, demand),
        "resilience": resilience(supplied, demand),
        "vulnerability": vulnerability(supplied, demand),
        "total_demand_mm3": _mean(demand),
        "total_withdrawal_mm3": _mean(withdrawal),
        "total_deficit_mm3": _mean(deficit),
        "inflow_mm3": mean_of("inflow_mm3"),
        "upstream_inflow_mm3": mean_of("upstream_inflow_mm3"),
        "outflow_mm3": mean_of("outflow_mm3"),
        "storage_end_mm3": mean_of("storage_end_mm3"),
        "final_storage_mm3": float(recs[-1].storage_end_mm3),
        "environmental_flow_mm3": mean_of("environmental_flow_mm3"),
        "env_flow_met_share": sum(1.0 for m in env_met if m) / n,
        "years_env_flow_unmet": sum(1 for m in env_met if not m),
        "entitlement_mm3": _mean([float(r.entitlement_mm3) for r in recs]),
        "water_stress_sdg642": mean_of("water_stress_sdg642"),
        "groundwater_stress": mean_of("groundwater_stress"),
        "per_capita_water_m3": mean_pc,
        "falkenmark": falkenmark_category(max(mean_pc, 0.0)),
        "water_value_usd": mean_of("water_value_usd"),
        # energy
        "hydropower_gwh": _mean(hydro),
        "total_hydropower_gwh": float(sum(hydro)),
        "thermal_gwh": mean_of("thermal_gwh"),
        "other_renewable_gwh": _mean(other),
        "energy_supply_gwh": _mean(e_supply),
        "energy_demand_gwh": _mean(e_demand),
        "energy_for_water_gwh": mean_of("energy_for_water_gwh"),
        "energy_deficit_gwh": mean_of("energy_deficit_gwh"),
        "emissions_t": mean_of("emissions_t"),
        "energy_self_sufficiency": _ratio(total_e_supply, sum(e_demand), 1.0),
        "renewable_share": min(max(_ratio(sum(hydro) + sum(other), total_e_supply, 0.0), 0.0), 1.0),
        "groundwater_pumping_gwh": mean_of("groundwater_pumping_gwh"),
        "other_renewable_share_of_demand": mean_of("other_renewable_share_of_demand"),
        # food
        "food_production_t": mean_of("food_production_t"),
        "food_kcal": _mean(kcal),
        "food_demand_kcal": _mean(kcal_demand),
        "food_self_sufficiency": _ratio(sum(kcal), sum(kcal_demand), 1.0),
        "et_ratio": mean_of("et_ratio"),
        "irrigation_requirement_mm3": mean_of("irrigation_requirement_mm3"),
        "crop_value_usd": mean_of("crop_value_usd"),
    }


def _summarise_basin(result: NexusResult, riparians: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    names = list(riparians)
    by_year = [result.for_year(y) for y in result.years]
    demand_y = [sum(r.total_demand() for r in recs) for recs in by_year]
    deficit_y = [sum(r.total_deficit() for r in recs) for recs in by_year]
    withdrawal_y = [sum(r.total_withdrawal() for r in recs) for recs in by_year]
    supplied_y = [max(d - df, 0.0) for d, df in zip(demand_y, deficit_y)]
    ratio_y = [_supply_ratio(d, df) for d, df in zip(demand_y, deficit_y)]
    pop_y = [sum(r.population for r in recs) for recs in by_year]
    e_supply_y = [sum(r.energy_supply_gwh for r in recs) for recs in by_year]
    e_demand_y = [sum(r.energy_demand_gwh for r in recs) for recs in by_year]
    renew_y = [sum(r.hydropower_gwh + r.other_renewable_gwh for r in recs) for recs in by_year]
    hydro_y = [sum(r.hydropower_gwh for r in recs) for recs in by_year]
    emissions_y = [sum(r.emissions_t for r in recs) for recs in by_year]
    kcal_y = [sum(r.food_kcal for r in recs) for recs in by_year]
    kcal_demand_y = [sum(r.food_demand_kcal for r in recs) for recs in by_year]
    env_flags = [bool(r.env_flow_met) for recs in by_year for r in recs]
    total_e_supply = sum(e_supply_y)

    mean_pop = {n: riparians[n]["population"] for n in names}
    total_pop = sum(mean_pop.values())

    def rip_mean(key: str) -> float:
        return _mean([riparians[n][key] for n in names])

    nexus_by_rip = {n: riparians[n]["nexus_index"] for n in names}
    if total_pop > 0.0:
        pop_weighted = sum(nexus_by_rip[n] * mean_pop[n] for n in names) / total_pop
    else:
        pop_weighted = rip_mean("nexus_index")

    out: Dict[str, Any] = {
        "riparians": len(names),
        "years": len(result.years),
        "first_year": result.years[0] if result.years else None,
        "last_year": result.years[-1] if result.years else None,
        "population": _mean(pop_y),
        "gdp_usd": _mean([sum(r.gdp_usd for r in recs) for recs in by_year]),
        # indices
        "water_security": rip_mean("water_security"),
        "energy_security": rip_mean("energy_security"),
        "food_security": rip_mean("food_security"),
        "nexus_index": rip_mean("nexus_index"),
        "nexus_index_population_weighted": float(pop_weighted),
        "min_nexus_index": min(nexus_by_rip.values()),
        "worst_riparian": min(names, key=lambda n: nexus_by_rip[n]),
        "equity_index": _sus.equity_index({n: riparians[n]["supply_ratio"] for n in names}),
        "equity_nexus_index": _sus.equity_index(nexus_by_rip),
        # water
        "supply_ratio": _supply_ratio(sum(demand_y), sum(deficit_y)),
        "min_supply_ratio": min(ratio_y),
        "supply_reliability": supply_reliability(supplied_y, demand_y),
        "resilience": resilience(supplied_y, demand_y),
        "vulnerability": vulnerability(supplied_y, demand_y),
        "total_demand_mm3": _mean(demand_y),
        "total_withdrawal_mm3": _mean(withdrawal_y),
        "total_deficit_mm3": _mean(deficit_y),
        "env_flow_met_share": sum(1.0 for m in env_flags if m) / len(env_flags),
        "years_env_flow_unmet": sum(1 for recs in by_year if not all(r.env_flow_met for r in recs)),
        "water_stress_sdg642": rip_mean("water_stress_sdg642"),
        "groundwater_stress": rip_mean("groundwater_stress"),
        "final_storage_mm3": float(sum(riparians[n]["final_storage_mm3"] for n in names)),
        "water_value_usd": _mean([sum(r.water_value_usd for r in recs) for recs in by_year]),
        # energy
        "hydropower_gwh": _mean(hydro_y),
        "total_hydropower_gwh": float(sum(hydro_y)),
        "energy_supply_gwh": _mean(e_supply_y),
        "energy_demand_gwh": _mean(e_demand_y),
        "energy_deficit_gwh": _mean([sum(r.energy_deficit_gwh for r in recs) for recs in by_year]),
        "energy_self_sufficiency": _ratio(total_e_supply, sum(e_demand_y), 1.0),
        "renewable_share": min(max(_ratio(sum(renew_y), total_e_supply, 0.0), 0.0), 1.0),
        "emissions_t": _mean(emissions_y),
        # food
        "food_production_t": _mean([sum(r.food_production_t for r in recs) for recs in by_year]),
        "food_kcal": _mean(kcal_y),
        "food_demand_kcal": _mean(kcal_demand_y),
        "food_self_sufficiency": _ratio(sum(kcal_y), sum(kcal_demand_y), 1.0),
        "crop_value_usd": _mean([sum(r.crop_value_usd for r in recs) for recs in by_year]),
        # from the routed balances
        "natural_flow_mm3": None,
        "outflow_to_sea_mm3": None,
        "mass_balance_error_mm3": None,
    }
    if result.balances:
        out["natural_flow_mm3"] = _mean([float(b.natural_flow_mm3) for b in result.balances])
        out["outflow_to_sea_mm3"] = _mean([float(b.outflow_to_sea_mm3) for b in result.balances])
        out["mass_balance_error_mm3"] = max(float(b.mass_balance_error()) for b in result.balances)
    return out


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class NexusModel:
    """Multi-year WEF nexus simulation of a basin under a scenario.

    Parameters
    ----------
    basin : Basin
        The basin to simulate (riparians ordered upstream -> downstream).
        Never mutated.
    scenario : Scenario
        Drivers of the horizon (flow trend, growth rates, efficiency and
        renewable targets, cooperation and allocation rule, stochastic
        setting).  Never mutated.
    allocation_rule : str, optional
        Overrides ``scenario.allocation_rule``: ``"treaty"``,
        ``"upstream_priority"`` (or ``None`` = no caps) or any bankruptcy
        rule of :data:`wefnexus.allocation.RULES` / its aliases
        (``"proportional"``, ``"cea"``, ``"cel"``, ``"talmud"``, ``"ap"``,
        ``"equal"``).  Default ``None`` uses the scenario's rule.
    reservoir_refill_fraction : float, optional
        Share of the river surplus above the environmental flow stored each
        year by every reservoir, in ``[0, 1]`` (see
        :func:`wefnexus.water.route_basin`).  Default 0.25.
    flow_factors : sequence of float, optional (keyword-only)
        Explicit natural-flow multipliers, one per year (>= 0), replacing
        the scenario trend / stochastic draw.  Default ``None``.

    Attributes
    ----------
    basin, scenario : Basin, Scenario
        The inputs, as given.
    allocation_rule : str
        Canonical name of the rule in force (``"upstream_priority"`` when
        the scenario is non-cooperative or no rule applies).
    requested_rule : str
        Canonical name of the rule asked for (``allocation_rule`` argument
        or ``scenario.allocation_rule``) before the cooperation switch.
    reservoir_refill_fraction : float

    Raises
    ------
    ValueError
        On a non-``Basin`` / non-``Scenario`` argument, an empty basin, an
        unknown rule, a refill fraction outside ``[0, 1]``, ill-sized /
        negative ``flow_factors``, or scenario drivers out of range (a
        horizon shorter than one year, ``drought_severity`` outside
        ``[0, 1]``, a flow or area change below -100 %, a growth rate
        ``<= -1``, an efficiency / renewable target out of range, a negative
        seed or drought-year offset, non-boolean flags; see
        :func:`wefnexus.scenarios.validate_scenario` for the same checks).

    Notes
    -----
    The model is a priority-driven annual simulation in the tradition of
    WEAP (Yates et al. 2005) extended with the water-energy and water-food
    links of the WEF Nexus Tool (Daher & Mohtar 2015) and the bankruptcy
    allocation of Mianabadi et al. (2014).  It is deterministic for a given
    scenario seed.

    Bankruptcy rules work on a *consumptive* basis (:meth:`claims`,
    :meth:`entitlements`): the estate is the water the basin can consume in
    the year (natural flow plus start-of-year reservoir storage minus the
    environmental flows) and the claims are the consumptive parts of the
    riparians' river-water demands, so the rules only ration when the basin
    is physically short.  ``EnergySystem.renewable_share`` is the share of
    electricity demand met by *non-hydro* renewables; the hydropower of the
    routed flow is added separately (see the module notes).

    Examples
    --------
    >>> from wefnexus.data import example_basin
    >>> from wefnexus.models import Scenario
    >>> result = NexusModel(example_basin(), Scenario(years=3)).run()
    >>> len(result.records), len(result.years)
    (9, 3)
    """

    def __init__(
        self,
        basin: Basin,
        scenario: Scenario,
        allocation_rule: Optional[str] = None,
        reservoir_refill_fraction: float = 0.25,
        *,
        flow_factors: Optional[Sequence[float]] = None,
    ) -> None:
        if not isinstance(basin, Basin):
            raise ValueError(f"basin must be a wefnexus.models.Basin, got {type(basin).__name__}")
        if not isinstance(scenario, Scenario):
            raise ValueError(f"scenario must be a wefnexus.models.Scenario, got {type(scenario).__name__}")
        if len(basin.riparians) == 0:
            raise ValueError(f"basin {basin.name!r} has no riparians to simulate")
        _validate_scenario(scenario)
        years = scenario.years
        self.basin = basin
        self.scenario = scenario
        rule = scenario.allocation_rule if allocation_rule is None else allocation_rule
        self.requested_rule = _canonical_rule(rule)
        self.allocation_rule = self.requested_rule if scenario.cooperation else "upstream_priority"
        self.reservoir_refill_fraction = _fraction(reservoir_refill_fraction, "reservoir_refill_fraction")
        if flow_factors is None:
            self._flow_factors: Optional[List[float]] = None
        else:
            try:
                factors = [float(f) for f in flow_factors]
            except (TypeError, ValueError):
                raise ValueError("flow_factors must be a sequence of numbers") from None
            if len(factors) != int(years):
                raise ValueError(
                    f"flow_factors must hold one factor per year ({int(years)}), got {len(factors)}"
                )
            self._flow_factors = [_nonneg(f, f"flow_factors[{i}]") for i, f in enumerate(factors)]
        # one-off validation of per-riparian attributes used directly here
        for r in basin.riparians:
            label = f"riparian {r.name!r}"
            _nonneg(r.population, f"{label} population")
            _nonneg(r.gdp_usd, f"{label} gdp_usd")
            _nonneg(r.groundwater_recharge_mm3, f"{label} groundwater_recharge_mm3")
            _nonneg(r.groundwater_abstraction_mm3, f"{label} groundwater_abstraction_mm3")
            _nonneg(r.energy.desalination_capacity_mm3, f"{label} energy.desalination_capacity_mm3")
            _nonneg(r.energy.demand_gwh, f"{label} energy.demand_gwh")
            _fraction(r.energy.turbined_fraction, f"{label} energy.turbined_fraction")
            _nonneg(r.food_demand_kcal_per_capita_day, f"{label} food_demand_kcal_per_capita_day")
            for s, v in r.demand.value_usd_per_m3.items():
                _nonneg(v, f"{label} demand.value_usd_per_m3[{Sector(s).value}]")

    # -- helpers ---------------------------------------------------------------
    @property
    def years(self) -> List[int]:
        """Calendar years of the horizon."""
        return [self.scenario.year(i) for i in range(int(self.scenario.years))]

    def _check_year_index(self, year_index: Any) -> int:
        if isinstance(year_index, bool) or not isinstance(year_index, int):
            raise ValueError(f"year_index must be an int, got {year_index!r}")
        n = int(self.scenario.years)
        if not 0 <= year_index < n:
            raise ValueError(f"year_index must lie in [0, {n - 1}], got {year_index}")
        return year_index

    def flow_factors(self) -> List[float]:
        """Natural-flow multiplier of every year of the horizon.

        Returns
        -------
        list of float
            The explicit ``flow_factors`` given at construction, otherwise
            :func:`wefnexus.water.stochastic_flow_factors` (the deterministic
            trend of ``Scenario.flow_factor`` with droughts, multiplied by a
            seeded lognormal draw when ``scenario.stochastic``).
        """
        if self._flow_factors is not None:
            return list(self._flow_factors)
        return [float(f) for f in stochastic_flow_factors(self.scenario, self.basin)]

    def _drivers(self, r: Riparian, i: int) -> Dict[str, Any]:
        """Year-``i`` drivers of riparian ``r`` (all scenario factors applied)."""
        sc = self.scenario
        label = f"riparian {r.name!r}"
        pf = _nonneg(sc.population_factor(i), f"scenario.population_factor({i})")
        gf = _nonneg(sc.gdp_factor(i), f"scenario.gdp_factor({i})")
        df = _nonneg(sc.demand_factor(i), f"scenario.demand_factor({i})")
        edf = _nonneg(sc.energy_demand_factor(i), f"scenario.energy_demand_factor({i})")
        af = _nonneg(sc.irrigated_area_factor(i), f"scenario.irrigated_area_factor({i})")
        eff = sc.irrigation_efficiency(r.irrigation_efficiency, i)
        eff = _as_float(eff, f"scenario.irrigation_efficiency({label}, {i})")
        if not 0.0 < eff <= 1.0:
            raise ValueError(f"irrigation efficiency of {label} in year {i} must lie in (0, 1], got {eff}")
        rs = _fraction(sc.renewable_share(r.energy.renewable_share, i), f"renewable share of {label} in year {i}")
        dem = r.demand
        if r.crops:
            ag = _food.riparian_irrigation_requirement_mm3(r, af, eff)
        else:
            ag = _nonneg(dem.agricultural, f"{label} demand.agricultural") * af
        demand = {
            Sector.MUNICIPAL: _nonneg(dem.municipal, f"{label} demand.municipal") * df,
            Sector.INDUSTRIAL: _nonneg(dem.industrial, f"{label} demand.industrial") * df,
            Sector.ENERGY: _nonneg(dem.energy, f"{label} demand.energy") * df,
            Sector.AGRICULTURAL: ag,
        }
        return {
            "population": r.population * pf,
            "gdp_usd": r.gdp_usd * gf,
            "demand": demand,
            "irrigation_requirement_mm3": ag,
            "area_factor": af,
            "irrigation_efficiency": eff,
            "irrigated_area_ha": r.irrigated_area_ha() * af,
            "energy_demand_gwh": r.energy.demand_gwh * edf,
            "renewable_share": rs,
        }

    def drivers(self, year_index: int) -> Dict[str, Dict[str, Any]]:
        """Scenario drivers of every riparian in year ``year_index``.

        Parameters
        ----------
        year_index : int
            0-based year offset in ``[0, scenario.years)``.

        Returns
        -------
        dict
            ``{riparian: {"population", "gdp_usd", "demand" (``{Sector:
            Mm3/yr}``), "irrigation_requirement_mm3", "area_factor",
            "irrigation_efficiency", "irrigated_area_ha", "energy_demand_gwh"
            (before energy-for-water), "renewable_share"}}``.
        """
        i = self._check_year_index(year_index)
        return {r.name: self._drivers(r, i) for r in self.basin.riparians}

    def demands(self, year_index: int) -> Dict[str, Dict[Sector, float]]:
        """Withdrawal demands of every riparian in year ``year_index`` (Mm3/yr).

        Municipal, industrial and energy demands are ``WaterDemand`` values
        times ``scenario.demand_factor(i)``; the agricultural demand is the
        crop irrigation requirement (area factor and scenario efficiency
        applied) when the riparian has crops, else ``demand.agricultural``
        times the irrigated-area factor.
        """
        i = self._check_year_index(year_index)
        return {r.name: dict(self._drivers(r, i)["demand"]) for r in self.basin.riparians}

    def natural_flow(self, year_index: int, flow_factor: Optional[float] = None) -> float:
        """Basin natural flow of year ``year_index`` (Mm3/yr).

        ``(headwater + sum of local inflows) * flow_factor``; the factor
        defaults to :meth:`flow_factors` ``[year_index]``.
        """
        i = self._check_year_index(year_index)
        ff = self.flow_factors()[i] if flow_factor is None else _nonneg(flow_factor, "flow_factor")
        flows = natural_flows(self.basin, ff)
        return float(flows[self.basin.riparians[-1].name]) if flows else 0.0

    def _demands_of_year(
        self, i: int, demands: Optional[Mapping[str, Mapping[Sector, float]]]
    ) -> Dict[str, Dict[Sector, float]]:
        """Validated ``{riparian: {Sector: Mm3/yr}}`` demands (default :meth:`demands`)."""
        names = self.basin.names()
        if demands is None:
            return self.demands(i)
        if not isinstance(demands, Mapping):
            raise ValueError("demands must be a dict of {riparian: {Sector: Mm3}}")
        unknown = [k for k in demands if k not in names]
        if unknown:
            raise ValueError(f"demands refers to unknown riparian(s) {unknown}; basin riparians are {names}")
        missing = [n for n in names if n not in demands]
        if missing:
            raise ValueError(f"demands lacks riparian(s) {missing}")
        return {n: _sector_dict(demands[n], f"demands[{n!r}]") for n in names}

    def _storages_start(self, storages: Optional[Mapping[str, float]]) -> Dict[str, float]:
        """Start-of-year reservoir storages (Mm3) as :func:`wefnexus.water.route_basin` sees them.

        Entries of ``storages`` are used where given; a missing riparian (or
        ``None``) falls back to its ``reservoir_storage_mm3``.
        """
        names = self.basin.names()
        if storages is None:
            given: Mapping[str, Any] = {}
        elif isinstance(storages, Mapping):
            unknown = [k for k in storages if k not in names]
            if unknown:
                raise ValueError(f"storages refers to unknown riparian(s) {unknown}; basin riparians are {names}")
            given = storages
        else:
            raise ValueError("storages must be a dict of {riparian: Mm3} or None")
        out: Dict[str, float] = {}
        for r in self.basin.riparians:
            v = given.get(r.name)
            if v is None:
                out[r.name] = _nonneg(r.reservoir_storage_mm3, f"riparian {r.name!r} reservoir_storage_mm3")
            else:
                out[r.name] = _nonneg(v, f"storages[{r.name!r}]")
        return out

    def claims(
        self,
        year_index: int,
        demands: Optional[Mapping[str, Mapping[Sector, float]]] = None,
    ) -> Dict[str, float]:
        """Consumptive river-water claims of every riparian for one year.

        A riparian's claim is the part of its withdrawal demand that would
        be taken from the river *and consumed*: the gross demand net of the
        non-river supply (groundwater abstraction + desalination capacity,
        offset proportionally across sectors exactly as
        :func:`wefnexus.water.route_basin` does), weighted by the sector
        consumption fractions::

            D               = sum_s demand[s]
            river_demand[s] = demand[s] * (1 - min(non_river, D) / D)      (0 if D == 0)
            claim           = sum_s river_demand[s] * consumption_fraction[s]

        Return flows are not claimed: they stay in the river and are re-used
        downstream.  This puts the claims on the same net basis as the
        estate of :meth:`entitlements` (consumptive-use accounting, as in the
        Colorado River Compact and Mianabadi et al. 2014).

        Parameters
        ----------
        year_index : int
            0-based year offset in ``[0, scenario.years)``.
        demands : dict, optional
            ``{riparian: {Sector: Mm3/yr}}`` gross demands of the year
            (default :meth:`demands`); missing sectors count as 0.

        Returns
        -------
        dict
            ``{riparian: claim}`` in Mm3/yr, in basin order; each claim lies
            in ``[0, sum of the riparian's river demand]``.

        Raises
        ------
        ValueError
            On a bad year index, demands naming unknown or lacking riparians,
            negative demands or consumption fractions outside ``[0, 1]``.
        """
        i = self._check_year_index(year_index)
        dem = self._demands_of_year(i, demands)
        return {
            r.name: _consumptive_claim(_river_demand(r, dem[r.name]), _consumption_fractions(r))
            for r in self.basin.riparians
        }

    def entitlements(
        self,
        year_index: int,
        natural_flow: float,
        demands: Optional[Mapping[str, Mapping[Sector, float]]] = None,
        storages: Optional[Mapping[str, float]] = None,
    ) -> Optional[Dict[str, Optional[float]]]:
        """Surface-withdrawal caps of every riparian for one year.

        Parameters
        ----------
        year_index : int
            0-based year offset (used to grow the demands that form the
            claims of a bankruptcy rule).
        natural_flow : float
            Basin natural flow of the year (Mm3/yr), e.g.
            :meth:`natural_flow`.
        demands : dict, optional
            Pre-computed ``{riparian: {Sector: Mm3/yr}}`` gross demands of
            the year (default :meth:`demands`).
        storages : dict, optional
            ``{riparian: Mm3}`` start-of-year reservoir storages carried into
            the year (``BasinBalance.storages_end()`` of the previous year);
            a missing riparian, or ``None``, uses its ``reservoir_storage_mm3``
            - the same convention as :func:`wefnexus.water.route_basin`.
            Only bankruptcy rules use it.

        Returns
        -------
        dict or None
            * ``None`` when ``scenario.cooperation`` is False or the rule is
              ``"upstream_priority"``: no caps, the upstream riparian
              withdraws first;
            * for ``"treaty"``: ``{riparian: treaty_allocation_mm3}`` with
              ``None`` for riparians without an entitlement (no cap).  Treaty
              volumes are fixed, not scaled with the flow;
            * for a bankruptcy rule: caps on gross surface withdrawal (from
              river *and* storage) derived from
              ``awards = apply_rule(rule, estate, claims)`` with

              - ``estate = max(natural_flow + sum(storages) - sum(environmental
                flows), 0)``, the water the basin can consume this year;
              - ``claims`` the consumptive river-water claims of
                :meth:`claims`;
              - ``cap = `` the gross withdrawal at which the riparian's
                consumption, serving sectors in
                :data:`~wefnexus.models.SECTOR_PRIORITY` order as the router
                does, equals its award.  An award equal to the claim gives a
                cap equal to the riparian's river demand (never binding).

              Awards sum to ``min(estate, sum of claims)`` and never exceed
              the claim, so a year in which the consumptive claims fit into
              the estate leaves every riparian uncapped at its demand and a
              rule only rations under physical scarcity; because the estate
              counts the carried storage, a zero-flow year with full
              reservoirs still lets the riparians draw them down.

        Raises
        ------
        ValueError
            On a bad year index, a negative natural flow, demands or
            storages naming unknown riparians, negative storages, or demands
            lacking riparians.

        References
        ----------
        Mianabadi et al. (2014); Ansink & Weikard (2012); Aumann & Maschler
        (1985) for the Talmud rule.
        """
        i = self._check_year_index(year_index)
        nat = _nonneg(natural_flow, "natural_flow")
        rule = self.allocation_rule
        if rule == "upstream_priority":
            return None
        if rule == "treaty":
            out: Dict[str, Optional[float]] = {}
            for r in self.basin.riparians:
                t = r.treaty_allocation_mm3
                out[r.name] = None if t is None else _nonneg(t, f"riparian {r.name!r} treaty_allocation_mm3", allow_inf=True)
            return out
        names = self.basin.names()
        dem = self._demands_of_year(i, demands)
        storage_total = sum(self._storages_start(storages).values())
        env_total = sum(_nonneg(r.demand.environmental, f"riparian {r.name!r} demand.environmental") for r in self.basin.riparians)
        estate = max(nat + storage_total - env_total, 0.0)
        river = {r.name: _river_demand(r, dem[r.name]) for r in self.basin.riparians}
        fractions = {r.name: _consumption_fractions(r) for r in self.basin.riparians}
        claims = {n: _consumptive_claim(river[n], fractions[n]) for n in names}
        awards = apply_rule(rule, estate, claims)
        return {n: _withdrawal_cap_for_award(river[n], fractions[n], float(awards[n])) for n in names}

    # -- simulation ------------------------------------------------------------
    def _record(
        self,
        r: Riparian,
        year: int,
        ff: float,
        reach: ReachResult,
        d: Dict[str, Any],
    ) -> RiparianYear:
        """Build the :class:`RiparianYear` of riparian ``r`` from its routed reach."""
        pop = d["population"]
        gdp = d["gdp_usd"]
        env = reach.environmental_flow_mm3

        # -- non-river supply split (groundwater vs desalination) --------------
        gw_abs = r.groundwater_abstraction_mm3
        desal_cap = r.energy.desalination_capacity_mm3
        non_river = reach.non_river_supply_mm3
        total_nr = gw_abs + desal_cap
        if total_nr > 0.0 and non_river > 0.0:
            gw_used = non_river * gw_abs / total_nr
            desal_used = max(non_river - gw_used, 0.0)
        else:
            gw_used = desal_used = 0.0

        # -- water indicators -------------------------------------------------
        surface_w = reach.total_withdrawal()
        renewable = reach.inflow_mm3 + r.groundwater_recharge_mm3
        per_capita = per_capita_water(renewable, pop) if pop > 0.0 else math.inf
        falk = falkenmark_category(per_capita)
        stress = sdg_642_water_stress(surface_w + gw_used, renewable, env)
        gw_stress = groundwater_stress(gw_used, r.groundwater_recharge_mm3)
        env_compliance = 1.0 if env <= 0.0 else min(max(reach.outflow_mm3 / env, 0.0), 1.0)
        supply_ratio = reach.supply_ratio()
        water_security = _sus.water_security_index(supply_ratio, stress, env_compliance, gw_stress)

        # -- energy ------------------------------------------------------------
        es = r.energy
        hydro = _energy.hydropower_gwh(
            reach.outflow_mm3 * es.turbined_fraction,
            es.hydropower_head_m,
            es.turbine_efficiency,
            capacity_mw=es.hydropower_capacity_mw,
        )
        thermal = _energy.thermal_generation_gwh(es.thermal_capacity_mw, es.thermal_capacity_factor)
        cooling = _energy.water_for_energy(r, thermal)
        base_demand = d["energy_demand_gwh"]
        # "other" renewables are the NON-hydro share of demand; the hydropower
        # of the routed flow is added separately (see the module notes)
        other = base_demand * d["renewable_share"]
        efw = dict(_energy.energy_for_water(r, reach.withdrawals, desal_used))
        # energy_for_water() lifts only the pumped share of the *surface*
        # withdrawals; the groundwater actually abstracted is lifted entirely
        gw_pumping = _energy.pumping_energy_gwh(gw_used, es.pumping_lift_m, es.pumping_efficiency)
        efw["groundwater"] = gw_pumping
        efw["total"] = efw["total"] + gw_pumping
        demand_total = base_demand + efw["total"]
        ebal = _energy.energy_balance(r, hydro, thermal, other, demand_total)
        supply = ebal["supply"]
        e_ratio = _energy.energy_security_index(supply, demand_total)
        ren_share = min(max(_ratio(hydro + other, supply, 0.0), 0.0), 1.0)
        intensity = _ratio(ebal["emissions_t"], supply, 0.0)
        energy_security = _sus.energy_security_index(e_ratio, ren_share, intensity)

        # -- food --------------------------------------------------------------
        ag_demand = reach.demands.get(Sector.AGRICULTURAL, 0.0)
        ag_deficit = reach.deficits.get(Sector.AGRICULTURAL, 0.0)
        ag_water = max(ag_demand - ag_deficit, 0.0)  # surface withdrawal + non-river share
        fp = _food.riparian_food_production(r, ag_water, d["area_factor"], d["irrigation_efficiency"])
        food_demand = _food.food_demand_kcal(pop, r.food_demand_kcal_per_capita_day)
        ssr = _food.food_self_sufficiency(fp["kcal"], food_demand)
        et_ratio = fp["et_ratio"] if r.crops else _supply_ratio(ag_demand, ag_deficit)
        food_security = _sus.food_security_index(ssr, et_ratio)

        nexus = _sus.wef_nexus_index(water_security, energy_security, food_security)
        values = r.demand.value_usd_per_m3
        water_value = float(sum(w * values.get(s, 0.0) * _M3_PER_MM3 for s, w in reach.withdrawals.items()))

        return RiparianYear(
            name=r.name,
            year=year,
            population=pop,
            gdp_usd=gdp,
            inflow_mm3=reach.inflow_mm3,
            upstream_inflow_mm3=reach.upstream_inflow_mm3,
            outflow_mm3=reach.outflow_mm3,
            storage_end_mm3=reach.storage_end_mm3,
            entitlement_mm3=reach.entitlement_mm3,
            demands=dict(reach.demands),
            withdrawals=dict(reach.withdrawals),
            deficits=dict(reach.deficits),
            environmental_flow_mm3=env,
            env_flow_met=bool(reach.env_flow_met),
            per_capita_water_m3=per_capita,
            falkenmark=falk,
            water_stress_sdg642=stress,
            groundwater_stress=gw_stress,
            hydropower_gwh=hydro,
            thermal_gwh=thermal,
            other_renewable_gwh=other,
            energy_supply_gwh=supply,
            energy_demand_gwh=demand_total,
            energy_for_water_gwh=efw["total"],
            energy_deficit_gwh=ebal["deficit"],
            emissions_t=ebal["emissions_t"],
            food_production_t=fp["production_t"],
            food_kcal=fp["kcal"],
            food_demand_kcal=food_demand,
            food_self_sufficiency=ssr,
            et_ratio=et_ratio,
            irrigation_requirement_mm3=d["irrigation_requirement_mm3"],
            crop_value_usd=fp["value_usd"],
            water_value_usd=water_value,
            water_security=water_security,
            energy_security=energy_security,
            food_security=food_security,
            nexus_index=nexus,
            storage_start_mm3=reach.storage_start_mm3,
            storage_release_mm3=reach.storage_release_mm3,
            storage_refill_mm3=reach.storage_refill_mm3,
            evaporation_mm3=reach.evaporation_mm3,
            non_river_supply_mm3=non_river,
            groundwater_used_mm3=gw_used,
            desalinated_mm3=desal_used,
            consumption=dict(reach.consumption),
            flow_factor=ff,
            renewable_water_mm3=renewable,
            agricultural_water_mm3=ag_water,
            cooling_water_mm3=cooling,
            renewable_share=ren_share,
            emission_intensity_t_per_gwh=intensity,
            energy_supply_ratio=e_ratio,
            env_flow_compliance=env_compliance,
            irrigated_area_ha=d["irrigated_area_ha"],
            irrigation_efficiency=d["irrigation_efficiency"],
            groundwater_pumping_gwh=gw_pumping,
            other_renewable_share_of_demand=d["renewable_share"],
        )

    def run(self) -> NexusResult:
        """Simulate the whole horizon.

        For every year ``i`` of the scenario: resolve the flow factor
        (:meth:`flow_factors`), grow the drivers (:meth:`drivers`), derive the
        caps (:meth:`entitlements`, with the carried storages in the estate
        of a bankruptcy rule), route the basin with
        :func:`wefnexus.water.route_basin` (carrying the previous year's
        end-of-year storages), and compute the energy, food and indicator
        block of every riparian (see the module docstring).

        Returns
        -------
        NexusResult
            Records in year-major order, one :class:`~wefnexus.water.BasinBalance`
            per year, the flow factors and the rule applied.  The basin and
            scenario passed to the model are left untouched; the result holds
            a copy of the scenario.

        Raises
        ------
        ValueError
            Propagated from the leaf modules on invalid riparian attributes
            (negative volumes, efficiencies outside their ranges, ...).
        """
        basin = self.basin
        sc = self.scenario
        factors = self.flow_factors()
        years = self.years
        records: List[RiparianYear] = []
        balances: List[BasinBalance] = []
        storages: Optional[Dict[str, float]] = None
        for i, ff in enumerate(factors):
            year = years[i]
            drivers = {r.name: self._drivers(r, i) for r in basin.riparians}
            demands = {name: dict(d["demand"]) for name, d in drivers.items()}
            natural = self.natural_flow(i, ff)
            caps = self.entitlements(i, natural, demands=demands, storages=storages)
            # ``None`` from entitlements() means *no caps*; route_basin would
            # read ``entitlements=None`` as "use each riparian's treaty
            # allocation", so the absence of caps is made explicit here.
            if caps is None:
                caps = {r.name: None for r in basin.riparians}
            balance = route_basin(
                basin,
                flow_factor=ff,
                entitlements=caps,
                demands=demands,
                storages=storages,
                reservoir_refill_fraction=self.reservoir_refill_fraction,
            )
            balances.append(balance)
            storages = balance.storages_end()
            for r in basin.riparians:
                records.append(self._record(r, year, ff, balance.reach(r.name), drivers[r.name]))
        scenario_copy = replace(sc, drought_years=list(sc.drought_years))
        return NexusResult(
            basin_name=basin.name,
            scenario=scenario_copy,
            years=years,
            records=records,
            balances=balances,
            flow_factors=factors,
            allocation_rule=self.allocation_rule,
        )


def run_nexus(basin: Basin, scenario: Scenario, **kw: Any) -> NexusResult:
    """Build a :class:`NexusModel` and run it.

    Parameters
    ----------
    basin : Basin
        Basin to simulate (never mutated).
    scenario : Scenario
        Scenario drivers (never mutated).
    **kw
        Passed to :class:`NexusModel` (``allocation_rule``,
        ``reservoir_refill_fraction``, ``flow_factors``).

    Returns
    -------
    NexusResult

    Examples
    --------
    >>> from wefnexus.data import example_basin
    >>> from wefnexus.models import Scenario
    >>> res = run_nexus(example_basin(), Scenario(name="baseline", years=5))
    >>> res.n_years, res.n_riparians, len(res.records)
    (5, 3, 15)
    """
    return NexusModel(basin, scenario, **kw).run()
