"""Scenario library, multi-scenario runner and comparison tables (composite module).

This module provides a small library of named *scenario factories* that
build :class:`~wefnexus.models.Scenario` driver sets for the integrated
nexus model of :mod:`wefnexus.nexus`, a runner that simulates every scenario
on a fresh copy of a basin, and helpers that condense the results into
comparison rows and plain-text tables.

Scenario families
-----------------
The library follows the scenario-analysis practice of the IPCC (Moss et al.
2010; O'Neill et al. 2014) and of transboundary-basin planning studies
(Mahmoud et al. 2009; Jeuland et al. 2014): a *baseline* (business as
usual) is contrasted with single-driver variants that isolate one pressure
or response, and with *combined* storylines.

======================  ==========================================================
factory                 storyline
======================  ==========================================================
``baseline``            current flows, demands, area and energy mix held constant
``climate_change``      natural flow declines linearly to ``flow_change_pct`` by the
                        final year, with discrete drought years
``growth``              population, municipal/industrial and electricity demand
                        grow at compound rates, irrigated area expands
``efficiency``          irrigation efficiency and renewable electricity share rise
                        linearly to targets (demand-side / supply-side response)
``unilateral``          no cooperation: upstream riparians withdraw first
``cooperative``         a bankruptcy sharing rule (default Talmud) divides the
                        basin's water among the riparians' claims
``combined_stress``     climate change + growth under the existing fixed-volume
                        treaty
``combined_adaptation`` climate change + growth + efficiency, shared by the
                        Talmud rule (the contract's composition; see below)
======================  ==========================================================

Every factory takes ``years`` (horizon length) plus its own storyline
parameters, and accepts any further :class:`~wefnexus.models.Scenario`
field as a keyword override (for example ``stochastic=True, seed=3`` or
``start_year=2030``).  Overrides may not duplicate a field that the
factory's own parameters already set (``ValueError``), except the labels
``name`` and ``description``.  All factories validate their inputs and
return a fresh :class:`~wefnexus.models.Scenario` with a descriptive
``description``.

A note on the adaptation package
--------------------------------
ARCHITECTURE.md section 9 defines ``combined_adaptation`` as "climate +
growth + efficiency + talmud" and the factory follows it
(:data:`DEFAULT_ADAPTATION_RULE`).  In :class:`~wefnexus.nexus.NexusModel` a
bankruptcy rule divides the water the basin can *consume* in the year - the
natural flow plus the carried reservoir storage, net of the environmental
flows - among the riparians' consumptive river-water claims (Mianabadi et
al. 2014; consumptive-use accounting as in the Colorado River Compact).
Claims that fit into that estate are awarded in full, so the rule rations
only under physical scarcity, whereas a fixed treaty volume below a
riparian's demand binds in every year.  On the stylised Azura basin the
Talmud package therefore serves every riparian in full (10-year Delta mean
supply ratio: stress 0.90, treaty + efficiency 0.97, Talmud + efficiency
1.00).  ``rule="treaty"`` gives the fixed-treaty variant, which differs from
:func:`combined_stress` only by its efficiency measures.

Units
-----
Growth rates are fractions per year (0.015 = 1.5 %/yr); ``flow_change_pct``
and ``irrigated_area_change_pct`` are percentages reached by the final year;
``drought_severity`` is the fractional flow reduction in a drought year;
efficiency and renewable targets are fractions in ``(0, 1]`` / ``[0, 1]``.
Comparison-table quantities are those of :meth:`wefnexus.nexus.NexusResult.summary`:
water in **Mm3/yr** (storage in **Mm3**), energy in **GWh/yr** (hydropower
total in **GWh** over the horizon), emissions in **t CO2/yr**, indices
dimensionless in ``[0, 1]``.

References
----------
Ansink, E. & Ruijs, A. (2008). Climate change and the stability of water
    allocation agreements. *Environmental and Resource Economics* 41,
    249-266.
Aumann, R. J. & Maschler, M. (1985). Game theoretic analysis of a bankruptcy
    problem from the Talmud. *Journal of Economic Theory* 36(2), 195-213.
Cooley, H. & Gleick, P. H. (2011). Climate-proofing transboundary water
    agreements. *Hydrological Sciences Journal* 56(4), 711-718.
Dinar, S., Katz, D., De Stefano, L. & Blankespoor, B. (2015). Climate
    change, conflict, and cooperation: global analysis of the effectiveness
    of international river treaties in addressing water variability.
    *Political Geography* 45, 55-66.
Drieschova, A., Giordano, M. & Fischhendler, I. (2008). Governance
    mechanisms to address flow variability in water treaties. *Global
    Environmental Change* 18(2), 285-295.
Hashimoto, T., Stedinger, J. R. & Loucks, D. P. (1982). Reliability,
    resiliency, and vulnerability criteria for water resource system
    performance evaluation. *Water Resources Research* 18(1), 14-20.
Jeuland, M., Baker, J., Bartlett, R. & Lacombe, G. (2014). The costs of
    uncoordinated infrastructure management in the Mekong basin.
    *Environmental Research Letters* 9, 105006.
Mahmoud, M. et al. (2009). A formal framework for scenario development in
    support of environmental decision-making. *Environmental Modelling &
    Software* 24(7), 798-808.
Mianabadi, H., Mostert, E., Zarghami, M. & van de Giesen, N. (2014). A new
    bankruptcy method for conflict resolution in water resources allocation.
    *Journal of Environmental Management* 144, 152-159.
Moss, R. H. et al. (2010). The next generation of scenarios for climate
    change research and assessment. *Nature* 463, 747-756.
O'Neill, B. C. et al. (2014). A new scenario framework for climate change
    research: the concept of shared socioeconomic pathways. *Climatic
    Change* 122, 387-400.
Wolf, A. T. (2007). Shared waters: conflict and cooperation. *Annual Review
    of Environment and Resources* 32, 241-269.
Yates, D., Sieber, J., Purkey, D. & Huber-Lee, A. (2005). WEAP21 - a
    demand-, priority-, and preference-driven water planning model.
    *Water International* 30(4), 487-500.
"""
from __future__ import annotations

import csv
import dataclasses
import inspect
import math
import numbers
import os
from dataclasses import replace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from wefnexus.allocation import RULE_ALIASES
from wefnexus.models import Basin, Scenario
from wefnexus.nexus import ALLOCATION_RULES, BASIN_KEY, NexusModel, NexusResult

__all__ = [
    # defaults and constants
    "DEFAULT_YEARS",
    "DEFAULT_FLOW_CHANGE_PCT",
    "DEFAULT_DROUGHT_YEARS",
    "DEFAULT_DROUGHT_SEVERITY",
    "DEFAULT_POPULATION_GROWTH",
    "DEFAULT_DEMAND_GROWTH",
    "DEFAULT_ENERGY_GROWTH",
    "DEFAULT_IRRIGATED_AREA_CHANGE_PCT",
    "DEFAULT_IRRIGATION_EFFICIENCY_TARGET",
    "DEFAULT_RENEWABLE_SHARE_TARGET",
    "DEFAULT_COOPERATIVE_RULE",
    "DEFAULT_ADAPTATION_RULE",
    "SCENARIO_FIELDS",
    "KEY_COLUMNS",
    "INFO_COLUMNS",
    "COMPARISON_COLUMNS",
    "BASIN_COLUMNS",
    "TABLE_COLUMNS",
    # factories
    "baseline",
    "climate_change",
    "growth",
    "efficiency",
    "unilateral",
    "cooperative",
    "combined_stress",
    "combined_adaptation",
    "SCENARIOS",
    "scenario_names",
    "get_scenario",
    "all_scenarios",
    "validate_scenario",
    # runner and tables
    "run_scenarios",
    "comparison_table",
    "scenario_differences",
    "rank_scenarios",
    "format_table",
    "write_table_csv",
]

# --------------------------------------------------------------------------- #
# Defaults (the values of ARCHITECTURE.md section 9)
# --------------------------------------------------------------------------- #
#: Default horizon length in years.
DEFAULT_YEARS: int = 25
#: Default change of natural flow by the final year, percent (negative = drier).
DEFAULT_FLOW_CHANGE_PCT: float = -20.0
#: Default 0-based year offsets of drought years.
DEFAULT_DROUGHT_YEARS: Tuple[int, ...] = (8, 15, 22)
#: Default fractional reduction of natural flow in a drought year.
DEFAULT_DROUGHT_SEVERITY: float = 0.4
#: Default population growth, fraction per year.
DEFAULT_POPULATION_GROWTH: float = 0.015
#: Default municipal + industrial (+ energy-sector water) demand growth, fraction per year.
DEFAULT_DEMAND_GROWTH: float = 0.02
#: Default electricity demand growth, fraction per year.
DEFAULT_ENERGY_GROWTH: float = 0.03
#: Default change of irrigated area by the final year, percent.
DEFAULT_IRRIGATED_AREA_CHANGE_PCT: float = 20.0
#: Default irrigation efficiency reached by the final year (fraction).
DEFAULT_IRRIGATION_EFFICIENCY_TARGET: float = 0.7
#: Default renewable electricity share reached by the final year (fraction).
DEFAULT_RENEWABLE_SHARE_TARGET: float = 0.5
#: Default sharing rule of :func:`cooperative`.
DEFAULT_COOPERATIVE_RULE: str = "talmud"
#: Default sharing rule of :func:`combined_adaptation`: the Talmud rule of
#: ARCHITECTURE.md section 9 ("climate + growth + efficiency + talmud").
DEFAULT_ADAPTATION_RULE: str = "talmud"

#: Field names of :class:`~wefnexus.models.Scenario`, in declaration order.
SCENARIO_FIELDS: Tuple[str, ...] = tuple(f.name for f in dataclasses.fields(Scenario))

#: Identifying columns of a comparison row.
KEY_COLUMNS: Tuple[str, ...] = ("scenario", "riparian")
#: Descriptive (non-numeric or horizon) columns of a comparison row.
INFO_COLUMNS: Tuple[str, ...] = ("allocation_rule", "years")
#: Numeric columns present (and numeric) in every comparison row.
COMPARISON_COLUMNS: Tuple[str, ...] = (
    "mean_supply_ratio",
    "min_supply_ratio",
    "supply_reliability",
    "mean_nexus_index",
    "min_nexus_index",
    "mean_water_security",
    "mean_energy_security",
    "mean_food_security",
    "years_env_flow_unmet",
    "env_flow_met_share",
    "total_hydropower_gwh",
    "mean_energy_deficit_gwh",
    "mean_emissions_t",
    "food_self_sufficiency",
    "final_storage_mm3",
    "mean_total_deficit_mm3",
    "mean_water_stress_sdg642",
)
#: Basin-wide columns: numeric in the basin row, ``None`` in riparian rows.
BASIN_COLUMNS: Tuple[str, ...] = (
    "equity_index",
    "natural_flow_mm3",
    "outflow_to_sea_mm3",
    "mass_balance_error_mm3",
)
#: All columns of a comparison row, in order.
TABLE_COLUMNS: Tuple[str, ...] = KEY_COLUMNS + INFO_COLUMNS + COMPARISON_COLUMNS + BASIN_COLUMNS

#: Comparison column -> key of :meth:`wefnexus.nexus.NexusResult.summary`.
_SUMMARY_KEYS: Dict[str, str] = {
    "mean_supply_ratio": "supply_ratio",
    "min_supply_ratio": "min_supply_ratio",
    "supply_reliability": "supply_reliability",
    "mean_nexus_index": "nexus_index",
    "min_nexus_index": "min_nexus_index",
    "mean_water_security": "water_security",
    "mean_energy_security": "energy_security",
    "mean_food_security": "food_security",
    "years_env_flow_unmet": "years_env_flow_unmet",
    "env_flow_met_share": "env_flow_met_share",
    "total_hydropower_gwh": "total_hydropower_gwh",
    "mean_energy_deficit_gwh": "energy_deficit_gwh",
    "mean_emissions_t": "emissions_t",
    "food_self_sufficiency": "food_self_sufficiency",
    "final_storage_mm3": "final_storage_mm3",
    "mean_total_deficit_mm3": "total_deficit_mm3",
    "mean_water_stress_sdg642": "water_stress_sdg642",
    "equity_index": "equity_index",
    "natural_flow_mm3": "natural_flow_mm3",
    "outflow_to_sea_mm3": "outflow_to_sea_mm3",
    "mass_balance_error_mm3": "mass_balance_error_mm3",
}


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
def _as_float(value: Any, label: str) -> float:
    """Coerce to a finite float; bools, NaN, inf and non-numbers raise ``ValueError``."""
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a number, got {value!r}")
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a number, got {value!r}") from None
    if math.isnan(v) or math.isinf(v):
        raise ValueError(f"{label} must be finite, got {v}")
    return v


def _as_int(value: Any, label: str) -> int:
    """Coerce an integral number (int or integral float) to ``int``."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{label} must be an integer, got {value!r}")
    v = float(value)
    if math.isnan(v) or math.isinf(v) or int(v) != v:
        raise ValueError(f"{label} must be an integer, got {value!r}")
    return int(v)


def _check_years(years: Any) -> int:
    n = _as_int(years, "years")
    if n < 1:
        raise ValueError(f"years must be >= 1, got {n}")
    return n


def _check_rate(value: Any, label: str) -> float:
    """A compound growth rate: finite and > -1 (a factor ``(1 + r) ** i`` stays positive)."""
    v = _as_float(value, label)
    if v <= -1.0:
        raise ValueError(f"{label} must be > -1 (fraction per year), got {v}")
    return v


def _check_pct_change(value: Any, label: str) -> float:
    """A percentage change by the final year: finite and >= -100."""
    v = _as_float(value, label)
    if v < -100.0:
        raise ValueError(f"{label} must be >= -100 percent, got {v}")
    return v


def _check_fraction(value: Any, label: str, *, lo_open: bool = False) -> float:
    v = _as_float(value, label)
    if v < 0.0 or v > 1.0 or (lo_open and v <= 0.0):
        rng = "(0, 1]" if lo_open else "[0, 1]"
        raise ValueError(f"{label} must lie in {rng}, got {v}")
    return v


def _check_droughts(droughts: Any) -> List[int]:
    """Sorted, de-duplicated list of non-negative integer year offsets."""
    if droughts is None:
        return []
    if isinstance(droughts, (str, bytes)) or not isinstance(droughts, Iterable):
        raise ValueError(f"drought years must be an iterable of non-negative integers, got {droughts!r}")
    out = set()
    for d in droughts:
        i = _as_int(d, "drought year offset")
        if i < 0:
            raise ValueError(f"drought year offsets must be >= 0, got {i}")
        out.add(i)
    return sorted(out)


def _canonical_rule(rule: Any) -> str:
    """Canonical allocation-rule name accepted by :class:`~wefnexus.nexus.NexusModel`."""
    if not isinstance(rule, str) or not rule.strip():
        raise ValueError(
            f"allocation rule must be a non-empty string, got {rule!r}; "
            f"expected one of {list(ALLOCATION_RULES)}"
        )
    key = rule.strip().lower().replace("-", "_").replace(" ", "_")
    key = RULE_ALIASES.get(key, key)
    if key not in ALLOCATION_RULES:
        raise ValueError(f"unknown allocation rule {rule!r}; expected one of {list(ALLOCATION_RULES)}")
    return key


def _check_bool(value: Any, label: str) -> bool:
    if not isinstance(value, (bool,)) and value not in (0, 1):
        raise ValueError(f"{label} must be a bool, got {value!r}")
    return bool(value)


def validate_scenario(scenario: Scenario) -> Scenario:
    """Validate a :class:`~wefnexus.models.Scenario` and return a clean copy.

    Parameters
    ----------
    scenario : Scenario
        The scenario to check.  It is never mutated.

    Returns
    -------
    Scenario
        A new object with the same drivers; ``allocation_rule`` is replaced
        by its canonical name (aliases of
        :data:`wefnexus.allocation.RULE_ALIASES` resolved, case folded) and
        ``drought_years`` by a sorted, de-duplicated list.

    Raises
    ------
    ValueError
        On a non-``Scenario`` argument; an empty ``name``; a non-integer
        ``start_year``; ``years < 1``; a ``flow_change_pct_by_end`` or
        ``irrigated_area_change_pct_by_end`` below -100 %; a growth rate
        ``<= -1``; an ``irrigation_efficiency_target`` outside ``(0, 1]``;
        a ``renewable_share_target`` outside ``[0, 1]``; an unknown
        ``allocation_rule``; a negative ``seed``; non-integer or negative
        ``drought_years``; a ``drought_severity`` outside ``[0, 1]``; or
        non-boolean ``cooperation`` / ``stochastic`` flags.  Non-finite
        numbers are rejected everywhere.

    Notes
    -----
    The checks mirror what :class:`~wefnexus.nexus.NexusModel` needs so
    that a bad scenario fails early, at construction, with a message naming
    the field, rather than deep inside a multi-year run.

    Examples
    --------
    >>> validate_scenario(Scenario(name="x", years=3, allocation_rule="Contested-Garment")).allocation_rule
    'talmud'
    """
    if not isinstance(scenario, Scenario):
        raise ValueError(f"scenario must be a wefnexus.models.Scenario, got {type(scenario).__name__}")
    if not isinstance(scenario.name, str) or not scenario.name.strip():
        raise ValueError(f"scenario.name must be a non-empty string, got {scenario.name!r}")
    if not isinstance(scenario.description, str):
        raise ValueError(f"scenario.description must be a string, got {scenario.description!r}")
    changes: Dict[str, Any] = {
        "start_year": _as_int(scenario.start_year, "scenario.start_year"),
        "years": _check_years(scenario.years),
        "flow_change_pct_by_end": _check_pct_change(scenario.flow_change_pct_by_end, "scenario.flow_change_pct_by_end"),
        "population_growth_rate": _check_rate(scenario.population_growth_rate, "scenario.population_growth_rate"),
        "gdp_growth_rate": _check_rate(scenario.gdp_growth_rate, "scenario.gdp_growth_rate"),
        "demand_growth_rate": _check_rate(scenario.demand_growth_rate, "scenario.demand_growth_rate"),
        "energy_demand_growth_rate": _check_rate(scenario.energy_demand_growth_rate, "scenario.energy_demand_growth_rate"),
        "irrigated_area_change_pct_by_end": _check_pct_change(
            scenario.irrigated_area_change_pct_by_end, "scenario.irrigated_area_change_pct_by_end"
        ),
        "irrigation_efficiency_target": (
            None
            if scenario.irrigation_efficiency_target is None
            else _check_fraction(scenario.irrigation_efficiency_target, "scenario.irrigation_efficiency_target", lo_open=True)
        ),
        "renewable_share_target": (
            None
            if scenario.renewable_share_target is None
            else _check_fraction(scenario.renewable_share_target, "scenario.renewable_share_target")
        ),
        "cooperation": _check_bool(scenario.cooperation, "scenario.cooperation"),
        "allocation_rule": _canonical_rule(scenario.allocation_rule),
        "stochastic": _check_bool(scenario.stochastic, "scenario.stochastic"),
        "seed": _as_int(scenario.seed, "scenario.seed"),
        "drought_years": _check_droughts(scenario.drought_years),
        "drought_severity": _check_fraction(scenario.drought_severity, "scenario.drought_severity"),
    }
    if changes["seed"] < 0:
        raise ValueError(f"scenario.seed must be >= 0, got {changes['seed']}")
    return replace(scenario, **changes)


# --------------------------------------------------------------------------- #
# Factory plumbing
# --------------------------------------------------------------------------- #
def _build(factory: str, name: str, description: str, fields: Dict[str, Any], overrides: Dict[str, Any]) -> Scenario:
    """Assemble, override-check and validate a scenario.

    ``fields`` are the Scenario fields set by the factory's own parameters;
    ``overrides`` are extra keyword arguments naming other Scenario fields.
    """
    unknown = [k for k in overrides if k not in SCENARIO_FIELDS]
    if unknown:
        raise ValueError(
            f"{factory}(): unknown Scenario field(s) {unknown}; "
            f"Scenario fields are {list(SCENARIO_FIELDS)}"
        )
    clash = [k for k in overrides if k in fields and k not in ("name", "description")]
    if clash:
        raise ValueError(
            f"{factory}(): {clash} are set by the factory's own parameters and cannot be "
            "overridden directly; use the factory's parameters instead"
        )
    sc = Scenario(name=name, description=description, **fields)
    if overrides:
        sc = replace(sc, **overrides)
    return validate_scenario(sc)


def _climate_fields(flow_change_pct: Any, droughts: Any, drought_severity: Any) -> Dict[str, Any]:
    return {
        "flow_change_pct_by_end": _check_pct_change(flow_change_pct, "flow_change_pct"),
        "drought_years": _check_droughts(droughts),
        "drought_severity": _check_fraction(drought_severity, "drought_severity"),
    }


def _growth_fields(
    population_growth: Any,
    demand_growth: Any,
    energy_growth: Any,
    irrigated_area_change_pct: Any,
    gdp_growth: Any,
) -> Dict[str, Any]:
    demand = _check_rate(demand_growth, "demand_growth")
    return {
        "population_growth_rate": _check_rate(population_growth, "population_growth"),
        "demand_growth_rate": demand,
        "energy_demand_growth_rate": _check_rate(energy_growth, "energy_growth"),
        "irrigated_area_change_pct_by_end": _check_pct_change(irrigated_area_change_pct, "irrigated_area_change_pct"),
        "gdp_growth_rate": demand if gdp_growth is None else _check_rate(gdp_growth, "gdp_growth"),
    }


def _efficiency_fields(irrigation_efficiency_target: Any, renewable_share_target: Any) -> Dict[str, Any]:
    return {
        "irrigation_efficiency_target": (
            None
            if irrigation_efficiency_target is None
            else _check_fraction(irrigation_efficiency_target, "irrigation_efficiency_target", lo_open=True)
        ),
        "renewable_share_target": (
            None
            if renewable_share_target is None
            else _check_fraction(renewable_share_target, "renewable_share_target")
        ),
    }


def _pct(v: float) -> str:
    return f"{v:+.0f} %" if float(v).is_integer() else f"{v:+.1f} %"


def _rate_text(v: float) -> str:
    return f"{100.0 * v:.1f} %/yr"


def _describe_climate(f: Dict[str, Any], years: int) -> str:
    droughts = f["drought_years"]
    in_horizon = [d for d in droughts if d < years]
    text = f"natural flow changes linearly by {_pct(f['flow_change_pct_by_end'])} by the final year"
    if in_horizon:
        text += (
            f", with drought years (flow -{100.0 * f['drought_severity']:.0f} %) at year offsets "
            f"{in_horizon}"
        )
    else:
        text += ", without drought years inside the horizon"
    return text


def _describe_growth(f: Dict[str, Any]) -> str:
    return (
        f"population grows {_rate_text(f['population_growth_rate'])}, municipal/industrial water demand "
        f"{_rate_text(f['demand_growth_rate'])}, electricity demand {_rate_text(f['energy_demand_growth_rate'])}, "
        f"GDP {_rate_text(f['gdp_growth_rate'])}; irrigated area changes by "
        f"{_pct(f['irrigated_area_change_pct_by_end'])} by the final year"
    )


def _describe_efficiency(f: Dict[str, Any]) -> str:
    parts = []
    if f["irrigation_efficiency_target"] is not None:
        parts.append(f"irrigation efficiency rises linearly to {f['irrigation_efficiency_target']:.2f}")
    if f["renewable_share_target"] is not None:
        parts.append(f"renewable electricity share rises linearly to {f['renewable_share_target']:.2f}")
    return " and ".join(parts) if parts else "no efficiency or renewable targets"


def _describe_rule(rule: str) -> str:
    if rule == "treaty":
        return "the riparians cooperate under the existing fixed-volume treaty entitlements"
    if rule == "upstream_priority":
        return "no sharing arrangement: the upstream riparian withdraws first"
    return (
        f"the riparians cooperate and the basin's consumable water (natural flow plus carried storage, "
        f"net of environmental flows) is shared by the {rule!r} bankruptcy rule applied to their "
        "consumptive claims"
    )


# --------------------------------------------------------------------------- #
# Scenario factories
# --------------------------------------------------------------------------- #
def baseline(years: int = DEFAULT_YEARS, **overrides: Any) -> Scenario:
    """Business-as-usual reference scenario.

    Natural flow, population, demands, irrigated area, irrigation
    efficiency and energy mix are held constant over the horizon; the
    riparians cooperate under the basin's existing treaty entitlements
    (``allocation_rule="treaty"``).

    Parameters
    ----------
    years : int, optional
        Horizon length in years (>= 1).  Default 25.
    **overrides
        Any other :class:`~wefnexus.models.Scenario` field (e.g.
        ``stochastic=True``, ``seed=7``, ``start_year=2030``, ``name=...``).

    Returns
    -------
    Scenario
        Named ``"baseline"`` with a descriptive ``description``.

    Raises
    ------
    ValueError
        On an invalid horizon or override (unknown field, or a field the
        factory sets itself).

    References
    ----------
    Mahmoud et al. (2009) on reference scenarios in environmental decision
    support; Yates et al. (2005) for the WEAP "reference" scenario concept.

    Examples
    --------
    >>> baseline(years=5).years, baseline().allocation_rule, baseline().cooperation
    (5, 'treaty', True)
    """
    n = _check_years(years)
    fields = {"years": n, "cooperation": True, "allocation_rule": "treaty"}
    description = (
        f"Business as usual over {n} years: natural flow, population, water and energy demands, "
        f"irrigated area and efficiencies held at current levels; {_describe_rule('treaty')}."
    )
    return _build("baseline", "baseline", description, fields, overrides)


def climate_change(
    years: int = DEFAULT_YEARS,
    flow_change_pct: float = DEFAULT_FLOW_CHANGE_PCT,
    droughts: Iterable[int] = DEFAULT_DROUGHT_YEARS,
    drought_severity: float = DEFAULT_DROUGHT_SEVERITY,
    **overrides: Any,
) -> Scenario:
    """Climate-change scenario: declining natural flow with discrete droughts.

    The natural flow multiplier falls linearly from 1 to
    ``1 + flow_change_pct / 100`` by the final year
    (:meth:`~wefnexus.models.Scenario.flow_factor`), and is further reduced
    by ``drought_severity`` in each drought year.  Demands are held
    constant; cooperation under the existing treaty.

    Parameters
    ----------
    years : int, optional
        Horizon length in years.  Default 25.
    flow_change_pct : float, optional
        Percent change of mean natural flow reached by the final year
        (negative = drier); must be >= -100.  Default -20.
    droughts : iterable of int, optional
        0-based year offsets of drought years; offsets beyond the horizon
        are kept but never occur.  Default ``(8, 15, 22)``.
    drought_severity : float, optional
        Fractional flow reduction in a drought year, in ``[0, 1]``.
        Default 0.4.
    **overrides
        Any other :class:`~wefnexus.models.Scenario` field.

    Returns
    -------
    Scenario
        Named ``"climate_change"``.

    Raises
    ------
    ValueError
        On invalid parameters or overrides.

    References
    ----------
    Moss et al. (2010); Cooley & Gleick (2011) on climate-proofing
    transboundary agreements; Dinar et al. (2015) on flow variability.

    Examples
    --------
    >>> sc = climate_change(years=11, flow_change_pct=-20, droughts=(5,))
    >>> round(sc.flow_factor(10), 3), round(sc.flow_factor(5), 3)
    (0.8, 0.54)
    """
    n = _check_years(years)
    fields = {"years": n, "cooperation": True, "allocation_rule": "treaty"}
    fields.update(_climate_fields(flow_change_pct, droughts, drought_severity))
    description = (
        f"Climate change over {n} years: {_describe_climate(fields, n)}; demands held constant; "
        f"{_describe_rule('treaty')}."
    )
    return _build("climate_change", "climate_change", description, fields, overrides)


def growth(
    years: int = DEFAULT_YEARS,
    population_growth: float = DEFAULT_POPULATION_GROWTH,
    demand_growth: float = DEFAULT_DEMAND_GROWTH,
    energy_growth: float = DEFAULT_ENERGY_GROWTH,
    irrigated_area_change_pct: float = DEFAULT_IRRIGATED_AREA_CHANGE_PCT,
    gdp_growth: Optional[float] = None,
    **overrides: Any,
) -> Scenario:
    """Socio-economic growth scenario (demand-side pressure).

    Population, municipal/industrial (and energy-sector) water demand and
    electricity demand grow at compound annual rates; irrigated area
    expands linearly to ``irrigated_area_change_pct`` by the final year.
    Natural flow is unchanged; cooperation under the existing treaty.

    Parameters
    ----------
    years : int, optional
        Horizon length in years.  Default 25.
    population_growth : float, optional
        Population growth, fraction per year (> -1).  Default 0.015.
    demand_growth : float, optional
        Municipal + industrial (+ energy-sector water) demand growth,
        fraction per year.  Default 0.02.
    energy_growth : float, optional
        Electricity demand growth, fraction per year.  Default 0.03.
    irrigated_area_change_pct : float, optional
        Percent change of irrigated area by the final year (>= -100).
        Default +20.
    gdp_growth : float, optional
        GDP growth, fraction per year; defaults to ``demand_growth``
        (economic growth drives municipal/industrial demand).
    **overrides
        Any other :class:`~wefnexus.models.Scenario` field.

    Returns
    -------
    Scenario
        Named ``"growth"``.

    Raises
    ------
    ValueError
        On invalid parameters or overrides.

    References
    ----------
    O'Neill et al. (2014) shared socio-economic pathways; Yates et al.
    (2005) demand growth in WEAP.

    Examples
    --------
    >>> sc = growth(years=3, population_growth=0.1, irrigated_area_change_pct=50)
    >>> round(sc.population_factor(2), 3), sc.irrigated_area_factor(2)
    (1.21, 1.5)
    """
    n = _check_years(years)
    fields = {"years": n, "cooperation": True, "allocation_rule": "treaty"}
    fields.update(_growth_fields(population_growth, demand_growth, energy_growth, irrigated_area_change_pct, gdp_growth))
    description = (
        f"Socio-economic growth over {n} years: {_describe_growth(fields)}; natural flow unchanged; "
        f"{_describe_rule('treaty')}."
    )
    return _build("growth", "growth", description, fields, overrides)


def efficiency(
    years: int = DEFAULT_YEARS,
    irrigation_efficiency_target: Optional[float] = DEFAULT_IRRIGATION_EFFICIENCY_TARGET,
    renewable_share_target: Optional[float] = DEFAULT_RENEWABLE_SHARE_TARGET,
    **overrides: Any,
) -> Scenario:
    """Efficiency (response) scenario: better irrigation, more renewables.

    Each riparian's irrigation efficiency moves linearly from its own value
    to ``irrigation_efficiency_target`` (lowering gross irrigation demand,
    FAO-56) and its renewable electricity share to
    ``renewable_share_target`` (lowering emissions) by the final year.
    Flows and demand drivers are otherwise unchanged; cooperation under the
    existing treaty.

    Parameters
    ----------
    years : int, optional
        Horizon length in years.  Default 25.
    irrigation_efficiency_target : float or None, optional
        Final irrigation efficiency in ``(0, 1]`` (``None`` = unchanged).
        Default 0.7 (sprinkler-class efficiency).
    renewable_share_target : float or None, optional
        Final renewable share of electricity demand in ``[0, 1]`` (``None``
        = unchanged).  Default 0.5.
    **overrides
        Any other :class:`~wefnexus.models.Scenario` field.

    Returns
    -------
    Scenario
        Named ``"efficiency"``.

    Raises
    ------
    ValueError
        On invalid parameters or overrides.

    References
    ----------
    Allen et al. (1998) FAO-56 irrigation efficiency; IRENA (2021)
    renewable targets; Hoff (2011) nexus efficiency responses.

    Examples
    --------
    >>> sc = efficiency(years=5, irrigation_efficiency_target=0.9)
    >>> sc.irrigation_efficiency(0.5, 4), sc.irrigation_efficiency(0.5, 0)
    (0.9, 0.5)
    """
    n = _check_years(years)
    fields = {"years": n, "cooperation": True, "allocation_rule": "treaty"}
    fields.update(_efficiency_fields(irrigation_efficiency_target, renewable_share_target))
    description = (
        f"Efficiency measures over {n} years: {_describe_efficiency(fields)}; flows and demand drivers "
        f"unchanged; {_describe_rule('treaty')}."
    )
    return _build("efficiency", "efficiency", description, fields, overrides)


def unilateral(years: int = DEFAULT_YEARS, **overrides: Any) -> Scenario:
    """Non-cooperative scenario: upstream priority, no entitlements.

    ``cooperation`` is False, so :class:`~wefnexus.nexus.NexusModel`
    applies no withdrawal caps and each riparian withdraws whatever is
    physically available before passing the remainder downstream (the
    "Harmon doctrine" / absolute territorial sovereignty outcome, Ansink &
    Weikard 2012).  Drivers are as in :func:`baseline`.

    Parameters
    ----------
    years : int, optional
        Horizon length in years.  Default 25.
    **overrides
        Any other :class:`~wefnexus.models.Scenario` field except
        ``cooperation`` / ``allocation_rule`` (set by the factory).

    Returns
    -------
    Scenario
        Named ``"unilateral"`` with ``cooperation=False`` and
        ``allocation_rule="upstream_priority"``.

    Raises
    ------
    ValueError
        On invalid parameters or overrides.

    References
    ----------
    Ansink, E. & Weikard, H.-P. (2012). Sequential sharing rules for river
    sharing problems. *Social Choice and Welfare* 38, 187-210; Wolf (2007).

    Examples
    --------
    >>> unilateral(years=2).cooperation
    False
    """
    n = _check_years(years)
    fields = {"years": n, "cooperation": False, "allocation_rule": "upstream_priority"}
    description = (
        f"Unilateral development over {n} years: {_describe_rule('upstream_priority')}; "
        "drivers held at current levels."
    )
    return _build("unilateral", "unilateral", description, fields, overrides)


def cooperative(years: int = DEFAULT_YEARS, rule: str = DEFAULT_COOPERATIVE_RULE, **overrides: Any) -> Scenario:
    """Cooperative scenario: a bankruptcy sharing rule divides the basin's water.

    ``cooperation`` is True and ``allocation_rule`` is ``rule``.  For a
    bankruptcy rule, :class:`~wefnexus.nexus.NexusModel` divides the water
    the basin can consume in the year (natural flow plus carried reservoir
    storage, net of environmental flows) among the riparians' consumptive
    river-water claims with the rule and turns each award into a cap on
    surface withdrawal (Mianabadi et al. 2014); ``"treaty"`` and
    ``"upstream_priority"`` are accepted as well.  Drivers are as in
    :func:`baseline`.

    Parameters
    ----------
    years : int, optional
        Horizon length in years.  Default 25.
    rule : str, optional
        One of :data:`wefnexus.nexus.ALLOCATION_RULES` or an alias of
        :data:`wefnexus.allocation.RULE_ALIASES` (case-insensitive).
        Default ``"talmud"`` (Aumann & Maschler 1985).
    **overrides
        Any other :class:`~wefnexus.models.Scenario` field.

    Returns
    -------
    Scenario
        Named ``"cooperative"``; ``allocation_rule`` holds the canonical
        rule name.  Pass ``name=...`` to run several rules side by side in
        :func:`run_scenarios` (result keys must be unique).

    Raises
    ------
    ValueError
        On an unknown rule or invalid parameters / overrides.

    References
    ----------
    Aumann & Maschler (1985); Mianabadi et al. (2014); Ansink & Ruijs
    (2008) on the stability of sharing agreements.

    Examples
    --------
    >>> cooperative(rule="Contested garment").allocation_rule
    'talmud'
    """
    n = _check_years(years)
    canonical = _canonical_rule(rule)
    fields = {"years": n, "cooperation": True, "allocation_rule": canonical}
    description = f"Cooperative sharing over {n} years: {_describe_rule(canonical)}; drivers held at current levels."
    return _build("cooperative", "cooperative", description, fields, overrides)


def combined_stress(
    years: int = DEFAULT_YEARS,
    rule: str = "treaty",
    *,
    climate_kw: Optional[Mapping[str, Any]] = None,
    growth_kw: Optional[Mapping[str, Any]] = None,
    **overrides: Any,
) -> Scenario:
    """Combined stress storyline: climate change + growth under the fixed treaty.

    The drivers of :func:`climate_change` and :func:`growth` act together
    while the riparians keep cooperating under their fixed-volume treaty
    entitlements - the classic "fixed allocations meet a changing
    hydrology" problem of Drieschova et al. (2008) and Cooley & Gleick
    (2011).

    Parameters
    ----------
    years : int, optional
        Horizon length in years.  Default 25.
    rule : str, optional
        Sharing arrangement (see :func:`cooperative`).  Default
        ``"treaty"``.
    climate_kw : mapping, optional
        Keyword arguments of :func:`climate_change` (``flow_change_pct``,
        ``droughts``, ``drought_severity``) overriding its defaults.
    growth_kw : mapping, optional
        Keyword arguments of :func:`growth` (``population_growth``,
        ``demand_growth``, ``energy_growth``, ``irrigated_area_change_pct``,
        ``gdp_growth``) overriding its defaults.
    **overrides
        Any other :class:`~wefnexus.models.Scenario` field.

    Returns
    -------
    Scenario
        Named ``"combined_stress"``.

    Raises
    ------
    ValueError
        On invalid parameters, unknown ``climate_kw`` / ``growth_kw`` keys
        or invalid overrides.

    Examples
    --------
    >>> sc = combined_stress(years=10, climate_kw={"flow_change_pct": -30})
    >>> sc.flow_change_pct_by_end, sc.population_growth_rate, sc.allocation_rule
    (-30.0, 0.015, 'treaty')
    """
    n = _check_years(years)
    canonical = _canonical_rule(rule)
    fields = {"years": n, "cooperation": True, "allocation_rule": canonical}
    fields.update(_climate_fields(**_component_kw("climate_kw", climate_kw, _CLIMATE_DEFAULTS)))
    fields.update(_growth_fields(**_component_kw("growth_kw", growth_kw, _GROWTH_DEFAULTS)))
    description = (
        f"Combined stress over {n} years: {_describe_climate(fields, n)}; {_describe_growth(fields)}; "
        f"no efficiency measures; {_describe_rule(canonical)}."
    )
    return _build("combined_stress", "combined_stress", description, fields, overrides)


def combined_adaptation(
    years: int = DEFAULT_YEARS,
    rule: str = DEFAULT_ADAPTATION_RULE,
    *,
    climate_kw: Optional[Mapping[str, Any]] = None,
    growth_kw: Optional[Mapping[str, Any]] = None,
    efficiency_kw: Optional[Mapping[str, Any]] = None,
    **overrides: Any,
) -> Scenario:
    """Combined adaptation storyline: climate change + growth + efficiency + Talmud.

    Same climate and growth pressures as :func:`combined_stress`, plus the
    efficiency and renewable responses of :func:`efficiency`, under the
    sharing arrangement ``rule`` - by default the Talmud rule of
    ARCHITECTURE.md section 9 (Aumann & Maschler 1985): the demand- and
    supply-side measures are paired with a flexible, claims-based
    re-sharing of the basin's water in place of the fixed treaty volumes
    (Drieschova et al. 2008; Cooley & Gleick 2011).

    As explained in the module notes, :class:`~wefnexus.nexus.NexusModel`
    applies the rule to a consumptive-use estate that counts the carried
    reservoir storage, so it rations only under physical scarcity; on the
    stylised basin the package serves every riparian in full.  Pass
    ``rule="treaty"`` for the fixed-treaty variant, which differs from
    :func:`combined_stress` only by its efficiency measures.

    Parameters
    ----------
    years : int, optional
        Horizon length in years.  Default 25.
    rule : str, optional
        Sharing arrangement (see :func:`cooperative`).  Default
        ``"talmud"`` (:data:`DEFAULT_ADAPTATION_RULE`).
    climate_kw, growth_kw, efficiency_kw : mapping, optional
        Keyword arguments of :func:`climate_change`, :func:`growth` and
        :func:`efficiency` overriding their defaults.
    **overrides
        Any other :class:`~wefnexus.models.Scenario` field.

    Returns
    -------
    Scenario
        Named ``"combined_adaptation"``.

    Raises
    ------
    ValueError
        On invalid parameters, unknown component keys or invalid overrides.

    References
    ----------
    Cooley & Gleick (2011); Hoff (2011) on nexus efficiency responses;
    Aumann & Maschler (1985) and Mianabadi et al. (2014) for the Talmud
    rule on a water-sharing estate.

    Examples
    --------
    >>> sc = combined_adaptation(years=10)
    >>> sc.irrigation_efficiency_target, sc.renewable_share_target, sc.allocation_rule
    (0.7, 0.5, 'talmud')
    >>> combined_adaptation(rule="treaty").allocation_rule
    'treaty'
    """
    n = _check_years(years)
    canonical = _canonical_rule(rule)
    fields = {"years": n, "cooperation": True, "allocation_rule": canonical}
    fields.update(_climate_fields(**_component_kw("climate_kw", climate_kw, _CLIMATE_DEFAULTS)))
    fields.update(_growth_fields(**_component_kw("growth_kw", growth_kw, _GROWTH_DEFAULTS)))
    fields.update(_efficiency_fields(**_component_kw("efficiency_kw", efficiency_kw, _EFFICIENCY_DEFAULTS)))
    description = (
        f"Combined adaptation over {n} years: {_describe_climate(fields, n)}; {_describe_growth(fields)}; "
        f"{_describe_efficiency(fields)}; {_describe_rule(canonical)}."
    )
    return _build("combined_adaptation", "combined_adaptation", description, fields, overrides)


_CLIMATE_DEFAULTS: Dict[str, Any] = {
    "flow_change_pct": DEFAULT_FLOW_CHANGE_PCT,
    "droughts": DEFAULT_DROUGHT_YEARS,
    "drought_severity": DEFAULT_DROUGHT_SEVERITY,
}
_GROWTH_DEFAULTS: Dict[str, Any] = {
    "population_growth": DEFAULT_POPULATION_GROWTH,
    "demand_growth": DEFAULT_DEMAND_GROWTH,
    "energy_growth": DEFAULT_ENERGY_GROWTH,
    "irrigated_area_change_pct": DEFAULT_IRRIGATED_AREA_CHANGE_PCT,
    "gdp_growth": None,
}
_EFFICIENCY_DEFAULTS: Dict[str, Any] = {
    "irrigation_efficiency_target": DEFAULT_IRRIGATION_EFFICIENCY_TARGET,
    "renewable_share_target": DEFAULT_RENEWABLE_SHARE_TARGET,
}


def _component_kw(label: str, given: Optional[Mapping[str, Any]], defaults: Mapping[str, Any]) -> Dict[str, Any]:
    """Merge a component factory's keyword overrides into its defaults."""
    if given is None:
        return dict(defaults)
    if not isinstance(given, Mapping):
        raise ValueError(f"{label} must be a mapping of keyword arguments, got {type(given).__name__}")
    unknown = [k for k in given if k not in defaults]
    if unknown:
        raise ValueError(f"{label}: unknown key(s) {unknown}; expected a subset of {list(defaults)}")
    merged = dict(defaults)
    merged.update(given)
    return merged


#: Registry of scenario factories, keyed by scenario name.
SCENARIOS: Dict[str, Callable[..., Scenario]] = {
    "baseline": baseline,
    "climate_change": climate_change,
    "growth": growth,
    "efficiency": efficiency,
    "unilateral": unilateral,
    "cooperative": cooperative,
    "combined_stress": combined_stress,
    "combined_adaptation": combined_adaptation,
}


def scenario_names() -> List[str]:
    """Names of the library scenarios, in :data:`SCENARIOS` order.

    Returns
    -------
    list of str

    Examples
    --------
    >>> scenario_names()[0], len(scenario_names())
    ('baseline', 8)
    """
    return list(SCENARIOS)


def _normalise_name(name: Any) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"scenario name must be a non-empty string, got {name!r}")
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def get_scenario(name: str, **kw: Any) -> Scenario:
    """Build a library scenario by name.

    Parameters
    ----------
    name : str
        A key of :data:`SCENARIOS` (case-insensitive; hyphens and spaces
        are read as underscores, e.g. ``"Climate-Change"``).
    **kw
        Passed to the factory (its own parameters such as ``years`` or
        ``rule``, and any :class:`~wefnexus.models.Scenario` field
        override).

    Returns
    -------
    Scenario

    Raises
    ------
    ValueError
        On an unknown name or invalid factory arguments.

    Examples
    --------
    >>> get_scenario("cooperative", years=3, rule="cea").allocation_rule
    'cea'
    """
    key = _normalise_name(name)
    if key not in SCENARIOS:
        raise ValueError(f"unknown scenario {name!r}; available scenarios: {scenario_names()}")
    return SCENARIOS[key](**kw)


def all_scenarios(years: int = DEFAULT_YEARS, **kw: Any) -> List[Scenario]:
    """Build every library scenario with a common horizon.

    Parameters
    ----------
    years : int, optional
        Horizon length passed to every factory.  Default 25.
    **kw
        :class:`~wefnexus.models.Scenario` field overrides applied to every
        factory (e.g. ``stochastic=True, seed=1``).  Factory-specific
        parameters are not accepted here because not every factory takes
        them.

    Returns
    -------
    list of Scenario
        In :data:`SCENARIOS` order, with unique names.

    Raises
    ------
    ValueError
        Propagated from the factories.

    Examples
    --------
    >>> [s.name for s in all_scenarios(years=2)][:3]
    ['baseline', 'climate_change', 'growth']
    """
    return [factory(years=years, **kw) for factory in SCENARIOS.values()]


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def _check_model_kw(model_kw: Mapping[str, Any]) -> None:
    allowed = [p for p in inspect.signature(NexusModel.__init__).parameters if p not in ("self", "basin", "scenario")]
    unknown = [k for k in model_kw if k not in allowed]
    if unknown:
        raise ValueError(f"unknown NexusModel keyword(s) {unknown}; accepted: {allowed}")


def _normalise_scenarios(scenarios: Any) -> List[Tuple[str, Scenario]]:
    """``(key, Scenario)`` pairs from a Scenario, a name, a sequence or a mapping."""
    if isinstance(scenarios, (Scenario, str)):
        scenarios = [scenarios]
    if isinstance(scenarios, Mapping):
        pairs = [(k, v) for k, v in scenarios.items()]
        keyed = True
    elif isinstance(scenarios, Iterable) and not isinstance(scenarios, (bytes,)):
        pairs = [(None, v) for v in scenarios]
        keyed = False
    else:
        raise ValueError(
            "scenarios must be a sequence of Scenario objects (or library names), or a mapping "
            f"{{key: Scenario}}, got {type(scenarios).__name__}"
        )
    if not pairs:
        raise ValueError("scenarios must not be empty")
    out: List[Tuple[str, Scenario]] = []
    seen: Dict[str, int] = {}
    for idx, (key, sc) in enumerate(pairs):
        if isinstance(sc, str):
            sc = get_scenario(sc)
        if not isinstance(sc, Scenario):
            raise ValueError(
                f"scenarios[{idx}] must be a wefnexus.models.Scenario or a library scenario name, "
                f"got {type(sc).__name__}"
            )
        sc = validate_scenario(sc)
        name = str(key) if keyed else sc.name
        if not name:
            raise ValueError(f"scenarios[{idx}] has an empty name")
        if name in seen:
            raise ValueError(
                f"duplicate scenario name {name!r} (positions {seen[name]} and {idx}); give each scenario a "
                "unique name (e.g. cooperative(rule='cea', name='cooperative_cea'))"
            )
        seen[name] = idx
        out.append((name, sc))
    return out


def run_scenarios(basin: Basin, scenarios: Sequence[Scenario], **model_kw: Any) -> Dict[str, NexusResult]:
    """Simulate several scenarios on fresh copies of one basin.

    Parameters
    ----------
    basin : Basin
        The basin; every scenario runs on its own ``basin.copy()``, so the
        argument is never touched and runs cannot interfere.
    scenarios : sequence of Scenario
        Scenarios to run, in order.  Library names (keys of
        :data:`SCENARIOS`, built with their defaults) are accepted in place
        of objects, as is a mapping ``{key: Scenario}`` (its keys name the
        results) or a single scenario.  Names must be unique.
    **model_kw
        Keyword arguments of :class:`~wefnexus.nexus.NexusModel` applied
        to every run (``allocation_rule`` override,
        ``reservoir_refill_fraction``, ``flow_factors``).

    Returns
    -------
    dict
        ``{scenario name: NexusResult}`` in input order.

    Raises
    ------
    ValueError
        On a non-``Basin`` argument, an empty or ill-typed scenario list,
        duplicate names, an unknown library name, an invalid scenario or an
        unknown model keyword; model errors propagate.

    Notes
    -----
    Runs are deterministic: a stochastic scenario draws its flow factors
    from ``numpy.random.default_rng(scenario.seed)``.

    Examples
    --------
    >>> from wefnexus.data import example_basin
    >>> res = run_scenarios(example_basin(), [baseline(years=2), unilateral(years=2)])
    >>> list(res), res["baseline"].n_years
    (['baseline', 'unilateral'], 2)
    """
    if not isinstance(basin, Basin):
        raise ValueError(f"basin must be a wefnexus.models.Basin, got {type(basin).__name__}")
    _check_model_kw(model_kw)
    items = _normalise_scenarios(scenarios)
    results: Dict[str, NexusResult] = {}
    for key, sc in items:
        results[key] = NexusModel(basin.copy(), sc, **model_kw).run()
    return results


# --------------------------------------------------------------------------- #
# Comparison tables
# --------------------------------------------------------------------------- #
def _normalise_results(results: Any) -> List[Tuple[str, NexusResult]]:
    if isinstance(results, NexusResult):
        results = [results]
    if isinstance(results, Mapping):
        pairs = [(str(k), v) for k, v in results.items()]
    elif isinstance(results, Iterable) and not isinstance(results, (str, bytes)):
        pairs = [(getattr(v, "scenario", None) and v.scenario.name, v) for v in results]
    else:
        raise ValueError(f"results must be a mapping {{name: NexusResult}}, got {type(results).__name__}")
    if not pairs:
        raise ValueError("results must not be empty")
    out: List[Tuple[str, NexusResult]] = []
    seen = set()
    for idx, (key, res) in enumerate(pairs):
        if not isinstance(res, NexusResult):
            raise ValueError(f"results[{key if key is not None else idx}] must be a NexusResult, got {type(res).__name__}")
        if not res.records:
            raise ValueError(f"result {key!r} holds no records")
        key = str(key)
        if key in seen:
            raise ValueError(f"duplicate result name {key!r}")
        seen.add(key)
        out.append((key, res))
    return out


def _row(key: str, riparian: str, res: NexusResult, summary: Mapping[str, Any], basin: bool) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "scenario": key,
        "riparian": riparian,
        "allocation_rule": res.allocation_rule,
        "years": int(res.n_years),
    }
    for col in COMPARISON_COLUMNS:
        value = summary[_SUMMARY_KEYS[col]]
        row[col] = int(value) if col == "years_env_flow_unmet" else float(value)
    for col in BASIN_COLUMNS:
        if basin:
            value = summary.get(_SUMMARY_KEYS[col])
            row[col] = None if value is None else float(value)
        else:
            row[col] = None
    return row


def comparison_table(results: Dict[str, NexusResult]) -> List[Dict[str, Any]]:
    """Condense scenario results into one row per (scenario, riparian) plus a basin row.

    Parameters
    ----------
    results : dict
        ``{scenario name: NexusResult}`` as returned by :func:`run_scenarios`
        (a sequence of results, keyed by their scenario names, is accepted
        too).

    Returns
    -------
    list of dict
        For every scenario, in input order: one row per riparian (basin
        order) followed by a basin row whose ``"riparian"`` is
        :data:`wefnexus.nexus.BASIN_KEY` (``"BASIN"``).  Every row has the
        same keys, :data:`TABLE_COLUMNS`:

        * :data:`KEY_COLUMNS` ``scenario``, ``riparian``;
        * :data:`INFO_COLUMNS` ``allocation_rule`` (canonical rule
          applied), ``years`` (int);
        * :data:`COMPARISON_COLUMNS`, numeric in every row, from
          :meth:`~wefnexus.nexus.NexusResult.summary`: ``mean_supply_ratio``,
          ``min_supply_ratio``, ``supply_reliability`` (Hashimoto et al.
          1982), ``mean_nexus_index``, ``min_nexus_index``,
          ``mean_water_security``, ``mean_energy_security``,
          ``mean_food_security`` (indices in ``[0, 1]``),
          ``years_env_flow_unmet`` (int; basin row: years with any reach
          unmet), ``env_flow_met_share``, ``total_hydropower_gwh`` (GWh over
          the horizon), ``mean_energy_deficit_gwh`` (GWh/yr),
          ``mean_emissions_t`` (t CO2/yr), ``food_self_sufficiency`` (ratio
          of horizon totals), ``final_storage_mm3`` (Mm3, basin row: sum),
          ``mean_total_deficit_mm3`` (Mm3/yr), ``mean_water_stress_sdg642``
          (percent; may be ``inf``);
        * :data:`BASIN_COLUMNS`, numeric in the basin row and ``None`` in
          riparian rows: ``equity_index`` (``1 - Gini`` of riparian supply
          ratios), ``natural_flow_mm3`` and ``outflow_to_sea_mm3`` (Mm3/yr
          means) and ``mass_balance_error_mm3`` (max over years; ``None``
          for a result without balances).

    Raises
    ------
    ValueError
        On an empty argument, a value that is not a
        :class:`~wefnexus.nexus.NexusResult`, an empty result or duplicate
        names.

    Examples
    --------
    >>> from wefnexus.data import example_basin
    >>> rows = comparison_table(run_scenarios(example_basin(), [baseline(years=2)]))
    >>> len(rows), rows[-1]["riparian"], rows[0]["scenario"]
    (4, 'BASIN', 'baseline')
    """
    rows: List[Dict[str, Any]] = []
    for key, res in _normalise_results(results):
        summary = res.summary()
        for name in res.riparian_names():
            rows.append(_row(key, name, res, summary[name], basin=False))
        rows.append(_row(key, BASIN_KEY, res, summary[BASIN_KEY], basin=True))
    return rows


def _check_rows(rows: Any) -> List[Mapping[str, Any]]:
    if isinstance(rows, Mapping) or isinstance(rows, (str, bytes)) or not isinstance(rows, Iterable):
        raise ValueError(f"rows must be a sequence of dictionaries, got {type(rows).__name__}")
    out = list(rows)
    if not out:
        raise ValueError("rows must not be empty")
    for i, r in enumerate(out):
        if not isinstance(r, Mapping):
            raise ValueError(f"rows[{i}] must be a dictionary, got {type(r).__name__}")
    return out


def _is_number(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def scenario_differences(
    rows: Sequence[Mapping[str, Any]],
    reference: str = "baseline",
    columns: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Differences of comparison rows from a reference scenario.

    For every row whose ``"scenario"`` is not ``reference``, the numeric
    ``columns`` are replaced by ``value - reference value`` of the row with
    the same ``"riparian"`` in the reference scenario; other keys are
    copied.  Positive differences of an index or ratio mean the scenario
    does better than the reference (e.g. an adaptation gain).

    Parameters
    ----------
    rows : sequence of dict
        Rows of :func:`comparison_table`.
    reference : str, optional
        Name of the reference scenario.  Default ``"baseline"``.
    columns : sequence of str, optional
        Columns to difference; default :data:`COMPARISON_COLUMNS` plus
        :data:`BASIN_COLUMNS`.  A column that is ``None`` (or missing) in
        either row yields ``None``.

    Returns
    -------
    list of dict
        Rows in input order, without the reference scenario's own rows.

    Raises
    ------
    ValueError
        If the reference scenario is absent, a riparian of another scenario
        has no reference row, or a column is non-numeric in some row.

    Examples
    --------
    >>> from wefnexus.data import example_basin
    >>> res = run_scenarios(example_basin(), [baseline(years=2), growth(years=2)])
    >>> d = scenario_differences(comparison_table(res))
    >>> [r["scenario"] for r in d] == ["growth"] * 4, d[-1]["riparian"]
    (True, 'BASIN')
    """
    table = _check_rows(rows)
    if not isinstance(reference, str):
        raise ValueError(f"reference must be a scenario name, got {reference!r}")
    cols = list(COMPARISON_COLUMNS + BASIN_COLUMNS) if columns is None else [str(c) for c in columns]
    ref_rows: Dict[Any, Mapping[str, Any]] = {}
    for r in table:
        if r.get("scenario") == reference:
            ref_rows[r.get("riparian")] = r
    if not ref_rows:
        names = sorted({str(r.get("scenario")) for r in table})
        raise ValueError(f"reference scenario {reference!r} not found; scenarios present: {names}")
    out: List[Dict[str, Any]] = []
    for r in table:
        if r.get("scenario") == reference:
            continue
        rip = r.get("riparian")
        if rip not in ref_rows:
            raise ValueError(f"riparian {rip!r} of scenario {r.get('scenario')!r} has no row in reference {reference!r}")
        ref = ref_rows[rip]
        new = dict(r)
        for c in cols:
            a, b = r.get(c), ref.get(c)
            if a is None or b is None:
                new[c] = None
                continue
            if not _is_number(a) or not _is_number(b):
                raise ValueError(f"column {c!r} is not numeric in scenario {r.get('scenario')!r} / {reference!r}")
            new[c] = a - b
        out.append(new)
    return out


def rank_scenarios(
    rows: Sequence[Mapping[str, Any]],
    column: str = "mean_nexus_index",
    riparian: str = BASIN_KEY,
    higher_is_better: bool = True,
) -> List[Tuple[str, float]]:
    """Order scenarios by one comparison column for one riparian (or the basin).

    Parameters
    ----------
    rows : sequence of dict
        Rows of :func:`comparison_table`.
    column : str, optional
        Numeric column to rank on.  Default ``"mean_nexus_index"``.
    riparian : str, optional
        Riparian name, or :data:`wefnexus.nexus.BASIN_KEY` for the basin
        row (default).
    higher_is_better : bool, optional
        Descending order when True (default), ascending otherwise.

    Returns
    -------
    list of (str, float)
        ``(scenario, value)`` pairs, best first; ties keep input order.

    Raises
    ------
    ValueError
        If no row matches ``riparian``, or the column is missing,
        ``None`` or non-numeric (NaN counts as non-numeric) in a matching
        row.

    Examples
    --------
    >>> rows = [{"scenario": "a", "riparian": "BASIN", "x": 0.2},
    ...         {"scenario": "b", "riparian": "BASIN", "x": 0.9}]
    >>> rank_scenarios(rows, "x")
    [('b', 0.9), ('a', 0.2)]
    """
    table = _check_rows(rows)
    if not isinstance(column, str) or not column:
        raise ValueError(f"column must be a non-empty string, got {column!r}")
    picked = [r for r in table if r.get("riparian") == riparian]
    if not picked:
        present = sorted({str(r.get("riparian")) for r in table})
        raise ValueError(f"no rows for riparian {riparian!r}; rows cover {present}")
    pairs: List[Tuple[str, float]] = []
    for r in picked:
        v = r.get(column)
        if not _is_number(v) or math.isnan(float(v)):
            raise ValueError(f"column {column!r} is missing or non-numeric for scenario {r.get('scenario')!r}")
        pairs.append((str(r.get("scenario")), float(v)))
    return sorted(pairs, key=lambda p: p[1], reverse=bool(higher_is_better))


def _format_cell(value: Any) -> str:
    """Compact text rendering of one table cell."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, str):
        return value
    if isinstance(value, numbers.Integral):
        return str(int(value))
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    a = abs(v)
    if a >= 1e6:
        return f"{v:.3e}"
    if a >= 100.0:
        return f"{v:.0f}"
    if a >= 10.0:
        return f"{v:.1f}"
    return f"{v:.3f}"


def format_table(rows: Sequence[Mapping[str, Any]], columns: Optional[Sequence[str]] = None) -> str:
    """Render rows as an aligned plain-text table.

    Parameters
    ----------
    rows : sequence of dict
        Rows of :func:`comparison_table` (or any list of dictionaries with
        a common set of keys).
    columns : sequence of str, optional
        Columns to print, in order; default: the keys of the first row.  A
        column missing from a row prints as ``"-"``; a column missing from
        *every* row raises.

    Returns
    -------
    str
        Header line, a rule of dashes and one line per row.  Text columns
        are left-aligned, numeric columns right-aligned; floats print with
        three decimals below 10, one decimal below 100, no decimals below
        1e6 and in scientific notation above; ``None`` prints as ``"-"``,
        booleans as ``yes``/``no``.  Lines are joined with ``"\\n"`` (no
        trailing newline).

    Raises
    ------
    ValueError
        On empty rows, a non-dictionary row, an empty column list or a
        column present in no row.

    Examples
    --------
    >>> print(format_table([{"scenario": "baseline", "riparian": "Delta", "mean_supply_ratio": 0.98}]))
    scenario  riparian  mean_supply_ratio
    -------------------------------------
    baseline  Delta                 0.980
    """
    table = _check_rows(rows)
    if columns is None:
        cols = list(table[0].keys())
    else:
        if isinstance(columns, (str, bytes)) or not isinstance(columns, Iterable):
            raise ValueError("columns must be a sequence of column names")
        cols = [str(c) for c in columns]
        if not cols:
            raise ValueError("columns must not be empty")
    missing = [c for c in cols if not any(c in r for r in table)]
    if missing:
        raise ValueError(f"column(s) {missing} present in no row; available: {sorted({k for r in table for k in r})}")
    cells = [[_format_cell(r.get(c)) for c in cols] for r in table]
    numeric = [
        all(_is_number(r.get(c)) for r in table if r.get(c) is not None) and any(r.get(c) is not None for r in table)
        for c in cols
    ]
    widths = [max([len(c)] + [len(row[i]) for row in cells]) for i, c in enumerate(cols)]

    def line(parts: Sequence[str]) -> str:
        return "  ".join(
            (p.rjust(w) if numeric[i] else p.ljust(w)) for i, (p, w) in enumerate(zip(parts, widths))
        ).rstrip()

    header = line(cols)
    out = [header, "-" * len(header)]
    out.extend(line(row) for row in cells)
    return "\n".join(out)


def write_table_csv(rows: Sequence[Mapping[str, Any]], path: Any, columns: Optional[Sequence[str]] = None) -> str:
    """Write comparison rows to a CSV file (standard library, no pandas).

    Parameters
    ----------
    rows : sequence of dict
        Rows of :func:`comparison_table`.
    path : str or path-like
        Destination file, overwritten if it exists.
    columns : sequence of str, optional
        Columns to write, in order; default: keys of the first row.
        ``None`` values are written as empty cells.

    Returns
    -------
    str
        The path written.

    Raises
    ------
    ValueError
        On empty rows or an empty column list.
    """
    table = _check_rows(rows)
    cols = list(table[0].keys()) if columns is None else [str(c) for c in columns]
    if not cols:
        raise ValueError("columns must not be empty")
    target = os.fspath(path)
    with open(target, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for r in table:
            writer.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in cols})
    return target
