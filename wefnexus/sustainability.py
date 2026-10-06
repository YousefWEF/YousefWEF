"""Composite WEF-nexus sustainability indices and SDG-style indicators.

This *composite* module turns the raw indicators of :mod:`wefnexus.water`,
:mod:`wefnexus.energy` and :mod:`wefnexus.food` into bounded, comparable
sustainability scores:

* **normalisation** - :func:`normalise` (min-max to ``[0, 1]``, clipped,
  with a reversible direction) following the OECD/JRC handbook on composite
  indicators (Nardo et al. 2005; OECD 2008);
* **aggregation** - :func:`weighted_geometric_mean`, the *geometric*
  aggregation used by the Human Development Index since 2010 (UNDP 2010) and
  by the WEF Nexus Index of Simpson et al. (2022): it is only partially
  compensatory, so a very weak pillar drags the whole score down and cannot
  be offset by strong ones (Ebert & Welsch 2004);
* **pillar indices** - :func:`water_security_index`,
  :func:`energy_security_index` and :func:`food_security_index`, each a
  geometric mean of two to four normalised components, and the overall
  :func:`wef_nexus_index` (Hoff 2011; Bizikova et al. 2013);
* **equity** - :func:`equity_index` ``= 1 - Gini`` of a per-riparian
  distribution (Cullis & van Koppen 2007);
* **SDG indicators** - :func:`sdg_indicators` for SDG 6.4.1 (water-use
  efficiency), 6.4.2 (water stress), 6.5.2 (transboundary cooperation), 7.2
  (renewable share) and 2.1 (food self-sufficiency as a proxy for the
  prevalence of undernourishment target);
* **reporting** - :class:`SustainabilityReport` and :func:`assess`, which
  summarise a multi-year :class:`~wefnexus.nexus.NexusResult` per riparian
  and for the whole basin (means over the horizon, Hashimoto
  reliability / resilience / vulnerability, equity, SDG indicators) with a
  plain-text table.

Units
-----
Water volumes in **Mm3/yr** (1 Mm3 = 1e6 m3), energy in **GWh/yr**,
emissions in **t CO2/yr**, emission intensity in **t CO2 per GWh**, money in
**USD**, food energy in **kcal/yr**, per-capita water in **m3/person/yr**,
SDG 6.4.2 water stress in **percent**.  Every index is dimensionless and
lies in ``[0, 1]`` with **1 = best**.

Conventions
-----------
* Index functions accept ratios slightly above 1 (floating-point noise, or
  genuine surpluses such as a food self-sufficiency of 1.3) and treat them as
  1; negative values raise ``ValueError``.
* Inside the geometric mean every component is clipped to
  ``[GEOMETRIC_FLOOR, 1] = [1e-6, 1]`` so that a zero component yields a very
  small but finite index (``1e-6 ** (1/3) = 0.01`` for three equal-weighted
  pillars) instead of ``log(0)``.
* Nothing here mutates its inputs.  :func:`assess` imports
  :class:`~wefnexus.nexus.NexusResult` *lazily* (inside the function) so that
  :mod:`wefnexus.nexus` can import this module without a circular import.
* SDG 6.4.1 in :func:`assess` / :func:`assess_records` divides GDP by the
  *total freshwater withdrawal* of FAO (2018b): surface withdrawal **plus**
  groundwater used (the optional ``groundwater_used_mm3`` record field, 0
  when a record does not carry it).  Desalinated water is not a freshwater
  withdrawal and is excluded.  This is the same withdrawal convention as the
  SDG 6.4.2 stress computed by :mod:`wefnexus.nexus`.
* Environmental-flow bookkeeping: ``env_flow_met_share`` is the share of
  riparian-years in which the reach outflow met its requirement, while the
  basin ``years_env_flow_unmet`` counts *years* in which any reach fell short
  (so it never exceeds the horizon length) - the definition of
  :meth:`wefnexus.nexus.NexusResult.summary` and
  :func:`wefnexus.scenarios.comparison_table`.

References
----------
Bizikova, L., Roy, D., Swanson, D., Venema, H. D. & McCandless, M. (2013).
    *The Water-Energy-Food Security Nexus: Towards a Practical Planning and
    Decision-Support Framework for Landscape Investment and Risk
    Management.* IISD, Winnipeg.
Cullis, J. & van Koppen, B. (2007). *Applying the Gini Coefficient to
    Measure Inequality of Water Use in the Olifants River Water Management
    Area, South Africa.* IWMI Research Report 113, Colombo.
Ebert, U. & Welsch, H. (2004). Meaningful environmental indices: a social
    choice approach. *Journal of Environmental Economics and Management*
    47(2), 270-283.
Falkenmark, M., Lundqvist, J. & Widstrand, C. (1989). Macro-scale water
    scarcity requires micro-scale approaches. *Natural Resources Forum*
    13(4), 258-267.
FAO (1996). *Rome Declaration on World Food Security and World Food Summit
    Plan of Action.* FAO, Rome (the four pillars of food security).
FAO (2001). *Food Balance Sheets: A Handbook.* FAO, Rome
    (self-sufficiency ratio).
FAO (2018). *Progress on Level of Water Stress - Global Baseline for SDG
    Indicator 6.4.2.* FAO/UN-Water, Rome.
FAO (2018b). *Progress on Water-Use Efficiency - Global Baseline for SDG
    Indicator 6.4.1.* FAO/UN-Water, Rome.
Gini, C. (1912). *Variabilita e mutabilita.* Cuppini, Bologna.
Gleeson, T., Wada, Y., Bierkens, M. F. P. & van Beek, L. P. H. (2012).
    Water balance of global aquifers revealed by groundwater footprint.
    *Nature* 488, 197-200.
Hashimoto, T., Stedinger, J. R. & Loucks, D. P. (1982). Reliability,
    resiliency, and vulnerability criteria for water resource system
    performance evaluation. *Water Resources Research* 18(1), 14-20.
Hoff, H. (2011). *Understanding the Nexus.* Background paper for the Bonn
    2011 Conference: The Water, Energy and Food Security Nexus. SEI,
    Stockholm.
IPCC (2014). *Climate Change 2014: Mitigation of Climate Change.* Annex III,
    Technology-specific cost and performance parameters (lifecycle emission
    factors: coal ~820, gas CC ~490, hydro ~24 t CO2eq/GWh).
Kruyt, B., van Vuuren, D. P., de Vries, H. J. M. & Groenenberg, H. (2009).
    Indicators for energy security. *Energy Policy* 37(6), 2166-2181.
Nardo, M., Saisana, M., Saltelli, A., Tarantola, S., Hoffman, A. &
    Giovannini, E. (2005). *Handbook on Constructing Composite Indicators:
    Methodology and User Guide.* OECD Statistics Working Paper 2005/3.
OECD/JRC (2008). *Handbook on Constructing Composite Indicators.* OECD,
    Paris.
Sen, A. (1973). *On Economic Inequality.* Clarendon Press, Oxford.
Simpson, G. B., Jewitt, G. P. W., Becker, W., Badenhorst, J., Masia, S.,
    Neves, A. R., Rovira, P. & Pascual, V. (2022). The Water-Energy-Food
    Nexus Index: a tool to support sustainable development. *Sustainability*
    14(1), 45.
Sovacool, B. K. & Mukherjee, I. (2011). Conceptualizing and measuring energy
    security: a synthesized approach. *Energy* 36(8), 5343-5355.
Sullivan, C. (2002). Calculating a water poverty index. *World Development*
    30(7), 1195-1210.
UNDP (2010). *Human Development Report 2010: The Real Wealth of Nations.*
    Technical note 1 (geometric aggregation of the HDI).
UN-Water (2018). *Progress on Transboundary Water Cooperation - Global
    Baseline for SDG Indicator 6.5.2.* UNECE/UNESCO, Paris.
Wolf, A. T., Yoffe, S. B. & Giordano, M. (2003). International waters:
    identifying basins at risk. *Water Policy* 5(1), 29-60.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from wefnexus.allocation import gini
from wefnexus.water import (
    falkenmark_category,
    resilience,
    supply_reliability,
    vulnerability,
)

__all__ = [
    "GEOMETRIC_FLOOR",
    "WATER_STRESS_RANGE",
    "GROUNDWATER_STRESS_RANGE",
    "EMISSION_INTENSITY_RANGE",
    "WATER_COMPONENTS",
    "ENERGY_COMPONENTS",
    "FOOD_COMPONENTS",
    "NEXUS_PILLARS",
    "INDEX_KEYS",
    "SDG_KEYS",
    "RIPARIAN_YEAR_FIELDS",
    "SUMMARY_COLUMNS",
    "COLUMN_LABELS",
    "normalise",
    "weighted_geometric_mean",
    "water_security_components",
    "water_security_index",
    "energy_security_components",
    "energy_security_index",
    "food_security_components",
    "food_security_index",
    "wef_nexus_index",
    "equity_index",
    "water_use_efficiency_usd_per_m3",
    "sdg_indicators",
    "SustainabilityReport",
    "assess_records",
    "assess",
]

#: Lower clip applied to every component inside :func:`weighted_geometric_mean`.
GEOMETRIC_FLOOR: float = 1e-6

#: ``(lo, hi)`` passed to :func:`normalise` for SDG 6.4.2 water stress (%):
#: 100 % (withdrawals equal the water available beyond the environmental
#: flow, FAO "critical" class) scores 0, 0 % scores 1.
WATER_STRESS_RANGE: Tuple[float, float] = (100.0, 0.0)

#: ``(lo, hi)`` for groundwater stress (abstraction / recharge): 1.5 (severe
#: overdraft) scores 0, 0.5 (half the recharge, a sustainable yield) scores
#: 1 (Gleeson et al. 2012 place depletion at a ratio of 1).
GROUNDWATER_STRESS_RANGE: Tuple[float, float] = (1.5, 0.5)

#: ``(lo, hi)`` for grid emission intensity (t CO2 per GWh): 1000 t/GWh (an
#: all-coal, sub-critical grid; IPCC 2014 coal lifecycle median 820) scores
#: 0, a carbon-free grid scores 1.
EMISSION_INTENSITY_RANGE: Tuple[float, float] = (1000.0, 0.0)

#: Component names of the water, energy and food security indices.
WATER_COMPONENTS: Tuple[str, ...] = ("supply", "stress", "environment", "groundwater")
ENERGY_COMPONENTS: Tuple[str, ...] = ("supply", "renewable", "emissions")
FOOD_COMPONENTS: Tuple[str, ...] = ("self_sufficiency", "et_ratio")
#: Pillars of :func:`wef_nexus_index` (keys of its ``weights``).
NEXUS_PILLARS: Tuple[str, ...] = ("water", "energy", "food")
#: Index fields averaged by :func:`assess`.
INDEX_KEYS: Tuple[str, ...] = ("water_security", "energy_security", "food_security", "nexus_index")

#: Short code -> key used in the dictionaries returned by :func:`sdg_indicators`.
SDG_KEYS: Dict[str, str] = {
    "6.4.1": "6.4.1_water_use_efficiency_usd_per_m3",
    "6.4.2": "6.4.2_water_stress_pct",
    "6.5.2": "6.5.2_transboundary_cooperation",
    "7.2": "7.2_renewable_share",
    "2.1": "2.1_food_self_sufficiency",
}

#: Fields of :class:`wefnexus.nexus.RiparianYear` read by :func:`assess`
#: (ARCHITECTURE.md section 7).
RIPARIAN_YEAR_FIELDS: Tuple[str, ...] = (
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

#: Default columns of :meth:`SustainabilityReport.summary`.
SUMMARY_COLUMNS: Tuple[str, ...] = (
    "water_security",
    "energy_security",
    "food_security",
    "nexus_index",
    "supply_ratio",
    "supply_reliability",
    "env_flow_met_share",
    SDG_KEYS["6.4.2"],
    SDG_KEYS["2.1"],
    SDG_KEYS["7.2"],
    SDG_KEYS["6.4.1"],
)

#: Short column headers used by :meth:`SustainabilityReport.summary`.
COLUMN_LABELS: Dict[str, str] = {
    "water_security": "water",
    "energy_security": "energy",
    "food_security": "food",
    "nexus_index": "nexus",
    "min_nexus_index": "nexus_min",
    "supply_ratio": "supply",
    "min_supply_ratio": "supply_min",
    "supply_reliability": "reliab",
    "resilience": "resil",
    "vulnerability": "vulner",
    "env_flow_met_share": "env_met",
    "years_env_flow_unmet": "env_unmet",
    "groundwater_used_mm3": "gw_used",
    "freshwater_withdrawal_mm3": "fresh_w",
    "equity_index": "equity",
    "food_self_sufficiency": "food_ss",
    "renewable_share": "renew",
    "per_capita_water_m3": "m3/cap",
    SDG_KEYS["6.4.1"]: "6.4.1$/m3",
    SDG_KEYS["6.4.2"]: "6.4.2%",
    SDG_KEYS["6.5.2"]: "6.5.2",
    SDG_KEYS["7.2"]: "7.2",
    SDG_KEYS["2.1"]: "2.1",
}

_M3_PER_MM3 = 1.0e6
_BASIN_ROW = "BASIN"


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
def _as_float(value: Any, label: str, *, allow_inf: bool = False) -> float:
    """Coerce ``value`` to a float; NaN, bools and non-numbers raise ``ValueError``."""
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


def _ratio_capped(value: Any, label: str) -> float:
    """A non-negative ratio; values above 1 (surplus, rounding) count as 1."""
    return min(_nonneg(value, label, allow_inf=True), 1.0)


def _clip_pillar(v: float) -> float:
    return min(max(v, GEOMETRIC_FLOOR), 1.0)


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        raise ValueError("cannot average an empty sequence")
    return float(sum(vals) / len(vals))


def _safe_ratio(numerator: float, denominator: float, when_zero: float) -> float:
    return numerator / denominator if denominator > 0.0 else when_zero


# --------------------------------------------------------------------------- #
# Normalisation and aggregation
# --------------------------------------------------------------------------- #
def normalise(value: float, lo: float, hi: float, higher_is_better: bool = True) -> float:
    """Min-max normalisation of ``value`` to ``[0, 1]``, clipped.

    ``lo`` maps to 0 and ``hi`` maps to 1 (linear in between); the scale may
    run downwards (``lo > hi``), which is the idiom used throughout this
    module for "lower is better" quantities, e.g.
    ``normalise(stress_pct, 100, 0)``.  With ``higher_is_better=False`` the
    result is flipped (``1 - t``), so ``normalise(x, 0, 100, False) ==
    normalise(x, 100, 0)``.

    Parameters
    ----------
    value : float
        Raw indicator value (any unit; ``+inf`` / ``-inf`` are accepted and
        clip to the appropriate end of the scale).
    lo, hi : float
        Finite reference values mapped to 0 and 1 respectively (same unit as
        ``value``); must differ.
    higher_is_better : bool, optional
        If False the normalised value is reversed (default True).

    Returns
    -------
    float
        Dimensionless score in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``value`` is NaN or not a number, ``lo``/``hi`` are not finite, or
        ``lo == hi``.

    References
    ----------
    Nardo et al. (2005), OECD/JRC (2008) §"Min-Max"; UNDP (2010) HDI
    technical note (goalposts).

    Examples
    --------
    >>> normalise(50.0, 0.0, 100.0)
    0.5
    >>> normalise(25.0, 100.0, 0.0)      # water stress 25 % -> 0.75
    0.75
    >>> normalise(150.0, 0.0, 100.0)     # clipped
    1.0
    """
    v = _as_float(value, "value", allow_inf=True)
    lo_f = _as_float(lo, "lo")
    hi_f = _as_float(hi, "hi")
    if lo_f == hi_f:
        raise ValueError(f"lo and hi must differ, both are {lo_f}")
    if math.isinf(v):
        # +inf lies beyond ``hi`` when the scale rises and beyond ``lo`` when it falls.
        t = 1.0 if (v > 0.0) == (hi_f > lo_f) else 0.0
    else:
        t = (v - lo_f) / (hi_f - lo_f)
    if not higher_is_better:
        t = 1.0 - t
    return float(min(max(t, 0.0), 1.0))


def weighted_geometric_mean(
    values: Mapping[str, float], weights: Optional[Mapping[str, float]] = None
) -> float:
    """Weighted geometric mean of component scores, each clipped to ``[1e-6, 1]``.

    ``G = exp( sum_i w_i ln(v_i) / sum_i w_i )`` with ``v_i`` clipped to
    ``[GEOMETRIC_FLOOR, 1]``.  The geometric mean is *partially
    compensatory*: by the AM-GM inequality it never exceeds the weighted
    arithmetic mean, and a single very low component pulls the result down
    sharply (one zero component among three equal-weighted ones gives
    ``1e-6 ** (1/3) = 0.01``).  This is the aggregation of the HDI (UNDP
    2010) and of the WEF Nexus Index (Simpson et al. 2022).

    Parameters
    ----------
    values : mapping of str -> float
        Component scores (dimensionless, nominally in ``[0, 1]``; anything
        outside is clipped, ``inf`` counts as 1).  Must be non-empty.
    weights : mapping of str -> float, optional
        Non-negative weights with exactly the same keys as ``values``; at
        least one must be positive.  Default: equal weights.

    Returns
    -------
    float
        Geometric mean in ``[1e-6, 1]``.

    Raises
    ------
    ValueError
        On an empty or non-mapping ``values``, NaN / non-numeric entries,
        negative weights, weights that sum to zero, or mismatched keys.

    References
    ----------
    Ebert & Welsch (2004); OECD/JRC (2008) §"Geometric aggregation"; UNDP
    (2010); Simpson et al. (2022).

    Examples
    --------
    >>> weighted_geometric_mean({"a": 0.25, "b": 1.0})
    0.5
    >>> weighted_geometric_mean({"a": 0.25, "b": 1.0}, {"a": 1.0, "b": 0.0})
    0.25
    """
    if not isinstance(values, Mapping):
        raise ValueError(f"values must be a mapping of component -> score, got {type(values).__name__}")
    if len(values) == 0:
        raise ValueError("values must contain at least one component")
    keys = list(values)
    vals = [_clip_pillar(_as_float(values[k], f"values[{k!r}]", allow_inf=True)) for k in keys]
    if weights is None:
        w = [1.0] * len(keys)
    else:
        if not isinstance(weights, Mapping):
            raise ValueError(f"weights must be a mapping of component -> weight, got {type(weights).__name__}")
        missing = [k for k in keys if k not in weights]
        extra = [k for k in weights if k not in values]
        if missing or extra:
            raise ValueError(
                f"weights keys must match values keys {sorted(map(str, keys))!r}; "
                f"missing {sorted(map(str, missing))!r}, unexpected {sorted(map(str, extra))!r}"
            )
        w = [_nonneg(weights[k], f"weights[{k!r}]") for k in keys]
    total_w = sum(w)
    if total_w <= 0.0:
        raise ValueError("weights must not all be zero")
    log_mean = sum(wi * math.log(vi) for wi, vi in zip(w, vals)) / total_w
    return float(_clip_pillar(math.exp(log_mean)))


# --------------------------------------------------------------------------- #
# Pillar indices
# --------------------------------------------------------------------------- #
def water_security_components(
    supply_ratio: float,
    stress_sdg642: float,
    env_flow_met_share: float,
    groundwater_stress: float,
) -> Dict[str, float]:
    """Normalised components of the water security index (each in ``[0, 1]``).

    Parameters
    ----------
    supply_ratio : float
        Share of withdrawal demand that was met, ``1 - deficit / demand``
        (dimensionless, >= 0; values above 1 count as 1).
    stress_sdg642 : float
        SDG 6.4.2 level of water stress in percent (>= 0, ``inf`` allowed),
        see :func:`wefnexus.water.sdg_642_water_stress`.
    env_flow_met_share : float
        Share of years (or reaches) in which the environmental flow
        requirement was met, in ``[0, 1]``.
    groundwater_stress : float
        Groundwater abstraction / recharge (>= 0, ``inf`` allowed), see
        :func:`wefnexus.water.groundwater_stress`.

    Returns
    -------
    dict
        ``{"supply": supply_ratio, "stress": normalise(stress, 100, 0),
        "environment": env_flow_met_share,
        "groundwater": normalise(gw_stress, 1.5, 0.5)}``
        (keys :data:`WATER_COMPONENTS`).

    References
    ----------
    Sullivan (2002) Water Poverty Index (resource / access / environment
    components); FAO (2018) SDG 6.4.2 classes; Gleeson et al. (2012).
    """
    return {
        "supply": _ratio_capped(supply_ratio, "supply_ratio"),
        "stress": normalise(_nonneg(stress_sdg642, "stress_sdg642", allow_inf=True), *WATER_STRESS_RANGE),
        "environment": _fraction(env_flow_met_share, "env_flow_met_share"),
        "groundwater": normalise(
            _nonneg(groundwater_stress, "groundwater_stress", allow_inf=True), *GROUNDWATER_STRESS_RANGE
        ),
    }


def water_security_index(
    supply_ratio: float,
    stress_sdg642: float,
    env_flow_met_share: float,
    groundwater_stress: float,
    *,
    weights: Optional[Mapping[str, float]] = None,
) -> float:
    """Water security index in ``[0, 1]`` (1 = fully secure).

    Geometric mean of the four :func:`water_security_components`: supply
    ratio, ``normalise(SDG 6.4.2 stress, 100, 0)``, environmental-flow
    compliance share and ``normalise(groundwater stress, 1.5, 0.5)``.

    Parameters
    ----------
    supply_ratio, stress_sdg642, env_flow_met_share, groundwater_stress : float
        See :func:`water_security_components` (dimensionless / percent).
    weights : mapping, optional
        Weights keyed by :data:`WATER_COMPONENTS` (default equal).

    Returns
    -------
    float
        Index in ``[0, 1]``; decreasing in both stress measures and
        increasing in the supply ratio and the environmental-flow share.

    Raises
    ------
    ValueError
        On negative inputs, ``env_flow_met_share`` outside ``[0, 1]`` or bad
        weights.

    References
    ----------
    Sullivan (2002); Hoff (2011); Simpson et al. (2022).

    Examples
    --------
    >>> water_security_index(1.0, 0.0, 1.0, 0.0)
    1.0
    >>> round(water_security_index(1.0, 50.0, 1.0, 1.0), 4)   # (1*0.5*1*0.5)**0.25
    0.7071
    """
    comps = water_security_components(supply_ratio, stress_sdg642, env_flow_met_share, groundwater_stress)
    return weighted_geometric_mean(comps, weights)


def energy_security_components(
    supply_ratio: float, renewable_share: float, emission_intensity_t_per_gwh: float
) -> Dict[str, float]:
    """Normalised components of the energy security index (each in ``[0, 1]``).

    Parameters
    ----------
    supply_ratio : float
        Share of electricity demand met, ``min(supply / demand, 1)``
        (dimensionless, >= 0; above 1 counts as 1), e.g. from
        :func:`wefnexus.energy.energy_security_index`.
    renewable_share : float
        Share of supply from hydropower and other renewables, in ``[0, 1]``.
    emission_intensity_t_per_gwh : float
        Emissions per unit of supply, t CO2 per GWh (>= 0; ``inf`` allowed).

    Returns
    -------
    dict
        ``{"supply": ..., "renewable": ..., "emissions": normalise(intensity,
        1000, 0)}`` (keys :data:`ENERGY_COMPONENTS`).

    References
    ----------
    Kruyt et al. (2009) availability / acceptability dimensions; Sovacool &
    Mukherjee (2011); IPCC (2014) Annex III emission factors.
    """
    return {
        "supply": _ratio_capped(supply_ratio, "supply_ratio"),
        "renewable": _fraction(renewable_share, "renewable_share"),
        "emissions": normalise(
            _nonneg(emission_intensity_t_per_gwh, "emission_intensity_t_per_gwh", allow_inf=True),
            *EMISSION_INTENSITY_RANGE,
        ),
    }


def energy_security_index(
    supply_ratio: float,
    renewable_share: float,
    emission_intensity_t_per_gwh: float,
    *,
    weights: Optional[Mapping[str, float]] = None,
) -> float:
    """Energy security index in ``[0, 1]`` (1 = fully secure, clean and renewable).

    Geometric mean of the supply ratio (availability), the renewable share
    (SDG 7.2, sustainability) and ``normalise(emission intensity, 1000, 0)``
    (environmental acceptability) - see :func:`energy_security_components`.

    Parameters
    ----------
    supply_ratio : float
        ``min(supply / demand, 1)`` (dimensionless).
    renewable_share : float
        Renewable share of supply in ``[0, 1]``.
    emission_intensity_t_per_gwh : float
        t CO2 per GWh of supply (>= 0).
    weights : mapping, optional
        Weights keyed by :data:`ENERGY_COMPONENTS` (default equal).

    Returns
    -------
    float
        Index in ``[0, 1]``; increasing in the supply ratio and renewable
        share, decreasing in the emission intensity.

    References
    ----------
    Kruyt et al. (2009); Sovacool & Mukherjee (2011); IPCC (2014).

    Examples
    --------
    >>> energy_security_index(1.0, 1.0, 0.0)
    1.0
    >>> energy_security_index(0.5, 0.5, 500.0)
    0.5
    """
    comps = energy_security_components(supply_ratio, renewable_share, emission_intensity_t_per_gwh)
    return weighted_geometric_mean(comps, weights)


def food_security_components(self_sufficiency: float, et_ratio: float) -> Dict[str, float]:
    """Normalised components of the food security index (each in ``[0, 1]``).

    Parameters
    ----------
    self_sufficiency : float
        Calorie self-sufficiency ratio, produced / demanded (dimensionless,
        >= 0; a surplus above 1 counts as 1), see
        :func:`wefnexus.food.food_self_sufficiency`.
    et_ratio : float
        Relative crop evapotranspiration ETa/ETm of the irrigated crops
        (dimensionless, in ``[0, 1]``; above 1 counts as 1), the FAO-33
        water-adequacy proxy for yield stability.

    Returns
    -------
    dict
        ``{"self_sufficiency": min(ssr, 1), "et_ratio": min(et_ratio, 1)}``
        (keys :data:`FOOD_COMPONENTS`).

    References
    ----------
    FAO (1996) availability and stability pillars; FAO (2001); Doorenbos &
    Kassam (1979) FAO-33.
    """
    return {
        "self_sufficiency": _ratio_capped(self_sufficiency, "self_sufficiency"),
        "et_ratio": _ratio_capped(et_ratio, "et_ratio"),
    }


def food_security_index(
    self_sufficiency: float, et_ratio: float, *, weights: Optional[Mapping[str, float]] = None
) -> float:
    """Food security index in ``[0, 1]`` (1 = self-sufficient and fully irrigated).

    Geometric mean of the capped calorie self-sufficiency ratio
    (availability) and the relative evapotranspiration of the irrigated
    crops (stability of production under water shortage).

    Parameters
    ----------
    self_sufficiency : float
        Produced / demanded food energy (dimensionless, >= 0).
    et_ratio : float
        ETa/ETm of the crops (dimensionless, >= 0).
    weights : mapping, optional
        Weights keyed by :data:`FOOD_COMPONENTS` (default equal).

    Returns
    -------
    float
        Index in ``[0, 1]``, increasing in both arguments.

    References
    ----------
    FAO (1996); FAO (2001); Hoff (2011).

    Examples
    --------
    >>> food_security_index(1.0, 1.0)
    1.0
    >>> food_security_index(0.25, 1.0)
    0.5
    >>> food_security_index(2.0, 0.5)     # surplus capped at 1
    0.7071067811865476
    """
    comps = food_security_components(self_sufficiency, et_ratio)
    return weighted_geometric_mean(comps, weights)


def wef_nexus_index(
    water: float, energy: float, food: float, weights: Optional[Mapping[str, float]] = None
) -> float:
    """Overall WEF nexus index: geometric mean of the three pillar indices.

    Parameters
    ----------
    water, energy, food : float
        Pillar indices in ``[0, 1]`` (dimensionless; values above 1 count as
        1, negative values raise).
    weights : mapping, optional
        Non-negative weights keyed ``"water"``, ``"energy"``, ``"food"``
        (:data:`NEXUS_PILLARS`), default equal.

    Returns
    -------
    float
        Index in ``[0, 1]``; symmetric and increasing in every pillar, and
        never above the weighted arithmetic mean of the pillars.

    References
    ----------
    Hoff (2011); Bizikova et al. (2013); Simpson et al. (2022) WEF Nexus
    Index; UNDP (2010).

    Examples
    --------
    >>> wef_nexus_index(1.0, 1.0, 1.0)
    1.0
    >>> wef_nexus_index(0.5, 0.5, 0.5)
    0.5
    >>> round(wef_nexus_index(1.0, 1.0, 0.0), 6)   # one collapsed pillar
    0.01
    """
    comps = {
        "water": _ratio_capped(water, "water"),
        "energy": _ratio_capped(energy, "energy"),
        "food": _ratio_capped(food, "food"),
    }
    return weighted_geometric_mean(comps, weights)


# --------------------------------------------------------------------------- #
# Equity and SDG indicators
# --------------------------------------------------------------------------- #
def equity_index(values_by_riparian: Union[Mapping[str, float], Sequence[float]]) -> float:
    """Distributional equity ``1 - Gini`` of a per-riparian distribution.

    Parameters
    ----------
    values_by_riparian : mapping of str -> float, or sequence of float
        Non-negative values per riparian (any unit: supply ratios, awards in
        Mm3/yr, per-capita water in m3/person/yr ...).  Must be non-empty.

    Returns
    -------
    float
        ``1 - gini(values)`` in ``[0, 1]``: 1 for a perfectly equal
        distribution (or a single riparian, or all zeros), ``1/n`` when one
        riparian holds everything.

    Raises
    ------
    ValueError
        On an empty input or negative / non-finite values.

    References
    ----------
    Gini (1912); Sen (1973); Cullis & van Koppen (2007) Gini of water use.

    Examples
    --------
    >>> equity_index({"A": 1.0, "B": 1.0, "C": 1.0})
    1.0
    >>> equity_index({"A": 1.0, "B": 0.0})
    0.5
    """
    if isinstance(values_by_riparian, Mapping):
        n = len(values_by_riparian)
    elif isinstance(values_by_riparian, (str, bytes)):
        raise ValueError("values_by_riparian must be a mapping or a sequence of numbers")
    else:
        try:
            n = len(list(values_by_riparian))
        except TypeError:
            raise ValueError("values_by_riparian must be a mapping or a sequence of numbers") from None
    if n == 0:
        raise ValueError("equity_index needs at least one value")
    return float(min(max(1.0 - gini(values_by_riparian), 0.0), 1.0))


def water_use_efficiency_usd_per_m3(gdp_usd: float, withdrawal_mm3: float) -> float:
    """SDG 6.4.1 water-use efficiency: value added per unit of water withdrawn.

    Parameters
    ----------
    gdp_usd : float
        Gross domestic product (or value added), USD/yr (>= 0).
    withdrawal_mm3 : float
        Total freshwater withdrawal (FAO "TFWW": surface water plus
        groundwater, excluding desalinated and reused water), Mm3/yr (>= 0).

    Returns
    -------
    float
        ``gdp_usd / (withdrawal_mm3 * 1e6)`` in USD per m3; ``inf`` when
        there is value added but no withdrawal, ``0.0`` when both are zero.

    References
    ----------
    FAO (2018b) SDG 6.4.1 metadata (change in water-use efficiency over
    time, USD/m3).
    """
    gdp = _nonneg(gdp_usd, "gdp_usd")
    w = _nonneg(withdrawal_mm3, "withdrawal_mm3")
    if w <= 0.0:
        return math.inf if gdp > 0.0 else 0.0
    return gdp / (w * _M3_PER_MM3)


def sdg_indicators(
    gdp_usd: float,
    withdrawal_mm3: float,
    water_stress_pct: float,
    renewable_share: float,
    food_self_sufficiency: float,
    transboundary_cooperation: float = 0.0,
) -> Dict[str, float]:
    """SDG-style indicator set for one riparian (or a whole basin).

    Parameters
    ----------
    gdp_usd : float
        Value added, USD/yr (>= 0) - numerator of SDG 6.4.1.
    withdrawal_mm3 : float
        Total freshwater withdrawal (surface + groundwater), Mm3/yr (>= 0) -
        denominator of 6.4.1.
    water_stress_pct : float
        SDG 6.4.2 level of water stress, percent (>= 0, ``inf`` allowed),
        from :func:`wefnexus.water.sdg_642_water_stress`.
    renewable_share : float
        SDG 7.2 share of renewables in energy supply, in ``[0, 1]``.
    food_self_sufficiency : float
        Calorie self-sufficiency ratio (>= 0, not capped), from
        :func:`wefnexus.food.food_self_sufficiency`; proxy for SDG 2.1.
    transboundary_cooperation : float or bool, optional
        SDG 6.5.2: 1 (or True) when an operational arrangement for water
        cooperation (treaty / agreed sharing rule) is in force, 0 when not,
        or the share of the basin covered (in ``[0, 1]``).  Default 0.

    Returns
    -------
    dict
        Keys (see :data:`SDG_KEYS`):

        * ``"6.4.1_water_use_efficiency_usd_per_m3"`` - ``gdp / withdrawal``
          in USD/m3 (:func:`water_use_efficiency_usd_per_m3`);
        * ``"6.4.2_water_stress_pct"`` - as given;
        * ``"6.5.2_transboundary_cooperation"`` - as given (0..1);
        * ``"7.2_renewable_share"`` - as given (0..1);
        * ``"2.1_food_self_sufficiency"`` - as given.

    References
    ----------
    FAO (2018, 2018b); UN-Water (2018); FAO (2001); UN SDG indicator
    framework (A/RES/71/313).
    """
    if isinstance(transboundary_cooperation, bool):
        coop = 1.0 if transboundary_cooperation else 0.0
    else:
        coop = _fraction(transboundary_cooperation, "transboundary_cooperation")
    return {
        SDG_KEYS["6.4.1"]: water_use_efficiency_usd_per_m3(gdp_usd, withdrawal_mm3),
        SDG_KEYS["6.4.2"]: _nonneg(water_stress_pct, "water_stress_pct", allow_inf=True),
        SDG_KEYS["6.5.2"]: coop,
        SDG_KEYS["7.2"]: _fraction(renewable_share, "renewable_share"),
        SDG_KEYS["2.1"]: _nonneg(food_self_sufficiency, "food_self_sufficiency"),
    }


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _format_cell(value: Any) -> str:
    """Compact text rendering of one table cell."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
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


def _format_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str], first: str = "riparian") -> str:
    """Fixed-width text table; ``rows`` must carry ``"name"`` plus the columns."""
    labels = [COLUMN_LABELS.get(c, c) for c in columns]
    cells = [[_format_cell(r.get(c)) for c in columns] for r in rows]
    names = [str(r.get("name", "")) for r in rows]
    w0 = max([len(first)] + [len(n) for n in names])
    widths = [max([len(lab)] + [row[i] and len(row[i]) or 0 for row in cells]) for i, lab in enumerate(labels)]
    header = first.ljust(w0) + "".join("  " + lab.rjust(w) for lab, w in zip(labels, widths))
    sep = "-" * len(header)
    lines = [header, sep]
    for name, row in zip(names, cells):
        line = name.ljust(w0) + "".join("  " + c.rjust(w) for c, w in zip(row, widths))
        if name == _BASIN_ROW:
            lines.append(sep)
        lines.append(line)
    return "\n".join(lines)


@dataclass
class SustainabilityReport:
    """Sustainability assessment of a multi-year nexus simulation.

    Attributes
    ----------
    basin_name : str
        Name of the basin.
    scenario_name : str
        Name of the scenario that produced the result.
    years : list of int
        Calendar years of the horizon.
    riparians : dict
        ``{riparian: {indicator: value}}`` - means over the horizon of the
        four indices (:data:`INDEX_KEYS`), supply ratio and Hashimoto
        reliability / resilience / vulnerability, water, energy and food
        totals, SDG indicators (:data:`SDG_KEYS`) and a Falkenmark class.
        See :func:`assess_records` for the full key list.
    basin : dict
        Basin-level aggregates: riparian means of the indices,
        population-weighted nexus index, equity indices (``1 - Gini`` of
        supply ratios, nexus indices and per-capita water), basin supply
        ratio / reliability, totals, ``env_flow_met_share`` (over
        riparian-years) and ``years_env_flow_unmet`` (years in which *any*
        reach missed its environmental flow, so at most ``basin["years"]``),
        SDG indicators and, when balances were supplied, outflow to sea,
        natural flow, per-capita water and the maximum water mass-balance
        error.
    """

    basin_name: str
    scenario_name: str
    years: List[int]
    riparians: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    basin: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.riparians, Mapping):
            raise ValueError("riparians must be a dict of {name: {indicator: value}}")
        for name, d in self.riparians.items():
            if not isinstance(d, Mapping):
                raise ValueError(f"riparians[{name!r}] must be a dict of indicators")
        if not isinstance(self.basin, Mapping):
            raise ValueError("basin must be a dict of indicators")
        self.years = list(self.years)

    # -- access --------------------------------------------------------------
    def names(self) -> List[str]:
        """Riparian names in report order."""
        return list(self.riparians)

    def riparian(self, name: str) -> Dict[str, Any]:
        """Indicator dict of riparian ``name`` (``KeyError`` if absent)."""
        try:
            return self.riparians[name]
        except KeyError:
            raise KeyError(f"no riparian named {name!r}; report covers {self.names()}") from None

    def indices(self, include_basin: bool = True) -> Dict[str, Dict[str, float]]:
        """``{name: {water_security, energy_security, food_security, nexus_index}}``.

        Parameters
        ----------
        include_basin : bool, optional
            Also include a ``"BASIN"`` entry (default True).
        """
        out = {name: {k: d.get(k) for k in INDEX_KEYS} for name, d in self.riparians.items()}
        if include_basin:
            out[_BASIN_ROW] = {k: self.basin.get(k) for k in INDEX_KEYS}
        return out

    def sdg(self, include_basin: bool = True) -> Dict[str, Dict[str, float]]:
        """``{name: {sdg_key: value}}`` for the five :data:`SDG_KEYS`."""
        keys = list(SDG_KEYS.values())
        out = {name: {k: d.get(k) for k in keys} for name, d in self.riparians.items()}
        if include_basin:
            out[_BASIN_ROW] = {k: self.basin.get(k) for k in keys}
        return out

    # -- export ----------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Deep-copied plain dictionary of the whole report."""
        return {
            "basin_name": self.basin_name,
            "scenario": self.scenario_name,
            "years": list(self.years),
            "riparians": copy.deepcopy(dict(self.riparians)),
            "basin": copy.deepcopy(dict(self.basin)),
        }

    def _resolve_columns(self, columns: Optional[Sequence[str]]) -> List[str]:
        """Default columns are rendered as-is (missing -> ``None``); explicit
        unknown columns raise ``ValueError``."""
        if columns is None:
            return list(SUMMARY_COLUMNS)
        cols = [str(c) for c in columns]
        if not cols:
            raise ValueError("columns must not be empty")
        known = set(self.basin)
        for d in self.riparians.values():
            known.update(d)
        unknown = [c for c in cols if c not in known]
        if unknown:
            raise ValueError(f"unknown column(s) {unknown!r}; available: {sorted(known)!r}")
        return cols

    def to_rows(self, columns: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
        """One row per riparian plus a ``"BASIN"`` row (``"name"`` + columns).

        Parameters
        ----------
        columns : sequence of str, optional
            Indicator keys to include (default :data:`SUMMARY_COLUMNS`).
            A key missing from a row yields ``None``.
        """
        cols = self._resolve_columns(columns)
        rows = [{"name": name, **{c: d.get(c) for c in cols}} for name, d in self.riparians.items()]
        rows.append({"name": _BASIN_ROW, **{c: self.basin.get(c) for c in cols}})
        return rows

    def to_dataframe(self, columns: Optional[Sequence[str]] = None):  # pragma: no cover - thin wrapper
        """Rows of :meth:`to_rows` as a :class:`pandas.DataFrame` indexed by name.

        pandas is imported lazily; ``ImportError`` if it is not installed.
        """
        import pandas as pd  # lazy: optional dependency

        return pd.DataFrame(self.to_rows(columns)).set_index("name")

    def summary(self, columns: Optional[Sequence[str]] = None) -> str:
        """Plain-text table of the main indicators per riparian and for the basin.

        Parameters
        ----------
        columns : sequence of str, optional
            Indicator keys (default :data:`SUMMARY_COLUMNS`); unknown keys
            raise ``ValueError``.

        Returns
        -------
        str
            A title line followed by a fixed-width table whose last row is
            the basin aggregate.
        """
        rows = self.to_rows(columns)
        cols = self._resolve_columns(columns)
        if self.years:
            span = f"{self.years[0]}-{self.years[-1]} ({len(self.years)} years)"
        else:
            span = "no years"
        title = f"Sustainability assessment: {self.basin_name} | scenario: {self.scenario_name} | {span}"
        return title + "\n" + _format_table(rows, cols)

    def __str__(self) -> str:
        return self.summary()


# --------------------------------------------------------------------------- #
# Assessment of nexus results
# --------------------------------------------------------------------------- #
def _rec_get(rec: Any, fld: str) -> Any:
    try:
        return getattr(rec, fld)
    except AttributeError:
        who = f"{getattr(rec, 'name', '?')!r}/{getattr(rec, 'year', '?')}"
        raise ValueError(
            f"record {who} lacks field {fld!r}; sustainability.assess() expects "
            f"wefnexus.nexus.RiparianYear records with fields {RIPARIAN_YEAR_FIELDS}"
        ) from None


def _rec_float(rec: Any, fld: str, *, allow_none_inf: bool = False) -> float:
    value = _rec_get(rec, fld)
    if value is None and allow_none_inf:
        return math.inf
    return _as_float(value, f"{getattr(rec, 'name', '?')}[{fld}]", allow_inf=True)


def _rec_sector_total(rec: Any, fld: str) -> float:
    value = _rec_get(rec, fld)
    if not isinstance(value, Mapping):
        raise ValueError(f"{getattr(rec, 'name', '?')}[{fld}] must be a dict of sector -> Mm3")
    return float(sum(_as_float(v, f"{fld}[{k!r}]", allow_inf=True) for k, v in value.items()))


def _rec_optional_nonneg(rec: Any, fld: str, default: float = 0.0) -> float:
    """Non-negative float of an *optional* (non-contract) record field.

    A missing attribute, or ``None``, yields ``default``; a present value
    must be a finite number >= 0, otherwise ``ValueError``.  Used for the
    ``groundwater_used_mm3`` diagnostic that :class:`wefnexus.nexus.RiparianYear`
    carries but that the section-7 contract does not require.
    """
    value = getattr(rec, fld, None)
    if value is None:
        return float(default)
    return _nonneg(value, f"{getattr(rec, 'name', '?')}[{fld}]")


def _validate_years(years: Any, record_years: Any) -> List[Any]:
    """Validate an explicit ``years`` argument of :func:`assess_records`.

    ``years`` must be a non-empty, duplicate-free subset of the years found
    in the records; it is returned sorted (chronological order, as the
    Hashimoto criteria need).  Anything else raises ``ValueError`` instead
    of silently falling back to the record years.
    """
    if isinstance(years, (str, bytes)):
        raise ValueError(f"years must be a sequence of years, not a string ({years!r})")
    try:
        year_list = list(years)
    except TypeError:
        raise ValueError(f"years must be a sequence of years, got {years!r}") from None
    if not year_list:
        raise ValueError("years must not be empty (pass None to use the years of the records)")
    try:
        unknown = [y for y in year_list if y not in record_years]
        n_distinct = len(set(year_list))
    except TypeError:
        raise ValueError(f"years must be hashable values such as integers, got {year_list!r}") from None
    if unknown:
        raise ValueError(
            f"years {unknown!r} have no records; the records cover {sorted(record_years)!r}"
        )
    if n_distinct != len(year_list):
        raise ValueError(f"years must not contain duplicates, got {year_list!r}")
    return sorted(year_list)


def _riparian_summary(name: str, recs: List[Any], cooperation: bool) -> Dict[str, Any]:
    """Horizon summary of one riparian's records (sorted by year)."""
    try:
        recs = sorted(recs, key=lambda r: _rec_get(r, "year"))
    except TypeError:
        raise ValueError(f"records of {name!r} have incomparable 'year' values") from None
    n = len(recs)

    def series(fld: str, **kw: Any) -> List[float]:
        return [_rec_float(r, fld, **kw) for r in recs]

    demand = [max(_rec_sector_total(r, "demands"), 0.0) for r in recs]
    deficit = [max(_rec_sector_total(r, "deficits"), 0.0) for r in recs]
    withdrawal = [max(_rec_sector_total(r, "withdrawals"), 0.0) for r in recs]  # surface water only
    groundwater = [_rec_optional_nonneg(r, "groundwater_used_mm3") for r in recs]
    # FAO total freshwater withdrawal (surface + groundwater): SDG 6.4.1 denominator
    freshwater = [w + g for w, g in zip(withdrawal, groundwater)]
    supplied = [max(d - df, 0.0) for d, df in zip(demand, deficit)]
    supply_ratio = [min(max(1.0 - df / d, 0.0), 1.0) if d > 0.0 else 1.0 for d, df in zip(demand, deficit)]
    env_met = [bool(_rec_get(r, "env_flow_met")) for r in recs]
    entitlement = series("entitlement_mm3", allow_none_inf=True)
    coop_years = [cooperation and math.isfinite(e) for e in entitlement]

    population = series("population")
    gdp = series("gdp_usd")
    nexus = series("nexus_index")
    supply_gwh = series("energy_supply_gwh")
    demand_gwh = series("energy_demand_gwh")
    hydro = series("hydropower_gwh")
    other_ren = series("other_renewable_gwh")
    emissions = series("emissions_t")
    kcal = series("food_kcal")
    kcal_demand = series("food_demand_kcal")
    stress = series("water_stress_sdg642")
    per_capita = series("per_capita_water_m3")

    sum_supply_gwh = sum(supply_gwh)
    renewable_share = min(max(_safe_ratio(sum(hydro) + sum(other_ren), sum_supply_gwh, 0.0), 0.0), 1.0)
    emission_intensity = _safe_ratio(sum(emissions), sum_supply_gwh, 0.0)
    energy_self_sufficiency = _safe_ratio(sum_supply_gwh, sum(demand_gwh), 1.0)
    food_ss = _safe_ratio(sum(kcal), sum(kcal_demand), 1.0)
    mean_stress = _mean(stress)
    mean_per_capita = _mean(per_capita)
    coop_share = sum(1.0 for c in coop_years if c) / n

    out: Dict[str, Any] = {
        "years": n,
        "first_year": _rec_get(recs[0], "year"),
        "last_year": _rec_get(recs[-1], "year"),
        "population": _mean(population),
        "gdp_usd": _mean(gdp),
        # indices (means over the horizon)
        "water_security": _mean(series("water_security")),
        "energy_security": _mean(series("energy_security")),
        "food_security": _mean(series("food_security")),
        "nexus_index": _mean(nexus),
        "min_nexus_index": min(nexus),
        "nexus_index_trend": nexus[-1] - nexus[0],
        # water
        "supply_ratio": _mean(supply_ratio),
        "min_supply_ratio": min(supply_ratio),
        "supply_reliability": supply_reliability(supplied, demand),
        "resilience": resilience(supplied, demand),
        "vulnerability": vulnerability(supplied, demand),
        "total_demand_mm3": _mean(demand),
        "total_withdrawal_mm3": _mean(withdrawal),
        "total_deficit_mm3": _mean(deficit),
        "groundwater_used_mm3": _mean(groundwater),
        "freshwater_withdrawal_mm3": _mean(freshwater),
        "inflow_mm3": _mean(series("inflow_mm3")),
        "upstream_inflow_mm3": _mean(series("upstream_inflow_mm3")),
        "outflow_mm3": _mean(series("outflow_mm3")),
        "storage_end_mm3": _mean(series("storage_end_mm3")),
        "final_storage_mm3": _rec_float(recs[-1], "storage_end_mm3"),
        "environmental_flow_mm3": _mean(series("environmental_flow_mm3")),
        "env_flow_met_share": sum(1.0 for m in env_met if m) / n,
        "years_env_flow_unmet": sum(1 for m in env_met if not m),
        "entitlement_mm3": _mean(entitlement),
        "water_stress_sdg642": mean_stress,
        "groundwater_stress": _mean(series("groundwater_stress")),
        "per_capita_water_m3": mean_per_capita,
        "falkenmark": falkenmark_category(max(mean_per_capita, 0.0)),
        "transboundary_cooperation": coop_share,
        # energy
        "hydropower_gwh": _mean(hydro),
        "thermal_gwh": _mean(series("thermal_gwh")),
        "other_renewable_gwh": _mean(other_ren),
        "energy_supply_gwh": _mean(supply_gwh),
        "energy_demand_gwh": _mean(demand_gwh),
        "energy_for_water_gwh": _mean(series("energy_for_water_gwh")),
        "energy_deficit_gwh": _mean(series("energy_deficit_gwh")),
        "emissions_t": _mean(emissions),
        "energy_self_sufficiency": energy_self_sufficiency,
        "renewable_share": renewable_share,
        "emission_intensity_t_per_gwh": emission_intensity,
        # food
        "food_production_t": _mean(series("food_production_t")),
        "food_kcal": _mean(kcal),
        "food_demand_kcal": _mean(kcal_demand),
        "food_self_sufficiency": food_ss,
        "et_ratio": _mean(series("et_ratio")),
        "irrigation_requirement_mm3": _mean(series("irrigation_requirement_mm3")),
        "crop_value_usd": _mean(series("crop_value_usd")),
        "water_value_usd": _mean(series("water_value_usd")),
    }
    out.update(
        sdg_indicators(
            gdp_usd=max(sum(gdp), 0.0),
            withdrawal_mm3=sum(freshwater),  # surface + groundwater (FAO 6.4.1)
            water_stress_pct=max(mean_stress, 0.0),
            renewable_share=renewable_share,
            food_self_sufficiency=max(food_ss, 0.0),
            transboundary_cooperation=coop_share,
        )
    )
    return out


def _equity_of_populated(per_capita_by_riparian: Mapping[str, float]) -> float:
    """``1 - Gini`` of per-capita water over the riparians that have people.

    A riparian with zero population has no per-capita availability
    (:class:`~wefnexus.nexus.RiparianYear` records it as ``inf``); nobody
    lives there among whom water could be shared unequally, so it is left
    out of the Gini.  When no riparian is populated the distribution is
    trivially equal and ``1.0`` is returned.

    Parameters
    ----------
    per_capita_by_riparian : mapping of str -> float
        Mean per-capita water of every riparian, m3/person/yr (``inf`` for
        an unpopulated riparian).

    Returns
    -------
    float
        :func:`equity_index` of the finite entries (negative values clipped
        to 0), or ``1.0`` when there is none.
    """
    populated = {
        name: max(float(v), 0.0)
        for name, v in per_capita_by_riparian.items()
        if math.isfinite(float(v))
    }
    return equity_index(populated) if populated else 1.0


def _basin_summary(
    riparians: Dict[str, Dict[str, Any]],
    recs: List[Any],
    years: List[Any],
    balances: Optional[Sequence[Any]],
) -> Dict[str, Any]:
    """Basin-level aggregates from the per-riparian summaries and all records."""
    names = list(riparians)
    n_rip = len(names)
    by_year: Dict[Any, List[Any]] = {}
    for r in recs:
        by_year.setdefault(_rec_get(r, "year"), []).append(r)
    year_keys = list(years)
    unknown = [y for y in year_keys if y not in by_year]
    if not year_keys or unknown:  # assess_records validates; guard the private entry point too
        raise ValueError(
            f"years must be a non-empty subset of the record years {sorted(by_year)!r}, "
            f"got {year_keys!r} (no records for {unknown!r})"
        )

    def yearly(fn: Any) -> List[float]:
        return [float(sum(fn(r) for r in by_year[y])) for y in year_keys]

    demand_y = yearly(lambda r: max(_rec_sector_total(r, "demands"), 0.0))
    deficit_y = yearly(lambda r: max(_rec_sector_total(r, "deficits"), 0.0))
    withdrawal_y = yearly(lambda r: max(_rec_sector_total(r, "withdrawals"), 0.0))  # surface only
    groundwater_y = yearly(lambda r: _rec_optional_nonneg(r, "groundwater_used_mm3"))
    freshwater_y = [w + g for w, g in zip(withdrawal_y, groundwater_y)]  # FAO TFWW (SDG 6.4.1)
    supplied_y = [max(d - df, 0.0) for d, df in zip(demand_y, deficit_y)]
    ratio_y = [min(max(1.0 - df / d, 0.0), 1.0) if d > 0.0 else 1.0 for d, df in zip(demand_y, deficit_y)]
    pop_y = yearly(lambda r: _rec_float(r, "population"))
    gdp_y = yearly(lambda r: _rec_float(r, "gdp_usd"))
    supply_gwh_y = yearly(lambda r: _rec_float(r, "energy_supply_gwh"))
    demand_gwh_y = yearly(lambda r: _rec_float(r, "energy_demand_gwh"))
    renewable_gwh_y = yearly(lambda r: _rec_float(r, "hydropower_gwh") + _rec_float(r, "other_renewable_gwh"))
    emissions_y = yearly(lambda r: _rec_float(r, "emissions_t"))
    kcal_y = yearly(lambda r: _rec_float(r, "food_kcal"))
    kcal_demand_y = yearly(lambda r: _rec_float(r, "food_demand_kcal"))

    # environmental flow: ``env_flow_met_share`` over the riparian-years of the
    # assessed horizon, ``years_env_flow_unmet`` over YEARS in which any reach
    # fell short (the convention of NexusResult.summary / comparison_table), so
    # the basin count can never exceed ``len(year_keys)``.
    env_by_year = {y: [bool(_rec_get(r, "env_flow_met")) for r in by_year[y]] for y in year_keys}
    env_flags = [met for y in year_keys for met in env_by_year[y]]
    years_env_unmet = sum(1 for y in year_keys if not all(env_by_year[y]))
    mean_pop = {name: riparians[name]["population"] for name in names}
    total_pop = sum(mean_pop.values())

    def rip_mean(key: str) -> float:
        return _mean(riparians[name][key] for name in names)

    def rip_pop_weighted(key: str) -> float:
        if total_pop <= 0.0:
            return rip_mean(key)
        return float(sum(riparians[name][key] * mean_pop[name] for name in names) / total_pop)

    nexus_by_rip = {name: riparians[name]["nexus_index"] for name in names}
    worst = min(names, key=lambda nm: nexus_by_rip[nm])
    total_supply_gwh = sum(supply_gwh_y)
    renewable_share = min(max(_safe_ratio(sum(renewable_gwh_y), total_supply_gwh, 0.0), 0.0), 1.0)
    food_ss = _safe_ratio(sum(kcal_y), sum(kcal_demand_y), 1.0)
    mean_stress = rip_mean("water_stress_sdg642")
    coop_share = rip_mean("transboundary_cooperation")

    out: Dict[str, Any] = {
        "riparians": n_rip,
        "years": len(year_keys),
        "population": _mean(pop_y),
        "gdp_usd": _mean(gdp_y),
        # indices: riparian means + population-weighted nexus
        "water_security": rip_mean("water_security"),
        "energy_security": rip_mean("energy_security"),
        "food_security": rip_mean("food_security"),
        "nexus_index": rip_mean("nexus_index"),
        "nexus_index_population_weighted": rip_pop_weighted("nexus_index"),
        "min_nexus_index": nexus_by_rip[worst],
        "worst_riparian": worst,
        "nexus_index_trend": rip_mean("nexus_index_trend"),
        # equity
        "equity_index": equity_index({name: riparians[name]["supply_ratio"] for name in names}),
        "equity_nexus_index": equity_index(nexus_by_rip),
        "equity_per_capita_water": _equity_of_populated(
            {name: riparians[name]["per_capita_water_m3"] for name in names}
        ),
        # water
        "supply_ratio": min(max(1.0 - _safe_ratio(sum(deficit_y), sum(demand_y), 0.0), 0.0), 1.0),
        "min_supply_ratio": min(ratio_y),
        "supply_reliability": supply_reliability(supplied_y, demand_y),
        "resilience": resilience(supplied_y, demand_y),
        "vulnerability": vulnerability(supplied_y, demand_y),
        "total_demand_mm3": _mean(demand_y),
        "total_withdrawal_mm3": _mean(withdrawal_y),
        "total_deficit_mm3": _mean(deficit_y),
        "groundwater_used_mm3": _mean(groundwater_y),
        "freshwater_withdrawal_mm3": _mean(freshwater_y),
        "env_flow_met_share": sum(1.0 for m in env_flags if m) / len(env_flags),
        "years_env_flow_unmet": years_env_unmet,
        "water_stress_sdg642": mean_stress,
        "groundwater_stress": rip_mean("groundwater_stress"),
        "transboundary_cooperation": coop_share,
        # energy
        "hydropower_gwh": _mean(yearly(lambda r: _rec_float(r, "hydropower_gwh"))),
        "energy_supply_gwh": _mean(supply_gwh_y),
        "energy_demand_gwh": _mean(demand_gwh_y),
        "energy_self_sufficiency": _safe_ratio(total_supply_gwh, sum(demand_gwh_y), 1.0),
        "renewable_share": renewable_share,
        "emission_intensity_t_per_gwh": _safe_ratio(sum(emissions_y), total_supply_gwh, 0.0),
        "emissions_t": _mean(emissions_y),
        # food
        "food_kcal": _mean(kcal_y),
        "food_demand_kcal": _mean(kcal_demand_y),
        "food_self_sufficiency": food_ss,
        "water_value_usd": _mean(yearly(lambda r: _rec_float(r, "water_value_usd"))),
        "crop_value_usd": _mean(yearly(lambda r: _rec_float(r, "crop_value_usd"))),
        # from routed balances (filled below when available)
        "outflow_to_sea_mm3": None,
        "natural_flow_mm3": None,
        "per_capita_water_m3": None,
        "falkenmark": None,
        "mass_balance_error_mm3": None,
    }
    out.update(
        sdg_indicators(
            gdp_usd=max(sum(gdp_y), 0.0),
            withdrawal_mm3=sum(freshwater_y),  # surface + groundwater (FAO 6.4.1)
            water_stress_pct=max(mean_stress, 0.0),
            renewable_share=renewable_share,
            food_self_sufficiency=max(food_ss, 0.0),
            transboundary_cooperation=min(max(coop_share, 0.0), 1.0),
        )
    )

    bals = list(balances) if balances else []
    if bals:
        usable = [b for b in bals if hasattr(b, "outflow_to_sea_mm3") and hasattr(b, "natural_flow_mm3")]
        if len(usable) != len(bals):
            raise ValueError("balances must be wefnexus.water.BasinBalance objects")
        out["outflow_to_sea_mm3"] = _mean(float(b.outflow_to_sea_mm3) for b in usable)
        out["natural_flow_mm3"] = _mean(float(b.natural_flow_mm3) for b in usable)
        errs = [float(b.mass_balance_error()) for b in usable if hasattr(b, "mass_balance_error")]
        out["mass_balance_error_mm3"] = max(errs) if errs else None
        if len(usable) == len(pop_y) and all(p > 0.0 for p in pop_y):
            pcs = [float(b.natural_flow_mm3) * _M3_PER_MM3 / p for b, p in zip(usable, pop_y)]
            out["per_capita_water_m3"] = _mean(pcs)
            out["falkenmark"] = falkenmark_category(max(out["per_capita_water_m3"], 0.0))
    return out


def assess_records(
    records: Iterable[Any],
    *,
    basin_name: str = "",
    scenario_name: str = "",
    years: Optional[Sequence[int]] = None,
    balances: Optional[Sequence[Any]] = None,
    cooperation: bool = True,
) -> SustainabilityReport:
    """Build a :class:`SustainabilityReport` from ``RiparianYear``-like records.

    This is the engine behind :func:`assess`; it only needs objects exposing
    the :data:`RIPARIAN_YEAR_FIELDS` attributes (``demands``,
    ``withdrawals`` and ``deficits`` as ``{sector: Mm3}`` dicts), so it can
    be used with any record source.  Records are grouped by ``name`` (first
    appearance order) and sorted by ``year`` within each group.

    Parameters
    ----------
    records : iterable
        Non-empty sequence of per-riparian, per-year records.
    basin_name, scenario_name : str, optional
        Labels for the report title.
    years : sequence of int, optional
        Years to assess (default: every distinct record year, sorted).  When
        given it must be a non-empty, duplicate-free subset of the record
        years; it is stored sorted as ``report.years`` and the per-riparian
        and basin aggregates then cover exactly those years
        (``basin["years"] == len(report.years)``).  Years that have no
        records raise ``ValueError`` - there is no silent fallback.
    balances : sequence of BasinBalance, optional
        One routed :class:`~wefnexus.water.BasinBalance` per year; adds
        outflow to sea, natural flow, basin per-capita water and the maximum
        mass-balance error to the basin dict.
    cooperation : bool, optional
        Whether the scenario ran with cooperation; SDG 6.5.2 per riparian is
        the share of years with ``cooperation`` and a finite
        ``entitlement_mm3`` (an operational sharing arrangement).

    Returns
    -------
    SustainabilityReport
        ``riparians[name]`` holds, per riparian: ``years``, ``first_year``,
        ``last_year``, ``population``, ``gdp_usd`` (means), the four index
        means (:data:`INDEX_KEYS`), ``min_nexus_index``,
        ``nexus_index_trend`` (last - first), ``supply_ratio`` (mean of
        annual ``1 - deficit/demand``), ``min_supply_ratio``,
        ``supply_reliability`` / ``resilience`` / ``vulnerability``
        (Hashimoto et al. 1982 on supplied vs demanded volumes),
        ``total_demand_mm3`` / ``total_withdrawal_mm3`` /
        ``total_deficit_mm3`` (means), ``inflow_mm3``,
        ``upstream_inflow_mm3``, ``outflow_mm3``, ``storage_end_mm3``
        (means), ``groundwater_used_mm3`` (mean of the optional record
        field, 0 when absent), ``freshwater_withdrawal_mm3`` (mean surface
        + groundwater withdrawal, the FAO total freshwater withdrawal),
        ``final_storage_mm3``, ``environmental_flow_mm3``,
        ``env_flow_met_share``, ``years_env_flow_unmet``,
        ``entitlement_mm3``, ``water_stress_sdg642`` (mean %, ``inf``
        propagates), ``groundwater_stress``, ``per_capita_water_m3``,
        ``falkenmark`` (class of the mean per-capita water),
        ``transboundary_cooperation``, energy means (``hydropower_gwh``,
        ``thermal_gwh``, ``other_renewable_gwh``, ``energy_supply_gwh``,
        ``energy_demand_gwh``, ``energy_for_water_gwh``,
        ``energy_deficit_gwh``, ``emissions_t``), ``energy_self_sufficiency``
        (total supply / total demand), ``renewable_share`` (renewable /
        total supply), ``emission_intensity_t_per_gwh``, food means
        (``food_production_t``, ``food_kcal``, ``food_demand_kcal``,
        ``et_ratio``, ``irrigation_requirement_mm3``, ``crop_value_usd``,
        ``water_value_usd``), ``food_self_sufficiency`` (total kcal / total
        demand) and the five :data:`SDG_KEYS` (6.4.1 = total GDP / total
        freshwater withdrawal, i.e. surface + groundwater, in USD/m3).
        ``basin`` holds the aggregates listed in
        :class:`SustainabilityReport`; its ``years_env_flow_unmet`` counts
        years with any unmet reach, its ``env_flow_met_share`` is over
        riparian-years.

    Raises
    ------
    ValueError
        On an empty record set, a record missing a contract field, a
        non-numeric value, malformed ``balances``, or a ``years`` argument
        that is empty, has duplicates or names years without records.
    """
    recs = list(records)
    if not recs:
        raise ValueError("records must not be empty")
    try:
        record_years = {_rec_get(r, "year") for r in recs}
        year_list = sorted(record_years)
    except TypeError:
        raise ValueError("records have incomparable or unhashable 'year' values") from None
    if years is not None:
        year_list = _validate_years(years, record_years)
        if len(year_list) != len(record_years):
            keep = set(year_list)  # restrict every aggregate to the requested horizon
            recs = [r for r in recs if _rec_get(r, "year") in keep]
    groups: Dict[str, List[Any]] = {}
    for rec in recs:
        name = _rec_get(rec, "name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"record name must be a non-empty string, got {name!r}")
        groups.setdefault(name, []).append(rec)
    coop = bool(cooperation)
    riparians = {name: _riparian_summary(name, group, coop) for name, group in groups.items()}
    basin = _basin_summary(riparians, recs, year_list, balances)
    return SustainabilityReport(
        basin_name=str(basin_name),
        scenario_name=str(scenario_name),
        years=year_list,
        riparians=riparians,
        basin=basin,
    )


def assess(result: "NexusResult") -> SustainabilityReport:  # noqa: F821 - lazy import
    """Sustainability assessment of a :class:`~wefnexus.nexus.NexusResult`.

    Iterates ``result.records``, groups them by riparian, averages the water
    / energy / food / nexus indices over the horizon, computes Hashimoto
    reliability-resilience-vulnerability of water supply, equity between
    riparians, SDG indicators and basin aggregates, and returns them as a
    :class:`SustainabilityReport` (``report.summary()`` gives a text table).

    :class:`~wefnexus.nexus.NexusResult` is imported lazily inside this
    function to avoid a circular import; any object with the same
    attributes (``records``, ``basin_name``, ``scenario``, ``years``,
    ``balances``) is accepted as well.  A missing or empty ``years``
    attribute means "the years of the records"; a non-empty one must match
    record years (see :func:`assess_records`).

    SDG 6.4.1 uses the FAO total freshwater withdrawal (surface withdrawal
    plus ``groundwater_used_mm3``) as its denominator, and the basin
    ``years_env_flow_unmet`` counts years with any unmet reach, exactly as
    :meth:`~wefnexus.nexus.NexusResult.summary` does.

    Parameters
    ----------
    result : NexusResult
        Output of :func:`wefnexus.nexus.run_nexus` /
        :meth:`wefnexus.nexus.NexusModel.run`.

    Returns
    -------
    SustainabilityReport
        See :func:`assess_records` for the keys of the per-riparian and basin
        dictionaries.

    Raises
    ------
    ValueError
        If ``result`` is not a nexus result (no ``records``), has no records,
        or its records lack the section-7 fields.

    References
    ----------
    Hoff (2011); Hashimoto et al. (1982); Simpson et al. (2022).
    """
    try:
        from wefnexus.nexus import NexusResult  # lazy: nexus imports this module
    except ImportError:  # pragma: no cover - nexus module not available
        NexusResult = None  # type: ignore[assignment]
    if (NexusResult is None or not isinstance(result, NexusResult)) and not hasattr(result, "records"):
        raise ValueError(
            "assess() expects a wefnexus.nexus.NexusResult (or an object exposing 'records', "
            f"'basin_name', 'scenario', 'years' and 'balances'), got {type(result).__name__}"
        )
    scenario = getattr(result, "scenario", None)
    if scenario is None:
        scenario_name = ""
    else:
        scenario_name = str(getattr(scenario, "name", scenario))
    cooperation = bool(getattr(scenario, "cooperation", True))
    years = getattr(result, "years", None)
    if years is not None:
        try:
            years = list(years) or None  # an empty horizon: take the years of the records
        except TypeError:
            raise ValueError(f"result.years must be a sequence of years, got {years!r}") from None
    return assess_records(
        result.records,
        basin_name=str(getattr(result, "basin_name", "")),
        scenario_name=scenario_name,
        years=years,
        balances=getattr(result, "balances", None),
        cooperation=cooperation,
    )
