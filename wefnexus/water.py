"""Water-security indicators and basin water-balance routing.

This module is a *leaf* of the :mod:`wefnexus` package: it depends only on
:mod:`wefnexus.models` and NumPy.  It provides

* classic water-scarcity indicators (Falkenmark per-capita availability,
  water exploitation index, SDG 6.4.2 water stress, FAO AQUASTAT dependency
  ratio, groundwater stress) and the Hashimoto reliability / resilience /
  vulnerability performance criteria;
* :func:`route_basin`, an annual upstream-to-downstream water-balance
  routing of a transboundary basin that honours treaty entitlements,
  environmental-flow requirements, sector priorities, reservoir operation
  and non-river (groundwater / desalination) supply, with an exact
  per-reach mass balance;
* :func:`natural_flows` and :func:`stochastic_flow_factors`, helpers used by
  the multi-year nexus simulation.

Units
-----
Water volumes are **Mm3/yr** (1 Mm3 = 1e6 m3) unless stated otherwise;
per-capita availability is **m3/person/yr**; indices are dimensionless.

References
----------
Falkenmark, M., Lundqvist, J. & Widstrand, C. (1989). Macro-scale water
    scarcity requires micro-scale approaches. *Natural Resources Forum*
    13(4), 258-267.
Raskin, P., Gleick, P., Kirshen, P., Pontius, G. & Strzepek, K. (1997).
    *Water Futures: Assessment of Long-range Patterns and Problems.*
    Stockholm Environment Institute (water-resources vulnerability index).
FAO (2018). *Progress on Level of Water Stress - Global baseline for SDG
    indicator 6.4.2.* FAO/UN-Water, Rome.
FAO AQUASTAT (2016). *Glossary - dependency ratio.*
Gleeson, T., Wada, Y., Bierkens, M.F.P. & van Beek, L.P.H. (2012). Water
    balance of global aquifers revealed by groundwater footprint. *Nature*
    488, 197-200.
Hashimoto, T., Stedinger, J.R. & Loucks, D.P. (1982). Reliability,
    resiliency, and vulnerability criteria for water resource system
    performance evaluation. *Water Resources Research* 18(1), 14-20.
Loucks, D.P. & van Beek, E. (2005). *Water Resources Systems Planning and
    Management.* UNESCO, Paris (ch. 11, performance criteria; ch. 12,
    simulation).
Yates, D., Sieber, J., Purkey, D. & Huber-Lee, A. (2005). WEAP21 - a
    demand-, priority-, and preference-driven water planning model.
    *Water International* 30(4), 487-500.
Wurbs, R.A. (1993). Reservoir-system simulation and optimization models.
    *Journal of Water Resources Planning and Management* 119(4), 455-472.
Stedinger, J.R. (1980). Fitting log normal distributions to hydrologic
    data. *Water Resources Research* 16(3), 481-490.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from wefnexus.models import (
    SECTOR_PRIORITY,
    Basin,
    Riparian,
    Scenario,
    Sector,
    WaterDemand,
)

__all__ = [
    "FALKENMARK_THRESHOLDS",
    "per_capita_water",
    "falkenmark_category",
    "water_exploitation_index",
    "sdg_642_water_stress",
    "dependency_ratio",
    "groundwater_stress",
    "supply_reliability",
    "resilience",
    "vulnerability",
    "ReachResult",
    "BasinBalance",
    "route_basin",
    "natural_flows",
    "stochastic_flow_factors",
]

#: Falkenmark (1989) per-capita renewable-water thresholds, m3/person/yr.
#: ``>= 1700`` no stress, ``1000-1700`` stress, ``500-1000`` scarcity,
#: ``< 500`` absolute scarcity.
FALKENMARK_THRESHOLDS: Dict[str, float] = {
    "no_stress": 1700.0,
    "stress": 1000.0,
    "scarcity": 500.0,
}

#: A year counts as a supply failure when ``supplied < RELIABILITY_TOL * demanded``.
RELIABILITY_TOL = 0.999

#: Tolerance (Mm3) used when deciding whether an environmental flow was met.
_ENV_TOL = 1e-9


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------
def _as_float(value: Any, label: str, *, allow_inf: bool = False) -> float:
    """Coerce ``value`` to a finite (or optionally infinite) float."""
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


def _series(values: Sequence[float], label: str) -> np.ndarray:
    try:
        arr = np.asarray(list(values), dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a sequence of numbers") from None
    if arr.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional")
    if arr.size == 0:
        raise ValueError(f"{label} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must contain only finite numbers")
    if np.any(arr < 0.0):
        raise ValueError(f"{label} must be non-negative")
    return arr


def _failure_mask(supplied: Sequence[float], demanded: Sequence[float]) -> np.ndarray:
    """Boolean mask of failure years (``supplied < 0.999 * demanded``)."""
    s = _series(supplied, "supplied")
    d = _series(demanded, "demanded")
    if s.shape != d.shape:
        raise ValueError(
            f"supplied and demanded must have the same length, got {s.size} and {d.size}"
        )
    return s < RELIABILITY_TOL * d


# ---------------------------------------------------------------------------
# indicators
# ---------------------------------------------------------------------------
def per_capita_water(renewable_mm3: float, population: float) -> float:
    """Renewable water availability per person.

    Parameters
    ----------
    renewable_mm3 : float
        Total renewable water resources, Mm3/yr (>= 0).
    population : float
        Number of people (> 0).

    Returns
    -------
    float
        Availability in m3 per person per year.

    Raises
    ------
    ValueError
        If ``population <= 0`` or ``renewable_mm3 < 0``.

    References
    ----------
    Falkenmark et al. (1989); FAO AQUASTAT "total renewable water resources
    per capita".

    Examples
    --------
    >>> per_capita_water(1000.0, 1_000_000)
    1000.0
    """
    renewable = _nonneg(renewable_mm3, "renewable_mm3")
    pop = _as_float(population, "population")
    if pop <= 0.0:
        raise ValueError(f"population must be > 0, got {pop}")
    return renewable * 1e6 / pop


def falkenmark_category(per_capita_m3: float) -> str:
    """Classify per-capita water availability on the Falkenmark scale.

    Parameters
    ----------
    per_capita_m3 : float
        Renewable water per person, m3/person/yr (>= 0).

    Returns
    -------
    str
        ``"no_stress"`` (>= 1700), ``"stress"`` (1000-1700),
        ``"scarcity"`` (500-1000) or ``"absolute_scarcity"`` (< 500).
        Boundaries belong to the less-stressed class (e.g. 1000 is
        ``"stress"``).

    References
    ----------
    Falkenmark, Lundqvist & Widstrand (1989), *Natural Resources Forum*
    13(4).
    """
    v = _nonneg(per_capita_m3, "per_capita_m3", allow_inf=True)
    if v >= FALKENMARK_THRESHOLDS["no_stress"]:
        return "no_stress"
    if v >= FALKENMARK_THRESHOLDS["stress"]:
        return "stress"
    if v >= FALKENMARK_THRESHOLDS["scarcity"]:
        return "scarcity"
    return "absolute_scarcity"


def water_exploitation_index(withdrawal_mm3: float, renewable_mm3: float) -> float:
    """Water exploitation index, WEI = withdrawal / renewable resources.

    Parameters
    ----------
    withdrawal_mm3 : float
        Total freshwater withdrawal, Mm3/yr (>= 0).
    renewable_mm3 : float
        Total renewable freshwater resources, Mm3/yr (>= 0).

    Returns
    -------
    float
        Dimensionless ratio.  Conventional thresholds: 0.2 = moderate
        stress, 0.4 = severe stress (Raskin et al. 1997; EEA WEI).
        Returns ``inf`` when ``renewable_mm3 <= 0`` and withdrawal is
        positive, ``0.0`` when both are zero.

    References
    ----------
    Raskin et al. (1997); European Environment Agency, indicator CSI 018.
    """
    w = _nonneg(withdrawal_mm3, "withdrawal_mm3")
    r = _nonneg(renewable_mm3, "renewable_mm3")
    if r <= 0.0:
        return math.inf if w > 0.0 else 0.0
    return w / r


def sdg_642_water_stress(
    withdrawal_mm3: float, renewable_mm3: float, environmental_flow_mm3: float
) -> float:
    """SDG indicator 6.4.2 "level of water stress" in percent.

    ``stress = 100 * withdrawal / (renewable - environmental_flow)``.

    Parameters
    ----------
    withdrawal_mm3 : float
        Total freshwater withdrawal of all sectors, Mm3/yr (>= 0).
    renewable_mm3 : float
        Total renewable freshwater resources, Mm3/yr (>= 0).
    environmental_flow_mm3 : float
        Environmental flow requirement, Mm3/yr (>= 0).

    Returns
    -------
    float
        Percent.  FAO classes: < 25 no stress, 25-50 low, 50-75 medium,
        75-100 high, > 100 critical.  Returns ``inf`` if the denominator
        ``renewable - environmental_flow`` is ``<= 0`` (no water is
        available beyond the environmental requirement).

    References
    ----------
    FAO (2018) *Progress on level of water stress*, SDG 6.4.2 metadata.
    """
    w = _nonneg(withdrawal_mm3, "withdrawal_mm3")
    r = _nonneg(renewable_mm3, "renewable_mm3")
    e = _nonneg(environmental_flow_mm3, "environmental_flow_mm3")
    denom = r - e
    if denom <= 0.0:
        return math.inf
    return 100.0 * w / denom


def dependency_ratio(external_inflow_mm3: float, total_renewable_mm3: float) -> float:
    """FAO AQUASTAT dependency ratio.

    Share of a territory's total renewable water resources that originates
    outside its borders (``external / total``).

    Parameters
    ----------
    external_inflow_mm3 : float
        Renewable water entering from outside the territory, Mm3/yr (>= 0).
    total_renewable_mm3 : float
        Total renewable water resources (internal + external), Mm3/yr.

    Returns
    -------
    float
        Ratio in ``[0, 1]``; ``0.0`` when there is no renewable water at all.

    Raises
    ------
    ValueError
        On negative inputs or if ``external > total`` (beyond rounding).

    References
    ----------
    FAO AQUASTAT glossary, "dependency ratio" (external renewable water
    resources as a percentage of total renewable water resources).
    """
    ext = _nonneg(external_inflow_mm3, "external_inflow_mm3")
    tot = _nonneg(total_renewable_mm3, "total_renewable_mm3")
    if tot <= 0.0:
        if ext > 0.0:
            raise ValueError(
                "external_inflow_mm3 is positive but total_renewable_mm3 is zero; "
                "external inflow is part of the total renewable resource"
            )
        return 0.0
    if ext > tot * (1.0 + 1e-9):
        raise ValueError(
            f"external_inflow_mm3 ({ext}) cannot exceed total_renewable_mm3 ({tot})"
        )
    return min(ext / tot, 1.0)


def groundwater_stress(abstraction_mm3: float, recharge_mm3: float) -> float:
    """Groundwater stress = abstraction / recharge (> 1 means overdraft).

    Parameters
    ----------
    abstraction_mm3 : float
        Groundwater abstraction, Mm3/yr (>= 0).
    recharge_mm3 : float
        Mean annual recharge, Mm3/yr (>= 0).

    Returns
    -------
    float
        Dimensionless.  ``inf`` if there is abstraction but no recharge,
        ``0.0`` if there is no abstraction.

    References
    ----------
    Gleeson et al. (2012) groundwater footprint, *Nature* 488; Wada et al.
    (2010) *Geophys. Res. Lett.* 37, L20402 (groundwater depletion).
    """
    a = _nonneg(abstraction_mm3, "abstraction_mm3")
    r = _nonneg(recharge_mm3, "recharge_mm3")
    if r <= 0.0:
        return math.inf if a > 0.0 else 0.0
    return a / r


def supply_reliability(supplied: Sequence[float], demanded: Sequence[float]) -> float:
    """Hashimoto reliability: share of years in which demand was met.

    A year is *satisfactory* when ``supplied >= 0.999 * demanded`` (the
    0.1 % tolerance absorbs floating-point noise).  Years with zero demand
    are always satisfactory.

    Parameters
    ----------
    supplied, demanded : sequence of float
        Annual supplied and demanded volumes (same units, same length,
        non-negative, non-empty).

    Returns
    -------
    float
        Reliability in ``[0, 1]``.

    References
    ----------
    Hashimoto, Stedinger & Loucks (1982), *Water Resour. Res.* 18(1), eq. 2.
    """
    fail = _failure_mask(supplied, demanded)
    return float(1.0 - fail.mean())


def resilience(supplied: Sequence[float], demanded: Sequence[float]) -> float:
    """Hashimoto resilience: probability of recovering from a failure year.

    Estimated as the number of transitions *failure -> success* divided by
    the number of failure years (Loucks & van Beek 2005, §11.3.3).  A
    failure in the final year that is never seen to recover counts as a
    non-recovery.  Returns ``1.0`` when there are no failures.

    Parameters
    ----------
    supplied, demanded : sequence of float
        Annual supplied and demanded volumes (same units and length).

    Returns
    -------
    float
        Resilience in ``[0, 1]``.

    References
    ----------
    Hashimoto, Stedinger & Loucks (1982), eq. 7; Loucks & van Beek (2005).
    """
    fail = _failure_mask(supplied, demanded)
    n_fail = int(fail.sum())
    if n_fail == 0:
        return 1.0
    recoveries = int(np.sum(fail[:-1] & ~fail[1:]))
    return recoveries / n_fail


def vulnerability(supplied: Sequence[float], demanded: Sequence[float]) -> float:
    """Hashimoto vulnerability: mean relative deficit over failure years.

    ``mean((demanded - supplied) / demanded)`` over years in which
    ``supplied < 0.999 * demanded``; ``0.0`` if there are no failures.

    Parameters
    ----------
    supplied, demanded : sequence of float
        Annual supplied and demanded volumes (same units and length).

    Returns
    -------
    float
        Dimensionless, in ``[0, 1]``.

    References
    ----------
    Hashimoto, Stedinger & Loucks (1982), eq. 9 (severity as relative
    deficit); Loucks & van Beek (2005).
    """
    fail = _failure_mask(supplied, demanded)
    if not fail.any():
        return 0.0
    s = np.asarray(list(supplied), dtype=float)[fail]
    d = np.asarray(list(demanded), dtype=float)[fail]
    rel = np.clip((d - s) / d, 0.0, 1.0)
    return float(rel.mean())


# ---------------------------------------------------------------------------
# routing results
# ---------------------------------------------------------------------------
@dataclass
class ReachResult:
    """Annual water balance of one riparian's reach (all volumes Mm3/yr).

    The fields satisfy the exact mass balance::

        inflow_mm3 + storage_start_mm3
            == total_consumption() + outflow_mm3 + storage_end_mm3 + evaporation_mm3

    Attributes
    ----------
    name : str
        Riparian name.
    inflow_mm3 : float
        Routed inflow from upstream plus local runoff this year.
    upstream_inflow_mm3 : float
        Part of ``inflow_mm3`` that arrived from upstream (headwater for the
        first reach).
    storage_start_mm3, storage_end_mm3 : float
        Reservoir storage at the start and end of the year.
    storage_release_mm3 : float
        Water taken out of the reservoir this year: releases to meet
        withdrawals that exceed river inflow plus releases that top the
        river up to the environmental flow.  Never exceeds
        ``storage_start_mm3``.
    storage_refill_mm3 : float
        Water diverted from the river into the reservoir this year.
    evaporation_mm3 : float
        Reservoir evaporation (fraction of end-of-year gross storage).
    non_river_supply_mm3 : float
        Groundwater abstraction + desalination applied to demand (capped at
        total demand).
    entitlement_mm3 : float
        Cap on surface withdrawal applied this year (``inf`` if none).
    withdrawals, consumption, deficits, demands : dict
        Per-sector surface withdrawal, consumptive use, unmet demand and
        gross demand (keys are the four withdrawal :class:`Sector` members).
        ``deficits[s] = demands[s] - withdrawals[s] - non-river share``.
    environmental_flow_mm3 : float
        In-stream flow requirement at the reach outlet.
    outflow_mm3 : float
        Flow leaving the reach to the next riparian (or the sea).
    env_flow_met : bool
        Whether ``outflow_mm3 >= environmental_flow_mm3`` (within 1e-9).
    """

    name: str
    inflow_mm3: float
    upstream_inflow_mm3: float
    storage_start_mm3: float
    storage_release_mm3: float
    storage_refill_mm3: float
    storage_end_mm3: float
    evaporation_mm3: float
    non_river_supply_mm3: float
    entitlement_mm3: float
    withdrawals: Dict[Sector, float] = field(default_factory=dict)
    consumption: Dict[Sector, float] = field(default_factory=dict)
    deficits: Dict[Sector, float] = field(default_factory=dict)
    demands: Dict[Sector, float] = field(default_factory=dict)
    environmental_flow_mm3: float = 0.0
    outflow_mm3: float = 0.0
    env_flow_met: bool = True

    # -- aggregates ---------------------------------------------------------
    def total_withdrawal(self) -> float:
        """Total surface-water withdrawal, Mm3/yr."""
        return float(sum(self.withdrawals.values()))

    def total_consumption(self) -> float:
        """Total consumptive use (not returned to the river), Mm3/yr."""
        return float(sum(self.consumption.values()))

    def total_deficit(self) -> float:
        """Total unmet withdrawal demand, Mm3/yr."""
        return float(sum(self.deficits.values()))

    def total_demand(self) -> float:
        """Total gross withdrawal demand (before non-river supply), Mm3/yr."""
        return float(sum(self.demands.values()))

    def return_flow(self) -> float:
        """Withdrawn water returned to the river, Mm3/yr."""
        return self.total_withdrawal() - self.total_consumption()

    def river_demand(self) -> Dict[Sector, float]:
        """Demand placed on the river after non-river supply, per sector."""
        return {s: self.withdrawals.get(s, 0.0) + self.deficits.get(s, 0.0) for s in self.demands}

    def supply_ratio(self) -> float:
        """``1 - total_deficit / total_demand`` (``1.0`` when demand is zero)."""
        demand = self.total_demand()
        if demand <= 0.0:
            return 1.0
        return max(0.0, min(1.0, 1.0 - self.total_deficit() / demand))

    def mass_balance_error(self) -> float:
        """Absolute violation of the reach mass balance, Mm3 (~0)."""
        lhs = self.inflow_mm3 + self.storage_start_mm3
        rhs = (
            self.total_consumption()
            + self.outflow_mm3
            + self.storage_end_mm3
            + self.evaporation_mm3
        )
        return abs(lhs - rhs)

    def to_dict(self) -> Dict[str, Any]:
        """Flat dictionary; sector dicts expand to ``withdrawal_municipal`` etc."""
        out: Dict[str, Any] = {
            "name": self.name,
            "inflow_mm3": self.inflow_mm3,
            "upstream_inflow_mm3": self.upstream_inflow_mm3,
            "storage_start_mm3": self.storage_start_mm3,
            "storage_release_mm3": self.storage_release_mm3,
            "storage_refill_mm3": self.storage_refill_mm3,
            "storage_end_mm3": self.storage_end_mm3,
            "evaporation_mm3": self.evaporation_mm3,
            "non_river_supply_mm3": self.non_river_supply_mm3,
            "entitlement_mm3": self.entitlement_mm3,
            "environmental_flow_mm3": self.environmental_flow_mm3,
            "outflow_mm3": self.outflow_mm3,
            "env_flow_met": self.env_flow_met,
            "total_withdrawal_mm3": self.total_withdrawal(),
            "total_consumption_mm3": self.total_consumption(),
            "total_deficit_mm3": self.total_deficit(),
            "total_demand_mm3": self.total_demand(),
            "supply_ratio": self.supply_ratio(),
        }
        for prefix, data in (
            ("withdrawal", self.withdrawals),
            ("consumption", self.consumption),
            ("deficit", self.deficits),
            ("demand", self.demands),
        ):
            for s, v in data.items():
                out[f"{prefix}_{Sector(s).value}"] = v
        return out


@dataclass
class BasinBalance:
    """Result of :func:`route_basin` for one year.

    Attributes
    ----------
    reaches : list of ReachResult
        Ordered upstream -> downstream.
    natural_flow_mm3 : float
        Headwater plus all local inflows after the flow factor (and local
        inflow factors) - the water that would reach the sea with zero use
        and no reservoirs.
    outflow_to_sea_mm3 : float
        Outflow of the most downstream reach.
    """

    reaches: List[ReachResult]
    natural_flow_mm3: float
    outflow_to_sea_mm3: float

    def reach(self, name: str) -> ReachResult:
        """Return the :class:`ReachResult` of riparian ``name`` (KeyError if absent)."""
        for r in self.reaches:
            if r.name == name:
                return r
        raise KeyError(f"no reach named {name!r}; reaches are {[r.name for r in self.reaches]}")

    def names(self) -> List[str]:
        return [r.name for r in self.reaches]

    def total_withdrawal(self) -> float:
        """Basin-wide surface withdrawal, Mm3/yr."""
        return float(sum(r.total_withdrawal() for r in self.reaches))

    def total_consumption(self) -> float:
        """Basin-wide consumptive use, Mm3/yr."""
        return float(sum(r.total_consumption() for r in self.reaches))

    def total_deficit(self) -> float:
        """Basin-wide unmet withdrawal demand, Mm3/yr."""
        return float(sum(r.total_deficit() for r in self.reaches))

    def total_demand(self) -> float:
        """Basin-wide gross withdrawal demand, Mm3/yr."""
        return float(sum(r.total_demand() for r in self.reaches))

    def supply_ratio(self) -> float:
        """Basin-wide ``1 - deficit / demand`` (``1.0`` if demand is zero)."""
        demand = self.total_demand()
        if demand <= 0.0:
            return 1.0
        return max(0.0, min(1.0, 1.0 - self.total_deficit() / demand))

    def withdrawals_by_riparian(self) -> Dict[str, float]:
        return {r.name: r.total_withdrawal() for r in self.reaches}

    def storages_end(self) -> Dict[str, float]:
        """End-of-year storages, ready to pass as ``storages`` next year."""
        return {r.name: r.storage_end_mm3 for r in self.reaches}

    def env_flow_met_share(self) -> float:
        """Share of reaches whose environmental flow was met (``1.0`` if none)."""
        if not self.reaches:
            return 1.0
        return sum(1 for r in self.reaches if r.env_flow_met) / len(self.reaches)

    def mass_balance_errors(self) -> Dict[str, float]:
        """Absolute mass-balance violation of each reach, Mm3."""
        return {r.name: r.mass_balance_error() for r in self.reaches}

    def mass_balance_error(self) -> float:
        """Largest absolute reach mass-balance violation, Mm3 (should be ~0).

        Each reach must satisfy
        ``inflow + storage_start == consumption + outflow + storage_end + evaporation``.
        """
        if not self.reaches:
            return 0.0
        return max(r.mass_balance_error() for r in self.reaches)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "natural_flow_mm3": self.natural_flow_mm3,
            "outflow_to_sea_mm3": self.outflow_to_sea_mm3,
            "total_withdrawal_mm3": self.total_withdrawal(),
            "total_consumption_mm3": self.total_consumption(),
            "total_deficit_mm3": self.total_deficit(),
            "supply_ratio": self.supply_ratio(),
            "mass_balance_error_mm3": self.mass_balance_error(),
            "reaches": [r.to_dict() for r in self.reaches],
        }


# ---------------------------------------------------------------------------
# routing
# ---------------------------------------------------------------------------
def _check_names(mapping: Optional[Mapping[str, Any]], names: Sequence[str], label: str) -> Dict[str, Any]:
    """Validate that an optional per-riparian mapping only names basin riparians."""
    if mapping is None:
        return {}
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{label} must be a dict keyed by riparian name, got {type(mapping).__name__}")
    unknown = [k for k in mapping if k not in names]
    if unknown:
        raise ValueError(
            f"{label} refers to unknown riparian(s) {unknown}; basin riparians are {list(names)}"
        )
    return dict(mapping)


def _normalise_demand(demand: Any, label: str) -> Dict[Sector, float]:
    """Return ``{sector: demand}`` for the four withdrawal sectors (missing = 0)."""
    if isinstance(demand, WaterDemand):
        demand = demand.withdrawals()
    if not isinstance(demand, Mapping):
        raise ValueError(f"{label} must be a dict mapping Sector -> Mm3/yr, got {type(demand).__name__}")
    out: Dict[Sector, float] = {s: 0.0 for s in SECTOR_PRIORITY}
    for key, value in demand.items():
        try:
            sector = Sector(key)
        except ValueError:
            raise ValueError(f"{label}: unknown sector {key!r}") from None
        if sector not in out:
            raise ValueError(
                f"{label}: {sector.value!r} is not a withdrawal sector "
                "(environmental flow is an in-stream requirement; use env_flows)"
            )
        out[sector] = _nonneg(value, f"{label}[{sector.value}]")
    return out


def _route_reach(
    r: Riparian,
    upstream_inflow: float,
    local_inflow: float,
    storage: float,
    demand: Dict[Sector, float],
    env: float,
    cap: float,
    refill_fraction: float,
) -> ReachResult:
    """Annual water balance of a single reach (see :func:`route_basin`)."""
    label = f"riparian {r.name!r}"
    fractions = {
        s: _fraction(r.demand.consumption_fraction.get(s, 0.0), f"{label} consumption_fraction[{s.value}]")
        for s in SECTOR_PRIORITY
    }
    capacity = _nonneg(r.reservoir_capacity_mm3, f"{label} reservoir_capacity_mm3")
    evap_fraction = _fraction(r.reservoir_evaporation_fraction, f"{label} reservoir_evaporation_fraction")
    groundwater = _nonneg(r.groundwater_abstraction_mm3, f"{label} groundwater_abstraction_mm3")
    desalination = _nonneg(r.energy.desalination_capacity_mm3, f"{label} energy.desalination_capacity_mm3")

    inflow = upstream_inflow + local_inflow

    # -- non-river supply offsets demand proportionally across sectors ------
    total_demand = sum(demand.values())
    non_river = min(groundwater + desalination, total_demand)
    if total_demand > 0.0:
        river_demand = {s: max(d - non_river * d / total_demand, 0.0) for s, d in demand.items()}
    else:
        river_demand = {s: 0.0 for s in demand}

    # -- serve sectors in priority order within the physical limits --------
    supply = inflow + storage                      # water available for withdrawal
    cap_withdrawal = min(cap, supply)              # entitlement and physical cap
    cap_consumption = max(supply - env, 0.0)       # environmental flow constraint
    withdrawals: Dict[Sector, float] = {}
    consumption: Dict[Sector, float] = {}
    cum_w = 0.0
    cum_c = 0.0
    for s in SECTOR_PRIORITY:
        f = fractions[s]
        rem_w = max(cap_withdrawal - cum_w, 0.0)
        rem_c = max(cap_consumption - cum_c, 0.0)
        limit_c = rem_c / f if f > 0.0 else math.inf
        w = max(min(river_demand[s], rem_w, limit_c), 0.0)
        c = w * f
        withdrawals[s] = w
        consumption[s] = c
        cum_w += w
        cum_c += c
    deficits = {s: max(river_demand[s] - withdrawals[s], 0.0) for s in SECTOR_PRIORITY}

    # -- reservoir operation ------------------------------------------------
    # release what withdrawals need beyond the river inflow (<= storage,
    # guaranteed because total withdrawal <= supply = inflow + storage)
    release = min(max(cum_w - inflow, 0.0), storage)
    storage_after = storage - release
    # water in the river after use: inflow + release - consumption
    # (= return flows + unused inflow); never negative because
    # consumption <= withdrawal <= inflow + release
    river = max(inflow + release - cum_c, 0.0)
    # top the river up to the environmental flow from storage if possible
    # (this is the spec's "outflow at least min(env, remaining)")
    env_release = min(max(env - river, 0.0), storage_after)
    release += env_release
    storage_after -= env_release
    river += env_release
    # refill: store a share of the river surplus above the environmental
    # flow, limited by the room left in the reservoir
    surplus = max(river - env, 0.0)
    room = max(capacity - storage_after, 0.0)
    refill = max(min(surplus * refill_fraction, room), 0.0)
    outflow = river - refill
    evaporation = (storage_after + refill) * evap_fraction
    storage_end = storage_after + refill - evaporation
    env_flow_met = bool(outflow >= env - _ENV_TOL)

    return ReachResult(
        name=r.name,
        inflow_mm3=inflow,
        upstream_inflow_mm3=upstream_inflow,
        storage_start_mm3=storage,
        storage_release_mm3=release,
        storage_refill_mm3=refill,
        storage_end_mm3=storage_end,
        evaporation_mm3=evaporation,
        non_river_supply_mm3=non_river,
        entitlement_mm3=cap,
        withdrawals=withdrawals,
        consumption=consumption,
        deficits=deficits,
        demands=dict(demand),
        environmental_flow_mm3=env,
        outflow_mm3=outflow,
        env_flow_met=env_flow_met,
    )


def route_basin(
    basin: Basin,
    flow_factor: float = 1.0,
    entitlements: Optional[Dict[str, float]] = None,
    demands: Optional[Dict[str, Dict[Sector, float]]] = None,
    storages: Optional[Dict[str, float]] = None,
    env_flows: Optional[Dict[str, float]] = None,
    local_inflow_factors: Optional[Dict[str, float]] = None,
    reservoir_refill_fraction: float = 0.25,
) -> BasinBalance:
    """Route one year of water through the basin, upstream to downstream.

    Each riparian's reach receives the routed outflow of its upstream
    neighbour plus its own local runoff, serves its sector demands in
    :data:`~wefnexus.models.SECTOR_PRIORITY` order (municipal, industrial,
    energy, agricultural) subject to its treaty entitlement and to the
    environmental flow requirement, operates its reservoir, and passes the
    remaining flow downstream.  This is a priority-driven annual water
    balance of the kind used in WEAP (Yates et al. 2005) and in the
    reservoir-system simulation models reviewed by Wurbs (1993).

    Algorithm for each reach (all in Mm3)::

        local   = local_inflow * flow_factor * local_inflow_factor
        inflow  = upstream_outflow + local
        supply  = inflow + storage_start
        nonriver = min(groundwater_abstraction + desalination, sum(demand))
        river_demand[s] = demand[s] * (1 - nonriver / sum(demand))

        serve sectors in SECTOR_PRIORITY order with
            cumulative withdrawal  <= min(entitlement, supply)
            cumulative consumption <= max(supply - env, 0)
            w_s = min(river_demand[s], remaining withdrawal cap,
                      remaining consumption cap / f_s)
            consumption_s = w_s * f_s          (f_s = consumption fraction)

        release  = max(withdrawal - inflow, 0)         # from storage
        river    = inflow + release - consumption      # incl. return flows
        river   += min(max(env - river, 0), storage - release)   # env release
        surplus  = max(river - env, 0)
        refill   = min(surplus * reservoir_refill_fraction, capacity - storage_after_release)
        outflow  = river - refill                      # >= env whenever supply >= env
        evaporation = (storage_after_release + refill) * evaporation_fraction
        storage_end = storage_after_release + refill - evaporation

    which satisfies, for every reach, the exact mass balance
    ``inflow + storage_start == consumption + outflow + storage_end + evaporation``.

    Parameters
    ----------
    basin : Basin
        Basin to route; never mutated.
    flow_factor : float, default 1.0
        Multiplier on headwater and local inflows (>= 0; 0 = no runoff).
    entitlements : dict, optional
        ``{riparian name: cap}`` overriding each riparian's
        ``treaty_allocation_mm3`` as the cap on *surface withdrawal* this
        year.  A value of ``None`` or ``math.inf`` removes the cap (pure
        upstream priority).  Riparians not listed keep their treaty
        allocation (``None`` = no cap).
    demands : dict, optional
        ``{riparian name: {Sector: Mm3}}`` overriding each riparian's
        ``demand.withdrawals()`` (a :class:`WaterDemand` is also accepted).
        Missing sectors count as zero; ``Sector.ENVIRONMENT`` is rejected.
    storages : dict, optional
        ``{riparian name: Mm3}`` start-of-year reservoir storage overriding
        ``reservoir_storage_mm3`` (use :meth:`BasinBalance.storages_end` to
        carry storage between years).
    env_flows : dict, optional
        ``{riparian name: Mm3}`` overriding ``demand.environmental``.
    local_inflow_factors : dict, optional
        ``{riparian name: factor}`` extra multiplier on that riparian's
        local inflow (e.g. spatially varying drought).
    reservoir_refill_fraction : float, default 0.25
        Share of the river surplus above the environmental flow that is
        stored each year, in ``[0, 1]``.

    Returns
    -------
    BasinBalance
        Per-reach results, natural flow and outflow to sea.

    Raises
    ------
    ValueError
        On negative or non-finite inputs, unknown riparian names, unknown or
        non-withdrawal sectors, fractions outside ``[0, 1]``.

    Notes
    -----
    * Environmental flow is a constraint on consumption, not a withdrawal:
      consumption is capped at ``supply - env`` and the reservoir releases
      water to keep the outflow at ``env``; hence ``env_flow_met`` is
      ``True`` whenever ``inflow + storage_start >= env``.
    * ``storage_release_mm3 <= storage_start_mm3`` always holds.
    * With ``flow_factor == 0`` only storage can be used; a basin with
      neither flow nor storage yields all-zero withdrawals.

    References
    ----------
    Yates et al. (2005) WEAP21, *Water International* 30(4); Wurbs (1993)
    *J. Water Resour. Plann. Manage.* 119(4); Loucks & van Beek (2005)
    ch. 12.
    """
    if not isinstance(basin, Basin):
        raise ValueError(f"basin must be a wefnexus.models.Basin, got {type(basin).__name__}")
    ff = _nonneg(flow_factor, "flow_factor")
    refill_fraction = _fraction(reservoir_refill_fraction, "reservoir_refill_fraction")
    names = basin.names()
    entitlements = _check_names(entitlements, names, "entitlements")
    demands = _check_names(demands, names, "demands")
    storages = _check_names(storages, names, "storages")
    env_flows = _check_names(env_flows, names, "env_flows")
    local_inflow_factors = _check_names(local_inflow_factors, names, "local_inflow_factors")

    headwater = _nonneg(basin.headwater_inflow_mm3, "basin.headwater_inflow_mm3")
    upstream = headwater * ff
    natural_flow = upstream
    reaches: List[ReachResult] = []
    for r in basin.riparians:
        label = f"riparian {r.name!r}"

        factor = local_inflow_factors.get(r.name)
        factor = 1.0 if factor is None else _nonneg(factor, f"local_inflow_factors[{r.name!r}]")
        local = _nonneg(r.local_inflow_mm3, f"{label} local_inflow_mm3") * ff * factor
        natural_flow += local

        storage = storages.get(r.name)
        storage = (
            _nonneg(r.reservoir_storage_mm3, f"{label} reservoir_storage_mm3")
            if storage is None
            else _nonneg(storage, f"storages[{r.name!r}]")
        )

        raw_demand = demands.get(r.name)
        demand = _normalise_demand(
            r.demand.withdrawals() if raw_demand is None else raw_demand,
            f"{label} demand" if raw_demand is None else f"demands[{r.name!r}]",
        )

        env = env_flows.get(r.name)
        env = (
            _nonneg(r.demand.environmental, f"{label} demand.environmental")
            if env is None
            else _nonneg(env, f"env_flows[{r.name!r}]")
        )

        if r.name in entitlements:
            cap_value = entitlements[r.name]
            cap = math.inf if cap_value is None else _nonneg(cap_value, f"entitlements[{r.name!r}]", allow_inf=True)
        elif r.treaty_allocation_mm3 is None:
            cap = math.inf
        else:
            cap = _nonneg(r.treaty_allocation_mm3, f"{label} treaty_allocation_mm3", allow_inf=True)

        reach = _route_reach(r, upstream, local, storage, demand, env, cap, refill_fraction)
        reaches.append(reach)
        upstream = reach.outflow_mm3

    return BasinBalance(reaches=reaches, natural_flow_mm3=natural_flow, outflow_to_sea_mm3=upstream)


def natural_flows(
    basin: Basin,
    flow_factor: float = 1.0,
    local_inflow_factors: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Natural (unregulated, zero-use) inflow at each reach, Mm3/yr.

    The natural inflow of reach *i* is the headwater inflow plus the local
    inflows of reaches ``0..i`` (cumulative downstream), all multiplied by
    ``flow_factor``; the last entry equals the basin's natural flow.

    Parameters
    ----------
    basin : Basin
        Basin (never mutated).
    flow_factor : float, default 1.0
        Multiplier on all inflows (>= 0).
    local_inflow_factors : dict, optional
        Extra per-riparian multipliers on local inflow, as in
        :func:`route_basin`.

    Returns
    -------
    dict
        ``{riparian name: natural inflow}`` ordered upstream -> downstream.
    """
    if not isinstance(basin, Basin):
        raise ValueError(f"basin must be a wefnexus.models.Basin, got {type(basin).__name__}")
    ff = _nonneg(flow_factor, "flow_factor")
    names = basin.names()
    local_inflow_factors = _check_names(local_inflow_factors, names, "local_inflow_factors")
    flow = _nonneg(basin.headwater_inflow_mm3, "basin.headwater_inflow_mm3") * ff
    out: Dict[str, float] = {}
    for r in basin.riparians:
        factor = local_inflow_factors.get(r.name)
        factor = 1.0 if factor is None else _nonneg(factor, f"local_inflow_factors[{r.name!r}]")
        flow += _nonneg(r.local_inflow_mm3, f"riparian {r.name!r} local_inflow_mm3") * ff * factor
        out[r.name] = flow
    return out


def stochastic_flow_factors(scenario: Scenario, basin: Basin) -> List[float]:
    """Annual natural-flow multipliers for a scenario, optionally stochastic.

    When ``scenario.stochastic`` is False the deterministic trend
    ``[scenario.flow_factor(i) for i in range(scenario.years)]`` is
    returned.  Otherwise each year's factor is the trend value multiplied by
    an independent lognormal variate with mean 1 and coefficient of
    variation ``basin.climate_cv`` (Stedinger 1980), so that
    ``E[factor_i] = scenario.flow_factor(i)`` and ``CV = climate_cv``.
    The draw uses ``numpy.random.default_rng(scenario.seed)`` and is
    therefore reproducible.

    Parameters
    ----------
    scenario : Scenario
        Provides ``years``, ``flow_factor(i)``, ``stochastic`` and ``seed``.
    basin : Basin
        Provides ``climate_cv`` (>= 0; 0 gives the deterministic trend).

    Returns
    -------
    list of float
        One non-negative multiplier per simulated year.

    References
    ----------
    Stedinger (1980) *Water Resour. Res.* 16(3); Loucks & van Beek (2005)
    ch. 7 (synthetic streamflow generation).
    """
    if not isinstance(scenario, Scenario):
        raise ValueError(f"scenario must be a wefnexus.models.Scenario, got {type(scenario).__name__}")
    if not isinstance(basin, Basin):
        raise ValueError(f"basin must be a wefnexus.models.Basin, got {type(basin).__name__}")
    years = int(scenario.years)
    if years < 0:
        raise ValueError(f"scenario.years must be >= 0, got {scenario.years}")
    trend = [float(scenario.flow_factor(i)) for i in range(years)]
    if any(t < 0.0 or math.isnan(t) or math.isinf(t) for t in trend):
        raise ValueError("scenario.flow_factor(i) must be finite and >= 0 for every year")
    if not scenario.stochastic or years == 0:
        return trend
    cv = _nonneg(basin.climate_cv, "basin.climate_cv")
    if cv == 0.0:
        return trend
    sigma = math.sqrt(math.log1p(cv * cv))      # lognormal with mean 1, CV = cv
    rng = np.random.default_rng(scenario.seed)
    multipliers = rng.lognormal(mean=-0.5 * sigma * sigma, sigma=sigma, size=years)
    return [t * float(m) for t, m in zip(trend, multipliers)]
