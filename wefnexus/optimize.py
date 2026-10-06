"""Linear-programming allocation of basin water and Pareto trade-off analysis.

This *composite* module formulates the annual allocation of surface water
among the riparians and water-using sectors of a transboundary basin as a
linear programme (LP) and solves it with :func:`scipy.optimize.linprog`
(``method="highs"``).  It is the "normative" counterpart of the
priority-driven simulation in :func:`wefnexus.water.route_basin`: instead
of serving demands upstream-first in a fixed sector order, the LP looks for
the basin-wide allocation that maximises

* economic **benefit** - the value of water delivered (Harou et al. 2009
  hydro-economic modelling; Loucks & van Beek 2005, ch. 4),
* **equity** - the smallest supply ratio among the riparians (max-min or
  Rawlsian fairness; Bertsimas, Farias & Trichakis 2011), or
* a **weighted** combination of the two,

subject to the physical routing of the river, environmental-flow
requirements, optional minimum-supply guarantees and optional treaty
entitlements.  :func:`pareto_front` traces the efficiency-equity trade-off
with the epsilon-constraint method (Haimes, Lasdon & Wismer 1971; Cohon
1978), which is the classic way to show negotiators the "price" of fairness
in water diplomacy (Sadoff & Grey 2002; Wu & Whittington 2006).

Decision variables and routing
------------------------------
``w[r, s] >= 0`` is the surface-water withdrawal (Mm3/yr) of riparian ``r``
for sector ``s`` (municipal, industrial, energy, agricultural).  With the
consumption fraction ``f[r, s]`` of each sector, the river is routed
linearly from upstream to downstream::

    in_0 = (headwater + local_0) * flow_factor
    out_r = in_r - sum_s f[r, s] * w[r, s]          # return flows re-enter the river
    in_r  = out_{r-1} + local_r * flow_factor

so that ``in_r = N_r - sum_{k<r} sum_s f[k, s] w[k, s]`` where ``N_r`` is the
natural (zero-use) inflow at reach ``r`` from
:func:`wefnexus.water.natural_flows`.  Reservoirs are ignored (annual
steady state: no net storage change), as in most annual hydro-economic
LPs.

Constraints
-----------
* ``0 <= w[r, s] <= river_demand[r, s]`` - the gross demand of the sector
  net of the riparian's non-river supply (groundwater abstraction plus
  desalination, spread proportionally over sectors exactly as
  :func:`wefnexus.water.route_basin` does);
* ``sum_s w[r, s] <= in_r`` - a reach cannot withdraw more than flows in;
* ``out_r >= env_r`` when ``env_flow_hard`` (environmental flow at the
  outlet of every reach);
* ``sum_s w[r, s] + non_river_r >= min_supply_ratio * gross_demand_r``;
* ``sum_s w[r, s] <= entitlement_r`` when entitlements are given.

The **supply ratio** of a riparian is ``(sum_s w[r, s] + non_river_r) /
gross_demand_r`` (``1.0`` when it has no demand), which is identical to
:meth:`wefnexus.water.ReachResult.supply_ratio` for the same withdrawals.

Units
-----
Water in **Mm3/yr**, values in **USD per m3**, benefits in **USD/yr**
(``benefit = w * 1e6 * value``), ratios dimensionless.

References
----------
Bertsimas, D., Farias, V.F. & Trichakis, N. (2011). The price of fairness.
    *Operations Research* 59(1), 17-31.
Cohon, J.L. (1978). *Multiobjective Programming and Planning.* Academic
    Press, New York.
Draper, A.J., Jenkins, M.W., Kirby, K.W., Lund, J.R. & Howitt, R.E. (2003).
    Economic-engineering optimization for California water management.
    *Journal of Water Resources Planning and Management* 129(3), 155-164.
Haimes, Y.Y., Lasdon, L.S. & Wismer, D.A. (1971). On a bicriterion
    formulation of the problems of integrated system identification and
    system optimization. *IEEE Trans. Systems, Man, and Cybernetics* 1(3),
    296-297.
Harou, J.J., Pulido-Velazquez, M., Rosenberg, D.E., Medellin-Azuara, J.,
    Lund, J.R. & Howitt, R.E. (2009). Hydro-economic models: concepts,
    design, applications, and future prospects. *Journal of Hydrology*
    375(3-4), 627-643.
Loucks, D.P. & van Beek, E. (2005). *Water Resources Systems Planning and
    Management.* UNESCO, Paris (ch. 4, optimisation; ch. 10, river basin
    planning).
Ringler, C., von Braun, J. & Rosegrant, M.W. (2004). Water policy analysis
    for the Mekong River basin. *Water International* 29(1), 30-42.
Sadoff, C.W. & Grey, D. (2002). Beyond the river: the benefits of
    cooperation on international rivers. *Water Policy* 4(5), 389-403.
Wu, X. & Whittington, D. (2006). Incentive compatibility and conflict
    resolution in international river basins: a case study of the Nile
    Basin. *Water Resources Research* 42, W02417.
Gini, C. (1912). *Variabilita e mutabilita.* Bologna.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from wefnexus.allocation import gini as _gini
from wefnexus.models import SECTOR_PRIORITY, Basin, Sector, WaterDemand
from wefnexus.water import BasinBalance, route_basin

__all__ = [
    "OBJECTIVES",
    "M3_PER_MM3",
    "OptimizationResult",
    "optimize_allocation",
    "pareto_front",
    "route_allocation",
]

#: Objective names accepted by :func:`optimize_allocation`.
OBJECTIVES: Tuple[str, ...] = ("benefit", "equity", "weighted")

#: Cubic metres per Mm3 (benefit USD = Mm3 * M3_PER_MM3 * USD/m3).
M3_PER_MM3: float = 1.0e6

#: Absolute tolerance (Mm3, scaled by magnitude) for "environmental flow met".
_ENV_TOL: float = 1e-6

#: Slack subtracted from the max-min supply ratio in the lexicographic
#: second stage of the equity objective (absolute, supply ratios are 0..1).
_LEXI_TOL: float = 1e-7

#: Mapping of :func:`scipy.optimize.linprog` status codes to result status.
_LINPROG_STATUS: Dict[int, str] = {
    0: "optimal",
    1: "iteration_limit",
    2: "infeasible",
    3: "unbounded",
    4: "numerical_error",
}


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


def _check_names(mapping: Any, names: Sequence[str], label: str) -> Dict[str, Any]:
    """Validate an optional per-riparian mapping (unknown names -> ValueError)."""
    if mapping is None:
        return {}
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{label} must be a dict keyed by riparian name, got {type(mapping).__name__}")
    unknown = [k for k in mapping if k not in names]
    if unknown:
        raise ValueError(f"{label} refers to unknown riparian(s) {unknown}; basin riparians are {list(names)}")
    return dict(mapping)


def _withdrawal_sector(key: Any, label: str) -> Sector:
    """Parse a withdrawal :class:`Sector` from a member or its string value."""
    try:
        sector = Sector(key)
    except ValueError:
        raise ValueError(f"{label}: unknown sector {key!r}") from None
    if sector not in SECTOR_PRIORITY:
        raise ValueError(
            f"{label}: {sector.value!r} is not a withdrawal sector "
            "(environmental flow is an in-stream requirement; use env_flows)"
        )
    return sector


def _normalise_demand(demand: Any, label: str) -> Dict[Sector, float]:
    """``{sector: Mm3}`` for the four withdrawal sectors (missing = 0)."""
    if isinstance(demand, WaterDemand):
        demand = demand.withdrawals()
    if not isinstance(demand, Mapping):
        raise ValueError(f"{label} must be a dict mapping Sector -> Mm3/yr, got {type(demand).__name__}")
    out: Dict[Sector, float] = {s: 0.0 for s in SECTOR_PRIORITY}
    for key, value in demand.items():
        sector = _withdrawal_sector(key, label)
        out[sector] = _nonneg(value, f"{label}[{sector.value}]")
    return out


def _normalise_values(values: Any, names: Sequence[str]) -> Dict[str, Dict[Sector, float]]:
    """Return ``{riparian: {sector: USD/m3}}`` overrides from ``values``.

    ``values`` may be ``None``, ``{riparian: {sector: value}}`` (partial
    overrides) or a flat ``{sector: value}`` applied to every riparian.
    """
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise ValueError(f"values must be a dict, got {type(values).__name__}")
    if len(values) == 0:
        return {}
    keys = list(values.keys())
    if all(k in names for k in keys):
        out: Dict[str, Dict[Sector, float]] = {}
        for name, inner in values.items():
            if not isinstance(inner, Mapping):
                raise ValueError(f"values[{name!r}] must be a dict mapping Sector -> USD/m3")
            out[name] = {
                _withdrawal_sector(s, f"values[{name!r}]"): _nonneg(v, f"values[{name!r}][{s!r}]")
                for s, v in inner.items()
            }
        return out
    try:
        flat = {_withdrawal_sector(s, "values"): _nonneg(v, f"values[{s!r}]") for s, v in values.items()}
    except ValueError:
        raise ValueError(
            f"values keys must all be riparian names {list(names)} or all withdrawal sectors; got {keys}"
        ) from None
    return {name: dict(flat) for name in names}


# ---------------------------------------------------------------------------
# LP data
# ---------------------------------------------------------------------------
@dataclass
class _Problem:
    """Validated, array-form data of one allocation LP (internal)."""

    names: List[str]
    sectors: List[Sector]
    flow_factor: float
    headwater: float              # Mm3, after flow factor
    local: np.ndarray             # (n,) Mm3, after flow factor
    natural: np.ndarray           # (n,) cumulative natural inflow N_r
    fractions: np.ndarray         # (n, m) consumption fractions
    gross: np.ndarray             # (n, m) gross withdrawal demand
    river: np.ndarray             # (n, m) demand on the river (upper bounds)
    non_river: np.ndarray         # (n,) groundwater + desalination applied
    gross_total: np.ndarray       # (n,)
    env: np.ndarray               # (n,) environmental flow at outlet
    caps: np.ndarray              # (n,) entitlement, inf = none
    values: np.ndarray            # (n, m) USD per m3

    @property
    def n(self) -> int:
        return len(self.names)

    @property
    def m(self) -> int:
        return len(self.sectors)


def _build_problem(
    basin: Basin,
    flow_factor: float,
    values: Any,
    demands: Any,
    env_flows: Any,
    entitlements: Any,
) -> _Problem:
    """Validate inputs and assemble the LP data (never mutates ``basin``)."""
    if not isinstance(basin, Basin):
        raise ValueError(f"basin must be a wefnexus.models.Basin, got {type(basin).__name__}")
    ff = _nonneg(flow_factor, "flow_factor")
    names = basin.names()
    sectors = list(SECTOR_PRIORITY)
    n, m = len(names), len(sectors)

    demands = _check_names(demands, names, "demands")
    env_flows = _check_names(env_flows, names, "env_flows")
    if isinstance(entitlements, str):
        if entitlements.lower() != "treaty":
            raise ValueError(f"entitlements must be None, 'treaty' or a dict, got {entitlements!r}")
        entitlements = {r.name: r.treaty_allocation_mm3 for r in basin.riparians}
    entitlements = _check_names(entitlements, names, "entitlements")
    overrides = _normalise_values(values, names)

    headwater = _nonneg(basin.headwater_inflow_mm3, "basin.headwater_inflow_mm3") * ff
    local = np.zeros(n)
    fractions = np.zeros((n, m))
    gross = np.zeros((n, m))
    river = np.zeros((n, m))
    non_river = np.zeros(n)
    env = np.zeros(n)
    caps = np.full(n, math.inf)
    vals = np.zeros((n, m))

    for i, r in enumerate(basin.riparians):
        label = f"riparian {r.name!r}"
        local[i] = _nonneg(r.local_inflow_mm3, f"{label} local_inflow_mm3") * ff
        for j, s in enumerate(sectors):
            fractions[i, j] = _fraction(
                r.demand.consumption_fraction.get(s, 0.0), f"{label} consumption_fraction[{s.value}]"
            )
        raw = demands.get(r.name)
        d = _normalise_demand(
            r.demand.withdrawals() if raw is None else raw,
            f"{label} demand" if raw is None else f"demands[{r.name!r}]",
        )
        gross[i] = [d[s] for s in sectors]
        e = env_flows.get(r.name)
        env[i] = (
            _nonneg(r.demand.environmental, f"{label} demand.environmental")
            if e is None
            else _nonneg(e, f"env_flows[{r.name!r}]")
        )
        if r.name in entitlements:
            cap = entitlements[r.name]
            caps[i] = math.inf if cap is None else _nonneg(cap, f"entitlements[{r.name!r}]", allow_inf=True)
        base_values = r.demand.value_usd_per_m3
        own = overrides.get(r.name, {})
        for j, s in enumerate(sectors):
            v = own.get(s, base_values.get(s, 0.0))
            vals[i, j] = _nonneg(v, f"{label} value_usd_per_m3[{s.value}]")
        groundwater = _nonneg(r.groundwater_abstraction_mm3, f"{label} groundwater_abstraction_mm3")
        desalination = _nonneg(r.energy.desalination_capacity_mm3, f"{label} energy.desalination_capacity_mm3")
        total = float(gross[i].sum())
        non_river[i] = min(groundwater + desalination, total)
        if total > 0.0:
            river[i] = np.maximum(gross[i] * (1.0 - non_river[i] / total), 0.0)

    natural = headwater + np.cumsum(local)
    return _Problem(
        names=names,
        sectors=sectors,
        flow_factor=ff,
        headwater=headwater,
        local=local,
        natural=natural,
        fractions=fractions,
        gross=gross,
        river=river,
        non_river=non_river,
        gross_total=gross.sum(axis=1),
        env=env,
        caps=caps,
        values=vals,
    )


# ---------------------------------------------------------------------------
# LP assembly and solution
# ---------------------------------------------------------------------------
def _constraint_rows(
    p: _Problem, msr: float, env_flow_hard: bool, with_t: bool
) -> Tuple[List[np.ndarray], List[float], List[Tuple[float, float]]]:
    """Build ``A_ub x <= b_ub`` rows and variable bounds for the LP.

    Variables are ``w[r, s]`` flattened riparian-major, optionally followed
    by the epigraph variable ``t`` (min supply ratio).
    """
    n, m = p.n, p.m
    nv = n * m + (1 if with_t else 0)
    rows: List[np.ndarray] = []
    rhs: List[float] = []
    cum = np.zeros(nv)                      # consumption coefficients of reaches < r
    for i in range(n):
        own = np.zeros(nv)
        own[i * m:(i + 1) * m] = 1.0
        # availability: sum_s w[r,s] + upstream consumption <= N_r
        rows.append(own + cum)
        rhs.append(float(p.natural[i]))
        cons = np.zeros(nv)
        cons[i * m:(i + 1) * m] = p.fractions[i]
        cum = cum + cons
        # environmental flow at the outlet: cumulative consumption <= N_r - env_r
        if env_flow_hard:
            rows.append(cum.copy())
            rhs.append(float(p.natural[i] - p.env[i]))
        total = float(p.gross_total[i])
        # minimum supply guarantee
        if msr > 0.0 and total > 0.0:
            need = msr * total - float(p.non_river[i])
            if need > 0.0:
                rows.append(-own)
                rhs.append(-need)
        # epigraph: t <= (sum_s w[r,s] + non_river) / gross
        if with_t and total > 0.0:
            row = -own / total
            row[-1] = 1.0
            rows.append(row)
            rhs.append(float(p.non_river[i] / total))
        # entitlement cap on total surface withdrawal
        if math.isfinite(p.caps[i]):
            rows.append(own.copy())
            rhs.append(float(p.caps[i]))
    bounds = [(0.0, float(p.river[i, j])) for i in range(n) for j in range(m)]
    if with_t:
        bounds.append((0.0, 1.0))
    return rows, rhs, bounds


def _solve(
    p: _Problem, c: np.ndarray, msr: float, env_flow_hard: bool, with_t: bool
) -> Tuple[str, Optional[np.ndarray], float, str]:
    """Minimise ``c @ x`` subject to the allocation constraints (HiGHS)."""
    try:
        from scipy.optimize import linprog
    except ImportError as exc:  # pragma: no cover - scipy is a declared dependency
        raise ImportError("wefnexus.optimize requires SciPy (pip install scipy)") from exc
    rows, rhs, bounds = _constraint_rows(p, msr, env_flow_hard, with_t)
    a_ub = np.vstack(rows) if rows else None
    b_ub = np.asarray(rhs, dtype=float) if rows else None
    res = linprog(np.asarray(c, dtype=float), A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
    status = _LINPROG_STATUS.get(int(res.status), "error")
    if status == "optimal" and res.x is not None:
        return status, np.asarray(res.x, dtype=float), float(res.fun), str(res.message)
    return status, None, math.nan, str(res.message)


def _benefit_costs(p: _Problem, with_t: bool) -> np.ndarray:
    """Objective vector minimising ``-sum value * w`` (benefit in M USD)."""
    c = np.zeros(p.n * p.m + (1 if with_t else 0))
    c[: p.n * p.m] = -p.values.reshape(-1)
    return c


def _equity_costs(p: _Problem) -> np.ndarray:
    c = np.zeros(p.n * p.m + 1)
    c[-1] = -1.0
    return c


def _withdrawals(p: _Problem, x: np.ndarray) -> np.ndarray:
    """Clip the LP solution to its bounds and reshape to ``(n, m)``."""
    w = np.asarray(x[: p.n * p.m], dtype=float).reshape(p.n, p.m)
    return np.clip(w, 0.0, p.river)


def _diagnose(p: _Problem, msr: float, env_flow_hard: bool) -> str:
    """Human-readable reasons why the LP is infeasible."""
    notes: List[str] = []
    for i, name in enumerate(p.names):
        if env_flow_hard and p.env[i] > p.natural[i] * (1.0 + 1e-12) + 1e-9:
            notes.append(
                f"environmental flow of {name!r} ({p.env[i]:.6g} Mm3) exceeds the natural inflow at its "
                f"outlet ({p.natural[i]:.6g} Mm3) at flow_factor {p.flow_factor:g}"
            )
        if msr > 0.0 and p.gross_total[i] > 0.0:
            need = msr * float(p.gross_total[i]) - float(p.non_river[i])
            if need > p.natural[i] * (1.0 + 1e-12) + 1e-9:
                notes.append(
                    f"min_supply_ratio {msr:g} requires {need:.6g} Mm3 of surface withdrawal for {name!r} "
                    f"but at most {p.natural[i]:.6g} Mm3 of natural flow reaches it"
                )
            elif need > p.caps[i] * (1.0 + 1e-12) + 1e-9:
                notes.append(
                    f"min_supply_ratio {msr:g} requires {need:.6g} Mm3 of surface withdrawal for {name!r} "
                    f"but its entitlement is {p.caps[i]:.6g} Mm3"
                )
    if not notes:
        notes.append(
            "environmental flows, minimum-supply requirements and entitlements cannot all be satisfied "
            f"simultaneously with the natural flow available at flow_factor {p.flow_factor:g}"
        )
    return "; ".join(notes)


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------
@dataclass
class OptimizationResult:
    """Solution of one allocation LP (all water in Mm3/yr, money in USD/yr).

    The first eight fields are the contract of ``ARCHITECTURE.md`` section 8;
    the remaining fields give the full context of the solution and are
    filled even when the LP failed (allocations and outflows are then empty
    and the scalar results ``nan``).

    Attributes
    ----------
    status : str
        ``"optimal"``, ``"infeasible"``, ``"unbounded"``,
        ``"iteration_limit"``, ``"numerical_error"`` or ``"error"``.
    objective : float
        Optimal objective value: total benefit in USD for ``"benefit"``,
        the max-min supply ratio for ``"equity"``, the dimensionless score
        ``(1 - equity_weight) * benefit / max_benefit + equity_weight * t``
        for ``"weighted"``.  ``nan`` when not optimal.
    allocations : dict
        ``{riparian: {Sector: withdrawal Mm3}}`` surface withdrawals.
    outflows : dict
        ``{riparian: outflow Mm3}`` at the outlet of each reach.
    total_benefit_usd : float
        ``sum w * 1e6 * value``.
    min_supply_ratio : float
        Smallest riparian supply ratio in the solution.
    gini : float
        Gini coefficient of the riparian supply ratios (0 = equal).
    message : str
        Solver message and diagnostics.
    objective_name, flow_factor, min_supply_ratio_constraint, equity_weight, env_flow_hard
        Echo of the call.
    inflows, natural_flows : dict
        Routed and natural (zero-use) inflow per reach.
    consumption : dict
        ``{riparian: {Sector: consumptive use Mm3}}``.
    demands, river_demands : dict
        Gross demand and demand placed on the river (= upper bounds of ``w``).
    non_river_supply : dict
        Groundwater + desalination applied to demand per riparian.
    supply_ratios : dict
        ``(withdrawal + non_river) / gross demand`` per riparian (1 if none).
    benefits_usd : dict
        Benefit per riparian.
    env_flows, env_flow_met : dict
        Environmental-flow requirement and whether the outflow meets it.
    entitlements : dict
        Cap on total withdrawal per riparian (``inf`` = none).
    values_usd_per_m3 : dict
        Values used in the objective per riparian and sector.
    """

    status: str
    objective: float
    allocations: Dict[str, Dict[Sector, float]]
    outflows: Dict[str, float]
    total_benefit_usd: float
    min_supply_ratio: float
    gini: float
    message: str
    objective_name: str = "benefit"
    flow_factor: float = 1.0
    min_supply_ratio_constraint: float = 0.0
    equity_weight: float = 0.0
    env_flow_hard: bool = True
    inflows: Dict[str, float] = field(default_factory=dict)
    natural_flows: Dict[str, float] = field(default_factory=dict)
    consumption: Dict[str, Dict[Sector, float]] = field(default_factory=dict)
    demands: Dict[str, Dict[Sector, float]] = field(default_factory=dict)
    river_demands: Dict[str, Dict[Sector, float]] = field(default_factory=dict)
    non_river_supply: Dict[str, float] = field(default_factory=dict)
    supply_ratios: Dict[str, float] = field(default_factory=dict)
    benefits_usd: Dict[str, float] = field(default_factory=dict)
    env_flows: Dict[str, float] = field(default_factory=dict)
    env_flow_met: Dict[str, bool] = field(default_factory=dict)
    entitlements: Dict[str, float] = field(default_factory=dict)
    values_usd_per_m3: Dict[str, Dict[Sector, float]] = field(default_factory=dict)

    # -- convenience --------------------------------------------------------
    @property
    def success(self) -> bool:
        """``True`` when the LP was solved to optimality."""
        return self.status == "optimal"

    @property
    def outflow_to_sea_mm3(self) -> float:
        """Outflow of the most downstream reach (``nan`` if not solved)."""
        if not self.outflows:
            return math.nan
        return float(list(self.outflows.values())[-1])

    def names(self) -> List[str]:
        return list(self.demands.keys())

    def withdrawals_by_riparian(self) -> Dict[str, float]:
        """Total surface withdrawal per riparian, Mm3/yr."""
        return {name: float(sum(alloc.values())) for name, alloc in self.allocations.items()}

    def total_withdrawal_mm3(self) -> float:
        """Basin-wide surface withdrawal, Mm3/yr (``nan`` if not solved)."""
        if not self.allocations:
            return math.nan
        return float(sum(self.withdrawals_by_riparian().values()))

    def total_consumption_mm3(self) -> float:
        """Basin-wide consumptive use, Mm3/yr (``nan`` if not solved)."""
        if not self.consumption:
            return math.nan
        return float(sum(sum(c.values()) for c in self.consumption.values()))

    def mean_supply_ratio(self) -> float:
        """Unweighted mean of the riparian supply ratios (``nan`` if not solved)."""
        if not self.supply_ratios:
            return math.nan
        return float(np.mean(list(self.supply_ratios.values())))

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable dictionary (sector keys as strings, ``inf`` -> ``None``)."""

        def sectors(d: Dict[str, Dict[Sector, float]]) -> Dict[str, Dict[str, float]]:
            return {name: {Sector(s).value: float(v) for s, v in inner.items()} for name, inner in d.items()}

        return {
            "status": self.status,
            "objective_name": self.objective_name,
            "objective": self.objective,
            "message": self.message,
            "flow_factor": self.flow_factor,
            "min_supply_ratio_constraint": self.min_supply_ratio_constraint,
            "equity_weight": self.equity_weight,
            "env_flow_hard": self.env_flow_hard,
            "total_benefit_usd": self.total_benefit_usd,
            "min_supply_ratio": self.min_supply_ratio,
            "mean_supply_ratio": self.mean_supply_ratio(),
            "gini": self.gini,
            "total_withdrawal_mm3": self.total_withdrawal_mm3(),
            "total_consumption_mm3": self.total_consumption_mm3(),
            "outflow_to_sea_mm3": self.outflow_to_sea_mm3,
            "allocations": sectors(self.allocations),
            "consumption": sectors(self.consumption),
            "demands": sectors(self.demands),
            "river_demands": sectors(self.river_demands),
            "values_usd_per_m3": sectors(self.values_usd_per_m3),
            "outflows": dict(self.outflows),
            "inflows": dict(self.inflows),
            "natural_flows": dict(self.natural_flows),
            "supply_ratios": dict(self.supply_ratios),
            "benefits_usd": dict(self.benefits_usd),
            "non_river_supply": dict(self.non_river_supply),
            "env_flows": dict(self.env_flows),
            "env_flow_met": dict(self.env_flow_met),
            "entitlements": {k: (None if math.isinf(v) else v) for k, v in self.entitlements.items()},
        }

    def summary(self) -> str:
        """Plain-text table of the solution, one row per riparian."""
        head = (
            f"{self.objective_name} allocation: status={self.status}, flow_factor={self.flow_factor:g}, "
            f"benefit={self.total_benefit_usd:,.0f} USD, min supply ratio={self.min_supply_ratio:.3f}, "
            f"gini={self.gini:.3f}"
        )
        if not self.success:
            return head + "\n" + self.message
        cols = ["riparian", "inflow", "withdrawal", "demand", "supply", "outflow", "env", "met", "benefit_MUSD"]
        widths = [max(10, max(len(n) for n in self.names())), 10, 10, 10, 7, 10, 10, 4, 12]
        lines = [head, "  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(cols, widths)))]
        totals = self.withdrawals_by_riparian()
        for name in self.names():
            row = [
                name.ljust(widths[0]),
                f"{self.inflows[name]:.1f}".rjust(widths[1]),
                f"{totals[name]:.1f}".rjust(widths[2]),
                f"{sum(self.demands[name].values()):.1f}".rjust(widths[3]),
                f"{self.supply_ratios[name]:.3f}".rjust(widths[4]),
                f"{self.outflows[name]:.1f}".rjust(widths[5]),
                f"{self.env_flows[name]:.1f}".rjust(widths[6]),
                ("yes" if self.env_flow_met[name] else "no").rjust(widths[7]),
                f"{self.benefits_usd[name] / 1e6:.2f}".rjust(widths[8]),
            ]
            lines.append("  ".join(row))
        return "\n".join(lines)


def _evaluate(p: _Problem, w: np.ndarray) -> Dict[str, np.ndarray]:
    """Route the river for withdrawals ``w`` and compute the solution metrics."""
    n = p.n
    cons = p.fractions * w
    total_w = w.sum(axis=1)
    total_c = cons.sum(axis=1)
    inflow = np.zeros(n)
    outflow = np.zeros(n)
    upstream = p.headwater
    for i in range(n):
        inflow[i] = upstream + p.local[i]
        outflow[i] = inflow[i] - total_c[i]
        upstream = outflow[i]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(p.gross_total > 0.0, (total_w + p.non_river) / np.where(p.gross_total > 0.0, p.gross_total, 1.0), 1.0)
    supply_ratio = np.clip(ratio, 0.0, 1.0)
    benefit = (p.values * w).sum(axis=1) * M3_PER_MM3
    env_met = outflow >= p.env - _ENV_TOL * np.maximum(1.0, p.env)
    return {
        "consumption": cons,
        "inflow": inflow,
        "outflow": outflow,
        "supply_ratio": supply_ratio,
        "benefit": benefit,
        "env_met": env_met,
    }


def _context(p: _Problem) -> Dict[str, Any]:
    """Result fields that describe the problem (available even on failure)."""
    return {
        "flow_factor": p.flow_factor,
        "natural_flows": {name: float(p.natural[i]) for i, name in enumerate(p.names)},
        "demands": {name: {s: float(p.gross[i, j]) for j, s in enumerate(p.sectors)} for i, name in enumerate(p.names)},
        "river_demands": {name: {s: float(p.river[i, j]) for j, s in enumerate(p.sectors)} for i, name in enumerate(p.names)},
        "non_river_supply": {name: float(p.non_river[i]) for i, name in enumerate(p.names)},
        "env_flows": {name: float(p.env[i]) for i, name in enumerate(p.names)},
        "entitlements": {name: float(p.caps[i]) for i, name in enumerate(p.names)},
        "values_usd_per_m3": {name: {s: float(p.values[i, j]) for j, s in enumerate(p.sectors)} for i, name in enumerate(p.names)},
    }


def _failure(
    p: _Problem, status: str, message: str, objective_name: str, msr: float, ew: float, env_flow_hard: bool
) -> OptimizationResult:
    return OptimizationResult(
        status=status,
        objective=math.nan,
        allocations={},
        outflows={},
        total_benefit_usd=math.nan,
        min_supply_ratio=math.nan,
        gini=math.nan,
        message=message,
        objective_name=objective_name,
        min_supply_ratio_constraint=msr,
        equity_weight=ew,
        env_flow_hard=env_flow_hard,
        **_context(p),
    )


def _success(
    p: _Problem,
    w: np.ndarray,
    objective: float,
    message: str,
    objective_name: str,
    msr: float,
    ew: float,
    env_flow_hard: bool,
) -> OptimizationResult:
    ev = _evaluate(p, w)
    names, sectors = p.names, p.sectors
    ratios = [float(v) for v in ev["supply_ratio"]]
    return OptimizationResult(
        status="optimal",
        objective=float(objective),
        allocations={name: {s: float(w[i, j]) for j, s in enumerate(sectors)} for i, name in enumerate(names)},
        outflows={name: float(ev["outflow"][i]) for i, name in enumerate(names)},
        total_benefit_usd=float(ev["benefit"].sum()),
        min_supply_ratio=float(min(ratios)) if ratios else 1.0,
        gini=float(_gini(ratios)) if ratios else 0.0,
        message=message,
        objective_name=objective_name,
        min_supply_ratio_constraint=msr,
        equity_weight=ew,
        env_flow_hard=env_flow_hard,
        inflows={name: float(ev["inflow"][i]) for i, name in enumerate(names)},
        consumption={name: {s: float(ev["consumption"][i, j]) for j, s in enumerate(sectors)} for i, name in enumerate(names)},
        supply_ratios={name: ratios[i] for i, name in enumerate(names)},
        benefits_usd={name: float(ev["benefit"][i]) for i, name in enumerate(names)},
        env_flow_met={name: bool(ev["env_met"][i]) for i, name in enumerate(names)},
        **_context(p),
    )


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def optimize_allocation(
    basin: Basin,
    flow_factor: float = 1.0,
    objective: str = "benefit",
    min_supply_ratio: float = 0.0,
    equity_weight: float = 0.0,
    env_flow_hard: bool = True,
    values: Optional[Mapping[Any, Any]] = None,
    *,
    demands: Optional[Mapping[str, Mapping[Any, float]]] = None,
    env_flows: Optional[Mapping[str, float]] = None,
    entitlements: Any = None,
) -> OptimizationResult:
    """Optimal annual allocation of surface water in a basin (LP, HiGHS).

    Solves, with :func:`scipy.optimize.linprog`, the linear programme
    described in the module docstring: withdrawals ``w[r, s]`` of every
    riparian and sector are chosen to maximise the objective subject to
    linear river routing (return flows re-enter the river), environmental
    flows, minimum-supply guarantees and optional entitlements.  Reservoirs
    are ignored (annual steady state).

    Parameters
    ----------
    basin : Basin
        Basin to allocate (never mutated); riparians ordered upstream to
        downstream.
    flow_factor : float, default 1.0
        Multiplier on headwater and local inflows (>= 0).
    objective : {"benefit", "equity", "weighted"}
        ``"benefit"`` maximises ``sum w[r,s] * 1e6 * value[r,s]`` (USD/yr).
        ``"equity"`` maximises the minimum riparian supply ratio ``t``
        (epigraph variable, ``t <= supply_ratio_r`` for every riparian with
        demand); among the max-min optimal allocations the benefit-maximal
        one is then selected (lexicographic max-min fairness).
        ``"weighted"`` maximises
        ``(1 - equity_weight) * benefit / max_benefit + equity_weight * t``
        where ``max_benefit`` is the optimum of the benefit objective under
        the same constraints (0 if that optimum is 0).
    min_supply_ratio : float, default 0.0
        Every riparian with demand must reach at least this supply ratio
        (epsilon-constraint), in ``[0, 1]``.
    equity_weight : float, default 0.0
        Weight of the equity term for ``"weighted"``, in ``[0, 1]``.
    env_flow_hard : bool, default True
        Enforce ``outflow_r >= environmental flow_r`` for every reach.  When
        False the requirement is only reported (``env_flow_met``).
    values : dict, optional
        Economic value of water in USD/m3 overriding
        ``riparian.demand.value_usd_per_m3``: either
        ``{riparian: {Sector: value}}`` (partial overrides allowed) or a
        flat ``{Sector: value}`` applied to every riparian.
    demands : dict, optional
        ``{riparian: {Sector: Mm3}}`` gross withdrawal demands overriding
        ``riparian.demand.withdrawals()`` (a :class:`WaterDemand` is also
        accepted; missing sectors are zero).  Same convention as
        :func:`wefnexus.water.route_basin`.
    env_flows : dict, optional
        ``{riparian: Mm3}`` overriding ``demand.environmental``.
    entitlements : None, "treaty" or dict, optional
        Caps on each riparian's total surface withdrawal.  ``None`` (the
        default) imposes no caps; ``"treaty"`` uses
        ``riparian.treaty_allocation_mm3`` (``None`` = no cap); a dict
        ``{riparian: cap}`` gives explicit caps (``None``/``inf`` = none).

    Returns
    -------
    OptimizationResult
        ``status == "optimal"`` with allocations, outflows, benefit, minimum
        supply ratio and Gini coefficient; ``status == "infeasible"`` (or
        another solver status) with an explanatory ``message`` and empty
        allocations when the constraints cannot be met.  Infeasibility never
        raises.

    Raises
    ------
    ValueError
        On an unknown objective, ``min_supply_ratio`` / ``equity_weight``
        outside ``[0, 1]``, negative or non-finite inputs, unknown riparian
        names or sectors.

    Notes
    -----
    * Non-river supply (groundwater abstraction + desalination capacity)
      offsets each riparian's gross demand proportionally across sectors
      before the LP, exactly as in :func:`wefnexus.water.route_basin`, so
      ``w`` is bounded by the demand placed on the river and the supply
      ratio ``(sum_s w + non_river) / gross`` equals
      :meth:`wefnexus.water.ReachResult.supply_ratio` for the same
      withdrawals.
    * With default values (municipal 1.5 > industrial 0.8 > energy 0.4 >
      agricultural 0.1 USD/m3) and consumption fractions, the benefit
      optimum never serves a riparian's agriculture while its municipal
      demand is short: shifting water from agriculture to municipal use
      within a reach raises the benefit and lowers consumption.
    * The LP is solved in Mm3 and million USD for conditioning; the
      reported benefit is in USD.

    References
    ----------
    Loucks & van Beek (2005) ch. 4; Harou et al. (2009); Draper et al.
    (2003); Bertsimas, Farias & Trichakis (2011) for max-min fairness.

    Examples
    --------
    >>> from wefnexus.data import example_basin
    >>> res = optimize_allocation(example_basin(), flow_factor=0.6, objective="equity")
    >>> res.status
    'optimal'
    """
    if not isinstance(objective, str):
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
    objective_name = objective.strip().lower()
    if objective_name not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
    msr = _fraction(min_supply_ratio, "min_supply_ratio")
    ew = _fraction(equity_weight, "equity_weight")
    hard = bool(env_flow_hard)
    p = _build_problem(basin, flow_factor, values, demands, env_flows, entitlements)

    if p.n == 0:
        return OptimizationResult(
            status="optimal",
            objective=0.0 if objective_name != "equity" else 1.0,
            allocations={},
            outflows={},
            total_benefit_usd=0.0,
            min_supply_ratio=1.0,
            gini=0.0,
            message="optimal: basin has no riparians",
            objective_name=objective_name,
            min_supply_ratio_constraint=msr,
            equity_weight=ew,
            env_flow_hard=hard,
            **_context(p),
        )

    if objective_name == "benefit":
        status, x, fun, msg = _solve(p, _benefit_costs(p, False), msr, hard, False)
        if status != "optimal":
            return _failure(p, status, _fail_message(status, msg, p, msr, hard), objective_name, msr, ew, hard)
        w = _withdrawals(p, x)
        benefit = float((p.values * w).sum() * M3_PER_MM3)
        return _success(p, w, benefit, f"optimal: {msg}", objective_name, msr, ew, hard)

    if objective_name == "equity" or ew >= 1.0:
        # "weighted" with equity_weight == 1 is the pure max-min problem: use
        # the same lexicographic tie-break so the result is reproducible.
        status, w, t_star, msg = _solve_max_min(p, msr, hard)
        if status != "optimal":
            return _failure(p, status, _fail_message(status, msg, p, msr, hard), objective_name, msr, ew, hard)
        return _success(p, w, t_star, f"optimal: max-min supply ratio {t_star:.6f}; {msg}", objective_name, msr, ew, hard)

    # weighted
    status_b, x_b, fun_b, msg_b = _solve(p, _benefit_costs(p, False), msr, hard, False)
    if status_b != "optimal":
        return _failure(p, status_b, _fail_message(status_b, msg_b, p, msr, hard), objective_name, msr, ew, hard)
    max_benefit = max(-fun_b, 0.0)                       # million USD
    c = np.zeros(p.n * p.m + 1)
    if max_benefit > 0.0:
        c[: p.n * p.m] = -(1.0 - ew) * p.values.reshape(-1) / max_benefit
    c[-1] = -ew
    status, x, fun, msg = _solve(p, c, msr, hard, True)
    if status != "optimal":
        return _failure(p, status, _fail_message(status, msg, p, msr, hard), objective_name, msr, ew, hard)
    w = _withdrawals(p, x)
    t = float(min(max(x[-1], 0.0), 1.0))
    benefit_m = float((p.values * w).sum())
    score = (1.0 - ew) * (benefit_m / max_benefit if max_benefit > 0.0 else 0.0) + ew * t
    return _success(
        p, w, score,
        f"optimal: weighted score {score:.6f} (equity_weight {ew:g}, max benefit {max_benefit * M3_PER_MM3:,.0f} USD)",
        objective_name, msr, ew, hard,
    )


def _solve_max_min(p: _Problem, msr: float, hard: bool) -> Tuple[str, Optional[np.ndarray], float, str]:
    """Lexicographic max-min: maximise ``t``, then benefit among the optima.

    Returns ``(status, withdrawals, t_star, message)``.
    """
    status, x, fun, msg = _solve(p, _equity_costs(p), msr, hard, True)
    if status != "optimal" or x is None:
        return status, None, math.nan, msg
    t_star = min(max(-fun, 0.0), 1.0)
    status2, x2, _, msg2 = _solve(p, _benefit_costs(p, False), max(msr, t_star - _LEXI_TOL), hard, False)
    if status2 == "optimal" and x2 is not None:
        return "optimal", _withdrawals(p, x2), t_star, "benefit-maximal allocation among max-min optima"
    # the stage-1 point is itself feasible for stage 2, so this is only a numerical fallback
    return "optimal", _withdrawals(p, x), t_star, f"max-min allocation (second stage not solved: {msg2})"


def _fail_message(status: str, solver_message: str, p: _Problem, msr: float, hard: bool) -> str:
    if status == "infeasible":
        return f"infeasible: {_diagnose(p, msr, hard)} (solver: {solver_message})"
    return f"{status}: {solver_message}"


def pareto_front(
    basin: Basin,
    flow_factor: float = 1.0,
    points: int = 11,
    *,
    start: float = 0.0,
    env_flow_hard: bool = True,
    values: Optional[Mapping[Any, Any]] = None,
    demands: Optional[Mapping[str, Mapping[Any, float]]] = None,
    env_flows: Optional[Mapping[str, float]] = None,
    entitlements: Any = None,
) -> List[Dict[str, float]]:
    """Benefit-equity Pareto front by the epsilon-constraint method.

    The minimum supply ratio is swept from ``start`` (default 0) to its
    feasible maximum (the optimum of the ``"equity"`` objective) in
    ``points`` equally spaced levels; at each level the ``"benefit"``
    objective is maximised subject to ``min_supply_ratio >= level``
    (Haimes et al. 1971; Cohon 1978).  Benefit is therefore non-increasing
    along the front and the last point is the lexicographic max-min
    allocation.  Levels below the minimum supply ratio of the unconstrained
    benefit optimum are not binding and repeat that optimum; pass
    ``start=<that ratio>`` to sweep only the binding range.

    Parameters
    ----------
    basin : Basin
        Basin to analyse (never mutated).
    flow_factor : float, default 1.0
        Multiplier on natural inflows (>= 0).
    points : int, default 11
        Number of epsilon levels (>= 2).
    start : float, default 0.0
        First epsilon level, in ``[0, 1]``; clipped to the feasible maximum.
    env_flow_hard, values, demands, env_flows, entitlements
        Passed to :func:`optimize_allocation`.

    Returns
    -------
    list of dict
        One dict per level with keys ``"min_supply_ratio"`` (the epsilon
        level), ``"total_benefit_usd"``, ``"gini"`` (of the supply ratios),
        ``"achieved_min_supply_ratio"``, ``"mean_supply_ratio"``,
        ``"total_withdrawal_mm3"`` and ``"outflow_to_sea_mm3"``.  An empty
        list is returned when the problem is infeasible even without a
        minimum-supply constraint (e.g. environmental flows exceed the
        natural flow); infeasibility never raises.

    Raises
    ------
    ValueError
        If ``points < 2`` or on invalid basin inputs.

    References
    ----------
    Haimes, Lasdon & Wismer (1971); Cohon (1978); Bertsimas et al. (2011)
    "price of fairness".
    """
    if isinstance(points, bool) or not isinstance(points, (int, np.integer)):
        raise ValueError(f"points must be an integer >= 2, got {points!r}")
    if points < 2:
        raise ValueError(f"points must be >= 2, got {points}")
    start = _fraction(start, "start")
    common: Dict[str, Any] = dict(
        env_flow_hard=env_flow_hard, values=values, demands=demands, env_flows=env_flows, entitlements=entitlements
    )
    equity = optimize_allocation(basin, flow_factor, "equity", 0.0, **common)
    if not equity.success:
        return []
    t_max = float(equity.min_supply_ratio)
    levels = np.linspace(min(start, t_max), t_max, int(points))
    front: List[Dict[str, float]] = []
    for k, eps in enumerate(levels):
        eps = float(eps)
        if k == len(levels) - 1:
            res = equity
        else:
            res = optimize_allocation(basin, flow_factor, "benefit", eps, **common)
            if not res.success:                      # numerical edge: relax slightly
                res = optimize_allocation(basin, flow_factor, "benefit", max(eps - _LEXI_TOL, 0.0), **common)
                if not res.success:
                    continue
        front.append(
            {
                "min_supply_ratio": eps,
                "total_benefit_usd": float(res.total_benefit_usd),
                "gini": float(res.gini),
                "achieved_min_supply_ratio": float(res.min_supply_ratio),
                "mean_supply_ratio": float(res.mean_supply_ratio()),
                "total_withdrawal_mm3": float(res.total_withdrawal_mm3()),
                "outflow_to_sea_mm3": float(res.outflow_to_sea_mm3),
            }
        )
    return front


def route_allocation(basin: Basin, result: OptimizationResult) -> BasinBalance:
    """Replay an LP allocation through :func:`wefnexus.water.route_basin`.

    The basin is routed with the LP's surface withdrawals imposed as the
    demand on the river, no reservoir operation (start storages 0,
    ``reservoir_refill_fraction=0``) and the same entitlements, flow factor
    and environmental flows the LP used, so that the resulting
    :class:`~wefnexus.water.BasinBalance` reproduces the LP's withdrawals,
    inflows and outflows and provides the simulation-side quantities
    (consumption, return flows, mass balance, ``env_flow_met``).

    Parameters
    ----------
    basin : Basin
        The basin the LP was solved for (never mutated).
    result : OptimizationResult
        A successful result of :func:`optimize_allocation`.

    Returns
    -------
    BasinBalance
        Routed balance.  ``reach.demands`` hold the LP withdrawals grossed
        up by the riparian's non-river supply (so that the demand placed on
        the river equals the LP withdrawal and ``deficits`` are zero by
        construction); ``reach.withdrawals`` equal ``result.allocations``
        and ``reach.outflow_mm3`` equals ``result.outflows`` up to solver
        tolerance.  When the LP was solved with ``env_flow_hard=False`` the
        environmental flows are passed as zero so the routing does not clip
        withdrawals the LP allowed.

    Raises
    ------
    ValueError
        If ``result`` is not optimal or does not match the basin's
        riparians.
    """
    if not isinstance(basin, Basin):
        raise ValueError(f"basin must be a wefnexus.models.Basin, got {type(basin).__name__}")
    if not isinstance(result, OptimizationResult):
        raise ValueError(f"result must be an OptimizationResult, got {type(result).__name__}")
    if not result.success:
        raise ValueError(f"cannot route a non-optimal result (status {result.status!r}): {result.message}")
    names = basin.names()
    if list(result.allocations.keys()) != names:
        raise ValueError(
            f"result riparians {list(result.allocations.keys())} do not match basin riparians {names}"
        )
    demands: Dict[str, Dict[Sector, float]] = {}
    for name in names:
        alloc = result.allocations[name]
        gross = result.demands.get(name, {})
        non_river = float(result.non_river_supply.get(name, 0.0))
        total_w = float(sum(alloc.values()))
        total_g = float(sum(gross.values()))
        d: Dict[Sector, float] = {}
        for s in SECTOR_PRIORITY:
            w = float(alloc.get(s, 0.0))
            if total_w > 0.0:
                share = w / total_w
            elif total_g > 0.0:
                share = float(gross.get(s, 0.0)) / total_g
            else:
                share = 0.0
            d[s] = w + non_river * share
        demands[name] = d
    caps = {name: (None if math.isinf(v) else v) for name, v in result.entitlements.items()}
    env = dict(result.env_flows) if result.env_flow_hard else {name: 0.0 for name in names}
    return route_basin(
        basin,
        flow_factor=result.flow_factor,
        entitlements={name: caps.get(name) for name in names},
        demands=demands,
        storages={name: 0.0 for name in names},
        env_flows=env,
        reservoir_refill_fraction=0.0,
    )
