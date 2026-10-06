"""Water sharing rules, cooperative game solutions and equity metrics (leaf module).

Transboundary water allocation is, formally, a *claims problem* (also called
a *bankruptcy problem*, O'Neill 1982): an *estate* ``E`` (the water that can
be withdrawn from the river in a given year, Mm3/yr) has to be divided among
``n`` claimants (riparians) whose *claims* ``c_1, ..., c_n`` (demands or
treaty entitlements, Mm3/yr) add up to more than the estate.  The module
offers

* **division rules** for claims problems – proportional, constrained equal
  awards (CEA), constrained equal losses (CEL), the Talmud rule of Aumann &
  Maschler (1985), the adjusted proportional rule of Curiel, Maschler & Tijs
  (1987), equal split and upstream (sequential) priority, together with a
  ``RULES`` registry and :func:`apply_rule` / :func:`compare_rules`
  dispatchers;
* **cooperative game theory** – the Shapley value, core membership tests, the
  nucleolus (sequential linear programmes, Maschler, Peleg & Shapley 1979)
  and the O'Neill (1982) bankruptcy game that links the two families;
* **bargaining** – the Nash (1950) bargaining solution for a divisible
  resource;
* **equity metrics** – the Gini coefficient, claim satisfaction, an
  equal-satisfaction (proportionality) test and Foley's no-envy test.

All rules accept the claims either as a sequence (list/tuple/array) or as a
mapping ``{name: claim}`` and return awards in the *same* container type (a
``dict`` for mappings, a ``tuple`` for tuples, a ``list`` otherwise), in the
same order.  Awards always satisfy

    0 <= a_i <= c_i   and   sum(a) == min(E, sum(c))

so that an estate larger than the total claim simply honours every claim in
full.  Units are whatever the caller uses for ``estate`` and ``claims`` –
throughout *wefnexus* that is **Mm3/yr** – and the rules are homogeneous of
degree one, i.e. independent of the unit.

Inputs are never mutated and bad inputs (negative or non-finite numbers,
booleans, unknown rule names, duplicate player names, ...) raise
``ValueError`` with an explicit message.  SciPy is imported lazily and only
by :func:`nucleolus` and the n-player branch of :func:`nash_bargaining`.

References
----------
Ansink, E. & Weikard, H.-P. (2012). Sequential sharing rules for river
    sharing problems. *Social Choice and Welfare* 38, 187-210.
Aumann, R. J. & Maschler, M. (1985). Game theoretic analysis of a bankruptcy
    problem from the Talmud. *Journal of Economic Theory* 36, 195-213.
Curiel, I. J., Maschler, M. & Tijs, S. H. (1987). Bankruptcy games.
    *Zeitschrift fur Operations Research* 31, A143-A159.
Dinar, A., Ratner, A. & Yaron, D. (1992). Evaluating cooperative game theory
    in water resources. *Theory and Decision* 32, 1-20.
Foley, D. K. (1967). Resource allocation and the public sector. *Yale
    Economic Essays* 7, 45-98.
Gini, C. (1912). *Variabilita e mutabilita*. Bologna: Cuppini.
Guajardo, M. & Jornsten, K. (2015). Common mistakes in computing the
    nucleolus. *European Journal of Operational Research* 241, 931-935.
Kohlberg, E. (1971). On the nucleolus of a characteristic function game.
    *SIAM Journal on Applied Mathematics* 20, 62-66.
Madani, K. (2010). Game theory and water resources. *Journal of Hydrology*
    381, 225-238.
Maschler, M., Peleg, B. & Shapley, L. S. (1979). Geometric properties of the
    kernel, nucleolus, and related solution concepts. *Mathematics of
    Operations Research* 4, 303-338.
Mianabadi, H., Mostert, E., Zarghami, M. & van de Giesen, N. (2014). A new
    bankruptcy method for conflict resolution in water resources allocation.
    *Journal of Environmental Management* 144, 152-159.
Nash, J. F. (1950). The bargaining problem. *Econometrica* 18, 155-162.
O'Neill, B. (1982). A problem of rights arbitration from the Talmud.
    *Mathematical Social Sciences* 2, 345-371.
Schmeidler, D. (1969). The nucleolus of a characteristic function game.
    *SIAM Journal on Applied Mathematics* 17, 1163-1170.
Sen, A. (1973). *On Economic Inequality*. Oxford: Clarendon Press.
Shapley, L. S. (1953). A value for n-person games. In Kuhn, H. W. & Tucker,
    A. W. (eds), *Contributions to the Theory of Games II*, 307-317.
Shapley, L. S. (1971). Cores of convex games. *International Journal of Game
    Theory* 1, 11-26.
Thomson, W. (2003). Axiomatic and game-theoretic analysis of bankruptcy and
    taxation problems: a survey. *Mathematical Social Sciences* 45, 249-297.
Thomson, W. (2015). Axiomatic and game-theoretic analysis of bankruptcy and
    taxation problems: an update. *Mathematical Social Sciences* 74, 41-59.
Young, H. P. (1994). *Equity: In Theory and Practice*. Princeton University
    Press.
"""
from __future__ import annotations

import math
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

__all__ = [
    "RULES",
    "RULE_ALIASES",
    "proportional",
    "constrained_equal_awards",
    "constrained_equal_losses",
    "talmud",
    "adjusted_proportional",
    "equal_split",
    "upstream_priority",
    "apply_rule",
    "compare_rules",
    "minimal_rights",
    "bankruptcy_game",
    "game_from_dict",
    "shapley_value",
    "core_constraints_violations",
    "is_in_core",
    "nucleolus",
    "nash_bargaining",
    "gini",
    "satisfaction",
    "equal_satisfaction",
    "envy_free",
    "no_envy",
]

#: Claims container accepted by every rule: a sequence or a mapping.
Claims = Union[Sequence[float], Mapping[Hashable, float]]
#: Characteristic function of a transferable-utility game: coalition -> worth.
CharacteristicFunction = Callable[[FrozenSet[Any]], float]

#: Maximum number of players for the exact Shapley value (2**n coalitions).
MAX_SHAPLEY_PLAYERS = 16
#: Maximum number of players for the sequential-LP nucleolus (spec: n <= 8).
MAX_NUCLEOLUS_PLAYERS = 8


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
def _as_float(name: str, value: Any, *, allow_inf: bool = False, allow_bool: bool = False) -> float:
    """Convert ``value`` to a float or raise ``ValueError``.

    Booleans are rejected by default because ``True``/``False`` passed as a
    volume is almost always a bug; characteristic functions of *simple games*
    legitimately return booleans, so :func:`_as_game_value` allows them.
    """
    if isinstance(value, (bool, np.bool_)):
        if not allow_bool:
            raise ValueError(f"{name} must be a number, got boolean {value!r}")
        return 1.0 if value else 0.0
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}") from None
    if math.isnan(out):
        raise ValueError(f"{name} must not be NaN")
    if math.isinf(out) and not allow_inf:
        raise ValueError(f"{name} must be finite, got {out!r}")
    return out


def _non_negative(name: str, value: Any, *, allow_inf: bool = False) -> float:
    out = _as_float(name, value, allow_inf=allow_inf)
    if out < 0.0:
        raise ValueError(f"{name} must be >= 0, got {out!r}")
    return out


def _as_game_value(name: str, value: Any) -> float:
    return _as_float(name, value, allow_bool=True)


def _unpack_claims(claims: Any, label: str = "claims") -> Tuple[str, Optional[List[Hashable]], np.ndarray]:
    """Validate a claims container and return ``(kind, keys, array)``.

    ``kind`` is ``"dict"``, ``"tuple"`` or ``"list"`` and drives
    :func:`_pack` so that a rule returns the container type it was given.
    """
    if isinstance(claims, (str, bytes)):
        raise ValueError(f"{label} must be a sequence or mapping of numbers, not a string")
    if isinstance(claims, Mapping):
        keys = list(claims.keys())
        vals = [_non_negative(f"{label}[{k!r}]", claims[k]) for k in keys]
        return "dict", keys, np.asarray(vals, dtype=float)
    if isinstance(claims, np.ndarray) and claims.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional, got shape {claims.shape}")
    try:
        items = list(claims)
    except TypeError:
        raise ValueError(
            f"{label} must be a sequence or mapping of numbers, got {type(claims).__name__}"
        ) from None
    kind = "tuple" if isinstance(claims, tuple) else "list"
    vals = [_non_negative(f"{label}[{i}]", v) for i, v in enumerate(items)]
    return kind, None, np.asarray(vals, dtype=float)


def _pack(kind: str, keys: Optional[List[Hashable]], awards: np.ndarray) -> Any:
    vals = [float(a) + 0.0 for a in awards]  # ``+ 0.0`` turns -0.0 into 0.0
    if kind == "dict":
        assert keys is not None
        return dict(zip(keys, vals))
    if kind == "tuple":
        return tuple(vals)
    return vals


def _clip_awards(awards: np.ndarray, claims: np.ndarray) -> np.ndarray:
    """Remove floating-point dust so that ``0 <= a_i <= c_i`` holds exactly."""
    return np.minimum(np.maximum(awards, 0.0), claims)


# --------------------------------------------------------------------------- #
# Array kernels (claims already validated)
# --------------------------------------------------------------------------- #
def _proportional_array(estate: float, c: np.ndarray) -> np.ndarray:
    total = float(c.sum())
    if total <= 0.0:
        return np.zeros_like(c)
    return _clip_awards(c * (min(estate, total) / total), c)


def _cea_array(estate: float, c: np.ndarray) -> np.ndarray:
    """Exact CEA by sorting: fill the smallest claims first.

    Walking the claims in increasing order, the current candidate level is
    ``remaining / (number of claimants not yet served)``.  A claim at or
    below that level is honoured in full; the first claim above it, and every
    larger claim, receives the level itself.
    """
    n = len(c)
    awards = np.zeros(n, dtype=float)
    if n == 0:
        return awards
    remaining = min(estate, float(c.sum()))
    order = np.argsort(c, kind="stable")
    for k, idx in enumerate(order):
        level = remaining / (n - k)
        if c[idx] <= level:
            awards[idx] = c[idx]
            remaining -= c[idx]
        else:
            awards[order[k:]] = level
            break
    return _clip_awards(awards, c)


def _cel_array(estate: float, c: np.ndarray) -> np.ndarray:
    """Exact CEL through its duality with CEA: ``CEL(E, c) = c - CEA(C - E, c)``.

    Equal losses capped at each claim are the constrained-equal-awards
    division of the total loss ``C - min(E, C)`` (Aumann & Maschler 1985;
    Thomson 2003, Sect. 2.1).
    """
    total = float(c.sum())
    loss = max(total - min(estate, total), 0.0)
    return _clip_awards(c - _cea_array(loss, c), c)


def _talmud_array(estate: float, c: np.ndarray) -> np.ndarray:
    total = float(c.sum())
    half = c / 2.0
    if estate <= total / 2.0:
        return _cea_array(estate, half)
    return _clip_awards(half + _cel_array(estate - total / 2.0, half), c)


def _minimal_rights_array(estate: float, c: np.ndarray) -> np.ndarray:
    total = float(c.sum())
    e = min(estate, total)
    return _clip_awards(np.maximum(e - (total - c), 0.0), c)


def _adjusted_proportional_array(estate: float, c: np.ndarray) -> np.ndarray:
    total = float(c.sum())
    e = min(estate, total)
    rights = _minimal_rights_array(e, c)
    rest = max(e - float(rights.sum()), 0.0)
    truncated = np.minimum(np.maximum(c - rights, 0.0), rest)
    return _clip_awards(rights + _proportional_array(rest, truncated), c)


def _upstream_priority_array(available: float, c: np.ndarray) -> np.ndarray:
    awards = np.zeros_like(c)
    remaining = available
    for i, ci in enumerate(c):
        a = min(float(ci), remaining)
        awards[i] = a
        remaining -= a
        if remaining <= 0.0:
            break
    return _clip_awards(awards, c)


# --------------------------------------------------------------------------- #
# Division rules for claims (bankruptcy) problems
# --------------------------------------------------------------------------- #
def proportional(estate: float, claims: Claims) -> Any:
    """Proportional rule: awards in proportion to claims.

    ``a_i = c_i * min(E, C) / C`` with ``C = sum(c)``; every claimant
    obtains the same *satisfaction* ``a_i / c_i``.  It is the rule written
    into most volumetric water treaties ("x % of the flow") and the one
    Aristotle's *Nicomachean Ethics* calls distributive justice.

    Parameters
    ----------
    estate : float
        Amount to divide (e.g. withdrawable water, Mm3/yr); ``>= 0`` and may
        be ``inf`` (every claim is then honoured).
    claims : sequence or mapping of float
        Non-negative claims in the same unit.

    Returns
    -------
    list, tuple or dict
        Awards in the same container type and order as ``claims``, summing to
        ``min(estate, sum(claims))`` with ``0 <= a_i <= c_i``.

    Raises
    ------
    ValueError
        On a negative, NaN or non-numeric estate or claim.

    References
    ----------
    O'Neill (1982); Thomson (2003), Sect. 2.1.
    """
    e = _non_negative("estate", estate, allow_inf=True)
    kind, keys, c = _unpack_claims(claims)
    return _pack(kind, keys, _proportional_array(e, c))


def constrained_equal_awards(estate: float, claims: Claims) -> Any:
    """Constrained equal awards (CEA) rule: ``a_i = min(c_i, lambda)``.

    The level ``lambda`` is chosen so that the awards exhaust
    ``min(E, C)``.  Small claimants are served in full and the surplus is
    shared equally among the large ones – Maimonides' rule, the rule that
    maximises the smallest award (leximin) and, in water terms, the rule
    that protects basic needs of the smallest users first.  Computed exactly
    by sorting the claims (no numerical bisection).

    Parameters
    ----------
    estate : float
        Amount to divide (Mm3/yr); ``>= 0`` (``inf`` allowed).
    claims : sequence or mapping of float
        Non-negative claims (Mm3/yr).

    Returns
    -------
    list, tuple or dict
        Awards in the same container type as ``claims``.

    Raises
    ------
    ValueError
        On negative, NaN or non-numeric inputs.

    References
    ----------
    Aumann & Maschler (1985); Thomson (2003), Sect. 2.1.
    """
    e = _non_negative("estate", estate, allow_inf=True)
    kind, keys, c = _unpack_claims(claims)
    return _pack(kind, keys, _cea_array(e, c))


def constrained_equal_losses(estate: float, claims: Claims) -> Any:
    """Constrained equal losses (CEL) rule: ``a_i = max(c_i - lambda, 0)``.

    Every claimant bears the same *loss* ``c_i - a_i``, except that nobody
    receives a negative award; ``lambda`` is set so that awards exhaust
    ``min(E, C)``.  CEL is the dual of CEA, ``CEL(E, c) = c - CEA(C - E, c)``
    (Aumann & Maschler 1985), which is how it is computed here (exactly).
    In water-allocation terms it protects large historical users (equal cuts
    in absolute volume) and is the rule behind "shared shortage" clauses.

    Parameters
    ----------
    estate : float
        Amount to divide (Mm3/yr); ``>= 0`` (``inf`` allowed).
    claims : sequence or mapping of float
        Non-negative claims (Mm3/yr).

    Returns
    -------
    list, tuple or dict
        Awards in the same container type as ``claims``.

    Raises
    ------
    ValueError
        On negative, NaN or non-numeric inputs.

    References
    ----------
    Aumann & Maschler (1985); Thomson (2003), Sect. 2.1.
    """
    e = _non_negative("estate", estate, allow_inf=True)
    kind, keys, c = _unpack_claims(claims)
    return _pack(kind, keys, _cel_array(e, c))


def talmud(estate: float, claims: Claims) -> Any:
    """Talmud rule of Aumann & Maschler (1985).

    ``T(E, c) = CEA(E, c/2)`` when ``E <= C/2`` and
    ``T(E, c) = c/2 + CEL(E - C/2, c/2)`` otherwise, with ``C = sum(c)``.
    It reproduces the Mishnah's *contested garment* and *marriage contract*
    tables (claims 100, 200, 300: estate 100 -> 33 1/3 each; 200 -> 50, 75,
    75; 300 -> 50, 100, 150), is self-dual (``T(E, c) = c - T(C - E, c)``),
    and coincides with the nucleolus of the O'Neill bankruptcy game
    (Aumann & Maschler 1985) – see :func:`bankruptcy_game` and
    :func:`nucleolus`.  For two claimants it equals the contested-garment
    rule and :func:`adjusted_proportional`.

    Parameters
    ----------
    estate : float
        Amount to divide (Mm3/yr); ``>= 0`` (``inf`` allowed).
    claims : sequence or mapping of float
        Non-negative claims (Mm3/yr).

    Returns
    -------
    list, tuple or dict
        Awards in the same container type as ``claims``.

    Raises
    ------
    ValueError
        On negative, NaN or non-numeric inputs.

    References
    ----------
    Aumann & Maschler (1985); Thomson (2003), Sect. 2.1.
    """
    e = _non_negative("estate", estate, allow_inf=True)
    kind, keys, c = _unpack_claims(claims)
    return _pack(kind, keys, _talmud_array(e, c))


def minimal_rights(estate: float, claims: Claims) -> Any:
    """Minimal rights ``m_i = max(E - sum_{j != i} c_j, 0)`` (truncated at ``c_i``).

    The minimal right is what claimant ``i`` keeps even if every other claim
    were honoured in full first – the "undisputed" part of the estate.  It
    is the first step of :func:`adjusted_proportional` and equals the worth
    of the singleton coalitions in :func:`bankruptcy_game`.

    Parameters
    ----------
    estate : float
        Amount to divide (Mm3/yr); ``>= 0`` (``inf`` allowed, then
        ``m = c``).
    claims : sequence or mapping of float
        Non-negative claims (Mm3/yr).

    Returns
    -------
    list, tuple or dict
        Minimal rights in the same container type as ``claims``.

    References
    ----------
    Curiel, Maschler & Tijs (1987); Thomson (2003), Sect. 2.1.
    """
    e = _non_negative("estate", estate, allow_inf=True)
    kind, keys, c = _unpack_claims(claims)
    return _pack(kind, keys, _minimal_rights_array(e, c))


def adjusted_proportional(estate: float, claims: Claims) -> Any:
    """Adjusted proportional (AP) rule of Curiel, Maschler & Tijs (1987).

    1. Each claimant first receives its minimal right
       ``m_i = max(E - sum_{j != i} c_j, 0)``.
    2. The remainder ``E' = E - sum(m)`` is divided *proportionally* to the
       revised claims ``c'_i = min(c_i - m_i, E')`` (claims net of what was
       already received and truncated at what is left).

    AP equals the tau-value of the bankruptcy game and, for two claimants,
    the contested-garment / Talmud rule.  It honours minimal rights and
    claim truncation, which plain proportionality does not.

    Parameters
    ----------
    estate : float
        Amount to divide (Mm3/yr); ``>= 0`` (``inf`` allowed).
    claims : sequence or mapping of float
        Non-negative claims (Mm3/yr).

    Returns
    -------
    list, tuple or dict
        Awards in the same container type as ``claims``.

    Raises
    ------
    ValueError
        On negative, NaN or non-numeric inputs.

    References
    ----------
    Curiel, Maschler & Tijs (1987); Thomson (2003), Sect. 2.1.
    """
    e = _non_negative("estate", estate, allow_inf=True)
    kind, keys, c = _unpack_claims(claims)
    return _pack(kind, keys, _adjusted_proportional_array(e, c))


def equal_split(estate: float, claims: Claims) -> Any:
    """Equal split with redistribution: ``min(c_i, E/n)`` and reallocate the unused share.

    Giving everybody ``E/n``, returning what exceeds a claim to the pool and
    repeating until nothing is left over is exactly the constrained equal
    awards rule, so this is a documented alias of
    :func:`constrained_equal_awards` (same awards, same container type).
    It expresses the "equal and reasonable utilisation" principle of the
    1997 UN Watercourses Convention in its most literal (per-riparian) form.

    Parameters
    ----------
    estate : float
        Amount to divide (Mm3/yr); ``>= 0`` (``inf`` allowed).
    claims : sequence or mapping of float
        Non-negative claims (Mm3/yr).

    Returns
    -------
    list, tuple or dict
        Awards in the same container type as ``claims``.

    References
    ----------
    Thomson (2003), Sect. 2.1 (CEA); UN (1997) Convention on the Law of the
    Non-navigational Uses of International Watercourses, Art. 5.
    """
    return constrained_equal_awards(estate, claims)


def upstream_priority(available: float, claims: Claims) -> Any:
    """Serve claims sequentially in index (upstream -> downstream) order.

    ``a_i = min(c_i, available - sum_{j < i} a_j)``: the first claimant
    takes what it wants, the second takes from what is left, and so on.
    This is the non-cooperative "absolute territorial sovereignty" (Harmon
    doctrine) outcome of unilateral upstream development and the natural
    *disagreement point* for negotiation (Ansink & Weikard 2012).  Claims
    given as a mapping are served in insertion order.

    Parameters
    ----------
    available : float
        Water available for withdrawal at the top of the sequence
        (Mm3/yr); ``>= 0`` (``inf`` allowed).
    claims : sequence or mapping of float
        Non-negative claims ordered from upstream to downstream (Mm3/yr).

    Returns
    -------
    list, tuple or dict
        Awards in the same container type as ``claims``, summing to
        ``min(available, sum(claims))``.

    Raises
    ------
    ValueError
        On negative, NaN or non-numeric inputs.

    References
    ----------
    Ansink & Weikard (2012); Madani (2010).
    """
    e = _non_negative("available", available, allow_inf=True)
    kind, keys, c = _unpack_claims(claims)
    return _pack(kind, keys, _upstream_priority_array(e, c))


#: Registry of division rules by canonical name (``apply_rule`` / ``compare_rules``).
RULES: Dict[str, Callable[[float, Claims], Any]] = {
    "proportional": proportional,
    "cea": constrained_equal_awards,
    "cel": constrained_equal_losses,
    "talmud": talmud,
    "ap": adjusted_proportional,
    "equal": equal_split,
    "upstream_priority": upstream_priority,
}

#: Accepted aliases (case-insensitive; ``-`` and spaces are read as ``_``).
RULE_ALIASES: Dict[str, str] = {
    "p": "proportional",
    "prop": "proportional",
    "pro_rata": "proportional",
    "constrained_equal_awards": "cea",
    "maimonides": "cea",
    "constrained_equal_losses": "cel",
    "aumann_maschler": "talmud",
    "contested_garment": "talmud",
    "cg": "talmud",
    "adjusted_proportional": "ap",
    "equal_split": "equal",
    "equal_division": "equal",
    "upstream": "upstream_priority",
    "priority": "upstream_priority",
    "sequential": "upstream_priority",
    "harmon": "upstream_priority",
}


def _resolve_rule(rule: Any) -> Tuple[str, Callable[[float, Claims], Any]]:
    """Map a rule name (or callable) to ``(canonical_name, function)``."""
    if callable(rule):
        return getattr(rule, "__name__", "custom"), rule
    if not isinstance(rule, str):
        raise ValueError(f"rule must be a rule name or callable, got {type(rule).__name__}")
    key = rule.strip().lower().replace("-", "_").replace(" ", "_")
    key = RULE_ALIASES.get(key, key)
    if key not in RULES:
        raise ValueError(f"unknown allocation rule {rule!r}; expected one of {sorted(RULES)}")
    return key, RULES[key]


def apply_rule(rule: str, estate: float, claims: Claims) -> Any:
    """Dispatch a division rule by name.

    Parameters
    ----------
    rule : str or callable
        One of ``"proportional"``, ``"cea"``, ``"cel"``, ``"talmud"``,
        ``"ap"``, ``"equal"``, ``"upstream_priority"`` (case-insensitive;
        aliases in :data:`RULE_ALIASES`, e.g. ``"adjusted_proportional"``),
        or any callable ``f(estate, claims)``.
    estate : float
        Amount to divide (Mm3/yr).
    claims : sequence or mapping of float
        Non-negative claims (Mm3/yr).

    Returns
    -------
    list, tuple or dict
        Awards in the same container type as ``claims``.

    Raises
    ------
    ValueError
        On an unknown rule name or invalid estate / claims.
    """
    _, fn = _resolve_rule(rule)
    return fn(estate, claims)


def compare_rules(
    estate: float,
    claims: Claims,
    rules: Optional[Iterable[Any]] = None,
) -> Dict[str, Any]:
    """Apply several division rules to the same claims problem.

    Parameters
    ----------
    estate : float
        Amount to divide (Mm3/yr).
    claims : mapping (or sequence) of float
        Non-negative claims (Mm3/yr); a mapping keyed by riparian name is
        the intended use.
    rules : iterable of str or callable, optional
        Rules to compare; default: every rule in :data:`RULES`, in registry
        order.

    Returns
    -------
    dict
        ``{canonical rule name: awards}`` with awards in the same container
        type as ``claims`` (e.g. ``{"talmud": {"Highland": ..., ...}}``).

    Raises
    ------
    ValueError
        On an unknown rule, a duplicated rule, or invalid estate / claims.
    """
    e = _non_negative("estate", estate, allow_inf=True)
    _unpack_claims(claims)  # validate once, with a clear message
    names = list(RULES) if rules is None else list(rules)
    out: Dict[str, Any] = {}
    for rule in names:
        name, fn = _resolve_rule(rule)
        if name in out:
            raise ValueError(f"rule {name!r} listed more than once")
        out[name] = fn(e, claims)
    return out


# --------------------------------------------------------------------------- #
# Cooperative games: helpers
# --------------------------------------------------------------------------- #
def _check_players(players: Any) -> List[Hashable]:
    if isinstance(players, (str, bytes)):
        raise ValueError("players must be a sequence of names, not a single string")
    try:
        names = list(players)
    except TypeError:
        raise ValueError(f"players must be a sequence of names, got {type(players).__name__}") from None
    if not names:
        raise ValueError("players must contain at least one player")
    try:
        unique = len(set(names))
    except TypeError:
        raise ValueError("player names must be hashable") from None
    if unique != len(names):
        raise ValueError(f"player names must be unique, got {names!r}")
    return names


def _coalition(mask: int, names: Sequence[Hashable]) -> FrozenSet[Hashable]:
    return frozenset(names[i] for i in range(len(names)) if (mask >> i) & 1)


def _evaluate_game(names: Sequence[Hashable], v: Any, label: str = "v") -> np.ndarray:
    """Evaluate ``v`` on every coalition; index ``mask`` holds ``v(S)`` for the
    coalition whose bit ``i`` is set iff ``names[i]`` is a member."""
    if not callable(v):
        raise ValueError(f"{label} must be callable on frozensets of players, got {type(v).__name__}")
    n = len(names)
    out = np.empty(1 << n, dtype=float)
    for mask in range(1 << n):
        coalition = _coalition(mask, names)
        out[mask] = _as_game_value(f"{label}({sorted(map(str, coalition))})", v(coalition))
    scale = float(np.max(np.abs(out))) if out.size else 0.0
    if abs(out[0]) > 1e-12 * max(1.0, scale):
        raise ValueError(f"{label}(empty coalition) must be 0, got {out[0]!r}")
    return out


def _check_allocation(allocation: Any, names: Sequence[Hashable]) -> np.ndarray:
    if not isinstance(allocation, Mapping):
        raise ValueError(f"allocation must be a mapping of player -> payoff, got {type(allocation).__name__}")
    missing = [p for p in names if p not in allocation]
    extra = [k for k in allocation if k not in set(names)]
    if missing or extra:
        raise ValueError(
            f"allocation keys must match the players; missing {missing!r}, unexpected {extra!r}"
        )
    return np.array([_as_float(f"allocation[{p!r}]", allocation[p]) for p in names], dtype=float)


def _popcount(masks: np.ndarray) -> np.ndarray:
    """Vectorised population count of non-negative integer masks."""
    counts = np.zeros(masks.shape, dtype=np.int64)
    m = masks.astype(np.int64, copy=True)
    while np.any(m):
        counts += m & 1
        m >>= 1
    return counts


def game_from_dict(
    values: Mapping[Any, float],
    default: Optional[float] = None,
) -> CharacteristicFunction:
    """Build a characteristic function from a ``{coalition: worth}`` table.

    Keys may be frozensets, sets, tuples or lists of player names, or a
    single player name (a bare string is a one-player coalition, not a
    sequence of characters).  The empty coalition defaults to 0.

    Parameters
    ----------
    values : mapping
        Coalition worths (any unit, e.g. USD/yr of benefits).
    default : float, optional
        Worth of coalitions absent from the table.  When ``None`` (default)
        asking for an absent coalition raises ``ValueError``.

    Returns
    -------
    callable
        ``v(frozenset) -> float`` usable with :func:`shapley_value`,
        :func:`nucleolus`, :func:`is_in_core` and
        :func:`core_constraints_violations`.

    Raises
    ------
    ValueError
        On a non-numeric worth, or on an absent coalition when ``default`` is
        ``None``.
    """
    if not isinstance(values, Mapping):
        raise ValueError(f"values must be a mapping of coalition -> worth, got {type(values).__name__}")
    table: Dict[FrozenSet[Any], float] = {}
    for key, worth in values.items():
        if isinstance(key, (str, bytes)) or not isinstance(key, Iterable):
            coalition = frozenset([key])
        else:
            coalition = frozenset(key)
        table[coalition] = _as_game_value(f"values[{sorted(map(str, coalition))}]", worth)
    table.setdefault(frozenset(), 0.0)
    default_value = None if default is None else _as_float("default", default)

    def v(coalition: Iterable[Any]) -> float:
        key = frozenset([coalition]) if isinstance(coalition, (str, bytes)) else frozenset(coalition)
        if key in table:
            return table[key]
        if default_value is None:
            raise ValueError(f"coalition {sorted(map(str, key))!r} is not in the game table")
        return default_value

    return v


def bankruptcy_game(estate: float, claims: Claims) -> CharacteristicFunction:
    """O'Neill (1982) bankruptcy game of a claims problem.

    ``v(S) = max(min(E, C) - sum_{j not in S} c_j, 0)`` with ``C = sum_j c_j``:
    a coalition is worth what is left of the estate after everybody outside
    it has been paid in full.  O'Neill's game is defined for ``E <= C``; a
    larger estate (a wet year in which every demand can be met) is truncated
    to ``C`` – exactly as every division rule in this module awards
    ``min(E, C)`` – so that ``v(empty) == 0`` and ``v(N) == min(E, C)``
    always hold.  For ``E >= C`` the game is additive, ``v(S) = sum_{i in S}
    c_i``, and every solution concept returns the claims themselves.

    The game is convex, so its core is non-empty; its nucleolus is the
    :func:`talmud` award vector (Aumann & Maschler 1985), its Shapley value
    is O'Neill's random-arrival rule and its tau-value is
    :func:`adjusted_proportional` (Curiel, Maschler & Tijs 1987).

    Parameters
    ----------
    estate : float
        Amount to divide (Mm3/yr); ``>= 0`` and finite.  Values above the
        total claim are treated as the total claim.
    claims : sequence or mapping of float
        Non-negative claims.  Players are the mapping keys, or the indices
        ``0 .. n-1`` for a sequence.

    Returns
    -------
    callable
        ``v(frozenset) -> float``.  Unknown players raise ``ValueError``.

    References
    ----------
    O'Neill (1982); Aumann & Maschler (1985); Curiel, Maschler & Tijs (1987).
    """
    e = _non_negative("estate", estate)
    kind, keys, c = _unpack_claims(claims)
    players: List[Hashable] = list(keys) if kind == "dict" else list(range(len(c)))
    index = {p: i for i, p in enumerate(players)}
    total = float(c.sum())
    # O'Neill's game is defined for E <= C: above C every claim is honoured in
    # full, so the estate is truncated exactly as the division rules do.  This
    # keeps v(empty) == 0 (required by every solver) and v(N) == min(E, C).
    e = min(e, total)

    def v(coalition: Iterable[Any]) -> float:
        members = frozenset([coalition]) if isinstance(coalition, (str, bytes)) else frozenset(coalition)
        unknown = [p for p in members if p not in index]
        if unknown:
            raise ValueError(f"unknown players {unknown!r}; expected a subset of {players!r}")
        outside = total - sum(c[index[p]] for p in members)
        return max(e - outside, 0.0)

    return v


# --------------------------------------------------------------------------- #
# Cooperative games: solutions
# --------------------------------------------------------------------------- #
def shapley_value(
    players: Sequence[Hashable],
    v: CharacteristicFunction,
    *,
    max_players: int = MAX_SHAPLEY_PLAYERS,
) -> Dict[Hashable, float]:
    """Shapley (1953) value of a transferable-utility game.

    ``phi_i = sum_{S not containing i} |S|! (n - |S| - 1)! / n! * [v(S + i) - v(S)]``
    evaluated exactly over all ``2**n`` coalitions (vectorised).  It is the
    unique allocation that is efficient (``sum phi = v(N)``), symmetric,
    gives dummies ``v({i})`` and is additive across games; for convex games
    it lies in the core (Shapley 1971).  In water diplomacy it values each
    riparian's average marginal contribution to cooperative gains (Dinar,
    Ratner & Yaron 1992).

    Parameters
    ----------
    players : sequence of hashable
        Distinct player names (grand coalition ``N``).
    v : callable
        Characteristic function ``v(frozenset(players)) -> float`` with
        ``v(frozenset()) == 0`` (booleans are accepted for simple games).
    max_players : int, optional
        Guard against accidental ``2**n`` blow-ups (default 16).

    Returns
    -------
    dict
        ``{player: value}`` in the unit of ``v``.

    Raises
    ------
    ValueError
        On duplicate / empty players, ``n > max_players``, a non-callable
        ``v``, a non-numeric worth or ``v(empty) != 0``.

    References
    ----------
    Shapley (1953); Shapley (1971); Dinar, Ratner & Yaron (1992).
    """
    names = _check_players(players)
    n = len(names)
    if n > max_players:
        raise ValueError(f"shapley_value supports at most {max_players} players (got {n}); raise max_players to override")
    vals = _evaluate_game(names, v)
    masks = np.arange(1 << n, dtype=np.int64)
    sizes = _popcount(masks)
    fact = [math.factorial(k) for k in range(n + 1)]
    weights = np.array([fact[s] * fact[n - s - 1] / fact[n] for s in range(n)] + [0.0])
    out: Dict[Hashable, float] = {}
    for i in range(n):
        bit = 1 << i
        without = masks[(masks & bit) == 0]
        marginal = vals[without | bit] - vals[without]
        out[names[i]] = float(np.dot(weights[sizes[without]], marginal))
    return out


def core_constraints_violations(
    allocation: Mapping[Hashable, float],
    players: Sequence[Hashable],
    v: CharacteristicFunction,
) -> Dict[FrozenSet[Hashable], float]:
    """Excess ``e(S, x) = v(S) - x(S)`` of every non-empty coalition.

    A positive value means the coalition ``S`` could do better on its own
    than under ``x`` – the core constraint ``x(S) >= v(S)`` is violated.  The
    grand coalition is included: its excess is ``v(N) - x(N)`` (positive when
    the allocation under-distributes).

    Parameters
    ----------
    allocation : mapping
        ``{player: payoff}`` covering exactly ``players``.
    players : sequence of hashable
        Distinct player names.
    v : callable
        Characteristic function with ``v(empty) == 0``.

    Returns
    -------
    dict
        ``{frozenset(coalition): v(S) - x(S)}`` for all ``2**n - 1`` non-empty
        coalitions, in the unit of ``v``.

    Raises
    ------
    ValueError
        On mismatched allocation keys or an invalid game.

    References
    ----------
    Gillies (1959); Schmeidler (1969); Madani (2010).
    """
    names = _check_players(players)
    x = _check_allocation(allocation, names)
    vals = _evaluate_game(names, v)
    out: Dict[FrozenSet[Hashable], float] = {}
    for mask in range(1, 1 << len(names)):
        members = [i for i in range(len(names)) if (mask >> i) & 1]
        out[_coalition(mask, names)] = float(vals[mask] - x[members].sum())
    return out


def is_in_core(
    allocation: Mapping[Hashable, float],
    players: Sequence[Hashable],
    v: CharacteristicFunction,
    tol: float = 1e-9,
) -> bool:
    """Test whether an allocation is in the core of the game.

    ``x`` is in the core iff it is efficient, ``|x(N) - v(N)| <= tol``, and
    coalitionally rational, ``x(S) >= v(S) - tol`` for every coalition.
    Core allocations are stable: no group of riparians can gain by leaving
    the basin agreement (Madani 2010; Dinar, Ratner & Yaron 1992).

    Parameters
    ----------
    allocation : mapping
        ``{player: payoff}`` covering exactly ``players``.
    players : sequence of hashable
        Distinct player names.
    v : callable
        Characteristic function with ``v(empty) == 0``.
    tol : float, optional
        Absolute tolerance in the unit of ``v`` (default 1e-9).

    Returns
    -------
    bool

    Raises
    ------
    ValueError
        On mismatched allocation keys, a negative tolerance or an invalid
        game.

    References
    ----------
    Gillies (1959); Shapley (1971); Madani (2010).
    """
    tol = _non_negative("tol", tol)
    names = _check_players(players)
    x = _check_allocation(allocation, names)
    vals = _evaluate_game(names, v)
    full = (1 << len(names)) - 1
    if abs(float(x.sum()) - vals[full]) > tol:
        return False
    for mask in range(1, full + 1):
        members = [i for i in range(len(names)) if (mask >> i) & 1]
        if vals[mask] - x[members].sum() > tol:
            return False
    return True


def nucleolus(
    players: Sequence[Hashable],
    v: CharacteristicFunction,
    *,
    imputation: bool = True,
    tol: float = 1e-7,
    max_players: int = MAX_NUCLEOLUS_PLAYERS,
) -> Dict[Hashable, float]:
    """Nucleolus of a transferable-utility game (Schmeidler 1969).

    The nucleolus is the imputation that lexicographically minimises the
    non-increasingly ordered vector of coalition excesses
    ``e(S, x) = v(S) - x(S)``: it first makes the unhappiest coalition as
    happy as possible, then the next, and so on.  It always exists, is
    unique, lies in the core whenever the core is non-empty, and for the
    O'Neill bankruptcy game it equals the :func:`talmud` rule (Aumann &
    Maschler 1985).

    Algorithm (Maschler, Peleg & Shapley 1979; Kohlberg 1971), with the
    safeguards of Guajardo & Jornsten (2015):

    1. ``min eps  s.t.  v(S) - x(S) <= eps`` for every unsettled proper
       coalition, ``x(N) = v(N)``, settled coalitions held at their level,
       and ``x_i >= v({i})`` when ``imputation`` is True – solved with
       ``scipy.optimize.linprog`` (HiGHS).
    2. A coalition is *settled* at the optimal ``eps*`` only if its excess
       cannot be pushed below ``eps*`` by *any* optimal solution; this is
       checked with one auxiliary LP per candidate (maximise ``x(S)`` at
       ``eps = eps*``), which avoids the classic error of fixing every
       constraint that merely happens to be tight at one vertex.
    3. Repeat until the settled equalities pin ``x`` down (rank ``n``).

    Worths are rescaled to unit magnitude before the LPs for numerical
    robustness.  With up to ``2**n - 2`` auxiliary LPs per level the solver
    is practical for ``n <= 8`` (the default guard).

    Parameters
    ----------
    players : sequence of hashable
        Distinct player names (``1 <= n <= max_players``).
    v : callable
        Characteristic function with ``v(empty) == 0``.
    imputation : bool, optional
        Enforce individual rationality ``x_i >= v({i})`` (the nucleolus
        proper; default).  ``False`` computes the *prenucleolus*, which also
        exists for inessential games.
    tol : float, optional
        Relative tolerance (after rescaling) used to decide whether an
        excess is tight (default 1e-7).
    max_players : int, optional
        Guard on ``n`` (default 8).

    Returns
    -------
    dict
        ``{player: payoff}`` in the unit of ``v``; ``sum == v(N)``.

    Raises
    ------
    ValueError
        On invalid players / game, ``n > max_players``, or (with
        ``imputation=True``) a game that is not essential
        (``sum v({i}) > v(N)``) so that no imputation exists.
    RuntimeError
        If an LP fails to solve (should not happen for finite games).

    References
    ----------
    Schmeidler (1969); Kohlberg (1971); Maschler, Peleg & Shapley (1979);
    Aumann & Maschler (1985); Guajardo & Jornsten (2015).
    """
    from scipy.optimize import linprog

    tol = _non_negative("tol", tol)
    names = _check_players(players)
    n = len(names)
    if n > max_players:
        raise ValueError(f"nucleolus supports at most {max_players} players (got {n}); raise max_players to override")
    raw = _evaluate_game(names, v)
    full = (1 << n) - 1
    if n == 1:
        return {names[0]: float(raw[full])}

    scale = float(np.max(np.abs(raw)))
    scale = scale if scale > 0.0 else 1.0
    w = raw / scale
    v_grand = float(w[full])
    singles = np.array([w[1 << i] for i in range(n)], dtype=float)
    if imputation and singles.sum() > v_grand + tol:
        raise ValueError(
            "no imputation exists: the game is not essential "
            f"(sum of singleton worths {float(singles.sum() * scale)!r} exceeds v(N) = {float(raw[full])!r}); "
            "use imputation=False for the prenucleolus"
        )

    proper = list(range(1, full))  # non-empty proper coalitions
    rows = {m: np.array([1.0 if (m >> i) & 1 else 0.0 for i in range(n)]) for m in proper}
    efficiency = np.ones(n, dtype=float)
    x_bounds: List[Tuple[Optional[float], Optional[float]]] = [
        (float(singles[i]), None) if imputation else (None, None) for i in range(n)
    ]
    unsettled: List[int] = list(proper)
    settled: List[int] = []
    levels: List[float] = []
    objective = np.append(np.zeros(n), 1.0)
    x = np.full(n, v_grand / n)

    def solve(c: np.ndarray, eps_bounds: Tuple[Optional[float], Optional[float]]) -> Any:
        a_ub = np.array([np.append(-rows[m], -1.0) for m in unsettled])
        b_ub = np.array([-w[m] for m in unsettled])
        a_eq = np.array([np.append(efficiency, 0.0)] + [np.append(rows[m], 0.0) for m in settled])
        b_eq = np.array([v_grand] + [w[m] - lvl for m, lvl in zip(settled, levels)])
        res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=x_bounds + [eps_bounds], method="highs")
        if res.status != 0 or res.x is None:
            raise RuntimeError(f"nucleolus LP failed: {res.message}")
        return res

    for _ in range(len(proper)):
        if not unsettled:
            break
        res = solve(objective, (None, None))
        x = np.asarray(res.x[:n], dtype=float)
        eps = float(res.x[n])
        newly: List[int] = []
        remaining: List[int] = []
        for m in unsettled:
            if w[m] - float(rows[m] @ x) < eps - tol:
                remaining.append(m)  # not even tight at this optimum
                continue
            aux = solve(np.append(-rows[m], 0.0), (None, eps + tol))
            min_excess = w[m] + float(aux.fun)  # fun = -max x(S)
            if min_excess >= eps - 2.0 * tol:
                newly.append(m)
            else:
                remaining.append(m)
        if not newly:  # numerical safeguard; cannot happen in exact arithmetic
            tightest = max(unsettled, key=lambda m: w[m] - float(rows[m] @ x))
            newly = [tightest]
            remaining = [m for m in unsettled if m != tightest]
        settled.extend(newly)
        levels.extend([eps] * len(newly))
        unsettled = remaining
        system = np.array([efficiency] + [rows[m] for m in settled])
        if np.linalg.matrix_rank(system) >= n:
            break

    return {names[i]: float(x[i] * scale) + 0.0 for i in range(n)}


# --------------------------------------------------------------------------- #
# Bargaining
# --------------------------------------------------------------------------- #
def _golden_max(f: Callable[[float], float], lo: float, hi: float, iters: int = 120) -> Tuple[float, float]:
    """Golden-section maximisation of ``f`` on ``[lo, hi]``; returns ``(x, f(x))``.

    ``f`` may return ``-inf`` for infeasible points.  Exact for unimodal
    functions; otherwise it returns the best point visited.
    """
    if hi < lo:
        lo, hi = hi, lo
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    c = hi - phi * (hi - lo)
    e = lo + phi * (hi - lo)
    fc, fe = f(c), f(e)
    best_x, best_f = (c, fc) if fc >= fe else (e, fe)
    span = max(abs(lo), abs(hi), 1.0)
    for _ in range(iters):
        if hi - lo <= 1e-15 * span:
            break
        if fc < fe:
            lo, c, fc = c, e, fe
            e = lo + phi * (hi - lo)
            fe = f(e)
            if fe > best_f:
                best_x, best_f = e, fe
        else:
            hi, e, fe = e, c, fc
            c = hi - phi * (hi - lo)
            fc = f(c)
            if fc > best_f:
                best_x, best_f = c, fc
    mid = (lo + hi) / 2.0
    fm = f(mid)
    if fm > best_f:
        best_x, best_f = mid, fm
    return float(best_x), float(best_f)
def _check_utility(name: Hashable, fn: Any, x: float) -> float:
    try:
        val = fn(x)
    except Exception as exc:  # pragma: no cover - message path
        raise ValueError(f"utility function of {name!r} failed at x={x!r}: {exc}") from exc
    return _as_float(f"utility of {name!r} at x={x!r}", val)


def _nash_two_player(
    names: List[Hashable],
    fns: List[Callable[[float], float]],
    d: np.ndarray,
    total: float,
    grid: int,
    refine: bool,
) -> Dict[Hashable, float]:
    feas_tol = 1e-12 * max(1.0, float(np.max(np.abs(d))))
    xs = np.linspace(0.0, total, grid)
    g1 = np.array([_check_utility(names[0], fns[0], float(x)) for x in xs]) - d[0]
    g2 = np.array([_check_utility(names[1], fns[1], float(total - x)) for x in xs]) - d[1]
    feasible = (g1 >= -feas_tol) & (g2 >= -feas_tol)
    if not feasible.any():
        raise ValueError(
            "no feasible agreement: no split of the total gives every party at least its disagreement utility"
        )
    product = np.where(feasible, np.maximum(g1, 0.0) * np.maximum(g2, 0.0), -np.inf)
    k = int(np.argmax(product))
    best_x, best_val = float(xs[k]), float(product[k])

    if refine and total > 0.0:
        def f(x: float) -> float:
            a = _check_utility(names[0], fns[0], x) - d[0]
            b = _check_utility(names[1], fns[1], total - x) - d[1]
            if a < -feas_tol or b < -feas_tol:
                return -np.inf
            return max(a, 0.0) * max(b, 0.0)

        lo, hi = float(xs[max(k - 1, 0)]), float(xs[min(k + 1, grid - 1)])
        cand, val = _golden_max(f, lo, hi)
        if val > best_val:
            best_x, best_val = float(min(max(cand, 0.0), total)), val

    return {names[0]: best_x, names[1]: float(total - best_x)}


def _polish_pairwise(
    x: np.ndarray,
    names: List[Hashable],
    fns: List[Callable[[float], float]],
    d: np.ndarray,
    feas_tol: float,
    sweeps: int = 40,
) -> np.ndarray:
    """Block-coordinate ascent on the log Nash product along pairwise transfers.

    For every pair ``(i, j)`` the transfer ``t`` (``x_i + t``, ``x_j - t``)
    is optimised exactly by golden section while the other shares stay
    fixed; the simplex constraint is preserved by construction.  Only
    improvements are accepted, so the result is never worse than ``x``.
    """
    n = len(names)
    x = x.copy()

    def log_gain(i: int, xi: float) -> float:
        g = _check_utility(names[i], fns[i], xi) - d[i]
        if g < -feas_tol:
            return -np.inf
        return math.log(g) if g > 0.0 else -np.inf

    logs = np.array([log_gain(i, float(x[i])) for i in range(n)])
    if not np.all(np.isfinite(logs)):
        return x
    for _ in range(sweeps):
        gain_total = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                xi, xj = float(x[i]), float(x[j])

                def f(t: float, i=i, j=j, xi=xi, xj=xj) -> float:
                    return log_gain(i, xi + t) + log_gain(j, xj - t)

                current = float(logs[i] + logs[j])
                t, ft = _golden_max(f, -xi, xj)
                if ft > current + 1e-15 * max(1.0, abs(current)):
                    x[i], x[j] = xi + t, xj - t
                    logs[i], logs[j] = log_gain(i, float(x[i])), log_gain(j, float(x[j]))
                    gain_total += ft - current
        if gain_total <= 1e-13:
            break
    return x


def _nash_n_player(
    names: List[Hashable],
    fns: List[Callable[[float], float]],
    d: np.ndarray,
    total: float,
    seed: int,
) -> Dict[Hashable, float]:
    from scipy.optimize import minimize

    n = len(names)
    feas_tol = 1e-9 * max(1.0, float(np.max(np.abs(d))))

    def gains(x: np.ndarray) -> np.ndarray:
        return np.array([_check_utility(names[i], fns[i], float(x[i])) for i in range(n)]) - d

    def product(x: np.ndarray) -> float:
        g = gains(x)
        if np.any(g < -feas_tol):
            return -np.inf
        return float(np.prod(np.maximum(g, 0.0)))

    def objective(x: np.ndarray) -> float:
        g = gains(x)
        return -float(np.sum(np.log(np.maximum(g, 1e-300))))

    if total <= 0.0:
        zeros = np.zeros(n)
        if product(zeros) == -np.inf:
            raise ValueError("no feasible agreement: total is 0 and the disagreement utilities are not met")
        return {names[i]: 0.0 for i in range(n)}

    rng = np.random.default_rng(seed)
    candidates = [np.full(n, total / n)]
    candidates += list(rng.dirichlet(np.ones(n), size=256) * total)
    scored = sorted(((product(x), i) for i, x in enumerate(candidates)), key=lambda t: -t[0])
    if scored[0][0] == -np.inf:
        raise ValueError(
            "no feasible agreement found: no sampled split gives every party at least its disagreement utility"
        )
    best_x = candidates[scored[0][1]].copy()
    best_val = scored[0][0]
    starts = [candidates[i] for val, i in scored[:3] if val > 0.0]
    constraints = [
        {"type": "eq", "fun": lambda x: float(np.sum(x) - total)},
        {"type": "ineq", "fun": gains},
    ]
    for x0 in starts:
        try:
            res = minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=[(0.0, total)] * n,
                constraints=constraints,
                options={"ftol": 1e-12, "maxiter": 500},
            )
        except (ValueError, FloatingPointError):  # pragma: no cover - solver hiccup
            continue
        x = np.minimum(np.maximum(np.asarray(res.x, dtype=float), 0.0), total)
        s = float(x.sum())
        if s > 0.0:
            x = x * (total / s)
        val = product(x)
        if val > best_val:
            best_x, best_val = x, val
    if best_val > 0.0:
        polished = _polish_pairwise(best_x, names, fns, d, feas_tol)
        if product(polished) >= best_val:
            best_x = polished
    return {names[i]: float(best_x[i]) + 0.0 for i in range(n)}


def nash_bargaining(
    utility_fns: Mapping[Hashable, Callable[[float], float]],
    disagreement: Mapping[Hashable, float],
    total: float,
    grid: int = 2001,
    *,
    refine: bool = True,
    seed: int = 0,
) -> Dict[Hashable, float]:
    """Nash (1950) bargaining solution for dividing a divisible resource.

    Finds the split ``x`` of ``total`` (``x_i >= 0``, ``sum x_i = total``)
    that maximises the Nash product ``prod_i (u_i(x_i) - d_i)`` subject to
    ``u_i(x_i) >= d_i`` for every party, where ``d_i`` is party ``i``'s
    utility at the disagreement point (e.g. the unilateral
    :func:`upstream_priority` outcome).  The solution is efficient,
    symmetric, invariant to affine rescaling of utilities and independent of
    irrelevant alternatives; with linear utilities it splits the *surplus*
    over the disagreement payoffs equally.

    * Two parties: exhaustive grid search over ``x_1 in [0, total]`` on
      ``grid`` points (``x_2 = total - x_1``), followed by a golden-section
      polish inside the best grid cell when ``refine`` is True.
    * Three or more parties: ``scipy.optimize.minimize`` (SLSQP) of
      ``-sum log(u_i - d_i)`` on the simplex from the best of several seeded
      feasible starts (Dirichlet samples), keeping the best feasible result
      and polishing it by exact pairwise (golden-section) transfers.

    Because the Nash product is flat at its maximum, derivative-free
    searches resolve the optimal shares to roughly ``sqrt(machine eps)``,
    i.e. about 1e-7 of ``total`` (two parties typically reach 1e-9); that
    is far below the precision of any water-allocation input.

    Parameters
    ----------
    utility_fns : mapping
        ``{party: u(x)}`` with ``u`` a scalar function of the party's share
        of the resource (Mm3/yr -> utility).
    disagreement : mapping
        ``{party: d}`` disagreement utilities (same keys, finite).
    total : float
        Divisible amount (Mm3/yr); ``>= 0`` and finite.
    grid : int, optional
        Number of grid points for the two-party search (``>= 3``, default
        2001; odd values put the midpoint on the grid).
    refine : bool, optional
        Polish the two-party grid optimum (default True).
    seed : int, optional
        Seed of the start-point sampler for ``n >= 3`` (default 0).

    Returns
    -------
    dict
        ``{party: share of total}`` in the unit of ``total``.

    Raises
    ------
    ValueError
        On mismatched keys, non-callable utilities, non-finite utility
        values, a negative ``total``, ``grid < 3``, or when no split meets
        every disagreement utility.

    References
    ----------
    Nash (1950); Young (1994), Ch. 8; Madani (2010).
    """
    if not isinstance(utility_fns, Mapping) or not utility_fns:
        raise ValueError("utility_fns must be a non-empty mapping of party -> callable")
    if not isinstance(disagreement, Mapping):
        raise ValueError("disagreement must be a mapping of party -> utility")
    names = list(utility_fns.keys())
    if set(disagreement.keys()) != set(names):
        raise ValueError(
            f"disagreement keys {sorted(map(str, disagreement))!r} must match utility_fns keys {sorted(map(str, names))!r}"
        )
    fns = []
    for name in names:
        fn = utility_fns[name]
        if not callable(fn):
            raise ValueError(f"utility_fns[{name!r}] must be callable, got {type(fn).__name__}")
        fns.append(fn)
    d = np.array([_as_float(f"disagreement[{name!r}]", disagreement[name]) for name in names], dtype=float)
    t = _non_negative("total", total)
    if isinstance(grid, (bool, np.bool_)) or int(grid) != grid or int(grid) < 3:
        raise ValueError(f"grid must be an integer >= 3, got {grid!r}")
    grid = int(grid)

    if len(names) == 1:
        if _check_utility(names[0], fns[0], t) < d[0] - 1e-12 * max(1.0, abs(float(d[0]))):
            raise ValueError("no feasible agreement: the single party's disagreement utility is not met")
        return {names[0]: t}
    if len(names) == 2:
        return _nash_two_player(names, fns, d, t, grid, refine)
    return _nash_n_player(names, fns, d, t, seed)


# --------------------------------------------------------------------------- #
# Equity metrics
# --------------------------------------------------------------------------- #
def gini(values: Union[Sequence[float], Mapping[Hashable, float]]) -> float:
    """Gini (1912) coefficient of a non-negative distribution.

    ``G = sum_i sum_j |x_i - x_j| / (2 n^2 mean)``, computed with the
    sorted-values formula ``G = 2 sum_i i x_(i) / (n sum x) - (n + 1) / n``.
    ``G = 0`` for perfect equality and ``G -> 1 - 1/n`` when one party holds
    everything (``[0, 0, 0, 1] -> 0.75``).  Empty or all-zero inputs return
    0.  Used on per-riparian awards, satisfactions or per-capita water to
    summarise distributional equity (Sen 1973; Young 1994).

    Parameters
    ----------
    values : sequence or mapping of float
        Non-negative amounts (any unit; the index is dimensionless).

    Returns
    -------
    float
        Gini coefficient in ``[0, 1]``.

    Raises
    ------
    ValueError
        On a negative, NaN, infinite or non-numeric value.

    References
    ----------
    Gini (1912); Sen (1973).
    """
    _, _, arr = _unpack_claims(values, label="values")
    n = arr.size
    if n == 0:
        return 0.0
    total = float(arr.sum())
    if total <= 0.0:
        return 0.0
    ranks = np.arange(1, n + 1, dtype=float)
    g = 2.0 * float(np.dot(ranks, np.sort(arr))) / (n * total) - (n + 1.0) / n
    return float(min(max(g, 0.0), 1.0))


def _align_awards(
    awards: Claims, claims: Claims
) -> Tuple[str, Optional[List[Hashable]], np.ndarray, np.ndarray]:
    """Validate ``awards`` against ``claims`` and return ``(kind, keys, a, c)``.

    Both containers must be of the same family (mappings with identical key
    sets, or sequences of equal length); ``a`` is re-ordered to the claims'
    keys so that ``a[i]`` and ``c[i]`` belong to the same claimant.
    """
    ckind, ckeys, c = _unpack_claims(claims, label="claims")
    akind, akeys, a = _unpack_claims(awards, label="awards")
    if (ckind == "dict") != (akind == "dict"):
        raise ValueError("awards and claims must both be mappings or both be sequences")
    if ckind == "dict":
        assert ckeys is not None and akeys is not None
        if set(ckeys) != set(akeys):
            raise ValueError(
                f"awards keys {sorted(map(str, akeys))!r} must match claims keys {sorted(map(str, ckeys))!r}"
            )
        index = {k: i for i, k in enumerate(akeys)}
        a = np.array([a[index[k]] for k in ckeys], dtype=float)
    elif a.size != c.size:
        raise ValueError(f"awards has {a.size} entries but claims has {c.size}")
    return ckind, ckeys, a, c


def satisfaction(awards: Claims, claims: Claims) -> Any:
    """Claim satisfaction ``a_i / c_i`` (1 when the claim is 0).

    Values above 1 are possible when an award exceeds its claim (e.g. actual
    over-abstraction against a treaty entitlement) and are returned as is.

    Parameters
    ----------
    awards : sequence or mapping of float
        Non-negative awards (Mm3/yr).
    claims : sequence or mapping of float
        Non-negative claims (Mm3/yr), same keys / length as ``awards``.

    Returns
    -------
    list, tuple or dict
        Dimensionless satisfaction ratios in the container type of
        ``claims`` (keyed and ordered like ``claims`` for mappings).

    Raises
    ------
    ValueError
        On mismatched keys / lengths, mixed container kinds, or invalid
        numbers.

    References
    ----------
    Thomson (2003), Sect. 2.
    """
    ckind, ckeys, a, c = _align_awards(awards, claims)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(c > 0.0, a / np.where(c > 0.0, c, 1.0), 1.0)
    return _pack(ckind, ckeys, ratio)


def equal_satisfaction(awards: Claims, claims: Claims, tol: float = 1e-9) -> bool:
    """Equal treatment of claims: every claimant is served the same share.

    ``max_i a_i/c_i - min_i a_i/c_i <= tol`` with the satisfaction of a
    zero claim taken as 1 (see :func:`satisfaction`).  This is the
    *proportionality* principle of the claims literature – awards in
    proportion to claims, so that riparians are treated equally relative to
    their demands or entitlements (Thomson 2003, Sect. 2; Young 1994).  The
    :func:`proportional` rule satisfies it for every problem without zero
    claims (a zero claim counts as fully served, which differs from the
    common positive share), every claims rule does when all claims are
    equal, and every rule does when the estate covers every claim; CEA, CEL
    and the Talmud rule generally fail it when claims differ.

    This is a share-based criterion.  It is *not* Foley's (1967) no-envy,
    which compares bundles: a small claimant served the same *share* as a
    large one still receives a smaller *amount* and may envy it.  Use
    :func:`no_envy` for the bundle-based test.  Empty inputs are trivially
    equal.

    Parameters
    ----------
    awards, claims : sequence or mapping of float
        As in :func:`satisfaction` (Mm3/yr).
    tol : float, optional
        Absolute tolerance on the dimensionless satisfaction spread
        (default 1e-9).

    Returns
    -------
    bool

    Raises
    ------
    ValueError
        On invalid inputs or a negative tolerance.

    See Also
    --------
    envy_free : the same test under the name used by the module contract.
    no_envy : Foley's no-envy adapted to claims problems.

    References
    ----------
    Thomson (2003), Sect. 2; Young (1994).
    """
    tol = _non_negative("tol", tol)
    sat = satisfaction(awards, claims)
    vals = list(sat.values()) if isinstance(sat, Mapping) else list(sat)
    if not vals:
        return True
    return (max(vals) - min(vals)) <= tol


def envy_free(awards: Claims, claims: Claims, tol: float = 1e-9) -> bool:
    """Equal-satisfaction test, under the name of the module contract.

    Identical to :func:`equal_satisfaction`: ``max_i a_i/c_i - min_i a_i/c_i
    <= tol``, i.e. proportionality / equal treatment of claims (Thomson
    2003).  The name follows ``docs/ARCHITECTURE.md`` ("envy-free in
    satisfaction terms") and is kept for compatibility; it is **not**
    envy-freeness in Foley's (1967) sense, which compares the bundles
    claimants receive rather than the shares of their claims.  For claims
    (100, 200) the proportional awards (50, 100) pass this test although the
    first claimant would prefer the second's bundle, while the CEA awards
    (100, 100) fail it although nobody wants to swap.  Use :func:`no_envy`
    for the Foley criterion.

    Parameters
    ----------
    awards, claims : sequence or mapping of float
        As in :func:`satisfaction` (Mm3/yr).
    tol : float, optional
        Absolute tolerance on the satisfaction spread (default 1e-9).

    Returns
    -------
    bool

    Raises
    ------
    ValueError
        On invalid inputs or a negative tolerance.

    See Also
    --------
    equal_satisfaction : the same test under its proper name.
    no_envy : Foley's no-envy adapted to claims problems.

    References
    ----------
    Thomson (2003), Sect. 2; Young (1994).
    """
    return equal_satisfaction(awards, claims, tol)


def no_envy(awards: Claims, claims: Claims, tol: float = 1e-9) -> bool:
    """Foley (1967) no-envy, adapted to claims problems.

    Claimant ``i`` envies claimant ``j`` when it would rather have ``j``'s
    award, of which it can only use up to its own claim:
    ``a_i < min(a_j, c_i) - tol``.  An award vector is envy-free when nobody
    envies anybody, i.e.

        ``a_i >= min(a_j, c_i) - tol   for all i, j``,

    equivalently every claimant that is not fully served (``a_i < c_i``)
    receives at least as much as anyone else (``a_i >= max_j a_j - tol``).
    The constrained equal awards rule (and hence :func:`equal_split`)
    satisfies it for every problem, and every rule does when the estate
    covers all claims; the proportional, CEL and Talmud rules generally fail
    it when claims differ – with claims (100, 200) the proportional awards
    (50, 100) leave the first claimant envying the second's 100, which it
    could use in full.  This is the bundle-based fairness notion of Foley
    (1967) and Young (1994); :func:`equal_satisfaction` (alias
    :func:`envy_free`) is the share-based one.  Claimants with a zero claim,
    and awards above the claim (over-abstraction), never envy.  Empty and
    single-claimant inputs are trivially envy-free.

    Parameters
    ----------
    awards, claims : sequence or mapping of float
        As in :func:`satisfaction` (Mm3/yr).
    tol : float, optional
        Absolute tolerance in the unit of the awards (default 1e-9).

    Returns
    -------
    bool

    Raises
    ------
    ValueError
        On invalid inputs or a negative tolerance.

    See Also
    --------
    equal_satisfaction : equal treatment of claims (share-based).

    References
    ----------
    Foley (1967); Young (1994); Thomson (2003).
    """
    tol = _non_negative("tol", tol)
    _, _, a, c = _align_awards(awards, claims)
    if a.size == 0:
        return True
    # a_i >= min(a_j, c_i) for all j  <=>  a_i >= min(max_j a_j, c_i)
    return bool(np.all(a >= np.minimum(float(a.max()), c) - tol))
