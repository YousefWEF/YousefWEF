"""Hydro-politics of transboundary basins (composite module).

This module turns the physical water balance of :mod:`wefnexus.water` and
the sharing rules of :mod:`wefnexus.allocation` into *water-diplomacy*
indicators:

* **hydro-hegemony** - the four pillars of riparian power (geographic
  position, material, bargaining and ideational power) of Zeitoun & Warner
  (2006) and Cascao & Zeitoun (2010), and the resulting power asymmetry;
* **event-based conflict / cooperation** - the Basins at Risk (BAR) water
  event intensity scale of Wolf, Yoffe & Giordano (2003), the net
  cooperation index (mean scale of every event) and the TWINS
  (Transboundary Waters Interaction NexuS) matrix of Mirumachi & Allan
  (2007), in which conflict and cooperation coexist and are therefore
  scored on two independent axes - :func:`cooperation_intensity` from the
  cooperative events alone and :func:`conflict_intensity` from the
  conflictive events alone;
* **treaties** - a :class:`Treaty` container, its *institutional resilience*
  after De Stefano et al. (2012), and compliance / delivery checks against a
  routed :class:`~wefnexus.water.BasinBalance`;
* **risk** - a Basins-at-Risk style composite :func:`conflict_risk_index`
  built from water stress, downstream dependency, power asymmetry, treaty
  weakness, recent conflict events, hydrological variability, unmet
  environmental flows and upstream dam filling;
* **benefits** - the Sadoff & Grey (2002) typology of benefits *to*, *from*,
  *because of* and *beyond* the river;
* **negotiation** - :func:`negotiate` frames the basin as a claims problem
  whose estate (:func:`bankruptcy_estate`) is the natural flow less the
  in-stream flow that must still reach the sea, applies a sharing rule,
  compares the proposal with every riparian's BATNA (the unilateral,
  upstream-priority outcome) and reports whether a zone of possible
  agreement (ZOPA) exists; :func:`compare_allocation_rules` does so for
  every rule.

Units
-----
Water volumes are **Mm3/yr**; money is **USD/yr**; energy is **GWh/yr**;
all indices are dimensionless.  BAR event scales are integers in ``-7..7``.

Nothing here mutates the :class:`~wefnexus.models.Basin`,
:class:`~wefnexus.models.Riparian` or :class:`~wefnexus.water.BasinBalance`
objects it receives; invalid inputs raise ``ValueError``.

References
----------
Ansink, E. & Weikard, H.-P. (2012). Sequential sharing rules for river
    sharing problems. *Social Choice and Welfare* 38, 187-210.
Azar, E. E. (1980). The Conflict and Peace Data Bank (COPDAB) project.
    *Journal of Conflict Resolution* 24(1), 143-152.
Cascao, A. E. & Zeitoun, M. (2010). Power, hegemony and critical
    hydropolitics. In Earle, A., Jagerskog, A. & Ojendal, J. (eds),
    *Transboundary Water Management: Principles and Practice*, Earthscan,
    27-42.
De Stefano, L., Duncan, J., Dinar, S., Stahl, K., Strzepek, K. M. & Wolf,
    A. T. (2012). Climate change and the institutional resilience of
    international river basins. *Journal of Peace Research* 49(1), 193-209.
Dinar, S., Katz, D., De Stefano, L. & Blankespoor, B. (2015). Climate change,
    conflict, and cooperation: global analysis of the effectiveness of
    international river treaties in addressing water variability.
    *Political Geography* 45, 55-66.
Drieschova, A., Giordano, M. & Fischhendler, I. (2008). Governance mechanisms
    to address flow variability in water treaties. *Global Environmental
    Change* 18(2), 285-295.
FAO (2018). *Progress on Level of Water Stress - Global baseline for SDG
    indicator 6.4.2.* FAO/UN-Water, Rome.
Fisher, R. & Ury, W. (1981). *Getting to Yes: Negotiating Agreement Without
    Giving In.* Houghton Mifflin (BATNA, zone of possible agreement).
Madani, K. (2010). Game theory and water resources. *Journal of Hydrology*
    381, 225-238.
Mirumachi, N. & Allan, J. A. (2007). Revisiting transboundary water
    governance: power, conflict, cooperation and the political economy.
    *Proceedings of the International Conference on Adaptive and Integrated
    Water Management (CAIWA)*, Basel.
Mirumachi, N. (2015). *Transboundary Water Politics in the Developing
    World.* Routledge, London.
Sadoff, C. W. & Grey, D. (2002). Beyond the river: the benefits of
    cooperation on international rivers. *Water Policy* 4(5), 389-403.
Sadoff, C. W. & Grey, D. (2005). Cooperation on international rivers: a
    continuum for securing and sharing benefits. *Water International*
    30(4), 420-427.
Warner, J. (2004). Plugging the GAP - working with Buzan: the Ilisu Dam as a
    security issue. *SOAS Water Issues Study Group Occasional Paper* 67.
Wolf, A. T., Yoffe, S. B. & Giordano, M. (2003). International waters:
    identifying basins at risk. *Water Policy* 5(1), 29-60.
Yoffe, S., Wolf, A. T. & Giordano, M. (2003). Conflict and cooperation over
    international freshwater resources: indicators of basins at risk.
    *Journal of the American Water Resources Association* 39(5), 1109-1126.
Zeitoun, M. & Warner, J. (2006). Hydro-hegemony - a framework for analysis
    of trans-boundary water conflicts. *Water Policy* 8(5), 435-460.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from wefnexus.allocation import compare_rules, gini, satisfaction
from wefnexus.energy import hydropower_gwh
from wefnexus.models import SECTOR_PRIORITY, Basin, Riparian
from wefnexus.water import (
    BasinBalance,
    dependency_ratio,
    natural_flows,
    route_basin,
    sdg_642_water_stress,
)

__all__ = [
    "BAR_SCALE",
    "HEGEMONY_PILLARS",
    "TWINS_COOPERATION_THRESHOLDS",
    "TWINS_CONFLICT_THRESHOLDS",
    "TREATY_RESILIENCE_WEIGHTS",
    "TREATY_PRESENCE_CREDIT",
    "CONFLICT_RISK_WEIGHTS",
    "CONFLICT_RISK_CATEGORIES",
    "VARIABILITY_CV_REFERENCE",
    "DEFAULT_HYDROPOWER_VALUE_USD_PER_KWH",
    "Treaty",
    "BarEvent",
    "bar_label",
    "hydro_hegemony",
    "power_asymmetry",
    "cooperation_index",
    "cooperation_intensity",
    "conflict_intensity",
    "bar_event_summary",
    "twins_cooperation_class",
    "twins_conflict_class",
    "twins_classification",
    "treaty_from_basin",
    "treaty_resilience",
    "treaty_compliance",
    "compliance_summary",
    "water_dependency",
    "conflict_risk_index",
    "risk_category",
    "benefit_sharing_matrix",
    "bankruptcy_estate",
    "negotiate",
    "compare_allocation_rules",
]

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
#: Basins at Risk (BAR) water event intensity scale (Yoffe, Wolf & Giordano
#: 2003, Table 1; adapted from Azar's COPDAB).  ``-7`` is the most
#: conflictive event, ``+7`` the most cooperative, ``0`` neutral.
BAR_SCALE: Dict[int, str] = {
    -7: "Formal declaration of war",
    -6: "Extensive war acts causing deaths, dislocation or high strategic cost",
    -5: "Small scale military acts",
    -4: "Political-military hostile actions",
    -3: "Diplomatic-economic hostile actions",
    -2: "Strong verbal expressions displaying hostility in interaction",
    -1: "Mild verbal expressions displaying discord in interaction",
    0: "Neutral or non-significant acts for the inter-nation situation",
    1: "Minor official exchanges, talks or policy expressions - mild verbal support",
    2: "Official verbal support of goals, values or regime",
    3: "Cultural or scientific agreement or support (non-strategic)",
    4: "Non-military economic, technological or industrial agreement",
    5: "Military, economic or strategic support",
    6: "International freshwater treaty; major strategic alliance (regional or international)",
    7: "Voluntary unification into one nation",
}

BAR_MIN: int = -7
BAR_MAX: int = 7

#: The four pillars of hydro-hegemony (Cascao & Zeitoun 2010).
HEGEMONY_PILLARS: Tuple[str, ...] = ("geographic", "material", "bargaining", "ideational")

#: Lower bound of the cooperation intensity (mean BAR scale of the
#: *cooperative* events only, ``0..7``, see :func:`cooperation_intensity`)
#: for each TWINS cooperation intensity class (Mirumachi & Allan 2007),
#: mapped onto the positive BAR levels of Yoffe, Wolf & Giordano (2003).
#: ``0.0`` means "> 0": any cooperative interaction is at least a
#: *confrontation of the issue*; without a cooperative event the class is
#: ``"none"``.
TWINS_COOPERATION_THRESHOLDS: Dict[str, float] = {
    "confrontation": 0.0,   # > 0: minor official exchanges, talks (BAR +1) put the issue on the table
    "ad_hoc": 2.0,          # official support of goals or regime (BAR +2): one-off joint positions
    "technical": 3.0,       # cultural, scientific or technical agreements (BAR +3)
    "risk_averting": 4.0,   # economic, technological or strategic agreements and support (BAR +4, +5)
    "risk_taking": 6.0,     # freshwater treaty, alliance, integration (BAR +6, +7)
}

#: Lower bound of the conflict intensity (mean ``|scale|`` of negative BAR
#: events, ``0..7``) for each TWINS conflict intensity class (Warner 2004;
#: Mirumachi & Allan 2007).  ``0`` is "none".
TWINS_CONFLICT_THRESHOLDS: Dict[str, float] = {
    "non-politicised": 0.0,  # > 0: mild verbal discord
    "politicised": 2.0,      # strong verbal hostility, diplomatic-economic sanctions
    "securitised": 4.0,      # political-military hostile actions
    "violent": 5.0,          # military acts, war
}

#: Default weights of the treaty mechanisms that make an agreement resilient
#: to hydrological change (De Stefano et al. 2012; Drieschova et al. 2008;
#: benefit sharing after Sadoff & Grey 2002).  They sum to 1.
TREATY_RESILIENCE_WEIGHTS: Dict[str, float] = {
    "variable_allocation": 0.20,
    "drought_provisions": 0.20,
    "data_sharing": 0.15,
    "joint_institution": 0.15,
    "dispute_resolution": 0.15,
    "benefit_sharing": 0.10,
    "review_period": 0.05,
}

#: Share of the institutional risk removed by the mere existence of a treaty
#: covering a riparian (Wolf et al. 2003 found treaties, even weak ones, to
#: be the single strongest predictor of cooperative relations); the rest is
#: removed in proportion to :func:`treaty_resilience`.
TREATY_PRESENCE_CREDIT: float = 0.5

#: Default weights of the :func:`conflict_risk_index` components (sum to 1).
CONFLICT_RISK_WEIGHTS: Dict[str, float] = {
    "water_stress": 0.20,
    "dependency": 0.15,
    "power_asymmetry": 0.10,
    "institutional": 0.20,
    "conflict_history": 0.10,
    "variability": 0.10,
    "environmental": 0.05,
    "dam_filling": 0.10,
}

#: Upper bounds (exclusive) of the risk categories; above the last one the
#: category is ``"very_high"``.
CONFLICT_RISK_CATEGORIES: Dict[str, float] = {
    "low": 0.25,
    "moderate": 0.50,
    "high": 0.75,
}

#: Coefficient of variation of annual flow at which the variability
#: component saturates at 1 (arid, highly variable rivers have CV >= 0.5;
#: humid temperate rivers 0.1-0.2).
VARIABILITY_CV_REFERENCE: float = 0.5

#: Illustrative wholesale value of hydropower, USD per kWh.
DEFAULT_HYDROPOWER_VALUE_USD_PER_KWH: float = 0.05

_M3_PER_MM3 = 1.0e6
_KWH_PER_GWH = 1.0e6
_ENV_TOL = 1e-9


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------
def _as_float(value: Any, label: str, *, allow_inf: bool = False) -> float:
    if isinstance(value, (bool, str, bytes)):
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


def _as_int(value: Any, label: str) -> int:
    if isinstance(value, (bool, str, bytes)):
        raise ValueError(f"{label} must be an integer, got {value!r}")
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be an integer, got {value!r}") from None
    if math.isnan(f) or math.isinf(f) or f != int(f):
        raise ValueError(f"{label} must be an integer, got {value!r}")
    return int(f)


def _as_bool(value: Any, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(f"{label} must be True or False, got {value!r}")


def _check_basin(basin: Any) -> Basin:
    if not isinstance(basin, Basin):
        raise ValueError(f"basin must be a wefnexus.models.Basin, got {type(basin).__name__}")
    return basin


def _check_balance(balance: Any, basin: Optional[Basin] = None, label: str = "balance") -> BasinBalance:
    if not isinstance(balance, BasinBalance):
        raise ValueError(f"{label} must be a wefnexus.water.BasinBalance, got {type(balance).__name__}")
    if basin is not None and balance.names() != basin.names():
        raise ValueError(
            f"{label} reaches {balance.names()} do not match basin riparians {basin.names()} "
            "(route the same basin with wefnexus.water.route_basin)"
        )
    return balance


def _check_treaty(treaty: Any) -> "Treaty":
    if not isinstance(treaty, Treaty):
        raise ValueError(f"treaty must be a wefnexus.diplomacy.Treaty, got {type(treaty).__name__}")
    treaty._validated_fields()  # re-validate: fields may have been edited since construction
    return treaty


def _check_weights(
    weights: Optional[Mapping[str, float]],
    defaults: Mapping[str, float],
    label: str = "weights",
) -> Dict[str, float]:
    """Merge user weights into ``defaults``; unknown keys, negative or
    non-finite values and an all-zero result raise ``ValueError``."""
    out = {k: float(v) for k, v in defaults.items()}
    if weights is None:
        return out
    if not isinstance(weights, Mapping):
        raise ValueError(f"{label} must be a mapping of component -> weight, got {type(weights).__name__}")
    unknown = [k for k in weights if k not in out]
    if unknown:
        raise ValueError(f"{label} has unknown keys {unknown!r}; expected a subset of {list(out)!r}")
    for k, v in weights.items():
        out[k] = _nonneg(v, f"{label}[{k!r}]")
    if sum(out.values()) <= 0.0:
        raise ValueError(f"{label} must not all be zero")
    return out


def _per_riparian(
    mapping: Optional[Mapping[str, Any]], names: Sequence[str], label: str
) -> Dict[str, Any]:
    """Validate an optional ``{riparian: value}`` mapping (unknown names raise)."""
    if mapping is None:
        return {}
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{label} must be a dict keyed by riparian name, got {type(mapping).__name__}")
    unknown = [k for k in mapping if k not in names]
    if unknown:
        raise ValueError(f"{label} refers to unknown riparian(s) {unknown!r}; basin riparians are {list(names)!r}")
    return dict(mapping)


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return float(sum(vals) / len(vals)) if vals else 0.0


def _clip01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


# ---------------------------------------------------------------------------
# data containers
# ---------------------------------------------------------------------------
@dataclass
class Treaty:
    """A transboundary water agreement.

    Attributes
    ----------
    name : str
        Name of the agreement.
    parties : list of str
        Riparian names that signed the treaty (unique, non-empty).
    allocations_mm3 : dict
        Fixed volumetric entitlements to withdraw surface water, Mm3/yr,
        keyed by party.  Parties without an entry have no volumetric
        entitlement (e.g. an upstream hydropower state).
    variable_allocation : bool
        Entitlements are expressed as shares of actual flow rather than fixed
        volumes (De Stefano et al. 2012 "flexible allocation").  When
        ``reference_flow_mm3`` is also given, :meth:`entitlements` scales
        the fixed volumes by ``natural_flow / reference_flow``.
    drought_provisions : bool
        Explicit rules for sharing shortage in dry years (variability
        management, Drieschova et al. 2008).
    data_sharing : bool
        Regular exchange of hydrological data.
    joint_institution : bool
        A joint commission / river basin organisation.
    dispute_resolution : bool
        A conflict-resolution mechanism (negotiation, mediation,
        arbitration, court).
    benefit_sharing : bool
        Benefits (hydropower, flood control, trade) rather than only water
        volumes are shared (Sadoff & Grey 2002).
    review_period_years : int, optional
        Period after which the treaty is reviewed / amended (> 0).
    year_signed : int, optional
        Year of signature.
    reference_flow_mm3 : float, optional
        Mean annual natural flow the fixed allocations refer to (> 0); used
        only when ``variable_allocation`` is True.

    Raises
    ------
    ValueError
        On construction with invalid fields (empty or duplicate parties,
        allocations for non-parties, negative volumes, non-boolean flags,
        non-positive review period or reference flow).

    References
    ----------
    De Stefano et al. (2012); Drieschova et al. (2008); Sadoff & Grey (2002).
    """

    name: str
    parties: List[str]
    allocations_mm3: Dict[str, float] = field(default_factory=dict)
    variable_allocation: bool = False
    drought_provisions: bool = False
    data_sharing: bool = False
    joint_institution: bool = False
    dispute_resolution: bool = False
    benefit_sharing: bool = False
    review_period_years: Optional[int] = None
    year_signed: Optional[int] = None
    reference_flow_mm3: Optional[float] = None

    #: Names of the boolean resilience features in the order of
    #: :data:`TREATY_RESILIENCE_WEIGHTS` (``review_period`` is derived).
    FEATURE_FLAGS = (
        "variable_allocation",
        "drought_provisions",
        "data_sharing",
        "joint_institution",
        "dispute_resolution",
        "benefit_sharing",
    )

    def __post_init__(self) -> None:
        for name, value in self._validated_fields().items():
            setattr(self, name, value)

    def _validated_fields(self) -> Dict[str, Any]:
        """Validate every field and return its normalised value (no mutation)."""
        if not isinstance(self.name, str):
            raise ValueError(f"Treaty.name must be a string, got {type(self.name).__name__}")
        if isinstance(self.parties, (str, bytes)) or not isinstance(self.parties, Iterable):
            raise ValueError("Treaty.parties must be a list of riparian names")
        parties = list(self.parties)
        if not parties:
            raise ValueError("Treaty.parties must contain at least one party")
        if any(not isinstance(p, str) or not p for p in parties):
            raise ValueError(f"Treaty.parties must be non-empty strings, got {parties!r}")
        if len(set(parties)) != len(parties):
            raise ValueError(f"Treaty.parties must be unique, got {parties!r}")
        if not isinstance(self.allocations_mm3, Mapping):
            raise ValueError("Treaty.allocations_mm3 must be a dict of party -> Mm3/yr")
        allocations: Dict[str, float] = {}
        for party, volume in self.allocations_mm3.items():
            if party not in parties:
                raise ValueError(
                    f"Treaty.allocations_mm3 names {party!r}, which is not a party {parties!r}"
                )
            allocations[party] = _nonneg(volume, f"Treaty.allocations_mm3[{party!r}]")
        out: Dict[str, Any] = {"parties": parties, "allocations_mm3": allocations}
        for flag in self.FEATURE_FLAGS:
            out[flag] = _as_bool(getattr(self, flag), f"Treaty.{flag}")
        period = self.review_period_years
        if period is not None:
            period = _as_int(period, "Treaty.review_period_years")
            if period <= 0:
                raise ValueError(f"Treaty.review_period_years must be > 0, got {period}")
        out["review_period_years"] = period
        out["year_signed"] = None if self.year_signed is None else _as_int(self.year_signed, "Treaty.year_signed")
        ref = self.reference_flow_mm3
        if ref is not None:
            ref = _as_float(ref, "Treaty.reference_flow_mm3")
            if ref <= 0.0:
                raise ValueError(f"Treaty.reference_flow_mm3 must be > 0, got {ref}")
        out["reference_flow_mm3"] = ref
        return out

    # -- helpers -------------------------------------------------------------
    def covers(self, name: str) -> bool:
        """Whether ``name`` is a party to the treaty."""
        return name in self.parties

    def features(self) -> Dict[str, bool]:
        """Presence of each resilience mechanism (keys of
        :data:`TREATY_RESILIENCE_WEIGHTS`)."""
        fields = self._validated_fields()
        out = {flag: bool(fields[flag]) for flag in self.FEATURE_FLAGS}
        out["review_period"] = fields["review_period_years"] is not None
        return out

    def entitlements(self, natural_flow_mm3: Optional[float] = None) -> Dict[str, float]:
        """Volumetric entitlements this year, Mm3/yr, keyed by party.

        Fixed volumes unless the treaty has ``variable_allocation`` and a
        ``reference_flow_mm3``, in which case they scale with
        ``natural_flow_mm3 / reference_flow_mm3`` ("shares of actual flow").
        Parties without an allocation are absent from the result.

        Parameters
        ----------
        natural_flow_mm3 : float, optional
            This year's natural basin flow (Mm3/yr, >= 0).

        Returns
        -------
        dict
        """
        fields = self._validated_fields()
        allocations: Dict[str, float] = fields["allocations_mm3"]
        reference = fields["reference_flow_mm3"]
        if fields["variable_allocation"] and reference is not None and natural_flow_mm3 is not None:
            scale = _nonneg(natural_flow_mm3, "natural_flow_mm3") / reference
            return {p: v * scale for p, v in allocations.items()}
        return dict(allocations)

    def to_dict(self) -> Dict[str, Any]:
        fields = self._validated_fields()
        return {
            "name": self.name,
            "parties": list(fields["parties"]),
            "allocations_mm3": dict(fields["allocations_mm3"]),
            **{flag: bool(fields[flag]) for flag in self.FEATURE_FLAGS},
            "review_period_years": fields["review_period_years"],
            "year_signed": fields["year_signed"],
            "reference_flow_mm3": fields["reference_flow_mm3"],
            "resilience": treaty_resilience(self),
        }


@dataclass
class BarEvent:
    """One water-related interaction between riparians on the BAR scale.

    Attributes
    ----------
    year : int
        Year of the event.
    parties : list of str
        Riparians involved (may be empty when unknown).
    scale : int
        Basins at Risk intensity, ``-7`` (formal declaration of war) to
        ``+7`` (voluntary unification); ``0`` is neutral.  See
        :data:`BAR_SCALE`.
    issue : str
        Issue area (e.g. ``"quantity"``, ``"infrastructure"``,
        ``"quality"``, ``"hydropower"``).
    description : str
        Free text.

    References
    ----------
    Wolf, Yoffe & Giordano (2003); Yoffe, Wolf & Giordano (2003), Table 1.
    """

    year: int
    parties: List[str]
    scale: int
    issue: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        self.year = _as_int(self.year, "BarEvent.year")
        if isinstance(self.parties, (str, bytes)) or not isinstance(self.parties, Iterable):
            raise ValueError("BarEvent.parties must be a list of riparian names")
        parties = list(self.parties)
        if any(not isinstance(p, str) for p in parties):
            raise ValueError(f"BarEvent.parties must be strings, got {parties!r}")
        self.parties = parties
        scale = _as_int(self.scale, "BarEvent.scale")
        if not BAR_MIN <= scale <= BAR_MAX:
            raise ValueError(f"BarEvent.scale must lie in [{BAR_MIN}, {BAR_MAX}], got {scale}")
        self.scale = scale
        if not isinstance(self.issue, str) or not isinstance(self.description, str):
            raise ValueError("BarEvent.issue and BarEvent.description must be strings")

    @property
    def label(self) -> str:
        """Verbal BAR scale label of the event."""
        return BAR_SCALE[self.scale]

    @property
    def is_conflictive(self) -> bool:
        return self.scale < 0

    @property
    def is_cooperative(self) -> bool:
        return self.scale > 0

    def involves(self, parties: Iterable[str]) -> bool:
        """Whether any of ``parties`` took part in the event."""
        wanted = set(parties)
        return any(p in wanted for p in self.parties)


def bar_label(scale: int) -> str:
    """Verbal label of a BAR scale value.

    Parameters
    ----------
    scale : int
        Event intensity in ``-7..7``.

    Returns
    -------
    str

    Raises
    ------
    ValueError
        If ``scale`` is not an integer in ``[-7, 7]``.

    References
    ----------
    Yoffe, Wolf & Giordano (2003), Table 1.
    """
    s = _as_int(scale, "scale")
    if s not in BAR_SCALE:
        raise ValueError(f"scale must lie in [{BAR_MIN}, {BAR_MAX}], got {s}")
    return BAR_SCALE[s]


# ---------------------------------------------------------------------------
# hydro-hegemony
# ---------------------------------------------------------------------------
def hydro_hegemony(
    riparian: Riparian,
    basin: Basin,
    weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """Four-pillar hydro-hegemony profile of a riparian (0..1 each).

    Zeitoun & Warner (2006) explain the control of shared rivers by
    *hydro-hegemony*: the riparian that combines a favourable position with
    superior power can impose its preferred water order.  Cascao & Zeitoun
    (2010) decompose this into four pillars, all scored on 0..1 here:

    * ``geographic`` - riparian position; an upstream state controls the
      flow physically.  ``1 - position / (n - 1)``: 1 for the most upstream,
      0 for the most downstream riparian (1 when the basin has a single
      riparian);
    * ``material`` - structural ("hard") power: economy, military,
      technology, finance (``riparian.material_power``);
    * ``bargaining`` - ability to set the agenda and the rules of the game
      (``riparian.bargaining_power``);
    * ``ideational`` - power over ideas, discourse and knowledge
      (``riparian.ideational_power``).

    ``score`` is the weighted mean of the four pillars.

    Parameters
    ----------
    riparian : Riparian
        The riparian to profile; its power attributes are read from this
        object (so a modified copy can be used for "what-if" analysis).
    basin : Basin
        Basin the riparian belongs to (by name); provides the position and
        the number of riparians.
    weights : mapping, optional
        ``{pillar: weight}`` overriding the default equal weights (1 each);
        unlisted pillars keep weight 1; negative weights raise.

    Returns
    -------
    dict
        ``{"geographic", "material", "bargaining", "ideational", "score"}``,
        each in ``[0, 1]``.

    Raises
    ------
    ValueError
        If the riparian is not in the basin, a power attribute lies outside
        ``[0, 1]`` or the weights are invalid.

    References
    ----------
    Zeitoun & Warner (2006); Cascao & Zeitoun (2010).
    """
    _check_basin(basin)
    if not isinstance(riparian, Riparian):
        raise ValueError(f"riparian must be a wefnexus.models.Riparian, got {type(riparian).__name__}")
    try:
        member = basin.riparian(riparian.name)
    except KeyError:
        raise ValueError(
            f"riparian {riparian.name!r} is not part of basin {basin.name!r} (riparians: {basin.names()})"
        ) from None
    n = len(basin)
    position = int(member.position)
    if n <= 1:
        geographic = 1.0
    else:
        geographic = _clip01(1.0 - position / (n - 1))
    pillars = {
        "geographic": geographic,
        "material": _fraction(riparian.material_power, f"riparian {riparian.name!r} material_power"),
        "bargaining": _fraction(riparian.bargaining_power, f"riparian {riparian.name!r} bargaining_power"),
        "ideational": _fraction(riparian.ideational_power, f"riparian {riparian.name!r} ideational_power"),
    }
    w = _check_weights(weights, {k: 1.0 for k in HEGEMONY_PILLARS}, "weights")
    score = sum(w[k] * pillars[k] for k in HEGEMONY_PILLARS) / sum(w.values())
    pillars["score"] = _clip01(score)
    return pillars


def power_asymmetry(basin: Basin, weights: Optional[Mapping[str, float]] = None) -> float:
    """Power asymmetry of a basin: spread of hydro-hegemony scores.

    ``max_i score_i - min_i score_i`` over the riparians, in ``[0, 1]``.
    Large asymmetry allows the hegemon to impose a unilateral order
    (Zeitoun & Warner 2006) and is one of the Basins-at-Risk indicators.

    Parameters
    ----------
    basin : Basin
        Basin (never mutated).
    weights : mapping, optional
        Pillar weights passed to :func:`hydro_hegemony`.

    Returns
    -------
    float
        ``0.0`` for a basin with fewer than two riparians.

    References
    ----------
    Zeitoun & Warner (2006); Wolf, Yoffe & Giordano (2003).
    """
    _check_basin(basin)
    scores = [hydro_hegemony(r, basin, weights)["score"] for r in basin.riparians]
    if len(scores) < 2:
        return 0.0
    return _clip01(max(scores) - min(scores))


# ---------------------------------------------------------------------------
# BAR events: cooperation and conflict
# ---------------------------------------------------------------------------
def _check_events(events: Any) -> List[BarEvent]:
    if isinstance(events, (BarEvent, str, bytes)):
        raise ValueError("events must be a sequence of BarEvent objects (wrap a single event in a list)")
    try:
        items = list(events)
    except TypeError:
        raise ValueError(f"events must be a sequence of BarEvent objects, got {type(events).__name__}") from None
    for i, e in enumerate(items):
        if not isinstance(e, BarEvent):
            raise ValueError(f"events[{i}] must be a wefnexus.diplomacy.BarEvent, got {type(e).__name__}")
        # re-validate without mutating: fields may have been edited since construction
        if isinstance(e.scale, bool) or not isinstance(e.scale, (int, np.integer)) or not BAR_MIN <= int(e.scale) <= BAR_MAX:
            raise ValueError(f"events[{i}].scale must be an integer in [{BAR_MIN}, {BAR_MAX}], got {e.scale!r}")
        if isinstance(e.year, bool) or not isinstance(e.year, (int, np.integer)):
            raise ValueError(f"events[{i}].year must be an integer, got {e.year!r}")
    return items


def _filter_events(
    events: Any,
    window_years: Optional[int],
    until: Optional[int],
    parties: Optional[Iterable[str]] = None,
) -> List[BarEvent]:
    """Events ``e`` with ``until - window_years < e.year <= until``.

    ``until`` defaults to the latest event year; ``parties`` keeps only events
    involving at least one of the named parties.
    """
    items = _check_events(events)
    if parties is not None:
        if isinstance(parties, (str, bytes)):
            raise ValueError("parties must be a sequence of names, not a single string")
        wanted = list(parties)
        items = [e for e in items if e.involves(wanted)]
    if window_years is not None:
        window = _as_int(window_years, "window_years")
        if window <= 0:
            raise ValueError(f"window_years must be >= 1, got {window}")
    else:
        window = None
    if until is not None:
        end = _as_int(until, "until")
    elif items:
        end = max(e.year for e in items)
    else:
        return []
    out = [e for e in items if e.year <= end]
    if window is not None:
        out = [e for e in out if e.year > end - window]
    return out


def cooperation_index(
    events: Sequence[BarEvent],
    window_years: Optional[int] = None,
    until: Optional[int] = None,
    *,
    parties: Optional[Iterable[str]] = None,
) -> float:
    """Mean BAR intensity of the events in a window (``-7..7``).

    Positive values indicate predominantly cooperative interaction, negative
    values predominantly conflictive interaction; this is the "average BAR
    scale" indicator of the Basins at Risk project, where the historical
    mean over all basins was mildly cooperative (Wolf et al. 2003).

    It is a *net* balance of the record, not an axis of the TWINS matrix: a
    freshwater treaty (``+6``) and extensive war acts (``-6``) in the same
    window average to ``0``.  Use :func:`cooperation_intensity` and
    :func:`conflict_intensity`, which score the cooperative and the
    conflictive events separately, as the inputs of
    :func:`twins_classification`.

    Parameters
    ----------
    events : sequence of BarEvent
        Event record of the basin.
    window_years : int, optional
        Length of the window (years) ending at ``until``; events with
        ``until - window_years < year <= until`` are kept.  ``None`` keeps
        every event up to ``until``.
    until : int, optional
        Last year of the window (default: the latest event year).
    parties : iterable of str, optional
        Keep only events involving at least one of these riparians.

    Returns
    -------
    float
        Mean scale, ``0.0`` when no event falls in the window.

    Raises
    ------
    ValueError
        On non-``BarEvent`` items or a non-positive window.

    References
    ----------
    Wolf, Yoffe & Giordano (2003); Yoffe, Wolf & Giordano (2003).
    """
    selected = _filter_events(events, window_years, until, parties)
    return _mean(float(e.scale) for e in selected)


def cooperation_intensity(
    events: Sequence[BarEvent],
    window_years: Optional[int] = None,
    until: Optional[int] = None,
    *,
    parties: Optional[Iterable[str]] = None,
) -> float:
    """Mean BAR intensity of the *cooperative* events in a window.

    Only events with a positive scale count; the result lies in ``[0, 7]``
    and is ``0.0`` when there is no cooperative event.  This is the
    cooperation axis of the TWINS matrix (Mirumachi & Allan 2007), scored
    from the cooperative events alone exactly as :func:`conflict_intensity`
    scores the conflict axis from the conflictive events alone.  TWINS was
    built on the observation that conflict and cooperation coexist, so a
    hostile event must not cancel a treaty signed in the same window - which
    the net :func:`cooperation_index` lets happen.

    Parameters
    ----------
    events : sequence of BarEvent
        Event record of the basin.
    window_years : int, optional
        Length of the window (years) ending at ``until``; events with
        ``until - window_years < year <= until`` are kept.  ``None`` keeps
        every event up to ``until``.
    until : int, optional
        Last year of the window (default: the latest event year).
    parties : iterable of str, optional
        Keep only events involving at least one of these riparians.

    Returns
    -------
    float
        Mean scale over positive events, ``0..7`` (``0.0`` if none).

    Raises
    ------
    ValueError
        On non-``BarEvent`` items or a non-positive window.

    References
    ----------
    Mirumachi & Allan (2007); Mirumachi (2015); Yoffe, Wolf & Giordano
    (2003).
    """
    selected = _filter_events(events, window_years, until, parties)
    return _mean(float(e.scale) for e in selected if e.scale > 0)


def conflict_intensity(
    events: Sequence[BarEvent],
    window_years: Optional[int] = None,
    until: Optional[int] = None,
    *,
    parties: Optional[Iterable[str]] = None,
) -> float:
    """Mean absolute BAR intensity of the *conflictive* events in a window.

    Only events with a negative scale count; the result lies in ``[0, 7]``
    and is ``0.0`` when there is no conflictive event.  This is the
    conflict axis of the TWINS matrix (Mirumachi & Allan 2007); its
    counterpart on the cooperation axis is :func:`cooperation_intensity`.

    Parameters
    ----------
    events, window_years, until, parties
        As in :func:`cooperation_index`.

    Returns
    -------
    float
        Mean ``|scale|`` over negative events, ``0..7``.

    References
    ----------
    Yoffe, Wolf & Giordano (2003); Mirumachi & Allan (2007).
    """
    selected = _filter_events(events, window_years, until, parties)
    return _mean(float(-e.scale) for e in selected if e.scale < 0)


def bar_event_summary(
    events: Sequence[BarEvent],
    window_years: Optional[int] = None,
    until: Optional[int] = None,
    *,
    parties: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Counts and indices of a BAR event record.

    Parameters
    ----------
    events, window_years, until, parties
        As in :func:`cooperation_index`.

    Returns
    -------
    dict
        ``n_events``, ``n_cooperative`` (scale > 0), ``n_conflictive``
        (scale < 0), ``n_neutral``, ``share_conflictive``,
        ``cooperation_index`` (net mean scale of every event, see
        :func:`cooperation_index`), ``cooperation_intensity`` (mean scale of
        the cooperative events - the TWINS cooperation axis),
        ``conflict_intensity`` (mean ``|scale|`` of the conflictive events -
        the TWINS conflict axis), ``most_conflictive`` (lowest scale,
        ``None`` if no events), ``most_cooperative`` and ``twins`` (the
        :func:`twins_classification` of the two intensities;
        ``"none/none"`` for an empty record).
    """
    selected = _filter_events(events, window_years, until, parties)
    n = len(selected)
    n_coop = sum(1 for e in selected if e.scale > 0)
    n_conf = sum(1 for e in selected if e.scale < 0)
    net = _mean(float(e.scale) for e in selected)
    coop = _mean(float(e.scale) for e in selected if e.scale > 0)
    conf = _mean(float(-e.scale) for e in selected if e.scale < 0)
    return {
        "n_events": n,
        "n_cooperative": n_coop,
        "n_conflictive": n_conf,
        "n_neutral": n - n_coop - n_conf,
        "share_conflictive": (n_conf / n) if n else 0.0,
        "cooperation_index": net,
        "cooperation_intensity": coop,
        "conflict_intensity": conf,
        "most_conflictive": min((e.scale for e in selected), default=None),
        "most_cooperative": max((e.scale for e in selected), default=None),
        "twins": twins_classification(coop, conf),
    }


def twins_cooperation_class(cooperation: float) -> str:
    """TWINS cooperation intensity class of a cooperation intensity.

    Mirumachi & Allan (2007) distinguish, with increasing intensity,
    *confrontation of the issue* (the shared problem is acknowledged and
    talked about), *ad hoc joint action*, *technical cooperation*,
    *risk-averting cooperation* and *risk-taking cooperation*; ``"none"`` is
    returned when there is no cooperative interaction at all.  The classes
    are mapped onto the mean BAR scale of the cooperative events through
    :data:`TWINS_COOPERATION_THRESHOLDS`, so each class is reached by the
    corresponding positive BAR level (``+1`` confrontation of the issue,
    ``+2`` ad hoc, ``+3`` technical, ``+4``/``+5`` risk-averting, ``+6``/``+7``
    risk-taking).

    Parameters
    ----------
    cooperation : float
        Cooperation intensity in ``[0, 7]`` (see
        :func:`cooperation_intensity`).  The net :func:`cooperation_index`
        (``-7..7``) mixes both TWINS axes and is not a valid input.

    Returns
    -------
    str
        One of ``"none"``, ``"confrontation"``, ``"ad_hoc"``,
        ``"technical"``, ``"risk_averting"``, ``"risk_taking"``.

    Raises
    ------
    ValueError
        If ``cooperation`` lies outside ``[0, 7]``.

    References
    ----------
    Mirumachi & Allan (2007); Mirumachi (2015); Yoffe, Wolf & Giordano
    (2003), Table 1.
    """
    c = _as_float(cooperation, "cooperation")
    if not 0.0 <= c <= BAR_MAX:
        raise ValueError(
            f"cooperation must be a cooperation intensity in [0, {BAR_MAX}] (see cooperation_intensity()), "
            f"got {c}; the net cooperation_index() is not the TWINS cooperation axis"
        )
    if c <= 0.0:
        return "none"
    label = "confrontation"
    for name, lower in TWINS_COOPERATION_THRESHOLDS.items():
        if c >= lower:
            label = name
    return label


def twins_conflict_class(conflict: float) -> str:
    """TWINS conflict intensity class of a conflict intensity.

    Following Warner (2004) and Mirumachi & Allan (2007) an issue is
    *non-politicised*, *politicised*, *securitised* (framed as an
    existential threat) or *violent* ("violised"); ``"none"`` when there is
    no conflictive interaction at all.  Classes are mapped onto the mean
    absolute BAR scale of negative events via
    :data:`TWINS_CONFLICT_THRESHOLDS`.

    Parameters
    ----------
    conflict : float
        Conflict intensity in ``[0, 7]`` (see :func:`conflict_intensity`).

    Returns
    -------
    str
        One of ``"none"``, ``"non-politicised"``, ``"politicised"``,
        ``"securitised"``, ``"violent"``.
    """
    k = _as_float(conflict, "conflict")
    if not 0.0 <= k <= BAR_MAX:
        raise ValueError(f"conflict must lie in [0, {BAR_MAX}], got {k}")
    if k <= 0.0:
        return "none"
    label = "non-politicised"
    for name, lower in TWINS_CONFLICT_THRESHOLDS.items():
        if k >= lower:
            label = name
    return label


def twins_classification(cooperation: float, conflict: float) -> str:
    """Position of a riparian relationship in the TWINS matrix.

    The Transboundary Waters Interaction NexuS (Mirumachi & Allan 2007;
    Mirumachi 2015) rejects the idea of a single conflict-cooperation
    continuum: conflict and cooperation *coexist*, so a relationship is
    located by two coordinates that are scored independently of each other
    - the intensity of the cooperative events and the intensity of the
    conflictive events.  A hostile event therefore never erases a treaty
    signed in the same window, nor the reverse.  The cell is returned as
    ``"<cooperation>/<conflict>"``, e.g. ``"technical/politicised"``.

    Parameters
    ----------
    cooperation : float
        Cooperation intensity in ``[0, 7]``: mean BAR scale of the
        *cooperative* events (see :func:`cooperation_intensity`).  The net
        :func:`cooperation_index` is not a TWINS coordinate and is rejected
        when negative.
    conflict : float
        Conflict intensity in ``[0, 7]``: mean ``|scale|`` of the
        *conflictive* events (see :func:`conflict_intensity`).

    Returns
    -------
    str
        Cooperation class (``none``, ``confrontation``, ``ad_hoc``,
        ``technical``, ``risk_averting``, ``risk_taking``) and conflict
        class (``none``, ``non-politicised``, ``politicised``,
        ``securitised``, ``violent``) joined by ``"/"``; ``"none/none"``
        when there is no interaction at all.

    Raises
    ------
    ValueError
        If a coordinate lies outside its range.

    References
    ----------
    Mirumachi & Allan (2007); Mirumachi (2015); Warner (2004).
    """
    return f"{twins_cooperation_class(cooperation)}/{twins_conflict_class(conflict)}"


# ---------------------------------------------------------------------------
# treaties
# ---------------------------------------------------------------------------
def treaty_from_basin(basin: Basin, name: Optional[str] = None, **features: Any) -> Optional[Treaty]:
    """Build a bare fixed-volume :class:`Treaty` from the riparians' entitlements.

    Riparians whose ``treaty_allocation_mm3`` is not ``None`` become the
    parties, with those volumes as ``allocations_mm3``.  Extra keyword
    arguments (``drought_provisions=True`` ...) are passed to
    :class:`Treaty`.

    Parameters
    ----------
    basin : Basin
        Basin (never mutated).
    name : str, optional
        Treaty name (default ``"<basin name> treaty"``).
    **features
        Any other :class:`Treaty` field.

    Returns
    -------
    Treaty or None
        ``None`` when no riparian has a treaty entitlement.
    """
    _check_basin(basin)
    parties = [r.name for r in basin.riparians if r.treaty_allocation_mm3 is not None]
    if not parties:
        return None
    allocations = {
        r.name: _nonneg(r.treaty_allocation_mm3, f"riparian {r.name!r} treaty_allocation_mm3")
        for r in basin.riparians
        if r.treaty_allocation_mm3 is not None
    }
    treaty_name = f"{basin.name} treaty" if name is None else name
    return Treaty(name=treaty_name, parties=parties, allocations_mm3=allocations, **features)


def treaty_resilience(treaty: Treaty, weights: Optional[Mapping[str, float]] = None) -> float:
    """Institutional resilience of a treaty to hydrological change (0..1).

    De Stefano et al. (2012) rate basins by the presence of treaty
    mechanisms that let the agreement absorb variability: a *flexible
    allocation* (shares of flow rather than fixed volumes), *variability
    management* (drought / flood provisions), *amendment and review*
    procedures, *conflict resolution* mechanisms, a *joint institution*
    (river basin organisation) and *information exchange*.  Benefit sharing
    (Sadoff & Grey 2002) is added as a seventh mechanism.  The score is the
    weight-sum of the mechanisms present divided by the total weight
    (:data:`TREATY_RESILIENCE_WEIGHTS`), so a bare fixed-volume treaty scores
    0 and a treaty with every mechanism scores 1.

    Parameters
    ----------
    treaty : Treaty
        The agreement.
    weights : mapping, optional
        Overrides of the default mechanism weights (keys of
        :data:`TREATY_RESILIENCE_WEIGHTS`); negative weights raise.

    Returns
    -------
    float
        Resilience in ``[0, 1]``.

    Raises
    ------
    ValueError
        On an invalid treaty or weights.

    References
    ----------
    De Stefano et al. (2012); Drieschova et al. (2008); Dinar et al. (2015);
    Sadoff & Grey (2002).
    """
    _check_treaty(treaty)
    w = _check_weights(weights, TREATY_RESILIENCE_WEIGHTS, "weights")
    present = treaty.features()
    score = sum(w[k] for k, is_present in present.items() if is_present) / sum(w.values())
    return _clip01(score)


def treaty_compliance(
    treaty: Treaty,
    balance: BasinBalance,
    tol: float = 1e-9,
) -> Dict[str, Dict[str, float]]:
    """Compliance of a routed year with a treaty's volumetric entitlements.

    For every party the function compares its actual surface withdrawal
    with its entitlement (over-abstraction when the ratio exceeds 1) and
    checks whether the water *arriving* at its reach (routed upstream outflow
    plus local runoff) was at least its entitlement - i.e. whether upstream
    parties *delivered* it.

    Parameters
    ----------
    treaty : Treaty
        The agreement; every party must be a reach of ``balance``.
    balance : BasinBalance
        A routed year from :func:`wefnexus.water.route_basin`.  Its
        ``natural_flow_mm3`` scales variable allocations (see
        :meth:`Treaty.entitlements`).
    tol : float, optional
        Relative tolerance on the comparisons (default 1e-9).

    Returns
    -------
    dict
        ``{party: {...}}`` with, all in Mm3/yr unless dimensionless:

        * ``entitlement`` - volumetric entitlement (``inf`` when the party
          has no volumetric allocation);
        * ``withdrawal`` - actual surface withdrawal;
        * ``compliance_ratio`` - ``withdrawal / entitlement``; ``> 1`` means
          over-abstraction.  Conventions: ``0.0`` for an unlimited
          entitlement; for a zero entitlement ``1.0`` if nothing was
          withdrawn, else ``inf``;
        * ``over_abstraction_mm3`` - ``max(withdrawal - entitlement, 0)``;
        * ``compliant`` - ``1.0`` if ``withdrawal <= entitlement``, else
          ``0.0``;
        * ``inflow`` and ``upstream_inflow`` - water arriving at the reach;
        * ``delivery_ratio`` - ``inflow / entitlement`` (``1.0`` when there
          is no finite positive entitlement to deliver);
        * ``delivery_shortfall_mm3`` - ``max(entitlement - inflow, 0)``;
        * ``delivered`` - ``1.0`` if ``inflow >= entitlement``, else ``0.0``.

    Raises
    ------
    ValueError
        If a party is not a reach of the balance or inputs are invalid.

    References
    ----------
    Wolf, Yoffe & Giordano (2003); De Stefano et al. (2012); UN (1997)
    Watercourses Convention, Art. 5-7 (equitable utilisation, no significant
    harm).
    """
    _check_treaty(treaty)
    _check_balance(balance)
    rel = _nonneg(tol, "tol")
    names = balance.names()
    missing = [p for p in treaty.parties if p not in names]
    if missing:
        raise ValueError(f"treaty parties {missing!r} are not reaches of the balance {names!r}")
    entitlements = treaty.entitlements(balance.natural_flow_mm3)
    out: Dict[str, Dict[str, float]] = {}
    for party in treaty.parties:
        reach = balance.reach(party)
        e = entitlements.get(party, math.inf)
        w = reach.total_withdrawal()
        inflow = reach.inflow_mm3
        if math.isinf(e):
            ratio, over, compliant = 0.0, 0.0, 1.0
            delivery, shortfall, delivered = 1.0, 0.0, 1.0
        elif e <= 0.0:
            ratio = 1.0 if w <= rel else math.inf
            over = max(w, 0.0)
            compliant = 1.0 if w <= rel else 0.0
            delivery, shortfall, delivered = 1.0, 0.0, 1.0
        else:
            ratio = w / e
            over = max(w - e, 0.0)
            compliant = 1.0 if w <= e * (1.0 + rel) else 0.0
            delivery = inflow / e
            shortfall = max(e - inflow, 0.0)
            delivered = 1.0 if inflow >= e * (1.0 - rel) else 0.0
        out[party] = {
            "entitlement": float(e),
            "withdrawal": float(w),
            "compliance_ratio": float(ratio),
            "over_abstraction_mm3": float(over),
            "compliant": compliant,
            "inflow": float(inflow),
            "upstream_inflow": float(reach.upstream_inflow_mm3),
            "delivery_ratio": float(delivery),
            "delivery_shortfall_mm3": float(shortfall),
            "delivered": delivered,
        }
    return out


def compliance_summary(compliance: Mapping[str, Mapping[str, float]]) -> Dict[str, float]:
    """Basin-level summary of a :func:`treaty_compliance` result.

    Parameters
    ----------
    compliance : mapping
        Output of :func:`treaty_compliance`.

    Returns
    -------
    dict
        ``n_parties``, ``share_compliant``, ``share_delivered`` (both 1.0
        when there are no parties), ``total_over_abstraction_mm3``,
        ``total_delivery_shortfall_mm3`` and ``operational`` (1.0 when every
        party is compliant and delivered, else 0.0) - a usable proxy for
        SDG 6.5.2 "operational arrangement".
    """
    if not isinstance(compliance, Mapping):
        raise ValueError("compliance must be the mapping returned by treaty_compliance")
    rows = list(compliance.values())
    n = len(rows)
    for row in rows:
        if not isinstance(row, Mapping) or "compliant" not in row or "delivered" not in row:
            raise ValueError("compliance rows must contain 'compliant' and 'delivered'")
    share_c = _mean(float(r["compliant"]) for r in rows) if n else 1.0
    share_d = _mean(float(r["delivered"]) for r in rows) if n else 1.0
    return {
        "n_parties": float(n),
        "share_compliant": share_c,
        "share_delivered": share_d,
        "total_over_abstraction_mm3": float(sum(float(r.get("over_abstraction_mm3", 0.0)) for r in rows)),
        "total_delivery_shortfall_mm3": float(sum(float(r.get("delivery_shortfall_mm3", 0.0)) for r in rows)),
        "operational": 1.0 if (share_c >= 1.0 and share_d >= 1.0) else 0.0,
    }


# ---------------------------------------------------------------------------
# dependency
# ---------------------------------------------------------------------------
def water_dependency(basin: Basin, balance: Optional[BasinBalance] = None) -> Dict[str, float]:
    """Share of each riparian's river inflow that originates upstream.

    ``upstream_inflow / (upstream_inflow + local_inflow)`` - the surface-water
    analogue of the FAO AQUASTAT *dependency ratio*, computed with
    :func:`wefnexus.water.dependency_ratio`.  With a routed ``balance`` the
    actual (post-use) upstream inflow is used; without one the natural,
    zero-use flows of :func:`wefnexus.water.natural_flows` are used.  A high
    downstream dependency is a classic Basins-at-Risk vulnerability.

    Parameters
    ----------
    basin : Basin
        Basin (never mutated).
    balance : BasinBalance, optional
        Routed year of the same basin.

    Returns
    -------
    dict
        ``{riparian: ratio in [0, 1]}`` upstream -> downstream; ``0.0`` when
        a reach receives no water at all.

    References
    ----------
    FAO AQUASTAT glossary (dependency ratio); Wolf, Yoffe & Giordano (2003).
    """
    _check_basin(basin)
    out: Dict[str, float] = {}
    if balance is None:
        flows = natural_flows(basin)
        upstream = _nonneg(basin.headwater_inflow_mm3, "basin.headwater_inflow_mm3")
        for r in basin.riparians:
            total = flows[r.name]
            out[r.name] = dependency_ratio(min(upstream, total), total)
            upstream = total
        return out
    _check_balance(balance, basin)
    for reach in balance.reaches:
        total = max(reach.inflow_mm3, 0.0)
        upstream = min(max(reach.upstream_inflow_mm3, 0.0), total)
        out[reach.name] = dependency_ratio(upstream, total)
    return out


# ---------------------------------------------------------------------------
# conflict risk
# ---------------------------------------------------------------------------
def risk_category(score: float) -> str:
    """Verbal category of a conflict-risk score (``low`` < 0.25 <=
    ``moderate`` < 0.5 <= ``high`` < 0.75 <= ``very_high``)."""
    s = _as_float(score, "score")
    if not 0.0 <= s <= 1.0:
        raise ValueError(f"score must lie in [0, 1], got {s}")
    for name, upper in CONFLICT_RISK_CATEGORIES.items():
        if s < upper:
            return name
    return "very_high"


def _hydropower_of_reach(riparian: Riparian, outflow_mm3: float) -> float:
    """Hydropower (GWh/yr) generated by a riparian from its reach outflow."""
    es = riparian.energy
    capacity = _nonneg(es.hydropower_capacity_mw, f"riparian {riparian.name!r} energy.hydropower_capacity_mw")
    head = _nonneg(es.hydropower_head_m, f"riparian {riparian.name!r} energy.hydropower_head_m")
    if capacity <= 0.0 or head <= 0.0:
        return 0.0
    turbined = _fraction(es.turbined_fraction, f"riparian {riparian.name!r} energy.turbined_fraction")
    return hydropower_gwh(
        max(outflow_mm3, 0.0) * turbined,
        head,
        efficiency=es.turbine_efficiency,
        capacity_mw=capacity,
    )


def conflict_risk_index(
    basin: Basin,
    balance: BasinBalance,
    treaty: Optional[Treaty] = None,
    events: Sequence[BarEvent] = (),
    climate_cv: Optional[float] = None,
    *,
    weights: Optional[Mapping[str, float]] = None,
    window_years: Optional[int] = None,
    until: Optional[int] = None,
) -> Dict[str, Any]:
    """Basins-at-Risk style composite conflict risk (0 = none, 1 = extreme).

    Wolf, Yoffe & Giordano (2003) found that the likelihood of water-related
    tension rises when *rapid or extreme change* in a basin's physical or
    institutional setting outpaces the *institutional capacity* to absorb
    it.  The index combines eight normalised components, each in ``[0, 1]``
    with higher = riskier, by a weighted mean
    (:data:`CONFLICT_RISK_WEIGHTS`):

    ``water_stress``
        Mean over riparians of ``min(SDG 6.4.2 / 100 %, 1)``, where SDG
        6.4.2 = withdrawal / (renewable - environmental flow) uses the
        reach's surface withdrawal plus groundwater use as withdrawal and
        its river inflow plus groundwater recharge as renewable resource
        (FAO 2018).  ``inf`` stress (no water beyond the environmental flow)
        counts as 1.
    ``dependency``
        Mean upstream dependency of :func:`water_dependency` (routed).
    ``power_asymmetry``
        :func:`power_asymmetry` of the basin.
    ``institutional``
        Treaty absence / weakness: ``1.0`` without a treaty; with one,
        ``1 - coverage * (c + (1 - c) * resilience)`` where ``coverage`` is
        the share of riparians that are parties, ``resilience`` is
        :func:`treaty_resilience` and ``c`` is :data:`TREATY_PRESENCE_CREDIT`
        (so a bare treaty covering everybody scores 0.5 and a fully
        resilient one 0).
    ``conflict_history``
        Mean over the BAR events in the window of ``max(-scale, 0) / 7``:
        the average conflictive intensity of the record (``0`` without
        events or with only cooperative ones, ``1`` if every event is a
        formal war).
    ``variability``
        ``min(climate_cv / VARIABILITY_CV_REFERENCE, 1)`` - hydrological
        variability, the "extreme change" driver (De Stefano et al. 2012).
    ``environmental``
        Mean over reaches of the relative environmental-flow shortfall
        ``max(env - outflow, 0) / env`` (0 when ``env`` is 0).
    ``dam_filling``
        Share of the natural flow diverted into the reservoirs of riparians
        that have someone downstream, ``sum(refill of non-terminal reaches)
        / natural flow`` (clipped to 1): unilateral storage of the flow by
        upstream states is the most frequently cited trigger of
        transboundary tension (Wolf et al. 2003).

    Parameters
    ----------
    basin : Basin
        Basin (never mutated).
    balance : BasinBalance
        Routed year of the same basin (e.g. ``route_basin(basin, 0.6)`` for
        a drought year).
    treaty : Treaty, optional
        Agreement in force; ``None`` means no treaty (the riparians'
        ``treaty_allocation_mm3`` are *not* consulted; use
        :func:`treaty_from_basin` to build one from them).
    events : sequence of BarEvent, optional
        Event record of the basin.
    climate_cv : float, optional
        Coefficient of variation of annual flow (default
        ``basin.climate_cv``), ``>= 0``.
    weights : mapping, optional
        Overrides of :data:`CONFLICT_RISK_WEIGHTS`.
    window_years, until : int, optional
        Event window as in :func:`cooperation_index`.

    Returns
    -------
    dict
        ``score`` (0..1), ``category`` (``"low"``, ``"moderate"``,
        ``"high"``, ``"very_high"``), ``components`` (the eight values),
        ``weights`` (as used) and ``details`` (per-riparian water stress in
        %, dependencies, hegemony scores, treaty resilience / coverage,
        event counts, environmental shortfalls, upstream refill).

    Raises
    ------
    ValueError
        If the balance does not belong to the basin or any input is invalid.

    References
    ----------
    Wolf, Yoffe & Giordano (2003); Yoffe, Wolf & Giordano (2003); De Stefano
    et al. (2012); Zeitoun & Warner (2006); FAO (2018).
    """
    _check_basin(basin)
    _check_balance(balance, basin)
    w = _check_weights(weights, CONFLICT_RISK_WEIGHTS, "weights")
    names = basin.names()
    n = len(names)

    # -- water stress (SDG 6.4.2) ------------------------------------------
    stress_pct: Dict[str, float] = {}
    for reach, rip in zip(balance.reaches, basin.riparians):
        label = f"riparian {rip.name!r}"
        groundwater = _nonneg(rip.groundwater_abstraction_mm3, f"{label} groundwater_abstraction_mm3")
        desalination = _nonneg(rip.energy.desalination_capacity_mm3, f"{label} energy.desalination_capacity_mm3")
        non_river = max(reach.non_river_supply_mm3, 0.0)
        gw_used = non_river * (groundwater / (groundwater + desalination)) if groundwater + desalination > 0.0 else 0.0
        withdrawal = reach.total_withdrawal() + gw_used
        renewable = max(reach.inflow_mm3, 0.0) + _nonneg(rip.groundwater_recharge_mm3, f"{label} groundwater_recharge_mm3")
        stress_pct[rip.name] = sdg_642_water_stress(withdrawal, renewable, max(reach.environmental_flow_mm3, 0.0))
    water_stress = _mean(1.0 if math.isinf(s) else _clip01(s / 100.0) for s in stress_pct.values())

    # -- downstream dependency ---------------------------------------------
    dependencies = water_dependency(basin, balance)
    dependency = _mean(dependencies.values())

    # -- power asymmetry ---------------------------------------------------
    hegemony = {r.name: hydro_hegemony(r, basin)["score"] for r in basin.riparians}
    asymmetry = power_asymmetry(basin)

    # -- institutional capacity --------------------------------------------
    if treaty is None:
        institutional = 1.0
        resilience: Optional[float] = None
        coverage = 0.0
    else:
        _check_treaty(treaty)
        resilience = treaty_resilience(treaty)
        coverage = (sum(1 for p in treaty.parties if p in names) / n) if n else 0.0
        institutional = _clip01(
            1.0 - coverage * (TREATY_PRESENCE_CREDIT + (1.0 - TREATY_PRESENCE_CREDIT) * resilience)
        )

    # -- conflict history --------------------------------------------------
    selected = _filter_events(events, window_years, until)
    conflict_history = _clip01(_mean(max(-e.scale, 0) / float(BAR_MAX) for e in selected))

    # -- hydrological variability ------------------------------------------
    cv = _nonneg(basin.climate_cv if climate_cv is None else climate_cv, "climate_cv")
    variability = _clip01(cv / VARIABILITY_CV_REFERENCE)

    # -- environmental flows -----------------------------------------------
    env_shortfall: Dict[str, float] = {}
    for reach in balance.reaches:
        env = max(reach.environmental_flow_mm3, 0.0)
        env_shortfall[reach.name] = _clip01(max(env - reach.outflow_mm3, 0.0) / env) if env > 0.0 else 0.0
    environmental = _mean(env_shortfall.values())

    # -- upstream dam filling ----------------------------------------------
    upstream_refill = float(sum(max(r.storage_refill_mm3, 0.0) for r in balance.reaches[:-1]))
    natural = max(balance.natural_flow_mm3, 0.0)
    if natural > 0.0:
        dam_filling = _clip01(upstream_refill / natural)
    else:
        dam_filling = 1.0 if upstream_refill > 0.0 else 0.0

    components = {
        "water_stress": water_stress,
        "dependency": dependency,
        "power_asymmetry": asymmetry,
        "institutional": institutional,
        "conflict_history": conflict_history,
        "variability": variability,
        "environmental": environmental,
        "dam_filling": dam_filling,
    }
    score = _clip01(sum(w[k] * components[k] for k in components) / sum(w.values()))
    return {
        "score": score,
        "category": risk_category(score),
        "components": components,
        "weights": w,
        "details": {
            "water_stress_pct": stress_pct,
            "dependency": dependencies,
            "hegemony": hegemony,
            "treaty_resilience": resilience,
            "treaty_coverage": coverage,
            "n_events": len(selected),
            "n_conflictive_events": sum(1 for e in selected if e.scale < 0),
            "env_shortfall": env_shortfall,
            "upstream_refill_mm3": upstream_refill,
            "natural_flow_mm3": natural,
            "climate_cv": cv,
        },
    }


# ---------------------------------------------------------------------------
# benefit sharing
# ---------------------------------------------------------------------------
def benefit_sharing_matrix(
    basin: Basin,
    balance: BasinBalance,
    cooperative_balance: Optional[BasinBalance] = None,
    *,
    treaty: Optional[Treaty] = None,
    events: Sequence[BarEvent] = (),
    hydropower_value_usd_per_kwh: float = DEFAULT_HYDROPOWER_VALUE_USD_PER_KWH,
) -> Dict[str, Dict[str, float]]:
    """Sadoff & Grey (2002) benefit typology per riparian.

    Sadoff & Grey distinguish four types of benefits of cooperation on
    international rivers, proxied here as follows:

    * ``to_the_river`` - environmental benefits: environmental-flow
      compliance ``min(outflow / env, 1)`` (1 when no requirement),
      dimensionless;
    * ``from_the_river`` - economic benefits *from* the river: value of the
      water withdrawn (``sum_s withdrawal_s * value_usd_per_m3[s] * 1e6``)
      plus hydropower generated from the reach outflow valued at
      ``hydropower_value_usd_per_kwh``; USD/yr;
    * ``because_of_the_river`` - reduced costs of non-cooperation:
      ``1 - conflict_risk_index(...)["score"]`` (basin-wide, dimensionless);
    * ``beyond_the_river`` - catalytic benefits (trade, integration), proxied
      by the riparian's gain in ``from_the_river`` under a cooperative
      allocation: ``from_the_river(cooperative) - from_the_river(balance)``
      in USD/yr (``0.0`` when no cooperative balance is given; may be
      negative for a riparian that gives up water).

    Parameters
    ----------
    basin : Basin
        Basin (never mutated).
    balance : BasinBalance
        Routed year under the status quo (e.g. unilateral entitlements).
    cooperative_balance : BasinBalance, optional
        Routed year under a cooperative allocation of the same basin.
    treaty, events
        Passed to :func:`conflict_risk_index` for the status-quo balance.
    hydropower_value_usd_per_kwh : float, optional
        Value of hydropower (USD/kWh, >= 0; default 0.05).

    Returns
    -------
    dict
        ``{riparian: {"to_the_river", "from_the_river",
        "because_of_the_river", "beyond_the_river", "water_value_usd",
        "hydropower_gwh", "hydropower_value_usd"}}``.

    References
    ----------
    Sadoff & Grey (2002); Sadoff & Grey (2005).
    """
    _check_basin(basin)
    _check_balance(balance, basin)
    if cooperative_balance is not None:
        _check_balance(cooperative_balance, basin, "cooperative_balance")
    price = _nonneg(hydropower_value_usd_per_kwh, "hydropower_value_usd_per_kwh")
    risk = conflict_risk_index(basin, balance, treaty, events)["score"]

    def economic(reach: Any, rip: Riparian) -> Tuple[float, float, float]:
        values = rip.demand.value_usd_per_m3
        water_usd = 0.0
        for s in SECTOR_PRIORITY:
            water_usd += max(reach.withdrawals.get(s, 0.0), 0.0) * _nonneg(
                values.get(s, 0.0), f"riparian {rip.name!r} demand.value_usd_per_m3[{s.value}]"
            ) * _M3_PER_MM3
        gwh = _hydropower_of_reach(rip, reach.outflow_mm3)
        return water_usd, gwh, gwh * _KWH_PER_GWH * price

    out: Dict[str, Dict[str, float]] = {}
    for reach, rip in zip(balance.reaches, basin.riparians):
        env = max(reach.environmental_flow_mm3, 0.0)
        to_river = 1.0 if env <= 0.0 else _clip01(reach.outflow_mm3 / env)
        water_usd, gwh, hydro_usd = economic(reach, rip)
        from_river = water_usd + hydro_usd
        beyond = 0.0
        if cooperative_balance is not None:
            c_water, _, c_hydro = economic(cooperative_balance.reach(rip.name), rip)
            beyond = (c_water + c_hydro) - from_river
        out[rip.name] = {
            "to_the_river": to_river,
            "from_the_river": float(from_river),
            "because_of_the_river": _clip01(1.0 - risk),
            "beyond_the_river": float(beyond),
            "water_value_usd": float(water_usd),
            "hydropower_gwh": float(gwh),
            "hydropower_value_usd": float(hydro_usd),
        }
    return out


# ---------------------------------------------------------------------------
# negotiation
# ---------------------------------------------------------------------------
def bankruptcy_estate(basin: Basin, flow_factor: float = 1.0) -> Dict[str, Any]:
    """Water a basin year can divide among its riparians - the bankruptcy *estate*.

    Bankruptcy rules (Mianabadi et al. 2014) divide a single number, the
    estate, among the claims.  Here the estate is the volume the basin can
    *consume* without violating any in-stream requirement.  Write ``Q_r``
    for the cumulative natural flow at the outlet of reach ``r``
    (:func:`wefnexus.water.natural_flows`), ``env_r`` for the environmental
    flow that must remain in the river there and ``c_k >= 0`` for the
    consumption of reach ``k``; the requirements read
    ``sum_{k <= r} c_k <= Q_r - env_r`` for every reach ``r``.  An in-stream
    flow is not a withdrawal: the water left at an upstream outlet flows on
    and may still be consumed by the riparians below, so the only volume
    nobody may consume is what must still be in the river at the terminal
    outlet.  The largest total consumption compatible with every
    satisfiable requirement is therefore

    ``estate = max(Q_terminal - env_terminal, 0) = max(natural_flow - env_terminal, 0)``

    (a reach whose requirement exceeds the natural flow at its own outlet
    cannot consume anything, exactly as the consumption cap of
    :func:`wefnexus.water.route_basin`; when that reach is the terminal one
    nothing can be divided).  Subtracting the requirements of *every* reach
    instead (``natural_flow - sum_r env_r``) counts the same water once per
    reach it passes and manufactures scarcity: on the stylised example basin
    it withholds 7 000 instead of 1 500 Mm3/yr at mean flow and caps the
    upstream riparians below their demands while far more than the terminal
    requirement flows to the sea.

    The estate ignores reservoir storage and, being consumptive, is a
    conservative bound when the awards are applied as *withdrawal* caps
    (consumption never exceeds withdrawal, so routed awards can never
    consume more than the estate in aggregate, while return flows let the
    actual withdrawals exceed it).  Where a claimant's water may physically
    be taken is not captured by a single number: ``headroom_mm3`` reports
    the cumulative consumption each outlet allows and
    :func:`wefnexus.water.route_basin` enforces it when awards are routed.

    Parameters
    ----------
    basin : Basin
        Basin (never mutated).
    flow_factor : float, optional
        Multiplier on natural flow (>= 0; e.g. 0.6 for a drought year).

    Returns
    -------
    dict
        Volumes in Mm3/yr:

        * ``estate`` - ``max(natural_flow - env_terminal, 0)``;
        * ``natural_flow_mm3`` - headwater plus every local inflow, times
          ``flow_factor``;
        * ``environmental_flow_mm3`` - sum of every reach's requirement
          (information only - it is *not* what is withheld);
        * ``terminal_environmental_flow_mm3`` - requirement at the terminal
          outlet (``0.0`` for a basin without riparians);
        * ``environmental_reserve_mm3`` - ``natural_flow - estate``, the
          water withheld from the claimants;
        * ``terminal_reach`` - name of the terminal riparian (``None`` if
          the basin has none);
        * ``headroom_mm3`` - ``{riparian: Q_r - env_r}``, the cumulative
          consumption allowed above each outlet (negative when the
          requirement cannot be met even with zero use);
        * ``infeasible_reaches`` - riparians whose requirement exceeds the
          natural flow at their outlet.

    Raises
    ------
    ValueError
        On a negative or non-finite flow factor, inflow or environmental
        flow.

    References
    ----------
    Mianabadi, H., Mostert, E., Zarghami, M. & van de Giesen, N. (2014). A
        new bankruptcy method for conflict resolution in water resources
        allocation. *Journal of Environmental Management* 144, 152-159.
    Ansink & Weikard (2012); Madani (2010); Tharme, R. E. (2003). A global
        perspective on environmental flow assessment. *River Research and
        Applications* 19(5-6), 397-441.
    """
    _check_basin(basin)
    ff = _nonneg(flow_factor, "flow_factor")
    cumulative = natural_flows(basin, ff)  # validates headwater and local inflows
    if cumulative:
        natural = float(cumulative[basin.riparians[-1].name])
    else:
        natural = _nonneg(basin.headwater_inflow_mm3, "basin.headwater_inflow_mm3") * ff
    env = {
        r.name: _nonneg(r.demand.environmental, f"riparian {r.name!r} demand.environmental")
        for r in basin.riparians
    }
    headroom = {name: float(cumulative[name] - env[name]) for name in cumulative}
    if basin.riparians:
        terminal: Optional[str] = basin.riparians[-1].name
        env_terminal = env[terminal]
    else:
        terminal = None
        env_terminal = 0.0
    estate = max(natural - env_terminal, 0.0)
    return {
        "estate": float(estate),
        "natural_flow_mm3": float(natural),
        "environmental_flow_mm3": float(sum(env.values())),
        "terminal_environmental_flow_mm3": float(env_terminal),
        "environmental_reserve_mm3": float(natural - estate),
        "terminal_reach": terminal,
        "headroom_mm3": headroom,
        "infeasible_reaches": [name for name, h in headroom.items() if h < 0.0],
    }


def _claims_problem(
    basin: Basin,
    flow_factor: float,
    claims: Optional[Mapping[str, float]],
    estate: Optional[float],
) -> Tuple[Dict[str, float], float, Dict[str, Any]]:
    """Return ``(claims, estate, estate_info)`` for a basin year.

    ``estate_info`` is the :func:`bankruptcy_estate` result; ``estate`` is
    its ``"estate"`` unless an explicit non-negative ``estate`` is given.
    """
    names = basin.names()
    overrides = _per_riparian(claims, names, "claims")
    out_claims: Dict[str, float] = {}
    for r in basin.riparians:
        label = f"riparian {r.name!r}"
        if r.name in overrides and overrides[r.name] is not None:
            out_claims[r.name] = _nonneg(overrides[r.name], f"claims[{r.name!r}]")
        elif r.treaty_allocation_mm3 is not None:
            out_claims[r.name] = _nonneg(r.treaty_allocation_mm3, f"{label} treaty_allocation_mm3")
        else:
            out_claims[r.name] = sum(
                _nonneg(v, f"{label} demand.{s.value}") for s, v in r.demand.withdrawals().items()
            )
    info = bankruptcy_estate(basin, flow_factor)
    if estate is None:
        estate_v = info["estate"]
    else:
        estate_v = _nonneg(estate, "estate")
    return out_claims, float(estate_v), info


def negotiate(
    basin: Basin,
    flow_factor: float = 1.0,
    rule: str = "talmud",
    claims: Optional[Mapping[str, float]] = None,
    batna: Optional[Mapping[str, float]] = None,
    *,
    estate: Optional[float] = None,
    tol: float = 1e-6,
) -> Dict[str, Any]:
    """Frame a basin year as a claims problem and test a proposal against BATNAs.

    1. **Claims**: each riparian's treaty entitlement
       (``treaty_allocation_mm3``) or, failing that, its total withdrawal
       demand; ``claims`` overrides individual entries.
    2. **Estate**: the water the basin can consume without violating an
       in-stream requirement - ``natural flow * flow_factor`` minus the
       environmental flow that must still be in the river at the terminal
       outlet, floored at 0 (:func:`bankruptcy_estate`) - or ``estate`` if
       given.  Upstream requirements are *not* subtracted as well: the water
       they keep in the river is an in-stream flow, not a withdrawal, and
       stays available to the riparians below.
    3. **Proposal**: ``allocation.apply_rule(rule, estate, claims)`` - a
       bankruptcy-rule division of the estate (Mianabadi et al. 2014).
    4. **BATNA** (best alternative to a negotiated agreement, Fisher & Ury
       1981): each riparian's surface withdrawal under unilateral,
       upstream-priority use - :func:`wefnexus.water.route_basin` with every
       entitlement removed (Ansink & Weikard 2012); ``batna`` overrides
       individual entries.
    5. **ZOPA**: the proposal is acceptable to a riparian when its award is
       at least its BATNA withdrawal (within ``tol`` Mm3); a zone of possible
       agreement exists when every riparian accepts.

    The proposal is also routed through the basin as entitlements to report
    the withdrawals it would actually deliver.

    Note that the estate is a *consumptive* volume that ignores reservoir
    storage and return flows, whereas the awards are applied as withdrawal
    caps and the BATNA is a withdrawal: the physical routing re-uses return
    flows downstream, so in a basin whose withdrawals rely on them a
    riparian's unilateral BATNA withdrawal can exceed even its own claim
    (the terminal riparian of the example basin withdraws 16 300 Mm3/yr
    against a 16 000 Mm3/yr entitlement) and no ZOPA is reported; pass
    ``estate`` to negotiate over a different pie (e.g. a treaty's reference
    flow).

    Parameters
    ----------
    basin : Basin
        Basin (never mutated).
    flow_factor : float, optional
        Multiplier on natural flow (>= 0; e.g. 0.6 for a drought year).
    rule : str or callable, optional
        Sharing rule accepted by :func:`wefnexus.allocation.apply_rule`
        (default ``"talmud"``).
    claims : mapping, optional
        ``{riparian: claim}`` overrides (Mm3/yr).
    batna : mapping, optional
        ``{riparian: disagreement withdrawal}`` overrides (Mm3/yr).
    estate : float, optional
        Explicit estate (Mm3/yr) instead of the natural-flow estimate.
    tol : float, optional
        Absolute tolerance (Mm3/yr) in the acceptability test.

    Returns
    -------
    dict
        ``rule`` (canonical name), ``flow_factor``, ``natural_flow_mm3``,
        ``environmental_flow_mm3`` (sum of every reach's requirement,
        information only), ``terminal_environmental_flow_mm3`` (the
        requirement at the terminal outlet), ``environmental_reserve_mm3``
        (natural flow minus the basin's own estate - the water withheld
        from the claimants, reported even when ``estate`` is overridden),
        ``estate``, ``claims``,
        ``proposal``, ``total_awarded_mm3`` (``== min(estate, sum claims)``),
        ``batna``, ``acceptable`` (``{riparian: bool}``), ``zopa`` (bool),
        ``gini`` (of the awards), ``gini_satisfaction``, ``satisfaction``
        (award / claim), ``routed_withdrawals``, ``outflow_to_sea_mm3`` and
        ``env_flow_met_share`` of the routed proposal.

    Raises
    ------
    ValueError
        On an unknown rule, unknown riparian names or invalid numbers.

    References
    ----------
    Fisher & Ury (1981); Ansink & Weikard (2012); Madani (2010); Mianabadi
    et al. (2014); Aumann & Maschler (1985) (Talmud rule).
    """
    _check_basin(basin)
    abs_tol = _nonneg(tol, "tol")
    claims_d, estate_v, info = _claims_problem(basin, flow_factor, claims, estate)
    ff = _nonneg(flow_factor, "flow_factor")
    names = basin.names()

    awards_by_rule = compare_rules(estate_v, claims_d, [rule])
    rule_name, proposal = next(iter(awards_by_rule.items()))
    proposal = {name: float(proposal[name]) for name in names}

    unilateral = route_basin(basin, ff, entitlements={name: None for name in names})
    batna_d = {name: float(v) for name, v in unilateral.withdrawals_by_riparian().items()}
    for name, value in _per_riparian(batna, names, "batna").items():
        if value is not None:
            batna_d[name] = _nonneg(value, f"batna[{name!r}]")

    acceptable = {name: bool(proposal[name] >= batna_d[name] - abs_tol) for name in names}
    zopa = all(acceptable.values())
    sat = satisfaction(proposal, claims_d)
    routed = route_basin(basin, ff, entitlements=proposal)
    return {
        "rule": rule_name,
        "flow_factor": ff,
        "natural_flow_mm3": info["natural_flow_mm3"],
        "environmental_flow_mm3": info["environmental_flow_mm3"],
        "terminal_environmental_flow_mm3": info["terminal_environmental_flow_mm3"],
        "environmental_reserve_mm3": info["environmental_reserve_mm3"],
        "estate": estate_v,
        "claims": claims_d,
        "proposal": proposal,
        "total_awarded_mm3": float(sum(proposal.values())),
        "batna": batna_d,
        "acceptable": acceptable,
        "zopa": zopa,
        "gini": gini(proposal),
        "gini_satisfaction": gini(list(sat.values())),
        "satisfaction": sat,
        "routed_withdrawals": routed.withdrawals_by_riparian(),
        "outflow_to_sea_mm3": routed.outflow_to_sea_mm3,
        "env_flow_met_share": routed.env_flow_met_share(),
    }


def compare_allocation_rules(
    basin: Basin,
    flow_factor: float = 1.0,
    rules: Optional[Iterable[Any]] = None,
    *,
    claims: Optional[Mapping[str, float]] = None,
    estate: Optional[float] = None,
) -> Dict[str, Dict[str, Any]]:
    """Apply every sharing rule to the basin's claims problem and route each.

    Claims and estate are built as in :func:`negotiate`; each rule's awards
    are then used as withdrawal entitlements in
    :func:`wefnexus.water.route_basin` to see what the basin would actually
    deliver (upstream riparians can always take their award; downstream
    ones only if the water arrives).

    Parameters
    ----------
    basin : Basin
        Basin (never mutated).
    flow_factor : float, optional
        Multiplier on natural flow (>= 0).
    rules : iterable of str or callable, optional
        Rules to compare (default: every rule in
        :data:`wefnexus.allocation.RULES`).
    claims, estate
        As in :func:`negotiate`.

    Returns
    -------
    dict
        ``{rule: {"awards", "total_awarded_mm3", "gini", "gini_satisfaction",
        "satisfaction", "min_satisfaction", "withdrawals",
        "total_withdrawal_mm3", "supply_ratio", "outflow_to_sea_mm3",
        "env_flow_met_share"}}`` with ``"estate"`` and ``"claims"`` repeated
        in every row for convenience.

    References
    ----------
    Thomson (2003); Mianabadi et al. (2014); Madani (2010).
    """
    _check_basin(basin)
    claims_d, estate_v, _ = _claims_problem(basin, flow_factor, claims, estate)
    ff = _nonneg(flow_factor, "flow_factor")
    names = basin.names()
    awards_by_rule = compare_rules(estate_v, claims_d, rules)
    out: Dict[str, Dict[str, Any]] = {}
    for rule_name, awards in awards_by_rule.items():
        awards = {name: float(awards[name]) for name in names}
        sat = satisfaction(awards, claims_d)
        bal = route_basin(basin, ff, entitlements=awards)
        out[rule_name] = {
            "estate": estate_v,
            "claims": dict(claims_d),
            "awards": awards,
            "total_awarded_mm3": float(sum(awards.values())),
            "gini": gini(awards),
            "gini_satisfaction": gini(list(sat.values())),
            "satisfaction": sat,
            "min_satisfaction": float(min(sat.values())) if sat else 1.0,
            "withdrawals": bal.withdrawals_by_riparian(),
            "total_withdrawal_mm3": bal.total_withdrawal(),
            "supply_ratio": bal.supply_ratio(),
            "outflow_to_sea_mm3": bal.outflow_to_sea_mm3,
            "env_flow_met_share": bal.env_flow_met_share(),
        }
    return out
