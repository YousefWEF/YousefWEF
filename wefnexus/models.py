"""Core data models for the WEF nexus and water diplomacy toolkit.

Units used throughout the package
---------------------------------
* Water volumes: **Mm3** (million cubic metres) per year unless stated otherwise.
* Energy: **GWh** per year (helper functions may use kWh and document it).
* Food: tonnes (t) and kilocalories (kcal).
* Areas: hectares (ha).  Depths: mm.  Heads/lifts: metres.
* Money: USD.

All dataclasses are plain containers.  Behaviour lives in the functional
modules (``water``, ``energy``, ``food``, ``allocation``, ``diplomacy``,
``sustainability``, ``nexus``, ``optimize``, ``scenarios``).
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field, fields, replace
from enum import Enum
from typing import Dict, Iterable, Iterator, List, Optional, Union

__all__ = [
    "Sector",
    "WaterDemand",
    "Crop",
    "EnergySystem",
    "Riparian",
    "Basin",
    "Scenario",
    "SECTOR_PRIORITY",
    "DEFAULT_CONSUMPTION_FRACTION",
    "DEFAULT_VALUE_USD_PER_M3",
]


class Sector(str, Enum):
    """Water-using sectors.  ``ENVIRONMENT`` is an in-stream requirement,
    not a withdrawal."""

    MUNICIPAL = "municipal"
    INDUSTRIAL = "industrial"
    AGRICULTURAL = "agricultural"
    ENERGY = "energy"
    ENVIRONMENT = "environment"


#: Order in which withdrawal demands are served when water is short.
#: Environmental flow is treated as a constraint on availability, not as a
#: withdrawal, so it is not in this list.
SECTOR_PRIORITY: List[Sector] = [
    Sector.MUNICIPAL,
    Sector.INDUSTRIAL,
    Sector.ENERGY,
    Sector.AGRICULTURAL,
]

#: Share of each sector's withdrawal that is consumed (evaporated / embedded
#: in products) rather than returned to the river.  Typical literature values.
DEFAULT_CONSUMPTION_FRACTION: Dict[Sector, float] = {
    Sector.MUNICIPAL: 0.20,
    Sector.INDUSTRIAL: 0.10,
    Sector.ENERGY: 0.03,
    Sector.AGRICULTURAL: 0.60,
}

#: Illustrative economic value of water by sector (USD per m3).
DEFAULT_VALUE_USD_PER_M3: Dict[Sector, float] = {
    Sector.MUNICIPAL: 1.50,
    Sector.INDUSTRIAL: 0.80,
    Sector.ENERGY: 0.40,
    Sector.AGRICULTURAL: 0.10,
}

#: :class:`WaterDemand` attribute holding each sector's annual volume.  The
#: enum value and the field name coincide for every withdrawal sector but
#: differ for the in-stream requirement (``"environment"`` vs
#: ``environmental``), so always go through this map rather than ``s.value``.
_SECTOR_FIELD: Dict[Sector, str] = {
    Sector.MUNICIPAL: "municipal",
    Sector.INDUSTRIAL: "industrial",
    Sector.AGRICULTURAL: "agricultural",
    Sector.ENERGY: "energy",
    Sector.ENVIRONMENT: "environmental",
}


@dataclass
class WaterDemand:
    """Annual water demands of one riparian, in Mm3/yr.

    ``environmental`` is the minimum in-stream flow that must be left in the
    river at the riparian's outlet; it is *not* a withdrawal.
    """

    municipal: float = 0.0
    industrial: float = 0.0
    agricultural: float = 0.0
    energy: float = 0.0
    environmental: float = 0.0
    consumption_fraction: Dict[Sector, float] = field(
        default_factory=lambda: dict(DEFAULT_CONSUMPTION_FRACTION)
    )
    value_usd_per_m3: Dict[Sector, float] = field(
        default_factory=lambda: dict(DEFAULT_VALUE_USD_PER_M3)
    )

    def withdrawals(self) -> Dict[Sector, float]:
        """Withdrawal demand per sector (excludes environmental flow)."""
        return {
            Sector.MUNICIPAL: self.municipal,
            Sector.INDUSTRIAL: self.industrial,
            Sector.ENERGY: self.energy,
            Sector.AGRICULTURAL: self.agricultural,
        }

    def total_withdrawal(self) -> float:
        return sum(self.withdrawals().values())

    def consumptive_demand(self) -> float:
        """Total consumptive (non-returned) demand in Mm3/yr."""
        return sum(
            v * self.consumption_fraction.get(s, 0.0)
            for s, v in self.withdrawals().items()
        )

    def copy(self) -> "WaterDemand":
        """Return an independent copy of this demand.

        The per-sector ``consumption_fraction`` and ``value_usd_per_m3``
        dictionaries are re-created, so editing them on the copy never
        touches the original (``dataclasses.replace`` alone would share
        them).

        Returns
        -------
        WaterDemand
            A new object comparing equal to ``self`` (volumes in Mm3/yr).
        """
        return replace(
            self,
            consumption_fraction=dict(self.consumption_fraction),
            value_usd_per_m3=dict(self.value_usd_per_m3),
        )

    def scaled(
        self,
        factor: float,
        sectors: Optional[Iterable[Union[Sector, str]]] = None,
    ) -> "WaterDemand":
        """Return a copy with the given sectors multiplied by ``factor``.

        Used for demand-growth what-ifs (e.g. municipal and industrial
        demand growing with population and GDP, Wada et al. 2016).

        Parameters
        ----------
        factor : float
            Dimensionless multiplier applied to each selected sector's
            annual volume (Mm3/yr); must be finite and ``>= 0``.
        sectors : iterable of Sector or str, optional
            Sectors to scale, as :class:`Sector` members or their string
            values.  ``None`` (default) scales every withdrawal sector in
            :data:`SECTOR_PRIORITY` and leaves the in-stream
            ``environmental`` requirement untouched.  ``Sector.ENVIRONMENT``
            may be passed explicitly to scale the environmental flow (its
            enum value ``"environment"`` maps onto the ``environmental``
            field).  An empty iterable scales nothing.

        Returns
        -------
        WaterDemand
            A new object with fresh sector dictionaries (see :meth:`copy`);
            ``self`` is not modified.

        Raises
        ------
        ValueError
            If ``factor`` is not a finite non-negative number or a sector
            is not a known :class:`Sector`.

        References
        ----------
        Wada, Y. et al. (2016). Modeling global water use for the 21st
        century: the Water Futures and Solutions (WFaS) initiative and its
        approaches. *Geoscientific Model Development*, 9, 175-222.
        """
        if (
            isinstance(factor, bool)
            or not isinstance(factor, numbers.Real)
            or not math.isfinite(factor)
            or factor < 0
        ):
            raise ValueError(
                f"factor must be a finite number >= 0, got {factor!r}"
            )
        if sectors is None:
            chosen: List[Sector] = list(SECTOR_PRIORITY)
        else:
            chosen = []
            for s in sectors:
                try:
                    chosen.append(s if isinstance(s, Sector) else Sector(s))
                except ValueError:
                    raise ValueError(
                        f"unknown sector {s!r}; expected one of "
                        f"{[m.value for m in Sector]}"
                    ) from None
        kwargs: Dict[str, float] = {}
        for s in chosen:
            name = _SECTOR_FIELD[s]
            kwargs[name] = getattr(self, name) * factor
        return replace(self.copy(), **kwargs)


@dataclass
class Crop:
    """A crop grown under irrigation in a riparian's territory.

    * ``kc`` – seasonal average FAO-56 crop coefficient.
    * ``season_days`` – growing season length.
    * ``ky`` – FAO-33 yield response factor (1 - Ya/Ym = ky (1 - ETa/ETm)).
    """

    name: str
    area_ha: float
    kc: float
    season_days: int
    yield_max_t_ha: float
    ky: float
    kcal_per_kg: float
    price_usd_t: float = 0.0


@dataclass
class EnergySystem:
    """Energy assets and demands of a riparian."""

    demand_gwh: float = 0.0
    hydropower_capacity_mw: float = 0.0
    hydropower_head_m: float = 0.0
    turbine_efficiency: float = 0.90
    #: share of annual outflow that passes through turbines (0..1)
    turbined_fraction: float = 1.0
    thermal_capacity_mw: float = 0.0
    thermal_capacity_factor: float = 0.5
    #: consumptive cooling water use per MWh of thermal generation
    thermal_water_intensity_m3_per_mwh: float = 1.5
    renewable_share: float = 0.0
    pumping_lift_m: float = 30.0
    pumping_efficiency: float = 0.60
    #: share of agricultural withdrawals that must be pumped (groundwater / lift)
    pumped_fraction: float = 0.3
    desalination_capacity_mm3: float = 0.0
    desal_energy_kwh_m3: float = 3.5
    treatment_energy_kwh_m3: float = 0.4
    wastewater_energy_kwh_m3: float = 0.6
    grid_emission_factor_t_per_gwh: float = 400.0


@dataclass
class Riparian:
    """A riparian state (or sub-basin unit) of a shared river basin.

    ``local_inflow_mm3`` is the natural runoff generated within this
    riparian's territory (tributaries, local rainfall-runoff), which enters
    the main stem in addition to whatever arrives from upstream.
    """

    name: str
    population: float
    gdp_usd: float
    local_inflow_mm3: float
    demand: WaterDemand
    groundwater_recharge_mm3: float = 0.0
    groundwater_abstraction_mm3: float = 0.0
    crops: List[Crop] = field(default_factory=list)
    energy: EnergySystem = field(default_factory=EnergySystem)
    reservoir_capacity_mm3: float = 0.0
    reservoir_storage_mm3: float = 0.0
    reservoir_evaporation_fraction: float = 0.05
    irrigation_efficiency: float = 0.50
    et0_mm_day: float = 5.0
    effective_rainfall_mm: float = 0.0
    food_demand_kcal_per_capita_day: float = 2500.0
    #: Zeitoun & Warner hydro-hegemony pillars, each on 0..1
    material_power: float = 0.5
    bargaining_power: float = 0.5
    ideational_power: float = 0.5
    #: allocation guaranteed by treaty (None = no treaty entitlement)
    treaty_allocation_mm3: Optional[float] = None
    #: position in the basin, 0 = most upstream; set by :class:`Basin`
    position: int = 0

    def irrigated_area_ha(self) -> float:
        return sum(c.area_ha for c in self.crops)

    def food_demand_kcal(self) -> float:
        """Annual food energy demand in kcal."""
        return self.population * self.food_demand_kcal_per_capita_day * 365.0

    def copy(self, **changes) -> "Riparian":
        """Return an independent copy, optionally with some fields changed.

        Unlike a bare ``dataclasses.replace``, the nested containers are
        re-created: ``demand`` via :meth:`WaterDemand.copy` (fresh sector
        dictionaries), ``crops`` as a new list of new :class:`Crop`
        objects and ``energy`` as a new :class:`EnergySystem`.  Editing
        any of them on the copy therefore never alters the original, as
        the no-mutation contract in ``docs/ARCHITECTURE.md`` requires.

        Parameters
        ----------
        **changes
            Field values overriding the copied ones (e.g.
            ``material_power=1.0`` or ``crops=[...]``).  A nested field
            given here is used as passed, not copied.

        Returns
        -------
        Riparian
            A new object; with no ``changes`` it compares equal to ``self``.

        Raises
        ------
        ValueError
            If ``changes`` names a field :class:`Riparian` does not have.
        """
        known = {f.name for f in fields(self)}
        unknown = sorted(set(changes) - known)
        if unknown:
            raise ValueError(
                f"unknown Riparian field(s) {unknown}; expected a subset of "
                f"{sorted(known)}"
            )
        if "demand" not in changes:
            changes["demand"] = self.demand.copy()
        if "crops" not in changes:
            changes["crops"] = [replace(c) for c in self.crops]
        if "energy" not in changes:
            changes["energy"] = replace(self.energy)
        return replace(self, **changes)


@dataclass
class Basin:
    """A transboundary river basin.

    ``riparians`` must be ordered from upstream to downstream along the
    main stem.  ``headwater_inflow_mm3`` is any flow entering above the
    first riparian (e.g. from glaciers or an outside catchment).
    """

    name: str
    riparians: List[Riparian]
    headwater_inflow_mm3: float = 0.0
    #: coefficient of variation of annual natural flow (for stochastic runs)
    climate_cv: float = 0.20

    def __post_init__(self) -> None:
        names = [r.name for r in self.riparians]
        if len(set(names)) != len(names):
            raise ValueError("riparian names must be unique")
        for i, r in enumerate(self.riparians):
            r.position = i

    def __iter__(self) -> Iterator[Riparian]:
        return iter(self.riparians)

    def __len__(self) -> int:
        return len(self.riparians)

    def names(self) -> List[str]:
        return [r.name for r in self.riparians]

    def riparian(self, name: str) -> Riparian:
        for r in self.riparians:
            if r.name == name:
                return r
        raise KeyError(f"unknown riparian {name!r} in basin {self.name!r}")

    def upstream_of(self, name: str) -> List[Riparian]:
        pos = self.riparian(name).position
        return [r for r in self.riparians if r.position < pos]

    def downstream_of(self, name: str) -> List[Riparian]:
        pos = self.riparian(name).position
        return [r for r in self.riparians if r.position > pos]

    def total_natural_flow(self) -> float:
        """Mean annual natural surface flow of the whole basin (Mm3/yr)."""
        return self.headwater_inflow_mm3 + sum(r.local_inflow_mm3 for r in self.riparians)

    def total_renewable_water(self) -> float:
        """Surface flow plus groundwater recharge (Mm3/yr)."""
        return self.total_natural_flow() + sum(
            r.groundwater_recharge_mm3 for r in self.riparians
        )

    def total_population(self) -> float:
        return sum(r.population for r in self.riparians)

    def copy(self) -> "Basin":
        """Return an independent deep copy of the basin.

        Every riparian is copied with :meth:`Riparian.copy`, so demands
        (including their ``consumption_fraction`` and ``value_usd_per_m3``
        dictionaries), crops and energy systems of the copy are separate
        objects.  This is the copy that modules use to obtain a modifiable
        basin without mutating the caller's object; scenario runs rely on
        it for isolation.

        Returns
        -------
        Basin
            A new basin comparing equal to ``self`` with riparian
            positions re-assigned upstream to downstream.
        """
        return Basin(
            name=self.name,
            riparians=[r.copy() for r in self.riparians],
            headwater_inflow_mm3=self.headwater_inflow_mm3,
            climate_cv=self.climate_cv,
        )


@dataclass
class Scenario:
    """Drivers applied over a multi-year simulation.

    Trends are linear from the start year to the end year unless stated.
    ``allocation_rule`` selects how the basin's water is shared between
    riparians when ``cooperation`` is True.  When ``cooperation`` is False
    the upstream riparian withdraws first ("upstream priority").
    """

    name: str = "baseline"
    start_year: int = 2025
    years: int = 25
    #: percentage change in natural flow reached by the final year (e.g. -15)
    flow_change_pct_by_end: float = 0.0
    population_growth_rate: float = 0.0
    gdp_growth_rate: float = 0.0
    #: municipal + industrial demand growth per year (fraction)
    demand_growth_rate: float = 0.0
    energy_demand_growth_rate: float = 0.0
    irrigated_area_change_pct_by_end: float = 0.0
    irrigation_efficiency_target: Optional[float] = None
    renewable_share_target: Optional[float] = None
    cooperation: bool = True
    #: one of "treaty", "proportional", "cea", "cel", "talmud",
    #: "upstream_priority", "equal"
    allocation_rule: str = "treaty"
    stochastic: bool = False
    seed: int = 42
    #: year offsets (0-based) in which a drought occurs
    drought_years: List[int] = field(default_factory=list)
    #: fractional reduction of natural flow in drought years (0..1)
    drought_severity: float = 0.4
    description: str = ""

    def year(self, i: int) -> int:
        return self.start_year + i

    def progress(self, i: int) -> float:
        """Fraction of the horizon elapsed at year index ``i`` (0..1)."""
        if self.years <= 1:
            return 1.0
        return min(max(i / (self.years - 1), 0.0), 1.0)

    def flow_factor(self, i: int) -> float:
        """Deterministic multiplier on natural flow in year index ``i``."""
        trend = 1.0 + (self.flow_change_pct_by_end / 100.0) * self.progress(i)
        if i in self.drought_years:
            trend *= 1.0 - self.drought_severity
        return max(trend, 0.0)

    def population_factor(self, i: int) -> float:
        return (1.0 + self.population_growth_rate) ** i

    def gdp_factor(self, i: int) -> float:
        return (1.0 + self.gdp_growth_rate) ** i

    def demand_factor(self, i: int) -> float:
        return (1.0 + self.demand_growth_rate) ** i

    def energy_demand_factor(self, i: int) -> float:
        return (1.0 + self.energy_demand_growth_rate) ** i

    def irrigated_area_factor(self, i: int) -> float:
        return 1.0 + (self.irrigated_area_change_pct_by_end / 100.0) * self.progress(i)

    def irrigation_efficiency(self, base: float, i: int) -> float:
        if self.irrigation_efficiency_target is None:
            return base
        return base + (self.irrigation_efficiency_target - base) * self.progress(i)

    def renewable_share(self, base: float, i: int) -> float:
        if self.renewable_share_target is None:
            return base
        return base + (self.renewable_share_target - base) * self.progress(i)
