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

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Dict, Iterator, List, Optional

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

    def scaled(self, factor: float, sectors: Optional[List[Sector]] = None) -> "WaterDemand":
        """Return a copy with the given sectors (default: all withdrawals)
        multiplied by ``factor``."""
        sectors = sectors or list(SECTOR_PRIORITY)
        kwargs = {}
        for s in sectors:
            kwargs[s.value] = getattr(self, s.value) * factor
        return replace(self, **kwargs)


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
        return Basin(
            name=self.name,
            riparians=[replace(r, demand=replace(r.demand), crops=[replace(c) for c in r.crops], energy=replace(r.energy)) for r in self.riparians],
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
