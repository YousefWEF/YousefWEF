"""Water-energy links of the WEF nexus (leaf module).

This module holds the *physical* conversions between water and energy that
the integrated nexus model relies on:

* **water for energy** – hydropower generation from turbined flow, thermal
  generation from installed capacity, and the consumptive cooling water that
  thermal generation requires;
* **energy for water** – the electricity needed to pump (lift) irrigation
  water, to desalinate seawater, and to treat drinking water and wastewater;
* **energy accounting** – a per-riparian energy balance (supply, demand,
  deficit, self-sufficiency, emissions) and an energy-security index.

Units
-----
Water volumes are in **Mm3/yr** (1 Mm3 = 1e6 m3), energy in **GWh/yr**
(1 GWh = 1e6 kWh = 3.6e12 J), heads and lifts in **m**, power in **MW**,
emissions in **t CO2** and energy intensities in **kWh/m3** or **m3/MWh** as
stated in each docstring.

Core formulas
-------------
Hydropower (e.g. Gulliver & Arndt 1991; Kumar et al. 2011):

    E [J] = rho * g * V * H * eta_turbine
    E [GWh] = rho * g * V_m3 * H_m * eta / 3.6e12

so that 1 Mm3 falling through 100 m at eta = 1 yields exactly 0.2725 GWh
(0.2725 kWh per m3).  Annual generation is capped by the installed capacity:
``capacity_mw * hours / 1000`` GWh.

Pumping (lifting) water is the mirror image, divided by the pump efficiency
(Plappally & Lienhard 2012):

    E [kWh] = rho * g * V_m3 * H_m / (eta_pump * 3.6e6)

i.e. 0.002725 kWh per m3 per metre of lift at 100 % efficiency.

Desalination, drinking-water treatment and wastewater treatment are modelled
with constant energy intensities in kWh/m3 (Plappally & Lienhard 2012;
Voutchkov 2018), and consumptive cooling water of thermal plants with a
constant water intensity in m3/MWh (Macknick et al. 2012).

References
----------
Bazilian, M. et al. (2011). Considering the energy, water and food nexus:
    towards an integrated modelling approach. *Energy Policy* 39, 7896-7906.
Gulliver, J. S. & Arndt, R. E. A. (1991). *Hydropower Engineering Handbook*.
    McGraw-Hill.
Hoff, H. (2011). *Understanding the Nexus*. Background paper for the Bonn 2011
    Conference: The Water, Energy and Food Security Nexus. SEI, Stockholm.
IPCC (2014). *Climate Change 2014: Mitigation of Climate Change*, Annex III
    (technology-specific cost and performance parameters).
Kruyt, B., van Vuuren, D. P., de Vries, H. J. M. & Groenenberg, H. (2009).
    Indicators for energy security. *Energy Policy* 37, 2166-2181.
Kumar, A. et al. (2011). Hydropower. In: *IPCC Special Report on Renewable
    Energy Sources and Climate Change Mitigation*, Ch. 5.
Macknick, J., Newmark, R., Heath, G. & Hallett, K. C. (2012). Operational
    water consumption and withdrawal factors for electricity generating
    technologies: a review of existing literature. *Environmental Research
    Letters* 7, 045802.
Plappally, A. K. & Lienhard V, J. H. (2012). Energy requirements for water
    production, treatment, end use, reclamation, and disposal. *Renewable and
    Sustainable Energy Reviews* 16, 4818-4848.
Voutchkov, N. (2018). Energy use for membrane seawater desalination - current
    status and trends. *Desalination* 431, 2-14.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

from wefnexus.models import Riparian, Sector

__all__ = [
    "HYDRO_G",
    "WATER_DENSITY",
    "J_PER_KWH",
    "J_PER_GWH",
    "KWH_PER_GWH",
    "HOURS_PER_YEAR",
    "M3_PER_MM3",
    "hydropower_gwh",
    "specific_hydropower_kwh_per_m3",
    "pumping_energy_gwh",
    "specific_pumping_energy_kwh_per_m3",
    "desalination_energy_gwh",
    "treatment_energy_gwh",
    "thermal_cooling_water_mm3",
    "thermal_generation_gwh",
    "energy_for_water",
    "water_for_energy",
    "energy_balance",
    "energy_security_index",
]

#: Gravitational acceleration (m/s2).
HYDRO_G: float = 9.81
#: Density of fresh water (kg/m3).
WATER_DENSITY: float = 1000.0
#: Joules per kilowatt-hour.
J_PER_KWH: float = 3.6e6
#: Joules per gigawatt-hour.
J_PER_GWH: float = 3.6e12
#: Kilowatt-hours per gigawatt-hour.
KWH_PER_GWH: float = 1.0e6
#: Hours in a (non-leap) year.
HOURS_PER_YEAR: float = 8760.0
#: Cubic metres per million cubic metres.
M3_PER_MM3: float = 1.0e6

#: Energy-for-water components returned by :func:`energy_for_water`.
_ENERGY_FOR_WATER_KEYS = ("pumping", "desalination", "treatment", "wastewater", "total")


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
def _as_float(name: str, value: Any) -> float:
    """Convert ``value`` to a finite float or raise ``ValueError``.

    Booleans are rejected because ``True``/``False`` passed as a physical
    quantity is almost always a bug.
    """
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, got boolean {value!r}")
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}") from None
    if math.isnan(out) or math.isinf(out):
        raise ValueError(f"{name} must be finite, got {out!r}")
    return out


def _non_negative(name: str, value: Any) -> float:
    out = _as_float(name, value)
    if out < 0.0:
        raise ValueError(f"{name} must be >= 0, got {out!r}")
    return out


def _fraction(name: str, value: Any, *, allow_zero: bool) -> float:
    """Validate a dimensionless share: ``(0, 1]`` or ``[0, 1]``."""
    out = _as_float(name, value)
    lo_ok = out >= 0.0 if allow_zero else out > 0.0
    if not lo_ok or out > 1.0:
        rng = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(f"{name} must be in {rng}, got {out!r}")
    return out


def _check_riparian(riparian: Any) -> Riparian:
    if not isinstance(riparian, Riparian):
        raise ValueError(
            f"riparian must be a wefnexus.models.Riparian, got {type(riparian).__name__}"
        )
    return riparian


def _normalise_withdrawals(withdrawals: Any) -> Dict[Sector, float]:
    """Return a fresh ``{Sector: Mm3}`` dict with every withdrawal sector present.

    Keys may be :class:`Sector` members or their string values.  Missing
    sectors default to 0.  ``Sector.ENVIRONMENT`` is an in-stream flow, not a
    withdrawal, so it is accepted but ignored.  Unknown keys and negative
    values raise ``ValueError``.
    """
    if withdrawals is None:
        withdrawals = {}
    if not isinstance(withdrawals, Mapping):
        raise ValueError(
            f"withdrawals must be a mapping of Sector -> Mm3, got {type(withdrawals).__name__}"
        )
    out: Dict[Sector, float] = {
        Sector.MUNICIPAL: 0.0,
        Sector.INDUSTRIAL: 0.0,
        Sector.ENERGY: 0.0,
        Sector.AGRICULTURAL: 0.0,
    }
    for key, value in withdrawals.items():
        try:
            sector = key if isinstance(key, Sector) else Sector(key)
        except ValueError:
            raise ValueError(
                f"unknown withdrawal sector {key!r}; expected one of "
                f"{[s.value for s in Sector]}"
            ) from None
        if sector is Sector.ENVIRONMENT:
            # Environmental flow stays in the river; it needs no energy here.
            _non_negative("withdrawals[environment]", value)
            continue
        out[sector] = _non_negative(f"withdrawals[{sector.value}]", value)
    return out


# --------------------------------------------------------------------------- #
# Water for energy
# --------------------------------------------------------------------------- #
def specific_hydropower_kwh_per_m3(head_m: float, efficiency: float = 0.9) -> float:
    """Electricity recovered per cubic metre of water falling through a head.

    Parameters
    ----------
    head_m : float
        Net hydraulic head (m), >= 0.
    efficiency : float, optional
        Overall turbine-generator efficiency, in (0, 1].  Default 0.9.

    Returns
    -------
    float
        Specific yield in **kWh per m3**:
        ``rho * g * H * eta / 3.6e6`` = ``0.002725 * H * eta``.

    Notes
    -----
    At 100 m head and unit efficiency the yield is 0.2725 kWh/m3, the
    textbook figure (Gulliver & Arndt 1991).
    """
    head = _non_negative("head_m", head_m)
    eta = _fraction("efficiency", efficiency, allow_zero=False)
    return WATER_DENSITY * HYDRO_G * head * eta / J_PER_KWH


def hydropower_gwh(
    volume_mm3: float,
    head_m: float,
    efficiency: float = 0.9,
    capacity_mw: Optional[float] = None,
    hours: float = 8760.0,
) -> float:
    """Annual hydropower generation from a turbined volume of water.

    Parameters
    ----------
    volume_mm3 : float
        Water passed through the turbines (Mm3 over the period), >= 0.
    head_m : float
        Net hydraulic head (m), >= 0.
    efficiency : float, optional
        Overall turbine-generator efficiency in (0, 1].  Default 0.9 (large
        modern Francis/Kaplan units reach 0.85-0.95).
    capacity_mw : float, optional
        Installed capacity (MW).  When given, generation is capped at
        ``capacity_mw * hours / 1000`` GWh (plant running flat out for
        ``hours``).  ``None`` (default) means no capacity limit.
    hours : float, optional
        Operating hours in the period used for the capacity cap, >= 0.
        Default 8760 (one year).

    Returns
    -------
    float
        Generation in **GWh**::

            E = rho * g * V * H * eta / 3.6e12
              = min(E, capacity_mw * hours / 1000)   if capacity_mw is given

    Raises
    ------
    ValueError
        On negative volume/head/capacity/hours or efficiency outside (0, 1].

    Notes
    -----
    1 Mm3 through 100 m at ``efficiency=1`` gives exactly 0.2725 GWh.
    The formula is the standard ``P = rho g Q H eta`` of hydropower
    engineering (Gulliver & Arndt 1991; Kumar et al. 2011, IPCC SRREN Ch. 5).
    Spillage, tailwater variation and part-load efficiency are ignored; use
    ``EnergySystem.turbined_fraction`` to express the share of outflow that
    actually reaches the turbines.

    Examples
    --------
    >>> round(hydropower_gwh(1.0, 100.0, efficiency=1.0), 6)
    0.2725
    >>> hydropower_gwh(1000.0, 100.0, efficiency=1.0, capacity_mw=10.0)
    87.6
    """
    volume = _non_negative("volume_mm3", volume_mm3)
    head = _non_negative("head_m", head_m)
    eta = _fraction("efficiency", efficiency, allow_zero=False)
    hrs = _non_negative("hours", hours)
    energy_gwh = WATER_DENSITY * HYDRO_G * volume * M3_PER_MM3 * head * eta / J_PER_GWH
    if capacity_mw is not None:
        cap_gwh = _non_negative("capacity_mw", capacity_mw) * hrs / 1000.0
        energy_gwh = min(energy_gwh, cap_gwh)
    return energy_gwh


def thermal_generation_gwh(
    capacity_mw: float, capacity_factor: float, hours: float = 8760.0
) -> float:
    """Annual generation of a thermal (or any dispatchable) plant.

    Parameters
    ----------
    capacity_mw : float
        Installed capacity (MW), >= 0.
    capacity_factor : float
        Average output as a share of capacity, in [0, 1].
    hours : float, optional
        Hours in the period, >= 0.  Default 8760 (one year).

    Returns
    -------
    float
        Generation in **GWh**: ``capacity_mw * capacity_factor * hours / 1000``.

    Notes
    -----
    Typical capacity factors: 0.5-0.85 for coal and combined-cycle gas,
    lower for peaking plants (IPCC 2014, Annex III).
    """
    cap = _non_negative("capacity_mw", capacity_mw)
    cf = _fraction("capacity_factor", capacity_factor, allow_zero=True)
    hrs = _non_negative("hours", hours)
    return cap * cf * hrs / 1000.0


def thermal_cooling_water_mm3(generation_gwh: float, intensity_m3_per_mwh: float) -> float:
    """Consumptive cooling-water use of thermal power generation.

    Parameters
    ----------
    generation_gwh : float
        Thermal generation (GWh), >= 0.
    intensity_m3_per_mwh : float
        Consumptive water intensity (m3 per MWh), >= 0.  Macknick et al.
        (2012) report roughly 0.4-1.0 m3/MWh for once-through cooling,
        1.5-3.0 m3/MWh for recirculating (wet-tower) cooling and near zero
        for dry cooling.

    Returns
    -------
    float
        Water consumed in **Mm3**: ``generation_gwh * 1000 * intensity / 1e6``.

    Examples
    --------
    >>> thermal_cooling_water_mm3(1000.0, 1.5)
    1.5
    """
    gen = _non_negative("generation_gwh", generation_gwh)
    intensity = _non_negative("intensity_m3_per_mwh", intensity_m3_per_mwh)
    return gen * 1000.0 * intensity / M3_PER_MM3


def water_for_energy(riparian: Riparian, thermal_generation_gwh: float) -> float:
    """Consumptive cooling water of a riparian's thermal generation.

    Parameters
    ----------
    riparian : Riparian
        Supplies ``energy.thermal_water_intensity_m3_per_mwh``.
    thermal_generation_gwh : float
        Thermal generation in the year (GWh), >= 0.

    Returns
    -------
    float
        Consumptive cooling water in **Mm3/yr** (see
        :func:`thermal_cooling_water_mm3`).  The riparian is not modified.
    """
    rip = _check_riparian(riparian)
    return thermal_cooling_water_mm3(
        thermal_generation_gwh, rip.energy.thermal_water_intensity_m3_per_mwh
    )


# --------------------------------------------------------------------------- #
# Energy for water
# --------------------------------------------------------------------------- #
def specific_pumping_energy_kwh_per_m3(lift_m: float, efficiency: float = 0.6) -> float:
    """Electricity needed to lift one cubic metre of water.

    Parameters
    ----------
    lift_m : float
        Total dynamic head / lift (m), >= 0.
    efficiency : float, optional
        Wire-to-water pump efficiency in (0, 1].  Default 0.6.

    Returns
    -------
    float
        **kWh per m3**: ``rho * g * H / (eta * 3.6e6)`` = ``0.002725 * H / eta``.

    Notes
    -----
    0.002725 kWh/m3 per metre at 100 % efficiency is the figure quoted by
    Plappally & Lienhard (2012); real groundwater pumping at 40-60 %
    wire-to-water efficiency therefore costs about 0.005 kWh/m3 per metre.
    """
    lift = _non_negative("lift_m", lift_m)
    eta = _fraction("efficiency", efficiency, allow_zero=False)
    return WATER_DENSITY * HYDRO_G * lift / (eta * J_PER_KWH)


def pumping_energy_gwh(volume_mm3: float, lift_m: float, efficiency: float = 0.6) -> float:
    """Electricity for pumping (lifting) a volume of water.

    Parameters
    ----------
    volume_mm3 : float
        Volume pumped (Mm3), >= 0.
    lift_m : float
        Total dynamic head / lift (m), >= 0.
    efficiency : float, optional
        Wire-to-water pump efficiency in (0, 1].  Default 0.6.

    Returns
    -------
    float
        Energy in **GWh**::

            E_kWh = rho * g * V_m3 * H / (eta * 3.6e6)
            E_GWh = E_kWh / 1e6

    Raises
    ------
    ValueError
        On negative volume/lift or efficiency outside (0, 1].

    Notes
    -----
    Mirror image of :func:`hydropower_gwh`: pumping ``V`` up ``H`` and
    turbining it back down returns ``eta_pump * eta_turbine`` of the input,
    the round-trip efficiency of pumped storage.  Reference: Plappally &
    Lienhard (2012).

    Examples
    --------
    >>> round(pumping_energy_gwh(1.0, 30.0, efficiency=0.6), 6)
    0.13625
    """
    volume = _non_negative("volume_mm3", volume_mm3)
    lift = _non_negative("lift_m", lift_m)
    eta = _fraction("efficiency", efficiency, allow_zero=False)
    energy_kwh = WATER_DENSITY * HYDRO_G * volume * M3_PER_MM3 * lift / (eta * J_PER_KWH)
    return energy_kwh / KWH_PER_GWH


def desalination_energy_gwh(volume_mm3: float, kwh_per_m3: float = 3.5) -> float:
    """Electricity for desalinating a volume of seawater or brackish water.

    Parameters
    ----------
    volume_mm3 : float
        Desalinated water produced (Mm3), >= 0.
    kwh_per_m3 : float, optional
        Specific energy consumption (kWh/m3), >= 0.  Default 3.5, typical of
        modern seawater reverse osmosis (3-4 kWh/m3, Voutchkov 2018);
        brackish RO is ~1 kWh/m3, thermal MSF/MED 10-20 kWh-equivalent/m3.

    Returns
    -------
    float
        Energy in **GWh**: ``volume_mm3 * 1e6 * kwh_per_m3 / 1e6`` =
        ``volume_mm3 * kwh_per_m3``.

    Examples
    --------
    >>> desalination_energy_gwh(300.0, 3.5)
    1050.0
    """
    volume = _non_negative("volume_mm3", volume_mm3)
    intensity = _non_negative("kwh_per_m3", kwh_per_m3)
    return volume * M3_PER_MM3 * intensity / KWH_PER_GWH


def treatment_energy_gwh(volume_mm3: float, kwh_per_m3: float) -> float:
    """Electricity for treating a volume of water at a fixed intensity.

    Used for drinking-water treatment (typically 0.2-0.6 kWh/m3 including
    distribution) and wastewater treatment (0.3-0.8 kWh/m3 for activated
    sludge), see Plappally & Lienhard (2012).

    Parameters
    ----------
    volume_mm3 : float
        Volume treated (Mm3), >= 0.
    kwh_per_m3 : float
        Specific energy consumption (kWh/m3), >= 0.

    Returns
    -------
    float
        Energy in **GWh**: ``volume_mm3 * kwh_per_m3``.
    """
    volume = _non_negative("volume_mm3", volume_mm3)
    intensity = _non_negative("kwh_per_m3", kwh_per_m3)
    return volume * M3_PER_MM3 * intensity / KWH_PER_GWH


def energy_for_water(
    riparian: Riparian,
    withdrawals: Dict[Sector, float],
    desalinated_mm3: float,
    pumped_fraction: Optional[float] = None,
) -> Dict[str, float]:
    """Electricity embedded in a riparian's water supply for one year.

    Parameters
    ----------
    riparian : Riparian
        Supplies the energy intensities from ``riparian.energy``
        (``pumping_lift_m``, ``pumping_efficiency``, ``pumped_fraction``,
        ``desal_energy_kwh_m3``, ``treatment_energy_kwh_m3``,
        ``wastewater_energy_kwh_m3``) and the sector consumption fractions
        from ``riparian.demand.consumption_fraction``.
    withdrawals : dict
        Actual withdrawals per sector (Mm3/yr), e.g. ``ReachResult.withdrawals``
        or ``WaterDemand.withdrawals()``.  Keys may be :class:`Sector`
        members or their string values; missing sectors count as 0 and
        ``Sector.ENVIRONMENT`` (an in-stream flow) is ignored.
    desalinated_mm3 : float
        Desalinated water produced (Mm3/yr), >= 0.
    pumped_fraction : float, optional
        Share of agricultural withdrawals that must be pumped/lifted, in
        [0, 1].  Defaults to ``riparian.energy.pumped_fraction``.

    Returns
    -------
    dict
        GWh/yr by component, with keys

        * ``"pumping"``     – ``pumping_energy_gwh(agri * pumped_fraction, lift, eta)``
        * ``"desalination"`` – ``desalination_energy_gwh(desalinated, desal_kwh_m3)``
        * ``"treatment"``   – drinking/process-water treatment of municipal +
          industrial withdrawals at ``treatment_energy_kwh_m3``
        * ``"wastewater"``  – treatment of the return flow
          ``municipal * (1 - f_mun) + industrial * (1 - f_ind)`` at
          ``wastewater_energy_kwh_m3``, where ``f_s`` is the sector's
          consumption fraction (missing fraction -> 0, i.e. fully returned)
        * ``"total"``       – sum of the four components.

    Raises
    ------
    ValueError
        On a non-``Riparian`` argument, unknown sector keys, negative volumes
        or a ``pumped_fraction`` outside [0, 1].

    Notes
    -----
    Energy for agricultural pumping is the dominant energy-for-water term in
    groundwater-dependent basins; desalination dominates in coastal arid
    states (Plappally & Lienhard 2012; Hoff 2011).  Inputs are not modified.
    """
    rip = _check_riparian(riparian)
    w = _normalise_withdrawals(withdrawals)
    desal = _non_negative("desalinated_mm3", desalinated_mm3)
    pf_source = rip.energy.pumped_fraction if pumped_fraction is None else pumped_fraction
    pf = _fraction("pumped_fraction", pf_source, allow_zero=True)

    es = rip.energy
    cf = rip.demand.consumption_fraction
    f_mun = _fraction("consumption_fraction[municipal]", cf.get(Sector.MUNICIPAL, 0.0), allow_zero=True)
    f_ind = _fraction("consumption_fraction[industrial]", cf.get(Sector.INDUSTRIAL, 0.0), allow_zero=True)

    municipal = w[Sector.MUNICIPAL]
    industrial = w[Sector.INDUSTRIAL]
    agricultural = w[Sector.AGRICULTURAL]

    pumping = pumping_energy_gwh(agricultural * pf, es.pumping_lift_m, es.pumping_efficiency)
    desalination = desalination_energy_gwh(desal, es.desal_energy_kwh_m3)
    treatment = treatment_energy_gwh(municipal + industrial, es.treatment_energy_kwh_m3)
    return_flow = municipal * (1.0 - f_mun) + industrial * (1.0 - f_ind)
    wastewater = treatment_energy_gwh(return_flow, es.wastewater_energy_kwh_m3)

    out = {
        "pumping": pumping,
        "desalination": desalination,
        "treatment": treatment,
        "wastewater": wastewater,
    }
    out["total"] = pumping + desalination + treatment + wastewater
    return out


# --------------------------------------------------------------------------- #
# Energy accounting
# --------------------------------------------------------------------------- #
def energy_balance(
    riparian: Riparian,
    hydropower_gwh: float,
    thermal_gwh: float,
    other_renewable_gwh: float,
    demand_gwh: float,
) -> Dict[str, float]:
    """Annual electricity balance of a riparian.

    Parameters
    ----------
    riparian : Riparian
        Supplies ``energy.grid_emission_factor_t_per_gwh`` (t CO2 per GWh of
        thermal generation).
    hydropower_gwh, thermal_gwh, other_renewable_gwh : float
        Generation by source (GWh/yr), each >= 0.
    demand_gwh : float
        Electricity demand (GWh/yr), >= 0.

    Returns
    -------
    dict
        * ``"supply"`` – hydropower + thermal + other renewables (GWh)
        * ``"demand"`` – demand (GWh)
        * ``"deficit"`` – ``max(demand - supply, 0)`` (GWh)
        * ``"self_sufficiency"`` – raw ratio ``supply / demand`` (not capped,
          so > 1 means an exportable surplus); 1.0 when demand is 0
        * ``"emissions_t"`` – ``thermal_gwh * grid_emission_factor_t_per_gwh``
          (t CO2/yr); hydropower and other renewables are counted as zero
          operational emissions.

    Notes
    -----
    Emission factors of 400-500 t/GWh correspond to gas/coal mixes
    (IPCC 2014, Annex III: ~490 t/GWh gas CC, ~820 t/GWh coal, lifecycle).
    """
    rip = _check_riparian(riparian)
    hydro = _non_negative("hydropower_gwh", hydropower_gwh)
    thermal = _non_negative("thermal_gwh", thermal_gwh)
    other = _non_negative("other_renewable_gwh", other_renewable_gwh)
    demand = _non_negative("demand_gwh", demand_gwh)
    factor = _non_negative(
        "riparian.energy.grid_emission_factor_t_per_gwh", rip.energy.grid_emission_factor_t_per_gwh
    )

    supply = hydro + thermal + other
    deficit = max(demand - supply, 0.0)
    self_sufficiency = supply / demand if demand > 0.0 else 1.0
    return {
        "supply": supply,
        "demand": demand,
        "deficit": deficit,
        "self_sufficiency": self_sufficiency,
        "emissions_t": thermal * factor,
    }


def energy_security_index(supply_gwh: float, demand_gwh: float) -> float:
    """Share of electricity demand that can be met, capped at 1.

    Parameters
    ----------
    supply_gwh : float
        Available supply (GWh/yr), >= 0.
    demand_gwh : float
        Demand (GWh/yr), >= 0.

    Returns
    -------
    float
        ``min(supply / demand, 1)``; 1.0 when demand is 0.

    Notes
    -----
    A simple availability indicator in the sense of Kruyt et al. (2009);
    the composite index with renewable share and emissions lives in
    ``wefnexus.sustainability.energy_security_index``.
    """
    supply = _non_negative("supply_gwh", supply_gwh)
    demand = _non_negative("demand_gwh", demand_gwh)
    if demand <= 0.0:
        return 1.0
    return min(supply / demand, 1.0)
