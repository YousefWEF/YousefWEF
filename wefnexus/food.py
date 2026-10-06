"""Water-food links of the WEF nexus (leaf module).

This module holds the *physical* conversions between water and food that the
integrated nexus model relies on:

* **crop water requirements** – FAO-56 crop evapotranspiration, net and
  gross irrigation requirement of a crop and of a whole riparian;
* **yield response to water** – the FAO-33 (Doorenbos & Kassam 1979) linear
  yield-response function, and the resulting production, calories and value;
* **food accounting** – distribution of a riparian's agricultural water
  across its crops, food self-sufficiency and security indices, virtual
  water content / imports and the farm energy of irrigated agriculture.

Units
-----
Water volumes are in **Mm3/yr** (1 Mm3 = 1e6 m3), water depths in **mm**,
reference evapotranspiration in **mm/day**, areas in **ha**, production in
**t** (metric tonnes), food energy in **kcal**, money in **USD**, farm energy
in **GWh** (1 GWh = 1e6 kWh).  1 mm of water over 1 ha is 10 m3, so a gross
depth ``d`` mm over ``A`` ha is ``d * A * 10 / 1e6`` Mm3.

Core formulas
-------------
Crop evapotranspiration (FAO-56, Allen et al. 1998), seasonal total with a
seasonal-average crop coefficient::

    ETc [mm] = ET0 [mm/day] * Kc * season_days

Net and gross irrigation requirement (FAO-24 / FAO-56; Brouwer et al. 1989)::

    IRn [mm] = max(ETc - Pe, 0)            Pe = effective rainfall
    IRg [mm] = IRn / efficiency            efficiency in (0, 1]
    IR  [Mm3] = IRg * area_ha * 10 / 1e6

Yield response to water (FAO-33, Doorenbos & Kassam 1979)::

    1 - Ya/Ym = ky * (1 - ETa/ETm)
    Ya = Ym * (1 - ky * (1 - ETa/ETm)),  clipped to [0, Ym]

where ``ky`` is the seasonal crop yield-response factor (0.7-1.3 for most
crops) and ``ETa/ETm`` the relative seasonal crop evapotranspiration.  The
FAO-33 total-season table (Doorenbos & Kassam 1979, Table 24) gives
winter wheat 1.00, spring wheat 1.15, maize 1.25, cotton 0.85, groundnut
0.70, potato 1.10, sorghum 0.90, soybean 0.85, sugarbeet 0.7-1.1, sugarcane
1.20; it does **not** tabulate a seasonal ``ky`` for flooded (paddy) rice,
whose values of about 1.0-1.2 come from later compilations (Steduto et al.
2012, FAO-66).  The example basin uses illustrative values in that range
(wheat 1.05, between the FAO-33 winter and spring values; rice 1.20).

``ETa`` is the water the crop actually evaporates: the effective rainfall
plus the **net** irrigation water that reaches the root zone, so with gross
irrigation ``supplied`` delivered at ``efficiency`` over ``area_ha``::

    ETm     = ETc
    ETa     = min(Pe + efficiency * supplied_mm, ETc)
    ETa/ETm = (Pe + efficiency * supplied_mm) / ETc,  capped at 1
            = 1 - (1 - supplied / required) * IRn / ETc      (required > 0)

The second form uses the gross irrigation supply ratio ``supplied /
required`` and the irrigated share of crop ET ``IRn / ETc``; the two are
identical whenever ``required`` is the gross requirement above.  A crop
whose ET is covered by rainfall (``ETc <= Pe``) has ``ETa/ETm = 1`` whatever
the irrigation supply.

Virtual water (Allan 1998; Hoekstra & Chapagain 2008; Mekonnen & Hoekstra
2011)::

    VWC [m3/t] = water_m3 / production_t
    import [Mm3] = kcal_deficit / (kcal_per_kg * 1000) * m3_per_t / 1e6

References
----------
Allan, J. A. (1998). Virtual water: a strategic resource. Global solutions to
    regional deficits. *Ground Water* 36, 545-546.
Allen, R. G., Pereira, L. S., Raes, D. & Smith, M. (1998). *Crop
    evapotranspiration: guidelines for computing crop water requirements*.
    FAO Irrigation and Drainage Paper 56, FAO, Rome.
Brouwer, C., Prins, K. & Heibloem, M. (1989). *Irrigation Water Management:
    Irrigation Scheduling*. Training Manual 4, FAO, Rome (irrigation
    efficiencies, net/gross irrigation requirement).
Doorenbos, J. & Kassam, A. H. (1979). *Yield response to water*. FAO
    Irrigation and Drainage Paper 33, FAO, Rome.
FAO (2001). *Food balance sheets: a handbook*. FAO, Rome (self-sufficiency
    ratio; calorie conversion of commodities).
FAO (2011). *Energy-smart food for people and climate*. Issue paper, FAO,
    Rome (energy inputs of crop production).
Hoekstra, A. Y. & Chapagain, A. K. (2008). *Globalization of Water: Sharing
    the Planet's Freshwater Resources*. Blackwell, Oxford.
Hoff, H. (2011). *Understanding the Nexus*. Background paper for the Bonn 2011
    Conference: The Water, Energy and Food Security Nexus. SEI, Stockholm.
Mekonnen, M. M. & Hoekstra, A. Y. (2011). The green, blue and grey water
    footprint of crops and derived crop products. *Hydrology and Earth System
    Sciences* 15, 1577-1600.
Molden, D. et al. (2010). Improving agricultural water productivity: between
    optimism and caution. *Agricultural Water Management* 97, 528-535.
Pelletier, N. et al. (2011). Energy intensity of agriculture and food
    systems. *Annual Review of Environment and Resources* 36, 223-246.
Steduto, P., Hsiao, T. C., Fereres, E. & Raes, D. (2012). *Crop yield
    response to water*. FAO Irrigation and Drainage Paper 66, FAO, Rome.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from wefnexus.models import Crop, Riparian

__all__ = [
    "M3_PER_HA_MM",
    "M3_PER_MM3",
    "KG_PER_T",
    "KWH_PER_GWH",
    "DAYS_PER_YEAR",
    "DEFAULT_KCAL_PER_KG",
    "DEFAULT_VIRTUAL_WATER_M3_PER_T",
    "DEFAULT_FARM_ENERGY_KWH_PER_HA",
    "crop_evapotranspiration_mm",
    "net_irrigation_mm",
    "gross_irrigation_mm",
    "irrigation_requirement_mm3",
    "riparian_crop_requirements_mm3",
    "riparian_irrigation_requirement_mm3",
    "fao33_relative_yield",
    "fao33_yield",
    "rainfall_et_share",
    "relative_evapotranspiration",
    "crop_production",
    "riparian_food_production",
    "food_demand_kcal",
    "food_self_sufficiency",
    "food_security_index",
    "virtual_water_content_m3_per_t",
    "water_productivity_kg_per_m3",
    "virtual_water_import_mm3",
    "energy_for_agriculture_gwh",
]

#: Cubic metres of water in a depth of 1 mm over 1 ha (1 ha = 10 000 m2).
M3_PER_HA_MM: float = 10.0
#: Cubic metres per million cubic metres.
M3_PER_MM3: float = 1.0e6
#: Kilograms per metric tonne.
KG_PER_T: float = 1000.0
#: Kilowatt-hours per gigawatt-hour.
KWH_PER_GWH: float = 1.0e6
#: Days in a (non-leap) year.
DAYS_PER_YEAR: float = 365.0
#: Energy content of a generic cereal (wheat), kcal per kg (FAO 2001).
DEFAULT_KCAL_PER_KG: float = 3400.0
#: Virtual water content of a generic cereal, m3 per tonne (Hoekstra &
#: Chapagain 2008 report ~1300 m3/t for wheat globally; Mekonnen & Hoekstra
#: 2011 ~1600 m3/t for cereals).
DEFAULT_VIRTUAL_WATER_M3_PER_T: float = 1500.0
#: Illustrative direct farm energy of irrigated crop production (machinery,
#: on-farm electricity), kWh per ha per year (FAO 2011; Pelletier et al. 2011).
DEFAULT_FARM_ENERGY_KWH_PER_HA: float = 150.0

#: Keys of the per-crop dictionary returned by :func:`crop_production`.
_CROP_PRODUCTION_KEYS = (
    "area_ha",
    "et_ratio",
    "irrigation_supply_ratio",
    "rainfall_et_share",
    "yield_t_ha",
    "production_t",
    "kcal",
    "value_usd",
    "water_required_mm3",
    "water_supplied_mm3",
)


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


def _positive(name: str, value: Any) -> float:
    out = _as_float(name, value)
    if out <= 0.0:
        raise ValueError(f"{name} must be > 0, got {out!r}")
    return out


def _efficiency(name: str, value: Any) -> float:
    """Validate an irrigation efficiency in ``(0, 1]``."""
    out = _as_float(name, value)
    if not 0.0 < out <= 1.0:
        raise ValueError(f"{name} must be in (0, 1], got {out!r}")
    return out


def _check_crop(crop: Any) -> Crop:
    if not isinstance(crop, Crop):
        raise ValueError(f"crop must be a wefnexus.models.Crop, got {type(crop).__name__}")
    return crop


def _check_riparian(riparian: Any) -> Riparian:
    if not isinstance(riparian, Riparian):
        raise ValueError(
            f"riparian must be a wefnexus.models.Riparian, got {type(riparian).__name__}"
        )
    return riparian


def _crop_label(crop: Crop) -> str:
    return f"crop {crop.name!r}"


# --------------------------------------------------------------------------- #
# Crop water requirements
# --------------------------------------------------------------------------- #
def crop_evapotranspiration_mm(et0_mm_day: float, kc: float, season_days: float) -> float:
    """Seasonal crop evapotranspiration from the FAO-56 single crop coefficient.

    Parameters
    ----------
    et0_mm_day : float
        Reference (grass) evapotranspiration averaged over the season
        (mm/day), >= 0.  Typical values: 2-4 mm/day humid temperate,
        5-7 mm/day arid.
    kc : float
        Seasonal-average FAO-56 crop coefficient (dimensionless), >= 0.
        Typical seasonal means: wheat 0.85, maize 0.9, rice 1.1, cotton 0.9.
    season_days : float
        Length of the growing season (days), >= 0.

    Returns
    -------
    float
        Seasonal crop evapotranspiration ``ETc`` in **mm**::

            ETc = ET0 * Kc * season_days

    Raises
    ------
    ValueError
        On a negative or non-finite argument.

    Notes
    -----
    This is the single-coefficient approach of FAO Irrigation and Drainage
    Paper 56 (Allen et al. 1998, Ch. 6) applied with one seasonal-mean
    ``Kc``.  Crop-stage variation of ``Kc`` is ignored.

    Examples
    --------
    >>> crop_evapotranspiration_mm(4.0, 0.85, 150)
    510.0
    """
    et0 = _non_negative("et0_mm_day", et0_mm_day)
    k = _non_negative("kc", kc)
    days = _non_negative("season_days", season_days)
    return et0 * k * days


def net_irrigation_mm(etc_mm: float, effective_rainfall_mm: float) -> float:
    """Net irrigation requirement: crop ET not met by effective rainfall.

    Parameters
    ----------
    etc_mm : float
        Seasonal crop evapotranspiration (mm), >= 0; see
        :func:`crop_evapotranspiration_mm`.
    effective_rainfall_mm : float
        Seasonal effective rainfall ``Pe`` (mm), >= 0: the part of rainfall
        stored in the root zone and usable by the crop (FAO-25, Dastane 1974;
        Brouwer et al. 1989).

    Returns
    -------
    float
        Net irrigation requirement ``IRn = max(ETc - Pe, 0)`` in **mm**.
        Zero when rainfall covers the whole crop ET (rainfed crop).

    Raises
    ------
    ValueError
        On a negative or non-finite argument.

    Examples
    --------
    >>> net_irrigation_mm(510.0, 300.0)
    210.0
    >>> net_irrigation_mm(200.0, 300.0)
    0.0
    """
    etc = _non_negative("etc_mm", etc_mm)
    pe = _non_negative("effective_rainfall_mm", effective_rainfall_mm)
    return max(etc - pe, 0.0)


def gross_irrigation_mm(net_mm: float, efficiency: float) -> float:
    """Gross irrigation requirement at the field or scheme intake.

    Parameters
    ----------
    net_mm : float
        Net irrigation requirement (mm), >= 0; see :func:`net_irrigation_mm`.
    efficiency : float
        Overall irrigation (conveyance x application) efficiency in (0, 1].
        Brouwer et al. (1989) give ~0.4-0.5 for surface irrigation, 0.6-0.75
        for sprinkler and 0.8-0.9 for drip systems.

    Returns
    -------
    float
        Gross irrigation requirement ``IRg = IRn / efficiency`` in **mm**,
        i.e. the depth that must be withdrawn so that ``IRn`` reaches the
        crop.  The losses ``IRg - IRn`` are return flow / percolation and
        are *not* consumed by the crop.

    Raises
    ------
    ValueError
        On a negative/non-finite ``net_mm`` or an efficiency outside (0, 1].

    Examples
    --------
    >>> gross_irrigation_mm(210.0, 0.5)
    420.0
    """
    net = _non_negative("net_mm", net_mm)
    eff = _efficiency("efficiency", efficiency)
    return net / eff


def irrigation_requirement_mm3(
    crop: Crop,
    et0_mm_day: float,
    effective_rainfall_mm: float,
    efficiency: float,
) -> float:
    """Gross annual irrigation water requirement of one crop (volume).

    Parameters
    ----------
    crop : Crop
        Supplies ``area_ha``, ``kc`` and ``season_days``.  Not modified.
    et0_mm_day : float
        Seasonal-mean reference evapotranspiration (mm/day), >= 0.
    effective_rainfall_mm : float
        Seasonal effective rainfall (mm), >= 0.
    efficiency : float
        Overall irrigation efficiency in (0, 1].

    Returns
    -------
    float
        Gross irrigation requirement in **Mm3/yr**::

            IR = gross_mm * area_ha * 10 / 1e6

        with ``gross_mm = max(ET0 * Kc * days - Pe, 0) / efficiency``.

    Raises
    ------
    ValueError
        If ``crop`` is not a :class:`~wefnexus.models.Crop`, has a negative
        area / ``kc`` / season length, or any other argument is invalid.

    Notes
    -----
    1 mm over 1 ha is 10 m3 (Allen et al. 1998, Ch. 1).  The requirement is
    linear in area, so scaling the area by a factor scales the volume by the
    same factor.

    Examples
    --------
    Wheat on 150 000 ha, ET0 4 mm/day, Kc 0.85, 150 days, 300 mm effective
    rainfall, 50 % efficiency: (4*0.85*150 - 300)/0.5 = 420 mm -> 630 Mm3.

    >>> from wefnexus.models import Crop
    >>> wheat = Crop("wheat", 150_000, 0.85, 150, 4.0, 1.05, 3400, 250)
    >>> irrigation_requirement_mm3(wheat, 4.0, 300.0, 0.5)
    630.0
    """
    c = _check_crop(crop)
    label = _crop_label(c)
    area = _non_negative(f"{label}.area_ha", c.area_ha)
    kc = _non_negative(f"{label}.kc", c.kc)
    days = _non_negative(f"{label}.season_days", c.season_days)
    etc = crop_evapotranspiration_mm(et0_mm_day, kc, days)
    net = net_irrigation_mm(etc, effective_rainfall_mm)
    gross = gross_irrigation_mm(net, efficiency)
    return gross * area * M3_PER_HA_MM / M3_PER_MM3


def _resolve_riparian_irrigation_inputs(
    riparian: Riparian, area_factor: float, efficiency: Optional[float]
) -> Tuple[float, float, float, float]:
    """Validate and return ``(et0, pe, efficiency, area_factor)`` for a riparian."""
    rip = _check_riparian(riparian)
    factor = _non_negative("area_factor", area_factor)
    eff_source = rip.irrigation_efficiency if efficiency is None else efficiency
    eff_name = "riparian.irrigation_efficiency" if efficiency is None else "efficiency"
    eff = _efficiency(eff_name, eff_source)
    et0 = _non_negative("riparian.et0_mm_day", rip.et0_mm_day)
    pe = _non_negative("riparian.effective_rainfall_mm", rip.effective_rainfall_mm)
    return et0, pe, eff, factor


def _unique_crop_keys(crops: List[Crop]) -> List[str]:
    """Dictionary keys for a crop list: the crop names, de-duplicated.

    A repeated name gets a ``#2``, ``#3``, ... suffix so that no crop is
    silently dropped from a name-keyed result.
    """
    keys: List[str] = []
    seen: Dict[str, int] = {}
    for crop in crops:
        base = str(crop.name)
        count = seen.get(base, 0) + 1
        seen[base] = count
        key = base if count == 1 else f"{base}#{count}"
        while key in keys:  # a literal "wheat#2" crop name could collide
            count += 1
            seen[base] = count
            key = f"{base}#{count}"
        keys.append(key)
    return keys


def riparian_crop_requirements_mm3(
    riparian: Riparian, area_factor: float = 1.0, efficiency: Optional[float] = None
) -> Dict[str, float]:
    """Gross irrigation requirement of each crop of a riparian.

    Parameters
    ----------
    riparian : Riparian
        Supplies ``crops``, ``et0_mm_day``, ``effective_rainfall_mm`` and the
        default ``irrigation_efficiency``.  Not modified.
    area_factor : float, optional
        Multiplier on every crop area (e.g. ``Scenario.irrigated_area_factor``),
        >= 0.  Default 1.
    efficiency : float, optional
        Overall irrigation efficiency in (0, 1] overriding
        ``riparian.irrigation_efficiency`` (e.g. from
        ``Scenario.irrigation_efficiency``).  ``None`` (default) uses the
        riparian's value.

    Returns
    -------
    dict
        ``{crop name: Mm3/yr}`` in the order of ``riparian.crops``; a repeated
        crop name is suffixed ``#2``, ``#3``, ... so that every crop appears.
        Empty when the riparian has no crops.

    Raises
    ------
    ValueError
        On a non-``Riparian`` argument, a negative ``area_factor``, an
        efficiency outside (0, 1], or invalid crop / climate attributes.

    See Also
    --------
    irrigation_requirement_mm3, riparian_irrigation_requirement_mm3
    """
    rip = _check_riparian(riparian)
    et0, pe, eff, factor = _resolve_riparian_irrigation_inputs(rip, area_factor, efficiency)
    crops = list(rip.crops)
    keys = _unique_crop_keys(crops)
    return {
        key: irrigation_requirement_mm3(crop, et0, pe, eff) * factor
        for key, crop in zip(keys, crops)
    }


def riparian_irrigation_requirement_mm3(
    riparian: Riparian, area_factor: float = 1.0, efficiency: Optional[float] = None
) -> float:
    """Total gross irrigation requirement of a riparian's crops.

    Parameters
    ----------
    riparian : Riparian
        Supplies ``crops``, ``et0_mm_day``, ``effective_rainfall_mm`` and the
        default ``irrigation_efficiency``.  Not modified.
    area_factor : float, optional
        Multiplier on every crop area, >= 0.  Default 1.
    efficiency : float, optional
        Overall irrigation efficiency in (0, 1] overriding
        ``riparian.irrigation_efficiency``; ``None`` (default) uses the
        riparian's value.

    Returns
    -------
    float
        ``sum_i irrigation_requirement_mm3(crop_i) * area_factor`` in
        **Mm3/yr**; 0 when the riparian grows no crops.

    Raises
    ------
    ValueError
        On a non-``Riparian`` argument, a negative ``area_factor``, an
        efficiency outside (0, 1], or invalid crop / climate attributes.

    Notes
    -----
    This is the agricultural *withdrawal* demand that the nexus model uses
    in place of ``WaterDemand.agricultural`` when crops are specified.  It is
    decreasing in ``efficiency`` and linear in ``area_factor``.

    Examples
    --------
    Highland in the example basin (ET0 4 mm/day, 300 mm effective rainfall,
    efficiency 0.5): wheat 630 + maize 168 = 798 Mm3/yr.

    >>> from wefnexus.data import example_basin
    >>> round(riparian_irrigation_requirement_mm3(example_basin().riparian("Highland")), 6)
    798.0
    """
    return sum(riparian_crop_requirements_mm3(riparian, area_factor, efficiency).values())


# --------------------------------------------------------------------------- #
# Yield response to water (FAO-33)
# --------------------------------------------------------------------------- #
def fao33_relative_yield(ky: float, et_ratio: float) -> float:
    """Relative yield ``Ya/Ym`` from the FAO-33 yield-response function.

    Parameters
    ----------
    ky : float
        Yield-response factor (dimensionless), >= 0.  The FAO-33
        total-season table (Doorenbos & Kassam 1979, Table 24) gives 1.00
        (winter wheat), 1.15 (spring wheat), 1.25 (maize), 0.85 (cotton),
        0.70 (groundnut), 1.10 (potato), 0.90 (sorghum), 0.85 (soybean),
        1.20 (sugarcane).  Flooded (paddy) rice is not tabulated in FAO-33;
        seasonal values of about 1.0-1.2 for rice come from later
        compilations such as Steduto et al. (2012, FAO-66).  ``ky > 1``
        means yield falls faster than ET, ``ky < 1`` slower.
    et_ratio : float
        Relative evapotranspiration ``ETa/ETm``, >= 0.  Values above 1 are
        treated as 1 (actual ET cannot exceed maximum ET).

    Returns
    -------
    float
        ``clip(1 - ky * (1 - min(et_ratio, 1)), 0, 1)`` (dimensionless).

    Raises
    ------
    ValueError
        On a negative or non-finite argument.

    Examples
    --------
    >>> fao33_relative_yield(1.25, 0.5)
    0.375
    >>> fao33_relative_yield(1.05, 0.0)
    0.0
    """
    k = _non_negative("ky", ky)
    ratio = min(_non_negative("et_ratio", et_ratio), 1.0)
    relative = 1.0 - k * (1.0 - ratio)
    return min(max(relative, 0.0), 1.0)


def fao33_yield(yield_max_t_ha: float, ky: float, et_ratio: float) -> float:
    """Actual yield under water deficit (FAO-33, Doorenbos & Kassam 1979).

    Parameters
    ----------
    yield_max_t_ha : float
        Maximum (fully watered) yield ``Ym`` (t/ha), >= 0.
    ky : float
        Yield-response factor, >= 0 (see :func:`fao33_relative_yield`).
    et_ratio : float
        Relative evapotranspiration ``ETa/ETm``, >= 0; values above 1 count
        as 1.

    Returns
    -------
    float
        Actual yield ``Ya`` in **t/ha**::

            Ya = Ym * (1 - ky * (1 - ETa/ETm)),  clipped to [0, Ym]

    Raises
    ------
    ValueError
        On a negative or non-finite argument.

    Notes
    -----
    The seasonal FAO-33 relation is linear in the ET deficit, so ``Ya`` is
    non-decreasing in ``et_ratio`` and reaches ``Ym`` at ``et_ratio = 1``.
    With ``ky > 1`` the yield is already zero at ``et_ratio = 1 - 1/ky``
    (0.13 for spring wheat with ky 1.15, 0.2 for maize with ky 1.25); with
    ``ky < 1`` a crop still yields ``Ym * (1 - ky)`` with no water at all
    (0.15 Ym for cotton with ky 0.85, 0.30 Ym for groundnut with ky 0.70),
    the well-known limitation of the linear seasonal model (Steduto et al.
    2012, FAO-66).  ``et_ratio`` must be the relative *crop
    evapotranspiration*, i.e. rainfall plus net irrigation over ``ETc``, not
    the gross irrigation supply ratio; see :func:`crop_production`.

    Examples
    --------
    >>> fao33_yield(4.0, 1.05, 1.0)
    4.0
    >>> round(fao33_yield(4.0, 1.05, 0.8), 6)
    3.16
    >>> fao33_yield(4.0, 1.05, 0.0)
    0.0
    """
    ym = _non_negative("yield_max_t_ha", yield_max_t_ha)
    return ym * fao33_relative_yield(ky, et_ratio)


# --------------------------------------------------------------------------- #
# Production
# --------------------------------------------------------------------------- #
def rainfall_et_share(etc_mm: float, effective_rainfall_mm: float) -> float:
    """Share of a crop's maximum evapotranspiration met by effective rainfall.

    Parameters
    ----------
    etc_mm : float
        Seasonal crop evapotranspiration ``ETc = ETm`` (mm), >= 0; see
        :func:`crop_evapotranspiration_mm`.
    effective_rainfall_mm : float
        Seasonal effective rainfall ``Pe`` (mm), >= 0.

    Returns
    -------
    float
        ``min(Pe / ETc, 1)`` (dimensionless, in ``[0, 1]``): the "green
        water" part of the crop's ET (Mekonnen & Hoekstra 2011).  Its
        complement ``1 - Pe/ETc = IRn/ETc`` is the share that irrigation
        must supply.  When ``ETc = 0`` there is no evaporative demand to
        meet and the function returns 1.

    Raises
    ------
    ValueError
        On a negative or non-finite argument.

    Examples
    --------
    >>> rainfall_et_share(510.0, 300.0)
    0.5882352941176471
    >>> rainfall_et_share(200.0, 300.0)
    1.0
    """
    etc = _non_negative("etc_mm", etc_mm)
    pe = _non_negative("effective_rainfall_mm", effective_rainfall_mm)
    if etc <= 0.0:
        return 1.0
    return min(pe / etc, 1.0)


def relative_evapotranspiration(
    supply_ratio: float, rainfall_share: float = 0.0
) -> float:
    """FAO-33 relative evapotranspiration ``ETa/ETm`` of an irrigated crop.

    Parameters
    ----------
    supply_ratio : float
        Gross irrigation supply ratio ``supplied / required`` (dimensionless),
        >= 0; values above 1 count as 1 (water beyond the requirement is not
        evaporated by the crop).
    rainfall_share : float, optional
        Share of the crop's maximum ET met by effective rainfall,
        ``min(Pe / ETc, 1)`` (see :func:`rainfall_et_share`), in ``[0, 1]``.
        Default 0 (no effective rainfall, every mm of crop ET comes from
        irrigation).

    Returns
    -------
    float
        ``ETa/ETm = 1 - (1 - supply_ratio) * (1 - rainfall_share)``, clipped
        to ``[0, 1]``.

    Raises
    ------
    ValueError
        On a negative / non-finite ``supply_ratio`` or a ``rainfall_share``
        outside ``[0, 1]``.

    Notes
    -----
    With ``ETc`` the maximum crop ET, ``Pe`` the effective rainfall,
    ``IRn = max(ETc - Pe, 0)`` the net and ``IRg = IRn / efficiency`` the
    gross requirement, delivering a fraction ``s`` of ``IRg`` at the same
    efficiency puts ``s * IRn`` into the root zone, so (Doorenbos & Kassam
    1979, Ch. 2)::

        ETa     = Pe + s * IRn = Pe + s * (ETc - Pe)
        ETa/ETm = Pe/ETc + s * (1 - Pe/ETc) = 1 - (1 - s) * (1 - Pe/ETc)

    A fully rain-fed crop (``Pe >= ETc``) therefore has ``ETa/ETm = 1`` for
    any ``s``, and with no rainfall ``ETa/ETm = s``.

    Examples
    --------
    Highland wheat of the example basin (ETc 510 mm, Pe 300 mm) with no
    irrigation at all still evaporates 300/510 of its maximum ET:

    >>> round(relative_evapotranspiration(0.0, 300.0 / 510.0), 6)
    0.588235
    >>> relative_evapotranspiration(0.5)
    0.5
    """
    s = min(_non_negative("supply_ratio", supply_ratio), 1.0)
    share = _as_float("rainfall_share", rainfall_share)
    if not 0.0 <= share <= 1.0:
        raise ValueError(f"rainfall_share must be in [0, 1], got {share!r}")
    ratio = 1.0 - (1.0 - s) * (1.0 - share)
    return min(max(ratio, 0.0), 1.0)


def crop_production(
    crop: Crop,
    water_supplied_mm3: float,
    water_required_mm3: float,
    area_factor: float = 1.0,
    et0_mm_day: Optional[float] = None,
    effective_rainfall_mm: Optional[float] = None,
) -> Dict[str, float]:
    """Production, calories and value of one crop given its water supply.

    Parameters
    ----------
    crop : Crop
        Supplies ``area_ha``, ``kc``, ``season_days``, ``yield_max_t_ha``,
        ``ky``, ``kcal_per_kg`` and ``price_usd_t``.  Not modified.
    water_supplied_mm3 : float
        Gross irrigation water actually delivered to the crop (Mm3/yr), >= 0.
    water_required_mm3 : float
        Gross irrigation requirement of the crop (Mm3/yr), >= 0; see
        :func:`irrigation_requirement_mm3`.  It must already include the
        same ``area_factor`` as applied here.
    area_factor : float, optional
        Multiplier on ``crop.area_ha``, >= 0.  Default 1.
    et0_mm_day : float, optional
        Seasonal-mean reference evapotranspiration (mm/day), >= 0, used
        with ``crop.kc`` and ``crop.season_days`` to obtain the maximum crop
        ET ``ETc`` (:func:`crop_evapotranspiration_mm`).  Must be given
        together with ``effective_rainfall_mm``.
    effective_rainfall_mm : float, optional
        Seasonal effective rainfall ``Pe`` (mm), >= 0, the part of the crop's
        ET that is met without irrigation.  When both climate arguments are
        omitted the crop is assumed to receive **no** effective rainfall, so
        the whole of its ET must come from irrigation and ``et_ratio``
        reduces to the gross supply ratio (exact only for ``Pe = 0``).

    Returns
    -------
    dict
        * ``"area_ha"`` – harvested area ``crop.area_ha * area_factor`` (ha)
        * ``"irrigation_supply_ratio"`` – ``min(supplied / required, 1)``;
          1.0 when the requirement is 0 (dimensionless)
        * ``"rainfall_et_share"`` – ``min(Pe / ETc, 1)``, the share of the
          crop's maximum ET met by effective rainfall (0 when no climate is
          given; 1 when ``ETc = 0``)
        * ``"et_ratio"`` – FAO-33 relative evapotranspiration ``ETa/ETm``
          (dimensionless, in ``[0, 1]``)::

              ETa/ETm = (Pe + efficiency * supplied_mm) / ETc,  capped at 1
                      = 1 - (1 - irrigation_supply_ratio) * (1 - rainfall_et_share)

          1.0 when the requirement is 0 (rainfall covers the crop's ET) or
          when ``ETc <= Pe``
        * ``"yield_t_ha"`` – :func:`fao33_yield` at that ``et_ratio`` (t/ha)
        * ``"production_t"`` – ``yield_t_ha * area_ha`` (t)
        * ``"kcal"`` – ``production_t * 1000 * kcal_per_kg`` (kcal)
        * ``"value_usd"`` – ``production_t * price_usd_t`` (USD)
        * ``"water_required_mm3"``, ``"water_supplied_mm3"`` – the inputs
          echoed back (Mm3/yr) for conservation checks.

    Raises
    ------
    ValueError
        On a non-``Crop`` argument, negative water volumes or area factor,
        negative crop attributes, a negative / non-finite climate argument,
        or when only one of ``et0_mm_day`` / ``effective_rainfall_mm`` is
        given.

    Notes
    -----
    FAO-33 (Doorenbos & Kassam 1979) defines the deficit on the *total*
    crop evapotranspiration, ``ETa = Pe + net irrigation applied``.  The
    gross supply ratio alone understates ``ETa/ETm`` by ``Pe/ETc`` whenever
    effective rainfall is positive: a mostly rain-fed crop would be reported
    as a total failure in a year without irrigation.  Using the gross ratio
    ``s = supplied / required`` and the net/gross decomposition of this
    module (``required = IRn / efficiency * area``), the net water reaching
    the root zone is ``s * IRn`` whatever the efficiency, hence the
    equivalent closed form ``1 - (1 - s) * IRn/ETc`` implemented here (see
    :func:`relative_evapotranspiration`).  Non-food crops such as cotton
    carry ``kcal_per_kg = 0`` and contribute only to ``value_usd``.

    Examples
    --------
    Highland wheat of the example basin: ET0 4 mm/day, Kc 0.85, 150 days
    (ETc 510 mm), Pe 300 mm, efficiency 0.5 -> IRn 210 mm, IRg 420 mm,
    requirement 630 Mm3 on 150 000 ha.

    >>> from wefnexus.models import Crop
    >>> wheat = Crop("wheat", 150_000, 0.85, 150, 4.0, 1.05, 3400, 250)
    >>> out = crop_production(wheat, 630.0, 630.0, et0_mm_day=4.0, effective_rainfall_mm=300.0)
    >>> out["production_t"], out["kcal"], out["value_usd"]
    (600000.0, 2040000000000.0, 150000000.0)

    Without any irrigation the crop still evaporates the rainfall, so
    ``ETa/ETm = 300/510`` and FAO-33 gives ``Ya/Ym = 1 - 1.05 * (1 - 0.588)``:

    >>> out = crop_production(wheat, 0.0, 630.0, et0_mm_day=4.0, effective_rainfall_mm=300.0)
    >>> round(out["et_ratio"], 3), round(out["yield_t_ha"] / 4.0, 3)
    (0.588, 0.568)
    """
    c = _check_crop(crop)
    label = _crop_label(c)
    supplied = _non_negative("water_supplied_mm3", water_supplied_mm3)
    required = _non_negative("water_required_mm3", water_required_mm3)
    factor = _non_negative("area_factor", area_factor)
    base_area = _non_negative(f"{label}.area_ha", c.area_ha)
    ym = _non_negative(f"{label}.yield_max_t_ha", c.yield_max_t_ha)
    ky = _non_negative(f"{label}.ky", c.ky)
    kcal_per_kg = _non_negative(f"{label}.kcal_per_kg", c.kcal_per_kg)
    price = _non_negative(f"{label}.price_usd_t", c.price_usd_t)

    if (et0_mm_day is None) != (effective_rainfall_mm is None):
        raise ValueError(
            "et0_mm_day and effective_rainfall_mm must be given together "
            f"(got et0_mm_day={et0_mm_day!r}, effective_rainfall_mm={effective_rainfall_mm!r})"
        )
    if et0_mm_day is None:
        rain_share = 0.0
    else:
        kc = _non_negative(f"{label}.kc", c.kc)
        days = _non_negative(f"{label}.season_days", c.season_days)
        etc = crop_evapotranspiration_mm(et0_mm_day, kc, days)
        rain_share = rainfall_et_share(etc, effective_rainfall_mm)

    area = base_area * factor
    supply_ratio = 1.0 if required <= 0.0 else min(supplied / required, 1.0)
    et_ratio = relative_evapotranspiration(supply_ratio, rain_share)
    yield_t_ha = fao33_yield(ym, ky, et_ratio)
    production_t = yield_t_ha * area
    return {
        "area_ha": area,
        "et_ratio": et_ratio,
        "irrigation_supply_ratio": supply_ratio,
        "rainfall_et_share": rain_share,
        "yield_t_ha": yield_t_ha,
        "production_t": production_t,
        "kcal": production_t * KG_PER_T * kcal_per_kg,
        "value_usd": production_t * price,
        "water_required_mm3": required,
        "water_supplied_mm3": supplied,
    }


def riparian_food_production(
    riparian: Riparian,
    agricultural_water_mm3: float,
    area_factor: float = 1.0,
    efficiency: Optional[float] = None,
) -> Dict[str, Any]:
    """Food production of a riparian from its available agricultural water.

    The available gross agricultural water is shared among the riparian's
    crops **in proportion to their gross irrigation requirement** (every
    irrigated crop receives the same relative supply), each crop's
    ``ETa/ETm`` follows from the riparian's effective rainfall plus the net
    irrigation it receives, yields follow FAO-33 and the per-crop results
    are summed.

    Parameters
    ----------
    riparian : Riparian
        Supplies ``crops``, ``et0_mm_day``, ``effective_rainfall_mm`` and the
        default ``irrigation_efficiency``.  Not modified.
    agricultural_water_mm3 : float
        Gross agricultural water available in the year (Mm3/yr), >= 0 – e.g.
        the agricultural withdrawal from the basin routing plus any
        groundwater / desalinated water used for irrigation.  Water in excess
        of the total requirement is not used.
    area_factor : float, optional
        Multiplier on every crop area, >= 0.  Default 1.
    efficiency : float, optional
        Overall irrigation efficiency in (0, 1] overriding
        ``riparian.irrigation_efficiency``; ``None`` (default) uses the
        riparian's value.  Higher efficiency lowers the gross requirement and
        so raises the relative supply obtained from the same water.

    Returns
    -------
    dict
        * ``"crops"`` – ``{crop name: crop_production dict}`` (see
          :func:`crop_production`); a repeated crop name is suffixed ``#2``,
          ``#3``, ...; empty when there are no crops
        * ``"production_t"`` – total production (t)
        * ``"kcal"`` – total food energy (kcal); non-food crops with
          ``kcal_per_kg = 0`` add nothing
        * ``"value_usd"`` – total farm-gate value (USD)
        * ``"requirement_mm3"`` – total gross irrigation requirement (Mm3/yr)
        * ``"supplied_mm3"`` – water actually applied,
          ``min(agricultural_water_mm3, requirement_mm3)`` (Mm3/yr)
        * ``"irrigation_supply_ratio"`` – overall gross supply ratio
          ``supplied / requirement`` (1.0 when the requirement is 0)
        * ``"etm_mm3"`` – maximum crop evapotranspiration of all crops,
          ``sum_i ETc_i * area_i * 10 / 1e6`` (Mm3/yr)
        * ``"eta_mm3"`` – actual crop evapotranspiration,
          ``sum_i (ETa/ETm)_i * ETc_i * area_i * 10 / 1e6`` (Mm3/yr),
          i.e. the effective rainfall evaporated by the crops plus the net
          irrigation water ``efficiency * supplied_mm3``
        * ``"et_ratio"`` – overall FAO-33 relative evapotranspiration of the
          cropped area, ``eta_mm3 / etm_mm3`` (1.0 when ``etm_mm3`` is 0,
          e.g. no crops); each crop's ``ETa/ETm`` weighted by its maximum ET
          volume
        * ``"available_mm3"`` – ``agricultural_water_mm3`` as given (Mm3/yr)
        * ``"deficit_mm3"`` – ``requirement_mm3 - supplied_mm3`` (Mm3/yr)
        * ``"area_ha"`` – total irrigated area after ``area_factor`` (ha).

    Raises
    ------
    ValueError
        On a non-``Riparian`` argument, negative water / area factor, an
        efficiency outside (0, 1], or invalid crop / climate attributes.

    Notes
    -----
    Proportional sharing is the "equal relative deficit" rule; it is what a
    scheme operator applies when rationing a canal pro rata.  Because the
    FAO-33 response is linear, the water applied to each crop sums exactly
    to ``supplied_mm3`` and production is non-decreasing and concave in
    the available water.  With the proportional rule the overall ratio has
    the closed form (Doorenbos & Kassam 1979)::

        et_ratio = (sum_i min(Pe, ETc_i) * area_i * 10/1e6 + efficiency * supplied_mm3) / etm_mm3

    so it equals the gross supply ratio only when there is no effective
    rainfall, and is 1 whenever every crop is rain-fed.  The calorie and
    value totals feed :func:`food_self_sufficiency` and the sustainability
    indices (Hoff 2011); ``et_ratio`` is the stability pillar of
    ``wefnexus.sustainability.food_security_index``.

    Examples
    --------
    >>> from wefnexus.data import example_basin
    >>> out = riparian_food_production(example_basin().riparian("Highland"), 798.0)
    >>> round(out["production_t"]), round(out["et_ratio"], 6)
    (900000, 1.0)

    With no irrigation at all Highland's 300 mm of effective rainfall still
    meets 600 of the 999 Mm3 of maximum crop ET (wheat 300/510, maize
    300/468), so the crops are far from a total failure:

    >>> out = riparian_food_production(example_basin().riparian("Highland"), 0.0)
    >>> round(out["et_ratio"], 4), round(out["irrigation_supply_ratio"], 4), round(out["production_t"])
    (0.6006, 0.0, 505973)
    """
    rip = _check_riparian(riparian)
    available = _non_negative("agricultural_water_mm3", agricultural_water_mm3)
    et0, pe, eff, factor = _resolve_riparian_irrigation_inputs(rip, area_factor, efficiency)

    crops = list(rip.crops)
    keys = _unique_crop_keys(crops)
    requirements = [irrigation_requirement_mm3(c, et0, pe, eff) * factor for c in crops]
    requirement = float(sum(requirements))
    supplied = min(available, requirement)
    supply_ratio = supplied / requirement if requirement > 0.0 else 1.0

    crop_results: Dict[str, Dict[str, float]] = {}
    production_t = kcal = value_usd = area_ha = 0.0
    etm_mm3 = eta_mm3 = 0.0
    for key, crop, req in zip(keys, crops, requirements):
        share = req * supply_ratio  # proportional to requirement
        result = crop_production(
            crop, share, req, area_factor=factor, et0_mm_day=et0, effective_rainfall_mm=pe
        )
        crop_results[key] = result
        production_t += result["production_t"]
        kcal += result["kcal"]
        value_usd += result["value_usd"]
        area_ha += result["area_ha"]
        etc_i = crop_evapotranspiration_mm(et0, crop.kc, crop.season_days)
        etm_i = etc_i * result["area_ha"] * M3_PER_HA_MM / M3_PER_MM3
        etm_mm3 += etm_i
        eta_mm3 += result["et_ratio"] * etm_i
    eta_mm3 = min(eta_mm3, etm_mm3)
    et_ratio = min(max(eta_mm3 / etm_mm3, 0.0), 1.0) if etm_mm3 > 0.0 else 1.0

    return {
        "crops": crop_results,
        "production_t": production_t,
        "kcal": kcal,
        "value_usd": value_usd,
        "requirement_mm3": requirement,
        "supplied_mm3": supplied,
        "irrigation_supply_ratio": supply_ratio,
        "etm_mm3": etm_mm3,
        "eta_mm3": eta_mm3,
        "et_ratio": et_ratio,
        "available_mm3": available,
        "deficit_mm3": max(requirement - supplied, 0.0),
        "area_ha": area_ha,
    }


# --------------------------------------------------------------------------- #
# Food security indicators
# --------------------------------------------------------------------------- #
def food_demand_kcal(
    population: float, kcal_per_capita_day: float = 2500.0, days: float = DAYS_PER_YEAR
) -> float:
    """Annual dietary energy demand of a population.

    Parameters
    ----------
    population : float
        Number of people, >= 0.
    kcal_per_capita_day : float, optional
        Dietary energy supply per person per day (kcal), >= 0.  Default
        2500 (FAO 2001 food balance sheets report ~2 100 kcal minimum dietary
        energy requirement and ~2 800 kcal average supply).
    days : float, optional
        Days in the period, >= 0.  Default 365.

    Returns
    -------
    float
        ``population * kcal_per_capita_day * days`` in **kcal**; identical to
        ``Riparian.food_demand_kcal()`` for the riparian's own values.
    """
    pop = _non_negative("population", population)
    per_cap = _non_negative("kcal_per_capita_day", kcal_per_capita_day)
    d = _non_negative("days", days)
    return pop * per_cap * d


def food_self_sufficiency(kcal_produced: float, kcal_demand: float) -> float:
    """Food self-sufficiency ratio: domestic production over demand.

    Parameters
    ----------
    kcal_produced : float
        Food energy produced (kcal/yr), >= 0.
    kcal_demand : float
        Food energy demand (kcal/yr), >= 0.

    Returns
    -------
    float
        ``kcal_produced / kcal_demand`` (dimensionless, **not capped**, so a
        value above 1 is an exportable surplus); 1.0 when demand is 0.

    Notes
    -----
    A calorie-based simplification of the FAO self-sufficiency ratio
    ``SSR = production / (production + imports - exports)`` (FAO 2001).
    """
    produced = _non_negative("kcal_produced", kcal_produced)
    demand = _non_negative("kcal_demand", kcal_demand)
    if demand <= 0.0:
        return 1.0
    return produced / demand


def food_security_index(kcal_produced: float, kcal_demand: float) -> float:
    """Share of food energy demand covered by domestic production, capped at 1.

    Parameters
    ----------
    kcal_produced : float
        Food energy produced (kcal/yr), >= 0.
    kcal_demand : float
        Food energy demand (kcal/yr), >= 0.

    Returns
    -------
    float
        ``min(kcal_produced / kcal_demand, 1)``; 1.0 when demand is 0.

    Notes
    -----
    This is the *availability* pillar of food security (FAO 2001; Hoff
    2011); the composite index that also accounts for the irrigation supply
    ratio lives in ``wefnexus.sustainability.food_security_index``.
    """
    return min(food_self_sufficiency(kcal_produced, kcal_demand), 1.0)


def virtual_water_content_m3_per_t(water_mm3: float, production_t: float) -> float:
    """Virtual (embedded) water content of a crop output.

    Parameters
    ----------
    water_mm3 : float
        Water used to grow the output (Mm3), >= 0 – here the gross
        irrigation (blue) water.
    production_t : float
        Output produced (t), >= 0.

    Returns
    -------
    float
        ``water_mm3 * 1e6 / production_t`` in **m3 per tonne**; ``inf`` when
        nothing is produced (even with zero water, by convention).

    Notes
    -----
    Hoekstra & Chapagain (2008) and Mekonnen & Hoekstra (2011) report global
    averages of roughly 1 300-1 800 m3/t for wheat, 900-1 200 m3/t for
    maize and 1 300-1 700 m3/t for rice (green + blue + grey).

    Examples
    --------
    >>> virtual_water_content_m3_per_t(798.0, 900_000.0)
    886.6666666666666
    """
    water = _non_negative("water_mm3", water_mm3)
    production = _non_negative("production_t", production_t)
    if production <= 0.0:
        return math.inf
    return water * M3_PER_MM3 / production


def water_productivity_kg_per_m3(production_t: float, water_mm3: float) -> float:
    """Crop water productivity: output per unit of water (Molden et al. 2010).

    Parameters
    ----------
    production_t : float
        Output produced (t), >= 0.
    water_mm3 : float
        Water used (Mm3), >= 0.

    Returns
    -------
    float
        ``production_t * 1000 / (water_mm3 * 1e6)`` in **kg per m3**; the
        reciprocal of :func:`virtual_water_content_m3_per_t` (with the t/kg
        conversion).  ``inf`` when output is produced with no water
        (rainfed) and 0 when nothing is produced.

    Notes
    -----
    Typical irrigated values are 0.5-1.5 kg/m3 for wheat and 1-2 kg/m3 for
    maize (Molden et al. 2010; Zwart & Bastiaanssen 2004).
    """
    production = _non_negative("production_t", production_t)
    water = _non_negative("water_mm3", water_mm3)
    if production <= 0.0:
        return 0.0
    if water <= 0.0:
        return math.inf
    return production * KG_PER_T / (water * M3_PER_MM3)


def virtual_water_import_mm3(
    kcal_deficit: float,
    kcal_per_kg: float = DEFAULT_KCAL_PER_KG,
    m3_per_t: float = DEFAULT_VIRTUAL_WATER_M3_PER_T,
) -> float:
    """Water embedded in the food imports needed to close a calorie deficit.

    Parameters
    ----------
    kcal_deficit : float
        Food energy demand not met by domestic production (kcal/yr).  A
        value <= 0 (surplus) needs no imports and returns 0.
    kcal_per_kg : float, optional
        Energy content of the imported staple (kcal/kg), > 0.  Default 3400
        (wheat).
    m3_per_t : float, optional
        Virtual water content of the imported staple (m3/t), >= 0.  Default
        1500 (cereal average, Hoekstra & Chapagain 2008).

    Returns
    -------
    float
        Virtual water import in **Mm3/yr**::

            tonnes = max(kcal_deficit, 0) / (kcal_per_kg * 1000)
            import = tonnes * m3_per_t / 1e6

    Raises
    ------
    ValueError
        On a non-finite deficit, ``kcal_per_kg <= 0`` or ``m3_per_t < 0``.

    Notes
    -----
    This is Allan's (1998) virtual-water argument: a water-scarce riparian
    can "import" the water it lacks as food.  The nexus model uses it to
    express a food deficit in water terms comparable to the basin flows.

    Examples
    --------
    3.4e12 kcal of wheat is 1e6 t, i.e. 1.5e9 m3 = 1500 Mm3 of virtual water.

    >>> virtual_water_import_mm3(3.4e12)
    1500.0
    """
    deficit = _as_float("kcal_deficit", kcal_deficit)
    per_kg = _positive("kcal_per_kg", kcal_per_kg)
    vwc = _non_negative("m3_per_t", m3_per_t)
    if deficit <= 0.0:
        return 0.0
    tonnes = deficit / (per_kg * KG_PER_T)
    return tonnes * vwc / M3_PER_MM3


def energy_for_agriculture_gwh(
    area_ha: float, kwh_per_ha: float = DEFAULT_FARM_ENERGY_KWH_PER_HA
) -> float:
    """Direct farm energy of crop production (excluding irrigation pumping).

    Parameters
    ----------
    area_ha : float
        Cultivated area (ha), >= 0.
    kwh_per_ha : float, optional
        Direct energy intensity (kWh per ha per year), >= 0.  Default 150,
        an illustrative value for machinery fuel and on-farm electricity of
        irrigated field crops (FAO 2011; Pelletier et al. 2011 report
        roughly 100-500 kWh/ha depending on mechanisation).

    Returns
    -------
    float
        ``area_ha * kwh_per_ha / 1e6`` in **GWh/yr**.  Pumping energy of the
        irrigation water itself is computed in
        ``wefnexus.energy.energy_for_water`` and must not be double counted.

    Examples
    --------
    >>> energy_for_agriculture_gwh(1_000_000.0)
    150.0
    """
    area = _non_negative("area_ha", area_ha)
    intensity = _non_negative("kwh_per_ha", kwh_per_ha)
    return area * intensity / KWH_PER_GWH
