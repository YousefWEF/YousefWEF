"""Serialisation of basins and scenarios to plain dictionaries and JSON (integration module).

The core models of :mod:`wefnexus.models` are dataclasses that hold
``Sector``-keyed dictionaries and nested objects (a :class:`~wefnexus.models.Basin`
holds :class:`~wefnexus.models.Riparian` objects, each with a
:class:`~wefnexus.models.WaterDemand`, a list of :class:`~wefnexus.models.Crop`
and an :class:`~wefnexus.models.EnergySystem`).  This module converts them to
and from JSON-compatible dictionaries so that basins and scenarios can be
stored in version control, edited by hand, exchanged between tools and fed to
the command-line interface (``python -m wefnexus run --basin my_basin.json``).

Conventions
-----------
* Every dataclass field is written, including the ones left at their
  defaults, so a file is self-describing and ``basin_from_dict(basin_to_dict(b))
  == b`` for any basin ``b``.
* ``Sector``-keyed dictionaries (``consumption_fraction``,
  ``value_usd_per_m3``) are written with the sector *values* as keys
  (``"municipal"``, ``"industrial"``, ``"agricultural"``, ``"energy"``,
  ``"environment"``); on reading, sector names (``"MUNICIPAL"``) are accepted
  too.
* ``None`` becomes JSON ``null`` (used by ``treaty_allocation_mm3``,
  ``irrigation_efficiency_target`` and ``renewable_share_target``).
* Numbers must be finite: JSON has no ``NaN`` / ``Infinity``, so writing a
  non-finite value raises :class:`ValueError` and so does reading one.
* Each document carries a ``"format"`` tag (``"wefnexus.basin"`` or
  ``"wefnexus.scenario"``) and a ``"format_version"`` so that mistakes such as
  loading a scenario file as a basin are reported clearly.  Both tags are
  optional on input.
* Unknown keys raise :class:`ValueError` (they are almost always typos such
  as ``"polulation"`` and would otherwise silently fall back to defaults).

Units follow the package conventions (see :mod:`wefnexus.models`): water in
**Mm3/yr**, energy in **GWh/yr**, area in **ha**, depth in **mm**, money in
**USD**; the dictionaries carry the same field names and units as the
dataclasses, so nothing is rescaled.

Nothing in this module mutates its inputs: the dictionaries returned by
:func:`basin_to_dict` / :func:`scenario_to_dict` are fresh (deep) copies and
the objects returned by :func:`basin_from_dict` / :func:`scenario_from_dict`
do not share containers with the input dictionary.

References
----------
* Bray, T. (ed.) (2017). *The JavaScript Object Notation (JSON) Data
  Interchange Format*, RFC 8259, IETF.
* Wilkinson, M. D. et al. (2016). The FAIR Guiding Principles for scientific
  data management and stewardship. *Scientific Data* 3, 160018 - the
  motivation for self-describing, machine-readable model inputs.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from wefnexus.models import Basin, Crop, EnergySystem, Riparian, Scenario, Sector, WaterDemand

__all__ = [
    "BASIN_FORMAT",
    "SCENARIO_FORMAT",
    "FORMAT_VERSION",
    "sector_key",
    "parse_sector",
    "basin_to_dict",
    "basin_to_json",
    "basin_from_dict",
    "load_basin",
    "scenario_to_dict",
    "scenario_to_json",
    "scenario_from_dict",
    "load_scenario",
]

#: ``"format"`` tag written into basin documents.
BASIN_FORMAT: str = "wefnexus.basin"

#: ``"format"`` tag written into scenario documents.
SCENARIO_FORMAT: str = "wefnexus.scenario"

#: ``"format_version"`` written into every document.
FORMAT_VERSION: int = 1

#: Document-level keys that are metadata rather than dataclass fields.
_META_KEYS = ("format", "format_version")

# Field name lists derived from the dataclasses (so a field added to a
# model is picked up automatically on *writing*; reading has explicit
# converters below).
_WATER_DEMAND_FIELDS = tuple(f.name for f in dataclasses.fields(WaterDemand))
_CROP_FIELDS = tuple(f.name for f in dataclasses.fields(Crop))
_ENERGY_FIELDS = tuple(f.name for f in dataclasses.fields(EnergySystem))
_RIPARIAN_FIELDS = tuple(f.name for f in dataclasses.fields(Riparian))
_BASIN_FIELDS = tuple(f.name for f in dataclasses.fields(Basin))
_SCENARIO_FIELDS = tuple(f.name for f in dataclasses.fields(Scenario))

_SECTOR_DICT_FIELDS = ("consumption_fraction", "value_usd_per_m3")


# --------------------------------------------------------------------------- #
# Small validation helpers
# --------------------------------------------------------------------------- #
def _type_name(value: Any) -> str:
    return type(value).__name__


def _check_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object (dict), got {_type_name(value)}")
    for k in value:
        if not isinstance(k, str):
            raise ValueError(f"{label} has a non-string key {k!r}")
    return value


def _check_keys(data: Mapping[str, Any], allowed: Sequence[str], label: str, *, extra: Sequence[str] = ()) -> None:
    unknown = sorted(k for k in data if k not in allowed and k not in extra)
    if unknown:
        raise ValueError(f"{label} has unknown key(s) {unknown}; allowed keys: {list(allowed)}")


def _require(data: Mapping[str, Any], key: str, label: str) -> Any:
    if key not in data:
        raise ValueError(f"{label} is missing required key {key!r}")
    return data[key]


def _as_number(value: Any, label: str) -> Union[int, float]:
    """A finite number: an ``int`` stays an ``int``, a ``float`` stays a ``float``.

    Keeping the JSON number type makes ``basin_from_dict(basin_to_dict(b))``
    reproduce ``b`` exactly (the example basin holds integer populations and
    areas).  Booleans and non-numbers are rejected.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number, got {value!r} ({_type_name(value)})")
    if isinstance(value, int):
        return int(value)
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return v


def _as_int(value: Any, label: str) -> int:
    """An integer (a float with an integral value is accepted; bools are not)."""
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer, got {value!r} (bool)")
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and math.isfinite(value) and float(value).is_integer():
        return int(value)
    raise ValueError(f"{label} must be an integer, got {value!r} ({_type_name(value)})")


def _as_optional_number(value: Any, label: str) -> Optional[Union[int, float]]:
    return None if value is None else _as_number(value, label)


def _as_optional_int(value: Any, label: str) -> Optional[int]:
    return None if value is None else _as_int(value, label)


def _as_str(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string, got {value!r} ({_type_name(value)})")
    return value


def _as_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be true or false, got {value!r} ({_type_name(value)})")
    return value


def _as_list(value: Any, label: str) -> List[Any]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a JSON array (list), got {_type_name(value)}")
    return list(value)


def _finite_out(value: Any, label: str) -> Any:
    """Validate a numeric field on output (finite, not bool-typed)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{label} is not finite ({value!r}) and cannot be written to JSON")
        return value
    return value


# --------------------------------------------------------------------------- #
# Sector keys
# --------------------------------------------------------------------------- #
def sector_key(sector: Union[Sector, str]) -> str:
    """JSON key of a sector: its :class:`~wefnexus.models.Sector` value.

    Parameters
    ----------
    sector : Sector or str
        A sector, its value (``"municipal"``) or its name (``"MUNICIPAL"``).

    Returns
    -------
    str
        ``sector.value``, e.g. ``"agricultural"``.

    Raises
    ------
    ValueError
        On an unknown sector.

    Examples
    --------
    >>> sector_key(Sector.AGRICULTURAL), sector_key("ENERGY")
    ('agricultural', 'energy')
    """
    return parse_sector(sector).value


def parse_sector(value: Union[Sector, str]) -> Sector:
    """Read a sector from a JSON key.

    Parameters
    ----------
    value : Sector or str
        A :class:`~wefnexus.models.Sector`, its value (``"municipal"``, as
        written by :func:`basin_to_dict`) or its name (``"MUNICIPAL"``);
        matching is case-insensitive and tolerant of surrounding blanks.

    Returns
    -------
    Sector

    Raises
    ------
    ValueError
        On an unknown sector or a non-string key.

    Examples
    --------
    >>> parse_sector("municipal") is Sector.MUNICIPAL, parse_sector(" Industrial ") is Sector.INDUSTRIAL
    (True, True)
    """
    if isinstance(value, Sector):
        return value
    if not isinstance(value, str):
        raise ValueError(f"sector key must be a string, got {value!r} ({_type_name(value)})")
    key = value.strip().lower()
    for s in Sector:
        if key == s.value or key == s.name.lower():
            return s
    raise ValueError(f"unknown sector {value!r}; expected one of {[s.value for s in Sector]}")


def _sector_dict_out(mapping: Any, label: str) -> Dict[str, float]:
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{label} must be a dict keyed by Sector, got {_type_name(mapping)}")
    out: Dict[str, float] = {}
    for s, v in mapping.items():
        key = sector_key(s)
        if key in out:
            raise ValueError(f"{label} names sector {key!r} twice")
        out[key] = _finite_out(v, f"{label}[{key!r}]")
    return out


def _sector_dict_in(mapping: Any, label: str) -> Dict[Sector, float]:
    data = _check_mapping(mapping, label)
    out: Dict[Sector, float] = {}
    for k, v in data.items():
        s = parse_sector(k)
        if s in out:
            raise ValueError(f"{label} names sector {s.value!r} twice")
        out[s] = _as_number(v, f"{label}[{k!r}]")
    return out


# --------------------------------------------------------------------------- #
# Basin -> dict
# --------------------------------------------------------------------------- #
def _water_demand_to_dict(d: WaterDemand, label: str) -> Dict[str, Any]:
    if not isinstance(d, WaterDemand):
        raise ValueError(f"{label} must be a WaterDemand, got {_type_name(d)}")
    out: Dict[str, Any] = {}
    for name in _WATER_DEMAND_FIELDS:
        value = getattr(d, name)
        if name in _SECTOR_DICT_FIELDS:
            out[name] = _sector_dict_out(value, f"{label}.{name}")
        else:
            out[name] = _finite_out(value, f"{label}.{name}")
    return out


def _crop_to_dict(c: Crop, label: str) -> Dict[str, Any]:
    if not isinstance(c, Crop):
        raise ValueError(f"{label} must be a Crop, got {_type_name(c)}")
    return {name: _finite_out(getattr(c, name), f"{label}.{name}") for name in _CROP_FIELDS}


def _energy_to_dict(e: EnergySystem, label: str) -> Dict[str, Any]:
    if not isinstance(e, EnergySystem):
        raise ValueError(f"{label} must be an EnergySystem, got {_type_name(e)}")
    return {name: _finite_out(getattr(e, name), f"{label}.{name}") for name in _ENERGY_FIELDS}


def _riparian_to_dict(r: Riparian) -> Dict[str, Any]:
    if not isinstance(r, Riparian):
        raise ValueError(f"basin.riparians entries must be Riparian objects, got {_type_name(r)}")
    label = f"riparian {getattr(r, 'name', '?')!r}"
    out: Dict[str, Any] = {}
    for name in _RIPARIAN_FIELDS:
        value = getattr(r, name)
        if name == "demand":
            out[name] = _water_demand_to_dict(value, f"{label}.demand")
        elif name == "crops":
            crops = list(value) if value is not None else []
            out[name] = [_crop_to_dict(c, f"{label}.crops[{i}]") for i, c in enumerate(crops)]
        elif name == "energy":
            out[name] = _energy_to_dict(value, f"{label}.energy")
        else:
            out[name] = _finite_out(value, f"{label}.{name}")
    return out


def basin_to_dict(basin: Basin) -> Dict[str, Any]:
    """Convert a basin to a JSON-compatible dictionary.

    Every field of :class:`~wefnexus.models.Basin`, its
    :class:`~wefnexus.models.Riparian` objects and their nested
    :class:`~wefnexus.models.WaterDemand`, :class:`~wefnexus.models.Crop` and
    :class:`~wefnexus.models.EnergySystem` is written (defaults included).
    ``Sector``-keyed dictionaries use the sector values as keys.  The
    document is tagged with ``"format": "wefnexus.basin"`` and
    ``"format_version"``.

    Parameters
    ----------
    basin : Basin
        The basin (never mutated).  Units are those of the dataclasses:
        water volumes in Mm3/yr, areas in ha, depths in mm, money in USD,
        energy in GWh/yr.

    Returns
    -------
    dict
        ``{"format", "format_version", "name", "riparians": [...],
        "headwater_inflow_mm3", "climate_cv"}``; each riparian dictionary
        holds the riparian's fields with ``"demand"`` (dict), ``"crops"``
        (list of dicts) and ``"energy"`` (dict) nested, and ``"position"``
        (informative; the order of the list is authoritative).  The
        dictionary shares no container with ``basin``.

    Raises
    ------
    ValueError
        If ``basin`` is not a :class:`~wefnexus.models.Basin`, a nested
        object has the wrong type, a sector key is unknown or a numeric
        field is not finite (JSON cannot hold ``NaN`` / ``inf``).

    Examples
    --------
    >>> from wefnexus.data import example_basin
    >>> d = basin_to_dict(example_basin())
    >>> d["format"], d["name"], [r["name"] for r in d["riparians"]]
    ('wefnexus.basin', 'Azura River (stylised)', ['Highland', 'Midland', 'Delta'])
    >>> sorted(d["riparians"][0]["demand"]["consumption_fraction"])
    ['agricultural', 'energy', 'industrial', 'municipal']
    """
    if not isinstance(basin, Basin):
        raise ValueError(f"basin must be a wefnexus.models.Basin, got {_type_name(basin)}")
    out: Dict[str, Any] = {"format": BASIN_FORMAT, "format_version": FORMAT_VERSION}
    for name in _BASIN_FIELDS:
        value = getattr(basin, name)
        if name == "riparians":
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                raise ValueError(f"basin.riparians must be a list of Riparian objects, got {_type_name(value)}")
            out[name] = [_riparian_to_dict(r) for r in value]
        else:
            out[name] = _finite_out(value, f"basin.{name}")
    return out


def _dumps(doc: Mapping[str, Any], indent: Optional[int], label: str) -> str:
    try:
        return json.dumps(doc, indent=indent, allow_nan=False, ensure_ascii=False)
    except ValueError as exc:  # pragma: no cover - guarded earlier by _finite_out
        raise ValueError(f"{label} cannot be written as JSON: {exc}") from None


def _write_text(path: Any, text: str) -> str:
    target = os.fspath(path)
    with open(target, "w", encoding="utf-8") as fh:
        fh.write(text)
        if not text.endswith("\n"):
            fh.write("\n")
    return target


def basin_to_json(basin: Basin, path: Optional[Any] = None, indent: Optional[int] = 2) -> str:
    """Serialise a basin to JSON text, optionally writing it to a file.

    Parameters
    ----------
    basin : Basin
        The basin (never mutated).
    path : str or path-like, optional
        When given, the JSON text (plus a trailing newline) is written to
        this file, overwriting it; the parent directory must exist.
    indent : int or None, optional
        Indentation passed to :func:`json.dumps` (default 2; ``None`` gives
        a single line).

    Returns
    -------
    str
        The JSON document (strict JSON: ``NaN`` / ``Infinity`` are refused,
        non-ASCII names are kept as-is).

    Raises
    ------
    ValueError
        As :func:`basin_to_dict`.
    OSError
        If the file cannot be written.

    Examples
    --------
    >>> from wefnexus.data import example_basin
    >>> text = basin_to_json(example_basin(), indent=None)
    >>> text.startswith('{"format": "wefnexus.basin"')
    True
    """
    text = _dumps(basin_to_dict(basin), indent, "basin")
    if path is not None:
        _write_text(path, text)
    return text


# --------------------------------------------------------------------------- #
# dict -> Basin
# --------------------------------------------------------------------------- #
def _water_demand_from_dict(data: Any, label: str) -> WaterDemand:
    d = _check_mapping(data, label)
    _check_keys(d, _WATER_DEMAND_FIELDS, label)
    kwargs: Dict[str, Any] = {}
    for name in ("municipal", "industrial", "agricultural", "energy", "environmental"):
        if name in d:
            kwargs[name] = _as_number(d[name], f"{label}.{name}")
    for name in _SECTOR_DICT_FIELDS:
        if name in d:
            kwargs[name] = _sector_dict_in(d[name], f"{label}.{name}")
    return WaterDemand(**kwargs)


def _crop_from_dict(data: Any, label: str) -> Crop:
    d = _check_mapping(data, label)
    _check_keys(d, _CROP_FIELDS, label)
    kwargs: Dict[str, Any] = {
        "name": _as_str(_require(d, "name", label), f"{label}.name"),
        "area_ha": _as_number(_require(d, "area_ha", label), f"{label}.area_ha"),
        "kc": _as_number(_require(d, "kc", label), f"{label}.kc"),
        "season_days": _as_int(_require(d, "season_days", label), f"{label}.season_days"),
        "yield_max_t_ha": _as_number(_require(d, "yield_max_t_ha", label), f"{label}.yield_max_t_ha"),
        "ky": _as_number(_require(d, "ky", label), f"{label}.ky"),
        "kcal_per_kg": _as_number(_require(d, "kcal_per_kg", label), f"{label}.kcal_per_kg"),
    }
    if "price_usd_t" in d:
        kwargs["price_usd_t"] = _as_number(d["price_usd_t"], f"{label}.price_usd_t")
    return Crop(**kwargs)


def _energy_from_dict(data: Any, label: str) -> EnergySystem:
    d = _check_mapping(data, label)
    _check_keys(d, _ENERGY_FIELDS, label)
    kwargs = {name: _as_number(d[name], f"{label}.{name}") for name in _ENERGY_FIELDS if name in d}
    return EnergySystem(**kwargs)


_RIPARIAN_REQUIRED = ("name", "population", "gdp_usd", "local_inflow_mm3", "demand")
_RIPARIAN_FLOAT_OPTIONAL = (
    "groundwater_recharge_mm3",
    "groundwater_abstraction_mm3",
    "reservoir_capacity_mm3",
    "reservoir_storage_mm3",
    "reservoir_evaporation_fraction",
    "irrigation_efficiency",
    "et0_mm_day",
    "effective_rainfall_mm",
    "food_demand_kcal_per_capita_day",
    "material_power",
    "bargaining_power",
    "ideational_power",
)


def _riparian_from_dict(data: Any, index: int) -> Riparian:
    label = f"riparians[{index}]"
    d = _check_mapping(data, label)
    _check_keys(d, _RIPARIAN_FIELDS, label)
    for key in _RIPARIAN_REQUIRED:
        _require(d, key, label)
    name = _as_str(d["name"], f"{label}.name")
    label = f"riparian {name!r}"
    kwargs: Dict[str, Any] = {
        "name": name,
        "population": _as_number(d["population"], f"{label}.population"),
        "gdp_usd": _as_number(d["gdp_usd"], f"{label}.gdp_usd"),
        "local_inflow_mm3": _as_number(d["local_inflow_mm3"], f"{label}.local_inflow_mm3"),
        "demand": _water_demand_from_dict(d["demand"], f"{label}.demand"),
    }
    for key in _RIPARIAN_FLOAT_OPTIONAL:
        if key in d:
            kwargs[key] = _as_number(d[key], f"{label}.{key}")
    if "crops" in d:
        crops = _as_list(d["crops"], f"{label}.crops")
        kwargs["crops"] = [_crop_from_dict(c, f"{label}.crops[{i}]") for i, c in enumerate(crops)]
    if "energy" in d:
        kwargs["energy"] = _energy_from_dict(d["energy"], f"{label}.energy")
    if "treaty_allocation_mm3" in d:
        kwargs["treaty_allocation_mm3"] = _as_optional_number(
            d["treaty_allocation_mm3"], f"{label}.treaty_allocation_mm3"
        )
    if "position" in d:
        # informative only: Basin.__post_init__ assigns positions from the
        # list order, but a wrong value is still a sign of a corrupt file.
        pos = _as_int(d["position"], f"{label}.position")
        if pos != index:
            raise ValueError(
                f"{label} has position {pos} but is entry {index} of 'riparians'; "
                "riparians must be listed upstream -> downstream with matching positions (or omit 'position')"
            )
    return Riparian(**kwargs)


def _check_format(data: Mapping[str, Any], expected: str, label: str) -> None:
    fmt = data.get("format")
    if fmt is not None and fmt != expected:
        raise ValueError(f"{label} has format {fmt!r}, expected {expected!r}")
    version = data.get("format_version")
    if version is not None:
        v = _as_int(version, f"{label}.format_version")
        if v > FORMAT_VERSION:
            raise ValueError(
                f"{label} has format_version {v}, newer than the supported version {FORMAT_VERSION}"
            )


def basin_from_dict(data: Mapping[str, Any]) -> Basin:
    """Build a basin from a dictionary written by :func:`basin_to_dict`.

    Only ``"name"`` and ``"riparians"`` are required at the top level and
    only ``name``, ``population``, ``gdp_usd``, ``local_inflow_mm3`` and
    ``demand`` for each riparian; every other field takes its dataclass
    default when absent.  Riparians must be listed upstream -> downstream.

    Parameters
    ----------
    data : dict
        JSON-compatible document (never mutated).  ``Sector``-keyed
        dictionaries may use sector values (``"municipal"``) or names
        (``"MUNICIPAL"``).  An optional ``"format"`` tag must equal
        ``"wefnexus.basin"`` when present.

    Returns
    -------
    Basin
        A new basin sharing no container with ``data``; ``Riparian.position``
        is assigned from the list order.  JSON integers stay ``int`` and JSON
        floats stay ``float`` (``season_days`` and ``position`` must be
        integral), so a written basin is reproduced exactly.

    Raises
    ------
    ValueError
        On a non-dict document, a wrong ``"format"``, missing required keys,
        unknown keys (typos), wrong types (strings for numbers, booleans for
        numbers, non-integer ``season_days``), non-finite numbers, unknown
        sector keys, a ``position`` that disagrees with the list order or
        duplicate riparian names.

    Examples
    --------
    >>> b = basin_from_dict({"name": "Toy", "riparians": [
    ...     {"name": "Up", "population": 1e6, "gdp_usd": 1e9, "local_inflow_mm3": 500.0,
    ...      "demand": {"municipal": 50.0, "environmental": 100.0}}]})
    >>> b.riparians[0].demand.municipal, b.riparians[0].irrigation_efficiency, b.riparians[0].position
    (50.0, 0.5, 0)
    """
    d = _check_mapping(data, "basin document")
    _check_format(d, BASIN_FORMAT, "basin document")
    _check_keys(d, _BASIN_FIELDS, "basin document", extra=_META_KEYS)
    name = _as_str(_require(d, "name", "basin document"), "basin.name")
    riparians_raw = _as_list(_require(d, "riparians", "basin document"), "basin.riparians")
    riparians = [_riparian_from_dict(r, i) for i, r in enumerate(riparians_raw)]
    kwargs: Dict[str, Any] = {"name": name, "riparians": riparians}
    if "headwater_inflow_mm3" in d:
        kwargs["headwater_inflow_mm3"] = _as_number(d["headwater_inflow_mm3"], "basin.headwater_inflow_mm3")
    if "climate_cv" in d:
        kwargs["climate_cv"] = _as_number(d["climate_cv"], "basin.climate_cv")
    return Basin(**kwargs)  # raises ValueError on duplicate names


def _read_json(path: Any, label: str) -> Any:
    target = os.fspath(path)
    with open(target, "r", encoding="utf-8") as fh:
        text = fh.read()

    def _refuse_constant(token: str) -> Any:
        raise ValueError(f"{label} {target!r} contains the non-finite value {token}, which is not valid JSON")

    try:
        return json.loads(text, parse_constant=_refuse_constant)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} {target!r} is not valid JSON: {exc}") from None


def load_basin(path: Any) -> Basin:
    """Read a basin from a JSON file written by :func:`basin_to_json`.

    Parameters
    ----------
    path : str or path-like
        Path of the JSON file (UTF-8).

    Returns
    -------
    Basin

    Raises
    ------
    OSError
        If the file cannot be read (``FileNotFoundError`` for a missing file).
    ValueError
        If the file is not valid JSON, contains ``NaN`` / ``Infinity``, or
        does not describe a basin (see :func:`basin_from_dict`).

    Examples
    --------
    >>> import os, tempfile
    >>> from wefnexus.data import example_basin
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     p = os.path.join(tmp, "basin.json")
    ...     _ = basin_to_json(example_basin(), p)
    ...     load_basin(p) == example_basin()
    True
    """
    return basin_from_dict(_read_json(path, "basin file"))


# --------------------------------------------------------------------------- #
# Scenario <-> dict
# --------------------------------------------------------------------------- #
def scenario_to_dict(scenario: Scenario) -> Dict[str, Any]:
    """Convert a scenario to a JSON-compatible dictionary.

    Parameters
    ----------
    scenario : Scenario
        The scenario (never mutated).  Rates are fractions per year,
        ``*_pct_by_end`` fields are percentages reached in the final year,
        ``drought_years`` are 0-based year offsets.

    Returns
    -------
    dict
        ``{"format": "wefnexus.scenario", "format_version", <every Scenario
        field>}`` with ``drought_years`` copied to a new list.

    Raises
    ------
    ValueError
        If ``scenario`` is not a :class:`~wefnexus.models.Scenario` or a
        numeric field is not finite.

    Examples
    --------
    >>> d = scenario_to_dict(Scenario(name="dry", years=3, drought_years=[1]))
    >>> d["format"], d["name"], d["years"], d["drought_years"], d["irrigation_efficiency_target"]
    ('wefnexus.scenario', 'dry', 3, [1], None)
    """
    if not isinstance(scenario, Scenario):
        raise ValueError(f"scenario must be a wefnexus.models.Scenario, got {_type_name(scenario)}")
    out: Dict[str, Any] = {"format": SCENARIO_FORMAT, "format_version": FORMAT_VERSION}
    for name in _SCENARIO_FIELDS:
        value = getattr(scenario, name)
        if name == "drought_years":
            years = list(value) if value is not None else []
            out[name] = [_as_int(y, f"scenario.drought_years[{i}]") for i, y in enumerate(years)]
        else:
            out[name] = _finite_out(value, f"scenario.{name}")
    return out


def scenario_to_json(scenario: Scenario, path: Optional[Any] = None, indent: Optional[int] = 2) -> str:
    """Serialise a scenario to JSON text, optionally writing it to a file.

    Parameters
    ----------
    scenario : Scenario
        The scenario (never mutated).
    path : str or path-like, optional
        File to write (overwritten); the JSON text gets a trailing newline.
    indent : int or None, optional
        Indentation for :func:`json.dumps` (default 2).

    Returns
    -------
    str
        The JSON document.

    Raises
    ------
    ValueError
        As :func:`scenario_to_dict`.
    OSError
        If the file cannot be written.
    """
    text = _dumps(scenario_to_dict(scenario), indent, "scenario")
    if path is not None:
        _write_text(path, text)
    return text


_SCENARIO_FLOATS = (
    "flow_change_pct_by_end",
    "population_growth_rate",
    "gdp_growth_rate",
    "demand_growth_rate",
    "energy_demand_growth_rate",
    "irrigated_area_change_pct_by_end",
    "drought_severity",
)
_SCENARIO_OPTIONAL_FLOATS = ("irrigation_efficiency_target", "renewable_share_target")
_SCENARIO_INTS = ("start_year", "years", "seed")
_SCENARIO_BOOLS = ("cooperation", "stochastic")
_SCENARIO_STRS = ("name", "allocation_rule", "description")


def scenario_from_dict(data: Mapping[str, Any]) -> Scenario:
    """Build a scenario from a dictionary written by :func:`scenario_to_dict`.

    Every key is optional (the :class:`~wefnexus.models.Scenario` defaults
    apply), so ``{}`` yields the default baseline scenario.

    Parameters
    ----------
    data : dict
        JSON-compatible document (never mutated); an optional ``"format"``
        tag must equal ``"wefnexus.scenario"``.

    Returns
    -------
    Scenario
        A new scenario (``drought_years`` is a fresh list).

    Raises
    ------
    ValueError
        On a non-dict document, a wrong ``"format"``, unknown keys, wrong
        types (``years`` not an integer, ``cooperation`` not a boolean,
        ``drought_years`` not a list of integers, ...) or non-finite numbers.
        The *values* are not range-checked here; the model and scenario
        validators do that when the scenario is used.

    Examples
    --------
    >>> s = scenario_from_dict({"name": "x", "years": 4, "cooperation": False, "drought_years": [2]})
    >>> s.name, s.years, s.cooperation, s.drought_years, s.allocation_rule
    ('x', 4, False, [2], 'treaty')
    """
    d = _check_mapping(data, "scenario document")
    _check_format(d, SCENARIO_FORMAT, "scenario document")
    _check_keys(d, _SCENARIO_FIELDS, "scenario document", extra=_META_KEYS)
    kwargs: Dict[str, Any] = {}
    for key in _SCENARIO_STRS:
        if key in d:
            kwargs[key] = _as_str(d[key], f"scenario.{key}")
    for key in _SCENARIO_INTS:
        if key in d:
            kwargs[key] = _as_int(d[key], f"scenario.{key}")
    for key in _SCENARIO_FLOATS:
        if key in d:
            kwargs[key] = _as_number(d[key], f"scenario.{key}")
    for key in _SCENARIO_OPTIONAL_FLOATS:
        if key in d:
            kwargs[key] = _as_optional_number(d[key], f"scenario.{key}")
    for key in _SCENARIO_BOOLS:
        if key in d:
            kwargs[key] = _as_bool(d[key], f"scenario.{key}")
    if "drought_years" in d:
        years = _as_list(d["drought_years"], "scenario.drought_years")
        kwargs["drought_years"] = [_as_int(y, f"scenario.drought_years[{i}]") for i, y in enumerate(years)]
    return Scenario(**kwargs)


def load_scenario(path: Any) -> Scenario:
    """Read a scenario from a JSON file written by :func:`scenario_to_json`.

    Parameters
    ----------
    path : str or path-like
        Path of the JSON file (UTF-8).

    Returns
    -------
    Scenario

    Raises
    ------
    OSError
        If the file cannot be read.
    ValueError
        If the file is not valid JSON or does not describe a scenario (see
        :func:`scenario_from_dict`).
    """
    return scenario_from_dict(_read_json(path, "scenario file"))
