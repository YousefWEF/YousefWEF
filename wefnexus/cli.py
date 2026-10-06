"""Command-line interface of *wefnexus* (``python -m wefnexus <command> ...``).

The CLI exposes the integrated nexus model, the water-diplomacy tools and
the optimiser of the package as plain-text commands so that a basin can be
analysed from a shell or a notebook without writing Python:

===============  ==============================================================
command          what it does
===============  ==============================================================
``run``          simulate one scenario (:func:`wefnexus.nexus.run_nexus`) and
                 print a per-riparian summary table; ``--csv`` / ``--json``
                 write the full results, ``--plot DIR`` the PNG figures of
                 :mod:`wefnexus.viz` (needs matplotlib)
``allocate``     divide an estate among claims with a bankruptcy rule
                 (:mod:`wefnexus.allocation`), from explicit ``--claims`` or
                 from a basin's river demands / treaty entitlements on a
                 consumptive basis (:func:`wefnexus.diplomacy.compare_allocation_rules`);
                 ``--rule all`` compares every rule
``negotiate``    frame a basin year as a consumptive claims problem, test the
                 gross proposal against each riparian's BATNA and report the
                 ZOPA (:func:`wefnexus.diplomacy.negotiate`)
``compare``      run several library scenarios and print the comparison table
                 of :func:`wefnexus.scenarios.comparison_table`
``pareto``       trace the benefit-equity Pareto front of the allocation LP
                 (:func:`wefnexus.optimize.pareto_front`)
``report``       sustainability assessment (:func:`wefnexus.sustainability.assess`)
                 and the Basins-at-Risk style conflict risk index
                 (:func:`wefnexus.diplomacy.conflict_risk_index`) of a run
``export-basin`` write the stylised example basin (or any basin file) as JSON
===============  ==============================================================

Every command that needs a basin takes ``--basin example`` (the stylised
Azura River of :mod:`wefnexus.data`) or ``--basin path.json`` (a file written
by :func:`wefnexus.io.basin_to_json`).  Scenario arguments accept a library
name of :data:`wefnexus.scenarios.SCENARIOS` or the path of a scenario JSON
file (:func:`wefnexus.io.scenario_to_json`).

Output is plain text without colour; tables are rendered by
:func:`wefnexus.scenarios.format_table`.  Units in the tables are the package
units: water in **Mm3/yr**, energy in **GWh/yr**, money in **USD** (benefit
columns in **MUSD**), indices and ratios dimensionless in ``[0, 1]`` unless
stated, water stress (SDG 6.4.2) in percent.

Exit codes
----------
``0``
    success;
``1``
    the computation produced no result (e.g. an infeasible Pareto front);
``2``
    usage or input error (bad option, unknown scenario or rule, unreadable
    basin file, invalid numbers) - the message goes to standard error.

:func:`main` can be called in-process (``main(["run", "--years", "3"])``)
and returns the exit code instead of exiting; only ``--help`` and
``--version`` raise :class:`SystemExit` (code 0) as argparse does.

References
----------
Aumann & Maschler (1985) (Talmud rule); Fisher & Ury (1981) (BATNA / ZOPA);
Wolf, Yoffe & Giordano (2003) (Basins at Risk); Haimes, Lasdon & Wismer
(1971) (epsilon-constraint Pareto fronts); Hoff (2011) (WEF nexus).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, TextIO, Tuple

import numpy as np

from wefnexus import __version__
from wefnexus import io as _io
from wefnexus.allocation import RULES, RULE_ALIASES, compare_rules, gini, satisfaction
from wefnexus.data import example_basin
from wefnexus.diplomacy import (
    CLAIM_BASES,
    Treaty,
    compare_allocation_rules,
    conflict_risk_index,
    negotiate,
    risk_category,
    treaty_from_basin,
)
from wefnexus.models import Basin, Scenario
from wefnexus.nexus import ALLOCATION_RULES, BASIN_KEY, NexusResult, run_nexus
from wefnexus.optimize import pareto_front
from wefnexus.scenarios import (
    SCENARIOS,
    TABLE_COLUMNS,
    comparison_table,
    format_table,
    get_scenario,
    run_scenarios,
    scenario_names,
    write_table_csv,
)
from wefnexus.sustainability import assess

__all__ = [
    "PROG",
    "EXIT_OK",
    "EXIT_NO_RESULT",
    "EXIT_USAGE",
    "RUN_COLUMNS",
    "COMPARE_DEFAULT_COLUMNS",
    "PARETO_COLUMNS",
    "RISK_COLUMNS",
    "PLOT_FILES",
    "CliError",
    "build_parser",
    "main",
    "load_basin_arg",
    "resolve_scenario",
    "parse_claims",
    "parse_list",
    "canonical_model_rule",
    "canonical_claims_rule",
    "summary_rows",
    "treaty_for_result",
    "conflict_risk_rows",
    "json_safe",
    "write_json",
    "write_plots",
    "cmd_run",
    "cmd_allocate",
    "cmd_negotiate",
    "cmd_compare",
    "cmd_pareto",
    "cmd_report",
    "cmd_export_basin",
]

#: Program name used in usage and error messages.
PROG: str = "wefnexus"

#: Exit code: success.
EXIT_OK: int = 0
#: Exit code: the computation produced no result (infeasible problem).
EXIT_NO_RESULT: int = 1
#: Exit code: usage or input error.
EXIT_USAGE: int = 2

#: ``(summary key, table label)`` pairs of the ``run`` summary table, taken
#: from :meth:`wefnexus.nexus.NexusResult.summary`.
RUN_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("supply_ratio", "supply"),
    ("min_supply_ratio", "min_supply"),
    ("supply_reliability", "reliability"),
    ("water_security", "water"),
    ("energy_security", "energy"),
    ("food_security", "food"),
    ("nexus_index", "nexus"),
    ("env_flow_met_share", "env_met"),
    ("water_stress_sdg642", "stress_pct"),
    ("total_hydropower_gwh", "hydro_GWh"),
    ("food_self_sufficiency", "food_ss"),
    ("final_storage_mm3", "storage_Mm3"),
)

#: Columns of :data:`wefnexus.scenarios.TABLE_COLUMNS` printed by ``compare``
#: by default (``--columns all`` prints every column).
COMPARE_DEFAULT_COLUMNS: Tuple[str, ...] = (
    "scenario",
    "riparian",
    "allocation_rule",
    "mean_supply_ratio",
    "min_supply_ratio",
    "supply_reliability",
    "mean_nexus_index",
    "mean_water_security",
    "mean_energy_security",
    "mean_food_security",
    "years_env_flow_unmet",
    "total_hydropower_gwh",
    "food_self_sufficiency",
    "final_storage_mm3",
    "equity_index",
)

#: ``(front key, table label)`` pairs of the ``pareto`` table.
PARETO_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("min_supply_ratio", "epsilon"),
    ("achieved_min_supply_ratio", "min_supply"),
    ("mean_supply_ratio", "mean_supply"),
    ("gini", "gini"),
    ("total_benefit_musd", "benefit_MUSD"),
    ("total_withdrawal_mm3", "withdrawal_Mm3"),
    ("outflow_to_sea_mm3", "outflow_Mm3"),
)

#: Component keys of :func:`wefnexus.diplomacy.conflict_risk_index` and their
#: table labels, in print order.
RISK_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("water_stress", "stress"),
    ("dependency", "dependency"),
    ("power_asymmetry", "asymmetry"),
    ("institutional", "institutional"),
    ("conflict_history", "conflict"),
    ("variability", "variability"),
    ("environmental", "env_short"),
    ("dam_filling", "dam_fill"),
)

#: File names of the PNG figures that ``run --plot DIR`` writes into ``DIR``
#: (see :func:`write_plots`), in write order: the annual supply ratio, the
#: security / nexus indices over the horizon and the radar profile of the
#: horizon means.
PLOT_FILES: Tuple[str, ...] = ("supply_ratio.png", "nexus_indices.png", "nexus_radar.png")

_RUN_LABELS = dict(RUN_COLUMNS)


# --------------------------------------------------------------------------- #
# Errors and parser plumbing
# --------------------------------------------------------------------------- #
class CliError(Exception):
    """Usage or input error that :func:`main` turns into exit code 2.

    Attributes
    ----------
    message : str
        Human-readable description.
    usage : str
        Usage line(s) of the (sub)parser that raised it (may be empty).
    prog : str
        Program / subcommand name for the ``<prog>: error:`` prefix.
    """

    def __init__(self, message: str, usage: str = "", prog: str = PROG) -> None:
        super().__init__(message)
        self.message = message
        self.usage = usage
        self.prog = prog


class _Parser(argparse.ArgumentParser):
    """ArgumentParser that raises :class:`CliError` instead of exiting on errors."""

    def error(self, message: str) -> None:  # type: ignore[override]
        raise CliError(message, self.format_usage(), self.prog)


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return value


def _points_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer >= 2, got {text!r}") from None
    if value < 2:
        raise argparse.ArgumentTypeError(f"expected an integer >= 2, got {value}")
    return value


def _nonneg_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a non-negative number, got {text!r}") from None
    if not math.isfinite(value) or value < 0.0:
        raise argparse.ArgumentTypeError(f"expected a finite non-negative number, got {text!r}")
    return value


def _fraction(text: str) -> float:
    value = _nonneg_float(text)
    if value > 1.0:
        raise argparse.ArgumentTypeError(f"expected a number in [0, 1], got {text!r}")
    return value


def _model_rule_arg(text: str) -> str:
    try:
        return canonical_model_rule(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _claims_rule_arg(text: str) -> str:
    try:
        return canonical_claims_rule(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _single_claims_rule_arg(text: str) -> str:
    rule = _claims_rule_arg(text)
    if rule == "all":
        raise argparse.ArgumentTypeError(f"expected a single rule, one of {sorted(RULES)}, got 'all'")
    return rule


def _claims_arg(text: str) -> Dict[str, float]:
    try:
        return parse_claims(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


# --------------------------------------------------------------------------- #
# Argument helpers (public, testable without argparse)
# --------------------------------------------------------------------------- #
def canonical_model_rule(rule: str) -> str:
    """Canonical allocation-rule name accepted by :class:`wefnexus.nexus.NexusModel`.

    Parameters
    ----------
    rule : str
        ``"treaty"``, ``"upstream_priority"`` or any bankruptcy rule of
        :data:`wefnexus.allocation.RULES` / alias of
        :data:`wefnexus.allocation.RULE_ALIASES`; case-insensitive, hyphens
        and blanks read as underscores.

    Returns
    -------
    str
        A member of :data:`wefnexus.nexus.ALLOCATION_RULES`.

    Raises
    ------
    ValueError
        On an unknown rule.

    Examples
    --------
    >>> canonical_model_rule("Contested-Garment"), canonical_model_rule("TREATY")
    ('talmud', 'treaty')
    """
    if not isinstance(rule, str) or not rule.strip():
        raise ValueError(f"allocation rule must be a non-empty string; expected one of {list(ALLOCATION_RULES)}")
    key = rule.strip().lower().replace("-", "_").replace(" ", "_")
    key = RULE_ALIASES.get(key, key)
    if key not in ALLOCATION_RULES:
        raise ValueError(f"unknown allocation rule {rule!r}; expected one of {list(ALLOCATION_RULES)}")
    return key


def canonical_claims_rule(rule: str) -> str:
    """Canonical name of a claims (bankruptcy) rule, or ``"all"``.

    Parameters
    ----------
    rule : str
        A key or alias of :data:`wefnexus.allocation.RULES`, or ``"all"``
        to compare every rule.

    Returns
    -------
    str

    Raises
    ------
    ValueError
        On an unknown rule (``"treaty"`` is not a claims rule).

    Examples
    --------
    >>> canonical_claims_rule("CEA"), canonical_claims_rule(" all ")
    ('cea', 'all')
    """
    if not isinstance(rule, str) or not rule.strip():
        raise ValueError(f"rule must be a non-empty string; expected 'all' or one of {sorted(RULES)}")
    key = rule.strip().lower().replace("-", "_").replace(" ", "_")
    if key == "all":
        return "all"
    key = RULE_ALIASES.get(key, key)
    if key not in RULES:
        raise ValueError(f"unknown allocation rule {rule!r}; expected 'all' or one of {sorted(RULES)}")
    return key


def load_basin_arg(spec: str) -> Basin:
    """Resolve a ``--basin`` argument.

    Parameters
    ----------
    spec : str
        ``"example"`` (case-insensitive) for a fresh copy of the stylised
        Azura River basin of :func:`wefnexus.data.example_basin`, otherwise
        the path of a basin JSON file read with
        :func:`wefnexus.io.load_basin`.

    Returns
    -------
    Basin
        A new basin object on every call.

    Raises
    ------
    ValueError
        On an empty argument, a missing file or an invalid basin file.

    Examples
    --------
    >>> load_basin_arg("example").names()
    ['Highland', 'Midland', 'Delta']
    """
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError("--basin must be 'example' or the path of a basin JSON file")
    if spec.strip().lower() == "example":
        return example_basin()
    if not os.path.isfile(spec):
        raise ValueError(f"basin file {spec!r} not found (use --basin example for the built-in basin)")
    return _io.load_basin(spec)


def _scenario_key(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def resolve_scenario(spec: str, years: Optional[int] = None) -> Scenario:
    """Resolve a ``--scenario`` argument to a :class:`~wefnexus.models.Scenario`.

    Parameters
    ----------
    spec : str
        A library name of :data:`wefnexus.scenarios.SCENARIOS`
        (case-insensitive, ``-``/blank read as ``_``) or the path of a
        scenario JSON file written by :func:`wefnexus.io.scenario_to_json`.
    years : int, optional
        Horizon override (>= 1).  For a library name it is passed to the
        factory; for a file it replaces the file's ``years``.

    Returns
    -------
    Scenario

    Raises
    ------
    ValueError
        If ``spec`` is neither a library name nor an existing file, or the
        file / years are invalid.

    Examples
    --------
    >>> resolve_scenario("Climate-Change", years=3).years
    3
    """
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError(f"--scenario must be one of {scenario_names()} or the path of a scenario JSON file")
    if years is not None and (isinstance(years, bool) or not isinstance(years, int) or years < 1):
        raise ValueError(f"years must be a positive integer, got {years!r}")
    key = _scenario_key(spec)
    if key in SCENARIOS:
        return get_scenario(key, years=years) if years is not None else get_scenario(key)
    if os.path.isfile(spec):
        sc = _io.load_scenario(spec)
        return dataclasses.replace(sc, years=years) if years is not None else sc
    raise ValueError(
        f"unknown scenario {spec!r}; expected one of {scenario_names()} or the path of a scenario JSON file"
    )


def parse_list(text: str) -> List[str]:
    """Split a comma-separated option value into non-empty, stripped items.

    Parameters
    ----------
    text : str
        E.g. ``"baseline, growth,,climate_change"``.

    Returns
    -------
    list of str
        ``["baseline", "growth", "climate_change"]``.

    Raises
    ------
    ValueError
        If no item remains.

    Examples
    --------
    >>> parse_list(" a, b ,,c ")
    ['a', 'b', 'c']
    """
    if not isinstance(text, str):
        raise ValueError(f"expected a comma-separated string, got {text!r}")
    items = [t.strip() for t in text.split(",")]
    items = [t for t in items if t]
    if not items:
        raise ValueError("expected a non-empty comma-separated list")
    return items


def parse_claims(text: str) -> Dict[str, float]:
    """Parse a ``--claims`` option.

    Parameters
    ----------
    text : str
        Comma-separated claims, either named (``"a=100,b=200,c=300"``) or
        bare numbers (``"100,200,300"``, named ``c1``, ``c2``, ...).  Mixing
        the two forms is an error.  Claims are in the caller's unit (Mm3/yr
        throughout the package) and must be finite and non-negative.

    Returns
    -------
    dict
        ``{claimant: claim}`` in input order.

    Raises
    ------
    ValueError
        On an empty list, a malformed item, a duplicate name, a non-numeric,
        negative or non-finite value.

    Examples
    --------
    >>> parse_claims("a=100, b=200,c=300")
    {'a': 100.0, 'b': 200.0, 'c': 300.0}
    >>> parse_claims("10,20")
    {'c1': 10.0, 'c2': 20.0}
    """
    items = parse_list(text)
    named = ["=" in t for t in items]
    if any(named) and not all(named):
        raise ValueError("claims must be either all named (a=1,b=2) or all bare numbers (1,2)")
    out: Dict[str, float] = {}
    for i, item in enumerate(items):
        if named[i]:
            name, _, raw = item.partition("=")
            name = name.strip()
            raw = raw.strip()
            if not name:
                raise ValueError(f"claim {item!r} has an empty name; use name=value")
        else:
            name, raw = f"c{i + 1}", item
        try:
            value = float(raw)
        except ValueError:
            raise ValueError(f"claim {item!r}: {raw!r} is not a number") from None
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"claim {item!r}: value must be finite and non-negative")
        if name in out:
            raise ValueError(f"duplicate claimant {name!r}")
        out[name] = value
    return out


# --------------------------------------------------------------------------- #
# Result helpers
# --------------------------------------------------------------------------- #
def summary_rows(result: NexusResult) -> List[Dict[str, Any]]:
    """Rows of the ``run`` summary table (one per riparian plus ``BASIN``).

    Parameters
    ----------
    result : NexusResult
        A finished simulation.

    Returns
    -------
    list of dict
        Each row has ``"riparian"`` and the labels of :data:`RUN_COLUMNS`
        (``supply``, ``min_supply``, ``reliability``, ``water``, ``energy``,
        ``food``, ``nexus``, ``env_met``, ``stress_pct``, ``hydro_GWh``
        (sum over the horizon), ``food_ss``, ``storage_Mm3`` (final)); the
        last row is the basin aggregate.

    Raises
    ------
    ValueError
        On an empty result.
    """
    if not isinstance(result, NexusResult):
        raise ValueError(f"result must be a wefnexus.nexus.NexusResult, got {type(result).__name__}")
    summary = result.summary()
    rows: List[Dict[str, Any]] = []
    for name in result.riparian_names() + [BASIN_KEY]:
        s = summary[name]
        row: Dict[str, Any] = {"riparian": name}
        for key, label in RUN_COLUMNS:
            row[label] = s.get(key)
        rows.append(row)
    return rows


def treaty_for_result(basin: Basin, result: NexusResult) -> Optional[Treaty]:
    """The agreement in force during a run, for the conflict risk index.

    Parameters
    ----------
    basin : Basin
        The simulated basin (never mutated).
    result : NexusResult
        The run; its ``allocation_rule`` decides:

        * ``"upstream_priority"`` (no caps / non-cooperative) - no treaty;
        * ``"treaty"`` - the riparians' fixed ``treaty_allocation_mm3`` via
          :func:`wefnexus.diplomacy.treaty_from_basin` (``None`` if no
          riparian has an entitlement);
        * a bankruptcy rule - a *variable* agreement with drought provisions
          (the rule re-divides every year's flow among all riparians), with
          the riparians' fixed volumes, if any, as reference entitlements.

    Returns
    -------
    Treaty or None
    """
    if not isinstance(basin, Basin):
        raise ValueError(f"basin must be a wefnexus.models.Basin, got {type(basin).__name__}")
    if not isinstance(result, NexusResult):
        raise ValueError(f"result must be a wefnexus.nexus.NexusResult, got {type(result).__name__}")
    rule = result.allocation_rule
    if rule == "upstream_priority":
        return None
    if rule == "treaty":
        return treaty_from_basin(basin)
    name = f"{basin.name} {rule} sharing arrangement"
    treaty = treaty_from_basin(basin, name=name, variable_allocation=True, drought_provisions=True)
    if treaty is not None:
        return treaty
    if not basin.riparians:
        return None
    return Treaty(name=name, parties=basin.names(), variable_allocation=True, drought_provisions=True)


def conflict_risk_rows(basin: Basin, result: NexusResult, treaty: Optional[Treaty] = None) -> List[Dict[str, Any]]:
    """Conflict risk index of every simulated year.

    Parameters
    ----------
    basin : Basin
        The simulated basin (never mutated).
    result : NexusResult
        A run that carries balances.
    treaty : Treaty, optional
        Agreement in force (default: :func:`treaty_for_result`).

    Returns
    -------
    list of dict
        One row per year: ``year``, ``score``, ``category`` and the eight
        component labels of :data:`RISK_COLUMNS`.

    Raises
    ------
    ValueError
        If the result carries no balances or the basin does not match.
    """
    if not isinstance(result, NexusResult):
        raise ValueError(f"result must be a wefnexus.nexus.NexusResult, got {type(result).__name__}")
    if not result.balances:
        raise ValueError("the result carries no routed balances; cannot compute the conflict risk index")
    if treaty is None:
        treaty = treaty_for_result(basin, result)
    rows: List[Dict[str, Any]] = []
    for year, balance in zip(result.years, result.balances):
        cri = conflict_risk_index(basin, balance, treaty)
        row: Dict[str, Any] = {"year": int(year), "score": float(cri["score"]), "category": str(cri["category"])}
        for key, label in RISK_COLUMNS:
            row[label] = float(cri["components"][key])
        rows.append(row)
    return rows


def json_safe(obj: Any) -> Any:
    """Convert a result object into strict-JSON-compatible data.

    Parameters
    ----------
    obj : object
        Nested dicts / lists / tuples / sets of numbers, strings, booleans,
        ``None``, enums (:class:`~wefnexus.models.Sector` keys become their
        values), NumPy scalars and arrays, :class:`~wefnexus.models.Scenario`
        (via :func:`wefnexus.io.scenario_to_dict`) and other dataclasses
        (via :func:`dataclasses.asdict`).  Anything else is rendered with
        ``str``.

    Returns
    -------
    object
        Data that :func:`json.dumps` accepts with ``allow_nan=False``:
        non-finite floats (``inf`` entitlements meaning "no cap", undefined
        water stress) become ``None``; non-string keys become strings.

    Examples
    --------
    >>> from wefnexus.models import Sector
    >>> json_safe({Sector.MUNICIPAL: float("inf"), "x": (1, 2.5), "b": True})
    {'municipal': None, 'x': [1, 2.5], 'b': True}
    """
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, Enum):
        return json_safe(obj.value)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.ndarray):
        return [json_safe(v) for v in obj.tolist()]
    if isinstance(obj, Scenario):
        return _io.scenario_to_dict(obj)
    if isinstance(obj, Mapping):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            key = k.value if isinstance(k, Enum) else k
            out[key if isinstance(key, str) else str(key)] = json_safe(v)
        return out
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [json_safe(v) for v in obj]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return json_safe(dataclasses.asdict(obj))
    return str(obj)


def write_json(path: Any, payload: Any) -> str:
    """Write ``payload`` (through :func:`json_safe`) as indented JSON.

    Parameters
    ----------
    path : str or path-like
        Destination file, overwritten.
    payload : object
        Anything :func:`json_safe` accepts.

    Returns
    -------
    str
        The path written.

    Raises
    ------
    OSError
        If the file cannot be written.
    """
    target = os.fspath(path)
    text = json.dumps(json_safe(payload), indent=2, allow_nan=False, ensure_ascii=False)
    with open(target, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.write("\n")
    return target


def _close_figure(fig: Any) -> None:
    """Close a matplotlib figure (imported lazily); failures are ignored."""
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:  # pragma: no cover - defensive
        pass


def write_plots(result: NexusResult, directory: Any) -> List[str]:
    """Write the standard PNG figures of a run into ``directory`` (``run --plot``).

    The figures are those of :mod:`wefnexus.viz`; matplotlib is imported
    lazily, so the core package does not need it.  In write order
    (:data:`PLOT_FILES`):

    * ``supply_ratio.png`` - :func:`wefnexus.viz.plot_supply_ratio`: the
      annual water supply ratio ``1 - deficit / demand`` (dimensionless,
      ``[0, 1]``) of every riparian and the basin, the complement of the
      relative shortfall of Hashimoto, Stedinger & Loucks (1982);
    * ``nexus_indices.png`` - :func:`wefnexus.viz.plot_nexus_indices`: the
      water, energy and food security indices and the WEF nexus index
      (``[0, 1]``; Hoff 2011) over the horizon;
    * ``nexus_radar.png`` - :func:`wefnexus.viz.plot_nexus_radar`: the radar
      profile of the horizon-mean indicators of
      :meth:`wefnexus.nexus.NexusResult.summary`.

    Every figure is closed after it is saved, so a long session does not
    accumulate open figures.  ``result`` is never mutated and the directory
    is only created once matplotlib is known to be importable.

    Parameters
    ----------
    result : NexusResult
        A finished simulation.
    directory : str or path-like
        Destination directory; created with its parents when missing.

    Returns
    -------
    list of str
        The paths written, ``os.path.join(directory, name)`` for each name of
        :data:`PLOT_FILES`.

    Raises
    ------
    ValueError
        If ``result`` is not a :class:`~wefnexus.nexus.NexusResult` or
        ``directory`` is empty.
    ImportError
        If matplotlib is not installed (the message says how to install it).
    OSError
        If the directory cannot be created (e.g. the path is an existing
        file) or a figure cannot be written.
    """
    if not isinstance(result, NexusResult):
        raise ValueError(f"result must be a wefnexus.nexus.NexusResult, got {type(result).__name__}")
    if directory is None or isinstance(directory, bool):
        raise ValueError("--plot must be a non-empty directory path")
    try:
        target = os.fsdecode(directory)
    except TypeError:
        raise ValueError(f"--plot must be a directory path, got {type(directory).__name__}") from None
    if not target.strip():
        raise ValueError("--plot must be a non-empty directory path")
    try:
        import matplotlib  # noqa: F401  (lazy: the core package never needs it)
    except ImportError as exc:
        raise ImportError(
            "--plot needs matplotlib; install it with `pip install matplotlib` or `pip install wefnexus[viz]`"
        ) from exc
    from wefnexus import viz

    os.makedirs(target, exist_ok=True)
    plotters = (
        (PLOT_FILES[0], lambda path: viz.plot_supply_ratio(result, save=path)),
        (PLOT_FILES[1], lambda path: viz.plot_nexus_indices(result, save=path)),
        (PLOT_FILES[2], lambda path: viz.plot_nexus_radar(result.summary(), save=path)),
    )
    written: List[str] = []
    for filename, plot in plotters:
        path = os.path.join(target, filename)
        fig = plot(path)
        _close_figure(fig)
        written.append(path)
    return written


def _fmt(value: float, digits: int = 1) -> str:
    """Compact number for free-text lines."""
    if value is None:
        return "-"
    v = float(value)
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    if math.isnan(v):
        return "nan"
    return f"{v:,.{digits}f}"


def _years_text(result: NexusResult) -> str:
    if not result.years:
        return "no years"
    return f"{result.years[0]}-{result.years[-1]} ({result.n_years} years)"


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_run(args: argparse.Namespace, out: TextIO) -> int:
    """``run``: simulate one scenario and print the summary table.

    Parameters
    ----------
    args : argparse.Namespace
        ``basin``, ``scenario``, ``years``, ``rule``, ``refill``, ``csv``,
        ``json``, ``plot`` (directory for the PNG figures of
        :func:`write_plots`, or ``None``).
    out : text stream
        Where to print.

    Returns
    -------
    int
        :data:`EXIT_OK`.  When ``--plot`` is given but matplotlib is not
        installed, the summary is still printed, the figures are skipped with
        a hint naming the missing package and the exit code stays
        :data:`EXIT_OK`; an unusable directory (empty, or an existing file)
        is an error (exit code 2 through :func:`main`).
    """
    basin = load_basin_arg(args.basin)
    scenario = resolve_scenario(args.scenario, args.years)
    kw: Dict[str, Any] = {"reservoir_refill_fraction": args.refill}
    if args.rule is not None:
        kw["allocation_rule"] = args.rule
    result = run_nexus(basin, scenario, **kw)
    summary = result.summary()
    basin_row = summary[BASIN_KEY]

    print(
        f"Basin: {result.basin_name} | scenario: {scenario.name} | years: {_years_text(result)} "
        f"| allocation rule: {result.allocation_rule}",
        file=out,
    )
    if scenario.description:
        print(f"  {scenario.description}", file=out)
    ff = result.flow_factors
    if ff:
        print(
            f"flow factors: min {min(ff):.3f} | mean {sum(ff) / len(ff):.3f} | max {max(ff):.3f}"
            f"{' (stochastic)' if scenario.stochastic else ''}",
            file=out,
        )
    print(format_table(summary_rows(result)), file=out)
    mbe = basin_row.get("mass_balance_error_mm3")
    print(
        f"natural flow {_fmt(basin_row.get('natural_flow_mm3'))} Mm3/yr | outflow to sea "
        f"{_fmt(basin_row.get('outflow_to_sea_mm3'))} Mm3/yr | equity index {basin_row['equity_index']:.3f} "
        f"| mass balance error {mbe if mbe is None else format(mbe, '.2e')} Mm3",
        file=out,
    )
    if args.csv:
        print(f"wrote CSV: {result.to_csv(args.csv)}", file=out)
    if args.json:
        payload = {
            "basin": result.basin_name,
            "scenario": scenario,
            "allocation_rule": result.allocation_rule,
            "years": list(result.years),
            "flow_factors": list(result.flow_factors),
            "summary": summary,
            "records": result.to_records(),
        }
        print(f"wrote JSON: {write_json(args.json, payload)}", file=out)
    if args.plot is not None:
        try:
            paths = write_plots(result, args.plot)
        except ImportError as exc:
            print(f"figures skipped: {exc}", file=out)
        else:
            print(f"wrote figures: {', '.join(paths)}", file=out)
    return EXIT_OK


def _allocation_table(
    rules: Mapping[str, Mapping[str, Any]],
    claimants: Sequence[str],
    extra: Sequence[Any] = (),
) -> str:
    """Side-by-side awards per rule; ``extra`` lists result keys or ``(column, key)`` pairs."""
    rows = []
    for rule, info in rules.items():
        row: Dict[str, Any] = {"rule": rule}
        for name in claimants:
            row[name] = info["awards"][name]
        row["total"] = info["total_awarded_mm3"]
        row["gini"] = info["gini"]
        row["min_satisfaction"] = info["min_satisfaction"]
        for item in extra:
            column, key = (item, item) if isinstance(item, str) else item
            row[column] = info.get(key)
        rows.append(row)
    return format_table(rows)


def cmd_allocate(args: argparse.Namespace, out: TextIO) -> int:
    """``allocate``: divide an estate among claims with a bankruptcy rule.

    With ``--claims`` the problem is explicit (``--estate`` required); with
    ``--basin`` the gross claims are the riparians' river demands
    (``--claims-basis demand``, default) or treaty entitlements
    (``--claims-basis treaty``), the rules divide the consumptive estate -
    natural flow times ``--flow-factor`` minus the in-stream requirement at
    the basin outlet (:func:`wefnexus.diplomacy.bankruptcy_estate`;
    ``--estate`` overrides) - among the consumptive claims, and each rule's
    awards are converted into gross withdrawal caps and routed through the
    basin (:func:`wefnexus.diplomacy.compare_allocation_rules`).

    Parameters
    ----------
    args : argparse.Namespace
        ``rule``, ``claims``, ``basin``, ``estate``, ``flow_factor``,
        ``claims_basis``, ``include_storage``, ``json``.
    out : text stream

    Returns
    -------
    int
        :data:`EXIT_OK`.
    """
    rule = args.rule
    rules_arg = None if rule == "all" else [rule]
    consumptive: Optional[Dict[str, float]] = None
    if args.claims is not None:
        if args.estate is None:
            raise CliError("--estate is required with --claims", prog=f"{PROG} allocate")
        if getattr(args, "claims_basis", None) is not None:
            raise CliError("--claims-basis is only valid with --basin", prog=f"{PROG} allocate")
        if getattr(args, "include_storage", False):
            raise CliError("--include-storage is only valid with --basin", prog=f"{PROG} allocate")
        claims = dict(args.claims)
        estate = float(args.estate)
        awards_by_rule = compare_rules(estate, claims, rules_arg)
        table: Dict[str, Dict[str, Any]] = {}
        for name, awards in awards_by_rule.items():
            sat = satisfaction(awards, claims)
            table[name] = {
                "awards": dict(awards),
                "total_awarded_mm3": float(sum(awards.values())),
                "gini": gini(awards),
                "gini_satisfaction": gini(list(sat.values())),
                "satisfaction": sat,
                "min_satisfaction": float(min(sat.values())) if sat else 1.0,
            }
        source = "explicit claims"
        extra: Tuple[Any, ...] = ()
    else:
        basin = load_basin_arg(args.basin)
        basis = args.claims_basis or "demand"
        table = compare_allocation_rules(
            basin,
            args.flow_factor,
            rules_arg,
            estate=args.estate,
            claim_basis=basis,
            include_storage=bool(args.include_storage),
        )
        first = next(iter(table.values()))
        claims = dict(first["claims"])
        consumptive = dict(first["consumptive_claims"])
        estate = float(first["estate"])
        source = (
            f"basin {basin.name!r} at flow factor {args.flow_factor:g}, {basis} claims, "
            f"storage {'included' if args.include_storage else 'excluded'}"
        )
        extra = (("c-total", "total_consumptive_award_mm3"), "outflow_to_sea_mm3", "env_flow_met_share")

    total_claims = float(sum(claims.values()))
    if consumptive is None:
        claims_text = f"total claims {_fmt(total_claims)}"
        shortfall = max(total_claims - estate, 0.0)
    else:
        total_consumptive = float(sum(consumptive.values()))
        claims_text = f"total claims {_fmt(total_claims)} (consumptive {_fmt(total_consumptive)})"
        shortfall = max(total_consumptive - estate, 0.0)
    print(
        f"Claims problem ({source}): estate {_fmt(estate)} | {claims_text} "
        f"| shortfall {_fmt(shortfall)} | claimants {len(claims)}",
        file=out,
    )
    claimants = list(claims)
    if rule == "all":
        print(_allocation_table(table, claimants, extra), file=out)
    else:
        info = table[rule]
        rows = []
        for name in claimants:
            row: Dict[str, Any] = {"claimant": name, "claim": claims[name]}
            if consumptive is not None:
                row["c-claim"] = consumptive[name]
                row["c-award"] = info["consumptive_awards"][name]
            row["award"] = info["awards"][name]
            row["satisfaction"] = info["satisfaction"][name]
            if "withdrawals" in info:
                row["routed_withdrawal"] = info["withdrawals"][name]
            rows.append(row)
        print(f"rule: {rule}", file=out)
        print(format_table(rows), file=out)
        line = f"total awarded {_fmt(info['total_awarded_mm3'])}"
        if consumptive is not None:
            line += f" (consumptive {_fmt(info['total_consumptive_award_mm3'])})"
        line += (
            f" | gini {info['gini']:.3f} "
            f"| gini (satisfaction) {info['gini_satisfaction']:.3f} | min satisfaction {info['min_satisfaction']:.3f}"
        )
        if "outflow_to_sea_mm3" in info:
            line += (
                f" | outflow to sea {_fmt(info['outflow_to_sea_mm3'])} Mm3"
                f" | env flow met share {info['env_flow_met_share']:.2f}"
            )
        print(line, file=out)
    if args.json:
        payload: Dict[str, Any] = {"source": source, "estate": estate, "claims": claims, "rules": table}
        if consumptive is not None:
            payload["consumptive_claims"] = consumptive
            payload["claim_basis"] = basis
            payload["include_storage"] = bool(args.include_storage)
        print(f"wrote JSON: {write_json(args.json, payload)}", file=out)
    return EXIT_OK


def cmd_negotiate(args: argparse.Namespace, out: TextIO) -> int:
    """``negotiate``: consumptive claims-problem proposal versus BATNAs and the ZOPA.

    Parameters
    ----------
    args : argparse.Namespace
        ``basin``, ``flow_factor``, ``rule``, ``estate``, ``claims``
        (claim basis), ``include_storage``, ``json``.
    out : text stream

    Returns
    -------
    int
        :data:`EXIT_OK`.
    """
    basin = load_basin_arg(args.basin)
    kw: Dict[str, Any] = {"claim_basis": args.claims, "include_storage": bool(args.include_storage)}
    if args.estate is not None:
        kw["estate"] = args.estate
    res = negotiate(basin, args.flow_factor, args.rule, **kw)
    print(
        f"Negotiation: {basin.name} | flow factor {res['flow_factor']:g} | rule {res['rule']} "
        f"| claims {res['claim_basis']} | storage {'included' if res['include_storage'] else 'excluded'}",
        file=out,
    )
    print(
        f"natural flow {_fmt(res['natural_flow_mm3'])} Mm3 | environmental flows {_fmt(res['environmental_flow_mm3'])} Mm3 "
        f"(reserve at the outlet {_fmt(res['environmental_reserve_mm3'])} Mm3) "
        f"| estate {_fmt(res['estate'])} Mm3 | total claims {_fmt(sum(res['claims'].values()))} Mm3 "
        f"(consumptive {_fmt(sum(res['consumptive_claims'].values()))} Mm3)",
        file=out,
    )
    rows = []
    for name in basin.names():
        rows.append(
            {
                "riparian": name,
                "claim": res["claims"][name],
                "c-claim": res["consumptive_claims"][name],
                "c-award": res["consumptive_awards"][name],
                "proposal": res["proposal"][name],
                "batna": res["batna"][name],
                "satisfaction": res["satisfaction"][name],
                "acceptable": bool(res["acceptable"][name]),
                "routed": res["routed_withdrawals"][name],
            }
        )
    print(format_table(rows), file=out)
    print(
        f"total awarded {_fmt(res['total_awarded_mm3'])} Mm3 (consumptive {_fmt(res['total_consumptive_award_mm3'])} Mm3) "
        f"| gini {res['gini']:.3f} | gini (satisfaction) {res['gini_satisfaction']:.3f} "
        f"| outflow to sea {_fmt(res['outflow_to_sea_mm3'])} Mm3 | env flow met share {res['env_flow_met_share']:.2f}",
        file=out,
    )
    if res["zopa"]:
        print("ZOPA: yes - every riparian is awarded at least its BATNA withdrawal", file=out)
    else:
        rejecting = [n for n, ok in res["acceptable"].items() if not ok]
        print(f"ZOPA: no - proposal below the BATNA of {', '.join(rejecting)}", file=out)
    if args.json:
        print(f"wrote JSON: {write_json(args.json, res)}", file=out)
    return EXIT_OK


def _compare_columns(spec: Optional[str]) -> List[str]:
    if spec is None:
        return list(COMPARE_DEFAULT_COLUMNS)
    items = parse_list(spec)
    if len(items) == 1 and items[0].lower() == "all":
        return list(TABLE_COLUMNS)
    unknown = [c for c in items if c not in TABLE_COLUMNS]
    if unknown:
        raise ValueError(f"unknown column(s) {unknown}; available: {list(TABLE_COLUMNS)}")
    cols = []
    for c in items:
        if c not in cols:
            cols.append(c)
    return cols


def cmd_compare(args: argparse.Namespace, out: TextIO) -> int:
    """``compare``: run several scenarios and print the comparison table.

    Parameters
    ----------
    args : argparse.Namespace
        ``basin``, ``scenarios`` (comma list or ``None`` for all), ``years``,
        ``rule``, ``columns``, ``csv``, ``json``.
    out : text stream

    Returns
    -------
    int
        :data:`EXIT_OK`.
    """
    basin = load_basin_arg(args.basin)
    specs = parse_list(args.scenarios) if args.scenarios else list(scenario_names())
    columns = _compare_columns(args.columns)
    scenarios: Dict[str, Scenario] = {}
    for spec in specs:
        sc = resolve_scenario(spec, args.years)
        key = sc.name if _scenario_key(spec) not in SCENARIOS else _scenario_key(spec)
        if key in scenarios:
            raise ValueError(f"scenario {key!r} is listed twice")
        scenarios[key] = sc
    kw: Dict[str, Any] = {}
    if args.rule is not None:
        kw["allocation_rule"] = args.rule
    results = run_scenarios(basin, scenarios, **kw)
    rows = comparison_table(results)
    n_years = sorted({r.n_years for r in results.values()})
    print(
        f"Scenario comparison: {basin.name} | scenarios: {', '.join(results)} | years: "
        f"{'/'.join(str(n) for n in n_years)}{' | rule override: ' + args.rule if args.rule else ''}",
        file=out,
    )
    print(format_table(rows, columns), file=out)
    if args.csv:
        print(f"wrote CSV: {write_table_csv(rows, args.csv)}", file=out)
    if args.json:
        payload = {
            "basin": basin.name,
            "scenarios": {k: sc for k, sc in scenarios.items()},
            "rows": rows,
            "summaries": {k: r.summary() for k, r in results.items()},
        }
        print(f"wrote JSON: {write_json(args.json, payload)}", file=out)
    return EXIT_OK


def cmd_pareto(args: argparse.Namespace, out: TextIO) -> int:
    """``pareto``: benefit-equity Pareto front of the allocation LP.

    Parameters
    ----------
    args : argparse.Namespace
        ``basin``, ``flow_factor``, ``points``, ``csv``, ``json``.
    out : text stream

    Returns
    -------
    int
        :data:`EXIT_OK`, or :data:`EXIT_NO_RESULT` when the problem is
        infeasible (empty front).
    """
    basin = load_basin_arg(args.basin)
    front = pareto_front(basin, args.flow_factor, args.points)
    print(
        f"Pareto front (epsilon-constraint on the minimum supply ratio): {basin.name} "
        f"| flow factor {args.flow_factor:g} | {len(front)} of {args.points} points feasible",
        file=out,
    )
    if not front:
        print(
            "no feasible allocation: the environmental flows cannot be met at this flow factor "
            "(try a larger --flow-factor or lower environmental flows)",
            file=out,
        )
        return EXIT_NO_RESULT
    rows = []
    for p in front:
        row = dict(p)
        row["total_benefit_musd"] = p["total_benefit_usd"] / 1e6
        rows.append({label: row[key] for key, label in PARETO_COLUMNS})
    print(format_table(rows), file=out)
    b_first, b_last = front[0]["total_benefit_usd"], front[-1]["total_benefit_usd"]
    price = (b_first - b_last) / b_first if b_first > 0 else 0.0
    print(
        f"price of fairness: benefit falls from {b_first / 1e6:,.1f} to {b_last / 1e6:,.1f} MUSD "
        f"({100.0 * price:.1f}%) as the minimum supply ratio rises from "
        f"{front[0]['achieved_min_supply_ratio']:.3f} to {front[-1]['achieved_min_supply_ratio']:.3f}",
        file=out,
    )
    if args.csv:
        print(f"wrote CSV: {write_table_csv(front, args.csv)}", file=out)
    if args.json:
        payload = {"basin": basin.name, "flow_factor": args.flow_factor, "points": args.points, "front": front}
        print(f"wrote JSON: {write_json(args.json, payload)}", file=out)
    return EXIT_OK


def cmd_report(args: argparse.Namespace, out: TextIO) -> int:
    """``report``: sustainability assessment and conflict risk of a run.

    Parameters
    ----------
    args : argparse.Namespace
        ``basin``, ``scenario``, ``years``, ``rule``, ``json``.
    out : text stream

    Returns
    -------
    int
        :data:`EXIT_OK`.
    """
    basin = load_basin_arg(args.basin)
    scenario = resolve_scenario(args.scenario, args.years)
    kw: Dict[str, Any] = {}
    if args.rule is not None:
        kw["allocation_rule"] = args.rule
    result = run_nexus(basin, scenario, **kw)
    report = assess(result)
    print(report.summary(), file=out)
    print("", file=out)

    treaty = treaty_for_result(basin, result)
    rows = conflict_risk_rows(basin, result, treaty)
    scores = [r["score"] for r in rows]
    mean_score = sum(scores) / len(scores)
    worst = max(rows, key=lambda r: r["score"])
    last = conflict_risk_index(basin, result.balances[-1], treaty)
    print(
        f"Conflict risk index (Basins at Risk style, 0 = none .. 1 = extreme) | allocation rule: "
        f"{result.allocation_rule} | agreement: {treaty.name if treaty is not None else 'none'}",
        file=out,
    )
    print(format_table(rows), file=out)
    print(
        f"mean score {mean_score:.3f} ({risk_category(mean_score)}) | worst year {worst['year']}: "
        f"{worst['score']:.3f} ({worst['category']}) | final year {rows[-1]['year']}: "
        f"{rows[-1]['score']:.3f} ({rows[-1]['category']})",
        file=out,
    )
    if args.json:
        payload = {
            "basin": basin.name,
            "scenario": scenario,
            "allocation_rule": result.allocation_rule,
            "sustainability": report.to_dict(),
            "conflict_risk": {
                "treaty": treaty.to_dict() if treaty is not None else None,
                "mean_score": mean_score,
                "mean_category": risk_category(mean_score),
                "years": rows,
                "final_year": last,
            },
        }
        print(f"wrote JSON: {write_json(args.json, payload)}", file=out)
    return EXIT_OK


def cmd_export_basin(args: argparse.Namespace, out: TextIO) -> int:
    """``export-basin``: write a basin (default: the example) as JSON.

    Parameters
    ----------
    args : argparse.Namespace
        ``basin`` (``example`` or a file to re-export / normalise), ``out``
        (path; standard output when omitted), ``indent``.
    out : text stream

    Returns
    -------
    int
        :data:`EXIT_OK`.
    """
    basin = load_basin_arg(args.basin)
    indent = None if args.indent is not None and args.indent < 0 else args.indent
    if args.out:
        _io.basin_to_json(basin, args.out, indent=indent)
        print(f"wrote basin {basin.name!r} ({len(basin)} riparians) to {args.out}", file=out)
    else:
        print(_io.basin_to_json(basin, indent=indent), file=out)
    return EXIT_OK


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def _add_basin(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--basin",
        default="example",
        metavar="example|PATH.json",
        help="'example' for the stylised Azura River basin, or a basin JSON file (default: example)",
    )


def _add_scenario(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--scenario",
        default="baseline",
        metavar="NAME|PATH.json",
        help=f"library scenario ({', '.join(scenario_names())}) or a scenario JSON file (default: baseline)",
    )
    p.add_argument("--years", type=_positive_int, default=None, help="horizon in years (default: the scenario's, 25)")
    p.add_argument(
        "--rule",
        type=_model_rule_arg,
        default=None,
        metavar="RULE",
        help=f"override the scenario's allocation rule: one of {', '.join(ALLOCATION_RULES)}",
    )


def _add_json(p: argparse.ArgumentParser) -> None:
    p.add_argument("--json", metavar="PATH", default=None, help="also write the full results as JSON")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with every subcommand.

    Returns
    -------
    argparse.ArgumentParser
        Its ``error`` raises :class:`CliError` instead of exiting, so
        :func:`main` can translate usage errors into exit code 2.  Each
        subparser stores its handler in ``args.func``.

    Examples
    --------
    >>> ns = build_parser().parse_args(["run", "--years", "3"])
    >>> ns.command, ns.years, ns.scenario
    ('run', 3, 'baseline')
    """
    parser = _Parser(
        prog=PROG,
        description="Water-energy-food nexus and water diplomacy toolkit for transboundary basins.",
        epilog="Units: water Mm3/yr, energy GWh/yr, money USD. Exit codes: 0 ok, 1 no result, 2 usage error.",
    )
    parser.add_argument("--version", action="version", version=f"{PROG} {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>", required=True)

    p = sub.add_parser("run", help="simulate one scenario and print a summary table")
    _add_basin(p)
    _add_scenario(p)
    p.add_argument(
        "--refill", type=_fraction, default=0.25, metavar="F", help="reservoir refill fraction in [0, 1] (default 0.25)"
    )
    p.add_argument("--csv", metavar="PATH", default=None, help="write every riparian-year record as CSV")
    _add_json(p)
    p.add_argument(
        "--plot",
        metavar="DIR",
        default=None,
        help=f"write PNG figures ({', '.join(PLOT_FILES)}) into this directory (needs matplotlib)",
    )
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("allocate", help="divide an estate among claims with a bankruptcy rule")
    p.add_argument(
        "--rule",
        type=_claims_rule_arg,
        default="talmud",
        metavar="RULE",
        help=f"one of {', '.join(sorted(RULES))} or 'all' (default: talmud)",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--claims", type=_claims_arg, default=None, metavar="a=1,b=2", help="explicit claims (Mm3/yr)")
    src.add_argument(
        "--basin",
        default=None,
        metavar="example|PATH.json",
        help="take the claims (river demands or treaty entitlements, on a consumptive basis) and the estate from a basin",
    )
    p.add_argument(
        "--estate", type=_nonneg_float, default=None, metavar="E", help="estate (Mm3/yr); required with --claims"
    )
    p.add_argument(
        "--flow-factor", type=_nonneg_float, default=1.0, metavar="F", help="natural-flow multiplier with --basin (default 1)"
    )
    p.add_argument(
        "--claims-basis",
        choices=list(CLAIM_BASES),
        default=None,
        help="with --basin: gross claims are the river demands (default) or the treaty entitlements",
    )
    p.add_argument(
        "--include-storage",
        action="store_true",
        help="with --basin: let the routed awards draw on the riparians' reservoir storage",
    )
    _add_json(p)
    p.set_defaults(func=cmd_allocate)

    p = sub.add_parser("negotiate", help="claims-problem proposal versus BATNAs; is there a ZOPA?")
    _add_basin(p)
    p.add_argument("--flow-factor", type=_nonneg_float, default=1.0, metavar="F", help="natural-flow multiplier (default 1)")
    p.add_argument(
        "--rule",
        type=_single_claims_rule_arg,
        default="talmud",
        metavar="RULE",
        help=f"sharing rule: one of {', '.join(sorted(RULES))} (default: talmud)",
    )
    p.add_argument("--estate", type=_nonneg_float, default=None, metavar="E", help="override the estate (Mm3/yr)")
    p.add_argument(
        "--claims",
        choices=list(CLAIM_BASES),
        default="demand",
        help="gross claims: the riparians' river demands (default) or their treaty entitlements",
    )
    p.add_argument(
        "--include-storage",
        action="store_true",
        help="let the unilateral BATNA and the routed proposal draw on reservoir storage (default: this year's flow only)",
    )
    _add_json(p)
    p.set_defaults(func=cmd_negotiate)

    p = sub.add_parser("compare", help="run several scenarios and print the comparison table")
    _add_basin(p)
    p.add_argument(
        "--scenarios",
        default=None,
        metavar="a,b,c",
        help=f"comma-separated scenario names or files (default: all of {', '.join(scenario_names())})",
    )
    p.add_argument("--years", type=_positive_int, default=None, help="horizon in years for every scenario")
    p.add_argument(
        "--rule",
        type=_model_rule_arg,
        default=None,
        metavar="RULE",
        help=f"override the allocation rule of every scenario: one of {', '.join(ALLOCATION_RULES)}",
    )
    p.add_argument(
        "--columns",
        default=None,
        metavar="c1,c2|all",
        help="columns to print (default: a compact selection; 'all' for every column)",
    )
    p.add_argument("--csv", metavar="PATH", default=None, help="write the full comparison table as CSV")
    _add_json(p)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("pareto", help="benefit-equity Pareto front of the allocation LP")
    _add_basin(p)
    p.add_argument("--flow-factor", type=_nonneg_float, default=1.0, metavar="F", help="natural-flow multiplier (default 1)")
    p.add_argument("--points", type=_points_int, default=11, metavar="N", help="number of epsilon levels, >= 2 (default 11)")
    p.add_argument("--csv", metavar="PATH", default=None, help="write the front as CSV")
    _add_json(p)
    p.set_defaults(func=cmd_pareto)

    p = sub.add_parser("report", help="sustainability assessment and conflict risk index of a run")
    _add_basin(p)
    _add_scenario(p)
    _add_json(p)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("export-basin", help="write the example basin (or any basin file) as JSON")
    _add_basin(p)
    p.add_argument("--out", metavar="PATH", default=None, help="destination file (default: print to standard output)")
    p.add_argument("--indent", type=int, default=2, metavar="N", help="JSON indentation (negative: single line; default 2)")
    p.set_defaults(func=cmd_export_basin)
    return parser


def main(argv: Optional[Sequence[str]] = None, stdout: Optional[TextIO] = None, stderr: Optional[TextIO] = None) -> int:
    """Entry point of ``python -m wefnexus`` / the ``wefnexus`` console script.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments without the program name (default: ``sys.argv[1:]``).
    stdout, stderr : text streams, optional
        Where normal output and error messages go (default: the current
        ``sys.stdout`` / ``sys.stderr``, so pytest's ``capsys`` sees them).

    Returns
    -------
    int
        :data:`EXIT_OK` (0) on success, :data:`EXIT_NO_RESULT` (1) when a
        command produced no result, :data:`EXIT_USAGE` (2) on a usage or
        input error (unknown option, scenario, rule or column; unreadable or
        invalid basin file; invalid numbers; unwritable output file).
        ``--help`` and ``--version`` raise :class:`SystemExit` with code 0.

    Examples
    --------
    >>> import io
    >>> buf = io.StringIO()
    >>> main(["allocate", "--rule", "talmud", "--estate", "200", "--claims", "a=100,b=200,c=300"], stdout=buf)
    0
    >>> "rule: talmud" in buf.getvalue()
    True
    """
    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except CliError as exc:
        if exc.usage:
            err.write(exc.usage)
        err.write(f"{exc.prog}: error: {exc.message}\n")
        return EXIT_USAGE
    try:
        return int(args.func(args, out))
    except CliError as exc:
        if exc.usage:
            err.write(exc.usage)
        err.write(f"{exc.prog}: error: {exc.message}\n")
        return EXIT_USAGE
    except (ValueError, OSError) as exc:
        err.write(f"{PROG} {args.command}: error: {exc}\n")
        return EXIT_USAGE


if __name__ == "__main__":  # pragma: no cover - exercised through __main__.py
    sys.exit(main())
