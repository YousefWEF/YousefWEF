"""Optional plots of :mod:`wefnexus` results (matplotlib) and the basin nexus graph (networkx).

Every ``plot_*`` function in this module

* imports matplotlib **lazily** inside the function - the core package never
  needs it - and switches to the non-interactive ``Agg`` backend when the
  process has no display (Linux without ``DISPLAY`` / ``WAYLAND_DISPLAY`` and
  no explicit ``MPLBACKEND``), so the functions work on servers and in CI;
* returns the :class:`matplotlib.figure.Figure` and **never** calls
  ``plt.show()`` - the caller decides whether to display, save or close it;
* accepts ``save=<path>`` to also write the figure (format by extension,
  e.g. ``.png`` / ``.svg`` / ``.pdf``; parent directories are created);
* never mutates the result, basin or dictionaries it is given.

The figures share one visual system: a fixed-order categorical palette in
which riparian *i* (upstream -> downstream) always wears palette slot *i*
and each water sector has its own fixed colour (:data:`CATEGORICAL`,
:data:`SECTOR_COLORS`); thin marks, hairline grids and axes; text in ink
colours (never in a series colour); a legend whenever two or more series are
drawn plus selective direct labels; basin-wide aggregates and thresholds in
muted greys; and never two y-scales on one axes.

:func:`nexus_graph` builds a :class:`networkx.DiGraph` of the basin - the
river chain from headwater to sea plus sector and non-river-source nodes,
with edge weights in Mm3/yr - and :func:`plot_nexus_graph` draws it.

Units follow the package: water in **Mm3/yr** (storage in Mm3), energy in
**GWh/yr**, money in **USD/yr**; indices are dimensionless in ``[0, 1]``.

References
----------
Hashimoto, Stedinger & Loucks (1982) - reliability of water supply;
Hoff (2011) - the water-energy-food nexus framing;
Sadoff & Grey (2002); Mianabadi et al. (2014) - allocation rules and
benefit sharing in transboundary basins;
Haimes, Lasdon & Wismer (1971); Bertsimas, Farias & Trichakis (2011) -
epsilon-constraint Pareto fronts and the "price of fairness";
Wolf, Yoffe & Giordano (2003) - basins at risk and basin network views.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from wefnexus.models import SECTOR_PRIORITY, Basin, Sector

__all__ = [
    "CATEGORICAL",
    "SECTOR_COLORS",
    "INDEX_KEYS",
    "DEFAULT_RADAR_KEYS",
    "riparian_color",
    "plot_supply_ratio",
    "plot_nexus_indices",
    "plot_water_balance",
    "plot_allocation_rules",
    "plot_pareto",
    "plot_nexus_radar",
    "nexus_graph",
    "plot_nexus_graph",
]

# --------------------------------------------------------------------------- #
# Visual system (validated categorical palette, chrome and ink tokens)
# --------------------------------------------------------------------------- #
#: Fixed-order categorical palette (eight slots).  Riparian *i* in basin
#: order always gets slot *i*; the order is colour-vision-deficiency safe for
#: adjacent series and is never cycled (a ninth series is drawn in grey).
CATEGORICAL: Tuple[str, ...] = (
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
)

#: Fixed colour of each withdrawal sector (same slots, assigned in
#: :data:`~wefnexus.models.SECTOR_PRIORITY` order).
SECTOR_COLORS: Dict[Sector, str] = {
    Sector.MUNICIPAL: CATEGORICAL[0],
    Sector.INDUSTRIAL: CATEGORICAL[1],
    Sector.ENERGY: CATEGORICAL[2],
    Sector.AGRICULTURAL: CATEGORICAL[3],
}

#: The four composite indices of a :class:`~wefnexus.nexus.RiparianYear`.
INDEX_KEYS: Tuple[str, ...] = ("water_security", "energy_security", "food_security", "nexus_index")

#: Default axes of :func:`plot_nexus_radar` (those present in the summary are used).
DEFAULT_RADAR_KEYS: Tuple[str, ...] = (
    "water_security",
    "energy_security",
    "food_security",
    "nexus_index",
    "supply_ratio",
    "supply_reliability",
    "env_flow_met_share",
)

# chrome & ink (light surface)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
DEEMPHASIS = "#c3c2b7"
DEFICIT_FILL = "#d9d8d2"
OTHER = "#898781"

_RC: Dict[str, Any] = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 9.5,
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 2.0,
    "lines.markersize": 6.0,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "font.family": "sans-serif",
    "font.size": 10,
    "text.color": INK,
    "figure.dpi": 100,
}

_NON_INTERACTIVE_BACKENDS = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}
_SOURCE_OFFSETS = {"runoff": -0.3, "groundwater": 0.0, "desalination": 0.3}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def riparian_color(index: int) -> str:
    """Colour of the riparian at basin position ``index`` (0 = most upstream).

    Parameters
    ----------
    index : int
        Position in the basin (``>= 0``).

    Returns
    -------
    str
        Hex colour: :data:`CATEGORICAL` slot ``index`` for the first eight
        riparians, the neutral grey :data:`OTHER` beyond that (palette slots
        are never cycled, so a ninth series cannot impersonate the first).

    Raises
    ------
    ValueError
        If ``index`` is negative or not an integer.
    """
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError(f"index must be a non-negative integer, got {index!r}")
    return CATEGORICAL[index] if index < len(CATEGORICAL) else OTHER


def _headless() -> bool:
    """True when no display is available and no backend was chosen explicitly."""
    if os.environ.get("MPLBACKEND"):
        return False
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return False
    return not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _pyplot():
    """Import matplotlib lazily, select ``Agg`` when headless, return ``(matplotlib, pyplot)``."""
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "wefnexus.viz needs matplotlib; install it with `pip install matplotlib` "
            "or `pip install wefnexus[viz]`"
        ) from exc
    if _headless():
        try:
            current = str(matplotlib.get_backend()).lower()
        except Exception:  # pragma: no cover - defensive
            current = ""
        if not (current.startswith("module://") or current in _NON_INTERACTIVE_BACKENDS):
            try:
                matplotlib.use("Agg")
            except Exception:  # pragma: no cover - defensive
                pass
    import matplotlib.pyplot as plt

    return matplotlib, plt


def _finish(fig: Any, save: Any, dpi: int = 150) -> Any:
    """Optionally save ``fig`` (creating parent directories) and return it."""
    if save is not None:
        path = os.fspath(save)
        if not path:
            raise ValueError("save must be a non-empty path")
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig


def _figure(plt: Any, ax: Any, figsize: Tuple[float, float]) -> Tuple[Any, Any]:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax
    if not hasattr(ax, "plot") or not hasattr(ax, "figure"):
        raise ValueError(f"ax must be a matplotlib Axes, got {type(ax).__name__}")
    return ax.figure, ax


def _check_result(result: Any) -> None:
    for attr in ("years", "riparian_names", "series", "basin_series", "scenario"):
        if not hasattr(result, attr):
            raise ValueError(
                f"result must be a wefnexus.nexus.NexusResult (missing {attr!r}), got {type(result).__name__}"
            )
    if not list(result.years):
        raise ValueError("result has no simulated years")
    if not result.riparian_names():
        raise ValueError("result has no riparian records")


def _scenario_name(result: Any) -> str:
    scenario = getattr(result, "scenario", None)
    name = getattr(scenario, "name", None)
    return str(name) if name else ""


def _float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be numeric, got {value!r}") from None


def _clip01(value: float) -> float:
    if math.isnan(value):
        return 0.0
    return min(max(value, 0.0), 1.0)


def _pretty(key: str) -> str:
    """``"water_security"`` -> ``"Water security"``; SDG-style keys keep their code."""
    text = str(key).replace("_", " ").strip()
    if not text:
        return str(key)
    return text[0].upper() + text[1:]


def _integer_years(ax: Any) -> None:
    from matplotlib.ticker import MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))


def _spread(values: Sequence[float], min_gap: float) -> List[float]:
    """Push label positions apart so adjacent labels are at least ``min_gap`` apart."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = list(values)
    last: Optional[float] = None
    for i in order:
        if last is not None and out[i] - last < min_gap:
            out[i] = last + min_gap
        last = out[i]
    return out


def _direct_labels(ax: Any, entries: Sequence[Tuple[float, float, str]]) -> None:
    """Label line endpoints in ink (never in the series colour), de-overlapped."""
    if not entries:
        return
    lo, hi = ax.get_ylim()
    gap = 0.045 * (hi - lo)
    ys = _spread([e[1] for e in entries], gap)
    excess = max(ys) - hi + gap * 0.5
    if excess > 0:
        ys = [y - excess for y in ys]
    for (x, _y, text), y in zip(entries, ys):
        ax.annotate(
            text,
            xy=(x, y),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=8.5,
            color=INK_SECONDARY,
            annotation_clip=False,
        )


def _money_scale(max_value: float) -> Tuple[float, str]:
    if max_value >= 1e9:
        return 1e9, "billion USD/yr"
    if max_value >= 1e6:
        return 1e6, "million USD/yr"
    if max_value >= 1e3:
        return 1e3, "thousand USD/yr"
    return 1.0, "USD/yr"


def _basin_supply_ratio(result: Any) -> List[float]:
    demand = result.basin_series("total_demand_mm3", "sum")
    deficit = result.basin_series("total_deficit_mm3", "sum")
    out = []
    for d, df in zip(demand, deficit):
        out.append(1.0 if d <= 0.0 else _clip01(1.0 - df / d))
    return out


def _legend_below(fig: Any, handles: Sequence[Any], labels: Sequence[str], ncol: Optional[int] = None) -> None:
    if not handles:
        return
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=ncol or min(len(handles), 5),
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
        fontsize=9,
    )


# --------------------------------------------------------------------------- #
# Multi-year simulation plots
# --------------------------------------------------------------------------- #
def plot_supply_ratio(
    result: Any,
    *,
    include_basin: bool = True,
    ax: Any = None,
    save: Any = None,
    title: Optional[str] = None,
) -> Any:
    """Line chart of the annual water supply ratio of every riparian.

    The supply ratio ``1 - deficit / demand`` (dimensionless, ``[0, 1]``) is
    the complement of the relative shortfall used in reliability analysis
    (Hashimoto, Stedinger & Loucks 1982).  One line per riparian in its
    fixed palette colour, direct-labelled at the last year, plus the basin
    ratio (total supplied / total demanded) as a dashed grey context line.

    Parameters
    ----------
    result : NexusResult
        Output of :func:`wefnexus.nexus.run_nexus` (never mutated).
    include_basin : bool, optional
        Draw the basin-wide ratio (default True).
    ax : matplotlib.axes.Axes, optional
        Draw into an existing axes; its figure is returned.
    save : str or path-like, optional
        Also write the figure to this path (format by extension).
    title : str, optional
        Axes title (default names the scenario).

    Returns
    -------
    matplotlib.figure.Figure
        The figure (``plt.show()`` is never called).

    Raises
    ------
    ValueError
        If ``result`` is not a nexus result or is empty.
    ImportError
        If matplotlib is not installed.
    """
    _check_result(result)
    _, plt = _pyplot()
    names = list(result.riparian_names())
    years = [int(y) for y in result.years]
    with plt.rc_context(_RC):
        fig, ax = _figure(plt, ax, (8.0, 4.2))
        marker = "o" if len(years) <= 12 else None
        endpoints: List[Tuple[float, float, str]] = []
        for i, name in enumerate(names):
            ys = [_clip01(_float(v, f"supply_ratio[{name}]")) for v in result.series(name, "supply_ratio")]
            ax.plot(years, ys, color=riparian_color(i), label=name, marker=marker, markeredgecolor=SURFACE,
                    markeredgewidth=1.0, zorder=3)
            endpoints.append((years[-1], ys[-1], name))
        if include_basin:
            basin = _basin_supply_ratio(result)
            ax.plot(years, basin, color=MUTED, linestyle="--", linewidth=1.5, label="Basin (total)", zorder=2)
            endpoints.append((years[-1], basin[-1], "Basin"))
        ax.set_ylim(0.0, 1.05)
        ax.set_xlim(years[0] - 0.5, years[-1] + 0.5)
        ax.set_xlabel("Year")
        ax.set_ylabel("Supply ratio (supplied / demanded)")
        scenario = _scenario_name(result)
        ax.set_title(title or ("Water supply ratio by riparian" + (f" - {scenario}" if scenario else "")))
        _integer_years(ax)
        ax.legend(loc="lower left", ncol=min(len(names) + 1, 4))
        _direct_labels(ax, endpoints)
    return _finish(fig, save)


def plot_nexus_indices(
    result: Any,
    *,
    keys: Sequence[str] = INDEX_KEYS,
    include_basin: bool = True,
    save: Any = None,
    title: Optional[str] = None,
) -> Any:
    """Small multiples of the composite security indices over the horizon.

    One panel per index (water security, energy security, food security and
    the WEF nexus index of :mod:`wefnexus.sustainability`, all in ``[0,
    1]``), one line per riparian in its fixed colour, with the riparian mean
    as a dashed grey context line.  Panels share both axes, so no panel has
    a second y-scale.

    Parameters
    ----------
    result : NexusResult
        Simulation output (never mutated).
    keys : sequence of str, optional
        Record fields to plot (default :data:`INDEX_KEYS`); each must be a
        numeric field of :class:`~wefnexus.nexus.RiparianYear` in ``[0, 1]``.
    include_basin : bool, optional
        Draw the riparian mean per panel (default True).
    save : str or path-like, optional
        Also write the figure to this path.
    title : str, optional
        Figure title (default names the basin and scenario).

    Returns
    -------
    matplotlib.figure.Figure
        Figure with ``len(keys)`` axes (plus a shared legend below).

    Raises
    ------
    ValueError
        On a non-result, an empty result or an empty ``keys``.
    """
    _check_result(result)
    keys = [str(k) for k in keys]
    if not keys:
        raise ValueError("keys must name at least one index field")
    _, plt = _pyplot()
    names = list(result.riparian_names())
    years = [int(y) for y in result.years]
    ncols = 2 if len(keys) > 1 else 1
    nrows = int(math.ceil(len(keys) / ncols))
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.0 * nrows + 0.6), sharex=True, sharey=True,
                                 squeeze=False)
        flat = [a for row in axes for a in row]
        handles: List[Any] = []
        labels: List[str] = []
        for k, key in enumerate(keys):
            ax = flat[k]
            for i, name in enumerate(names):
                ys = [_clip01(_float(v, f"{key}[{name}]")) for v in result.series(name, key)]
                (line,) = ax.plot(years, ys, color=riparian_color(i), label=name, zorder=3)
                if k == 0:
                    handles.append(line)
                    labels.append(name)
            if include_basin:
                mean = [_clip01(v) for v in result.basin_series(key, "mean")]
                (line,) = ax.plot(years, mean, color=MUTED, linestyle="--", linewidth=1.5, label="Riparian mean")
                if k == 0:
                    handles.append(line)
                    labels.append("Riparian mean")
            ax.set_title(_pretty(key))
            ax.set_ylim(0.0, 1.0)
            ax.set_xlim(years[0] - 0.5, years[-1] + 0.5)
            _integer_years(ax)
        for ax in flat[len(keys):]:
            ax.set_visible(False)
        for row in axes:
            row[0].set_ylabel("Index (0-1)")
        for ax in axes[-1]:
            ax.set_xlabel("Year")
        scenario = _scenario_name(result)
        fig.suptitle(
            title or (f"WEF security indices - {result.basin_name}" + (f" - {scenario}" if scenario else "")),
            x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK,
        )
        fig.tight_layout(rect=(0, 0.06, 1, 0.96))
        _legend_below(fig, handles, labels)
    return _finish(fig, save)


def plot_water_balance(
    result: Any,
    riparian: str,
    *,
    basin: Optional[Basin] = None,
    save: Any = None,
    title: Optional[str] = None,
) -> Any:
    """Annual water balance of one riparian: flows (stacked use) and storage.

    Upper panel (Mm3/yr): stacked bars of surface withdrawals by sector
    (fixed sector colours, :data:`SECTOR_COLORS`), topped by non-river supply
    (groundwater + desalination, grey) and unmet demand (light grey) so the
    bar height equals the gross demand; lines for the reach inflow, the
    outflow passed downstream and the environmental-flow requirement at the
    outlet.  Lower panel (Mm3): end-of-year reservoir storage, with the
    reservoir capacity as a hairline when ``basin`` is given.  Both panels
    share the year axis and each has a single scale.

    Parameters
    ----------
    result : NexusResult
        Simulation output (never mutated).
    riparian : str
        Name of the riparian to plot.
    basin : Basin, optional
        Basin of the run, used only to draw the reservoir capacity.
    save : str or path-like, optional
        Also write the figure to this path.
    title : str, optional
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with two axes.

    Raises
    ------
    ValueError
        On a non-result or an empty result.
    KeyError
        If the riparian is not in the result (or not in ``basin``).
    """
    _check_result(result)
    if not isinstance(riparian, str) or not riparian:
        raise ValueError(f"riparian must be a non-empty name, got {riparian!r}")
    if riparian not in result.riparian_names():
        raise KeyError(f"no riparian named {riparian!r}; result covers {result.riparian_names()}")
    capacity: Optional[float] = None
    if basin is not None:
        if not isinstance(basin, Basin):
            raise ValueError(f"basin must be a wefnexus.models.Basin, got {type(basin).__name__}")
        capacity = float(basin.riparian(riparian).reservoir_capacity_mm3)
    _, plt = _pyplot()
    years = [int(y) for y in result.years]
    series = lambda fld: [_float(v, f"{fld}[{riparian}]") for v in result.series(riparian, fld)]  # noqa: E731
    withdrawals = {s: series(f"withdrawal_{s.value}") for s in SECTOR_PRIORITY}
    non_river = [max(v, 0.0) for v in series("non_river_supply_mm3")]
    deficit = [max(v, 0.0) for v in series("total_deficit_mm3")]
    inflow = series("inflow_mm3")
    outflow = series("outflow_mm3")
    env = series("environmental_flow_mm3")
    storage = series("storage_end_mm3")
    with plt.rc_context(_RC):
        fig, (ax, ax2) = plt.subplots(
            2, 1, figsize=(9.0, 6.4), sharex=True, gridspec_kw={"height_ratios": [3.0, 1.4], "hspace": 0.12}
        )
        bottom = [0.0] * len(years)
        for s in SECTOR_PRIORITY:
            vals = [max(v, 0.0) for v in withdrawals[s]]
            ax.bar(years, vals, bottom=bottom, width=0.78, color=SECTOR_COLORS[s], edgecolor=SURFACE,
                   linewidth=1.0, label=f"{_pretty(s.value)} withdrawal", zorder=3)
            bottom = [b + v for b, v in zip(bottom, vals)]
        if any(v > 0 for v in non_river):
            ax.bar(years, non_river, bottom=bottom, width=0.78, color=MUTED, edgecolor=SURFACE, linewidth=1.0,
                   label="Non-river supply (groundwater + desalination)", zorder=3)
            bottom = [b + v for b, v in zip(bottom, non_river)]
        if any(v > 0 for v in deficit):
            ax.bar(years, deficit, bottom=bottom, width=0.78, color=DEFICIT_FILL, edgecolor=SURFACE, linewidth=1.0,
                   label="Unmet demand", zorder=3)
        ax.plot(years, inflow, color=INK, linewidth=2.0, label="Reach inflow", zorder=4)
        ax.plot(years, outflow, color=INK_SECONDARY, linewidth=1.6, linestyle="-.", label="Outflow downstream",
                zorder=4)
        ax.plot(years, env, color=MUTED, linewidth=1.4, linestyle=":", label="Environmental flow requirement",
                zorder=4)
        ax.set_ylabel("Water (Mm3/yr)")
        ax.set_ylim(bottom=0.0)
        ax.set_xlim(years[0] - 0.6, years[-1] + 0.6)
        scenario = _scenario_name(result)
        ax.set_title(title or (f"Water balance of {riparian}" + (f" - {scenario}" if scenario else "")))
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=1)
        _integer_years(ax)
        ax2.fill_between(years, 0.0, storage, color=CATEGORICAL[0], alpha=0.15, linewidth=0, zorder=2)
        ax2.plot(years, storage, color=CATEGORICAL[0], label="Reservoir storage (end of year)", zorder=3)
        if capacity is not None and capacity > 0:
            ax2.axhline(capacity, color=MUTED, linewidth=1.2, linestyle="--", label="Reservoir capacity", zorder=2)
        ax2.set_ylabel("Storage (Mm3)")
        ax2.set_xlabel("Year")
        ax2.set_ylim(bottom=0.0)
        ax2.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return _finish(fig, save)


# --------------------------------------------------------------------------- #
# Diplomacy and optimisation plots
# --------------------------------------------------------------------------- #
def _check_comparison(comparison: Any) -> Tuple[List[str], List[str], Dict[str, Dict[str, Any]]]:
    if not isinstance(comparison, Mapping) or not comparison:
        raise ValueError("comparison must be a non-empty dict {rule: {...}} from diplomacy.compare_allocation_rules")
    rules = [str(k) for k in comparison]
    rows: Dict[str, Dict[str, Any]] = {}
    names: List[str] = []
    for rule in rules:
        row = comparison[rule]
        if not isinstance(row, Mapping) or not isinstance(row.get("awards"), Mapping) or not row["awards"]:
            raise ValueError(f"comparison[{rule!r}] must be a dict with a non-empty 'awards' mapping")
        awards = {str(n): _float(v, f"awards[{n}]") for n, v in row["awards"].items()}
        if not names:
            names = list(awards)
        missing = [n for n in names if n not in awards]
        if missing:
            raise ValueError(f"comparison[{rule!r}]['awards'] lacks riparian(s) {missing}")
        sat = row.get("satisfaction")
        if isinstance(sat, Mapping):
            satisfaction = {n: _float(sat.get(n, 1.0), f"satisfaction[{n}]") for n in names}
        elif isinstance(row.get("claims"), Mapping):
            claims = {n: _float(row["claims"].get(n, 0.0), f"claims[{n}]") for n in names}
            satisfaction = {n: (awards[n] / claims[n] if claims[n] > 0 else 1.0) for n in names}
        else:
            raise ValueError(f"comparison[{rule!r}] needs a 'satisfaction' or 'claims' mapping")
        gini = row.get("gini")
        c_awards = row.get("consumptive_awards")
        consumptive = isinstance(c_awards, Mapping) and all(n in c_awards for n in names)
        rows[rule] = {
            "awards": awards,
            "consumptive_awards": None if not consumptive else {
                n: _float(c_awards[n], f"consumptive_awards[{n}]") for n in names
            },
            "satisfaction": satisfaction,
            "gini": None if gini is None else _float(gini, "gini"),
            "estate": None if row.get("estate") is None else _float(row["estate"], "estate"),
            "claims": None if not isinstance(row.get("claims"), Mapping) else {
                n: _float(row["claims"].get(n, 0.0), f"claims[{n}]") for n in names
            },
            "consumptive_claims": None if not isinstance(row.get("consumptive_claims"), Mapping) else {
                n: _float(row["consumptive_claims"].get(n, 0.0), f"consumptive_claims[{n}]") for n in names
            },
        }
    # the bars and the claims line share one basis: the consumptive awards (what the rules divide, the basis
    # of the estate) when every rule carries them, the gross awards (legacy / explicit input) otherwise
    consumptive = all(rows[r]["consumptive_awards"] is not None for r in rules)
    for r in rules:
        rows[r]["bars"] = rows[r]["consumptive_awards"] if consumptive else rows[r]["awards"]
        rows[r]["claims_line"] = rows[r]["consumptive_claims"] if consumptive else rows[r]["claims"]
        rows[r]["consumptive"] = consumptive
    return rules, names, rows


def plot_allocation_rules(
    comparison: Mapping[str, Mapping[str, Any]],
    *,
    save: Any = None,
    title: Optional[str] = None,
) -> Any:
    """Compare sharing rules: awards per riparian and satisfaction per rule.

    Left panel: horizontal stacked bars of the awards (Mm3/yr) of every
    riparian under each rule, with the estate and the total claims as grey
    reference lines - the bankruptcy-problem view of a basin (Mianabadi et
    al. 2014; Thomson 2003).  The three are drawn on one basis: when every
    rule carries ``"consumptive_awards"`` (the rows of
    :func:`wefnexus.diplomacy.compare_allocation_rules`, whose estate is
    the consumptive water the year can divide) the bars are the consumptive
    awards and the claims line the consumptive claims, so the bars sum to
    ``min(estate, total claims)``; otherwise the gross ``"awards"`` and
    ``"claims"`` are drawn (legacy or explicit input).  Right panel: a dot
    plot of each riparian's satisfaction ``award / claim`` per rule (the
    same on either basis), with the Gini coefficient of the awards printed
    beside each row and a line at full satisfaction.  Riparians keep their
    fixed colours in both panels.

    Parameters
    ----------
    comparison : dict
        Output of :func:`wefnexus.diplomacy.compare_allocation_rules`:
        ``{rule: {"awards": {...}, "satisfaction": {...}, "gini": ...,
        "estate": ..., "claims": {...}, "consumptive_awards": {...},
        "consumptive_claims": {...}}}``; ``"satisfaction"`` is derived from
        ``"claims"`` when missing, the other extras are optional.
    save : str or path-like, optional
        Also write the figure to this path.
    title : str, optional
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with two axes.

    Raises
    ------
    ValueError
        On an empty or malformed comparison.
    """
    rules, names, rows = _check_comparison(comparison)
    _, plt = _pyplot()
    n_rules = len(rules)
    with plt.rc_context(_RC):
        fig, (ax, ax2) = plt.subplots(
            1, 2, figsize=(11.0, 1.0 + 0.55 * n_rules + 1.2), sharey=True,
            gridspec_kw={"width_ratios": [1.7, 1.0], "wspace": 0.08},
        )
        for a in (ax, ax2):
            a.grid(True, axis="x")
            a.grid(False, axis="y")
        ypos = list(range(n_rules))[::-1]
        consumptive = rows[rules[0]]["consumptive"]
        left = [0.0] * n_rules
        for i, name in enumerate(names):
            vals = [max(rows[r]["bars"][name], 0.0) for r in rules]
            ax.barh(ypos, vals, left=left, height=0.62, color=riparian_color(i), edgecolor=SURFACE, linewidth=1.0,
                    label=name, zorder=3)
            left = [l + v for l, v in zip(left, vals)]
        x_max = max(left) if left else 0.0
        estates = [rows[r]["estate"] for r in rules if rows[r]["estate"] is not None]
        claims_rows = [rows[r]["claims_line"] for r in rules if rows[r]["claims_line"] is not None]
        total_claims = sum(claims_rows[0].values()) if claims_rows else None
        for ref in (estates[0] if estates else None, total_claims):
            if ref is not None:
                x_max = max(x_max, ref)
        x_max = x_max * 1.12 if x_max > 0 else 1.0
        if estates:
            ax.axvline(estates[0], color=INK_SECONDARY, linewidth=1.2, linestyle="--", zorder=4)
            ax.annotate("estate", xy=(estates[0], ypos[0] + 0.5), xytext=(-4, 0), textcoords="offset points",
                        fontsize=8.5, color=INK_SECONDARY, va="center", ha="right")
        if total_claims is not None:
            ax.axvline(total_claims, color=MUTED, linewidth=1.2, linestyle=":", zorder=4)
            ax.annotate("consumptive claims" if consumptive else "total claims",
                        xy=(total_claims, ypos[-1] - 0.5), xytext=(-4, 0),
                        textcoords="offset points", fontsize=8.5, color=INK_SECONDARY, va="center", ha="right")
        ax.set_yticks(ypos)
        ax.set_yticklabels(rules)
        ax.set_xlabel("Consumptive award (Mm3/yr)" if consumptive else "Award (Mm3/yr)")
        ax.set_xlim(0.0, x_max)
        ax.set_title("Consumptive awards by riparian" if consumptive else "Awards by riparian")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=min(len(names), 4))

        jitter = 0.5 / max(len(names), 1)
        for i, name in enumerate(names):
            xs = [_clip01(rows[r]["satisfaction"][name]) for r in rules]
            ys = [y + (i - (len(names) - 1) / 2.0) * -jitter for y in ypos]
            ax2.plot(xs, ys, linestyle="none", marker="o", markersize=8, color=riparian_color(i),
                     markeredgecolor=SURFACE, markeredgewidth=1.5, label=name, zorder=3)
        ax2.axvline(1.0, color=MUTED, linewidth=1.0, zorder=2)
        for y, rule in zip(ypos, rules):
            gini = rows[rule]["gini"]
            if gini is not None:
                ax2.annotate(f"Gini {gini:.2f}", xy=(1.0, y), xytext=(12, 0), textcoords="offset points",
                             fontsize=8.5, color=INK_SECONDARY, va="center", annotation_clip=False)
        ax2.set_xlim(0.0, 1.08)
        ax2.set_xlabel("Satisfaction (award / claim)")
        ax2.set_title("Claim satisfaction")
        ax2.tick_params(axis="y", length=0)
        fig.suptitle(title or "Allocation rules compared", x=0.01, ha="left", fontsize=12,
                     fontweight="bold", color=INK)
        fig.subplots_adjust(top=0.84, bottom=0.26 if n_rules <= 4 else 0.2, right=0.9)
    return _finish(fig, save)


def plot_pareto(
    front: Sequence[Mapping[str, Any]],
    *,
    show_gini: bool = True,
    save: Any = None,
    title: Optional[str] = None,
) -> Any:
    """Benefit-equity Pareto front from :func:`wefnexus.optimize.pareto_front`.

    Upper panel: total economic benefit (USD/yr, auto-scaled to millions or
    billions) against the guaranteed minimum supply ratio (the
    epsilon-constraint level), with the unconstrained optimum and the
    max-min (Rawlsian) end labelled and the "price of fairness" (benefit
    foregone between them, Bertsimas et al. 2011) annotated.  Lower panel
    (optional): the Gini coefficient of the riparian supply ratios along
    the same x axis.

    Parameters
    ----------
    front : sequence of dict
        Points with ``"min_supply_ratio"`` and ``"total_benefit_usd"``
        (``"gini"`` optional); drawn in increasing x.
    show_gini : bool, optional
        Add the Gini panel when every point has a ``"gini"`` (default True).
    save : str or path-like, optional
        Also write the figure to this path.
    title : str, optional
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with one or two axes.

    Raises
    ------
    ValueError
        On an empty front or points lacking the required keys.
    """
    if isinstance(front, (str, bytes, Mapping)) or not isinstance(front, Iterable):
        raise ValueError("front must be a list of points from optimize.pareto_front")
    points: List[Tuple[float, float, Optional[float]]] = []
    for k, p in enumerate(front):
        if not isinstance(p, Mapping) or "min_supply_ratio" not in p or "total_benefit_usd" not in p:
            raise ValueError(f"front[{k}] must be a dict with 'min_supply_ratio' and 'total_benefit_usd'")
        x = _float(p["min_supply_ratio"], f"front[{k}]['min_supply_ratio']")
        y = _float(p["total_benefit_usd"], f"front[{k}]['total_benefit_usd']")
        g = None if p.get("gini") is None else _float(p["gini"], f"front[{k}]['gini']")
        points.append((x, y, g))
    if not points:
        raise ValueError("front is empty (the allocation problem was infeasible?)")
    points.sort(key=lambda t: t[0])
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ginis = [p[2] for p in points]
    with_gini = bool(show_gini) and all(g is not None for g in ginis)
    scale, unit = _money_scale(max(abs(v) for v in ys) if ys else 0.0)
    ys_s = [y / scale for y in ys]
    _, plt = _pyplot()
    with plt.rc_context(_RC):
        if with_gini:
            fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.5, 6.2), sharex=True,
                                          gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.12})
        else:
            fig, ax = plt.subplots(figsize=(7.5, 4.4))
            ax2 = None
        ax.plot(xs, ys_s, color=CATEGORICAL[0], marker="o", markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=3,
                label="Pareto front (epsilon-constraint)")
        ax.set_ylabel(f"Total benefit ({unit})")
        ax.set_title(title or "Efficiency-equity trade-off of basin allocation")
        if len(points) >= 2:
            ax.annotate("max benefit", xy=(xs[0], ys_s[0]), xytext=(6, 8), textcoords="offset points",
                        fontsize=8.5, color=INK_SECONDARY)
            ax.annotate("max-min equity", xy=(xs[-1], ys_s[-1]), xytext=(-10, 0), textcoords="offset points",
                        fontsize=8.5, color=INK_SECONDARY, ha="right", va="center")
            if ys[0] > 0:
                drop = (ys[0] - ys[-1]) / ys[0] * 100.0
                ax.text(0.02, 0.05, f"price of fairness: {drop:.1f}% of max benefit", transform=ax.transAxes,
                        ha="left", va="bottom", fontsize=8.5, color=INK_SECONDARY)
        lo, hi = min(ys_s), max(ys_s)
        pad = 0.08 * (hi - lo) if hi > lo else (0.1 * abs(hi) if hi else 1.0)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(-0.02, 1.02)
        if ax2 is not None:
            ax2.plot(xs, ginis, color=CATEGORICAL[0], marker="o", markeredgecolor=SURFACE, markeredgewidth=1.0,
                     zorder=3)
            ax2.set_ylabel("Gini of supply ratios")
            ax2.set_ylim(0.0, max(0.05, max(g for g in ginis if g is not None) * 1.15))
            ax2.set_xlabel("Guaranteed minimum supply ratio (epsilon)")
        else:
            ax.set_xlabel("Guaranteed minimum supply ratio (epsilon)")
    return _finish(fig, save)


def _radar_entries(summary: Any) -> Dict[str, Mapping[str, Any]]:
    """Normalise a summary dict, SustainabilityReport or NexusResult to ``{name: {key: value}}``."""
    if hasattr(summary, "records") and callable(getattr(summary, "summary", None)):
        summary = summary.summary()
    elif hasattr(summary, "riparians") and hasattr(summary, "basin") and isinstance(summary.riparians, Mapping):
        entries = {str(k): v for k, v in summary.riparians.items()}
        if isinstance(summary.basin, Mapping) and summary.basin:
            entries["BASIN"] = summary.basin
        summary = entries
    if not isinstance(summary, Mapping) or not summary:
        raise ValueError(
            "summary must be NexusResult.summary() ({riparian: {indicator: value}}), a "
            "SustainabilityReport or a NexusResult"
        )
    out: Dict[str, Mapping[str, Any]] = {}
    for name, row in summary.items():
        if isinstance(row, Mapping):
            out[str(name)] = row
    if not out:
        raise ValueError("summary holds no {riparian: {indicator: value}} entries")
    return out


def plot_nexus_radar(
    summary: Any,
    *,
    keys: Optional[Sequence[str]] = None,
    include_basin: bool = True,
    save: Any = None,
    title: Optional[str] = None,
) -> Any:
    """Radar (spider) chart of horizon-mean indicators per riparian.

    Each axis is one indicator in ``[0, 1]`` (values are clipped); each
    riparian is one polygon in its fixed colour and the basin row, when
    present, a dashed grey polygon.  A radar is used for *profiles* - the
    shape tells which pillar of security a riparian lacks - not for precise
    comparison, which the summary table provides.

    Parameters
    ----------
    summary : dict, SustainabilityReport or NexusResult
        ``NexusResult.summary()`` (``{riparian: {...}, "BASIN": {...}}``), a
        :class:`~wefnexus.sustainability.SustainabilityReport` or a result
        (summarised on the fly).
    keys : sequence of str, optional
        Indicator keys to use as axes (default: those of
        :data:`DEFAULT_RADAR_KEYS` present in every riparian entry); at
        least three are required.
    include_basin : bool, optional
        Draw the ``"BASIN"`` entry when present (default True).
    save : str or path-like, optional
        Also write the figure to this path.
    title : str, optional
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with one polar axes.

    Raises
    ------
    ValueError
        On an empty summary, fewer than three usable axes or a requested
        key missing from a riparian.
    """
    entries = _radar_entries(summary)
    riparians = [n for n in entries if n != "BASIN"]
    if not riparians:
        raise ValueError("summary has no riparian entries (only 'BASIN')")

    def usable(key: str) -> bool:
        for n in riparians:
            v = entries[n].get(key)
            if v is None or isinstance(v, (str, bytes)):
                return False
            try:
                float(v)
            except (TypeError, ValueError):
                return False
        return True

    if keys is None:
        axes_keys = [k for k in DEFAULT_RADAR_KEYS if usable(k)]
    else:
        axes_keys = [str(k) for k in keys]
        bad = [k for k in axes_keys if not usable(k)]
        if bad:
            raise ValueError(f"key(s) {bad} missing or non-numeric for some riparian; available: "
                             f"{sorted(k for k in entries[riparians[0]] if usable(k))}")
    if len(axes_keys) < 3:
        raise ValueError(f"a radar needs at least three numeric indicators, found {axes_keys}")
    n = len(axes_keys)
    angles = [2.0 * math.pi * k / n for k in range(n)]
    closed = angles + angles[:1]
    _, plt = _pyplot()
    with plt.rc_context(_RC):
        fig = plt.figure(figsize=(7.2, 6.0))
        ax = fig.add_subplot(111, polar=True)
        ax.set_theta_offset(math.pi / 2.0)
        ax.set_theta_direction(-1)
        ax.set_facecolor(SURFACE)
        ax.grid(True, axis="both", color=GRID, linewidth=0.8)
        ax.spines["polar"].set_color(AXIS)
        ax.set_xticks(angles)
        ax.set_xticklabels([_pretty(k) for k in axes_keys], fontsize=9, color=INK_SECONDARY)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7.5, color=MUTED)
        ax.set_rlabel_position(0)
        for i, name in enumerate(riparians):
            vals = [_clip01(float(entries[name][k])) for k in axes_keys]
            vals = vals + vals[:1]
            ax.plot(closed, vals, color=riparian_color(i), linewidth=2.0, marker="o", markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=0.8, label=name, zorder=3)
            ax.fill(closed, vals, color=riparian_color(i), alpha=0.07, zorder=2)
        if include_basin and "BASIN" in entries:
            row = entries["BASIN"]
            if all(row.get(k) is not None for k in axes_keys):
                try:
                    vals = [_clip01(float(row[k])) for k in axes_keys]
                except (TypeError, ValueError):
                    vals = []
                if vals:
                    vals = vals + vals[:1]
                    ax.plot(closed, vals, color=MUTED, linewidth=1.5, linestyle="--", label="Basin", zorder=3)
        ax.set_title(title or "WEF security profile by riparian (horizon means)", pad=22, loc="center")
        ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.08))
    return _finish(fig, save)


# --------------------------------------------------------------------------- #
# Basin nexus graph (networkx)
# --------------------------------------------------------------------------- #
def nexus_graph(basin: Basin, *, balance: Any = None, flow_factor: float = 1.0) -> Any:
    """Directed graph of the basin's water flows: river chain, sectors and sources.

    Nodes (attribute ``kind``):

    * ``"headwater"`` and ``"sea"`` (``kind="terminal"``) and one node per
      riparian (``kind="riparian"``, with ``position``, ``population``,
      ``gdp_usd``, ``local_inflow_mm3``, ``environmental_flow_mm3``,
      ``reservoir_capacity_mm3`` and ``treaty_allocation_mm3``);
    * one node per withdrawal sector of each riparian, named
      ``"<riparian>:<sector>"`` (``kind="sector"``, ``riparian``, ``sector``);
    * non-river sources per riparian, ``"<riparian>:runoff"`` (local
      inflow), ``"<riparian>:groundwater"`` (abstraction) and
      ``"<riparian>:desalination"`` (capacity), ``kind="source"`` with a
      ``source`` attribute, added only when the volume is positive.

    Edges carry ``weight`` in **Mm3/yr** and ``kind``:

    * river edges (``kind="river"``) headwater -> riparian 0 -> ... -> sea,
      weighted by the natural (zero-use) flow at that point
      (:func:`wefnexus.water.natural_flows`) or, when ``balance`` is given,
      by the routed flow (upstream inflow of the first reach, each reach's
      outflow); they also carry ``env_flow_mm3``, the requirement at the
      upstream reach's outlet;
    * withdrawal edges (``kind="withdrawal"``) riparian -> sector, weighted
      by the sector's withdrawal demand or, with ``balance``, the actual
      surface withdrawal; they also carry ``demand_mm3``,
      ``consumption_mm3`` and ``return_flow_mm3``;
    * source edges (``kind="source"``) source -> riparian, weighted by the
      volume supplied.

    Parameters
    ----------
    basin : Basin
        Basin to describe (never mutated).
    balance : BasinBalance, optional
        A routed year of the same basin; its flows replace the natural
        flows and demands.
    flow_factor : float, optional
        Multiplier on natural flows when no ``balance`` is given (``>= 0``).

    Returns
    -------
    networkx.DiGraph
        Graph attributes ``basin``, ``riparians`` (names in order),
        ``flow_factor`` and ``routed``.

    Raises
    ------
    ValueError
        On a non-basin, a negative flow factor or a balance of a different
        basin.
    ImportError
        If networkx is not installed.

    References
    ----------
    Wolf, Yoffe & Giordano (2003); Hoff (2011) (nexus linkages as a network).
    """
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("nexus_graph needs networkx; install it with `pip install networkx`") from exc
    from wefnexus.water import natural_flows

    if not isinstance(basin, Basin):
        raise ValueError(f"basin must be a wefnexus.models.Basin, got {type(basin).__name__}")
    ff = _float(flow_factor, "flow_factor")
    if math.isnan(ff) or ff < 0.0:
        raise ValueError(f"flow_factor must be >= 0, got {flow_factor!r}")
    names = basin.names()
    reaches: Dict[str, Any] = {}
    if balance is not None:
        if not hasattr(balance, "reaches") or not hasattr(balance, "outflow_to_sea_mm3"):
            raise ValueError(f"balance must be a wefnexus.water.BasinBalance, got {type(balance).__name__}")
        reaches = {r.name: r for r in balance.reaches}
        if list(reaches) != names:
            raise ValueError(f"balance reaches {list(reaches)} do not match basin riparians {names}")
    natural = natural_flows(basin, ff)

    G = nx.DiGraph()
    G.graph.update({"basin": basin.name, "riparians": list(names), "flow_factor": ff, "routed": balance is not None})
    G.add_node("headwater", kind="terminal", label="Headwater")
    G.add_node("sea", kind="terminal", label="Sea / outlet")
    for r in basin.riparians:
        G.add_node(
            r.name,
            kind="riparian",
            label=r.name,
            position=int(r.position),
            population=float(r.population),
            gdp_usd=float(r.gdp_usd),
            local_inflow_mm3=float(r.local_inflow_mm3),
            environmental_flow_mm3=float(r.demand.environmental),
            reservoir_capacity_mm3=float(r.reservoir_capacity_mm3),
            treaty_allocation_mm3=None if r.treaty_allocation_mm3 is None else float(r.treaty_allocation_mm3),
        )
    # river chain
    head_weight = float(basin.headwater_inflow_mm3) * ff
    if reaches:
        head_weight = float(reaches[names[0]].upstream_inflow_mm3)
    G.add_edge("headwater", names[0], kind="river", weight=max(head_weight, 0.0), env_flow_mm3=0.0)
    for k, r in enumerate(basin.riparians):
        if reaches:
            weight = float(reaches[r.name].outflow_mm3)
        else:
            weight = float(natural[r.name])
        downstream = names[k + 1] if k + 1 < len(names) else "sea"
        G.add_edge(r.name, downstream, kind="river", weight=max(weight, 0.0),
                   env_flow_mm3=float(r.demand.environmental))
    # sectors and sources
    for r in basin.riparians:
        demand = r.demand.withdrawals()
        reach = reaches.get(r.name)
        for s in SECTOR_PRIORITY:
            node = f"{r.name}:{s.value}"
            G.add_node(node, kind="sector", riparian=r.name, sector=s.value, label=_pretty(s.value))
            dem = float(demand.get(s, 0.0))
            if reach is not None:
                w = float(reach.withdrawals.get(s, 0.0))
                cons = float(reach.consumption.get(s, 0.0))
                dem = float(reach.demands.get(s, dem))
            else:
                w = dem
                cons = w * float(r.demand.consumption_fraction.get(s, 0.0))
            G.add_edge(r.name, node, kind="withdrawal", weight=max(w, 0.0), demand_mm3=dem,
                       consumption_mm3=max(cons, 0.0), return_flow_mm3=max(w - cons, 0.0))
        if reach is not None:
            local = float(reach.inflow_mm3) - float(reach.upstream_inflow_mm3)
        else:
            local = float(r.local_inflow_mm3) * ff
        sources = (
            ("runoff", local, "Runoff"),
            ("groundwater", float(r.groundwater_abstraction_mm3), "Groundwater"),
            ("desalination", float(r.energy.desalination_capacity_mm3), "Desalination"),
        )
        for key, volume, label in sources:
            if volume > 0.0:
                node = f"{r.name}:{key}"
                G.add_node(node, kind="source", riparian=r.name, source=key, label=label)
                G.add_edge(node, r.name, kind="source", weight=float(volume))
    return G


def _graph_layout(G: Any) -> Dict[Any, Tuple[float, float]]:
    riparians = [n for n, d in G.nodes(data=True) if d.get("kind") == "riparian"]
    riparians.sort(key=lambda n: G.nodes[n].get("position", 0))
    pos: Dict[Any, Tuple[float, float]] = {}
    xs: Dict[Any, float] = {}
    for k, n in enumerate(riparians):
        x = float(G.nodes[n].get("position", k))
        pos[n] = (x, 0.0)
        xs[n] = x
    if "headwater" in G:
        pos["headwater"] = ((min(xs.values()) if xs else 0.0) - 1.0, 0.0)
    if "sea" in G:
        pos["sea"] = ((max(xs.values()) if xs else 0.0) + 1.0, 0.0)
    sector_index = {s.value: i for i, s in enumerate(SECTOR_PRIORITY)}
    for n, d in G.nodes(data=True):
        if n in pos:
            continue
        rip = d.get("riparian")
        if d.get("kind") == "sector" and rip in xs:
            k = sector_index.get(d.get("sector"), len(sector_index))
            pos[n] = (xs[rip] + (k - (len(sector_index) - 1) / 2.0) * 0.22, -1.0)
        elif d.get("kind") == "source" and rip in xs:
            pos[n] = (xs[rip] + _SOURCE_OFFSETS.get(d.get("source"), 0.0), 1.0)
    missing = [n for n in G if n not in pos]
    if missing:
        import networkx as nx

        extra = nx.spring_layout(G.subgraph(missing), seed=0)
        pos.update({n: (float(x), float(y)) for n, (x, y) in extra.items()})
    return pos


def plot_nexus_graph(
    graph: Any,
    *,
    balance: Any = None,
    flow_factor: float = 1.0,
    ax: Any = None,
    save: Any = None,
    title: Optional[str] = None,
) -> Any:
    """Draw the basin nexus graph of :func:`nexus_graph`.

    Riparians sit on a horizontal river line (headwater left, sea right) in
    their fixed colours; local-runoff, groundwater and desalination sources
    hang above them in grey; the four sector nodes hang below in the fixed
    sector colours.  Edge width is proportional to the edge weight (Mm3/yr)
    and river edges are labelled with their flow.

    Parameters
    ----------
    graph : networkx.DiGraph or Basin
        A graph from :func:`nexus_graph`, or a basin (the graph is built
        with ``balance`` and ``flow_factor``).
    balance, flow_factor
        Passed to :func:`nexus_graph` when ``graph`` is a basin.
    ax : matplotlib.axes.Axes, optional
        Draw into an existing axes.
    save : str or path-like, optional
        Also write the figure to this path.
    title : str, optional
        Axes title.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If ``graph`` is neither a basin nor a graph with nodes.
    ImportError
        If matplotlib or networkx is missing.
    """
    import networkx as nx

    if isinstance(graph, Basin):
        G = nexus_graph(graph, balance=balance, flow_factor=flow_factor)
    elif isinstance(graph, nx.Graph):
        G = graph
    else:
        raise ValueError(f"graph must be a networkx DiGraph or a Basin, got {type(graph).__name__}")
    if G.number_of_nodes() == 0:
        raise ValueError("graph has no nodes")
    _, plt = _pyplot()
    from matplotlib.lines import Line2D

    pos = _graph_layout(G)
    riparians = [n for n, d in G.nodes(data=True) if d.get("kind") == "riparian"]
    riparians.sort(key=lambda n: G.nodes[n].get("position", 0))
    rip_index = {n: k for k, n in enumerate(riparians)}
    sectors = [n for n, d in G.nodes(data=True) if d.get("kind") == "sector"]
    sources = [n for n, d in G.nodes(data=True) if d.get("kind") == "source"]
    terminals = [n for n, d in G.nodes(data=True) if d.get("kind") == "terminal"]
    others = [n for n in G if n not in rip_index and n not in sectors and n not in sources and n not in terminals]
    sector_color = {s.value: c for s, c in SECTOR_COLORS.items()}
    weights = [float(d.get("weight", 0.0)) for _, _, d in G.edges(data=True)]
    wmax = max(weights) if weights and max(weights) > 0 else 1.0

    def width(w: float) -> float:
        return 0.8 + 5.2 * max(w, 0.0) / wmax

    n_rip = max(len(riparians), 1)
    with plt.rc_context(_RC):
        fig, ax = _figure(plt, ax, (2.6 * n_rip + 4.0, 5.2))
        ax.set_facecolor(SURFACE)
        ax.grid(False)
        longest = max([len(str(G.nodes[n].get("label", n))) for n in riparians] or [6])
        diameter_pt = max(48.0, 5.4 * longest + 12.0)
        rip_size = min(diameter_pt * diameter_pt, 6400.0)
        node_size = {}
        for n in G:
            kind = G.nodes[n].get("kind")
            node_size[n] = rip_size if kind == "riparian" else 380 if kind == "sector" else 320
        for kind_edges, color_fn in (
            ("river", lambda u, v, d: INK_SECONDARY),
            ("source", lambda u, v, d: MUTED),
            ("withdrawal", lambda u, v, d: sector_color.get(G.nodes[v].get("sector"), MUTED)),
        ):
            edges = [(u, v, d) for u, v, d in G.edges(data=True) if d.get("kind") == kind_edges]
            if not edges:
                continue
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v) for u, v, _ in edges], width=[width(float(d.get("weight", 0.0))) for _, _, d in edges],
                edge_color=[color_fn(u, v, d) for u, v, d in edges], arrows=True, arrowstyle="-|>", arrowsize=11,
                nodelist=list(G.nodes()), node_size=[node_size[n] for n in G.nodes()], ax=ax, alpha=0.9,
                connectionstyle="arc3,rad=0.0",
            )
        other_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("kind") not in ("river", "source", "withdrawal")]
        if other_edges:
            nx.draw_networkx_edges(G, pos, edgelist=other_edges, width=1.0, edge_color=MUTED, arrows=True, ax=ax,
                                   nodelist=list(G.nodes()), node_size=[node_size[n] for n in G.nodes()])
        if riparians:
            nx.draw_networkx_nodes(G, pos, nodelist=riparians, node_color=[riparian_color(rip_index[n]) for n in riparians],
                                   node_size=[node_size[n] for n in riparians], edgecolors=SURFACE, linewidths=2.0, ax=ax)
        if sectors:
            nx.draw_networkx_nodes(G, pos, nodelist=sectors,
                                   node_color=[sector_color.get(G.nodes[n].get("sector"), MUTED) for n in sectors],
                                   node_size=[node_size[n] for n in sectors], edgecolors=SURFACE, linewidths=1.5, ax=ax)
        if sources:
            nx.draw_networkx_nodes(G, pos, nodelist=sources, node_color=MUTED, node_size=[node_size[n] for n in sources],
                                   edgecolors=SURFACE, linewidths=1.5, ax=ax)
        if terminals:
            nx.draw_networkx_nodes(G, pos, nodelist=terminals, node_color=DEEMPHASIS, node_shape="s",
                                   node_size=[node_size[n] for n in terminals], edgecolors=SURFACE, linewidths=1.5, ax=ax)
        if others:
            nx.draw_networkx_nodes(G, pos, nodelist=others, node_color=DEEMPHASIS, node_size=300, ax=ax)
        nx.draw_networkx_labels(G, pos, labels={n: str(G.nodes[n].get("label", n)) for n in riparians}, font_size=8,
                                font_color=INK, font_weight="bold", ax=ax)
        small = {n: str(G.nodes[n].get("label", n)) for n in sectors + sources + terminals + others}
        sector_rank = {s.value: i for i, s in enumerate(SECTOR_PRIORITY)}
        source_rank = {"runoff": 0, "groundwater": 1, "desalination": 2}
        offset = {}
        for n in small:
            x, y = pos[n]
            d = G.nodes[n]
            if n in sectors:
                row = sector_rank.get(d.get("sector"), 0) % 2
                offset[n] = (x, y - 0.2 - 0.14 * row)
            elif n in sources:
                row = source_rank.get(d.get("source"), 0) % 2
                offset[n] = (x, y + 0.2 + 0.14 * row)
            elif n in terminals:
                offset[n] = (x, y - 0.24)
            else:
                offset[n] = (x, y + 0.2)
        nx.draw_networkx_labels(G, offset, labels=small, font_size=7.5, font_color=INK_SECONDARY, ax=ax)
        river_labels = {(u, v): f"{float(d.get('weight', 0.0)):,.0f}" for u, v, d in G.edges(data=True)
                        if d.get("kind") == "river"}
        if river_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=river_labels, font_size=7.5, font_color=INK_SECONDARY,
                                         bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.0}, ax=ax,
                                         rotate=False, label_pos=0.5)
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        ax.set_xlim(min(xs) - 0.6, max(xs) + 0.6)
        ax.set_ylim(min(ys) - 0.55, max(ys) + 0.55)
        ax.set_axis_off()
        routed = bool(G.graph.get("routed", False))
        ax.set_title(title or (f"Water-energy-food nexus network - {G.graph.get('basin', '')} "
                               f"({'routed flows' if routed else 'natural flows and demands'}, Mm3/yr)"))
        handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=8, color=sector_color[s.value],
                          label=f"{_pretty(s.value)} withdrawal") for s in SECTOR_PRIORITY]
        handles.append(Line2D([0], [0], color=INK_SECONDARY, linewidth=3, label="River flow (width ~ Mm3/yr)"))
        handles.append(Line2D([0], [0], color=MUTED, linewidth=2, label="Local runoff / groundwater / desalination"))
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=8.5)
    return _finish(fig, save)
