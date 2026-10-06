"""Tests for :mod:`wefnexus.viz`, ``examples/run_example.py`` and the demo notebook.

The plotting tests skip cleanly when matplotlib (or networkx for the graph
tests) is not installed; the example-script test runs regardless, because
the script itself skips the figures without matplotlib.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from wefnexus import viz as V
from wefnexus.data import example_basin
from wefnexus.models import SECTOR_PRIORITY, Scenario, Sector
from wefnexus.nexus import run_nexus

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "examples" / "run_example.py"
NOTEBOOK = REPO / "examples" / "wef_nexus_demo.ipynb"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


# --------------------------------------------------------------------------- #
# fixtures and helpers
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def mpl():
    """matplotlib with the Agg backend, or skip the test."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    return matplotlib


@pytest.fixture(scope="module")
def nx():
    return pytest.importorskip("networkx")


@pytest.fixture(scope="module")
def result():
    """A short deterministic run with one drought year (so deficits exist)."""
    return run_nexus(example_basin(), Scenario(name="viz-test", years=6, drought_years=[2], drought_severity=0.5))


@pytest.fixture(scope="module")
def comparison():
    from wefnexus.diplomacy import compare_allocation_rules

    return compare_allocation_rules(example_basin(), 0.7)


@pytest.fixture(scope="module")
def front():
    from wefnexus.optimize import pareto_front

    return pareto_front(example_basin(), 0.6, points=4)


@pytest.fixture(autouse=True)
def _close_all_figures():
    yield
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return
    plt.close("all")


def is_figure(fig, mpl) -> bool:
    return isinstance(fig, mpl.figure.Figure)


def assert_png(path: Path) -> None:
    assert path.is_file(), f"{path} was not written"
    data = path.read_bytes()
    assert data[:8] == PNG_MAGIC, "not a PNG file"
    assert len(data) > 1000


def visible_axes(fig):
    return [a for a in fig.axes if a.get_visible()]


def title_of(ax) -> str:
    """Axes title regardless of its location (the viz rc puts titles on the left)."""
    return ax.get_title(loc="left") or ax.get_title(loc="center") or ax.get_title(loc="right")


# --------------------------------------------------------------------------- #
# palette and backend helpers
# --------------------------------------------------------------------------- #
def test_palette_slots_are_distinct_and_fixed():
    assert len(V.CATEGORICAL) == 8
    assert len(set(V.CATEGORICAL)) == 8
    assert all(c.startswith("#") and len(c) == 7 for c in V.CATEGORICAL)
    assert V.riparian_color(0) == V.CATEGORICAL[0]
    assert V.riparian_color(7) == V.CATEGORICAL[7]
    # the palette is never cycled: a ninth series is grey, not slot 1 again
    assert V.riparian_color(8) == V.OTHER
    assert V.riparian_color(8) not in V.CATEGORICAL


@pytest.mark.parametrize("bad", [-1, 1.5, "0", True, None])
def test_riparian_color_rejects_bad_index(bad):
    with pytest.raises(ValueError):
        V.riparian_color(bad)


def test_sector_colors_cover_all_withdrawal_sectors():
    assert set(V.SECTOR_COLORS) == set(SECTOR_PRIORITY)
    assert Sector.ENVIRONMENT not in V.SECTOR_COLORS
    assert len(set(V.SECTOR_COLORS.values())) == len(SECTOR_PRIORITY)


def test_headless_detection(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    for var in ("MPLBACKEND", "DISPLAY", "WAYLAND_DISPLAY"):
        monkeypatch.delenv(var, raising=False)
    assert V._headless() is True
    monkeypatch.setenv("DISPLAY", ":0")
    assert V._headless() is False
    monkeypatch.delenv("DISPLAY")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    assert V._headless() is False
    monkeypatch.delenv("WAYLAND_DISPLAY")
    monkeypatch.setenv("MPLBACKEND", "module://matplotlib_inline.backend_inline")
    assert V._headless() is False  # an explicit backend (e.g. Jupyter inline) is never overridden
    monkeypatch.delenv("MPLBACKEND")
    monkeypatch.setattr(sys, "platform", "darwin")
    assert V._headless() is False
    monkeypatch.setattr(sys, "platform", "win32")
    assert V._headless() is False


def test_pyplot_uses_agg_when_headless(monkeypatch, mpl):
    monkeypatch.setattr(sys, "platform", "linux")
    for var in ("MPLBACKEND", "DISPLAY", "WAYLAND_DISPLAY"):
        monkeypatch.delenv(var, raising=False)
    matplotlib, plt = V._pyplot()
    assert matplotlib is mpl
    assert plt.get_backend().lower() == "agg"


# --------------------------------------------------------------------------- #
# plot_supply_ratio
# --------------------------------------------------------------------------- #
def test_plot_supply_ratio_returns_figure_and_saves(result, mpl, tmp_path):
    path = tmp_path / "supply.png"
    fig = V.plot_supply_ratio(result, save=path)
    assert is_figure(fig, mpl)
    assert_png(path)
    (ax,) = fig.axes
    names = result.riparian_names()
    assert len(ax.lines) == len(names) + 1  # riparians + basin line
    assert ax.get_legend() is not None
    assert [t.get_text() for t in ax.get_legend().get_texts()][: len(names)] == names
    lo, hi = ax.get_ylim()
    assert lo == 0.0 and hi >= 1.0
    assert ax.get_xlabel() == "Year"
    assert "viz-test" in title_of(ax)
    # the plotted data are the result's supply ratios, in year order
    for i, name in enumerate(names):
        ys = list(ax.lines[i].get_ydata())
        assert ys == pytest.approx(result.series(name, "supply_ratio"))
        assert list(ax.lines[i].get_xdata()) == result.years
        assert ax.lines[i].get_color() == V.riparian_color(i)


def test_plot_supply_ratio_options(result, mpl, tmp_path):
    fig = V.plot_supply_ratio(result, include_basin=False, title="custom", save=str(tmp_path / "s.png"))
    (ax,) = fig.axes
    assert len(ax.lines) == result.n_riparians
    assert title_of(ax) == "custom"
    assert_png(tmp_path / "s.png")
    # into an existing axes
    import matplotlib.pyplot as plt

    fig2, ax2 = plt.subplots()
    out = V.plot_supply_ratio(result, ax=ax2)
    assert out is fig2
    assert len(ax2.lines) == result.n_riparians + 1


def test_plot_supply_ratio_rejects_bad_input(mpl):
    with pytest.raises(ValueError, match="NexusResult"):
        V.plot_supply_ratio({"years": [1]})
    with pytest.raises(ValueError, match="NexusResult"):
        V.plot_supply_ratio(None)


def test_plot_supply_ratio_single_year(mpl):
    res = run_nexus(example_basin(), Scenario(name="one", years=1))
    fig = V.plot_supply_ratio(res)
    (ax,) = fig.axes
    assert all(len(line.get_xdata()) == 1 for line in ax.lines)


# --------------------------------------------------------------------------- #
# plot_nexus_indices
# --------------------------------------------------------------------------- #
def test_plot_nexus_indices_small_multiples(result, mpl, tmp_path):
    path = tmp_path / "indices.png"
    fig = V.plot_nexus_indices(result, save=path)
    assert is_figure(fig, mpl)
    assert_png(path)
    axes = visible_axes(fig)
    assert len(axes) == 4
    assert [title_of(a) for a in axes] == ["Water security", "Energy security", "Food security", "Nexus index"]
    for a in axes:
        assert len(a.lines) == result.n_riparians + 1
        assert a.get_ylim() == (0.0, 1.0)
        for line in a.lines:
            assert all(0.0 <= y <= 1.0 for y in line.get_ydata())
    assert fig.legends, "a shared legend is drawn below the panels"


def test_plot_nexus_indices_custom_keys(result, mpl):
    fig = V.plot_nexus_indices(result, keys=["water_security", "food_security"], include_basin=False)
    axes = visible_axes(fig)
    assert len(axes) == 2
    assert all(len(a.lines) == result.n_riparians for a in axes)
    fig = V.plot_nexus_indices(result, keys=["nexus_index"])
    assert len(visible_axes(fig)) == 1
    with pytest.raises(ValueError):
        V.plot_nexus_indices(result, keys=[])
    with pytest.raises(ValueError):
        V.plot_nexus_indices(result, keys=["no_such_field"])


# --------------------------------------------------------------------------- #
# plot_water_balance
# --------------------------------------------------------------------------- #
def test_plot_water_balance(result, mpl, tmp_path):
    basin = example_basin()
    path = tmp_path / "wb.png"
    fig = V.plot_water_balance(result, "Delta", basin=basin, save=path)
    assert is_figure(fig, mpl)
    assert_png(path)
    ax, ax2 = fig.axes
    # one bar container per sector (+ non-river supply and unmet demand, both present for Delta)
    assert len(ax.containers) >= len(SECTOR_PRIORITY)
    assert len(ax.lines) == 3  # inflow, outflow, environmental flow
    assert ax.get_legend() is not None
    assert "Delta" in title_of(ax)
    # the stack height equals the gross demand of each year
    totals = [0.0] * result.n_years
    for container in ax.containers:
        for k, bar in enumerate(container):
            totals[k] += bar.get_height()
    assert totals == pytest.approx(result.series("Delta", "total_demand_mm3"), rel=1e-9)
    # storage panel with capacity hairline
    assert ax2.get_ylim()[0] == 0.0
    capacity = basin.riparian("Delta").reservoir_capacity_mm3
    assert any(line.get_ydata()[0] == pytest.approx(capacity) for line in ax2.lines)
    assert list(ax2.lines[0].get_ydata()) == pytest.approx(result.series("Delta", "storage_end_mm3"))


def test_plot_water_balance_without_basin_and_errors(result, mpl):
    fig = V.plot_water_balance(result, "Highland")
    ax, ax2 = fig.axes
    assert len(ax2.lines) == 1  # no capacity line without the basin
    with pytest.raises(KeyError):
        V.plot_water_balance(result, "Atlantis")
    with pytest.raises(ValueError):
        V.plot_water_balance(result, "")
    with pytest.raises(ValueError):
        V.plot_water_balance(result, "Delta", basin="not a basin")
    with pytest.raises(ValueError):
        V.plot_water_balance(object(), "Delta")


# --------------------------------------------------------------------------- #
# plot_allocation_rules
# --------------------------------------------------------------------------- #
def test_plot_allocation_rules(comparison, mpl, tmp_path):
    path = tmp_path / "rules.png"
    fig = V.plot_allocation_rules(comparison, save=path)
    assert is_figure(fig, mpl)
    assert_png(path)
    ax, ax2 = fig.axes
    rules = list(comparison)
    assert [t.get_text() for t in ax.get_yticklabels()] == rules
    names = list(next(iter(comparison.values()))["awards"])
    assert len(ax.containers) == len(names)
    # the rows of compare_allocation_rules carry consumptive awards: the bars, the estate line and the claims
    # line are drawn on that one basis, so every stacked bar sums to the rule's total consumptive award
    assert ax.get_xlabel() == "Consumptive award (Mm3/yr)"
    for k, rule in enumerate(rules):
        total = sum(container[k].get_width() for container in ax.containers)
        assert total == pytest.approx(comparison[rule]["total_consumptive_award_mm3"])
        assert total <= comparison[rule]["estate"] + 1e-6
        assert total < comparison[rule]["total_awarded_mm3"]  # the gross caps exceed the consumptive awards
    assert ax.get_legend() is not None
    assert [t.get_text() for t in ax.get_legend().get_texts()] == names
    assert len(ax2.lines) >= len(names)
    assert ax2.get_xlim()[1] >= 1.0


def test_plot_allocation_rules_minimal_and_errors(mpl):
    minimal = {
        "proportional": {"awards": {"A": 2.0, "B": 4.0}, "claims": {"A": 4.0, "B": 4.0}},
        "cea": {"awards": {"A": 3.0, "B": 3.0}, "claims": {"A": 4.0, "B": 4.0}},
    }
    fig = V.plot_allocation_rules(minimal, title="t")
    assert fig._suptitle.get_text() == "t"
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Award (Mm3/yr)"  # no consumptive awards: the gross awards are drawn
    assert sum(container[0].get_width() for container in ax.containers) == pytest.approx(6.0)
    with pytest.raises(ValueError):
        V.plot_allocation_rules({})
    with pytest.raises(ValueError):
        V.plot_allocation_rules([("a", 1)])
    with pytest.raises(ValueError):
        V.plot_allocation_rules({"rule": {"gini": 0.1}})
    with pytest.raises(ValueError):  # neither satisfaction nor claims
        V.plot_allocation_rules({"rule": {"awards": {"A": 1.0}}})
    with pytest.raises(ValueError):  # inconsistent riparians between rules
        V.plot_allocation_rules({"r1": {"awards": {"A": 1.0}, "claims": {"A": 1.0}},
                                 "r2": {"awards": {"B": 1.0}, "claims": {"B": 1.0}}})


# --------------------------------------------------------------------------- #
# plot_pareto
# --------------------------------------------------------------------------- #
def test_plot_pareto(front, mpl, tmp_path):
    assert front, "the drought-year front should be feasible"
    path = tmp_path / "pareto.png"
    fig = V.plot_pareto(front, save=path)
    assert is_figure(fig, mpl)
    assert_png(path)
    ax, ax2 = fig.axes
    xs = list(ax.lines[0].get_xdata())
    assert xs == sorted(xs)
    assert xs == pytest.approx(sorted(p["min_supply_ratio"] for p in front))
    assert "USD" in ax.get_ylabel()
    assert "Gini" in ax2.get_ylabel()
    assert list(ax2.lines[0].get_ydata()) == pytest.approx([p["gini"] for p in sorted(front, key=lambda p: p["min_supply_ratio"])])


def test_plot_pareto_options_and_errors(mpl):
    synthetic = [
        {"min_supply_ratio": 0.5, "total_benefit_usd": 2.0e6},
        {"min_supply_ratio": 0.0, "total_benefit_usd": 3.0e6},
        {"min_supply_ratio": 1.0, "total_benefit_usd": 1.0e6},
    ]
    fig = V.plot_pareto(synthetic)  # no gini -> single panel, unsorted input is sorted
    (ax,) = fig.axes
    assert list(ax.lines[0].get_xdata()) == [0.0, 0.5, 1.0]
    assert list(ax.lines[0].get_ydata()) == pytest.approx([3.0, 2.0, 1.0])
    assert "million" in ax.get_ylabel()
    fig = V.plot_pareto([dict(p, gini=0.1) for p in synthetic], show_gini=False)
    assert len(fig.axes) == 1
    fig = V.plot_pareto([{"min_supply_ratio": 0.2, "total_benefit_usd": 50.0}])
    assert "USD/yr" in fig.axes[0].get_ylabel()
    with pytest.raises(ValueError):
        V.plot_pareto([])
    with pytest.raises(ValueError):
        V.plot_pareto([{"min_supply_ratio": 0.1}])
    with pytest.raises(ValueError):
        V.plot_pareto({"min_supply_ratio": 0.1, "total_benefit_usd": 1.0})
    with pytest.raises(ValueError):
        V.plot_pareto([{"min_supply_ratio": "x", "total_benefit_usd": 1.0}])


# --------------------------------------------------------------------------- #
# plot_nexus_radar
# --------------------------------------------------------------------------- #
def test_plot_nexus_radar_from_summary_dict(result, mpl, tmp_path):
    path = tmp_path / "radar.png"
    fig = V.plot_nexus_radar(result.summary(), save=path)
    assert is_figure(fig, mpl)
    assert_png(path)
    (ax,) = fig.axes
    assert ax.name == "polar"
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == result.riparian_names() + ["Basin"]
    assert len(ax.get_xticks()) == len(V.DEFAULT_RADAR_KEYS)
    assert ax.get_ylim() == (0.0, 1.0)
    for line in ax.lines:
        assert all(0.0 <= y <= 1.0 for y in line.get_ydata())


def test_plot_nexus_radar_accepts_report_and_result(result, mpl):
    from wefnexus.sustainability import assess

    fig = V.plot_nexus_radar(assess(result))
    assert [t.get_text() for t in fig.axes[0].get_legend().get_texts()] == result.riparian_names() + ["Basin"]
    fig = V.plot_nexus_radar(result, include_basin=False)
    assert [t.get_text() for t in fig.axes[0].get_legend().get_texts()] == result.riparian_names()


def test_plot_nexus_radar_keys_and_errors(result, mpl):
    summary = result.summary()
    fig = V.plot_nexus_radar(summary, keys=["water_security", "energy_security", "food_security"], title="T")
    (ax,) = fig.axes
    assert len(ax.get_xticks()) == 3
    assert [t.get_text() for t in ax.get_xticklabels()] == ["Water security", "Energy security", "Food security"]
    assert title_of(ax) == "T"
    with pytest.raises(ValueError):
        V.plot_nexus_radar({})
    with pytest.raises(ValueError):
        V.plot_nexus_radar(summary, keys=["water_security", "energy_security"])
    with pytest.raises(ValueError):
        V.plot_nexus_radar(summary, keys=["water_security", "energy_security", "falkenmark"])  # text field
    with pytest.raises(ValueError):
        V.plot_nexus_radar({"BASIN": {"water_security": 0.5}})
    with pytest.raises(ValueError):
        V.plot_nexus_radar("not a summary")


# --------------------------------------------------------------------------- #
# nexus_graph / plot_nexus_graph
# --------------------------------------------------------------------------- #
def test_nexus_graph_structure_and_weights(nx):
    from wefnexus.water import natural_flows

    basin = example_basin()
    G = V.nexus_graph(basin)
    assert isinstance(G, nx.DiGraph)
    names = basin.names()
    assert G.graph["basin"] == basin.name
    assert G.graph["riparians"] == names
    assert G.graph["routed"] is False
    for n in names:
        assert G.nodes[n]["kind"] == "riparian"
        assert G.nodes[n]["position"] == basin.riparian(n).position
    # river chain headwater -> Highland -> Midland -> Delta -> sea with natural-flow weights
    flows = natural_flows(basin)
    chain = ["headwater"] + names + ["sea"]
    for u, v in zip(chain, chain[1:]):
        assert G.has_edge(u, v)
        assert G.edges[u, v]["kind"] == "river"
    assert G.edges["headwater", names[0]]["weight"] == pytest.approx(basin.headwater_inflow_mm3)
    for k, n in enumerate(names):
        assert G.edges[n, chain[k + 2]]["weight"] == pytest.approx(flows[n])
        assert G.edges[n, chain[k + 2]]["env_flow_mm3"] == basin.riparian(n).demand.environmental
    # sector nodes with demand weights
    for r in basin.riparians:
        for s in SECTOR_PRIORITY:
            node = f"{r.name}:{s.value}"
            assert G.nodes[node]["kind"] == "sector"
            assert G.nodes[node]["sector"] == s.value
            e = G.edges[r.name, node]
            assert e["kind"] == "withdrawal"
            assert e["weight"] == pytest.approx(r.demand.withdrawals()[s])
            assert e["consumption_mm3"] + e["return_flow_mm3"] == pytest.approx(e["weight"])
        assert G.edges[f"{r.name}:runoff", r.name]["weight"] == pytest.approx(r.local_inflow_mm3)
        assert G.edges[f"{r.name}:groundwater", r.name]["weight"] == pytest.approx(r.groundwater_abstraction_mm3)
    assert "Delta:desalination" in G and "Highland:desalination" not in G  # only positive volumes
    n_sources = sum(1 for _, d in G.nodes(data=True) if d.get("kind") == "source")
    assert G.number_of_nodes() == 2 + len(names) * (1 + len(SECTOR_PRIORITY)) + n_sources
    assert G.number_of_edges() == (len(names) + 1) + len(names) * len(SECTOR_PRIORITY) + n_sources


def test_nexus_graph_flow_factor_and_balance(nx):
    from wefnexus.water import route_basin

    basin = example_basin()
    G = V.nexus_graph(basin, flow_factor=0.5)
    assert G.edges["headwater", "Highland"]["weight"] == pytest.approx(0.5 * basin.headwater_inflow_mm3)
    assert G.edges["Highland:runoff", "Highland"]["weight"] == pytest.approx(0.5 * basin.riparian("Highland").local_inflow_mm3)
    G0 = V.nexus_graph(basin, flow_factor=0.0)
    assert G0.edges["headwater", "Highland"]["weight"] == 0.0
    assert "Highland:runoff" not in G0  # zero-volume sources are omitted
    bal = route_basin(basin, 0.6)
    G = V.nexus_graph(basin, balance=bal)
    assert G.graph["routed"] is True
    for reach in bal.reaches:
        for s in SECTOR_PRIORITY:
            assert G.edges[reach.name, f"{reach.name}:{s.value}"]["weight"] == pytest.approx(reach.withdrawals[s])
    assert G.edges["Highland", "Midland"]["weight"] == pytest.approx(bal.reach("Highland").outflow_mm3)
    assert G.edges["Delta", "sea"]["weight"] == pytest.approx(bal.outflow_to_sea_mm3)
    assert G.edges["headwater", "Highland"]["weight"] == pytest.approx(bal.reach("Highland").upstream_inflow_mm3)


def test_nexus_graph_does_not_mutate_basin_and_validates(nx):
    from wefnexus.models import Basin, Riparian, WaterDemand
    from wefnexus.water import route_basin

    basin = example_basin()
    V.nexus_graph(basin)
    V.nexus_graph(basin, balance=route_basin(basin))
    assert basin == example_basin()
    with pytest.raises(ValueError):
        V.nexus_graph("basin")
    with pytest.raises(ValueError):
        V.nexus_graph(basin, flow_factor=-1.0)
    with pytest.raises(ValueError):
        V.nexus_graph(basin, balance="balance")
    other = Basin("Other", [Riparian("Solo", 1e6, 1e9, 100.0, WaterDemand(municipal=10.0))])
    with pytest.raises(ValueError):
        V.nexus_graph(basin, balance=route_basin(other))
    # a single-riparian basin without reservoir or crops still gives a valid chain
    G = V.nexus_graph(other)
    assert list(nx.shortest_path(G, "headwater", "sea")) == ["headwater", "Solo", "sea"]


def test_plot_nexus_graph(nx, mpl, tmp_path):
    from wefnexus.water import route_basin

    basin = example_basin()
    path = tmp_path / "graph.png"
    fig = V.plot_nexus_graph(V.nexus_graph(basin), save=path)
    assert is_figure(fig, mpl)
    assert_png(path)
    (ax,) = fig.axes
    assert not ax.axison
    assert ax.get_legend() is not None
    assert "natural flows" in title_of(ax)
    fig = V.plot_nexus_graph(basin, balance=route_basin(basin, 0.6), title="routed")
    assert title_of(fig.axes[0]) == "routed"
    with pytest.raises(ValueError):
        V.plot_nexus_graph("basin")
    with pytest.raises(ValueError):
        V.plot_nexus_graph(nx.DiGraph())


# --------------------------------------------------------------------------- #
# cross-cutting: no plt.show(), saving formats, no mutation
# --------------------------------------------------------------------------- #
def test_plots_never_call_show(result, comparison, front, mpl, monkeypatch):
    import matplotlib.pyplot as plt

    def boom(*args, **kwargs):  # pragma: no cover - should never run
        raise AssertionError("plt.show() must not be called by wefnexus.viz")

    monkeypatch.setattr(plt, "show", boom)
    V.plot_supply_ratio(result)
    V.plot_nexus_indices(result)
    V.plot_water_balance(result, "Midland")
    V.plot_allocation_rules(comparison)
    V.plot_pareto(front)
    V.plot_nexus_radar(result.summary())
    pytest.importorskip("networkx")
    V.plot_nexus_graph(example_basin())


def test_save_creates_directories_and_other_formats(result, mpl, tmp_path):
    nested = tmp_path / "a" / "b" / "supply.png"
    V.plot_supply_ratio(result, save=nested)
    assert_png(nested)
    svg = tmp_path / "supply.svg"
    V.plot_supply_ratio(result, save=svg)
    head = svg.read_bytes()[:200].lower()
    assert b"<svg" in head or b"<?xml" in head
    with pytest.raises(ValueError):
        V.plot_supply_ratio(result, save="")


def test_plots_do_not_mutate_inputs(result, comparison, front, mpl):
    import copy

    before_records = [r.to_dict() for r in result.records]
    before_cmp = copy.deepcopy(comparison)
    before_front = copy.deepcopy(front)
    summary = result.summary()
    before_summary = copy.deepcopy(summary)
    V.plot_supply_ratio(result)
    V.plot_nexus_indices(result)
    V.plot_water_balance(result, "Delta")
    V.plot_allocation_rules(comparison)
    V.plot_pareto(front)
    V.plot_nexus_radar(summary)
    assert [r.to_dict() for r in result.records] == before_records
    assert comparison == before_cmp
    assert front == before_front
    assert summary == before_summary


# --------------------------------------------------------------------------- #
# examples/run_example.py and the notebook
# --------------------------------------------------------------------------- #
def _run_example(tmp_path: Path, extra_args, env_extra):
    env = dict(os.environ)
    env.update(env_extra)
    env.setdefault("MPLBACKEND", "Agg")
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, str(EXAMPLE), "--outdir", str(tmp_path)] + list(extra_args)
    return subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=600)


def test_run_example_fast_mode_subprocess(tmp_path):
    """WEFNEXUS_FAST=1 runs the whole script on a 3-year horizon."""
    proc = _run_example(tmp_path, [], {"WEFNEXUS_FAST": "1"})
    assert proc.returncode == 0, proc.stderr[-4000:]
    out = proc.stdout
    assert "Horizon: 3 years (WEFNEXUS_FAST=1)" in out
    assert "Scenario comparison" in out
    assert "Zone of possible agreement" in out
    assert "Price of fairness" in out
    assert "Done" in out
    assert "Traceback" not in proc.stderr
    # CSV outputs
    comparison = tmp_path / "scenario_comparison.csv"
    assert comparison.is_file()
    import csv

    with open(comparison, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3 * 4  # 3 scenarios x (3 riparians + basin row)
    assert {r["scenario"] for r in rows} == {"baseline", "climate_change", "combined_adaptation"}
    assert all(r["years"] == "3" for r in rows)
    for name in ("records_baseline.csv", "sustainability_baseline.csv", "allocation_rules_normal.csv",
                 "conflict_risk.csv", "pareto_front_drought.csv"):
        assert (tmp_path / name).is_file(), name
    with open(tmp_path / "records_baseline.csv", newline="") as fh:
        assert len(list(csv.DictReader(fh))) == 3 * 3  # 3 years x 3 riparians
    # figures, when matplotlib is available
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        assert not list(tmp_path.glob("*.png"))
    else:
        pngs = sorted(p.name for p in tmp_path.glob("*.png"))
        assert "supply_ratio_baseline.png" in pngs
        assert "allocation_rules_drought.png" in pngs
        assert "pareto_front.png" in pngs
        for p in tmp_path.glob("*.png"):
            assert_png(p)
        try:
            import networkx  # noqa: F401
        except ImportError:
            assert "nexus_graph.png" not in pngs
        else:
            assert "nexus_graph.png" in pngs


def test_run_example_years_override_and_no_plots(tmp_path):
    proc = _run_example(tmp_path, ["--years", "2", "--no-plots", "--pareto-points", "2"], {"WEFNEXUS_FAST": "1"})
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert "Horizon: 2 years" in proc.stdout
    assert not list(tmp_path.glob("*.png"))
    assert (tmp_path / "scenario_comparison.csv").is_file()
    assert "0 figures" in proc.stdout


def test_run_example_rejects_bad_years(tmp_path):
    proc = _run_example(tmp_path, ["--years", "0", "--no-plots"], {})
    assert proc.returncode == 2


def test_notebook_is_valid_and_documented():
    nbformat = pytest.importorskip("nbformat")
    nb = nbformat.read(str(NOTEBOOK), as_version=4)
    nbformat.validate(nb)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    md_cells = [c for c in nb.cells if c.cell_type == "markdown"]
    assert len(md_cells) >= 10
    assert len(code_cells) >= 10
    first = code_cells[0].source
    assert "# !pip install" in first and "YousefWEF" in first  # install line present but commented out
    assert "Colab" in first
    assert all(not line.lstrip().startswith("!") for c in code_cells for line in c.source.splitlines())
    joined = "\n".join(c.source for c in code_cells)
    for snippet in ("run_scenarios", "compare_allocation_rules", "negotiate", "conflict_risk_index",
                    "pareto_front", "nexus_graph", "plot_supply_ratio", "assess("):
        assert snippet in joined, snippet
    assert nb.metadata["kernelspec"]["language"] == "python"


def test_notebook_code_runs_top_to_bottom(tmp_path):
    """Execute the notebook's code cells in order in a fresh interpreter (no kernel needed)."""
    nbformat = pytest.importorskip("nbformat")
    pytest.importorskip("matplotlib")
    nb = nbformat.read(str(NOTEBOOK), as_version=4)
    chunks = []
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        lines = [l for l in cell.source.splitlines() if not l.lstrip().startswith(("%", "!"))]
        chunks.append("\n".join(lines))
    script = tmp_path / "notebook_cells.py"
    script.write_text("\n\n".join(chunks), encoding="utf-8")
    env = dict(os.environ, MPLBACKEND="Agg")
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, str(script)], cwd=str(tmp_path), env=env, capture_output=True,
                          text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert (tmp_path / "output" / "scenario_comparison.csv").is_file()
    assert (tmp_path / "output" / "supply_ratio.png").is_file()
