"""Tests for :mod:`wefnexus.cli` and ``python -m wefnexus``.

Every subcommand is driven in-process through ``cli.main([...])`` with
``capsys`` / ``tmp_path`` on the example basin with ``--years 3``:
``run``, ``allocate``, ``negotiate``, ``compare``, ``pareto``, ``report`` and
``export-basin``.  Checks cover exit codes (0 success, 1 no result, 2 usage
error), the printed tables (parsed back and compared with closed-form
literature results such as the Talmud contested-garment awards), the CSV /
JSON side outputs (strict JSON, no ``Infinity``), file-based basins and
scenarios, argument helpers, and no-mutation of basin / result objects.
"""
from __future__ import annotations

import csv
import dataclasses
import doctest
import io as pyio
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

import wefnexus.cli as cli_module
from wefnexus import __version__, cli
from wefnexus import io as IO
from wefnexus import scenarios as S
from wefnexus.cli import main
from wefnexus.data import example_basin
from wefnexus.diplomacy import Treaty
from wefnexus.models import Scenario, Sector
from wefnexus.nexus import ALLOCATION_RULES, NexusResult, run_nexus
from wefnexus.scenarios import TABLE_COLUMNS

RIPARIANS = ["Highland", "Midland", "Delta"]
SCENARIO_NAMES = list(S.SCENARIOS)
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def run_cli(capsys, *args):
    """Run ``main`` in-process; return ``(exit code, stdout, stderr)``."""
    code = main(list(args))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def table(out, first_col):
    """Parse the first ``format_table`` block whose header starts with ``first_col``.

    Returns ``(header, rows)`` with rows as ``{column: cell string}``.
    """
    lines = out.splitlines()
    for i, line in enumerate(lines):
        parts = line.split()
        if parts and parts[0] == first_col and i + 1 < len(lines) and lines[i + 1].strip() and set(lines[i + 1].strip()) == {"-"}:
            header = parts
            rows = []
            for row in lines[i + 2:]:
                cells = row.split()
                if not cells or set(row.strip()) == {"-"} or len(cells) != len(header):
                    break
                rows.append(dict(zip(header, cells)))
            return header, rows
    raise AssertionError(f"no table starting with {first_col!r} in output:\n{out}")


def strict_json_load(path):
    """Load JSON refusing NaN / Infinity tokens."""

    def refuse(token):
        raise AssertionError(f"non-strict JSON token {token} in {path}")

    with open(path, encoding="utf-8") as fh:
        return json.load(fh, parse_constant=refuse)


@pytest.fixture
def basin_file(tmp_path):
    path = tmp_path / "azura.json"
    IO.basin_to_json(example_basin(), path)
    return path


@pytest.fixture
def result3():
    return run_nexus(example_basin(), Scenario(name="baseline", years=3))


# ---------------------------------------------------------------------------
# module hygiene
# ---------------------------------------------------------------------------
def test_doctests_pass():
    failed, attempted = doctest.testmod(cli_module)
    assert attempted > 0
    assert failed == 0


def test_all_exports_exist_with_docstrings():
    for name in cli_module.__all__:
        obj = getattr(cli_module, name)
        if callable(obj):
            assert obj.__doc__ and obj.__doc__.strip(), name


def test_module_imports_cleanly():
    subprocess.run([sys.executable, "-c", "import wefnexus.cli, wefnexus.__main__"], check=True)


def test_console_script_target_is_main():
    assert callable(cli.main)
    assert cli.PROG == "wefnexus"
    assert (cli.EXIT_OK, cli.EXIT_NO_RESULT, cli.EXIT_USAGE) == (0, 1, 2)


# ---------------------------------------------------------------------------
# global behaviour
# ---------------------------------------------------------------------------
def test_no_command_is_usage_error(capsys):
    code, out, err = run_cli(capsys)
    assert code == 2
    assert out == ""
    assert "usage" in err and "error" in err


def test_unknown_command_is_usage_error(capsys):
    code, out, err = run_cli(capsys, "bogus")
    assert code == 2
    assert "invalid choice" in err


def test_unknown_option_is_usage_error(capsys):
    code, _, err = run_cli(capsys, "run", "--bogus")
    assert code == 2
    assert "unrecognized" in err


def test_version_and_help_exit_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert f"wefnexus {__version__}" in capsys.readouterr().out
    for args in (["--help"], ["run", "--help"], ["allocate", "-h"], ["export-basin", "--help"]):
        with pytest.raises(SystemExit) as exc:
            main(args)
        assert exc.value.code == 0
        assert "usage" in capsys.readouterr().out


def test_explicit_streams_bypass_sys_streams(capsys):
    out, err = pyio.StringIO(), pyio.StringIO()
    assert main(["run", "--years", "3"], stdout=out, stderr=err) == 0
    assert "BASIN" in out.getvalue() and err.getvalue() == ""
    assert main(["run", "--years", "0"], stdout=out, stderr=err) == 2
    assert "error" in err.getvalue()
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_python_m_wefnexus_subprocess():
    proc = subprocess.run([sys.executable, "-m", "wefnexus", "--version"], capture_output=True, text=True)
    assert proc.returncode == 0 and __version__ in proc.stdout
    proc = subprocess.run([sys.executable, "-m", "wefnexus", "run", "--years", "2"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "BASIN" in proc.stdout and proc.stderr == ""
    proc = subprocess.run([sys.executable, "-m", "wefnexus"], capture_output=True, text=True)
    assert proc.returncode == 2 and "usage" in proc.stderr


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def test_run_example_three_years(capsys):
    code, out, err = run_cli(capsys, "run", "--basin", "example", "--scenario", "baseline", "--years", "3")
    assert code == 0 and err == ""
    assert "Basin: Azura River (stylised)" in out
    assert "scenario: baseline" in out and "2025-2027 (3 years)" in out
    assert "allocation rule: treaty" in out
    header, rows = table(out, "riparian")
    assert [r["riparian"] for r in rows] == RIPARIANS + ["BASIN"]
    assert header == ["riparian"] + [label for _, label in cli.RUN_COLUMNS]
    for r in rows:
        for col in ("supply", "min_supply", "reliability", "water", "energy", "food", "nexus", "env_met"):
            assert 0.0 <= float(r[col]) <= 1.0, (r["riparian"], col)
        assert float(r["hydro_GWh"]) >= 0.0 and float(r["storage_Mm3"]) >= 0.0
    assert "mass balance error" in out and "equity index" in out
    assert "\x1b[" not in out  # no colour escapes


def test_run_matches_library_summary(capsys, result3):
    _, out, _ = run_cli(capsys, "run", "--years", "3")
    _, rows = table(out, "riparian")
    summary = result3.summary()
    for r in rows:
        assert float(r["nexus"]) == pytest.approx(summary[r["riparian"]]["nexus_index"], abs=5e-4)
        assert float(r["supply"]) == pytest.approx(summary[r["riparian"]]["supply_ratio"], abs=5e-4)


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_run_every_library_scenario(capsys, name):
    code, out, err = run_cli(capsys, "run", "--scenario", name, "--years", "3")
    assert code == 0 and err == ""
    assert f"scenario: {name}" in out and "(3 years)" in out
    assert "BASIN" in out


def test_run_scenario_name_is_case_and_separator_insensitive(capsys):
    code, out, _ = run_cli(capsys, "run", "--scenario", "Climate-Change", "--years", "3")
    assert code == 0 and "scenario: climate_change" in out


@pytest.mark.parametrize("rule", list(ALLOCATION_RULES) + ["CG", "maimonides"])
def test_run_rule_override(capsys, rule):
    code, out, _ = run_cli(capsys, "run", "--years", "3", "--rule", rule)
    assert code == 0
    assert f"allocation rule: {cli.canonical_model_rule(rule)}" in out


def test_run_writes_csv_and_json(capsys, tmp_path):
    csv_path, json_path = tmp_path / "run.csv", tmp_path / "run.json"
    code, out, _ = run_cli(capsys, "run", "--years", "3", "--csv", str(csv_path), "--json", str(json_path))
    assert code == 0
    assert f"wrote CSV: {csv_path}" in out and f"wrote JSON: {json_path}" in out
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 9
    assert {"name", "year", "nexus_index"} <= set(rows[0])
    assert sorted({r["name"] for r in rows}) == sorted(RIPARIANS)
    data = strict_json_load(json_path)
    assert data["basin"] == "Azura River (stylised)"
    assert data["scenario"]["name"] == "baseline" and data["scenario"]["years"] == 3
    assert data["scenario"]["format"] == IO.SCENARIO_FORMAT
    assert data["allocation_rule"] == "treaty"
    assert data["years"] == [2025, 2026, 2027]
    assert len(data["flow_factors"]) == 3
    assert set(data["summary"]) == set(RIPARIANS) | {"BASIN"}
    assert len(data["records"]) == 9
    assert 0.0 <= data["summary"]["BASIN"]["nexus_index"] <= 1.0


def test_run_json_is_strict_without_infinity(capsys, tmp_path):
    json_path = tmp_path / "uni.json"
    code, _, _ = run_cli(capsys, "run", "--scenario", "unilateral", "--years", "3", "--json", str(json_path))
    assert code == 0
    text = json_path.read_text(encoding="utf-8")
    assert "Infinity" not in text and "NaN" not in text
    data = strict_json_load(json_path)
    # no caps under upstream priority: the inf entitlement is written as null
    assert all(rec["entitlement_mm3"] is None for rec in data["records"])
    assert data["summary"]["Highland"]["entitlement_mm3"] is None


def test_run_from_basin_file_matches_example(capsys, basin_file):
    _, expected, _ = run_cli(capsys, "run", "--years", "3")
    code, out, err = run_cli(capsys, "run", "--basin", str(basin_file), "--years", "3")
    assert code == 0 and err == ""
    assert out == expected


def test_run_from_scenario_file(capsys, tmp_path):
    sc = Scenario(name="custom-dry", years=2, flow_change_pct_by_end=-30.0, drought_years=[1], description="hand made")
    path = tmp_path / "scenario.json"
    IO.scenario_to_json(sc, path)
    code, out, _ = run_cli(capsys, "run", "--scenario", str(path))
    assert code == 0
    assert "scenario: custom-dry" in out and "(2 years)" in out and "hand made" in out
    assert "flow factors: min 0.420" in out  # year 1: (1 - 0.3) * (1 - 0.4)
    code, out, _ = run_cli(capsys, "run", "--scenario", str(path), "--years", "3")
    assert code == 0 and "(3 years)" in out


@pytest.mark.parametrize(
    "args",
    [
        ["run", "--years", "0"],
        ["run", "--years", "-3"],
        ["run", "--years", "two"],
        ["run", "--rule", "bogus"],
        ["run", "--refill", "2"],
        ["run", "--refill", "-0.1"],
        ["run", "--scenario", "nope"],
        ["run", "--scenario", ""],
        ["run", "--basin", "/no/such/basin.json"],
        ["run", "--basin", ""],
        ["run", "--plot", ""],
        ["run", "--plot", "   "],
        ["run", "--plot"],
    ],
)
def test_run_usage_errors(capsys, args):
    code, out, err = run_cli(capsys, *args)
    assert code == 2
    assert "error" in err


def test_run_invalid_basin_files(capsys, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    code, _, err = run_cli(capsys, "run", "--basin", str(bad), "--years", "3")
    assert code == 2 and "not valid JSON" in err
    sc = tmp_path / "sc.json"
    IO.scenario_to_json(Scenario(), sc)
    code, _, err = run_cli(capsys, "run", "--basin", str(sc), "--years", "3")
    assert code == 2 and "format" in err
    typo = tmp_path / "typo.json"
    doc = IO.basin_to_dict(example_basin())
    doc["riparians"][0]["polulation"] = doc["riparians"][0].pop("population")
    typo.write_text(json.dumps(doc), encoding="utf-8")
    code, _, err = run_cli(capsys, "run", "--basin", str(typo), "--years", "3")
    assert code == 2 and "polulation" in err


def test_run_unwritable_output_is_error(capsys, tmp_path):
    code, _, err = run_cli(capsys, "run", "--years", "3", "--csv", str(tmp_path / "nodir" / "x.csv"))
    assert code == 2 and "error" in err


# ---------------------------------------------------------------------------
# run --plot DIR (ARCHITECTURE.md section 10) and write_plots
# ---------------------------------------------------------------------------
def assert_png(path):
    assert path.is_file(), path
    with open(path, "rb") as fh:
        assert fh.read(8) == PNG_MAGIC, path


def open_figure_count():
    import matplotlib.pyplot as plt

    return len(plt.get_fignums())


def test_run_plot_option_is_documented():
    assert cli.PLOT_FILES == ("supply_ratio.png", "nexus_indices.png", "nexus_radar.png")
    assert all(name.endswith(".png") for name in cli.PLOT_FILES)
    assert len(set(cli.PLOT_FILES)) == len(cli.PLOT_FILES)
    help_text = cli.build_parser().parse_args(["run", "--plot", "x"])
    assert help_text.plot == "x"
    # the option appears in the subcommand help with its file names
    sub = [a for a in cli.build_parser()._subparsers._group_actions[0].choices.values()]
    run_help = [p for p in sub if p.prog.endswith(" run")][0].format_help()
    assert "--plot DIR" in run_help and "supply_ratio.png" in run_help and "matplotlib" in run_help


def test_run_plot_writes_png_figures(capsys, tmp_path):
    pytest.importorskip("matplotlib")
    figs = tmp_path / "nested" / "figs"  # does not exist yet: created with parents
    before = open_figure_count()
    code, out, err = run_cli(capsys, "run", "--years", "3", "--plot", str(figs))
    assert code == 0 and err == ""
    # the summary table is still printed before the figures
    _, rows = table(out, "riparian")
    assert [r["riparian"] for r in rows] == RIPARIANS + ["BASIN"]
    assert "wrote figures: " in out
    written = out.split("wrote figures: ", 1)[1].splitlines()[0].split(", ")
    assert written == [str(figs / name) for name in cli.PLOT_FILES]
    for name in cli.PLOT_FILES:
        assert_png(figs / name)
        assert f"{name}" in out
    assert sorted(p.name for p in figs.iterdir()) == sorted(cli.PLOT_FILES)
    # every figure is closed again
    assert open_figure_count() == before


def test_run_plot_with_csv_and_json(capsys, tmp_path):
    pytest.importorskip("matplotlib")
    csv_path, json_path, figs = tmp_path / "r.csv", tmp_path / "r.json", tmp_path / "figs"
    code, out, _ = run_cli(
        capsys, "run", "--years", "3", "--scenario", "climate_change", "--rule", "talmud",
        "--csv", str(csv_path), "--json", str(json_path), "--plot", str(figs),
    )
    assert code == 0
    assert out.index("wrote CSV: ") < out.index("wrote JSON: ") < out.index("wrote figures: ")
    assert csv_path.is_file() and strict_json_load(json_path)["allocation_rule"] == "talmud"
    for name in cli.PLOT_FILES:
        assert_png(figs / name)


def test_run_plot_into_existing_directory_overwrites(capsys, tmp_path):
    pytest.importorskip("matplotlib")
    figs = tmp_path / "figs"
    figs.mkdir()
    stale = figs / cli.PLOT_FILES[0]
    stale.write_bytes(b"stale")
    extra = figs / "keep.txt"
    extra.write_text("keep", encoding="utf-8")
    code, out, _ = run_cli(capsys, "run", "--years", "2", "--plot", str(figs))
    assert code == 0 and "wrote figures" in out
    assert_png(stale)  # overwritten with a real PNG
    assert extra.read_text(encoding="utf-8") == "keep"  # other files untouched


def test_run_plot_accepts_relative_directory(capsys, tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    monkeypatch.chdir(tmp_path)
    code, out, _ = run_cli(capsys, "run", "--years", "2", "--plot", "out")
    assert code == 0
    assert f"wrote figures: {os.path.join('out', cli.PLOT_FILES[0])}" in out
    for name in cli.PLOT_FILES:
        assert_png(tmp_path / "out" / name)


def test_run_plot_without_matplotlib_prints_hint(capsys, tmp_path, monkeypatch):
    figs = tmp_path / "figs"
    monkeypatch.setitem(sys.modules, "matplotlib", None)  # makes `import matplotlib` raise ImportError
    code, out, err = run_cli(capsys, "run", "--years", "3", "--plot", str(figs))
    assert code == 0 and err == ""
    _, rows = table(out, "riparian")  # the run itself still succeeds
    assert len(rows) == 4
    assert "figures skipped" in out and "matplotlib" in out and "pip install" in out
    assert "wrote figures" not in out
    assert not figs.exists()  # nothing is created when the figures cannot be drawn


def test_run_plot_path_is_a_file_is_error(capsys, tmp_path):
    pytest.importorskip("matplotlib")
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    code, _, err = run_cli(capsys, "run", "--years", "2", "--plot", str(blocker))
    assert code == 2 and "error" in err
    assert blocker.read_text(encoding="utf-8") == "x"


def test_write_plots_helper(result3, tmp_path):
    pytest.importorskip("matplotlib")
    before_summary = result3.summary()
    before_records = result3.to_records()
    n_open = open_figure_count()
    paths = cli.write_plots(result3, tmp_path / "a" / "b")  # path-like accepted, parents created
    assert paths == [str(tmp_path / "a" / "b" / name) for name in cli.PLOT_FILES]
    for p in paths:
        assert_png(tmp_path / "a" / "b" / os.path.basename(p))
    assert open_figure_count() == n_open  # figures closed
    # no mutation of the result
    assert result3.summary() == before_summary
    assert result3.to_records() == before_records
    # a plain string works too and re-running overwrites in place
    again = cli.write_plots(result3, str(tmp_path / "a" / "b"))
    assert again == paths
    assert sorted(p.name for p in (tmp_path / "a" / "b").iterdir()) == sorted(cli.PLOT_FILES)


def test_write_plots_validation(result3, tmp_path):
    with pytest.raises(ValueError, match="NexusResult"):
        cli.write_plots({"not": "a result"}, tmp_path)
    for bad in ("", "  ", None, True):
        with pytest.raises(ValueError, match="plot"):
            cli.write_plots(result3, bad)
    with pytest.raises(ValueError, match="directory path"):
        cli.write_plots(result3, 3.5)
    assert sorted(p.name for p in tmp_path.iterdir()) == []  # nothing written on invalid input


def test_write_plots_import_error_is_explicit(result3, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    target = tmp_path / "figs"
    with pytest.raises(ImportError, match="pip install matplotlib"):
        cli.write_plots(result3, target)
    assert not target.exists()


# ---------------------------------------------------------------------------
# allocate
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "estate, expected",
    [(100.0, (100 / 3, 100 / 3, 100 / 3)), (200.0, (50.0, 75.0, 75.0)), (300.0, (50.0, 100.0, 150.0))],
)
def test_allocate_talmud_contested_garment(capsys, estate, expected):
    """Aumann & Maschler (1985): claims (100, 200, 300)."""
    code, out, err = run_cli(capsys, "allocate", "--rule", "talmud", "--estate", str(estate), "--claims", "a=100,b=200,c=300")
    assert code == 0 and err == ""
    assert "rule: talmud" in out
    _, rows = table(out, "claimant")
    assert [r["claimant"] for r in rows] == ["a", "b", "c"]
    for r, e in zip(rows, expected):
        assert float(r["award"]) == pytest.approx(e, abs=0.06)
        assert float(r["satisfaction"]) == pytest.approx(e / float(r["claim"]), abs=1e-3)
    assert f"total awarded {estate:,.1f}" in out
    assert "gini" in out


def test_allocate_equal_awards_have_zero_gini(capsys):
    _, out, _ = run_cli(capsys, "allocate", "--rule", "cea", "--estate", "90", "--claims", "a=100,b=200,c=300")
    assert "gini 0.000" in out


def test_allocate_estate_above_claims_honours_every_claim(capsys):
    _, out, _ = run_cli(capsys, "allocate", "--rule", "proportional", "--estate", "1000", "--claims", "a=100,b=200,c=300")
    _, rows = table(out, "claimant")
    assert [float(r["award"]) for r in rows] == pytest.approx([100.0, 200.0, 300.0])
    assert "total awarded 600.0" in out and "shortfall 0.0" in out


def test_allocate_bare_numbers_and_aliases(capsys):
    code, out, _ = run_cli(capsys, "allocate", "--rule", "CG", "--estate", "200", "--claims", "100,200,300")
    assert code == 0 and "rule: talmud" in out
    _, rows = table(out, "claimant")
    assert [r["claimant"] for r in rows] == ["c1", "c2", "c3"]
    assert [float(r["award"]) for r in rows] == pytest.approx([50.0, 75.0, 75.0])


def test_allocate_all_rules(capsys):
    code, out, _ = run_cli(capsys, "allocate", "--rule", "all", "--estate", "200", "--claims", "a=100,b=200,c=300")
    assert code == 0
    header, rows = table(out, "rule")
    assert header[:4] == ["rule", "a", "b", "c"] and "gini" in header and "min_satisfaction" in header
    assert {r["rule"] for r in rows} == {"proportional", "cea", "cel", "talmud", "ap", "equal", "upstream_priority"}
    for r in rows:
        assert float(r["total"]) == pytest.approx(200.0, abs=0.6)
        assert 0.0 <= float(r["gini"]) <= 1.0
    by_rule = {r["rule"]: r for r in rows}
    assert float(by_rule["upstream_priority"]["a"]) == pytest.approx(100.0)
    assert float(by_rule["upstream_priority"]["c"]) == pytest.approx(0.0)
    assert float(by_rule["cea"]["a"]) == pytest.approx(200 / 3, abs=0.06)


def test_allocate_from_basin(capsys):
    code, out, err = run_cli(capsys, "allocate", "--basin", "example", "--rule", "talmud")
    assert code == 0 and err == ""
    # estate = natural flow minus the terminal in-stream requirement (1 500), not minus the 7 000 sum;
    # gross claims = river demands (1 500 / 7 350 / 16 300), consumptive claims = claims * consumption ratio
    assert "basin 'Azura River (stylised)'" in out and "demand claims" in out and "storage excluded" in out
    assert "estate 26,500.0" in out and "total claims 25,150.0 (consumptive 11,468.1)" in out and "shortfall 0.0" in out
    header, rows = table(out, "claimant")
    assert header == ["claimant", "claim", "c-claim", "c-award", "award", "satisfaction", "routed_withdrawal"]
    assert [r["claimant"] for r in rows] == RIPARIANS
    assert [float(r["claim"]) for r in rows] == pytest.approx([1500.0, 7350.0, 16300.0])
    assert [float(r["c-claim"]) for r in rows] == pytest.approx([549.7, 3322.2, 7596.2], abs=0.6)
    # no rationing at mean flow: consumptive awards = consumptive claims, gross caps = river demands
    assert [float(r["c-award"]) for r in rows] == pytest.approx([549.7, 3322.2, 7596.2], abs=0.6)
    assert [float(r["award"]) for r in rows] == pytest.approx([1500.0, 7350.0, 16300.0], abs=0.6)
    assert [float(r["routed_withdrawal"]) for r in rows] == pytest.approx([1500.0, 7350.0, 16300.0], abs=0.6)
    assert all(float(r["satisfaction"]) == pytest.approx(1.0) for r in rows)
    assert "total awarded 25,150.0 (consumptive 11,468.1)" in out
    assert "outflow to sea" in out and "env flow met share" in out


def test_allocate_from_basin_all_rules_and_estate_override(capsys):
    code, out, _ = run_cli(capsys, "allocate", "--basin", "example", "--rule", "all", "--estate", "1000", "--flow-factor", "0.5")
    assert code == 0 and "estate 1,000.0" in out and "flow factor 0.5" in out
    header, rows = table(out, "rule")
    assert "c-total" in header and "outflow_to_sea_mm3" in header and "env_flow_met_share" in header
    assert len(rows) == 7
    for r in rows:
        assert float(r["c-total"]) == pytest.approx(1000.0, abs=1.0)  # the consumptive estate is exhausted
        assert float(r["total"]) > 1000.0  # the gross caps exceed the consumptive volume they deliver


def test_allocate_from_basin_treaty_basis_and_drought(capsys):
    code, out, _ = run_cli(capsys, "allocate", "--basin", "example", "--claims-basis", "treaty")
    assert code == 0 and "treaty claims" in out and "total claims 27,500.0 (consumptive 12,440.5)" in out
    _, rows = table(out, "claimant")
    assert [float(r["claim"]) for r in rows] == pytest.approx([2500.0, 9000.0, 16000.0])
    assert [float(r["award"]) for r in rows] == pytest.approx([2500.0, 9000.0, 16000.0], abs=0.6)
    # flow factor 0.4: the 9 700 estate rations the 11 468 consumptive claims (Talmud)
    code, out, _ = run_cli(capsys, "allocate", "--basin", "example", "--flow-factor", "0.4", "--rule", "talmud")
    assert code == 0 and "estate 9,700.0" in out and "shortfall 1,768.1" in out
    _, rows = table(out, "claimant")
    assert [float(r["c-award"]) for r in rows] == pytest.approx([274.9, 2575.6, 6849.5], abs=0.6)
    assert [float(r["award"]) for r in rows] == pytest.approx([750.0, 5698.2, 14697.9], abs=0.6)
    assert "total awarded 21,146.2 (consumptive 9,700.0)" in out
    code, out, _ = run_cli(capsys, "allocate", "--basin", "example", "--flow-factor", "0.4", "--include-storage")
    assert code == 0 and "storage included" in out


def test_allocate_from_basin_json(capsys, tmp_path):
    path = tmp_path / "alloc_basin.json"
    code, out, _ = run_cli(capsys, "allocate", "--basin", "example", "--flow-factor", "0.4", "--rule", "all", "--json", str(path))
    assert code == 0 and f"wrote JSON: {path}" in out
    data = strict_json_load(path)
    assert data["claim_basis"] == "demand" and data["include_storage"] is False
    assert data["estate"] == pytest.approx(9700.0) and set(data["consumptive_claims"]) == set(RIPARIANS)
    row = data["rules"]["talmud"]
    assert set(row) >= {"claims", "consumptive_claims", "consumptive_awards", "consumption_ratio",
                        "total_consumptive_award_mm3", "awards", "withdrawals", "consumption"}
    assert row["total_consumptive_award_mm3"] == pytest.approx(9700.0)
    assert row["awards"]["Highland"] == pytest.approx(750.0)


def test_allocate_from_basin_file(capsys, basin_file):
    _, expected, _ = run_cli(capsys, "allocate", "--basin", "example")
    _, out, _ = run_cli(capsys, "allocate", "--basin", str(basin_file))
    assert out == expected


def test_allocate_json(capsys, tmp_path):
    path = tmp_path / "alloc.json"
    code, out, _ = run_cli(capsys, "allocate", "--estate", "200", "--claims", "a=100,b=200,c=300", "--json", str(path))
    assert code == 0 and f"wrote JSON: {path}" in out
    data = strict_json_load(path)
    assert data["estate"] == 200.0 and data["claims"] == {"a": 100.0, "b": 200.0, "c": 300.0}
    assert data["rules"]["talmud"]["awards"] == pytest.approx({"a": 50.0, "b": 75.0, "c": 75.0})


@pytest.mark.parametrize(
    "args",
    [
        ["allocate"],
        ["allocate", "--estate", "100"],
        ["allocate", "--claims", "a=1"],
        ["allocate", "--claims", "a=1", "--basin", "example", "--estate", "1"],
        ["allocate", "--estate", "-1", "--claims", "a=1"],
        ["allocate", "--estate", "x", "--claims", "a=1"],
        ["allocate", "--estate", "1", "--claims", "a=x"],
        ["allocate", "--estate", "1", "--claims", "a=1,a=2"],
        ["allocate", "--estate", "1", "--claims", "a=1,2"],
        ["allocate", "--estate", "1", "--claims", "=1"],
        ["allocate", "--estate", "1", "--claims", ","],
        ["allocate", "--estate", "1", "--claims", "a=-1"],
        ["allocate", "--estate", "1", "--claims", "a=inf"],
        ["allocate", "--estate", "1", "--claims", "a=1", "--rule", "treaty"],
        ["allocate", "--estate", "1", "--claims", "a=1", "--rule", "bogus"],
        ["allocate", "--estate", "1", "--claims", "a=1", "--claims-basis", "treaty"],
        ["allocate", "--estate", "1", "--claims", "a=1", "--include-storage"],
        ["allocate", "--basin", "example", "--claims-basis", "bogus"],
        ["allocate", "--basin", "example", "--flow-factor", "-1"],
        ["allocate", "--basin", "/no/such.json"],
    ],
)
def test_allocate_usage_errors(capsys, args):
    code, _, err = run_cli(capsys, *args)
    assert code == 2 and "error" in err


# ---------------------------------------------------------------------------
# negotiate
# ---------------------------------------------------------------------------
def test_negotiate_default(capsys):
    code, out, err = run_cli(capsys, "negotiate", "--basin", "example", "--flow-factor", "1.0", "--rule", "talmud")
    assert code == 0 and err == ""
    assert "Negotiation: Azura River (stylised)" in out and "rule talmud" in out
    assert "claims demand" in out and "storage excluded" in out
    # estate = natural flow minus the terminal in-stream requirement (1 500), not minus the 7 000 sum
    assert "natural flow 28,000.0" in out and "environmental flows 7,000.0" in out
    assert "reserve at the outlet 1,500.0" in out and "estate 26,500.0" in out
    assert "total claims 25,150.0 Mm3 (consumptive 11,468.1 Mm3)" in out
    header, rows = table(out, "riparian")
    assert header == ["riparian", "claim", "c-claim", "c-award", "proposal", "batna", "satisfaction", "acceptable", "routed"]
    assert [r["riparian"] for r in rows] == RIPARIANS
    assert [float(r["claim"]) for r in rows] == pytest.approx([1500.0, 7350.0, 16300.0])
    assert [float(r["c-claim"]) for r in rows] == pytest.approx([549.7, 3322.2, 7596.2], abs=0.6)
    for r in rows:
        assert r["acceptable"] == "yes"
        assert float(r["satisfaction"]) == pytest.approx(1.0)
        assert float(r["c-award"]) == pytest.approx(float(r["c-claim"]), abs=0.6)
        assert float(r["proposal"]) == pytest.approx(float(r["claim"]), abs=0.6)
        assert float(r["batna"]) == pytest.approx(float(r["claim"]), abs=0.6)
        assert float(r["routed"]) == pytest.approx(float(r["claim"]), abs=0.6)
    assert sum(float(r["proposal"]) for r in rows) == pytest.approx(25150.0, abs=1.5)
    assert "total awarded 25,150.0 Mm3 (consumptive 11,468.1 Mm3)" in out
    assert "ZOPA: yes" in out and "BATNA" in out


def test_negotiate_drought_year_and_rules(capsys):
    # flow factor 0.6: the 15 300 estate still covers the consumptive claims -> no rationing, ZOPA
    code, out, _ = run_cli(capsys, "negotiate", "--flow-factor", "0.6", "--rule", "cea")
    assert code == 0 and "flow factor 0.6" in out and "rule cea" in out and "estate 15,300.0" in out
    _, rows = table(out, "riparian")
    assert sum(float(r["c-award"]) for r in rows) == pytest.approx(11468.1, abs=1.5)
    assert sum(float(r["proposal"]) for r in rows) == pytest.approx(25150.0, abs=1.5)
    assert all(r["acceptable"] == "yes" for r in rows)
    assert "ZOPA: yes" in out


def test_negotiate_severe_drought_has_no_zopa(capsys):
    # flow factor 0.4: the 9 700 estate rations; the upstream riparians fall below their BATNA
    code, out, _ = run_cli(capsys, "negotiate", "--flow-factor", "0.4")
    assert code == 0 and "estate 9,700.0" in out
    _, rows = table(out, "riparian")
    by = {r["riparian"]: r for r in rows}
    assert float(by["Highland"]["proposal"]) == pytest.approx(750.0, abs=0.6)
    assert float(by["Highland"]["batna"]) == pytest.approx(1500.0, abs=0.6)
    assert float(by["Midland"]["proposal"]) == pytest.approx(5698.2, abs=0.6)
    assert float(by["Midland"]["batna"]) == pytest.approx(7350.0, abs=0.6)
    assert float(by["Delta"]["proposal"]) == pytest.approx(14697.9, abs=0.6)
    assert float(by["Delta"]["batna"]) == pytest.approx(7328.1, abs=0.6)
    assert [r["acceptable"] for r in rows] == ["no", "no", "yes"]
    assert float(by["Highland"]["satisfaction"]) == pytest.approx(0.5)
    assert sum(float(r["c-award"]) for r in rows) == pytest.approx(9700.0, abs=1.5)
    assert "total awarded 21,146.2 Mm3 (consumptive 9,700.0 Mm3)" in out
    assert "ZOPA: no - proposal below the BATNA of Highland, Midland" in out
    # with reservoir storage Delta's unilateral alternative improves; the upstream riparians still reject
    code, out2, _ = run_cli(capsys, "negotiate", "--flow-factor", "0.4", "--include-storage")
    assert code == 0 and "storage included" in out2
    _, rows2 = table(out2, "riparian")
    # Delta's supply with the reservoirs: 5 386.6 Mm3 of inflow (the upstream reservoirs refill 1 112.6 and
    # 828.9 of the surplus above their requirements) plus its 6 000 Mm3 of storage
    assert float(rows2[2]["batna"]) == pytest.approx(11386.6, abs=0.6)
    assert float(rows2[2]["batna"]) > float(by["Delta"]["batna"]) + 3000.0
    assert "ZOPA: no - proposal below the BATNA of Highland, Midland" in out2


def test_negotiate_treaty_claims(capsys):
    code, out, _ = run_cli(capsys, "negotiate", "--claims", "treaty")
    assert code == 0 and "claims treaty" in out and "total claims 27,500.0 Mm3 (consumptive 12,440.5 Mm3)" in out
    _, rows = table(out, "riparian")
    assert [float(r["claim"]) for r in rows] == pytest.approx([2500.0, 9000.0, 16000.0])
    assert [float(r["proposal"]) for r in rows] == pytest.approx([2500.0, 9000.0, 16000.0], abs=0.6)
    # Delta's entitlement is below the 16 300 it withdraws unilaterally
    assert [r["acceptable"] for r in rows] == ["yes", "yes", "no"]
    assert "ZOPA: no - proposal below the BATNA of Delta" in out


def test_negotiate_treaty_basis_falls_back_to_demand(capsys, tmp_path):
    """``--claims treaty`` on a basin without entitlements: claims = river demands >= every BATNA."""
    b = example_basin()
    for r in b.riparians:
        r.treaty_allocation_mm3 = None
    path = tmp_path / "no_treaty.json"
    IO.basin_to_json(b, path)
    code, out, _ = run_cli(capsys, "negotiate", "--basin", str(path), "--claims", "treaty", "--estate", "100000")
    assert code == 0
    assert "ZOPA: yes" in out and "estate 100,000.0" in out and "total claims 25,150.0" in out
    _, rows = table(out, "riparian")
    assert all(r["acceptable"] == "yes" for r in rows)
    assert all(float(r["satisfaction"]) == pytest.approx(1.0) for r in rows)


def test_negotiate_json(capsys, tmp_path):
    path = tmp_path / "neg.json"
    code, out, _ = run_cli(capsys, "negotiate", "--json", str(path))
    assert code == 0 and f"wrote JSON: {path}" in out
    data = strict_json_load(path)
    assert data["rule"] == "talmud" and data["estate"] == 26500.0 and data["claim_basis"] == "demand"
    assert data["include_storage"] is False and data["zopa"] is True
    assert set(data["proposal"]) == set(RIPARIANS) and isinstance(data["zopa"], bool)
    assert set(data) >= {"claims", "consumptive_claims", "consumptive_awards", "consumption_ratio",
                         "total_consumptive_award_mm3", "total_awarded_mm3", "batna", "acceptable", "gini",
                         "satisfaction", "routed_withdrawals", "routed_consumption"}
    assert data["total_consumptive_award_mm3"] == pytest.approx(11468.06, abs=0.01)
    assert data["total_awarded_mm3"] == pytest.approx(25150.0)


@pytest.mark.parametrize(
    "args",
    [
        ["negotiate", "--flow-factor", "-1"],
        ["negotiate", "--flow-factor", "nan"],
        ["negotiate", "--rule", "treaty"],
        ["negotiate", "--rule", "all"],
        ["negotiate", "--estate", "-5"],
        ["negotiate", "--claims", "bogus"],
        ["negotiate", "--claims", "a=1"],
        ["negotiate", "--basin", "/no/such.json"],
    ],
)
def test_negotiate_usage_errors(capsys, args):
    code, _, err = run_cli(capsys, *args)
    assert code == 2 and "error" in err


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------
def test_compare_selected_scenarios(capsys):
    code, out, err = run_cli(capsys, "compare", "--basin", "example", "--scenarios", "baseline,unilateral,cooperative", "--years", "3")
    assert code == 0 and err == ""
    assert "scenarios: baseline, unilateral, cooperative" in out and "years: 3" in out
    header, rows = table(out, "scenario")
    assert header == list(cli.COMPARE_DEFAULT_COLUMNS)
    assert len(rows) == 12
    assert [r["riparian"] for r in rows[:4]] == RIPARIANS + ["BASIN"]
    assert [r["scenario"] for r in rows[::4]] == ["baseline", "unilateral", "cooperative"]
    by = {(r["scenario"], r["riparian"]): r for r in rows}
    assert by[("baseline", "BASIN")]["allocation_rule"] == "treaty"
    assert by[("unilateral", "BASIN")]["allocation_rule"] == "upstream_priority"
    assert by[("cooperative", "BASIN")]["allocation_rule"] == "talmud"
    for r in rows:
        assert 0.0 <= float(r["mean_nexus_index"]) <= 1.0
        assert 0.0 <= float(r["mean_supply_ratio"]) <= 1.0
        assert r["equity_index"] == "-" if r["riparian"] != "BASIN" else 0.0 <= float(r["equity_index"]) <= 1.0


def test_compare_default_runs_every_scenario(capsys):
    code, out, _ = run_cli(capsys, "compare", "--years", "3")
    assert code == 0
    _, rows = table(out, "scenario")
    assert len(rows) == 4 * len(SCENARIO_NAMES)
    assert [r["scenario"] for r in rows[::4]] == SCENARIO_NAMES


def test_compare_columns_option(capsys):
    code, out, _ = run_cli(capsys, "compare", "--scenarios", "baseline", "--years", "3", "--columns", "scenario,riparian,mean_nexus_index,mean_nexus_index")
    assert code == 0
    header, rows = table(out, "scenario")
    assert header == ["scenario", "riparian", "mean_nexus_index"] and len(rows) == 4
    code, out, _ = run_cli(capsys, "compare", "--scenarios", "baseline", "--years", "3", "--columns", "all")
    assert code == 0
    header, _ = table(out, "scenario")
    assert header == list(TABLE_COLUMNS)


def test_compare_rule_override(capsys):
    code, out, _ = run_cli(capsys, "compare", "--scenarios", "baseline,unilateral", "--years", "3", "--rule", "cea")
    assert code == 0 and "rule override: cea" in out
    _, rows = table(out, "scenario")
    by = {(r["scenario"], r["riparian"]): r for r in rows}
    assert by[("baseline", "BASIN")]["allocation_rule"] == "cea"
    assert by[("unilateral", "BASIN")]["allocation_rule"] == "upstream_priority"  # non-cooperative


def test_compare_csv_and_json(capsys, tmp_path):
    csv_path, json_path = tmp_path / "cmp.csv", tmp_path / "cmp.json"
    code, out, _ = run_cli(capsys, "compare", "--scenarios", "baseline,growth", "--years", "3", "--csv", str(csv_path), "--json", str(json_path))
    assert code == 0 and f"wrote CSV: {csv_path}" in out and f"wrote JSON: {json_path}" in out
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        assert reader.fieldnames == list(TABLE_COLUMNS)
    assert len(rows) == 8
    data = strict_json_load(json_path)
    assert list(data["scenarios"]) == ["baseline", "growth"]
    assert data["scenarios"]["growth"]["years"] == 3
    assert len(data["rows"]) == 8 and set(data["summaries"]) == {"baseline", "growth"}


def test_compare_with_scenario_file(capsys, tmp_path):
    path = tmp_path / "mine.json"
    IO.scenario_to_json(Scenario(name="mine", years=2, cooperation=False), path)
    code, out, _ = run_cli(capsys, "compare", "--scenarios", f"baseline,{path}", "--years", "3")
    assert code == 0
    _, rows = table(out, "scenario")
    assert [r["scenario"] for r in rows[::4]] == ["baseline", "mine"]
    assert rows[-1]["allocation_rule"] == "upstream_priority"
    assert rows[-1]["scenario"] == "mine"


@pytest.mark.parametrize(
    "args",
    [
        ["compare", "--scenarios", "baseline,nope", "--years", "3"],
        ["compare", "--scenarios", "baseline,baseline", "--years", "3"],
        ["compare", "--scenarios", ",", "--years", "3"],
        ["compare", "--years", "0"],
        ["compare", "--scenarios", "baseline", "--years", "3", "--columns", "scenario,nope"],
        ["compare", "--scenarios", "baseline", "--years", "3", "--columns", ","],
        ["compare", "--scenarios", "baseline", "--years", "3", "--rule", "bogus"],
        ["compare", "--basin", "/no/such.json", "--years", "3"],
    ],
)
def test_compare_usage_errors(capsys, args):
    code, _, err = run_cli(capsys, *args)
    assert code == 2 and "error" in err


# ---------------------------------------------------------------------------
# pareto
# ---------------------------------------------------------------------------
def test_pareto_default_points(capsys):
    code, out, err = run_cli(capsys, "pareto", "--basin", "example", "--flow-factor", "1.0", "--points", "3")
    assert code == 0 and err == ""
    assert "Pareto front" in out and "3 of 3 points feasible" in out
    header, rows = table(out, "epsilon")
    assert header == [label for _, label in cli.PARETO_COLUMNS]
    assert len(rows) == 3
    eps = [float(r["epsilon"]) for r in rows]
    assert eps == sorted(eps) and eps[0] == 0.0
    benefits = [float(r["benefit_MUSD"]) for r in rows]
    assert all(b1 >= b2 - 1e-6 for b1, b2 in zip(benefits, benefits[1:]))  # non-increasing along the front
    for r in rows:
        assert 0.0 <= float(r["gini"]) <= 1.0
        assert 0.0 <= float(r["min_supply"]) <= 1.0 + 1e-9
        assert float(r["min_supply"]) >= float(r["epsilon"]) - 1e-6
    assert "price of fairness" in out


def test_pareto_drought_has_a_binding_trade_off(capsys):
    code, out, _ = run_cli(capsys, "pareto", "--flow-factor", "0.4", "--points", "5")
    assert code == 0
    _, rows = table(out, "epsilon")
    assert len(rows) >= 2
    benefits = [float(r["benefit_MUSD"]) for r in rows]
    mins = [float(r["min_supply"]) for r in rows]
    assert benefits[0] >= benefits[-1]
    assert mins[-1] >= mins[0]


def test_pareto_infeasible_returns_one(capsys):
    code, out, err = run_cli(capsys, "pareto", "--flow-factor", "0", "--points", "3")
    assert code == 1 and err == ""
    assert "no feasible allocation" in out and "0 of 3 points feasible" in out


def test_pareto_csv_and_json(capsys, tmp_path):
    csv_path, json_path = tmp_path / "front.csv", tmp_path / "front.json"
    code, out, _ = run_cli(capsys, "pareto", "--points", "3", "--csv", str(csv_path), "--json", str(json_path))
    assert code == 0 and f"wrote CSV: {csv_path}" in out
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3 and {"min_supply_ratio", "total_benefit_usd", "gini"} <= set(rows[0])
    data = strict_json_load(json_path)
    assert data["points"] == 3 and len(data["front"]) == 3
    assert data["front"][0]["total_benefit_usd"] == pytest.approx(float(rows[0]["total_benefit_usd"]))


@pytest.mark.parametrize(
    "args",
    [
        ["pareto", "--points", "1"],
        ["pareto", "--points", "0"],
        ["pareto", "--points", "many"],
        ["pareto", "--flow-factor", "-0.5"],
        ["pareto", "--basin", "/no/such.json"],
    ],
)
def test_pareto_usage_errors(capsys, args):
    code, _, err = run_cli(capsys, *args)
    assert code == 2 and "error" in err


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def test_report_baseline(capsys):
    code, out, err = run_cli(capsys, "report", "--basin", "example", "--scenario", "baseline", "--years", "3")
    assert code == 0 and err == ""
    assert "Sustainability assessment: Azura River (stylised) | scenario: baseline | 2025-2027 (3 years)" in out
    for name in RIPARIANS + ["BASIN"]:
        assert name in out
    assert "Conflict risk index" in out
    assert "allocation rule: treaty" in out and "agreement: Azura River (stylised) treaty" in out
    header, rows = table(out, "year")
    assert header == ["year", "score", "category"] + [label for _, label in cli.RISK_COLUMNS]
    assert [r["year"] for r in rows] == ["2025", "2026", "2027"]
    for r in rows:
        assert 0.0 <= float(r["score"]) <= 1.0
        assert r["category"] in ("low", "moderate", "high", "very_high")
        for _, label in cli.RISK_COLUMNS:
            assert 0.0 <= float(r[label]) <= 1.0
        assert float(r["institutional"]) == pytest.approx(0.5, abs=5e-4)  # bare treaty covering everybody
    assert "mean score" in out and "worst year" in out and "final year 2027" in out


def test_report_unilateral_has_no_agreement(capsys):
    code, out, _ = run_cli(capsys, "report", "--scenario", "unilateral", "--years", "3")
    assert code == 0
    assert "allocation rule: upstream_priority" in out and "agreement: none" in out
    _, rows = table(out, "year")
    assert all(float(r["institutional"]) == pytest.approx(1.0) for r in rows)


def test_report_sharing_rule_is_a_variable_agreement(capsys):
    code, out, _ = run_cli(capsys, "report", "--years", "3", "--rule", "talmud")
    assert code == 0
    assert "allocation rule: talmud" in out and "talmud sharing arrangement" in out
    _, rows = table(out, "year")
    assert all(float(r["institutional"]) < 0.5 for r in rows)


def test_report_json(capsys, tmp_path):
    path = tmp_path / "report.json"
    code, out, _ = run_cli(capsys, "report", "--years", "3", "--json", str(path))
    assert code == 0 and f"wrote JSON: {path}" in out
    data = strict_json_load(path)
    assert data["basin"] == "Azura River (stylised)" and data["scenario"]["years"] == 3
    assert set(data["sustainability"]["riparians"]) == set(RIPARIANS)
    assert "nexus_index" in data["sustainability"]["basin"]
    cr = data["conflict_risk"]
    assert cr["treaty"]["name"] == "Azura River (stylised) treaty"
    assert len(cr["years"]) == 3 and 0.0 <= cr["mean_score"] <= 1.0
    assert cr["final_year"]["category"] in ("low", "moderate", "high", "very_high")
    assert set(cr["final_year"]["components"]) == {k for k, _ in cli.RISK_COLUMNS}


def test_report_handles_unpopulated_riparian(capsys, tmp_path):
    # regression: a riparian without population (``inf`` per-capita water) used to
    # make ``report`` fail with "values['Midland'] must be finite" after the
    # simulation itself had succeeded
    b = example_basin()
    b.riparian("Midland").population = 0.0
    path = tmp_path / "unpopulated.json"
    IO.basin_to_json(b, path)
    code, out, err = run_cli(capsys, "report", "--basin", str(path), "--years", "2")
    assert code == 0 and err == ""
    assert "Sustainability assessment" in out and "Conflict risk index" in out
    for name in RIPARIANS + ["BASIN"]:
        assert name in out
    _, rows = table(out, "year")
    assert [r["year"] for r in rows] == ["2025", "2026"]


@pytest.mark.parametrize(
    "args",
    [
        ["report", "--years", "0"],
        ["report", "--scenario", "nope", "--years", "3"],
        ["report", "--rule", "bogus", "--years", "3"],
        ["report", "--basin", "/no/such.json", "--years", "3"],
    ],
)
def test_report_usage_errors(capsys, args):
    code, _, err = run_cli(capsys, *args)
    assert code == 2 and "error" in err


# ---------------------------------------------------------------------------
# export-basin
# ---------------------------------------------------------------------------
def test_export_basin_to_file(capsys, tmp_path):
    path = tmp_path / "exported.json"
    code, out, err = run_cli(capsys, "export-basin", "--out", str(path))
    assert code == 0 and err == ""
    assert "wrote basin 'Azura River (stylised)' (3 riparians)" in out and str(path) in out
    assert IO.load_basin(path) == example_basin()
    data = strict_json_load(path)
    assert data["format"] == IO.BASIN_FORMAT


def test_export_basin_to_stdout(capsys):
    code, out, err = run_cli(capsys, "export-basin")
    assert code == 0 and err == ""
    assert IO.basin_from_dict(json.loads(out)) == example_basin()
    code, out, _ = run_cli(capsys, "export-basin", "--indent", "-1")
    assert code == 0
    assert out.count("\n") == 1 and IO.basin_from_dict(json.loads(out)) == example_basin()


def test_export_basin_reexports_a_file(capsys, basin_file, tmp_path):
    out_path = tmp_path / "again.json"
    code, _, _ = run_cli(capsys, "export-basin", "--basin", str(basin_file), "--out", str(out_path))
    assert code == 0
    assert out_path.read_text(encoding="utf-8") == basin_file.read_text(encoding="utf-8")


def test_export_basin_errors(capsys, tmp_path):
    code, _, err = run_cli(capsys, "export-basin", "--out", str(tmp_path / "nodir" / "b.json"))
    assert code == 2 and "error" in err
    code, _, err = run_cli(capsys, "export-basin", "--basin", "/no/such.json")
    assert code == 2 and "not found" in err


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def test_parse_claims():
    assert cli.parse_claims("a=100, b=200,c=300") == {"a": 100.0, "b": 200.0, "c": 300.0}
    assert cli.parse_claims("10,20,,30") == {"c1": 10.0, "c2": 20.0, "c3": 30.0}
    assert cli.parse_claims("x = 5") == {"x": 5.0}
    for bad in ["", ",", "a=x", "a=1,2", "a=1,a=2", "=1", "a=-1", "a=nan", "a=inf", None]:
        with pytest.raises(ValueError):
            cli.parse_claims(bad)


def test_parse_list():
    assert cli.parse_list(" a, b ,,c ") == ["a", "b", "c"]
    for bad in ["", " , ", None]:
        with pytest.raises(ValueError):
            cli.parse_list(bad)


def test_canonical_rules():
    assert cli.canonical_model_rule("Treaty") == "treaty"
    assert cli.canonical_model_rule("contested-garment") == "talmud"
    assert cli.canonical_model_rule("upstream priority") == "upstream_priority"
    assert all(cli.canonical_model_rule(r) == r for r in ALLOCATION_RULES)
    assert cli.canonical_claims_rule("ALL") == "all"
    assert cli.canonical_claims_rule("maimonides") == "cea"
    for bad in ["", "treaty", "bogus", None, 3]:
        with pytest.raises(ValueError):
            cli.canonical_claims_rule(bad)
    for bad in ["", "bogus", "all", None]:
        with pytest.raises(ValueError):
            cli.canonical_model_rule(bad)


def test_load_basin_arg(basin_file):
    a, b = cli.load_basin_arg("example"), cli.load_basin_arg("EXAMPLE")
    assert a == b == example_basin() and a is not b
    assert cli.load_basin_arg(str(basin_file)) == example_basin()
    for bad in ["", "   ", "/no/such.json", None]:
        with pytest.raises(ValueError):
            cli.load_basin_arg(bad)


def test_resolve_scenario(tmp_path):
    assert cli.resolve_scenario("baseline").years == 25
    assert cli.resolve_scenario("Climate-Change", 3) == S.climate_change(years=3)
    path = tmp_path / "s.json"
    IO.scenario_to_json(Scenario(name="f", years=7, seed=3), path)
    assert cli.resolve_scenario(str(path)) == Scenario(name="f", years=7, seed=3)
    assert cli.resolve_scenario(str(path), 2) == Scenario(name="f", years=2, seed=3)
    for spec, years in [("nope", None), ("", None), ("baseline", 0), ("baseline", True), (None, None)]:
        with pytest.raises(ValueError):
            cli.resolve_scenario(spec, years)


def test_summary_rows(result3):
    rows = cli.summary_rows(result3)
    assert [r["riparian"] for r in rows] == RIPARIANS + ["BASIN"]
    assert set(rows[0]) == {"riparian"} | {label for _, label in cli.RUN_COLUMNS}
    summary = result3.summary()
    assert rows[1]["nexus"] == summary["Midland"]["nexus_index"]
    assert rows[-1]["storage_Mm3"] == summary["BASIN"]["final_storage_mm3"]
    with pytest.raises(ValueError):
        cli.summary_rows("not a result")


def test_treaty_for_result():
    b = example_basin()
    uni = run_nexus(b, Scenario(years=2, cooperation=False))
    assert cli.treaty_for_result(b, uni) is None
    fixed = run_nexus(b, Scenario(years=2, allocation_rule="treaty"))
    t = cli.treaty_for_result(b, fixed)
    assert isinstance(t, Treaty) and t.parties == RIPARIANS
    assert t.allocations_mm3 == {"Highland": 2500.0, "Midland": 9000.0, "Delta": 16000.0}
    assert not t.variable_allocation
    shared = run_nexus(b, Scenario(years=2, allocation_rule="talmud"))
    t = cli.treaty_for_result(b, shared)
    assert t.variable_allocation and t.drought_provisions and "talmud" in t.name
    # basin without entitlements
    nb = example_basin()
    for r in nb.riparians:
        r.treaty_allocation_mm3 = None
    assert cli.treaty_for_result(nb, run_nexus(nb, Scenario(years=2, allocation_rule="treaty"))) is None
    t = cli.treaty_for_result(nb, run_nexus(nb, Scenario(years=2, allocation_rule="cea")))
    assert t.parties == RIPARIANS and t.allocations_mm3 == {} and t.variable_allocation
    with pytest.raises(ValueError):
        cli.treaty_for_result("basin", fixed)
    with pytest.raises(ValueError):
        cli.treaty_for_result(b, "result")


def test_conflict_risk_rows(result3):
    b = example_basin()
    rows = cli.conflict_risk_rows(b, result3)
    assert [r["year"] for r in rows] == [2025, 2026, 2027]
    for r in rows:
        assert 0.0 <= r["score"] <= 1.0
        assert all(0.0 <= r[label] <= 1.0 for _, label in cli.RISK_COLUMNS)
    none = cli.conflict_risk_rows(b, result3, treaty=None)
    assert none[0]["institutional"] == pytest.approx(0.5)  # falls back to treaty_for_result
    empty = dataclasses.replace(result3, balances=[])
    with pytest.raises(ValueError, match="balances"):
        cli.conflict_risk_rows(b, empty)
    with pytest.raises(ValueError):
        cli.conflict_risk_rows(b, "result")


def test_helpers_do_not_mutate(result3):
    b = example_basin()
    # ``asdict`` deep-copies the nested sector dictionaries (``value_usd_per_m3``
    # read by ``json_safe`` / the risk rows), so the guard cannot be tautological
    before = dataclasses.asdict(b)
    records_before = [dataclasses.replace(r) for r in result3.records]
    cli.summary_rows(result3)
    cli.treaty_for_result(b, result3)
    cli.conflict_risk_rows(b, result3)
    cli.json_safe({"r": result3.summary(), "s": result3.scenario})
    assert dataclasses.asdict(b) == before
    assert b.riparians[2].demand.value_usd_per_m3[Sector.MUNICIPAL] == 1.5
    assert b.riparians[2].demand.consumption_fraction[Sector.AGRICULTURAL] == 0.6
    assert result3.records == records_before
    assert result3.scenario == Scenario(name="baseline", years=3)


def test_json_safe():
    assert cli.json_safe({Sector.MUNICIPAL: math.inf, "n": math.nan, "x": (1, 2.5), "b": True, 3: {"s": {1, }}}) == {
        "municipal": None,
        "n": None,
        "x": [1, 2.5],
        "b": True,
        "3": {"s": [1]},
    }
    assert cli.json_safe(np.float64(1.5)) == 1.5 and isinstance(cli.json_safe(np.int64(2)), int)
    assert cli.json_safe(np.array([1.0, math.inf])) == [1.0, None]
    assert cli.json_safe(Sector.ENERGY) == "energy"
    assert cli.json_safe(Scenario(name="z"))["format"] == IO.SCENARIO_FORMAT
    t = Treaty("t", ["a"], {"a": 1.0})
    assert cli.json_safe(t)["parties"] == ["a"]
    assert cli.json_safe(object()).startswith("<object")
    assert cli.json_safe(None) is None and cli.json_safe("s") == "s"
    json.dumps(cli.json_safe({"inf": math.inf}), allow_nan=False)


def test_write_json(tmp_path):
    path = tmp_path / "w.json"
    assert cli.write_json(path, {"a": math.inf, "b": [Sector.ENERGY]}) == str(path)
    assert strict_json_load(path) == {"a": None, "b": ["energy"]}
    with pytest.raises(OSError):
        cli.write_json(tmp_path / "nodir" / "w.json", {})


def test_build_parser_defaults():
    parser = cli.build_parser()
    ns = parser.parse_args(["run"])
    assert (ns.basin, ns.scenario, ns.years, ns.rule, ns.refill, ns.csv, ns.json, ns.plot) == (
        "example", "baseline", None, None, 0.25, None, None, None,
    )
    assert parser.parse_args(["run", "--plot", "figs"]).plot == "figs"
    ns = parser.parse_args(["pareto"])
    assert ns.points == 11 and ns.flow_factor == 1.0
    ns = parser.parse_args(["allocate", "--claims", "a=1", "--estate", "2", "--rule", "CEL"])
    assert ns.claims == {"a": 1.0} and ns.estate == 2.0 and ns.rule == "cel"
    with pytest.raises(cli.CliError) as exc:
        parser.parse_args(["run", "--years", "0"])
    assert "positive integer" in exc.value.message and "usage" in exc.value.usage


def test_cli_error_attributes():
    e = cli.CliError("boom", "usage: x\n", "x")
    assert str(e) == "boom" and e.usage == "usage: x\n" and e.prog == "x"
    assert cli.CliError("m").prog == cli.PROG
