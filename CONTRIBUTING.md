# Contributing to wefnexus

Thank you for helping improve the toolkit. This is a short guide; the module
contract lives in `docs/ARCHITECTURE.md` and the formulas in
`docs/METHODOLOGY.md`.

## Set up

```bash
git clone https://github.com/YousefWEF/YousefWEF.git
cd YousefWEF
pip install -e ".[dev]"        # numpy, scipy, pandas, matplotlib, networkx, pytest
```

Python 3.9 or newer is required (use `typing.Optional` / `Dict` / `List`;
`from __future__ import annotations` is fine).

## Run the tests

```bash
python -m pytest                       # whole suite (several minutes)
python -m pytest tests/test_water.py -q   # one module
python -m pytest -q -k talmud          # by keyword
WEFNEXUS_FAST=1 python examples/run_example.py --no-plots   # 3-year end-to-end smoke run
```

There is one test file per module (`tests/test_<module>.py`) and the shared
fixtures `basin` (a fresh Azura River) and `baseline` (a 5-year scenario) in
`tests/conftest.py`. The suite is deterministic: stochastic runs are seeded
through `Scenario.seed`.

Every change must keep the suite green. New or changed behaviour needs tests
that cover

- property checks: bounds (indices in `[0, 1]`, awards in `[0, claim]`),
  conservation (water mass balance below 1e-6 Mm3, awards summing to
  `min(estate, claims)`), monotonicity (more water never lowers a supply ratio);
- closed-form literature examples (e.g. the Talmud's claims 100/200/300, 1 Mm3
  through 100 m = 0.2725 GWh, FAO-33 yields);
- edge cases: zero flow, zero demand, missing reservoir or crops, `inf`
  entitlements, empty event lists;
- no-mutation checks: the `Basin`, `Riparian` and `Scenario` passed in are
  unchanged afterwards (compare against a `basin.copy()`).

## Style

- Plain functions over plain data: inputs are the dataclasses of
  `wefnexus.models`, floats, lists and dicts; results are floats, dicts or
  small result dataclasses. Never mutate an input; use `basin.copy()` or
  `dataclasses.replace`.
- Validate inputs and raise `ValueError` with a message that names the
  offending argument and the accepted range.
- NumPy-style docstrings on every public function: parameters with units,
  returns with units, the formula, and at least one literature reference.
- Units are fixed package-wide: water in Mm3/yr, energy in GWh/yr (say so when
  a helper returns kWh), area in ha, depth in mm, food in t and kcal, money in
  USD. 1 mm over 1 ha = 10 m3.
- Dependencies: NumPy in the core; SciPy only in `optimize` and the
  cooperative-game solvers of `allocation`; pandas, matplotlib and networkx
  only through lazy imports inside functions of `viz`, `io` and result helpers.
- Import direction: `models` <- leaf modules (`water`, `energy`, `food`,
  `allocation`) <- composite modules (`diplomacy`, `sustainability`, `nexus`,
  `optimize`, `scenarios`) <- integration (`viz`, `io`, `cli`). No module may
  import `nexus` at import time except `scenarios`, `cli`, `viz` and
  `optimize`; `sustainability.assess` imports it lazily.
- Keep public signatures listed in `docs/ARCHITECTURE.md` stable; extend with
  keyword-only arguments that have defaults.
- Formatting: 4-space indentation, lines up to ~110 characters, `snake_case`
  names, module-level constants in `UPPER_CASE` with a `#:` comment. No
  emojis, no colour codes in CLI output.
- Avoid committing generated files (`examples/output/`, `*.png`, `*.egg-info`
  are git-ignored).

## Submitting a change

1. Open an issue or describe the change in the pull request: which indicator
   or method, which reference, which module.
2. Add or update tests and docstrings; update `docs/METHODOLOGY.md` when a
   formula, default weight or threshold changes, and `README.md` when the CLI
   or the public API changes (README snippets are meant to run as written).
3. Run `python -m pytest` and make sure it passes.
