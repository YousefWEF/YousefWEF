"""Tests for :mod:`wefnexus.food` (water-food links).

Covers closed-form FAO-56 / FAO-33 / virtual-water literature values, the
example-basin numbers quoted in the architecture spec, physical bounds and
conservation, monotonicity, edge cases (zero water / zero demand / no crops /
rainfed crops / non-food crops), input validation and no-mutation guarantees.
"""
import copy
import math
import pathlib
from dataclasses import replace

import numpy as np
import pytest

from wefnexus import food
from wefnexus.food import (
    DAYS_PER_YEAR,
    DEFAULT_FARM_ENERGY_KWH_PER_HA,
    DEFAULT_KCAL_PER_KG,
    DEFAULT_VIRTUAL_WATER_M3_PER_T,
    KG_PER_T,
    KWH_PER_GWH,
    M3_PER_HA_MM,
    M3_PER_MM3,
    crop_evapotranspiration_mm,
    crop_production,
    energy_for_agriculture_gwh,
    fao33_relative_yield,
    fao33_yield,
    food_demand_kcal,
    food_security_index,
    food_self_sufficiency,
    gross_irrigation_mm,
    irrigation_requirement_mm3,
    net_irrigation_mm,
    rainfall_et_share,
    relative_evapotranspiration,
    riparian_crop_requirements_mm3,
    riparian_food_production,
    riparian_irrigation_requirement_mm3,
    virtual_water_content_m3_per_t,
    virtual_water_import_mm3,
    water_productivity_kg_per_m3,
)
from wefnexus.models import Crop, Riparian, WaterDemand

BAD_NUMBERS = [-1.0, -1e-9, float("nan"), float("inf"), -float("inf"), "abc", None, True]

# Spec keys that every crop_production dict must carry.
CROP_KEYS = {"area_ha", "et_ratio", "yield_t_ha", "production_t", "kcal", "value_usd"}
# Spec keys that every riparian_food_production dict must carry.
RIPARIAN_KEYS = {
    "crops", "production_t", "kcal", "value_usd", "requirement_mm3", "supplied_mm3", "et_ratio",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _wheat(area_ha=150_000.0) -> Crop:
    """Highland wheat from the example basin."""
    return Crop("wheat", area_ha=area_ha, kc=0.85, season_days=150,
                yield_max_t_ha=4.0, ky=1.05, kcal_per_kg=3_400, price_usd_t=250)


def _maize(area_ha=50_000.0) -> Crop:
    """Highland maize from the example basin."""
    return Crop("maize", area_ha=area_ha, kc=0.90, season_days=130,
                yield_max_t_ha=6.0, ky=1.25, kcal_per_kg=3_600, price_usd_t=200)


def _cotton(area_ha=100_000.0) -> Crop:
    """Midland cotton: a non-food crop (kcal_per_kg 0) with ky < 1."""
    return Crop("cotton", area_ha=area_ha, kc=0.90, season_days=180,
                yield_max_t_ha=2.5, ky=0.85, kcal_per_kg=0.0, price_usd_t=1_600)


def _riparian(crops, et0=4.0, pe=300.0, eff=0.5, population=8_000_000, **kw) -> Riparian:
    return Riparian(
        name="R",
        population=population,
        gdp_usd=1e9,
        local_inflow_mm3=1000.0,
        demand=WaterDemand(agricultural=800.0),
        crops=list(crops),
        irrigation_efficiency=eff,
        et0_mm_day=et0,
        effective_rainfall_mm=pe,
        **kw,
    )


def _expected_requirement(crop: Crop, et0: float, pe: float, eff: float) -> float:
    """Independent re-implementation of the spec formula for cross-checks."""
    gross_mm = max(et0 * crop.kc * crop.season_days - pe, 0.0) / eff
    return gross_mm * crop.area_ha * 10.0 / 1e6


# Example-basin closed forms quoted in the spec / computed by hand.
HIGHLAND_WHEAT_MM3 = (4 * 0.85 * 150 - 300) / 0.5 * 150_000 * 10 / 1e6   # 630
HIGHLAND_MAIZE_MM3 = (4 * 0.9 * 130 - 300) / 0.5 * 50_000 * 10 / 1e6     # 168
HIGHLAND_TOTAL_MM3 = HIGHLAND_WHEAT_MM3 + HIGHLAND_MAIZE_MM3             # 798


# --------------------------------------------------------------------------- #
# Constants and module surface
# --------------------------------------------------------------------------- #
def test_constants_and_all():
    assert M3_PER_HA_MM == 10.0           # 1 mm over 1 ha = 10 m3
    assert M3_PER_MM3 == 1e6
    assert KG_PER_T == 1000.0
    assert KWH_PER_GWH == 1e6
    assert DAYS_PER_YEAR == 365.0
    assert DEFAULT_KCAL_PER_KG == 3400.0
    assert DEFAULT_VIRTUAL_WATER_M3_PER_T == 1500.0
    assert DEFAULT_FARM_ENERGY_KWH_PER_HA == 150.0
    for name in food.__all__:
        assert hasattr(food, name), name
    # every public function has a docstring mentioning a unit
    for name in food.__all__:
        obj = getattr(food, name)
        if callable(obj):
            assert obj.__doc__ and len(obj.__doc__) > 50, name


def test_module_is_a_leaf_without_heavy_imports():
    """food.py is a leaf: no SciPy/plotting/pandas and no sibling wefnexus modules."""
    src = pathlib.Path(food.__file__).read_text(encoding="utf-8")
    for forbidden in ("import scipy", "import matplotlib", "import pandas", "import networkx",
                      "from wefnexus.energy", "from wefnexus.water", "from wefnexus.nexus",
                      "import numpy"):
        assert forbidden not in src, forbidden


# --------------------------------------------------------------------------- #
# Crop evapotranspiration (FAO-56)
# --------------------------------------------------------------------------- #
def test_etc_closed_form_highland_wheat():
    """ETc = ET0 * Kc * days: 4 mm/day * 0.85 * 150 d = 510 mm."""
    assert crop_evapotranspiration_mm(4.0, 0.85, 150) == pytest.approx(510.0, rel=1e-12)
    assert crop_evapotranspiration_mm(4.0, 0.90, 130) == pytest.approx(468.0, rel=1e-12)
    # Delta rice: 6.5 * 1.1 * 150 = 1072.5 mm
    assert crop_evapotranspiration_mm(6.5, 1.1, 150) == pytest.approx(1072.5, rel=1e-12)


def test_etc_is_trilinear_and_zero_cases():
    base = crop_evapotranspiration_mm(5.0, 1.0, 100)
    assert base == pytest.approx(500.0)
    assert crop_evapotranspiration_mm(10.0, 1.0, 100) == pytest.approx(2 * base)
    assert crop_evapotranspiration_mm(5.0, 0.5, 100) == pytest.approx(0.5 * base)
    assert crop_evapotranspiration_mm(5.0, 1.0, 200) == pytest.approx(2 * base)
    assert crop_evapotranspiration_mm(0.0, 1.0, 100) == 0.0
    assert crop_evapotranspiration_mm(5.0, 0.0, 100) == 0.0
    assert crop_evapotranspiration_mm(5.0, 1.0, 0) == 0.0


def test_etc_monotone_in_each_argument():
    grid = np.linspace(0.0, 10.0, 25)
    out = [crop_evapotranspiration_mm(float(x), 0.9, 120) for x in grid]
    assert all(b >= a for a, b in zip(out, out[1:]))
    out = [crop_evapotranspiration_mm(5.0, float(x) / 10, 120) for x in grid]
    assert all(b >= a for a, b in zip(out, out[1:]))
    out = [crop_evapotranspiration_mm(5.0, 0.9, float(x) * 30) for x in grid]
    assert all(b >= a for a, b in zip(out, out[1:]))


def test_etc_accepts_numpy_and_int_scalars():
    out = crop_evapotranspiration_mm(np.float64(4.0), np.float32(0.85), np.int64(150))
    assert out == pytest.approx(510.0, rel=1e-6)
    assert isinstance(crop_evapotranspiration_mm(4, 1, 150), float)


@pytest.mark.parametrize("bad", BAD_NUMBERS)
def test_etc_rejects_bad_inputs(bad):
    with pytest.raises(ValueError):
        crop_evapotranspiration_mm(bad, 0.85, 150)
    with pytest.raises(ValueError):
        crop_evapotranspiration_mm(4.0, bad, 150)
    with pytest.raises(ValueError):
        crop_evapotranspiration_mm(4.0, 0.85, bad)


def test_etc_error_messages_name_the_argument():
    with pytest.raises(ValueError, match="et0_mm_day"):
        crop_evapotranspiration_mm(-1.0, 0.85, 150)
    with pytest.raises(ValueError, match="kc"):
        crop_evapotranspiration_mm(4.0, -0.85, 150)
    with pytest.raises(ValueError, match="season_days"):
        crop_evapotranspiration_mm(4.0, 0.85, -150)


# --------------------------------------------------------------------------- #
# Net and gross irrigation requirement
# --------------------------------------------------------------------------- #
def test_net_irrigation_closed_form():
    assert net_irrigation_mm(510.0, 300.0) == pytest.approx(210.0, rel=1e-12)
    assert net_irrigation_mm(468.0, 300.0) == pytest.approx(168.0, rel=1e-12)
    # rainfall exceeds crop ET -> rainfed, no irrigation needed
    assert net_irrigation_mm(200.0, 300.0) == 0.0
    assert net_irrigation_mm(300.0, 300.0) == 0.0
    assert net_irrigation_mm(0.0, 0.0) == 0.0
    assert net_irrigation_mm(510.0, 0.0) == pytest.approx(510.0)


def test_net_irrigation_bounds_and_monotonicity():
    rng = np.random.default_rng(10)
    for _ in range(200):
        etc = float(rng.uniform(0, 2000))
        pe = float(rng.uniform(0, 2000))
        net = net_irrigation_mm(etc, pe)
        assert 0.0 <= net <= etc
        assert net == pytest.approx(max(etc - pe, 0.0))
    pes = np.linspace(0, 1000, 50)
    out = [net_irrigation_mm(600.0, float(p)) for p in pes]
    assert all(b <= a for a, b in zip(out, out[1:]))  # non-increasing in rainfall
    etcs = np.linspace(0, 1000, 50)
    out = [net_irrigation_mm(float(e), 300.0) for e in etcs]
    assert all(b >= a for a, b in zip(out, out[1:]))  # non-decreasing in ETc


@pytest.mark.parametrize("bad", BAD_NUMBERS)
def test_net_irrigation_rejects_bad_inputs(bad):
    with pytest.raises(ValueError):
        net_irrigation_mm(bad, 100.0)
    with pytest.raises(ValueError):
        net_irrigation_mm(100.0, bad)


def test_gross_irrigation_closed_form():
    assert gross_irrigation_mm(210.0, 0.5) == pytest.approx(420.0, rel=1e-12)
    assert gross_irrigation_mm(168.0, 0.5) == pytest.approx(336.0, rel=1e-12)
    assert gross_irrigation_mm(210.0, 1.0) == pytest.approx(210.0)   # perfect efficiency
    assert gross_irrigation_mm(0.0, 0.3) == 0.0
    # drip (0.9) vs surface (0.45): half the gross water
    assert gross_irrigation_mm(100.0, 0.9) == pytest.approx(gross_irrigation_mm(100.0, 0.45) / 2)


def test_gross_irrigation_monotone_decreasing_in_efficiency_and_at_least_net():
    etas = np.linspace(0.05, 1.0, 40)
    out = [gross_irrigation_mm(100.0, float(e)) for e in etas]
    assert all(b <= a for a, b in zip(out, out[1:]))
    assert all(g >= 100.0 - 1e-12 for g in out)


@pytest.mark.parametrize("eta", [0.0, -0.1, 1.0001, 2.0, float("nan"), float("inf"), "x", None, True])
def test_gross_irrigation_rejects_bad_efficiency(eta):
    with pytest.raises(ValueError, match="efficiency"):
        gross_irrigation_mm(100.0, eta)


@pytest.mark.parametrize("bad", BAD_NUMBERS)
def test_gross_irrigation_rejects_bad_net(bad):
    with pytest.raises(ValueError):
        gross_irrigation_mm(bad, 0.5)


# --------------------------------------------------------------------------- #
# Crop irrigation requirement (volume)
# --------------------------------------------------------------------------- #
def test_irrigation_requirement_spec_examples():
    """Spec: Highland wheat 630 Mm3 and maize 168 Mm3 at ET0 4, Pe 300, eff 0.5."""
    assert HIGHLAND_WHEAT_MM3 == pytest.approx(630.0, rel=1e-12)
    assert HIGHLAND_MAIZE_MM3 == pytest.approx(168.0, rel=1e-12)
    assert irrigation_requirement_mm3(_wheat(), 4.0, 300.0, 0.5) == pytest.approx(630.0, rel=1e-12)
    assert irrigation_requirement_mm3(_maize(), 4.0, 300.0, 0.5) == pytest.approx(168.0, rel=1e-12)


def test_irrigation_requirement_chain_equals_component_functions():
    crop = _wheat()
    etc = crop_evapotranspiration_mm(4.0, crop.kc, crop.season_days)
    net = net_irrigation_mm(etc, 300.0)
    gross = gross_irrigation_mm(net, 0.5)
    assert irrigation_requirement_mm3(crop, 4.0, 300.0, 0.5) == pytest.approx(
        gross * crop.area_ha * M3_PER_HA_MM / M3_PER_MM3, rel=1e-12
    )


def test_irrigation_requirement_linear_in_area_and_zero_cases():
    base = irrigation_requirement_mm3(_wheat(), 4.0, 300.0, 0.5)
    assert irrigation_requirement_mm3(_wheat(300_000), 4.0, 300.0, 0.5) == pytest.approx(2 * base)
    assert irrigation_requirement_mm3(_wheat(0.0), 4.0, 300.0, 0.5) == 0.0
    # rainfall covers the whole ET: rainfed crop
    assert irrigation_requirement_mm3(_wheat(), 4.0, 510.0, 0.5) == 0.0
    assert irrigation_requirement_mm3(_wheat(), 4.0, 5000.0, 0.5) == 0.0
    # no evaporative demand at all
    assert irrigation_requirement_mm3(_wheat(), 0.0, 0.0, 0.5) == 0.0
    # zero-length season
    crop = Crop("x", 1000.0, 1.0, 0, 1.0, 1.0, 1000.0)
    assert irrigation_requirement_mm3(crop, 5.0, 0.0, 0.5) == 0.0


def test_irrigation_requirement_random_cross_check():
    rng = np.random.default_rng(11)
    for _ in range(100):
        crop = Crop("c", float(rng.uniform(0, 5e5)), float(rng.uniform(0.3, 1.3)),
                    int(rng.integers(30, 365)), 5.0, 1.0, 3000.0)
        et0 = float(rng.uniform(0, 9))
        pe = float(rng.uniform(0, 800))
        eff = float(rng.uniform(0.3, 1.0))
        got = irrigation_requirement_mm3(crop, et0, pe, eff)
        assert got >= 0.0
        assert got == pytest.approx(_expected_requirement(crop, et0, pe, eff), rel=1e-12, abs=1e-12)


def test_irrigation_requirement_monotone_in_et0_and_decreasing_in_pe_and_eff():
    et0s = np.linspace(0, 10, 30)
    out = [irrigation_requirement_mm3(_wheat(), float(e), 300.0, 0.5) for e in et0s]
    assert all(b >= a for a, b in zip(out, out[1:]))
    pes = np.linspace(0, 800, 30)
    out = [irrigation_requirement_mm3(_wheat(), 4.0, float(p), 0.5) for p in pes]
    assert all(b <= a for a, b in zip(out, out[1:]))
    effs = np.linspace(0.2, 1.0, 30)
    out = [irrigation_requirement_mm3(_wheat(), 4.0, 300.0, float(f)) for f in effs]
    assert all(b <= a for a, b in zip(out, out[1:]))


def test_irrigation_requirement_does_not_mutate_crop():
    crop = _wheat()
    before = copy.deepcopy(crop)
    irrigation_requirement_mm3(crop, 4.0, 300.0, 0.5)
    assert crop == before


@pytest.mark.parametrize("not_a_crop", [None, 3.0, "wheat", {"area_ha": 1.0}, (1, 2)])
def test_irrigation_requirement_rejects_non_crop(not_a_crop):
    with pytest.raises(ValueError, match="Crop"):
        irrigation_requirement_mm3(not_a_crop, 4.0, 300.0, 0.5)


def test_irrigation_requirement_rejects_bad_crop_attributes():
    with pytest.raises(ValueError, match="area_ha"):
        irrigation_requirement_mm3(_wheat(-1.0), 4.0, 300.0, 0.5)
    bad_kc = Crop("w", 1.0, -0.5, 100, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="kc"):
        irrigation_requirement_mm3(bad_kc, 4.0, 300.0, 0.5)
    bad_days = Crop("w", 1.0, 0.5, -100, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="season_days"):
        irrigation_requirement_mm3(bad_days, 4.0, 300.0, 0.5)


@pytest.mark.parametrize("bad", BAD_NUMBERS)
def test_irrigation_requirement_rejects_bad_climate(bad):
    with pytest.raises(ValueError):
        irrigation_requirement_mm3(_wheat(), bad, 300.0, 0.5)
    with pytest.raises(ValueError):
        irrigation_requirement_mm3(_wheat(), 4.0, bad, 0.5)


@pytest.mark.parametrize("eta", [0.0, -0.5, 1.5, float("nan"), None])
def test_irrigation_requirement_rejects_bad_efficiency(eta):
    with pytest.raises(ValueError, match="efficiency"):
        irrigation_requirement_mm3(_wheat(), 4.0, 300.0, eta)


# --------------------------------------------------------------------------- #
# Riparian irrigation requirement
# --------------------------------------------------------------------------- #
def test_riparian_requirement_highland_spec_value(basin):
    """Spec: Highland crops require about 798 Mm3 (wheat 630 + maize 168)."""
    highland = basin.riparian("Highland")
    assert HIGHLAND_TOTAL_MM3 == pytest.approx(798.0, rel=1e-12)
    assert riparian_irrigation_requirement_mm3(highland) == pytest.approx(798.0, rel=1e-9)
    per_crop = riparian_crop_requirements_mm3(highland)
    assert list(per_crop) == ["wheat", "maize"]
    assert per_crop["wheat"] == pytest.approx(630.0, rel=1e-9)
    assert per_crop["maize"] == pytest.approx(168.0, rel=1e-9)
    assert sum(per_crop.values()) == pytest.approx(riparian_irrigation_requirement_mm3(highland))
    # the stylised agricultural demand (800) is deliberately close to the crop requirement
    assert abs(highland.demand.agricultural - 798.0) < 5.0


def test_riparian_requirement_midland_and_delta_closed_forms(basin):
    midland = basin.riparian("Midland")   # ET0 5.5, Pe 150, eff 0.5
    expected_mid = {
        "wheat": (5.5 * 0.85 * 150 - 150) / 0.5 * 300_000 * 10 / 1e6,       # 3307.5
        "cotton": (5.5 * 0.90 * 180 - 150) / 0.5 * 100_000 * 10 / 1e6,      # 1482
        "vegetables": (5.5 * 0.95 * 120 - 150) / 0.5 * 80_000 * 10 / 1e6,   # 763.2
    }
    assert expected_mid["wheat"] == pytest.approx(3307.5)
    assert expected_mid["cotton"] == pytest.approx(1482.0)
    assert expected_mid["vegetables"] == pytest.approx(763.2)
    got = riparian_crop_requirements_mm3(midland)
    assert got.keys() == expected_mid.keys()
    for k, v in expected_mid.items():
        assert got[k] == pytest.approx(v, rel=1e-9)
    assert riparian_irrigation_requirement_mm3(midland) == pytest.approx(5552.7, rel=1e-9)

    delta = basin.riparian("Delta")       # ET0 6.5, Pe 20, eff 0.55
    expected_delta = sum(
        _expected_requirement(c, 6.5, 20.0, 0.55) for c in delta.crops
    )
    assert expected_delta == pytest.approx(13039.5, abs=0.1)
    assert riparian_irrigation_requirement_mm3(delta) == pytest.approx(expected_delta, rel=1e-9)


def test_riparian_requirement_every_riparian_matches_sum_of_crops(basin):
    for r in basin:
        expected = sum(irrigation_requirement_mm3(c, r.et0_mm_day, r.effective_rainfall_mm,
                                                  r.irrigation_efficiency) for c in r.crops)
        assert riparian_irrigation_requirement_mm3(r) == pytest.approx(expected, rel=1e-12)


def test_riparian_requirement_area_factor_is_linear(basin):
    highland = basin.riparian("Highland")
    base = riparian_irrigation_requirement_mm3(highland)
    assert riparian_irrigation_requirement_mm3(highland, area_factor=2.0) == pytest.approx(2 * base)
    assert riparian_irrigation_requirement_mm3(highland, area_factor=0.5) == pytest.approx(0.5 * base)
    assert riparian_irrigation_requirement_mm3(highland, area_factor=0.0) == 0.0
    # equivalent to physically enlarging every crop area by the same factor
    enlarged = highland.copy(crops=[replace(c, area_ha=c.area_ha * 1.2) for c in highland.crops])
    assert riparian_irrigation_requirement_mm3(highland, 1.2) == pytest.approx(
        riparian_irrigation_requirement_mm3(enlarged), rel=1e-12
    )


def test_riparian_requirement_efficiency_override(basin):
    highland = basin.riparian("Highland")
    base = riparian_irrigation_requirement_mm3(highland)
    assert riparian_irrigation_requirement_mm3(highland, efficiency=0.5) == pytest.approx(base)
    assert riparian_irrigation_requirement_mm3(highland, efficiency=1.0) == pytest.approx(base / 2)
    assert riparian_irrigation_requirement_mm3(highland, efficiency=0.25) == pytest.approx(2 * base)
    effs = np.linspace(0.2, 1.0, 20)
    out = [riparian_irrigation_requirement_mm3(highland, efficiency=float(e)) for e in effs]
    assert all(b <= a for a, b in zip(out, out[1:]))
    # None means the riparian's own efficiency
    assert riparian_irrigation_requirement_mm3(highland, efficiency=None) == pytest.approx(base)


def test_riparian_requirement_no_crops_and_rainfed():
    assert riparian_irrigation_requirement_mm3(_riparian([])) == 0.0
    assert riparian_crop_requirements_mm3(_riparian([])) == {}
    # rainfall covers ET of every crop: requirement 0 even though crops exist
    r = _riparian([_wheat(), _maize()], et0=1.0, pe=1000.0)
    assert riparian_irrigation_requirement_mm3(r) == 0.0
    assert riparian_crop_requirements_mm3(r) == {"wheat": 0.0, "maize": 0.0}


def test_riparian_requirement_duplicate_crop_names_are_all_counted():
    r = _riparian([_wheat(), _wheat(), _wheat()])
    per_crop = riparian_crop_requirements_mm3(r)
    assert list(per_crop) == ["wheat", "wheat#2", "wheat#3"]
    assert all(v == pytest.approx(630.0) for v in per_crop.values())
    assert riparian_irrigation_requirement_mm3(r) == pytest.approx(3 * 630.0)


def test_riparian_requirement_does_not_mutate(basin):
    snapshot = copy.deepcopy(basin)
    for r in basin:
        riparian_irrigation_requirement_mm3(r, area_factor=1.3, efficiency=0.7)
        riparian_crop_requirements_mm3(r, area_factor=0.3)
    assert basin == snapshot


@pytest.mark.parametrize("bad", [None, 1.0, "Highland", object()])
def test_riparian_requirement_rejects_non_riparian(bad):
    with pytest.raises(ValueError, match="Riparian"):
        riparian_irrigation_requirement_mm3(bad)
    with pytest.raises(ValueError, match="Riparian"):
        riparian_crop_requirements_mm3(bad)


@pytest.mark.parametrize("factor", [-0.1, float("nan"), float("inf"), "x", None])
def test_riparian_requirement_rejects_bad_area_factor(basin, factor):
    with pytest.raises(ValueError, match="area_factor"):
        riparian_irrigation_requirement_mm3(basin.riparian("Highland"), area_factor=factor)


@pytest.mark.parametrize("eta", [0.0, -0.5, 1.5, float("nan"), "x"])
def test_riparian_requirement_rejects_bad_efficiency_override(basin, eta):
    with pytest.raises(ValueError, match="efficiency"):
        riparian_irrigation_requirement_mm3(basin.riparian("Highland"), efficiency=eta)


def test_riparian_requirement_rejects_bad_riparian_attributes():
    with pytest.raises(ValueError, match="irrigation_efficiency"):
        riparian_irrigation_requirement_mm3(_riparian([_wheat()], eff=0.0))
    with pytest.raises(ValueError, match="et0_mm_day"):
        riparian_irrigation_requirement_mm3(_riparian([_wheat()], et0=-1.0))
    with pytest.raises(ValueError, match="effective_rainfall_mm"):
        riparian_irrigation_requirement_mm3(_riparian([_wheat()], pe=-1.0))
    with pytest.raises(ValueError, match="area_ha"):
        riparian_irrigation_requirement_mm3(_riparian([_wheat(-5.0)]))


# --------------------------------------------------------------------------- #
# FAO-33 yield response
# --------------------------------------------------------------------------- #
def test_fao33_closed_form_literature_examples():
    """FAO-33: 1 - Ya/Ym = ky (1 - ETa/ETm)."""
    # full water -> maximum yield
    assert fao33_yield(4.0, 1.05, 1.0) == pytest.approx(4.0)
    # wheat ky 1.05, 20 % ET deficit -> Ya = 4 * (1 - 0.21) = 3.16 t/ha
    assert fao33_yield(4.0, 1.05, 0.8) == pytest.approx(3.16, rel=1e-12)
    # maize ky 1.25, 50 % deficit -> relative yield 0.375 (Doorenbos & Kassam 1979)
    assert fao33_relative_yield(1.25, 0.5) == pytest.approx(0.375, rel=1e-12)
    assert fao33_yield(6.0, 1.25, 0.5) == pytest.approx(2.25, rel=1e-12)
    # cotton ky 0.85 keeps 15 % of yield with no water at all
    assert fao33_yield(2.5, 0.85, 0.0) == pytest.approx(0.375, rel=1e-12)
    assert fao33_relative_yield(0.85, 0.0) == pytest.approx(0.15, rel=1e-12)
    # ky 1: yield proportional to ET
    assert fao33_yield(10.0, 1.0, 0.3) == pytest.approx(3.0)


def test_fao33_clipping():
    # ky > 1 and no water -> negative formula value clipped to 0
    assert fao33_yield(4.0, 1.05, 0.0) == 0.0
    assert fao33_relative_yield(1.05, 0.0) == 0.0
    # zero yield exactly at et_ratio = 1 - 1/ky and below
    assert fao33_yield(4.0, 1.25, 0.2) == pytest.approx(0.0, abs=1e-12)
    assert fao33_yield(4.0, 1.25, 0.1) == 0.0
    # et_ratio above 1 counts as full ET -> Ym (clipped to [0, Ym])
    assert fao33_yield(4.0, 1.05, 1.5) == pytest.approx(4.0)
    assert fao33_relative_yield(1.05, 7.0) == 1.0
    # ky 0: water-insensitive, always Ym
    assert fao33_yield(4.0, 0.0, 0.0) == pytest.approx(4.0)
    # Ym 0 -> 0
    assert fao33_yield(0.0, 1.05, 1.0) == 0.0


def test_fao33_bounds_and_monotonicity():
    rng = np.random.default_rng(12)
    for _ in range(300):
        ym = float(rng.uniform(0, 30))
        ky = float(rng.uniform(0, 2))
        ratio = float(rng.uniform(0, 1.5))
        ya = fao33_yield(ym, ky, ratio)
        assert 0.0 <= ya <= ym + 1e-12
        rel = fao33_relative_yield(ky, ratio)
        assert 0.0 <= rel <= 1.0
        assert ya == pytest.approx(ym * rel)
    ratios = np.linspace(0, 1.2, 60)
    for ky in (0.0, 0.85, 1.0, 1.05, 1.25, 2.0):
        out = [fao33_yield(5.0, ky, float(r)) for r in ratios]
        assert all(b >= a for a, b in zip(out, out[1:]))
        assert out[-1] == pytest.approx(5.0)
    # higher ky -> lower yield for the same deficit
    kys = np.linspace(0, 2, 30)
    out = [fao33_yield(5.0, float(k), 0.6) for k in kys]
    assert all(b <= a for a, b in zip(out, out[1:]))


def test_fao33_linear_segment_matches_formula():
    rng = np.random.default_rng(13)
    for _ in range(100):
        ym = float(rng.uniform(1, 20))
        ky = float(rng.uniform(0.5, 1.3))
        # keep to the unclipped region
        ratio = float(rng.uniform(max(1 - 1 / ky, 0.0) + 0.01, 1.0))
        assert fao33_yield(ym, ky, ratio) == pytest.approx(ym * (1 - ky * (1 - ratio)), rel=1e-12)


@pytest.mark.parametrize("bad", BAD_NUMBERS)
def test_fao33_rejects_bad_inputs(bad):
    with pytest.raises(ValueError):
        fao33_yield(bad, 1.0, 1.0)
    with pytest.raises(ValueError):
        fao33_yield(4.0, bad, 1.0)
    with pytest.raises(ValueError):
        fao33_yield(4.0, 1.0, bad)
    with pytest.raises(ValueError):
        fao33_relative_yield(bad, 1.0)
    with pytest.raises(ValueError):
        fao33_relative_yield(1.0, bad)


def test_fao33_error_messages_name_the_argument():
    with pytest.raises(ValueError, match="yield_max_t_ha"):
        fao33_yield(-4.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="ky"):
        fao33_yield(4.0, -1.0, 1.0)
    with pytest.raises(ValueError, match="et_ratio"):
        fao33_yield(4.0, 1.0, -0.5)


# --------------------------------------------------------------------------- #
# FAO-33 relative evapotranspiration from rainfall + net irrigation
# --------------------------------------------------------------------------- #
def test_rainfall_et_share_closed_form_and_edges():
    assert rainfall_et_share(510.0, 300.0) == pytest.approx(300.0 / 510.0, rel=1e-12)
    assert rainfall_et_share(510.0, 0.0) == 0.0
    assert rainfall_et_share(200.0, 300.0) == 1.0      # rain-fed: capped at 1
    assert rainfall_et_share(300.0, 300.0) == 1.0
    assert rainfall_et_share(0.0, 0.0) == 1.0          # no evaporative demand to meet
    assert rainfall_et_share(0.0, 100.0) == 1.0


def test_rainfall_et_share_bounds_and_monotonicity():
    rng = np.random.default_rng(20)
    for _ in range(200):
        etc, pe = float(rng.uniform(0, 1500)), float(rng.uniform(0, 1500))
        share = rainfall_et_share(etc, pe)
        assert 0.0 <= share <= 1.0
        # its complement is the irrigated share of crop ET, IRn / ETc
        if etc > 0:
            assert 1.0 - share == pytest.approx(net_irrigation_mm(etc, pe) / etc, abs=1e-12)
    pes = np.linspace(0, 800, 40)
    out = [rainfall_et_share(500.0, float(p)) for p in pes]
    assert all(b >= a for a, b in zip(out, out[1:]))
    etcs = np.linspace(1, 1000, 40)
    out = [rainfall_et_share(float(e), 300.0) for e in etcs]
    assert all(b <= a for a, b in zip(out, out[1:]))


@pytest.mark.parametrize("bad", BAD_NUMBERS)
def test_rainfall_et_share_rejects_bad_inputs(bad):
    with pytest.raises(ValueError):
        rainfall_et_share(bad, 100.0)
    with pytest.raises(ValueError):
        rainfall_et_share(100.0, bad)


def test_relative_evapotranspiration_closed_form():
    """ETa/ETm = Pe/ETc + s (1 - Pe/ETc): Highland wheat, Pe/ETc = 300/510."""
    share = 300.0 / 510.0
    assert relative_evapotranspiration(0.0, share) == pytest.approx(share, rel=1e-12)
    assert relative_evapotranspiration(0.5, share) == pytest.approx((300 + 105) / 510, rel=1e-12)
    assert relative_evapotranspiration(0.25, share) == pytest.approx((300 + 52.5) / 510, rel=1e-12)
    assert relative_evapotranspiration(1.0, share) == 1.0
    # no rainfall: reduces to the gross supply ratio
    assert relative_evapotranspiration(0.3) == pytest.approx(0.3)
    assert relative_evapotranspiration(0.3, 0.0) == pytest.approx(0.3)
    # rain-fed crop: always 1
    assert relative_evapotranspiration(0.0, 1.0) == 1.0
    # supply above the requirement counts as 1
    assert relative_evapotranspiration(2.0, 0.2) == 1.0
    assert relative_evapotranspiration(0.0, 0.0) == 0.0


def test_relative_evapotranspiration_bounds_and_monotonicity():
    rng = np.random.default_rng(21)
    for _ in range(300):
        s, share = float(rng.uniform(0, 1.5)), float(rng.uniform(0, 1))
        et = relative_evapotranspiration(s, share)
        assert 0.0 <= et <= 1.0
        assert et >= min(s, 1.0) - 1e-12 and et >= share - 1e-12
        assert et == pytest.approx(1.0 - (1.0 - min(s, 1.0)) * (1.0 - share), abs=1e-12)
    grid = np.linspace(0, 1.2, 50)
    out = [relative_evapotranspiration(float(s), 0.4) for s in grid]
    assert all(b >= a for a, b in zip(out, out[1:]))
    shares = np.linspace(0, 1, 50)
    out = [relative_evapotranspiration(0.4, float(x)) for x in shares]
    assert all(b >= a for a, b in zip(out, out[1:]))


@pytest.mark.parametrize("bad", BAD_NUMBERS)
def test_relative_evapotranspiration_rejects_bad_supply_ratio(bad):
    with pytest.raises(ValueError, match="supply_ratio"):
        relative_evapotranspiration(bad, 0.5)


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), float("inf"), "x", None, True])
def test_relative_evapotranspiration_rejects_bad_rainfall_share(bad):
    with pytest.raises(ValueError, match="rainfall_share"):
        relative_evapotranspiration(0.5, bad)


def test_fao33_docstrings_quote_the_fao33_ky_table_correctly():
    """Doorenbos & Kassam (1979) Table 24: groundnut 0.70, winter wheat 1.00, spring wheat 1.15,
    maize 1.25, cotton 0.85, potato 1.10; no seasonal ky for flooded rice (a later source such
    as Steduto et al. 2012, FAO-66, has to be cited for rice)."""
    for doc in (fao33_relative_yield.__doc__, food.__doc__):
        text = " ".join(doc.split())
        assert "0.70 (groundnut)" in text or "groundnut 0.70" in text
        assert "1.00 (winter wheat)" in text or "winter wheat 1.00" in text
        assert "1.15 (spring wheat)" in text or "spring wheat 1.15" in text
        assert "1.25 (maize)" in text or "maize 1.25" in text
        assert "0.85 (cotton)" in text or "cotton 0.85" in text
        assert "1.10 (potato)" in text or "potato 1.10" in text
        assert "FAO-66" in text and "rice" in text
        # the mis-quoted values of the earlier docstring must be gone
        assert "0.8 (groundnut)" not in text
        assert "1.05 (wheat)" not in text
        assert "1.2 (rice)" not in text


# --------------------------------------------------------------------------- #
# Crop production
# --------------------------------------------------------------------------- #
def test_crop_production_full_supply_highland_wheat():
    out = crop_production(_wheat(), 630.0, 630.0)
    assert CROP_KEYS <= set(out)
    assert out["area_ha"] == pytest.approx(150_000.0)
    assert out["et_ratio"] == pytest.approx(1.0)
    assert out["yield_t_ha"] == pytest.approx(4.0)
    assert out["production_t"] == pytest.approx(600_000.0)
    assert out["kcal"] == pytest.approx(600_000.0 * 1000 * 3400)      # 2.04e12
    assert out["value_usd"] == pytest.approx(600_000.0 * 250)         # 1.5e8
    assert out["water_required_mm3"] == 630.0
    assert out["water_supplied_mm3"] == 630.0
    assert all(isinstance(v, float) for v in out.values())


def test_crop_production_partial_supply():
    out = crop_production(_wheat(), 315.0, 630.0)
    assert out["et_ratio"] == pytest.approx(0.5)
    assert out["yield_t_ha"] == pytest.approx(fao33_yield(4.0, 1.05, 0.5))
    assert out["yield_t_ha"] == pytest.approx(4.0 * (1 - 1.05 * 0.5), rel=1e-12)
    assert out["production_t"] == pytest.approx(out["yield_t_ha"] * 150_000)
    assert out["kcal"] == pytest.approx(out["production_t"] * 1000 * 3400)
    assert out["value_usd"] == pytest.approx(out["production_t"] * 250)


def test_crop_production_zero_and_excess_supply():
    # no water: wheat (ky 1.05) yields nothing
    out = crop_production(_wheat(), 0.0, 630.0)
    assert out["et_ratio"] == 0.0
    assert out["yield_t_ha"] == 0.0 and out["production_t"] == 0.0
    assert out["kcal"] == 0.0 and out["value_usd"] == 0.0
    # no water: cotton (ky 0.85) still yields 15 %
    out = crop_production(_cotton(), 0.0, 1482.0)
    assert out["yield_t_ha"] == pytest.approx(0.375)
    assert out["production_t"] == pytest.approx(37_500.0)
    # excess water is capped: et_ratio 1, same as exact supply
    full = crop_production(_wheat(), 630.0, 630.0)
    excess = crop_production(_wheat(), 10_000.0, 630.0)
    assert excess["et_ratio"] == 1.0
    for k in CROP_KEYS:
        assert excess[k] == pytest.approx(full[k])
    assert excess["water_supplied_mm3"] == 10_000.0  # echoed input, not the capped value


def test_crop_production_zero_requirement_means_rainfed_full_yield():
    # required == 0 -> et_ratio 1 regardless of supply (even 0)
    out = crop_production(_wheat(), 0.0, 0.0)
    assert out["et_ratio"] == 1.0
    assert out["yield_t_ha"] == pytest.approx(4.0)
    assert out["production_t"] == pytest.approx(600_000.0)
    out = crop_production(_wheat(), 123.0, 0.0)
    assert out["et_ratio"] == 1.0


def test_crop_production_area_factor():
    base = crop_production(_wheat(), 630.0, 630.0)
    doubled = crop_production(_wheat(), 1260.0, 1260.0, area_factor=2.0)
    assert doubled["area_ha"] == pytest.approx(2 * base["area_ha"])
    assert doubled["production_t"] == pytest.approx(2 * base["production_t"])
    assert doubled["kcal"] == pytest.approx(2 * base["kcal"])
    assert doubled["value_usd"] == pytest.approx(2 * base["value_usd"])
    assert doubled["yield_t_ha"] == pytest.approx(base["yield_t_ha"])
    zero = crop_production(_wheat(), 0.0, 0.0, area_factor=0.0)
    assert zero["area_ha"] == 0.0 and zero["production_t"] == 0.0 and zero["kcal"] == 0.0


def test_crop_production_non_food_crop_has_zero_kcal_but_value():
    out = crop_production(_cotton(), 1482.0, 1482.0)
    assert out["kcal"] == 0.0
    assert out["production_t"] == pytest.approx(250_000.0)
    assert out["value_usd"] == pytest.approx(250_000.0 * 1600)
    # zero price crop: value 0 but kcal present
    free = Crop("free", 1000.0, 1.0, 100, 2.0, 1.0, 3000.0)   # price_usd_t defaults to 0
    out = crop_production(free, 1.0, 1.0)
    assert out["value_usd"] == 0.0
    assert out["kcal"] == pytest.approx(2000.0 * 1000 * 3000)


def test_crop_production_bounds_and_monotone_in_supply():
    supplies = np.linspace(0, 800, 50)
    out = [crop_production(_maize(), float(s), 168.0) for s in supplies]
    prod = [o["production_t"] for o in out]
    kcal = [o["kcal"] for o in out]
    assert all(b >= a for a, b in zip(prod, prod[1:]))
    assert all(b >= a for a, b in zip(kcal, kcal[1:]))
    assert all(0.0 <= o["et_ratio"] <= 1.0 for o in out)
    assert all(0.0 <= o["yield_t_ha"] <= 6.0 + 1e-12 for o in out)
    assert max(prod) == pytest.approx(6.0 * 50_000)


def test_crop_production_random_consistency():
    rng = np.random.default_rng(14)
    for _ in range(100):
        crop = Crop("c", float(rng.uniform(0, 1e5)), 0.9, 120, float(rng.uniform(0, 10)),
                    float(rng.uniform(0, 1.5)), float(rng.uniform(0, 4000)), float(rng.uniform(0, 1000)))
        s, r = float(rng.uniform(0, 100)), float(rng.uniform(0, 100))
        out = crop_production(crop, s, r)
        ratio = min(s / r, 1.0) if r > 0 else 1.0
        assert out["et_ratio"] == pytest.approx(ratio)
        assert out["yield_t_ha"] == pytest.approx(fao33_yield(crop.yield_max_t_ha, crop.ky, ratio))
        assert out["production_t"] == pytest.approx(out["yield_t_ha"] * crop.area_ha)
        assert out["kcal"] == pytest.approx(out["production_t"] * KG_PER_T * crop.kcal_per_kg)
        assert out["value_usd"] == pytest.approx(out["production_t"] * crop.price_usd_t)


def test_crop_production_does_not_mutate_crop():
    crop = _wheat()
    before = copy.deepcopy(crop)
    out = crop_production(crop, 100.0, 630.0, area_factor=3.0)
    out["area_ha"] = -1.0  # editing the result must not touch the crop either
    assert crop == before


@pytest.mark.parametrize("not_a_crop", [None, 3.0, "wheat", {"area_ha": 1.0}])
def test_crop_production_rejects_non_crop(not_a_crop):
    with pytest.raises(ValueError, match="Crop"):
        crop_production(not_a_crop, 1.0, 1.0)


@pytest.mark.parametrize("bad", BAD_NUMBERS)
def test_crop_production_rejects_bad_water(bad):
    with pytest.raises(ValueError):
        crop_production(_wheat(), bad, 1.0)
    with pytest.raises(ValueError):
        crop_production(_wheat(), 1.0, bad)


@pytest.mark.parametrize("factor", [-0.1, float("nan"), "x", None])
def test_crop_production_rejects_bad_area_factor(factor):
    with pytest.raises(ValueError, match="area_factor"):
        crop_production(_wheat(), 1.0, 1.0, area_factor=factor)


def test_crop_production_rejects_bad_crop_attributes():
    with pytest.raises(ValueError, match="yield_max_t_ha"):
        crop_production(Crop("w", 1.0, 1.0, 100, -1.0, 1.0, 1.0), 1.0, 1.0)
    with pytest.raises(ValueError, match="ky"):
        crop_production(Crop("w", 1.0, 1.0, 100, 1.0, -1.0, 1.0), 1.0, 1.0)
    with pytest.raises(ValueError, match="kcal_per_kg"):
        crop_production(Crop("w", 1.0, 1.0, 100, 1.0, 1.0, -1.0), 1.0, 1.0)
    with pytest.raises(ValueError, match="price_usd_t"):
        crop_production(Crop("w", 1.0, 1.0, 100, 1.0, 1.0, 1.0, -5.0), 1.0, 1.0)
    with pytest.raises(ValueError, match="area_ha"):
        crop_production(_wheat(-1.0), 1.0, 1.0)


def test_crop_production_fao33_counts_effective_rainfall_worked_example():
    """Highland wheat: ETc 510 mm, Pe 300 mm, IRn 210, IRg 420 mm (630 Mm3 on 150 000 ha), ky 1.05.

    FAO-33 defines the deficit on total crop ET, ETa/ETm = (Pe + net applied) / ETc, so with no
    irrigation ETa/ETm = 300/510 = 0.588 and Ya/Ym = 1 - 1.05 (1 - 0.588) = 0.568 - not 0 as the
    gross supply ratio proxy gave; 50 % gross supply -> 0.784, 25 % -> 0.676.
    """
    wheat = _wheat()
    cases = {0.0: (300.0 / 510.0, 0.5676), 0.25: (352.5 / 510.0, 0.6757),
             0.5: (405.0 / 510.0, 0.7838), 1.0: (1.0, 1.0)}
    for share, (et_expected, rel_yield_expected) in cases.items():
        out = crop_production(wheat, share * 630.0, 630.0, et0_mm_day=4.0, effective_rainfall_mm=300.0)
        assert out["irrigation_supply_ratio"] == pytest.approx(share)
        assert out["rainfall_et_share"] == pytest.approx(300.0 / 510.0, rel=1e-12)
        assert out["et_ratio"] == pytest.approx(et_expected, rel=1e-12)
        assert out["yield_t_ha"] / 4.0 == pytest.approx(rel_yield_expected, abs=5e-4)
        assert out["yield_t_ha"] == pytest.approx(fao33_yield(4.0, 1.05, et_expected))
        # literal FAO-33 form: ETa = Pe + efficiency * supplied depth (efficiency 0.5)
        supplied_mm = share * 630.0 * M3_PER_MM3 / (150_000 * M3_PER_HA_MM)
        assert out["et_ratio"] == pytest.approx(min((300.0 + 0.5 * supplied_mm) / 510.0, 1.0), rel=1e-12)
    # the old gross-supply proxy understated ETa/ETm by Pe/ETc and reported a total failure
    proxy = crop_production(wheat, 0.0, 630.0)
    assert proxy["et_ratio"] == 0.0 and proxy["production_t"] == 0.0
    with_rain = crop_production(wheat, 0.0, 630.0, et0_mm_day=4.0, effective_rainfall_mm=300.0)
    assert with_rain["production_t"] == pytest.approx(fao33_yield(4.0, 1.05, 300.0 / 510.0) * 150_000)
    assert with_rain["production_t"] > 0.5 * 600_000.0


def test_crop_production_climate_matches_literal_fao33_form_randomly():
    """ETa/ETm = min((Pe + eff * supplied_mm) / ETc, 1) when `required` is the gross requirement."""
    rng = np.random.default_rng(22)
    for _ in range(200):
        crop = Crop("c", float(rng.uniform(1, 1e5)), float(rng.uniform(0.3, 1.3)),
                    int(rng.integers(30, 365)), float(rng.uniform(0, 10)),
                    float(rng.uniform(0, 1.5)), 3000.0, 100.0)
        et0, pe, eff = float(rng.uniform(0.1, 9)), float(rng.uniform(0, 800)), float(rng.uniform(0.3, 1.0))
        factor = float(rng.uniform(0.1, 2.0))
        required = irrigation_requirement_mm3(crop, et0, pe, eff) * factor
        supplied = float(rng.uniform(0, 1.3)) * required
        out = crop_production(crop, supplied, required, area_factor=factor,
                              et0_mm_day=et0, effective_rainfall_mm=pe)
        etc = crop_evapotranspiration_mm(et0, crop.kc, crop.season_days)
        area = crop.area_ha * factor
        if required > 0:
            supplied_mm = min(supplied, required) * M3_PER_MM3 / (area * M3_PER_HA_MM)
            literal = min((pe + eff * supplied_mm) / etc, 1.0)
        else:
            literal = 1.0
        assert out["et_ratio"] == pytest.approx(literal, abs=1e-9)
        assert out["rainfall_et_share"] == pytest.approx(min(pe / etc, 1.0))
        assert out["et_ratio"] >= out["irrigation_supply_ratio"] - 1e-12
        assert out["et_ratio"] >= out["rainfall_et_share"] - 1e-12
        assert 0.0 <= out["et_ratio"] <= 1.0
        assert out["yield_t_ha"] == pytest.approx(fao33_yield(crop.yield_max_t_ha, crop.ky, out["et_ratio"]))
        assert out["production_t"] == pytest.approx(out["yield_t_ha"] * area)


def test_crop_production_rainfed_crop_has_full_et_whatever_the_supply():
    # rainfall covers the whole ET (ETc 510 <= Pe 600): no deficit even with zero irrigation,
    # even if a caller passes a spurious positive requirement
    for supplied, required in ((0.0, 0.0), (0.0, 100.0), (50.0, 100.0)):
        out = crop_production(_wheat(), supplied, required, et0_mm_day=4.0, effective_rainfall_mm=600.0)
        assert out["rainfall_et_share"] == 1.0
        assert out["et_ratio"] == 1.0
        assert out["production_t"] == pytest.approx(600_000.0)
    # exactly ETc == Pe is rain-fed as well
    out = crop_production(_wheat(), 0.0, 0.0, et0_mm_day=4.0, effective_rainfall_mm=510.0)
    assert out["et_ratio"] == 1.0 and out["rainfall_et_share"] == 1.0
    # no evaporative demand at all (ETc 0): nothing to be short of
    out = crop_production(_wheat(), 0.0, 0.0, et0_mm_day=0.0, effective_rainfall_mm=0.0)
    assert out["et_ratio"] == 1.0 and out["rainfall_et_share"] == 1.0
    zero_season = Crop("x", 1000.0, 1.0, 0, 2.0, 1.0, 1000.0)
    out = crop_production(zero_season, 0.0, 0.0, et0_mm_day=5.0, effective_rainfall_mm=0.0)
    assert out["et_ratio"] == 1.0
    # zero requirement keeps et_ratio 1 regardless of the climate given
    out = crop_production(_wheat(), 0.0, 0.0, et0_mm_day=4.0, effective_rainfall_mm=0.0)
    assert out["et_ratio"] == 1.0 and out["irrigation_supply_ratio"] == 1.0


def test_crop_production_without_rainfall_reduces_to_supply_ratio():
    """With Pe = 0 the climate path and the no-climate default agree exactly."""
    # Pe 0 -> IRn = ETc = 510 mm, IRg 1020 mm at efficiency 0.5 -> 1530 Mm3
    for s in (0.0, 0.2, 0.5, 0.9, 1.0, 1.7):
        a = crop_production(_wheat(), s * 1530.0, 1530.0)
        b = crop_production(_wheat(), s * 1530.0, 1530.0, et0_mm_day=4.0, effective_rainfall_mm=0.0)
        assert a["rainfall_et_share"] == 0.0 and b["rainfall_et_share"] == 0.0
        assert a["et_ratio"] == pytest.approx(min(s, 1.0))
        assert b["et_ratio"] == pytest.approx(min(s, 1.0))
        for k in CROP_KEYS:
            assert a[k] == pytest.approx(b[k])
    # the no-climate default is documented as "no effective rainfall"
    assert crop_production(_wheat(), 100.0, 630.0)["rainfall_et_share"] == 0.0


def test_crop_production_et_ratio_monotone_in_rainfall_and_supply():
    pes = np.linspace(0, 600, 40)
    out = [crop_production(_wheat(), 100.0, 630.0, et0_mm_day=4.0, effective_rainfall_mm=float(p))
           for p in pes]
    et = [o["et_ratio"] for o in out]
    prod = [o["production_t"] for o in out]
    assert all(b >= a for a, b in zip(et, et[1:]))
    assert all(b >= a for a, b in zip(prod, prod[1:]))
    assert et[-1] == 1.0  # Pe 600 >= ETc 510
    supplies = np.linspace(0, 800, 40)
    out = [crop_production(_wheat(), float(s), 630.0, et0_mm_day=4.0, effective_rainfall_mm=300.0)
           for s in supplies]
    et = [o["et_ratio"] for o in out]
    assert all(b >= a for a, b in zip(et, et[1:]))
    assert et[0] == pytest.approx(300.0 / 510.0) and et[-1] == 1.0
    assert all(300.0 / 510.0 - 1e-12 <= e <= 1.0 for e in et)


def test_crop_production_climate_arguments_must_come_together():
    with pytest.raises(ValueError, match="together"):
        crop_production(_wheat(), 100.0, 630.0, et0_mm_day=4.0)
    with pytest.raises(ValueError, match="together"):
        crop_production(_wheat(), 100.0, 630.0, effective_rainfall_mm=300.0)


@pytest.mark.parametrize("bad", BAD_NUMBERS)
def test_crop_production_rejects_bad_climate(bad):
    with pytest.raises(ValueError):
        crop_production(_wheat(), 100.0, 630.0, et0_mm_day=bad, effective_rainfall_mm=300.0)
    with pytest.raises(ValueError):
        crop_production(_wheat(), 100.0, 630.0, et0_mm_day=4.0, effective_rainfall_mm=bad)


def test_crop_production_with_climate_validates_kc_and_season():
    bad_kc = Crop("w", 1.0, -0.5, 100, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="kc"):
        crop_production(bad_kc, 1.0, 1.0, et0_mm_day=4.0, effective_rainfall_mm=0.0)
    bad_days = Crop("w", 1.0, 0.5, -100, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="season_days"):
        crop_production(bad_days, 1.0, 1.0, et0_mm_day=4.0, effective_rainfall_mm=0.0)


def test_crop_production_result_keys_and_types_with_climate():
    out = crop_production(_wheat(), 315.0, 630.0, et0_mm_day=4.0, effective_rainfall_mm=300.0)
    assert CROP_KEYS | {"irrigation_supply_ratio", "rainfall_et_share",
                        "water_required_mm3", "water_supplied_mm3"} == set(out)
    assert all(isinstance(v, float) and math.isfinite(v) for v in out.values())
    crop = _wheat()
    before = copy.deepcopy(crop)
    crop_production(crop, 315.0, 630.0, et0_mm_day=4.0, effective_rainfall_mm=300.0)
    assert crop == before


# --------------------------------------------------------------------------- #
# Riparian food production
# --------------------------------------------------------------------------- #
def test_riparian_food_production_highland_full_supply(basin):
    highland = basin.riparian("Highland")
    out = riparian_food_production(highland, 798.0)
    assert RIPARIAN_KEYS <= set(out)
    assert set(out["crops"]) == {"wheat", "maize"}
    assert out["requirement_mm3"] == pytest.approx(798.0, rel=1e-9)
    assert out["supplied_mm3"] == pytest.approx(798.0, rel=1e-9)
    assert out["et_ratio"] == pytest.approx(1.0)
    # wheat 150 000 ha * 4 t/ha + maize 50 000 ha * 6 t/ha
    assert out["production_t"] == pytest.approx(600_000.0 + 300_000.0)
    assert out["kcal"] == pytest.approx(600_000 * 1000 * 3400 + 300_000 * 1000 * 3600)  # 3.12e12
    assert out["value_usd"] == pytest.approx(600_000 * 250 + 300_000 * 200)             # 2.1e8
    assert out["crops"]["wheat"]["production_t"] == pytest.approx(600_000.0)
    assert out["crops"]["maize"]["production_t"] == pytest.approx(300_000.0)
    assert out["deficit_mm3"] == pytest.approx(0.0, abs=1e-9)
    assert out["available_mm3"] == 798.0
    assert out["area_ha"] == pytest.approx(200_000.0)


def test_riparian_food_production_excess_water_is_capped(basin):
    highland = basin.riparian("Highland")
    full = riparian_food_production(highland, 798.0)
    excess = riparian_food_production(highland, 50_000.0)
    assert excess["supplied_mm3"] == pytest.approx(798.0, rel=1e-9)
    assert excess["available_mm3"] == 50_000.0
    assert excess["et_ratio"] == pytest.approx(1.0)
    for k in ("production_t", "kcal", "value_usd", "requirement_mm3", "supplied_mm3"):
        assert excess[k] == pytest.approx(full[k])


def test_riparian_food_production_proportional_sharing(basin):
    highland = basin.riparian("Highland")
    out = riparian_food_production(highland, 399.0)   # exactly half of 798
    assert out["irrigation_supply_ratio"] == pytest.approx(0.5)
    assert out["supplied_mm3"] == pytest.approx(399.0)
    assert out["deficit_mm3"] == pytest.approx(399.0)
    crops = out["crops"]
    # every irrigated crop gets the same relative (gross) supply ...
    assert crops["wheat"]["irrigation_supply_ratio"] == pytest.approx(0.5)
    assert crops["maize"]["irrigation_supply_ratio"] == pytest.approx(0.5)
    # ... proportional to its requirement (wheat 630 -> 315, maize 168 -> 84)
    assert crops["wheat"]["water_supplied_mm3"] == pytest.approx(315.0)
    assert crops["maize"]["water_supplied_mm3"] == pytest.approx(84.0)
    assert crops["wheat"]["water_required_mm3"] == pytest.approx(630.0)
    assert crops["maize"]["water_required_mm3"] == pytest.approx(168.0)
    # conservation: water handed to the crops sums to the water supplied
    assert sum(c["water_supplied_mm3"] for c in crops.values()) == pytest.approx(out["supplied_mm3"])
    assert sum(c["water_required_mm3"] for c in crops.values()) == pytest.approx(out["requirement_mm3"])
    # FAO-33 ETa/ETm counts the 300 mm of effective rainfall: ETa = Pe + 0.5 * IRn
    # wheat ETc 510, IRn 210 -> (300 + 105)/510; maize ETc 468, IRn 168 -> (300 + 84)/468
    wheat_et = (300.0 + 0.5 * 210.0) / 510.0
    maize_et = (300.0 + 0.5 * 168.0) / 468.0
    assert crops["wheat"]["et_ratio"] == pytest.approx(wheat_et, rel=1e-12)
    assert crops["maize"]["et_ratio"] == pytest.approx(maize_et, rel=1e-12)
    assert crops["wheat"]["rainfall_et_share"] == pytest.approx(300.0 / 510.0, rel=1e-12)
    assert crops["maize"]["rainfall_et_share"] == pytest.approx(300.0 / 468.0, rel=1e-12)
    # yields follow FAO-33 at those relative ETs (not at the gross supply ratio 0.5)
    assert crops["wheat"]["yield_t_ha"] == pytest.approx(fao33_yield(4.0, 1.05, wheat_et))
    assert crops["maize"]["yield_t_ha"] == pytest.approx(fao33_yield(6.0, 1.25, maize_et))
    assert crops["wheat"]["yield_t_ha"] > fao33_yield(4.0, 1.05, 0.5)
    # overall ETa/ETm weights each crop by its maximum ET volume:
    # ETm = 510 mm * 150 000 ha + 468 mm * 50 000 ha = 765 + 234 = 999 Mm3
    # ETa = Pe volume 300 mm * 200 000 ha (600) + net irrigation 0.5 * 399 (199.5) = 799.5 Mm3
    assert out["etm_mm3"] == pytest.approx(999.0, rel=1e-12)
    assert out["eta_mm3"] == pytest.approx(799.5, rel=1e-12)
    assert out["et_ratio"] == pytest.approx(799.5 / 999.0, rel=1e-12)
    assert out["et_ratio"] == pytest.approx((wheat_et * 765.0 + maize_et * 234.0) / 999.0, rel=1e-12)
    # totals are the sums of the crops
    for k in ("production_t", "kcal", "value_usd"):
        assert out[k] == pytest.approx(sum(c[k] for c in crops.values()))


def test_riparian_food_production_zero_water(basin):
    highland = basin.riparian("Highland")
    out = riparian_food_production(highland, 0.0)
    assert out["supplied_mm3"] == 0.0
    assert out["irrigation_supply_ratio"] == 0.0
    assert out["requirement_mm3"] == pytest.approx(798.0, rel=1e-9)
    assert out["deficit_mm3"] == pytest.approx(798.0, rel=1e-9)
    # Highland's 300 mm of effective rainfall still meets 600 of 999 Mm3 of crop ET,
    # so a dry year is a partial, not a total, crop failure (FAO-33 on total ET)
    assert out["eta_mm3"] == pytest.approx(600.0, rel=1e-12)
    assert out["et_ratio"] == pytest.approx(600.0 / 999.0, rel=1e-12)
    wheat_t = fao33_yield(4.0, 1.05, 300.0 / 510.0) * 150_000
    maize_t = fao33_yield(6.0, 1.25, 300.0 / 468.0) * 50_000
    assert out["crops"]["wheat"]["production_t"] == pytest.approx(wheat_t, rel=1e-12)
    assert out["crops"]["maize"]["production_t"] == pytest.approx(maize_t, rel=1e-12)
    assert out["production_t"] == pytest.approx(wheat_t + maize_t)
    assert out["production_t"] == pytest.approx(505_973.0, abs=1.0)
    assert out["kcal"] == pytest.approx(wheat_t * 1000 * 3400 + maize_t * 1000 * 3600)
    assert out["value_usd"] == pytest.approx(wheat_t * 250 + maize_t * 200)
    # Midland (Pe 150 mm): cotton (ky 0.85) keeps 15 % plus its rain-fed ET share
    midland = basin.riparian("Midland")
    out = riparian_food_production(midland, 0.0)
    assert out["irrigation_supply_ratio"] == 0.0
    cotton_et = 150.0 / (5.5 * 0.90 * 180)   # Pe / ETc = 150 / 891
    assert out["crops"]["cotton"]["et_ratio"] == pytest.approx(cotton_et, rel=1e-12)
    assert out["crops"]["cotton"]["production_t"] == pytest.approx(
        fao33_yield(2.5, 0.85, cotton_et) * 100_000, rel=1e-12)
    assert out["crops"]["cotton"]["production_t"] > 0.15 * 2.5 * 100_000
    assert out["crops"]["cotton"]["kcal"] == 0.0           # cotton has no calories
    assert out["kcal"] == pytest.approx(out["crops"]["wheat"]["kcal"] + out["crops"]["vegetables"]["kcal"])
    assert out["kcal"] > 0.0
    # without any effective rainfall the old behaviour is recovered:
    # wheat & maize (ky > 1) give nothing at all without irrigation
    dry = _riparian([_wheat(), _maize()], et0=4.0, pe=0.0)
    out = riparian_food_production(dry, 0.0)
    assert out["et_ratio"] == 0.0 and out["irrigation_supply_ratio"] == 0.0
    assert out["eta_mm3"] == 0.0 and out["etm_mm3"] > 0.0
    assert out["production_t"] == 0.0 and out["kcal"] == 0.0 and out["value_usd"] == 0.0
    # ... and cotton (ky 0.85) keeps exactly 15 %
    out = riparian_food_production(_riparian([_cotton()], et0=5.5, pe=0.0), 0.0)
    assert out["et_ratio"] == 0.0
    assert out["crops"]["cotton"]["production_t"] == pytest.approx(0.15 * 2.5 * 100_000)
    assert out["kcal"] == 0.0
    assert out["value_usd"] == pytest.approx(0.15 * 2.5 * 100_000 * 1600)


def test_riparian_food_production_midland_cotton_has_no_calories(basin):
    midland = basin.riparian("Midland")
    out = riparian_food_production(midland, 1e6)
    assert set(out["crops"]) == {"wheat", "cotton", "vegetables"}
    assert out["crops"]["cotton"]["kcal"] == 0.0
    assert out["crops"]["cotton"]["value_usd"] == pytest.approx(2.5 * 100_000 * 1600)
    expected_kcal = 4.5 * 300_000 * 1000 * 3400 + 25.0 * 80_000 * 1000 * 300
    assert out["kcal"] == pytest.approx(expected_kcal)
    assert out["kcal"] == pytest.approx(out["crops"]["wheat"]["kcal"] + out["crops"]["vegetables"]["kcal"])
    assert out["production_t"] == pytest.approx(4.5 * 300_000 + 2.5 * 100_000 + 25.0 * 80_000)
    assert math.isfinite(out["kcal"]) and math.isfinite(out["value_usd"])


def test_riparian_food_production_all_example_riparians_are_well_formed(basin):
    for r in basin:
        for water in (0.0, 0.3 * r.demand.agricultural, r.demand.agricultural, 1e9):
            out = riparian_food_production(r, water)
            assert RIPARIAN_KEYS <= set(out)
            assert set(out["crops"]) == {c.name for c in r.crops}
            assert 0.0 <= out["et_ratio"] <= 1.0
            assert out["supplied_mm3"] == pytest.approx(min(water, out["requirement_mm3"]))
            assert out["requirement_mm3"] == pytest.approx(riparian_irrigation_requirement_mm3(r))
            assert out["production_t"] >= 0 and out["kcal"] >= 0 and out["value_usd"] >= 0
            assert out["production_t"] == pytest.approx(sum(c["production_t"] for c in out["crops"].values()))
            assert out["kcal"] == pytest.approx(sum(c["kcal"] for c in out["crops"].values()))
            assert out["value_usd"] == pytest.approx(sum(c["value_usd"] for c in out["crops"].values()))
            assert out["area_ha"] == pytest.approx(r.irrigated_area_ha())
            for c in out["crops"].values():
                assert CROP_KEYS <= set(c)
                assert all(math.isfinite(v) for v in c.values())


def test_riparian_food_production_monotone_and_bounded_in_water(basin):
    for r in basin:
        req = riparian_irrigation_requirement_mm3(r)
        waters = np.linspace(0, 1.5 * req, 40)
        outs = [riparian_food_production(r, float(w)) for w in waters]
        prod = [o["production_t"] for o in outs]
        kcal = [o["kcal"] for o in outs]
        value = [o["value_usd"] for o in outs]
        ratio = [o["et_ratio"] for o in outs]
        for seq in (prod, kcal, value, ratio):
            assert all(b >= a - 1e-9 for a, b in zip(seq, seq[1:]))
        max_prod = sum(c.yield_max_t_ha * c.area_ha for c in r.crops)
        assert max(prod) == pytest.approx(max_prod)
        assert all(p <= max_prod + 1e-6 for p in prod)
        # beyond the requirement production is flat (FAO-33 is linear then saturates)
        saturated = [p for w, p in zip(waters, prod) if w >= req]
        assert all(p == pytest.approx(max_prod) for p in saturated)


def test_riparian_food_production_area_factor(basin):
    highland = basin.riparian("Highland")
    base = riparian_food_production(highland, 798.0)
    big = riparian_food_production(highland, 2 * 798.0, area_factor=2.0)
    assert big["requirement_mm3"] == pytest.approx(2 * base["requirement_mm3"])
    assert big["area_ha"] == pytest.approx(2 * base["area_ha"])
    assert big["production_t"] == pytest.approx(2 * base["production_t"])
    assert big["kcal"] == pytest.approx(2 * base["kcal"])
    assert big["et_ratio"] == pytest.approx(1.0)
    # same water, double the area -> half the relative (gross) supply; ETa/ETm is the
    # same as for half the water on the base area (ratios do not depend on area)
    half = riparian_food_production(highland, 798.0, area_factor=2.0)
    assert half["irrigation_supply_ratio"] == pytest.approx(0.5)
    assert half["et_ratio"] == pytest.approx(riparian_food_production(highland, 399.0)["et_ratio"])
    assert half["et_ratio"] == pytest.approx(799.5 / 999.0, rel=1e-12)
    assert half["etm_mm3"] == pytest.approx(2 * 999.0, rel=1e-12)
    assert half["crops"]["wheat"]["area_ha"] == pytest.approx(300_000.0)
    # area factor 0: nothing to irrigate, nothing produced, no deficit
    none = riparian_food_production(highland, 798.0, area_factor=0.0)
    assert none["requirement_mm3"] == 0.0 and none["supplied_mm3"] == 0.0
    assert none["et_ratio"] == 1.0
    assert none["production_t"] == 0.0 and none["kcal"] == 0.0 and none["area_ha"] == 0.0


def test_riparian_food_production_efficiency_override(basin):
    highland = basin.riparian("Highland")
    base = riparian_food_production(highland, 399.0)             # eff 0.5 -> ratio 0.5
    better = riparian_food_production(highland, 399.0, efficiency=1.0)
    assert better["requirement_mm3"] == pytest.approx(399.0)     # requirement halves
    assert better["et_ratio"] == pytest.approx(1.0)
    assert better["production_t"] > base["production_t"]
    assert better["kcal"] == pytest.approx(3.12e12)
    same = riparian_food_production(highland, 399.0, efficiency=0.5)
    assert same["et_ratio"] == pytest.approx(base["et_ratio"])
    assert same["production_t"] == pytest.approx(base["production_t"])
    effs = np.linspace(0.2, 1.0, 20)
    out = [riparian_food_production(highland, 399.0, efficiency=float(e))["kcal"] for e in effs]
    assert all(b >= a for a, b in zip(out, out[1:]))


def test_riparian_food_production_no_crops():
    r = _riparian([])
    out = riparian_food_production(r, 500.0)
    assert out["crops"] == {}
    assert out["production_t"] == 0.0 and out["kcal"] == 0.0 and out["value_usd"] == 0.0
    assert out["requirement_mm3"] == 0.0 and out["supplied_mm3"] == 0.0
    assert out["deficit_mm3"] == 0.0 and out["area_ha"] == 0.0
    assert out["et_ratio"] == 1.0
    assert out["available_mm3"] == 500.0


def test_riparian_food_production_rainfed_crops_need_no_water():
    r = _riparian([_wheat(), _maize()], et0=1.0, pe=1000.0)
    out = riparian_food_production(r, 0.0)
    assert out["requirement_mm3"] == 0.0 and out["supplied_mm3"] == 0.0
    assert out["et_ratio"] == 1.0
    assert out["crops"]["wheat"]["et_ratio"] == 1.0
    assert out["production_t"] == pytest.approx(600_000.0 + 300_000.0)


def test_riparian_food_production_mixed_rainfed_and_irrigated():
    # maize fully rainfed (season ET 1*0.9*130 = 117 < Pe 120), wheat not (1*0.85*150=127.5 > 120)
    r = _riparian([_wheat(), _maize()], et0=1.0, pe=120.0, eff=0.5)
    req = riparian_crop_requirements_mm3(r)
    assert req["maize"] == 0.0 and req["wheat"] > 0.0
    out = riparian_food_production(r, 0.0)
    assert out["crops"]["maize"]["et_ratio"] == 1.0
    assert out["crops"]["maize"]["rainfall_et_share"] == 1.0
    assert out["crops"]["maize"]["production_t"] == pytest.approx(300_000.0)
    # wheat gets no irrigation but rainfall still covers 120 of its 127.5 mm of ET
    assert out["crops"]["wheat"]["irrigation_supply_ratio"] == 0.0
    assert out["crops"]["wheat"]["et_ratio"] == pytest.approx(120.0 / 127.5, rel=1e-12)
    assert out["crops"]["wheat"]["production_t"] == pytest.approx(
        fao33_yield(4.0, 1.05, 120.0 / 127.5) * 150_000, rel=1e-12)
    # the overall gross ratio describes the irrigated requirement only ...
    assert out["irrigation_supply_ratio"] == 0.0
    # ... while ETa/ETm weights both crops by their maximum ET volume:
    # ETm wheat 127.5 mm * 150 000 ha = 191.25, maize 117 mm * 50 000 ha = 58.5 Mm3
    assert out["etm_mm3"] == pytest.approx(191.25 + 58.5, rel=1e-12)
    assert out["eta_mm3"] == pytest.approx(120.0 * 1.5 + 58.5, rel=1e-12)
    assert out["et_ratio"] == pytest.approx((180.0 + 58.5) / (191.25 + 58.5), rel=1e-12)
    assert 120.0 / 127.5 < out["et_ratio"] < 1.0


def test_riparian_food_production_duplicate_crop_names():
    r = _riparian([_wheat(), _wheat(), Crop("wheat#2", 1000.0, 0.85, 150, 4.0, 1.05, 3400, 250)])
    out = riparian_food_production(r, 1e6)
    assert len(out["crops"]) == 3                              # nothing dropped
    assert len(set(out["crops"])) == 3                          # keys unique
    assert "wheat" in out["crops"] and "wheat#2" in out["crops"]
    assert out["production_t"] == pytest.approx(2 * 600_000.0 + 4.0 * 1000.0)
    assert out["requirement_mm3"] == pytest.approx(riparian_irrigation_requirement_mm3(r))


def test_riparian_food_production_does_not_mutate_inputs(basin):
    snapshot = copy.deepcopy(basin)
    for r in basin:
        out = riparian_food_production(r, 0.5 * r.demand.agricultural, area_factor=1.1, efficiency=0.6)
        # mutating the result must not leak into the basin
        out["crops"].clear()
        out["kcal"] = -1.0
    assert basin == snapshot


@pytest.mark.parametrize("bad", [None, 1.0, "Highland", object()])
def test_riparian_food_production_rejects_non_riparian(bad):
    with pytest.raises(ValueError, match="Riparian"):
        riparian_food_production(bad, 1.0)


@pytest.mark.parametrize("bad", BAD_NUMBERS)
def test_riparian_food_production_rejects_bad_water(basin, bad):
    with pytest.raises(ValueError, match="agricultural_water_mm3"):
        riparian_food_production(basin.riparian("Highland"), bad)


@pytest.mark.parametrize("factor", [-0.1, float("nan"), float("inf"), "x", None])
def test_riparian_food_production_rejects_bad_area_factor(basin, factor):
    with pytest.raises(ValueError, match="area_factor"):
        riparian_food_production(basin.riparian("Highland"), 1.0, area_factor=factor)


@pytest.mark.parametrize("eta", [0.0, -0.5, 1.5, float("nan"), "x"])
def test_riparian_food_production_rejects_bad_efficiency(basin, eta):
    with pytest.raises(ValueError, match="efficiency"):
        riparian_food_production(basin.riparian("Highland"), 1.0, efficiency=eta)


def test_riparian_food_production_rejects_bad_crop_attributes():
    r = _riparian([_wheat(), Crop("bad", 1.0, 0.9, 100, -3.0, 1.0, 1.0)])
    with pytest.raises(ValueError, match="yield_max_t_ha"):
        riparian_food_production(r, 1.0)


def test_riparian_food_production_et_ratio_closed_form_all_riparians(basin):
    """et_ratio = (sum_i min(Pe, ETc_i) area_i + efficiency * supplied) / ETm, all in Mm3."""
    rng = np.random.default_rng(23)
    for r in basin:
        req = riparian_irrigation_requirement_mm3(r)
        etc = {c.name: crop_evapotranspiration_mm(r.et0_mm_day, c.kc, c.season_days) for c in r.crops}
        pe_vol = sum(min(r.effective_rainfall_mm, etc[c.name]) * c.area_ha * M3_PER_HA_MM / M3_PER_MM3
                     for c in r.crops)
        etm = sum(etc[c.name] * c.area_ha * M3_PER_HA_MM / M3_PER_MM3 for c in r.crops)
        for water in [0.0, req, 2 * req] + [float(w) for w in rng.uniform(0, 1.2 * req, 10)]:
            out = riparian_food_production(r, water)
            assert out["etm_mm3"] == pytest.approx(etm, rel=1e-12)
            assert out["eta_mm3"] == pytest.approx(
                pe_vol + r.irrigation_efficiency * out["supplied_mm3"], rel=1e-9)
            assert out["et_ratio"] == pytest.approx(out["eta_mm3"] / out["etm_mm3"], rel=1e-12)
            assert 0.0 <= out["eta_mm3"] <= out["etm_mm3"] + 1e-9
            assert out["irrigation_supply_ratio"] == pytest.approx(min(water, req) / req)
            assert out["et_ratio"] >= out["irrigation_supply_ratio"] - 1e-12
            # per-crop ETa volumes sum to the riparian ETa (conservation)
            eta_crops = sum(c["et_ratio"] * etc[crop.name] * c["area_ha"] * M3_PER_HA_MM / M3_PER_MM3
                            for c, crop in zip(out["crops"].values(), r.crops))
            assert eta_crops == pytest.approx(out["eta_mm3"], rel=1e-12)
            for c in out["crops"].values():
                assert c["irrigation_supply_ratio"] == pytest.approx(out["irrigation_supply_ratio"])


def test_riparian_food_production_efficiency_changes_net_water_per_gross_unit(basin):
    """The same gross water at a higher efficiency puts more net water into the root zone."""
    highland = basin.riparian("Highland")
    effs = np.linspace(0.3, 1.0, 15)
    outs = [riparian_food_production(highland, 300.0, efficiency=float(e)) for e in effs]
    for e, o in zip(effs, outs):
        # ETa = effective-rainfall volume (300 mm on 200 000 ha = 600 Mm3) + efficiency * gross applied
        assert o["eta_mm3"] == pytest.approx(600.0 + float(e) * o["supplied_mm3"], rel=1e-9)
    et = [o["et_ratio"] for o in outs]
    assert all(b >= a - 1e-12 for a, b in zip(et, et[1:]))


def test_riparian_food_production_dry_year_on_mostly_rainfed_crop_is_not_a_total_failure():
    """A crop whose ET is mostly rain-fed loses only its irrigated share in a year without water."""
    # ETc 510 mm, Pe 450 mm: irrigation covers only 60/510 = 12 % of the crop's ET
    r = _riparian([_wheat()], et0=4.0, pe=450.0, eff=0.5)
    req = riparian_irrigation_requirement_mm3(r)
    assert req == pytest.approx(60.0 / 0.5 * 150_000 * 10 / 1e6)   # 180 Mm3
    dry = riparian_food_production(r, 0.0)
    assert dry["irrigation_supply_ratio"] == 0.0
    assert dry["et_ratio"] == pytest.approx(450.0 / 510.0, rel=1e-12)
    assert dry["production_t"] == pytest.approx(fao33_yield(4.0, 1.05, 450.0 / 510.0) * 150_000)
    assert dry["production_t"] > 0.85 * 600_000.0        # ~87.6 % of the full yield
    assert dry["kcal"] > 0.0
    wet = riparian_food_production(r, req)
    assert wet["et_ratio"] == 1.0 and wet["production_t"] == pytest.approx(600_000.0)


def test_riparian_food_production_no_rainfall_reduces_to_gross_supply_ratio():
    r = _riparian([_wheat(), _maize()], et0=4.0, pe=0.0, eff=0.5)
    req = riparian_irrigation_requirement_mm3(r)
    for share in (0.0, 0.3, 0.75, 1.0, 1.5):
        out = riparian_food_production(r, share * req)
        assert out["et_ratio"] == pytest.approx(min(share, 1.0))
        assert out["et_ratio"] == pytest.approx(out["irrigation_supply_ratio"])
        for c in out["crops"].values():
            assert c["rainfall_et_share"] == 0.0
            assert c["et_ratio"] == pytest.approx(min(share, 1.0))


def test_riparian_food_production_new_keys_are_finite_floats(basin):
    for r in basin:
        for water in (0.0, 1000.0, 1e9):
            out = riparian_food_production(r, water)
            for k in ("irrigation_supply_ratio", "etm_mm3", "eta_mm3", "et_ratio"):
                assert isinstance(out[k], float) and math.isfinite(out[k])
            assert 0.0 <= out["irrigation_supply_ratio"] <= 1.0
    none = riparian_food_production(_riparian([]), 10.0)
    assert none["etm_mm3"] == 0.0 and none["eta_mm3"] == 0.0
    assert none["et_ratio"] == 1.0 and none["irrigation_supply_ratio"] == 1.0


# --------------------------------------------------------------------------- #
# Food demand and security indices
# --------------------------------------------------------------------------- #
def test_food_demand_kcal_matches_riparian_method(basin):
    for r in basin:
        assert food_demand_kcal(r.population, r.food_demand_kcal_per_capita_day) == pytest.approx(
            r.food_demand_kcal(), rel=1e-12
        )
    # Highland: 8e6 * 2500 * 365 = 7.3e12 kcal
    assert food_demand_kcal(8_000_000) == pytest.approx(7.3e12, rel=1e-12)
    assert food_demand_kcal(0.0) == 0.0
    assert food_demand_kcal(100.0, 2000.0, days=1.0) == pytest.approx(200_000.0)


@pytest.mark.parametrize("bad", BAD_NUMBERS)
def test_food_demand_rejects_bad_inputs(bad):
    with pytest.raises(ValueError):
        food_demand_kcal(bad)
    with pytest.raises(ValueError):
        food_demand_kcal(1.0, bad)
    with pytest.raises(ValueError):
        food_demand_kcal(1.0, 2500.0, days=bad)


def test_food_self_sufficiency_ratio_not_capped():
    assert food_self_sufficiency(50.0, 100.0) == pytest.approx(0.5)
    assert food_self_sufficiency(100.0, 100.0) == pytest.approx(1.0)
    assert food_self_sufficiency(300.0, 100.0) == pytest.approx(3.0)   # surplus not capped
    assert food_self_sufficiency(0.0, 100.0) == 0.0
    assert food_self_sufficiency(0.0, 0.0) == 1.0                      # no demand -> 1
    assert food_self_sufficiency(10.0, 0.0) == 1.0


def test_food_security_index_capped():
    assert food_security_index(50.0, 100.0) == pytest.approx(0.5)
    assert food_security_index(100.0, 100.0) == 1.0
    assert food_security_index(300.0, 100.0) == 1.0
    assert food_security_index(0.0, 100.0) == 0.0
    assert food_security_index(0.0, 0.0) == 1.0
    rng = np.random.default_rng(15)
    for _ in range(200):
        p, d = float(rng.uniform(0, 1e13)), float(rng.uniform(0, 1e13))
        idx = food_security_index(p, d)
        assert 0.0 <= idx <= 1.0
        assert idx == pytest.approx(min(food_self_sufficiency(p, d), 1.0))
    # monotone non-decreasing in production
    prods = np.linspace(0, 2e12, 30)
    out = [food_security_index(float(p), 1e12) for p in prods]
    assert all(b >= a for a, b in zip(out, out[1:]))


@pytest.mark.parametrize("bad", BAD_NUMBERS)
def test_food_indices_reject_bad_inputs(bad):
    for fn in (food_self_sufficiency, food_security_index):
        with pytest.raises(ValueError):
            fn(bad, 1.0)
        with pytest.raises(ValueError):
            fn(1.0, bad)


def test_highland_self_sufficiency_example(basin):
    """Highland grows 3.12e12 kcal against a 7.3e12 kcal demand -> ~0.427."""
    highland = basin.riparian("Highland")
    out = riparian_food_production(highland, 798.0)
    ssr = food_self_sufficiency(out["kcal"], highland.food_demand_kcal())
    assert ssr == pytest.approx(3.12e12 / 7.3e12, rel=1e-9)
    assert 0.42 < ssr < 0.43
    assert food_security_index(out["kcal"], highland.food_demand_kcal()) == pytest.approx(ssr)


# --------------------------------------------------------------------------- #
# Virtual water and water productivity
# --------------------------------------------------------------------------- #
def test_virtual_water_content_closed_form():
    # Highland: 798 Mm3 for 900 000 t -> 886.7 m3/t
    assert virtual_water_content_m3_per_t(798.0, 900_000.0) == pytest.approx(798e6 / 900_000, rel=1e-12)
    assert virtual_water_content_m3_per_t(1.5, 1000.0) == pytest.approx(1500.0)
    assert virtual_water_content_m3_per_t(0.0, 1000.0) == 0.0          # rainfed: no blue water
    assert virtual_water_content_m3_per_t(10.0, 0.0) == math.inf        # nothing produced
    assert virtual_water_content_m3_per_t(0.0, 0.0) == math.inf         # spec: inf if production 0


def test_virtual_water_content_scaling():
    base = virtual_water_content_m3_per_t(100.0, 50_000.0)
    assert virtual_water_content_m3_per_t(200.0, 50_000.0) == pytest.approx(2 * base)
    assert virtual_water_content_m3_per_t(100.0, 100_000.0) == pytest.approx(base / 2)
    # more water per tonne is a lower productivity: reciprocal relation (kg vs t)
    assert water_productivity_kg_per_m3(50_000.0, 100.0) == pytest.approx(1000.0 / base)


def test_water_productivity_closed_form_and_edges():
    # 600 000 t wheat from 630 Mm3 -> 0.95 kg/m3 (within Zwart & Bastiaanssen's 0.6-1.7 range)
    assert water_productivity_kg_per_m3(600_000.0, 630.0) == pytest.approx(600_000 * 1000 / 630e6, rel=1e-12)
    assert 0.6 < water_productivity_kg_per_m3(600_000.0, 630.0) < 1.7
    assert water_productivity_kg_per_m3(0.0, 100.0) == 0.0
    assert water_productivity_kg_per_m3(0.0, 0.0) == 0.0
    assert water_productivity_kg_per_m3(10.0, 0.0) == math.inf


@pytest.mark.parametrize("bad", BAD_NUMBERS)
def test_virtual_water_content_and_productivity_reject_bad_inputs(bad):
    with pytest.raises(ValueError):
        virtual_water_content_m3_per_t(bad, 1.0)
    with pytest.raises(ValueError):
        virtual_water_content_m3_per_t(1.0, bad)
    with pytest.raises(ValueError):
        water_productivity_kg_per_m3(bad, 1.0)
    with pytest.raises(ValueError):
        water_productivity_kg_per_m3(1.0, bad)


def test_virtual_water_import_closed_form():
    """3.4e12 kcal of wheat = 1e6 t; at 1500 m3/t that is 1.5e9 m3 = 1500 Mm3."""
    assert virtual_water_import_mm3(3.4e12) == pytest.approx(1500.0, rel=1e-12)
    assert virtual_water_import_mm3(3.4e12, kcal_per_kg=3400.0, m3_per_t=1500.0) == pytest.approx(1500.0)
    # rice at 3600 kcal/kg and 1670 m3/t (Mekonnen & Hoekstra 2011)
    expected = 1e9 / (3600.0 * 1000.0) * 1670.0 / 1e6
    assert virtual_water_import_mm3(1e9, 3600.0, 1670.0) == pytest.approx(expected, rel=1e-12)
    # linear in deficit and in m3/t, inverse in kcal/kg
    base = virtual_water_import_mm3(1e12)
    assert virtual_water_import_mm3(2e12) == pytest.approx(2 * base)
    assert virtual_water_import_mm3(1e12, m3_per_t=3000.0) == pytest.approx(2 * base)
    assert virtual_water_import_mm3(1e12, kcal_per_kg=6800.0) == pytest.approx(base / 2)


def test_virtual_water_import_zero_and_surplus():
    assert virtual_water_import_mm3(0.0) == 0.0
    assert virtual_water_import_mm3(-1e12) == 0.0        # surplus: nothing to import
    assert virtual_water_import_mm3(1e12, m3_per_t=0.0) == 0.0
    deficits = np.linspace(-1e12, 1e13, 40)
    out = [virtual_water_import_mm3(float(d)) for d in deficits]
    assert all(b >= a for a, b in zip(out, out[1:]))
    assert all(o >= 0.0 for o in out)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "x", None, True])
def test_virtual_water_import_rejects_non_numeric_deficit(bad):
    with pytest.raises(ValueError, match="kcal_deficit"):
        virtual_water_import_mm3(bad)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), "x", None])
def test_virtual_water_import_rejects_bad_kcal_per_kg(bad):
    with pytest.raises(ValueError, match="kcal_per_kg"):
        virtual_water_import_mm3(1e12, kcal_per_kg=bad)


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), "x", None])
def test_virtual_water_import_rejects_bad_m3_per_t(bad):
    with pytest.raises(ValueError, match="m3_per_t"):
        virtual_water_import_mm3(1e12, m3_per_t=bad)


def test_virtual_water_round_trip_closes_the_deficit(basin):
    """Importing the deficit at the riparian's own VWC costs the water it would have needed."""
    highland = basin.riparian("Highland")
    out = riparian_food_production(highland, 798.0)
    demand = highland.food_demand_kcal()
    deficit = demand - out["kcal"]
    assert deficit > 0
    # importing the missing calories as the riparian's own crop mix at its own VWC
    kcal_per_kg_mix = out["kcal"] / (out["production_t"] * 1000.0)
    vwc = virtual_water_content_m3_per_t(out["supplied_mm3"], out["production_t"])
    imported = virtual_water_import_mm3(deficit, kcal_per_kg_mix, vwc)
    # water to grow the deficit domestically = requirement scaled by deficit share
    assert imported == pytest.approx(798.0 * deficit / out["kcal"], rel=1e-9)
    # with the default cereal parameters the figure is of the same order
    assert 0.5 < virtual_water_import_mm3(deficit) / imported < 2.5


# --------------------------------------------------------------------------- #
# Energy for agriculture
# --------------------------------------------------------------------------- #
def test_energy_for_agriculture_closed_form():
    assert energy_for_agriculture_gwh(1_000_000.0) == pytest.approx(150.0, rel=1e-12)
    assert energy_for_agriculture_gwh(1_000_000.0, 150.0) == pytest.approx(150.0)
    assert energy_for_agriculture_gwh(200_000.0) == pytest.approx(30.0)      # Highland area
    assert energy_for_agriculture_gwh(200_000.0, 300.0) == pytest.approx(60.0)
    assert energy_for_agriculture_gwh(0.0) == 0.0
    assert energy_for_agriculture_gwh(1e6, 0.0) == 0.0
    rng = np.random.default_rng(16)
    for _ in range(50):
        a, k = float(rng.uniform(0, 1e7)), float(rng.uniform(0, 1000))
        assert energy_for_agriculture_gwh(a, k) == pytest.approx(a * k / KWH_PER_GWH, rel=1e-12)


def test_energy_for_agriculture_example_basin(basin):
    for r in basin:
        got = energy_for_agriculture_gwh(r.irrigated_area_ha())
        assert got == pytest.approx(r.irrigated_area_ha() * 150.0 / 1e6)
        # farm energy is a small share of the riparian's electricity demand
        assert got < 0.05 * r.energy.demand_gwh


@pytest.mark.parametrize("bad", BAD_NUMBERS)
def test_energy_for_agriculture_rejects_bad_inputs(bad):
    with pytest.raises(ValueError):
        energy_for_agriculture_gwh(bad)
    with pytest.raises(ValueError):
        energy_for_agriculture_gwh(1.0, bad)


# --------------------------------------------------------------------------- #
# Integration: the chain the nexus model will run per riparian-year
# --------------------------------------------------------------------------- #
def test_nexus_style_chain_for_each_riparian(basin):
    for r in basin:
        requirement = riparian_irrigation_requirement_mm3(r)
        vwi_by_share = []
        # the routed agricultural withdrawal could be anything up to the demand
        for share in (0.0, 0.4, 0.8, 1.0):
            water = share * requirement
            out = riparian_food_production(r, water)
            demand = food_demand_kcal(r.population, r.food_demand_kcal_per_capita_day)
            ssr = food_self_sufficiency(out["kcal"], demand)
            fsi = food_security_index(out["kcal"], demand)
            vwc = virtual_water_content_m3_per_t(out["supplied_mm3"], out["production_t"])
            vwi = virtual_water_import_mm3(demand - out["kcal"])
            farm_energy = energy_for_agriculture_gwh(out["area_ha"])
            assert out["irrigation_supply_ratio"] == pytest.approx(share)
            # ETa/ETm is at least the gross supply ratio (rainfall adds to crop ET) ...
            assert out["et_ratio"] >= share - 1e-12
            assert out["et_ratio"] <= 1.0
            # ... and exceeds it strictly whenever the riparian has effective rainfall
            if r.effective_rainfall_mm > 0 and share < 1.0:
                assert out["et_ratio"] > share
            assert 0.0 <= fsi <= 1.0 and fsi == pytest.approx(min(ssr, 1.0))
            assert vwc >= 0.0
            assert vwi >= 0.0 and farm_energy >= 0.0
            if out["production_t"] > 0:
                assert math.isfinite(vwc)
            # closed form with the module defaults (Allan 1998; Hoekstra & Chapagain
            # 2008): unmet kcal as tonnes of wheat at 3 400 kcal/kg, 1 500 m3/t, in Mm3
            kcal_deficit = demand - out["kcal"]
            if kcal_deficit > 0.0:
                assert vwi == pytest.approx(kcal_deficit / (3400.0 * 1000.0) * 1500.0 / 1e6, rel=1e-12)
                assert vwi > 0.0
            else:
                assert vwi == 0.0
            vwi_by_share.append(vwi)
        # more water -> more food produced -> less virtual water import needed
        assert all(later <= earlier + 1e-9 for earlier, later in zip(vwi_by_share, vwi_by_share[1:]))
        if requirement > 0.0 and vwi_by_share[0] > 0.0:
            assert vwi_by_share[-1] < vwi_by_share[0]
