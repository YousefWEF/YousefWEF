"""Tests for :mod:`wefnexus.energy` (water-energy links).

Covers closed-form literature values, physical bounds and conservation,
monotonicity, edge cases (zero flow / zero demand / missing sectors or
plants), input validation and no-mutation guarantees.
"""
import copy
import math

import numpy as np
import pytest

from wefnexus import energy
from wefnexus.energy import (
    HOURS_PER_YEAR,
    HYDRO_G,
    J_PER_GWH,
    J_PER_KWH,
    WATER_DENSITY,
    desalination_energy_gwh,
    energy_balance,
    energy_for_water,
    energy_security_index,
    hydropower_gwh,
    pumping_energy_gwh,
    specific_hydropower_kwh_per_m3,
    specific_pumping_energy_kwh_per_m3,
    thermal_cooling_water_mm3,
    thermal_generation_gwh,
    treatment_energy_gwh,
    water_for_energy,
)
from wefnexus.models import EnergySystem, Riparian, Sector, WaterDemand

# Reference value from the spec: 1 Mm3 through 100 m at eta = 1.
#   1e6 m3 * 1000 kg/m3 * 9.81 m/s2 * 100 m / 3.6e12 J/GWh = 0.2725 GWh
ONE_MM3_100M_GWH = 1e6 * 9.81 * 100.0 / 3.6e9

BAD_NUMBERS = [-1.0, -1e-9, float("nan"), float("inf"), -float("inf"), "abc", None, True]


def _plain_riparian(**energy_kwargs) -> Riparian:
    """Minimal riparian with no crops/reservoir and explicit energy parameters."""
    return Riparian(
        name="Plain",
        population=1_000,
        gdp_usd=1e6,
        local_inflow_mm3=10.0,
        demand=WaterDemand(municipal=1.0, industrial=2.0, agricultural=10.0, energy=0.5),
        energy=EnergySystem(**energy_kwargs),
    )


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
def test_constants():
    assert HYDRO_G == 9.81
    assert WATER_DENSITY == 1000.0
    assert J_PER_KWH == 3.6e6
    assert J_PER_GWH == 3.6e12
    assert HOURS_PER_YEAR == 8760.0
    # every public name in __all__ exists on the module
    for name in energy.__all__:
        assert hasattr(energy, name), name


# --------------------------------------------------------------------------- #
# Hydropower
# --------------------------------------------------------------------------- #
def test_hydropower_closed_form_spec_example():
    """1 Mm3 through 100 m at eta = 1 -> 0.2725 GWh (0.2725 kWh/m3)."""
    assert ONE_MM3_100M_GWH == pytest.approx(0.2725, rel=1e-12)
    assert hydropower_gwh(1.0, 100.0, efficiency=1.0) == pytest.approx(0.2725, rel=1e-12)
    # equivalently via the explicit rho g V H eta / J_per_GWh form
    expected = WATER_DENSITY * HYDRO_G * 1e6 * 100.0 * 1.0 / J_PER_GWH
    assert hydropower_gwh(1.0, 100.0, efficiency=1.0) == pytest.approx(expected, rel=1e-12)


def test_hydropower_default_efficiency_is_0_9():
    assert hydropower_gwh(1.0, 100.0) == pytest.approx(0.2725 * 0.9, rel=1e-12)
    assert hydropower_gwh(1.0, 100.0) == hydropower_gwh(1.0, 100.0, efficiency=0.9)


def test_hydropower_linear_in_volume_head_and_efficiency():
    base = hydropower_gwh(1.0, 100.0, efficiency=1.0)
    assert hydropower_gwh(2.0, 100.0, efficiency=1.0) == pytest.approx(2 * base)
    assert hydropower_gwh(1.0, 200.0, efficiency=1.0) == pytest.approx(2 * base)
    assert hydropower_gwh(1.0, 100.0, efficiency=0.5) == pytest.approx(0.5 * base)
    assert hydropower_gwh(3.0, 50.0, efficiency=0.8) == pytest.approx(3 * 0.5 * 0.8 * base)


def test_hydropower_specific_yield_literature_value():
    """Specific hydropower yield: 0.002725 kWh/m3 per metre of head at eta = 1."""
    assert specific_hydropower_kwh_per_m3(1.0, efficiency=1.0) == pytest.approx(0.002725, rel=1e-12)
    assert specific_hydropower_kwh_per_m3(100.0, efficiency=1.0) == pytest.approx(0.2725, rel=1e-12)
    # consistency between the specific and the volumetric function
    v, h, eta = 37.5, 83.0, 0.87
    assert hydropower_gwh(v, h, eta) == pytest.approx(
        specific_hydropower_kwh_per_m3(h, eta) * v * 1e6 / 1e6, rel=1e-12
    )


def test_hydropower_capacity_cap():
    uncapped = hydropower_gwh(1000.0, 100.0, efficiency=1.0)
    assert uncapped == pytest.approx(272.5)
    # 10 MW * 8760 h / 1000 = 87.6 GWh cap binds
    assert hydropower_gwh(1000.0, 100.0, efficiency=1.0, capacity_mw=10.0) == pytest.approx(87.6)
    # generous capacity does not bind
    assert hydropower_gwh(1000.0, 100.0, efficiency=1.0, capacity_mw=1e6) == pytest.approx(uncapped)
    # exactly at the boundary: cap == uncapped
    cap_mw = uncapped * 1000.0 / 8760.0
    assert hydropower_gwh(1000.0, 100.0, efficiency=1.0, capacity_mw=cap_mw) == pytest.approx(uncapped)
    # custom hours: 10 MW for 100 h -> 1 GWh
    assert hydropower_gwh(1000.0, 100.0, efficiency=1.0, capacity_mw=10.0, hours=100.0) == pytest.approx(1.0)
    # capacity 0 (no plant) -> no generation at all
    assert hydropower_gwh(1000.0, 100.0, capacity_mw=0.0) == 0.0
    # hours 0 with a capacity -> cap of 0
    assert hydropower_gwh(1000.0, 100.0, capacity_mw=10.0, hours=0.0) == 0.0
    # None cap == no cap
    assert hydropower_gwh(1000.0, 100.0, capacity_mw=None) == hydropower_gwh(1000.0, 100.0)


def test_hydropower_capped_is_min_of_uncapped_and_cap():
    rng = np.random.default_rng(1)
    for _ in range(200):
        v = float(rng.uniform(0, 20_000))
        h = float(rng.uniform(0, 300))
        eta = float(rng.uniform(0.5, 1.0))
        cap_mw = float(rng.uniform(0, 3000))
        hrs = float(rng.uniform(0, 8760))
        got = hydropower_gwh(v, h, eta, capacity_mw=cap_mw, hours=hrs)
        unc = hydropower_gwh(v, h, eta)
        assert got == pytest.approx(min(unc, cap_mw * hrs / 1000.0))
        assert 0.0 <= got <= unc + 1e-12


def test_hydropower_zero_cases():
    assert hydropower_gwh(0.0, 100.0) == 0.0
    assert hydropower_gwh(100.0, 0.0) == 0.0
    assert hydropower_gwh(0.0, 0.0, capacity_mw=0.0) == 0.0


def test_hydropower_monotonic_in_volume_and_head():
    vols = np.sort(np.random.default_rng(0).uniform(0, 1000, 50))
    out = [hydropower_gwh(float(v), 80.0) for v in vols]
    assert all(b >= a for a, b in zip(out, out[1:]))
    heads = np.sort(np.random.default_rng(2).uniform(0, 500, 50))
    out = [hydropower_gwh(10.0, float(h)) for h in heads]
    assert all(b >= a for a, b in zip(out, out[1:]))
    # with a cap it is still non-decreasing and bounded
    out = [hydropower_gwh(float(v), 80.0, capacity_mw=5.0) for v in vols]
    assert all(b >= a for a, b in zip(out, out[1:]))
    assert max(out) <= 5.0 * 8760 / 1000 + 1e-12


def test_hydropower_accepts_numpy_and_int_scalars():
    assert hydropower_gwh(np.float64(1.0), np.int64(100), efficiency=np.float32(1.0)) == pytest.approx(0.2725, rel=1e-6)
    assert hydropower_gwh(1, 100, 1) == pytest.approx(0.2725)
    assert isinstance(hydropower_gwh(1, 100, 1), float)


@pytest.mark.parametrize("bad", BAD_NUMBERS)
def test_hydropower_rejects_bad_volume(bad):
    with pytest.raises(ValueError):
        hydropower_gwh(bad, 100.0)


@pytest.mark.parametrize("bad", BAD_NUMBERS)
def test_hydropower_rejects_bad_head(bad):
    with pytest.raises(ValueError):
        hydropower_gwh(1.0, bad)


@pytest.mark.parametrize("eta", [0.0, -0.1, 1.0001, 2.0, float("nan"), "x", None])
def test_hydropower_rejects_bad_efficiency(eta):
    with pytest.raises(ValueError):
        hydropower_gwh(1.0, 100.0, efficiency=eta)


@pytest.mark.parametrize("cap", [-1.0, float("nan"), "big"])
def test_hydropower_rejects_bad_capacity(cap):
    with pytest.raises(ValueError):
        hydropower_gwh(1.0, 100.0, capacity_mw=cap)


@pytest.mark.parametrize("hrs", [-1.0, float("inf"), "year"])
def test_hydropower_rejects_bad_hours(hrs):
    with pytest.raises(ValueError):
        hydropower_gwh(1.0, 100.0, hours=hrs)


def test_hydropower_error_messages_name_the_argument():
    with pytest.raises(ValueError, match="volume_mm3"):
        hydropower_gwh(-1.0, 100.0)
    with pytest.raises(ValueError, match="head_m"):
        hydropower_gwh(1.0, -100.0)
    with pytest.raises(ValueError, match="efficiency"):
        hydropower_gwh(1.0, 100.0, efficiency=0.0)
    with pytest.raises(ValueError, match="capacity_mw"):
        hydropower_gwh(1.0, 100.0, capacity_mw=-5.0)


# --------------------------------------------------------------------------- #
# Pumping
# --------------------------------------------------------------------------- #
def test_pumping_closed_form():
    """E_kWh = rho g V H / (eta 3.6e6); 1 Mm3 up 30 m at eta 0.6 = 136 250 kWh."""
    e_kwh = WATER_DENSITY * HYDRO_G * 1e6 * 30.0 / (0.6 * J_PER_KWH)
    assert e_kwh == pytest.approx(136_250.0, rel=1e-12)
    assert pumping_energy_gwh(1.0, 30.0, efficiency=0.6) == pytest.approx(e_kwh / 1e6, rel=1e-12)
    assert pumping_energy_gwh(1.0, 30.0) == pytest.approx(0.13625, rel=1e-12)  # default eta 0.6


def test_pumping_mirrors_hydropower_at_unit_efficiency():
    for v, h in [(1.0, 100.0), (250.0, 12.5), (0.3, 999.0)]:
        assert pumping_energy_gwh(v, h, efficiency=1.0) == pytest.approx(hydropower_gwh(v, h, efficiency=1.0), rel=1e-12)


def test_pumped_storage_round_trip_efficiency():
    """Hydropower recovered = pumping input * eta_pump * eta_turbine."""
    v, h, eta_p, eta_t = 500.0, 300.0, 0.85, 0.9
    pumped = pumping_energy_gwh(v, h, efficiency=eta_p)
    recovered = hydropower_gwh(v, h, efficiency=eta_t)
    assert recovered == pytest.approx(pumped * eta_p * eta_t, rel=1e-12)
    assert recovered < pumped


def test_pumping_specific_energy_literature_value():
    """0.002725 kWh per m3 per metre of lift at 100 % efficiency (Plappally & Lienhard 2012)."""
    assert specific_pumping_energy_kwh_per_m3(1.0, efficiency=1.0) == pytest.approx(0.002725, rel=1e-12)
    # ~0.0045 kWh/m3/m at 60 % wire-to-water efficiency
    assert specific_pumping_energy_kwh_per_m3(1.0) == pytest.approx(0.002725 / 0.6, rel=1e-12)
    v, h, eta = 12.0, 55.0, 0.7
    assert pumping_energy_gwh(v, h, eta) == pytest.approx(specific_pumping_energy_kwh_per_m3(h, eta) * v, rel=1e-12)


def test_pumping_scaling_and_zero():
    base = pumping_energy_gwh(1.0, 10.0, 0.5)
    assert pumping_energy_gwh(2.0, 10.0, 0.5) == pytest.approx(2 * base)
    assert pumping_energy_gwh(1.0, 20.0, 0.5) == pytest.approx(2 * base)
    assert pumping_energy_gwh(1.0, 10.0, 0.25) == pytest.approx(2 * base)  # lower efficiency -> more energy
    assert pumping_energy_gwh(0.0, 10.0) == 0.0
    assert pumping_energy_gwh(10.0, 0.0) == 0.0


def test_pumping_monotone_decreasing_in_efficiency():
    etas = np.linspace(0.05, 1.0, 40)
    out = [pumping_energy_gwh(5.0, 40.0, float(e)) for e in etas]
    assert all(b <= a for a, b in zip(out, out[1:]))


@pytest.mark.parametrize("bad", BAD_NUMBERS)
def test_pumping_rejects_bad_volume(bad):
    with pytest.raises(ValueError):
        pumping_energy_gwh(bad, 10.0)


@pytest.mark.parametrize("bad", BAD_NUMBERS)
def test_pumping_rejects_bad_lift(bad):
    with pytest.raises(ValueError):
        pumping_energy_gwh(1.0, bad)


@pytest.mark.parametrize("eta", [0.0, -0.5, 1.5, float("nan")])
def test_pumping_rejects_bad_efficiency(eta):
    with pytest.raises(ValueError, match="efficiency"):
        pumping_energy_gwh(1.0, 10.0, efficiency=eta)


@pytest.mark.parametrize("eta", [0.0, -0.5, 1.5])
def test_specific_functions_reject_bad_efficiency(eta):
    with pytest.raises(ValueError):
        specific_pumping_energy_kwh_per_m3(10.0, efficiency=eta)
    with pytest.raises(ValueError):
        specific_hydropower_kwh_per_m3(10.0, efficiency=eta)
    with pytest.raises(ValueError):
        specific_pumping_energy_kwh_per_m3(-10.0)
    with pytest.raises(ValueError):
        specific_hydropower_kwh_per_m3(-10.0)


# --------------------------------------------------------------------------- #
# Desalination and treatment
# --------------------------------------------------------------------------- #
def test_desalination_delta_example():
    """Delta: 300 Mm3 at 3.5 kWh/m3 = 300e6 m3 * 3.5 kWh/m3 = 1.05e9 kWh = 1050 GWh."""
    assert 300e6 * 3.5 / 1e6 == pytest.approx(1050.0)
    assert desalination_energy_gwh(300.0, 3.5) == pytest.approx(1050.0, rel=1e-12)
    assert desalination_energy_gwh(300.0) == pytest.approx(1050.0, rel=1e-12)  # default 3.5


def test_desalination_and_treatment_are_volume_times_intensity():
    rng = np.random.default_rng(3)
    for _ in range(50):
        v = float(rng.uniform(0, 500))
        k = float(rng.uniform(0, 20))
        assert desalination_energy_gwh(v, k) == pytest.approx(v * k, rel=1e-12)
        assert treatment_energy_gwh(v, k) == pytest.approx(v * k, rel=1e-12)
    assert treatment_energy_gwh(1.0, 0.4) == pytest.approx(0.4)
    assert treatment_energy_gwh(2.5, 0.6) == pytest.approx(1.5)


def test_desalination_and_treatment_zero_cases():
    assert desalination_energy_gwh(0.0) == 0.0
    assert desalination_energy_gwh(10.0, 0.0) == 0.0
    assert treatment_energy_gwh(0.0, 0.4) == 0.0
    assert treatment_energy_gwh(5.0, 0.0) == 0.0


@pytest.mark.parametrize("bad", BAD_NUMBERS)
def test_desalination_rejects_bad_inputs(bad):
    with pytest.raises(ValueError):
        desalination_energy_gwh(bad)
    with pytest.raises(ValueError):
        desalination_energy_gwh(1.0, bad)


@pytest.mark.parametrize("bad", BAD_NUMBERS)
def test_treatment_rejects_bad_inputs(bad):
    with pytest.raises(ValueError):
        treatment_energy_gwh(bad, 0.4)
    with pytest.raises(ValueError):
        treatment_energy_gwh(1.0, bad)


def test_treatment_requires_intensity_argument():
    with pytest.raises(TypeError):
        treatment_energy_gwh(1.0)  # type: ignore[call-arg]


# --------------------------------------------------------------------------- #
# Thermal generation and cooling water
# --------------------------------------------------------------------------- #
def test_thermal_generation_closed_form():
    assert thermal_generation_gwh(1000.0, 0.5) == pytest.approx(4380.0)
    assert thermal_generation_gwh(1000.0, 1.0) == pytest.approx(8760.0)
    assert thermal_generation_gwh(1000.0, 0.0) == 0.0
    assert thermal_generation_gwh(0.0, 0.9) == 0.0
    assert thermal_generation_gwh(100.0, 0.5, hours=100.0) == pytest.approx(5.0)
    assert thermal_generation_gwh(100.0, 0.5, hours=0.0) == 0.0


def test_thermal_generation_bounded_by_nameplate():
    rng = np.random.default_rng(4)
    for _ in range(100):
        cap = float(rng.uniform(0, 10_000))
        cf = float(rng.uniform(0, 1))
        g = thermal_generation_gwh(cap, cf)
        assert 0.0 <= g <= cap * 8760 / 1000 + 1e-9
        assert g == pytest.approx(cap * cf * 8.76)


@pytest.mark.parametrize("cf", [-0.01, 1.01, float("nan"), "half"])
def test_thermal_generation_rejects_bad_capacity_factor(cf):
    with pytest.raises(ValueError, match="capacity_factor"):
        thermal_generation_gwh(100.0, cf)


@pytest.mark.parametrize("bad", BAD_NUMBERS)
def test_thermal_generation_rejects_bad_capacity(bad):
    with pytest.raises(ValueError):
        thermal_generation_gwh(bad, 0.5)


def test_thermal_cooling_water_closed_form():
    # 1000 GWh = 1e6 MWh; * 1.5 m3/MWh = 1.5e6 m3 = 1.5 Mm3
    assert thermal_cooling_water_mm3(1000.0, 1.5) == pytest.approx(1.5, rel=1e-12)
    assert thermal_cooling_water_mm3(0.0, 1.5) == 0.0
    assert thermal_cooling_water_mm3(1000.0, 0.0) == 0.0  # dry cooling
    rng = np.random.default_rng(5)
    for _ in range(50):
        g = float(rng.uniform(0, 1e5))
        i = float(rng.uniform(0, 5))
        assert thermal_cooling_water_mm3(g, i) == pytest.approx(g * i / 1000.0, rel=1e-12)


@pytest.mark.parametrize("bad", BAD_NUMBERS)
def test_thermal_cooling_water_rejects_bad_inputs(bad):
    with pytest.raises(ValueError):
        thermal_cooling_water_mm3(bad, 1.5)
    with pytest.raises(ValueError):
        thermal_cooling_water_mm3(1.0, bad)


def test_water_for_energy_midland(basin):
    midland = basin.riparian("Midland")
    before = copy.deepcopy(midland)
    gen = thermal_generation_gwh(midland.energy.thermal_capacity_mw, midland.energy.thermal_capacity_factor)
    assert gen == pytest.approx(13_140.0)  # 3000 MW * 0.5 * 8760 / 1000
    water = water_for_energy(midland, gen)
    assert water == pytest.approx(19.71, rel=1e-12)  # 13 140 GWh * 1.5 m3/MWh
    assert water == pytest.approx(thermal_cooling_water_mm3(gen, midland.energy.thermal_water_intensity_m3_per_mwh))
    assert water_for_energy(midland, 0.0) == 0.0
    assert midland == before  # no mutation


def test_water_for_energy_uses_riparian_intensity_and_validates():
    r = _plain_riparian(thermal_water_intensity_m3_per_mwh=2.0)
    assert water_for_energy(r, 500.0) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        water_for_energy(r, -1.0)
    with pytest.raises(ValueError, match="Riparian"):
        water_for_energy("Midland", 1.0)  # type: ignore[arg-type]
    bad = _plain_riparian(thermal_water_intensity_m3_per_mwh=-2.0)
    with pytest.raises(ValueError):
        water_for_energy(bad, 500.0)


# --------------------------------------------------------------------------- #
# energy_for_water
# --------------------------------------------------------------------------- #
def test_energy_for_water_delta_example(basin):
    """Hand-computed components for Delta with full demand withdrawals and 300 Mm3 desalination."""
    delta = basin.riparian("Delta")
    before = copy.deepcopy(delta)
    withdrawals = delta.demand.withdrawals()
    w_before = dict(withdrawals)
    out = energy_for_water(delta, withdrawals, desalinated_mm3=300.0)

    # pumping: 13 000 Mm3 * 0.35 = 4 550 Mm3 lifted 40 m at eta 0.6
    pumping = 4_550e6 * WATER_DENSITY * HYDRO_G * 40.0 / (0.6 * J_PER_KWH) / 1e6
    assert pumping == pytest.approx(826.5833333, rel=1e-9)
    assert out["pumping"] == pytest.approx(pumping, rel=1e-12)
    # desalination: 300e6 m3 * 3.5 kWh/m3 = 1.05e9 kWh = 1050 GWh (NOT 1.05 GWh)
    assert out["desalination"] == pytest.approx(1050.0, rel=1e-12)
    # treatment: (3500 + 1500) Mm3 * 0.4 kWh/m3 = 2000 GWh
    assert out["treatment"] == pytest.approx(2000.0, rel=1e-12)
    # wastewater: return flow (3500*0.8 + 1500*0.9) = 4150 Mm3 * 0.6 kWh/m3 = 2490 GWh
    assert out["wastewater"] == pytest.approx(2490.0, rel=1e-12)
    assert out["total"] == pytest.approx(pumping + 1050.0 + 2000.0 + 2490.0, rel=1e-12)
    assert out["total"] == pytest.approx(6366.5833333, rel=1e-9)
    # energy for water is a few percent of Delta's 160 000 GWh demand - sanity on magnitude
    assert 0.02 < out["total"] / delta.energy.demand_gwh < 0.08

    assert set(out) == {"pumping", "desalination", "treatment", "wastewater", "total"}
    assert all(isinstance(v, float) for v in out.values())
    assert delta == before
    assert withdrawals == w_before


def test_energy_for_water_highland_and_midland(basin):
    """Every riparian: components match the helper functions on the same inputs."""
    for r in basin:
        w = r.demand.withdrawals()
        desal = r.energy.desalination_capacity_mm3
        out = energy_for_water(r, w, desal)
        es = r.energy
        cf = r.demand.consumption_fraction
        assert out["pumping"] == pytest.approx(
            pumping_energy_gwh(w[Sector.AGRICULTURAL] * es.pumped_fraction, es.pumping_lift_m, es.pumping_efficiency)
        )
        assert out["desalination"] == pytest.approx(desalination_energy_gwh(desal, es.desal_energy_kwh_m3))
        mi = w[Sector.MUNICIPAL] + w[Sector.INDUSTRIAL]
        assert out["treatment"] == pytest.approx(treatment_energy_gwh(mi, es.treatment_energy_kwh_m3))
        rf = w[Sector.MUNICIPAL] * (1 - cf[Sector.MUNICIPAL]) + w[Sector.INDUSTRIAL] * (1 - cf[Sector.INDUSTRIAL])
        assert out["wastewater"] == pytest.approx(treatment_energy_gwh(rf, es.wastewater_energy_kwh_m3))
        assert out["total"] == pytest.approx(sum(out[k] for k in ("pumping", "desalination", "treatment", "wastewater")))
        assert all(v >= 0.0 for v in out.values())
    # Highland and Midland have no desalination
    assert energy_for_water(basin.riparian("Highland"), basin.riparian("Highland").demand.withdrawals(), 0.0)["desalination"] == 0.0


def test_energy_for_water_pumped_fraction_override(basin):
    delta = basin.riparian("Delta")
    w = delta.demand.withdrawals()
    default = energy_for_water(delta, w, 0.0)
    assert default["pumping"] == pytest.approx(energy_for_water(delta, w, 0.0, pumped_fraction=0.35)["pumping"])
    none = energy_for_water(delta, w, 0.0, pumped_fraction=0.0)
    full = energy_for_water(delta, w, 0.0, pumped_fraction=1.0)
    assert none["pumping"] == 0.0
    assert full["pumping"] == pytest.approx(default["pumping"] / 0.35, rel=1e-12)
    assert full["pumping"] == pytest.approx(pumping_energy_gwh(13_000.0, 40.0, 0.6))
    # other components are unaffected by the pumped fraction
    for k in ("desalination", "treatment", "wastewater"):
        assert none[k] == default[k] == full[k]
    # pumping is monotone in the pumped fraction
    pf = np.linspace(0, 1, 11)
    p = [energy_for_water(delta, w, 0.0, pumped_fraction=float(x))["pumping"] for x in pf]
    assert all(b >= a for a, b in zip(p, p[1:]))


def test_energy_for_water_missing_sectors_and_empty_dict(basin):
    delta = basin.riparian("Delta")
    # only municipal -> pumping 0, treatment & wastewater on municipal only
    out = energy_for_water(delta, {Sector.MUNICIPAL: 100.0}, 0.0)
    assert out["pumping"] == 0.0
    assert out["desalination"] == 0.0
    assert out["treatment"] == pytest.approx(100.0 * 0.4)
    assert out["wastewater"] == pytest.approx(100.0 * 0.8 * 0.6)
    # empty dict: only desalination
    out = energy_for_water(delta, {}, 10.0)
    assert out["pumping"] == out["treatment"] == out["wastewater"] == 0.0
    assert out["desalination"] == pytest.approx(35.0)
    assert out["total"] == pytest.approx(35.0)
    # None behaves like an empty dict
    assert energy_for_water(delta, None, 10.0) == out  # type: ignore[arg-type]
    # all zero
    out = energy_for_water(delta, {s: 0.0 for s in Sector if s is not Sector.ENVIRONMENT}, 0.0)
    assert out == {"pumping": 0.0, "desalination": 0.0, "treatment": 0.0, "wastewater": 0.0, "total": 0.0}


def test_energy_for_water_accepts_string_keys_and_ignores_environment(basin):
    delta = basin.riparian("Delta")
    w_enum = {Sector.MUNICIPAL: 100.0, Sector.AGRICULTURAL: 1000.0}
    w_str = {"municipal": 100.0, "agricultural": 1000.0}
    assert energy_for_water(delta, w_str, 1.0) == energy_for_water(delta, w_enum, 1.0)
    # environmental flow is an in-stream requirement, not a withdrawal: ignored
    w_env = dict(w_enum)
    w_env[Sector.ENVIRONMENT] = 5_000.0
    assert energy_for_water(delta, w_env, 1.0) == energy_for_water(delta, w_enum, 1.0)
    # energy sector withdrawals need no pumping/treatment in this accounting
    w_energy = dict(w_enum)
    w_energy[Sector.ENERGY] = 600.0
    assert energy_for_water(delta, w_energy, 1.0) == energy_for_water(delta, w_enum, 1.0)


def test_energy_for_water_wastewater_uses_riparian_consumption_fractions():
    r = _plain_riparian(treatment_energy_kwh_m3=1.0, wastewater_energy_kwh_m3=1.0, pumped_fraction=0.0)
    r.demand.consumption_fraction[Sector.MUNICIPAL] = 0.5
    r.demand.consumption_fraction[Sector.INDUSTRIAL] = 0.0
    out = energy_for_water(r, {Sector.MUNICIPAL: 10.0, Sector.INDUSTRIAL: 4.0}, 0.0)
    assert out["treatment"] == pytest.approx(14.0)
    assert out["wastewater"] == pytest.approx(10.0 * 0.5 + 4.0 * 1.0)
    # fully consumed -> no wastewater
    r.demand.consumption_fraction[Sector.MUNICIPAL] = 1.0
    r.demand.consumption_fraction[Sector.INDUSTRIAL] = 1.0
    assert energy_for_water(r, {Sector.MUNICIPAL: 10.0, Sector.INDUSTRIAL: 4.0}, 0.0)["wastewater"] == 0.0
    # missing fraction -> treated as fully returned (same convention as WaterDemand.consumptive_demand)
    del r.demand.consumption_fraction[Sector.MUNICIPAL]
    out = energy_for_water(r, {Sector.MUNICIPAL: 10.0}, 0.0)
    assert out["wastewater"] == pytest.approx(10.0)


def test_energy_for_water_riparian_without_crops_or_desalination():
    r = _plain_riparian()
    assert r.crops == [] and r.energy.desalination_capacity_mm3 == 0.0
    out = energy_for_water(r, r.demand.withdrawals(), r.energy.desalination_capacity_mm3)
    assert out["desalination"] == 0.0
    assert out["pumping"] == pytest.approx(pumping_energy_gwh(10.0 * 0.3, 30.0, 0.6))
    assert out["total"] > 0.0


def test_energy_for_water_bounds_and_scaling(basin):
    """Doubling all water doubles every component (linear, non-negative)."""
    rng = np.random.default_rng(6)
    for r in basin:
        for _ in range(10):
            w = {s: float(rng.uniform(0, 5000)) for s in (Sector.MUNICIPAL, Sector.INDUSTRIAL, Sector.AGRICULTURAL, Sector.ENERGY)}
            d = float(rng.uniform(0, 500))
            a = energy_for_water(r, w, d)
            b = energy_for_water(r, {s: 2 * v for s, v in w.items()}, 2 * d)
            for k in a:
                assert a[k] >= 0.0
                assert b[k] == pytest.approx(2 * a[k], rel=1e-12)
            assert a["total"] == pytest.approx(a["pumping"] + a["desalination"] + a["treatment"] + a["wastewater"])


def test_energy_for_water_validation(basin):
    delta = basin.riparian("Delta")
    w = delta.demand.withdrawals()
    with pytest.raises(ValueError, match="Riparian"):
        energy_for_water("Delta", w, 0.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="withdrawals"):
        energy_for_water(delta, [1.0, 2.0], 0.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown withdrawal sector"):
        energy_for_water(delta, {"mining": 1.0}, 0.0)
    with pytest.raises(ValueError, match="unknown withdrawal sector"):
        energy_for_water(delta, {42: 1.0}, 0.0)  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="agricultural"):
        energy_for_water(delta, {Sector.AGRICULTURAL: -1.0}, 0.0)
    with pytest.raises(ValueError, match="municipal"):
        energy_for_water(delta, {Sector.MUNICIPAL: float("nan")}, 0.0)
    with pytest.raises(ValueError, match="environment"):
        energy_for_water(delta, {Sector.ENVIRONMENT: -1.0}, 0.0)
    with pytest.raises(ValueError, match="desalinated_mm3"):
        energy_for_water(delta, w, -0.5)
    for pf in (-0.1, 1.1, float("nan"), "all"):
        with pytest.raises(ValueError, match="pumped_fraction"):
            energy_for_water(delta, w, 0.0, pumped_fraction=pf)
    # bad riparian parameters are also caught
    r = _plain_riparian(pumping_efficiency=0.0)
    with pytest.raises(ValueError, match="efficiency"):
        energy_for_water(r, r.demand.withdrawals(), 0.0)
    r = _plain_riparian(pumped_fraction=1.5)
    with pytest.raises(ValueError, match="pumped_fraction"):
        energy_for_water(r, r.demand.withdrawals(), 0.0)
    r = _plain_riparian(desal_energy_kwh_m3=-3.5)
    with pytest.raises(ValueError, match="kwh_per_m3"):
        energy_for_water(r, r.demand.withdrawals(), 1.0)
    r = _plain_riparian()
    r.demand.consumption_fraction[Sector.MUNICIPAL] = 1.2
    with pytest.raises(ValueError, match="consumption_fraction"):
        energy_for_water(r, r.demand.withdrawals(), 0.0)


def test_energy_for_water_does_not_mutate_inputs(basin):
    snapshot = copy.deepcopy(basin)
    for r in basin:
        w = r.demand.withdrawals()
        w_copy = dict(w)
        energy_for_water(r, w, 12.0, pumped_fraction=0.5)
        energy_for_water(r, w, 12.0)
        assert w == w_copy
    assert basin == snapshot


# --------------------------------------------------------------------------- #
# energy_balance and energy_security_index
# --------------------------------------------------------------------------- #
def test_energy_balance_highland_example(basin):
    highland = basin.riparian("Highland")
    before = copy.deepcopy(highland)
    out = energy_balance(highland, hydropower_gwh=4000.0, thermal_gwh=1000.0, other_renewable_gwh=2000.0, demand_gwh=12_000.0)
    assert set(out) == {"supply", "demand", "deficit", "self_sufficiency", "emissions_t"}
    assert out["supply"] == pytest.approx(7000.0)
    assert out["demand"] == pytest.approx(12_000.0)
    assert out["deficit"] == pytest.approx(5000.0)
    assert out["self_sufficiency"] == pytest.approx(7000.0 / 12_000.0)
    assert out["emissions_t"] == pytest.approx(1000.0 * 150.0)  # Highland grid factor 150 t/GWh
    assert highland == before


def test_energy_balance_surplus_is_not_capped(basin):
    midland = basin.riparian("Midland")
    out = energy_balance(midland, 20_000.0, 30_000.0, 20_000.0, 60_000.0)
    assert out["deficit"] == 0.0
    assert out["self_sufficiency"] == pytest.approx(70_000.0 / 60_000.0)
    assert out["self_sufficiency"] > 1.0
    assert out["emissions_t"] == pytest.approx(30_000.0 * 450.0)


def test_energy_balance_zero_demand_and_zero_supply(basin):
    r = basin.riparian("Delta")
    out = energy_balance(r, 0.0, 0.0, 0.0, 0.0)
    assert out == {"supply": 0.0, "demand": 0.0, "deficit": 0.0, "self_sufficiency": 1.0, "emissions_t": 0.0}
    out = energy_balance(r, 10.0, 0.0, 0.0, 0.0)
    assert out["self_sufficiency"] == 1.0 and out["deficit"] == 0.0
    out = energy_balance(r, 0.0, 0.0, 0.0, 100.0)
    assert out["self_sufficiency"] == 0.0 and out["deficit"] == 100.0 and out["emissions_t"] == 0.0


def test_energy_balance_identities(basin):
    rng = np.random.default_rng(7)
    for r in basin:
        for _ in range(30):
            h, t, o, d = (float(x) for x in rng.uniform(0, 1e5, 4))
            out = energy_balance(r, h, t, o, d)
            assert out["supply"] == pytest.approx(h + t + o)
            assert out["demand"] == pytest.approx(d)
            assert out["deficit"] == pytest.approx(max(d - (h + t + o), 0.0))
            assert out["deficit"] >= 0.0
            assert out["deficit"] == pytest.approx(max(d - out["supply"], 0.0))
            assert out["self_sufficiency"] == pytest.approx(out["supply"] / d)
            assert out["emissions_t"] == pytest.approx(t * r.energy.grid_emission_factor_t_per_gwh)
            # security index is the capped self-sufficiency
            assert energy_security_index(out["supply"], d) == pytest.approx(min(out["self_sufficiency"], 1.0))
            # a riparian is either short (deficit > 0) or exporting (self-sufficiency > 1),
            # never both: the two keys must agree on which side of the balance it is
            assert not (out["deficit"] > 0.0 and out["self_sufficiency"] > 1.0)
            assert (out["deficit"] > 0.0) == (out["self_sufficiency"] < 1.0)


@pytest.mark.parametrize(
    "hydro, thermal, other, demand, surplus, deficit",
    [
        # Highland-like shortfall: 7 000 GWh of supply against 12 000 GWh of demand
        (4000.0, 1000.0, 2000.0, 12_000.0, 0.0, 5000.0),
        # Midland-like surplus: 70 000 GWh of supply against 60 000 GWh of demand
        (20_000.0, 30_000.0, 20_000.0, 60_000.0, 10_000.0, 0.0),
        # exactly balanced: neither surplus nor deficit
        (500.0, 250.0, 250.0, 1000.0, 0.0, 0.0),
        # no generation at all: the whole demand is the deficit
        (0.0, 0.0, 0.0, 2500.0, 0.0, 2500.0),
        # no demand at all: the whole generation is surplus
        (300.0, 0.0, 200.0, 0.0, 500.0, 0.0),
    ],
)
def test_energy_balance_surplus_and_deficit_are_complementary(
    basin, hydro, thermal, other, demand, surplus, deficit
):
    """``supply - demand = surplus - deficit`` against hand-computed values.

    ``surplus`` and ``deficit`` are literals worked out by hand, so an
    unclamped (``demand - supply``) or absolute (``|demand - supply|``)
    deficit would fail the surplus row, and a wrong ``demand`` echo would
    fail every row.
    """
    out = energy_balance(basin.riparian("Delta"), hydro, thermal, other, demand)
    assert out["deficit"] == pytest.approx(deficit)
    assert out["supply"] - out["demand"] == pytest.approx(surplus - deficit)
    assert out["supply"] - out["demand"] + out["deficit"] == pytest.approx(surplus)
    # at most one of surplus / deficit is positive
    assert surplus == 0.0 or out["deficit"] == 0.0
    if demand > 0.0:
        assert (out["self_sufficiency"] > 1.0) == (surplus > 0.0)
    else:
        # convention: nothing demanded -> fully self-sufficient, whatever the surplus
        assert out["self_sufficiency"] == 1.0


def test_energy_balance_emissions_only_from_thermal(basin):
    r = basin.riparian("Delta")
    assert energy_balance(r, 1000.0, 0.0, 1000.0, 1.0)["emissions_t"] == 0.0
    assert energy_balance(r, 0.0, 1.0, 0.0, 1.0)["emissions_t"] == pytest.approx(500.0)
    r2 = _plain_riparian(grid_emission_factor_t_per_gwh=0.0)
    assert energy_balance(r2, 0.0, 1000.0, 0.0, 1.0)["emissions_t"] == 0.0


def test_energy_balance_validation(basin):
    r = basin.riparian("Delta")
    with pytest.raises(ValueError, match="Riparian"):
        energy_balance(None, 1.0, 1.0, 1.0, 1.0)  # type: ignore[arg-type]
    for name, args in [
        ("hydropower_gwh", (-1.0, 0.0, 0.0, 1.0)),
        ("thermal_gwh", (0.0, -1.0, 0.0, 1.0)),
        ("other_renewable_gwh", (0.0, 0.0, -1.0, 1.0)),
        ("demand_gwh", (0.0, 0.0, 0.0, -1.0)),
        ("demand_gwh", (0.0, 0.0, 0.0, float("nan"))),
    ]:
        with pytest.raises(ValueError, match=name):
            energy_balance(r, *args)
    bad = _plain_riparian(grid_emission_factor_t_per_gwh=-1.0)
    with pytest.raises(ValueError, match="grid_emission_factor"):
        energy_balance(bad, 0.0, 1.0, 0.0, 1.0)


def test_energy_security_index_values():
    assert energy_security_index(50.0, 100.0) == pytest.approx(0.5)
    assert energy_security_index(100.0, 100.0) == 1.0
    assert energy_security_index(150.0, 100.0) == 1.0  # capped
    assert energy_security_index(0.0, 100.0) == 0.0
    assert energy_security_index(0.0, 0.0) == 1.0  # no demand -> secure
    assert energy_security_index(10.0, 0.0) == 1.0


def test_energy_security_index_bounds_and_monotonicity():
    rng = np.random.default_rng(8)
    for _ in range(100):
        s, d = (float(x) for x in rng.uniform(0, 1e4, 2))
        v = energy_security_index(s, d)
        assert 0.0 <= v <= 1.0
    supplies = np.linspace(0, 300, 31)
    out = [energy_security_index(float(s), 100.0) for s in supplies]
    assert all(b >= a for a, b in zip(out, out[1:]))
    assert out[0] == 0.0 and out[-1] == 1.0
    demands = np.linspace(1, 300, 31)
    out = [energy_security_index(100.0, float(d)) for d in demands]
    assert all(b <= a for a, b in zip(out, out[1:]))


@pytest.mark.parametrize("bad", BAD_NUMBERS)
def test_energy_security_index_validation(bad):
    with pytest.raises(ValueError):
        energy_security_index(bad, 1.0)
    with pytest.raises(ValueError):
        energy_security_index(1.0, bad)


# --------------------------------------------------------------------------- #
# Integration-style sanity on the example basin
# --------------------------------------------------------------------------- #
def test_example_basin_hydropower_magnitudes(basin):
    """Plausible annual hydropower from each riparian's mean outflow and plant parameters."""
    highland, midland, delta = basin.riparians
    # Highland: ~ (2000 + 18000) Mm3 natural flow, 80 % turbined, 120 m head, 2000 MW
    flow = (basin.headwater_inflow_mm3 + highland.local_inflow_mm3) * highland.energy.turbined_fraction
    e = hydropower_gwh(flow, highland.energy.hydropower_head_m, highland.energy.turbine_efficiency,
                       capacity_mw=highland.energy.hydropower_capacity_mw)
    assert e == pytest.approx(16_000.0 * 0.2725 * 1.2 * 0.9)  # 4 709 GWh, below the 17 520 GWh cap
    assert e < highland.energy.hydropower_capacity_mw * 8.76
    # Delta: low head (20 m), 300 MW cap = 2 628 GWh
    e_delta = hydropower_gwh(20_000.0, delta.energy.hydropower_head_m, delta.energy.turbine_efficiency,
                             capacity_mw=delta.energy.hydropower_capacity_mw)
    assert e_delta == pytest.approx(min(20_000.0 * 0.2725 * 0.2 * 0.9, 2628.0))
    assert e_delta == pytest.approx(981.0)
    # Midland: 600 MW cap = 5 256 GWh binds for a large flow
    assert hydropower_gwh(1e6, midland.energy.hydropower_head_m, capacity_mw=midland.energy.hydropower_capacity_mw) == pytest.approx(5256.0)


def test_example_basin_full_energy_chain(basin):
    """Thermal generation -> cooling water, energy for water -> balance, for each riparian."""
    snapshot = copy.deepcopy(basin)
    for r in basin:
        es = r.energy
        thermal = thermal_generation_gwh(es.thermal_capacity_mw, es.thermal_capacity_factor)
        cooling = water_for_energy(r, thermal)
        assert cooling == pytest.approx(thermal * es.thermal_water_intensity_m3_per_mwh / 1000.0)
        hydro = hydropower_gwh(r.local_inflow_mm3 * es.turbined_fraction, es.hydropower_head_m,
                               es.turbine_efficiency, capacity_mw=es.hydropower_capacity_mw)
        other = es.demand_gwh * es.renewable_share
        efw = energy_for_water(r, r.demand.withdrawals(), es.desalination_capacity_mm3)
        bal = energy_balance(r, hydro, thermal, other, es.demand_gwh + efw["total"])
        assert bal["supply"] == pytest.approx(hydro + thermal + other)
        assert bal["demand"] == pytest.approx(es.demand_gwh + efw["total"])
        assert 0.0 <= energy_security_index(bal["supply"], bal["demand"]) <= 1.0
        assert bal["emissions_t"] == pytest.approx(thermal * es.grid_emission_factor_t_per_gwh)
    assert basin == snapshot
