"""Stylised example basin used in tests, examples and the CLI.

The *Azura River* is a fictional transboundary basin shared by three
riparians, ordered upstream to downstream:

* **Highland** – mountainous upstream state, generates most of the runoff,
  has large hydropower potential and little irrigation.
* **Midland** – mid-stream state with fast-growing cities, thermal power
  plants and an expanding irrigation sector.
* **Delta** – arid downstream state with a very large population, the
  biggest irrigation sector, groundwater overdraft and desalination.

All numbers are illustrative but of realistic magnitude for a basin of
roughly 28 km3/yr mean annual flow.  They are *not* data for any real river.
"""
from __future__ import annotations

from wefnexus.models import Basin, Crop, EnergySystem, Riparian, WaterDemand

__all__ = ["example_basin", "EXAMPLE_BASIN_NAME"]

EXAMPLE_BASIN_NAME = "Azura River (stylised)"


def example_basin() -> Basin:
    """Build and return a fresh copy of the stylised Azura River basin."""
    highland = Riparian(
        name="Highland",
        population=8_000_000,
        gdp_usd=40e9,
        local_inflow_mm3=18_000.0,
        groundwater_recharge_mm3=800.0,
        groundwater_abstraction_mm3=200.0,
        demand=WaterDemand(
            municipal=600.0,
            industrial=200.0,
            agricultural=800.0,
            energy=100.0,
            environmental=3_000.0,
        ),
        crops=[
            Crop("wheat", area_ha=150_000, kc=0.85, season_days=150,
                 yield_max_t_ha=4.0, ky=1.05, kcal_per_kg=3_400, price_usd_t=250),
            Crop("maize", area_ha=50_000, kc=0.90, season_days=130,
                 yield_max_t_ha=6.0, ky=1.25, kcal_per_kg=3_600, price_usd_t=200),
        ],
        energy=EnergySystem(
            demand_gwh=12_000.0,
            hydropower_capacity_mw=2_000.0,
            hydropower_head_m=120.0,
            turbined_fraction=0.8,
            thermal_capacity_mw=300.0,
            renewable_share=0.75,
            pumping_lift_m=20.0,
            pumped_fraction=0.2,
            grid_emission_factor_t_per_gwh=150.0,
        ),
        reservoir_capacity_mm3=6_000.0,
        reservoir_storage_mm3=3_000.0,
        irrigation_efficiency=0.50,
        et0_mm_day=4.0,
        effective_rainfall_mm=300.0,
        material_power=0.40,
        bargaining_power=0.65,
        ideational_power=0.40,
        treaty_allocation_mm3=2_500.0,
    )

    midland = Riparian(
        name="Midland",
        population=20_000_000,
        gdp_usd=150e9,
        local_inflow_mm3=7_000.0,
        groundwater_recharge_mm3=1_200.0,
        groundwater_abstraction_mm3=900.0,
        demand=WaterDemand(
            municipal=1_500.0,
            industrial=900.0,
            agricultural=5_550.0,
            energy=300.0,
            environmental=2_500.0,
        ),
        crops=[
            Crop("wheat", area_ha=300_000, kc=0.85, season_days=150,
                 yield_max_t_ha=4.5, ky=1.05, kcal_per_kg=3_400, price_usd_t=250),
            Crop("cotton", area_ha=100_000, kc=0.90, season_days=180,
                 yield_max_t_ha=2.5, ky=0.85, kcal_per_kg=0.0, price_usd_t=1_600),
            Crop("vegetables", area_ha=80_000, kc=0.95, season_days=120,
                 yield_max_t_ha=25.0, ky=1.10, kcal_per_kg=300, price_usd_t=400),
        ],
        energy=EnergySystem(
            demand_gwh=60_000.0,
            hydropower_capacity_mw=600.0,
            hydropower_head_m=40.0,
            turbined_fraction=0.6,
            thermal_capacity_mw=3_000.0,
            renewable_share=0.25,
            pumping_lift_m=35.0,
            pumped_fraction=0.3,
            grid_emission_factor_t_per_gwh=450.0,
        ),
        reservoir_capacity_mm3=3_000.0,
        reservoir_storage_mm3=1_500.0,
        irrigation_efficiency=0.50,
        et0_mm_day=5.5,
        effective_rainfall_mm=150.0,
        material_power=0.70,
        bargaining_power=0.60,
        ideational_power=0.60,
        treaty_allocation_mm3=9_000.0,
    )

    delta = Riparian(
        name="Delta",
        population=45_000_000,
        gdp_usd=300e9,
        local_inflow_mm3=1_000.0,
        groundwater_recharge_mm3=1_500.0,
        groundwater_abstraction_mm3=2_000.0,
        demand=WaterDemand(
            municipal=3_500.0,
            industrial=1_500.0,
            agricultural=13_000.0,
            energy=600.0,
            environmental=1_500.0,
        ),
        crops=[
            Crop("rice", area_ha=200_000, kc=1.10, season_days=150,
                 yield_max_t_ha=7.0, ky=1.20, kcal_per_kg=3_600, price_usd_t=350),
            Crop("wheat", area_ha=400_000, kc=0.85, season_days=150,
                 yield_max_t_ha=5.0, ky=1.05, kcal_per_kg=3_400, price_usd_t=250),
            Crop("maize", area_ha=150_000, kc=0.90, season_days=130,
                 yield_max_t_ha=7.0, ky=1.25, kcal_per_kg=3_600, price_usd_t=200),
            Crop("vegetables", area_ha=100_000, kc=0.95, season_days=120,
                 yield_max_t_ha=28.0, ky=1.10, kcal_per_kg=300, price_usd_t=400),
        ],
        energy=EnergySystem(
            demand_gwh=160_000.0,
            hydropower_capacity_mw=300.0,
            hydropower_head_m=20.0,
            turbined_fraction=0.5,
            thermal_capacity_mw=8_000.0,
            renewable_share=0.12,
            pumping_lift_m=40.0,
            pumped_fraction=0.35,
            desalination_capacity_mm3=300.0,
            grid_emission_factor_t_per_gwh=500.0,
        ),
        reservoir_capacity_mm3=10_000.0,
        reservoir_storage_mm3=6_000.0,
        irrigation_efficiency=0.55,
        et0_mm_day=6.5,
        effective_rainfall_mm=20.0,
        material_power=0.80,
        bargaining_power=0.50,
        ideational_power=0.70,
        treaty_allocation_mm3=16_000.0,
    )

    return Basin(
        name=EXAMPLE_BASIN_NAME,
        riparians=[highland, midland, delta],
        headwater_inflow_mm3=2_000.0,
        climate_cv=0.20,
    )
