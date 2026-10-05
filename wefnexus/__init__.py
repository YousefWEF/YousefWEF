"""wefnexus - Water-Energy-Food nexus and water diplomacy sustainability toolkit.

Sub-modules
-----------
models          dataclasses: Sector, WaterDemand, Crop, EnergySystem, Riparian, Basin, Scenario
water           water stress indicators and basin water-balance routing
energy          water-for-energy and energy-for-water calculations
food            crop water requirements, FAO-33 yield response, food security
allocation      bankruptcy rules, cooperative game solutions, equity metrics
diplomacy       hydro-hegemony, treaties, benefit sharing, conflict/cooperation indices
sustainability  composite WEF security and SDG-style indicators
nexus           integrated multi-year WEF nexus simulation
optimize        LP-based basin allocation and Pareto trade-offs
scenarios       scenario library and comparison runner
viz             optional matplotlib plots
cli             command-line interface (``python -m wefnexus``)
"""
from wefnexus.models import (  # noqa: F401
    Basin,
    Crop,
    EnergySystem,
    Riparian,
    Scenario,
    Sector,
    WaterDemand,
)

__version__ = "0.1.0"

__all__ = [
    "Basin",
    "Crop",
    "EnergySystem",
    "Riparian",
    "Scenario",
    "Sector",
    "WaterDemand",
    "__version__",
]
