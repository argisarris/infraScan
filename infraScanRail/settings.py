"""
infraScanRail — Pipeline Settings
Last modified: 2026-05-15

Central configuration file. Sections mirror the main pipeline order.
Edit values here; all downstream modules read from this file.
"""

from shapely.geometry import Polygon

# ═══════════════════════════════════════════════════════════════════════════════
# 1. STUDY AREA
# ═══════════════════════════════════════════════════════════════════════════════

# 'coordinates' — use perimeter_infra_generation polygon defined below
# 'admin'       — dissolve SwissBoundaries admin units (fill STUDY_AREA_ADMIN_* below)
STUDY_AREA_METHOD = 'coordinates'

# Polygon used when STUDY_AREA_METHOD = 'coordinates'  (EPSG:2056)
perimeter_infra_generation = Polygon([
    (2700989.862, 1235663.403),
    (2708491.515, 1239608.529),
    (2694972.602, 1255514.900),
    (2687415.817, 1251056.404),
])

# Used when STUDY_AREA_METHOD = 'admin'
# Admin level: 'national' | 'cantonal' | 'bezirke' | 'municipal'
STUDY_AREA_ADMIN_LEVEL = 'municipal'
STUDY_AREA_ADMIN_NAMES = [           
    'Dübendorf', 'Fällanden', 'Fehraltorf',
    'Gossau (ZH)', 'Greifensee', 'Grüningen',
    'Hinwil', 'Illnau-Effretikon', 'Mönchaltorf',
    'Pfäffikon', 'Schwerzenbach', 'Seegräben',
    'Uster',  'Volketswil', 'Wangen-Brüttisellen',
    'Wetzikon (ZH)',
]
STUDY_AREA_ADMIN_SUBDIVISIONS = {'bezirke': [], 'municipal': []}  # optional extra units in subdivisions of the choosen admin level
STUDY_AREA_BUFFER_M = 3000                    # margin (m) for feeder network edge handling

# ═══════════════════════════════════════════════════════════════════════════════
# 2. CATCHMENT AREA
# ═══════════════════════════════════════════════════════════════════════════════

# Catchment boundary is always admin-based (study area must be fully contained within)
# Admin level: 'national' | 'cantonal' | 'bezirke' | 'municipal'
CATCHMENT_AREA_ADMIN_LEVEL = 'cantonal'
CATCHMENT_AREA_ADMIN_NAMES = ['Zürich']       # list of admin entity names to dissolve
CATCHMENT_AREA_ADMIN_SUBDIVISIONS = {'bezirke': [], 'municipal': []}
CATCHMENT_AREA_BUFFER_M = 5000                # buffer (m) around boundary for GTFS spatial filter
CATCHMENT_CANTON_ABBREV = 'ZH'               # canton abbreviation — used in folder and file naming

# ═══════════════════════════════════════════════════════════════════════════════
# 3. INFRASTRUCTURE VERSION
# ═══════════════════════════════════════════════════════════════════════════════

# Named output folders for the two filter scripts (Phase 2 skip-guards)
# infrabuild_filter_network output: data/Infrastructure/<INFRA_RAW_VERSION>/
INFRA_RAW_VERSION   = 'Raw_ZH'

# 'Build_New'           — run full infrabuild pipeline to create a new named version
# 'AS_2026_ZH'          — BAV network as-is for 2026 (must already exist on disk)
# 'AS_2026_ZH_enhanced' — AS_2026_ZH enriched with projected service travel times and corrections from the svc projections
# 'AS_2035_ZH'          — BAV network as-is for 2035 (must already exist on disk)
# 'AS_2035_ZH_enhanced' — AS_2035_ZH enriched with projected service travel times and corrections from the svc projections
INFRA_VERSION = 'AS_2035_ZH'

# Name for the new version to create — used only when INFRA_VERSION = 'Build_New'
INFRA_BUILD_NEW_NAME = 'AS_2035_ZH'

# ═══════════════════════════════════════════════════════════════════════════════
# 4. SERVICES VERSION
# ═══════════════════════════════════════════════════════════════════════════════

# services_filter_gtfs output: data/Network/GTFS_Timetable/<GTFS_FILTER_VERSION>/
GTFS_FILTER_VERSION = 'GTFS_SVC2026_ZH_S18'

# 'Build_New'   — run full services pipeline to create a new named version
# 'AK_2026'     — scheduled services as of 2026 timetable
# 'AK_2026_S18' — AK_2026 with S18 line included
# 'AK_2035'     — scheduled services as of 2035 timetable
# 'AK_2035_S18' — AK_2035 with S18 line included
SVC_VERSION = 'AK_2026_S18'

# Name for the new version to create — used only when SVC_VERSION = 'Build_New'
SVC_BUILD_NEW_NAME = 'AK_2026_S18'

# ── Enhancement conflict resolution ─────────────────────────────────────────
# When both infra and service data are Tier 1 (real data: infra/gtfs vs
# gtfs/infra), this setting decides which side is the source of truth.
# 'service' — GTFS service recalibrates infra TT/speed → speed_source='gtfs'
# 'infra'   — infra speeds recalibrate service TT     → tt_source='infra'
ENHANCEMENT_CONFLICT_T1 = 'service'

# ═══════════════════════════════════════════════════════════════════════════════
# 5. CAPACITY
# ═══════════════════════════════════════════════════════════════════════════════

# 'None'      — skip capacity phases entirely
# 'Set_Value' — apply a fixed trains/hour/direction threshold to all sections
# 'Dynamic'   — full iterative capacity calculator workflow (run_capacity_analysis.py)
CAPACITY_MODE = 'Dynamic'

CAPACITY_SET_VALUE = 6             # trains/hour/direction — used when CAPACITY_MODE = 'Set_Value'
capacity_threshold = 2.0           # minimum available capacity (tphpd) — Dynamic only
max_enhancement_iterations = 10    # max Phase 4 enhancement iterations — Dynamic only

# Internal — set dynamically in main (do not edit)
baseline_network_for_developments = None

# ═══════════════════════════════════════════════════════════════════════════════
# 6. CATCHMENT METHOD
# ═══════════════════════════════════════════════════════════════════════════════

# 'Municipal' — each commune assigned wholly to one rail station (centroid-based)
# 'PT_Feeder' — access-time raster decides which cells belong to which station
# OD allocation method is derived automatically:
#   Municipal  ->  commune-to-station lookup table
#   PT_Feeder  ->  communal OD re-weighted by PT-feeder catchment shares
CATCHMENT_METHOD = 'PT_Feeder'

# When True, all access times expressed as equivalent IVT seconds (generalised cost)
USE_GENERALISED_COST = True

# Transfer cost model — requires USE_GENERALISED_COST = True
# 'fixed_value' — flat 12.1 min eq. IVT penalty (Axhausen 2014, Fuchs 2025)
# 'explicit'    — W_TRANSFER x (transfer walk time + wait time based on connecting headway)
TRANSFER_COST_MODEL = 'fixed_value'

# When True, OD demand is filtered to only include trips from/to the study area perimeter
only_demand_from_to_perimeter = True

# ═══════════════════════════════════════════════════════════════════════════════
# 7. OD & DEMAND
# ═══════════════════════════════════════════════════════════════════════════════

# 'canton_ZH'              — station OD derived from commune-level cantonal survey data
# 'pt_catchment_perimeter' — OD derived from PT catchment area
OD_TYPE = 'canton_ZH'

# Base year for the population/employment grids, OD matrices, and scenario scaling
POPULATION_BASE_YEAR = 2023

# OD scaling weights — control how population and employment growth are blended
# when computing per-commune OD growth factors.
# OD_SCALING_EMPL_WEIGHT = 0.0 because employment is derived from population
# (empl scales with pop), so adding employment weight changes nothing.
# Set OD_SCALING_EMPL_WEIGHT > 0 only when an independent employment projection
# is available (e.g. BFS STATENT projections beyond 2023).
OD_SCALING_POP_WEIGHT  = 1.0
OD_SCALING_EMPL_WEIGHT = 0.0

# ═══════════════════════════════════════════════════════════════════════════════
# 8. SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════════

# 'GENERATED' — Monte Carlo random scenarios from population growth models
# 'STATIC_9'  — fixed set of 9 canonical scenarios
# 'dummy'     — minimal placeholder scenarios for testing
scenario_type = 'GENERATED'

amount_of_scenarios = 100
start_year_scenario = 2018
end_year_scenario = 2100
start_valuation_year = 2050

# ═══════════════════════════════════════════════════════════════════════════════
# 9. VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════

# Global visualisation mode — controls how optional plots are handled at runtime
# 'manual' — prompt the user for each optional plot decision
# 'none'   — skip all optional plots without prompting
# 'all'    — generate all optional plots without prompting
VISUALIZATION_MODE = 'manual'

# Capacity grouping strategy — controls automated decisions in the capacity workflow
# 'manual'       — prompt for each capacity grouping decision
# 'conservative' — always choose the lowest capacity option
# 'baseline'     — always choose the middle option
# 'optimal'      — always choose the highest capacity option
CAPACITY_GROUPING_STRATEGY = 'manual'

# Per-pipeline plot toggles — set False to suppress all plots for that pipeline area
# without changing the global VISUALIZATION_MODE for interactive decisions elsewhere
PLOT_CATCHMENT = True   # catchment_base population/employment maps (Phase 2)
PLOT_INFRA     = True   # infrabuild network plots (Phase 3A)
PLOT_SERVICES  = True   # services pipeline plots (Phase 3B)
PLOT_CAPACITY  = True   # capacity analysis plots (Phase 3C)
PLOT_RESULTS   = True   # final CBA/result visualisations

plot_passenger_flow = True
plot_railway_line_load = True

# ═══════════════════════════════════════════════════════════════════════════════
# 10. CACHE
# ═══════════════════════════════════════════════════════════════════════════════

# Set to True to load pre-computed outputs from disk instead of recomputing
use_cache_network = False
use_cache_pt_catchment = False
use_cache_developments = False
use_cache_catchmentOD = False
use_cache_stationsOD = False
use_cache_traveltime_graph = False
use_cache_scenarios = False
use_cache_tts_calc = False
