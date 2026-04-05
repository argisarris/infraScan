import os
from pathlib import Path

# Use relative path from the script location
MAIN = str(Path(__file__).parent.resolve())
RAIL_SERVICES_AK2035_PATH= r'data\temp\railway_services_ak2035.gpkg'
RAIL_SERVICES_AK2035_EXTENDED_PATH = r'data\temp\railway_services_ak2035_extended.gpkg'
RAIL_SERVICES_2024_PATH= r'data/temp/network_railway-services.gpkg'
RAIL_SERVICES_AK2024_EXTENDED_PATH = r'data/temp/network2024_railway_services_extended.gpkg'
NEW_LINKS_UPDATED_PATH = r"data\Network\processed\updated_new_links.gpkg"
NEW_RAILWAY_LINES_PATH = r"data\Network\processed\new_railway_lines.gpkg"
NETWORK_WITH_ALL_MODIFICATIONS = r"data\Network\processed\combined_network_with_all_modifications.gpkg"
DEVELOPMENT_DIRECTORY = r"data\Network\processed\developments"

RAIL_NODES_PATH = r"data\Network\Rail_Node.csv"
RAIL_POINTS_PATH = r"data\Network\processed\points.gpkg"
OD_KT_ZH_PATH = r'data/_basic_data/KTZH_00001982_00003903.xlsx'
OD_STATIONS_KT_ZH_PATH = r'data/traffic_flow/od/rail/ktzh/od_matrix_stations_ktzh_20.csv'
COMMUNE_TO_STATION_PATH = r"data\Network\processed\Communes_to_railway_stations_ZH.xlsx"
GRAPH_POS_PATH = r"data\Network\processed\graph_data.pkl"

# --- BAV Geopackages (Official Swiss Railway Infrastructure) ---
BAV_RAIL_NODES_GPKG = r"data/Spatial_Data/Railway_Infrastructure/Rail_Nodes.gpkg"
BAV_RAIL_SEGMENTS_GPKG = r"data/Spatial_Data/Railway_Infrastructure/Rail_Edges_Segments.gpkg"
BAV_RAIL_ROUTES_GPKG = r"data/Spatial_Data/Railway_Infrastructure/Rail_Edges_Routes.gpkg"
HALTESTELLEN_OEV_GPKG = r"data/Spatial_Data/Railway_Infrastructure/HaltestellenOeV.gpkg"

# --- TLMRegio Railway (Supplementary data for tunnels/bridges) ---
TLMREGIO_RAILWAY_SHP = r"data/Spatial_Data/Land_Use/Transportation/swissTLMRegio_Railway.shp"

# --- Network Infrastructure ---
# Root directory — all version subfolders live here
NETWORK_INFRASTRUCTURE_DIR  = r"data/Infrastructure"
# Raw/  : spatial-filtered BAV output (infrabuild_filter_network.py stage 1)
NETWORK_INFRASTRUCTURE_RAW  = r"data/Infrastructure/Raw"
NETWORK_INFRASTRUCTURE_RAW_NODES                = r"data/Infrastructure/Raw/nodes.gpkg"
NETWORK_INFRASTRUCTURE_RAW_SEGMENTS             = r"data/Infrastructure/Raw/segments.gpkg"
NETWORK_INFRASTRUCTURE_RAW_SEGMENTS_COMPOSITION = r"data/Infrastructure/Raw/segments_composition.gpkg"
# Base/ : macroscopic-simplified network (infrabuild_filter_network.py stage 2)
#         This is the selectable base version for network_builder and version_manager
NETWORK_INFRASTRUCTURE_BASE          = r"data/Infrastructure/Base"
NETWORK_INFRASTRUCTURE_BASE_NODES    = r"data/Infrastructure/Base/nodes.gpkg"
NETWORK_INFRASTRUCTURE_BASE_SEGMENTS = r"data/Infrastructure/Base/segments.gpkg"

# --- Infrastructure Plots ---
INFRASTRUCTURE_PLOTS_DIR = r"plots/Infrastructure"

POPULATION_RASTER = r"data\independent_variable\processed\replacement.pop20_ArcGisExport.tif"
EMPLOYMENT_RASTER = r"data\independent_variable\processed\replacement.empl20_ArcGisExport.tif"
POPULATION_SCENARIO_CANTON_ZH_2050 = r"data\Scenario\KTZH_00000705_00001741.csv"
POPULATION_SCENARIO_CH_BFS_2055 = r"data\Scenario\pop_scenario_switzerland_2055.csv"
POPULATION_SCENARIO_CH_EUROSTAT_2100 = r"data\Scenario\Eurostat_population_CH_2100.xlsx"
POPULATION_PER_COMMUNE_ZH_2018 = r"data\Scenario\population_by_gemeinde_2018.csv"
RANDOM_SCENARIO_CACHE_PATH = r"data\Scenario\cache"
DISTRICT_PATH = r"data\_basic_data\Gemeindegrenzen\UP_BEZIRKE_F.shp"
CANTON_BOUNDARIES_GPKG = r"data/Spatial_Data/Boundaries/Swissboundaries_Cantons_2026_CH.gpkg"
BEZIRKE_BOUNDARIES_GPKG = r"data/Spatial_Data/Boundaries/SwissBoundaries_Bezirke_2026_CH.gpkg"
MUNICIPAL_BOUNDARIES_GPKG = r"data/Spatial_Data/Boundaries/SwissBoundaries_Municipalities_2026_CH.gpkg"
GTFS_TRANSIT_DIR = r"data/Network/GTFS_Timetable"
BUS_LINES_DIR = r"data/Network/Buslines"
FEEDER_LINES_DIR = r"data/Network/Feeder_Lines"
RAIL_PROCESSED_DIR = r"data/Network/processed"
EDGES_IN_CORRIDOR_GPKG = r"data/Network/processed/edges_in_corridor.gpkg"

STUDY_AREA_DIR           = r"data/Study_Area"
STUDY_AREA_BOUNDARY_GPKG = r"data/Study_Area/study_area_boundary.gpkg"
STUDY_AREA_BUFFER_GPKG   = r"data/Study_Area/study_area_buffer.gpkg"

CATCHMENT_AREA_DIR           = r"data/Catchment_Area"
CATCHMENT_AREA_BOUNDARY_GPKG = r"data/Catchment_Area/catchment_area_boundary.gpkg"
CATCHMENT_AREA_BUFFER_GPKG   = r"data/Catchment_Area/catchment_area_buffer.gpkg"
POPULATION_CSV_2023 = r"data/Spatial_Data/Land_Use/Population/Inhabitants_2023_CH.csv"
EMPLOYMENT_CSV_2023 = r"data/Spatial_Data/Land_Use/Employment/Employment_FTE_2023_CH.csv"
LAKES_SHP          = r"data/Spatial_Data/Land_Use/Hydrography/swissTLMRegio_Lake.shp"

CONSTRUCTION_COSTS =  r"data/costs/construction_cost.csv"
TOTAL_COST_WITH_GEOMETRY = r"data/costs/total_costs_with_geometry.csv"
TOTAL_COST_RAW = r"data/costs/total_costs_raw.csv"
COST_AND_BENEFITS_DISCOUNTED = r"data/costs/costs_and_benefits_dev_discounted.csv"
COSTS_CONNECTION_CURVES = r"data/costs/costs_connection_curves.xlsx"

TTS_CACHE = r"data/Network/travel_time/cache/compute_tts_cache.pkl"

PLOT_DIRECTORY = r"plots"
PLOT_SCENARIOS = r"plots/scenarios"

def get_rail_services_path(rail_network_settings):
    """
    Returns the path to the rail services file based on the rail network settings.
    """
    if rail_network_settings == 'AK_2035':
        return RAIL_SERVICES_AK2035_PATH
    elif rail_network_settings == 'AK_2035_extended':
        return RAIL_SERVICES_AK2035_EXTENDED_PATH
    elif rail_network_settings == 'current':
        return RAIL_SERVICES_2024_PATH
    elif rail_network_settings == '2024_extended':
        return RAIL_SERVICES_AK2024_EXTENDED_PATH
    else:
        raise ValueError("Invalid rail network settings provided.")