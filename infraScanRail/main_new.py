"""
infraScanRail - New Main Pipeline
Last modified: 2026-05-15

Orchestrates Phases 1 and 2 of the new pipeline.
Phases 3A/3B (infra/services network building, Workflow 2) and
Phase 3C (capacity, Workflow 3) are stubs pending future workflows.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# RUN CONTROL
# ═══════════════════════════════════════════════════════════════════════════════
# 'all'  — run the complete pipeline from start to finish
# list   — run only the listed phases, e.g. ['1', '2'] or ['3A'] or ['1', '3A']
# Valid phase names: '1', '2', '3A', '3B', '3C'
# Each phase requires its predecessor's outputs to already be on disk.
RUN_MODE = 'all'
# ═══════════════════════════════════════════════════════════════════════════════

import os
import sys
import time
import warnings
import subprocess
from pathlib import Path

import geopandas as gpd
import pandas as pd

import paths
import settings
from catchment_base import (
    _dissolve_admin_polygon,
    _export_gpkg,
    _validate_containment,
    STUDY_AREA_DEFAULT_BUFFER_M,
    CATCHMENT_AREA_DEFAULT_BUFFER_M,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Configuration
# ═══════════════════════════════════════════════════════════════════════════════

class PipelineConfig:
    """Holds resolved pipeline state set during Phase 1."""

    def __init__(self):
        self.visualization_mode    = settings.VISUALIZATION_MODE
        self.grouping_strategy     = settings.CAPACITY_GROUPING_STRATEGY
        self.needs_projection      = False   # set True if svc version needs projection
        self.infra_version         = None    # resolved in phase_1_initialisation
        self.svc_version           = None    # resolved in phase_1_initialisation
        self._original_input       = None    # stored in infrascanrail_new for smart_input

    def should_generate_plots(self, default_yes: bool = False):
        """Return True/False/None based on visualization_mode.

        None signals the caller to prompt the user (manual mode).
        """
        if self.visualization_mode == 'all':
            return True
        if self.visualization_mode == 'none':
            return False
        return None  # manual: caller decides


PIPELINE_CONFIG = PipelineConfig()


def get_catchment_od_method() -> str:
    """Return the OD allocation method that matches the active catchment method.

    Municipal  -> 'municipal'       (commune-to-station lookup table)
    PT_Feeder  -> 'feeder_weighted' (communal OD re-weighted by feeder catchment shares)
    """
    if settings.CATCHMENT_METHOD == 'Municipal':
        return 'municipal'
    return 'feeder_weighted'


def get_routing_od_method() -> str:
    """Translate the settings OD method to the internal string used by
    catchment_OD_rail_network.route_od_matrices() when selecting W3 CSV files.

    'feeder_weighted' -> 'pt_feeder'   (reads OD_STATIONS_PT_FEEDER_* paths)
    'municipal'       -> 'municipal'   (reads OD_STATIONS_MUNICIPAL_* paths)
    """
    return 'pt_feeder' if get_catchment_od_method() == 'feeder_weighted' else 'municipal'


# ═══════════════════════════════════════════════════════════════════════════════
# Study / Catchment area helpers  (Option X -no input() patching)
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_study_area():
    """Build study area polygon from settings without interactive prompts.

    Returns (polygon, buffered_polygon, admin_level, primary_names, buffer_m).
    """
    if settings.STUDY_AREA_METHOD == 'coordinates':
        polygon       = settings.perimeter_infra_generation
        admin_level   = 'coordinates'
        primary_names = []
    else:
        polygon = _dissolve_admin_polygon(
            settings.STUDY_AREA_ADMIN_LEVEL,
            settings.STUDY_AREA_ADMIN_NAMES,
            settings.STUDY_AREA_ADMIN_SUBDIVISIONS,
        )
        admin_level   = settings.STUDY_AREA_ADMIN_LEVEL
        primary_names = settings.STUDY_AREA_ADMIN_NAMES

    buffer_m = settings.STUDY_AREA_BUFFER_M
    return polygon, polygon.buffer(buffer_m), admin_level, primary_names, buffer_m


def _resolve_catchment_area():
    """Build catchment area polygon from settings without interactive prompts.

    Returns (polygon, buffered_polygon, admin_level, primary_names, buffer_m).
    """
    polygon = _dissolve_admin_polygon(
        settings.CATCHMENT_AREA_ADMIN_LEVEL,
        settings.CATCHMENT_AREA_ADMIN_NAMES,
        settings.CATCHMENT_AREA_ADMIN_SUBDIVISIONS,
    )
    buffer_m = settings.CATCHMENT_AREA_BUFFER_M
    return polygon, polygon.buffer(buffer_m), settings.CATCHMENT_AREA_ADMIN_LEVEL, \
           settings.CATCHMENT_AREA_ADMIN_NAMES, buffer_m


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 -Initialisation
# ═══════════════════════════════════════════════════════════════════════════════

def phase_1_initialisation(runtimes: dict) -> tuple:
    """Resolve study/catchment areas, validate version settings, print run summary.

    Args:
        runtimes: Dict tracking phase execution times.

    Returns:
        (sa_boundary, sa_buffer, ca_boundary, ca_buffer) -Shapely polygons.
    """
    print("\n" + "=" * 80)
    print("PHASE 1: INITIALISATION")
    print("=" * 80 + "\n")
    st = time.time()

    # ── Step 1.1: Study area ──────────────────────────────────────────────────
    print("--- Step 1.1: Study Area ---\n")
    sa_boundary_path = os.path.join(paths.MAIN, paths.STUDY_AREA_BOUNDARY_GPKG)
    sa_buffer_path   = os.path.join(paths.MAIN, paths.STUDY_AREA_BUFFER_GPKG)

    sa_boundary, sa_buffer, sa_admin, sa_names, sa_buf_m = _resolve_study_area()
    _export_gpkg(sa_boundary, 'study_area_boundary', sa_admin, sa_names, 0,         sa_boundary_path)
    _export_gpkg(sa_buffer,   'study_area_buffer',   sa_admin, sa_names, sa_buf_m,  sa_buffer_path)
    print(f"  Study area built from settings and saved.")
    print(f"  Bounds: {sa_boundary.bounds}\n")

    # ── Step 1.2: Catchment area ──────────────────────────────────────────────
    print("--- Step 1.2: Catchment Area ---\n")
    ca_boundary_path = os.path.join(paths.MAIN, paths.CATCHMENT_AREA_BOUNDARY_GPKG)
    ca_buffer_path   = os.path.join(paths.MAIN, paths.CATCHMENT_AREA_BUFFER_GPKG)

    ca_boundary, ca_buffer, ca_admin, ca_names, ca_buf_m = _resolve_catchment_area()

    valid = _validate_containment(sa_boundary, ca_boundary)
    if not valid:
        print("  WARNING: Study area is not fully within catchment area.")
        print("  Update CATCHMENT_AREA_* in settings.py and re-run.")

    _export_gpkg(ca_boundary, 'catchment_area_boundary', ca_admin, ca_names, 0,        ca_boundary_path)
    _export_gpkg(ca_buffer,   'catchment_area_buffer',   ca_admin, ca_names, ca_buf_m, ca_buffer_path)
    print(f"  Catchment area built from settings and saved.")
    print(f"  Bounds: {ca_boundary.bounds}\n")

    # ── Step 1.3: Infrastructure version check ────────────────────────────────
    print("--- Step 1.3: Infrastructure Version ---\n")
    infra_v = settings.INFRA_VERSION

    if infra_v == 'Build_New':
        print(f"  INFRA_VERSION = 'Build_New' → will build '{settings.INFRA_BUILD_NEW_NAME}' in Phase 3A.")
        PIPELINE_CONFIG.infra_version = settings.INFRA_BUILD_NEW_NAME
    elif paths.infra_version_exists(infra_v):
        print(f"  Infrastructure version '{infra_v}' found on disk.")
        PIPELINE_CONFIG.infra_version = infra_v
    else:
        print(f"  WARNING: Infrastructure version '{infra_v}' not found.")
        print(f"    Expected: {paths.get_infra_version_dir(infra_v)}")
        ans = input(f"  Switch to Build_New (b) or abort (a)? [b]: ").strip().lower() or 'b'
        if ans == 'b':
            print(f"  Switching INFRA_VERSION to Build_New.")
            infra_v = 'Build_New'
            PIPELINE_CONFIG.infra_version = settings.INFRA_BUILD_NEW_NAME
        else:
            raise SystemExit("Aborted: infrastructure version not found.")

    # ── Step 1.4: Services version check ─────────────────────────────────────
    print("\n--- Step 1.4: Services Version ---\n")
    svc_v = settings.SVC_VERSION

    if svc_v == 'Build_New':
        print(f"  SVC_VERSION = 'Build_New' → will build '{settings.SVC_BUILD_NEW_NAME}' in Phase 3B.")
        PIPELINE_CONFIG.svc_version       = settings.SVC_BUILD_NEW_NAME
        PIPELINE_CONFIG.needs_projection  = True
    elif not paths.svc_version_exists(svc_v):
        print(f"  WARNING: Services network '{svc_v}_network' not found.")
        print(f"    Expected: data/Network/Rail_Lines/{svc_v}_network/Unprojected/"
              "{rail_lines,rail_segments,rail_stops}.gpkg")
        ans = input(f"  Switch to Build_New (b) or abort (a)? [b]: ").strip().lower() or 'b'
        if ans == 'b':
            print(f"  Switching SVC_VERSION to Build_New.")
            PIPELINE_CONFIG.svc_version      = settings.SVC_BUILD_NEW_NAME
            PIPELINE_CONFIG.needs_projection = True
        else:
            raise SystemExit("Aborted: services version not found.")
    else:
        PIPELINE_CONFIG.svc_version = svc_v
        print(f"  Rail network '{svc_v}_network/Unprojected/' found.")

        if settings.CATCHMENT_METHOD == 'PT_Feeder':
            if paths.svc_feeder_exists(svc_v):
                print(f"  PT-Feeder network '{svc_v}_network/Unprojected/' found.")
            else:
                print(f"  WARNING: PT-Feeder network '{svc_v}_network/Unprojected/' incomplete.")
                print(f"    Expected: data/Network/Feeder_Lines/{svc_v}_network/Unprojected/"
                      "{pt_feeder_lines,pt_feeder_segments,pt_feeder_stops}.gpkg")
                print(f"  Re-run services_network_builder.py with mode 'all' or 'pt_feeder'.")

        resolved_infra = PIPELINE_CONFIG.infra_version
        if resolved_infra and resolved_infra != 'Build_New' and \
                not paths.svc_projected_exists(svc_v, resolved_infra):
            print(f"  Services version '{svc_v}' found but not yet projected to '{resolved_infra}'.")
            print(f"  Projection will run in Phase 3B.")
            PIPELINE_CONFIG.needs_projection = True
        else:
            print(f"  Services version '{svc_v}' found and projected.")

    # ── Step 1.5: Run configuration table ────────────────────────────────────
    infra_tag      = (f"  -->  build as '{settings.INFRA_BUILD_NEW_NAME}'"
                      if settings.INFRA_VERSION == 'Build_New' else "")
    needs_proj_tag = "  [needs projection]" if PIPELINE_CONFIG.needs_projection else ""

    config_lines = [
        "=" * 80,
        "  RUN CONFIGURATION",
        "=" * 80,
        f"  Infrastructure   : {settings.INFRA_VERSION}{infra_tag}",
        f"  Raw version      : {settings.INFRA_RAW_VERSION}",
        f"  Services         : {settings.SVC_VERSION}{needs_proj_tag}",
        f"  GTFS version     : {settings.GTFS_FILTER_VERSION}",
        f"  Canton           : {settings.CATCHMENT_CANTON_ABBREV}",
        f"  Catchment method : {settings.CATCHMENT_METHOD}  (OD: {get_catchment_od_method()})",
        f"  Capacity mode    : {settings.CAPACITY_MODE}",
        f"  OD type          : {settings.OD_TYPE}  (base year {settings.POPULATION_BASE_YEAR})",
        f"  Population base  : {settings.POPULATION_BASE_YEAR}",
        f"  Scenarios        : {settings.amount_of_scenarios} x "
        f"[{settings.start_year_scenario}-{settings.end_year_scenario}]",
        f"  Visualisation    : {settings.VISUALIZATION_MODE}",
        "-" * 80,
    ]
    print()
    for line in config_lines:
        print(line)
    print()

    # Write configuration header to report_new.txt immediately so it is
    # present even if the pipeline is interrupted before completion.
    rt_file = os.path.join(paths.MAIN, 'report_new.txt')
    with open(rt_file, 'w', encoding='utf-8') as f:
        f.write("INFRASCANRAIL NEW PIPELINE - RUN LOG\n\n")
        for line in config_lines:
            f.write(line + "\n")
        f.write("\n")

    runtimes["Phase 1: Initialisation"] = time.time() - st
    return sa_boundary, sa_buffer, ca_boundary, ca_buffer


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 -Data Preparation
# ═══════════════════════════════════════════════════════════════════════════════

def phase_2_data_preparation(
    sa_boundary,
    sa_buffer,
    ca_boundary,
    ca_buffer,
    runtimes: dict,
) -> None:
    """Import base datasets needed by all downstream phases.

    Args:
        sa_boundary: Study area polygon (Shapely).
        sa_buffer:   Study area polygon + margin (Shapely).
        ca_boundary: Catchment area polygon (Shapely).
        ca_buffer:   Catchment area polygon + GTFS buffer (Shapely).
        runtimes:    Dict tracking phase execution times.
    """
    print("\n" + "=" * 80)
    print("PHASE 2: DATA PREPARATION")
    print("=" * 80 + "\n")
    st = time.time()
    loaded = []
    skipped = []

    # ── Step 2.1: Lake clipping ───────────────────────────────────────────────
    print("--- Step 2.1: Lake Clipping ---\n")
    lakes_src_path = os.path.join(paths.MAIN, paths.LAKES_SHP)
    if os.path.isfile(lakes_src_path):
        os.makedirs(os.path.join(paths.MAIN, os.path.dirname(paths.LAKES_CA_GPKG)),
                    exist_ok=True)
        lakes_src = gpd.read_file(lakes_src_path).to_crs('EPSG:2056')

        lakes_ca = lakes_src[lakes_src.intersects(ca_boundary)].copy()
        lakes_ca = gpd.clip(lakes_ca, ca_boundary)
        lakes_ca.to_file(os.path.join(paths.MAIN, paths.LAKES_CA_GPKG), driver='GPKG')
        loaded.append("lakes (CA)")

        lakes_sa = lakes_src[lakes_src.intersects(sa_boundary)].copy()
        lakes_sa = gpd.clip(lakes_sa, sa_boundary)
        lakes_sa.to_file(os.path.join(paths.MAIN, paths.LAKES_SA_GPKG), driver='GPKG')
        loaded.append("lakes (SA)")
        print(f"  Lakes clipped to CA ({len(lakes_ca)} features) and "
              f"SA ({len(lakes_sa)} features) and saved.")
    else:
        print(f"  WARNING: Lake source not found: {paths.LAKES_SHP}")
    print()

    # ── Step 2.2: Population & employment grid (catchment_base) ──────────────
    print("--- Step 2.2: Population & Employment Grid ---\n")
    print(f"  Running catchment_base for POPULATION_BASE_YEAR={settings.POPULATION_BASE_YEAR} ...")
    import catchment_base as _cb
    scale_report = _cb.main(
        year=settings.POPULATION_BASE_YEAR,
        do_plots=settings.PLOT_CATCHMENT,
    )
    if scale_report:
        rt_file = os.path.join(paths.MAIN, 'report_new.txt')
        with open(rt_file, 'a', encoding='utf-8') as f:
            f.write("\n--- Grid Scaling (Step 2.2) ---\n")
            f.write(scale_report + "\n")
    loaded.append(f"pop/empl grid ({settings.POPULATION_BASE_YEAR})")
    print()

    # ── Step 2.3: BAV infrastructure filter ──────────────────────────────────
    print("--- Step 2.3: BAV Infrastructure Filter ---\n")
    raw_dir = paths.get_infra_raw_dir(settings.INFRA_RAW_VERSION)
    required_raw = [
        'nodes.gpkg', 'segments.gpkg',
        'segments_composition.gpkg', 'osm_maxspeed_segments.gpkg',
    ]
    missing_raw = [f for f in required_raw
                   if not os.path.isfile(os.path.join(raw_dir, f))]

    if not missing_raw:
        print(f"  {settings.INFRA_RAW_VERSION}/ complete -- skipping BAV filter.")
        skipped.append("BAV filter")
    else:
        print(f"  Missing in {settings.INFRA_RAW_VERSION}/: {missing_raw}")
        print(f"  Running infrabuild_filter_network ...")
        from infrabuild_filter_network import run_filter_network
        ca_buf_path = os.path.join(paths.MAIN, paths.CATCHMENT_AREA_BUFFER_GPKG)
        run_filter_network(
            output_dir=raw_dir,
            catchment_filepath=ca_buf_path,
        )
        loaded.append("BAV Raw network")
        print(f"  BAV filter complete.\n")

    # ── Step 2.4: GTFS filter ─────────────────────────────────────────────────
    print("--- Step 2.4: GTFS Filter ---\n")
    gtfs_dir      = os.path.join(paths.MAIN, paths.GTFS_TRANSIT_DIR,
                                 settings.GTFS_FILTER_VERSION)
    gtfs_key_file = os.path.join(gtfs_dir, 'stop_times.txt')

    if os.path.isfile(gtfs_key_file):
        print(f"  {settings.GTFS_FILTER_VERSION} found -- skipping GTFS filter.")
        skipped.append("GTFS filter")
    else:
        print(f"  {settings.GTFS_FILTER_VERSION} not found -- running services_filter_gtfs ...")
        print(f"  NOTE: services_filter_gtfs.py requires interactive input.")
        print(f"  It will prompt for GTFS input folder; output folder: {settings.GTFS_FILTER_VERSION}")
        script_path = os.path.join(paths.MAIN, 'services_filter_gtfs.py')
        result = subprocess.run([sys.executable, script_path], cwd=paths.MAIN)
        if result.returncode != 0:
            print(f"  WARNING: services_filter_gtfs.py exited with code {result.returncode}.")
        else:
            loaded.append("GTFS filtered data")
            print(f"  GTFS filter complete.\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("-" * 80)
    print("  DATA PREPARATION SUMMARY")
    print("-" * 80)
    if loaded:
        print(f"  Loaded  : {', '.join(loaded)}")
    if skipped:
        print(f"  Skipped : {', '.join(skipped)}  (already on disk)")
    print("-" * 80 + "\n")

    runtimes["Phase 2: Data Preparation"] = time.time() - st


# ═══════════════════════════════════════════════════════════════════════════════
# Runtime writer
# ═══════════════════════════════════════════════════════════════════════════════

def _save_runtimes(runtimes: dict, filename: str) -> None:
    """Append phase runtimes to the run log file started by Phase 1."""
    total_time = sum(runtimes.values())
    # Use append mode: the config header was written at end of Phase 1
    with open(filename, 'a', encoding='utf-8') as f:
        f.write("PHASE RUNTIMES\n")
        f.write("=" * 80 + "\n\n")
        for part, runtime in runtimes.items():
            mins = int(runtime // 60)
            secs = int(runtime % 60)
            f.write(f"{part:.<60} {mins}m {secs}s ({runtime:.2f}s)\n")
        f.write("\n" + "=" * 80 + "\n")
        total_mins = int(total_time // 60)
        total_secs = int(total_time % 60)
        f.write(f"{'TOTAL TIME':.<60} {total_mins}m {total_secs}s ({total_time:.2f}s)\n")
        f.write("=" * 80 + "\n")
    print(f"Runtimes saved to: {filename}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def _should_run(phase: str) -> bool:
    """Return True if this phase should execute given RUN_MODE."""
    if RUN_MODE == 'all':
        return True
    return phase in RUN_MODE


def infrascanrail_new():
    """New InfraScanRail pipeline orchestrator.

    Runs all phases or a subset defined by RUN_MODE at the top of this file.
    Each phase function can also be called directly for isolated debugging:
        phase_1_initialisation(runtimes)
        phase_2_data_preparation(sa_boundary, sa_buffer, ca_boundary, ca_buffer, runtimes)
    """
    os.chdir(paths.MAIN)
    warnings.filterwarnings("ignore")
    runtimes = {}

    sa_boundary = sa_buffer = ca_boundary = ca_buffer = None

    # Phase 1: Initialisation
    if _should_run('1'):
        sa_boundary, sa_buffer, ca_boundary, ca_buffer = phase_1_initialisation(runtimes)
    else:
        # Load area geometries from disk so later phases have them available
        _sa  = gpd.read_file(os.path.join(paths.MAIN, paths.STUDY_AREA_BOUNDARY_GPKG))
        _sab = gpd.read_file(os.path.join(paths.MAIN, paths.STUDY_AREA_BUFFER_GPKG))
        _ca  = gpd.read_file(os.path.join(paths.MAIN, paths.CATCHMENT_AREA_BOUNDARY_GPKG))
        _cab = gpd.read_file(os.path.join(paths.MAIN, paths.CATCHMENT_AREA_BUFFER_GPKG))
        sa_boundary = _sa.geometry.iloc[0]
        sa_buffer   = _sab.geometry.iloc[0]
        ca_boundary = _ca.geometry.iloc[0]
        ca_buffer   = _cab.geometry.iloc[0]

    # Phase 2: Data Preparation
    if _should_run('2'):
        phase_2_data_preparation(sa_boundary, sa_buffer, ca_boundary, ca_buffer, runtimes)

    # Phase 3A: Infrastructure Network Building  (Workflow 2 - not yet implemented)
    # if _should_run('3A'):
    #     phase_3a_infrastructure(runtimes)

    # Phase 3B: Services Network Building        (Workflow 2 - not yet implemented)
    # if _should_run('3B'):
    #     phase_3b_services(runtimes)

    # Phase 3C: Capacity Analysis                (Workflow 3 - not yet implemented)
    # if _should_run('3C'):
    #     phase_3c_capacity(runtimes)

    if runtimes:
        _save_runtimes(runtimes, 'report_new.txt')

    print("\n" + "=" * 80)
    phases_run = RUN_MODE if RUN_MODE == 'all' else ', '.join(RUN_MODE)
    print(f"PIPELINE COMPLETE  (phases: {phases_run})")
    print("=" * 80)
    if runtimes:
        total = sum(runtimes.values())
        print(f"\nTotal runtime: {int(total // 60)}m {int(total % 60)}s")
        print(f"Runtimes saved to: report_new.txt")
    print()


if __name__ == '__main__':
    infrascanrail_new()
