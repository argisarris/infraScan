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
# Phase dependencies:
#   '2'  requires '1' outputs on disk (study/catchment area gpkgs, BAV raw, GTFS)
#   '3A' requires '2' outputs on disk (Raw_ZH infra gpkgs)
#   '3B' requires '3A' outputs on disk (named infra version) + '2' GTFS output
#   '3C' requires '3B' outputs on disk (enhanced infra, projected services)
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
        print(f"  1) Switch to Build_New")
        print(f"  2) Abort")
        ans = input(f"  Select [1]: ").strip() or '1'
        if ans == '1':
            print(f"  Switching INFRA_VERSION to Build_New.")
            settings.INFRA_VERSION = 'Build_New'
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
        print(f"  1) Switch to Build_New")
        print(f"  2) Abort")
        ans = input(f"  Select [1]: ").strip() or '1'
        if ans == '1':
            print(f"  Switching SVC_VERSION to Build_New.")
            settings.SVC_VERSION             = 'Build_New'
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
# Phase 3A -Infrastructure Network Build
# ═══════════════════════════════════════════════════════════════════════════════

def phase_3a_infrastructure(runtimes: dict) -> None:
    """Build the macroscopic Base network if missing and resolve the infra version.

    For Build_New: launches infrabuild_version_manager interactively so the user
    can create the named scenario version before projection and enhancement.
    For named versions: validates the version exists on disk.

    Args:
        runtimes: Dict tracking phase execution times.
    """
    print("\n" + "=" * 80)
    print("PHASE 3A: INFRASTRUCTURE NETWORK BUILD")
    print("=" * 80 + "\n")
    st = time.time()

    base_name = f'Base_{settings.CATCHMENT_CANTON_ABBREV}'
    infra_v   = PIPELINE_CONFIG.infra_version  # resolved by Phase 1

    # ── Step 3A.1: Base network ───────────────────────────────────────────────
    print("--- Step 3A.1: Base Network ---\n")
    if paths.infra_version_exists(base_name):
        print(f"  Base network '{base_name}' found — skipping build.")
    else:
        print(f"  '{base_name}' not found — building macroscopic Base "
              f"from '{settings.INFRA_RAW_VERSION}' ...")
        from infrabuild_network_builder import run_build_base
        raw_dir = paths.get_infra_raw_dir(settings.INFRA_RAW_VERSION)
        out_dir = paths.get_infra_version_dir(base_name)
        run_build_base(raw_dir=raw_dir, output_dir=out_dir)
        print(f"  Base network built → {out_dir}")
    print()

    # ── Step 3A.2: Infrastructure version ─────────────────────────────────────
    print("--- Step 3A.2: Infrastructure Version ---\n")
    if settings.INFRA_VERSION == 'Build_New':
        print(f"  Opening Infrastructure Version Manager to create '{infra_v}'.")
        print(f"  ┌─ INSTRUCTIONS ──────────────────────────────────────────────────")
        print(f"  │  Create from '{base_name}' or copy an existing named version.")
        print(f"  │  Version name : {infra_v}")
        print(f"  │  Edit nodes/segments as needed, then Save and close.")
        print(f"  └─────────────────────────────────────────────────────────────────\n")
        script_path = os.path.join(paths.MAIN, 'infrabuild_version_manager.py')
        result = subprocess.run([sys.executable, script_path], cwd=paths.MAIN)
        if result.returncode != 0:
            print(f"  WARNING: infrabuild_version_manager.py exited with code "
                  f"{result.returncode}.")
        elif not paths.infra_version_exists(infra_v):
            print(f"  WARNING: Version '{infra_v}' not found on disk after version "
                  f"manager exited.")
            print(f"  Ensure you saved before closing (Phase 3 → Save and close).")
        else:
            print(f"  Infrastructure version '{infra_v}' created successfully.")
    else:
        if paths.infra_version_exists(infra_v):
            print(f"  Infrastructure version '{infra_v}' found on disk.")
        else:
            print(f"  ERROR: Infrastructure version '{infra_v}' not found.")
            print(f"  Phase 1 should have caught this — re-run from Phase 1.")
            raise SystemExit("Aborted: infrastructure version not found in Phase 3A.")
    print()

    # ── Step 3A.3: Pre-enhancement network visualisation ──────────────────────
    # Skipped for _enhanced versions (no pre-enhancement state to capture).
    if not infra_v.endswith('_enhanced'):
        print("--- Step 3A.3: Pre-Enhancement Network Visualisation ---\n")
        if not settings.PLOT_INFRA:
            print(f"  PLOT_INFRA = False — skipping infrastructure plots.")
        elif not paths.infra_version_exists(infra_v):
            print(f"  Version '{infra_v}' not on disk — skipping plots.")
        else:
            from infrabuild_network_builder import (
                load_version,
                build_networkx_graph,
                plot_infrastructure_canonical,
                plot_gauge_map,
                plot_electrification_map,
                plot_speed_map,
                NetworkData,
                _build_infra_qgz,
            )
            import matplotlib.pyplot as plt
            print(f"  Building QGIS project and plots for '{infra_v}' ...")

            nodes, segments = load_version(infra_v)
            G = build_networkx_graph(nodes, segments)

            _ca_bdry_path = os.path.join(paths.MAIN, paths.CATCHMENT_AREA_BOUNDARY_GPKG)
            _sa_bdry_path = os.path.join(paths.MAIN, paths.STUDY_AREA_BOUNDARY_GPKG)
            ca_bdry_gdf = gpd.read_file(_ca_bdry_path) if os.path.isfile(_ca_bdry_path) else None
            sa_bdry_gdf = gpd.read_file(_sa_bdry_path) if os.path.isfile(_sa_bdry_path) else None

            def _extent_from_gdf(gdf, margin_m: int = 2000):
                if gdf is None:
                    return None
                b = gdf.total_bounds
                return (b[0] - margin_m, b[2] + margin_m, b[1] - margin_m, b[3] + margin_m)

            ca_ext = _extent_from_gdf(ca_bdry_gdf)
            sa_ext = _extent_from_gdf(sa_bdry_gdf)

            net_ca = NetworkData(nodes=nodes, segments=segments, graph=G,
                                 version=infra_v, boundary=ca_bdry_gdf)
            net_sa = NetworkData(nodes=nodes, segments=segments, graph=G,
                                 version=infra_v, boundary=sa_bdry_gdf)

            version_dir = Path(paths.get_infra_version_dir(infra_v))
            qgz_path = str(version_dir / f'{infra_v}.qgz')
            _build_infra_qgz(qgz_path, version_dir)
            print(f"  QGIS project written: {qgz_path}")

            plot_dir = Path(paths.MAIN) / paths.INFRASTRUCTURE_PLOTS_DIR / infra_v
            plot_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Generating plots → {plot_dir}")

            _plots = [
                (plot_infrastructure_canonical, net_ca, ca_ext,
                 'ca_infrastructure.pdf', {'is_catchment': True, 'show_labels': False}),
                (plot_gauge_map,               net_ca, ca_ext,
                 'ca_gauge.pdf',          {'is_catchment': True}),
                (plot_electrification_map,     net_ca, ca_ext,
                 'ca_electrification.pdf', {'is_catchment': True}),
                (plot_speed_map,               net_ca, ca_ext,
                 'ca_speed.pdf',           {'is_catchment': True}),
                (plot_infrastructure_canonical, net_sa, sa_ext,
                 'sa_infrastructure.pdf', {'show_outside': True}),
                (plot_gauge_map,               net_sa, sa_ext,
                 'sa_gauge.pdf',          {'show_outside': True}),
                (plot_electrification_map,     net_sa, sa_ext,
                 'sa_electrification.pdf', {'show_outside': True}),
                (plot_speed_map,               net_sa, sa_ext,
                 'sa_speed.pdf',           {'show_outside': True}),
            ]
            for _fn, _net, _ext, _fname, _kw in _plots:
                print(f"    {_fname} ...")
                _fig = _fn(_net, extent=_ext, output_path=plot_dir / _fname, **_kw)
                plt.close(_fig)

            print(f"  Plots complete.\n")

    runtimes["Phase 3A: Infrastructure Network Build"] = time.time() - st


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3B -Services Network Build
# ═══════════════════════════════════════════════════════════════════════════════

def phase_3b_services(
    sa_boundary,
    ca_boundary,
    runtimes: dict,
) -> None:
    """Build services network, project onto infra, enhance, and plot.

    Handles the full integration pipeline:
      3B.1  Build Unprojected services network (if missing)
      3B.2  Services version manager (Build_New only)
      3B.3  Project services onto infrastructure graph
      3B.4  Enhance infrastructure with GTFS travel times
      3B.5  Build QGIS project and infrastructure plots

    Args:
        sa_boundary: Study area polygon (Shapely) — held for future use.
        ca_boundary: Catchment area polygon (Shapely) — held for future use.
        runtimes:    Dict tracking phase execution times.
    """
    print("\n" + "=" * 80)
    print("PHASE 3B: SERVICES NETWORK BUILD")
    print("=" * 80 + "\n")
    st = time.time()

    infra_v = PIPELINE_CONFIG.infra_version  # e.g. 'AS_2026_ZH' or 'AS_2026_ZH_enhanced'
    svc_v   = PIPELINE_CONFIG.svc_version    # e.g. 'SVC2026_ZH_S18'

    already_enhanced = infra_v.endswith('_enhanced')
    base_infra_v     = infra_v.removesuffix('_enhanced')
    enhanced_v       = f'{base_infra_v}_enhanced'

    # ── Step 3B.1: Services network build ────────────────────────────────────
    print("--- Step 3B.1: Services Network Build ---\n")
    if settings.SVC_VERSION == 'Build_New':
        # For Build_New: check if any complete network already exists that can be
        # copied in the version manager; if not, build one from Phase 2 GTFS data.
        need_rail   = not paths.any_svc_rail_exists()
        need_feeder = (
            settings.CATCHMENT_METHOD == 'PT_Feeder'
            and not paths.any_svc_feeder_exists()
        )
    else:
        need_rail   = not paths.svc_version_exists(svc_v)
        need_feeder = (
            settings.CATCHMENT_METHOD == 'PT_Feeder'
            and not paths.svc_feeder_exists(svc_v)
        )

    if not need_rail and not need_feeder:
        print(f"  Services network found — skipping build.")
    else:
        missing = []
        if need_rail:
            missing.append("rail network")
        if need_feeder:
            missing.append("PT-feeder network")
        print(f"  Missing: {', '.join(missing)} — building from GTFS data ...")
        print(f"  ┌─ INSTRUCTIONS ──────────────────────────────────────────────────")
        print(f"  │  GTFS input folder : {settings.GTFS_FILTER_VERSION}")
        print(f"  │  Mode              : all  (builds rail + feeder)")
        print(f"  │  Output folder name: {svc_v}")
        print(f"  └─────────────────────────────────────────────────────────────────\n")
        script_path = os.path.join(paths.MAIN, 'services_network_builder.py')
        result = subprocess.run([sys.executable, script_path], cwd=paths.MAIN)
        if result.returncode != 0:
            print(f"  WARNING: services_network_builder.py exited with code "
                  f"{result.returncode}.")
        else:
            print(f"  Services network build complete.\n")
    print()

    # ── Step 3B.2: Services version manager (Build_New only) ─────────────────
    if settings.SVC_VERSION == 'Build_New':
        print("--- Step 3B.2: Services Version Manager ---\n")
        print(f"  Opening services_version_manager to create/edit service scenario.")
        print(f"  ┌─ INSTRUCTIONS ──────────────────────────────────────────────────")
        print(f"  │  Build from an existing network or copy an existing service version.")
        print(f"  │  Version name     : {svc_v}")
        print(f"  │  Infra version    : {infra_v}")
        print(f"  │  Edit lines/segments, then Save.")
        print(f"  └─────────────────────────────────────────────────────────────────\n")
        script_path = os.path.join(paths.MAIN, 'services_version_manager.py')
        result = subprocess.run(
            [sys.executable, script_path, '--infra-version', infra_v],
            cwd=paths.MAIN,
        )
        if result.returncode != 0:
            print(f"  WARNING: services_version_manager.py exited with code "
                  f"{result.returncode}.")
        else:
            print(f"  Services version manager complete.\n")
        print()

    # ── Step 3B.3: Service projection ─────────────────────────────────────
    # Always project onto infra_v (the active version). When already_enhanced,
    # infra_v is the enhanced network which is the correct routing base.
    # When not enhanced, infra_v == base_infra_v — same result, enhancement
    # will then read the projected data from the base_infra_v path.
    print("--- Step 3B.3: Service Projection ---\n")
    if paths.svc_projected_exists(svc_v, infra_v):
        print(f"  Projected services for '{svc_v}' on '{infra_v}' found "
              f"— skipping.")
    else:
        print(f"  Projecting '{svc_v}' onto '{infra_v}' (non-interactive) ...")
        script_path = os.path.join(paths.MAIN, 'services_service_projection.py')
        _proj_cmd = [sys.executable, script_path,
                     '--svc-version',   svc_v,
                     '--infra-version', infra_v]
        if settings.CATCHMENT_METHOD != 'PT_Feeder':
            _proj_cmd.append('--no-feeder-plots')
        result = subprocess.run(_proj_cmd, cwd=paths.MAIN)
        if result.returncode != 0:
            print(f"  WARNING: services_service_projection.py exited with code "
                  f"{result.returncode}.")
        else:
            print(f"  Service projection complete.\n")
    print()

    # ── Step 3B.4: Infrastructure enhancement ────────────────────────────
    print("--- Step 3B.4: Infrastructure Enhancement ---\n")
    if already_enhanced:
        print(f"  Infrastructure version '{infra_v}' is already enhanced.")
        print(f"  Skipping enhancement — network assumed complete.")
        print(f"  To re-calibrate, set INFRA_VERSION = '{base_infra_v}' and re-run.\n")
    else:
        mode_hint = "(extend — fills null slots)" if paths.infra_version_exists(enhanced_v) \
                    else "(initial — full calibration)"
        print(f"  Enhancing '{base_infra_v}' with GTFS-calibrated travel times "
              f"{mode_hint} (non-interactive) ...")
        script_path = os.path.join(
            paths.MAIN, 'infrabuild_infrastructure_enhancement.py'
        )
        result = subprocess.run(
            [sys.executable, script_path,
             '--infra-version',  base_infra_v,
             '--svc-version',    svc_v + '_network',
             '--enhanced-name',  enhanced_v],
            cwd=paths.MAIN,
        )
        if result.returncode != 0:
            print(f"  WARNING: infrabuild_infrastructure_enhancement.py exited with "
                  f"code {result.returncode}.")
        else:
            print(f"  Enhancement complete.\n")
            # ── Write enhancement stats to report ─────────────────────────────
            _enh_segs_path = os.path.join(
                paths.get_infra_version_dir(enhanced_v), 'segments.gpkg'
            )
            if os.path.isfile(_enh_segs_path):
                _enh_segs = gpd.read_file(_enh_segs_path)
                _n_total      = len(_enh_segs)
                _n_tt         = int(_enh_segs['TT_Stopping'].notna().sum()) \
                                if 'TT_Stopping' in _enh_segs.columns else 0
                _n_gtfs       = int((_enh_segs.get('speed_source', '') == 'GTFS').sum())
                _n_osm        = int((_enh_segs.get('speed_source', '') == 'OSM').sum())
                _n_infra      = int((_enh_segs.get('speed_source', '') == 'infra').sum())
                _n_feeder     = int(
                    _enh_segs['Segment_ID'].astype(str).str.startswith('feeder_').sum()
                ) if 'Segment_ID' in _enh_segs.columns else 0
                _rt_file = os.path.join(paths.MAIN, 'report_new.txt')
                with open(_rt_file, 'a', encoding='utf-8') as _f:
                    _f.write(f"\n--- Enhancement Stats: {enhanced_v} ---\n")
                    _f.write(f"  Total segments       : {_n_total}\n")
                    _f.write(f"  TT_Stopping filled   : {_n_tt} "
                             f"({_n_tt / _n_total * 100:.1f}%)\n")
                    _f.write(f"  speed_source = GTFS  : {_n_gtfs}\n")
                    _f.write(f"  speed_source = OSM   : {_n_osm}\n")
                    _f.write(f"  speed_source = infra : {_n_infra}\n")
                    _f.write(f"  Feeder-derived segs  : {_n_feeder}\n")
        print()

    # ── Update resolved infra version ─────────────────────────────────────
    PIPELINE_CONFIG.infra_version = enhanced_v
    print(f"  Active infrastructure version updated to: {enhanced_v}\n")

    # ── Step 3B.5: QGIS project + plots ──────────────────────────────────────
    print("--- Step 3B.5: Network Visualisation ---\n")
    final_infra_v = PIPELINE_CONFIG.infra_version

    if not settings.PLOT_INFRA:
        print(f"  PLOT_INFRA = False — skipping infrastructure plots.")
    else:
        from infrabuild_network_builder import (
            load_version,
            build_networkx_graph,
            plot_infrastructure_canonical,
            plot_infrastructure_diff,
            plot_gauge_map,
            plot_electrification_map,
            plot_speed_map,
            NetworkData,
            _build_infra_qgz,
        )
        import matplotlib.pyplot as plt
        print(f"  Building QGIS project and plots for '{final_infra_v}' ...")

        nodes, segments = load_version(final_infra_v)
        G = build_networkx_graph(nodes, segments)

        # Load boundary GeoDataFrames from disk (NetworkData.boundary is a GDF)
        _ca_bdry_path = os.path.join(paths.MAIN, paths.CATCHMENT_AREA_BOUNDARY_GPKG)
        _sa_bdry_path = os.path.join(paths.MAIN, paths.STUDY_AREA_BOUNDARY_GPKG)
        ca_bdry_gdf = gpd.read_file(_ca_bdry_path) if os.path.isfile(_ca_bdry_path) else None
        sa_bdry_gdf = gpd.read_file(_sa_bdry_path) if os.path.isfile(_sa_bdry_path) else None

        def _extent_from_gdf(gdf, margin_m: int = 2000):
            if gdf is None:
                return None
            b = gdf.total_bounds
            return (b[0] - margin_m, b[2] + margin_m, b[1] - margin_m, b[3] + margin_m)

        ca_ext = _extent_from_gdf(ca_bdry_gdf)
        sa_ext = _extent_from_gdf(sa_bdry_gdf)

        net_ca = NetworkData(nodes=nodes, segments=segments, graph=G,
                             version=final_infra_v, boundary=ca_bdry_gdf)
        net_sa = NetworkData(nodes=nodes, segments=segments, graph=G,
                             version=final_infra_v, boundary=sa_bdry_gdf)

        # QGIS project
        version_dir = Path(paths.get_infra_version_dir(final_infra_v))
        qgz_path = str(version_dir / f'{final_infra_v}.qgz')
        _build_infra_qgz(qgz_path, version_dir)
        print(f"  QGIS project written: {qgz_path}")

        # Plots
        plot_dir = Path(paths.MAIN) / paths.INFRASTRUCTURE_PLOTS_DIR / final_infra_v
        plot_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Generating plots → {plot_dir}")

        _plots = [
            (plot_infrastructure_canonical, net_ca, ca_ext,
             'ca_infrastructure.pdf', {'is_catchment': True, 'show_labels': False}),
            (plot_gauge_map,               net_ca, ca_ext,
             'ca_gauge.pdf',          {'is_catchment': True}),
            (plot_electrification_map,     net_ca, ca_ext,
             'ca_electrification.pdf', {'is_catchment': True}),
            (plot_speed_map,               net_ca, ca_ext,
             'ca_speed.pdf',           {'is_catchment': True}),
            (plot_infrastructure_canonical, net_sa, sa_ext,
             'sa_infrastructure.pdf', {'show_outside': True}),
            (plot_gauge_map,               net_sa, sa_ext,
             'sa_gauge.pdf',          {'show_outside': True}),
            (plot_electrification_map,     net_sa, sa_ext,
             'sa_electrification.pdf', {'show_outside': True}),
            (plot_speed_map,               net_sa, sa_ext,
             'sa_speed.pdf',           {'show_outside': True}),
        ]
        for _fn, _net, _ext, _fname, _kw in _plots:
            print(f"    {_fname} ...")
            _fig = _fn(_net, extent=_ext, output_path=plot_dir / _fname, **_kw)
            plt.close(_fig)

        # Diff plot vs original (base_infra_v) — always generated for enhanced versions
        _diff_base = base_infra_v if not already_enhanced else \
                     final_infra_v.removesuffix('_enhanced')
        if paths.infra_version_exists(_diff_base):
            print(f"    diff vs '{_diff_base}' ...")
            _base_nodes, _base_segs = load_version(_diff_base)
            _base_G = build_networkx_graph(_base_nodes, _base_segs)
            _net_base_ca = NetworkData(nodes=_base_nodes, segments=_base_segs,
                                       graph=_base_G, version=_diff_base,
                                       boundary=ca_bdry_gdf)
            _fig = plot_infrastructure_diff(
                _net_base_ca, net_ca,
                extent=ca_ext,
                output_path=plot_dir / f'ca_diff_vs_{_diff_base}.pdf',
                is_catchment=True,
            )
            plt.close(_fig)
            _net_base_sa = NetworkData(nodes=_base_nodes, segments=_base_segs,
                                       graph=_base_G, version=_diff_base,
                                       boundary=sa_bdry_gdf)
            _fig = plot_infrastructure_diff(
                _net_base_sa, net_sa,
                extent=sa_ext,
                output_path=plot_dir / f'sa_diff_vs_{_diff_base}.pdf',
                show_outside=True,
            )
            plt.close(_fig)

        print(f"  Plots complete.\n")

    runtimes["Phase 3B: Services Network Build"] = time.time() - st


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

    # Phase 3A: Infrastructure Network Building
    if _should_run('3A'):
        phase_3a_infrastructure(runtimes)

    # Phase 3B: Services Network Building
    if _should_run('3B'):
        phase_3b_services(sa_boundary, ca_boundary, runtimes)

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
