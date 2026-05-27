"""catchment_OD_preparation.py
Last modified: 2026-05-05

Station-pair OD matrix preparation for two methods (PT-feeder, Municipal)
across three time windows (AM peak, off-peak, all-day). Both methods consume
the same scaled communal OD; differ only in cell-to-station attribution.

Public entry point: prepare_all_od_matrices(use_cache: bool) -> None

Pipeline (W3):
  1. Load catchment boundary, rail stations (within boundary), name lookup.
  2. Load daily communal OD via scoring.GetOevDemandPerCommune(tau=1).
  3. Apply a single 2018→2023 catchment-wide population scale factor.
  4. PT-feeder branch: cell-level disaggregation → station re-aggregation.
  5. Municipal branch (Phase 3): commune→station 1:1 mapping (Phase 3).
  6. Emit per (method, time-window) name-keyed station OD CSV.
  7. Print conservation diagnostics for each method.
"""

import os
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import paths
import settings
import scoring
import cost_parameters as cp
import catchment_base
import catchment_allocate

_CODEBASE_CRS = 'EPSG:2056'

# Diagnostic plots — updated to the versioned path by prepare_all_od_matrices()
# when catchment_base.setup_versioned_dirs() has been called beforehand.
OD_COMPARISON_PLOT_DIR = catchment_base.OD_COMPARISON_PLOT_DIR


# ===============================================================================
# PUBLIC ENTRY POINT
# ===============================================================================

def prepare_all_od_matrices(use_cache: bool = False, svc_version: str = '') -> None:
    """Generate station-pair OD matrices for both methods × three time windows.

    Args:
        use_cache:   If True, skip writing any CSV that already exists at its
                     target path. The long-format build is still done (cheap).
        svc_version: Service version folder name, e.g. 'SVC2026_ZH_S18_network'.
                     Required when running standalone; inferred automatically when
                     called after catchment_allocate.get_catchment().

    Writes (under paths.MAIN):
        data/traffic_flow/od/rail/pt_feeder/od_matrix_stations_pt_feeder_{am_peak,off_peak,all_day}.csv
        data/traffic_flow/od/rail/municipal/od_matrix_stations_municipal_{am_peak,off_peak,all_day}.csv
    """
    global OD_COMPARISON_PLOT_DIR

    if svc_version:
        catchment_base.setup_versioned_dirs(svc_version)
        catchment_allocate._RAIL_BASE = os.path.join(
            paths.RAIL_LINES_DIR, svc_version, paths.SERVICES_UNPROJECTED_SUBDIR)

    OD_COMPARISON_PLOT_DIR = catchment_base.OD_COMPARISON_PLOT_DIR

    print("\n=== Preparing station-pair OD matrices (W3) ===")

    boundary      = catchment_base._load_catchment_boundary()
    rail_stations = catchment_allocate._load_rail_stations(boundary, 'all', buffer=0)
    name_lookup   = _build_station_name_lookup(rail_stations)

    pop_factor    = _compute_pop_scale_factor(boundary)

    communal_od   = _load_communal_od()
    communal_od['wert'] *= pop_factor
    print(f"  Communal OD scaled by 2018→2023 factor {pop_factor:.4f} "
          f"(applied uniformly to every OD pair)")

    # --- PT-feeder branch ---
    pt_long, pt_orig_weights = _run_pt_feeder_branch(communal_od)
    _diagnose_conservation(communal_od, pt_long, pt_orig_weights, 'PT-Feeder')

    # --- Municipal branch ---
    muni_long = _run_municipal_branch(communal_od)
    _diagnose_conservation(communal_od, muni_long, None, 'Municipal')

    # --- Emit CSVs per (method, window) ---
    targets = [
        (cp.TAU_AM_PEAK_SHARE,  'am_peak',
         paths.OD_STATIONS_PT_FEEDER_AM_PEAK_PATH,
         paths.OD_STATIONS_MUNICIPAL_AM_PEAK_PATH),
        (cp.TAU_OFF_PEAK_SHARE, 'off_peak',
         paths.OD_STATIONS_PT_FEEDER_OFF_PEAK_PATH,
         paths.OD_STATIONS_MUNICIPAL_OFF_PEAK_PATH),
        (cp.TAU_ALL_DAY_SHARE,  'all_day',
         paths.OD_STATIONS_PT_FEEDER_ALL_DAY_PATH,
         paths.OD_STATIONS_MUNICIPAL_ALL_DAY_PATH),
    ]
    print("\n  Writing CSVs ...")
    for tau, suffix, pt_path, muni_path in targets:
        if use_cache and (Path(paths.MAIN) / pt_path).exists():
            print(f"    cached: {pt_path}")
        else:
            _apply_window_scaling(pt_long, tau, name_lookup, pt_path,
                                  label=f'PT-feeder {suffix}')
        if use_cache and (Path(paths.MAIN) / muni_path).exists():
            print(f"    cached: {muni_path}")
        else:
            _apply_window_scaling(muni_long, tau, name_lookup, muni_path,
                                  label=f'Municipal {suffix}')

    # --- Diagnostic plots (uses the unscaled τ=1 long-format OD) ---
    _plot_od_diagnostics(pt_long, muni_long, rail_stations, boundary,
                          name_lookup)

    print("\n=== W3 OD matrices done ===")


# Backwards-compat alias — DEPRECATED. New callers should use
# prepare_all_od_matrices, which produces six matrices instead of one.
prepare_od_pt_feeder_a = prepare_all_od_matrices


# ===============================================================================
# NEW SHARED HELPERS (W3)
# ===============================================================================

def _compute_pop_scale_factor(boundary) -> float:
    """Return total_pop_2023_in_catchment / total_pop_{POPULATION_BASE_YEAR}_in_catchment.

    Single global scalar applied uniformly to every OD pair. The base year
    is settings.POPULATION_BASE_YEAR; the current reference is the
    2023 federal population grid cached by catchment_base.main().

    Base year source : catchment_base.load_commune_pop() using
                       POPULATION_CANTON_ZH_XLSX (commune level, 1962-2025)
    2023 source      : catchment_base.load_population_grid_cached() (cell level,
                       boundary-filtered, from data/Catchment_Area/ cache)
    """
    base_year = settings.POPULATION_BASE_YEAR
    print(f"  Computing {base_year}->2023 population scale factor ...")

    # Commune BFS codes that intersect the catchment boundary
    commune_gdf, bfs_col = _load_commune_boundaries()
    bnd_gdf = gpd.GeoDataFrame(geometry=[boundary], crs=_CODEBASE_CRS)
    in_bnd_communes = gpd.sjoin(commune_gdf, bnd_gdf, predicate='intersects',
                                how='inner')
    in_bnd_bfs = set(
        pd.to_numeric(in_bnd_communes[bfs_col], errors='coerce')
          .dropna().astype(int).tolist()
    )

    # Base year population from the new cantonal xlsx via catchment_base
    pop_base_series = catchment_base.load_commune_pop(base_year)
    pop_base = float(
        pop_base_series[pop_base_series.index.isin(in_bnd_bfs)].sum()
    )

    # 2023 cell-level grid (boundary-filtered, from catchment_base.main() cache)
    pop_grid_2023 = catchment_base.load_population_grid_cached()
    pop_2023 = float(pop_grid_2023['NUMMER'].sum())

    if pop_base <= 0:
        print(f"    WARNING: {base_year} population total is zero -- using factor=1.0")
        return 1.0

    factor = pop_2023 / pop_base
    print(f"    Population {base_year} in catchment: {pop_base:>12,.0f}")
    print(f"    Population 2023 in catchment:  {pop_2023:>12,.0f}")
    print(f"    Scale factor {base_year}->2023:         {factor:>12.4f}")
    return factor


def _build_station_name_lookup(rail_stations) -> dict:
    """Return dict[id_point_str → readable_name].

    Disambiguates collisions: if a stop_name appears more than once, append
    `(id_point)` so the CSV index/columns remain unique.
    """
    df = rail_stations[['id_point', 'stop_name']].copy()
    df['id_point']  = df['id_point'].astype(str)
    df['stop_name'] = df['stop_name'].fillna('').astype(str).str.strip()
    df = df.drop_duplicates('id_point')

    name_counts = df['stop_name'].value_counts()
    duplicated = set(name_counts[name_counts > 1].index)

    if duplicated:
        print(f"  Station-name lookup: {len(duplicated)} name collision(s); "
              f"appending (id_point) to disambiguate.")

    lookup = {}
    for _, row in df.iterrows():
        sid  = row['id_point']
        name = row['stop_name'] or sid
        if name in duplicated:
            name = f"{name} ({sid})"
        lookup[sid] = name
    return lookup


def _apply_window_scaling(long_df: pd.DataFrame, tau_window: float,
                          name_lookup: dict, output_path: str,
                          label: str = '') -> None:
    """Scale a long-format station-pair OD by tau, pivot to wide, rename axes
    to station names via `name_lookup`, and write CSV to paths.MAIN / output_path.
    """
    if long_df is None or long_df.empty:
        print(f"    ({label}): no data — skipping {output_path}")
        return

    scaled = long_df.copy()
    scaled['trips'] = scaled['trips'] * tau_window
    scaled = scaled[scaled['trips'] > 0].copy()

    matrix = scaled.pivot_table(
        index='origin_station_id',
        columns='dest_station_id',
        values='trips',
        aggfunc='sum',
        fill_value=0.0,
    )
    matrix.index   = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    matrix = matrix.rename(index=name_lookup, columns=name_lookup)
    matrix.index.name   = 'origin'
    matrix.columns.name = 'destination'

    out_full = Path(paths.MAIN) / output_path
    out_full.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(out_full, encoding='utf-8-sig')
    print(f"    Saved → {out_full}")
    print(f"      {label}: {matrix.shape[0]}×{matrix.shape[1]} stations, "
          f"total trips = {matrix.values.sum():,.1f}  (τ={tau_window})")


def _diagnose_conservation(communal_od: pd.DataFrame,
                           station_long_df: pd.DataFrame,
                           orig_weights,
                           method_label: str) -> None:
    """Print conservation diagnostic for a method.

    Always reports total communal vs total station-pair trips. When
    `orig_weights` is provided (PT-feeder), additionally lists the top-5
    origin communes by share-loss (cells with no PT assignment).
    """
    total_communal = float(communal_od['wert'].sum())
    total_station  = float(station_long_df['trips'].sum())
    delta          = total_communal - total_station
    pct            = 100.0 * delta / max(total_communal, 1e-9)

    print(f"\n  Conservation diagnostic [{method_label}]:")
    print(f"    Communal OD total: {total_communal:>14,.1f}")
    print(f"    Station OD total:  {total_station:>14,.1f}")
    print(f"    Residual:          {delta:>+14,.1f}  ({pct:+.2f}%)")

    if orig_weights is None or len(orig_weights) == 0:
        if method_label.lower().startswith('munici'):
            print(f"    Municipal: 1:1 commune→station mapping; residual = "
                  f"trips for communes outside the assignment lookup.")
        return

    # Per-commune origin-side weight loss (PT-feeder)
    bfs_key = 'BFS' if 'BFS' in orig_weights.columns else 'quelle_code'
    sum_w = orig_weights.groupby(bfs_key)['orig_weight'].sum()
    losses = (1.0 - sum_w).clip(lower=0.0)
    losses = losses[losses > 1e-6].sort_values(ascending=False)
    if losses.empty:
        print(f"    PT-feeder: every origin commune fully covered "
              f"(no cells with NoPT assignment).")
        return

    row_total_per_bfs = communal_od.groupby('quelle_code')['wert'].sum()
    print(f"    Top-5 origin communes by share loss "
          f"(cells with no PT assignment):")
    for bfs, loss in losses.head(5).items():
        bfs_int   = int(bfs)
        row_total = float(row_total_per_bfs.get(bfs_int, 0.0))
        lost      = row_total * float(loss)
        print(f"      BFS {bfs_int:>5}  loss={float(loss)*100:5.1f}%  "
              f"row_total={row_total:>10,.0f}  lost≈{lost:>9,.1f}")


def _load_municipal_assignment() -> pd.DataFrame:
    """Read the municipal commune→station lookup written by
    catchment_allocate._run_municipal_method.

    Returns:
        DataFrame[BFS_NR (int), station_id (int), station_name (str)].
        Communes without a valid station assignment are dropped.
    """
    path = os.path.join(paths.MAIN, catchment_base.MUNICIPAL_DATA_DIR, 'station_assignment.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Municipal commune→station lookup not found at {path}. "
            f"Run catchment_allocate.get_catchment() with the municipal "
            f"method first."
        )
    print(f"  Loading municipal commune→station assignment ...")
    df = pd.read_csv(path, encoding='utf-8-sig')
    keep = ['BFS_NR', 'station_id', 'station_name']
    for c in keep:
        if c not in df.columns:
            raise ValueError(
                f"station_assignment.csv missing column '{c}'. "
                f"Available: {list(df.columns)}"
            )
    df = df[keep].copy()
    df['BFS_NR']     = pd.to_numeric(df['BFS_NR'],     errors='coerce')
    df['station_id'] = pd.to_numeric(df['station_id'], errors='coerce')
    df = df.dropna(subset=['BFS_NR', 'station_id'])
    df = df[df['station_id'] > 0]
    df['BFS_NR']     = df['BFS_NR'].astype(int)
    df['station_id'] = df['station_id'].astype(int)
    print(f"    {len(df):,} communes with a valid station assignment")
    return df


def _run_municipal_branch(communal_od: pd.DataFrame) -> pd.DataFrame:
    """Map communal OD to station-pair OD via the municipal commune→station
    1:1 lookup. Origin commune → origin station, destination commune →
    destination station; trips are summed by (origin_station, dest_station).

    Returns:
        Long-format DataFrame: origin_station_id (int), dest_station_id (int),
        trips (float).
    """
    print("\n  --- Municipal branch ---")
    assign = _load_municipal_assignment()
    bfs_to_stn = dict(zip(assign['BFS_NR'].astype(int),
                          assign['station_id'].astype(int)))

    od = communal_od.copy()
    od['origin_station_id'] = od['quelle_code'].map(bfs_to_stn)
    od['dest_station_id']   = od['ziel_code']  .map(bfs_to_stn)

    n_in  = len(od)
    od    = od.dropna(subset=['origin_station_id', 'dest_station_id'])
    n_out = len(od)
    if n_out < n_in:
        print(f"    {n_in - n_out:,} of {n_in:,} OD pairs dropped "
              f"(commune outside the assignment lookup).")

    od['origin_station_id'] = od['origin_station_id'].astype(int)
    od['dest_station_id']   = od['dest_station_id'].astype(int)

    n_before_intra = len(od)
    od = od[od['origin_station_id'] != od['dest_station_id']]
    n_intra = n_before_intra - len(od)
    if n_intra:
        print(f"    {n_intra:,} intra-station pairs dropped (origin == dest station).")

    station_od = (od.groupby(['origin_station_id', 'dest_station_id'],
                             as_index=False)['wert'].sum()
                    .rename(columns={'wert': 'trips'}))
    station_od = station_od[station_od['trips'] > 0].copy()
    print(f"    {len(station_od):,} non-zero station OD pairs")
    return station_od


def _run_pt_feeder_branch(communal_od: pd.DataFrame) -> tuple:
    """Disaggregate→reaggregate communal OD to (rail-station, rail-station)
    pairs using separate cell-station assignments for each side (W4a).

    Origin (production) weights: population cells → pop assignment.
    Destination (attraction) weights: employment cells → empl assignment.

    Returns:
        (long_df, orig_weights)
        long_df:      DataFrame[origin_station_id, dest_station_id, trips]
        orig_weights: DataFrame[BFS, station_id, orig_weight] — for diagnostic.
    """
    print("\n  --- PT-feeder branch ---")
    commune_gdf, bfs_col = _load_commune_boundaries()
    cells_df             = _load_and_join_cells(commune_gdf, bfs_col)

    pop_assign  = _load_pt_feeder_assignment()
    empl_assign = _load_pt_feeder_empl_assignment()

    orig_weights = _compute_orig_weights(cells_df, pop_assign)
    dest_weights = _compute_dest_weights(cells_df, empl_assign)

    print(f"    Origin weights: {orig_weights['BFS'].nunique()} communes, "
          f"{orig_weights['station_id'].nunique()} stations")
    print(f"    Dest  weights: {dest_weights['BFS'].nunique()} communes, "
          f"{dest_weights['station_id'].nunique()} stations")

    long_df = _reaggregate_to_stations(communal_od, orig_weights, dest_weights)
    return long_df, orig_weights


# ===============================================================================
# EXISTING HELPERS (kept; one rename: _load_cell_station_assignment →
# _load_pt_feeder_assignment for clarity vs. the upcoming _load_municipal_*)
# ===============================================================================

def _load_commune_boundaries() -> tuple:
    """Load municipal boundary GeoPackage, detect BFS column, project to LV95.

    Returns:
        (muni_gdf, bfs_col_name)
    """
    print("  Loading commune boundaries ...")
    muni = gpd.read_file(paths.MUNICIPAL_BOUNDARIES_GPKG).to_crs(_CODEBASE_CRS)
    if 'objektart' in muni.columns:
        muni = muni[muni['objektart'] == 'Gemeindegebiet'].copy()

    bfs_col = None
    for candidate in ['BFS_NR', 'bfs_nr', 'BFS_NUMMER', 'bfs_nummer', 'GMDNR', 'gmdnr']:
        if candidate in muni.columns:
            bfs_col = candidate
            break
    if bfs_col is None:
        raise ValueError(
            f"No BFS column found in commune boundaries. "
            f"Available columns: {list(muni.columns)}"
        )

    muni[bfs_col] = pd.to_numeric(muni[bfs_col], errors='coerce')
    muni = muni[[bfs_col, 'geometry']].dropna(subset=[bfs_col])
    print(f"    {len(muni)} communes loaded (BFS column: '{bfs_col}')")
    return muni, bfs_col


def _load_communal_od() -> pd.DataFrame:
    """Load canton-level PT OD (tau=1 = full daily demand), drop intrazonal pairs.

    Returns:
        DataFrame with columns: quelle_code (int), ziel_code (int), wert (float).
    """
    print("  Loading communal OD ...")
    od = scoring.GetOevDemandPerCommune(tau=1)
    od = od[od['quelle_code'] != od['ziel_code']].copy()
    od = od[od['wert'] > 0][['quelle_code', 'ziel_code', 'wert']].copy()
    od['quelle_code'] = od['quelle_code'].astype(int)
    od['ziel_code']   = od['ziel_code'].astype(int)
    print(f"    {len(od)} OD pairs with positive demand (intrazonal removed)")
    return od


def _load_and_join_cells(commune_gdf: gpd.GeoDataFrame, bfs_col: str) -> pd.DataFrame:
    """Load population and employment CSVs, build cell centroids, join to communes.

    Returns:
        DataFrame with columns: RELI (int), BFS (int), Pop (float), Empl (float).
    """
    print("  Loading population and employment rasters ...")
    pop_raw  = pd.read_csv(paths.POPULATION_CSV_2023,  sep=';')
    empl_raw = pd.read_csv(paths.EMPLOYMENT_CSV_2023, sep=';')

    # Standardise value column — source files use 'NUMMER'
    pop_raw  = pop_raw.rename(columns={'NUMMER': 'Pop'})
    empl_raw = empl_raw.rename(columns={'NUMMER': 'Empl'})

    pop_cols  = [c for c in ['RELI', 'E_KOORD', 'N_KOORD', 'Pop']  if c in pop_raw.columns]
    empl_cols = [c for c in ['RELI', 'Empl']                         if c in empl_raw.columns]

    cells = (
        pop_raw[pop_cols]
        .merge(empl_raw[empl_cols], on='RELI', how='outer')
        .fillna({'Pop': 0.0, 'Empl': 0.0})
    )
    for col in ['E_KOORD', 'N_KOORD', 'Pop', 'Empl', 'RELI']:
        cells[col] = pd.to_numeric(cells[col], errors='coerce')
    cells = cells.dropna(subset=['RELI', 'E_KOORD', 'N_KOORD'])
    cells['RELI'] = cells['RELI'].astype(int)

    print(f"    Spatially joining {len(cells):,} cells to communes ...")
    gdf = gpd.GeoDataFrame(
        cells,
        geometry=gpd.points_from_xy(cells['E_KOORD'], cells['N_KOORD']),
        crs=_CODEBASE_CRS
    )
    joined = gpd.sjoin(
        gdf[['RELI', 'Pop', 'Empl', 'geometry']],
        commune_gdf[[bfs_col, 'geometry']],
        how='left',
        predicate='within'
    )
    joined = joined.rename(columns={bfs_col: 'BFS'}).dropna(subset=['BFS'])
    joined['BFS'] = joined['BFS'].astype(int)

    result = joined[['RELI', 'BFS', 'Pop', 'Empl']].drop_duplicates('RELI').copy()
    print(f"    {len(result):,} cells joined to a commune")
    return result


def _load_pt_feeder_assignment() -> pd.DataFrame:
    """Read cell_station_candidates.csv; return (RELI, Station_1_ID) as int.

    Cells with NO_PT sentinel (Station_1_ID = -1) are excluded.

    Returns:
        DataFrame with columns: RELI (int), Station_1_ID (int).
    """
    cand_path = os.path.join(paths.MAIN, catchment_base.PT_FEEDER_DATA_DIR, 'cell_station_candidates.csv')
    print(f"  Loading cell→station assignment (production / population side) ...")
    cand = pd.read_csv(cand_path)[['RELI', 'Station_1_ID']].copy()
    cand['RELI']         = pd.to_numeric(cand['RELI'],         errors='coerce')
    cand['Station_1_ID'] = pd.to_numeric(cand['Station_1_ID'], errors='coerce')
    cand = cand.dropna()
    cand = cand[cand['Station_1_ID'] > 0]   # exclude noPT sentinel (-1)
    cand['RELI']         = cand['RELI'].astype(int)
    cand['Station_1_ID'] = cand['Station_1_ID'].astype(int)
    print(f"    {len(cand):,} cells with a valid station assignment")
    return cand


def _load_pt_feeder_empl_assignment() -> pd.DataFrame:
    """Read the unified cell_station_candidates.csv and filter to cells with
    Emp > 0. Returns (RELI, Station_1_ID) as int.

    The single CSV (written by catchment_allocate._run_pt_feeder_method) covers
    every cell with Pop > 0 OR Emp > 0; the station assignment is GC-based and
    identical for production and attraction sides — only the row-set filter
    differs.

    Returns:
        DataFrame with columns: RELI (int), Station_1_ID (int).
    """
    cand_path = os.path.join(paths.MAIN, catchment_base.PT_FEEDER_DATA_DIR,
                             'cell_station_candidates.csv')
    if not os.path.exists(cand_path):
        raise FileNotFoundError(
            f"PT-feeder cell→station candidates not found at {cand_path}. "
            f"Run catchment_allocate.get_catchment() first to generate it."
        )
    print(f"  Loading cell→station assignment (attraction / employment side) ...")
    cand = pd.read_csv(cand_path)[['RELI', 'Emp', 'Station_1_ID']].copy()
    cand['RELI']         = pd.to_numeric(cand['RELI'],         errors='coerce')
    cand['Emp']          = pd.to_numeric(cand['Emp'],          errors='coerce')
    cand['Station_1_ID'] = pd.to_numeric(cand['Station_1_ID'], errors='coerce')
    cand = cand.dropna(subset=['RELI', 'Station_1_ID'])
    cand = cand[(cand['Station_1_ID'] > 0) & (cand['Emp'].fillna(0) > 0)]
    cand = cand[['RELI', 'Station_1_ID']].copy()
    cand['RELI']         = cand['RELI'].astype(int)
    cand['Station_1_ID'] = cand['Station_1_ID'].astype(int)
    print(f"    {len(cand):,} cells with a valid employment station assignment")
    return cand


def _compute_orig_weights(cells_df: pd.DataFrame,
                          pop_assign_df: pd.DataFrame) -> pd.DataFrame:
    """Production (origin) weights: population share of each cell within its
    commune, aggregated to (commune, station) level using the pop assignment.

    Only cells with Pop > 0 contribute. Weights sum to ≤ 1.0 per BFS;
    shortfall = share of commune population in cells with no PT access.

    Args:
        cells_df:     DataFrame[RELI, BFS, Pop, Empl] from _load_and_join_cells.
        pop_assign_df: DataFrame[RELI, Station_1_ID] from _load_pt_feeder_assignment.

    Returns:
        DataFrame (BFS, station_id, orig_weight).
    """
    merged = (
        cells_df[cells_df['Pop'] > 0]
        .merge(pop_assign_df, on='RELI', how='inner')
    ).copy()
    commune_pop      = merged.groupby('BFS')['Pop'].transform('sum').clip(lower=1e-9)
    merged['pop_share'] = merged['Pop'] / commune_pop
    return (
        merged.groupby(['BFS', 'Station_1_ID'])['pop_share']
        .sum().reset_index()
        .rename(columns={'Station_1_ID': 'station_id', 'pop_share': 'orig_weight'})
    )


def _compute_dest_weights(cells_df: pd.DataFrame,
                          empl_assign_df: pd.DataFrame) -> pd.DataFrame:
    """Attraction (destination) weights: FTE share of each cell within its
    commune, aggregated to (commune, station) level using the employment
    assignment.

    Only cells with Empl > 0 contribute. Uses the employment-specific station
    assignment (cell_station_candidates_empl.csv) so FTE-dense cells correctly
    pull flows toward their nearest rail station even when pop = 0.

    Args:
        cells_df:      DataFrame[RELI, BFS, Pop, Empl] from _load_and_join_cells.
        empl_assign_df: DataFrame[RELI, Station_1_ID] from _load_pt_feeder_empl_assignment.

    Returns:
        DataFrame (BFS, station_id, dest_weight).
    """
    merged = (
        cells_df[cells_df['Empl'] > 0]
        .merge(empl_assign_df, on='RELI', how='inner')
    ).copy()
    commune_empl        = merged.groupby('BFS')['Empl'].transform('sum').clip(lower=1e-9)
    merged['empl_share'] = merged['Empl'] / commune_empl
    return (
        merged.groupby(['BFS', 'Station_1_ID'])['empl_share']
        .sum().reset_index()
        .rename(columns={'Station_1_ID': 'station_id', 'empl_share': 'dest_weight'})
    )


def _reaggregate_to_stations(
    communal_od: pd.DataFrame,
    orig_weights: pd.DataFrame,
    dest_weights: pd.DataFrame
) -> pd.DataFrame:
    """Merge communal OD with (commune, station) weights; compute station-pair flows.

    Formula:
        trips(A, C) = Σ_{i,j} T_ij × orig_weight(i, A) × dest_weight(j, C)

    Returns:
        Long-format DataFrame: origin_station_id (int), dest_station_id (int), trips (float).
    """
    print("  Reaggregating communal OD to station pairs ...")

    merged = communal_od.merge(
        orig_weights.rename(columns={
            'BFS': 'quelle_code',
            'station_id': 'origin_station_id',
            'orig_weight': 'ow'
        }),
        on='quelle_code',
        how='inner'
    )

    merged = merged.merge(
        dest_weights.rename(columns={
            'BFS': 'ziel_code',
            'station_id': 'dest_station_id',
            'dest_weight': 'dw'
        }),
        on='ziel_code',
        how='inner'
    )

    merged['trips'] = merged['wert'] * merged['ow'] * merged['dw']

    n_before_intra = len(merged)
    merged = merged[merged['origin_station_id'] != merged['dest_station_id']]
    n_intra = n_before_intra - len(merged)
    if n_intra:
        print(f"    {n_intra:,} intra-station rows dropped (origin == dest station).")

    station_od = (
        merged.groupby(['origin_station_id', 'dest_station_id'])['trips']
        .sum()
        .reset_index()
    )
    station_od = station_od[station_od['trips'] > 0].copy()
    print(f"    {len(station_od):,} non-zero station OD pairs")
    return station_od


# ===============================================================================
# DIAGNOSTIC PLOTS (W3 Phase 4)
# ===============================================================================

def _build_station_summary(rail_stations, pt_long, muni_long, name_lookup):
    """Combine per-station attracted trips for both methods + geometry + name.

    Attracted trips = sum over destinations (i.e., each value is the total
    number of incoming PT trips per station, all-day τ=1).
    """
    pt_attr = (pt_long.groupby('dest_station_id')['trips'].sum()
               if pt_long is not None and not pt_long.empty else pd.Series(dtype=float))
    mu_attr = (muni_long.groupby('dest_station_id')['trips'].sum()
               if muni_long is not None and not muni_long.empty else pd.Series(dtype=float))

    df = rail_stations[['id_point', 'stop_name', 'geometry']].copy()
    df['id_point']      = df['id_point'].astype(int)
    df['readable_name'] = (df['id_point'].astype(str).map(name_lookup)
                                         .fillna(df['stop_name']))
    df['pt_feeder_attracted'] = df['id_point'].map(pt_attr).fillna(0).astype(float)
    df['municipal_attracted'] = df['id_point'].map(mu_attr).fillna(0).astype(float)
    df['diff_pt_minus_muni']  = df['pt_feeder_attracted'] - df['municipal_attracted']
    return gpd.GeoDataFrame(df, geometry='geometry',
                            crs=catchment_base.CODEBASE_CRS)


def _plot_bar_attracted(summary: gpd.GeoDataFrame, top_n: int = 20) -> None:
    """Top-N stations bar chart, two bars per station (PT-Feeder vs Municipal)."""
    df = summary.copy()
    df['_max_attr'] = df[['pt_feeder_attracted', 'municipal_attracted']].max(axis=1)
    df = df.sort_values('_max_attr', ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(df))
    w = 0.4
    ax.bar(x - w/2, df['pt_feeder_attracted'], width=w,
           color='#1565C0', label='PT-Feeder')
    ax.bar(x + w/2, df['municipal_attracted'], width=w,
           color='#E65100', label='Municipal')
    ax.set_xticks(x)
    ax.set_xticklabels(df['readable_name'].tolist(), rotation=60, ha='right')
    ax.set_ylabel('Attracted trips per day (all-day, τ=1)')
    ax.set_title(f'Top {top_n} stations by attracted PT trips — '
                 f'PT-Feeder vs Municipal')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(OD_COMPARISON_PLOT_DIR,
                            'od_compare_attracted_trips_bar.pdf')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"    Saved → {out_path}")


def _plot_spatial_attracted(summary: gpd.GeoDataFrame, boundary) -> None:
    """Two-panel spatial map: PT-feeder vs Municipal, stations as circles
    sized by attracted trips. Shared scale across panels."""
    max_val = float(max(summary['pt_feeder_attracted'].max(),
                        summary['municipal_attracted'].max(), 1.0))

    lakes = None
    if os.path.exists(paths.LAKES_SHP):
        lakes = gpd.read_file(paths.LAKES_SHP).to_crs(catchment_base.CODEBASE_CRS)
        lakes = lakes[lakes.geometry.intersects(boundary)].copy()
    boundary_gdf = gpd.GeoDataFrame(geometry=[boundary],
                                    crs=catchment_base.CODEBASE_CRS)

    fig, axes = plt.subplots(1, 2, figsize=(20, 11))
    panels = [
        (axes[0], 'pt_feeder_attracted', 'PT-Feeder', '#1565C0'),
        (axes[1], 'municipal_attracted', 'Municipal', '#E65100'),
    ]
    for ax, col, title, color in panels:
        ax.set_facecolor('#E8E8E8')
        boundary_gdf.plot(ax=ax, color='white', edgecolor='none', zorder=0)
        if lakes is not None and not lakes.empty:
            lakes.plot(ax=ax, color='#A8D8EA', edgecolor='none', zorder=2)
        boundary_gdf.boundary.plot(ax=ax, color='black', linewidth=1.8,
                                    linestyle='--', zorder=4)
        sizes = (summary[col] / max_val) * 800 + 5
        ax.scatter(summary.geometry.x, summary.geometry.y,
                   s=sizes, c=color, alpha=0.65,
                   edgecolors='black', linewidths=0.4, zorder=5)
        bx_min, by_min, bx_max, by_max = boundary.bounds
        pad = 200
        ax.set_xlim(bx_min - pad, bx_max + pad)
        ax.set_ylim(by_min - pad, by_max + pad)
        ax.set_aspect('equal')
        ax.set_title(f'{title}: attracted trips per day (all-day)', fontsize=13)
        ax.set_xlabel('E [m]')
        ax.set_ylabel('N [m]')
        catchment_base._add_map_elements(ax)

    fig.suptitle('Per-station attracted PT trips — PT-Feeder vs Municipal',
                 fontsize=15, y=0.98)
    out_path = os.path.join(OD_COMPARISON_PLOT_DIR,
                            'od_compare_attracted_trips_map.pdf')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"    Saved → {out_path}")


def _plot_diff_map(summary: gpd.GeoDataFrame, boundary,
                    top_n_label: int = 5) -> None:
    """Signed difference map: PT-feeder − Municipal. Circle size by |diff|,
    colour by signed diff (RdBu_r). Top-N |diff| stations annotated."""
    diffs   = summary['diff_pt_minus_muni']
    abs_max = float(max(abs(diffs.min()), abs(diffs.max()), 1.0))

    lakes = None
    if os.path.exists(paths.LAKES_SHP):
        lakes = gpd.read_file(paths.LAKES_SHP).to_crs(catchment_base.CODEBASE_CRS)
        lakes = lakes[lakes.geometry.intersects(boundary)].copy()
    boundary_gdf = gpd.GeoDataFrame(geometry=[boundary],
                                    crs=catchment_base.CODEBASE_CRS)

    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_facecolor('#E8E8E8')
    boundary_gdf.plot(ax=ax, color='white', edgecolor='none', zorder=0)
    if lakes is not None and not lakes.empty:
        lakes.plot(ax=ax, color='#A8D8EA', edgecolor='none', zorder=2)
    boundary_gdf.boundary.plot(ax=ax, color='black', linewidth=1.8,
                                linestyle='--', zorder=4)

    sizes = (diffs.abs() / abs_max) * 800 + 5
    sc = ax.scatter(summary.geometry.x, summary.geometry.y,
                    s=sizes, c=diffs, cmap='RdBu_r',
                    vmin=-abs_max, vmax=abs_max, alpha=0.85,
                    edgecolors='black', linewidths=0.4, zorder=5)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label('PT-Feeder − Municipal (trips/day)', fontsize=10)

    # Label top-N |diff| outliers
    top_outliers = summary.reindex(
        diffs.abs().sort_values(ascending=False).index).head(top_n_label)
    for _, row in top_outliers.iterrows():
        ax.annotate(row['readable_name'],
                    xy=(row.geometry.x, row.geometry.y),
                    xytext=(8, 8), textcoords='offset points',
                    fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor='black', alpha=0.85),
                    zorder=6)

    bx_min, by_min, bx_max, by_max = boundary.bounds
    pad = 200
    ax.set_xlim(bx_min - pad, bx_max + pad)
    ax.set_ylim(by_min - pad, by_max + pad)
    ax.set_aspect('equal')
    ax.set_title('Difference: PT-Feeder − Municipal '
                 '(per-station attracted trips, all-day)', fontsize=14)
    ax.set_xlabel('E [m]')
    ax.set_ylabel('N [m]')
    catchment_base._add_map_elements(ax)

    out_path = os.path.join(OD_COMPARISON_PLOT_DIR,
                            'od_compare_attracted_trips_diff.pdf')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"    Saved → {out_path}")


def _plot_od_diagnostics(pt_long, muni_long, rail_stations, boundary,
                          name_lookup) -> None:
    """Three comparison figures from the all-day station-pair OD matrices.

    Outputs (under plots/Catchment_Area/OD_Comparison/):
      - od_compare_attracted_trips_bar.pdf   — top-20 station bar chart
      - od_compare_attracted_trips_map.pdf   — two-panel spatial map
      - od_compare_attracted_trips_diff.pdf  — diff map with top-5 labels
    Plots are built from the unscaled (τ=1) long-format OD so the comparison
    is window-independent.
    """
    print("\n  Building OD comparison plots ...")
    os.makedirs(OD_COMPARISON_PLOT_DIR, exist_ok=True)

    summary = _build_station_summary(rail_stations, pt_long, muni_long,
                                      name_lookup)
    if summary.empty:
        print("    No station data — skipping plots.")
        return
    if (summary['pt_feeder_attracted'].sum() == 0
        and summary['municipal_attracted'].sum() == 0):
        print("    Both methods have zero attracted trips — skipping plots.")
        return

    _plot_bar_attracted(summary, top_n=20)
    _plot_spatial_attracted(summary, boundary)
    _plot_diff_map(summary, boundary, top_n_label=5)


# ===============================================================================
# STANDALONE ENTRY POINT
# ===============================================================================

if __name__ == '__main__':
    os.chdir(paths.MAIN)
    import settings

    # Discover available service versions (same logic as catchment_allocate)
    _feeder_root = os.path.join(paths.MAIN, paths.FEEDER_LINES_DIR)
    _svc_versions = sorted([
        d for d in os.listdir(_feeder_root)
        if os.path.isdir(os.path.join(_feeder_root, d))
        and os.path.exists(os.path.join(_feeder_root, d,
                                        paths.SERVICES_UNPROJECTED_SUBDIR,
                                        'pt_feeder_stops.gpkg'))
    ]) if os.path.isdir(_feeder_root) else []

    _svc = ''
    if not _svc_versions:
        print("WARNING: no service versions found — _RAIL_BASE will be empty.")
    elif len(_svc_versions) == 1:
        _svc = _svc_versions[0]
        print(f"Service version: {_svc}")
    else:
        print("Available service versions:")
        for _i, _sv in enumerate(_svc_versions, 1):
            print(f"  {_i}) {_sv}")
        while True:
            _raw = input("Select service version [1]: ").strip() or '1'
            if _raw.isdigit() and 1 <= int(_raw) <= len(_svc_versions):
                _svc = _svc_versions[int(_raw) - 1]
                break
            print(f"  Invalid — enter 1–{len(_svc_versions)}.")

    prepare_all_od_matrices(use_cache=settings.use_cache_stationsOD, svc_version=_svc)
