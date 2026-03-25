# catchment_allocate.py
#
# Station catchment area generation for infraScanRail.
# Houses both the municipal (centroid-based) and PT-Feeder (GTFS multimodal)
# methods. Both methods are always run so their results can be compared.
#
# Public entry point: get_catchment(use_cache: bool) -> None
#
# CRS: EPSG:2056 (LV95 - Swiss National Grid) throughout.
#
# Directory layout:
#   Data outputs  -> data/Catchment_Area/            (shared Step 1)
#                    data/Catchment_Area/Municipal/   (municipal method)
#                    data/Catchment_Area/PT_Feeder/   (PT-feeder method)
#   Plot outputs  -> plots/Catchment_Area/            (all plots)

import os
import time

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import rasterio
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.spatial import cKDTree
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

import paths
import settings
import cost_parameters as cp

# ===============================================================================
# CONSTANTS
# ===============================================================================

CODEBASE_CRS = 'EPSG:2056'

# ARE-aligned buffer radii (walking access to PT stops)
BUFFER_RAIL_M  = 1000    # m - walking to rail station
BUFFER_TRAM_M  =  750    # m - walking to tram stop
BUFFER_BUS_M   =  500    # m - walking to bus/feeder stop

# Access speeds
WALK_SPEED_MS  = 1.389   # m/s  (5 km/h)
CYCLE_SPEED_MS = 4.167   # m/s  (15 km/h)
CYCLE_RADIUS_M = 2500    # m - cycling search radius to rail

# Transfer penalty (Axhausen 2014, comfort-weighted)
# Applied at each feeder-to-rail mode switch (walk->bus->rail or walk->tram->rail)
# Consistent with TT_Delay.py line change penalty in the travel time graph.
TRANSFER_PENALTY_SEC = cp.comfort_weighted_change_time * 60  # 12 min -> 720 sec

# Raster cell size
CELL_SIZE_M    = 100     # m

# noPT sentinel
NO_PT_ID       = -1

# Access mode codes (written to access_mode.tif)
MODE_NO_PT  = 0
MODE_WALK   = 1
MODE_BUS    = 2
MODE_TRAM   = 3
MODE_CYCLE  = 4

# Station search constraint (PT-Feeder method)
MAX_CANDIDATE_STATIONS = 5

# GTFS network folder (output of catchment_build_network.py)
GTFS_NETWORK_FOLDER = 'FP2026_ZH_network'

# Output directories
CATCHMENT_DATA_DIR  = paths.CATCHMENT_AREA_DIR               # data/Catchment_Area
CATCHMENT_PLOT_DIR  = os.path.join('plots', 'Catchment_Area') # plots/Catchment_Area
MUNICIPAL_DATA_DIR  = os.path.join(CATCHMENT_DATA_DIR, 'Municipal')
PT_FEEDER_DATA_DIR  = os.path.join(CATCHMENT_DATA_DIR, 'PT_Feeder')


def _ensure_dirs():
    """Create all output directories."""
    for d in [CATCHMENT_DATA_DIR, CATCHMENT_PLOT_DIR,
              MUNICIPAL_DATA_DIR, PT_FEEDER_DATA_DIR]:
        os.makedirs(d, exist_ok=True)


# ===============================================================================
# STEP 1: SHARED DATA LOADING (both methods)
# ===============================================================================

def _load_catchment_boundary():
    """Load the catchment area boundary polygon from the GPKG produced by
    catchment_filter_gtfs.py. Single source of truth for the spatial extent.

    Returns
    -------
    shapely.Polygon
        Study area boundary in EPSG:2056.
    """
    boundary_path = paths.CATCHMENT_AREA_BOUNDARY_GPKG
    print(f"  Loading catchment boundary from {boundary_path} ...")
    gdf = gpd.read_file(boundary_path)
    gdf = gdf.to_crs(CODEBASE_CRS)
    boundary = gdf.geometry.unary_union
    print(f"  Boundary loaded - area = {boundary.area / 1e6:.1f} km2")
    return boundary


def _load_population_grid(boundary):
    """Load the Swiss population CSV, filter to study area, write GeoTIFF."""
    print("  Loading population grid ...")
    df = pd.read_csv(paths.POPULATION_CSV_2023, sep=';')
    return _load_grid(df, boundary, 'population')


def _load_employment_grid(boundary):
    """Load the Swiss employment CSV, filter to study area, write GeoTIFF."""
    print("  Loading employment grid ...")
    df = pd.read_csv(paths.EMPLOYMENT_CSV_2023, sep=';')
    return _load_grid(df, boundary, 'employment')


def _load_grid(df, boundary, label):
    """Shared loader for population / employment CSVs."""
    # Swiss CSVs may use comma as decimal separator (e.g. "5,376" = 5.376).
    # Replace commas before numeric conversion so these are not lost.
    for col in ['E_KOORD', 'N_KOORD', 'NUMMER']:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(',', '.', regex=False),
            errors='coerce',
        )
    df = df.dropna(subset=['E_KOORD', 'N_KOORD', 'NUMMER'])
    df = df[df['NUMMER'] > 0].copy()

    # Compute cell centroids (bottom-left corner + 50m)
    df['cx'] = df['E_KOORD'] + 50
    df['cy'] = df['N_KOORD'] + 50
    geometry = gpd.points_from_xy(df['cx'], df['cy'])
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=CODEBASE_CRS)

    # Clip to study area
    from shapely.prepared import prep
    prep_boundary = prep(boundary)
    mask = gdf.geometry.apply(lambda pt: prep_boundary.contains(pt))
    gdf = gdf[mask].copy()
    print(f"    {label}: {len(gdf):,} cells within study area")

    area_tag = '_'.join(settings.CATCHMENT_CANTON).replace(' ', '')

    # Save filtered CSV
    csv_path = os.path.join(CATCHMENT_DATA_DIR, f'{label}_2023_{area_tag}.csv')
    gdf[['RELI', 'E_KOORD', 'N_KOORD', 'NUMMER']].to_csv(csv_path, index=False, sep=';')
    print(f"    Saved filtered CSV -> {csv_path}")

    # Write GeoTIFF
    _write_grid_tif(gdf, label)

    return gdf


def _write_grid_tif(gdf, label):
    """Rasterise a filtered grid GeoDataFrame into a single-band GeoTIFF."""
    e_min = int(gdf['E_KOORD'].min())
    e_max = int(gdf['E_KOORD'].max()) + CELL_SIZE_M
    n_min = int(gdf['N_KOORD'].min())
    n_max = int(gdf['N_KOORD'].max()) + CELL_SIZE_M

    width  = (e_max - e_min) // CELL_SIZE_M
    height = (n_max - n_min) // CELL_SIZE_M

    raster = np.zeros((height, width), dtype=np.float32)

    for _, row in gdf.iterrows():
        col = int((row['E_KOORD'] - e_min) // CELL_SIZE_M)
        r   = int((n_max - row['N_KOORD'] - CELL_SIZE_M) // CELL_SIZE_M)
        if 0 <= r < height and 0 <= col < width:
            raster[r, col] = row['NUMMER']

    transform = from_origin(e_min, n_max, CELL_SIZE_M, CELL_SIZE_M)
    tif_path = os.path.join(CATCHMENT_DATA_DIR, f'{label}_2023.tif')

    with rasterio.open(tif_path, 'w', driver='GTiff',
                       height=height, width=width, count=1,
                       dtype='float32', crs=CODEBASE_CRS,
                       transform=transform, nodata=0) as dst:
        dst.write(raster, 1)

    print(f"    Saved GeoTIFF -> {tif_path}  ({width}x{height})")


def _cumulate_per_municipality(pop_grid, empl_grid, boundary):
    """Spatial join of population/employment cells to municipalities, with
    nearest-municipality fallback for edge cells.

    Returns
    -------
    gpd.GeoDataFrame
        Summary with columns [BFS_NR, NAME, total_population, total_employment, geometry].
    """
    print("  Cumulating population/employment per municipality ...")

    muni = gpd.read_file(paths.MUNICIPAL_BOUNDARIES_GPKG)
    muni = muni.to_crs(CODEBASE_CRS)
    if 'objektart' in muni.columns:
        muni = muni[muni['objektart'] == 'Gemeindegebiet']
    muni = muni[muni.geometry.intersects(boundary)].copy()
    print(f"    {len(muni)} municipalities in study area")

    # Identify BFS number column
    bfs_col = None
    for candidate in ['BFS_NR', 'bfs_nr', 'BFS_NUMMER', 'bfs_nummer', 'GMDNR', 'gmdnr']:
        if candidate in muni.columns:
            bfs_col = candidate
            break
    if bfs_col is None:
        for c in muni.columns:
            if muni[c].dtype in ['int64', 'int32', 'float64'] and c != 'geometry':
                bfs_col = c
                break
    if bfs_col is None:
        raise ValueError("Cannot identify BFS number column in municipal boundaries")

    # Identify name column
    name_col = None
    for candidate in ['NAME', 'name', 'GMDNAME', 'gmdname', 'GEMEINDENAME']:
        if candidate in muni.columns:
            name_col = candidate
            break
    if name_col is None:
        name_col = bfs_col

    def _assign_cells_to_muni(grid, value_col='NUMMER'):
        joined = gpd.sjoin(grid, muni[[bfs_col, name_col, 'geometry']],
                           how='left', predicate='within')
        unassigned = joined[joined[bfs_col].isna()]
        if len(unassigned) > 0:
            print(f"    {len(unassigned)} cells outside all municipalities - assigning to nearest")
            for idx in unassigned.index:
                pt = grid.loc[idx, 'geometry']
                dists = muni.geometry.boundary.distance(pt)
                nearest_idx = dists.idxmin()
                joined.loc[idx, bfs_col] = muni.loc[nearest_idx, bfs_col]
                joined.loc[idx, name_col] = muni.loc[nearest_idx, name_col]
        agg = joined.groupby(bfs_col).agg({value_col: 'sum', name_col: 'first'}).reset_index()
        return agg

    pop_agg  = _assign_cells_to_muni(pop_grid)
    empl_agg = _assign_cells_to_muni(empl_grid)

    summary = pop_agg.rename(columns={'NUMMER': 'total_population'})
    empl_renamed = empl_agg.rename(columns={'NUMMER': 'total_employment'})
    summary = summary.merge(empl_renamed[[bfs_col, 'total_employment']],
                            on=bfs_col, how='outer')
    summary = summary.fillna(0)

    # Recover name for outer-join rows that only have employment
    missing_name = summary[name_col].isna() | (summary[name_col] == 0)
    if missing_name.any():
        name_lookup = muni.set_index(bfs_col)[name_col]
        summary.loc[missing_name, name_col] = (
            summary.loc[missing_name, bfs_col].map(name_lookup)
        )

    # Rename canonical columns before geometry merge so everything is consistent
    summary = summary.rename(columns={bfs_col: 'BFS_NR', name_col: 'NAME'})

    muni_geom = muni[[bfs_col, 'geometry']].rename(columns={bfs_col: 'BFS_NR'})
    summary = summary.merge(muni_geom, on='BFS_NR', how='left')
    summary = gpd.GeoDataFrame(summary, geometry='geometry', crs=CODEBASE_CRS)

    # Enforce column order: BFS_NR, NAME, total_population, total_employment
    csv_cols = ['BFS_NR', 'NAME', 'total_population', 'total_employment']
    csv_path = os.path.join(CATCHMENT_DATA_DIR, 'municipal_pop_empl_summary.csv')
    summary[csv_cols].to_csv(csv_path, index=False)
    print(f"    Saved -> {csv_path}")
    print(f"    Total pop = {summary['total_population'].sum():,.0f}, "
          f"empl = {summary['total_employment'].sum():,.0f}")

    return summary


def _plot_municipal_distributions(summary_df, boundary):
    """Produce choropleth maps for population and employment per municipality.

    Uses the same discrete FSO class breaks as the raster plots so that the
    municipal and raster maps share a common visual language.
    Saved to plots/Catchment_Area/.
    """
    print("  Plotting municipal distributions ...")

    # Cumulated bins reflecting total pop/empl per municipality
    pop_cfg = dict(
        bins   = [0, 2000, 5000, 10000, 20000, 50000, np.inf],
        labels = ['1–2,000', '2,001–5,000', '5,001–10,000',
                  '10,001–20,000', '20,001–50,000', 'more than 50,000'],
        colors = ['#FFFFB2', '#FECC5C', '#FD8D3C', '#F03B20', '#BD0026', '#800026'],
        col_header = 'Total inhabitants',
    )
    empl_cfg = dict(
        bins   = [0, 500, 1500, 5000, 20000, np.inf],
        labels = ['1–500', '501–1,500', '1,501–5,000',
                  '5,001–20,000', 'more than 20,000'],
        colors = ['#C7E9C0', '#74C476', '#238B45', '#31A354', '#2C7FB8', '#253494'],
        col_header = 'Total full-time equivalents',
    )
    grey = '#D3D3D3'

    for col, title_word, cfg in [
        ('total_population', 'Population', pop_cfg),
        ('total_employment', 'Employment', empl_cfg),
    ]:
        bins   = cfg['bins']
        labels = cfg['labels']
        colors = cfg['colors']

        # Assign each municipality to a class (0-indexed into labels/colors).
        # Use pd.cut so the open-ended top bin (np.inf) is handled correctly.
        vals = summary_df[col].fillna(0).values
        class_idx = (pd.cut(vals, bins=bins, labels=False, right=True)
                       .clip(0, len(labels) - 1)
                       .astype(int))

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        # Plot grey background for zero/missing first
        zero_mask = vals <= 0
        if zero_mask.any():
            summary_df[zero_mask].plot(ax=ax, color=grey,
                                       edgecolor='grey', linewidth=0.4)

        for ci, color in enumerate(colors):
            mask = (class_idx == ci) & (~zero_mask)
            if not mask.any():
                continue
            summary_df[mask].plot(ax=ax, color=color,
                                  edgecolor='grey', linewidth=0.4)

        # Catchment area boundary
        boundary_gdf = gpd.GeoDataFrame(geometry=[boundary], crs=CODEBASE_CRS)
        boundary_gdf.boundary.plot(ax=ax, color='black', linewidth=1.8,
                                   linestyle='--', zorder=5)

        # Legend (0 first, then data classes, then boundary)
        legend_handles = [Patch(facecolor=grey, edgecolor='none', label='0')]
        for color, lbl in zip(colors, labels):
            legend_handles.append(Patch(facecolor=color, edgecolor='none', label=lbl))
        legend_handles.append(Line2D([0], [0], color='black', linewidth=1.8,
                                     linestyle='--', label='Catchment area boundary'))

        ax.legend(handles=legend_handles,
                  title=cfg['col_header'],
                  title_fontsize=8,
                  fontsize=7,
                  loc='upper right',
                  framealpha=0.9)

        ax.set_title(f'{title_word} per Municipality', fontsize=14)
        ax.set_xlabel('E [m]')
        ax.set_ylabel('N [m]')
        ax.set_aspect('equal')

        fname = f'plot_{title_word.lower()}_by_municipality.pdf'
        out_path = os.path.join(CATCHMENT_PLOT_DIR, fname)
        fig.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"    Saved -> {out_path}")


def _plot_raster_map(gdf, label, boundary):
    """Render a population or employment grid as a coloured raster map,
    styled to match the corresponding map.geo.admin FSO layer.

    Population  → FSO "Population Statistics: Inhabitants"
                  6 discrete classes + grey for 0, yellow→red palette
    Employment  → FSO "Enterprise Statistics: Employment (FTE)"
                  5 discrete classes + grey for 0, yellow→green→blue palette

    Cells are drawn as 100 m × 100 m square patches matching the raster grid.
    Saved to plots/Catchment_Area/.
    """
    print(f"  Plotting {label} raster map ...")

    is_pop = (label == 'population')

    # --- FSO-aligned discrete class breaks and exact hex colours ---
    if is_pop:
        # FSO Population Statistics: Inhabitants (per ha / 100 m cell)
        bins   = [0, 3, 6, 15, 40, 120, np.inf]
        labels = ['1–3', '4–6', '7–15', '16–40', '41–120', 'more than 120']
        colors = ['#FFFFB2', '#FECC5C', '#FD8D3C', '#F03B20', '#BD0026', '#800026']
        col_header = 'Inhabitants per ha'
        title      = 'Population'
    else:
        # FSO Enterprise Statistics: Employment FTE (per ha / 100 m cell)
        bins   = [0, 40, 75, 150, 300, np.inf]
        labels = ['0.1–40', '40.1–75', '75.1–150', '150.1–300', 'more than 300']
        colors = ['#C7E9C0', '#74C476', '#238B45', '#2C7FB8', '#253494']
        col_header = 'Full-time equivalents per ha'
        title      = 'Employment (FTE)'

    grey = '#D3D3D3'
    n_classes = len(labels)

    # Assign each cell to a 0-based class index.
    # np.digitize with bins[1:] returns 0 for val < first break, 1 for first bin, etc.
    # That result is already the correct 0-based index — just clip for safety.
    vals = gdf['NUMMER'].values
    class_idx = np.digitize(vals, bins[1:], right=False)
    class_idx = np.clip(class_idx, 0, n_classes - 1)

    # Build square polygon geometry: bottom-left = (E_KOORD, N_KOORD), size = CELL_SIZE_M
    from shapely.geometry import box as _box
    squares = [
        _box(row['E_KOORD'], row['N_KOORD'],
             row['E_KOORD'] + CELL_SIZE_M, row['N_KOORD'] + CELL_SIZE_M)
        for _, row in gdf.iterrows()
    ]
    plot_gdf = gpd.GeoDataFrame(
        {'class_idx': class_idx},
        geometry=squares,
        crs=CODEBASE_CRS,
    )

    # Load municipal boundaries for overlay — clip to study area so that
    # adjacent municipality lines do not extend beyond the catchment boundary
    muni = gpd.read_file(paths.MUNICIPAL_BOUNDARIES_GPKG)
    muni = muni.to_crs(CODEBASE_CRS)
    if 'objektart' in muni.columns:
        muni = muni[muni['objektart'] == 'Gemeindegebiet']
    muni = muni[muni.geometry.intersects(boundary)].copy()
    muni = gpd.clip(muni, boundary)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Plot grey background covering the full study area for zero-value cells
    boundary_gdf = gpd.GeoDataFrame(geometry=[boundary], crs=CODEBASE_CRS)
    boundary_gdf.plot(ax=ax, color=grey, edgecolor='none')

    # Plot each class as filled squares (no edge to avoid grid lines at small scale)
    for ci, (color, lbl) in enumerate(zip(colors, labels)):
        mask = plot_gdf['class_idx'] == ci
        if not mask.any():
            continue
        plot_gdf[mask].plot(ax=ax, color=color, edgecolor='none', linewidth=0)

    # Municipal boundaries as thin dark grey lines
    # Drop point artefacts that arise where clipped municipalities only touch the
    # catchment boundary at a single point — their .boundary is a MultiPoint which
    # matplotlib renders as filled circles on the border.
    muni_lines = muni.boundary.explode(index_parts=False)
    muni_lines = muni_lines[~muni_lines.geom_type.isin(['Point', 'MultiPoint'])]
    muni_lines.plot(ax=ax, color='#404040', linewidth=0.3, zorder=4)

    # Catchment area boundary
    boundary_gdf.boundary.plot(ax=ax, color='black', linewidth=1.8,
                               linestyle='--', zorder=5)

    # --- Legend (0 first, then data classes, then boundary) ---
    legend_handles = [Patch(facecolor=grey, edgecolor='none', label='0')]
    for color, lbl in zip(colors, labels):
        legend_handles.append(Patch(facecolor=color, edgecolor='none', label=lbl))
    legend_handles.append(Line2D([0], [0], color='#404040', linewidth=0.3,
                                 label='Municipal boundary'))
    legend_handles.append(Line2D([0], [0], color='black', linewidth=1.8,
                                 linestyle='--', label='Catchment area boundary'))

    leg = ax.legend(handles=legend_handles,
                    title=col_header,
                    title_fontsize=8,
                    fontsize=7,
                    loc='upper right',
                    framealpha=0.9,
                    ncol=1)
    leg._legend_box.align = 'left'

    ax.set_title(title, fontsize=13)
    ax.set_xlabel('E [m]')
    ax.set_ylabel('N [m]')
    ax.set_aspect('equal')

    out_path = os.path.join(CATCHMENT_PLOT_DIR, f'plot_{label}_raster.pdf')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"    Saved -> {out_path}")


# ===============================================================================
# STEP 2-MUN: MUNICIPAL METHOD
# ===============================================================================

def _run_municipal_method(boundary, pop_grid, empl_grid, rail_stations_for_diff=None):
    """Replicate the existing canton_ZH flow: reads the hand-curated commune->station
    lookup and aggregates commune OD to station OD.

    Additionally builds catchment geometry by dissolving municipality polygons per
    station assignment, enabling comparison with the PT-Feeder method.

    Returns
    -------
    gpd.GeoDataFrame or None
        Municipal catchment polygons (for diff plot), or None if geometry cannot be built.
    """
    from scoring import aggregate_commune_od_to_station_od

    print("\n--- Municipal Method ---")
    print("  Loading commune OD matrix ...")
    commune_od_df = pd.read_excel(paths.OD_KT_ZH_PATH)

    print("  Loading commune->station lookup ...")
    commune_station_df = pd.read_excel(paths.COMMUNE_TO_STATION_PATH)

    print("  Aggregating commune OD -> station OD ...")
    station_od_matrix = aggregate_commune_od_to_station_od(commune_od_df, commune_station_df)

    out_path = os.path.join(MUNICIPAL_DATA_DIR, 'od_matrix_stations_municipal.csv')
    station_od_matrix.to_csv(out_path)
    print(f"  Station OD matrix saved -> {out_path}")
    print(f"  Shape: {station_od_matrix.shape}")

    # Also save to the legacy path for downstream compatibility
    legacy_path = paths.OD_STATIONS_KT_ZH_PATH
    os.makedirs(os.path.dirname(legacy_path), exist_ok=True)
    station_od_matrix.to_csv(legacy_path)

    # Build catchment geometry from commune->station assignment for comparison
    print("  Building municipal catchment polygons for comparison ...")
    muni_catchment = _build_municipal_catchment_geometry(
        commune_station_df, boundary, pop_grid, empl_grid)

    return muni_catchment


def _build_municipal_catchment_geometry(commune_station_df, boundary, pop_grid, empl_grid):
    """Dissolve municipality polygons by their assigned station to create
    catchment areas for the municipal method.

    Returns
    -------
    gpd.GeoDataFrame
        Columns: [id_point, geometry] dissolved per station.
    """
    muni = gpd.read_file(paths.MUNICIPAL_BOUNDARIES_GPKG)
    muni = muni.to_crs(CODEBASE_CRS)
    if 'objektart' in muni.columns:
        muni = muni[muni['objektart'] == 'Gemeindegebiet']
    muni = muni[muni.geometry.intersects(boundary)].copy()

    # Identify BFS column
    bfs_col = None
    for candidate in ['BFS_NR', 'bfs_nr', 'BFS_NUMMER', 'bfs_nummer', 'GMDNR', 'gmdnr']:
        if candidate in muni.columns:
            bfs_col = candidate
            break
    if bfs_col is None:
        for c in muni.columns:
            if muni[c].dtype in ['int64', 'int32', 'float64'] and c != 'geometry':
                bfs_col = c
                break
    if bfs_col is None:
        print("    WARNING: Cannot identify BFS column - skipping municipal geometry")
        return None

    # Identify the commune BFS code column in lookup
    commune_bfs_col = None
    for candidate in ['Commune_BFS_code', 'BFS_NR', 'bfs_nr', 'GMDNR']:
        if candidate in commune_station_df.columns:
            commune_bfs_col = candidate
            break
    if commune_bfs_col is None:
        print("    WARNING: Cannot identify BFS column in lookup - skipping municipal geometry")
        return None

    # Merge municipality polygons with station assignment
    lookup = commune_station_df[[commune_bfs_col, 'ID_point']].copy()
    lookup = lookup.rename(columns={commune_bfs_col: bfs_col})
    muni[bfs_col] = pd.to_numeric(muni[bfs_col], errors='coerce')
    lookup[bfs_col] = pd.to_numeric(lookup[bfs_col], errors='coerce')

    merged = muni.merge(lookup, on=bfs_col, how='left')
    merged = merged.dropna(subset=['ID_point'])
    merged['ID_point'] = merged['ID_point'].astype(int)

    # Dissolve by station
    dissolved = merged.dissolve(by='ID_point', as_index=False)
    dissolved = dissolved[['ID_point', 'geometry']].copy()
    dissolved = dissolved.rename(columns={'ID_point': 'id_point'})

    # Save
    out_path = os.path.join(MUNICIPAL_DATA_DIR, 'catchement.gpkg')
    # Format for downstream compatibility
    save_df = dissolved.copy()
    save_df['train_station'] = save_df['id_point']
    save_df = save_df.rename(columns={'id_point': 'id'})
    save_df = save_df[['train_station', 'id', 'geometry']]
    save_df.to_file(out_path, driver='GPKG')
    print(f"    Municipal catchment saved -> {out_path}  ({len(dissolved)} stations)")

    # Restore column name for return
    dissolved_return = dissolved.copy()
    return dissolved_return


def _assign_cells_to_municipal_catchment(pop_grid, empl_grid, muni_catchment):
    """Spatial-join population/employment cells to dissolved municipal catchment
    polygons to obtain a cell-level station assignment for the municipal method.

    Returns
    -------
    pd.DataFrame
        Columns: [RELI, id_point] — one row per unique cell that has pop or empl,
        with the municipal-method station assignment.
    """
    if muni_catchment is None or muni_catchment.empty:
        return pd.DataFrame(columns=['RELI', 'id_point'])

    # Combine pop and empl RELIs (union — a cell counts if it has either)
    pop_relis = set(pop_grid['RELI'].values)
    empl_relis = set(empl_grid['RELI'].values)
    all_relis = pop_relis | empl_relis

    # Build a point GeoDataFrame from both grids (deduplicated)
    combined = pd.concat([
        pop_grid[['RELI', 'geometry']],
        empl_grid[['RELI', 'geometry']],
    ]).drop_duplicates(subset='RELI')
    combined = combined[combined['RELI'].isin(all_relis)].copy()
    combined = gpd.GeoDataFrame(combined, geometry='geometry', crs=CODEBASE_CRS)

    # Spatial join: which municipal catchment polygon contains each cell centroid
    catchment = muni_catchment[['id_point', 'geometry']].copy()
    joined = gpd.sjoin(combined, catchment, how='left', predicate='within')

    return joined[['RELI', 'id_point']].copy()


def _plot_municipal_catchments(muni_catchment, pop_grid, empl_grid,
                               rail_stations, boundary):
    """Plot dissolved municipal catchment boundaries with grey pop/empl cells
    inside and station markers (black = has catchment, red = no catchment)."""
    if muni_catchment is None or muni_catchment.empty:
        print("  Skipping municipal catchment plot - no geometry available")
        return
    print("  Building municipal catchment plot ...")

    fig, ax = plt.subplots(1, 1, figsize=(14, 12))

    # Light grey background for study area
    boundary_gdf = gpd.GeoDataFrame(geometry=[boundary], crs=CODEBASE_CRS)
    boundary_gdf.plot(ax=ax, color='#F0F0F0', edgecolor='none')

    # Catchment area outlines
    muni_catchment.boundary.plot(ax=ax, color='#2C3E50', linewidth=1.0, zorder=3)

    # Grey cells: union of pop and empl grids
    pop_relis = set(pop_grid['RELI'].values)
    empl_relis = set(empl_grid['RELI'].values)
    all_relis = pop_relis | empl_relis
    combined = pd.concat([
        pop_grid[['RELI', 'E_KOORD', 'N_KOORD']],
        empl_grid[['RELI', 'E_KOORD', 'N_KOORD']],
    ]).drop_duplicates(subset='RELI')
    combined = combined[combined['RELI'].isin(all_relis)].copy()
    cell_geoms = [
        Polygon([
            (r['E_KOORD'], r['N_KOORD']),
            (r['E_KOORD'] + CELL_SIZE_M, r['N_KOORD']),
            (r['E_KOORD'] + CELL_SIZE_M, r['N_KOORD'] + CELL_SIZE_M),
            (r['E_KOORD'], r['N_KOORD'] + CELL_SIZE_M),
        ])
        for _, r in combined.iterrows()
    ]
    cells_gdf = gpd.GeoDataFrame(geometry=cell_geoms, crs=CODEBASE_CRS)
    cells_gdf.plot(ax=ax, color='#A0A0A0', edgecolor='none', zorder=2)

    # Station markers: black if has catchment, red if not
    assigned_ids = set(muni_catchment['id_point'].astype(str).values)
    has_catchment = rail_stations[rail_stations['id_point'].astype(str).isin(assigned_ids)]
    no_catchment = rail_stations[~rail_stations['id_point'].astype(str).isin(assigned_ids)]

    if len(has_catchment) > 0:
        has_catchment.plot(ax=ax, color='black', markersize=20, marker='^', zorder=5)
    if len(no_catchment) > 0:
        no_catchment.plot(ax=ax, color='red', markersize=20, marker='^', zorder=5)

    # Study area boundary
    boundary_gdf.boundary.plot(ax=ax, color='black', linewidth=1.8, linestyle='--', zorder=4)

    legend_handles = [
        Patch(facecolor='#A0A0A0', edgecolor='none', label='Pop / empl cells'),
        Line2D([0], [0], color='#2C3E50', linewidth=1.0, label='Municipal catchment boundary'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='black',
               markersize=8, label='Station (with catchment)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='red',
               markersize=8, label='Station (no catchment)'),
        Line2D([0], [0], color='black', linewidth=1.8, linestyle='--',
               label='Study area boundary'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=8, framealpha=0.9)
    ax.set_title('Municipal Catchment Areas', fontsize=14)
    ax.set_xlabel('E [m]')
    ax.set_ylabel('N [m]')
    ax.set_aspect('equal')

    out_path = os.path.join(CATCHMENT_PLOT_DIR, 'catchment_municipal_areas.pdf')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"    Saved -> {out_path}")


def _load_feeder_lines(boundary, temporal='all'):
    """Load PT-feeder line geometries from the lines GPKG for visualisation.

    Returns
    -------
    gpd.GeoDataFrame
        Line geometries with columns [geometry, mode].
    """
    subfolder_map = {'all': '', 'peak': 'Peak', 'offpeak': 'Off_Peak'}
    suffix_map    = {'all': '', 'peak': '_peak', 'offpeak': '_offpeak'}
    subfolder = subfolder_map.get(temporal, '')
    suffix    = suffix_map.get(temporal, '')
    base = os.path.join(paths.FEEDER_LINES_DIR, GTFS_NETWORK_FOLDER)
    lines_path = os.path.join(base, subfolder, f'pt_feeder_lines{suffix}.gpkg') \
                 if subfolder else os.path.join(base, f'pt_feeder_lines{suffix}.gpkg')

    if not os.path.exists(lines_path):
        print(f"    WARNING: {lines_path} not found — skipping feeder lines")
        return gpd.GeoDataFrame(columns=['geometry', 'mode'], crs=CODEBASE_CRS)

    import fiona
    layers = fiona.listlayers(lines_path)
    frames = []
    for layer in layers:
        gdf = gpd.read_file(lines_path, layer=layer)
        gdf = gdf.to_crs(CODEBASE_CRS)
        gdf['mode'] = layer
        frames.append(gdf)
    all_lines = pd.concat(frames, ignore_index=True) if frames else gpd.GeoDataFrame()
    if all_lines.empty:
        return gpd.GeoDataFrame(columns=['geometry', 'mode'], crs=CODEBASE_CRS)
    all_lines = gpd.GeoDataFrame(all_lines, geometry='geometry', crs=CODEBASE_CRS)

    # Clip to study area (with small buffer for edge lines)
    expanded = boundary.buffer(500)
    all_lines = all_lines[all_lines.geometry.intersects(expanded)].copy()
    return all_lines


def _plot_municipal_catchments_network(muni_catchment, rail_stations,
                                       boundary, temporal='all'):
    """Plot dissolved municipal catchment boundaries with the PT-feeder network
    lines overlaid, showing where the actual network extends relative to
    administrative catchment boundaries."""
    if muni_catchment is None or muni_catchment.empty:
        print("  Skipping municipal catchment + network plot - no geometry available")
        return
    print("  Building municipal catchment + network plot ...")

    feeder_lines = _load_feeder_lines(boundary, temporal)

    fig, ax = plt.subplots(1, 1, figsize=(14, 12))

    # Light grey background for study area
    boundary_gdf = gpd.GeoDataFrame(geometry=[boundary], crs=CODEBASE_CRS)
    boundary_gdf.plot(ax=ax, color='#F0F0F0', edgecolor='none')

    # Catchment area outlines
    muni_catchment.boundary.plot(ax=ax, color='#2C3E50', linewidth=1.0, zorder=3)

    # PT-feeder network lines coloured by mode
    mode_colours = {
        'bus':          '#0000FF',
        'express_bus':  '#0000FF',
        'on_demand_bus':'#0000FF',
        'tram':         '#FF66CC',
        'metro':        '#00246B',
        'ship':         '#0099FF',
        'funicular':    '#000000',
    }
    plotted_modes = []
    if not feeder_lines.empty:
        for mode_name, colour in mode_colours.items():
            subset = feeder_lines[feeder_lines['mode'].str.lower() == mode_name]
            if len(subset) > 0:
                subset.plot(ax=ax, color=colour, linewidth=0.6, alpha=0.7, zorder=2)
                plotted_modes.append((mode_name, colour))

    # Station markers
    assigned_ids = set(muni_catchment['id_point'].astype(str).values)
    has_catchment = rail_stations[rail_stations['id_point'].astype(str).isin(assigned_ids)]
    no_catchment = rail_stations[~rail_stations['id_point'].astype(str).isin(assigned_ids)]

    if len(has_catchment) > 0:
        has_catchment.plot(ax=ax, color='black', markersize=20, marker='^', zorder=5)
    if len(no_catchment) > 0:
        no_catchment.plot(ax=ax, color='red', markersize=20, marker='^', zorder=5)

    # Study area boundary
    boundary_gdf.boundary.plot(ax=ax, color='black', linewidth=1.8, linestyle='--', zorder=4)

    # Legend
    legend_handles = [
        Line2D([0], [0], color='#2C3E50', linewidth=1.0, label='Municipal catchment boundary'),
    ]
    # Add one entry per plotted mode
    mode_labels = {
        'bus': 'Bus', 'express_bus': 'Express bus', 'on_demand_bus': 'On-demand bus',
        'tram': 'Tram', 'metro': 'Metro', 'ship': 'Ship', 'funicular': 'Funicular',
    }
    for mode_name, colour in plotted_modes:
        legend_handles.append(
            Line2D([0], [0], color=colour, linewidth=1.0, alpha=0.7,
                   label=mode_labels.get(mode_name, mode_name)))
    legend_handles += [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='black',
               markersize=8, label='Station (with catchment)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='red',
               markersize=8, label='Station (no catchment)'),
        Line2D([0], [0], color='black', linewidth=1.8, linestyle='--',
               label='Study area boundary'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=8, framealpha=0.9)
    ax.set_title('Municipal Catchment Areas with PT-Feeder Network', fontsize=14)
    ax.set_xlabel('E [m]')
    ax.set_ylabel('N [m]')
    ax.set_aspect('equal')

    out_path = os.path.join(CATCHMENT_PLOT_DIR, 'catchment_municipal_areas_network.pdf')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"    Saved -> {out_path}")


# ===============================================================================
# STEP 2-PT: PT-FEEDER BUFFERS
# ===============================================================================

def _load_feeder_stops(boundary, temporal='all'):
    """Load PT-feeder stops from the multi-layer GPKG, assign buffer radii.

    Parameters
    ----------
    temporal : 'all' | 'peak' | 'offpeak'
        Which temporal variant of the network file to load.
    """
    print("  Loading feeder stops ...")
    subfolder_map = {'all': '', 'peak': 'Peak', 'offpeak': 'Off_Peak'}
    suffix_map    = {'all': '', 'peak': '_peak', 'offpeak': '_offpeak'}
    subfolder = subfolder_map[temporal]
    suffix    = suffix_map[temporal]
    base = os.path.join(paths.FEEDER_LINES_DIR, GTFS_NETWORK_FOLDER)
    stops_path = os.path.join(base, subfolder, f'pt_feeder_stops{suffix}.gpkg') \
                 if subfolder else os.path.join(base, f'pt_feeder_stops{suffix}.gpkg')

    import fiona
    layers = fiona.listlayers(stops_path)
    frames = []
    for layer in layers:
        gdf = gpd.read_file(stops_path, layer=layer)
        gdf['mode'] = layer
        frames.append(gdf)
    all_stops = pd.concat(frames, ignore_index=True)
    all_stops = gpd.GeoDataFrame(all_stops, geometry='geometry', crs=CODEBASE_CRS)

    all_stops['buffer_radius_m'] = all_stops['mode'].apply(
        lambda m: BUFFER_TRAM_M if 'tram' in m.lower() else BUFFER_BUS_M
    )

    max_buffer = max(BUFFER_BUS_M, BUFFER_TRAM_M)
    expanded = boundary.buffer(max_buffer)
    all_stops = all_stops[all_stops.geometry.within(expanded)].copy()

    keep_cols = ['stop_id', 'stop_name', 'mode', 'buffer_radius_m', 'geometry']
    for c in keep_cols:
        if c not in all_stops.columns and c != 'geometry':
            all_stops[c] = None
    all_stops = all_stops[keep_cols].copy()

    print(f"    {len(all_stops):,} feeder stops loaded")
    return all_stops


def _load_rail_stations(boundary, temporal='all'):
    """Load rail stations from the rail_stops GPKG produced by catchment_build_network.py.
    Maps stop_id -> ID_point by spatial join (100 m buffer) against the same
    rail_stops file — no external ZVV file required.

    Parameters
    ----------
    temporal : 'all' | 'peak' | 'offpeak'
        Which temporal variant of the network file to load.
    """
    print("  Loading rail stations ...")
    subfolder_map = {'all': '', 'peak': 'Peak', 'offpeak': 'Off_Peak'}
    suffix_map    = {'all': '', 'peak': '_peak', 'offpeak': '_offpeak'}
    subfolder = subfolder_map[temporal]
    suffix    = suffix_map[temporal]
    base = os.path.join(paths.RAIL_PROCESSED_DIR, GTFS_NETWORK_FOLDER)
    rail_path = os.path.join(base, subfolder, f'rail_stops{suffix}.gpkg') \
                if subfolder else os.path.join(base, f'rail_stops{suffix}.gpkg')

    import fiona
    layers = fiona.listlayers(rail_path)
    frames = []
    for layer in layers:
        gdf = gpd.read_file(rail_path, layer=layer)
        gdf['mode'] = layer
        frames.append(gdf)
    rail = pd.concat(frames, ignore_index=True)
    rail = gpd.GeoDataFrame(rail, geometry='geometry', crs=CODEBASE_CRS)

    expanded = boundary.buffer(BUFFER_RAIL_M)
    rail = rail[rail.geometry.within(expanded)].copy()

    rail['stop_id'] = rail['stop_id'].astype(str)
    rail = rail.drop_duplicates(subset='stop_id').copy()

    # The rail_stops.gpkg produced by catchment_build_network.py uses the
    # GTFS parent-station numeric ID as stop_id (e.g. '8502224'). This is
    # the same identifier used as the network node key downstream, so we
    # use it directly as id_point without any external crosswalk file.
    rail['diva_nr'] = None
    rail['id_point'] = rail['stop_id']

    keep_cols = ['stop_id', 'stop_name', 'diva_nr', 'id_point', 'mode', 'geometry']
    for c in keep_cols:
        if c not in rail.columns and c != 'geometry':
            rail[c] = None
    rail = rail[keep_cols].copy()

    n_mapped = rail['id_point'].notna().sum()
    print(f"    {len(rail)} rail stations loaded, {n_mapped} mapped to ID_point")
    return rail


def _load_feeder_segments(temporal='all'):
    """Load PT-feeder segment edges from the multi-layer GPKG.

    Parameters
    ----------
    temporal : 'all' | 'peak' | 'offpeak'
        Which temporal variant of the network file to load.
    """
    print("  Loading feeder segments ...")
    subfolder_map = {'all': '', 'peak': 'Peak', 'offpeak': 'Off_Peak'}
    suffix_map    = {'all': '', 'peak': '_peak', 'offpeak': '_offpeak'}
    subfolder = subfolder_map[temporal]
    suffix    = suffix_map[temporal]
    base = os.path.join(paths.FEEDER_LINES_DIR, GTFS_NETWORK_FOLDER)
    seg_path = os.path.join(base, subfolder, f'pt_feeder_segments{suffix}.gpkg') \
               if subfolder else os.path.join(base, f'pt_feeder_segments{suffix}.gpkg')

    import fiona
    layers = fiona.listlayers(seg_path)
    frames = []
    for layer in layers:
        gdf = gpd.read_file(seg_path, layer=layer)
        frames.append(gdf)
    segments = pd.concat(frames, ignore_index=True)

    keep = ['from_stop_id', 'to_stop_id', 'travel_time_min', 'mode_label', 'line_short_name']
    for c in keep:
        if c not in segments.columns:
            segments[c] = None
    segments = segments[keep].copy()

    segments['travel_time_min'] = pd.to_numeric(segments['travel_time_min'], errors='coerce')
    segments = segments.dropna(subset=['travel_time_min'])
    segments = segments[segments['travel_time_min'] > 0].copy()

    print(f"    {len(segments):,} feeder segments loaded")
    return segments


def _build_pt_buffers(feeder_stops, rail_stations):
    """Generate buffer geometries for visualisation and validation."""
    print("  Building PT buffers for visualisation ...")
    records = []

    bus_stops = feeder_stops[~feeder_stops['mode'].str.lower().str.contains('tram')]
    for _, row in bus_stops.iterrows():
        records.append({
            'stop_id': row['stop_id'],
            'buffer_type': 'walk_to_feeder_bus',
            'geometry': row.geometry.buffer(BUFFER_BUS_M)
        })

    tram_stops = feeder_stops[feeder_stops['mode'].str.lower().str.contains('tram')]
    for _, row in tram_stops.iterrows():
        records.append({
            'stop_id': row['stop_id'],
            'buffer_type': 'walk_to_feeder_tram',
            'geometry': row.geometry.buffer(BUFFER_TRAM_M)
        })

    for _, row in rail_stations.iterrows():
        records.append({
            'stop_id': row['stop_id'],
            'buffer_type': 'walk_to_rail',
            'geometry': row.geometry.buffer(BUFFER_RAIL_M)
        })

    for _, row in rail_stations.iterrows():
        records.append({
            'stop_id': row['stop_id'],
            'buffer_type': 'cycle_to_rail',
            'geometry': row.geometry.buffer(CYCLE_RADIUS_M)
        })

    buffers_gdf = gpd.GeoDataFrame(records, geometry='geometry', crs=CODEBASE_CRS)

    out_path = os.path.join(PT_FEEDER_DATA_DIR, 'buffers_visualisation.gpkg')
    buffers_gdf.to_file(out_path, driver='GPKG')
    print(f"    Saved -> {out_path}  ({len(buffers_gdf)} features)")

    return buffers_gdf


# ===============================================================================
# STEP 3: FEEDER NETWORK GRAPH (PT-Feeder only)
# ===============================================================================

def _build_feeder_graph(feeder_stops, feeder_segments, rail_stations):
    """Build a node-split NetworkX graph and pre-compute shortest paths from
    each rail station to all reachable feeder stops, with correct transfer
    penalties.

    Node-splitting approach
    -----------------------
    Each physical feeder stop is split into one node per service line that
    serves it:  node id = (stop_id, line_short_name)
    Consecutive segments on the *same* line connect those split nodes at zero
    extra cost (just travel_time_min * 60 s).
    Changing line at the same physical stop adds TRANSFER_PENALTY_SEC as the
    edge weight between the two split nodes for that stop.

    Rail stations are single unsplit nodes.  Walk-entry edges connect each
    rail station node to every split node of nearby feeder stops (proximity
    ≤ 200 m, or ≤ 350 m when the stop name contains 'Bahnhof'/'HB').

    Dijkstra is run *from* each rail station, so it naturally finds the
    minimum-cost path (walk-on-board + in-vehicle + transfer penalties) to
    every reachable split node.  The result for a physical feeder stop is the
    minimum across all its split nodes.

    Returns
    -------
    dict
        feeder_stop_to_rail_times[feeder_stop_id] = {rail_stop_id: time_sec, ...}
    """
    import re

    print("  Building feeder network graph (node-split) ...")
    G = nx.Graph()

    # --- Rail station nodes (unsplit) ---
    rail_ids = set()
    for _, row in rail_stations.iterrows():
        sid = str(row['stop_id'])
        rail_ids.add(sid)
        G.add_node(sid, type='rail')

    # --- Feeder split nodes and segment edges ---
    # Collect all (stop_id, line) pairs that appear in segments
    feeder_ids = set(feeder_stops['stop_id'].astype(str))

    # Build a lookup: stop_id -> set of lines serving it
    stop_lines = {}   # stop_id (str) -> set of line_short_name (str)
    for _, row in feeder_segments.iterrows():
        line = str(row['line_short_name']) if pd.notna(row['line_short_name']) else '__unknown__'
        for sid in (str(row['from_stop_id']), str(row['to_stop_id'])):
            stop_lines.setdefault(sid, set()).add(line)

    def _split(stop_id, line):
        return (stop_id, line)

    # Add split nodes for every (stop, line) combination
    for sid, lines in stop_lines.items():
        for line in lines:
            G.add_node(_split(sid, line), type='feeder', stop_id=sid)

    # Segment edges — same line, directed travel time
    n_seg_edges = 0
    for _, row in feeder_segments.iterrows():
        from_id = str(row['from_stop_id'])
        to_id   = str(row['to_stop_id'])
        line    = str(row['line_short_name']) if pd.notna(row['line_short_name']) else '__unknown__'
        weight  = float(row['travel_time_min']) * 60  # seconds
        u = _split(from_id, line)
        v = _split(to_id,   line)
        if G.has_node(u) and G.has_node(v):
            # Keep minimum travel time if duplicate edges exist
            if G.has_edge(u, v):
                if G[u][v]['weight'] > weight:
                    G[u][v]['weight'] = weight
            else:
                G.add_edge(u, v, weight=weight)
                n_seg_edges += 1

    # Transfer edges — change line at the same physical stop
    n_transfer_edges = 0
    for sid, lines in stop_lines.items():
        line_list = list(lines)
        for a in range(len(line_list)):
            for b in range(a + 1, len(line_list)):
                u = _split(sid, line_list[a])
                v = _split(sid, line_list[b])
                if not G.has_edge(u, v):
                    G.add_edge(u, v, weight=TRANSFER_PENALTY_SEC)
                    n_transfer_edges += 1

    # --- Walk-entry edges: rail station -> feeder split nodes ---
    PROX_RADIUS_M = 200
    NAME_RADIUS_M = 350
    _BHF_RE = re.compile(r'\bBahnhof\b|\bHB\b', re.IGNORECASE)

    f_coords = np.column_stack([feeder_stops.geometry.x, feeder_stops.geometry.y])
    r_coords = np.column_stack([rail_stations.geometry.x, rail_stations.geometry.y])
    f_ids_arr = feeder_stops['stop_id'].astype(str).values
    r_ids_arr = rail_stations['stop_id'].astype(str).values
    f_names   = feeder_stops['stop_name'].fillna('').astype(str).values

    rail_tree = cKDTree(r_coords)

    n_prox_edges = 0
    n_name_edges = 0

    for i, (fsid, fname) in enumerate(zip(f_ids_arr, f_names)):
        is_named = bool(_BHF_RE.search(fname))
        radius = NAME_RADIUS_M if is_named else PROX_RADIUS_M
        near_rail = rail_tree.query_ball_point(f_coords[i], r=radius)
        if not near_rail:
            continue
        lines_at_stop = stop_lines.get(fsid, set())
        if not lines_at_stop:
            continue
        for j in near_rail:
            rsid = r_ids_arr[j]
            dist = np.sqrt(((f_coords[i] - r_coords[j]) ** 2).sum())
            walk_sec = dist / WALK_SPEED_MS
            # Connect rail node to every split node of this feeder stop
            for line in lines_at_stop:
                u = _split(fsid, line)
                if not G.has_node(u):
                    continue
                if G.has_edge(rsid, u):
                    if G[rsid][u]['weight'] > walk_sec:
                        G[rsid][u]['weight'] = walk_sec
                else:
                    G.add_edge(rsid, u, weight=walk_sec)
                    if is_named:
                        n_name_edges += 1
                    else:
                        n_prox_edges += 1

    print(f"    Graph: {G.number_of_nodes()} nodes, "
          f"{n_seg_edges} segment edges, "
          f"{n_transfer_edges} transfer edges ({TRANSFER_PENALTY_SEC}s penalty), "
          f"{n_prox_edges} proximity walk edges (≤{PROX_RADIUS_M}m), "
          f"{n_name_edges} name-match walk edges (Bahnhof/HB ≤{NAME_RADIUS_M}m)")
    print(f"    Rail stations: {len(rail_ids)}, Physical feeder stops: {len(feeder_ids)}")

    # --- Shortest paths from each rail station ---
    print("    Computing shortest paths from each rail station ...")
    # Result keyed by physical stop_id
    feeder_stop_to_rail_times = {sid: {} for sid in feeder_ids}

    for rail_id in rail_ids:
        if rail_id not in G.nodes:
            continue
        try:
            lengths = nx.single_source_dijkstra_path_length(G, rail_id, weight='weight')
        except nx.NetworkXError:
            continue
        for node_id, dist in lengths.items():
            # node_id is either a plain str (rail) or a (stop_id, line) tuple (feeder)
            if isinstance(node_id, tuple):
                physical_sid = node_id[0]
                if physical_sid in feeder_stop_to_rail_times:
                    existing = feeder_stop_to_rail_times[physical_sid].get(rail_id, np.inf)
                    if dist < existing:
                        feeder_stop_to_rail_times[physical_sid][rail_id] = dist

    n_reachable = sum(1 for v in feeder_stop_to_rail_times.values() if v)
    print(f"    {n_reachable}/{len(feeder_ids)} feeder stops reachable from at least one rail station")

    return feeder_stop_to_rail_times


# ===============================================================================
# STEP 4: CATCHMENT ALLOCATION (PT-Feeder only)
# ===============================================================================

def _compute_walk_to_rail_times(grid, rail_stations):
    """For each cell centroid, find rail stations within BUFFER_RAIL_M and
    compute walk time."""
    coords_grid = np.column_stack([grid.geometry.x, grid.geometry.y])
    coords_rail = np.column_stack([rail_stations.geometry.x, rail_stations.geometry.y])

    tree = cKDTree(coords_rail)
    results = []
    neighbors = tree.query_ball_point(coords_grid, r=BUFFER_RAIL_M)

    reli_values = grid['RELI'].values
    rail_stop_ids = rail_stations['stop_id'].values

    for i, near_idxs in enumerate(neighbors):
        for j in near_idxs:
            dist = np.sqrt((coords_grid[i, 0] - coords_rail[j, 0])**2 +
                           (coords_grid[i, 1] - coords_rail[j, 1])**2)
            time_sec = dist / WALK_SPEED_MS
            results.append({
                'RELI': reli_values[i],
                'rail_stop_id': str(rail_stop_ids[j]),
                'total_time_sec': time_sec,
                'access_mode': MODE_WALK
            })

    return pd.DataFrame(results)


def _compute_cycle_to_rail_times(grid, rail_stations):
    """For each cell centroid, find rail stations within CYCLE_RADIUS_M and
    compute cycling time."""
    coords_grid = np.column_stack([grid.geometry.x, grid.geometry.y])
    coords_rail = np.column_stack([rail_stations.geometry.x, rail_stations.geometry.y])

    tree = cKDTree(coords_rail)
    results = []
    neighbors = tree.query_ball_point(coords_grid, r=CYCLE_RADIUS_M)

    reli_values = grid['RELI'].values
    rail_stop_ids = rail_stations['stop_id'].values

    for i, near_idxs in enumerate(neighbors):
        for j in near_idxs:
            dist = np.sqrt((coords_grid[i, 0] - coords_rail[j, 0])**2 +
                           (coords_grid[i, 1] - coords_rail[j, 1])**2)
            time_sec = dist / CYCLE_SPEED_MS
            results.append({
                'RELI': reli_values[i],
                'rail_stop_id': str(rail_stop_ids[j]),
                'total_time_sec': time_sec,
                'access_mode': MODE_CYCLE
            })

    return pd.DataFrame(results)


def _compute_feeder_to_rail_times(grid, feeder_stops, feeder_stop_to_rail_times):
    """For each cell centroid, find reachable feeder stops within their buffer
    radii, then compute walk-to-stop + graph-derived time to each rail station.

    Transfer penalties are already embedded in feeder_stop_to_rail_times by
    _build_feeder_graph: a penalty of TRANSFER_PENALTY_SEC is added only when
    a path changes service line at an intermediate stop.  No flat penalty is
    applied here.
    """
    coords_grid = np.column_stack([grid.geometry.x, grid.geometry.y])
    results = []
    reli_values = grid['RELI'].values

    for radius, mode_code in [(BUFFER_BUS_M, MODE_BUS), (BUFFER_TRAM_M, MODE_TRAM)]:
        if mode_code == MODE_TRAM:
            subset = feeder_stops[feeder_stops['mode'].str.lower().str.contains('tram')]
        else:
            subset = feeder_stops[~feeder_stops['mode'].str.lower().str.contains('tram')]

        if len(subset) == 0:
            continue

        coords_fs = np.column_stack([subset.geometry.x, subset.geometry.y])
        tree = cKDTree(coords_fs)
        neighbors = tree.query_ball_point(coords_grid, r=radius)

        fs_stop_ids = subset['stop_id'].astype(str).values

        for i, near_idxs in enumerate(neighbors):
            for j in near_idxs:
                feeder_sid = fs_stop_ids[j]
                rail_times = feeder_stop_to_rail_times.get(feeder_sid, {})
                if not rail_times:
                    continue

                dist = np.sqrt((coords_grid[i, 0] - coords_fs[j, 0])**2 +
                               (coords_grid[i, 1] - coords_fs[j, 1])**2)
                walk_sec = dist / WALK_SPEED_MS

                for rail_id, graph_time_sec in rail_times.items():
                    # walk to first feeder stop + graph time (in-vehicle + any transfer penalties)
                    total = walk_sec + graph_time_sec
                    results.append({
                        'RELI': reli_values[i],
                        'rail_stop_id': rail_id,
                        'total_time_sec': total,
                        'access_mode': mode_code
                    })

    return pd.DataFrame(results)


def _allocate_cells(walk_df, cycle_df, feeder_df, grid, rail_stations, output_dir):
    """Winner-takes-all allocation: for each cell, pick the station with
    minimum total access time. Also save top-N candidates.

    Parameters
    ----------
    output_dir : str
        Method-specific data directory for saving candidates CSV.

    Returns
    -------
    pd.DataFrame
        Allocation with columns: RELI, E_KOORD, N_KOORD, station_id, id_point,
        access_time_sec, access_mode.
    """
    print("    Running winner-takes-all allocation ...")

    rail_id_map = dict(zip(rail_stations['stop_id'].astype(str),
                           rail_stations['id_point']))

    frames = [df for df in [walk_df, cycle_df, feeder_df] if len(df) > 0]
    if len(frames) == 0:
        print("    WARNING: No access routes found - all cells get NO_PT")
        combined = pd.DataFrame(columns=['RELI', 'rail_stop_id', 'total_time_sec', 'access_mode'])
    else:
        combined = pd.concat(frames, ignore_index=True)

    if len(combined) > 0:
        allocation = combined.loc[combined.groupby('RELI')['total_time_sec'].idxmin()].copy()

        top_n = (combined.sort_values('total_time_sec')
                 .groupby('RELI').head(MAX_CANDIDATE_STATIONS).copy())
        top_n['rank'] = top_n.groupby('RELI').cumcount() + 1
        top_n['id_point'] = top_n['rail_stop_id'].map(rail_id_map)
        top_n = top_n.rename(columns={'rail_stop_id': 'station_id',
                                       'total_time_sec': 'access_time_sec'})
        candidates_path = os.path.join(output_dir, 'cell_station_candidates.csv')
        top_n[['RELI', 'rank', 'station_id', 'id_point', 'access_time_sec', 'access_mode']].to_csv(
            candidates_path, index=False)
        print(f"    Top-{MAX_CANDIDATE_STATIONS} candidates saved -> {candidates_path}")
    else:
        allocation = pd.DataFrame(columns=['RELI', 'rail_stop_id', 'total_time_sec', 'access_mode'])

    allocation = allocation.rename(columns={'rail_stop_id': 'station_id',
                                             'total_time_sec': 'access_time_sec'})
    allocation['id_point'] = allocation['station_id'].map(rail_id_map)

    grid_coords = grid[['RELI', 'E_KOORD', 'N_KOORD']].drop_duplicates(subset='RELI')
    allocation = allocation.merge(grid_coords, on='RELI', how='right')

    allocation['station_id'] = allocation['station_id'].fillna(str(NO_PT_ID))
    allocation['id_point'] = allocation['id_point'].fillna(NO_PT_ID)
    allocation['access_time_sec'] = allocation['access_time_sec'].fillna(99999)
    allocation['access_mode'] = allocation['access_mode'].fillna(MODE_NO_PT)

    allocation['id_point'] = allocation['id_point'].astype(int)
    allocation['access_mode'] = allocation['access_mode'].astype(int)

    n_assigned = (allocation['id_point'] != NO_PT_ID).sum()
    print(f"    {n_assigned}/{len(allocation)} cells assigned to a rail station")

    return allocation


# ===============================================================================
# OUTPUTS
# ===============================================================================

def _build_catchment_gpkg(allocation, grid, output_dir):
    """Dissolve cells per station into catchment polygons.
    Schema: [train_station, id, geometry] for downstream compatibility."""
    print("  Building catchment GeoPackage ...")

    merged = allocation.merge(grid[['RELI', 'geometry']], on='RELI', how='left')
    merged = gpd.GeoDataFrame(merged, geometry='geometry', crs=CODEBASE_CRS)
    merged['geometry'] = merged.geometry.buffer(CELL_SIZE_M / 2, cap_style=3)

    dissolved = merged.dissolve(by='id_point', as_index=False)
    dissolved = dissolved.rename(columns={'id_point': 'id'})
    dissolved['train_station'] = dissolved['id'].copy()
    dissolved = dissolved[['train_station', 'id', 'geometry']].copy()
    dissolved['id'] = dissolved['id'].astype(int)
    dissolved['train_station'] = dissolved['train_station'].astype(int)

    out_path = os.path.join(output_dir, 'catchement.gpkg')
    dissolved.to_file(out_path, driver='GPKG')
    print(f"    Saved -> {out_path}  ({len(dissolved)} features)")

    return dissolved


def _build_catchment_tif(allocation, output_dir):
    """Write a 2-band GeoTIFF: Band 1 = access time (sec), Band 2 = station ID (ID_point)."""
    print("  Building catchment raster ...")

    e_min = int(allocation['E_KOORD'].min())
    e_max = int(allocation['E_KOORD'].max()) + CELL_SIZE_M
    n_min = int(allocation['N_KOORD'].min())
    n_max = int(allocation['N_KOORD'].max()) + CELL_SIZE_M

    width  = (e_max - e_min) // CELL_SIZE_M
    height = (n_max - n_min) // CELL_SIZE_M

    time_raster    = np.full((height, width), 99999, dtype=np.float32)
    station_raster = np.full((height, width), NO_PT_ID, dtype=np.float32)

    for _, row in allocation.iterrows():
        col = int((row['E_KOORD'] - e_min) // CELL_SIZE_M)
        r   = int((n_max - row['N_KOORD'] - CELL_SIZE_M) // CELL_SIZE_M)
        if 0 <= r < height and 0 <= col < width:
            time_raster[r, col]    = row['access_time_sec']
            station_raster[r, col] = row['id_point']

    transform = from_origin(e_min, n_max, CELL_SIZE_M, CELL_SIZE_M)
    out_path = os.path.join(output_dir, 'catchement.tif')

    with rasterio.open(out_path, 'w', driver='GTiff',
                       height=height, width=width, count=2,
                       dtype='float32', crs=CODEBASE_CRS,
                       transform=transform) as dst:
        dst.write(time_raster, 1)
        dst.write(station_raster, 2)

    print(f"    Saved -> {out_path}  ({width}x{height}, 2 bands)")


def _build_access_mode_tif(allocation, output_dir):
    """Write a single-band int8 GeoTIFF with access mode codes (0-4)."""
    print("  Building access mode raster ...")

    e_min = int(allocation['E_KOORD'].min())
    e_max = int(allocation['E_KOORD'].max()) + CELL_SIZE_M
    n_min = int(allocation['N_KOORD'].min())
    n_max = int(allocation['N_KOORD'].max()) + CELL_SIZE_M

    width  = (e_max - e_min) // CELL_SIZE_M
    height = (n_max - n_min) // CELL_SIZE_M

    mode_raster = np.full((height, width), MODE_NO_PT, dtype=np.int8)

    for _, row in allocation.iterrows():
        col = int((row['E_KOORD'] - e_min) // CELL_SIZE_M)
        r   = int((n_max - row['N_KOORD'] - CELL_SIZE_M) // CELL_SIZE_M)
        if 0 <= r < height and 0 <= col < width:
            mode_raster[r, col] = int(row['access_mode'])

    transform = from_origin(e_min, n_max, CELL_SIZE_M, CELL_SIZE_M)
    out_path = os.path.join(output_dir, 'access_mode.tif')

    with rasterio.open(out_path, 'w', driver='GTiff',
                       height=height, width=width, count=1,
                       dtype='int8', crs=CODEBASE_CRS,
                       transform=transform, nodata=MODE_NO_PT) as dst:
        dst.write(mode_raster, 1)

    print(f"    Saved -> {out_path}  ({width}x{height})")


def _build_visualisation(allocation, grid, rail_stations, boundary, method_label):
    """Produce a thesis-quality catchment visualisation map.
    Saved to plots/Catchment_Area/."""
    print(f"  Building {method_label} catchment visualisation ...")

    merged = allocation.merge(grid[['RELI', 'geometry']], on='RELI', how='left')
    merged = gpd.GeoDataFrame(merged, geometry='geometry', crs=CODEBASE_CRS)

    fig, ax = plt.subplots(1, 1, figsize=(14, 12))

    mode_colors = {
        MODE_NO_PT: '#d9d9d9',
        MODE_WALK:  '#2ca02c',
        MODE_BUS:   '#1f77b4',
        MODE_TRAM:  '#ff7f0e',
        MODE_CYCLE: '#9467bd'
    }
    mode_labels = {
        MODE_NO_PT: 'No PT access',
        MODE_WALK:  'Walk to rail',
        MODE_BUS:   'Bus feeder',
        MODE_TRAM:  'Tram feeder',
        MODE_CYCLE: 'Cycle to rail'
    }

    for mode, color in mode_colors.items():
        subset = merged[merged['access_mode'] == mode]
        if len(subset) > 0:
            subset.plot(ax=ax, color=color, markersize=1, alpha=0.6)

    rail_stations.plot(ax=ax, color='red', markersize=30, marker='^',
                       zorder=5, label='Rail stations')

    boundary_gdf = gpd.GeoDataFrame(geometry=[boundary], crs=CODEBASE_CRS)
    boundary_gdf.boundary.plot(ax=ax, color='black', linewidth=1.5, linestyle='--')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
               markersize=8, label=mode_labels[mode])
        for mode, color in mode_colors.items()
    ]
    legend_elements.append(
        Line2D([0], [0], marker='^', color='w', markerfacecolor='red',
               markersize=10, label='Rail station')
    )
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    ax.set_title(f'Catchment Area Allocation - {method_label}', fontsize=14)
    ax.set_xlabel('E [m]')
    ax.set_ylabel('N [m]')
    ax.set_aspect('equal')

    out_path = os.path.join(CATCHMENT_PLOT_DIR,
                            f'catchment_visualisation_{method_label.lower().replace(" ", "_")}.pdf')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"    Saved -> {out_path}")


# ===============================================================================
# DIFF COMPARISON PLOT
# ===============================================================================

def _build_diff_plot(muni_catchment, pt_catchment, pt_allocation,
                     pop_grid, empl_grid, rail_stations, boundary):
    """Cell-level comparison plot: Municipal vs PT-Feeder station assignment.

    Only cells with population and/or employment are coloured.  Colours:
      - Same station in both methods       → light grey
      - Different station in both methods   → orange (reassigned)
      - Municipal only (no PT-Feeder)       → red
      - PT-Feeder only (no Municipal)       → blue

    Combined station catchment boundaries from both methods are overlaid.

    Parameters
    ----------
    muni_catchment : gpd.GeoDataFrame
        Municipal catchment with columns [id_point, geometry], dissolved per station.
    pt_catchment : gpd.GeoDataFrame
        PT-Feeder catchment GPKG with columns [train_station, id, geometry].
    pt_allocation : pd.DataFrame
        PT-Feeder cell allocation with columns [RELI, id_point, E_KOORD, N_KOORD].
    pop_grid, empl_grid : gpd.GeoDataFrame
        Population / employment grids with RELI column.
    rail_stations : gpd.GeoDataFrame
    boundary : shapely.Polygon
    """
    if muni_catchment is None or pt_catchment is None:
        print("  Skipping diff plot - one method did not produce catchment geometry")
        return

    print("  Building Municipal vs PT-Feeder diff plot (cell-level) ...")

    # --- Cell-level municipal assignment via spatial join ---
    muni_cells = _assign_cells_to_municipal_catchment(
        pop_grid, empl_grid, muni_catchment)
    muni_cells = muni_cells.rename(columns={'id_point': 'muni_station'})
    muni_cells['muni_station'] = pd.to_numeric(
        muni_cells['muni_station'], errors='coerce')

    # --- PT-Feeder cell assignment ---
    pt_cells = pt_allocation[['RELI', 'id_point', 'E_KOORD', 'N_KOORD']].copy()
    pt_cells = pt_cells.rename(columns={'id_point': 'pt_station'})

    # Only cells with pop or empl
    pop_relis = set(pop_grid['RELI'].values)
    empl_relis = set(empl_grid['RELI'].values)
    active_relis = pop_relis | empl_relis
    pt_cells = pt_cells[pt_cells['RELI'].isin(active_relis)].copy()

    # Merge both assignments
    merged = pt_cells.merge(muni_cells, on='RELI', how='outer')

    # Fill coordinates for cells only in municipal (need grid coords)
    coord_lookup = pd.concat([
        pop_grid[['RELI', 'E_KOORD', 'N_KOORD']],
        empl_grid[['RELI', 'E_KOORD', 'N_KOORD']],
    ]).drop_duplicates(subset='RELI')
    missing_coords = merged['E_KOORD'].isna()
    if missing_coords.any():
        fill = merged.loc[missing_coords, ['RELI']].merge(
            coord_lookup, on='RELI', how='left', suffixes=('_old', ''))
        merged.loc[missing_coords, 'E_KOORD'] = fill['E_KOORD'].values
        merged.loc[missing_coords, 'N_KOORD'] = fill['N_KOORD'].values

    merged = merged.dropna(subset=['E_KOORD', 'N_KOORD'])

    # Replace NO_PT sentinel with NaN for cleaner logic
    merged.loc[merged['pt_station'] == NO_PT_ID, 'pt_station'] = np.nan

    # Classify each cell
    has_muni = merged['muni_station'].notna()
    has_pt   = merged['pt_station'].notna()
    same     = has_muni & has_pt & (merged['muni_station'] == merged['pt_station'])
    diff     = has_muni & has_pt & (merged['muni_station'] != merged['pt_station'])
    muni_only = has_muni & ~has_pt
    pt_only   = ~has_muni & has_pt

    CLR_SAME      = '#D3D3D3'
    CLR_DIFFERENT = '#E67E22'
    CLR_MUNI_ONLY = '#B2182B'
    CLR_PT_ONLY   = '#2166AC'

    merged['category'] = ''
    merged.loc[same, 'category']      = 'same'
    merged.loc[diff, 'category']      = 'different'
    merged.loc[muni_only, 'category'] = 'muni_only'
    merged.loc[pt_only, 'category']   = 'pt_only'
    # Cells with neither assignment are ignored
    merged = merged[merged['category'] != ''].copy()

    # Build cell square geometries
    merged['geometry'] = [
        Polygon([
            (r['E_KOORD'], r['N_KOORD']),
            (r['E_KOORD'] + CELL_SIZE_M, r['N_KOORD']),
            (r['E_KOORD'] + CELL_SIZE_M, r['N_KOORD'] + CELL_SIZE_M),
            (r['E_KOORD'], r['N_KOORD'] + CELL_SIZE_M),
        ])
        for _, r in merged.iterrows()
    ]
    merged = gpd.GeoDataFrame(merged, geometry='geometry', crs=CODEBASE_CRS)

    # Summary stats
    n_same = same.sum()
    n_diff = diff.sum()
    n_muni = muni_only.sum()
    n_pt   = pt_only.sum()
    print(f"    Cells — same station: {n_same:,}, different: {n_diff:,}, "
          f"municipal only: {n_muni:,}, PT-feeder only: {n_pt:,}")

    # --- Plot ---
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))

    # Light grey background
    boundary_gdf = gpd.GeoDataFrame(geometry=[boundary], crs=CODEBASE_CRS)
    boundary_gdf.plot(ax=ax, color='#F0F0F0', edgecolor='none')

    # Plot cells by category
    cat_cfg = [
        ('same',      CLR_SAME,      'Same station'),
        ('different', CLR_DIFFERENT, 'Different station (reassigned)'),
        ('muni_only', CLR_MUNI_ONLY, 'Municipal only'),
        ('pt_only',   CLR_PT_ONLY,   'PT-Feeder only'),
    ]
    legend_handles = []
    for cat, colour, label in cat_cfg:
        subset = merged[merged['category'] == cat]
        if len(subset) > 0:
            subset.plot(ax=ax, color=colour, edgecolor='none', zorder=2)
            legend_handles.append(
                Patch(facecolor=colour, edgecolor='none', label=label))

    # Normalise catchment column names
    muni_c = muni_catchment.copy()
    if 'id_point' not in muni_c.columns and 'id' in muni_c.columns:
        muni_c = muni_c.rename(columns={'id': 'id_point'})
    muni_c = muni_c[muni_c['id_point'] != NO_PT_ID]

    pt_c = pt_catchment.copy()
    if 'id_point' not in pt_c.columns and 'id' in pt_c.columns:
        pt_c = pt_c.rename(columns={'id': 'id_point'})
    pt_c = pt_c[pt_c['id_point'] != NO_PT_ID]

    # Combined station catchment boundaries
    muni_c.boundary.plot(ax=ax, color='#d6604d', linewidth=0.5, linestyle='--',
                         alpha=0.7, zorder=3)
    pt_c.boundary.plot(ax=ax, color='#4393c3', linewidth=0.5, linestyle='-',
                       alpha=0.7, zorder=3)

    legend_handles += [
        Line2D([0], [0], color='#d6604d', linewidth=1, linestyle='--',
               label='Municipal catchment boundary'),
        Line2D([0], [0], color='#4393c3', linewidth=1, linestyle='-',
               label='PT-Feeder catchment boundary'),
    ]

    # Rail stations
    rail_stations.plot(ax=ax, color='black', markersize=20, marker='^', zorder=5)
    legend_handles.append(
        Line2D([0], [0], marker='^', color='w', markerfacecolor='black',
               markersize=8, label='Rail station'))

    # Study area boundary
    boundary_gdf.boundary.plot(ax=ax, color='black', linewidth=1.8, linestyle='--', zorder=4)
    legend_handles.append(
        Line2D([0], [0], color='black', linewidth=1.8, linestyle='--',
               label='Study area boundary'))

    ax.legend(handles=legend_handles, loc='upper right', fontsize=8, framealpha=0.9)
    ax.set_title('Catchment Comparison: Municipal vs PT-Feeder (populated cells)',
                 fontsize=14)
    ax.set_xlabel('E [m]')
    ax.set_ylabel('N [m]')
    ax.set_aspect('equal')

    out_path = os.path.join(CATCHMENT_PLOT_DIR, 'catchment_diff_municipal_vs_pt_feeder.pdf')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"    Saved -> {out_path}")


# ===============================================================================
# ACCESS TIME PLOTS
# ===============================================================================

# Shared bin edges and labels for all access-time plots (minutes)
_ACCESS_BINS   = [0, 3, 6, 9, 12, 15, 20, 25, 30, np.inf]
_ACCESS_LABELS = ['0–3', '3–6', '6–9', '9–12', '12–15', '15–20', '20–25', '25–30', '≥ 30']
_ACCESS_GREY   = '#BDBDBD'


def _plot_access_times(walk_df, cycle_df, feeder_df, alloc_pop, pop_grid,
                       rail_stations, feeder_stops, boundary):
    """Produce four access-time maps (plasma_r palette, discrete 3/5-min bins).

    Plots
    -----
    1. Walk        — cells within BUFFER_RAIL_M of a rail station
    2. Cycle       — cells within CYCLE_RADIUS_M of a rail station
    3. Feeder      — cells within bus/tram buffer whose feeder stop is graph-reachable
    4. Best choice — winner-takes-all result (alloc_pop), all in-buffer cells

    Cells outside their respective buffer are shown in grey.
    Colour scale: plasma_r (yellow = fast, dark purple/blue = slow).
    Bins: 0–3, 3–6, 6–9, 9–12, 12–15 min (3-min steps) then
          15–20, 20–25, 25–30, ≥30 min (5-min steps).
    """
    print("  Plotting access time maps ...")

    n_bins  = len(_ACCESS_LABELS)
    cmap    = plt.get_cmap('plasma_r', n_bins)
    colours = [cmap(k / (n_bins - 1)) for k in range(n_bins)]

    # Pre-build square polygon geometry for the grid (100 m × 100 m)
    from shapely.geometry import box as _box
    squares = [
        _box(row['E_KOORD'], row['N_KOORD'],
             row['E_KOORD'] + CELL_SIZE_M, row['N_KOORD'] + CELL_SIZE_M)
        for _, row in pop_grid.iterrows()
    ]
    grid_plot = gpd.GeoDataFrame(
        pop_grid[['RELI', 'E_KOORD', 'N_KOORD']].copy(),
        geometry=squares, crs=CODEBASE_CRS
    )

    def _class(time_sec):
        """Convert array of seconds to 0-based class index into _ACCESS_BINS."""
        mins = np.array(time_sec, dtype=float) / 60.0
        idx  = np.digitize(mins, _ACCESS_BINS[1:], right=False)
        return np.clip(idx, 0, n_bins - 1)

    def _render(title, fname, reli_series, time_series):
        """Plot one access-time map given aligned RELI and time_sec arrays."""
        # Merge onto grid_plot
        df = pd.DataFrame({'RELI': reli_series, 'time_sec': time_series})
        # Keep best time per cell (minimum across candidate stations)
        df = df.groupby('RELI', as_index=False)['time_sec'].min()
        gdf = grid_plot.merge(df, on='RELI', how='left')

        in_buffer = gdf['time_sec'].notna()
        gdf['class_idx'] = -1  # -1 = outside buffer (grey)
        gdf.loc[in_buffer, 'class_idx'] = _class(gdf.loc[in_buffer, 'time_sec'].values)

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        # Grey for out-of-buffer cells
        if (~in_buffer).any():
            gdf[~in_buffer].plot(ax=ax, color=_ACCESS_GREY, edgecolor='none', linewidth=0)

        # Coloured cells per class
        for ci, colour in enumerate(colours):
            mask = gdf['class_idx'] == ci
            if mask.any():
                gdf[mask].plot(ax=ax, color=colour, edgecolor='none', linewidth=0)

        # Boundary
        boundary_gdf = gpd.GeoDataFrame(geometry=[boundary], crs=CODEBASE_CRS)
        boundary_gdf.boundary.plot(ax=ax, color='black', linewidth=1.8,
                                   linestyle='--', zorder=5)

        # Legend — only include colours that actually appear in the plot
        used_classes = set(gdf.loc[in_buffer, 'class_idx'].astype(int).unique())
        legend_handles = [
            Patch(facecolor=colours[ci], edgecolor='none', label=_ACCESS_LABELS[ci])
            for ci in sorted(used_classes)
        ]
        legend_handles.append(Patch(facecolor=_ACCESS_GREY, edgecolor='none',
                                    label='Outside buffer'))
        legend_handles.append(Line2D([0], [0], color='black', linewidth=1.8,
                                     linestyle='--', label='Catchment area boundary'))

        ax.legend(handles=legend_handles, title='Access time (min)',
                  title_fontsize=8, fontsize=7,
                  loc='upper right', framealpha=0.9)

        ax.set_title(title, fontsize=13)
        ax.set_xlabel('E [m]')
        ax.set_ylabel('N [m]')
        ax.set_aspect('equal')

        out_path = os.path.join(CATCHMENT_PLOT_DIR, fname)
        fig.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"    Saved -> {out_path}")

    # --- 1. Walk ---
    if len(walk_df) > 0:
        _render('Access Time — Walk to Rail Station',
                'plot_access_time_walk.pdf',
                walk_df['RELI'].values,
                walk_df['total_time_sec'].values)

    # --- 2. Cycle ---
    if len(cycle_df) > 0:
        _render('Access Time — Cycle to Rail Station',
                'plot_access_time_cycle.pdf',
                cycle_df['RELI'].values,
                cycle_df['total_time_sec'].values)

    # --- 3. Feeder (bus/tram) ---
    if len(feeder_df) > 0:
        _render('Access Time — PT Feeder to Rail Station',
                'plot_access_time_feeder.pdf',
                feeder_df['RELI'].values,
                feeder_df['total_time_sec'].values)

    # --- 4. Best choice (winner-takes-all from alloc_pop) ---
    # alloc_pop already has the minimum-time station per cell
    best_mask = alloc_pop['id_point'] != NO_PT_ID
    if best_mask.any():
        _render('Access Time — Best Mode (Walk / Cycle / PT)',
                'plot_access_time_best.pdf',
                alloc_pop.loc[best_mask, 'RELI'].values,
                alloc_pop.loc[best_mask, 'access_time_sec'].values)


# ===============================================================================
# ORCHESTRATORS
# ===============================================================================

def _run_pt_feeder_method(boundary, pop_grid, empl_grid, temporal='all'):
    """Full PT-Feeder allocation pipeline.

    Parameters
    ----------
    temporal : 'all' | 'peak' | 'offpeak'
        Which temporal variant of the network files to load.

    Returns
    -------
    tuple(gpd.GeoDataFrame, pd.DataFrame)
        PT-Feeder catchment polygons and the population cell allocation.
    """
    print("\n--- PT-Feeder Method ---")
    st = time.time()

    # Step 2-PT: Load network data
    feeder_stops = _load_feeder_stops(boundary, temporal)
    rail_stations = _load_rail_stations(boundary, temporal)
    feeder_segments = _load_feeder_segments(temporal)
    _build_pt_buffers(feeder_stops, rail_stations)

    print(f"  [Step 2-PT complete: {time.time() - st:.1f}s]")
    st = time.time()

    # Step 3: Build feeder graph
    feeder_stop_to_rail_times = _build_feeder_graph(
        feeder_stops, feeder_segments, rail_stations)

    print(f"  [Step 3 complete: {time.time() - st:.1f}s]")
    st = time.time()

    # Step 4: Allocate cells - population grid
    print("\n  Allocating population cells ...")
    walk_pop  = _compute_walk_to_rail_times(pop_grid, rail_stations)
    cycle_pop = _compute_cycle_to_rail_times(pop_grid, rail_stations)
    feeder_pop = _compute_feeder_to_rail_times(pop_grid, feeder_stops,
                                                feeder_stop_to_rail_times)
    alloc_pop = _allocate_cells(walk_pop, cycle_pop, feeder_pop,
                                pop_grid, rail_stations, PT_FEEDER_DATA_DIR)

    print(f"  [Pop allocation complete: {time.time() - st:.1f}s]")
    st = time.time()

    # Step 4: Allocate cells - employment grid
    print("\n  Allocating employment cells ...")
    walk_empl  = _compute_walk_to_rail_times(empl_grid, rail_stations)
    cycle_empl = _compute_cycle_to_rail_times(empl_grid, rail_stations)
    feeder_empl = _compute_feeder_to_rail_times(empl_grid, feeder_stops,
                                                 feeder_stop_to_rail_times)
    alloc_empl = _allocate_cells(walk_empl, cycle_empl, feeder_empl,
                                  empl_grid, rail_stations, PT_FEEDER_DATA_DIR)

    empl_alloc_path = os.path.join(PT_FEEDER_DATA_DIR, 'employment_allocation.csv')
    alloc_empl.to_csv(empl_alloc_path, index=False)
    print(f"  Employment allocation saved -> {empl_alloc_path}")

    print(f"  [Empl allocation complete: {time.time() - st:.1f}s]")
    st = time.time()

    # Outputs - produced from population allocation (primary)
    pt_catchment = _build_catchment_gpkg(alloc_pop, pop_grid, PT_FEEDER_DATA_DIR)
    _build_catchment_tif(alloc_pop, PT_FEEDER_DATA_DIR)
    _build_access_mode_tif(alloc_pop, PT_FEEDER_DATA_DIR)
    _build_visualisation(alloc_pop, pop_grid, rail_stations, boundary, 'PT-Feeder')
    _plot_access_times(walk_pop, cycle_pop, feeder_pop, alloc_pop, pop_grid,
                       rail_stations, feeder_stops, boundary)

    print(f"  [Outputs complete: {time.time() - st:.1f}s]")

    return pt_catchment, alloc_pop


# ===============================================================================
# PUBLIC ENTRY POINT
# ===============================================================================

def _interactive_config():
    """Prompt the user to choose method, network folder, and temporal variant.

    Returns
    -------
    dict with keys:
        method        : 'municipal' | 'pt_feeder' | 'both'
        network_folder: str  – subfolder name under FEEDER_LINES_DIR / RAIL_PROCESSED_DIR
        temporal      : 'all' | 'peak' | 'offpeak'
    """
    print("=" * 70)
    print("catchment_allocate.py — PIPELINE CONFIGURATION")
    print("=" * 70)

    # --- A. Method selection ---
    print("\nA. CATCHMENT METHOD")
    print("   1) Both        — run both and produce comparison diff plot")
    print("   2) Municipal   — commune centroid-based assignment")
    print("   3) PT-Feeder   — GTFS multimodal network routing")

    while True:
        choice = input("\n   Select method (1-3) [1]: ").strip() or "1"
        if choice in ('1', '2', '3'):
            break
        print("   Invalid selection. Please enter 1, 2, or 3.")

    method_map = {'1': 'both', '2': 'municipal', '3': 'pt_feeder'}
    method = method_map[choice]

    # --- B. Network folder (only needed for PT-Feeder) ---
    network_folder = GTFS_NETWORK_FOLDER
    temporal = 'all'
    subfolders = []
    if method in ('pt_feeder', 'both'):
        print("\nB. GTFS NETWORK FOLDER")
        feeder_base = os.path.join(paths.MAIN, paths.FEEDER_LINES_DIR)
        if os.path.isdir(feeder_base):
            subfolders = sorted([
                d for d in os.listdir(feeder_base)
                if os.path.isdir(os.path.join(feeder_base, d))
            ])
            if subfolders:
                print(f"   Available subfolders under {paths.FEEDER_LINES_DIR}:")
                for i, sf in enumerate(subfolders, 1):
                    print(f"     {i}) {sf}")

        while True:
            raw = input(
                f"\n   Enter network folder name [{GTFS_NETWORK_FOLDER}]: "
            ).strip() or GTFS_NETWORK_FOLDER
            # Allow numeric selection
            if raw.isdigit() and 1 <= int(raw) <= len(subfolders):
                raw = subfolders[int(raw) - 1]
            stops_check = os.path.join(feeder_base, raw, 'pt_feeder_stops.gpkg')
            if os.path.exists(stops_check):
                network_folder = raw
                break
            print(f"   Folder not found or missing pt_feeder_stops.gpkg: {raw}")

        # --- C. Temporal network variant ---
        print("\nC. TEMPORAL NETWORK VARIANT")
        print("   1) All-day  — full-day network (pt_feeder_stops.gpkg / rail_stops.gpkg)")
        print("   2) Peak     — AM+PM peak services only  (_peak files)")
        print("   3) Off-peak — off-peak services only    (_offpeak files)")

        # Check which variants are available
        feeder_folder = os.path.join(feeder_base, network_folder)
        has_peak    = os.path.exists(os.path.join(feeder_folder, 'Peak',     'pt_feeder_stops_peak.gpkg'))
        has_offpeak = os.path.exists(os.path.join(feeder_folder, 'Off_Peak', 'pt_feeder_stops_offpeak.gpkg'))
        if not has_peak:
            print("   (peak files not found in folder — option 2 unavailable)")
        if not has_offpeak:
            print("   (offpeak files not found in folder — option 3 unavailable)")

        valid = {'1'}
        if has_peak:
            valid.add('2')
        if has_offpeak:
            valid.add('3')

        while True:
            tchoice = input("\n   Select temporal variant (1-3) [1]: ").strip() or "1"
            if tchoice in valid:
                break
            print(f"   Invalid or unavailable selection. Choose from: {sorted(valid)}")

        temporal_map = {'1': 'all', '2': 'peak', '3': 'offpeak'}
        temporal = temporal_map[tchoice]

    # --- Summary ---
    method_labels = {
        'municipal': 'Municipal (commune centroid)',
        'pt_feeder': 'PT-Feeder (GTFS multimodal)',
        'both':      'Both + diff comparison plot',
    }
    temporal_labels = {'all': 'All-day', 'peak': 'Peak only', 'offpeak': 'Off-peak only'}
    print("\n" + "-" * 70)
    print("  CONFIGURATION SUMMARY")
    print("-" * 70)
    print(f"  Method         : {method_labels[method]}")
    if method in ('pt_feeder', 'both'):
        print(f"  Network folder : {network_folder}")
        print(f"  Temporal       : {temporal_labels[temporal]}")
    print("-" * 70)

    return {'method': method, 'network_folder': network_folder, 'temporal': temporal}


def get_catchment(use_cache: bool, method: str = 'both',
                  network_folder: str = GTFS_NETWORK_FOLDER,
                  temporal: str = 'all') -> None:
    """Run the selected catchment method(s) and produce outputs.

    Parameters
    ----------
    use_cache      : If True, skip regeneration when all output files exist.
    method         : 'municipal' | 'pt_feeder' | 'both'
    network_folder : Subfolder under FEEDER_LINES_DIR / RAIL_PROCESSED_DIR
                     containing the pre-built GTFS network.
    temporal       : 'all' | 'peak' | 'offpeak' — which temporal variant
                     of the network files (stops + segments) to load.
    """
    global GTFS_NETWORK_FOLDER
    GTFS_NETWORK_FOLDER = network_folder

    os.chdir(paths.MAIN)
    _ensure_dirs()

    temporal_labels = {'all': 'All-day', 'peak': 'Peak only', 'offpeak': 'Off-peak only'}
    print("=" * 70)
    print("CATCHMENT ALLOCATION")
    print(f"  Method : {method}")
    if method in ('pt_feeder', 'both'):
        print(f"  Network  : {network_folder}")
        print(f"  Temporal : {temporal_labels.get(temporal, temporal)}")
        print(f"  Transfer penalty: {TRANSFER_PENALTY_SEC:.0f}s "
              f"({cp.comfort_weighted_change_time} min, Axhausen 2014)")
    print("=" * 70)

    total_start = time.time()

    # Cache check (only meaningful when running both methods)
    if use_cache and method == 'both':
        expected_files = [
            os.path.join(CATCHMENT_DATA_DIR, 'population_2023.tif'),
            os.path.join(CATCHMENT_DATA_DIR, 'employment_2023.tif'),
            os.path.join(CATCHMENT_DATA_DIR, 'municipal_pop_empl_summary.csv'),
            os.path.join(PT_FEEDER_DATA_DIR, 'catchement.gpkg'),
            os.path.join(PT_FEEDER_DATA_DIR, 'catchement.tif'),
            os.path.join(PT_FEEDER_DATA_DIR, 'access_mode.tif'),
            os.path.join(MUNICIPAL_DATA_DIR, 'catchement.gpkg'),
            os.path.join(CATCHMENT_PLOT_DIR, 'catchment_diff_municipal_vs_pt_feeder.pdf'),
        ]
        all_exist = all(os.path.exists(f) for f in expected_files)
        if all_exist:
            print("Using cached catchment files - skipping generation.")
            return
        else:
            missing = [f for f in expected_files if not os.path.exists(f)]
            print(f"Cache enabled but {len(missing)} file(s) missing. Regenerating ...")

    # --- Step 1: Shared data loading ---
    print("\n[Step 1] Loading shared data ...")
    boundary  = _load_catchment_boundary()
    pop_grid  = _load_population_grid(boundary)
    empl_grid = _load_employment_grid(boundary)
    summary   = _cumulate_per_municipality(pop_grid, empl_grid, boundary)
    _plot_municipal_distributions(summary, boundary)
    _plot_raster_map(pop_grid,  'population', boundary)
    _plot_raster_map(empl_grid, 'employment', boundary)

    muni_catchment = None
    pt_catchment   = None
    pt_allocation  = None

    if method in ('municipal', 'both'):
        muni_catchment = _run_municipal_method(boundary, pop_grid, empl_grid)

        # Municipal catchment visualisation plots
        if muni_catchment is not None:
            rail_stations = _load_rail_stations(boundary, temporal)
            _plot_municipal_catchments(muni_catchment, pop_grid, empl_grid,
                                       rail_stations, boundary)
            _plot_municipal_catchments_network(muni_catchment, rail_stations,
                                               boundary, temporal)

    if method in ('pt_feeder', 'both'):
        pt_catchment, pt_allocation = _run_pt_feeder_method(
            boundary, pop_grid, empl_grid, temporal)

    if method == 'both':
        print("\n[Comparison] Building diff plot ...")
        rail_stations = _load_rail_stations(boundary, temporal)
        _build_diff_plot(muni_catchment, pt_catchment, pt_allocation,
                         pop_grid, empl_grid, rail_stations, boundary)

    elapsed = time.time() - total_start
    print(f"\nCatchment allocation complete - {elapsed:.1f}s total")


if __name__ == '__main__':
    cfg = _interactive_config()
    get_catchment(use_cache=False,
                  method=cfg['method'],
                  network_folder=cfg['network_folder'],
                  temporal=cfg['temporal'])
