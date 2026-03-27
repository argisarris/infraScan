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
#   Plot outputs  -> plots/Catchment_Area/            (shared Step 1 plots)
#                    plots/Catchment_Area/Municipal/  (municipal method)
#                    plots/Catchment_Area/PT_Feeder/  (PT-feeder method)

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
from rasterio.transform import from_origin
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
from shapely.prepared import prep

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
MUNICIPAL_PLOT_DIR  = os.path.join(CATCHMENT_PLOT_DIR, 'Municipal')
PT_FEEDER_PLOT_DIR  = os.path.join(CATCHMENT_PLOT_DIR, 'PT_Feeder')


def _ensure_dirs():
    """Create all output directories."""
    for d in [CATCHMENT_DATA_DIR, CATCHMENT_PLOT_DIR,
              MUNICIPAL_DATA_DIR, PT_FEEDER_DATA_DIR,
              MUNICIPAL_PLOT_DIR, PT_FEEDER_PLOT_DIR]:
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
    """Shared loader for population / employment CSVs.

    Cells are included if their centroid is strictly inside the boundary OR
    if at least 50 % of the 100×100 m cell area intersects the boundary
    (edge-cell rule).  A fast centroid pre-filter limits the expensive area
    check to cells within one cell-width of the boundary.
    """
    # Swiss CSVs may use comma as decimal separator (e.g. "5,376" → 5.376).
    for col in ['E_KOORD', 'N_KOORD', 'NUMMER']:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(',', '.', regex=False),
            errors='coerce',
        )
    if 'CLASS' in df.columns:
        df['CLASS'] = pd.to_numeric(df['CLASS'], errors='coerce')
    df = df.dropna(subset=['E_KOORD', 'N_KOORD', 'NUMMER'])
    df = df[df['NUMMER'] > 0].copy()

    # Compute cell centroids (bottom-left corner + 50 m)
    df['cx'] = df['E_KOORD'] + 50
    df['cy'] = df['N_KOORD'] + 50
    geometry = gpd.points_from_xy(df['cx'], df['cy'])
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=CODEBASE_CRS)

    # Step 1: fast pre-filter — drop cells more than one cell-width outside
    from shapely.prepared import prep
    from shapely.geometry import box as _cell_box
    prep_expanded = prep(boundary.buffer(CELL_SIZE_M))
    gdf = gdf[gdf.geometry.apply(prep_expanded.contains)].copy()

    # Step 2: centroid strictly inside → keep unconditionally
    prep_boundary = prep(boundary)
    inside_mask = gdf.geometry.apply(prep_boundary.contains)
    inside = gdf[inside_mask]

    # Step 3: edge cells — keep if ≥50 % of cell area intersects boundary
    edge_gdf = gdf[~inside_mask]
    cell_area = CELL_SIZE_M * CELL_SIZE_M
    if len(edge_gdf) > 0:
        keep_idx = [
            idx for idx, row in edge_gdf.iterrows()
            if _cell_box(row['E_KOORD'], row['N_KOORD'],
                         row['E_KOORD'] + CELL_SIZE_M,
                         row['N_KOORD'] + CELL_SIZE_M
                         ).intersection(boundary).area / cell_area >= 0.5
        ]
        gdf = pd.concat([inside, edge_gdf.loc[keep_idx]]).copy()
    else:
        gdf = inside.copy()

    print(f"    {label}: {len(gdf):,} cells within study area (≥50 % overlap)")

    area_tag = '_'.join(settings.CATCHMENT_CANTON).replace(' ', '')

    # Save filtered CSV (include CLASS if present)
    save_cols = ['RELI', 'E_KOORD', 'N_KOORD', 'NUMMER']
    if 'CLASS' in gdf.columns:
        save_cols.append('CLASS')
    csv_path = os.path.join(CATCHMENT_DATA_DIR, f'{label}_2023_{area_tag}.csv')
    gdf[save_cols].to_csv(csv_path, index=False, sep=';')
    print(f"    Saved filtered CSV -> {csv_path}")

    return gdf


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
    For MultiPolygon boundaries a separate plot is produced per component.
    """
    print("  Plotting municipal distributions ...")

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
        colors = ['#C7E9C0', '#74C476', '#238B45', '#2C7FB8', '#253494'],
        col_header = 'Total full-time equivalents',
    )

    # Load lakes once for all plots in this function
    lakes = None
    if os.path.exists(paths.LAKES_SHP):
        lakes = gpd.read_file(paths.LAKES_SHP).to_crs(CODEBASE_CRS)
        lakes = lakes[lakes.geometry.intersects(boundary)].copy()

    components = (list(boundary.geoms)
                  if boundary.geom_type == 'MultiPolygon' else [boundary])

    for col, title_word, cfg in [
        ('total_population', 'Population', pop_cfg),
        ('total_employment', 'Employment', empl_cfg),
    ]:
        bins   = cfg['bins']
        labels = cfg['labels']
        colors = cfg['colors']

        for part_idx, component in enumerate(components):
            part_suffix = f'_part{part_idx + 1}' if len(components) > 1 else ''

            clipped_summary = gpd.clip(summary_df, component)
            vals      = clipped_summary[col].fillna(0).values
            zero_mask = vals <= 0
            # fillna(0) before astype(int) handles the NaN pd.cut returns
            # for exact-zero values (which fall below the open lower bin edge)
            class_idx = (pd.Series(pd.cut(vals, bins=bins, labels=False, right=True))
                           .clip(0, len(labels) - 1)
                           .fillna(0)
                           .astype(int)
                           .values)

            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            ax.set_facecolor('#E8E8E8')   # grey outside the catchment

            # White fill for catchment interior (zero-value munis show as white)
            comp_gdf = gpd.GeoDataFrame(geometry=[component], crs=CODEBASE_CRS)
            comp_gdf.plot(ax=ax, color='white', edgecolor='none', zorder=0)

            for ci, color in enumerate(colors):
                mask = (class_idx == ci) & (~zero_mask)
                if not mask.any():
                    continue
                clipped_summary[mask].plot(ax=ax, color=color,
                                           edgecolor='grey', linewidth=0.4,
                                           zorder=1)

            # Lakes above fill, below boundary line
            if lakes is not None and not lakes.empty:
                lakes_clip = gpd.clip(lakes, component)
                if not lakes_clip.empty:
                    lakes_clip.plot(ax=ax, color='#A8D8EA', edgecolor='none',
                                    zorder=4)

            # Catchment area boundary
            comp_gdf.boundary.plot(ax=ax, color='black', linewidth=1.8,
                                   linestyle='--', zorder=5)

            bx_min, by_min, bx_max, by_max = component.bounds
            pad = 200
            ax.set_xlim(bx_min - pad, bx_max + pad)
            ax.set_ylim(by_min - pad, by_max + pad)

            legend_handles = []
            for color, lbl in zip(colors, labels):
                legend_handles.append(
                    Patch(facecolor=color, edgecolor='none', label=lbl))
            legend_handles.append(
                Line2D([0], [0], color='black', linewidth=1.8,
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

            fname = f'plot_{title_word.lower()}_by_municipality{part_suffix}.pdf'
            out_path = os.path.join(CATCHMENT_PLOT_DIR, fname)
            fig.savefig(out_path, bbox_inches='tight', dpi=150)
            plt.close(fig)
            print(f"    Saved -> {out_path}")


def _plot_raster_map(gdf, label, boundary):
    """Render a population or employment grid as a coloured raster map,
    styled to match the corresponding map.geo.admin FSO layer.

    Population  → FSO "Population Statistics: Inhabitants"
                  6 discrete classes, yellow→red palette
    Employment  → FSO "Enterprise Statistics: Employment (FTE)"
                  5 discrete classes, yellow→green→blue palette

    Cells are drawn as 100 m × 100 m square patches.
    Grey axes background outside the catchment; white interior for zero cells.
    Saved to plots/Catchment_Area/.
    For MultiPolygon boundaries a separate plot is produced per component.
    """
    print(f"  Plotting {label} raster map ...")

    is_pop = (label == 'population')

    if is_pop:
        bins   = [0, 3, 6, 15, 40, 120, np.inf]
        labels = ['1–3', '4–6', '7–15', '16–40', '41–120', 'more than 120']
        colors = ['#FFFFB2', '#FECC5C', '#FD8D3C', '#F03B20', '#BD0026', '#800026']
        col_header = 'Inhabitants per ha'
        title      = 'Population'
    else:
        bins   = [0, 40, 75, 150, 300, np.inf]
        labels = ['0.1–40', '40.1–75', '75.1–150', '150.1–300', 'more than 300']
        colors = ['#C7E9C0', '#74C476', '#238B45', '#2C7FB8', '#253494']
        col_header = 'Full-time equivalents per ha'
        title      = 'Employment (FTE)'

    n_classes = len(labels)

    vals = gdf['NUMMER'].values
    class_idx = np.digitize(vals, bins[1:], right=False)
    class_idx = np.clip(class_idx, 0, n_classes - 1)

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

    # Load municipal boundaries once (pre-filter to full boundary extent)
    muni = gpd.read_file(paths.MUNICIPAL_BOUNDARIES_GPKG).to_crs(CODEBASE_CRS)
    if 'objektart' in muni.columns:
        muni = muni[muni['objektart'] == 'Gemeindegebiet']
    muni = muni[muni.geometry.intersects(boundary)].copy()

    # Load lakes once
    lakes = None
    if os.path.exists(paths.LAKES_SHP):
        lakes = gpd.read_file(paths.LAKES_SHP).to_crs(CODEBASE_CRS)
        lakes = lakes[lakes.geometry.intersects(boundary)].copy()

    components = (list(boundary.geoms)
                  if boundary.geom_type == 'MultiPolygon' else [boundary])

    for part_idx, component in enumerate(components):
        part_suffix = f'_part{part_idx + 1}' if len(components) > 1 else ''

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.set_facecolor('#E8E8E8')   # grey outside the catchment

        # White fill for catchment interior (zero-value cells show as white)
        comp_gdf = gpd.GeoDataFrame(geometry=[component], crs=CODEBASE_CRS)
        comp_gdf.plot(ax=ax, color='white', edgecolor='none', zorder=0)

        # Clip and plot cell squares for this component
        plot_comp = gpd.clip(plot_gdf, component)
        for ci, (color, lbl) in enumerate(zip(colors, labels)):
            mask = plot_comp['class_idx'] == ci
            if not mask.any():
                continue
            plot_comp[mask].plot(ax=ax, color=color, edgecolor='none',
                                 linewidth=0, zorder=1)

        # Municipal boundaries (clipped to component)
        muni_comp  = gpd.clip(muni, component)
        muni_lines = muni_comp.boundary.explode(index_parts=False)
        muni_lines = muni_lines[
            ~muni_lines.geom_type.isin(['Point', 'MultiPoint'])]
        muni_lines.plot(ax=ax, color='#404040', linewidth=0.3, zorder=3)

        # Lakes above municipal lines, below catchment boundary
        if lakes is not None and not lakes.empty:
            lakes_clip = gpd.clip(lakes, component)
            if not lakes_clip.empty:
                lakes_clip.plot(ax=ax, color='#A8D8EA', edgecolor='none',
                                zorder=4)

        # Catchment boundary
        comp_gdf.boundary.plot(ax=ax, color='black', linewidth=1.8,
                               linestyle='--', zorder=5)

        legend_handles = []
        for color, lbl in zip(colors, labels):
            legend_handles.append(
                Patch(facecolor=color, edgecolor='none', label=lbl))
        legend_handles.append(
            Line2D([0], [0], color='#404040', linewidth=0.3,
                   label='Municipal boundary'))
        legend_handles.append(
            Line2D([0], [0], color='black', linewidth=1.8,
                   linestyle='--', label='Catchment area boundary'))

        leg = ax.legend(handles=legend_handles,
                        title=col_header,
                        title_fontsize=8,
                        fontsize=7,
                        loc='upper right',
                        framealpha=0.9,
                        ncol=1)
        leg._legend_box.align = 'left'

        bx_min, by_min, bx_max, by_max = component.bounds
        pad = 200
        ax.set_xlim(bx_min - pad, bx_max + pad)
        ax.set_ylim(by_min - pad, by_max + pad)

        ax.set_title(title, fontsize=13)
        ax.set_xlabel('E [m]')
        ax.set_ylabel('N [m]')
        ax.set_aspect('equal')

        out_path = os.path.join(CATCHMENT_PLOT_DIR,
                                f'plot_{label}_raster{part_suffix}.pdf')
        fig.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
    print(f"    Saved -> {out_path}")


# ===============================================================================
# STEP 2-MUN: MUNICIPAL METHOD
# ===============================================================================

def _assign_stations_to_municipalities(muni_gdf, rail_stations, bfs_col, name_col):
    """Assign each municipality to one rail station based on spatial proximity.

    Logic per municipality:
      - 0 stations inside polygon -> nearest station (Euclidean from centroid)
      - 1 station inside polygon  -> auto-assign
      - 2+ stations inside polygon -> interactive user prompt

    Returns
    -------
    pd.DataFrame
        Columns: [BFS_NR, NAME, station_id, station_name]
    """
    assignments = []

    for _, row in muni_gdf.iterrows():
        muni_name = row[name_col]
        bfs_nr = row[bfs_col]
        centroid = row.geometry.centroid

        # Find stations whose point geometry falls within this municipality
        mask = rail_stations.geometry.within(row.geometry)
        stations_within = rail_stations[mask]

        if len(stations_within) == 0:
            # Nearest station by Euclidean distance from municipality centroid
            dists = rail_stations.geometry.distance(centroid)
            nearest_idx = dists.idxmin()
            station = rail_stations.loc[nearest_idx]
            sname = (str(station['stop_name'])
                     if pd.notna(station.get('stop_name'))
                     else f"Station {station['id_point']}")
            assignments.append({
                'BFS_NR': bfs_nr, 'NAME': muni_name,
                'station_id': station['id_point'], 'station_name': sname,
            })
            print(f"    {muni_name}: no station inside "
                  f"-> nearest = {sname} ({dists[nearest_idx]:.0f}m)")

        elif len(stations_within) == 1:
            station = stations_within.iloc[0]
            sname = (str(station['stop_name'])
                     if pd.notna(station.get('stop_name'))
                     else f"Station {station['id_point']}")
            assignments.append({
                'BFS_NR': bfs_nr, 'NAME': muni_name,
                'station_id': station['id_point'], 'station_name': sname,
            })
            print(f"    {muni_name}: 1 station -> {sname}")

        else:
            # Multiple stations - interactive prompt
            sorted_st = stations_within.copy()
            sorted_st['_dist'] = sorted_st.geometry.distance(centroid)
            sorted_st = sorted_st.sort_values('_dist')

            print(f"\n  Municipality '{muni_name}' has "
                  f"{len(sorted_st)} stations:")
            for i, (_, st) in enumerate(sorted_st.iterrows()):
                sn = (str(st['stop_name'])
                      if pd.notna(st.get('stop_name'))
                      else f"Station {st['id_point']}")
                print(f"    [{i + 1}] {sn} "
                      f"({st['_dist']:.0f}m from centroid)")

            while True:
                choice = input(f"  Select station for '{muni_name}' "
                               f"[1-{len(sorted_st)}]: ")
                try:
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(sorted_st):
                        break
                except ValueError:
                    pass
                print(f"  Invalid choice. Please enter a number "
                      f"between 1 and {len(sorted_st)}.")

            station = sorted_st.iloc[choice_idx]
            sname = (str(station['stop_name'])
                     if pd.notna(station.get('stop_name'))
                     else f"Station {station['id_point']}")
            assignments.append({
                'BFS_NR': bfs_nr, 'NAME': muni_name,
                'station_id': station['id_point'], 'station_name': sname,
            })
            print(f"    {muni_name}: user selected -> {sname}")

    return pd.DataFrame(assignments)


def _run_municipal_method(boundary, rail_stations):
    """Automatic municipality-to-station assignment based on spatial proximity.

    For each municipality in the study area:
      - 0 stations inside -> assign nearest station (Euclidean from centroid)
      - 1 station inside  -> auto-assign
      - 2+ stations inside -> interactive prompt

    Builds catchment geometry and exports station summary CSV.

    Returns
    -------
    gpd.GeoDataFrame or None
        Municipal catchment polygons (for diff plot), or None on failure.
    """
    print("\n--- Municipal Method ---")

    # Load municipalities — only those whose centroid falls within the
    # study-area boundary (drops peripheral municipalities that merely
    # touch the edge and would otherwise be assigned to an internal station)
    muni = gpd.read_file(paths.MUNICIPAL_BOUNDARIES_GPKG)
    muni = muni.to_crs(CODEBASE_CRS)
    if 'objektart' in muni.columns:
        muni = muni[muni['objektart'] == 'Gemeindegebiet']
    muni = muni[muni.geometry.centroid.within(boundary)].copy()
    print(f"    {len(muni)} municipalities in study area (centroid within boundary)")

    # Identify BFS column
    bfs_col = None
    for candidate in ['BFS_NR', 'bfs_nr', 'BFS_NUMMER', 'bfs_nummer',
                       'GMDNR', 'gmdnr']:
        if candidate in muni.columns:
            bfs_col = candidate
            break
    if bfs_col is None:
        for c in muni.columns:
            if muni[c].dtype in ['int64', 'int32', 'float64'] \
                    and c != 'geometry':
                bfs_col = c
                break
    if bfs_col is None:
        print("    WARNING: Cannot identify BFS column "
              "- skipping municipal method")
        return None

    # Identify name column
    name_col = None
    for candidate in ['NAME', 'name', 'GMDNAME', 'gmdname',
                       'GEMEINDENAME']:
        if candidate in muni.columns:
            name_col = candidate
            break
    if name_col is None:
        name_col = bfs_col

    # Clip rail stations to the study-area boundary (same filter applied to muni)
    rail_stations = rail_stations[
        rail_stations.geometry.within(boundary)].copy()
    print(f"    {len(rail_stations)} rail stations within study area boundary")

    # Assign stations to municipalities (load cached assignment if available)
    assignment_csv = os.path.join(MUNICIPAL_DATA_DIR, 'station_assignment.csv')
    assignment_df = None
    if os.path.exists(assignment_csv):
        print(f"  Existing station assignment found: {assignment_csv}")
        while True:
            choice = input("  Use existing assignment? [1] Yes  [2] Reassign: ").strip()
            if choice == '1':
                assignment_df = pd.read_csv(assignment_csv)
                assignment_df['BFS_NR'] = pd.to_numeric(
                    assignment_df['BFS_NR'], errors='coerce')
                print(f"    Loaded {len(assignment_df)} assignments from cache")
                break
            elif choice == '2':
                break
            else:
                print("  Please enter 1 or 2.")
    if assignment_df is None:
        print("  Assigning stations to municipalities ...")
        assignment_df = _assign_stations_to_municipalities(
            muni, rail_stations, bfs_col, name_col)
        assignment_df.to_csv(assignment_csv, index=False)
        print(f"    Assignment saved -> {assignment_csv}")
    n_stations = assignment_df['station_id'].nunique()
    print(f"    {len(assignment_df)} municipalities assigned to "
          f"{n_stations} stations")

    # Build catchment geometry
    print("  Building municipal catchment polygons ...")
    muni_catchment = _build_municipal_catchment_geometry(
        assignment_df, muni, bfs_col)

    # Export station catchment summary CSV
    _export_station_catchment_csv(assignment_df, rail_stations)

    return muni_catchment, assignment_df, muni, bfs_col


def _build_municipal_catchment_geometry(assignment_df, muni_gdf, bfs_col):
    """Dissolve municipality polygons by their assigned station to create
    catchment areas for the municipal method.

    Returns
    -------
    gpd.GeoDataFrame
        Columns: [id_point, geometry] dissolved per station.
    """
    muni_with_station = muni_gdf[[bfs_col, 'geometry']].copy()
    muni_with_station[bfs_col] = pd.to_numeric(
        muni_with_station[bfs_col], errors='coerce')

    lookup = assignment_df[['BFS_NR', 'station_id']].copy()
    lookup['BFS_NR'] = pd.to_numeric(lookup['BFS_NR'], errors='coerce')
    lookup = lookup.rename(columns={'BFS_NR': bfs_col})

    merged = muni_with_station.merge(lookup, on=bfs_col, how='left')
    merged = merged.dropna(subset=['station_id'])

    # Dissolve by station
    dissolved = merged.dissolve(by='station_id', as_index=False)
    dissolved = dissolved[['station_id', 'geometry']].copy()
    dissolved = dissolved.rename(columns={'station_id': 'id_point'})

    # Save GeoPackage (downstream-compatible format)
    out_path = os.path.join(MUNICIPAL_DATA_DIR, 'catchement.gpkg')
    save_df = dissolved.copy()
    save_df['train_station'] = save_df['id_point']
    save_df = save_df.rename(columns={'id_point': 'id'})
    save_df[['train_station', 'id', 'geometry']].to_file(
        out_path, driver='GPKG')
    print(f"    Municipal catchment saved -> {out_path}  "
          f"({len(dissolved)} stations)")

    return dissolved


def _export_station_catchment_csv(assignment_df, rail_stations):
    """Export station catchment summary CSV.

    Columns: Station_Code, Station_Name, Pop, Empl, Municipalities.
    Includes all stations; those with no assignment get 0 / 0 / '--'.
    """
    print("  Exporting station catchment summary ...")

    # Read the per-municipality pop/empl summary produced by Step 1
    summary_path = os.path.join(
        CATCHMENT_DATA_DIR, 'municipal_pop_empl_summary.csv')
    if os.path.exists(summary_path):
        muni_summary = pd.read_csv(summary_path)
        muni_summary['BFS_NR'] = pd.to_numeric(
            muni_summary['BFS_NR'], errors='coerce')
    else:
        print("    WARNING: municipal_pop_empl_summary.csv not found "
              "- pop/empl will be zero")
        muni_summary = pd.DataFrame(
            columns=['BFS_NR', 'NAME',
                     'total_population', 'total_employment'])

    # Merge assignment with pop/empl
    assign = assignment_df.copy()
    assign['BFS_NR'] = pd.to_numeric(assign['BFS_NR'], errors='coerce')
    merged = assign.merge(
        muni_summary[['BFS_NR', 'total_population', 'total_employment']],
        on='BFS_NR', how='left',
    ).fillna(0)

    # Coerce numeric types to guard against non-numeric values from CSV load
    merged['total_population'] = pd.to_numeric(
        merged['total_population'], errors='coerce').fillna(0)
    merged['total_employment'] = pd.to_numeric(
        merged['total_employment'], errors='coerce').fillna(0)

    # Aggregate by station
    station_agg = merged.groupby(
        ['station_id', 'station_name'], sort=False,
    ).agg(
        Pop=('total_population', 'sum'),
        Empl=('total_employment', 'sum'),
        Municipalities=('NAME',
                        lambda x: ', '.join(sorted(x.astype(str)))),
    ).reset_index()
    station_agg = station_agg.rename(columns={
        'station_id': 'Station_Code',
        'station_name': 'Station_Name',
    })
    station_agg['Pop'] = station_agg['Pop'].astype(int)
    station_agg['Empl'] = station_agg['Empl'].astype(int)

    # Add unassigned stations
    assigned_ids = set(assignment_df['station_id'].values)
    unassigned = rail_stations[
        ~rail_stations['id_point'].isin(assigned_ids)]
    if len(unassigned) > 0:
        names = unassigned['stop_name'].apply(
            lambda x: str(x) if pd.notna(x) else '—')
        unassigned_rows = pd.DataFrame({
            'Station_Code': unassigned['id_point'].values,
            'Station_Name': names.values,
            'Pop': 0,
            'Empl': 0,
            'Municipalities': '—',
        })
        station_agg = pd.concat(
            [station_agg, unassigned_rows], ignore_index=True)

    station_agg = station_agg.sort_values(
        'Station_Name').reset_index(drop=True)

    out_path = os.path.join(
        MUNICIPAL_DATA_DIR, 'station_catchment_summary.csv')
    station_agg.to_csv(out_path, index=False, encoding='utf-8-sig')
    n_with = (station_agg['Municipalities'] != '—').sum()
    print(f"    Saved -> {out_path}  ({len(station_agg)} stations, "
          f"{n_with} with catchment)")


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
                               rail_stations, boundary,
                               assignment_df=None, muni_gdf=None,
                               bfs_col=None):
    """Plot municipal catchment areas with graph-colouring.  Each coloured
    area is a dissolved municipal catchment (one per assigned station).
    Only cells with population or employment are filled."""
    if muni_catchment is None or muni_catchment.empty:
        print("  Skipping municipal catchment plot - no geometry available")
        return
    print("  Building municipal catchment plot ...")

    # Clip dissolved catchment geometries to the study area boundary
    plot_catchments = gpd.clip(muni_catchment, boundary).reset_index(drop=True)

    # --- Adjacency graph of catchment areas ----------------------------------
    G = nx.Graph()
    for i in range(len(plot_catchments)):
        G.add_node(plot_catchments.loc[i, 'id_point'])

    for i in range(len(plot_catchments)):
        for j in range(i + 1, len(plot_catchments)):
            geom_i = plot_catchments.loc[i, 'geometry']
            geom_j = plot_catchments.loc[j, 'geometry']
            if geom_i.intersects(geom_j):
                inter = geom_i.intersection(geom_j)
                if hasattr(inter, 'length') and inter.length > 0:
                    G.add_edge(plot_catchments.loc[i, 'id_point'],
                               plot_catchments.loc[j, 'id_point'])

    # --- Graph colouring -----------------------------------------------------
    coloring = nx.coloring.greedy_color(G, strategy='largest_first')
    n_colors = max(coloring.values()) + 1 if coloring else 1

    base_palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#bcbd22', '#17becf', '#aec7e8',
        '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94',
        '#f7b6d2', '#dbdb8d', '#9edae5', '#393b79', '#637939',
    ]
    palette = base_palette[:max(n_colors, 1)]
    color_map = {sid: base_palette[cidx % len(base_palette)]
                 for sid, cidx in coloring.items()}
    max_deg = max(dict(G.degree()).values()) if G.degree() else 0
    print(f"    Graph colouring: {n_colors} colours for "
          f"{len(plot_catchments)} catchment areas "
          f"(max adjacency {max_deg})")

    # --- Build cell geometries -----------------------------------------------
    pop_relis = set(pop_grid['RELI'].values)
    empl_relis = set(empl_grid['RELI'].values)
    all_relis = pop_relis | empl_relis
    combined = pd.concat([
        pop_grid[['RELI', 'E_KOORD', 'N_KOORD', 'geometry']],
        empl_grid[['RELI', 'E_KOORD', 'N_KOORD', 'geometry']],
    ]).drop_duplicates(subset='RELI')
    combined = combined[combined['RELI'].isin(all_relis)].copy()
    combined = gpd.GeoDataFrame(combined, geometry='geometry',
                                crs=CODEBASE_CRS)

    # Spatial join cells -> dissolved catchment polygons
    joined = gpd.sjoin(
        combined, plot_catchments[['id_point', 'geometry']],
        how='left', predicate='within')
    joined['color'] = joined['id_point'].map(color_map).fillna('#D3D3D3')

    # Build square cell polygons
    cell_geoms = [
        Polygon([
            (e, n), (e + CELL_SIZE_M, n),
            (e + CELL_SIZE_M, n + CELL_SIZE_M), (e, n + CELL_SIZE_M),
        ])
        for e, n in zip(joined['E_KOORD'], joined['N_KOORD'])
    ]
    cells_gdf = gpd.GeoDataFrame(
        joined, geometry=cell_geoms, crs=CODEBASE_CRS)

    # --- Plot ----------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))

    # Study area background
    boundary_gdf = gpd.GeoDataFrame(geometry=[boundary], crs=CODEBASE_CRS)
    boundary_gdf.plot(ax=ax, color='#F0F0F0', edgecolor='none')

    # Coloured cells
    for color, group in cells_gdf.groupby('color'):
        group.plot(ax=ax, color=color, edgecolor='none', zorder=2)

    # Catchment area borders
    plot_catchments.boundary.plot(ax=ax, color='#2C3E50', linewidth=0.8,
                                 zorder=4)

    # Municipal borders — thin, light, dashed so individual municipalities
    # within dissolved catchments remain traceable
    muni_plot = gpd.read_file(paths.MUNICIPAL_BOUNDARIES_GPKG).to_crs(CODEBASE_CRS)
    if 'objektart' in muni_plot.columns:
        muni_plot = muni_plot[muni_plot['objektart'] == 'Gemeindegebiet']
    muni_plot = muni_plot[muni_plot.geometry.intersects(boundary)].copy()
    muni_plot = gpd.clip(muni_plot, boundary)
    muni_lines = muni_plot.boundary.explode(index_parts=False)
    muni_lines = muni_lines[~muni_lines.geom_type.isin(['Point', 'MultiPoint'])]
    muni_lines.plot(ax=ax, color='#B0B0B0', linewidth=0.25,
                    linestyle='--', zorder=3)

    # Connecting lines: drawn when a municipality is not contiguous with the
    # primary component of its dissolved catchment (islands) or when the
    # assigned station falls outside the dissolved catchment entirely (cutoff).

    # Pre-compute the primary component for each station:
    # the sub-polygon of the dissolved catchment that contains the station point.
    station_primary = {}  # station_id -> primary component geometry or None
    station_pt_lookup = dict(zip(rail_stations['id_point'], rail_stations.geometry))
    for _, crow in muni_catchment.iterrows():
        sid = crow['id_point']
        diss_geom = crow['geometry']
        st_pt = station_pt_lookup.get(sid)
        if st_pt is None or diss_geom is None:
            station_primary[sid] = None
            continue
        components = (list(diss_geom.geoms)
                      if diss_geom.geom_type == 'MultiPolygon'
                      else [diss_geom])
        station_primary[sid] = next(
            (c for c in components if c.contains(st_pt)), None)

    has_connect_lines = False
    if assignment_df is not None and muni_gdf is not None and bfs_col is not None:
        # Build lookup dict once to avoid per-row GDF boolean-index filtering
        _bfs_num = pd.to_numeric(muni_gdf[bfs_col], errors='coerce')
        muni_geom_lookup = dict(zip(_bfs_num, muni_gdf.geometry))

        line_geoms = []
        for _, arow in assignment_df.iterrows():
            sid = arow['station_id']
            bfs_numeric = pd.to_numeric(arow['BFS_NR'], errors='coerce')
            muni_geom = muni_geom_lookup.get(bfs_numeric)
            st_pt = station_pt_lookup.get(sid)
            if muni_geom is None or st_pt is None:
                continue
            primary = station_primary.get(sid)
            needs_line = (primary is None) or (not muni_geom.intersects(primary))
            if needs_line:
                line_geoms.append(LineString([
                    (st_pt.x, st_pt.y),
                    (muni_geom.centroid.x, muni_geom.centroid.y),
                ]))

        if line_geoms:
            gpd.GeoDataFrame(geometry=line_geoms, crs=CODEBASE_CRS).plot(
                ax=ax, color='black', linewidth=0.8, linestyle='-', zorder=5)
            has_connect_lines = True

    # Study area boundary
    boundary_gdf.boundary.plot(ax=ax, color='black', linewidth=1.8,
                               linestyle='--', zorder=5)

    # Clip view to study area bounds
    bx_min, by_min, bx_max, by_max = boundary.bounds
    pad = 200
    ax.set_xlim(bx_min - pad, bx_max + pad)
    ax.set_ylim(by_min - pad, by_max + pad)

    # Lakes — above catchment fills, below stations; clipped to boundary
    if os.path.exists(paths.LAKES_SHP):
        lakes = gpd.read_file(paths.LAKES_SHP).to_crs(CODEBASE_CRS)
        lakes = lakes[lakes.geometry.intersects(boundary)].copy()
        if not lakes.empty:
            gpd.clip(lakes, boundary).plot(
                ax=ax, color='#A8D4F0', edgecolor='none', zorder=6)

    # Station markers (circles) — only stations within boundary
    prep_bnd = prep(boundary)
    stations_in_boundary = rail_stations[
        rail_stations.geometry.apply(lambda p: prep_bnd.contains(p))]
    assigned_ids = set(plot_catchments['id_point'].astype(str).values)
    has_catchment = stations_in_boundary[
        stations_in_boundary['id_point'].astype(str).isin(assigned_ids)]
    no_catchment = stations_in_boundary[
        ~stations_in_boundary['id_point'].astype(str).isin(assigned_ids)]

    if len(has_catchment) > 0:
        ax.scatter(has_catchment.geometry.x, has_catchment.geometry.y,
                   s=25, c='white', edgecolors='black', linewidths=0.8,
                   marker='o', zorder=7)
    if len(no_catchment) > 0:
        ax.scatter(no_catchment.geometry.x, no_catchment.geometry.y,
                   s=25, c='red', edgecolors='black', linewidths=0.8,
                   marker='o', zorder=7)

    # Legend
    legend_handles = [
        Patch(facecolor=palette[0], edgecolor='none',
              label='Catchment area (coloured)'),
        Line2D([0], [0], color='#2C3E50', linewidth=0.8,
               label='Catchment boundary'),
        Line2D([0], [0], color='#B0B0B0', linewidth=0.25, linestyle='--',
               label='Municipal boundary'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='black', markeredgewidth=0.8,
               markersize=8, label='Station (with catchment)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markeredgecolor='black', markeredgewidth=0.8,
               markersize=8, label='Station (no catchment)'),
        Line2D([0], [0], color='black', linewidth=1.8, linestyle='--',
               label='Study area boundary'),
    ]
    if has_connect_lines:
        legend_handles.insert(-1, Line2D(
            [0], [0], color='black', linewidth=0.8,
            label='External station assignment'))
    ax.legend(handles=legend_handles, loc='upper right', fontsize=8,
              framealpha=0.9)
    ax.set_title('Municipal Catchment Areas', fontsize=14)
    ax.set_xlabel('E [m]')
    ax.set_ylabel('N [m]')
    ax.set_aspect('equal')

    out_path = os.path.join(MUNICIPAL_PLOT_DIR,
                            'catchment_municipal_areas.pdf')
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

    # Clip catchment geometries to the study area boundary
    clipped_catchment = gpd.clip(muni_catchment, boundary)

    # Catchment area outlines
    clipped_catchment.boundary.plot(ax=ax, color='#2C3E50', linewidth=1.0, zorder=2)

    # PT-feeder network lines coloured by mode
    mode_colours = {
        'bus':          '#0000FF',
        'express_bus':  '#0000FF',
        'on_demand_bus':'#0000FF',
        'tram':         '#FF66CC',
        'metro':        '#00246B',
        'ship':         '#004B8D',   # darker blue to distinguish from lake fill
        'funicular':    '#000000',
    }
    plotted_modes = []
    if not feeder_lines.empty:
        for mode_name, colour in mode_colours.items():
            subset = feeder_lines[
                feeder_lines['mode'].fillna('').str.lower() == mode_name]
            if len(subset) > 0:
                subset.plot(ax=ax, color=colour, linewidth=0.6, alpha=0.7, zorder=3)
                plotted_modes.append((mode_name, colour))

    # Lakes — above feeder network, below stations; clipped to boundary
    if os.path.exists(paths.LAKES_SHP):
        lakes = gpd.read_file(paths.LAKES_SHP).to_crs(CODEBASE_CRS)
        lakes = lakes[lakes.geometry.intersects(boundary)].copy()
        if not lakes.empty:
            gpd.clip(lakes, boundary).plot(
                ax=ax, color='#A8D4F0', edgecolor='none', zorder=4)

    # Station markers (circles) — only stations within boundary
    prep_bnd_net = prep(boundary)
    stations_in_bnd = rail_stations[
        rail_stations.geometry.apply(lambda p: prep_bnd_net.contains(p))]
    assigned_ids = set(clipped_catchment['id_point'].astype(str).values)
    has_catchment = stations_in_bnd[stations_in_bnd['id_point'].astype(str).isin(assigned_ids)]
    no_catchment = stations_in_bnd[~stations_in_bnd['id_point'].astype(str).isin(assigned_ids)]

    if len(has_catchment) > 0:
        ax.scatter(has_catchment.geometry.x, has_catchment.geometry.y,
                   s=25, c='white', edgecolors='black', linewidths=0.8,
                   marker='o', zorder=6)
    if len(no_catchment) > 0:
        ax.scatter(no_catchment.geometry.x, no_catchment.geometry.y,
                   s=25, c='red', edgecolors='black', linewidths=0.8,
                   marker='o', zorder=6)

    # Study area boundary
    boundary_gdf.boundary.plot(ax=ax, color='black', linewidth=1.8, linestyle='--', zorder=5)

    # Clip view to study area bounds
    bx_min, by_min, bx_max, by_max = boundary.bounds
    pad = 200
    ax.set_xlim(bx_min - pad, bx_max + pad)
    ax.set_ylim(by_min - pad, by_max + pad)

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
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='black', markeredgewidth=0.8,
               markersize=8, label='Station (with catchment)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markeredgecolor='black', markeredgewidth=0.8,
               markersize=8, label='Station (no catchment)'),
        Line2D([0], [0], color='black', linewidth=1.8, linestyle='--',
               label='Study area boundary'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=8, framealpha=0.9)
    ax.set_title('Municipal Catchment Areas with PT-Feeder Network', fontsize=14)
    ax.set_xlabel('E [m]')
    ax.set_ylabel('N [m]')
    ax.set_aspect('equal')

    out_path = os.path.join(MUNICIPAL_PLOT_DIR, 'catchment_municipal_areas_network.pdf')
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


def _load_rail_stations(boundary, temporal='all', buffer=BUFFER_RAIL_M):
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

    expanded = boundary.buffer(buffer) if buffer > 0 else boundary
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

    out_path = os.path.join(PT_FEEDER_PLOT_DIR,
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

    # Clip catchment boundaries to study area
    muni_c = gpd.clip(muni_c, boundary)
    pt_c = gpd.clip(pt_c, boundary)

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

    # Rail stations — only within boundary
    from shapely.prepared import prep as _prep_diff
    prep_bnd_diff = _prep_diff(boundary)
    stations_in_bnd = rail_stations[
        rail_stations.geometry.apply(lambda p: prep_bnd_diff.contains(p))]
    stations_in_bnd.plot(ax=ax, color='black', markersize=20, marker='^', zorder=5)
    legend_handles.append(
        Line2D([0], [0], marker='^', color='w', markerfacecolor='black',
               markersize=8, label='Rail station'))

    # Study area boundary
    boundary_gdf.boundary.plot(ax=ax, color='black', linewidth=1.8, linestyle='--', zorder=4)
    legend_handles.append(
        Line2D([0], [0], color='black', linewidth=1.8, linestyle='--',
               label='Study area boundary'))

    # Clip view to study area bounds
    bx_min, by_min, bx_max, by_max = boundary.bounds
    pad = 200
    ax.set_xlim(bx_min - pad, bx_max + pad)
    ax.set_ylim(by_min - pad, by_max + pad)

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

        # Clip view to study area bounds
        bx_min, by_min, bx_max, by_max = boundary.bounds
        pad = 200
        ax.set_xlim(bx_min - pad, bx_max + pad)
        ax.set_ylim(by_min - pad, by_max + pad)

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

        out_path = os.path.join(PT_FEEDER_PLOT_DIR, fname)
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
        rail_stations = _load_rail_stations(boundary, temporal, buffer=0)
        result = _run_municipal_method(boundary, rail_stations)
        if result is not None:
            muni_catchment, assignment_df, muni_gdf, bfs_col = result
        else:
            muni_catchment = None

        # Municipal catchment visualisation plots
        if muni_catchment is not None:
            _plot_municipal_catchments(muni_catchment, pop_grid, empl_grid,
                                       rail_stations, boundary,
                                       assignment_df, muni_gdf, bfs_col)
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
