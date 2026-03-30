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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib_scalebar.scalebar import ScaleBar
from scipy.spatial import cKDTree
import fiona
import re
import pyogrio
from shapely.geometry import LineString, Point, Polygon, box
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

# Access mode codes (used in allocation logic and visualisation)
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
GUETEKLASSEN_PLOT_DIR = os.path.join(CATCHMENT_PLOT_DIR, 'Gueteklassen')

# ÖV-Güteklassen classification constants (ARE 2022)
# Operational window for headway calculation (ARE standard: 06:00–20:00 = 840 min)
GK_WINDOW_MIN  = 840

# Maximum walk-access buffer radius (m) per Haltestellenkategorie (1=I … 5=V)
GK_MAX_RADIUS  = {1: 1000, 2: 1000, 3: 750, 4: 500, 5: 300}

# Distance bands for Güteklasse ring assignment
GK_DIST_BANDS  = [(0, 300), (300, 500), (500, 750), (750, 1000)]
GK_DIST_LABELS = ['0-300', '300-500', '500-750', '750-1000']

# Güteklasse lookup: (Kat_int, dist_band_0based) → 'A'/'B'/'C'/'D'
# Kat: 1=I, 2=II, 3=III, 4=IV, 5=V  |  band: 0=(0-300m), 1=(300-500m), …
_GK_LOOKUP = {
    (1, 0): 'A', (1, 1): 'A', (1, 2): 'B', (1, 3): 'C',
    (2, 0): 'A', (2, 1): 'B', (2, 2): 'C', (2, 3): 'D',
    (3, 0): 'B', (3, 1): 'C', (3, 2): 'D',
    (4, 0): 'C', (4, 1): 'D',
    (5, 0): 'D',
}

GK_PLOT_COLOURS = {'A': '#2D6A2D', 'B': '#1565C0', 'C': '#E65100', 'D': '#B71C1C'}

_GUETEKLASSEN_ARE_GPKG = os.path.join(
    'data', 'Spatial_Data', 'Transit_Network',
    'Gueteklassen_oev_2026_2056.gpkg', 'OeV_Gueteklassen_ARE.gpkg')


def _ensure_dirs():
    """Create all output directories."""
    for d in [CATCHMENT_DATA_DIR, CATCHMENT_PLOT_DIR,
              MUNICIPAL_DATA_DIR, PT_FEEDER_DATA_DIR,
              MUNICIPAL_PLOT_DIR, PT_FEEDER_PLOT_DIR]:
        os.makedirs(d, exist_ok=True)


# ===============================================================================
# MAP CARTOGRAPHIC ELEMENTS (North arrow & scale bar)
# ===============================================================================

def _add_north_arrow(ax, x=0.03, y=0.97, arrow_length=0.047, fontsize=10):
    """Add a compass-style north arrow to a matplotlib axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to add the arrow to.
    x, y : float
        Position in axes coordinates (0-1). Default is top-left.
    arrow_length : float
        Length of the arrow in axes fraction (reduced by 1/3 from 0.07).
    fontsize : int
        Font size for the 'N' label.
    """
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.transforms import Bbox, TransformedBbox, BboxTransform

    # Arrow dimensions in axes fraction
    half_width = arrow_length * 0.35
    tip_y = y
    base_y = y - arrow_length
    label_y = base_y - 0.015

    # Define the two triangular halves of the compass needle
    # Left half (dark/filled)
    left_triangle = [(x, tip_y), (x - half_width, base_y), (x, base_y)]
    # Right half (light/white)
    right_triangle = [(x, tip_y), (x + half_width, base_y), (x, base_y)]

    # Draw left half (dark grey fill)
    left_patch = MplPolygon(
        left_triangle, closed=True,
        facecolor='#4a4a4a', edgecolor='black', linewidth=0.8,
        transform=ax.transAxes, zorder=1000
    )
    ax.add_patch(left_patch)

    # Draw right half (white fill)
    right_patch = MplPolygon(
        right_triangle, closed=True,
        facecolor='white', edgecolor='black', linewidth=0.8,
        transform=ax.transAxes, zorder=1000
    )
    ax.add_patch(right_patch)

    # Add 'N' label below the arrow
    ax.text(
        x, label_y, 'N',
        transform=ax.transAxes,
        ha='center', va='top', fontsize=fontsize, fontweight='bold',
        zorder=1000
    )


def _add_scale_bar(ax):
    """Add a custom scale bar with two boxes (0–5 km and 5–10 km), always lower right."""
    from matplotlib.patches import Rectangle

    xlim = ax.get_xlim()

    # Scale bar dimensions in metres — two bins only
    bins_km = [5, 10]  # cumulative distances
    bins_m = [b * 1000 for b in bins_km]

    # Always lower right
    y_offset = 0.04
    x_offset = 0.75

    # Convert metres to axes fraction
    data_width = xlim[1] - xlim[0]
    total_width_frac = bins_m[-1] / data_width

    # Draw boxes
    box_height_frac = 0.012
    prev_frac = 0
    colours = ['black', 'white']  # alternating

    for i, dist_m in enumerate(bins_m):
        width_frac = (dist_m / data_width) - prev_frac
        rect = Rectangle(
            (x_offset + prev_frac, y_offset),
            width_frac, box_height_frac,
            facecolor=colours[i], edgecolor='black', linewidth=0.8,
            transform=ax.transAxes, zorder=999
        )
        ax.add_patch(rect)
        prev_frac = dist_m / data_width

    # Add tick labels below boxes
    label_y = y_offset - 0.018
    ax.text(x_offset, label_y, '0', transform=ax.transAxes,
            ha='center', va='top', fontsize=7, zorder=1000)
    for dist_m, dist_km in zip(bins_m, bins_km):
        x_pos = x_offset + dist_m / data_width
        ax.text(x_pos, label_y, f'{dist_km}', transform=ax.transAxes,
                ha='center', va='top', fontsize=7, zorder=1000)

    # Add 'km' unit label
    ax.text(x_offset + total_width_frac / 2, y_offset + box_height_frac + 0.008, 'km',
            transform=ax.transAxes, ha='center', va='bottom', fontsize=7, zorder=1000)


def _add_map_elements(ax):
    """Add scale bar (lower right, 0–5–10 km) and north arrow (upper left) to an axes."""
    _add_scale_bar(ax)
    _add_north_arrow(ax)


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
            if box(row['E_KOORD'], row['N_KOORD'],
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
    summary[csv_cols].to_csv(csv_path, index=False, encoding='utf-8-sig')
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

            # Add cartographic elements
            _add_map_elements(ax)

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

    e_arr = gdf['E_KOORD'].values
    n_arr = gdf['N_KOORD'].values
    squares = [box(e, n, e + CELL_SIZE_M, n + CELL_SIZE_M)
               for e, n in zip(e_arr, n_arr)]
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

        # Add cartographic elements
        _add_map_elements(ax)

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
                # station_id must stay as str to match rail_stations['id_point']
                assignment_df['station_id'] = (
                    assignment_df['station_id'].astype(str))
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
        assignment_df.to_csv(assignment_csv, index=False, encoding='utf-8-sig')
        print(f"    Assignment saved -> {assignment_csv}")
    n_stations = assignment_df['station_id'].nunique()
    print(f"    {len(assignment_df)} municipalities assigned to "
          f"{n_stations} stations")

    # Build catchment geometry (geometry only)
    print("  Building municipal catchment polygons ...")
    muni_catchment = _build_municipal_catchment_geometry(
        assignment_df, muni, bfs_col)

    # Export station catchment summary CSV; get per-station stats for GPKG enrichment
    station_agg = _export_station_catchment_csv(assignment_df, rail_stations)

    # Write enriched catchment GPKG: geometry + station stats
    out_path = os.path.join(MUNICIPAL_DATA_DIR, 'catchment.gpkg')
    save_df = muni_catchment.rename(columns={'id_point': 'id'}).copy()
    save_df['train_station'] = save_df['id']
    stats_idx = station_agg.set_index('Station_Code')
    id_str = save_df['id'].astype(str)
    save_df['station_name']   = id_str.map(stats_idx['Station_Name']).fillna('—')
    save_df['pop']            = id_str.map(stats_idx['Pop']).fillna(0).astype(int)
    save_df['empl']           = id_str.map(stats_idx['Empl']).fillna(0).astype(int)
    save_df['municipalities'] = id_str.map(stats_idx['Municipalities']).fillna('—')
    save_df[['train_station', 'id', 'station_name', 'pop', 'empl',
             'municipalities', 'geometry']].to_file(out_path, driver='GPKG')
    print(f"    Municipal catchment GPKG saved -> {out_path}  ({len(save_df)} stations)")

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

    # Add unassigned stations (cast both sides to str to avoid int/str mismatches)
    assigned_ids = set(str(x) for x in assignment_df['station_id'].values)
    unassigned = rail_stations[
        ~rail_stations['id_point'].astype(str).isin(assigned_ids)]
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
    return station_agg


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

    # Nearest-catchment fallback for edge cells not strictly within any polygon
    unassigned_mask = joined['id_point'].isna()
    if unassigned_mask.any():
        unassigned = combined.loc[joined.index[unassigned_mask]]
        catchment_reset = catchment.reset_index(drop=True)
        for idx, row in unassigned.iterrows():
            dists = catchment_reset.geometry.boundary.distance(row.geometry)
            nearest = catchment_reset.loc[dists.idxmin(), 'id_point']
            joined.loc[idx, 'id_point'] = nearest

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
            (c for c in components if c.covers(st_pt)), None)

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

        print(f"    Connecting lines: {len(line_geoms)} detached municipality "
              f"assignments found")
        if line_geoms:
            gpd.GeoDataFrame(geometry=line_geoms, crs=CODEBASE_CRS).plot(
                ax=ax, color='black', linewidth=0.4, linestyle='-', zorder=8)
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
            [0], [0], color='black', linewidth=0.4,
            label='External station assignment'))
    ax.legend(handles=legend_handles, loc='upper right', fontsize=8,
              framealpha=0.9)

    # Add cartographic elements
    _add_map_elements(ax)

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

    # Add cartographic elements
    _add_map_elements(ax)

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

    layers = fiona.listlayers(stops_path)
    frames = []
    for layer in layers:
        gdf = gpd.read_file(stops_path, layer=layer)
        gdf['mode'] = layer
        frames.append(gdf)
    all_stops = pd.concat(frames, ignore_index=True)
    all_stops = gpd.GeoDataFrame(all_stops, geometry='geometry', crs=CODEBASE_CRS)

    # buffer_radius_m is assigned later by _compute_stop_gueteklassen
    max_buffer = max(GK_MAX_RADIUS.values())  # 1000 m — max possible Güteklassen radius
    expanded = boundary.buffer(max_buffer)
    all_stops = all_stops[all_stops.geometry.within(expanded)].copy()

    keep_cols = ['stop_id', 'stop_name', 'mode', 'geometry']
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


def _build_pt_buffers(feeder_stops, rail_stations, boundary, pop_grid, empl_grid):
    """Generate buffer geometries for visualisation and validation.

    Produces three dissolved, boundary-clipped buffer polygons:
      - walk_to_rail  : Güteklassen walk radius around each rail station
      - pt_feeder     : walk-to-stop buffer for all bus + tram feeder stops
      - cycle_to_rail : fixed CYCLE_RADIUS_M cycling buffer around rail stations

    Attributes per feature: buffer_type, area_m2, population, employment.
    Saved as three separate named layers in buffers_visualisation.gpkg so that
    QGIS loads them with walk_to_rail on top (last written = top of panel).
    """
    print("  Building PT buffers for visualisation ...")
    records = []

    bus_mask  = ~feeder_stops['mode'].str.lower().str.contains('tram')
    tram_mask =  feeder_stops['mode'].str.lower().str.contains('tram')

    for _, mask, buf_type in [
        ('bus',  bus_mask,  'walk_to_feeder_bus'),
        ('tram', tram_mask, 'walk_to_feeder_tram'),
    ]:
        subset = feeder_stops[mask]
        if subset.empty:
            continue
        buf_geoms = [geom.buffer(float(r))
                     for geom, r in zip(subset.geometry, subset['buffer_radius_m'])]
        records.extend([
            {'stop_id': sid, 'buffer_type': buf_type, 'geometry': g}
            for sid, g in zip(subset['stop_id'], buf_geoms)
        ])

    # Rail: walk buffer (Güteklassen radius) + cycling buffer (fixed)
    rail_walk_geoms  = [g.buffer(float(r))
                        for g, r in zip(rail_stations.geometry,
                                        rail_stations['buffer_radius_m'])]
    rail_cycle_geoms = [g.buffer(CYCLE_RADIUS_M) for g in rail_stations.geometry]

    records.extend([
        {'stop_id': sid, 'buffer_type': 'walk_to_rail',  'geometry': g}
        for sid, g in zip(rail_stations['stop_id'], rail_walk_geoms)
    ])
    records.extend([
        {'stop_id': sid, 'buffer_type': 'cycle_to_rail', 'geometry': g}
        for sid, g in zip(rail_stations['stop_id'], rail_cycle_geoms)
    ])

    buffers_gdf = gpd.GeoDataFrame(records, geometry='geometry', crs=CODEBASE_CRS)

    # Dissolve into 3 categories, clip to boundary, compute stats
    pop_pts  = pop_grid[['geometry', 'NUMMER']].copy()
    empl_pts = empl_grid[['geometry', 'NUMMER']].copy()

    def _buffer_stats(geom):
        """Return (area_m2, population, employment) for a dissolved buffer geom."""
        clipped = geom.intersection(boundary)
        if clipped.is_empty:
            return 0.0, 0, 0
        buf_gdf = gpd.GeoDataFrame([{'geometry': clipped}],
                                   geometry='geometry', crs=CODEBASE_CRS)
        pop_in  = int(gpd.sjoin(pop_pts,  buf_gdf, how='inner',
                                predicate='within')['NUMMER'].sum())
        empl_in = int(gpd.sjoin(empl_pts, buf_gdf, how='inner',
                                predicate='within')['NUMMER'].sum())
        return round(clipped.area, 0), pop_in, empl_in

    # Layer order: cycle_to_rail first (bottom in QGIS), pt_feeder second,
    # walk_to_rail last (top in QGIS panel when loaded from file)
    layer_order = [
        (['walk_to_feeder_bus', 'walk_to_feeder_tram'], 'pt_feeder'),
        (['walk_to_rail'],                              'walk_to_rail'),
        (['cycle_to_rail'],                             'cycle_to_rail'),
    ]

    dissolved_by_label = {}
    for buf_types, label in layer_order:
        sub = buffers_gdf[buffers_gdf['buffer_type'].isin(buf_types)]
        if not sub.empty:
            merged_geom = unary_union(sub.geometry.values)
            area_m2, pop_in, empl_in = _buffer_stats(merged_geom)
            clipped_geom = merged_geom.intersection(boundary)
            dissolved_by_label[label] = {
                'buffer_type': label,
                'area_m2':     area_m2,
                'population':  pop_in,
                'employment':  empl_in,
                'geometry':    clipped_geom,
            }

    out_path = os.path.join(PT_FEEDER_DATA_DIR, 'buffers_visualisation.gpkg')
    if os.path.exists(out_path):
        os.remove(out_path)

    # Write in render order: cycle_to_rail → pt_feeder → walk_to_rail
    for label in ['cycle_to_rail', 'pt_feeder', 'walk_to_rail']:
        if label in dissolved_by_label:
            row = dissolved_by_label[label]
            layer_gdf = gpd.GeoDataFrame([row], geometry='geometry', crs=CODEBASE_CRS)
            layer_gdf.to_file(out_path, layer=label, driver='GPKG')
            print(f"    Layer '{label}': area={row['area_m2']:,.0f} m2, "
                  f"pop={row['population']:,}, empl={row['employment']:,}")

    print(f"    Saved -> {out_path}  ({len(dissolved_by_label)} layers)")

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
    """For each cell centroid, find rail stations within their Güteklassen walk
    buffer radius and compute walk time."""
    coords_grid = np.column_stack([grid.geometry.x, grid.geometry.y])
    coords_rail = np.column_stack([rail_stations.geometry.x, rail_stations.geometry.y])
    rail_radii  = rail_stations['buffer_radius_m'].values.astype(float)

    # Query with the largest radius present; filter per-station below
    max_radius = rail_radii.max() if len(rail_radii) > 0 else float(BUFFER_RAIL_M)

    tree = cKDTree(coords_rail)
    results = []
    neighbors = tree.query_ball_point(coords_grid, r=max_radius)

    reli_values   = grid['RELI'].values
    rail_stop_ids = rail_stations['stop_id'].values

    for i, near_idxs in enumerate(neighbors):
        for j in near_idxs:
            dist = np.sqrt((coords_grid[i, 0] - coords_rail[j, 0])**2 +
                           (coords_grid[i, 1] - coords_rail[j, 1])**2)
            if dist > rail_radii[j]:
                continue
            results.append({
                'RELI': reli_values[i],
                'rail_stop_id': str(rail_stop_ids[j]),
                'total_time_sec': dist / WALK_SPEED_MS,
                'access_mode': 'Walk'
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
                'access_mode': 'Cycle'
            })

    return pd.DataFrame(results)


def _compute_feeder_to_rail_times(grid, feeder_stops, feeder_stop_to_rail_times):
    """For each cell centroid, find reachable feeder stops within their
    Güteklassen walk buffer radii, then compute walk-to-stop + graph-derived
    time to each rail station.

    Transfer penalties are already embedded in feeder_stop_to_rail_times by
    _build_feeder_graph: a penalty of TRANSFER_PENALTY_SEC is added only when
    a path changes service line at an intermediate stop.  No flat penalty is
    applied here.
    """
    coords_grid = np.column_stack([grid.geometry.x, grid.geometry.y])
    results = []
    reli_values = grid['RELI'].values

    if feeder_stops.empty:
        return pd.DataFrame(results)

    def _feeder_mode_label(mode_str):
        m = mode_str.lower()
        if 'funicular' in m: return 'Funicular'
        if 'tram'      in m: return 'Tram'
        if 'ship'      in m: return 'Ship'
        return 'Bus'

    coords_fs   = np.column_stack([feeder_stops.geometry.x, feeder_stops.geometry.y])
    fs_radii    = feeder_stops['buffer_radius_m'].values.astype(float)
    fs_stop_ids = feeder_stops['stop_id'].astype(str).values
    fs_modes    = np.array([_feeder_mode_label(m)
                            for m in feeder_stops['mode'].astype(str)])

    # Query with the largest per-stop radius present; filter individually below
    max_radius = fs_radii.max() if len(fs_radii) > 0 else 1000.0
    tree = cKDTree(coords_fs)
    neighbors = tree.query_ball_point(coords_grid, r=max_radius)

    for i, near_idxs in enumerate(neighbors):
        for j in near_idxs:
            dist = np.sqrt((coords_grid[i, 0] - coords_fs[j, 0])**2 +
                           (coords_grid[i, 1] - coords_fs[j, 1])**2)
            if dist > fs_radii[j]:
                continue

            feeder_sid = fs_stop_ids[j]
            rail_times = feeder_stop_to_rail_times.get(feeder_sid, {})
            if not rail_times:
                continue

            walk_sec = dist / WALK_SPEED_MS
            for rail_id, graph_time_sec in rail_times.items():
                results.append({
                    'RELI': reli_values[i],
                    'rail_stop_id': rail_id,
                    'total_time_sec': walk_sec + graph_time_sec,
                    'access_mode': fs_modes[j]
                })

    return pd.DataFrame(results)


def _allocate_cells(walk_df, cycle_df, feeder_df, grid, rail_stations):
    """Hierarchical allocation: walk and PT-feeder compete on minimum time;
    cycling only fills cells that neither walk nor feeder can reach.

    Returns
    -------
    pd.DataFrame
        Allocation with columns: RELI, E_KOORD, N_KOORD, station_id, id_point,
        access_time_sec, access_mode.
    """
    print("    Running hierarchical allocation (walk+PT primary, cycle fallback) ...")

    rail_id_map = dict(zip(rail_stations['stop_id'].astype(str),
                           rail_stations['id_point']))

    # --- Primary competition: walk vs PT-feeder ---
    primary_frames = [df for df in [walk_df, feeder_df] if len(df) > 0]
    if primary_frames:
        primary = pd.concat(primary_frames, ignore_index=True)
        allocation = primary.loc[primary.groupby('RELI')['total_time_sec'].idxmin()].copy()
    else:
        allocation = pd.DataFrame(columns=['RELI', 'rail_stop_id', 'total_time_sec', 'access_mode'])

    # --- Cycling fallback: only for cells unreached by walk or PT ---
    if len(cycle_df) > 0:
        covered = set(allocation['RELI'])
        cycle_uncovered = cycle_df[~cycle_df['RELI'].isin(covered)]
        if len(cycle_uncovered) > 0:
            cycle_best = cycle_uncovered.loc[
                cycle_uncovered.groupby('RELI')['total_time_sec'].idxmin()].copy()
            allocation = pd.concat([allocation, cycle_best], ignore_index=True)

    if len(allocation) == 0:
        print("    WARNING: No access routes found - all cells get NO_PT")

    allocation = allocation.rename(columns={'rail_stop_id': 'station_id',
                                             'total_time_sec': 'access_time_sec'})
    allocation['id_point'] = allocation['station_id'].map(rail_id_map)

    grid_coords = grid[['RELI', 'E_KOORD', 'N_KOORD']].drop_duplicates(subset='RELI')
    allocation = allocation.merge(grid_coords, on='RELI', how='right')

    allocation['station_id'] = allocation['station_id'].fillna(str(NO_PT_ID))
    allocation['id_point'] = allocation['id_point'].fillna(NO_PT_ID)
    allocation['access_time_sec'] = allocation['access_time_sec'].fillna(99999)
    allocation['access_mode'] = allocation['access_mode'].fillna('No PT')

    allocation['id_point'] = allocation['id_point'].astype(int)

    n_assigned = (allocation['id_point'] != NO_PT_ID).sum()
    print(f"    {n_assigned}/{len(allocation)} cells assigned to a rail station")

    return allocation


def _build_candidates_csv(walk_df, cycle_df, feeder_df,
                           pop_grid, empl_grid, rail_stations, output_dir):
    """Build wide-format cell-station candidates CSV combining population and
    employment for each cell.

    Columns: RELI, Pop, Emp,
             Station_1_ID, Access_Time_1_min, Access_Mode_1, … (up to 5 slots).

    Slot ordering: nearest station first (by best access time across modes),
    then all available modes to that station in priority order
    (Walk → PT-Feeder → Cycle), then next nearest station, and so on.
    Cells with no access to any station are excluded.

    In the CSV output all PT feeder modes (Bus, Tram, Ship, Funicular) are
    collapsed to 'PT-Feeder'.  The plots use the original mode names.
    """
    _PT_FEEDER_MODES = {'Bus', 'Tram', 'Ship', 'Funicular'}
    _MODE_ORDER = {'Walk': 0, 'PT-Feeder': 1, 'Cycle': 2}

    rail_id_map = dict(zip(rail_stations['stop_id'].astype(str),
                           rail_stations['id_point']))

    pop_dict  = pop_grid.groupby('RELI')['NUMMER'].sum().to_dict()
    empl_dict = empl_grid.groupby('RELI')['NUMMER'].sum().to_dict()

    all_frames = [df for df in [walk_df, feeder_df, cycle_df] if len(df) > 0]
    if not all_frames:
        print("    WARNING: No access data — candidates CSV not written")
        return

    combined = pd.concat(all_frames, ignore_index=True)
    combined['id_point'] = combined['rail_stop_id'].astype(str).map(rail_id_map)

    # One entry per (cell, station, mode): keep the fastest path
    combined = (combined
                .groupby(['RELI', 'rail_stop_id', 'access_mode'], as_index=False)
                .agg(total_time_sec=('total_time_sec', 'min'),
                     id_point       =('id_point',       'first')))

    # Collapse all PT feeder mode names to 'PT-Feeder' for CSV output, then
    # re-deduplicate so Bus+Tram to the same station don't consume two slots.
    combined['access_mode'] = combined['access_mode'].apply(
        lambda m: 'PT-Feeder' if m in _PT_FEEDER_MODES else m)
    combined = (combined
                .groupby(['RELI', 'rail_stop_id', 'access_mode'], as_index=False)
                .agg(total_time_sec=('total_time_sec', 'min'),
                     id_point       =('id_point',       'first')))

    # Station rank within cell: minimum access time to that station across all modes
    station_min = (combined.groupby(['RELI', 'rail_stop_id'])['total_time_sec']
                   .min().rename('station_min_sec').reset_index())
    combined = combined.merge(station_min, on=['RELI', 'rail_stop_id'])

    # Sort: nearest station first, then Walk → Bus → Tram → Cycle within station
    combined['mode_ord'] = combined['access_mode'].map(_MODE_ORDER).fillna(9).astype(int)
    combined = combined.sort_values(['RELI', 'station_min_sec', 'mode_ord'],
                                    kind='mergesort')

    # Assign slot numbers (1-based) and keep top-N per cell
    combined['slot'] = combined.groupby('RELI').cumcount() + 1
    top_n = combined[combined['slot'] <= MAX_CANDIDATE_STATIONS].copy()

    top_n['time_min']   = (top_n['total_time_sec'] / 60).round(1)
    top_n['station_id'] = top_n['id_point'].fillna(NO_PT_ID).astype(int)

    # Pivot to wide format (fully vectorised)
    piv_id   = top_n.pivot(index='RELI', columns='slot', values='station_id')
    piv_time = top_n.pivot(index='RELI', columns='slot', values='time_min')
    piv_mode = top_n.pivot(index='RELI', columns='slot', values='access_mode')

    piv_id.columns   = [f'Station_{c}_ID'      for c in piv_id.columns]
    piv_time.columns = [f'Access_Time_{c}_min' for c in piv_time.columns]
    piv_mode.columns = [f'Access_Mode_{c}'     for c in piv_mode.columns]

    wide = pd.concat([piv_id, piv_time, piv_mode], axis=1)

    # Interleave into Station_1_ID, Access_Time_1_min, Access_Mode_1, Station_2_ID, …
    slot_cols = []
    for i in range(1, MAX_CANDIDATE_STATIONS + 1):
        for col in (f'Station_{i}_ID', f'Access_Time_{i}_min', f'Access_Mode_{i}'):
            if col in wide.columns:
                slot_cols.append(col)

    wide = wide[slot_cols].reset_index()
    wide['Pop'] = wide['RELI'].map(pop_dict).fillna(0).round().astype(int)
    wide['Emp'] = wide['RELI'].map(empl_dict).fillna(0).round().astype(int)
    wide = wide[['RELI', 'Pop', 'Emp'] + slot_cols]

    out_path = os.path.join(output_dir, 'cell_station_candidates.csv')
    wide.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"    Cell candidates ({len(wide):,} cells, ≤{MAX_CANDIDATE_STATIONS} slots each)"
          f" saved -> {out_path}")


# ===============================================================================
# OUTPUTS
# ===============================================================================

def _build_catchment_gpkg(allocation, pop_grid, empl_grid, rail_stations, output_dir):
    """Dissolve cells per station into catchment polygons.
    Schema: [train_station, id, station_name, pop, empl, geometry]."""
    print("  Building catchment GeoPackage ...")

    merged = allocation.merge(pop_grid[['RELI', 'geometry']], on='RELI', how='left')
    merged = gpd.GeoDataFrame(merged, geometry='geometry', crs=CODEBASE_CRS)
    merged['geometry'] = merged.geometry.buffer(CELL_SIZE_M / 2, cap_style=3)

    dissolved = merged.dissolve(by='id_point', as_index=False)
    dissolved = dissolved.rename(columns={'id_point': 'id'})
    dissolved['train_station'] = dissolved['id'].copy()
    dissolved['id'] = dissolved['id'].astype(int)
    dissolved['train_station'] = dissolved['train_station'].astype(int)

    # Per-station pop and empl from allocated cells
    pop_map  = pop_grid.set_index('RELI')['NUMMER']
    empl_map = empl_grid.set_index('RELI')['NUMMER']
    alloc = allocation.copy()
    alloc['_pop']  = alloc['RELI'].map(pop_map).fillna(0)
    alloc['_empl'] = alloc['RELI'].map(empl_map).fillna(0)
    stats = alloc.groupby('id_point').agg(
        pop=('_pop', 'sum'), empl=('_empl', 'sum')).astype(int).reset_index()
    stats['id'] = stats['id_point'].astype(int)

    # Station names
    name_map = rail_stations.set_index('id_point')['stop_name'].to_dict()
    dissolved['station_name'] = dissolved['id'].map(name_map).fillna('—')

    dissolved = dissolved.merge(stats[['id', 'pop', 'empl']], on='id', how='left')
    dissolved['pop']  = dissolved['pop'].fillna(0).astype(int)
    dissolved['empl'] = dissolved['empl'].fillna(0).astype(int)

    dissolved = dissolved[['train_station', 'id', 'station_name',
                            'pop', 'empl', 'geometry']].copy()

    out_path = os.path.join(output_dir, 'catchment.gpkg')
    dissolved.to_file(out_path, driver='GPKG')
    print(f"    Saved -> {out_path}  ({len(dissolved)} features)")

    return dissolved




def _build_visualisation(allocation, grid, rail_stations, boundary, method_label,
                         empl_grid=None,
                         walk_df=None, cycle_df=None, feeder_df=None,
                         catchment_gdf=None):
    """Produce a thesis-quality catchment visualisation as two side-by-side panels.

    Left panel:  cells coloured by access mode (Walk / Bus / Tram / Cycle / No PT).
    Right panel: cells coloured by station allocation (graph-coloured catchment fills).

    A 3-column summary table (Modal Access | Hierarchical | Non-Hierarchical) is
    rendered below the figure.

    Saved to plots/Catchment_Area/PT_Feeder/.
    """
    print(f"  Building {method_label} catchment visualisation ...")

    # --- Build 100×100 m square cells from allocation coordinates ---
    alloc = allocation.copy()
    alloc['geometry'] = [
        box(e, n, e + CELL_SIZE_M, n + CELL_SIZE_M)
        for e, n in zip(alloc['E_KOORD'], alloc['N_KOORD'])
    ]
    alloc = gpd.GeoDataFrame(alloc, geometry='geometry', crs=CODEBASE_CRS)
    alloc = gpd.clip(alloc, boundary)

    mode_colors = {
        'No PT':     '#d9d9d9',
        'Walk':      '#2ca02c',
        'Bus':       '#1f77b4',
        'Tram':      '#ff7f0e',
        'Ship':      '#004B8D',
        'Funicular': '#7B3F00',
        'Cycle':     '#9467bd',
    }
    mode_labels = {
        'No PT':     'No access',
        'Walk':      'Walk to rail',
        'Bus':       'Bus feeder',
        'Tram':      'Tram feeder',
        'Ship':      'Ship feeder',
        'Funicular': 'Funicular feeder',
        'Cycle':     'Cycle to rail',
    }

    # Graph-colouring palette (shared between both panels)
    boundary_palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#bcbd22', '#17becf', '#393b79',
        '#637939', '#8c6d31', '#843c39', '#7b4173', '#5254a3',
    ]

    # --- Build graph colouring for station catchments (right panel) ---
    color_map = {}   # id_point (int) -> hex colour
    clipped_catchment = None
    if catchment_gdf is not None and not catchment_gdf.empty:
        clipped_catchment = gpd.clip(catchment_gdf, boundary).reset_index(drop=True)
        id_col = 'id' if 'id' in clipped_catchment.columns else 'train_station'

        G = nx.Graph()
        for i in range(len(clipped_catchment)):
            G.add_node(clipped_catchment.loc[i, id_col])
        for i in range(len(clipped_catchment)):
            for j in range(i + 1, len(clipped_catchment)):
                geom_i = clipped_catchment.loc[i, 'geometry']
                geom_j = clipped_catchment.loc[j, 'geometry']
                if geom_i.intersects(geom_j):
                    inter = geom_i.intersection(geom_j)
                    if hasattr(inter, 'length') and inter.length > 0:
                        G.add_edge(clipped_catchment.loc[i, id_col],
                                   clipped_catchment.loc[j, id_col])

        coloring = nx.coloring.greedy_color(G, strategy='largest_first')
        n_colors = max(coloring.values()) + 1 if coloring else 1
        color_map = {sid: boundary_palette[cidx % len(boundary_palette)]
                     for sid, cidx in coloring.items()}
        max_deg = max(dict(G.degree()).values()) if G.degree() else 0
        print(f"    Graph colouring: {n_colors} colours for "
              f"{len(clipped_catchment)} station catchments (max adjacency {max_deg})")

    # --- Common base layers (loaded once) ---
    boundary_gdf = gpd.GeoDataFrame(geometry=[boundary], crs=CODEBASE_CRS)
    prep_bnd_vis = prep(boundary)
    stations_in_bnd = rail_stations[
        rail_stations.geometry.apply(lambda p: prep_bnd_vis.contains(p))]

    lakes_gdf = None
    if os.path.exists(paths.LAKES_SHP):
        lakes_raw = gpd.read_file(paths.LAKES_SHP).to_crs(CODEBASE_CRS)
        lakes_raw = lakes_raw[lakes_raw.geometry.intersects(boundary)]
        if not lakes_raw.empty:
            lakes_gdf = gpd.clip(lakes_raw, boundary)

    bx_min, by_min, bx_max, by_max = boundary.bounds
    pad = 200

    def _base_setup(ax, ylabel='N [m]'):
        """Apply grey background, boundary, lakes, stations, limits, map elements."""
        ax.set_facecolor('#E8E8E8')
        boundary_gdf.plot(ax=ax, color='white', edgecolor='none', zorder=0)
        if lakes_gdf is not None and not lakes_gdf.empty:
            lakes_gdf.plot(ax=ax, color='#A8D8EA', edgecolor='none', zorder=3)
        boundary_gdf.boundary.plot(ax=ax, color='black', linewidth=1.8,
                                   linestyle='--', zorder=5)
        if not stations_in_bnd.empty:
            ax.scatter(stations_in_bnd.geometry.x, stations_in_bnd.geometry.y,
                       s=25, c='white', edgecolors='black', linewidths=0.8,
                       marker='o', zorder=6)
        ax.set_xlim(bx_min - pad, bx_max + pad)
        ax.set_ylim(by_min - pad, by_max + pad)
        ax.set_aspect('equal')
        ax.set_xlabel('E [m]')
        ax.set_ylabel(ylabel)
        _add_map_elements(ax)

    # --- Figure: 1×2 panels ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(26, 11), sharey=True)
    fig.suptitle(f'Catchment Area Allocation — {method_label}', fontsize=14)

    # ---- Left panel: Access mode ----
    legend_left = []
    for mode, color in mode_colors.items():
        subset = alloc[alloc['access_mode'] == mode]
        if len(subset) > 0:
            subset.plot(ax=ax1, color=color, edgecolor='none', alpha=0.85, zorder=2)
            legend_left.append(
                Patch(facecolor=color, edgecolor='none', label=mode_labels[mode]))
    _base_setup(ax1, ylabel='N [m]')
    legend_left.append(
        Line2D([0], [0], color='black', linewidth=1.8, linestyle='--',
               label='Study area boundary'))
    legend_left.append(
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='black', markersize=8, label='Rail station'))
    ax1.set_title('Access Mode', fontsize=13)
    ax1.legend(handles=legend_left, loc='upper center',
               bbox_to_anchor=(0.5, -0.06), bbox_transform=ax1.transAxes,
               ncol=3, fontsize=8, framealpha=0.9, borderaxespad=0)

    # ---- Right panel: Station allocation ----
    legend_right = []
    if color_map:
        for sid, color in color_map.items():
            subset = alloc[alloc['id_point'] == sid]
            if len(subset) > 0:
                subset.plot(ax=ax2, color=color, edgecolor='none', alpha=0.85, zorder=2)
        legend_right.append(
            Patch(facecolor=boundary_palette[0], edgecolor='none',
                  label='Station catchment'))
    no_pt_cells = alloc[alloc['id_point'] == NO_PT_ID]
    if len(no_pt_cells) > 0:
        no_pt_cells.plot(ax=ax2, color='#d9d9d9', edgecolor='none', alpha=0.85, zorder=2)
    legend_right.append(
        Patch(facecolor='#d9d9d9', edgecolor='none', label='No access'))
    legend_right.append(
        Line2D([0], [0], color='black', linewidth=1.8, linestyle='--',
               label='Study area boundary'))
    legend_right.append(
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='black', markersize=8, label='Rail station'))
    _base_setup(ax2, ylabel='')
    ax2.set_title('Station Allocation', fontsize=13)
    ax2.legend(handles=legend_right, loc='upper center',
               bbox_to_anchor=(0.5, -0.06), bbox_transform=ax2.transAxes,
               ncol=3, fontsize=8, framealpha=0.9, borderaxespad=0)

    # --- Summary table: Modal Access | Hierarchical | Non-Hierarchical ---
    pop_map_d  = grid.set_index('RELI')['NUMMER'].to_dict()
    empl_map_d = empl_grid.set_index('RELI')['NUMMER'].to_dict() if empl_grid is not None else {}

    alloc['_pop']  = alloc['RELI'].map(pop_map_d).fillna(0)
    alloc['_empl'] = alloc['RELI'].map(empl_map_d).fillna(0)
    total_pop  = alloc['_pop'].sum()
    total_empl = alloc['_empl'].sum()

    # No access: same across all three columns (cells no mode can reach at all)
    no_acc_relis = set(alloc.loc[alloc['id_point'] == NO_PT_ID, 'RELI'].values)

    reli_pop_d  = alloc.set_index('RELI')['_pop'].to_dict()
    reli_empl_d = alloc.set_index('RELI')['_empl'].to_dict()

    def _reli_stats(relis):
        p  = sum(reli_pop_d.get(r,  0) for r in relis)
        e  = sum(reli_empl_d.get(r, 0) for r in relis)
        pp = p / total_pop  * 100 if total_pop  > 0 else 0.0
        ep = e / total_empl * 100 if total_empl > 0 else 0.0
        return int(p), pp, int(e), ep

    # Modal Access: cells counted in ALL reachable modes simultaneously
    walk_relis   = set(walk_df['RELI'].values)   if walk_df   is not None and len(walk_df)   > 0 else set()
    feeder_relis = set(feeder_df['RELI'].values) if feeder_df is not None and len(feeder_df) > 0 else set()
    cycle_relis  = set(cycle_df['RELI'].values)  if cycle_df  is not None and len(cycle_df)  > 0 else set()
    modal_access_stats = {
        'Walk':      _reli_stats(walk_relis),
        'PT Feeder': _reli_stats(feeder_relis),
        'Cycling':   _reli_stats(cycle_relis),
        'No access': _reli_stats(no_acc_relis),
    }

    # Hierarchical: winner-takes-all (Walk/PT primary; Cycle only fills uncovered cells)
    pt_modes = {'Bus', 'Tram', 'Ship', 'Funicular'}

    def _group_mode(mode):
        if mode == 'Walk':    return 'Walk'
        if mode in pt_modes:  return 'PT Feeder'
        if mode == 'Cycle':   return 'Cycling'
        return 'No access'

    alloc['_group'] = alloc['access_mode'].apply(_group_mode)
    hier_stats = {}
    for grp in ('Walk', 'PT Feeder', 'Cycling', 'No access'):
        sub = alloc[alloc['_group'] == grp]
        p  = int(sub['_pop'].sum())
        e  = int(sub['_empl'].sum())
        pp = p / total_pop  * 100 if total_pop  > 0 else 0.0
        ep = e / total_empl * 100 if total_empl > 0 else 0.0
        hier_stats[grp] = (p, pp, e, ep)

    # Non-Hierarchical: all three modes compete equally on raw travel time
    nh_frames = [df for df in [walk_df, feeder_df, cycle_df]
                 if df is not None and len(df) > 0]
    if nh_frames:
        nh_combined = pd.concat(nh_frames, ignore_index=True)
        nh_best = nh_combined.loc[
            nh_combined.groupby('RELI')['total_time_sec'].idxmin()].copy()
        nh_best['_group'] = nh_best['access_mode'].apply(_group_mode)
    else:
        nh_best = pd.DataFrame(columns=['RELI', '_group'])
    nh_reli_sets = {grp: set(nh_best.loc[nh_best['_group'] == grp, 'RELI'].values)
                    for grp in ('Walk', 'PT Feeder', 'Cycling')}
    nonhier_stats = {
        'Walk':      _reli_stats(nh_reli_sets['Walk']),
        'PT Feeder': _reli_stats(nh_reli_sets['PT Feeder']),
        'Cycling':   _reli_stats(nh_reli_sets['Cycling']),
        'No access': _reli_stats(no_acc_relis),
    }

    # Build 3-column table
    col_w = 9
    hdr = f"{'Pop':>{col_w}}{'Pop%':>{col_w}}{'FTE':>{col_w}}{'FTE%':>{col_w}}"
    tbl  = (f"{'':14}"
            f"{'── Modal Access ──':>{4*col_w}}  "
            f"{'── Hierarchical ──':>{4*col_w}}  "
            f"{'── Non-Hierarchical ──':>{4*col_w}}\n")
    tbl += f"{'Mode':<14}{hdr}  {hdr}  {hdr}\n"
    tbl += "─" * (14 + 3 * (4*col_w) + 4) + "\n"

    def _fmt(p, pp, e, ep):
        return f"{p:>{col_w},}{pp:>{col_w-1}.1f}%{e:>{col_w},}{ep:>{col_w-1}.1f}%"

    for grp in ('Walk', 'PT Feeder', 'Cycling', 'No access'):
        tbl += f"{grp:<14}{_fmt(*modal_access_stats[grp])}  {_fmt(*hier_stats[grp])}  {_fmt(*nonhier_stats[grp])}\n"
    tbl += "─" * (14 + 3 * (4*col_w) + 4) + "\n"
    tbl += ("Modal Access: all reachable modes counted per cell  |  "
            "Hierarchical: walk/PT primary + cycle fallback  |  "
            "Non-Hierarchical: fastest mode wins (cycle competes equally)")

    fig.subplots_adjust(top=0.93, bottom=0.24, wspace=0.08)
    fig.text(0.5, 0.02, tbl, ha='center', va='bottom',
             fontsize=7.5, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))

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
    e_arr = merged['E_KOORD'].values
    n_arr = merged['N_KOORD'].values
    merged['geometry'] = [box(e, n, e + CELL_SIZE_M, n + CELL_SIZE_M)
                          for e, n in zip(e_arr, n_arr)]
    merged = gpd.GeoDataFrame(merged, geometry='geometry', crs=CODEBASE_CRS)

    # Summary stats
    n_same = same.sum()
    n_diff = diff.sum()
    n_muni = muni_only.sum()
    n_pt   = pt_only.sum()
    print(f"    Cells — same station: {n_same:,}, different: {n_diff:,}, "
          f"municipal only: {n_muni:,}, PT-feeder only: {n_pt:,}")

    # --- Pop and empl per category (for summary table) ---
    pop_map  = pop_grid.set_index('RELI')['NUMMER'].to_dict()
    empl_map = empl_grid.set_index('RELI')['NUMMER'].to_dict()
    merged['_pop']  = merged['RELI'].map(pop_map).fillna(0)
    merged['_empl'] = merged['RELI'].map(empl_map).fillna(0)
    total_pop  = merged['_pop'].sum()
    total_empl = merged['_empl'].sum()

    def _cat_stats(cat):
        sub = merged[merged['category'] == cat]
        p = sub['_pop'].sum()
        e = sub['_empl'].sum()
        return (p / total_pop  * 100 if total_pop  > 0 else 0.0,
                e / total_empl * 100 if total_empl > 0 else 0.0)

    pct_same  = _cat_stats('same')
    pct_diff  = _cat_stats('different')
    pct_muni  = _cat_stats('muni_only')
    pct_pt    = _cat_stats('pt_only')

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

    # --- Station markers: white fill + black outline for all; green fill for
    #     stations that have a PT-feeder catchment but no municipalities assigned
    #     in the municipal method (Municipalities == '—' in station_catchment_summary.csv) ---
    prep_bnd_diff = prep(boundary)
    stations_in_bnd = rail_stations[
        rail_stations.geometry.apply(lambda p: prep_bnd_diff.contains(p))].copy()

    # Load no-municipality stations from the municipal summary CSV.
    # All ID sets use strings to match rail_stations['id_point'] (str from stop_id).
    summary_csv = os.path.join(MUNICIPAL_DATA_DIR, 'station_catchment_summary.csv')
    no_muni_ids: set = set()
    if os.path.exists(summary_csv):
        _sum = pd.read_csv(summary_csv)
        no_muni_mask = _sum['Municipalities'].astype(str).str.strip() == '—'
        no_muni_ids = set(
            _sum.loc[no_muni_mask, 'Station_Code'].astype(str).values
        )

    # A station is "PT-feeder new" if it has no municipal assignment AND exists
    # in the PT-feeder catchment
    pt_ids = {str(x) for x in pt_c['id_point'].values
              if str(x) != str(NO_PT_ID)}
    pt_new_ids = no_muni_ids & pt_ids

    sta_all    = stations_in_bnd
    sta_pt_new = stations_in_bnd[
        stations_in_bnd['id_point'].astype(str).isin(pt_new_ids)]

    # Draw all stations white+black first, then overlay green for new PT stations
    if not sta_all.empty:
        ax.scatter(sta_all.geometry.x, sta_all.geometry.y,
                   s=25, c='white', edgecolors='black', linewidths=0.8,
                   marker='o', zorder=6)
    if not sta_pt_new.empty:
        ax.scatter(sta_pt_new.geometry.x, sta_pt_new.geometry.y,
                   s=25, c='#2ca02c', edgecolors='black', linewidths=0.8,
                   marker='o', zorder=7)

    legend_handles += [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='black', markersize=8,
               label='Rail station (both methods)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c',
               markeredgecolor='black', markersize=8,
               label='Rail station (PT-Feeder only)'),
    ]

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

    ax.set_title('Catchment Comparison: Municipal vs PT-Feeder', fontsize=14)
    ax.set_xlabel('E [m]')
    ax.set_ylabel('N [m]')
    ax.set_aspect('equal')

    # Add cartographic elements
    _add_map_elements(ax)

    # --- Summary table ---
    col_w = 9
    tbl  = "Population & employment shift: Municipal → PT-Feeder\n"
    tbl += f"{'Category':<28}{'Pop (%)':>{col_w}}{'FTE (%)':>{col_w}}\n"
    tbl += "─" * (28 + 2 * col_w) + "\n"
    rows = [
        ('Same station',              pct_same),
        ('Changed station',           pct_diff),
        ('Lost access (Mun. only)',   pct_muni),
        ('Gained access (PT only)',   pct_pt),
    ]
    for label, (pp, ep) in rows:
        tbl += f"{label:<28}{pp:>{col_w-1}.1f}%{ep:>{col_w-1}.1f}%\n"

    # Legend and summary stacked vertically, centered below the plot
    fig.subplots_adjust(bottom=0.30)
    ax.legend(handles=legend_handles, loc='upper center',
              bbox_to_anchor=(0.5, -0.08),
              bbox_transform=ax.transAxes,
              ncol=2, fontsize=8, framealpha=0.9, borderaxespad=0)
    fig.text(0.5, 0.18, tbl, ha='center', va='top',
             fontsize=9, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))

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
                       rail_stations, feeder_stops, boundary, empl_grid=None):
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

    # Load lakes once for all access-time plots
    lakes = None
    if os.path.exists(paths.LAKES_SHP):
        lakes = gpd.read_file(paths.LAKES_SHP).to_crs(CODEBASE_CRS)
        lakes = lakes[lakes.geometry.intersects(boundary)].copy()

    # Pre-build square polygon geometry for the grid (100 m × 100 m)
    _e = pop_grid['E_KOORD'].values
    _n = pop_grid['N_KOORD'].values
    squares = [box(e, n, e + CELL_SIZE_M, n + CELL_SIZE_M)
               for e, n in zip(_e, _n)]
    grid_plot = gpd.GeoDataFrame(
        pop_grid[['RELI', 'E_KOORD', 'N_KOORD']].copy(),
        geometry=squares, crs=CODEBASE_CRS
    )

    def _class(time_sec):
        """Convert array of seconds to 0-based class index into _ACCESS_BINS."""
        mins = np.array(time_sec, dtype=float) / 60.0
        idx  = np.digitize(mins, _ACCESS_BINS[1:], right=False)
        return np.clip(idx, 0, n_bins - 1)

    pop_map  = pop_grid.set_index('RELI')['NUMMER'].to_dict()
    empl_map = empl_grid.set_index('RELI')['NUMMER'].to_dict() if empl_grid is not None else {}

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

        # Attach pop/empl values for the summary
        gdf['_pop']  = gdf['RELI'].map(pop_map).fillna(0)
        gdf['_empl'] = gdf['RELI'].map(empl_map).fillna(0)
        total_pop_grid  = gdf['_pop'].sum()
        total_empl_grid = gdf['_empl'].sum()
        total_pop  = gdf.loc[in_buffer, '_pop'].sum()
        total_empl = gdf.loc[in_buffer, '_empl'].sum()

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        # Grey for out-of-buffer cells
        if (~in_buffer).any():
            gdf[~in_buffer].plot(ax=ax, color=_ACCESS_GREY, edgecolor='none', linewidth=0)

        # Coloured cells per class
        for ci, colour in enumerate(colours):
            mask = gdf['class_idx'] == ci
            if mask.any():
                gdf[mask].plot(ax=ax, color=colour, edgecolor='none', linewidth=0)

        # Lakes — above coloured cells, below boundary
        if lakes is not None and not lakes.empty:
            gpd.clip(lakes, boundary).plot(ax=ax, color='#A8D8EA', edgecolor='none', zorder=3)

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
        if lakes is not None and not lakes.empty:
            legend_handles.append(Patch(facecolor='#A8D8EA', edgecolor='none', label='Lake'))
        legend_handles.append(Line2D([0], [0], color='black', linewidth=1.8,
                                     linestyle='--', label='Catchment area boundary'))

        ax.set_title(title, fontsize=13)
        ax.set_xlabel('E [m]')
        ax.set_ylabel('N [m]')
        ax.set_aspect('equal')

        # Add cartographic elements
        _add_map_elements(ax)

        # --- Summary table: pop and FTE by access-time bin ---
        # Percentages are relative to total grid (in-buffer + outside buffer)
        col_w = 9
        tbl  = f"{'Bin (min)':<10}{'Pop':>{col_w}}{'Pop%':>{col_w}}{'FTE':>{col_w}}{'FTE%':>{col_w}}\n"
        tbl += "─" * (10 + 4 * col_w) + "\n"
        for ci in sorted(used_classes):
            mask_ci = gdf['class_idx'] == ci
            p  = int(gdf.loc[mask_ci, '_pop'].sum())
            e  = int(gdf.loc[mask_ci, '_empl'].sum())
            pp = p / total_pop_grid  * 100 if total_pop_grid  > 0 else 0.0
            ep = e / total_empl_grid * 100 if total_empl_grid > 0 else 0.0
            tbl += (f"{_ACCESS_LABELS[ci]:<10}"
                    f"{p:>{col_w},}{pp:>{col_w-1}.1f}%"
                    f"{e:>{col_w},}{ep:>{col_w-1}.1f}%\n")
        p_out = int(total_pop_grid  - total_pop)
        e_out = int(total_empl_grid - total_empl)
        pp_out = p_out / total_pop_grid  * 100 if total_pop_grid  > 0 else 0.0
        ep_out = e_out / total_empl_grid * 100 if total_empl_grid > 0 else 0.0
        tbl += (f"{'Outside':<10}"
                f"{p_out:>{col_w},}{pp_out:>{col_w-1}.1f}%"
                f"{e_out:>{col_w},}{ep_out:>{col_w-1}.1f}%\n")
        p_tot  = int(total_pop_grid)
        e_tot  = int(total_empl_grid)
        tbl += "─" * (10 + 4 * col_w) + "\n"
        tbl += f"{'Total':<10}{p_tot:>{col_w},}{'100.0':>{col_w}}%{e_tot:>{col_w},}{'100.0':>{col_w-1}}%\n"

        # Legend and summary stacked vertically, centered below the plot
        fig.subplots_adjust(bottom=0.30)
        ax.legend(handles=legend_handles, title='Access time (min)',
                  title_fontsize=8, fontsize=7,
                  loc='upper center',
                  bbox_to_anchor=(0.5, -0.08),
                  bbox_transform=ax.transAxes,
                  ncol=3, framealpha=0.9, borderaxespad=0)
        fig.text(0.5, 0.16, tbl, ha='center', va='top',
                 fontsize=8, fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))

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
# STEP 2-PT: ÖV-GÜTEKLASSEN CLASSIFICATION (ARE 2022)
# ===============================================================================

def _headway_to_kat(headway_min, mode_group, is_bahnknoten=False):
    """Convert headway (min) to Haltestellenkategorie integer (1=I … 5=V).

    Returns 0 if headway > 60 min (no category) or mode_group is unknown.
    """
    if not np.isfinite(headway_min) or headway_min <= 0 or headway_min > 60:
        return 0

    if mode_group == 'C':       # Funicular: always Kat. V
        return 5

    # Headway bucket: 0 = <5, 1 = 5–9, 2 = 10–19, 3 = 20–39, 4 = 40–60
    if   headway_min <  5: bucket = 0
    elif headway_min < 10: bucket = 1
    elif headway_min < 20: bucket = 2
    elif headway_min < 40: bucket = 3
    else:                  bucket = 4

    if mode_group == 'A':
        return [1, 1, 2, 3, 4][bucket] if is_bahnknoten else [1, 2, 3, 4, 5][bucket]
    else:                               # mode_group == 'B'
        return [2, 3, 4, 5, 5][bucket]


def _kat_to_ring_records(stop_row, boundary_geom=None):
    """Return a list of ring-polygon records for one stop.

    Each record is a dict with keys:
        stop_id, stop_name, hst_kat, gk_class, dist_band,
        headway_min, n_lines, is_bahnknoten, geometry
    """
    kat = int(stop_row['hst_kat'])
    if kat == 0:
        return []

    max_radius = GK_MAX_RADIUS[kat]
    records = []

    for band_idx, (r_inner, r_outer) in enumerate(GK_DIST_BANDS):
        if r_inner >= max_radius:
            break
        r_outer_clamp = min(r_outer, max_radius)
        gk = _GK_LOOKUP.get((kat, band_idx))
        if gk is None:
            continue

        outer = stop_row.geometry.buffer(r_outer_clamp)
        geom  = outer.difference(stop_row.geometry.buffer(r_inner)) if r_inner > 0 else outer
        if boundary_geom is not None:
            geom = geom.intersection(boundary_geom)
        if geom.is_empty:
            continue

        records.append({
            'stop_id':       str(stop_row['stop_id']),
            'stop_name':     stop_row.get('stop_name', ''),
            'hst_kat':       kat,
            'gk_class':      gk,
            'dist_band':     GK_DIST_LABELS[band_idx],
            'headway_min':   round(float(stop_row.get('headway_min', np.inf)), 1),
            'n_lines':       int(stop_row.get('n_lines', 0)),
            'is_bahnknoten': bool(stop_row.get('is_bahnknoten', False)),
            'geometry':      geom,
        })

    return records


def _compute_stop_gueteklassen(feeder_stops, rail_stops, temporal='all'):
    """Classify every feeder and rail stop according to the ARE ÖV-Güteklassen
    methodology (ARE 2022) and add the following columns to both GDFs:

        hst_kat         int   0 (none) or 1–5  (I–V)
        buffer_radius_m int   maximum walk-access radius derived from hst_kat
        headway_min     float effective combined headway across all lines at stop
        n_lines         int   number of distinct routes serving the stop
        is_bahnknoten   bool  True if ≥ 2 distinct rail routes serve the stop

    Headway method: for each route compute a bidirectional average dep_value,
    then sum across all routes in the same mode group at the stop:
        temporal='all'     → dep_value = mean(total_dep, dir0, dir1)
                             headway   = GK_WINDOW_MIN / sum_per_stop
        temporal='peak'    → dep_value = mean((freq_am_peak + freq_pm_peak)/2, dir0, dir1)
                             headway   = 60 / sum_per_stop
        temporal='offpeak' → dep_value = mean(freq_offpeak, dir0, dir1)
                             headway   = 60 / sum_per_stop

    For stops served by multiple mode groups the best (lowest) Kat wins.
    Bahnknoten threshold: ≥ 2 distinct rail route_ids.
    """
    print("  Computing ÖV-Güteklassen stop categories ...")

    subfolder_map = {'all': '',       'peak': 'Peak',     'offpeak': 'Off_Peak'}
    suffix_map    = {'all': '',       'peak': '_peak',    'offpeak': '_offpeak'}
    subfolder = subfolder_map[temporal]
    suffix    = suffix_map[temporal]

    feeder_base = os.path.join(paths.FEEDER_LINES_DIR, GTFS_NETWORK_FOLDER)
    rail_base   = os.path.join(paths.RAIL_PROCESSED_DIR, GTFS_NETWORK_FOLDER)

    def _versioned(base, name):
        if subfolder:
            return os.path.join(base, subfolder, f'{name}{suffix}.gpkg')
        return os.path.join(base, f'{name}.gpkg')

    # --- Load lines (frequency data) ---
    feeder_mode_grp = {'tram': 'B', 'bus': 'B', 'ship': 'B', 'funicular': 'C'}
    rail_mode_grp   = {k: 'A' for k in
                       ['sbahn', 'long_distance_rail',
                        'inter_regional_rail', 'regional_rail']}

    def _load_lines(path, mode_grp_map):
        frames = []
        for layer_name, _ in pyogrio.list_layers(path):
            gdf = gpd.read_file(path, layer=layer_name)
            gdf['mode_group'] = mode_grp_map.get(layer_name, 'B')
            frames.append(gdf)
        return pd.concat(frames, ignore_index=True)

    feeder_lines = _load_lines(_versioned(feeder_base, 'pt_feeder_lines'), feeder_mode_grp)
    rail_lines   = _load_lines(_versioned(rail_base,   'rail_lines'),      rail_mode_grp)
    all_lines = pd.concat([feeder_lines, rail_lines], ignore_index=True)

    # Compute per-row dep_value, keeping both direction_ids.
    # The bidirectional average is produced by mean() in the route_dep aggregation.
    if temporal == 'peak':
        # Average AM-peak and PM-peak frequencies per row
        all_lines['dep_value'] = (
            pd.to_numeric(all_lines['freq_am_peak_dep_hr'], errors='coerce').fillna(0) +
            pd.to_numeric(all_lines['freq_pm_peak_dep_hr'], errors='coerce').fillna(0)
        ) / 2
        use_rate = True    # headway = 60 / sum_per_stop
    elif temporal == 'offpeak':
        all_lines['dep_value'] = pd.to_numeric(
            all_lines['freq_offpeak_dep_hr'], errors='coerce').fillna(0)
        use_rate = True    # headway = 60 / sum_per_stop
    else:  # 'all'
        all_lines['dep_value'] = pd.to_numeric(
            all_lines['total_dep'], errors='coerce').fillna(0)
        use_rate = False   # headway = GK_WINDOW_MIN / sum_per_stop

    # One dep_value per (route_id, variant_rank): mean across direction_ids
    # gives the bidirectional average naturally (both dirs present → (dir0+dir1)/2).
    route_dep = (all_lines.groupby(['route_id', 'variant_rank'], as_index=False)
                 .agg(dep_value=('dep_value', 'mean'),
                      mode_group=('mode_group', 'first')))

    # --- Load segments to map stops → routes ---
    def _load_segs(path):
        frames = []
        for layer_name, _ in pyogrio.list_layers(path):
            gdf = gpd.read_file(path, layer=layer_name)
            frames.append(
                gdf[['from_stop_id', 'to_stop_id',
                     'route_id', 'direction_id', 'variant_rank']].copy())
        return pd.concat(frames, ignore_index=True)

    feeder_segs = _load_segs(_versioned(feeder_base, 'pt_feeder_segments'))
    rail_segs   = _load_segs(_versioned(rail_base,   'rail_segments'))
    all_segs = pd.concat([feeder_segs, rail_segs], ignore_index=True)
    # Keep both directions so terminal stops that only appear in direction_id=1 are included.

    # Build unique (stop_id, route_id, variant_rank) pairs
    from_p = all_segs[['from_stop_id', 'route_id', 'variant_rank']].rename(
        columns={'from_stop_id': 'stop_id'})
    to_p   = all_segs[['to_stop_id',   'route_id', 'variant_rank']].rename(
        columns={'to_stop_id':   'stop_id'})
    pairs = (pd.concat([from_p, to_p])
               .drop_duplicates()
               .astype({'stop_id': str}))

    # Join departure counts
    pairs = pairs.merge(route_dep, on=['route_id', 'variant_rank'], how='left')
    pairs = pairs.dropna(subset=['dep_value'])
    pairs = pairs[pairs['dep_value'] > 0]

    # --- Aggregate per stop per mode group ---
    stop_grp = (pairs.groupby(['stop_id', 'mode_group'], as_index=False)
                .agg(dep_sum  =('dep_value', 'sum'),
                     n_routes =('route_id',  'nunique')))

    if use_rate:
        stop_grp['headway_min'] = 60.0 / stop_grp['dep_sum'].clip(lower=1e-6)
    else:
        stop_grp['headway_min'] = GK_WINDOW_MIN / stop_grp['dep_sum'].clip(lower=1e-6)

    # Bahnknoten: rail stops with ≥ 2 distinct route_ids
    rail_route_cnt = (pairs[pairs['mode_group'] == 'A']
                      .groupby('stop_id')['route_id']
                      .nunique()
                      .rename('rail_n_routes')
                      .reset_index())

    # Pivot headways to wide format: one row per stop
    pivot = stop_grp.pivot(index='stop_id', columns='mode_group',
                           values='headway_min').reset_index()
    pivot.columns.name = None
    for col in ('A', 'B', 'C'):
        if col not in pivot.columns:
            pivot[col] = np.nan
    pivot = pivot.rename(columns={'A': 'hw_A', 'B': 'hw_B', 'C': 'hw_C'})
    pivot = pivot.merge(rail_route_cnt, on='stop_id', how='left')
    pivot['is_bahnknoten'] = pivot['rail_n_routes'].fillna(0) >= 2

    # Per-stop n_lines (all groups combined)
    n_lines = (pairs.groupby('stop_id')['route_id']
               .nunique().rename('n_lines').reset_index())
    pivot = pivot.merge(n_lines, on='stop_id', how='left')
    pivot['n_lines'] = pivot['n_lines'].fillna(0).astype(int)

    # Best headway across groups (for display)
    pivot['headway_min'] = pivot[['hw_A', 'hw_B', 'hw_C']].min(axis=1)

    # Classify: best (lowest) Kat across all mode groups at the stop
    def _best_kat(row):
        kats = [
            _headway_to_kat(row['hw_A'], 'A', row['is_bahnknoten']),
            _headway_to_kat(row['hw_B'], 'B'),
            _headway_to_kat(row['hw_C'], 'C'),
        ]
        valid = [k for k in kats if k > 0]
        return min(valid) if valid else 0

    pivot['hst_kat'] = pivot.apply(_best_kat, axis=1)
    pivot['buffer_radius_m'] = pivot['hst_kat'].map(GK_MAX_RADIUS).fillna(0).astype(int)

    gk_cols = ['stop_id', 'hst_kat', 'buffer_radius_m',
               'headway_min', 'n_lines', 'is_bahnknoten']
    gk = pivot[gk_cols].copy()

    def _apply_gk(stops_gdf):
        out = stops_gdf.copy()
        out['_sid'] = out['stop_id'].astype(str)
        out = out.merge(gk, left_on='_sid', right_on='stop_id',
                        how='left', suffixes=('', '_gk'))
        out = out.drop(columns=['_sid', 'stop_id_gk'], errors='ignore')
        out['hst_kat']       = out['hst_kat'].fillna(0).astype(int)
        out['buffer_radius_m'] = out['buffer_radius_m'].fillna(0).astype(int)
        out['headway_min']   = out['headway_min'].fillna(np.inf)
        out['n_lines']       = out['n_lines'].fillna(0).astype(int)
        out['is_bahnknoten'] = out['is_bahnknoten'].fillna(False)
        return gpd.GeoDataFrame(out, geometry='geometry', crs=CODEBASE_CRS)

    feeder_stops = _apply_gk(feeder_stops)
    rail_stops   = _apply_gk(rail_stops)

    n_cls = ((feeder_stops['hst_kat'] > 0).sum()
             + (rail_stops['hst_kat'] > 0).sum())
    n_tot = len(feeder_stops) + len(rail_stops)
    print(f"    Classified {n_cls}/{n_tot} stops  "
          f"(window={GK_WINDOW_MIN} min, method={'rate' if use_rate else 'total_dep'})")

    kat_names = {0: 'none', 1: 'I', 2: 'II', 3: 'III', 4: 'IV', 5: 'V'}
    for k in range(6):
        n = ((feeder_stops['hst_kat'] == k).sum()
             + (rail_stops['hst_kat'] == k).sum())
        if n > 0:
            print(f"      Kat {kat_names[k]}: {n:,} stops")

    return feeder_stops, rail_stops


def _dissolve_hierarchical(rings_gdf):
    """Return a GeoDataFrame with one row per Güteklasse (A–D), where each
    class geometry has been clipped to remove overlap with better classes.

    A is best: unchanged.  B = B_raw − A_raw.  C = C_raw − (A∪B)_raw.  Etc.
    """
    class_order = ['A', 'B', 'C', 'D']
    raw_union = {}
    for cls in class_order:
        sub = rings_gdf[rings_gdf['gk_class'] == cls]
        raw_union[cls] = unary_union(sub.geometry.values) if not sub.empty else None

    rows = []
    cumulative_mask = None
    for cls in class_order:
        geom = raw_union[cls]
        if geom is not None and not geom.is_empty:
            if cumulative_mask is not None:
                geom = geom.difference(cumulative_mask)
            if not geom.is_empty:
                rows.append({'gk_class': cls, 'geometry': geom})
        if raw_union[cls] is not None:
            cumulative_mask = (raw_union[cls] if cumulative_mask is None
                               else cumulative_mask.union(raw_union[cls]))
    return gpd.GeoDataFrame(rows, geometry='geometry', crs=CODEBASE_CRS)


def _build_gueteklassen_gpkgs(feeder_stops, rail_stops, boundary, pop_grid, empl_grid):
    """Save two GeoPackages to PT_FEEDER_DATA_DIR:

    gueteklassen_by_stop.gpkg   — ring polygons per stop (one layer per mode)
                                  with cross-stop hierarchy applied: lower-class
                                  rings are clipped where higher-class rings overlap.
    gueteklassen_by_class.gpkg  — single layer, 4 rows (one per Güteklasse A–D)
                                  after hierarchical masking; columns:
                                  fid, Class, Class_Area_m2, Pop, Empl.
    """
    print("  Building Güteklassen GeoPackages ...")

    # Tag each stop with its transport-mode layer name
    feeder_s = feeder_stops.copy()
    feeder_s['_layer'] = feeder_s['mode'].str.lower().apply(
        lambda m: ('funicular' if 'funicular' in m else
                   'tram'      if 'tram'      in m else
                   'ship'      if 'ship'      in m else 'bus'))
    rail_s = rail_stops.copy()
    rail_s['_layer'] = 'rail'

    all_stops = gpd.GeoDataFrame(
        pd.concat([feeder_s, rail_s], ignore_index=True),
        geometry='geometry', crs=CODEBASE_CRS)

    all_records = []
    for _, row in all_stops.iterrows():
        rings = _kat_to_ring_records(row, boundary_geom=boundary)
        for r in rings:
            r['transport_mode'] = row['_layer']
        all_records.extend(rings)

    if not all_records:
        print("    WARNING: no classified stops — skipping GPKG output")
        return

    rings_gdf = gpd.GeoDataFrame(all_records, geometry='geometry', crs=CODEBASE_CRS)

    # Pre-compute raw class unions for hierarchy masking
    class_order = ['A', 'B', 'C', 'D']
    raw_union = {}
    for cls in class_order:
        sub = rings_gdf[rings_gdf['gk_class'] == cls]
        raw_union[cls] = unary_union(sub.geometry.values) if not sub.empty else None

    # --- GPKG 1: by stop — cross-stop hierarchy applied to individual rings ---
    gpkg1 = os.path.join(PT_FEEDER_DATA_DIR, 'gueteklassen_by_stop.gpkg')
    if os.path.exists(gpkg1):
        os.remove(gpkg1)

    clipped_frames = []
    cumulative_mask = None
    for cls in class_order:
        cls_rings = rings_gdf[rings_gdf['gk_class'] == cls].copy()
        if not cls_rings.empty and cumulative_mask is not None:
            cls_rings['geometry'] = cls_rings.geometry.apply(
                lambda g: g.difference(cumulative_mask))
            cls_rings = cls_rings[~cls_rings.geometry.is_empty].copy()
        clipped_frames.append(cls_rings)
        if raw_union[cls] is not None:
            cumulative_mask = (raw_union[cls] if cumulative_mask is None
                               else cumulative_mask.union(raw_union[cls]))

    clipped_rings = gpd.GeoDataFrame(
        pd.concat(clipped_frames, ignore_index=True),
        geometry='geometry', crs=CODEBASE_CRS)

    for layer in ['rail', 'bus', 'tram', 'ship', 'funicular']:
        subset = clipped_rings[clipped_rings['transport_mode'] == layer].drop(
            columns=['transport_mode'], errors='ignore')
        if not subset.empty:
            subset.to_file(gpkg1, layer=layer, driver='GPKG')
    print(f"    Saved -> {gpkg1}  ({len(clipped_rings):,} features)")

    # --- GPKG 2: by class — single layer, 4 rows, hierarchical, with stats ---
    gpkg2 = os.path.join(PT_FEEDER_DATA_DIR, 'gueteklassen_by_class.gpkg')
    if os.path.exists(gpkg2):
        os.remove(gpkg2)

    # Build exclusive geometries (A intact, each lower class minus better classes)
    exclusive_union = {}
    cumulative_mask = None
    for cls in class_order:
        geom = raw_union[cls]
        if geom is not None and not geom.is_empty:
            if cumulative_mask is not None:
                geom = geom.difference(cumulative_mask)
            exclusive_union[cls] = geom if not geom.is_empty else None
        else:
            exclusive_union[cls] = None
        if raw_union[cls] is not None:
            cumulative_mask = (raw_union[cls] if cumulative_mask is None
                               else cumulative_mask.union(raw_union[cls]))

    pop_pts  = pop_grid[['geometry', 'NUMMER']].copy()
    empl_pts = empl_grid[['geometry', 'NUMMER']].copy()

    class_rows = []
    for i, cls in enumerate(class_order, 1):
        geom = exclusive_union.get(cls)
        if geom is None or geom.is_empty:
            continue
        cls_poly = gpd.GeoDataFrame([{'geometry': geom}],
                                    geometry='geometry', crs=CODEBASE_CRS)
        pop_in  = int(gpd.sjoin(pop_pts,  cls_poly, how='inner',
                                predicate='within')['NUMMER'].sum())
        empl_in = int(gpd.sjoin(empl_pts, cls_poly, how='inner',
                                predicate='within')['NUMMER'].sum())
        class_rows.append({
            'fid':           i,
            'Class':         cls,
            'Class_Area_m2': round(geom.area, 0),
            'Pop':           pop_in,
            'Empl':          empl_in,
            'geometry':      geom,
        })

    if class_rows:
        by_class = gpd.GeoDataFrame(class_rows, geometry='geometry', crs=CODEBASE_CRS)
        by_class.to_file(gpkg2, layer='gueteklassen', driver='GPKG')
        print(f"    Saved -> {gpkg2}  ({len(by_class)} classes)")
    else:
        print(f"    WARNING: no classified zones — skipping {gpkg2}")


def _plot_gueteklassen_comparison(feeder_stops, rail_stops, boundary, pop_grid, empl_grid):
    """Side-by-side map: our computed Güteklassen vs. official ARE 2026 data,
    plus a diff panel showing improvement/worsening per cell.
    Includes a summary table in the figure with coverage of inhabited area,
    pop%, and FTE% per class, plus diff statistics.
    """
    print("  Plotting Güteklassen comparison ...")

    are_path = os.path.join(paths.MAIN, _GUETEKLASSEN_ARE_GPKG)
    if not os.path.exists(are_path):
        print(f"    WARNING: official ARE data not found at {are_path} — skipping")
        return

    official = gpd.read_file(are_path, layer='OeV_Gueteklassen_ARE').to_crs(CODEBASE_CRS)
    official = gpd.clip(official, boundary)

    # Build computed rings from classified stops
    all_stops = gpd.GeoDataFrame(
        pd.concat([feeder_stops, rail_stops], ignore_index=True),
        geometry='geometry', crs=CODEBASE_CRS)

    all_records = []
    for _, row in all_stops.iterrows():
        all_records.extend(_kat_to_ring_records(row, boundary_geom=boundary))

    if not all_records:
        print("    WARNING: no classified stops — skipping comparison plot")
        return

    rings = gpd.GeoDataFrame(all_records, geometry='geometry', crs=CODEBASE_CRS)
    # Apply hierarchical dissolving so lower classes don't overlap higher ones
    computed = _dissolve_hierarchical(rings)

    # --- Inhabited area: union of 100 m cell squares that contain pop or empl ---
    inhab_cells = pd.concat([
        pop_grid[pop_grid['NUMMER'] > 0][['E_KOORD', 'N_KOORD', 'RELI']],
        empl_grid[empl_grid['NUMMER'] > 0][['E_KOORD', 'N_KOORD', 'RELI']],
    ]).drop_duplicates(subset='RELI')
    inhabited_geom = unary_union([
        box(e, n, e + CELL_SIZE_M, n + CELL_SIZE_M)
        for e, n in zip(inhab_cells['E_KOORD'], inhab_cells['N_KOORD'])
    ])
    inhabited_area = inhabited_geom.area

    # --- Coverage statistics: % of inhabited area ---
    def _area_pct(gdf, col):
        """% of inhabited area covered by each class."""
        result = {}
        for gk in ('A', 'B', 'C', 'D'):
            if (gdf[col] == gk).any():
                class_geom = unary_union(gdf.loc[gdf[col] == gk, 'geometry'])
                result[gk] = class_geom.intersection(inhabited_geom).area / inhabited_area * 100
            else:
                result[gk] = 0.0
        # No-class: inhabited area not covered by any class
        all_covered = unary_union(gdf['geometry'])
        result['No class'] = (
            inhabited_geom.difference(all_covered).area / inhabited_area * 100
        )
        return result

    comp_pct = _area_pct(computed, 'gk_class')
    off_pct  = _area_pct(official, 'KLASSE')

    # --- Pop % and FTE % per class ---
    total_pop  = pop_grid['NUMMER'].sum()
    total_empl = empl_grid['NUMMER'].sum()
    pop_pts  = pop_grid[['geometry', 'NUMMER', 'RELI']].copy()
    empl_pts = empl_grid[['geometry', 'NUMMER', 'RELI']].copy()

    def _pop_empl_pct(gdf, col):
        """% of total pop and empl in each class (no-class = remainder)."""
        pop_j  = gpd.sjoin(pop_pts,  gdf[['geometry', col]],
                           how='left', predicate='within')
        pop_j  = pop_j[~pop_j.index.duplicated(keep='first')]
        empl_j = gpd.sjoin(empl_pts, gdf[['geometry', col]],
                           how='left', predicate='within')
        empl_j = empl_j[~empl_j.index.duplicated(keep='first')]

        pop_res  = {}
        empl_res = {}
        for gk in ('A', 'B', 'C', 'D'):
            p  = pop_j.loc[pop_j[col]  == gk, 'NUMMER'].sum()
            e  = empl_j.loc[empl_j[col] == gk, 'NUMMER'].sum()
            pop_res[gk]  = p  / total_pop  * 100 if total_pop  > 0 else 0.0
            empl_res[gk] = e  / total_empl * 100 if total_empl > 0 else 0.0
        # No class = not matched by any class
        pop_res['No class']  = pop_j[pop_j[col].isna()]['NUMMER'].sum() / total_pop  * 100 \
            if total_pop  > 0 else 0.0
        empl_res['No class'] = empl_j[empl_j[col].isna()]['NUMMER'].sum() / total_empl * 100 \
            if total_empl > 0 else 0.0
        return pop_res, empl_res

    comp_pop_pct, comp_empl_pct = _pop_empl_pct(computed, 'gk_class')
    off_pop_pct,  off_empl_pct  = _pop_empl_pct(official, 'KLASSE')

    # --- Build diff cells for third panel ---
    # Class hierarchy: A > B > C > D > None (A is best = 0, None = 4)
    class_rank = {'A': 0, 'B': 1, 'C': 2, 'D': 3, None: 4}

    # Build cell geometries for inhabited cells
    cell_geoms = [
        box(e, n, e + CELL_SIZE_M, n + CELL_SIZE_M)
        for e, n in zip(inhab_cells['E_KOORD'], inhab_cells['N_KOORD'])
    ]
    cells_gdf = gpd.GeoDataFrame(
        inhab_cells[['RELI', 'E_KOORD', 'N_KOORD']].copy(),
        geometry=cell_geoms, crs=CODEBASE_CRS
    )

    # Area-based majority classification: assign each cell to the class covering
    # the majority of its area (eliminates false "neither" for boundary cells)
    def _classify_by_area_majority(cells, class_gdf, class_col):
        """For each cell, compute intersection area with each class polygon
        and return the class that covers the majority of the cell's area."""
        results = [None] * len(cells)
        class_rank = {'A': 0, 'B': 1, 'C': 2, 'D': 3}  # lower rank = better class

        for idx, cell_row in enumerate(cells.itertuples()):
            cell_geom = cell_row.geometry
            cell_area = cell_geom.area
            best_class = None
            best_area = 0.0

            # Find all intersecting class polygons
            candidates = class_gdf[class_gdf.geometry.intersects(cell_geom)]
            for _, poly_row in candidates.iterrows():
                try:
                    inter_area = cell_geom.intersection(poly_row.geometry).area
                except Exception:
                    inter_area = 0.0
                gk = poly_row[class_col]
                # If this class covers more area, or equal area but better rank, use it
                if inter_area > best_area or (
                    inter_area == best_area and
                    class_rank.get(gk, 99) < class_rank.get(best_class, 99)
                ):
                    best_area = inter_area
                    best_class = gk

            # Only assign if at least some area is covered
            if best_area > 0:
                results[idx] = best_class
        return results

    cells_gdf['comp_class'] = _classify_by_area_majority(
        cells_gdf, computed[['geometry', 'gk_class']], 'gk_class')
    cells_gdf['off_class'] = _classify_by_area_majority(
        cells_gdf, official[['geometry', 'KLASSE']], 'KLASSE')

    # Classify diff: improved (green), worsened (red), same (light grey), neither (dark grey)
    def _classify_diff(comp, off):
        comp_rank = class_rank.get(comp, 4)
        off_rank  = class_rank.get(off, 4)
        if comp_rank == 4 and off_rank == 4:
            return 'neither'
        elif comp_rank < off_rank:
            return 'improved'
        elif comp_rank > off_rank:
            return 'worsened'
        else:
            return 'same'

    cells_gdf['diff_class'] = [
        _classify_diff(c, o)
        for c, o in zip(cells_gdf['comp_class'], cells_gdf['off_class'])
    ]

    # Diff colour scheme
    diff_colours = {
        'improved':  '#2ca02c',  # green
        'worsened':  '#d62728',  # red
        'same':      '#d9d9d9',  # light grey
        'neither':   '#808080',  # dark grey
    }

    # Compute diff statistics (pop and FTE)
    pop_map  = pop_grid.set_index('RELI')['NUMMER'].to_dict()
    empl_map = empl_grid.set_index('RELI')['NUMMER'].to_dict()
    cells_gdf['_pop']  = cells_gdf['RELI'].map(pop_map).fillna(0)
    cells_gdf['_empl'] = cells_gdf['RELI'].map(empl_map).fillna(0)
    total_diff_pop  = cells_gdf['_pop'].sum()
    total_diff_empl = cells_gdf['_empl'].sum()

    diff_stats = {}
    for dc in ('improved', 'worsened', 'same', 'neither'):
        sub = cells_gdf[cells_gdf['diff_class'] == dc]
        p  = int(sub['_pop'].sum())
        e  = int(sub['_empl'].sum())
        pp = p / total_diff_pop  * 100 if total_diff_pop  > 0 else 0.0
        ep = e / total_diff_empl * 100 if total_diff_empl > 0 else 0.0
        diff_stats[dc] = (p, pp, e, ep)

    # --- Create 3-panel figure ---
    fig, axes = plt.subplots(1, 3, figsize=(28, 12), sharey=True)
    fig.subplots_adjust(top=0.93, bottom=0.24, wspace=0.08)
    bx0, by0, bx1, by1 = boundary.bounds
    pad = 500
    boundary_gdf = gpd.GeoDataFrame(geometry=[boundary], crs=CODEBASE_CRS)

    # Panel 1 & 2: Computed and Official
    panel_data = [
        (axes[0], computed, 'gk_class', 'Computed (infraScanRail)'),
        (axes[1], official, 'KLASSE',   'Official ARE (2026)'),
    ]
    for ax, gdf, col, title in panel_data:
        boundary_gdf.plot(ax=ax, color='#F5F5F5', edgecolor='none', zorder=0)
        for gk in ('D', 'C', 'B', 'A'):   # D first → A on top
            sub = gdf[gdf[col] == gk]
            if not sub.empty:
                sub.plot(ax=ax, color=GK_PLOT_COLOURS[gk],
                         edgecolor='none', alpha=0.80, zorder=1)
        if os.path.exists(paths.LAKES_SHP):
            lakes = gpd.read_file(paths.LAKES_SHP).to_crs(CODEBASE_CRS)
            lakes = lakes[lakes.geometry.intersects(boundary)]
            if not lakes.empty:
                gpd.clip(lakes, boundary).plot(
                    ax=ax, color='#A8D4F0', edgecolor='none', zorder=2)
        boundary_gdf.boundary.plot(
            ax=ax, color='black', linewidth=1.8, linestyle='--', zorder=3)
        ax.set_xlim(bx0 - pad, bx1 + pad)
        ax.set_ylim(by0 - pad, by1 + pad)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=13)
        ax.set_xlabel('E [m]')
        ax.set_ylabel('N [m]')
        # Add cartographic elements
        _add_map_elements(ax)

    # Panel 3: Diff (improvement / worsening)
    ax3 = axes[2]
    boundary_gdf.plot(ax=ax3, color='#F5F5F5', edgecolor='none', zorder=0)

    # Plot diff cells by category
    for dc, colour in diff_colours.items():
        sub = cells_gdf[cells_gdf['diff_class'] == dc]
        if not sub.empty:
            sub.plot(ax=ax3, color=colour, edgecolor='none', alpha=0.85, zorder=1)

    if os.path.exists(paths.LAKES_SHP):
        lakes = gpd.read_file(paths.LAKES_SHP).to_crs(CODEBASE_CRS)
        lakes = lakes[lakes.geometry.intersects(boundary)]
        if not lakes.empty:
            gpd.clip(lakes, boundary).plot(
                ax=ax3, color='#A8D4F0', edgecolor='none', zorder=2)
    boundary_gdf.boundary.plot(
        ax=ax3, color='black', linewidth=1.8, linestyle='--', zorder=3)
    ax3.set_xlim(bx0 - pad, bx1 + pad)
    ax3.set_ylim(by0 - pad, by1 + pad)
    ax3.set_aspect('equal')
    ax3.set_title('Difference (Computed vs. Official)', fontsize=13)
    ax3.set_xlabel('E [m]')

    # Add cartographic elements to panel 3
    _add_map_elements(ax3)

    # Legend for panels 1 & 2
    legend_handles_gk = [
        Patch(facecolor=GK_PLOT_COLOURS[gk], edgecolor='none',
              label=f'Class {gk}')
        for gk in ('A', 'B', 'C', 'D')
    ]
    legend_handles_gk.append(
        Line2D([0], [0], color='black', linewidth=1.8,
               linestyle='--', label='Study area boundary'))
    axes[1].legend(handles=legend_handles_gk, loc='upper right',
                   fontsize=8, framealpha=0.9)

    # Legend for panel 3 (diff)
    legend_handles_diff = [
        Patch(facecolor=diff_colours['improved'], edgecolor='none', label='Improved'),
        Patch(facecolor=diff_colours['worsened'], edgecolor='none', label='Worsened'),
        Patch(facecolor=diff_colours['same'], edgecolor='none', label='Same class'),
        Patch(facecolor=diff_colours['neither'], edgecolor='none', label='Neither classified'),
        Line2D([0], [0], color='black', linewidth=1.8,
               linestyle='--', label='Study area boundary'),
    ]
    ax3.legend(handles=legend_handles_diff, loc='upper right',
               fontsize=8, framealpha=0.9)

    # --- Summary table (% of inhabited area + pop% + FTE% + diff stats) ---
    col_w = 8
    hdr = (f"{'':9}"
           f"{'— Computed —':>{3*col_w}}"
           f"  {'— Official —':>{3*col_w}}\n")
    hdr += (f"{'Class':<9}"
            f"{'Area%':>{col_w}}{'Pop%':>{col_w}}{'FTE%':>{col_w}}"
            f"  {'Area%':>{col_w}}{'Pop%':>{col_w}}{'FTE%':>{col_w}}\n")
    hdr += "─" * (9 + 3*col_w + 2 + 3*col_w) + "\n"

    tbl = f"Coverage (% of inhabited area)\n{hdr}"
    for gk in ('A', 'B', 'C', 'D', 'No class'):
        label = f'Class {gk}' if gk != 'No class' else 'No class'
        tbl += (f"{label:<9}"
                f"{comp_pct[gk]:>{col_w-1}.1f}%"
                f"{comp_pop_pct[gk]:>{col_w-1}.1f}%"
                f"{comp_empl_pct[gk]:>{col_w-1}.1f}%"
                f"  "
                f"{off_pct[gk]:>{col_w-1}.1f}%"
                f"{off_pop_pct[gk]:>{col_w-1}.1f}%"
                f"{off_empl_pct[gk]:>{col_w-1}.1f}%\n")

    # Add diff statistics
    tbl += "\n"
    tbl += f"{'Diff':9}{'Pop':>{col_w}}{'Pop%':>{col_w}}{'FTE':>{col_w}}{'FTE%':>{col_w}}\n"
    tbl += "─" * (9 + 4*col_w) + "\n"
    for dc, label in [('improved', 'Improved'), ('worsened', 'Worsened'),
                      ('same', 'Same'), ('neither', 'Neither')]:
        p, pp, e, ep = diff_stats[dc]
        tbl += f"{label:<9}{p:>{col_w},}{pp:>{col_w-1}.1f}%{e:>{col_w},}{ep:>{col_w-1}.1f}%\n"

    fig.text(0.5, 0.02, tbl, ha='center', fontsize=7.5, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))
    fig.suptitle('ÖV-Güteklassen: Computed vs. Official ARE (2026)',
                 fontsize=15, y=0.97)

    out_path = os.path.join(CATCHMENT_PLOT_DIR, 'gueteklassen_comparison.pdf')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"    Saved -> {out_path}")


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
    feeder_stops    = _load_feeder_stops(boundary, temporal)
    rail_stations   = _load_rail_stations(boundary, temporal)
    feeder_segments = _load_feeder_segments(temporal)

    # Güteklassen classification → assigns buffer_radius_m per stop
    feeder_stops, rail_stations = _compute_stop_gueteklassen(
        feeder_stops, rail_stations, temporal)
    _build_gueteklassen_gpkgs(feeder_stops, rail_stations, boundary, pop_grid, empl_grid)
    _plot_gueteklassen_comparison(feeder_stops, rail_stations, boundary, pop_grid, empl_grid)

    _build_pt_buffers(feeder_stops, rail_stations, boundary, pop_grid, empl_grid)

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
                                pop_grid, rail_stations)
    _build_candidates_csv(walk_pop, cycle_pop, feeder_pop,
                          pop_grid, empl_grid, rail_stations, PT_FEEDER_DATA_DIR)

    print(f"  [Pop allocation + candidates complete: {time.time() - st:.1f}s]")
    st = time.time()

    # Step 4b: Allocate employment-only cells (RELIs in empl_grid but not in pop_grid).
    # These cells are absent from alloc_pop; running the same allocation for them
    # ensures the diff plot can compare all inhabited cells, not just pop cells.
    pop_relis_set = set(pop_grid['RELI'].values)
    empl_only_grid = empl_grid[~empl_grid['RELI'].isin(pop_relis_set)].copy()
    if not empl_only_grid.empty:
        print(f"\n  Allocating {len(empl_only_grid):,} employment-only cells ...")
        walk_empl   = _compute_walk_to_rail_times(empl_only_grid, rail_stations)
        cycle_empl  = _compute_cycle_to_rail_times(empl_only_grid, rail_stations)
        feeder_empl = _compute_feeder_to_rail_times(empl_only_grid, feeder_stops,
                                                     feeder_stop_to_rail_times)
        alloc_empl = _allocate_cells(walk_empl, cycle_empl, feeder_empl,
                                     empl_only_grid, rail_stations)
        alloc_combined = pd.concat([alloc_pop, alloc_empl], ignore_index=True)
        print(f"  [Empl-only allocation complete: {time.time() - st:.1f}s]")
    else:
        alloc_combined = alloc_pop.copy()
        print("  No employment-only cells found — combined allocation equals pop allocation")
    st = time.time()

    # Outputs - produced from population allocation (primary)
    pt_catchment = _build_catchment_gpkg(
        alloc_pop, pop_grid, empl_grid, rail_stations, PT_FEEDER_DATA_DIR)
    _build_visualisation(alloc_pop, pop_grid, rail_stations, boundary, 'PT-Feeder',
                         empl_grid,
                         walk_df=walk_pop, cycle_df=cycle_pop, feeder_df=feeder_pop,
                         catchment_gdf=pt_catchment)
    _plot_access_times(walk_pop, cycle_pop, feeder_pop, alloc_pop, pop_grid,
                       rail_stations, feeder_stops, boundary, empl_grid=empl_grid)

    print(f"  [Outputs complete: {time.time() - st:.1f}s]")

    return pt_catchment, alloc_combined


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
            os.path.join(PT_FEEDER_DATA_DIR, 'catchment.gpkg'),
            os.path.join(MUNICIPAL_DATA_DIR, 'catchment.gpkg'),
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
