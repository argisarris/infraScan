"""
Network Builder Module

Two-phase interactive pipeline:

  Phase 1 (Build Base) — optional
    Loads the raw filtered network from data/Infrastructure/Raw/,
    applies macroscopic simplification, and writes the selectable
    Base version to data/Infrastructure/Base/.
    Raw/ is produced by infrabuild_filter_network.py.

  Phase 2 (Load & Analyse)
    Selects a prepared version from data/Infrastructure/<version>/,
    builds a NetworkX graph, and optionally generates infrastructure plots.
    Named versions (beyond Base) are produced by infrabuild_version_manager.py.

Usage (interactive):
    python infrabuild_network_builder.py
"""

import os
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter, deque
from shapely.ops import linemerge, substring as shp_substring
from shapely.geometry import MultiLineString, LineString
import math
import sys
import uuid
import zipfile
from matplotlib.patches import Rectangle
from matplotlib_map_utils.core.north_arrow import NorthArrow, north_arrow

sys.path.insert(0, str(Path(__file__).parent))
import paths


# =============================================================================
# Color Schemes
# =============================================================================

GAUGE_COLORS = {
    1435: '#000000',  # black           – Standard gauge
    1668: '#B8860B',  # dark goldenrod  – Iberian gauge
    1520: '#8B0000',  # dark red        – Russian/broad gauge
    1000: '#7B00D4',  # violet          – Metre gauge
    1067: "#0011FF",  # dark blue       - Cape gauge
    1676: "#D6F814",  # orange-red
    1600: '#FF6347',  # tomato
    600:  '#FF69B4',  # hot pink
}
GAUGE_DEFAULT = '#808080'

# Category-based electrification colours (keyed by canonical class).
# Raw BAV Elektrifizierung strings are classified via _classify_electrification().
ELECTRIFICATION_COLORS = {
    'no_electrification': '#000000',  # black       – nicht elektrifiziert
    'dc':                 "#046DAA",  # light blue  – Gleichstrom
    'ac_16_7hz':          '#2ca02c',  # green       – Wechselstrom 15 kV / 16.7 Hz
    'ac_25kv':            '#d62728',  # red         – Wechselstrom 25 kV
    'unknown':            '#7f7f7f',  # grey
}
ELECTRIFICATION_LABELS = {
    'no_electrification': 'No electrification',
    'dc':                 'DC (Gleichstrom)',
    'ac_16_7hz':          'AC 15 kV / 16.7 Hz',
    'ac_25kv':            'AC 25 kV / 50 Hz',
    'unknown':            'Unknown',
}
ELECTRIFICATION_DEFAULT = '#7f7f7f'


def _classify_electrification(val) -> str:
    """Map a raw BAV Elektrifizierung string to a canonical electrification class."""
    if pd.isna(val):
        return 'unknown'
    s = (str(val).lower()
         .replace('_', '').replace('-', '').replace(' ', '').replace(',', '.'))
    if 'nicht' in s or s in ('na', 'nan', '', 'none', 'unknown'):
        return 'no_electrification'
    if 'gleichstrom' in s or (s.startswith('dc') and 'dec' not in s):
        return 'dc'
    if '16.7' in s or '167' in s or '15kv' in s or '15000' in s:
        return 'ac_16_7hz'
    if '25kv' in s or '25000' in s:
        return 'ac_25kv'
    if 'wechselstrom' in s or s.startswith('ac'):
        return 'ac_16_7hz'  # Swiss default AC is 15 kV / 16.7 Hz
    return 'unknown'

# Speed colour bins (upper bound km/h → colour).  Ordered from slow to fast.
# Segments with no speed data (NaN) use SPEED_NO_DATA_COLOR.
SPEED_BINS: list = [
    (30,  '#0000AA', '≤ 30 km/h'),
    (60,  '#0066FF', '31–60 km/h'),
    (80,  '#00AA44', '61–80 km/h'),
    (100, '#AACC00', '81–100 km/h'),
    (120, '#FFAA00', '101–120 km/h'),
    (160, '#FF4400', '121–160 km/h'),
    (999, '#CC0000', '161+ km/h'),
]
SPEED_NO_DATA_COLOR = '#888888'
SPEED_NO_DATA_LABEL = 'No data'


TRACK_COLORS = {
    1: '#FF0000',
    2: '#0000FF',
    3: '#00FF00',
    4: '#800080',
}
TRACK_DEFAULT = '#FFA500'

CONSTRUCT_COLORS = {
    'tunnel':  '#8B4513',
    'bridge':  '#4169E1',
    'gallery': '#708090',
    'normal':  '#2E8B57',
}
CONSTRUCT_DEFAULT = '#808080'

# High-contrast palette for owner map — 30 perceptually distinct colours.
# SBB always gets '#E3000F'; remaining owners cycle through this list.
_OWNER_PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b',
    '#e377c2', '#17becf', '#bcbd22', '#393b79', '#637939',
    '#8c6d31', '#843c39', '#7b4173', '#5254a3', '#6b6ecf',
    '#b5cf6b', '#cedb9c', '#e7969c', '#de9ed6', '#3182bd',
    '#31a354', '#756bb1', '#636363', '#fd8d3c', '#fdae6b',
    '#c7e9c0', '#dadaeb', '#fee6ce', '#deebf7', '#d9d9d9',
]


# =============================================================================
# OSM Maxspeed Enrichment Constants
# =============================================================================

OSM_BUFFER_M         = 10     # buffer radius (m) around each BAV segment
OSM_MAX_BEARING_DIFF = 45     # degrees — max angular divergence to accept
OSM_NODE_SNAP_M      = 15     # BAV node snap radius for conflict resolution
OSM_COVERAGE_MIN     = 0.10   # min coverage fraction to record speed

# BAV gauge (mm) → compatible OSM railway types.
# Segments with gauge not in this dict (or gauge = None) accept all types.
GAUGE_TO_OSM_TYPES: Dict[int, set] = {
    1435: {"rail", "light_rail", "subway"},
    1000: {"tram", "narrow_gauge", "light_rail"},
    900:  {"narrow_gauge", "funicular"},
    800:  {"narrow_gauge", "funicular"},
    750:  {"narrow_gauge", "funicular"},
    600:  {"narrow_gauge", "funicular"},
    1520: {"rail"},
    1668: {"rail"},
}
_ALL_OSM_TYPES = {t for s in GAUGE_TO_OSM_TYPES.values() for t in s}

# Mode-to-OSM-type mapping used after transport_mode propagation.
# Tighter than GAUGE_TO_OSM_TYPES: distinguishes 1000 mm trains from trams.
# Gauge is kept as fallback only for NaN-mode segments.
MODE_TO_OSM_TYPES: Dict[str, set] = {
    "train":       {"rail"},
    "tram":        {"tram", "light_rail"},
    "funicular":   {"funicular"},
    "cog_railway": {"narrow_gauge"},
}
_OSM_COLS = [
    "osm_id", "railway_type", "maxspeed",
    "maxspeed_forward", "maxspeed_backward", "osm_name",
]


NODE_COLORS = {
    'station':           '#FF0000',
    'abandoned_station': '#888888',
    'junction':          '#0000FF',
}
NODE_DEFAULT = '#AAAAAA'


# =============================================================================
# Data Class
# =============================================================================

@dataclass
class NetworkData:
    """Container for a built infrastructure version."""
    nodes:    gpd.GeoDataFrame
    segments: gpd.GeoDataFrame
    graph:    nx.Graph
    version:  str
    boundary: Optional[gpd.GeoDataFrame] = field(default=None)

    def __repr__(self):
        return (
            f"NetworkData(version='{self.version}': "
            f"{len(self.nodes)} nodes, {len(self.segments)} segments, "
            f"{self.graph.number_of_edges()} edges)"
        )


# =============================================================================
# Version Discovery
# =============================================================================

def list_versions(infra_dir: Optional[str] = None) -> List[str]:
    """
    Scan data/Infrastructure/ for selectable network versions.

    A subfolder qualifies when it contains both nodes.gpkg and segments.gpkg.
    Raw/ is excluded (intermediate artifact, not a selectable network state).

    Returns version names sorted alphabetically, with 'Base' first when present.
    """
    root = Path(infra_dir or (Path(paths.MAIN) / paths.NETWORK_INFRASTRUCTURE_DIR))
    if not root.exists():
        return []

    versions = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        if sub.name == 'Raw':
            continue
        if (sub / 'nodes.gpkg').exists() and (sub / 'segments.gpkg').exists():
            versions.append(sub.name)

    # Ensure 'Base' sorts first for readability
    if 'Base' in versions:
        versions = ['Base'] + [v for v in versions if v != 'Base']

    return versions


# =============================================================================
# Loading
# =============================================================================

def load_version(
    version: str,
    infra_dir: Optional[str] = None,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load nodes.gpkg and segments.gpkg for a named version."""
    root = Path(infra_dir or (Path(paths.MAIN) / paths.NETWORK_INFRASTRUCTURE_DIR))
    folder = root / version

    nodes_path = folder / "nodes.gpkg"
    segs_path  = folder / "segments.gpkg"

    if not nodes_path.exists():
        raise FileNotFoundError(f"nodes.gpkg not found in {folder}")
    if not segs_path.exists():
        raise FileNotFoundError(f"segments.gpkg not found in {folder}")

    nodes    = gpd.read_file(nodes_path)
    segments = gpd.read_file(segs_path)
    print(f"Loaded '{version}': {len(nodes)} nodes, {len(segments)} segments")
    return nodes, segments


# =============================================================================
# Macroscopic Simplification
# =============================================================================



def _sub_lines(geom):
    """Return list of LineString parts from a LineString or MultiLineString."""
    if geom.geom_type == 'MultiLineString':
        return list(geom.geoms)
    return [geom]


def _reverse_line(geom):
    """Reverse coordinate order of a LineString."""
    return LineString(list(geom.coords)[::-1])


def _build_merged_segment(s1: pd.Series, s2: pd.Series, node_name: str) -> Optional[dict]:
    """
    Merge two segments sharing node_name into a single segment A→B.
    Returns None when the merge would create a self-loop (A == B).
    """
    if s1['from_name'] == node_name:
        A = s1['to_name']
        A_N, A_E = s1['to_N'], s1['to_E']
        lines1 = [_reverse_line(g) for g in reversed(_sub_lines(s1.geometry))]
    else:
        A = s1['from_name']
        A_N, A_E = s1['from_N'], s1['from_E']
        lines1 = _sub_lines(s1.geometry)

    if s2['to_name'] == node_name:
        B = s2['from_name']
        B_N, B_E = s2['from_N'], s2['from_E']
        lines2 = [_reverse_line(g) for g in reversed(_sub_lines(s2.geometry))]
    else:
        B = s2['to_name']
        B_N, B_E = s2['to_N'], s2['to_E']
        lines2 = _sub_lines(s2.geometry)

    if A == B:
        return None

    merged_geom = linemerge(MultiLineString(lines1 + lines2))

    return {
        'segment_id':      f"{s1['segment_id']}+{s2['segment_id']}",
        'segment_name':    pd.NA,
        'from_name':       A,
        'to_name':         B,
        'from_N':          A_N,
        'from_E':          A_E,
        'to_N':            B_N,
        'to_E':            B_E,
        'length_m':        s1['length_m'] + s2['length_m'],
        'num_tracks':      s1['num_tracks'],
        'gauge':           s1['gauge'],
        'electrification': s1['electrification'],
        'km_start':        s1.get('km_start', pd.NA),
        'km_end':          s2.get('km_end', pd.NA),
        'route_number':    s1.get('route_number', pd.NA),
        'route_name':      s1.get('route_name', pd.NA),
        'route_owner':     s1.get('route_owner', pd.NA),
        'geometry':        merged_geom,
    }


def filter_macroscopic_nodes(
    nodes: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Simplify raw filtered network to macroscopic representation.

    Removal candidates
    ------------------
    freight_yard, operational_yard, depot, km_change  – always removed
    switch                                             – removed if degree < 3

    Degree rules
    ------------
    degree 0–1 : node + dangling segment dropped.
    degree 2   : flanking segments merged when num_tracks + gauge +
                 electrification all match; otherwise kept as pass-through.
    degree >= 3: kept (topological junction).
    """
    print("Applying macroscopic simplification...")

    segs_df = segments.copy().reset_index(drop=True)

    degree: Counter = Counter()
    for fn, tn in zip(segs_df['from_name'], segs_df['to_name']):
        degree[fn] += 1
        degree[tn] += 1

    _MACRO_KEEP = {'station', 'junction', 'abandoned_station'}

    def _is_candidate(row) -> bool:
        nc   = str(row.get('node_class', ''))
        name = row.get('NAME')
        if pd.isna(name):
            return False
        if nc not in _MACRO_KEEP:
            return True
        return False

    candidate_names: set = set(
        nodes.loc[nodes.apply(_is_candidate, axis=1), 'NAME'].dropna()
    )
    print(f"  Removal candidates: {len(candidate_names)}")

    drop_names:   set  = set()
    segs_to_add:  list = []
    segs_to_drop: set  = set()
    n_merged = 0
    n_passthrough = 0

    for name in candidate_names:
        deg = degree[name]
        
        # Drop nodes by default if they are candidates
        drop_names.add(name)

        if deg <= 1:
            dangling = segs_df[
                (segs_df['from_name'] == name) | (segs_df['to_name'] == name)
            ]
            segs_to_drop.update(dangling['segment_id'])
            continue

        if deg >= 3:
            n_passthrough += 1
            continue

        connected = segs_df[
            (segs_df['from_name'] == name) | (segs_df['to_name'] == name)
        ]
        available = connected[~connected['segment_id'].isin(segs_to_drop)]
        if len(available) != 2:
            n_passthrough += 1
            continue

        s1, s2 = available.iloc[0], available.iloc[1]
        attrs_match = (
            s1['num_tracks']      == s2['num_tracks'] and
            s1['gauge']           == s2['gauge'] and
            s1['electrification'] == s2['electrification']
        )

        if attrs_match:
            merged = _build_merged_segment(s1, s2, name)
            if merged is None:
                n_passthrough += 1
                continue
            segs_to_add.append(merged)
            segs_to_drop.update([s1['segment_id'], s2['segment_id']])
            n_merged += 1
        else:
            n_passthrough += 1

    print(f"  Merged: {n_merged}  |  Pass-through: {n_passthrough}  "
          f"|  Dead-end dropped: {len(drop_names) - n_merged}")

    macro_nodes = (
        nodes[~nodes['NAME'].isin(drop_names)].copy().reset_index(drop=True)
    )
    remaining = segs_df[~segs_df['segment_id'].isin(segs_to_drop)].copy()
    if segs_to_add:
        added = gpd.GeoDataFrame(segs_to_add, crs=segments.crs)
        macro_segs = pd.concat([remaining, added], ignore_index=True)
    else:
        macro_segs = remaining

    # Build segment ID remap: raw_segment_id → final_macro_segment_id.
    # Used by run_build_base to update segment references in segments_composition.
    seg_id_remap: dict = {}
    for sid in remaining['segment_id'].dropna():
        seg_id_remap[str(sid)] = str(sid)
    for merged_row in segs_to_add:
        merged_id = str(merged_row['segment_id'])
        for part in merged_id.split('+'):
            seg_id_remap[part] = merged_id

    print(f"  → {len(macro_nodes)} nodes, {len(macro_segs)} segments")
    return macro_nodes, macro_segs.reset_index(drop=True), seg_id_remap


# =============================================================================
# OSM Maxspeed Enrichment
# =============================================================================

def _parse_speed(val) -> float:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    try:
        return float(str(val).replace("km/h", "").replace("mph", "").strip())
    except ValueError:
        return np.nan


def _best_speed(row) -> float:
    for col in ("maxspeed", "maxspeed_forward", "maxspeed_backward"):
        v = _parse_speed(row.get(col))
        if not np.isnan(v):
            return v
    return np.nan


def _compute_osm_bearing(geom) -> float:
    """Overall orientation [0, 180) using start-to-end direction."""
    if geom is None or geom.is_empty:
        return np.nan
    if geom.geom_type in ("MultiLineString", "GeometryCollection"):
        lines = [g for g in geom.geoms if g.geom_type == "LineString" and not g.is_empty]
        if not lines:
            return np.nan
        merged = linemerge(lines)
        coords = (
            [c for g in merged.geoms for c in g.coords]
            if merged.geom_type == "MultiLineString"
            else list(merged.coords)
        )
    elif geom.geom_type == "LineString":
        coords = list(geom.coords)
    else:
        return np.nan
    if len(coords) < 2:
        return np.nan
    dx = coords[-1][0] - coords[0][0]
    dy = coords[-1][1] - coords[0][1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return np.nan
    return np.degrees(np.arctan2(dy, dx)) % 180


def _angular_diff_osm(b1: float, b2: float) -> float:
    if np.isnan(b1) or np.isnan(b2):
        return np.nan
    diff = abs(b1 - b2) % 180
    return min(diff, 180 - diff)


def _resolve_osm_conflicts_at_nodes(
    pairs: gpd.GeoDataFrame,
    bav_nodes: gpd.GeoDataFrame,
    buf_lookup: dict,
) -> gpd.GeoDataFrame:
    """
    Split OSM ways that appear in multiple BAV buffers at BAV node positions,
    then reassign each piece to the buffer whose midpoint it falls in.
    Unresolvable conflicts (no nearby nodes) are kept as-is.
    """
    conflict_ids = set(pairs[pairs.duplicated("osm_id", keep=False)]["osm_id"].unique())
    if not conflict_ids:
        return pairs

    print(f"  Resolving {len(conflict_ids)} OSM ways spanning multiple buffers...")

    non_conflict  = pairs[~pairs["osm_id"].isin(conflict_ids)].copy()
    conflict_rows = pairs[pairs["osm_id"].isin(conflict_ids)].copy()

    osm_unique = (
        conflict_rows[["osm_id", "geometry"]]
        .drop_duplicates("osm_id")
        .set_index("osm_id")
    )

    resolved: List[dict] = []
    kept_as_is: List[pd.DataFrame] = []

    for osm_id, osm_row in osm_unique.iterrows():
        osm_geom = osm_row["geometry"]
        if osm_geom.geom_type != "LineString":
            kept_as_is.append(conflict_rows[conflict_rows["osm_id"] == osm_id])
            continue

        nearby = bav_nodes[bav_nodes.distance(osm_geom) <= OSM_NODE_SNAP_M]
        total_len = osm_geom.length
        split_dists = sorted({
            d for pt in nearby.geometry
            if 1.0 < (d := osm_geom.project(pt)) < total_len - 1.0
        })

        if not split_dists:
            kept_as_is.append(conflict_rows[conflict_rows["osm_id"] == osm_id])
            continue

        template = conflict_rows[conflict_rows["osm_id"] == osm_id].iloc[0]
        osm_attrs = {col: template[col] for col in _OSM_COLS if col in template.index}
        matched_segs = conflict_rows[conflict_rows["osm_id"] == osm_id]["segment_id"].tolist()

        for start, end in zip([0.0] + split_dists, split_dists + [total_len]):
            piece = shp_substring(osm_geom, start, end)
            if piece.is_empty or piece.length < 0.5:
                continue
            midpoint = piece.interpolate(0.5, normalized=True)
            assigned = next(
                (sid for sid in matched_segs if buf_lookup[sid].contains(midpoint)),
                None,
            )
            if assigned is None:
                continue
            resolved.append({**osm_attrs, "segment_id": assigned, "geometry": piece})

    parts = [non_conflict]
    if resolved:
        parts.append(gpd.GeoDataFrame(resolved, crs=pairs.crs))
    parts.extend(kept_as_is)

    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=pairs.crs)


def _empty_osm_result(bav: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = bav.copy()
    out["average_speed"]                 = np.nan
    out["predominant_speed"]             = np.nan
    out["predominant_speed_coverage_pct"] = 0.0
    return out


def _join_osm_speeds(
    bav: gpd.GeoDataFrame,
    bav_nodes: gpd.GeoDataFrame,
    osm: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Overlay OSM railway ways onto BAV segments and aggregate speed data.

    Returns bav with three new columns:
      average_speed                — length-weighted mean speed (km/h) or NaN
      predominant_speed            — modal speed by length (km/h) or NaN
      predominant_speed_coverage_pct — fraction of segment length covered by
                                       the predominant speed value
    """
    buf_series  = bav.geometry.buffer(OSM_BUFFER_M)
    bav_buf     = gpd.GeoDataFrame(bav[["segment_id"]].copy(), geometry=buf_series, crs=bav.crs)
    buf_lookup  = bav_buf.set_index("segment_id")["geometry"]
    bav_lengths = bav.set_index("segment_id")["length_m"]
    bav_bearings = bav.set_index("segment_id")["geometry"].apply(_compute_osm_bearing)
    seg_gauge   = bav.set_index("segment_id")["gauge"].to_dict()

    osm_join = osm[_OSM_COLS + ["geometry"]].copy()
    pairs = gpd.sjoin(osm_join, bav_buf, how="inner", predicate="intersects")
    pairs = pairs.drop(columns=["index_right"], errors="ignore").reset_index(drop=True)
    print(f"  {len(pairs)} candidate (buffer, OSM way) pairs")

    if pairs.empty:
        return _empty_osm_result(bav)

    pairs = _resolve_osm_conflicts_at_nodes(pairs, bav_nodes, buf_lookup)

    pairs["clipped_geom"] = pairs.apply(
        lambda r: r.geometry.intersection(buf_lookup[r["segment_id"]]), axis=1
    )
    is_line = pairs["clipped_geom"].geom_type.isin(["LineString", "MultiLineString"])
    pairs = pairs[is_line].copy()
    pairs["clipped_length_m"] = pairs["clipped_geom"].length
    pairs = pairs[pairs["clipped_length_m"] > 0].copy()

    pairs["osm_bearing"] = pairs["clipped_geom"].apply(_compute_osm_bearing)
    pairs["bav_bearing"] = pairs["segment_id"].map(bav_bearings)
    pairs["bearing_diff"] = pairs.apply(
        lambda r: _angular_diff_osm(r["osm_bearing"], r["bav_bearing"]), axis=1
    )
    n_before = len(pairs)
    pairs = pairs[
        pairs["bearing_diff"].isna() | (pairs["bearing_diff"] <= OSM_MAX_BEARING_DIFF)
    ].copy()
    print(f"  {n_before - len(pairs)} pairs removed by bearing filter")

    seg_mode = (
        bav.set_index("segment_id")["transport_mode"].to_dict()
        if "transport_mode" in bav.columns else {}
    )
    pairs["bav_mode"]  = pairs["segment_id"].map(seg_mode)
    pairs["bav_gauge"] = pairs["segment_id"].map(seg_gauge)

    def _allowed_osm_types(mode, gauge) -> set:
        """Return the set of OSM railway types compatible with this segment."""
        if pd.notna(mode) and str(mode).strip():
            types: set = set()
            for m in str(mode).split('/'):
                types |= MODE_TO_OSM_TYPES.get(m.strip(), set())
            if types:
                return types
        return GAUGE_TO_OSM_TYPES.get(gauge, _ALL_OSM_TYPES)

    compatible_mask = pairs.apply(
        lambda r: r["railway_type"] in _allowed_osm_types(r["bav_mode"], r["bav_gauge"]),
        axis=1,
    )
    n_before = len(pairs)
    pairs = pairs[compatible_mask].drop(columns=["bav_mode", "bav_gauge"]).copy()
    print(f"  {n_before - len(pairs)} pairs removed by railway-type filter ({len(pairs)} remain)")

    pairs["speed_kmh"] = pairs.apply(_best_speed, axis=1)

    def _aggregate(g):
        seg_len = bav_lengths.get(g.name, np.nan)
        valid   = g[g["speed_kmh"].notna()]
        if valid.empty:
            return pd.Series({
                "average_speed":                  np.nan,
                "predominant_speed":              np.nan,
                "predominant_speed_coverage_pct": 0.0,
            })

        avg = (
            (valid["speed_kmh"] * valid["clipped_length_m"]).sum()
            / valid["clipped_length_m"].sum()
        )
        spd_lengths = valid.groupby("speed_kmh")["clipped_length_m"].sum()
        predominant = spd_lengths.idxmax()
        pct = (
            float(spd_lengths[predominant] / seg_len)
            if seg_len and seg_len > 0
            else 0.0
        )
        return pd.Series({
            "average_speed":                  round(avg, 1),
            "predominant_speed":              predominant,
            "predominant_speed_coverage_pct": round(min(pct, 1.0), 3),
        })

    if pairs.empty:
        return _empty_osm_result(bav)

    agg = pairs.groupby("segment_id").apply(_aggregate, include_groups=False)

    _SPEED_COLS = ("average_speed", "predominant_speed", "predominant_speed_coverage_pct")
    bav_base = bav.drop(columns=[c for c in _SPEED_COLS if c in bav.columns])
    result = bav_base.merge(agg.reset_index(), on="segment_id", how="left")
    result["predominant_speed_coverage_pct"] = (
        result["predominant_speed_coverage_pct"].fillna(0.0)
    )
    return result


def enrich_segments_with_osm_speed(
    segments: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    raw_dir: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """
    Load OSM maxspeed data fetched by infrabuild_filter_network.py and attach
    speed columns to segments.

    Reads osm_maxspeed_segments.gpkg from data/Infrastructure/Raw/ (written by
    infrabuild_filter_network.run_filter_network).  If the file is absent the
    function returns segments unchanged with NaN speed columns and prints a
    warning — re-run infrabuild_filter_network.py to populate it.

    Adds three columns to the returned GeoDataFrame:
      average_speed                — length-weighted mean OSM speed (km/h) or NaN
      predominant_speed            — modal speed by segment length (km/h) or NaN
      predominant_speed_coverage_pct — fraction of segment under predominant speed

    Args:
        segments: BAV segment GeoDataFrame (must have segment_id, length_m, gauge, geometry)
        nodes:    BAV node GeoDataFrame (used for conflict resolution at junctions)
        raw_dir:  Override for Raw/ directory path (defaults to paths.NETWORK_INFRASTRUCTURE_RAW)

    Returns:
        segments GeoDataFrame with speed columns added.
    """
    osm_path = (
        Path(raw_dir) if raw_dir
        else Path(paths.MAIN) / paths.NETWORK_INFRASTRUCTURE_RAW
    ) / "osm_maxspeed_segments.gpkg"

    if not osm_path.exists():
        print(
            f"  WARNING: {osm_path.name} not found in Raw/.\n"
            "  Run infrabuild_filter_network.py to fetch OSM data.\n"
            "  Speed columns will be NaN for this build."
        )
        return _empty_osm_result(segments)

    osm = gpd.read_file(osm_path)
    if osm.crs is None or osm.crs.to_epsg() != 2056:
        osm = osm.to_crs("EPSG:2056")
    print(f"  Loaded {len(osm)} OSM ways from {osm_path.name}")

    if osm.empty:
        print("  WARNING: OSM file is empty — speed columns will be NaN.")
        return _empty_osm_result(segments)

    joined = _join_osm_speeds(segments, nodes, osm)

    # When 100% of the segment runs at the predominant speed, the weighted
    # average is identical — use the exact value to avoid float rounding drift.
    full_cov = joined["predominant_speed_coverage_pct"] == 1.0
    joined.loc[full_cov, "average_speed"] = joined.loc[full_cov, "predominant_speed"]

    total     = len(joined)
    has_speed = joined["average_speed"].notna().sum()
    no_speed  = (joined["average_speed"].isna() & (joined["predominant_speed_coverage_pct"] > 0)).sum()
    no_osm    = (joined["predominant_speed_coverage_pct"] == 0).sum()
    print(f"\n  OSM speed coverage:")
    print(f"    Speed assigned     : {has_speed:>4}  ({100*has_speed/total:.1f}%)")
    print(f"    OSM found, no speed: {no_speed:>4}  ({100*no_speed/total:.1f}%)")
    print(f"    No OSM match       : {no_osm:>4}  ({100*no_osm/total:.1f}%)")

    return joined


def derive_segment_transport_mode(
    nodes: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Propagate transport_mode from station nodes to segments and junctions via BFS.

    Single-mode stations seed the traversal.  Composite-mode stations
    (e.g. "train / tram") are left unseeded so propagation fills them from
    their neighbours.  Each segment and junction node receives the union of
    all modes reaching it.  Segments or junctions unreachable from any seed
    retain NaN; OSM matching falls back to gauge for those.

    Args:
        nodes:    macro-level nodes GeoDataFrame (must have NAME, transport_mode,
                  node_class columns)
        segments: macro-level segments GeoDataFrame (must have from_name, to_name)

    Returns:
        (nodes, segments) — nodes has derived transport_mode filled on previously-NaN
        junctions; segments has a new transport_mode column.
    """
    # Only modes that correspond to tracked rail infrastructure are propagated.
    # Bus, ship, etc. are excluded — they don't run on the BAV rail graph.
    _RAIL_MODES = frozenset(MODE_TO_OSM_TYPES.keys())

    def _parse_all(val) -> set:
        if pd.isna(val) or not str(val).strip():
            return set()
        return {m.strip() for m in str(val).split('/') if m.strip()}

    def _parse_rail(val) -> set:
        return _parse_all(val) & _RAIL_MODES

    def _fmt(modes: set):
        return ' / '.join(sorted(modes)) if modes else pd.NA

    nodes    = nodes.copy()
    segments = segments.copy()

    # Adjacency: node_name → [(seg_index, other_node_name), ...]
    adj: Dict[str, list] = {}
    for idx, row in segments.iterrows():
        fn, tn = row.get('from_name'), row.get('to_name')
        if fn and tn:
            adj.setdefault(fn, []).append((idx, tn))
            adj.setdefault(tn, []).append((idx, fn))

    # Collect OeV-sourced rail modes (non-rail modes stripped at source)
    node_modes: Dict[str, set] = {}
    for _, row in nodes.iterrows():
        name = row.get('NAME')
        if not name:
            continue
        m = _parse_rail(row.get('transport_mode'))
        if m:
            node_modes[name] = m

    # Nodes with OeV-sourced rail data act as hard propagation boundaries —
    # the BFS never overwrites them, preventing modes bleeding across network types.
    original_rail_seeds: frozenset = frozenset(node_modes)

    seg_modes: Dict[int, set] = {}

    # Seed: only single-mode stations (composite stations are unseeded)
    queue: deque = deque(n for n, m in node_modes.items() if len(m) == 1)

    while queue:
        name = queue.popleft()
        n_modes = node_modes.get(name, set())
        if not n_modes:
            continue
        for seg_idx, other in adj.get(name, []):
            old_s = seg_modes.get(seg_idx, set())
            new_s = old_s | n_modes
            if new_s != old_s:
                seg_modes[seg_idx] = new_s
                # Hard stop: never cross into a node that already has OeV rail data
                if other not in original_rail_seeds:
                    old_o = node_modes.get(other, set())
                    new_o = old_o | new_s
                    if new_o != old_o:
                        node_modes[other] = new_o
                        queue.append(other)

    # Write segment modes (all segments start without transport_mode)
    segments['transport_mode'] = [_fmt(seg_modes.get(i, set())) for i in segments.index]

    # Update junctions: only fill nodes that had no OeV rail data
    name_to_idx: Dict[str, int] = {
        row.get('NAME'): idx
        for idx, row in nodes.iterrows()
        if row.get('NAME')
    }
    for name, modes in node_modes.items():
        if name in original_rail_seeds:
            continue  # station already has OeV data — do not overwrite
        idx = name_to_idx.get(name)
        if idx is None:
            continue
        existing = _parse_all(
            nodes.at[idx, 'transport_mode'] if 'transport_mode' in nodes.columns else pd.NA
        )
        if not existing and modes:
            nodes.at[idx, 'transport_mode'] = _fmt(modes)

    n_segs = segments['transport_mode'].notna().sum()
    n_junc = (
        nodes[nodes.get('node_class', pd.Series(dtype=str)) == 'junction']['transport_mode'].notna().sum()
        if 'node_class' in nodes.columns else 0
    )
    print(f"  Mode assigned: {n_segs}/{len(segments)} segments, {n_junc} junctions derived")
    return nodes, segments


# =============================================================================
# ASP Station Import + Parent-Child Assignment
# =============================================================================

ASP_IMPORT_RADIUS_M    = 200  # max distance from prefix-matching parent station
PARENT_CHILD_RADIUS_M  = 200  # max distance for prefix-based parent detection


def _import_asp_stations(
    macro_nodes: gpd.GeoDataFrame,
    macro_segs:  gpd.GeoDataFrame,
    raw_nodes:   gpd.GeoDataFrame,
    raw_segs:    gpd.GeoDataFrame,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Import assigned_service_point nodes from raw_nodes into the macroscopic network.

    A node qualifies if:
      - its node_class is 'assigned_service_point'
      - it lies within ASP_IMPORT_RADIUS_M of a station node in macro_nodes
      - the station's NAME is a word-for-word prefix of the ASP's NAME

    Qualifying nodes are reclassified as 'station'. Their raw connecting segments
    are added where the other endpoint is already in the (updated) macro_nodes.
    """
    asp_raw = raw_nodes[raw_nodes['node_class'] == 'assigned_service_point'].copy()
    if asp_raw.empty:
        return macro_nodes, macro_segs

    macro_station_mask = macro_nodes['node_class'].isin({'station', 'abandoned_station'})
    macro_stations = macro_nodes[macro_station_mask].copy()
    macro_names = set(macro_nodes['NAME'].dropna())

    # ── Pass 1: identify which ASP nodes to import ────────────────────────────
    to_import = []
    for _, asp in asp_raw.iterrows():
        asp_name = str(asp['NAME'])
        if asp_name in macro_names:
            continue  # already present
        asp_words = asp_name.lower().split()
        aE = float(asp['E'])
        aN = float(asp['N'])

        best_parent_name = None
        best_len = 0
        for _, stn in macro_stations.iterrows():
            stn_name = str(stn['NAME'])
            stn_words = stn_name.lower().split()
            if len(stn_words) >= len(asp_words):
                continue  # parent must have fewer words
            if asp_words[:len(stn_words)] != stn_words:
                continue
            dist = np.hypot(float(stn['E']) - aE, float(stn['N']) - aN)
            if dist <= ASP_IMPORT_RADIUS_M and len(stn_words) > best_len:
                best_parent_name = stn_name
                best_len = len(stn_words)

        if best_parent_name is not None:
            new_node = asp.copy()
            new_node['node_class'] = 'station'
            to_import.append((new_node, best_parent_name))

    if not to_import:
        return macro_nodes, macro_segs

    # ── Pass 2: add all qualifying nodes then add their segments ─────────────
    new_node_rows = [n for n, _ in to_import]
    new_nodes_gdf = gpd.GeoDataFrame(new_node_rows, crs=macro_nodes.crs)
    macro_nodes = pd.concat([macro_nodes, new_nodes_gdf], ignore_index=True)
    macro_names = set(macro_nodes['NAME'].dropna())

    new_segs = []
    seen_seg_ids = set(macro_segs['segment_id'].dropna())
    for new_node, parent_name in to_import:
        asp_name = str(new_node['NAME'])
        mask = (raw_segs['from_name'] == asp_name) | (raw_segs['to_name'] == asp_name)
        for _, seg in raw_segs[mask].iterrows():
            other = seg['to_name'] if seg['from_name'] == asp_name else seg['from_name']
            if other in macro_names and seg['segment_id'] not in seen_seg_ids:
                new_segs.append(seg)
                seen_seg_ids.add(seg['segment_id'])
        print(
            f"  [ASP] imported '{asp_name}' → station"
            f"  (prefix-parent: '{parent_name}',"
            f" dist: {np.hypot(float(new_node['E']) - float(macro_nodes[macro_nodes['NAME'] == parent_name]['E'].iloc[0]), float(new_node['N']) - float(macro_nodes[macro_nodes['NAME'] == parent_name]['N'].iloc[0])):.0f}m)"
        )

    if new_segs:
        new_segs_gdf = gpd.GeoDataFrame(new_segs, crs=macro_segs.crs)
        macro_segs = pd.concat([macro_segs, new_segs_gdf], ignore_index=True)

    # Remove merged bypass segments whose constituent IDs were just rescued.
    # filter_macroscopic_nodes() merges degree-2 non-KEEP nodes by joining their
    # segment IDs with '+'. When an ASP node is rescued, those original segments
    # are restored, making the bypass a duplicate. Drop it.
    rescued_ids = {str(s['segment_id']) for s in new_segs}
    if rescued_ids:
        def _is_bypass(sid):
            sid_str = str(sid)
            if '+' not in sid_str:
                return False
            return bool(rescued_ids.intersection(sid_str.split('+')))

        bypass_mask = macro_segs['segment_id'].apply(_is_bypass)
        n_bypass = int(bypass_mask.sum())
        if n_bypass:
            macro_segs = macro_segs[~bypass_mask].reset_index(drop=True)
            print(f"  [ASP] removed {n_bypass} merged bypass segment(s) superseded by rescued originals")

    print(f"  [ASP] {len(to_import)} node(s) imported, {len(new_segs)} segment(s) added")
    return macro_nodes.reset_index(drop=True), macro_segs.reset_index(drop=True)


def _assign_parent_child(macro_nodes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Detect parent-child station relationships via word-prefix name matching.

    For each station with a NULL parent_node: if another station's NAME is a
    strict word-for-word prefix of this station's NAME and lies within
    PARENT_CHILD_RADIUS_M, the longer-named station becomes the child and the
    shorter-named station's Betriebspunkt_Nummer is written into parent_node.

    Only fills NULL parent_node values; never overwrites existing relationships.
    """
    station_mask = macro_nodes['node_class'].isin({'station', 'abandoned_station'})
    stations = macro_nodes[station_mask].reset_index()  # keep original index

    updated = 0
    for idx, row in macro_nodes.iterrows():
        if not station_mask.get(idx, False):
            continue
        existing_parent = row.get('parent_node')
        if pd.notna(existing_parent) and str(existing_parent).strip().lower() not in ('', 'none', 'nan'):
            continue  # already has a parent

        child_name = str(row.get('NAME', ''))
        if not child_name:
            continue
        child_words = child_name.lower().split()
        cE = float(row['E'])
        cN = float(row['N'])

        best_bpnr = None
        best_len = 0
        for _, stn in stations.iterrows():
            stn_name = str(stn['NAME'])
            if stn_name == child_name:
                continue
            stn_words = stn_name.lower().split()
            if len(stn_words) >= len(child_words):
                continue
            if child_words[:len(stn_words)] != stn_words:
                continue
            dist = np.hypot(float(stn['E']) - cE, float(stn['N']) - cN)
            if dist > PARENT_CHILD_RADIUS_M:
                continue
            bpnr = stn.get('Betriebspunkt_Nummer')
            if pd.notna(bpnr) and len(stn_words) > best_len:
                best_bpnr = int(float(bpnr))
                best_len = len(stn_words)

        if best_bpnr is not None:
            macro_nodes.at[idx, 'parent_node'] = best_bpnr
            updated += 1

    print(f"  [parent-child] {updated} relationship(s) assigned")
    return macro_nodes


def run_build_base(
    raw_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Load Raw/ network, apply macroscopic simplification, export to Base/.

    Reads from:  data/Infrastructure/Raw/   (written by infrabuild_filter_network.py)
    Writes to:   data/Infrastructure/Base/  (nodes.gpkg, segments.gpkg,
                                             segments_composition.gpkg, Base.qgz)
    Returns (macro_nodes, macro_segs, composition).
    """
    print("=" * 60)
    print("Build Macroscopic Base Network")
    print("=" * 60)

    raw_path = Path(raw_dir  or (Path(paths.MAIN) / paths.NETWORK_INFRASTRUCTURE_RAW))
    out_path = Path(output_dir or (Path(paths.MAIN) / paths.NETWORK_INFRASTRUCTURE_BASE))

    nodes_path = raw_path / "nodes.gpkg"
    segs_path  = raw_path / "segments.gpkg"

    if not nodes_path.exists() or not segs_path.exists():
        raise FileNotFoundError(
            f"Raw network not found in {raw_path}.\n"
            "Run infrabuild_filter_network.py first."
        )

    print("\n--- Loading raw network ---")
    nodes    = gpd.read_file(nodes_path)
    segments = gpd.read_file(segs_path)
    print(f"  {len(nodes)} nodes, {len(segments)} segments")

    comp_raw_path = raw_path / "segments_composition.gpkg"
    composition_raw = (
        gpd.read_file(comp_raw_path) if comp_raw_path.exists() else gpd.GeoDataFrame()
    )
    if not composition_raw.empty:
        print(f"  {len(composition_raw)} composition pieces (raw)")

    print("\n--- Macroscopic simplification ---")
    macro_nodes, macro_segs, seg_id_remap = filter_macroscopic_nodes(nodes, segments)

    composition = (
        _filter_composition_for_macro(composition_raw, seg_id_remap)
        if not composition_raw.empty else gpd.GeoDataFrame()
    )

    print("\n--- ASP station import ---")
    macro_nodes, macro_segs = _import_asp_stations(macro_nodes, macro_segs, nodes, segments)

    print("\n--- Parent-child assignment ---")
    macro_nodes = _assign_parent_child(macro_nodes)

    print("\n--- Transport mode propagation ---")
    macro_nodes, macro_segs = derive_segment_transport_mode(macro_nodes, macro_segs)

    print("\n--- OSM speed enrichment ---")
    macro_segs = enrich_segments_with_osm_speed(macro_segs, macro_nodes)

    out_path.mkdir(parents=True, exist_ok=True)
    print("\n--- Exporting ---")
    macro_nodes.to_file(out_path / "nodes.gpkg", driver="GPKG")
    macro_segs.to_file(out_path  / "segments.gpkg", driver="GPKG")
    print(f"  nodes.gpkg    → {out_path / 'nodes.gpkg'}")
    print(f"  segments.gpkg → {out_path / 'segments.gpkg'}")
    if not composition.empty:
        composition.to_file(out_path / "segments_composition.gpkg", driver="GPKG")
        print(f"  segments_composition.gpkg → {out_path / 'segments_composition.gpkg'}"
              f"  ({len(composition)} pieces)")

    print("\n--- QGIS project ---")
    _build_infra_qgz(str(out_path / "Base.qgz"), out_path)
    print(f"  Base.qgz → {out_path / 'Base.qgz'}")

    print("\n" + "=" * 60)
    print(f"Done  |  {len(macro_nodes)} nodes  |  {len(macro_segs)} segments")
    print("=" * 60)

    return macro_nodes, macro_segs, composition


# =============================================================================
# Composition Handling
# =============================================================================

def _filter_composition_for_macro(
    composition: gpd.GeoDataFrame,
    seg_id_remap: dict,
) -> gpd.GeoDataFrame:
    """Filter and remap segments_composition after macroscopic simplification.

    Drops pieces for segments that were removed (dead-ends); remaps segment_id
    to the merged ID for segments that were joined through an intermediate node.
    """
    if composition.empty or 'segment_id' not in composition.columns:
        return composition
    mask = composition['segment_id'].astype(str).isin(seg_id_remap)
    result = composition[mask].copy()
    result['segment_id'] = result['segment_id'].astype(str).map(seg_id_remap)
    return result.reset_index(drop=True)


def _filter_composition_for_version(
    composition_base: gpd.GeoDataFrame,
    version_segment_ids: set,
) -> gpd.GeoDataFrame:
    """Filter Base composition to segments present in a named version.

    Segments removed by infrabuild_version_manager are absent from
    version_segment_ids and are simply dropped.  Manually added segments
    (new_From_To IDs) have no composition entry, so they are absent already.
    """
    if composition_base.empty or 'segment_id' not in composition_base.columns:
        return composition_base
    mask = composition_base['segment_id'].isin(version_segment_ids)
    return composition_base[mask].reset_index(drop=True)


# =============================================================================
# QGIS Project Generation
# =============================================================================

_QGIS_VERSION = "3.44.9-Solothurn"

_INFRA_SRS_BLOCK = """<spatialrefsys nativeFormat="wkt">
      <proj4>+proj=somerc +lat_0=46.9524055555556 +lon_0=7.43958333333333 +k_0=1 +x_0=2600000 +y_0=1200000 +ellps=bessel +towgs84=674.374,15.056,405.346,0,0,0,0 +units=m +no_defs</proj4>
      <srsid>47</srsid>
      <srid>2056</srid>
      <authid>EPSG:2056</authid>
      <description>CH1903+ / LV95</description>
      <projectionacronym>somerc</projectionacronym>
      <ellipsoidacronym>bessel</ellipsoidacronym>
    </spatialrefsys>"""

_INFRA_WMS_ID   = 'Swisstopo_NationalMap_grey_infrabuild'
_INFRA_WMS_NAME = 'Swisstopo National Map (grey)'
_INFRA_WMS_SRC  = (
    'contextualWMSLegend=0&amp;crs=EPSG:2056&amp;dpiMode=7'
    '&amp;featureCount=10&amp;format=image/png'
    '&amp;layers=ch.swisstopo.pixelkarte-grau'
    '&amp;styles=&amp;url=http://wms.geo.admin.ch/'
)


def _multi_track_symbol_xml(sym_name: str, num_tracks: int) -> str:
    """XML for a line symbol with num_tracks parallel SimpleLine sub-layers."""
    width = '0.40'
    if num_tracks == 1:
        offsets = [0.0]
    elif num_tracks == 2:
        offsets = [0.400, -0.400]
    elif num_tracks == 3:
        offsets = [0.800, 0.0, -0.800]
    else:
        offsets = [1.200, 0.400, -0.400, -1.200]

    layers_xml = ''
    for off in offsets:
        layers_xml += f"""
          <layer pass="0" class="SimpleLine" locked="0" enabled="1">
            <prop k="line_color" v="0,0,0,255"/>
            <prop k="line_width" v="{width}"/>
            <prop k="line_width_unit" v="MM"/>
            <prop k="line_style" v="solid"/>
            <prop k="offset" v="{off:.3f}"/>
            <prop k="offset_unit" v="MM"/>
            <prop k="capstyle" v="square"/>
            <prop k="joinstyle" v="bevel"/>
          </layer>"""
    return (
        f'    <symbol alpha="1" clip_to_extent="1" type="line"'
        f' name="{sym_name}" force_rhr="0">\n'
        f'{layers_xml}\n'
        f'    </symbol>'
    )


def _segments_maplayer_xml(layer_id: str, gpkg_relpath: str,
                            display_name: str) -> str:
    """<maplayer> XML for segments with a rule-based multi-track renderer."""
    _RULES = [
        (1,  '&quot;num_tracks&quot; = 1',    '1 track',  '0'),
        (2,  '&quot;num_tracks&quot; = 2',    '2 tracks', '1'),
        (3,  '&quot;num_tracks&quot; = 3',    '3 tracks', '2'),
        (99, '&quot;num_tracks&quot; &gt;= 4','4+ tracks','3'),
    ]
    root_key = uuid.uuid4().hex
    rules_parts, symbols_parts = [], []
    for n_tracks, flt, label, sym_name in _RULES:
        rkey = uuid.uuid4().hex
        rules_parts.append(
            f'      <rule key="{rkey}" filter="{flt}" label="{label}" symbol="{sym_name}"/>'
        )
        symbols_parts.append(_multi_track_symbol_xml(sym_name, n_tracks))

    renderer_xml = (
        f'    <renderer-v2 type="RuleRenderer" symbollevels="0"'
        f' forceraster="0" enableorderby="0">\n'
        f'      <rules key="{root_key}">\n'
        + '\n'.join(rules_parts) + '\n'
        f'      </rules>\n'
        f'      <symbols>\n'
        + '\n'.join(symbols_parts) + '\n'
        f'      </symbols>\n'
        f'      <rotation/>\n'
        f'      <sizescale/>\n'
        f'    </renderer-v2>'
    )
    return (
        f'  <maplayer geometry="Line" type="vector" hasScaleBasedVisibilityFlag="0">\n'
        f'    <id>{layer_id}</id>\n'
        f'    <datasource>{gpkg_relpath}|layername=segments</datasource>\n'
        f'    <layername>{display_name}</layername>\n'
        f'    <provider encoding="UTF-8">ogr</provider>\n'
        f'    <srs>{_INFRA_SRS_BLOCK}</srs>\n'
        f'{renderer_xml}\n'
        f'  </maplayer>'
    )


def _nodes_maplayer_xml(layer_id: str, gpkg_relpath: str, display_name: str) -> str:
    """<maplayer> XML for all nodes using a RuleRenderer."""
    root_key = uuid.uuid4().hex
    rules_xml = f"""\
      <rules key="{root_key}">\
        <rule key="{uuid.uuid4().hex}" filter="&quot;node_class&quot; = 'station' AND &quot;transport_mode&quot; LIKE '%train%'" label="Train Stations" symbol="0"/>\
        <rule key="{uuid.uuid4().hex}" filter="&quot;node_class&quot; = 'station' AND (&quot;transport_mode&quot; LIKE '%tram%' OR &quot;transport_mode&quot; LIKE '%funicular%' OR &quot;transport_mode&quot; LIKE '%cog_railway%')" label="Tram / Funicular" symbol="1"/>\
        <rule key="{uuid.uuid4().hex}" filter="&quot;node_class&quot; = 'junction'" label="Junctions" symbol="2"/>\
      </rules>"""

    marker_xml = f"""\
    <renderer-v2 forceraster="0" symbollevels="0" type="RuleRenderer" enableorderby="0">
{rules_xml}
      <symbols>
        <symbol alpha="1" clip_to_extent="1" type="marker" name="0" force_rhr="0">
          <layer pass="0" class="SimpleMarker" locked="0" enabled="1">
            <prop k="angle" v="0"/>
            <prop k="color" v="255,255,255,255"/>
            <prop k="joinstyle" v="bevel"/>
            <prop k="name" v="circle"/>
            <prop k="offset" v="0,0"/>
            <prop k="offset_unit" v="MM"/>
            <prop k="outline_color" v="0,0,0,255"/>
            <prop k="outline_style" v="solid"/>
            <prop k="outline_width" v="0.25"/>
            <prop k="outline_width_unit" v="MM"/>
            <prop k="size" v="2.5"/>
            <prop k="size_unit" v="MM"/>
          </layer>
        </symbol>
        <symbol alpha="1" clip_to_extent="1" type="marker" name="1" force_rhr="0">
          <layer pass="0" class="SimpleMarker" locked="0" enabled="1">
            <prop k="angle" v="0"/>
            <prop k="color" v="0,0,0,255"/>
            <prop k="joinstyle" v="bevel"/>
            <prop k="name" v="circle"/>
            <prop k="offset" v="0,0"/>
            <prop k="offset_unit" v="MM"/>
            <prop k="outline_color" v="0,0,0,255"/>
            <prop k="outline_style" v="solid"/>
            <prop k="outline_width" v="0.25"/>
            <prop k="outline_width_unit" v="MM"/>
            <prop k="size" v="1.75"/>
            <prop k="size_unit" v="MM"/>
          </layer>
        </symbol>
        <symbol alpha="1" clip_to_extent="1" type="marker" name="2" force_rhr="0">
          <layer pass="0" class="SimpleMarker" locked="0" enabled="1">
            <prop k="angle" v="0"/>
            <prop k="color" v="150,150,150,255"/>
            <prop k="joinstyle" v="bevel"/>
            <prop k="name" v="circle"/>
            <prop k="offset" v="0,0"/>
            <prop k="offset_unit" v="MM"/>
            <prop k="outline_color" v="120,120,120,255"/>
            <prop k="outline_style" v="solid"/>
            <prop k="outline_width" v="0.20"/>
            <prop k="outline_width_unit" v="MM"/>
            <prop k="size" v="1.25"/>
            <prop k="size_unit" v="MM"/>
          </layer>
        </symbol>
      </symbols>
      <rotation/>
      <sizescale/>
    </renderer-v2>"""

    labeling_xml = """\
    <labeling type="rule-based">
      <rules>
        <rule filter="&quot;node_class&quot; = 'station' AND &quot;transport_mode&quot; LIKE '%train%'">
          <settings calloutType="simple">
            <text-style fieldName="CODE" fontFamily="Arial" fontSize="8" fontWeight="0" fontItalic="0" fontUnderline="0" fontStrikeout="0" textColor="35,35,35,255" textOpacity="1" blendMode="0" namedStyle="Regular" isExpression="0" useSubstitutions="0" multilineHeight="1" fontCapitals="0" fontLetterSpacing="0" fontWordSpacing="0" fontSizeUnit="Point" textOrientation="horizontal" previewBkgrdColor="255,255,255,255"/>
            <text-buffer bufferDraw="1" bufferSize="1" bufferColor="255,255,255,255" bufferOpacity="1" bufferJoinStyle="128" bufferNoFill="1" bufferSizeUnits="MM" bufferSizeMapUnitScale="3x:0,0,0,0,0,0" bufferBlendMode="0"/>
            <background shapeDraw="0"/>
            <shadow shadowDraw="0"/>
            <placement placement="1" centroidWhole="0" placementFlags="10" priority="5" offsetType="0" quadOffset="4" xOffset="2" yOffset="2" offsetUnits="MM" dist="1" distInMapUnits="0" distMapUnitScale="3x:0,0,0,0,0,0" rotationAngle="0" geometryGenerator="" geometryGeneratorEnabled="0" geometryGeneratorType="PointGeometry" isExpression="0" labelPerPart="0"/>
            <rendering drawLabels="1" obstacle="1" obstacleFactor="1" obstacleType="1" limitNumLabels="0" maxNumLabels="2000" minFeatureSize="0" fontMinPixelSize="3" fontMaxPixelSize="10000" displayAll="0" upsidedownLabels="0" mergeLines="0" zIndex="0" scaleVisibility="0" scaleMin="1" scaleMax="10000000" labelPerPart="0"/>
            <dd_properties/>
          </settings>
        </rule>
      </rules>
    </labeling>"""

    return (
        f'  <maplayer geometry="Point" type="vector" hasScaleBasedVisibilityFlag="0">\n'
        f'    <id>{layer_id}</id>\n'
        f'    <datasource>{gpkg_relpath}|layername=nodes</datasource>\n'
        f'    <layername>{display_name}</layername>\n'
        f'    <provider encoding="UTF-8">ogr</provider>\n'
        f'    <srs>{_INFRA_SRS_BLOCK}</srs>\n'
        f'{marker_xml}\n'
        f'{labeling_xml}\n'
        f'  </maplayer>'
    )


def _wms_maplayer_xml() -> str:
    return (
        f'  <maplayer type="raster" hasScaleBasedVisibilityFlag="0">\n'
        f'    <id>{_INFRA_WMS_ID}</id>\n'
        f'    <datasource>{_INFRA_WMS_SRC}</datasource>\n'
        f'    <layername>{_INFRA_WMS_NAME}</layername>\n'
        f'    <provider encoding="">wms</provider>\n'
        f'    <srs>{_INFRA_SRS_BLOCK}</srs>\n'
        f'  </maplayer>'
    )


def _boundary_maplayer_xml(layer_id: str, gpkg_relpath: str,
                            display_name: str, color: str = '0,0,0,255') -> str:
    """<maplayer> XML for a polygon boundary: dashed outline, no fill."""
    return (
        f'  <maplayer geometry="Polygon" type="vector" hasScaleBasedVisibilityFlag="0">\n'
        f'    <id>{layer_id}</id>\n'
        f'    <datasource>{gpkg_relpath}</datasource>\n'
        f'    <layername>{display_name}</layername>\n'
        f'    <provider encoding="UTF-8">ogr</provider>\n'
        f'    <srs>{_INFRA_SRS_BLOCK}</srs>\n'
        f'    <renderer-v2 type="singleSymbol" symbollevels="0" forceraster="0" enableorderby="0">\n'
        f'      <symbols>\n'
        f'        <symbol alpha="1" clip_to_extent="1" type="fill" name="0" force_rhr="0">\n'
        f'          <layer pass="0" class="SimpleFill" locked="0" enabled="1">\n'
        f'            <prop k="color" v="0,0,0,0"/>\n'
        f'            <prop k="style" v="no"/>\n'
        f'            <prop k="outline_color" v="{color}"/>\n'
        f'            <prop k="outline_style" v="dash"/>\n'
        f'            <prop k="outline_width" v="0.6"/>\n'
        f'            <prop k="outline_width_unit" v="MM"/>\n'
        f'            <prop k="joinstyle" v="miter"/>\n'
        f'            <prop k="offset" v="0,0"/>\n'
        f'            <prop k="offset_unit" v="MM"/>\n'
        f'          </layer>\n'
        f'        </symbol>\n'
        f'      </symbols>\n'
        f'      <rotation/>\n'
        f'      <sizescale/>\n'
        f'    </renderer-v2>\n'
        f'  </maplayer>'
    )


def _build_infra_qgz(qgz_path: str, version_dir: Path) -> None:
    """Write a QGIS .qgz infrastructure project alongside the network geopackages.

    Layer order (top = rendered last = on top):
      1. Stations/halts — white circles, black outline, CODE label
      2. Catchment area boundary — dashed black outline, no fill
      3. Study area boundary    — dashed dark-red outline, no fill
      4. Segments — black parallel lines scaled by num_tracks
      5. Swisstopo WMS basemap
    """
    version = version_dir.name
    seg_id  = f'seg_{uuid.uuid4().hex[:8]}'
    nd_id   = f'nd_{uuid.uuid4().hex[:8]}'
    ca_id   = f'ca_{uuid.uuid4().hex[:8]}'
    sa_id   = f'sa_{uuid.uuid4().hex[:8]}'

    # Relative paths from the version directory to boundary files
    ca_relpath = '../../Catchment_Area/catchment_area_boundary.gpkg'
    sa_relpath = '../../Catchment_Area/study_area_boundary.gpkg'

    seg_block = _segments_maplayer_xml(seg_id, 'segments.gpkg',
                                        f'Segments — {version}')
    nd_block  = _nodes_maplayer_xml(nd_id, 'nodes.gpkg',
                                    f'Nodes — {version}')
    ca_block  = _boundary_maplayer_xml(ca_id, ca_relpath,
                                        'Catchment Area Boundary', '0,0,0,255')
    sa_block  = _boundary_maplayer_xml(sa_id, sa_relpath,
                                        'Study Area Boundary', '139,0,0,255')
    wms_block = _wms_maplayer_xml()

    tree_xml = (
        f'    <layer-tree-layer id="{nd_id}" name="Nodes — {version}" '
        f'checked="Qt::Checked" expanded="1" '
        f'source="nodes.gpkg|layername=nodes" providerKey="ogr"/>\n'
        f'    <layer-tree-layer id="{ca_id}" name="Catchment Area Boundary" '
        f'checked="Qt::Checked" expanded="0" '
        f'source="{ca_relpath}" providerKey="ogr"/>\n'
        f'    <layer-tree-layer id="{sa_id}" name="Study Area Boundary" '
        f'checked="Qt::Checked" expanded="0" '
        f'source="{sa_relpath}" providerKey="ogr"/>\n'
        f'    <layer-tree-layer id="{seg_id}" name="Segments — {version}" '
        f'checked="Qt::Checked" expanded="1" '
        f'source="segments.gpkg|layername=segments" providerKey="ogr"/>\n'
        f'    <layer-tree-layer id="{_INFRA_WMS_ID}" name="{_INFRA_WMS_NAME}" '
        f'checked="Qt::Checked" expanded="0" '
        f'source="{_INFRA_WMS_SRC}" providerKey="wms"/>'
    )

    qgs = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis projectname="{version} Infrastructure" version="{_QGIS_VERSION}">
  <homePath path=""/>
  <title>{version} Infrastructure</title>
  <autotransaction active="0"/>
  <evaluateDefaultValues active="0"/>
  <trust active="0"/>
  <projectCrs>
    {_INFRA_SRS_BLOCK}
  </projectCrs>
  <layer-tree-group>
    <customproperties/>
{tree_xml}
    <custom-order enabled="0"/>
  </layer-tree-group>
  <projectlayers>
{nd_block}
{ca_block}
{sa_block}
{seg_block}
{wms_block}
  </projectlayers>
  <mapcanvas name="theMapCanvas">
    <units>meters</units>
    <rotation>0</rotation>
    <destinationsrs>
      {_INFRA_SRS_BLOCK}
    </destinationsrs>
  </mapcanvas>
</qgis>
"""
    with zipfile.ZipFile(qgz_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('project.qgs', qgs)


# =============================================================================
# Graph Building
# =============================================================================

def build_networkx_graph(
    nodes: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
) -> nx.Graph:
    """
    Build a NetworkX graph from macroscopic nodes and segments.

    Node key  : NAME (station/junction name)
    Node attrs: CODE, N, E, node_class, geometry
    Edge attrs: segment_id, length_m, num_tracks, gauge, electrification, geometry
    """
    G = nx.Graph()

    for _, row in nodes.iterrows():
        name = row.get('NAME')
        if pd.isna(name):
            continue
        G.add_node(
            name,
            CODE=row.get('CODE', ''),
            N=row.get('N', 0),
            E=row.get('E', 0),
            node_class=row.get('node_class', 'unknown'),
            geometry=row.geometry,
        )

    skipped = 0
    for _, row in segments.iterrows():
        fn = row.get('from_name')
        tn = row.get('to_name')
        if pd.isna(fn) or pd.isna(tn):
            skipped += 1
            continue
        if fn not in G or tn not in G:
            skipped += 1
            continue
        G.add_edge(
            fn, tn,
            segment_id=row.get('segment_id'),
            length_m=row.get('length_m', 0),
            num_tracks=row.get('num_tracks', 1),
            gauge=row.get('gauge', 1435),
            electrification=row.get('electrification', 'unknown'),
            geometry=row.geometry,
        )

    if skipped:
        print(f"  Skipped {skipped} segments (missing node reference)")
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# =============================================================================
# Node Classification
# =============================================================================

def _classify_nodes(nodes: gpd.GeoDataFrame):
    """Split nodes GDF into (train_stations, tram_funicular, junctions).

    train_stations : node_class == 'station' AND transport_mode contains 'train'
                     OR node_class == 'abandoned_station'
    tram_funicular : node_class == 'station' AND transport_mode contains
                     'tram', 'funicular', or 'cog_railway' (and NOT 'train')
    junctions      : node_class == 'junction'
    """
    if nodes is None or nodes.empty:
        empty = gpd.GeoDataFrame()
        return empty, empty, empty

    nc = (nodes['node_class']     if 'node_class'     in nodes.columns
          else pd.Series([''] * len(nodes), index=nodes.index))
    tm = (nodes['transport_mode'] if 'transport_mode' in nodes.columns
          else pd.Series([''] * len(nodes), index=nodes.index))

    is_station = nc.astype(str) == 'station'
    is_abandoned = nc.astype(str) == 'abandoned_station'
    has_train  = tm.astype(str).str.contains('train', na=False)
    has_tram_f = tm.astype(str).str.contains(
        'tram|funicular|cog_railway', na=False, regex=True)

    train_stations = nodes[is_station & has_train | is_abandoned]
    tram_funicular = nodes[is_station & has_tram_f & ~has_train]
    junctions      = nodes[nc.astype(str) == 'junction']
    return train_stations, tram_funicular, junctions


# =============================================================================
# Plotting Utilities
# =============================================================================

def _get_line_width(num_tracks: int) -> float:
    return 1.0 + (num_tracks - 1) * 0.5


_SCALE_BAR_NICE_KM = [1, 2, 5, 10, 20, 50, 100, 200, 500]


def _add_north_arrow(ax, location='upper left', scale=0.5):
    """Draw a north arrow using matplotlib_map_utils."""
    north_arrow(ax, location=location, scale=scale, rotation={"degrees": 0})


def _add_scale_bar(ax, location=(0.755, 0.012)):
    """Adaptive scale bar with 2-3 alternating black/white cells.

    Automatically picks a round km value based on the current axes extent.
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    map_w = xlim[1] - xlim[0]
    map_h = ylim[1] - ylim[0]

    target_km = (map_w / 4.0) / 1000.0
    total_km  = min(_SCALE_BAR_NICE_KM, key=lambda v: abs(v - target_km))
    n_cells   = 4 if total_km >= 4 else 2
    cell_m    = (total_km * 1000.0) / n_cells

    x0    = xlim[0] + map_w * location[0]
    y0    = ylim[0] + map_h * location[1]
    bar_h = map_h * 0.008

    for i in range(n_cells):
        color = 'black' if i % 2 == 0 else 'white'
        rect = Rectangle(
            (x0 + i * cell_m, y0), cell_m, bar_h,
            facecolor=color, edgecolor='black', linewidth=0.6, zorder=7,
        )
        ax.add_patch(rect)

    for i in range(n_cells + 1):
        val_km = (i * cell_m) / 1000.0
        label  = (f'{val_km:.0f} km' if val_km == int(val_km)
                  else f'{val_km:.1f} km')
        ax.text(
            x0 + i * cell_m, y0 + bar_h * 1.6,
            label, ha='center', va='bottom', fontsize=7, zorder=7,
        )


def _clip_to_boundary(gdf: gpd.GeoDataFrame,
                      boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Clip gdf geometries to boundary; returns original if clip fails."""
    if gdf is None or gdf.empty or boundary is None or boundary.empty:
        return gdf
    try:
        clipped = gpd.clip(gdf, boundary)
        return clipped if not clipped.empty else gdf
    except Exception:
        return gdf


def _clip_to_extent(gdf: gpd.GeoDataFrame, extent) -> gpd.GeoDataFrame:
    """Clip gdf geometries precisely to the extent bounding box to fix mask bounds."""
    if gdf is None or gdf.empty or extent is None:
        return gdf
    try:
        from shapely.geometry import box as shapely_box
        bbox = gpd.GeoDataFrame(
            geometry=[shapely_box(extent[0], extent[2], extent[1], extent[3])],
            crs=gdf.crs
        )
        clipped = gpd.clip(gdf, bbox)
        return clipped if not clipped.empty else gpd.GeoDataFrame(columns=gdf.columns).set_crs(gdf.crs)
    except Exception:
        return gdf


def _plot_lakes(ax, boundary: Optional[gpd.GeoDataFrame] = None,
                extent=None) -> None:
    """Overlay lakes from swissTLMRegio.

    Clipping priority:
      1. extent (xmin, xmax, ymin, ymax) — used for SA plots, clips to plot frame
      2. boundary GeoDataFrame            — used for CA plots, clips to study polygon
      3. Neither provided                 — no clipping (full coverage)
    """
    from shapely.geometry import box as shapely_box
    lakes_path = Path(paths.MAIN) / paths.LAKES_SHP
    if not lakes_path.exists():
        return
    try:
        lakes = gpd.read_file(lakes_path)
        if lakes.crs is None:
            lakes = lakes.set_crs('EPSG:2056')
        elif lakes.crs.to_epsg() != 2056:
            lakes = lakes.to_crs('EPSG:2056')

        if extent is not None:
            clip_geom = gpd.GeoDataFrame(
                geometry=[shapely_box(extent[0], extent[2], extent[1], extent[3])],
                crs='EPSG:2056',
            )
            lakes = gpd.clip(lakes, clip_geom)
        elif boundary is not None and not boundary.empty:
            lakes = _clip_to_boundary(lakes, boundary)

        if not lakes.empty:
            lakes.plot(ax=ax, facecolor='#a8cfe0', edgecolor='#5a9ab5',
                       linewidth=0.4, zorder=1)
    except Exception as exc:
        print(f"  Warning: could not load lakes — {exc}")


def _base_plot(network: NetworkData, title: str, figsize: Tuple[int, int],
               extent=None):
    """Create figure, draw boundary, return (fig, ax).

    extent : optional (xmin, xmax, ymin, ymax) in map coordinates to set the
             axes limits.  When None the view auto-fits the plotted data.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('E [m]', fontsize=10)
    ax.set_ylabel('N [m]', fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    if network.boundary is not None:
        network.boundary.plot(
            ax=ax, facecolor='none', edgecolor='black',
            linewidth=1.5, linestyle='--', alpha=0.6,
        )
    if extent is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    return fig, ax


# =============================================================================
# Plot Functions
# =============================================================================

def plot_infrastructure(
    network: NetworkData,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12),
    show_labels: bool = True,
    is_catchment: bool = False,
) -> plt.Figure:
    """Infrastructure overview: segments coloured by track count, nodes by class."""
    if title is None:
        title = f"Infrastructure — {network.version}"
    fig, ax = _base_plot(network, title, figsize)
    _plot_lakes(ax, network.boundary)

    segs  = _clip_to_boundary(network.segments, network.boundary)
    nodes = _clip_to_boundary(network.nodes,    network.boundary)

    tracks_present = set()
    has_other_tracks = False

    # Segments by track count
    if len(segs) > 0:
        for n_tracks, color in TRACK_COLORS.items():
            mask = segs['num_tracks'] == n_tracks
            if mask.any():
                tracks_present.add(n_tracks)
                for _, row in segs[mask].iterrows():
                    _draw_parallel_tracks(
                        ax,
                        geom=row.geometry,
                        num_tracks=n_tracks,
                        color='black',
                        linewidth=1.5,
                        alpha=0.75,
                        zorder=2,
                        track_spacing_m=_TRACK_SPACING_M_CA if is_catchment else _TRACK_SPACING_M_SA
                    )
        other = ~segs['num_tracks'].isin(TRACK_COLORS)
        if other.any():
            has_other_tracks = True
            for _, row in segs[other].iterrows():
                _draw_parallel_tracks(
                    ax,
                    geom=row.geometry,
                    num_tracks=1,
                    color=TRACK_DEFAULT,
                    linewidth=1.5,
                    alpha=0.7,
                    zorder=2,
                    track_spacing_m=_TRACK_SPACING_M_CA if is_catchment else _TRACK_SPACING_M_SA
                )

    nodes_present = set()

    # Nodes by class
    if len(nodes) > 0:
        for cls, color in NODE_COLORS.items():
            mask = nodes['node_class'] == cls
            if mask.any():
                nodes_present.add(cls)
                size   = 80 if cls == 'station' else 30
                marker = 'o' if cls == 'station' else 's'
                nodes[mask].plot(
                    ax=ax, color=color, markersize=size, marker=marker,
                    alpha=0.85, zorder=4,
                    edgecolor='black' if cls == 'station' else None,
                    linewidth=1 if cls == 'station' else 0,
                )

        if show_labels:
            stations = nodes[nodes['node_class'] == 'station']
            for _, row in stations.iterrows():
                name = row.get('NAME') or row.get('CODE', '')
                if pd.notna(name):
                    ax.annotate(
                        str(name)[:25],
                        xy=(row.geometry.x, row.geometry.y),
                        xytext=(8, 8), textcoords='offset points',
                        fontsize=7, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                  edgecolor='gray', alpha=0.8),
                        zorder=6,
                    )

    # Legend
    legend = []
    handler_map = {}
    
    if tracks_present or has_other_tracks:
        legend.append(Line2D([0], [0], color='none', label='─ Tracks ─'))
        for n in sorted(tracks_present):
            agg_linewidth = 1.5 if n == 1 else n * 1.35 
            handle = Line2D([0], [0], color='black', linewidth=agg_linewidth, label=f'{n} track(s)')
            legend.append(handle)
            handler_map[handle] = _MultiTrackLegendHandler()
            handle.track_count = n
            handle.total_width = agg_linewidth
            handle.color = 'black'
            
        if has_other_tracks:
            legend.append(Line2D([0], [0], color=TRACK_DEFAULT, linewidth=1, label='Other'))
    
    if nodes_present:
        if tracks_present or has_other_tracks:
            legend.append(Line2D([0], [0], color='none', label=''))
        legend.append(Line2D([0], [0], color='none', label='─ Nodes ─'))
        for cls, color in NODE_COLORS.items():
            if cls in nodes_present:
                marker = 'o' if cls == 'station' else 's'
                size   = 10 if cls == 'station' else 7
                legend.append(Line2D([0], [0], marker=marker, color='w',
                                     markerfacecolor=color, markersize=size,
                                     markeredgecolor='black' if cls == 'station' else None,
                                     label=cls.replace('_', ' ').title()))

    if legend:
        _lgnd = ax.legend(handles=legend, handler_map=handler_map,
                          loc='upper right', fontsize=8, title='Infrastructure')
        _lgnd.get_title().set_fontweight('bold')

    _add_north_arrow(ax)
    _add_scale_bar(ax)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        print(f"  Saved → {output_path}")
    return fig


def plot_gauge_map(
    network: NetworkData,
    extent=None,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12),
    show_outside: bool = False,
    is_catchment: bool = False,
) -> plt.Figure:
    """OpenRailwayMap-style gauge map."""
    if title is None:
        title = f"Gauge Map — {network.version}"
    fig, ax = _base_plot(network, title, figsize, extent=extent)
    ax.set_facecolor('#f5f5f5')
    if show_outside:
        _plot_lakes(ax, extent=extent)
    else:
        _plot_lakes(ax, boundary=network.boundary)

    GAUGE_LABELS = {
        1435: '1435 mm (Standard)',
        1668: '1668 mm (Iberian)',
        1520: '1520 mm (Russian)',
        1000: '1000 mm (Metre)',
        900:  '900 mm',
        800:  '800 mm',
        750:  '750 mm',
        600:  '600 mm',
    }

    segs_all = _clip_to_extent(network.segments, extent)
    segs_in  = (_clip_to_boundary(network.segments, network.boundary)
                if (show_outside and network.boundary is not None
                    and not network.boundary.empty)
                else None)
    segs     = _clip_to_boundary(network.segments, network.boundary)

    gauges_present = set()
    if show_outside and segs_in is not None:
        # Ghost pass
        if len(segs_all) > 0:
            for gauge, color in GAUGE_COLORS.items():
                mask = segs_all['gauge'] == gauge
                if mask.any():
                    gauges_present.add(gauge)
                    segs_all[mask].plot(ax=ax, color=color, linewidth=2.5, alpha=0.40)
            unknown = ~segs_all['gauge'].isin(GAUGE_COLORS)
            if unknown.any():
                segs_all[unknown].plot(ax=ax, color=GAUGE_DEFAULT, linewidth=2, alpha=0.40)
                gauges_present.add('unknown')
        # Solid pass
        if len(segs_in) > 0:
            for gauge, color in GAUGE_COLORS.items():
                mask = segs_in['gauge'] == gauge
                if mask.any():
                    segs_in[mask].plot(ax=ax, color=color, linewidth=2.5, alpha=1.0)
            unknown = ~segs_in['gauge'].isin(GAUGE_COLORS)
            if unknown.any():
                segs_in[unknown].plot(ax=ax, color=GAUGE_DEFAULT, linewidth=2, alpha=1.0)
    else:
        if len(segs) > 0:
            for gauge, color in GAUGE_COLORS.items():
                mask = segs['gauge'] == gauge
                if mask.any():
                    gauges_present.add(gauge)
                    segs[mask].plot(ax=ax, color=color, linewidth=2.5, alpha=0.85)
            unknown = ~segs['gauge'].isin(GAUGE_COLORS)
            if unknown.any():
                segs[unknown].plot(ax=ax, color=GAUGE_DEFAULT, linewidth=2, alpha=0.7)
                gauges_present.add('unknown')

    legend = []
    for gauge, color in GAUGE_COLORS.items():
        if gauge in gauges_present:
            legend.append(Line2D([0], [0], color=color, linewidth=3,
                                 label=GAUGE_LABELS.get(gauge, f'{gauge} mm')))
    if 'unknown' in gauges_present:
        legend.append(Line2D([0], [0], color=GAUGE_DEFAULT, linewidth=2, label='Unknown'))
    _lgnd = ax.legend(handles=legend, loc='upper right', fontsize=10,
                      title='Track Gauge', title_fontsize=11)
    _lgnd.get_title().set_fontweight('bold')

    _add_north_arrow(ax)
    _add_scale_bar(ax)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        print(f"  Saved → {output_path}")
    return fig


def plot_electrification_map(
    network: NetworkData,
    extent=None,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12),
    show_outside: bool = False,
    is_catchment: bool = False,
) -> plt.Figure:
    """OpenRailwayMap-style electrification map (English legend, German-value-aware)."""
    if title is None:
        title = f"Electrification — {network.version}"
    fig, ax = _base_plot(network, title, figsize, extent=extent)
    ax.set_facecolor('#f5f5f5')

    if show_outside:
        _plot_lakes(ax, extent=extent)
    else:
        _plot_lakes(ax, boundary=network.boundary)

    segs_all = _clip_to_extent(network.segments, extent)
    segs_in  = (_clip_to_boundary(network.segments, network.boundary)
                if (show_outside and network.boundary is not None and not network.boundary.empty)
                else None)
    segs     = _clip_to_boundary(network.segments, network.boundary)

    cats_present = set()
    
    if show_outside and segs_in is not None:
        if len(segs_all) > 0 and 'electrification' in segs_all.columns:
            elec_cat_all = segs_all['electrification'].apply(_classify_electrification)
            for cat, color in ELECTRIFICATION_COLORS.items():
                mask = elec_cat_all == cat
                if mask.any():
                    cats_present.add(cat)
                    lw = 2 if cat == 'no_electrification' else 3
                    segs_all[mask].plot(ax=ax, color=color, linewidth=lw, alpha=0.40, zorder=2)
                    
        if len(segs_in) > 0 and 'electrification' in segs_in.columns:
            elec_cat_in = segs_in['electrification'].apply(_classify_electrification)
            for cat, color in ELECTRIFICATION_COLORS.items():
                mask = elec_cat_in == cat
                if mask.any():
                    cats_present.add(cat)
                    lw = 2 if cat == 'no_electrification' else 3
                    segs_in[mask].plot(ax=ax, color=color, linewidth=lw, alpha=0.85, zorder=3)
    else:
        if len(segs) > 0 and 'electrification' in segs.columns:
            elec_cat = segs['electrification'].apply(_classify_electrification)
            for cat, color in ELECTRIFICATION_COLORS.items():
                mask = elec_cat == cat
                if mask.any():
                    cats_present.add(cat)
                    lw = 2 if cat == 'no_electrification' else 3
                    segs[mask].plot(ax=ax, color=color, linewidth=lw, alpha=0.85, zorder=2)


    legend = [
        Line2D([0], [0], color=ELECTRIFICATION_COLORS[cat], linewidth=3,
               label=ELECTRIFICATION_LABELS[cat])
        for cat in ELECTRIFICATION_COLORS
        if cat in cats_present
    ]
    _lgnd = ax.legend(handles=legend, loc='upper right', fontsize=9,
                      title='Electrification', title_fontsize=10)
    _lgnd.get_title().set_fontweight('bold')

    _add_north_arrow(ax)
    _add_scale_bar(ax)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        print(f"  Saved → {output_path}")
    return fig


def plot_construct_type_map(
    network: NetworkData,
    extent=None,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12),
    show_outside: bool = False,
    is_catchment: bool = False,
) -> plt.Figure:
    """Tunnel / bridge / gallery construct type map."""
    if title is None:
        title = f"Construct Type — {network.version}"
    fig, ax = _base_plot(network, title, figsize, extent=extent)
    ax.set_facecolor('#f5f5f5')
    
    if show_outside:
        _plot_lakes(ax, extent=extent)
    else:
        _plot_lakes(ax, boundary=network.boundary)

    segs     = _clip_to_boundary(network.segments, network.boundary)

    construct_present = set()
    if len(segs) > 0 and 'construct_type' in segs.columns:
        for ctype, color in CONSTRUCT_COLORS.items():
            mask = segs['construct_type'] == ctype
            if mask.any():
                construct_present.add(ctype)
                lw = 3 if ctype in ('tunnel', 'bridge') else 2
                segs[mask].plot(ax=ax, color=color, linewidth=lw, alpha=0.85)
        unknown = ~segs['construct_type'].isin(CONSTRUCT_COLORS)
        if unknown.any():
            segs[unknown].plot(ax=ax, color=CONSTRUCT_DEFAULT, linewidth=2, alpha=0.7)
            construct_present.add('unknown')
    else:
        segs.plot(ax=ax, color='gray', linewidth=2, alpha=0.7)

    if construct_present:
        legend = []
        for ctype, color in CONSTRUCT_COLORS.items():
            if ctype in construct_present:
                legend.append(Line2D([0], [0], color=color, linewidth=3,
                                     label=ctype.title()))
        if 'unknown' in construct_present:
            legend.append(Line2D([0], [0], color=CONSTRUCT_DEFAULT, linewidth=2,
                                 label='Unknown'))
        _lgnd = ax.legend(handles=legend, loc='upper right', fontsize=10,
                          title='Construct Type', title_fontsize=11)
        _lgnd.get_title().set_fontweight('bold')

    _add_north_arrow(ax)
    _add_scale_bar(ax)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        print(f"  Saved → {output_path}")
    return fig


# =============================================================================
# Canonical Infrastructure Plot
# =============================================================================

class _MultiTrackLegendHandle:
    """Placeholder object for rendering multi-track legend entries."""

    def __init__(self, total_width: float, track_count: int, color: str = "black"):
        self.total_width = total_width
        self.track_count = track_count
        self.color = color


class _MultiTrackLegendHandler(HandlerBase):
    """Custom legend handler that renders multiple parallel lines for tracks."""

    def create_artists(
        self,
        legend,
        orig_handle: "_MultiTrackLegendHandle",
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ):
        x0 = xdescent
        x1 = xdescent + width
        y = ydescent + height / 2.0

        track_count = orig_handle.track_count
        gap_factor = 0.4
        
        if track_count == 1:
            individual_line_width = orig_handle.total_width
            line_spacing = 0
        else:
            individual_line_width = orig_handle.total_width / (track_count + (track_count - 1) * gap_factor)
            line_spacing = individual_line_width * (1 + gap_factor)

        artists = []
        for track_idx in range(track_count):
            offset_y = y + (track_idx - (track_count - 1) / 2.0) * line_spacing
            line = Line2D([x0, x1], [offset_y, offset_y], color=orig_handle.color, linewidth=individual_line_width, solid_capstyle="round")
            line.set_transform(trans)
            artists.append(line)

        return artists


_TRACK_SPACING_M_CA = 140   # CA track spacing
_TRACK_SPACING_M_SA = 60    # SA track spacing


def _draw_parallel_tracks(ax, geom, num_tracks: int,
                           color: str = 'black', linewidth: float = 1.05, alpha: float = 1.0, zorder: int = 3, track_spacing_m: float = 30, linestyle: str = 'solid') -> None:
    """Draw num_tracks parallel offset lines for one segment geometry."""
    if num_tracks <= 0 or geom is None or geom.is_empty:
        return
    sub_lines = list(geom.geoms) if geom.geom_type == 'MultiLineString' else [geom]
    offsets = [(i - (num_tracks - 1) / 2) * track_spacing_m
               for i in range(num_tracks)]

    for line in sub_lines:
        if line.length < 1:
            continue
        for offset_m in offsets:
            if abs(offset_m) < 0.01:
                draw_line = line
            else:
                side = 'left' if offset_m > 0 else 'right'
                try:
                    draw_line = line.parallel_offset(
                        abs(offset_m), side, resolution=8,
                        join_style=2, mitre_limit=5,
                    )
                    if draw_line is None or draw_line.is_empty:
                        draw_line = line
                except Exception:
                    draw_line = line

            parts = (list(draw_line.geoms)
                     if draw_line.geom_type == 'MultiLineString' else [draw_line])
            for part in parts:
                if not part.is_empty and len(part.coords) >= 2:
                    xs, ys = zip(*[(c[0], c[1]) for c in part.coords])
                    ax.plot(xs, ys, color=color, linewidth=linewidth, linestyle=linestyle,
                            solid_capstyle='butt', alpha=alpha, zorder=zorder)


def _draw_parallel_tracks_mixed(
    ax, geom, colors: list,
    linewidth: float = 1.05, alpha: float = 1.0,
    zorder: int = 3, track_spacing_m: float = 30,
) -> None:
    """Like _draw_parallel_tracks but each track gets its own colour.

    len(colors) determines how many parallel lines are drawn.
    The outermost track (last offset) gets the last colour — use this for the
    diff track (green = gained, red = lost).
    """
    num_tracks = len(colors)
    if num_tracks == 0 or geom is None or geom.is_empty:
        return
    sub_lines = list(geom.geoms) if geom.geom_type == 'MultiLineString' else [geom]
    offsets = [(i - (num_tracks - 1) / 2) * track_spacing_m for i in range(num_tracks)]

    for line in sub_lines:
        if line.length < 1:
            continue
        for offset_m, color in zip(offsets, colors):
            if abs(offset_m) < 0.01:
                draw_line = line
            else:
                side = 'left' if offset_m > 0 else 'right'
                try:
                    draw_line = line.parallel_offset(
                        abs(offset_m), side, resolution=8,
                        join_style=2, mitre_limit=5,
                    )
                    if draw_line is None or draw_line.is_empty:
                        draw_line = line
                except Exception:
                    draw_line = line

            parts = (list(draw_line.geoms)
                     if draw_line.geom_type == 'MultiLineString' else [draw_line])
            for part in parts:
                if not part.is_empty and len(part.coords) >= 2:
                    xs, ys = zip(*[(c[0], c[1]) for c in part.coords])
                    ax.plot(xs, ys, color=color, linewidth=linewidth,
                            solid_capstyle='butt', alpha=alpha, zorder=zorder)


def plot_infrastructure_canonical(
    network: NetworkData,
    extent=None,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12),
    show_labels: bool = True,
    is_catchment: bool = False,
    show_outside: bool = False,
) -> plt.Figure:
    """Infrastructure overview: black parallel tracks by count,
    white station circles with CODE labels."""
    if title is None:
        title = f"Infrastructure — {network.version}"
    fig, ax = _base_plot(network, title, figsize, extent=extent)
    if show_outside:
        _plot_lakes(ax, extent=extent)
    else:
        _plot_lakes(ax, boundary=network.boundary)

    segs_all = _clip_to_extent(network.segments, extent)
    segs_in  = (_clip_to_boundary(network.segments, network.boundary)
                if (show_outside and network.boundary is not None and not network.boundary.empty)
                else None)
    segs  = _clip_to_boundary(network.segments, network.boundary)

    tracks_present = set()

    # Segments as back parallel lines (one line per track)
    if show_outside and segs_in is not None:
        if len(segs_all) > 0:
            for _, row in segs_all.iterrows():
                n = int(row.get('num_tracks', 1) or 1)
                n = max(1, min(n, 4))
                tracks_present.add(n)
                _draw_parallel_tracks(ax, row.geometry, n, color='black', alpha=0.40, zorder=2, track_spacing_m=_TRACK_SPACING_M_CA if is_catchment else _TRACK_SPACING_M_SA)
        if len(segs_in) > 0:
            for _, row in segs_in.iterrows():
                n = int(row.get('num_tracks', 1) or 1)
                n = max(1, min(n, 4))
                tracks_present.add(n)
                _draw_parallel_tracks(ax, row.geometry, n, color='black', alpha=1.0, zorder=3, track_spacing_m=_TRACK_SPACING_M_CA if is_catchment else _TRACK_SPACING_M_SA)
    else:
        if len(segs) > 0:
            for _, row in segs.iterrows():
                n = int(row.get('num_tracks', 1) or 1)
                n = max(1, min(n, 4))
                tracks_present.add(n)
                _draw_parallel_tracks(ax, row.geometry, n, color='black', alpha=1.0, zorder=3, track_spacing_m=_TRACK_SPACING_M_CA if is_catchment else _TRACK_SPACING_M_SA)

    # Nodes: three layers via _classify_nodes
    ts_all, tf_all, jn_all = _classify_nodes(network.nodes)

    if show_outside and network.boundary is not None and not network.boundary.empty:
        nodes_in    = _clip_to_boundary(network.nodes, network.boundary)
        ts_in, tf_in, jn_in = _classify_nodes(nodes_in)
        # Ghost pass — full network at 40%
        _ms_ts = 20  if is_catchment else 60
        _ms_tf = 5   if is_catchment else 25
        _ms_jn = 5  if is_catchment else 5
        
        if len(ts_all) > 0:
            ts_all.plot(ax=ax, facecolor='white', edgecolor='black',
                        markersize=_ms_ts, marker='o', linewidth=1.2,
                        alpha=0.40, zorder=4)
        if len(tf_all) > 0:
            tf_all.plot(ax=ax, facecolor='black', edgecolor='black',
                        markersize=_ms_tf, marker='o', linewidth=0.8,
                        alpha=0.40, zorder=4)
        if len(jn_all) > 0:
            jn_all.plot(ax=ax, color='#888888', markersize=_ms_jn, marker='o',
                        alpha=0.40, zorder=4)
        # Solid pass — clipped network at full opacity
        ts_draw, tf_draw, jn_draw = ts_in, tf_in, jn_in
    else:
        nodes_disp = _clip_to_boundary(network.nodes, network.boundary)
        ts_draw, tf_draw, jn_draw = _classify_nodes(nodes_disp)

    _ms_ts = 20  if is_catchment else 60
    _ms_tf = 5   if is_catchment else 25
    _ms_jn = 5  if is_catchment else 5

    if len(ts_draw) > 0:
        ts_draw.plot(ax=ax, facecolor='white', edgecolor='black',
                     markersize=_ms_ts, marker='o', linewidth=1.2,
                     alpha=1.0, zorder=5)
        if show_labels and not is_catchment:
            for _, row in ts_draw.iterrows():
                code = row.get('CODE', '')
                if pd.notna(code) and str(code).strip():
                    ax.annotate(
                        str(code),
                        xy=(row.geometry.x, row.geometry.y),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=7, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                  edgecolor='none', alpha=0.7),
                        zorder=6,
                    )

    if len(tf_draw) > 0:
        tf_draw.plot(ax=ax, facecolor='black', edgecolor='black',
                     markersize=_ms_tf, marker='o', linewidth=0.8,
                     alpha=1.0, zorder=5)

    if len(jn_draw) > 0:
        jn_draw.plot(ax=ax, color='#888888', markersize=_ms_jn, marker='o',
                     alpha=1.0, zorder=4)

    legend = []
    handler_map = {}
    
    # Add track styles based on what was plotted
    for n in sorted(tracks_present):
        label_text = f'{n} track{"s" if n > 1 else ""}'
        if n == 4:
            label_text = '4+ tracks'
        
        # Increase total visual width for handler map sizing
        agg_linewidth = 1.5 if n == 1 else n * 1.35 
        handle = Line2D([0], [0], color='black', linewidth=agg_linewidth, label=label_text)
        legend.append(handle)
        handler_map[handle] = _MultiTrackLegendHandler()
        handle.track_count = n
        handle.total_width = agg_linewidth
        handle.color = 'black'
        
    has_ts = len(ts_all) > 0 if show_outside else len(ts_draw) > 0
    if has_ts:
        legend.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                             markeredgecolor='black', markersize=8, label='Train station'))
    
    has_tf = len(tf_all) > 0 if show_outside else len(tf_draw) > 0
    if has_tf:
        legend.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                             markeredgecolor='black', markersize=6, label='Tram / Funicular'))

    has_jn = len(jn_all) > 0 if show_outside else len(jn_draw) > 0
    if has_jn:
        legend.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#888888',
                   markersize=5, label='Junction / turning loop')
        )
    
    if legend:
        ax.legend(handles=legend, handler_map=handler_map, loc='upper right', fontsize=8)
    _add_north_arrow(ax)
    _add_scale_bar(ax)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        print(f"  Saved → {output_path}")
    return fig


# =============================================================================
# Diff Plot
# =============================================================================

_DIFF_RED    = '#d62728'
_DIFF_GREEN  = '#2ca02c'
_DIFF_GREY   = '#aaaaaa'
_DIFF_YELLOW = '#B8860B'  # dark goldenrod — modified elements


def plot_infrastructure_diff(
    net_a: NetworkData,
    net_b: NetworkData,
    extent=None,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12),
    show_labels: bool = True,
    is_catchment: bool = False,
) -> plt.Figure:
    """Diff plot: net_a is the reference, net_b is the comparison.

    Segments
    --------
    - Removed (in A, not in B)          : red tracks
    - Added   (in B, not in A)          : green tracks
    - Track gained (n_b > n_a)          : n_b tracks drawn; outermost = green
    - Track lost   (n_b < n_a)          : max(n_a,n_b)+1 tracks drawn; outermost = red
    - Unchanged                         : black tracks, low opacity

    Nodes
    -----
    - Removed : red markers
    - Added   : green markers
    - Unchanged: grey markers, low opacity
    """
    if title is None:
        title = f"Infrastructure diff — {net_b.version} vs {net_a.version}"

    # Use net_b's boundary/graph for the base plot frame
    fig, ax = _base_plot(net_b, title, figsize, extent=extent)
    if is_catchment:
        _plot_lakes(ax, boundary=net_b.boundary)
    else:
        _plot_lakes(ax, extent=extent)

    ts = _TRACK_SPACING_M_CA if is_catchment else _TRACK_SPACING_M_SA

    # ── Clip ──────────────────────────────────────────────────────────────────
    def _clip(gdf, network):
        if extent is not None:
            return _clip_to_extent(gdf, extent)
        return _clip_to_boundary(gdf, network.boundary)

    segs_a = _clip(net_a.segments, net_a)
    segs_b = _clip(net_b.segments, net_b)
    nodes_a = _clip(net_a.nodes, net_a)
    nodes_b = _clip(net_b.nodes, net_b)

    # ── Segment diff ──────────────────────────────────────────────────────────
    ids_a = set(segs_a['segment_id'].dropna())
    ids_b = set(segs_b['segment_id'].dropna())

    removed_segs  = segs_a[segs_a['segment_id'].isin(ids_a - ids_b)]
    added_segs    = segs_b[segs_b['segment_id'].isin(ids_b - ids_a)]
    common_ids    = ids_a & ids_b

    # For common segments compare num_tracks
    common_a = segs_a[segs_a['segment_id'].isin(common_ids)].set_index('segment_id')
    common_b = segs_b[segs_b['segment_id'].isin(common_ids)].set_index('segment_id')

    _SEG_MOD_ATTRS = (
        'gauge', 'electrification', 'average_speed',
        'km_start', 'km_end', 'route_number', 'route_name', 'route_owner',
    )

    def _norm(v):
        try:
            return '' if pd.isna(v) else str(v)
        except (TypeError, ValueError):
            return str(v)

    unchanged_ids   = []
    modified_ids    = []
    track_gained    = []   # list of (geom, n_a, n_b)
    track_lost      = []   # list of (geom, n_a, n_b)

    for sid in common_ids:
        if sid not in common_a.index or sid not in common_b.index:
            continue
        n_a  = int(common_a.loc[sid].get('num_tracks', 1) or 1)
        n_b  = int(common_b.loc[sid].get('num_tracks', 1) or 1)
        geom = common_b.loc[sid].geometry
        if n_b > n_a:
            track_gained.append((geom, n_a, n_b))
        elif n_b < n_a:
            track_lost.append((geom, n_a, n_b))
        else:
            geom_changed = not common_a.loc[sid].geometry.equals(geom)
            attr_changed = any(
                _norm(common_a.loc[sid].get(a)) != _norm(common_b.loc[sid].get(a))
                for a in _SEG_MOD_ATTRS
            )
            if geom_changed or attr_changed:
                modified_ids.append(sid)
            else:
                unchanged_ids.append(sid)

    unchanged_segs = segs_b[segs_b['segment_id'].isin(unchanged_ids)]
    modified_segs  = segs_b[segs_b['segment_id'].isin(modified_ids)]

    # ── Draw segments ─────────────────────────────────────────────────────────
    # 1) Unchanged — black, faded
    for _, row in unchanged_segs.iterrows():
        n = int(row.get('num_tracks', 1) or 1)
        _draw_parallel_tracks(ax, row.geometry, n, color='black',
                              linewidth=1.05, alpha=0.20, zorder=2,
                              track_spacing_m=ts)

    # 2) Modified — dark yellow, above unchanged
    for _, row in modified_segs.iterrows():
        n = int(row.get('num_tracks', 1) or 1)
        _draw_parallel_tracks(ax, row.geometry, n, color=_DIFF_YELLOW,
                              linewidth=1.4, alpha=0.9, zorder=3,
                              track_spacing_m=ts)

    # 3) Removed — red, full opacity
    for _, row in removed_segs.iterrows():
        n = int(row.get('num_tracks', 1) or 1)
        _draw_parallel_tracks(ax, row.geometry, n, color=_DIFF_RED,
                              linewidth=1.4, alpha=0.9, zorder=4,
                              track_spacing_m=ts)

    # 4) Added — green, full opacity
    for _, row in added_segs.iterrows():
        n = int(row.get('num_tracks', 1) or 1)
        _draw_parallel_tracks(ax, row.geometry, n, color=_DIFF_GREEN,
                              linewidth=1.4, alpha=0.9, zorder=4,
                              track_spacing_m=ts)

    # 4) Track gained: draw n_b tracks, outermost = green
    for geom, n_a, n_b in track_gained:
        colors = ['black'] * n_a + [_DIFF_GREEN] * (n_b - n_a)
        _draw_parallel_tracks_mixed(ax, geom, colors,
                                    linewidth=1.2, alpha=0.9, zorder=3,
                                    track_spacing_m=ts)

    # 5) Track lost: draw n_a tracks (the original count), outermost = red
    for geom, n_a, n_b in track_lost:
        colors = ['black'] * n_b + [_DIFF_RED] * (n_a - n_b)
        _draw_parallel_tracks_mixed(ax, geom, colors,
                                    linewidth=1.2, alpha=0.9, zorder=3,
                                    track_spacing_m=ts)

    # ── Node diff ─────────────────────────────────────────────────────────────
    names_a = set(nodes_a['NAME'].dropna()) if not nodes_a.empty else set()
    names_b = set(nodes_b['NAME'].dropna()) if not nodes_b.empty else set()

    removed_nodes = nodes_a[nodes_a['NAME'].isin(names_a - names_b)]
    added_nodes   = nodes_b[nodes_b['NAME'].isin(names_b - names_a)]

    # Split common nodes into unchanged vs modified (geometry or node_class changed)
    _NODE_MOD_ATTRS = ('node_class',)
    modified_node_names  = []
    unchanged_node_names = []
    if not nodes_a.empty and not nodes_b.empty:
        for _name in names_a & names_b:
            _ra = nodes_a[nodes_a['NAME'] == _name]
            _rb = nodes_b[nodes_b['NAME'] == _name]
            if _ra.empty or _rb.empty:
                unchanged_node_names.append(_name)
                continue
            _ra, _rb = _ra.iloc[0], _rb.iloc[0]
            _geom_changed = not _ra.geometry.equals(_rb.geometry)
            _attr_changed = any(
                _norm(_ra.get(a)) != _norm(_rb.get(a)) for a in _NODE_MOD_ATTRS
            )
            if _geom_changed or _attr_changed:
                modified_node_names.append(_name)
            else:
                unchanged_node_names.append(_name)
    else:
        unchanged_node_names = list(names_a & names_b)

    modified_nodes  = nodes_b[nodes_b['NAME'].isin(modified_node_names)]
    unchanged_nodes = nodes_b[nodes_b['NAME'].isin(unchanged_node_names)]

    ms_ts = 20 if is_catchment else 55
    ms_jn = 5  if is_catchment else 8

    def _plot_node_set(nodes_gdf, facecolor, edgecolor, alpha, zorder):
        if nodes_gdf.empty:
            return
        ts_n, tf_n, jn_n = _classify_nodes(nodes_gdf)
        for gdf in (ts_n, tf_n):
            if not gdf.empty:
                gdf.plot(ax=ax, facecolor=facecolor, edgecolor=edgecolor,
                         markersize=ms_ts, marker='o', linewidth=1.2,
                         alpha=alpha, zorder=zorder)
        if not jn_n.empty:
            jn_n.plot(ax=ax, color=facecolor, markersize=ms_jn, marker='o',
                      alpha=alpha, zorder=zorder - 1)

    _plot_node_set(unchanged_nodes, _DIFF_GREY,   _DIFF_GREY,   0.25, 5)
    _plot_node_set(modified_nodes,  _DIFF_YELLOW, '#7a6000',    0.9,  6)
    _plot_node_set(removed_nodes,   _DIFF_RED,    '#7f0000',    0.9,  7)
    _plot_node_set(added_nodes,     _DIFF_GREEN,  '#005a00',    0.9,  7)

    # ── Labels (added/removed/modified stations only) ─────────────────────────
    if show_labels and not is_catchment:
        for nodes_gdf, color in ((added_nodes,    _DIFF_GREEN),
                                 (removed_nodes,  _DIFF_RED),
                                 (modified_nodes, _DIFF_YELLOW)):
            ts_n, tf_n, _ = _classify_nodes(nodes_gdf)
            for _, row in pd.concat([ts_n, tf_n]).iterrows():
                code = row.get('CODE', '')
                if pd.notna(code) and str(code).strip():
                    ax.annotate(
                        str(code),
                        xy=(row.geometry.x, row.geometry.y),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=7, fontweight='bold', color=color,
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                  edgecolor='none', alpha=0.7),
                        zorder=8,
                    )

    # ── Legend ────────────────────────────────────────────────────────────────
    legend = [Line2D([0], [0], color='none', label=r'$\bf{Segments}$')]
    legend.append(Line2D([0], [0], color='black',      linewidth=1.0, alpha=0.4,
                         label='Unchanged'))
    legend.append(Line2D([0], [0], color=_DIFF_YELLOW, linewidth=1.4,
                         label='Modified'))
    legend.append(Line2D([0], [0], color=_DIFF_GREEN,  linewidth=1.4,
                         label='Added'))
    legend.append(Line2D([0], [0], color=_DIFF_RED,    linewidth=1.4,
                         label='Removed'))

    legend.append(Line2D([0], [0], color='none', label=r'$\bf{Nodes}$'))
    legend.append(Line2D([0], [0], marker='o', color='w',
                         markerfacecolor=_DIFF_GREY, markersize=6,
                         alpha=0.5, label='Unchanged'))
    legend.append(Line2D([0], [0], marker='o', color='w',
                         markerfacecolor=_DIFF_YELLOW,
                         markeredgecolor='#7a6000', markersize=8,
                         label='Modified'))
    legend.append(Line2D([0], [0], marker='o', color='w',
                         markerfacecolor=_DIFF_GREEN,
                         markeredgecolor='#005a00', markersize=8,
                         label='Added'))
    legend.append(Line2D([0], [0], marker='o', color='w',
                         markerfacecolor=_DIFF_RED,
                         markeredgecolor='#7f0000', markersize=8,
                         label='Removed'))

    ax.legend(handles=legend, loc='upper right', fontsize=8)
    _add_north_arrow(ax)
    _add_scale_bar(ax)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        print(f"  Saved → {output_path}")
    return fig


# =============================================================================
# Construction Components Plot
# =============================================================================

_CONSTRUCT_TRACK_COLOR  = '#FF6600'   # orange — OpenRailwayMap track colour
_CONSTRUCT_BRIDGE_RAIL  = '#000000'   # black border lines for bridges
_CONSTRUCT_PORTAL_COLOR = '#000000'   # black portal bars for tunnels
_BRIDGE_OFFSET_M = 30                 # offset of bridge border lines from centre (m)
_PORTAL_BAR_M  = 4 * _BRIDGE_OFFSET_M # = 120 m — portal spans 4× bridge half-width


def _draw_geom(ax, geom, color, linewidth, linestyle='solid', zorder=3,
               alpha: float = 1.0) -> None:
    """Draw a LineString or MultiLineString on ax."""
    if geom is None or geom.is_empty:
        return
    parts = list(geom.geoms) if geom.geom_type == 'MultiLineString' else [geom]
    for part in parts:
        if not part.is_empty and len(part.coords) >= 2:
            xs, ys = zip(*part.coords)
            ax.plot(xs, ys, color=color, linewidth=linewidth, linestyle=linestyle,
                    solid_capstyle='butt', zorder=zorder, alpha=alpha)


def _draw_tunnel_portals(ax, geom, num_tracks: int = 1, track_spacing_m: float = 60, alpha: float = 1.0, zorder: int = 6) -> None:
    """Draw perpendicular portal bars at both ends of a tunnel piece."""
    if geom is None or geom.is_empty:
        return
    if geom.geom_type == 'MultiLineString':
        all_coords = [c for sub in geom.geoms for c in sub.coords]
    else:
        all_coords = list(geom.coords)
    if len(all_coords) < 2:
        return

    for i, (pt, ref) in enumerate(((all_coords[0], all_coords[1]),
                                   (all_coords[-1], all_coords[-2]))):
        dx, dy = ref[0] - pt[0], ref[1] - pt[1]
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1e-6:
            continue
        ux, uy = dx / length, dy / length          # Inward pointing unit vector
        px, py = -uy, ux                           # Perpendicular
        
        width = (num_tracks - 1) * track_spacing_m + 2 * _BRIDGE_OFFSET_M
        h = width / 2.0
        depth = width / 3.0
        
        p1 = (pt[0] - px * h - ux * depth, pt[1] - py * h - uy * depth)
        p2 = (pt[0] - px * h, pt[1] - py * h)
        p3 = (pt[0] + px * h, pt[1] + py * h)
        p4 = (pt[0] + px * h - ux * depth, pt[1] + py * h - uy * depth)

        ax.plot(
            [p1[0], p2[0]], [p1[1], p2[1]],
            color=_CONSTRUCT_PORTAL_COLOR, linewidth=2.5,
            solid_capstyle='butt', zorder=zorder, alpha=alpha,
        )
        ax.plot(
            [p3[0], p4[0]], [p3[1], p4[1]],
            color=_CONSTRUCT_PORTAL_COLOR, linewidth=2.5,
            solid_capstyle='butt', zorder=zorder, alpha=alpha,
        )


def plot_construction_components(
    network: NetworkData,
    composition: gpd.GeoDataFrame,
    extent=None,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12),
    show_outside: bool = False,
) -> plt.Figure:
    """Study-area construction components map (OpenRailwayMap-inspired).

    Normal  — solid orange track
    Bridge  — solid orange track + parallel black border lines on each side
    Tunnel  — dashed orange track + perpendicular portal bar at each end
    Gallery — same treatment as tunnel
    """
    composition_all = gpd.GeoDataFrame()
    composition_in = None
    if not composition.empty:
        if 'num_tracks' not in composition.columns and 'segment_id' in composition.columns and not network.segments.empty:
            composition = composition.merge(network.segments[['segment_id', 'num_tracks']], on='segment_id', how='left')

        composition_all = composition.copy()
        composition_in = (_clip_to_boundary(composition, network.boundary)
                          if show_outside and network.boundary is not None and not network.boundary.empty
                          else None)
        composition = _clip_to_boundary(composition, network.boundary)

    if title is None:
        title = f"Construction Components — {network.version}"
    fig, ax = _base_plot(network, title, figsize, extent=extent)

    if show_outside:
        _plot_lakes(ax, extent=extent)
    else:
        _plot_lakes(ax, boundary=network.boundary)

    legend_handles: list = []
    legend_seen:    set  = set()

    def _add_legend(label, color, lw, ls='solid'):
        if label not in legend_seen:
            legend_seen.add(label)
            legend_handles.append(
                Line2D([0], [0], color=color, linewidth=lw,
                       linestyle=ls, label=label)
            )

    def _plot_comp_pass(comp_df, alpha, z_base):
        if comp_df.empty or 'construct_type' not in comp_df.columns:
            return
        for _, piece in comp_df.iterrows():
            geom  = piece.geometry
            if geom is None or geom.is_empty:
                continue
            ctype = str(piece.get('construct_type', 'normal')).lower()
            try:
                nt = piece.get('num_tracks', 1)
                num_tracks = int(float(nt)) if pd.notna(nt) else 1
            except (ValueError, TypeError):
                num_tracks = 1
            num_tracks = max(1, min(num_tracks, 4))
            
            # Construct type map only generated for sa
            track_dist = _TRACK_SPACING_M_SA 

            if ctype == 'normal':
                _draw_parallel_tracks(ax, geom, num_tracks, color=_CONSTRUCT_TRACK_COLOR, linewidth=1.1, zorder=z_base, alpha=alpha, track_spacing_m=track_dist)
                if alpha == 1.0: _add_legend('Normal track', _CONSTRUCT_TRACK_COLOR, 2)

            elif ctype == 'bridge':
                _draw_parallel_tracks(ax, geom, num_tracks, color=_CONSTRUCT_TRACK_COLOR, linewidth=1.1, zorder=z_base+1, alpha=alpha, track_spacing_m=track_dist)
                
                # Push the bridge rail to the extremeties of the N tracks
                outer_offset = ((num_tracks - 1) / 2.0) * track_dist + _BRIDGE_OFFSET_M
                for side in ('left', 'right'):
                    try:
                        sub_lines = list(geom.geoms) if geom.geom_type == 'MultiLineString' else [geom]
                        for sub in sub_lines:
                            border = sub.parallel_offset(outer_offset, side, resolution=8, join_style=2, mitre_limit=5)
                            if border and not border.is_empty:
                                _draw_geom(ax, border, _CONSTRUCT_BRIDGE_RAIL, 1.0, zorder=z_base+2, alpha=alpha)
                    except Exception:
                        pass
                if alpha == 1.0:
                    _add_legend('Bridge (track)', _CONSTRUCT_TRACK_COLOR, 2)
                    if 'Bridge (border)' not in legend_seen:
                        legend_seen.add('Bridge (border)')
                        legend_handles.append(
                            Line2D([0], [0], color=_CONSTRUCT_BRIDGE_RAIL,
                                   linewidth=2.0, label='Bridge (border rail)')
                        )

            elif ctype in ('tunnel', 'gallery'):
                _draw_parallel_tracks(ax, geom, num_tracks, color=_CONSTRUCT_TRACK_COLOR, linewidth=1.1,
                           linestyle='dashed', zorder=z_base, alpha=alpha, track_spacing_m=track_dist)
                _draw_tunnel_portals(ax, geom, num_tracks=num_tracks, track_spacing_m=track_dist, alpha=alpha, zorder=z_base+3)
                if alpha == 1.0:
                    lbl = 'Tunnel' if ctype == 'tunnel' else 'Gallery'
                    _add_legend(lbl, _CONSTRUCT_TRACK_COLOR, 2, ls='dashed')

            else:
                _draw_parallel_tracks(ax, geom, num_tracks, color='#888888', linewidth=1.2, zorder=max(1, z_base-1), alpha=alpha, track_spacing_m=track_dist)

    if show_outside and composition_in is not None:
        _plot_comp_pass(composition_all, alpha=0.40, z_base=2)
        _plot_comp_pass(composition_in, alpha=1.0, z_base=10)
    elif not composition.empty and 'construct_type' in composition.columns:
        _plot_comp_pass(composition, alpha=1.0, z_base=3)
    else:
        if len(network.segments) > 0:
            network.segments.plot(ax=ax, color='#888888', linewidth=1.2, zorder=2)
        print("  Warning: no composition data — falling back to plain segment plot.")

    # Stations overlay
    _ms_ts, _ms_tf, _ms_jn = 60, 20, 10
    
    ts_all, tf_all, jn_all = _classify_nodes(network.nodes)
    if show_outside and network.boundary is not None and not network.boundary.empty:
        ts_draw = _clip_to_boundary(ts_all, network.boundary)
        tf_draw = _clip_to_boundary(tf_all, network.boundary)
        jn_draw = _clip_to_boundary(jn_all, network.boundary)
        
        # Ghost pass — full network at 40%
        if len(ts_all) > 0:
            ts_all.plot(ax=ax, facecolor='white', edgecolor='black',
                        markersize=_ms_ts, marker='o', linewidth=1.2,
                        alpha=0.40, zorder=14)
        if len(tf_all) > 0:
            tf_all.plot(ax=ax, facecolor='black', edgecolor='black',
                        markersize=_ms_tf, marker='o', linewidth=0.8,
                        alpha=0.40, zorder=14)
        if len(jn_all) > 0:
            jn_all.plot(ax=ax, color='#888888', markersize=_ms_jn, marker='o',
                        alpha=0.40, zorder=13)
    else:
        nodes_disp = _clip_to_boundary(network.nodes, network.boundary)
        ts_draw, tf_draw, jn_draw = _classify_nodes(nodes_disp)
        
    if len(ts_draw) > 0:
        ts_draw.plot(ax=ax, facecolor='white', edgecolor='black',
                     markersize=_ms_ts, marker='o', linewidth=1.2,
                     alpha=1.0, zorder=16)
        for _, row in ts_draw.iterrows():
            code = row.get('CODE', '')
            if pd.notna(code) and str(code).strip():
                ax.annotate(
                    str(code),
                    xy=(row.geometry.x, row.geometry.y),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=7, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                              edgecolor='none', alpha=0.7),
                    zorder=17,
                )

    if len(tf_draw) > 0:
        tf_draw.plot(ax=ax, facecolor='black', edgecolor='black',
                     markersize=_ms_tf, marker='o', linewidth=0.8,
                     alpha=1.0, zorder=16)

    if len(jn_draw) > 0:
        jn_draw.plot(ax=ax, color='#888888', markersize=_ms_jn, marker='o',
                     alpha=1.0, zorder=15)

    # Adding legends for nodes
    has_ts = len(ts_all) > 0 if show_outside else len(ts_draw) > 0
    if has_ts:
        legend_handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                             markeredgecolor='black', markersize=8, label='Train station'))
    
    has_tf = len(tf_all) > 0 if show_outside else len(tf_draw) > 0
    if has_tf:
        legend_handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                             markeredgecolor='black', markersize=6, label='Tram / Funicular'))

    has_jn = len(jn_all) > 0 if show_outside else len(jn_draw) > 0
    if has_jn:
        legend_handles.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#888888',
                   markersize=5, label='Junction / turning loop')
        )
    _lgnd = ax.legend(handles=legend_handles, loc='upper right', fontsize=8,
                      title='Construction Components')
    _lgnd.get_title().set_fontweight('bold')
    _add_north_arrow(ax)
    _add_scale_bar(ax)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        print(f"  Saved → {output_path}")
    return fig


# =============================================================================

# =============================================================================
# Owner Map Plot
# =============================================================================

def plot_owner_map(
    network: NetworkData,
    extent=None,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12),
    show_outside: bool = False,
    is_catchment: bool = False,
) -> plt.Figure:
    import matplotlib.cm as cm
    
    if title is None:
        title = f"Owner Map - {network.version}"
    fig, ax = _base_plot(network, title, figsize, extent=extent)
    
    if show_outside:
        _plot_lakes(ax, extent=extent)
    else:
        _plot_lakes(ax, boundary=network.boundary)

    segs_all = _clip_to_extent(network.segments, extent)
    segs_in  = (_clip_to_boundary(network.segments, network.boundary)
                if (show_outside and network.boundary is not None and not network.boundary.empty)
                else None)
    segs = _clip_to_boundary(network.segments, network.boundary)

    # Owners are drawn from catchment-clipped `segs`; `segs_all` used only for
    # background rendering when show_outside=True.
    catchment_segs = segs if len(segs) > 0 else segs_all
    if len(catchment_segs) > 0 and 'route_owner' in catchment_segs.columns:
        # Determine owner set from catchment boundary only
        unique_owners = catchment_segs['route_owner'].dropna().unique()

        colors = {}
        palette_idx = 0
        for o in sorted(unique_owners, key=str):
            if 'SBB' in str(o).upper():
                colors[o] = '#E3000F'
            else:
                colors[o] = _OWNER_PALETTE[palette_idx % len(_OWNER_PALETTE)]
                palette_idx += 1

        plotted_owners = set()

        if show_outside and segs_in is not None:
            for owner, color in colors.items():
                mask_all = segs_all['route_owner'] == owner
                if mask_all.any():
                    plotted_owners.add(owner)
                    segs_all[mask_all].plot(ax=ax, color=color, linewidth=2, alpha=0.40, zorder=2)

            if len(segs_in) > 0:
                for owner, color in colors.items():
                    mask_in = segs_in['route_owner'] == owner
                    if mask_in.any():
                        plotted_owners.add(owner)
                        segs_in[mask_in].plot(ax=ax, color=color, linewidth=2, alpha=0.8, zorder=3)
        else:
            if len(segs) > 0:
                for owner, color in colors.items():
                    mask = segs['route_owner'] == owner
                    if mask.any():
                        plotted_owners.add(owner)
                        segs[mask].plot(ax=ax, color=color, linewidth=2, alpha=0.8, zorder=2)

        legend = [
            Line2D([0], [0], color=colors[own], linewidth=2, label=str(own))
            for own in sorted(plotted_owners, key=str)
        ]
        if len(legend) <= 40:
            _lgnd = ax.legend(handles=legend, loc='upper right', fontsize=8,
                              ncol=2, title='Route Owner')
            _lgnd.get_title().set_fontweight('bold')
            
    _add_north_arrow(ax)
    _add_scale_bar(ax)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        print(f"  Saved -> {output_path}")
    return fig

def plot_speed_map(
    network: NetworkData,
    extent=None,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12),
    show_outside: bool = False,
    is_catchment: bool = False,
) -> plt.Figure:
    """Speed map coloured by predominant OSM maxspeed per segment."""
    if title is None:
        title = f"Speed Map — {network.version}"
    fig, ax = _base_plot(network, title, figsize, extent=extent)
    ax.set_facecolor('#f5f5f5')

    if show_outside:
        _plot_lakes(ax, extent=extent)
    else:
        _plot_lakes(ax, boundary=network.boundary)

    segs_all = _clip_to_extent(network.segments, extent)
    segs_in  = (
        _clip_to_boundary(network.segments, network.boundary)
        if (show_outside and network.boundary is not None and not network.boundary.empty)
        else None
    )
    segs = _clip_to_boundary(network.segments, network.boundary)

    if 'predominant_speed' not in network.segments.columns:
        ax.text(0.5, 0.5, 'No speed data\n(run network builder with OSM enrichment)',
                transform=ax.transAxes, ha='center', va='center', fontsize=12, color='grey')
        _add_north_arrow(ax)
        _add_scale_bar(ax)
        plt.tight_layout()
        if output_path:
            fig.savefig(output_path, bbox_inches='tight')
            print(f"  Saved → {output_path}  (no speed data)")
        return fig

    bins_present: set = set()

    def _plot_segs(gdf, alpha):
        if gdf is None or gdf.empty:
            return
        no_data = gdf['predominant_speed'].isna()
        if no_data.any():
            gdf[no_data].plot(ax=ax, color=SPEED_NO_DATA_COLOR, linewidth=2, alpha=alpha)
            bins_present.add('no_data')
        prev_upper = 0
        for upper, color, _ in SPEED_BINS:
            mask = (
                ~gdf['predominant_speed'].isna()
                & (gdf['predominant_speed'] > prev_upper)
                & (gdf['predominant_speed'] <= upper)
            )
            if mask.any():
                gdf[mask].plot(ax=ax, color=color, linewidth=2.5, alpha=alpha)
                bins_present.add(upper)
            prev_upper = upper

    if show_outside and segs_in is not None:
        _plot_segs(segs_all, alpha=0.35)
        _plot_segs(segs_in,  alpha=1.0)
    else:
        _plot_segs(segs, alpha=0.85)

    legend = []
    if 'no_data' in bins_present:
        legend.append(Line2D([0], [0], color=SPEED_NO_DATA_COLOR, linewidth=3,
                             label=SPEED_NO_DATA_LABEL))
    for upper, color, label in SPEED_BINS:
        if upper in bins_present:
            legend.append(Line2D([0], [0], color=color, linewidth=3, label=label))
    _lgnd = ax.legend(handles=legend, loc='upper right', fontsize=10,
                      title='Predominant Speed', title_fontsize=11)
    _lgnd.get_title().set_fontweight('bold')

    _add_north_arrow(ax)
    _add_scale_bar(ax)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        print(f"  Saved → {output_path}")
    return fig


# Entry Point
# =============================================================================

if __name__ == "__main__":
    os.chdir(paths.MAIN)

    print("=" * 60)
    print("infraScanRail — Network Builder")
    print("=" * 60)

    _infra_root = Path(paths.MAIN) / paths.NETWORK_INFRASTRUCTURE_DIR
    _raw_path   = Path(paths.MAIN) / paths.NETWORK_INFRASTRUCTURE_RAW
    _base_path  = Path(paths.MAIN) / paths.NETWORK_INFRASTRUCTURE_BASE

    _raw_ready  = (_raw_path  / "nodes.gpkg").exists() and (_raw_path  / "segments.gpkg").exists()
    _base_ready = (_base_path / "nodes.gpkg").exists() and (_base_path / "segments.gpkg").exists()

    # ── Prerequisite check ────────────────────────────────────────────────────
    if not _raw_ready:
        print("\n  No filtered network found in data/Infrastructure/Raw/.")
        print("  Run infrabuild_filter_network.py first, then return here.")
        raise SystemExit(1)

    # ── Step 0 / Q1: Which network version to build? ─────────────────────────
    print("\n" + "─" * 60)
    print("[Step 0 / Q1]  Which network version to build?")
    print("─" * 60)

    if not _base_ready:
        print("\n  Note: No Base network exists yet.")
        print("  Build Base first before creating or analysing named versions.")
        _versions_avail: List[str] = ['Base']
    else:
        _versions_avail = list_versions()
        if not _versions_avail:
            _versions_avail = ['Base']

    print("\n  Available:")
    for _i, _v in enumerate(_versions_avail, 1):
        _tag = "  [base]" if _v == 'Base' else ""
        print(f"    {_i}) {_v}{_tag}")

    while True:
        _sel = input("\n  Select version (number or name) [1]: ").strip() or "1"
        if _sel.isdigit() and 1 <= int(_sel) <= len(_versions_avail):
            _chosen = _versions_avail[int(_sel) - 1]
            break
        elif _sel in _versions_avail:
            _chosen = _sel
            break
        print(f"  Invalid — enter 1–{len(_versions_avail)} or an exact name.")
    print(f"  → {_chosen}")

    # ── Step 0 / Q2: Which plots to generate? ────────────────────────────────
    print("\n" + "─" * 60)
    print("[Step 0 / Q2]  Which plots to generate?")
    print("─" * 60)
    print("\n  Catchment area (extent = catchment_area_boundary):")
    print("    Infrastructure · Gauge · Electrification · Speed · Track Owner")
    print("  Study area (extent = study_area_boundary):")
    print("    Infrastructure · Gauge · Electrification · Speed · Track Owner · Construction components")
    print("\n  Options:")
    print("    1) All")
    print("    2) Catchment area only")
    print("    3) Study area only")
    print("    4) Manual choice")
    print("    5) None")

    while True:
        _scope = input("\n  Select (1–5) [1]: ").strip() or "1"
        if _scope in ('1', '2', '3', '4', '5'):
            break
        print("  Invalid — enter 1–5.")

    _ALL_PLOTS = [
        ('ca_infra',     'Catchment — Infrastructure'),
        ('ca_gauge',     'Catchment — Gauge'),
        ('ca_elec',      'Catchment — Electrification'),
        ('ca_speed',     'Catchment — Speed'),
        ('ca_owner',     'Catchment — Track Owner'),
        ('sa_infra',     'Study area — Infrastructure'),
        ('sa_gauge',     'Study area — Gauge'),
        ('sa_elec',      'Study area — Electrification'),
        ('sa_speed',     'Study area — Speed'),
        ('sa_owner',     'Study area — Track Owner'),
        ('sa_construct', 'Study area — Construction components'),
    ]
    _CA_KEYS = [k for k, _ in _ALL_PLOTS if k.startswith('ca_')]
    _SA_KEYS = [k for k, _ in _ALL_PLOTS if k.startswith('sa_')]

    if _scope == '1':
        _plot_set = [k for k, _ in _ALL_PLOTS]
    elif _scope == '2':
        _plot_set = _CA_KEYS[:]
    elif _scope == '3':
        _plot_set = _SA_KEYS[:]
    elif _scope == '5':
        _plot_set = []
    else:  # manual
        print("\n  Select plots:")
        for _i, (_k, _lbl) in enumerate(_ALL_PLOTS, 1):
            print(f"    {_i}) {_lbl}")
        while True:
            _raw_sel = input("\n  Numbers (comma-sep) or 'all' [all]: ").strip() or "all"
            if _raw_sel.lower() == 'all':
                _plot_set = [k for k, _ in _ALL_PLOTS]
                break
            _parts = [p.strip() for p in _raw_sel.split(',') if p.strip()]
            if _parts and all(p.isdigit() and 1 <= int(p) <= len(_ALL_PLOTS) for p in _parts):
                _plot_set = [_ALL_PLOTS[int(p) - 1][0] for p in _parts]
                break
            if not _raw_sel:
                _plot_set = []
                break
            print(f"  Invalid — enter numbers 1–{len(_ALL_PLOTS)}, 'all', or leave empty.")

    _plot_label_map = dict(_ALL_PLOTS)

    # ── Step 0 / Q3: Diff plot? ───────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("[Step 0 / Q3]  Diff plot")
    print("─" * 60)

    _do_diff = input("\n  Generate a diff plot? (y/n) [n]: ").strip().lower() or "n"
    _ref_version  = None
    _diff_scope   = None

    if _do_diff == "y":
        _diff_candidates = [
            v for v in sorted(
                d.name for d in _infra_root.iterdir()
                if d.is_dir() and d.name != "Raw"
                and (d / "nodes.gpkg").exists()
                and (d / "segments.gpkg").exists()
            )
            if v != _chosen
        ]
        if not _diff_candidates:
            print("  No other versions available — skipping diff.")
            _do_diff = "n"
        else:
            print("\n  Compare against (reference version):")
            for _i, _v in enumerate(_diff_candidates, 1):
                _marker = "  [Base]" if _v == "Base" else ""
                print(f"    {_i}) {_v}{_marker}")
            while True:
                _ref_raw = input("  Select (number): ").strip()
                if _ref_raw.isdigit() and 1 <= int(_ref_raw) <= len(_diff_candidates):
                    _ref_version = _diff_candidates[int(_ref_raw) - 1]
                    break
                print(f"  Invalid — enter 1–{len(_diff_candidates)}.")

            print("\n  Extent:")
            print("    1) Catchment area")
            print("    2) Study area")
            print("    3) Both")
            while True:
                _diff_scope = input("  Select (1–3) [3]: ").strip() or "3"
                if _diff_scope in ("1", "2", "3"):
                    break
                print("  Enter 1, 2, or 3.")

    # ── Step 1: Build ─────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("[Step 1]  Network building")
    print("─" * 60)

    if _chosen == 'Base':
        if _base_ready:
            _rebuild = input(
                "\n  Base already exists. Rebuild from Raw/? (y/n) [n]: "
            ).strip().lower() or "n"
            if _rebuild == "y":
                _nodes, _segments, _composition = run_build_base()
            else:
                print("\n--- Loading existing Base ---")
                _nodes, _segments = load_version('Base')
                _comp_path = _base_path / "segments_composition.gpkg"
                _composition = (gpd.read_file(_comp_path)
                                if _comp_path.exists() else gpd.GeoDataFrame())
                print(f"  {len(_nodes)} nodes, {len(_segments)} segments")
                # Always regenerate the QGIS project
                print("\n--- QGIS project ---")
                _build_infra_qgz(str(_base_path / "Base.qgz"), _base_path)
                print(f"  Base.qgz → {_base_path / 'Base.qgz'}")
        else:
            _nodes, _segments, _composition = run_build_base()

        _version_dir = _base_path

    else:
        _version_dir = _infra_root / _chosen
        if not (_version_dir / "nodes.gpkg").exists():
            print(f"\n  Version '{_chosen}' not found in data/Infrastructure/.")
            print("  Use infrabuild_version_manager.py to create it first.")
            raise SystemExit(1)

        print(f"\n--- Loading version '{_chosen}' ---")
        _nodes, _segments = load_version(_chosen)
        print(f"  {len(_nodes)} nodes, {len(_segments)} segments")

        # Enrich any segments that were added via version manager and have no OSM speed
        _speed_col_missing = 'average_speed' not in _segments.columns
        if _speed_col_missing:
            _unmatched_mask = pd.Series([True] * len(_segments), index=_segments.index)
        else:
            _unmatched_mask = _segments['average_speed'].isna()
        if _unmatched_mask.any():
            _n_new = int(_unmatched_mask.sum())
            print(f"\n--- OSM speed enrichment for {_n_new} segment(s) missing speed ---")
            _new_segs = _segments[_unmatched_mask].copy()
            _enriched = enrich_segments_with_osm_speed(_new_segs, _nodes)
            for _col in ('average_speed', 'predominant_speed', 'predominant_speed_coverage_pct'):
                if _col in _enriched.columns:
                    _segments.loc[_unmatched_mask, _col] = _enriched[_col].values
            _segments.to_file(_version_dir / 'segments.gpkg', driver='GPKG')
            print(f"  segments.gpkg updated → {_version_dir / 'segments.gpkg'}")

        # Derive and export composition for this version
        _base_comp_path = _base_path / "segments_composition.gpkg"
        if _base_comp_path.exists():
            _base_comp    = gpd.read_file(_base_comp_path)
            _ver_sids     = set(_segments['segment_id'].dropna())
            _composition  = _filter_composition_for_version(_base_comp, _ver_sids)
            _composition.to_file(_version_dir / "segments_composition.gpkg", driver="GPKG")
            print(f"  segments_composition.gpkg → {_version_dir / 'segments_composition.gpkg'}"
                  f"  ({len(_composition)} pieces)")
        else:
            _composition = gpd.GeoDataFrame()
            print("  Warning: Base composition not found — skipping composition export.")

        # QGIS project
        print("\n--- QGIS project ---")
        _qgz = _version_dir / f"{_chosen}.qgz"
        _build_infra_qgz(str(_qgz), _version_dir)
        print(f"  {_chosen}.qgz → {_qgz}")

    # Build NetworkX graph
    print("\n--- Building NetworkX graph ---")
    G = build_networkx_graph(_nodes, _segments)

    # ── Step 2: Plots ─────────────────────────────────────────────────────────
    if not _plot_set:
        print("\n  No plots requested.")
    else:
        print("\n" + "─" * 60)
        print("[Step 2]  Generating plots")
        print("─" * 60)

        _plot_dir = Path(paths.MAIN) / paths.INFRASTRUCTURE_PLOTS_DIR / _chosen
        _plot_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  Output: {_plot_dir}")

        # Load boundaries
        _ca_boundary, _sa_boundary = None, None
        _ca_bdry_path = Path(paths.MAIN) / paths.CATCHMENT_AREA_BOUNDARY_GPKG
        _sa_bdry_path = Path(paths.MAIN) / "data/Catchment_Area/study_area_boundary.gpkg"
        if _ca_bdry_path.exists():
            _ca_boundary = gpd.read_file(_ca_bdry_path)
        if _sa_bdry_path.exists():
            _sa_boundary = gpd.read_file(_sa_bdry_path)

        def _extent_from_gdf(gdf, margin_m: int = 2000):
            if gdf is None:
                return None
            b = gdf.total_bounds          # [minx, miny, maxx, maxy]
            return (b[0] - margin_m, b[2] + margin_m,
                    b[1] - margin_m, b[3] + margin_m)

        _ca_ext = _extent_from_gdf(_ca_boundary)
        _sa_ext = _extent_from_gdf(_sa_boundary)

        _net_ca = NetworkData(nodes=_nodes, segments=_segments, graph=G,
                              version=_chosen, boundary=_ca_boundary)
        _net_sa = NetworkData(nodes=_nodes, segments=_segments, graph=G,
                              version=_chosen, boundary=_sa_boundary)

        _plot_dispatch = {
            'ca_infra':  (plot_infrastructure_canonical, _net_ca, _ca_ext,
                          'ca_infrastructure.pdf'),
            'ca_gauge':  (plot_gauge_map,                _net_ca, _ca_ext,
                          'ca_gauge.pdf'),
            'ca_elec':   (plot_electrification_map,      _net_ca, _ca_ext,
                          'ca_electrification.pdf'),
            'ca_speed':  (plot_speed_map,                _net_ca, _ca_ext,
                          'ca_speed.pdf'),
            'ca_owner':  (plot_owner_map,                _net_ca, _ca_ext,
                          'ca_owner.pdf'),
            'sa_infra':  (plot_infrastructure_canonical, _net_sa, _sa_ext,
                          'sa_infrastructure.pdf'),
            'sa_gauge':  (plot_gauge_map,                _net_sa, _sa_ext,
                          'sa_gauge.pdf'),
            'sa_elec':   (plot_electrification_map,      _net_sa, _sa_ext,
                          'sa_electrification.pdf'),
            'sa_speed':  (plot_speed_map,                _net_sa, _sa_ext,
                          'sa_speed.pdf'),
            'sa_owner':  (plot_owner_map,                _net_sa, _sa_ext,
                          'sa_owner.pdf'),
        }

        for _pk in _plot_set:
            if _pk == 'sa_construct':
                if _composition.empty:
                    print("  Skipping construction components — no composition data.")
                    continue
                print(f"  {_plot_label_map[_pk]} ...")
                _fig = plot_construction_components(
                    _net_sa, _composition, extent=_sa_ext,
                    output_path=_plot_dir / "sa_construction_components.pdf",
                    show_outside=True,
                )
                plt.close(_fig)
            else:
                _fn, _net, _ext, _fname = _plot_dispatch[_pk]
                print(f"  {_plot_label_map[_pk]} ...")
                
                kwargs = {}
                if _pk.startswith('sa_'):
                    kwargs['show_outside'] = True
                if _pk.startswith('ca_'):
                    kwargs['is_catchment'] = True
                if _pk == 'ca_infra':
                    kwargs['show_labels'] = False
                
                _fig = _fn(_net, extent=_ext, output_path=_plot_dir / _fname, **kwargs)
                plt.close(_fig)

    # ── Step 3: Diff plot ─────────────────────────────────────────────────────
    if _do_diff == "y" and _ref_version is not None:
        print("\n" + "─" * 60)
        print("[Step 3]  Diff plot")
        print("─" * 60)

        print(f"\n  Loading reference '{_ref_version}'...")
        _ref_nodes, _ref_segs = load_version(_ref_version)
        _ref_G = build_networkx_graph(_ref_nodes, _ref_segs)

        _diff_dir = Path(paths.MAIN) / paths.INFRASTRUCTURE_PLOTS_DIR / _chosen
        _diff_dir.mkdir(parents=True, exist_ok=True)

        _diff_pairs = []
        if _diff_scope in ("1", "3"):
            _diff_pairs.append(("ca", _ca_boundary, _ca_ext, True))
        if _diff_scope in ("2", "3"):
            _diff_pairs.append(("sa", _sa_boundary, _sa_ext, False))

        for _scope_key, _bdry, _ext, _is_ca in _diff_pairs:
            _ref_net  = NetworkData(nodes=_ref_nodes, segments=_ref_segs,
                                    graph=_ref_G, version=_ref_version,
                                    boundary=_bdry)
            _comp_net = NetworkData(nodes=_nodes, segments=_segments,
                                    graph=G, version=_chosen,
                                    boundary=_bdry)
            _diff_fname = f"diff_{_chosen}_vs_{_ref_version}_{_scope_key}.pdf"
            print(f"  Diff ({_scope_key.upper()}) ...")
            _fig = plot_infrastructure_diff(
                net_a=_ref_net,
                net_b=_comp_net,
                extent=_ext,
                output_path=_diff_dir / _diff_fname,
                is_catchment=_is_ca,
            )
            plt.close(_fig)
        print(f"  Diff plot(s) saved → {_diff_dir}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Version : {_chosen}")
    print(f"Nodes   : {G.number_of_nodes()}")
    print(f"Edges   : {G.number_of_edges()}")
    if _plot_set:
        print(f"Plots   : {_plot_dir}")
    print("=" * 60)
