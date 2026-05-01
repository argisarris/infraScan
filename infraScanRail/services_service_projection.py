"""
Service Projection Module

Maps rail, tram, and funicular services onto the BAV infrastructure network.
Routes GTFS-derived service links along shortest paths on the BAV graph,
enriching existing service files with real segment geometry and via-node annotations.

Workflow (interactive):
  Phase 0 — CLI setup: choose infra version + service version
  Phase 1 — Projection: match stops → route paths → enrich geopackages
  Phase 2 — Corrections: inspect combined lines, reroute services interactively
  Phase 3 — Plotting: overview plots for study area and catchment area

Usage:
    python services_service_projection.py
"""

import sys
import os
import json
import shutil
from pathlib import Path
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge, unary_union, substring
from scipy.spatial import cKDTree

from matplotlib.patches import Rectangle
from matplotlib_map_utils.core.north_arrow import NorthArrow, north_arrow

sys.path.insert(0, str(Path(__file__).parent))
import paths

# =============================================================================
# Constants
# =============================================================================

SWISS_CRS = "EPSG:2056"

NAME_MATCH_THRESHOLD = 0.85
SPATIAL_MATCH_THRESHOLD = 200  # metres

RAIL_ROUTE_TYPES = {
    0: "tram",
    1: "metro",
    2: "rail",                   # Standard GTFS fallback
    100: "rail",                 # Extended GTFS: Generic Railway Service
    101: "long_distance_rail",    # High Speed Rail Service
    102: "long_distance_rail",    # Long Distance Trains
    103: "inter_regional_rail",   # Inter Regional Rail Service
    106: "regional_rail",         # Regional Rail Service
    109: "sbahn",                # Suburban Railway (S-Bahn)
    400: "metro",
    900: "tram",
    1000: "ferry",
    1300: "cable",
    1400: "funicular",
}

# Mode colours for plotting
MODE_COLOURS = {
    "rail": "#1f3a6e",       # dark blue
    "tram": "#8b0000",       # dark red
    "funicular": "#1a5c2e",  # dark green
    "fallback": "#e07b00",   # orange (straight-line / unmatched)
}

# Maximum extra travel time (seconds) to still prefer a dead-end terminal hub node
# over a cheaper through-running child.  If the cheapest terminal node costs
# less than this much more than the cheapest candidate overall, the terminal is
# preferred for terminal (first/last) stops at hub stations.
# (~1500 m at 45 km/h ≈ 120 s; set to 300 s for a wider tolerance margin.)
TERMINAL_PREFERENCE_S: int = 300

# Default routing speeds (km/h) used when OSM maxspeed data is absent.
# Mixed-mode segments use the minimum (most conservative) of the applicable values.
RAIL_DEFAULT_SPEED_KMH: int        = 50   # train (all gauges)
TRAM_DEFAULT_SPEED_KMH: int        = 30   # urban tram / light rail
FUNICULAR_DEFAULT_SPEED_KMH: int   = 20   # funicular
COG_RAILWAY_DEFAULT_SPEED_KMH: int = 15   # cog railway

_MODE_DEFAULT_SPEEDS: Dict[str, int] = {
    "train":       RAIL_DEFAULT_SPEED_KMH,
    "tram":        TRAM_DEFAULT_SPEED_KMH,
    "funicular":   FUNICULAR_DEFAULT_SPEED_KMH,
    "cog_railway": COG_RAILWAY_DEFAULT_SPEED_KMH,
}

# Layers subject to boundary gateway rerouting (Phase 1.5).
# Services in these layers that start/end outside the buffer are re-routed
# via the nearest confirmed boundary station.
_GATEWAY_LAYERS: frozenset = frozenset({
    "long_distance_rail",
    "inter_regional_rail",
    "regional_rail",
    "sbahn",
})

# =============================================================================
# QGIS project styling constants (mirrored from services_network_builder.py)
# =============================================================================
_RAIL_LINE_TYPES: Dict[int, str] = {
    102: "Long-Distance Rail",
    103: "Inter-Regional Rail",
    106: "Regional Rail",
    109: "S-Bahn / Suburban Rail",
}
_PT_FEEDER_LINE_TYPES: Dict[int, str] = {
    900:  "Tram",
    401:  "Metro",
    700:  "Bus",
    702:  "Express Bus",
    715:  "On-demand Bus",
    1000: "Ship",
    1400: "Funicular",
}
_QGZ_LINE_COLOURS: Dict[int, str] = {
    102:  "#FF0000",   # Long-Distance Rail  — red
    103:  "#FF0000",   # Inter-Regional Rail — red
    106:  "#000000",   # Regional Rail       — black
    109:  "#000000",   # S-Bahn              — black
    401:  "#00246B",   # Metro               — dark blue
    900:  "#FF66CC",   # Tram                — pink
    700:  "#0000FF",   # Bus                 — blue
    702:  "#0000FF",   # Express Bus         — blue
    715:  "#0000FF",   # On-demand Bus       — blue
    1000: "#0099FF",   # Ship                — blue dashed
    1400: "#000000",   # Funicular           — black dashed
}
_QGZ_LINE_STYLE: Dict[int, str] = {
    1000: "dashed",
    1400: "dashed",
}
_QGZ_LAYER_NAMES: Dict[int, str] = {
    102:  "long_distance_rail",
    103:  "inter_regional_rail",
    106:  "regional_rail",
    109:  "sbahn",
    900:  "tram",
    700:  "bus",
    702:  "express_bus",
    715:  "ondemand_bus",
    1000: "ship",
    1400: "funicular",
    401:  "metro",
}
# Reverse: layer name in gpkg → route_type int (for re-splitting concatenated frames)
_LAYER_NAME_TO_RT: Dict[str, int] = {v: k for k, v in _QGZ_LAYER_NAMES.items()}
_RAIL_STOP_FILL    = "#FFFFFF"
_RAIL_STOP_OUTLINE = "#000000"


# =============================================================================
# ZVV Geometry — used as primary source for non-track modes (bus/ship) and
# as post-projection check/fallback for track-based modes
# =============================================================================

ZVV_LINES_GPKG        = os.path.join('data', 'Spatial_Data', 'Transit_Network',
                                     'ZVV_Lines_2026.gpkg')
ZVV_SEGMENT_LAYER     = 'ZVV_LINIEN_L'
ZVV_SBAHN_LAYER       = 'ZVV_S_BAHN_LINIEN_L'
ZVV_MATCH_TOLERANCE_M = 150    # max distance (m) for spatial stop crosswalk
ZVV_DIVERGENCE_THRESHOLD: float = 0.30  # flag when |proj/zvv − 1| exceeds this

SBAHN_SNAP_TOLERANCE_M = 2000
SBAHN_MAX_SINUOSITY    = 3.5
SBAHN_MIN_SINUOSITY    = 0.85
SBAHN_MIN_PART_LENGTH  = 50

# Track-based feeder modes — use BAV routing + ZVV post-pass check/replace
TRACK_BASED_FEEDER_LAYERS: frozenset = frozenset({"tram", "funicular", "metro"})
# Non-track feeder modes — use ZVV directly as geometry source
NON_TRACK_FEEDER_LAYERS: frozenset = frozenset({"bus", "express_bus", "ondemand_bus", "ship"})

# Module-level ZVV geometry state (populated in _run_phase1 before projection)
_zvv_seg_index:    Dict = {}  # (line_name, from_sid, to_sid) → LineString
_zvv_chain_index:  Dict = {}  # (line_name, from_sid) → [(to_sid, geom, seq), ...]
_zvv_sbahn_index:  Dict = {}  # line_name → [LineString, ...]
_zvv_sbahn_jgraph: Dict = {}  # line_name → {junction_id → [(part_idx, other_junction, geom)]}
_zvv_match_counts: Dict = defaultdict(int)

# Module-level stop coordinate dict — built from unprojected feeder segments
stop_coord: Dict[str, Point] = {}  # stop_id → Point(E, N) in EPSG:2056


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class MatchResult:
    """Result of matching a service stop to a BAV network node."""
    node_id: Optional[int]         # BAV Betriebspunkt_Nummer; None if unmatched
    confidence: float
    method: str  # 'id' | 'code' | 'name' | 'spatial' | 'raw_bav' | 'unmatched'
    distance_m: Optional[float] = None
    candidates: List[int] = field(default_factory=list)


@dataclass
class ProjectionConfig:
    """All path information needed to run the projection."""
    infra_version: str
    svc_version: str
    infra_dir: Path          # data/Infrastructure/<infra_version>/
    svc_dir: Path            # data/Network/Feeder_Lines/<svc_version>/
    rail_input: Path         # data/Network/Rail_Lines/<svc_version>/rail_segments.gpkg
    rail_output_dir: Path    # data/Network/Rail_Lines/<svc>/<infra>/
    feeder_output_dir: Path  # data/Network/Feeder_Lines/<svc>/<infra>/
    raw_infra_dir: Path      # data/Infrastructure/Raw/


# =============================================================================
# Infrastructure Graph
# =============================================================================

def _default_speed(mode, gauge=None) -> float:
    """Return the default routing speed (km/h) for a segment.

    Prefers transport_mode; falls back to gauge when mode is absent.
    Mixed-mode segments use the minimum (most conservative) value.
    """
    if mode is not None and not (isinstance(mode, float) and np.isnan(mode)):
        mode_str = str(mode).strip()
        if mode_str:
            speeds = [
                _MODE_DEFAULT_SPEEDS.get(m.strip(), RAIL_DEFAULT_SPEED_KMH)
                for m in mode_str.split('/')
                if m.strip()
            ]
            if speeds:
                return float(min(speeds))
    # Gauge fallback for NaN-mode segments
    if gauge is None or (isinstance(gauge, float) and np.isnan(gauge)):
        return float(RAIL_DEFAULT_SPEED_KMH)
    g = int(gauge)
    if g <= 900:
        return float(FUNICULAR_DEFAULT_SPEED_KMH)
    if g == 1000:
        return float(TRAM_DEFAULT_SPEED_KMH)
    return float(RAIL_DEFAULT_SPEED_KMH)


def _build_name_to_id(nodes: gpd.GeoDataFrame) -> Dict[str, int]:
    """
    Build a stable name → Betriebspunkt_Nummer lookup.

    Nodes that have a valid Betriebspunkt_Nummer use it directly.  Nodes that
    were manually inserted without one (e.g. synthetic junction nodes added via
    the version manager) receive a synthetic integer >= 9_000_000, assigned in
    DataFrame order so the result is identical across all callers within the same
    run.
    """
    existing_ids: set = {
        int(r["Betriebspunkt_Nummer"])
        for _, r in nodes.iterrows()
        if pd.notna(r.get("Betriebspunkt_Nummer"))
    }
    synth_counter = max(existing_ids, default=0)
    synth_counter = max(synth_counter, 9_000_000 - 1)

    name_to_id: Dict[str, int] = {}
    for _, row in nodes.iterrows():
        name = row.get("NAME", "")
        if not name or not pd.notna(name):
            continue
        if pd.notna(row.get("Betriebspunkt_Nummer")):
            name_to_id[name] = int(row["Betriebspunkt_Nummer"])
        else:
            synth_counter += 1
            while synth_counter in existing_ids:
                synth_counter += 1
            existing_ids.add(synth_counter)
            name_to_id[name] = synth_counter
    return name_to_id



# =============================================================================
# ZVV Geometry helpers — stop_coord builder, geometry loaders, post-pass
# =============================================================================

def _build_stop_coord_from_segments(svc_dir) -> dict:
    """Build stop_id -> Point(E, N) from unprojected feeder segment columns."""
    import fiona as _fiona
    gpkg = svc_dir / "pt_feeder_segments.gpkg"
    coord = {}
    if not gpkg.exists():
        return coord
    for layer in _fiona.listlayers(str(gpkg)):
        gdf = gpd.read_file(gpkg, layer=layer)
        for _, row in gdf.iterrows():
            for id_col, e_col, n_col in [
                ("from_stop_id", "from_stop_E", "from_stop_N"),
                ("to_stop_id",   "to_stop_E",   "to_stop_N"),
            ]:
                sid = str(row.get(id_col, "")).strip()
                E = row.get(e_col)
                N = row.get(n_col)
                if sid and pd.notna(E) and pd.notna(N):
                    coord[sid] = Point(float(E), float(N))
    return coord


def _load_zvv_geometry() -> bool:
    """Load ZVV GeoPackage and build geometry lookup indices.

    Uses module-level stop_coord to build a spatial crosswalk between ZVV
    internal stop numbering and GTFS DiDok/UIC stop IDs.
    Populates _zvv_seg_index, _zvv_chain_index, _zvv_sbahn_index.
    Returns True on success, False on failure.
    """
    global _zvv_seg_index, _zvv_chain_index, _zvv_sbahn_index

    zvv_path = os.path.join(paths.MAIN, ZVV_LINES_GPKG)
    if not os.path.isfile(zvv_path):
        print(f"  WARNING: ZVV GeoPackage not found at {zvv_path}")
        print("           Falling back to straight-line geometry.")
        return False

    print(f"  Loading {ZVV_SEGMENT_LAYER} ...", end=" ", flush=True)
    zvv_seg = gpd.read_file(zvv_path, layer=ZVV_SEGMENT_LAYER)
    print(f"{len(zvv_seg):,} segments")

    print(f"  Loading {ZVV_SBAHN_LAYER} ...", end=" ", flush=True)
    zvv_sbahn = gpd.read_file(zvv_path, layer=ZVV_SBAHN_LAYER)
    print(f"{len(zvv_sbahn):,} lines")

    print("  Building spatial stop crosswalk (ZVV -> GTFS) ...")

    zvv_stop_obs = defaultdict(list)
    for _, row in zvv_seg.iterrows():
        if row.geometry is None or row.geometry.is_empty:
            continue
        coords = list(row.geometry.coords)
        zvv_stop_obs[int(row["VONHALTESTELLENNR"])].append(coords[0])
        zvv_stop_obs[int(row["BISHALTESTELLENNR"])].append(coords[-1])

    zvv_stops = {}
    for vid, obs in zvv_stop_obs.items():
        xs = [c[0] for c in obs]
        ys = [c[1] for c in obs]
        zvv_stops[vid] = (float(np.median(xs)), float(np.median(ys)))

    gtfs_ids = list(stop_coord.keys())
    if not gtfs_ids:
        print("  WARNING: stop_coord is empty — ZVV crosswalk cannot be built.")
        return False
    gtfs_pts = np.array([(stop_coord[sid].x, stop_coord[sid].y) for sid in gtfs_ids])
    tree = cKDTree(gtfs_pts)

    zvv_ids = list(zvv_stops.keys())
    zvv_pts = np.array([zvv_stops[vid] for vid in zvv_ids])
    dists, idxs = tree.query(zvv_pts)

    crosswalk = {}
    matched_dists = []
    for i, zvv_id in enumerate(zvv_ids):
        if dists[i] <= ZVV_MATCH_TOLERANCE_M:
            crosswalk[zvv_id] = gtfs_ids[idxs[i]]
            matched_dists.append(dists[i])

    med_dist = float(np.median(matched_dists)) if matched_dists else float("nan")
    print(
        f"    {len(zvv_stops):,} ZVV stops -> "
        f"{len(crosswalk):,} matched within {ZVV_MATCH_TOLERANCE_M}m "
        f"(median {med_dist:.0f}m)"
    )

    if not crosswalk:
        print("  WARNING: No stops matched -- ZVV geometry unavailable.")
        return False

    print("  Building ZVV segment index ...")
    seg_index   = {}
    chain_index = defaultdict(list)

    for _, row in zvv_seg.iterrows():
        if row.geometry is None or row.geometry.is_empty:
            continue
        line_name = str(row["LINIENNUMMER"]).strip()
        von = int(row["VONHALTESTELLENNR"])
        bis = int(row["BISHALTESTELLENNR"])
        seq_nr = int(row["SEQUENZNR"]) if pd.notna(row.get("SEQUENZNR")) else 0
        gtfs_from = crosswalk.get(von)
        gtfs_to   = crosswalk.get(bis)
        if gtfs_from is None or gtfs_to is None:
            continue
        geom = row.geometry
        key  = (line_name, gtfs_from, gtfs_to)
        if key not in seg_index:
            seg_index[key] = geom
        chain_index[(line_name, gtfs_from)].append((gtfs_to, geom, seq_nr))

    for k in chain_index:
        chain_index[k].sort(key=lambda x: x[2])

    print("  Validating segment geometry directions ...")
    n_correct = n_reversed = n_unknown = 0
    for key, geom in list(seg_index.items()):
        _, from_sid, to_sid = key
        from_pt = stop_coord.get(from_sid)
        to_pt   = stop_coord.get(to_sid)
        if from_pt is None or to_pt is None:
            n_unknown += 1
            continue
        geom_start = Point(geom.coords[0])
        geom_end   = Point(geom.coords[-1])
        dist_ok  = geom_start.distance(from_pt) + geom_end.distance(to_pt)
        dist_rev = geom_start.distance(to_pt)   + geom_end.distance(from_pt)
        if dist_rev < dist_ok:
            seg_index[key] = LineString(list(geom.coords)[::-1])
            n_reversed += 1
        else:
            n_correct += 1
    print(
        f"    Direction: {n_correct:,} correct, "
        f"{n_reversed:,} reversed, {n_unknown:,} unknown"
    )

    for k, entries in chain_index.items():
        new_entries = []
        for to_sid, geom, seq_nr in entries:
            seg_key = (k[0], k[1], to_sid)
            if seg_key in seg_index:
                new_entries.append((to_sid, seg_index[seg_key], seq_nr))
            else:
                new_entries.append((to_sid, geom, seq_nr))
        chain_index[k] = new_entries

    print("  Building S-Bahn line index ...")
    sbahn_index = defaultdict(list)
    for _, row in zvv_sbahn.iterrows():
        name = str(row["LINIESBAHN"]).strip()
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if isinstance(geom, MultiLineString):
            merged = linemerge(geom)
            if isinstance(merged, MultiLineString):
                for part in merged.geoms:
                    if (part and not part.is_empty
                            and len(part.coords) >= 2
                            and part.length >= SBAHN_MIN_PART_LENGTH):
                        sbahn_index[name].append(part)
            else:
                if merged.length >= SBAHN_MIN_PART_LENGTH:
                    sbahn_index[name].append(merged)
        else:
            if geom.length >= SBAHN_MIN_PART_LENGTH:
                sbahn_index[name].append(geom)

    n_sbahn = len(sbahn_index)
    print(f"    {n_sbahn} S-Bahn lines indexed")

    JUNCTION_TOLERANCE = 100
    jgraph = {}
    n_jgraph = 0
    for name, parts in sbahn_index.items():
        if len(parts) < 2:
            continue
        endpoints = []
        for idx, p in enumerate(parts):
            c = list(p.coords)
            endpoints.append((idx, 0, c[0][0],  c[0][1]))
            endpoints.append((idx, 1, c[-1][0], c[-1][1]))
        ep_pts = np.array([(e[2], e[3]) for e in endpoints])
        junc_ids = list(range(len(endpoints)))
        for i in range(len(endpoints)):
            for j in range(i + 1, len(endpoints)):
                if np.hypot(
                    ep_pts[i, 0] - ep_pts[j, 0],
                    ep_pts[i, 1] - ep_pts[j, 1],
                ) < JUNCTION_TOLERANCE:
                    old_id = max(junc_ids[i], junc_ids[j])
                    new_id = min(junc_ids[i], junc_ids[j])
                    for k_i in range(len(junc_ids)):
                        if junc_ids[k_i] == old_id:
                            junc_ids[k_i] = new_id
        graph = defaultdict(list)
        for idx, p in enumerate(parts):
            start_j = junc_ids[idx * 2]
            end_j   = junc_ids[idx * 2 + 1]
            if start_j == end_j:
                continue
            graph[start_j].append((idx, end_j, p))
            graph[end_j].append((idx, start_j, p))
        if graph:
            jgraph[name] = dict(graph)
            n_jgraph += 1
    print(f"    {n_jgraph} S-Bahn junction graphs built")

    _zvv_seg_index.update(seg_index)
    _zvv_chain_index.update(chain_index)
    _zvv_sbahn_index.update(sbahn_index)
    _zvv_sbahn_jgraph.update(jgraph)

    n_lines_matched = len({k[0] for k in seg_index})
    print(
        f"  ZVV geometry ready: {len(seg_index):,} segment mappings "
        f"across {n_lines_matched} lines, {n_sbahn} S-Bahn lines"
    )
    return True


def _chain_zvv_segments(line_name, from_sid, to_sid, max_hops=30):
    """Follow ZVV segment chain from from_sid to to_sid."""
    if not _zvv_chain_index:
        return None
    visited = set()
    current = from_sid
    geoms   = []
    for _ in range(max_hops):
        if current == to_sid:
            break
        next_segs = _zvv_chain_index.get((line_name, current), [])
        if not next_segs:
            return None
        found = False
        for nxt, geom, _ in next_segs:
            if nxt not in visited:
                geoms.append(geom)
                visited.add(current)
                current = nxt
                found = True
                break
        if not found:
            return None
    if current != to_sid or not geoms:
        return None
    coords = []
    for g in geoms:
        c = list(g.coords)
        if coords:
            coords.extend(c[1:])
        else:
            coords.extend(c)
    return LineString(coords) if len(coords) >= 2 else None


def _snap_to_sbahn(from_pt, to_pt, line_name):
    """Project from_pt/to_pt onto an S-Bahn whole-line geometry."""
    geom_list = _zvv_sbahn_index.get(line_name)
    if not geom_list:
        return None
    from_point = Point(from_pt.x, from_pt.y)
    to_point   = Point(to_pt.x, to_pt.y)
    straight_dist = from_point.distance(to_point)
    if straight_dist < 1.0:
        return None
    best_sub   = None
    best_score = float("inf")
    for geom in geom_list:
        if geom is None or geom.is_empty or len(geom.coords) < 2:
            continue
        d_from = geom.distance(from_point)
        d_to   = geom.distance(to_point)
        if d_from > SBAHN_SNAP_TOLERANCE_M or d_to > SBAHN_SNAP_TOLERANCE_M:
            continue
        orig_from_proj = geom.project(from_point)
        orig_to_proj   = geom.project(to_point)
        geom_len       = geom.length
        orig_sub_len   = abs(orig_from_proj - orig_to_proj)
        orig_sinuosity = orig_sub_len / straight_dist
        if orig_sinuosity < SBAHN_MIN_SINUOSITY:
            close_d = min(d_from, d_to)
            both_near_start = orig_from_proj < 1000 and orig_to_proj < 1000
            both_near_end   = (
                orig_from_proj > geom_len - 1000
                and orig_to_proj > geom_len - 1000
            )
            if not (close_d < 100 and (both_near_start or both_near_end)):
                continue
        work_geom   = geom
        work_coords = list(geom.coords)
        for pt, d_pt in [(from_point, d_from), (to_point, d_to)]:
            if d_pt <= 50:
                continue
            proj = geom.project(pt)
            if proj < 50:
                work_coords = [(pt.x, pt.y)] + work_coords
            elif proj > geom_len - 50:
                work_coords = work_coords + [(pt.x, pt.y)]
        if len(work_coords) > len(geom.coords):
            work_geom = LineString(work_coords)
        from_proj = work_geom.project(from_point)
        to_proj   = work_geom.project(to_point)
        if abs(from_proj - to_proj) < 10.0:
            continue
        if from_proj <= to_proj:
            sub = substring(work_geom, from_proj, to_proj)
        else:
            sub = substring(work_geom, to_proj, from_proj)
            sub = LineString(list(sub.coords)[::-1])
        if sub is None or sub.is_empty or len(sub.coords) < 2:
            continue
        sinuosity = sub.length / straight_dist
        if sinuosity > SBAHN_MAX_SINUOSITY:
            continue
        if sub.length < best_score:
            best_score = sub.length
            best_sub   = sub
    return best_sub


def _snap_to_sbahn_chained(from_pt, to_pt, line_name, max_hops=3):
    """Chain multiple S-Bahn geometry parts via junction graph."""
    graph = _zvv_sbahn_jgraph.get(line_name)
    geom_list = _zvv_sbahn_index.get(line_name)
    if not graph or not geom_list or len(geom_list) < 2:
        return None
    from_point = Point(from_pt.x, from_pt.y)
    to_point   = Point(to_pt.x, to_pt.y)
    straight_dist = from_point.distance(to_point)
    if straight_dist < 1.0:
        return None
    from_parts = []
    to_parts   = []
    for idx, geom in enumerate(geom_list):
        d_from = geom.distance(from_point)
        d_to   = geom.distance(to_point)
        if d_from <= SBAHN_SNAP_TOLERANCE_M:
            from_parts.append((idx, geom.project(from_point), d_from))
        if d_to <= SBAHN_SNAP_TOLERANCE_M:
            to_parts.append((idx, geom.project(to_point), d_to))
    if not from_parts or not to_parts:
        return None
    part_juncs = {}
    for junc, neighbours in graph.items():
        for pidx, other_junc, _ in neighbours:
            if pidx not in part_juncs:
                part_juncs[pidx] = (junc, other_junc)
    best_result = None
    best_score  = float("inf")
    for from_idx, from_proj, _ in from_parts:
        for to_idx, to_proj, _ in to_parts:
            if from_idx == to_idx:
                continue
            fj = part_juncs.get(from_idx)
            tj = part_juncs.get(to_idx)
            if fj is None or tj is None:
                continue
            target_juncs = set(tj)
            for start_junc in fj:
                queue = [(start_junc, [])]
                visited = {start_junc}
                found_path = None
                while queue:
                    cur, path = queue.pop(0)
                    if cur in target_juncs and len(path) > 0:
                        found_path = (path, cur)
                        break
                    if len(path) >= max_hops:
                        continue
                    for pidx, nxt_junc, _ in graph.get(cur, []):
                        if nxt_junc not in visited:
                            visited.add(nxt_junc)
                            queue.append((nxt_junc, path + [pidx]))
                if found_path is None:
                    continue
                path_parts, end_junc = found_path
                from_geom = geom_list[from_idx]
                to_geom   = geom_list[to_idx]
                _, from_end_j = part_juncs[from_idx]
                from_fwd = (from_end_j == start_junc)
                if from_fwd:
                    sub_from = substring(from_geom, from_proj, from_geom.length)
                else:
                    sub_from = substring(from_geom, 0, from_proj)
                if (
                    sub_from is None or sub_from.is_empty
                    or sub_from.geom_type != "LineString"
                    or len(sub_from.coords) < 2
                ):
                    continue
                if not from_fwd:
                    sub_from = LineString(list(sub_from.coords)[::-1])
                chain_geoms = [sub_from]
                prev_junc = start_junc
                valid = True
                for pidx in path_parts:
                    p_start_j, p_end_j = part_juncs.get(pidx, (None, None))
                    p_geom = geom_list[pidx]
                    if pidx == from_idx or pidx == to_idx:
                        if pidx == to_idx:
                            break
                        continue
                    if p_start_j == prev_junc:
                        chain_geoms.append(p_geom)
                        prev_junc = p_end_j
                    elif p_end_j == prev_junc:
                        chain_geoms.append(LineString(list(p_geom.coords)[::-1]))
                        prev_junc = p_start_j
                    else:
                        valid = False
                        break
                if not valid:
                    continue
                to_start_j, _ = part_juncs[to_idx]
                to_fwd = (to_start_j == end_junc)
                if to_fwd:
                    sub_to = substring(to_geom, 0, to_proj)
                else:
                    sub_to = substring(to_geom, to_proj, to_geom.length)
                if (
                    sub_to is None or sub_to.is_empty
                    or sub_to.geom_type != "LineString"
                    or len(sub_to.coords) < 2
                ):
                    continue
                if not to_fwd:
                    sub_to = LineString(list(sub_to.coords)[::-1])
                chain_geoms.append(sub_to)
                all_coords = []
                for g in chain_geoms:
                    c = list(g.coords)
                    if all_coords:
                        all_coords.extend(c[1:])
                    else:
                        all_coords.extend(c)
                if len(all_coords) < 2:
                    continue
                result = LineString(all_coords)
                if result.length / straight_dist > SBAHN_MAX_SINUOSITY:
                    continue
                if result.length < best_score:
                    best_score  = result.length
                    best_result = result
    return best_result


def _get_zvv_segment_geom(from_sid: str, to_sid: str, line_name: str,
                          allow_crossline: bool = False):
    """Look up ZVV geometry for a single stop-to-stop segment.

    allow_crossline: try snapping to a *different* S-Bahn line when the
    segment's own line lookup fails.  Restricted to sbahn segments that
    need correction and lie within the catchment buffer — callers set it.
    """
    geom = _zvv_seg_index.get((line_name, from_sid, to_sid))
    if geom is not None:
        _zvv_match_counts["direct"] += 1
        return geom
    chained = _chain_zvv_segments(line_name, from_sid, to_sid)
    if chained is not None:
        _zvv_match_counts["chain"] += 1
        return chained
    if line_name in _zvv_sbahn_index:
        from_pt = stop_coord.get(from_sid)
        to_pt   = stop_coord.get(to_sid)
        if from_pt is not None and to_pt is not None:
            snapped = _snap_to_sbahn(from_pt, to_pt, line_name)
            if snapped is not None:
                _zvv_match_counts["sbahn"] += 1
                return snapped
    if line_name in _zvv_sbahn_jgraph:
        from_pt = stop_coord.get(from_sid)
        to_pt   = stop_coord.get(to_sid)
        if from_pt is not None and to_pt is not None:
            chained_sbahn = _snap_to_sbahn_chained(from_pt, to_pt, line_name)
            if chained_sbahn is not None:
                _zvv_match_counts["sbahn_chain"] += 1
                return chained_sbahn
    if allow_crossline:
        from_pt = stop_coord.get(from_sid)
        to_pt   = stop_coord.get(to_sid)
        if from_pt is not None and to_pt is not None:
            for other_line in _zvv_sbahn_index:
                if other_line == line_name:
                    continue
                snapped = _snap_to_sbahn(from_pt, to_pt, other_line)
                if snapped is not None:
                    _zvv_match_counts["sbahn_crossline"] += 1
                    return snapped
    _zvv_match_counts["fallback"] += 1
    return None


def _print_zvv_match_summary(label: str) -> None:
    """Print and reset ZVV geometry match statistics."""
    total = sum(_zvv_match_counts.values())
    if total == 0:
        return
    parts = []
    for src in ["direct", "chain", "sbahn", "sbahn_chain", "sbahn_crossline", "fallback"]:
        n = _zvv_match_counts.get(src, 0)
        if n > 0:
            parts.append(f"{src} {n:,} ({100 * n / total:.0f}%)")
    print(f"  ZVV [{label}]: {' | '.join(parts)}")
    _zvv_match_counts.clear()


def _apply_zvv_postpass(
    enriched_gdf: "gpd.GeoDataFrame",
    layer_name: str,
    is_track_based: bool,
    buffer_geom=None,
) -> "gpd.GeoDataFrame":
    """Apply ZVV geometry as post-projection check/replacement.

    Track-based (is_track_based=True):
      needs_correction=True rows -> replace with ZVV if found
      good rows -> flag zvv_divergence if |proj/zvv - 1| > ZVV_DIVERGENCE_THRESHOLD

    Non-track (is_track_based=False):
      Apply ZVV directly; keep straight-line where ZVV unavailable.

    sbahn_crossline is enabled only for sbahn segments that need correction
    and whose geometry lies within buffer_geom (or everywhere when buffer_geom
    is None).  All other modes go straight to fallback.
    """
    rows = enriched_gdf.to_dict("records")
    updated = 0
    flagged = 0

    for rec in rows:
        from_sid  = str(rec.get("from_stop_id", "")).strip()
        to_sid    = str(rec.get("to_stop_id",   "")).strip()
        line_name = str(rec.get("line_short_name", "")).strip()
        if not from_sid or not to_sid:
            continue

        # Determine the specific mode layer for this record.
        # Rail records carry _source_layer (e.g. "sbahn", "regional_rail");
        # feeder records are identified by layer_name directly.
        seg_layer  = rec.get("_source_layer", layer_name)
        needs_corr = bool(rec.get("needs_correction", False))

        # sbahn_crossline: only sbahn segments that need correction and are
        # within the catchment area buffer.
        if seg_layer == "sbahn" and needs_corr:
            seg_geom = rec.get("geometry")
            in_buf = (
                buffer_geom is None
                or (seg_geom is not None
                    and not seg_geom.is_empty
                    and seg_geom.intersects(buffer_geom))
            )
            allow_crossline = in_buf
        else:
            allow_crossline = False

        zvv_geom = _get_zvv_segment_geom(from_sid, to_sid, line_name,
                                         allow_crossline=allow_crossline)

        if is_track_based:
            if needs_corr and zvv_geom is not None:
                rec["geometry"] = zvv_geom
                rec["needs_correction"] = False
                rec["zvv_source"] = True
                updated += 1
            elif zvv_geom is not None and not needs_corr:
                proj_geom = rec.get("geometry")
                proj_len  = proj_geom.length if proj_geom else 0
                zvv_len   = zvv_geom.length
                if zvv_len > 0 and abs(proj_len / zvv_len - 1) > ZVV_DIVERGENCE_THRESHOLD:
                    rec["zvv_divergence"] = True
                    flagged += 1
        else:
            if zvv_geom is not None:
                rec["geometry"] = zvv_geom
                rec["zvv_source"] = True
                updated += 1

    print(f"  ZVV post-pass [{layer_name}]: {updated} replaced, {flagged} divergence flags")
    _print_zvv_match_summary(layer_name)
    return gpd.GeoDataFrame(rows, crs=SWISS_CRS)



def build_infra_graph(
    nodes: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
    raw_nodes: Optional[gpd.GeoDataFrame] = None,
) -> nx.Graph:
    """
    Build an undirected NetworkX graph from BAV nodes and segments.

    Nodes are identified by Betriebspunkt_Nummer (int). Segments connect via from_name/to_name,
    which are resolved to Betriebspunkt_Nummer through a name lookup. Segments whose endpoints
    are not present in nodes are silently skipped, UNLESS `raw_nodes` is provided,
    in which case missing nodes are added dynamically with `node_class='removed'`
    to heal broken graphs and enable continuous routing.

    Edge attribute 'weight' = length_m.
    Node attributes: name, code, E, N, node_class.
    """
    name_to_id = _build_name_to_id(nodes)

    raw_name_lookup = {}
    if raw_nodes is not None:
        for _, row in raw_nodes.iterrows():
            if pd.notna(row.get("NAME")) and row["NAME"] and pd.notna(row.get("Betriebspunkt_Nummer")):
                raw_name_lookup[row["NAME"]] = row

    G = nx.Graph()

    # Add all active nodes (synthetic-ID nodes included via name_to_id)
    for _, row in nodes.iterrows():
        name = row.get("NAME", "")
        nid = name_to_id.get(name) if name else None
        if nid is None:
            continue
        G.add_node(
            nid,
            name=row.get("NAME", ""),
            code=row.get("CODE", ""),
            E=float(row.get("E", 0)),
            N=float(row.get("N", 0)),
            node_class=row.get("node_class", ""),
        )

    # Add edges from segments, auto-healing missing nodes from raw_nodes
    skipped = 0
    healed = 0
    for _, seg in segments.iterrows():
        fn_name = seg.get("from_name")
        tn_name = seg.get("to_name")

        fn = name_to_id.get(fn_name)
        tn = name_to_id.get(tn_name)

        if (fn is None or tn is None) and raw_nodes is not None:
            # Try to heal missing nodes using raw_nodes
            for name, missing_id_var in [(fn_name, 'fn'), (tn_name, 'tn')]:
                if locals()[missing_id_var] is None and name in raw_name_lookup:
                    raw_row = raw_name_lookup[name]
                    raw_id = int(raw_row["Betriebspunkt_Nummer"])
                    if raw_id not in G:
                        G.add_node(
                            raw_id,
                            name=raw_row.get("NAME", ""),
                            code=raw_row.get("CODE", ""),
                            E=float(raw_row.get("E", 0)),
                            N=float(raw_row.get("N", 0)),
                            node_class="removed",  # Flag as a healed virtual node
                        )
                    if missing_id_var == 'fn':
                        fn = raw_id
                    else:
                        tn = raw_id
                    healed += 1

        if fn is None or tn is None:
            skipped += 1
            continue

        seg_length_m = float(seg.get("length_m", 0))
        raw_speed    = seg.get("average_speed")
        speed_kmh    = (
            float(raw_speed)
            if raw_speed is not None
            and not (isinstance(raw_speed, float) and np.isnan(raw_speed))
            else _default_speed(seg.get("transport_mode"), seg.get("gauge"))
        )
        travel_time_s = (
            seg_length_m / (speed_kmh / 3.6)
            if speed_kmh > 0
            else seg_length_m / (RAIL_DEFAULT_SPEED_KMH / 3.6)
        )
        G.add_edge(
            fn, tn,
            weight=travel_time_s,      # routing metric: travel time in seconds
            length_m=seg_length_m,     # physical distance for path_length_m output
            segment_id=seg.get("segment_id", ""),
            geometry=seg.geometry,
            gauge=seg.get("gauge"),
            electrification=seg.get("electrification"),
        )

    if healed:
        print(f"  [graph] {healed} missing node endpoints dynamically healed from raw_nodes.")
    if skipped:
        print(f"  [graph] {skipped} segments skipped (from_name or to_name unresolvable).")
    return G


def build_segment_lookup(
    nodes: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
    raw_nodes: Optional[gpd.GeoDataFrame] = None,
) -> Dict[Tuple[int, int], pd.Series]:
    """
    Build {(from_id, to_id): segment_row} dict for fast geometry retrieval.
    Both directions are stored (undirected). Includes healed virtual nodes if `raw_nodes` provided.
    """
    name_to_id = _build_name_to_id(nodes)

    raw_name_lookup = {}
    if raw_nodes is not None:
        raw_name_lookup = {
            row["NAME"]: int(row["Betriebspunkt_Nummer"])
            for _, row in raw_nodes.iterrows()
            if pd.notna(row.get("NAME")) and row["NAME"] and pd.notna(row.get("Betriebspunkt_Nummer"))
        }

    lookup: Dict[Tuple[int, int], pd.Series] = {}
    for _, seg in segments.iterrows():
        fn_name = seg.get("from_name")
        tn_name = seg.get("to_name")

        fn = name_to_id.get(fn_name)
        tn = name_to_id.get(tn_name)

        if fn is None and raw_nodes is not None:
            fn = raw_name_lookup.get(fn_name)
        if tn is None and raw_nodes is not None:
            tn = raw_name_lookup.get(tn_name)

        if fn is None or tn is None:
            continue
        lookup[(fn, tn)] = seg
        lookup[(tn, fn)] = seg
    return lookup


def build_node_attrs(nodes: gpd.GeoDataFrame) -> Dict[int, Dict]:
    """
    Build {Betriebspunkt_Nummer: {name, code, E, N, node_class}} dict for fast attribute lookup.
    """
    return {
        int(row["Betriebspunkt_Nummer"]): {
            "name": row.get("NAME", ""),
            "code": row.get("CODE", ""),
            "E": float(row.get("E", 0)),
            "N": float(row.get("N", 0)),
            "node_class": row.get("node_class", ""),
        }
        for _, row in nodes.iterrows() if pd.notna(row.get("Betriebspunkt_Nummer"))
    }


# =============================================================================
# Gauge / electrification routing helpers
# =============================================================================

def _node_gauge(node_id: int, G: nx.Graph) -> Optional[int]:
    """Return the single gauge (mm) shared by all edges incident to node_id.
    Returns None when values are mixed, all-null, or node absent."""
    gauges: set = set()
    for _, _, edata in G.edges(node_id, data=True):
        g = edata.get("gauge")
        if g is not None and not (isinstance(g, float) and np.isnan(g)):
            try:
                gauges.add(int(float(g)))
            except (ValueError, TypeError):
                pass
    return next(iter(gauges)) if len(gauges) == 1 else None


def _node_electrification(node_id: int, G: nx.Graph) -> Optional[str]:
    """Return the single electrification shared by all edges incident to node_id.
    Returns None when values are mixed, all-null, or node absent."""
    elecs: set = set()
    for _, _, edata in G.edges(node_id, data=True):
        e = edata.get("electrification")
        if e is not None and not (isinstance(e, float) and np.isnan(e)):
            s = str(e).strip()
            if s:
                elecs.add(s)
    return next(iter(elecs)) if len(elecs) == 1 else None


def _build_gauge_graphs(G: nx.Graph) -> Dict[int, nx.Graph]:
    """Pre-build one gauge-filtered subgraph view per distinct gauge value.

    Each view includes edges whose gauge matches the key plus all edges with
    null/missing gauge (permissive — treated as compatible with any gauge).
    Built once per Phase 1 run; O(1) per routing call thereafter.
    """
    gauge_values: set = set()
    for _, _, edata in G.edges(data=True):
        g = edata.get("gauge")
        if g is not None and not (isinstance(g, float) and np.isnan(g)):
            try:
                gauge_values.add(int(float(g)))
            except (ValueError, TypeError):
                pass

    def _make_filter(target: int):
        def _f(u, v):
            edata = G.edges[u, v]
            g = edata.get("gauge")
            if g is None or (isinstance(g, float) and np.isnan(g)):
                return True  # null gauge: permissive
            try:
                return int(float(g)) == target
            except (ValueError, TypeError):
                return True  # unparseable: permissive
        return _f

    gauge_graphs: Dict[int, nx.Graph] = {
        gv: nx.subgraph_view(G, filter_edge=_make_filter(gv))
        for gv in gauge_values
    }
    if gauge_graphs:
        print(f"  Gauge-filtered graphs: {sorted(gauge_graphs.keys())} mm")
    return gauge_graphs


# =============================================================================
# Hub Topology
# =============================================================================

def _bfs_outlying_stations(
    G: nx.Graph,
    gateway: int,
    cluster: frozenset,
) -> frozenset:
    """
    Branch-exhausting BFS from `gateway` outward through G, excluding all
    cluster nodes.  Returns the frozenset of station-class nodes that are the
    first station encountered on each outward branch.

    Stopping rule per branch: once a station node is found that branch
    terminates — we do not traverse past it.

    Args:
        G:       infrastructure graph
        gateway: starting node (first node outside the hub cluster)
        cluster: frozenset of all hub cluster node IDs (hub + children)

    Returns:
        frozenset of outlying station node IDs (may be empty if no station
        is reachable outward from this gateway).
    """
    if gateway not in G:
        return frozenset()

    node_classes = nx.get_node_attributes(G, "node_class")

    if node_classes.get(gateway) == "station":
        return frozenset({gateway})

    found: set = set()
    visited: set = set(cluster) | {gateway}
    queue: deque = deque([gateway])

    while queue:
        current = queue.popleft()
        for nbr in G.neighbors(current):
            if nbr in visited:
                continue
            visited.add(nbr)
            if node_classes.get(nbr) == "station":
                found.add(nbr)
                # Do not explore past this station on this branch
            else:
                queue.append(nbr)

    return frozenset(found)


def build_hub_topology(
    nodes: gpd.GeoDataFrame,
    G: nx.Graph,
) -> Dict[int, Dict]:
    """
    Precompute hub topology for all hub stations.

    A hub is a station node (node_class='station') that has at least one child
    that is also a station.  For each hub, computes:

      hub_node_type  — 'terminal' or 'through' classification of the hub parent.

      children       — per station child: node_type ('terminal'/'through') and,
                       per gateway, forced-routing flag + set of outlying stations
                       found by branch-exhausting BFS (_bfs_outlying_stations).

      crossing_table — maps frozenset({outlying_a, outlying_b}) to the through
                       child that connects them without backtracking.

      all_outlying   — union of all outlying station sets across the hub cluster
                       (children + hub parent perimeter).

    Returns
    -------
    {
      hub_id: {
        "hub_node_type": "terminal" | "through",
        "children": {
          child_id: {
            "node_type": "terminal" | "through",
            "gateways": {
              gateway_id: {
                "forced": bool,
                "outlying_stations": frozenset[int]
              }
            }
          }
        },
        "crossing_table": {frozenset({a, b}): child_id, ...},
        "all_outlying":   frozenset[int]
      }
    }
    """
    # ── Build parent → [children] mapping from nodes table ───────────────────
    node_uuid_to_bpnr: Dict[str, int] = {}
    for _, row in nodes.iterrows():
        if pd.notna(row.get("Betriebspunkt_Nummer")) and pd.notna(row.get("node_id")):
            node_uuid_to_bpnr[str(row["node_id"]).strip()] = int(row["Betriebspunkt_Nummer"])

    children_by_parent: Dict[int, List[int]] = {}
    node_class_map: Dict[int, str] = {}
    for _, row in nodes.iterrows():
        if pd.isna(row.get("Betriebspunkt_Nummer")):
            continue
        nid = int(row["Betriebspunkt_Nummer"])
        node_class_map[nid] = str(row.get("node_class", ""))
        pn = row.get("parent_node")
        if pd.isna(pn):
            continue
        raw = str(pn).strip()
        if not raw or raw.lower() in ("none", "nan"):
            continue
        try:
            pid = int(float(raw))
        except ValueError:
            pid = node_uuid_to_bpnr.get(raw)
        if pid is not None:
            children_by_parent.setdefault(pid, []).append(nid)

    hub_topology: Dict[int, Dict] = {}

    for hub_id, children in children_by_parent.items():
        if node_class_map.get(hub_id) != "station":
            continue
        station_children = [c for c in children if node_class_map.get(c) == "station"]
        if not station_children:
            continue

        cluster = frozenset([hub_id] + children)

        # ── Hub parent node type (step a) ─────────────────────────────────────
        hub_perimeter = (
            [n for n in G.neighbors(hub_id) if n not in cluster]
            if hub_id in G else []
        )
        hub_node_type = "through" if len(hub_perimeter) >= 2 else "terminal"

        # ── Station children: node type + gateway data (steps a + b) ─────────
        child_data: Dict[int, Dict] = {}
        for c in station_children:
            if c not in G:
                continue
            gateways = [n for n in G.neighbors(c) if n not in cluster]
            if not gateways:
                continue

            node_type = "through" if len(gateways) >= 2 else "terminal"

            gateway_data: Dict[int, Dict] = {}
            for gw in gateways:
                # Forced-routing flag: is there a surface path shorter than
                # the direct edge? If so Dijkstra would bypass the segment
                # (e.g. a DML tunnel), requiring forced_via routing.
                direct_w = G[gw][c].get("weight", 0.0)
                G_tmp = G.copy()
                G_tmp.remove_edge(gw, c)
                try:
                    surface_len = nx.shortest_path_length(G_tmp, gw, c, weight="weight")
                    forced = surface_len < direct_w
                except nx.NetworkXNoPath:
                    forced = False

                gateway_data[gw] = {
                    "forced": forced,
                    "outlying_stations": _bfs_outlying_stations(G, gw, cluster),
                }

            child_data[c] = {"node_type": node_type, "gateways": gateway_data}

        if not child_data:
            continue

        # ── Crossing table + all_outlying (step c) ───────────────────────────
        all_outlying: set = set()
        crossing_table: Dict = {}

        # Collect all gateway node IDs across ALL station children so we can
        # exclude them from through-child outlying when building the crossing
        # table cross-products.  This prevents a through-child from stealing
        # crossing-table pairs that belong to another child's exclusive gateway
        # corridor (e.g. ZLOE claiming Stadelhofen pairs that belong to ZMUS).
        all_child_gateway_nodes: set = set()
        for cdata in child_data.values():
            all_child_gateway_nodes |= set(cdata["gateways"].keys())

        for child_id, cdata in child_data.items():
            # Collect outlying stations from every child (terminal + through)
            for gdata in cdata["gateways"].values():
                all_outlying |= gdata["outlying_stations"]

            if cdata["node_type"] != "through":
                continue

            # Gateway nodes that belong to OTHER children — exclude from this
            # child's cross-product so each child owns its exclusive corridor.
            own_gateways    = set(cdata["gateways"].keys())
            other_gateways  = all_child_gateway_nodes - own_gateways

            gws = list(cdata["gateways"].keys())
            for i in range(len(gws)):
                for j in range(i + 1, len(gws)):
                    os_i = cdata["gateways"][gws[i]]["outlying_stations"] - other_gateways
                    os_j = cdata["gateways"][gws[j]]["outlying_stations"] - other_gateways
                    for o_a in os_i:
                        for o_b in os_j:
                            key = frozenset({o_a, o_b})
                            if key not in crossing_table:  # first match wins
                                crossing_table[key] = child_id

        # Also collect outlying stations reachable via the hub parent's own
        # perimeter edges (hub parent may have direct outside connections)
        for gw in hub_perimeter:
            all_outlying |= _bfs_outlying_stations(G, gw, cluster)

        hub_topology[hub_id] = {
            "hub_node_type": hub_node_type,
            "children": child_data,
            "crossing_table": crossing_table,
            "all_outlying": frozenset(all_outlying),
        }

    # ── Print summary ─────────────────────────────────────────────────────────
    def _hub_name(nid: int) -> str:
        return next(
            (str(r.get("NAME", nid)) for _, r in nodes.iterrows()
             if pd.notna(r.get("Betriebspunkt_Nummer"))
             and int(r["Betriebspunkt_Nummer"]) == nid),
            str(nid),
        )

    for hub_id, hub_data in hub_topology.items():
        children = hub_data["children"]
        print(
            f"  [hub] {_hub_name(hub_id)} ({hub_id}) [{hub_data['hub_node_type']}]: "
            f"{len(children)} station child(ren), "
            f"{len(hub_data['all_outlying'])} outlying station(s), "
            f"{len(hub_data['crossing_table'])} crossing path(s)"
        )
        for child_id, cdata in children.items():
            print(
                f"       {_hub_name(child_id)} [{cdata['node_type']}]: "
                f"{len(cdata['gateways'])} gateway(s)"
            )
            for gw, gdata in cdata["gateways"].items():
                print(
                    f"         gateway {gw}: forced={gdata['forced']}, "
                    f"outlying={len(gdata['outlying_stations'])} station(s)"
                )

    return hub_topology


# =============================================================================
# Boundary Station Detection
# =============================================================================

def detect_boundary_station_candidates(
    G: nx.Graph,
    nodes: gpd.GeoDataFrame,
    buffer_geom,
    threshold_m: float = 5000,
) -> List[int]:
    """
    Detect candidate boundary stations: leaf station nodes (degree=1 in the
    working graph) whose geometry lies within `threshold_m` metres of the
    catchment area buffer boundary edge.

    These stations mark where rail lines enter the catchment area from outside —
    the BAV network is spatially clipped at the buffer, so through-stations at
    the boundary appear as degree-1 leaf nodes.

    Returns a sorted list of Betriebspunkt_Nummer (int).
    """
    if buffer_geom is None:
        return []
    buf_boundary = buffer_geom.boundary
    candidates: List[int] = []
    for _, row in nodes.iterrows():
        if pd.isna(row.get("Betriebspunkt_Nummer")):
            continue
        nid = int(row["Betriebspunkt_Nummer"])
        if G.nodes.get(nid, {}).get("node_class") != "station":
            continue
        if nid not in G or G.degree(nid) != 1:
            continue
        pt = row.geometry
        if pt is None or pt.is_empty:
            continue
        if pt.distance(buf_boundary) <= threshold_m:
            candidates.append(nid)
    return sorted(candidates)


# =============================================================================
# Boundary Station — Persistence & Confirmation CLI
# =============================================================================

def _save_boundary_stations(boundary_ids: List[int], path: Path) -> None:
    """Persist confirmed boundary station node IDs to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(boundary_ids, f, indent=2)


def _load_boundary_stations(path: Path) -> Optional[List[int]]:
    """Load persisted boundary station node IDs from JSON. Returns None if absent."""
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_boundary_mapping(mapping: Dict[str, int], path: Path) -> None:
    """Persist stop_id → boundary_node_id mapping to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)


def _load_boundary_mapping(path: Path) -> Optional[Dict[str, int]]:
    """Load persisted stop_id → boundary_node_id mapping. Returns None if absent."""
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _run_boundary_station_confirmation_cli(
    candidates: List[int],
    node_attrs: Dict[int, Dict],
) -> List[int]:
    """
    Show the auto-detected boundary station candidates and let the user
    remove false positives by entering comma-separated list indices.

    Returns the confirmed list of node IDs.
    """
    print("\n  Candidate boundary stations (leaf stations ≤5 km from buffer edge):")
    for i, nid in enumerate(candidates, 1):
        name = node_attrs.get(nid, {}).get("name", str(nid))
        print(f"    {i:3}) {name}  (node {nid})")

    print(
        "\n  Enter comma-separated indices to REMOVE (false positives), "
        "or Enter to accept all:"
    )
    raw = input("  Remove: ").strip()
    if not raw:
        print(f"  All {len(candidates)} candidate(s) accepted.")
        return list(candidates)

    to_remove: set = set()
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit() and 1 <= int(part) <= len(candidates):
            to_remove.add(int(part) - 1)

    confirmed = [nid for i, nid in enumerate(candidates) if i not in to_remove]
    print(f"  Confirmed {len(confirmed)} boundary station(s).")
    return confirmed


# =============================================================================
# Stop Matching
# =============================================================================

_STOP_NAME_SUFFIXES = [
    " bahnhof/hb", " bahnhof", " bhf", ", bahnhof",
    " hb", " station", " gare", " stazione",
]


def normalize_stop_name(name: str) -> str:
    """Lowercase and strip common Swiss station name suffixes for fuzzy matching."""
    s = str(name).lower().strip()
    for suffix in _STOP_NAME_SUFFIXES:
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return s


def build_stop_lookups(nodes: gpd.GeoDataFrame) -> Dict[str, dict]:
    """
    Build lookup tables for stop-to-node matching.

    Returns:
        {
          'by_id':   {Betriebspunkt_Nummer (int): Betriebspunkt_Nummer},
          'by_name': {normalized_name (str): Betriebspunkt_Nummer},
          'by_code': {CODE (str): Betriebspunkt_Nummer},
          'children':{parent_node_id (int): [child_node_id...]}
        }

    Parent-child relationships are built from the `parent_node` column.  The BAV
    geopackage stores parent references as the UUID value found in the `node_id`
    column of the parent node — not as a numeric Betriebspunkt_Nummer.  A first
    pass therefore builds a node_id-UUID → Betriebspunkt_Nummer reverse-lookup so
    that UUID-style parent_node values can be resolved to their numeric parent ID.
    """
    # First pass: build node_id (UUID) → Betriebspunkt_Nummer reverse-lookup
    node_uuid_to_bpnr: Dict[str, int] = {}
    for _, row in nodes.iterrows():
        if pd.isna(row.get("Betriebspunkt_Nummer")):
            continue
        raw_uuid = str(row.get("node_id", "")).strip() if pd.notna(row.get("node_id")) else ""
        if raw_uuid:
            node_uuid_to_bpnr[raw_uuid] = int(row["Betriebspunkt_Nummer"])

    # Second pass: build all lookup tables
    by_id: Dict[int, int] = {}
    by_name: Dict[str, int] = {}
    by_code: Dict[str, int] = {}
    children_by_parent: Dict[int, List[int]] = {}

    for _, row in nodes.iterrows():
        if pd.isna(row.get("Betriebspunkt_Nummer")):
            continue
        nid = int(row["Betriebspunkt_Nummer"])
        by_id[nid] = nid

        # Build parent-child relationships
        if "parent_node" in row and pd.notna(row["parent_node"]):
            raw_pid = str(row["parent_node"]).strip()
            if raw_pid and raw_pid.lower() not in ["none", "nan"]:
                try:
                    # Numeric Betriebspunkt_Nummer stored directly
                    pid = int(float(raw_pid))
                    children_by_parent.setdefault(pid, []).append(nid)
                except ValueError:
                    # UUID — resolve via node_id reverse-lookup
                    pid = node_uuid_to_bpnr.get(raw_pid)
                    if pid is not None:
                        children_by_parent.setdefault(pid, []).append(nid)

        if pd.notna(row.get("NAME")) and row["NAME"]:
            by_name[normalize_stop_name(str(row["NAME"]))] = nid
        if pd.notna(row.get("CODE")) and row["CODE"]:
            by_code[str(row["CODE"]).strip()] = nid

    return {"by_id": by_id, "by_name": by_name, "by_code": by_code, "children": children_by_parent}


def match_stop_to_node(
    stop_id: str,
    stop_name: str,
    E: float,
    N: float,
    nodes: gpd.GeoDataFrame,
    lookups: Dict[str, dict],
) -> MatchResult:
    """
    Match a service stop to a BAV infrastructure node using a 3-tier strategy.

    Tier 1a — Numeric ID:  strip GTFS prefix, compare to Betriebspunkt_Nummer.
    Tier 1b — CODE match:  stop_id treated as a station code (e.g. 'ZUE').
    Tier 2  — Name match:  SequenceMatcher on normalized names (threshold 0.85).
    Tier 3  — Spatial:     nearest node within 200 m.

    Returns MatchResult with node_id=None and method='unmatched' if all tiers fail.
    """
    sid = str(stop_id).strip()

    def _make_result(nid: int, conf: float, meth: str, dist: Optional[float] = None) -> MatchResult:
        candidates = [nid]
        if nid in lookups.get("children", {}):
            candidates.extend(lookups["children"][nid])
        return MatchResult(node_id=nid, confidence=conf, method=meth, distance_m=dist, candidates=candidates)

    # --- Tier 1a: numeric ID match ---
    try:
        numeric_id = int(sid.split(":")[-1]) if ":" in sid else int(sid)
        if numeric_id in lookups["by_id"]:
            return _make_result(numeric_id, 1.0, "id")
    except ValueError:
        pass

    # --- Tier 1b: exact CODE match (for rail station codes like 'ZUE') ---
    if sid in lookups["by_code"]:
        return _make_result(lookups["by_code"][sid], 1.0, "code")

    # --- Tier 2: fuzzy name match ---
    norm_query = normalize_stop_name(stop_name)
    best_ratio = 0.0
    best_id: Optional[int] = None
    for norm_name, nid in lookups["by_name"].items():
        ratio = SequenceMatcher(None, norm_query, norm_name).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_id = nid
    if best_ratio >= NAME_MATCH_THRESHOLD and best_id is not None:
        return _make_result(best_id, best_ratio, "name")

    # --- Tier 3: spatial proximity ---
    if pd.notna(E) and pd.notna(N):
        stop_pt = Point(float(E), float(N))
        distances = nodes.geometry.distance(stop_pt)
        min_idx = distances.idxmin()
        min_dist = float(distances[min_idx])
        if min_dist <= SPATIAL_MATCH_THRESHOLD:
            return _make_result(
                int(nodes.loc[min_idx, "Betriebspunkt_Nummer"]),
                0.7 * (1 - min_dist / SPATIAL_MATCH_THRESHOLD),
                "spatial",
                min_dist,
            )

    return MatchResult(node_id=None, confidence=0.0, method="unmatched")


# =============================================================================
# Tier 4 — Raw BAV Fallback (interactive)
# =============================================================================

def _tier4_raw_bav_fallback(
    stop_id: str,
    stop_name: str,
    E: float,
    N: float,
    raw_nodes: gpd.GeoDataFrame,
    working_nodes: gpd.GeoDataFrame,
    working_segments: gpd.GeoDataFrame,
    infra_version_dir: Path,
) -> Tuple[Optional[MatchResult], gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Tier 4 matching: search the raw (pre-filter) BAV nodes for this stop.

    If a match is found in raw data, ask the user whether to add it (and its
    connecting raw segments) to the working infrastructure version.

    Returns:
        (MatchResult or None, updated_working_nodes, updated_working_segments)

    The caller must rebuild the graph + lookups if working_nodes was extended.
    """
    raw_lookups = build_stop_lookups(raw_nodes)
    raw_result = match_stop_to_node(stop_id, stop_name, E, N, raw_nodes, raw_lookups)

    if raw_result.node_id is None:
        return None, working_nodes, working_segments

    # Found in raw — explain and ask user
    raw_row = raw_nodes[raw_nodes["Betriebspunkt_Nummer"] == raw_result.node_id].iloc[0]
    print(f"\n  [Tier 4] Stop '{stop_name}' not in current version but found in Raw:")
    print(f"    Name:   {raw_row.get('NAME', '?')}")
    print(f"    Code:   {raw_row.get('CODE', '?')}")
    print(f"    Match:  {raw_result.method}  (confidence {raw_result.confidence:.2f})")

    ans = input(
        f"  Add this node and its connecting segments to the working version? (y/n) [n]: "
    ).strip().lower() or "n"

    if ans != "y":
        print(f"  Skipped — '{stop_name}' will use straight-line geometry.")
        return None, working_nodes, working_segments

    # Load raw segments and find those connecting this node
    raw_segments_path = infra_version_dir.parent / "Raw" / "segments.gpkg"
    if not raw_segments_path.exists():
        print(f"  WARNING: Raw segments not found at {raw_segments_path}. Cannot add.")
        return None, working_nodes, working_segments

    raw_segs = gpd.read_file(raw_segments_path)
    node_name = raw_row["NAME"]
    conn_mask = (
        (raw_segs["from_name"] == node_name) | (raw_segs["to_name"] == node_name)
    )
    conn_segs = raw_segs[conn_mask]

    # Append node to working_nodes
    new_node_row = raw_row.to_frame().T.reset_index(drop=True)
    new_node_row = gpd.GeoDataFrame(new_node_row, geometry="geometry", crs=SWISS_CRS)
    working_nodes = pd.concat([working_nodes, new_node_row], ignore_index=True)

    # Append connecting segments to working_segments (only those connecting to
    # existing working nodes — avoids dangling chain additions)
    existing_names = set(working_nodes["NAME"].dropna().tolist())
    for _, seg in conn_segs.iterrows():
        other_name = (
            seg["to_name"] if seg["from_name"] == node_name else seg["from_name"]
        )
        if other_name in existing_names:
            new_seg = seg.to_frame().T.reset_index(drop=True)
            new_seg_gdf = gpd.GeoDataFrame(new_seg, geometry="geometry", crs=SWISS_CRS)
            working_segments = pd.concat(
                [working_segments, new_seg_gdf], ignore_index=True
            )

    # Persist to disk
    working_nodes.to_file(infra_version_dir / "nodes.gpkg", driver="GPKG")
    working_segments.to_file(infra_version_dir / "segments.gpkg", driver="GPKG")
    print(f"  Added node '{node_name}' to working version. Infrastructure files updated.")

    final_result = MatchResult(
        node_id=int(raw_result.node_id),
        confidence=raw_result.confidence,
        method="raw_bav",
        distance_m=raw_result.distance_m,
    )
    return final_result, working_nodes, working_segments


# =============================================================================
# Path Routing
# =============================================================================

def route_between_nodes(
    G: nx.Graph,
    nodes_a: List[int],
    nodes_b: List[int],
    seg_lookup: Dict[Tuple[int, int], pd.Series],
    node_attrs: Dict[int, Dict],
    forced_via: Optional[List[int]] = None,
) -> Tuple[Optional[object], str, str, float, Optional[int], Optional[int], str]:
    """
    Find the shortest path between two BAV nodes and return routing metadata.

    Returns:
        (geometry, via_stations_str, via_junctions_str, path_length_m, chosen_a, chosen_b,
         path_nodes_str)

    path_nodes_str: ';'-joined Betriebspunkt_Nummer values for every node on the path,
        including endpoints. Empty string when no path is found.

    'via_stations_str' — ';'-joined NAMEs of intermediate nodes with node_class='station'.
    'via_junctions_str' — ';'-joined NAMEs of all other intermediate nodes.
    Returns (None, '', '', 0.0, None, None) when no path exists.

    forced_via — optional ordered list of node IDs that must appear on the path.
        The path is stitched as shortest(a, via[0]) + shortest(via[0], via[1]) + …
        + shortest(via[-1], b).  This overrides pure Dijkstra and is used for
        DML through-service routing where the physically longer tunnel must be
        preferred over the shorter surface path.  Falls back to normal Dijkstra
        if any via-node is absent from G or no path exists through the waypoints.
    """
    best_path = None
    best_length = float('inf')

    # Validate forced_via — all waypoints must be graph nodes
    use_forced_via = bool(forced_via) and all(v in G for v in forced_via)

    for a in nodes_a:
        for b in nodes_b:
            if a == b:
                continue
            try:
                if use_forced_via:
                    checkpoints = [a] + list(forced_via) + [b]
                    stitched: List[int] = []
                    for k in range(len(checkpoints) - 1):
                        c_from, c_to = checkpoints[k], checkpoints[k + 1]
                        # Use the direct edge when consecutive waypoints share one —
                        # prevents Dijkstra from substituting a cheaper surface detour
                        # for a physically longer but operationally required segment
                        # (e.g. DML tunnel ZOES↔ZLOE = 5 060 m vs surface ~4 500 m).
                        if G.has_edge(c_from, c_to):
                            sub_path = [c_from, c_to]
                        else:
                            sub_path = nx.shortest_path(G, c_from, c_to, weight="weight")
                        if k == 0:
                            stitched.extend(sub_path)
                        else:
                            stitched.extend(sub_path[1:])  # drop duplicate junction node
                    path: List[int] = stitched
                else:
                    path = nx.shortest_path(G, a, b, weight="weight")
                path_length = sum(float(G[path[i]][path[i+1]].get("weight", 0)) for i in range(len(path)-1))
                if path_length < best_length:
                    best_length = path_length
                    best_path = path

            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

    if not best_path:
        return None, "", "", 0.0, None, None, ""

    path = best_path
    chosen_a = path[0]
    chosen_b = path[-1]

    # Collect segment geometries along path
    geoms = []
    total_length = 0.0
    for i in range(len(path) - 1):
        seg = seg_lookup.get((path[i], path[i + 1]))
        if seg is not None and seg.geometry is not None:
            raw_geom = seg.geometry
            if raw_geom.geom_type == "LineString":
                geoms.append(raw_geom)
            elif hasattr(raw_geom, 'geoms'):
                geoms.extend([g for g in raw_geom.geoms if g.geom_type == "LineString"])
            total_length += float(G[path[i]][path[i + 1]].get("length_m", 0))

    final_geom = linemerge(geoms) if geoms else None

    # Classify intermediate nodes
    via_stations: List[str] = []
    via_junctions: List[str] = []
    for nid in path[1:-1]:
        attrs = node_attrs.get(nid, {})
        name = attrs.get("name", str(nid))
        if attrs.get("node_class") == "station":
            via_stations.append(name)
        else:
            via_junctions.append(name)

    path_nodes_str = ";".join(str(n) for n in path)

    return (
        final_geom,
        ";".join(via_stations),
        ";".join(via_junctions),
        total_length,
        chosen_a,
        chosen_b,
        path_nodes_str,
    )


# =============================================================================
# Stop Sequence Builders
# =============================================================================

def build_stop_sequence_rail(
    edges: gpd.GeoDataFrame,
) -> Dict[Tuple[str, str], List[Dict]]:
    """
    Build ordered stop sequences from edges_in_corridor.gpkg.

    Groups edges by (Service, Direction), sorts by 'Link NR', then builds a
    de-duplicated ordered list of stops.

    Each stop dict: {stop_id, stop_name, E, N}
    stop_id = FromCode (e.g. 'WS'), used for matching by code.

    Returns: {(service, direction): [stop_dict, ...]}
    """
    sequences: Dict[Tuple[str, str], List[Dict]] = {}

    for (svc, direction), group in edges.groupby(["Service", "Direction"]):
        group_sorted = group.sort_values("Link NR")
        stops: List[Dict] = []
        seen_ids: set = set()

        for _, row in group_sorted.iterrows():
            for prefix in ("From", "To"):
                sid = str(row.get(f"{prefix}Code", "")).strip()
                sname = str(row.get(f"{prefix}Station", "")).strip()
                E_col = "x_origin" if prefix == "From" else "x_dest"
                N_col = "y_origin" if prefix == "From" else "y_dest"
                E = float(row.get(E_col, 0))
                N = float(row.get(N_col, 0))

                if sid and sid not in seen_ids:
                    seen_ids.add(sid)
                    stops.append(
                        {"stop_id": sid, "stop_name": sname, "E": E, "N": N}
                    )

        if len(stops) >= 2:
            sequences[(str(svc), str(direction))] = stops

    return sequences


def build_stop_sequence_feeder(
    segments: gpd.GeoDataFrame,
) -> Dict[Tuple[str, int, int], List[Dict]]:
    """
    Build ordered stop sequences from pt_feeder_segments.gpkg.

    Groups by (route_id, direction_id, variant_rank) and chains consecutive
    from_stop_id → to_stop_id pairs into a de-duplicated ordered list.

    Each stop dict: {stop_id, stop_name, E, N}

    Returns: {(route_id, direction_id, variant_rank): [stop_dict, ...]}
    """
    sequences: Dict[Tuple[str, int, int], List[Dict]] = {}

    group_cols = ["route_id", "direction_id", "variant_rank"]
    for key_vals, group in segments.groupby(group_cols):
        key = (str(key_vals[0]), int(key_vals[1]), int(key_vals[2]))
        stops: List[Dict] = []
        seen_ids: set = set()

        for _, row in group.iterrows():
            for prefix in ("from", "to"):
                sid = str(row.get(f"{prefix}_stop_id", "")).strip()
                sname = str(row.get(f"{prefix}_stop_name", "")).strip()
                E = float(row.get(f"{prefix}_stop_E", 0))
                N = float(row.get(f"{prefix}_stop_N", 0))

                if sid and sid not in seen_ids:
                    seen_ids.add(sid)
                    stops.append(
                        {"stop_id": sid, "stop_name": sname, "E": E, "N": N}
                    )

        if len(stops) >= 2:
            sequences[key] = stops

    return sequences


# =============================================================================
# Service Enrichment
# =============================================================================

_NEW_COLS = [
    "node_id_from", "node_id_to",
    "match_method_from", "match_method_to",
    "Via_Station", "Via_Junction",
    "path_length_m", "needs_correction",
    "path_nodes",
    "elec_mismatch",
]


def _apply_enrichment(
    row_idx,
    from_id: str,
    from_name: str,
    from_E: float,
    from_N: float,
    to_id: str,
    to_name: str,
    to_E: float,
    to_N: float,
    nodes: gpd.GeoDataFrame,
    bav_segments: gpd.GeoDataFrame,
    G: nx.Graph,
    seg_lookup: Dict,
    node_attrs: Dict,
    lookups: Dict,
    buffer_geom,
    match_cache: Dict[str, MatchResult],
    raw_nodes: Optional[gpd.GeoDataFrame],
    raw_segments: Optional[gpd.GeoDataFrame],
    infra_version_dir: Optional[Path],
    stop_overrides: Optional[Dict[str, int]] = None,
    hub_topology: Optional[Dict] = None,
    gauge_graphs: Optional[Dict] = None,
) -> Tuple[Dict, gpd.GeoDataFrame, gpd.GeoDataFrame, nx.Graph, Dict, Dict]:
    """
    Core enrichment logic for a single service link (from-stop → to-stop).

    Returns:
        (enrichment_dict, possibly-updated nodes, segments, G, seg_lookup, match_cache)
    """
    _gauge_graphs = gauge_graphs  # local copy — may be rebuilt after Tier 4
    # Match FROM stop (use cache)
    if from_id not in match_cache:
        r = match_stop_to_node(from_id, from_name, from_E, from_N, nodes, lookups)
        # Tier 4 if unmatched + inside buffer
        if r.method == "unmatched" and raw_nodes is not None:
            from_pt = Point(from_E, from_N)
            if buffer_geom is not None and from_pt.within(buffer_geom):
                r4, nodes, bav_segments = _tier4_raw_bav_fallback(
                    from_id, from_name, from_E, from_N,
                    raw_nodes, nodes, bav_segments, infra_version_dir
                )
                if r4 is not None:
                    r = r4
                    # Rebuild graph + lookups after infra change
                    G = build_infra_graph(nodes, bav_segments, raw_nodes)
                    seg_lookup = build_segment_lookup(nodes, bav_segments, raw_nodes)
                    node_attrs.update(build_node_attrs(nodes))
                    lookups = build_stop_lookups(nodes)
                    if _gauge_graphs is not None:
                        _gauge_graphs = _build_gauge_graphs(G)
        match_cache[from_id] = r
    match_from = match_cache[from_id]

    # Match TO stop (use cache)
    if to_id not in match_cache:
        r = match_stop_to_node(to_id, to_name, to_E, to_N, nodes, lookups)
        if r.method == "unmatched" and raw_nodes is not None:
            to_pt = Point(to_E, to_N)
            if buffer_geom is not None and to_pt.within(buffer_geom):
                r4, nodes, bav_segments = _tier4_raw_bav_fallback(
                    to_id, to_name, to_E, to_N,
                    raw_nodes, nodes, bav_segments, infra_version_dir
                )
                if r4 is not None:
                    r = r4
                    G = build_infra_graph(nodes, bav_segments, raw_nodes)
                    seg_lookup = build_segment_lookup(nodes, bav_segments, raw_nodes)
                    node_attrs.update(build_node_attrs(nodes))
                    lookups = build_stop_lookups(nodes)
        match_cache[to_id] = r
    match_to = match_cache[to_id]

    # Apply per-service pre-selected node overrides: replace candidates with the
    # single pre-chosen child node so routing is consistent across directions.
    if stop_overrides:
        if from_id in stop_overrides:
            forced = stop_overrides[from_id]
            match_from = MatchResult(
                node_id=forced, confidence=match_from.confidence,
                method=match_from.method, candidates=[forced],
            )
        if to_id in stop_overrides:
            forced = stop_overrides[to_id]
            match_to = MatchResult(
                node_id=forced, confidence=match_to.confidence,
                method=match_to.method, candidates=[forced],
            )

    # Decide geometry: route if both matched + both inside buffer; else straight-line
    from_pt = Point(from_E, from_N)
    to_pt = Point(to_E, to_N)
    inside_from = buffer_geom is None or from_pt.within(buffer_geom)
    inside_to = buffer_geom is None or to_pt.within(buffer_geom)

    via_st, via_jn, path_len = "", "", 0.0
    path_nodes_str = ""
    needs_correction = False
    elec_mismatch = False

    node_id_from = match_from.node_id
    node_id_to = match_to.node_id

    if match_from.candidates and match_to.candidates and inside_from and inside_to:
        # Generalised forced_via: for hub children whose direct approach edge is
        # physically longer than the surface detour (e.g. the DML tunnel), find
        # the natural gateway the train passes through on its approach and force
        # routing through it when that gateway has forced=True.
        #
        # "Natural gateway" is determined by removing all forced direct edges
        # (gateway→child edges flagged forced=True) from a temporary graph and
        # running Dijkstra from other_id to child_id.  The first gateway of
        # child_id that appears on that surface path is the approach gateway.
        # Forced_via is only applied when that gateway's forced flag is True —
        # this prevents stations that happen to be adjacent to a forced gateway
        # (e.g. Wipkingen adjacent to ZOES) from being routed back through the
        # DML tunnel they did not arrive via.
        forced_via: Optional[List[int]] = None
        if hub_topology:
            for child_id, other_id in [
                (match_from.node_id, match_to.node_id),
                (match_to.node_id,   match_from.node_id),
            ]:
                if child_id is None or other_id is None:
                    continue
                for hub_id, hub_data in hub_topology.items():
                    children = hub_data.get("children", {})
                    if child_id not in children:
                        continue
                    gw_entries = children[child_id]["gateways"]

                    # Build a graph with all forced gateway→child direct edges removed
                    # so that Dijkstra reveals the natural surface approach path.
                    G_surface = G.copy()
                    for gw, gdata in gw_entries.items():
                        if gdata.get("forced") and G_surface.has_edge(gw, child_id):
                            G_surface.remove_edge(gw, child_id)

                    try:
                        surface_path = nx.shortest_path(
                            G_surface, other_id, child_id, weight="weight"
                        )
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        surface_path = []

                    # Find the first gateway of child_id on this surface path
                    natural_gw: Optional[int] = None
                    for node in surface_path:
                        if node in gw_entries:
                            natural_gw = node
                            break

                    if (
                        natural_gw is not None
                        and gw_entries[natural_gw].get("forced")
                        and natural_gw in G
                    ):
                        forced_via = [natural_gw]
                    break
                if forced_via is not None:
                    break

        # Hard constraint: route on gauge-filtered graph when gauge is determinable.
        G_route = G
        expected_gauge = None
        if _gauge_graphs and match_from.node_id is not None:
            expected_gauge = _node_gauge(match_from.node_id, G)
            if expected_gauge is not None and expected_gauge in _gauge_graphs:
                G_route = _gauge_graphs[expected_gauge]

        # Relaxed constraint: expected electrification from from-node incident edges.
        expected_elec = None
        if match_from.node_id is not None:
            expected_elec = _node_electrification(match_from.node_id, G)

        geom, via_st, via_jn, path_len, chosen_from, chosen_to, path_nodes_str = route_between_nodes(
            G_route, match_from.candidates, match_to.candidates, seg_lookup, node_attrs,
            forced_via=forced_via,
        )
        if geom is not None:
            node_id_from = chosen_from
            node_id_to = chosen_to
        if geom is None:
            # No path found despite matched nodes
            geom = LineString([(from_E, from_N), (to_E, to_N)])
            path_nodes_str = ""
            needs_correction = True

        # Electrification post-check along routed path (relaxed — flag only).
        if path_nodes_str and expected_elec is not None:
            node_ids = [int(n) for n in path_nodes_str.split(";") if n.strip()]
            for i in range(len(node_ids) - 1):
                u, v = node_ids[i], node_ids[i + 1]
                try:
                    edata = G.edges[u, v]
                except KeyError:
                    continue
                e = edata.get("electrification")
                if e is not None and not (isinstance(e, float) and np.isnan(e)):
                    if str(e).strip() != expected_elec:
                        elec_mismatch = True
                        break
    elif not inside_from or not inside_to:
        # Outside buffer — straight-line is expected
        geom = LineString([(from_E, from_N), (to_E, to_N)])
    else:
        # Unmatched stop inside buffer — flag for correction
        geom = LineString([(from_E, from_N), (to_E, to_N)])
        needs_correction = match_from.node_id is None or match_to.node_id is None
        if needs_correction:
            print(
                f"  WARNING: Stop unmatched — straight-line fallback "
                f"({from_name or from_id} → {to_name or to_id})"
            )

    enrichment = {
        "node_id_from": node_id_from,
        "node_id_to": node_id_to,
        "match_method_from": match_from.method,
        "match_method_to": match_to.method,
        "Via_Station": via_st,
        "Via_Junction": via_jn,
        "path_length_m": path_len,
        "needs_correction": needs_correction,
        "path_nodes": path_nodes_str,
        "elec_mismatch": elec_mismatch,
        "geometry": geom,
    }
    return enrichment, nodes, bav_segments, G, seg_lookup, match_cache


def _get_hub_node_type(c: int, hub_id: int, hub: dict) -> str:
    """
    Return the node_type ('terminal' or 'through') for candidate c within hub.

    If c is the hub parent itself, returns hub['hub_node_type'].
    If c is a known child, returns its recorded node_type.
    Defaults to 'terminal' for any node that is neither the hub parent nor a known child (safe fallback).
    """
    if c == hub_id:
        return hub.get("hub_node_type", "terminal")
    return hub.get("children", {}).get(c, {}).get("node_type", "terminal")


def _nearest_outlying(
    ref: Optional[int],
    all_outlying: frozenset,
    G: nx.Graph,
) -> Optional[int]:
    """
    Return the node in all_outlying with the shortest graph distance to ref.

    If ref is itself in all_outlying it is returned directly (distance = 0).
    Returns None if ref is None or all_outlying is empty.
    Returns None also if no outlying station is graph-reachable from ref.
    """
    if ref is None or not all_outlying:
        return None
    if ref in all_outlying:
        return ref
    best: Optional[int] = None
    best_d = float("inf")
    for o in all_outlying:
        try:
            d = nx.shortest_path_length(G, ref, o, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            d = float("inf")
        if d < best_d:
            best_d, best = d, o
    return best


def _is_same_gateway(
    child_id: int,
    prev_node: Optional[int],
    next_node: Optional[int],
    hub_children: dict,
    G: nx.Graph,
) -> bool:
    """
    Return True if prev_node and next_node both approach child_id from the
    same physical gateway — i.e. the service would enter and exit via the
    same segment (backtracking).

    Uses a surface-path approach: forced gateway→child edges are removed from
    the graph so that the natural (surface-level) path to child_id is found.
    The first gateway node encountered on that path is the natural gateway for
    that approach direction.  If both prev_node and next_node resolve to the
    same natural gateway, the service is backtracking.

    Falls back to True (safe / conservative) when either node is None, when
    child_id has fewer than two gateways, or when no surface path exists.
    """
    gw_entries = hub_children.get(child_id, {}).get("gateways", {})
    gateways = list(gw_entries.keys())
    if len(gateways) < 2:
        return True  # terminal child — no valid through route

    # Build surface graph: remove direct forced-gateway→child edges so that
    # Dijkstra must use the real track approach rather than shortcuts.
    G_surface = G.copy()
    for gw, gdata in gw_entries.items():
        if gdata.get("forced") and G_surface.has_edge(gw, child_id):
            G_surface.remove_edge(gw, child_id)

    def _natural_gw(ref: Optional[int]) -> Optional[int]:
        if ref is None:
            return None
        # If the stop itself is a gateway (e.g. Stadelhofen as next stop for
        # ZMUS), return it directly.
        if ref in gw_entries:
            return ref
        # Check exclusive BFS outlying membership: if ref appears in exactly one
        # gateway's outlying set, that gateway is the unambiguous approach side
        # (e.g. Oerlikon → ZOES only, Wiedikon → Langstrasse only).
        # This is robust to forced gateways that have no surface path alternative
        # (where removing the forced edge would disconnect them from child_id).
        containing = [gw for gw in gateways
                      if ref in gw_entries[gw].get("outlying_stations", frozenset())]
        if len(containing) == 1:
            return containing[0]
        if len(containing) > 1:
            # Ambiguous (in multiple outlying sets, e.g. Wipkingen reachable from
            # both ZOES and Langstrasse): use surface-path to determine the
            # natural approach direction.
            try:
                path = nx.shortest_path(G_surface, ref, child_id, weight="weight")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return None
            for node in path[1:]:
                if node in gw_entries:
                    return node
            return None
        # In zero outlying sets (e.g. Altstetten not reached by any gateway BFS,
        # or Oerlikon not reached when ZOES's BFS is blocked by an intermediate
        # station): fall back to nearest gateway by graph distance on the full G.
        # This was the original working approach and is correct for unambiguous
        # nodes that simply fall outside all BFS-reachable outlying sets.
        best_gw, best_d = None, float("inf")
        for gw in gateways:
            try:
                d = nx.shortest_path_length(G, ref, gw, weight="weight")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            if d < best_d:
                best_d, best_gw = d, gw
        return best_gw

    gw_prev = _natural_gw(prev_node)
    gw_next = _natural_gw(next_node)

    if gw_prev is None or gw_next is None:
        return True  # cannot determine direction → assume backtracking (safe)
    return gw_prev == gw_next


def _preselect_rail_stop_nodes(
    edges: gpd.GeoDataFrame,
    G: nx.Graph,
    lookups: Dict,
    nodes: gpd.GeoDataFrame,
    hub_topology: Dict,
) -> Dict[Tuple[str, str], Dict[str, int]]:
    """
    For each (Service, Direction) group, pre-select the best BAV node for every
    stop that has multiple routing candidates (i.e. parent nodes with children).

    Uses hub topology (build_hub_topology) to distinguish:
      - Valid through-running candidates: prev and next stops lie in different
        exclusive approach zones of the candidate child node.
      - Reversal candidates: prev and next both approach from the same side —
        service legitimately reverses at this hub.  In this case terminal nodes
        (dead-end platforms) are preferred over through-running children.

    For terminal stops (first or last in the sequence) the candidate with the
    shortest approach distance is chosen, with a preference boost of
    TERMINAL_PREFERENCE_S for terminal (single-approach) nodes.

    For stops not associated with any hub in hub_topology the original bilateral
    cost minimisation is used unchanged.

    Returns: {(service, direction): {stop_id: forced_node_id}}
    Only stops where the selected node differs from the default match are included.
    """

    def _bilateral(c: int, prev_node: Optional[int], next_node: Optional[int]) -> float:
        cost = 0.0
        for ref in (prev_node, next_node):
            if ref is None or c not in G:
                continue
            try:
                cost += nx.shortest_path_length(G, ref, c, weight="weight")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                cost += 1e9
        return cost

    def _approach_cost(c: int, ref: Optional[int]) -> float:
        if ref is None or c not in G:
            return 1e9
        try:
            return nx.shortest_path_length(G, ref, c, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return 1e9

    overrides: Dict[Tuple, Dict[str, int]] = {}

    # Group by variant as well when available so that multi-variant services
    # (e.g. S21 with variant_rank 1 and 2) are not merged into one stop sequence.
    # Merging would cause a variant's branch stops to appear after the terminal,
    # incorrectly promoting the terminal stop to a through-service position.
    has_variant = "variant_rank" in edges.columns
    group_cols = ["Service", "Direction", "variant_rank"] if has_variant else ["Service", "Direction"]

    for group_key, group in edges.groupby(group_cols):
        svc       = group_key[0]
        direction = group_key[1]
        group_sorted = group.sort_values("Link NR") if "Link NR" in group.columns else group

        # Build ordered stop list with coordinates
        stop_list: List[Dict] = []
        seen: set = set()
        for _, row in group_sorted.iterrows():
            for code_col, name_col, e_col, n_col in [
                ("FromCode", "FromStation", "x_origin", "y_origin"),
                ("ToCode",   "ToStation",   "x_dest",   "y_dest"),
            ]:
                sid = str(row.get(code_col, "")).strip()
                if sid and sid not in seen:
                    seen.add(sid)
                    stop_list.append({
                        "stop_id":   sid,
                        "stop_name": str(row.get(name_col, "")),
                        "E":         float(row.get(e_col, 0)),
                        "N":         float(row.get(n_col, 0)),
                    })

        if len(stop_list) < 2:
            continue

        # Quick-match each stop (no Tier 4 — pre-pass only)
        stop_matches: Dict[str, MatchResult] = {}
        for s in stop_list:
            stop_matches[s["stop_id"]] = match_stop_to_node(
                s["stop_id"], s["stop_name"], s["E"], s["N"], nodes, lookups
            )

        # Skip service if no stop has multiple candidates
        if not any(len(r.candidates) > 1 for r in stop_matches.values()):
            continue

        # Greedy left-to-right selection
        stop_node_choice: Dict[str, int] = {}
        for i, s in enumerate(stop_list):
            sid = s["stop_id"]
            r = stop_matches[sid]

            if not r.candidates or r.node_id is None:
                stop_node_choice[sid] = r.node_id
                continue

            if len(r.candidates) == 1:
                stop_node_choice[sid] = r.candidates[0]
                continue

            prev_node = stop_node_choice.get(stop_list[i - 1]["stop_id"]) if i > 0 else None
            next_r    = stop_matches.get(stop_list[i + 1]["stop_id"]) if i < len(stop_list) - 1 else None
            next_node = next_r.node_id if next_r else None

            if prev_node is None and next_node is None:
                stop_node_choice[sid] = r.node_id
                continue

            hub_id = r.node_id  # default match is the parent node
            # i==0 (first stop) is treated as terminating: no prev_node available,
            # so the crossing table lookup would degrade to bilateral anyway, and
            # terminal-node preference gives better results for origin stops.
            is_through = (0 < i < len(stop_list) - 1)

            if hub_id in hub_topology:
                # ── Hub-topology-aware selection (DML logic) ──────────────────
                hub = hub_topology[hub_id]
                valid_cands = [c for c in r.candidates if c is not None and c in G]
                if not valid_cands:
                    stop_node_choice[sid] = r.node_id
                    continue

                if is_through:
                    # Step (d): through service — use crossing table
                    all_outlying    = hub.get("all_outlying", frozenset())
                    crossing_table  = hub.get("crossing_table", {})
                    from_outlying   = _nearest_outlying(prev_node, all_outlying, G)
                    to_outlying     = _nearest_outlying(next_node,  all_outlying, G)

                    # Build crossing key only when the two outlying stations differ
                    key = (
                        frozenset({from_outlying, to_outlying})
                        if from_outlying and to_outlying and from_outlying != to_outlying
                        else None
                    )
                    through_child = crossing_table.get(key) if key else None

                    # One-sided fallback: exactly one of from_outlying / to_outlying
                    # is None because one adjacent stop is outside the study buffer
                    # (e.g. IC1/IC5 approaching from Bern/Aarau or departing to Bern/
                    # Aarau).  Use the resolved side to scan the crossing table.
                    # all_outlying includes hub-parent perimeter stations that never
                    # appear in crossing table keys, so re-resolve against ct_outlying
                    # (the set of stations that actually appear in crossing table keys).
                    # If the scan is ambiguous, prefer the child with a forced gateway —
                    # forced gateways mark DML-type tunnels used exclusively by
                    # long-distance through services.
                    # _skip_backtrack bypasses _is_same_gateway which cannot evaluate
                    # direction when one adjacent node is None.
                    _skip_backtrack = False
                    _one_sided = (
                        through_child is None
                        and (from_outlying is None) != (to_outlying is None)
                    )
                    if _one_sided:
                        ct_children = hub.get("children", {})
                        ct_outlying = frozenset(
                            o for key_ct in crossing_table.keys() for o in key_ct
                        )
                        # Use whichever side IS resolved; re-resolve against ct_outlying
                        ref_node = next_node if from_outlying is None else prev_node
                        resolved_ct = (
                            _nearest_outlying(ref_node, ct_outlying, G)
                            if ct_outlying else None
                        )
                        if resolved_ct is not None:
                            matching = {
                                child for key_ct, child in crossing_table.items()
                                if resolved_ct in key_ct and child in valid_cands
                            }
                            if len(matching) == 1:
                                through_child = next(iter(matching))
                                _skip_backtrack = True
                            elif len(matching) > 1:
                                forced_side = {
                                    c for c in matching
                                    if any(
                                        gdata.get("forced")
                                        for gdata in ct_children.get(c, {}).get("gateways", {}).values()
                                    )
                                }
                                if len(forced_side) == 1:
                                    through_child = next(iter(forced_side))
                                    _skip_backtrack = True

                    if through_child is not None and through_child in valid_cands:
                        # Backtracking check: do prev and next approach from the
                        # same physical gateway of through_child?  Skip when the
                        # child was resolved via forced-gateway fallback — the forced
                        # gateway is physical proof of through-running and
                        # _is_same_gateway would give a false positive with
                        # prev_node=None.
                        children = hub.get("children", {})
                        if not _skip_backtrack and _is_same_gateway(
                            through_child, prev_node, next_node, children, G
                        ):
                            is_through = False  # falls through to terminating branch below
                        else:
                            best = through_child
                    else:
                        # No crossing table match — bilateral cost fallback with
                        # terminal preference.  Pairs with no crossing entry are
                        # typically same-side (both approaching from Langstrasse,
                        # etc.) and would backtrack through any through-running
                        # child.  Apply TERMINAL_PREFERENCE_S tolerance so a
                        # terminal node wins over a through-running child by a
                        # thin cost margin.
                        _blt = lambda c: _bilateral(c, prev_node, next_node)
                        _cheapest_all_b  = min(valid_cands, key=_blt)
                        _term_cands_b    = [
                            c for c in valid_cands
                            if _get_hub_node_type(c, hub_id, hub) == "terminal"
                        ]
                        if _term_cands_b:
                            _cheapest_term_b = min(_term_cands_b, key=_blt)
                            _extra_b = _blt(_cheapest_term_b) - _blt(_cheapest_all_b)
                            best = _cheapest_term_b if _extra_b <= TERMINAL_PREFERENCE_S else _cheapest_all_b
                        else:
                            best = _cheapest_all_b

                # Note: 'if not is_through' (not elif) — is_through may have been
                # mutated to False inside the through-branch (backtracking case).
                if not is_through:
                    # Step (e): terminating service (or backtracking through service)
                    # Prefer terminal child nodes; apply TERMINAL_PREFERENCE_S tolerance.
                    approaching    = prev_node if i > 0 else next_node
                    cheapest_all   = min(valid_cands, key=lambda c: _approach_cost(c, approaching))
                    terminal_cands = [
                        c for c in valid_cands
                        if _get_hub_node_type(c, hub_id, hub) == "terminal"
                    ]
                    if terminal_cands:
                        cheapest_term = min(terminal_cands, key=lambda c: _approach_cost(c, approaching))
                        extra = (
                            _approach_cost(cheapest_term, approaching)
                            - _approach_cost(cheapest_all, approaching)
                        )
                        best = cheapest_term if extra <= TERMINAL_PREFERENCE_S else cheapest_all
                    else:
                        best = cheapest_all

            else:
                # ── Standard bilateral cost minimisation (non-hub stops) ──────
                best = r.node_id
                best_cost = float("inf")
                for candidate in r.candidates:
                    if candidate is None or candidate not in G:
                        continue
                    cost = _bilateral(candidate, prev_node, next_node)
                    if cost < best_cost:
                        best_cost = cost
                        best = candidate

            stop_node_choice[sid] = best

        # Pin ALL multi-candidate stops to their pre-selected node, even when
        # the pre-selected node happens to equal the default match node_id.
        # Without this, _apply_enrichment passes the full candidate list to
        # route_between_nodes, which picks the cheapest graph distance —
        # e.g. a through-running child 61 m closer than the correct terminal
        # platform — overriding the hub-topology decision made here.
        svc_overrides = {
            sid: chosen
            for sid, chosen in stop_node_choice.items()
            if chosen is not None
            and len(stop_matches[sid].candidates) > 1
        }
        if svc_overrides:
            key = (str(svc), str(direction), str(group_key[2])) if has_variant else (str(svc), str(direction))
            overrides[key] = svc_overrides

    return overrides


def enrich_rail_links(
    edges: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    bav_segments: gpd.GeoDataFrame,
    G: nx.Graph,
    seg_lookup: Dict,
    node_attrs: Dict,
    lookups: Dict,
    buffer_geom,
    raw_nodes: Optional[gpd.GeoDataFrame],
    raw_segments: Optional[gpd.GeoDataFrame],
    infra_version_dir: Optional[Path],
    hub_topology: Optional[Dict] = None,
    gauge_graphs: Optional[Dict] = None,
) -> gpd.GeoDataFrame:
    """
    Enrich edges_in_corridor.gpkg with real infrastructure geometry.
    Returns the input GeoDataFrame with new columns appended and geometry replaced.
    """
    stop_overrides_by_service = _preselect_rail_stop_nodes(
        edges, G, lookups, nodes, hub_topology or {}
    )

    enriched_rows = []
    match_cache: Dict[str, MatchResult] = {}
    _has_variant = "variant_rank" in edges.columns

    for idx, row in edges.iterrows():
        if _has_variant:
            svc_key = (
                str(row.get("Service", "")),
                str(row.get("Direction", "")),
                str(row.get("variant_rank", "")),
            )
        else:
            svc_key = (str(row.get("Service", "")), str(row.get("Direction", "")))
        stop_overrides = stop_overrides_by_service.get(svc_key, {})

        enrichment, nodes, bav_segments, G, seg_lookup, match_cache = _apply_enrichment(
            idx,
            str(row.get("FromCode", "")), str(row.get("FromStation", "")),
            float(row.get("x_origin", 0)), float(row.get("y_origin", 0)),
            str(row.get("ToCode", "")), str(row.get("ToStation", "")),
            float(row.get("x_dest", 0)), float(row.get("y_dest", 0)),
            nodes, bav_segments, G, seg_lookup, node_attrs, lookups,
            buffer_geom, match_cache, raw_nodes, raw_segments, infra_version_dir,
            stop_overrides=stop_overrides,
            hub_topology=hub_topology,
            gauge_graphs=gauge_graphs,
        )
        new_row = row.to_dict()
        new_row.update(enrichment)
        enriched_rows.append(new_row)

    result = gpd.GeoDataFrame(enriched_rows, crs=SWISS_CRS)
    return result


def enrich_feeder_segments(
    segments: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    bav_segments: gpd.GeoDataFrame,
    G: nx.Graph,
    seg_lookup: Dict,
    node_attrs: Dict,
    lookups: Dict,
    buffer_geom,
    raw_nodes: Optional[gpd.GeoDataFrame],
    raw_segs: Optional[gpd.GeoDataFrame],
    infra_version_dir: Optional[Path],
    hub_topology: Optional[Dict] = None,
    gauge_graphs: Optional[Dict] = None,
) -> gpd.GeoDataFrame:
    """
    Enrich pt_feeder_segments.gpkg with real infrastructure geometry.
    Returns the input GeoDataFrame with new columns appended and geometry replaced.
    """
    enriched_rows = []
    match_cache: Dict[str, MatchResult] = {}

    for idx, row in segments.iterrows():
        enrichment, nodes, bav_segments, G, seg_lookup, match_cache = _apply_enrichment(
            idx,
            str(row.get("from_stop_id", "")), str(row.get("from_stop_name", "")),
            float(row.get("from_stop_E", 0)), float(row.get("from_stop_N", 0)),
            str(row.get("to_stop_id", "")), str(row.get("to_stop_name", "")),
            float(row.get("to_stop_E", 0)), float(row.get("to_stop_N", 0)),
            nodes, bav_segments, G, seg_lookup, node_attrs, lookups,
            buffer_geom, match_cache, raw_nodes, raw_segs, infra_version_dir,
            hub_topology=hub_topology,
            gauge_graphs=gauge_graphs,
        )
        new_row = row.to_dict()
        new_row.update(enrichment)
        enriched_rows.append(new_row)

    result = gpd.GeoDataFrame(enriched_rows, crs=SWISS_CRS)
    return result

# =============================================================================
# QGIS Project (.qgz) Helpers
# =============================================================================
# Functions and templates below are the minimal subset needed to produce styled
# QGIS project files.  They mirror the implementation in services_network_builder.py
# so both scripts produce visually identical projects.

import zipfile as _zipfile

_QGIS_VERSION = "3.44.9-Solothurn"

_QGZ_SRS_BLOCK = """<spatialrefsys nativeFormat="wkt">
      <proj4>+proj=somerc +lat_0=46.9524055555556 +lon_0=7.43958333333333 +k_0=1 +x_0=2600000 +y_0=1200000 +ellps=bessel +towgs84=674.374,15.056,405.346,0,0,0,0 +units=m +no_defs</proj4>
      <srsid>47</srsid>
      <srid>2056</srid>
      <authid>EPSG:2056</authid>
      <description>CH1903+ / LV95</description>
      <projectionacronym>somerc</projectionacronym>
      <ellipsoidacronym>bessel</ellipsoidacronym>
    </spatialrefsys>"""

_QGZ_WMS_LAYER_ID   = "Swisstopo_National_Map__grey__e16b0296_87b7_4e32_b8e8_b46b5990275e"
_QGZ_WMS_LAYER_NAME = "Swisstopo National Map (grey)"
_QGZ_WMS_SOURCE     = (
    "contextualWMSLegend=0&amp;crs=EPSG:2056&amp;dpiMode=7"
    "&amp;featureCount=10&amp;format=image/png"
    "&amp;layers=ch.swisstopo.pixelkarte-grau"
    "&amp;styles=&amp;url=http://wms.geo.admin.ch/"
)


def _qgz_hex_to_rgba(hex_colour: str, alpha: int = 255) -> str:
    """Convert '#RRGGBB' to QGIS RGBA string 'R,G,B,A'."""
    h = hex_colour.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b},{alpha}"


def _qgz_line_maplayer(layer_id, gpkg_relpath, layer_name, display_name, rgba, line_style, width="0.5"):
    pen = "dash" if line_style == "dashed" else "solid"
    return f"""  <maplayer geometry="Line" type="vector" hasScaleBasedVisibilityFlag="0">
    <id>{layer_id}</id>
    <datasource>{gpkg_relpath}|layername={layer_name}</datasource>
    <layername>{display_name}</layername>
    <provider encoding="UTF-8">ogr</provider>
    <srs>{_QGZ_SRS_BLOCK}</srs>
    <renderer-v2 forceraster="0" symbollevels="0" type="singleSymbol" enableorderby="0">
      <symbols>
        <symbol alpha="1" clip_to_extent="1" type="line" name="0" force_rhr="0">
          <layer pass="0" class="SimpleLine" locked="0" enabled="1">
            <prop k="capstyle" v="square"/>
            <prop k="customdash" v="5;2"/>
            <prop k="customdash_map_unit_scale" v="3x:0,0,0,0,0,0"/>
            <prop k="customdash_unit" v="MM"/>
            <prop k="draw_inside_polygon" v="0"/>
            <prop k="joinstyle" v="bevel"/>
            <prop k="line_color" v="{rgba}"/>
            <prop k="line_style" v="{pen}"/>
            <prop k="line_width" v="{width}"/>
            <prop k="line_width_unit" v="MM"/>
            <prop k="offset" v="0"/>
            <prop k="offset_map_unit_scale" v="3x:0,0,0,0,0,0"/>
            <prop k="offset_unit" v="MM"/>
            <prop k="use_custom_dash" v="0"/>
            <prop k="width_map_unit_scale" v="3x:0,0,0,0,0,0"/>
          </layer>
        </symbol>
      </symbols>
      <rotation/>
      <sizescale/>
    </renderer-v2>
  </maplayer>"""


def _qgz_marker_maplayer(layer_id, gpkg_relpath, layer_name, display_name, fill_rgba, outline_rgba, size="2", outline_width="0.2"):
    return f"""  <maplayer geometry="Point" type="vector" hasScaleBasedVisibilityFlag="0">
    <id>{layer_id}</id>
    <datasource>{gpkg_relpath}|layername={layer_name}</datasource>
    <layername>{display_name}</layername>
    <provider encoding="UTF-8">ogr</provider>
    <srs>{_QGZ_SRS_BLOCK}</srs>
    <renderer-v2 forceraster="0" symbollevels="0" type="singleSymbol" enableorderby="0">
      <symbols>
        <symbol alpha="1" clip_to_extent="1" type="marker" name="0" force_rhr="0">
          <layer pass="0" class="SimpleMarker" locked="0" enabled="1">
            <prop k="angle" v="0"/>
            <prop k="color" v="{fill_rgba}"/>
            <prop k="horizontal_anchor_point" v="1"/>
            <prop k="joinstyle" v="bevel"/>
            <prop k="name" v="circle"/>
            <prop k="offset" v="0,0"/>
            <prop k="offset_map_unit_scale" v="3x:0,0,0,0,0,0"/>
            <prop k="offset_unit" v="MM"/>
            <prop k="outline_color" v="{outline_rgba}"/>
            <prop k="outline_style" v="solid"/>
            <prop k="outline_width" v="{outline_width}"/>
            <prop k="outline_width_map_unit_scale" v="3x:0,0,0,0,0,0"/>
            <prop k="outline_width_unit" v="MM"/>
            <prop k="scale_method" v="diameter"/>
            <prop k="size" v="{size}"/>
            <prop k="size_map_unit_scale" v="3x:0,0,0,0,0,0"/>
            <prop k="size_unit" v="MM"/>
            <prop k="vertical_anchor_point" v="1"/>
          </layer>
        </symbol>
      </symbols>
      <rotation/>
      <sizescale/>
    </renderer-v2>
  </maplayer>"""


def _qgz_wms_maplayer():
    return f"""  <maplayer type="raster" hasScaleBasedVisibilityFlag="0">
    <id>{_QGZ_WMS_LAYER_ID}</id>
    <datasource>{_QGZ_WMS_SOURCE}</datasource>
    <layername>{_QGZ_WMS_LAYER_NAME}</layername>
    <provider encoding="">wms</provider>
    <srs>{_QGZ_SRS_BLOCK}</srs>
  </maplayer>"""


def _build_qgz(qgz_path: str, layers: List[dict]) -> None:
    """Write a QGIS .qgz project file.

    Parameters
    ----------
    qgz_path : str
        Output path for the .qgz file.
    layers : list of dict
        Each dict: layer_id, gpkg_relpath, layer_name, display_name,
        geom_type ('line'|'point'), colour (hex), line_style, fill_colour,
        outline_colour.  Layers are listed top-to-bottom in the legend.
    """
    tree_entries: List[str] = []
    maplayer_blocks: List[str] = []

    for lyr in layers:
        lid  = lyr["layer_id"]
        src  = f"{lyr['gpkg_relpath']}|layername={lyr['layer_name']}"
        name = lyr["display_name"]
        tree_entries.append(
            f'    <layer-tree-layer id="{lid}" name="{name}" '
            f'checked="Qt::Checked" expanded="1" source="{src}" providerKey="ogr"/>'
        )
        if lyr["geom_type"] == "line":
            rgba = _qgz_hex_to_rgba(lyr["colour"])
            maplayer_blocks.append(
                _qgz_line_maplayer(lid, lyr["gpkg_relpath"], lyr["layer_name"],
                                   name, rgba, lyr.get("line_style", "solid"))
            )
        else:
            fill_rgba    = _qgz_hex_to_rgba(lyr["fill_colour"])
            outline_rgba = _qgz_hex_to_rgba(lyr["outline_colour"])
            maplayer_blocks.append(
                _qgz_marker_maplayer(lid, lyr["gpkg_relpath"], lyr["layer_name"],
                                     name, fill_rgba, outline_rgba)
            )

    # Swisstopo WMS — always at the bottom of the layer tree
    tree_entries.append(
        f'    <layer-tree-layer id="{_QGZ_WMS_LAYER_ID}" name="{_QGZ_WMS_LAYER_NAME}" '
        f'checked="Qt::Checked" expanded="0" source="{_QGZ_WMS_SOURCE}" providerKey="wms"/>'
    )
    maplayer_blocks.append(_qgz_wms_maplayer())

    tree_xml   = "\n".join(tree_entries)
    layers_xml = "\n".join(maplayer_blocks)

    qgs = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis projectname="Network Build" version="{_QGIS_VERSION}">
  <homePath path=""/>
  <title>Network Build</title>
  <autotransaction active="0"/>
  <evaluateDefaultValues active="0"/>
  <trust active="0"/>
  <projectCrs>
    {_QGZ_SRS_BLOCK}
  </projectCrs>
  <layer-tree-group>
    <customproperties/>
{tree_xml}
    <custom-order enabled="0"/>
  </layer-tree-group>
  <projectlayers>
{layers_xml}
  </projectlayers>
  <mapcanvas name="theMapCanvas">
    <units>meters</units>
    <rotation>0</rotation>
    <destinationsrs>
      {_QGZ_SRS_BLOCK}
    </destinationsrs>
  </mapcanvas>
</qgis>
"""
    with _zipfile.ZipFile(qgz_path, "w", _zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("project.qgs", qgs)


def _collect_qgz_line_layers(
    by_type: Dict[int, "gpd.GeoDataFrame"],
    gpkg_relpath: str,
    label_map: Dict[int, str],
    suffix: str = "Segments",
) -> List[dict]:
    """Build line layer descriptor list for _build_qgz from a route_type→GDF dict."""
    layers = []
    counter = 0
    for rt, gdf in sorted(by_type.items()):
        if gdf is None or gdf.empty:
            continue
        layer_name = _QGZ_LAYER_NAMES.get(rt, f"type_{rt}")
        counter += 1
        layers.append({
            "layer_id":     f"{layer_name}_seg_{counter:04d}",
            "gpkg_relpath": gpkg_relpath,
            "layer_name":   layer_name,
            "display_name": f"{label_map.get(rt, layer_name)} {suffix}",
            "geom_type":    "line",
            "colour":       _QGZ_LINE_COLOURS.get(rt, "#888888"),
            "line_style":   _QGZ_LINE_STYLE.get(rt, "solid"),
        })
    return layers


def _collect_qgz_stop_layers(
    by_type: Dict[int, "gpd.GeoDataFrame"],
    gpkg_relpath: str,
    label_map: Dict[int, str],
    is_rail: bool = False,
) -> List[dict]:
    """Build point layer descriptor list for _build_qgz from a route_type→GDF dict."""
    layers = []
    counter = 0
    for rt, gdf in sorted(by_type.items()):
        if gdf is None or gdf.empty:
            continue
        layer_name = _QGZ_LAYER_NAMES.get(rt, f"type_{rt}")
        counter += 1
        if is_rail:
            fill, outline = _RAIL_STOP_FILL, _RAIL_STOP_OUTLINE
        else:
            fill, outline = _QGZ_LINE_COLOURS.get(rt, "#888888"), "#000000"
        layers.append({
            "layer_id":       f"{layer_name}_stops_{counter:04d}",
            "gpkg_relpath":   gpkg_relpath,
            "layer_name":     layer_name,
            "display_name":   f"{label_map.get(rt, layer_name)} Stops",
            "geom_type":      "point",
            "fill_colour":    fill,
            "outline_colour": outline,
        })
    return layers


# =============================================================================
# Phase 0 — CLI Setup
# =============================================================================

def _check_prerequisites() -> bool:
    """
    Verify that both infrastructure and service network outputs exist.
    Prints an error and returns False if any check fails.
    """
    main = Path(paths.MAIN)
    infra_root = main / paths.NETWORK_INFRASTRUCTURE_DIR
    feeder_root = main / paths.FEEDER_LINES_DIR

    ok = True
    # Check at least one non-Raw infra version exists
    infra_versions = [
        d for d in infra_root.iterdir()
        if d.is_dir() and d.name != "Raw"
        and (d / "nodes.gpkg").exists() and (d / "segments.gpkg").exists()
    ] if infra_root.exists() else []

    if not infra_versions:
        print(
            "\n  ERROR: No infrastructure version found under "
            f"{infra_root}\n"
            "  Run infrabuild_network_builder.py first."
        )
        ok = False

    # Check at least one svc_version with Unprojected/ base exists
    feeder_versions = [
        d for d in feeder_root.iterdir()
        if d.is_dir()
        and (d / paths.SERVICES_UNPROJECTED_SUBDIR / "pt_feeder_segments.gpkg").exists()
    ] if feeder_root.exists() else []

    if not feeder_versions:
        print(
            "\n  ERROR: No service version found under "
            f"{feeder_root}\n"
            "  Run services_network_builder.py first."
        )
        ok = False

    return ok


def _list_infra_versions() -> List[str]:
    """Return sorted list of infrastructure version names (excludes Raw)."""
    root = Path(paths.MAIN) / paths.NETWORK_INFRASTRUCTURE_DIR
    versions = [
        d.name for d in sorted(root.iterdir())
        if d.is_dir() and d.name != "Raw"
        and (d / "nodes.gpkg").exists() and (d / "segments.gpkg").exists()
    ]
    # Put Base first if present
    if "Base" in versions:
        versions = ["Base"] + [v for v in versions if v != "Base"]
    return versions


def _list_svc_versions() -> List[str]:
    """Return svc_version names that have an Unprojected/ base."""
    root = Path(paths.MAIN) / paths.FEEDER_LINES_DIR
    if not root.exists():
        return []
    return [
        d.name for d in sorted(root.iterdir())
        if d.is_dir()
        and (d / paths.SERVICES_UNPROJECTED_SUBDIR / "pt_feeder_segments.gpkg").exists()
    ]


def _list_source_options(svc_version: str) -> List[Tuple[str, Path]]:
    """Return (label, path) pairs for Unprojected/ and any named Versions/."""
    root = Path(paths.MAIN) / paths.FEEDER_LINES_DIR / svc_version
    options: List[Tuple[str, Path]] = [
        ("Unprojected (canonical base)", root / paths.SERVICES_UNPROJECTED_SUBDIR)
    ]
    versions_dir = root / paths.SERVICES_VERSIONS_SUBDIR
    if versions_dir.exists():
        for d in sorted(versions_dir.iterdir()):
            if d.is_dir() and (d / "pt_feeder_segments.gpkg").exists():
                options.append((f"Version: {d.name}", d))
    return options


def _list_projection_outputs() -> List[Tuple[str, str]]:
    """Return (svc_version, infra_version) tuples for already-projected outputs.

    Scans Feeder_Lines/<svc>/<infra>/ for pt_feeder_segments.gpkg,
    skipping the reserved subdirectory names.
    """
    _SKIP = {
        paths.SERVICES_UNPROJECTED_SUBDIR,
        paths.SERVICES_VERSIONS_SUBDIR,
        paths.SERVICES_PROJECTED_SUBDIR,
    }
    results = []
    root = Path(paths.MAIN) / paths.FEEDER_LINES_DIR
    if not root.exists():
        return results
    for svc_dir in sorted(root.iterdir()):
        if not svc_dir.is_dir():
            continue
        for infra_dir in sorted(svc_dir.iterdir()):
            if (
                infra_dir.is_dir()
                and infra_dir.name not in _SKIP
                and (infra_dir / "pt_feeder_segments.gpkg").exists()
            ):
                results.append((svc_dir.name, infra_dir.name))
    return results


def _pick_one(labels: List[str], prompt: str = "Select") -> Optional[int]:
    """Display numbered list; return 0-based index or None on empty Enter."""
    for i, lbl in enumerate(labels, 1):
        print(f"     {i}) {lbl}")
    while True:
        raw = input(f"   {prompt} (number): ").strip()
        if not raw:
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(labels):
            return int(raw) - 1
        print(f"   Invalid — enter 1–{len(labels)} or press Enter to cancel.")


def _run_phase0() -> Optional[Tuple[ProjectionConfig, str]]:
    """
    Interactive CLI setup.

    Returns:
        (ProjectionConfig, mode)  where mode is 'map' or 'correct'
        None if user cancels.
    """
    main = Path(paths.MAIN)
    print("\n" + "─" * 60)
    print("  Service Projection")
    print("─" * 60)

    # Q1 — operation mode
    print("\n  What do you want to do?")
    print("    1) Map services (full pipeline: Phase 1 → 2 → 3)")
    print("    2) Correct an existing projection (Phase 2 onwards)")
    while True:
        choice = input("  Select (1/2): ").strip()
        if choice in ("1", "2"):
            break
        print("  Enter 1 or 2.")

    if choice == "2":
        existing = _list_projection_outputs()
        if not existing:
            print("  No existing projections found. Run mapping first.")
            return None
        print("\n  Choose an existing projection to correct:")
        labels = [f"{svc}  /  {infra}" for svc, infra in existing]
        idx = _pick_one(labels, "Projection")
        if idx is None:
            return None
        svc_version, infra_version = existing[idx]
        source_feeder_path = main / paths.FEEDER_LINES_DIR / svc_version / infra_version
        source_rail_path   = main / paths.RAIL_LINES_DIR   / svc_version / infra_version
        mode = "correct"
    else:
        # Q2 — svc_version
        svc_versions = _list_svc_versions()
        if not svc_versions:
            print("  No service versions found. Run services_network_builder.py first.")
            return None
        print("\n  Choose service version:")
        idx = _pick_one(svc_versions, "Service version")
        if idx is None:
            return None
        svc_version = svc_versions[idx]

        # Q3 — source (Unprojected or named Version)
        source_options = _list_source_options(svc_version)
        print("\n  Choose source:")
        idx = _pick_one([label for label, _ in source_options], "Source")
        if idx is None:
            return None
        source_label, source_feeder_path = source_options[idx]

        # Derive parallel rail source path from feeder source path
        # e.g. Feeder_Lines/<svc>/Unprojected -> Rail_Lines/<svc>/Unprojected
        feeder_base = main / paths.FEEDER_LINES_DIR / svc_version
        try:
            rel = source_feeder_path.relative_to(feeder_base)
            source_rail_path = main / paths.RAIL_LINES_DIR / svc_version / rel
        except ValueError:
            source_rail_path = main / paths.RAIL_LINES_DIR / svc_version / paths.SERVICES_UNPROJECTED_SUBDIR

        # Q4 — infrastructure version
        infra_versions = _list_infra_versions()
        print("\n  Choose infrastructure version:")
        idx = _pick_one(infra_versions, "Infrastructure version")
        if idx is None:
            return None
        infra_version = infra_versions[idx]
        mode = "map"

    rail_output_dir   = main / paths.RAIL_LINES_DIR   / svc_version / infra_version
    feeder_output_dir = main / paths.FEEDER_LINES_DIR / svc_version / infra_version
    rail_output_dir.mkdir(parents=True, exist_ok=True)
    feeder_output_dir.mkdir(parents=True, exist_ok=True)

    config = ProjectionConfig(
        infra_version=infra_version,
        svc_version=svc_version,
        infra_dir=main / paths.NETWORK_INFRASTRUCTURE_DIR / infra_version,
        svc_dir=source_feeder_path,
        rail_input=source_rail_path / "rail_segments.gpkg",
        rail_output_dir=rail_output_dir,
        feeder_output_dir=feeder_output_dir,
        raw_infra_dir=main / paths.NETWORK_INFRASTRUCTURE_RAW,
    )

    print(f"\n  Service version: {svc_version}")
    print(f"  Source         : {source_feeder_path.name}")
    print(f"  Infrastructure : {infra_version}")
    print(f"  Rail output    : {rail_output_dir}")
    print(f"  Feeder output  : {feeder_output_dir}")
    return config, mode

# =============================================================================
# Phase 1 — Projection Orchestrator
# =============================================================================

def _run_phase1(
    config: ProjectionConfig,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Phase 1: load data, match stops, route paths, save enriched geopackages.

    Returns:
        (rail_enriched, tram_enriched, funicular_enriched)
    """
    main = Path(paths.MAIN)
    print("\n" + "─" * 60)
    print("  Phase 1 — Service Projection")
    print("─" * 60)

    # 1a. Load infrastructure
    print("\n  Loading infrastructure...")
    nodes = gpd.read_file(config.infra_dir / "nodes.gpkg").reset_index(drop=True)
    bav_segments = gpd.read_file(config.infra_dir / "segments.gpkg").reset_index(drop=True)
    print(f"  {len(nodes)} nodes, {len(bav_segments)} segments loaded.")

    # Load raw infrastructure for Tier 4 fallback
    raw_nodes_path = config.raw_infra_dir / "nodes.gpkg"
    raw_segs_path = config.raw_infra_dir / "segments.gpkg"
    raw_nodes = gpd.read_file(raw_nodes_path) if raw_nodes_path.exists() else None
    raw_segs = gpd.read_file(raw_segs_path) if raw_segs_path.exists() else None
    if raw_nodes is None:
        print("  NOTE: Raw nodes not found — Tier 4 fallback disabled.")

    # 1b. Build graph, lookups, node attributes
    print("  Building infrastructure graph (with missing node healing from raw_nodes)...")
    G = build_infra_graph(nodes, bav_segments, raw_nodes)
    seg_lookup = build_segment_lookup(nodes, bav_segments, raw_nodes)
    node_attrs = build_node_attrs(nodes)
    lookups = build_stop_lookups(nodes)
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    gauge_graphs = _build_gauge_graphs(G)

    # 1b-ii. Precompute hub topology (forced-routing flags for hub stations)
    print("  Building hub topology...")
    hub_topology = build_hub_topology(nodes, G)
    print(f"  Hub topology: {len(hub_topology)} hub(s) identified.")

    # 1c. Load buffer geometry for inside/outside decision
    buffer_geom = None
    buf_path = main / paths.CATCHMENT_AREA_BUFFER_GPKG
    if buf_path.exists():
        buf_gdf = gpd.read_file(buf_path)
        buffer_geom = buf_gdf.geometry.union_all()
        print(f"  Buffer loaded: {buf_path.name}")
    else:
        print("  WARNING: Catchment area buffer not found — all stops treated as inside.")

    # 1d. Build stop_coord from feeder segments + load ZVV geometry
    print("\n  Building stop coordinate index from feeder segments...")
    global stop_coord
    stop_coord = _build_stop_coord_from_segments(config.svc_dir)
    print(f"  {len(stop_coord):,} stop coordinates indexed.")

    print("\n  Loading ZVV geometry...")
    _zvv_seg_index.clear()
    _zvv_chain_index.clear()
    _zvv_sbahn_index.clear()
    _zvv_sbahn_jgraph.clear()
    _zvv_geometry_available = _load_zvv_geometry()
    if not _zvv_geometry_available:
        print("  ZVV geometry unavailable — straight-line fallback for non-track modes.")

    # 1e. Load service data (all feeder layers dynamically)
    print("\n  Loading service data...")

    import fiona as _fiona
    rail_layers = _fiona.listlayers(str(config.rail_input))
    rail_gdfs = []
    for layer in rail_layers:
        gdf = gpd.read_file(config.rail_input, layer=layer)
        gdf["_source_layer"] = layer
        rail_gdfs.append(gdf)
    rail_segments = pd.concat(rail_gdfs, ignore_index=True) if rail_gdfs else gpd.GeoDataFrame()

    col_mapping = {
        'route_id': 'Service',
        'direction_id': 'Direction',
        'line_short_name': 'TrainType',
        'from_stop_id': 'FromCode',
        'to_stop_id': 'ToCode',
        'from_stop_name': 'FromStation',
        'to_stop_name': 'ToStation',
        'from_stop_E': 'x_origin',
        'from_stop_N': 'y_origin',
        'to_stop_E': 'x_dest',
        'to_stop_N': 'y_dest',
        'travel_time_min': 'TravelTime',
        'InVehWait_min': 'InVehWait',
    }
    rail_edges = rail_segments.rename(columns=col_mapping)
    if 'TrainType' in rail_edges.columns:
        rail_edges['line_short_name'] = rail_edges['TrainType']
    for req_col in [
        'E_KOORD_O', 'N_KOORD_O', 'E_KOORD_D', 'N_KOORD_D',
        'Peak', 'OffPeak', 'Capacity', 'Speed',
        'FromGde', 'ToGde', 'NR_x', 'NR_y', 'Link NR',
        'FromNode', 'ToNode', 'Via', 'FromEnd', 'ToEnd',
        'TotalPeakCapacity', 'Frequency', 'PeakTrainLength',
    ]:
        if req_col not in rail_edges.columns:
            rail_edges[req_col] = pd.NA
    if 'x_origin' in rail_edges.columns:
        rail_edges['E_KOORD_O'] = rail_edges['x_origin']
        rail_edges['N_KOORD_O'] = rail_edges['y_origin']
        rail_edges['E_KOORD_D'] = rail_edges['x_dest']
        rail_edges['N_KOORD_D'] = rail_edges['y_dest']

    feeder_gpkg = config.svc_dir / "pt_feeder_segments.gpkg"
    all_feeder_layers = _fiona.listlayers(str(feeder_gpkg))
    track_feeder_gdfs: Dict[str, gpd.GeoDataFrame] = {}
    non_track_feeder_gdfs: Dict[str, gpd.GeoDataFrame] = {}
    for layer in all_feeder_layers:
        gdf = gpd.read_file(feeder_gpkg, layer=layer)
        if layer in TRACK_BASED_FEEDER_LAYERS:
            track_feeder_gdfs[layer] = gdf
        else:
            non_track_feeder_gdfs[layer] = gdf

    print(
        f"  Rail: {len(rail_edges)} links | "
        f"Track feeders: {sum(len(v) for v in track_feeder_gdfs.values())} segs "
        f"({', '.join(track_feeder_gdfs.keys())}) | "
        f"Non-track: {sum(len(v) for v in non_track_feeder_gdfs.values())} segs "
        f"({', '.join(non_track_feeder_gdfs.keys())})"
    )

    # 1f. Enrich rail (BAV projection)
    print("\n  Enriching rail links...")
    rail_enriched = enrich_rail_links(
        rail_edges, nodes, bav_segments, G, seg_lookup, node_attrs,
        lookups, buffer_geom, raw_nodes, raw_segs, config.infra_dir,
        hub_topology=hub_topology, gauge_graphs=gauge_graphs,
    )
    n_corrected = rail_enriched["needs_correction"].sum()
    print(f"  Rail done. {n_corrected} links need correction.")

    # 1g. Enrich track-based feeders (BAV projection)
    track_feeder_enriched: Dict[str, gpd.GeoDataFrame] = {}
    for layer_name, segs in track_feeder_gdfs.items():
        print(f"\n  Enriching {layer_name} segments...")
        enriched = enrich_feeder_segments(
            segs, nodes, bav_segments, G, seg_lookup, node_attrs,
            lookups, buffer_geom, raw_nodes, raw_segs, config.infra_dir,
            hub_topology=hub_topology, gauge_graphs=gauge_graphs,
        )
        print(f"  {layer_name} done. {enriched['needs_correction'].sum()} need correction.")
        track_feeder_enriched[layer_name] = enriched

    # 1h. ZVV post-pass: replace straight-line results + check divergence
    if _zvv_geometry_available:
        print("\n  Applying ZVV post-pass to rail...")
        rail_enriched = _apply_zvv_postpass(
            rail_enriched, "rail", is_track_based=True, buffer_geom=buffer_geom
        )

        for layer_name in list(track_feeder_enriched.keys()):
            print(f"  Applying ZVV post-pass to {layer_name}...")
            track_feeder_enriched[layer_name] = _apply_zvv_postpass(
                track_feeder_enriched[layer_name], layer_name,
                is_track_based=True, buffer_geom=buffer_geom
            )

    # 1i. ZVV geometry for non-track modes (bus, ship, etc.)
    non_track_feeder_processed: Dict[str, gpd.GeoDataFrame] = {}
    if non_track_feeder_gdfs:
        print("\n  Applying ZVV geometry to non-track feeder modes...")
        for layer_name, segs in non_track_feeder_gdfs.items():
            processed = _apply_zvv_postpass(segs, layer_name, is_track_based=False)
            non_track_feeder_processed[layer_name] = processed

    # 1j. Save enriched files
    print("\n  Saving enriched geopackages...")
    rail_out = config.rail_output_dir / "edges_in_corridor.gpkg"
    if "_source_layer" in rail_enriched.columns:
        saved_layers = []
        for layer_name, layer_gdf in rail_enriched.groupby("_source_layer"):
            layer_gdf = layer_gdf.drop(columns=["_source_layer"])
            layer_gdf.to_file(rail_out, driver="GPKG", layer=layer_name)
            saved_layers.append(layer_name)
        print(f"  Rail -> {rail_out}  (layers: {', '.join(saved_layers)})")
    else:
        rail_enriched.to_file(rail_out, driver="GPKG")
        print(f"  Rail -> {rail_out}")

    feeder_segs_out = config.feeder_output_dir / "pt_feeder_segments.gpkg"
    for layer_name, enriched_gdf in track_feeder_enriched.items():
        enriched_gdf.to_file(feeder_segs_out, driver="GPKG", layer=layer_name)
    for layer_name, processed_gdf in non_track_feeder_processed.items():
        processed_gdf.to_file(feeder_segs_out, driver="GPKG", layer=layer_name)
    print(f"  Feeder -> {feeder_segs_out}")

    # 1k. Copy stops for all processed feeder layers
    stops_out = config.feeder_output_dir / "pt_feeder_stops.gpkg"
    stops_gpkg = config.svc_dir / "pt_feeder_stops.gpkg"
    if stops_gpkg.exists():
        all_stop_layers = _fiona.listlayers(str(stops_gpkg))
        processed_layer_names = (
            set(track_feeder_enriched.keys()) | set(non_track_feeder_processed.keys())
        )
        for layer in all_stop_layers:
            if layer in processed_layer_names:
                gpd.read_file(stops_gpkg, layer=layer).to_file(
                    stops_out, driver="GPKG", layer=layer
                )
        print(f"  Stops -> {stops_out}")

    print(f"\n  Phase 1 complete.")
    # Return for backward compatibility with Phase 1.5 callers
    tram_enriched  = track_feeder_enriched.get("tram",      gpd.GeoDataFrame())
    func_enriched  = track_feeder_enriched.get("funicular", gpd.GeoDataFrame())
    return rail_enriched, tram_enriched, func_enriched

# =============================================================================
# Phase 1.5 — Boundary Station Routing
# =============================================================================

def _collect_outside_stops_gateway(
    rail_enriched: gpd.GeoDataFrame,
    buffer_geom,
) -> Dict[str, Dict]:
    """
    Collect unique stops from gateway-layer services (LD, IR, RE, S-Bahn) whose
    coordinates lie outside buffer_geom.  Only rows whose _source_layer is in
    _GATEWAY_LAYERS are examined.

    Returns {stop_id: {name, E, N, services: set()}}
    """
    outside: Dict[str, Dict] = {}
    if "_source_layer" not in rail_enriched.columns:
        return outside

    mask = rail_enriched["_source_layer"].isin(_GATEWAY_LAYERS)
    for _, row in rail_enriched[mask].iterrows():
        svc = str(row.get("Service", "?"))

        from_E = float(row.get("x_origin", 0) or 0)
        from_N = float(row.get("y_origin", 0) or 0)
        to_E   = float(row.get("x_dest",   0) or 0)
        to_N   = float(row.get("y_dest",   0) or 0)

        from_inside = buffer_geom is None or Point(from_E, from_N).within(buffer_geom)
        to_inside   = buffer_geom is None or Point(to_E,   to_N  ).within(buffer_geom)

        # Only collect a stop if it is outside AND its partner in this link is
        # inside — this restricts the mapping to boundary-crossing stops only.
        # Stops where both endpoints are outside are skipped; they keep their
        # Phase-1 straight-line geometry unchanged.
        for sid_col, sname_col, e_col, n_col, this_inside, partner_inside in [
            ("FromCode", "FromStation", "x_origin", "y_origin", from_inside, to_inside),
            ("ToCode",   "ToStation",   "x_dest",   "y_dest",   to_inside,   from_inside),
        ]:
            if this_inside or not partner_inside:
                continue  # stop is inside, or partner is also outside
            sid   = str(row.get(sid_col,   "")).strip()
            sname = str(row.get(sname_col, "")).strip()
            E     = float(row.get(e_col, 0) or 0)
            N     = float(row.get(n_col, 0) or 0)
            if not sid:
                continue
            if sid not in outside:
                outside[sid] = {"name": sname, "E": E, "N": N, "services": set()}
            outside[sid]["services"].add(svc)

    return outside


def _run_destination_mapping_cli(
    outside_stops: Dict[str, Dict],
    confirmed_boundary_ids: List[int],
    node_attrs: Dict[int, Dict],
    existing_mapping: Optional[Dict[str, int]] = None,
) -> Dict[str, int]:
    """
    Interactive CLI: assign each unique outside stop to a boundary station.

    Displays existing assignments when editing a saved mapping so the user can
    skip unchanged entries with Enter.  Returns {stop_id: boundary_node_id}.
    """
    if not outside_stops:
        print("  No outside destinations found — nothing to map.")
        return {}
    if not confirmed_boundary_ids:
        print("  No boundary stations confirmed — cannot map.")
        return {}

    mapping: Dict[str, int] = dict(existing_mapping or {})

    boundary_id_set = set(confirmed_boundary_ids)

    # Auto-assign outside stops that are themselves boundary stations.
    # stop_id is the Betriebspunkt_Nummer as a string (e.g. "8506137").
    auto_assigned_stops: set = set()
    for stop_id, info in outside_stops.items():
        try:
            numeric_id = int(stop_id.split(":")[-1]) if ":" in stop_id else int(stop_id)
        except (ValueError, AttributeError):
            continue
        if numeric_id in boundary_id_set:
            mapping[stop_id] = numeric_id
            auto_assigned_stops.add(stop_id)
    if auto_assigned_stops:
        print(f"\n  {len(auto_assigned_stops)} outside stop(s) are boundary stations — auto-assigned to themselves.")

    print("\n  Available boundary stations:")
    for i, nid in enumerate(confirmed_boundary_ids, 1):
        name = node_attrs.get(nid, {}).get("name", str(nid))
        print(f"    {i:3}) {name}  (node {nid})")

    print(f"\n  Outside stops on gateway-layer services ({len(outside_stops)} unique):")
    for stop_id, info in outside_stops.items():
        if stop_id in auto_assigned_stops:
            continue

        svc_sample = ", ".join(sorted(info["services"])[:5])
        n_svc      = len(info["services"])
        existing   = mapping.get(stop_id)
        cur_str    = ""
        if existing is not None:
            cur_name = node_attrs.get(existing, {}).get("name", str(existing))
            cur_str  = f"  [currently → {cur_name}]"

        raw = input(
            f"\n  '{info['name']}' ({n_svc} service(s): {svc_sample}){cur_str}\n"
            f"    → boundary station number (or Enter to skip): "
        ).strip()

        if not raw:
            continue
        if raw.isdigit() and 1 <= int(raw) <= len(confirmed_boundary_ids):
            chosen_nid  = confirmed_boundary_ids[int(raw) - 1]
            mapping[stop_id] = chosen_nid
            chosen_name = node_attrs.get(chosen_nid, {}).get("name", str(chosen_nid))
            print(f"    → Assigned to: {chosen_name}")
        else:
            print(f"    Invalid input — skipped.")

    return mapping


def _apply_boundary_rerouting(
    rail_enriched: gpd.GeoDataFrame,
    boundary_mapping: Dict[str, int],
    node_attrs: Dict[int, Dict],
    G: nx.Graph,
    seg_lookup: Dict,
    buffer_geom,
) -> gpd.GeoDataFrame:
    """
    Post-process rail_enriched: for gateway-layer links (LD, IR, RE, S-Bahn)
    where from_stop or to_stop has a boundary mapping, replace straight-line
    geometry with:

        straight_line(outside_coords → boundary_node)
        + routed_path(boundary_node → inside_node)

    Both entry and exit cases are handled symmetrically.  When both stops are
    outside (service passing through), both segments are stitched together with
    the graph-routed middle portion.

    New columns added: boundary_entry_node, boundary_exit_node (pd.NA when unused).
    Clears needs_correction on updated rows.
    """
    for col in ("boundary_entry_node", "boundary_exit_node"):
        if col not in rail_enriched.columns:
            rail_enriched[col] = pd.NA

    if "_source_layer" not in rail_enriched.columns:
        return rail_enriched

    mask    = rail_enriched["_source_layer"].isin(_GATEWAY_LAYERS)
    updated = 0

    for idx, row in rail_enriched[mask].iterrows():
        from_id = str(row.get("FromCode", "")).strip()
        to_id   = str(row.get("ToCode",   "")).strip()
        from_E  = float(row.get("x_origin", 0) or 0)
        from_N  = float(row.get("y_origin", 0) or 0)
        to_E    = float(row.get("x_dest",   0) or 0)
        to_N    = float(row.get("y_dest",   0) or 0)

        from_bnode = boundary_mapping.get(from_id)
        to_bnode   = boundary_mapping.get(to_id)
        if from_bnode is None and to_bnode is None:
            continue  # no mapping for either stop — nothing to do

        from_pt      = Point(from_E, from_N)
        to_pt        = Point(to_E,   to_N)
        from_outside = buffer_geom is None or not from_pt.within(buffer_geom)
        to_outside   = buffer_geom is None or not to_pt.within(buffer_geom)

        if from_outside and to_outside:
            continue  # both stops outside — not a boundary-crossing link, keep Phase-1 geometry

        # Determine effective routing endpoints on the BAV graph
        if from_outside and from_bnode is not None:
            route_from: Optional[int] = from_bnode
        else:
            nf = row.get("node_id_from")
            route_from = int(nf) if pd.notna(nf) else None

        if to_outside and to_bnode is not None:
            route_to: Optional[int] = to_bnode
        else:
            nt = row.get("node_id_to")
            route_to = int(nt) if pd.notna(nt) else None

        if route_from is None or route_to is None:
            continue

        # Degenerate: boundary node equals the other matched endpoint
        if route_from == route_to:
            bE = node_attrs.get(route_from, {}).get("E", from_E)
            bN = node_attrs.get(route_from, {}).get("N", from_N)
            rail_enriched.at[idx, "geometry"] = LineString(
                [(from_E, from_N), (bE, bN)]
            )
            if from_outside and from_bnode is not None:
                rail_enriched.at[idx, "boundary_entry_node"] = str(route_from)
            rail_enriched.at[idx, "needs_correction"] = False
            updated += 1
            continue

        # Route between the two effective endpoints on the BAV graph
        routed_geom, via_st, via_jn, path_len, _, _, _ = route_between_nodes(
            G, [route_from], [route_to], seg_lookup, node_attrs,
        )
        if routed_geom is None:
            continue  # no graph path found — keep existing geometry

        # Build combined geometry parts
        geom_parts: List = []

        if from_outside and from_bnode is not None:
            bE = node_attrs.get(from_bnode, {}).get("E", from_E)
            bN = node_attrs.get(from_bnode, {}).get("N", from_N)
            geom_parts.append(LineString([(from_E, from_N), (bE, bN)]))
            rail_enriched.at[idx, "boundary_entry_node"] = str(from_bnode)

        if routed_geom.geom_type == "LineString":
            geom_parts.append(routed_geom)
        elif hasattr(routed_geom, "geoms"):
            geom_parts.extend(
                g for g in routed_geom.geoms if g.geom_type == "LineString"
            )

        if to_outside and to_bnode is not None:
            bE = node_attrs.get(to_bnode, {}).get("E", to_E)
            bN = node_attrs.get(to_bnode, {}).get("N", to_N)
            geom_parts.append(LineString([(bE, bN), (to_E, to_N)]))
            rail_enriched.at[idx, "boundary_exit_node"] = str(to_bnode)

        combined = (
            linemerge(geom_parts) if len(geom_parts) > 1
            else (geom_parts[0] if geom_parts else routed_geom)
        )

        rail_enriched.at[idx, "geometry"]      = combined
        rail_enriched.at[idx, "Via_Station"]   = via_st
        rail_enriched.at[idx, "Via_Junction"]  = via_jn
        rail_enriched.at[idx, "path_length_m"] = path_len
        rail_enriched.at[idx, "needs_correction"] = False
        updated += 1

    print(f"  Boundary rerouting: {updated} link(s) updated.")
    return rail_enriched


def _run_phase1_5(
    config: ProjectionConfig,
    rail_enriched: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Phase 1.5: Boundary station detection and outside-destination rerouting.

    Workflow
    --------
    a) Auto-detect boundary station candidates (leaf stations ≤5 km from buffer
       edge); user confirms the list once — saved to boundary_stations.json.
    b) Collect unique outside stops from LD/IR service links in rail_enriched.
    c) User maps each outside stop to a boundary station — saved to
       boundary_mapping.json.  On re-run, existing mapping is loaded and the
       user can skip unchanged assignments.
    d) Re-route affected links: straight_line(outside→boundary) + graph route
       (boundary→inside stop), replacing the Phase-1 straight-line geometry.
    e) Updated rail_enriched is saved to disk, overwriting the Phase-1 output
       so Phase 2 and QGIS inspection see the improved geometry.

    Returns the updated rail_enriched GeoDataFrame.
    """
    main_dir  = Path(paths.MAIN)
    bs_path   = config.rail_output_dir / "boundary_stations.json"
    bm_path   = config.rail_output_dir / "boundary_mapping.json"

    print("\n" + "─" * 60)
    print("  Phase 1.5 — Boundary Station Mapping")
    print("─" * 60)

    # Re-load infrastructure (may have been updated by Tier 4 during Phase 1).
    # raw_nodes must be passed so missing junction nodes (e.g. Winterthur Nord)
    # are healed into the graph — without them boundary rerouting takes long
    # detours on severed corridors.
    raw_nodes_path = config.raw_infra_dir / "nodes.gpkg"
    raw_nodes_1_5  = gpd.read_file(raw_nodes_path) if raw_nodes_path.exists() else None
    nodes      = gpd.read_file(config.infra_dir / "nodes.gpkg")
    bav_segs   = gpd.read_file(config.infra_dir / "segments.gpkg")
    G          = build_infra_graph(nodes, bav_segs, raw_nodes_1_5)
    seg_lookup = build_segment_lookup(nodes, bav_segs, raw_nodes_1_5)
    node_attrs = build_node_attrs(nodes)

    # Load buffer geometry (same source as Phase 1)
    buffer_geom = None
    buf_path    = main_dir / paths.CATCHMENT_AREA_BUFFER_GPKG
    if buf_path.exists():
        buf_gdf     = gpd.read_file(buf_path)
        buffer_geom = buf_gdf.geometry.union_all()

    # ── a. Boundary station list ──────────────────────────────────────────────
    existing_bs = _load_boundary_stations(bs_path)
    if existing_bs is not None:
        print(f"\n  Loaded {len(existing_bs)} boundary station(s) from {bs_path.name}.")
        ans = input("  Re-detect and re-confirm? (y/n) [n]: ").strip().lower() or "n"
        confirmed_bs: Optional[List[int]] = existing_bs if ans != "y" else None
    else:
        confirmed_bs = None

    if confirmed_bs is None:
        candidates = detect_boundary_station_candidates(G, nodes, buffer_geom)
        print(f"\n  {len(candidates)} candidate boundary station(s) detected.")
        if not candidates:
            print("  No candidates found — Phase 1.5 skipped.")
            return rail_enriched
        confirmed_bs = _run_boundary_station_confirmation_cli(candidates, node_attrs)
        _save_boundary_stations(confirmed_bs, bs_path)
        print(f"  Boundary stations saved to {bs_path.name}.")

    if not confirmed_bs:
        print("  No boundary stations confirmed — Phase 1.5 skipped.")
        return rail_enriched

    # ── b. Collect outside stops from gateway-layer services ─────────────────
    outside_stops = _collect_outside_stops_gateway(rail_enriched, buffer_geom)
    print(f"\n  {len(outside_stops)} unique outside stop(s) found on gateway-layer services.")
    if not outside_stops:
        print("  Nothing to map — Phase 1.5 skipped.")
        return rail_enriched

    # ── c. Destination → boundary station mapping ─────────────────────────────
    existing_bm = _load_boundary_mapping(bm_path)
    if existing_bm is not None:
        print(f"  Loaded {len(existing_bm)} mapping(s) from {bm_path.name}.")
        ans = input("  Edit existing mapping? (y/n) [n]: ").strip().lower() or "n"
        if ans == "y":
            boundary_mapping = _run_destination_mapping_cli(
                outside_stops, confirmed_bs, node_attrs,
                existing_mapping=existing_bm,
            )
        else:
            boundary_mapping = existing_bm
    else:
        boundary_mapping = _run_destination_mapping_cli(
            outside_stops, confirmed_bs, node_attrs,
        )

    if not boundary_mapping:
        print("  No mappings provided — Phase 1.5 skipped.")
        return rail_enriched

    _save_boundary_mapping(boundary_mapping, bm_path)
    print(f"  Mapping saved to {bm_path.name}  ({len(boundary_mapping)} entry/ies).")

    # ── d. Re-route affected gateway-layer links ─────────────────────────────
    print("\n  Applying boundary rerouting to gateway-layer links...")
    rail_enriched = _apply_boundary_rerouting(
        rail_enriched, boundary_mapping, node_attrs, G, seg_lookup, buffer_geom,
    )

    # ── e. Overwrite Phase-1 on-disk output with updated geometry ─────────────
    rail_out = config.rail_output_dir / "edges_in_corridor.gpkg"
    if "_source_layer" in rail_enriched.columns:
        for layer_name, layer_gdf in rail_enriched.groupby("_source_layer"):
            layer_gdf = layer_gdf.drop(columns=["_source_layer"])
            layer_gdf.to_file(rail_out, driver="GPKG", layer=layer_name)
    else:
        rail_enriched.to_file(rail_out, driver="GPKG")
    print(f"  Updated rail output saved → {rail_out}")

    print("\n  Phase 1.5 complete.")
    return rail_enriched


# =============================================================================
# Phase 2a — QGIS Projects and Clipped Segment Geopackages
# =============================================================================

def _save_phase2_outputs(
    config: ProjectionConfig,
    rail_enriched: gpd.GeoDataFrame,
    tram_enriched: gpd.GeoDataFrame,
    func_enriched: gpd.GeoDataFrame,
) -> None:
    """
    Build two QGIS project files (.qgz) and two clipped segment geopackages.

    Outputs (written next to the existing Phase-1 enriched files):
      rail_output_dir/   rail_segments.qgz
      feeder_output_dir/ pt_feeder_segments.qgz
      feeder_output_dir/ projected_segments_study.gpkg
      feeder_output_dir/ projected_segments_catchment.gpkg

    The rail QGZ contains enriched rail segments + rail stops.
    The PT-Feeder QGZ contains enriched tram/funicular segments and, for bus/ship
    (modes not touched by projection), inherits geometry from the parent svc_version
    geopackage one directory level up.
    """
    import fiona as _fiona
    main = Path(paths.MAIN)
    print("\n  Building QGIS projects and clipped segment geopackages...")

    # ── 1. Rail QGZ ────────────────────────────────────────────────────────────
    # Group enriched rail by source layer → {route_type_int: GeoDataFrame}
    rail_by_type: Dict[int, gpd.GeoDataFrame] = {}
    if "_source_layer" in rail_enriched.columns:
        for layer_name, gdf in rail_enriched.groupby("_source_layer"):
            rt = _LAYER_NAME_TO_RT.get(str(layer_name))
            if rt is not None:
                rail_by_type[rt] = gdf.drop(columns=["_source_layer"])
    if not rail_by_type:
        # Fallback when _source_layer was lost (e.g. correction-mode reload)
        rail_by_type[100] = rail_enriched

    # Rail stops — from svc_version output (one level above infra-versioned dir)
    rail_stops_path = config.rail_output_dir.parent / "rail_stops.gpkg"
    rail_stops_by_type: Dict[int, gpd.GeoDataFrame] = {}
    if rail_stops_path.exists():
        for lname in _fiona.listlayers(str(rail_stops_path)):
            rt = _LAYER_NAME_TO_RT.get(lname)
            if rt is not None:
                rail_stops_by_type[rt] = gpd.read_file(str(rail_stops_path), layer=lname)

    rail_layers_list = []
    if rail_stops_by_type:
        rail_layers_list += _collect_qgz_stop_layers(
            rail_stops_by_type, "../rail_stops.gpkg", _RAIL_LINE_TYPES, is_rail=True
        )
    rail_layers_list += _collect_qgz_line_layers(
        rail_by_type, "./edges_in_corridor.gpkg", _RAIL_LINE_TYPES, suffix="Segments"
    )

    rail_qgz = config.rail_output_dir / "rail_segments.qgz"
    _build_qgz(str(rail_qgz), rail_layers_list)
    print(f"  Rail QGZ → {rail_qgz}  ({len(rail_layers_list)} layer(s))")

    # ── 2. PT-Feeder QGZ ───────────────────────────────────────────────────────
    orig_seg_path   = config.feeder_output_dir / "pt_feeder_segments.gpkg"
    orig_stops_path = config.feeder_output_dir / "pt_feeder_stops.gpkg"

    # Enriched modes written by Phase 1
    enriched_feeder_by_type: Dict[int, gpd.GeoDataFrame] = {
        900:  tram_enriched,
        1400: func_enriched,
    }
    # Pass-through modes (bus, ship, …) — original geometry from svc_version output
    passthrough_by_type: Dict[int, gpd.GeoDataFrame] = {}
    if orig_seg_path.exists():
        for lname in _fiona.listlayers(str(orig_seg_path)):
            rt = _LAYER_NAME_TO_RT.get(lname)
            if rt is not None and rt not in enriched_feeder_by_type:
                passthrough_by_type[rt] = gpd.read_file(str(orig_seg_path), layer=lname)

    # Stops — all from original (stop locations are unchanged by projection)
    feeder_stops_by_type: Dict[int, gpd.GeoDataFrame] = {}
    if orig_stops_path.exists():
        for lname in _fiona.listlayers(str(orig_stops_path)):
            rt = _LAYER_NAME_TO_RT.get(lname)
            if rt is not None:
                feeder_stops_by_type[rt] = gpd.read_file(str(orig_stops_path), layer=lname)

    feeder_layers_list: List[dict] = []
    if feeder_stops_by_type:
        feeder_layers_list += _collect_qgz_stop_layers(
            feeder_stops_by_type, "./pt_feeder_stops.gpkg", _PT_FEEDER_LINE_TYPES, is_rail=False
        )
    feeder_layers_list += _collect_qgz_line_layers(
        enriched_feeder_by_type, "./pt_feeder_segments.gpkg",
        _PT_FEEDER_LINE_TYPES, suffix="Segments (projected)"
    )
    if passthrough_by_type:
        feeder_layers_list += _collect_qgz_line_layers(
            passthrough_by_type, "./pt_feeder_segments.gpkg",
            _PT_FEEDER_LINE_TYPES, suffix="Segments"
        )

    feeder_qgz = config.feeder_output_dir / "pt_feeder_segments.qgz"
    _build_qgz(str(feeder_qgz), feeder_layers_list)
    print(f"  PT-Feeder QGZ → {feeder_qgz}  ({len(feeder_layers_list)} layer(s))")

    # ── 3. Clipped segment geopackages ─────────────────────────────────────────
    # Merge all enriched modes, drop internal tag
    all_frames = []
    for mode, gdf in [("rail", rail_enriched), ("tram", tram_enriched), ("funicular", func_enriched)]:
        sub = gdf.copy()
        if "_source_layer" in sub.columns:
            sub = sub.drop(columns=["_source_layer"])
        sub["mode"] = mode
        all_frames.append(sub)
    combined = gpd.GeoDataFrame(pd.concat(all_frames, ignore_index=True), crs=SWISS_CRS)

    def _clip_and_save(boundary_path: Path, out_path: Path, area_label: str) -> None:
        if not boundary_path.exists():
            print(f"  Skipping {area_label} segments — boundary not found: {boundary_path.name}")
            return
        boundary_poly = gpd.read_file(boundary_path).geometry.union_all()
        total = 0
        for mode in ["rail", "tram", "funicular"]:
            sub = combined[combined["mode"] == mode].copy()
            if sub.empty:
                continue
            try:
                clipped = gpd.clip(sub, boundary_poly)
            except Exception:
                clipped = sub[sub.geometry.intersects(boundary_poly)]
            if not clipped.empty:
                clipped.to_file(str(out_path), driver="GPKG", layer=mode)
                total += len(clipped)
        if total:
            print(f"  {area_label} segments → {out_path}  ({total} features)")
        else:
            print(f"  {area_label} segments — no features within boundary.")

    study_path = main / paths.CATCHMENT_AREA_DIR / "study_area_boundary.gpkg"
    if not study_path.exists():
        study_path = main / paths.STUDY_AREA_BOUNDARY_GPKG
    _clip_and_save(
        study_path,
        config.feeder_output_dir / "projected_segments_study.gpkg",
        "Study area",
    )
    _clip_and_save(
        main / paths.CATCHMENT_AREA_BOUNDARY_GPKG,
        config.feeder_output_dir / "projected_segments_catchment.gpkg",
        "Catchment area",
    )


# =============================================================================
# Phase 2b — Corrections TUI
# =============================================================================

def _show_service_stops(
    enriched: gpd.GeoDataFrame,
    service_code: str,
    route_col: str,
    from_name_col: str,
    to_name_col: str,
    from_method_col: str,
    path_len_col: str,
) -> List[int]:
    """
    Print stop sequence for a service. Returns list of row indices in sequence order.
    """
    subset = enriched[enriched[route_col] == service_code]
    if subset.empty:
        print(f"  No links found for service '{service_code}'.")
        return []

    print(f"\n  Stop sequence for '{service_code}':")
    indices = list(subset.index)
    stop_num = 1

    for i, idx in enumerate(indices):
        row = enriched.loc[idx]
        from_name = row.get(from_name_col, "?")
        method = row.get(from_method_col, "?")
        node_id = row.get("node_id_from", None)
        km = row.get(path_len_col, 0) / 1000.0
        flag = " ← UNMATCHED" if row.get("needs_correction", False) else ""
        node_str = f"node {node_id}" if node_id else "UNMATCHED"
        print(
            f"    {stop_num:2}. {from_name:<30}  [{method:8}, {node_str}]  "
            f"→ {km:.1f} km{flag}"
        )
        stop_num += 1

        # Print the to-stop of the last link
        if i == len(indices) - 1:
            to_name = row.get(to_name_col, "?")
            to_method = row.get("match_method_to", "?")
            to_node = row.get("node_id_to", None)
            to_node_str = f"node {to_node}" if to_node else "UNMATCHED"
            print(f"    {stop_num:2}. {to_name:<30}  [{to_method:8}, {to_node_str}]")

    return indices


def _reroute_link(
    enriched: gpd.GeoDataFrame,
    link_idx: int,
    from_node_id: int,
    to_node_id: int,
    G: nx.Graph,
    seg_lookup: Dict,
    node_attrs: Dict,
) -> gpd.GeoDataFrame:
    """
    Interactively build a new path from from_node_id towards to_node_id by
    letting the user pick segments step by step.

    Updates the row at link_idx in enriched and returns the modified GeoDataFrame.
    """
    current_node = from_node_id
    path_nodes: List[int] = [current_node]

    print(f"\n  Building new path from node {from_node_id} → target node {to_node_id}")
    print("  At each step, pick the next segment. Type DONE to confirm when ready.\n")

    while True:
        neighbours = list(G.neighbors(current_node))
        if not neighbours:
            print("  Dead end — no reachable neighbours. Path confirmed as-is.")
            break

        labels = []
        for nb in neighbours:
            nb_name = node_attrs.get(nb, {}).get("name", str(nb))
            seg = seg_lookup.get((current_node, nb))
            seg_id = seg["segment_id"] if seg is not None else "?"
            km = G[current_node][nb].get("weight", 0) / 1000.0
            labels.append(f"{nb_name}  [{seg_id}, {km:.2f} km]")

        current_name = node_attrs.get(current_node, {}).get("name", str(current_node))
        print(f"  From: {current_name}")
        for i, lbl in enumerate(labels, 1):
            print(f"    {i}) {lbl}")
        print("    d) DONE — confirm path up to here")

        raw = input("  Pick next segment (number or d): ").strip().lower()
        if raw == "d":
            if current_node != to_node_id:
                print(
                    f"  WARNING: Path ends at node {current_node} "
                    f"(target was {to_node_id})."
                )
                confirm = input("  Confirm anyway? (y/n) [n]: ").strip().lower() or "n"
                if confirm != "y":
                    continue
            break
        if raw.isdigit() and 1 <= int(raw) <= len(neighbours):
            current_node = neighbours[int(raw) - 1]
            path_nodes.append(current_node)
            if current_node == to_node_id:
                print(f"  Reached target node {to_node_id}.")
                break
        else:
            print(f"  Invalid — enter 1–{len(neighbours)} or d.")

    if len(path_nodes) < 2:
        print("  No path built — no changes made.")
        return enriched

    # Reconstruct geometry and via columns from path_nodes
    geoms = []
    path_length = 0.0
    via_st: List[str] = []
    via_jn: List[str] = []

    for i in range(len(path_nodes) - 1):
        seg = seg_lookup.get((path_nodes[i], path_nodes[i + 1]))
        if seg is not None and seg.geometry is not None:
            g = seg.geometry
            if g.geom_type == "LineString":
                geoms.append(g)
            elif hasattr(g, 'geoms'):
                geoms.extend([sub_g for sub_g in g.geoms if sub_g.geom_type == "LineString"])
        path_length += float(G[path_nodes[i]][path_nodes[i + 1]].get("weight", 0))

    for nid in path_nodes[1:-1]:
        attrs = node_attrs.get(nid, {})
        name = attrs.get("name", str(nid))
        if attrs.get("node_class") == "station":
            via_st.append(name)
        else:
            via_jn.append(name)

    new_geom = linemerge(geoms) if geoms else enriched.at[link_idx, "geometry"]

    enriched.at[link_idx, "geometry"] = new_geom
    enriched.at[link_idx, "Via_Station"] = ";".join(via_st)
    enriched.at[link_idx, "Via_Junction"] = ";".join(via_jn)
    enriched.at[link_idx, "path_length_m"] = path_length
    enriched.at[link_idx, "needs_correction"] = False

    print(
        f"  Rerouted: {len(path_nodes)-1} segments, {path_length/1000:.2f} km. "
        f"Row {link_idx} updated."
    )
    return enriched


def _run_phase2(
    config: ProjectionConfig,
    rail_enriched: gpd.GeoDataFrame,
    tram_enriched: gpd.GeoDataFrame,
    func_enriched: gpd.GeoDataFrame,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Phase 2: build combined-line outputs for QGIS inspection, then run rerouting TUI.
    Returns updated (rail_enriched, tram_enriched, func_enriched).
    """
    print("\n" + "─" * 60)
    print("  Phase 2 — Corrections")
    print("─" * 60)

    _save_phase2_outputs(config, rail_enriched, tram_enriched, func_enriched)

    print("\n  Open the .qgz files in QGIS to inspect routing and identify any errors before making corrections.")
    ans = input("\n  Do you want to reroute any service? (y/n) [n]: ").strip().lower() or "n"
    if ans != "y":
        print("  No corrections made.")
        return rail_enriched, tram_enriched, func_enriched

    # Load graph for corrections
    nodes = gpd.read_file(config.infra_dir / "nodes.gpkg")
    bav_segs = gpd.read_file(config.infra_dir / "segments.gpkg")
    G = build_infra_graph(nodes, bav_segs)
    seg_lookup = build_segment_lookup(nodes, bav_segs)
    node_attrs = build_node_attrs(nodes)

    mode_map = {
        "rail": ("rail", rail_enriched, "Service", "FromStation", "ToStation",
                 "match_method_from", "path_length_m"),
        "tram": ("tram", tram_enriched, "route_id", "from_stop_name", "to_stop_name",
                 "match_method_from", "path_length_m"),
        "funicular": ("funicular", func_enriched, "route_id", "from_stop_name",
                      "to_stop_name", "match_method_from", "path_length_m"),
    }

    while True:
        print("\n  Which mode? (rail / tram / funicular / done)")
        mode_input = input("  Mode: ").strip().lower()
        if mode_input in ("done", "d", ""):
            break
        if mode_input not in mode_map:
            print("  Enter rail, tram, funicular, or done.")
            continue

        mode_label, enriched_df, route_col, fn_col, tn_col, meth_col, len_col = (
            mode_map[mode_input]
        )

        svc = input(f"  Service code (e.g. S14, 10, Polybahn): ").strip()
        link_indices = _show_service_stops(
            enriched_df, svc, route_col, fn_col, tn_col, meth_col, len_col
        )
        if not link_indices:
            continue

        raw_from = input(
            "\n  FROM stop number to begin rerouting (or Enter to cancel): "
        ).strip()
        if not raw_from:
            continue
        if not raw_from.isdigit() or not (1 <= int(raw_from) <= len(link_indices)):
            print("  Invalid stop number.")
            continue

        link_pos = int(raw_from) - 1
        link_idx = link_indices[link_pos]
        row = enriched_df.loc[link_idx]
        from_node = row.get("node_id_from")
        to_node = row.get("node_id_to")

        if pd.isna(from_node) or pd.isna(to_node):
            print(
                "  Cannot reroute — from or to node is unmatched. "
                "Fix the matching first."
            )
            continue

        enriched_df = _reroute_link(
            enriched_df, link_idx, int(from_node), int(to_node),
            G, seg_lookup, node_attrs
        )

        # Update the mode map reference
        if mode_input == "rail":
            rail_enriched = enriched_df
            rail_enriched.to_file(
                config.rail_output_dir / "edges_in_corridor.gpkg", driver="GPKG"
            )
        elif mode_input == "tram":
            tram_enriched = enriched_df
            tram_enriched.to_file(
                config.feeder_output_dir / "pt_feeder_segments.gpkg",
                driver="GPKG", layer="tram"
            )
        else:
            func_enriched = enriched_df
            func_enriched.to_file(
                config.feeder_output_dir / "pt_feeder_segments.gpkg",
                driver="GPKG", layer="funicular"
            )
        print("  Changes saved.")
        mode_map[mode_input] = (
            mode_label, enriched_df, route_col, fn_col, tn_col, meth_col, len_col
        )

        ans = input("\n  Reroute another service? (y/n) [n]: ").strip().lower() or "n"
        if ans != "y":
            break

    return rail_enriched, tram_enriched, func_enriched

# =============================================================================
# Phase 3 — Plotting
# =============================================================================

_SCALE_BAR_NICE_KM = [1, 2, 5, 10, 20, 50, 100, 200, 500]

def _extent_from_gdf(gdf, margin_m: int = 2000):
    if gdf is None or gdf.empty:
        return None
    b = gdf.total_bounds
    return (b[0] - margin_m, b[2] + margin_m, b[1] - margin_m, b[3] + margin_m)

def _add_north_arrow(ax, location='upper left', scale=0.5):
    north_arrow(ax, location=location, scale=scale, rotation={"degrees": 0})

def _add_scale_bar(ax, location=(0.72, 0.04)):
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    map_w, map_h = xlim[1] - xlim[0], ylim[1] - ylim[0]
    target_km = (map_w / 4.0) / 1000.0
    total_km  = min(_SCALE_BAR_NICE_KM, key=lambda v: abs(v - target_km))
    n_cells   = 4 if total_km >= 4 else 2
    cell_m    = (total_km * 1000.0) / n_cells
    x0, y0 = xlim[0] + map_w * location[0], ylim[0] + map_h * location[1]
    bar_h = map_h * 0.008
    for i in range(n_cells):
        color = 'black' if i % 2 == 0 else 'white'
        ax.add_patch(Rectangle((x0 + i * cell_m, y0), cell_m, bar_h, facecolor=color, edgecolor='black', linewidth=0.6, zorder=7))
    for i in range(n_cells + 1):
        val_km = (i * cell_m) / 1000.0
        label = f'{val_km:.0f} km' if val_km == int(val_km) else f'{val_km:.1f} km'
        ax.text(x0 + i * cell_m, y0 + bar_h * 1.6, label, ha='center', va='bottom', fontsize=7, zorder=7)

def _plot_service_overview(
    config: ProjectionConfig,
    rail_enriched: gpd.GeoDataFrame,
    tram_enriched: gpd.GeoDataFrame,
    func_enriched: gpd.GeoDataFrame,
    boundary_gpkg: Path,
    boundary_name: str,
) -> None:
    """
    Produce one overview plot for the given boundary (study area or catchment area).
    """
    if not boundary_gpkg.exists():
        print(f"  Skipping {boundary_name} plot — boundary file not found.")
        return

    main = Path(paths.MAIN)
    boundary_gdf = gpd.read_file(boundary_gpkg)
    boundary_poly = boundary_gdf.geometry.union_all()
    is_sa = boundary_name == "study_area"
    extent = _extent_from_gdf(boundary_gdf, margin_m=2000)

    # Load infrastructure
    bav_segs = gpd.read_file(config.infra_dir / "segments.gpkg")
    node_gdf = gpd.read_file(config.infra_dir / "nodes.gpkg")

    # Collect train stations to filter unused infrastructure
    train_stations = node_gdf
    if 'node_class' in node_gdf.columns and 'transport_mode' in node_gdf.columns:
        train_stations = node_gdf[(node_gdf['node_class'] == 'station') & (node_gdf['transport_mode'].astype(str).str.contains('train', case=False, na=False))]
    elif 'node_class' in node_gdf.columns:
        train_stations = node_gdf[node_gdf['node_class'] == 'station']

    train_names = set(train_stations['NAME'].tolist())
    if train_names:
        bav_segs_filtered = bav_segs[bav_segs['from_name'].isin(train_names) | bav_segs['to_name'].isin(train_names)].dropna(subset=['from_name', 'to_name'])
    else:
        bav_segs_filtered = bav_segs

    # Load lakes
    lakes_path = main / paths.LAKES_SHP
    lakes_gdf = gpd.read_file(lakes_path) if lakes_path.exists() else None

    # Collect all used segment geometries per mode
    mode_segments = {
        "rail": rail_enriched,
        "tram": tram_enriched,
        "funicular": func_enriched,
    }
    
    # Pre-calculate final destinations per (service, direction) to be direction-aware
    final_destinations = {}
    for mode, enriched in mode_segments.items():
        if mode == "rail":
            for (service_id, direction), rows in enriched.groupby(['Service', 'Direction']):
                if pd.isna(service_id): continue
                final_destinations[(str(service_id), str(direction))] = str(rows.iloc[-1].get('ToStation', ''))
        else:
            for (route_id, direction_id), rows in enriched.groupby(['route_id', 'direction_id']):
                if pd.isna(route_id): continue
                final_destinations[(str(route_id), str(direction_id))] = str(rows.iloc[-1].get('to_stop_name', ''))

    # Calculate terminus stations per (service, direction) using line_short_name as label
    station_termini_texts = {}
    for mode, enriched in mode_segments.items():
        if mode == "rail":
            group_cols = ['Service', 'Direction']
            label_col, fallback_col, id_col = 'line_short_name', 'TrainType', 'Service'
        else:
            group_cols = ['route_id', 'direction_id']
            label_col, fallback_col, id_col = 'line_short_name', 'route_id', 'route_id'

        for key_vals, rows in enriched.groupby(group_cols):
            if any(pd.isna(k) for k in key_vals):
                continue
            label = str(rows.iloc[0].get(label_col,
                        rows.iloc[0].get(fallback_col,
                        str(rows.iloc[0].get(id_col, '')))))
            f_nodes = set(rows['node_id_from'].dropna().unique())
            t_nodes = set(rows['node_id_to'].dropna().unique())
            for t_node in t_nodes - f_nodes:
                try:
                    station_termini_texts.setdefault(int(t_node), set()).add(label)
                except (ValueError, TypeError):
                    pass

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_aspect("equal")
    ax.set_xlabel('E [m]', fontsize=10)
    ax.set_ylabel('N [m]', fontsize=10)
    ax.grid(True, alpha=0.3)

    boundary_gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.5, linestyle='--', alpha=0.6)

    if lakes_gdf is not None:
        try:
            if is_sa and extent is not None:
                from shapely.geometry import box as _sbox
                clip_geom = gpd.GeoDataFrame(
                    geometry=[_sbox(extent[0], extent[2], extent[1], extent[3])],
                    crs=SWISS_CRS)
                lakes_clipped = gpd.clip(lakes_gdf, clip_geom)
            else:
                lakes_clipped = gpd.clip(lakes_gdf, boundary_gdf)
            if not lakes_clipped.empty:
                lakes_clipped.plot(ax=ax, color="#c8e8f5", linewidth=0.3, edgecolor="#99c4d8")
        except Exception:
            pass

    # Plot infrastructure background — SA: ghost all within extent + solid inside boundary;
    # CA: solid inside boundary only (mirrors infra builder show_outside/is_catchment logic)
    if is_sa and extent is not None:
        from shapely.geometry import box as _sbox
        bbox_gdf = gpd.GeoDataFrame(
            geometry=[_sbox(extent[0], extent[2], extent[1], extent[3])],
            crs=bav_segs.crs if bav_segs.crs else SWISS_CRS)
        if not bav_segs.empty:
            try:
                segs_extent = gpd.clip(bav_segs, bbox_gdf)
                if not segs_extent.empty:
                    segs_extent.plot(ax=ax, color="#d0d0d0", linewidth=0.4, alpha=0.40, zorder=1)
            except Exception:
                pass
        if not bav_segs_filtered.empty:
            try:
                segs_inside = gpd.clip(bav_segs_filtered, boundary_gdf)
                if not segs_inside.empty:
                    segs_inside.plot(ax=ax, color="#d0d0d0", linewidth=0.5, alpha=1.0, zorder=2)
            except Exception:
                pass
    else:
        if not bav_segs_filtered.empty:
            try:
                segs_inside = gpd.clip(bav_segs_filtered, boundary_gdf)
                if not segs_inside.empty:
                    segs_inside.plot(ax=ax, color="#d0d0d0", linewidth=0.4, zorder=1)
            except Exception:
                bav_segs_filtered.plot(ax=ax, color="#d0d0d0", linewidth=0.4, zorder=1)

    crossing_points = []
    
    for mode, enriched in mode_segments.items():
        colour = MODE_COLOURS[mode]
        for _, row in enriched.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            is_fallback = bool(row.get("needs_correction", False))
            line_colour = MODE_COLOURS["fallback"] if is_fallback else colour
            linestyle = "--" if is_fallback else "-"
            base_width = 1.2 if is_fallback else 1.8
            linewidth = base_width * 0.5 if mode in ("tram", "funicular") else base_width

            try:
                clipped = geom.intersection(boundary_poly)
            except Exception:
                clipped = geom

            if clipped.is_empty:
                continue

            seg_gdf = gpd.GeoDataFrame({"geometry": [clipped]}, crs=SWISS_CRS)
            seg_gdf.plot(ax=ax, color=line_colour, linewidth=linewidth, linestyle=linestyle, zorder=2)

            if not geom.within(boundary_poly):
                diff = geom.difference(boundary_poly)
                if not diff.is_empty:
                    try:
                        cross_pt = boundary_poly.boundary.intersection(geom)
                    except Exception:
                        cross_pt = None
                    if cross_pt is not None and not cross_pt.is_empty:
                        cp = None
                        if cross_pt.geom_type == "Point":
                            cp = cross_pt
                        elif hasattr(cross_pt, "geoms") and list(cross_pt.geoms):
                            cp_geom = list(cross_pt.geoms)[0]
                            cp = cp_geom if cp_geom.geom_type == "Point" else Point(cp_geom.coords[0])
                        elif cross_pt.geom_type in ("LineString", "MultiLineString"):
                            cp = Point(cross_pt.coords[0])

                        if cp is not None and getattr(cp, "geom_type", "") == "Point":
                            # Only record exiting crossings (first coord inside, last coord outside).
                            # Use covers() so stations exactly on the boundary edge are treated as
                            # inside rather than outside (contains() is strictly interior-only).
                            try:
                                if geom.geom_type == 'LineString':
                                    coords = list(geom.coords)
                                else:
                                    coords = []
                                    for _part in geom.geoms:
                                        coords.extend(list(_part.coords))
                                first_inside = boundary_poly.covers(Point(coords[0]))
                                last_inside  = boundary_poly.covers(Point(coords[-1]))
                                if first_inside and not last_inside:
                                    starts_inside = True   # exiting
                                elif not first_inside and last_inside:
                                    starts_inside = False  # entering — skip
                                else:
                                    starts_inside = True   # passthrough or ambiguous — include
                            except Exception:
                                starts_inside = True
                            if starts_inside:
                                svc_text = str(row.get("line_short_name",
                                               row.get("TrainType",
                                               row.get("Service", ""))))
                                _last = coords[-1]
                                _dx = _last[0] - cp.x
                                _dy = _last[1] - cp.y
                                _d  = (_dx**2 + _dy**2)**0.5
                                _udx, _udy = (_dx/_d, _dy/_d) if _d > 0 else (1.0, 0.0)
                                crossing_points.append({"pt": cp, "text": svc_text, "colour": line_colour, "udx": _udx, "udy": _udy})

    # For SA: show all stations within the plot extent (ghost outside boundary);
    # for CA: only show stations clipped to the boundary polygon.
    if is_sa and extent is not None:
        from shapely.geometry import box as _sbox
        _ext_box = gpd.GeoDataFrame(
            geometry=[_sbox(extent[0], extent[2], extent[1], extent[3])],
            crs=train_stations.crs if train_stations.crs else SWISS_CRS)
        train_gdf_clip = gpd.clip(train_stations, _ext_box)
    else:
        train_gdf_clip = gpd.clip(train_stations, boundary_gdf)

    train_gdf_inside = gpd.clip(train_stations, boundary_gdf)
    inside_ids = set(train_gdf_inside.index.tolist()) if not train_gdf_inside.empty else set()

    _ms = 20 if not is_sa else 30
    if is_sa:
        # Ghost pass — all nodes within extent at 40% opacity, then solid inside on top.
        train_gdf_clip.plot(ax=ax, facecolor='white', edgecolor='black',
                            markersize=_ms, marker='o', linewidth=0.8, alpha=0.40, zorder=5)
        train_gdf_inside.plot(ax=ax, facecolor='white', edgecolor='black',
                              markersize=_ms, marker='o', linewidth=0.8, alpha=1.0, zorder=6)
    else:
        train_gdf_clip.plot(ax=ax, facecolor='white', edgecolor='black',
                            markersize=_ms, marker='o', linewidth=0.8, alpha=1.0, zorder=5)

    for idx, row in train_gdf_clip.iterrows():
        code = str(row.get("CODE", "")).strip()
        bp_num = int(row.get("Betriebspunkt_Nummer", 0))
        is_inside = idx in inside_ids
        termini = sorted(station_termini_texts.get(bp_num, set()))

        if not is_inside:
            continue

        # SA: always show code; CA: only when a service terminates here.
        if not (is_sa or termini):
            continue

        # Station code — bold, matching SA infrastructure plot style.
        ax.annotate(
            code,
            xy=(row.geometry.x, row.geometry.y),
            xytext=(5, 5), textcoords="offset points",
            fontsize=7, fontweight='bold', color="#333333", zorder=7,
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                      edgecolor='none', alpha=0.7),
        )

        # Terminating services — separate box below the station code.
        if termini:
            ax.annotate(
                "  ".join(termini),
                xy=(row.geometry.x, row.geometry.y),
                xytext=(5, -10), textcoords="offset points",
                fontsize=5, color="#333333", zorder=7, va="top",
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#f5f5f5',
                          edgecolor='#cccccc', linewidth=0.4, alpha=0.85),
            )

    grouped_cps = {}
    for cp_info in crossing_points:
        cp = cp_info["pt"]
        key = (round(cp.x / 100) * 100, round(cp.y / 100) * 100)
        if key not in grouped_cps:
            grouped_cps[key] = {"pt": cp, "texts": set(), "colour": cp_info["colour"],
                                 "udx": cp_info["udx"], "udy": cp_info["udy"]}
        grouped_cps[key]["texts"].add(cp_info["text"])

    if extent is not None:
        _xmin, _xmax, _ymin, _ymax = extent[0], extent[1], extent[2], extent[3]
    else:
        _xmin, _xmax = ax.get_xlim()
        _ymin, _ymax = ax.get_ylim()
    # Stub length: 3% of shorter map dimension, at least 375 m.
    _map_min = min(_xmax - _xmin, _ymax - _ymin)
    _preferred_stub = max(375, _map_min * 0.03)

    for cp_info in list(grouped_cps.values()):
        cp = cp_info["pt"]
        ux = cp_info["udx"]
        uy = cp_info["udy"]
        # Cap stub so endpoint stays within the plot extent (90% of available space).
        _margins = []
        if ux > 0  and _xmax > cp.x:  _margins.append((_xmax - cp.x) / ux)
        elif ux < 0 and _xmin < cp.x: _margins.append((_xmin - cp.x) / ux)
        if uy > 0  and _ymax > cp.y:  _margins.append((_ymax - cp.y) / uy)
        elif uy < 0 and _ymin < cp.y: _margins.append((_ymin - cp.y) / uy)
        _avail = min(_margins) * 0.90 if _margins else _preferred_stub
        stub_len = min(_preferred_stub, max(100, _avail))
        stub_end = Point(cp.x + ux * stub_len, cp.y + uy * stub_len)

        if abs(ux) >= abs(uy):
            ha     = "right" if ux < 0 else "left"
            va     = "center"
            offset = (-4, 0) if ux < 0 else (4, 0)
        else:
            ha     = "center"
            va     = "top" if uy < 0 else "bottom"
            offset = (0, -4) if uy < 0 else (0, 4)

        stub_line = LineString([(cp.x, cp.y), (stub_end.x, stub_end.y)])
        stub_gdf = gpd.GeoDataFrame({"geometry": [stub_line]}, crs=SWISS_CRS)
        stub_gdf.plot(ax=ax, color=cp_info["colour"], linewidth=1.0, linestyle="--", zorder=3)

        sorted_texts = sorted(list(cp_info["texts"]))
        table_text = "\n".join(sorted_texts)

        ax.annotate(
            table_text,
            xy=(stub_end.x, stub_end.y),
            xytext=offset, textcoords="offset points",
            fontsize=4, color="#333333", va=va, ha=ha,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="grey", alpha=0.7), zorder=5
        )

    legend_handles = [
        Line2D([0], [0], color=MODE_COLOURS["rail"], linewidth=2, label="Rail"),
        Line2D([0], [0], color=MODE_COLOURS["tram"], linewidth=2, label="Tram"),
        Line2D([0], [0], color=MODE_COLOURS["funicular"], linewidth=2, label="Funicular"),
        Line2D([0], [0], color=MODE_COLOURS["fallback"], linewidth=1.5, linestyle="--", label="Straight-line"),
        Line2D([0], [0], color="#d0d0d0", linewidth=1.5, label="Unused infrastructure"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7)

    ax.set_title(f"Service Projection — {config.svc_version} on {config.infra_version}\nBoundary: {boundary_name}", fontsize=14, fontweight='bold')

    if extent is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        
    _add_north_arrow(ax, location='upper left', scale=0.5)
    _add_scale_bar(ax, location=(0.755, 0.012))
    plt.tight_layout()

    out_dir = main / paths.NETWORK_PLOTS_DIR / "Rail_Lines" / config.svc_version / config.infra_version
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{config.svc_version}_{config.infra_version}_{boundary_name.replace(' ', '_')}.pdf"
    out_path = out_dir / fname
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {out_path}")

def _run_phase3(
    config: ProjectionConfig,
    rail_enriched: gpd.GeoDataFrame,
    tram_enriched: gpd.GeoDataFrame,
    func_enriched: gpd.GeoDataFrame,
) -> None:
    """Phase 3: produce overview plots for study area and catchment area."""
    print("\n" + "─" * 60)
    print("  Phase 3 — Plotting")
    print("─" * 60)

    main = Path(paths.MAIN)
    
    # Check Catchment_Area folder first for study_area_boundary (per user paths)
    study_area_boundary = main / paths.CATCHMENT_AREA_DIR / "study_area_boundary.gpkg"
    if not study_area_boundary.exists():
        study_area_boundary = main / paths.STUDY_AREA_BOUNDARY_GPKG
        
    for boundary_path, label in [
        (study_area_boundary, "study_area"),
        (main / paths.CATCHMENT_AREA_BOUNDARY_GPKG, "catchment_area"),
    ]:
        print(f"\n  Plotting {label}...")
        _plot_service_overview(
            config, rail_enriched, tram_enriched, func_enriched,
            boundary_path, label,
        )

    print("\n  Phase 3 complete.")

# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Interactive entry point for the service projection pipeline."""
    if not _check_prerequisites():
        raise SystemExit(1)

    result = _run_phase0()
    if result is None:
        raise SystemExit(0)

    config, mode = result

    if mode == "map":
        rail_enriched, tram_enriched, func_enriched = _run_phase1(config)
        rail_enriched = _run_phase1_5(config, rail_enriched)
    else:
        # Load existing projection for correction
        rail_out = config.rail_output_dir / "edges_in_corridor.gpkg"
        feeder_out = config.feeder_output_dir / "pt_feeder_segments.gpkg"
        if not rail_out.exists() or not feeder_out.exists():
            print(f"\n  ERROR: Projected files not found in {config.rail_output_dir}.")
            raise SystemExit(1)
        print(f"\n  Loading existing projection...")
        import fiona as _fiona
        _rail_layers = _fiona.listlayers(rail_out)
        _rail_gdfs   = []
        for _lyr in _rail_layers:
            _gdf = gpd.read_file(rail_out, layer=_lyr)
            _gdf["_source_layer"] = _lyr
            _rail_gdfs.append(_gdf)
        rail_enriched = (
            gpd.GeoDataFrame(pd.concat(_rail_gdfs, ignore_index=True), crs=SWISS_CRS)
            if _rail_gdfs else gpd.GeoDataFrame()
        )
        tram_enriched = gpd.read_file(feeder_out, layer="tram")
        func_enriched = gpd.read_file(feeder_out, layer="funicular")
        print(
            f"  Loaded: {len(rail_enriched)} rail links, "
            f"{len(tram_enriched)} tram segments, "
            f"{len(func_enriched)} funicular segments."
        )
        rail_enriched = _run_phase1_5(config, rail_enriched)

    rail_enriched, tram_enriched, func_enriched = _run_phase2(
        config, rail_enriched, tram_enriched, func_enriched
    )
    _run_phase3(config, rail_enriched, tram_enriched, func_enriched)

    print("\n" + "─" * 60)
    print("  Service projection complete.")
    print(f"  Rail output    : {config.rail_output_dir}")
    print(f"  Feeder output  : {config.feeder_output_dir}")
    print("─" * 60)


if __name__ == "__main__":
    main()
