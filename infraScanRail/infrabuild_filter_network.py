"""
Filter Infrastructure Network Module

Loads Swiss BAV railway geopackages, cleans the data, joins route attributes
onto segments, filters everything to the catchment area buffer, and produces
two output geopackages:

  - nodes.gpkg             : Operating points within (and at the edge of) the catchment area buffer
  - segments.gpkg          : Track segments with BAV + route attributes
  - segments_composition.gpkg : TLMRegio breakdown per BAV segment (one row per
                                 physical piece: tunnel section, bridge section, etc.)

Join keys (verified against BAV data):
  segments.rKmLinie       → routes.xtf_id         (route attributes)
  segments.rAnfangsknoten → nodes.xtf_id           (from-node topology)
  segments.rEndknoten     → nodes.xtf_id           (to-node topology)

All data in EPSG:2056 (Swiss LV95).
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from collections import defaultdict
import warnings
import sys
from shapely.ops import substring
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).parent))
import paths


# =============================================================================
# Constants
# =============================================================================

SWISS_CRS = "EPSG:2056"

NODE_CLASSIFICATION_PATTERNS = {
    '(Abzw)':     'junction',
    '(Wds)':      'turning_loop',
    '(Vzw)':      'junction_tram',
    '(Gbf)':      'freight_yard',
    '(Bbf)':      'operational_yard',
    '(Dep)':      'depot',
    '(Grenze)':   'border',
    '(Halt)':     'halt',
    '(Spw)':      'switch',          # Spurwechsel – track switch point
    '(km-Sprung)': 'km_change',      # Kilometer jump – km reference discontinuity
}

GAUGE_MAPPING = {
    'mm1435': 1435,
    'mm1668': 1668,
    'mm1520': 1520,
    'mm1000': 1000,
    'mm900':  900,
    'mm800':  800,
    'mm750':  750,
    'mm600':  600,
}

# Betriebspunkttyp_Bezeichnung (German, from HaltestellenOeV) → English node_class
BETRIEBSPUNKT_TYPE_MAPPING = {
    # Approved / Main Categories
    'Haltestelle': 'station',
    'Haltestelle und Bedienpunkt': 'station',
    'Verzweigung, Abzweigung, Spaltweiche': 'junction',
    'Anschlusspunkt': 'junction',
    'Spurtrennung': 'junction',
    'Ausweiche': 'junction',
    'Wendeschleife': 'turning_loop',
    
    # Will be dropped during macroscopic filtering:
    'Haltestelle ausser Betrieb': 'abandoned_station',
    'Dienststation': 'operational_yard',
    'Bedienpunkt': 'service_point',
    'Zugeordneter Betriebspunkt': 'assigned_service_point',
    'Spurwechsel': 'switch',
    'Fehlerprofil/Kilometer-Sprung': 'km_change',
    'Gleisende': 'track_end',
    'Eigentumsgrenze': 'property_border',
    'Landesgrenze': 'border',
    
    # Existing ones
    'Bahnhof': 'station',
    'Haltepunkt': 'halt',
    'Abzweigung': 'junction',
    'Blockstelle': 'junction',
    'Betriebsanlage': 'operational_yard',
    'Güterbahnhof': 'freight_yard',
    'Güteranlage': 'freight_yard',
    'Depot': 'depot',
    'Depotanlage': 'depot',
    'Fahrzeugdepot': 'depot',
    'Grenzbahnhof': 'border',
    'Grenzpunkt': 'border',
    'km-Sprung': 'km_change',
}

TRANSPORT_MODE_MAPPING = {
    'Zug': 'train',
    'Bus': 'bus',
    'Tram': 'tram',
    'Zahnradbahn': 'cog_railway',
    'Standseilbahn': 'funicular',
}

# TLMRegio CONSTRUCT field → English label
CONSTRUCT_MAPPING = {
    'Keine Kunstbaute': 'normal',
    'Brücke':           'bridge',
    'Tunnel':           'tunnel',
    'Galerie':          'gallery',
}

# TLMRegio OBJVAL field → English label
OBJVAL_MAPPING = {
    'NS_Bahn':     'standard_gauge',
    'SS_Bahn':     'narrow_gauge',
    'MS_Bahn':     'meter_gauge',
    'NS_BahnAuto': 'car_transport_standard_gauge',
    'SS_BahnAuto': 'car_transport_narrow_gauge',
    'Luftseilbahn':  'gondola',
    'Standseilbahn': 'funicular',
}

# --- Projection-based composition constants -----------------------------------

# Maximum tolerated offset (m) between TLMRegio centreline and BAV segment.
# Covers independently-digitized coordinate discrepancies (observed max ~99 m,
# 75th percentile ~17 m).  Set conservatively to catch outliers without
# pulling in features from a completely different nearby line.
MAX_SNAP_DISTANCE = 60

# Slivers shorter than this (m) are discarded after interval arithmetic.
MIN_PIECE_LENGTH = 5

# TLMRegio OBJVAL values that correspond to actual railway lines (excludes
# aerial lifts and funiculars which have no BAV counterpart).
RAIL_OBJVALS = frozenset({'NS_Bahn', 'SS_Bahn', 'MS_Bahn', 'NS_BahnAuto', 'SS_BahnAuto'})

# Edge case #4: map BAV gauge (integer mm) to the set of TLMRegio OBJVAL values
# that represent the same gauge group.  Prevents a standard-gauge BAV segment
# from inheriting tunnel/bridge attributes of a nearby narrow-gauge line.
GAUGE_TO_OBJVAL = {
    1435: frozenset({'NS_Bahn', 'NS_BahnAuto'}),
    1000: frozenset({'MS_Bahn', 'SS_Bahn', 'SS_BahnAuto'}),
    900:  frozenset({'SS_Bahn', 'SS_BahnAuto'}),
    800:  frozenset({'SS_Bahn', 'SS_BahnAuto'}),
    750:  frozenset({'SS_Bahn', 'SS_BahnAuto'}),
    600:  frozenset({'SS_Bahn', 'SS_BahnAuto'}),
}


# =============================================================================
# Helper functions
# =============================================================================

def classify_node(name: str) -> str:
    if pd.isna(name):
        return 'unknown'
    name_str = str(name)
    for pattern, cls in NODE_CLASSIFICATION_PATTERNS.items():
        if pattern in name_str:
            return cls
    return 'station'


def parse_gauge(gauge_str) -> int:
    if pd.isna(gauge_str):
        return 1435
    s = str(gauge_str).strip().lower()
    if s in GAUGE_MAPPING:
        return GAUGE_MAPPING[s]
    import re
    m = re.search(r'(\d+)', s)
    return int(m.group(1)) if m else 1435


def _line_endpoints(geom):
    """
    Return ((from_N, from_E), (to_N, to_E)) for a LineString or MultiLineString.
    Coordinates in EPSG:2056 convention: x=Easting, y=Northing.
    """
    if geom is None or geom.is_empty:
        return (None, None), (None, None)
    if geom.geom_type == 'MultiLineString':
        first = list(geom.geoms[0].coords)
        last  = list(geom.geoms[-1].coords)
    else:
        coords = list(geom.coords)
        first = coords
        last  = coords
    if not first or not last:
        return (None, None), (None, None)
    # x = Easting (E), y = Northing (N)
    return (first[0][1], first[0][0]), (last[-1][1], last[-1][0])


def _project_onto_line(tlm_geom, bav_line) -> Optional[Tuple[float, float]]:
    """
    Project a TLMRegio geometry onto a BAV LineString using linear referencing.

    Projects every vertex of tlm_geom onto bav_line and returns the
    (d_start, d_end) span in metres along bav_line.  Returns None when the
    projected span is shorter than MIN_PIECE_LENGTH (degenerate case #5).

    Args:
        tlm_geom : Shapely LineString or MultiLineString (TLMRegio feature)
        bav_line : Shapely LineString (one sub-line of a BAV MultiLineString)
    """
    if bav_line.length < 1:
        return None
    if tlm_geom.geom_type == 'LineString':
        coords = list(tlm_geom.coords)
    elif tlm_geom.geom_type == 'MultiLineString':
        coords = [c for sub in tlm_geom.geoms for c in sub.coords]
    else:
        return None

    dists = [bav_line.project(Point(c)) for c in coords]
    d_start = max(0.0, min(dists))
    d_end   = min(bav_line.length, max(dists))

    if d_end - d_start < MIN_PIECE_LENGTH:
        return None
    return d_start, d_end


def _merge_intervals(intervals):
    """
    Merge overlapping [(d_start, d_end, attrs_dict), ...] intervals.

    Sorted by d_start; when two intervals overlap the first one's attrs are
    kept (the structure that starts earlier takes precedence).
    """
    if not intervals:
        return []
    srt = sorted(intervals, key=lambda x: x[0])
    merged = [list(srt[0])]
    for d_start, d_end, attrs in srt[1:]:
        if d_start <= merged[-1][1]:          # overlap or touching — extend
            merged[-1][1] = max(merged[-1][1], d_end)
        else:
            merged.append([d_start, d_end, attrs])
    return [tuple(m) for m in merged]


def _fill_normal_gaps(intervals, total_length):
    """
    Insert normal-type intervals for every gap between projected structures.

    Any portion of the BAV line not covered by a tunnel/bridge/gallery
    interval is labelled construct_type='normal'.  Slivers shorter than
    MIN_PIECE_LENGTH are silently dropped.
    """
    normal_attrs = {
        'construct_type': 'normal',
        'edge_level': 1,
        'under_construction': 0,
        'track_config': None,
        'railway_type': None,
    }
    result = []
    pos = 0.0
    for d_start, d_end, attrs in intervals:
        if d_start - pos > MIN_PIECE_LENGTH:
            result.append((pos, d_start, dict(normal_attrs)))
        result.append((d_start, d_end, attrs))
        pos = d_end
    if total_length - pos > MIN_PIECE_LENGTH:
        result.append((pos, total_length, dict(normal_attrs)))
    return result


def _ensure_crs(gdf: gpd.GeoDataFrame, filepath: str) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        warnings.warn(f"No CRS in {filepath}, assuming {SWISS_CRS}")
        gdf = gdf.set_crs(SWISS_CRS)
    elif gdf.crs.to_epsg() != 2056:
        warnings.warn(f"Reprojecting {filepath} from {gdf.crs} to {SWISS_CRS}")
        gdf = gdf.to_crs(SWISS_CRS)
    return gdf


# =============================================================================
# Loaders
# =============================================================================

def load_bav_nodes(filepath: Optional[str] = None) -> gpd.GeoDataFrame:
    if filepath is None:
        filepath = Path(paths.MAIN) / paths.BAV_RAIL_NODES_GPKG
    gdf = gpd.read_file(filepath)
    gdf = _ensure_crs(gdf, filepath)
    print(f"Loaded {len(gdf)} nodes")
    return gdf


def load_bav_segments(filepath: Optional[str] = None) -> gpd.GeoDataFrame:
    if filepath is None:
        filepath = Path(paths.MAIN) / paths.BAV_RAIL_SEGMENTS_GPKG
    gdf = gpd.read_file(filepath)
    gdf = _ensure_crs(gdf, filepath)
    print(f"Loaded {len(gdf)} segments")
    return gdf


def load_bav_routes(filepath: Optional[str] = None) -> gpd.GeoDataFrame:
    if filepath is None:
        filepath = Path(paths.MAIN) / paths.BAV_RAIL_ROUTES_GPKG
    gdf = gpd.read_file(filepath)
    gdf = _ensure_crs(gdf, filepath)
    print(f"Loaded {len(gdf)} routes")
    return gdf


def load_tlmregio(filepath: Optional[str] = None) -> gpd.GeoDataFrame:
    if filepath is None:
        filepath = Path(paths.MAIN) / paths.TLMREGIO_RAILWAY_SHP
    gdf = gpd.read_file(filepath)
    gdf = _ensure_crs(gdf, filepath)
    print(f"Loaded {len(gdf)} TLMRegio segments")
    return gdf


def load_catchment_boundary(filepath: Optional[str] = None) -> gpd.GeoDataFrame:
    if filepath is None:
        filepath = Path(paths.MAIN) / paths.CATCHMENT_AREA_BUFFER_GPKG
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Catchment boundary not found: {filepath}")
    gdf = gpd.read_file(filepath)
    gdf = _ensure_crs(gdf, filepath)
    print(f"Loaded catchment area buffer ({len(gdf)} polygon(s))")
    return gdf


def load_haltestellen_oev(filepath: Optional[str] = None):
    """
    Load the HaltestellenOeV geopackage.

    Returns:
        betriebspunkte : GeoDataFrame — operating points with type classification
        haltekanten    : GeoDataFrame — platform edges linked to operating points
    """
    if filepath is None:
        filepath = Path(paths.MAIN) / paths.HALTESTELLEN_OEV_GPKG
    betriebspunkte = gpd.read_file(filepath, layer='Betriebspunkt')
    haltekanten    = gpd.read_file(filepath, layer='Haltekante')
    print(f"Loaded HaltestellenOeV: {len(betriebspunkte)} Betriebspunkte, "
          f"{len(haltekanten)} Haltekanten")
    return betriebspunkte, haltekanten


# =============================================================================
# Cleaning
# =============================================================================

def clean_nodes(raw_nodes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Standardise BAV node data.

    Output columns:
        node_id     – xtf_id (used for topology joins with segments)
        ID_point    – Betriebspunkt_Nummer (used for GTFS and HaltestellenOeV matching)
        NAME, CODE  – station name and abbreviation
        N, E        – Northing and Easting (EPSG:2056)
        parent_node – rUebergeordnet

    node_class, node_type_source, platform_count are added by enrich_nodes().
    """
    print("Cleaning nodes...")
    print(f"  Columns: {list(raw_nodes.columns)}")

    out = gpd.GeoDataFrame(geometry=raw_nodes.geometry.copy(), crs=raw_nodes.crs)

    out['node_id']  = raw_nodes['xtf_id']
    out['ID_point'] = raw_nodes.get('Betriebspunkt_Nummer', pd.NA)
    out['NAME']     = raw_nodes.get('Betriebspunkt_Name', pd.NA)
    out['CODE']     = raw_nodes.get('Betriebspunkt_Abkuerzung', pd.NA)
    out['E']        = out.geometry.x
    out['N']        = out.geometry.y
    out['parent_node']  = raw_nodes.get('rUebergeordnet', pd.NA)

    n_before = len(out)
    out = out.drop_duplicates(subset=['node_id'], keep='first')
    if len(out) < n_before:
        print(f"  Removed {n_before - len(out)} duplicate nodes")

    print(f"  → {len(out)} nodes")
    return out.reset_index(drop=True)


def clean_segments(raw_segments: gpd.GeoDataFrame,
                   raw_routes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Standardise BAV segment data and join route attributes.

    Route join:  segments.rKmLinie → routes.xtf_id

    Output columns:
        segment_id              – xtf_id
        from_node, to_node      – xtf_id references to nodes
        from_N/E, to_N/E        – endpoint coordinates
        length_m                – calculated from geometry
        num_tracks              – AnzahlStreckengleise
        gauge                   – Spurweite as integer mm
        electrification         – Elektrifizierung string
        km_start, km_end        – KmAnfang / KmEnde
        route_number            – Nummer (from routes)
        route_name              – Name (from routes)
        route_owner             – Datenherr_TUAbkuerzung (from routes)
        geometry                – MultiLineString preserved
    """
    print("Cleaning segments...")
    print(f"  Columns: {list(raw_segments.columns)}")

    out = gpd.GeoDataFrame(geometry=raw_segments.geometry.copy(), crs=raw_segments.crs)

    out['segment_id']   = raw_segments['xtf_id']
    out['segment_name'] = raw_segments.get('Name', pd.NA)
    out['from_node']    = raw_segments['rAnfangsknoten']   # xtf_id, replaced by name after filtering
    out['to_node']      = raw_segments['rEndknoten']       # xtf_id, replaced by name after filtering

    # Endpoint coordinates from geometry
    endpoints = out.geometry.apply(_line_endpoints)
    out['from_N'] = endpoints.apply(lambda x: x[0][0])
    out['from_E'] = endpoints.apply(lambda x: x[0][1])
    out['to_N']   = endpoints.apply(lambda x: x[1][0])
    out['to_E']   = endpoints.apply(lambda x: x[1][1])

    out['length_m'] = out.geometry.length

    out['num_tracks'] = (pd.to_numeric(raw_segments.get('AnzahlStreckengleise'), errors='coerce')
                           .fillna(1).astype(int))
    out['gauge'] = raw_segments.get('Spurweite', pd.NA).apply(parse_gauge)
    out['electrification'] = raw_segments.get('Elektrifizierung', 'unknown')
    out['km_start'] = raw_segments.get('KmAnfang', pd.NA)
    out['km_end']   = raw_segments.get('KmEnde', pd.NA)

    # ---- Route attribute join ------------------------------------------------
    route_lookup = raw_routes.set_index('xtf_id')[['Nummer', 'Name', 'Datenherr_TUAbkuerzung']]
    km_line = raw_segments['rKmLinie']
    out['route_number'] = km_line.map(route_lookup['Nummer'])
    out['route_name']   = km_line.map(route_lookup['Name'])
    out['route_owner']  = km_line.map(route_lookup['Datenherr_TUAbkuerzung'])

    matched = out['route_number'].notna().sum()
    print(f"  → {len(out)} segments  |  route join: {matched}/{len(out)} matched")
    print(f"  Total length: {out['length_m'].sum() / 1000:.0f} km")
    print(f"  Gauge distribution: {out['gauge'].value_counts().to_dict()}")

    return out.reset_index(drop=True)


# =============================================================================
# Node enrichment from HaltestellenOeV
# =============================================================================

def enrich_nodes(nodes: gpd.GeoDataFrame,
                 betriebspunkte: gpd.GeoDataFrame,
                 haltekanten: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Enrich nodes with authoritative type classification, transport mode, and
    platform counts from HaltestellenOeV.gpkg.

    node_class:
        Betriebspunkttyp_Bezeichnung (German, as-is) for matched nodes (~98.6 %).
        Falls back to pattern-match on station name abbreviations for the rest.

    transport_mode:
        Verkehrsmittel_Bezeichnung – the transport mode serving the node
        (e.g. "Eisenbahn", "Tram", "Bus", …).  NaN for unmatched nodes.

    platform_count:
        Number of Haltekanten per operating point.
        NaN for nodes that are not public transport stops (junctions, switches, …).

    Join keys:
        nodes.ID_point (Betriebspunkt_Nummer) → betriebspunkte.Nummer
        haltekanten.rHaltestelle              → betriebspunkte.xtf_id
    """
    print("Enriching nodes from HaltestellenOeV...")
    nodes = nodes.copy()

    bp_num  = betriebspunkte['Nummer'].astype('Int64')
    node_ids = nodes['ID_point'].astype('Int64')

    # --- node_class: authoritative classification ------------------------------
    bp_lookup = dict(zip(bp_num, betriebspunkte['Betriebspunkttyp_Bezeichnung']))
    authoritative = node_ids.map(bp_lookup)

    nodes['node_class'] = authoritative.where(authoritative.notna(),
                                               nodes['NAME'].apply(classify_node))

    # Normalise German Betriebspunkttyp_Bezeichnung values to English
    nodes['node_class'] = nodes['node_class'].apply(
        lambda x: BETRIEBSPUNKT_TYPE_MAPPING.get(str(x), x) if pd.notna(x) else x
    )

    matched_type = authoritative.notna().sum()
    print(f"  Node classification: {matched_type}/{len(nodes)} from HaltestellenOeV "
          f"({len(nodes) - matched_type} pattern fallback)")

    # --- transport_mode: Verkehrsmittel_Bezeichnung ----------------------------
    mode_lookup = dict(zip(bp_num, betriebspunkte['Verkehrsmittel_Bezeichnung']))
    nodes['transport_mode'] = node_ids.map(mode_lookup)

    # Normalise German Verkehrsmittel_Bezeichnung values to English
    def translate_modes(val):
        if pd.isna(val):
            return val
        parts = [TRANSPORT_MODE_MAPPING.get(p.strip(), p.strip()) for p in str(val).split('/')]
        return ' / '.join(parts)

    nodes['transport_mode'] = nodes['transport_mode'].apply(translate_modes)

    # --- platform_count: Haltekante aggregation --------------------------------
    # Haltekante.rHaltestelle → Betriebspunkt.xtf_id → Betriebspunkt.Nummer
    bp_xtf_to_nummer = betriebspunkte.set_index('xtf_id')['Nummer'].astype('Int64').to_dict()

    hk_counts = (haltekanten
                 .groupby('rHaltestelle')
                 .size()
                 .reset_index(name='platform_count'))
    hk_counts['Nummer'] = hk_counts['rHaltestelle'].map(bp_xtf_to_nummer)

    nummer_to_platforms = (hk_counts
                           .dropna(subset=['Nummer'])
                           .set_index('Nummer')['platform_count']
                           .to_dict())

    nodes['platform_count'] = node_ids.map(nummer_to_platforms)

    matched_plats = nodes['platform_count'].notna().sum()
    print(f"  Platform counts: {matched_plats} nodes have platform data")

    return nodes


# =============================================================================
# Catchment area filtering
# =============================================================================

def filter_to_catchment(
    nodes: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Filter nodes and segments to the catchment area.

    Rules:
      - Nodes within the boundary are kept.
      - Segments are kept if at least one endpoint node is inside the boundary.
        Cross-boundary segments are kept with full (unclipped) geometry.
      - Endpoint nodes that lie outside the boundary are also kept when they
        belong to a kept cross-boundary segment.

    Returns:
        (filtered_nodes, filtered_segments)
    """
    print("Filtering to catchment boundary...")

    boundary_geom = boundary.geometry.union_all()

    # --- Classify nodes -------------------------------------------------------
    inside_mask   = nodes.geometry.within(boundary_geom)
    inside_ids    = set(nodes.loc[inside_mask, 'node_id'])
    print(f"  Nodes inside boundary: {inside_mask.sum()} / {len(nodes)}")

    # --- Filter segments (at least one endpoint inside) -----------------------
    seg_mask = (
        segments['from_node'].isin(inside_ids) |
        segments['to_node'].isin(inside_ids)
    )
    filtered_segs = segments[seg_mask].copy()
    print(f"  Segments kept: {seg_mask.sum()} / {len(segments)}")

    # --- Add outside endpoint nodes for cross-boundary segments ---------------
    all_endpoint_ids     = set(filtered_segs['from_node']) | set(filtered_segs['to_node'])
    outside_endpoint_ids = all_endpoint_ids - inside_ids
    outside_nodes        = nodes[nodes['node_id'].isin(outside_endpoint_ids)]
    print(f"  Outside endpoint nodes included: {len(outside_nodes)}")

    filtered_nodes = pd.concat(
        [nodes[inside_mask], outside_nodes]
    ).drop_duplicates(subset=['node_id']).reset_index(drop=True)

    print(f"  → {len(filtered_nodes)} nodes, {len(filtered_segs)} segments after filter")
    return filtered_nodes, filtered_segs


# =============================================================================
# Segments composition (TLMRegio breakdown)
# =============================================================================

def build_segments_composition(
    filtered_segments: gpd.GeoDataFrame,
    tlmregio: Optional[gpd.GeoDataFrame] = None
) -> gpd.GeoDataFrame:
    """
    Build a per-piece breakdown of each BAV segment using TLMRegio data.

    Uses linear-referencing projection rather than polygon overlay, so the
    output is gap-free: every metre of every BAV segment gets a label.

    Algorithm
    ---------
    1. Filter TLMRegio to rail-only non-normal features (tunnels/bridges/galleries).
    2. Explode BAV MultiLineStrings to individual sub-lines (edge case #1).
    3. Spatial join with MAX_SNAP_DISTANCE buffer to find candidate pairs
       (handles independently-digitised coordinate offsets up to ~60 m).
    4. Filter pairs by gauge-compatible OBJVAL (edge case #4).
    5. For each candidate pair project the TLMRegio endpoints onto the BAV
       sub-line → (d_start, d_end) interval on the BAV line (independent of
       coordinate offset).
    6. A TLMRegio feature may project onto multiple BAV sub-lines (edge
       case #2: long tunnel spanning two segments).
    7. Per sub-line: merge overlapping intervals (edge case #3: multiple
       features on the same segment), then fill the remaining gaps as 'normal'.
    8. Extract geometry via shapely.ops.substring; discard slivers < MIN_PIECE_LENGTH.

    Output columns
    --------------
        segment_id, from_node, to_node,
        construct_type, edge_level, under_construction,
        track_config, railway_type, piece_length_m, geometry
    """
    print("Building segments_composition (projection-based)...")

    if tlmregio is None:
        try:
            tlmregio = load_tlmregio()
        except FileNotFoundError:
            warnings.warn("TLMRegio file not found – segments_composition will not be written")
            return gpd.GeoDataFrame()

    # ------------------------------------------------------------------
    # 1. Prepare TLMRegio: rail-only, non-normal features
    # ------------------------------------------------------------------
    tlm_rail = tlmregio[tlmregio['OBJVAL'].isin(RAIL_OBJVALS)].copy()
    non_normal = tlm_rail[tlm_rail['CONSTRUCT'] != 'Keine Kunstbaute'].copy()
    non_normal = non_normal.reset_index(drop=True)

    non_normal['construct_type']     = non_normal['CONSTRUCT'].map(CONSTRUCT_MAPPING).fillna('normal')
    non_normal['edge_level']         = pd.to_numeric(non_normal['EDGELEVEL'], errors='coerce').fillna(1).astype(int)
    non_normal['under_construction'] = pd.to_numeric(non_normal['UNDERCONST'], errors='coerce').fillna(0).astype(int)
    non_normal['track_config']       = non_normal['FCO'].fillna('Unknown')
    non_normal['railway_type']       = non_normal['OBJVAL'].map(OBJVAL_MAPPING).fillna(non_normal['OBJVAL'])

    n_bridge  = non_normal['construct_type'].eq('bridge').sum()
    n_tunnel  = non_normal['construct_type'].eq('tunnel').sum()
    n_gallery = non_normal['construct_type'].eq('gallery').sum()
    print(f"  Non-normal rail TLMRegio: {len(non_normal)}  "
          f"(bridges: {n_bridge}, tunnels: {n_tunnel}, galleries: {n_gallery})")

    # ------------------------------------------------------------------
    # 2. Explode BAV MultiLineStrings → one row per sub-line (edge case #1)
    # ------------------------------------------------------------------
    segs_for_comp = filtered_segments[
        ['segment_id', 'from_node', 'to_node', 'gauge', 'geometry']
    ].copy()
    segs_expl = segs_for_comp.explode(index_parts=False).reset_index(drop=True)
    segs_expl['_sub_idx'] = segs_expl.groupby('segment_id').cumcount()
    print(f"  BAV sub-lines: {len(segs_expl)}  (from {len(filtered_segments)} segments)")

    # ------------------------------------------------------------------
    # 3. Spatial match: buffer non-normal TLMRegio → find candidate pairs
    #    (edge case #2: a single TLMRegio feature can match multiple sub-lines)
    # ------------------------------------------------------------------
    nn_search = non_normal[
        ['construct_type', 'edge_level', 'under_construction',
         'track_config', 'railway_type', 'OBJVAL', 'geometry']
    ].copy()
    nn_search['geometry'] = nn_search.geometry.buffer(MAX_SNAP_DISTANCE)

    matches = gpd.sjoin(segs_expl, nn_search, how='inner', predicate='intersects')
    print(f"  Candidate (BAV sub-line × TLMRegio) pairs: {len(matches)}")

    # ------------------------------------------------------------------
    # 4. Gauge-compatibility filter (edge case #4)
    # ------------------------------------------------------------------
    def _gauge_ok(row):
        allowed = GAUGE_TO_OBJVAL.get(int(row['gauge']), RAIL_OBJVALS)
        return row['OBJVAL'] in allowed

    compat_mask = matches.apply(_gauge_ok, axis=1)
    n_dropped = (~compat_mask).sum()
    if n_dropped:
        print(f"  Gauge-incompatible pairs removed: {n_dropped}")
    matches = matches[compat_mask].copy()

    # ------------------------------------------------------------------
    # 5. Project TLMRegio endpoints onto each BAV sub-line
    # ------------------------------------------------------------------
    # matches.geometry is the BAV sub-line geometry (left side of sjoin).
    # index_right indexes into non_normal for the original (unbuffered) TLMRegio geom.
    def _compute_interval(row):
        tlm_geom = non_normal.geometry.at[int(row['index_right'])]
        return _project_onto_line(tlm_geom, row.geometry)

    matches['_interval'] = matches.apply(_compute_interval, axis=1)
    valid = matches[matches['_interval'].notna()].copy()
    valid['_d_start'] = valid['_interval'].apply(lambda x: x[0])
    valid['_d_end']   = valid['_interval'].apply(lambda x: x[1])
    print(f"  Valid projected intervals: {len(valid)}  (of {len(matches)} candidates)")

    # ------------------------------------------------------------------
    # 6 & 7. Collect intervals per sub-line, merge overlaps, fill normal gaps
    #         (edge cases #2 and #3)
    # ------------------------------------------------------------------
    # Build lookup: (segment_id, _sub_idx) → list of (d_start, d_end, attrs)
    interval_map = defaultdict(list)
    attr_cols = ['construct_type', 'edge_level', 'under_construction',
                 'track_config', 'railway_type']
    for _, row in valid[['segment_id', '_sub_idx', '_d_start', '_d_end'] + attr_cols].iterrows():
        key   = (row['segment_id'], row['_sub_idx'])
        attrs = {c: row[c] for c in attr_cols}
        interval_map[key].append((row['_d_start'], row['_d_end'], attrs))

    # Build sub-line geometry + topology lookup
    subline_info = {}
    for _, row in segs_expl.iterrows():
        subline_info[(row['segment_id'], row['_sub_idx'])] = {
            'geometry': row.geometry,
            'from_node': row['from_node'],
            'to_node':   row['to_node'],
        }

    # ------------------------------------------------------------------
    # 8. Extract geometry pieces via linear referencing
    # ------------------------------------------------------------------
    output_rows = []
    for key, info in subline_info.items():
        bav_line = info['geometry']
        total    = bav_line.length

        if key in interval_map:
            merged = _merge_intervals(interval_map[key])
            pieces = _fill_normal_gaps(merged, total)
        else:
            # No non-normal structure found → entire sub-line is normal
            pieces = [(0.0, total, {
                'construct_type': 'normal', 'edge_level': 1,
                'under_construction': 0, 'track_config': None, 'railway_type': None,
            })]

        for d_start, d_end, attrs in pieces:
            try:
                geom = substring(bav_line, d_start, d_end)
            except Exception:
                continue
            if geom is None or geom.is_empty or geom.length < MIN_PIECE_LENGTH:
                continue
            output_rows.append({
                'segment_id':        key[0],
                'from_node':         info['from_node'],
                'to_node':           info['to_node'],
                'construct_type':    attrs['construct_type'],
                'edge_level':        attrs['edge_level'],
                'under_construction': attrs['under_construction'],
                'track_config':      attrs['track_config'],
                'railway_type':      attrs['railway_type'],
                'piece_length_m':    geom.length,
                'geometry':          geom,
            })

    if not output_rows:
        print("  Warning: no composition pieces generated")
        return gpd.GeoDataFrame()

    composition = gpd.GeoDataFrame(output_rows, crs=SWISS_CRS)
    print(f"  → {len(composition)} composition pieces")
    print(f"     construct_type: {composition['construct_type'].value_counts().to_dict()}")
    return composition


# =============================================================================
# Node name substitution
# =============================================================================

def substitute_node_names(df: gpd.GeoDataFrame,
                          nodes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Replace xtf_id values in from_node / to_node with human-readable node NAMEs.
    Renames the columns to from_name / to_name.

    Called after filtering so that all referenced nodes are present in the
    filtered node table.  Unmatched IDs (should not occur) are left as-is.
    """
    id_to_name = nodes.set_index('node_id')['NAME'].to_dict()
    df = df.copy()
    df['from_node'] = df['from_node'].map(id_to_name).fillna(df['from_node'])
    df['to_node']   = df['to_node'].map(id_to_name).fillna(df['to_node'])
    df = df.rename(columns={'from_node': 'from_name', 'to_node': 'to_name'})
    return df


# =============================================================================
# Lookup tables
# =============================================================================

def build_node_lookups(nodes: gpd.GeoDataFrame) -> Dict[str, dict]:
    """Build lookup tables for downstream GTFS matching."""
    lookups = {'by_node_id': {}, 'by_bp_nummer': {}, 'by_name': {}, 'by_code': {}}
    for _, row in nodes.iterrows():
        nid = row['node_id']
        lookups['by_node_id'][nid] = nid
        if pd.notna(row.get('ID_point')):
            lookups['by_bp_nummer'][row['ID_point']] = nid
        if pd.notna(row.get('NAME')) and row['NAME']:
            lookups['by_name'][str(row['NAME']).lower().strip()] = nid
        if pd.notna(row.get('CODE')) and row['CODE']:
            lookups['by_code'][str(row['CODE']).strip()] = nid
    print(f"Built lookups: {len(lookups['by_node_id'])} node_id, "
          f"{len(lookups['by_bp_nummer'])} BP_Nummer, "
          f"{len(lookups['by_name'])} by name, {len(lookups['by_code'])} by code")
    return lookups


# =============================================================================
# Main pipeline
# =============================================================================

def run_filter_network(
    output_dir: Optional[str] = None,
    catchment_filepath: Optional[str] = None,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, Dict]:
    """
    Full pipeline: load → clean → filter → compose → substitute names → export.

    Args:
        output_dir:         Where to write the three geopackages.
                            Defaults to paths.NETWORK_INFRASTRUCTURE_BASE.
        catchment_filepath: Catchment area buffer file.
                            Defaults to paths.CATCHMENT_AREA_BUFFER_GPKG.

    Returns:
        (filtered_nodes, filtered_segments, composition, lookups)
    """
    print("=" * 60)
    print("Filter Infrastructure Network")
    print("=" * 60)

    output_dir = Path(output_dir or (Path(paths.MAIN) / paths.NETWORK_INFRASTRUCTURE_RAW))
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load -----------------------------------------------------------------
    print("\n--- Loading BAV data ---")
    raw_nodes    = load_bav_nodes()
    raw_segments = load_bav_segments()
    raw_routes   = load_bav_routes()
    boundary     = load_catchment_boundary(catchment_filepath)
    betriebspunkte, haltekanten = load_haltestellen_oev()

    # --- Clean ----------------------------------------------------------------
    print("\n--- Cleaning ---")
    nodes    = clean_nodes(raw_nodes)
    segments = clean_segments(raw_segments, raw_routes)

    # --- Enrich nodes ---------------------------------------------------------
    print("\n--- Enriching nodes ---")
    nodes = enrich_nodes(nodes, betriebspunkte, haltekanten)

    # --- Filter ---------------------------------------------------------------
    print("\n--- Filtering to catchment ---")
    nodes, segments = filter_to_catchment(nodes, segments, boundary)

    # --- Composition (built while from/to are still xtf_ids for overlay) ------
    print("\n--- Building segments_composition ---")
    composition = build_segments_composition(segments)

    # --- Replace xtf_id references with node names ----------------------------
    print("\n--- Substituting node names ---")
    segments    = substitute_node_names(segments, nodes)
    if not composition.empty:
        composition = substitute_node_names(composition, nodes)

    # --- Lookups --------------------------------------------------------------
    print("\n--- Building node lookups ---")
    lookups = build_node_lookups(nodes)

    # --- Export ---------------------------------------------------------------
    print("\n--- Exporting ---")
    nodes_path       = output_dir / "nodes.gpkg"
    segments_path    = output_dir / "segments.gpkg"
    composition_path = output_dir / "segments_composition.gpkg"

    nodes.to_file(nodes_path, driver="GPKG")
    print(f"  nodes.gpkg                → {nodes_path}")

    segments.to_file(segments_path, driver="GPKG")
    print(f"  segments.gpkg             → {segments_path}")

    if not composition.empty:
        composition.to_file(composition_path, driver="GPKG")
        print(f"  segments_composition.gpkg → {composition_path}")

    print("\n" + "=" * 60)
    print(f"Done  |  {len(nodes)} nodes  |  {len(segments)} segments  |  "
          f"{len(composition)} composition pieces")
    print("=" * 60)

    return nodes, segments, composition, lookups


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import os
    os.chdir(paths.MAIN)

    print("=" * 60)
    print("infraScanRail — Filter Infrastructure Network")
    print("=" * 60)
    print("Loads BAV railway geopackages, cleans and enriches the data,")
    print("filters to the catchment area, and exports to Raw/.")
    print("\nOutputs: data/Infrastructure/Raw/nodes.gpkg")
    print("         data/Infrastructure/Raw/segments.gpkg")
    print("         data/Infrastructure/Raw/segments_composition.gpkg")

    raw_path = Path(paths.MAIN) / paths.NETWORK_INFRASTRUCTURE_RAW
    raw_ready = (
        (raw_path / "nodes.gpkg").exists() and
        (raw_path / "segments.gpkg").exists()
    )

    if raw_ready:
        print("\n   Raw/ already exists.")
        ans = input("   Re-run the filter and overwrite Raw/? (y/n) [n]: ").strip().lower() or "n"
        if ans != 'y':
            print("   Nothing to do. Exiting.")
            raise SystemExit(0)

    run_filter_network()
