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
import requests
from pathlib import Path
from typing import Dict, Tuple, Optional
from collections import defaultdict
import warnings
import sys
from shapely.ops import substring
from shapely.geometry import LineString, Point

sys.path.insert(0, str(Path(__file__).parent))
import paths


# =============================================================================
# OpenStreetMap (OSM) Fetch Constants
# =============================================================================
# Railway speed data is sourced from OpenStreetMap via the Overpass API.
# OpenRailwayMap (ORM) is a rendering layer on top of OSM and has no
# separate data API — the maxspeed tags live in the OSM database itself.
# The bounding box passed to Overpass is a rectangle derived from the
# catchment buffer extent.  Extra ways outside the buffer are harmless:
# they are filtered out during spatial joining in the network builder.

# Overpass API endpoints — tried in order until one responds.
OVERPASS_INSTANCES = [
    "https://overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
OVERPASS_TIMEOUT = 120
RAILWAY_TYPES    = "rail|light_rail|tram|subway|narrow_gauge|funicular"


def _osm_bbox_wgs84(gdf: gpd.GeoDataFrame) -> Tuple[float, float, float, float]:
    """Return (south, west, north, east) bounding box in WGS84 for a GeoDataFrame."""
    bounds = gdf.to_crs("EPSG:4326").total_bounds  # minx, miny, maxx, maxy
    return bounds[1], bounds[0], bounds[3], bounds[2]


def fetch_osm_ways(bbox: Tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    """
    Fetch all railway ways from OpenStreetMap (OSM) via the Overpass API.

    The query uses a rectangular bounding box derived from the filtered BAV
    segments.  Ways outside the catchment area are discarded automatically
    during spatial joining in the network builder.

    Args:
        bbox: (south, west, north, east) in WGS84 degrees.

    Returns:
        GeoDataFrame in EPSG:2056 with columns:
        osm_id, railway_type, maxspeed, maxspeed_forward,
        maxspeed_backward, osm_name, geometry.
    """
    south, west, north, east = bbox
    query = (
        f'[out:json][bbox:{south:.4f},{west:.4f},{north:.4f},{east:.4f}]'
        f'[timeout:{OVERPASS_TIMEOUT}];'
        f'way["railway"~"^({RAILWAY_TYPES})$"];'
        f'out geom;'
    )
    headers = {"User-Agent": "infraScanRail/1.0", "Accept": "application/json"}

    print(f"  Querying Overpass (S={south:.3f} W={west:.3f} N={north:.3f} E={east:.3f})...")
    last_error = None
    r = None
    for url in OVERPASS_INSTANCES:
        try:
            print(f"    Trying {url} ...")
            r = requests.post(url, data={"data": query}, headers=headers,
                              timeout=OVERPASS_TIMEOUT + 30)
            r.raise_for_status()
            break
        except requests.exceptions.RequestException as e:
            print(f"    Failed ({e}), trying next instance...")
            last_error = e

    if r is None or not r.ok:
        raise RuntimeError(f"All Overpass instances failed. Last error: {last_error}")

    elements = r.json().get("elements", [])
    print(f"  Received {len(elements)} railway ways")

    rows = []
    for el in elements:
        if el.get("type") != "way":
            continue
        pts = el.get("geometry", [])
        if len(pts) < 2:
            continue
        tags = el.get("tags", {})
        rows.append({
            "osm_id":            el["id"],
            "railway_type":      tags.get("railway"),
            "maxspeed":          tags.get("maxspeed"),
            "maxspeed_forward":  tags.get("maxspeed:forward"),
            "maxspeed_backward": tags.get("maxspeed:backward"),
            "osm_name":          tags.get("name"),
            "geometry":          LineString([(p["lon"], p["lat"]) for p in pts]),
        })

    if not rows:
        return gpd.GeoDataFrame(
            columns=["osm_id", "railway_type", "maxspeed", "geometry"],
            crs="EPSG:4326",
        )
    return gpd.GeoDataFrame(rows, crs="EPSG:4326").to_crs("EPSG:2056")


# =============================================================================
# Constants
# =============================================================================

SWISS_CRS = "EPSG:2056"

# BAV node name patterns used by classify_node() — documented here for reference.
# The function implements priority / spacing rules directly; this dict is not iterated.
NODE_CLASSIFICATION_PATTERNS = {
    # turning_loop — checked first (Wds beats any junction suffix in same name)
    '(Wds)':      'turning_loop',    # also: 'Wds ' (space after), ' Wds)' (before closing paren)
    '(boucle)':   'turning_loop',    # French
    # junction — German and French variants
    '(Abzw)':     'junction',
    '(Vzw':       'junction',        # catches (Vzw), (Vzw Ost/Nord/Süd/West/…)
    '(Verzw)':    'junction',        # long-form German variant
    '(bif':       'junction',        # French bifurcation; catches (bif), (bif sud), (bif nord)
    '(embr)':     'junction',        # French embranchement
    # other
    ' GB':        'freight_yard',    # space before GB, case-sensitive, no brackets
    'Depot':      'depot',           # substring match
    ' Grenze':    'border',          # space before Grenze, case-sensitive
    '(Km-Sprung)': 'km_change',      # also: (saut-km), (saut km)
    '(saut-km)':  'km_change',
    '(saut km)':  'km_change',       # French variant with space instead of hyphen
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
# Only values verified to exist in the geopackage are included.
BETRIEBSPUNKT_TYPE_MAPPING = {
    'Haltestelle':                          'station',
    'Haltestelle und Bedienpunkt':          'station',
    'Haltestelle ausser Betrieb':           'abandoned_station',
    'Bedienpunkt':                          'service_point',
    'Zugeordneter Betriebspunkt':           'assigned_service_point',
    'Verzweigung, Abzweigung, Spaltweiche': 'junction',
    'Anschlusspunkt':                       'junction',
    'Spurtrennung':                         'junction',
    'Ausweiche':                            'junction',
    'Blockstelle':                          'junction',
    'Wendeschleife':                        'turning_loop',
    'Dienststation':                        'operational_yard',
    'Eigentumsgrenze':                      'property_border',
    'Landesgrenze':                         'border',
    'Spurwechsel':                          'switch',
    'Fehlerprofil/Kilometer-Sprung':        'km_change',
    'Gleisende':                            'track_end',
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
    """
    Classify a BAV node from its Betriebspunkt_Name using Phase-1 pattern matching.

    All patterns are case-sensitive (matching the source data).
    Wds / boucle → turning_loop takes priority over all junction patterns.
    Returns 'unclassified' when no pattern matches.

    Junction patterns cover German and French naming conventions:
        (Abzw), (Vzw*), (Verzw)  – German Abzweigung / Verzweigung
        (bif*)                   – French bifurcation
        (embr)                   – French embranchement

    Turning-loop patterns:
        (Wds), Wds <space>, <space>Wds)  – catches (Wds), bare 'Wds ', and
                                            combined forms like (Vzw Wds)
        (boucle)                          – French boucle
    """
    if pd.isna(name):
        return 'unclassified'
    s = str(name)

    # turning_loop — checked first; Wds beats any junction suffix in same name
    has_wds = '(Wds)' in s or 'Wds ' in s or ' Wds)' in s
    if has_wds or '(boucle)' in s:
        return 'turning_loop'

    # junction — German and French variants
    if '(Vzw' in s or '(Abzw)' in s or '(Verzw)' in s or '(bif' in s or '(embr)' in s:
        return 'junction'

    if ' GB' in s:
        return 'freight_yard'
    if 'Depot' in s:
        return 'depot'
    if ' Grenze' in s:
        return 'border'
    if '(Km-Sprung)' in s or '(saut-km)' in s or '(saut km)' in s:
        return 'km_change'
    return 'unclassified'


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
        node_id              – xtf_id (used for topology joins with segments)
        Betriebspunkt_Nummer – stable numeric public ID (matching key for HaltestellenOeV)
        NAME, CODE           – station name and abbreviation
        N, E                 – Northing and Easting (EPSG:2056)
        parent_node          – rUebergeordnet

    node_class, transport_mode, platform_count, track_count are added later.
    """
    print("Cleaning nodes...")
    print(f"  Columns: {list(raw_nodes.columns)}")

    out = gpd.GeoDataFrame(geometry=raw_nodes.geometry.copy(), crs=raw_nodes.crs)

    out['node_id']              = raw_nodes['xtf_id']
    out['Betriebspunkt_Nummer'] = raw_nodes.get('Betriebspunkt_Nummer', pd.NA)
    out['NAME']                 = raw_nodes.get('Betriebspunkt_Name', pd.NA)
    out['CODE']                 = raw_nodes.get('Betriebspunkt_Abkuerzung', pd.NA)
    out['E']                    = out.geometry.x
    out['N']                    = out.geometry.y
    out['parent_node']          = raw_nodes.get('rUebergeordnet', pd.NA)

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
# Node classification and enrichment
# =============================================================================

def classify_nodes_bav(nodes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Phase-1 node classification from BAV Betriebspunkt_Name patterns.

    Applied after catchment filtering.  Sets node_class for all nodes that
    match a known pattern; unmatched nodes are labelled 'unclassified' and
    will be resolved in Phase 2 (enrich_nodes_oev).
    """
    print("Phase-1: classifying nodes from BAV name patterns...")
    nodes = nodes.copy()
    nodes['node_class'] = nodes['NAME'].apply(classify_node)
    n_classified   = (nodes['node_class'] != 'unclassified').sum()
    n_unclassified = (nodes['node_class'] == 'unclassified').sum()
    print(f"  Classified: {n_classified}  |  Unclassified (→ OeV Phase 2): {n_unclassified}")
    return nodes


def enrich_nodes_oev(
    nodes: gpd.GeoDataFrame,
    betriebspunkte: gpd.GeoDataFrame,
    haltekanten: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Phase-2 node enrichment from HaltestellenOeV.gpkg.

    Steps
    -----
    1. Match BAV nodes via Betriebspunkt_Nummer ↔ betriebspunkte.Nummer
    2. Import transport_mode (translated to English)
    3. Fill node_class for 'unclassified' nodes from Betriebspunkttyp_Bezeichnung
    4. Count platforms (Haltekanten) per operating point; enforce only for stations
    5. Add OeV-only nodes: Verkehrsmittel_Code ∋ {B,C,E,F}, not in BAV, within boundary

    Join keys:
        nodes.Betriebspunkt_Nummer → betriebspunkte.Nummer
        haltekanten.rHaltestelle   → betriebspunkte.xtf_id
    """
    print("Phase-2: enriching nodes from HaltestellenOeV...")
    nodes = nodes.copy()

    # Ensure betriebspunkte CRS matches nodes
    if betriebspunkte.crs is None:
        betriebspunkte = betriebspunkte.set_crs(SWISS_CRS)
    elif betriebspunkte.crs.to_epsg() != 2056:
        betriebspunkte = betriebspunkte.to_crs(SWISS_CRS)

    bp_num   = betriebspunkte['Nummer'].astype('Int64')
    node_ids = nodes['Betriebspunkt_Nummer'].astype('Int64')

    # --- transport_mode -------------------------------------------------------
    def _translate_modes(val):
        if pd.isna(val):
            return val
        parts = [TRANSPORT_MODE_MAPPING.get(p.strip(), p.strip()) for p in str(val).split('/')]
        return ' / '.join(parts)

    mode_lookup = dict(zip(bp_num, betriebspunkte['Verkehrsmittel_Bezeichnung']))
    nodes['transport_mode'] = node_ids.map(mode_lookup).apply(_translate_modes)

    # --- node_class: fill only unclassified nodes ----------------------------
    bp_type_lookup = dict(zip(bp_num, betriebspunkte['Betriebspunkttyp_Bezeichnung']))
    oev_class_raw  = node_ids.map(bp_type_lookup)
    oev_class      = oev_class_raw.apply(
        lambda x: BETRIEBSPUNKT_TYPE_MAPPING.get(str(x), 'unclassified') if pd.notna(x) else 'unclassified'
    )

    unclassified_mask = nodes['node_class'] == 'unclassified'
    nodes.loc[unclassified_mask, 'node_class'] = oev_class[unclassified_mask]

    n_filled  = int((unclassified_mask & (nodes['node_class'] != 'unclassified')).sum())
    still_unc = int((nodes['node_class'] == 'unclassified').sum())
    print(f"  OeV match: {oev_class_raw.notna().sum()}/{len(nodes)} nodes found in HaltestellenOeV")
    print(f"  Classification: {n_filled} filled from OeV  |  {still_unc} still unclassified")

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

    # Enforce: only stations get a platform_count value
    nodes.loc[nodes['node_class'] != 'station', 'platform_count'] = pd.NA

    matched_plats = nodes['platform_count'].notna().sum()
    print(f"  Platform counts: {matched_plats} station nodes have platform data")

    # --- Add OeV-only nodes (B/C/E/F mode, not in BAV set, within boundary) --
    existing_nummers = set(nodes['Betriebspunkt_Nummer'].dropna().astype('Int64'))
    rail_mode_mask   = betriebspunkte['Verkehrsmittel_Code'].fillna('').apply(
        lambda c: bool(set(str(c)) & {'B', 'C', 'E', 'F'})
    )
    new_bp = betriebspunkte[rail_mode_mask].copy()
    new_bp['_Nummer_int'] = new_bp['Nummer'].astype('Int64')
    new_bp = new_bp[~new_bp['_Nummer_int'].isin(existing_nummers)].copy()

    boundary_geom = boundary.geometry.union_all()
    new_bp        = new_bp[new_bp.geometry.within(boundary_geom)].copy()

    if len(new_bp) > 0:
        def _plat_count(nummer_int):
            n = int(nummer_int) if pd.notna(nummer_int) else None
            return nummer_to_platforms.get(n, pd.NA) if n is not None else pd.NA

        oev_class_new = new_bp['Betriebspunkttyp_Bezeichnung'].apply(
            lambda x: BETRIEBSPUNKT_TYPE_MAPPING.get(str(x), 'unclassified') if pd.notna(x) else 'unclassified'
        )

        oev_nodes = gpd.GeoDataFrame({
            'node_id':              new_bp['_Nummer_int'].values,
            'Betriebspunkt_Nummer': new_bp['_Nummer_int'].values,
            'NAME':                 new_bp['Name'].values,
            'CODE':                 new_bp['Abkuerzung'].values,
            'E':                    new_bp.geometry.x.values,
            'N':                    new_bp.geometry.y.values,
            'parent_node':          new_bp['rUebergeordneteHaltestelle'].values,
            'node_class':           oev_class_new.values,
            'transport_mode':       new_bp['Verkehrsmittel_Bezeichnung'].apply(_translate_modes).values,
            'platform_count':       new_bp['_Nummer_int'].apply(_plat_count).values,
            'geometry':             new_bp.geometry.values,
        }, crs=nodes.crs)

        # Enforce: only stations get a platform_count value
        oev_nodes.loc[oev_nodes['node_class'] != 'station', 'platform_count'] = pd.NA

        nodes = pd.concat([nodes, oev_nodes], ignore_index=True)
        print(f"  Added {len(oev_nodes)} OeV-only nodes (not in BAV, within boundary)")
    else:
        print("  No new OeV-only nodes to add")

    return nodes


def add_track_count(
    nodes: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Add track_count column and enforce nullable-integer types on both track columns.

    track_count (stations only):
        max(platform_count, max num_tracks of adjacent filtered segments).
        Defaults to 0 when neither source has data.  Non-station nodes get pd.NA.

    Both platform_count and track_count are stored as Int64 (no decimal point;
    NA preserved for non-stations).

    Must be called while segments still carry from_node / to_node as BAV xtf_ids
    (i.e. before substitute_node_names).
    """
    print("Computing track counts...")
    nodes = nodes.copy()

    # Build node_id → max(num_tracks) from all adjacent filtered segments
    node_max_tracks: dict = {}
    for row in segments[['from_node', 'to_node', 'num_tracks']].itertuples(index=False):
        for nid in (row.from_node, row.to_node):
            if nid is None or (isinstance(nid, float) and pd.isna(nid)):
                continue
            node_max_tracks[nid] = max(node_max_tracks.get(nid, 0), int(row.num_tracks))

    def _track_count(row):
        if row.get('node_class') != 'station':
            return pd.NA
        seg_tracks  = node_max_tracks.get(row['node_id'], 0)
        plat_tracks = int(row['platform_count']) if pd.notna(row.get('platform_count')) else 0
        return max(seg_tracks, plat_tracks)

    nodes['track_count'] = nodes.apply(_track_count, axis=1)

    # Cast both columns to nullable integer (no decimal point, NA preserved)
    nodes['platform_count'] = nodes['platform_count'].astype('Int64')
    nodes['track_count']    = nodes['track_count'].astype('Int64')

    n_tracks = nodes['track_count'].notna().sum()
    print(f"  track_count set for {n_tracks} station nodes")
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

    # Replace TLMRegio-derived track_config / railway_type with the authoritative
    # values from the BAV segments (num_tracks, gauge), matched by segment_id.
    seg_lookup = (
        filtered_segments[['segment_id', 'num_tracks', 'gauge']]
        .drop_duplicates(subset='segment_id')
        .set_index('segment_id')
    )
    composition['num_tracks'] = composition['segment_id'].map(seg_lookup['num_tracks'])
    composition['gauge']      = composition['segment_id'].map(seg_lookup['gauge'])
    composition = composition.drop(columns=['track_config', 'railway_type'])

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
        if pd.notna(row.get('Betriebspunkt_Nummer')):
            lookups['by_bp_nummer'][row['Betriebspunkt_Nummer']] = nid
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

    # --- Filter ---------------------------------------------------------------
    print("\n--- Filtering to catchment ---")
    nodes, segments = filter_to_catchment(nodes, segments, boundary)

    # --- Phase-1 classification: BAV name patterns ----------------------------
    print("\n--- Classifying nodes (BAV patterns) ---")
    nodes = classify_nodes_bav(nodes)

    # --- Phase-2 enrichment: HaltestellenOeV ---------------------------------
    print("\n--- Enriching nodes (HaltestellenOeV) ---")
    nodes = enrich_nodes_oev(nodes, betriebspunkte, haltekanten, boundary)

    # --- Track counts ---------------------------------------------------------
    print("\n--- Computing track counts ---")
    nodes = add_track_count(nodes, segments)

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

    # --- Fetch OSM railway ways for speed enrichment --------------------------
    print("\n--- Fetching OSM railway ways ---")
    osm_path = output_dir / "osm_maxspeed_segments.gpkg"
    try:
        bbox     = _osm_bbox_wgs84(segments)
        osm_ways = fetch_osm_ways(bbox)
        osm_ways.to_file(osm_path, driver="GPKG")
        print(f"  osm_maxspeed_segments.gpkg → {osm_path}  ({len(osm_ways)} ways)")
    except Exception as e:
        print(f"  WARNING: OSM fetch failed ({e})")
        print(f"  osm_maxspeed_segments.gpkg not written — speed enrichment will be skipped in network builder.")

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
