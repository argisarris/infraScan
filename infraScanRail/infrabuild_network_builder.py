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
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter
from shapely.ops import linemerge
from shapely.geometry import MultiLineString, LineString
import math
import sys
import uuid
import zipfile

sys.path.insert(0, str(Path(__file__).parent))
import paths


# =============================================================================
# Color Schemes
# =============================================================================

GAUGE_COLORS = {
    1435: '#000000',  # black           – standard gauge
    1668: '#B8860B',  # dark goldenrod  – Iberian gauge
    1520: '#8B0000',  # dark red        – Russian/broad gauge
    1000: '#7B00D4',  # violet          – metre gauge
    900:  '#FFD700',  # gold
    800:  '#FF4500',  # orange-red
    750:  '#FF6347',  # tomato
    600:  '#FF69B4',  # hot pink
}
GAUGE_DEFAULT = '#808080'

# Category-based electrification colours (keyed by canonical class).
# Raw BAV Elektrifizierung strings are classified via _classify_electrification().
ELECTRIFICATION_COLORS = {
    'no_electrification': '#000000',  # black       – nicht elektrifiziert
    'dc':                 '#ADD8E6',  # light blue  – Gleichstrom
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

NODE_COLORS = {
    'station':       '#FF0000',
    'junction':      '#0000FF',
    'turning_loop':  '#FFA500',
    'junction_tram': '#00FF00',
    'halt':          '#FF69B4',
    'border':        '#808080',
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

# Node classes always removed from the macroscopic network
_MACRO_DROP_ALWAYS = {'freight_yard', 'operational_yard', 'depot', 'km_change', 'abandoned', 'turning_loop', 'service_point'}
# Node classes removed only when degree < 3
_MACRO_DROP_IF_LOW_DEGREE = {'switch'}


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

    _MACRO_KEEP = {'station', 'junction', 'turning_loop'}

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
    spacing = 0.40  # MM between adjacent track centre lines
    if num_tracks == 1:
        offsets, width = [0.0], '0.50'
    elif num_tracks == 2:
        offsets, width = [-spacing, spacing], '0.40'
    elif num_tracks == 3:
        offsets, width = [-spacing, 0.0, spacing], '0.50'
    else:  # 4+
        offsets = [(-1.5 + i) * spacing for i in range(4)]
        width = '0.40'

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
            <prop k="offset_map_unit_scale" v="3x:0,0,0,0,0,0"/>
            <prop k="capstyle" v="square"/>
            <prop k="joinstyle" v="bevel"/>
            <prop k="use_custom_dash" v="0"/>
            <prop k="width_map_unit_scale" v="3x:0,0,0,0,0,0"/>
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


def _nodes_maplayer_xml(layer_id: str, gpkg_relpath: str,
                         display_name: str) -> str:
    """<maplayer> XML for nodes: white circles, black outline, CODE labels."""
    marker_xml = (
        '    <renderer-v2 forceraster="0" symbollevels="0" type="singleSymbol"'
        ' enableorderby="0">\n'
        '      <symbols>\n'
        '        <symbol alpha="1" clip_to_extent="1" type="marker" name="0" force_rhr="0">\n'
        '          <layer pass="0" class="SimpleMarker" locked="0" enabled="1">\n'
        '            <prop k="angle" v="0"/>\n'
        '            <prop k="color" v="255,255,255,255"/>\n'
        '            <prop k="joinstyle" v="bevel"/>\n'
        '            <prop k="name" v="circle"/>\n'
        '            <prop k="offset" v="0,0"/>\n'
        '            <prop k="offset_unit" v="MM"/>\n'
        '            <prop k="outline_color" v="0,0,0,255"/>\n'
        '            <prop k="outline_style" v="solid"/>\n'
        '            <prop k="outline_width" v="0.25"/>\n'
        '            <prop k="outline_width_unit" v="MM"/>\n'
        '            <prop k="size" v="2.5"/>\n'
        '            <prop k="size_unit" v="MM"/>\n'
        '          </layer>\n'
        '        </symbol>\n'
        '      </symbols>\n'
        '      <rotation/>\n'
        '      <sizescale/>\n'
        '    </renderer-v2>'
    )
    labeling_xml = (
        '    <labeling type="simple">\n'
        '      <settings calloutType="simple">\n'
        '        <text-style fieldName="CODE" fontFamily="Arial" fontSize="8"\n'
        '                    fontWeight="0" fontItalic="0" fontUnderline="0"\n'
        '                    fontStrikeout="0" textColor="35,35,35,255"\n'
        '                    textOpacity="1" blendMode="0" namedStyle="Regular"\n'
        '                    isExpression="0" useSubstitutions="0"\n'
        '                    multilineHeight="1" fontCapitals="0"\n'
        '                    fontLetterSpacing="0" fontWordSpacing="0"\n'
        '                    fontSizeUnit="Point" textOrientation="horizontal"\n'
        '                    previewBkgrdColor="255,255,255,255"/>\n'
        '        <text-buffer bufferDraw="1" bufferSize="1"\n'
        '                     bufferColor="255,255,255,255" bufferOpacity="1"\n'
        '                     bufferJoinStyle="128" bufferNoFill="1"\n'
        '                     bufferSizeUnits="MM"\n'
        '                     bufferSizeMapUnitScale="3x:0,0,0,0,0,0"\n'
        '                     bufferBlendMode="0"/>\n'
        '        <background shapeDraw="0"/>\n'
        '        <shadow shadowDraw="0"/>\n'
        '        <placement placement="1" centroidWhole="0" placementFlags="10"\n'
        '                   priority="5" offsetType="0" quadOffset="4"\n'
        '                   xOffset="2" yOffset="2" offsetUnits="MM"\n'
        '                   dist="1" distInMapUnits="0"\n'
        '                   distMapUnitScale="3x:0,0,0,0,0,0"\n'
        '                   rotationAngle="0" geometryGenerator=""\n'
        '                   geometryGeneratorEnabled="0"\n'
        '                   geometryGeneratorType="PointGeometry"\n'
        '                   isExpression="0" labelPerPart="0"/>\n'
        '        <rendering drawLabels="1" obstacle="1" obstacleFactor="1"\n'
        '                   obstacleType="1" limitNumLabels="0" maxNumLabels="2000"\n'
        '                   minFeatureSize="0" fontMinPixelSize="3"\n'
        '                   fontMaxPixelSize="10000" displayAll="0"\n'
        '                   upsidedownLabels="0" mergeLines="0" zIndex="0"\n'
        '                   scaleVisibility="0" scaleMin="1"\n'
        '                   scaleMax="10000000" labelPerPart="0"/>\n'
        '        <dd_properties/>\n'
        '      </settings>\n'
        '    </labeling>'
    )
    return (
        f'  <maplayer geometry="Point" type="vector" hasScaleBasedVisibilityFlag="0">\n'
        f'    <id>{layer_id}</id>\n'
        f'    <datasource>{gpkg_relpath}|layername=nodes</datasource>\n'
        f'    <layername>{display_name}</layername>\n'
        f'    <provider encoding="UTF-8">ogr</provider>\n'
        f'    <subset>&quot;node_class&quot; = \'station\' AND &quot;transport_mode&quot; LIKE \'%train%\'</subset>\n'
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
    nod_id  = f'nod_{uuid.uuid4().hex[:8]}'
    ca_id   = f'ca_{uuid.uuid4().hex[:8]}'
    sa_id   = f'sa_{uuid.uuid4().hex[:8]}'

    # Relative paths from the version directory to boundary files
    ca_relpath = '../../Catchment_Area/catchment_area_boundary.gpkg'
    sa_relpath = '../../Catchment_Area/study_area_boundary.gpkg'

    seg_block = _segments_maplayer_xml(seg_id, 'segments.gpkg',
                                        f'Segments — {version}')
    nod_block = _nodes_maplayer_xml(nod_id, 'nodes.gpkg',
                                     f'Stations &amp; Halts — {version}')
    ca_block  = _boundary_maplayer_xml(ca_id, ca_relpath,
                                        'Catchment Area Boundary', '0,0,0,255')
    sa_block  = _boundary_maplayer_xml(sa_id, sa_relpath,
                                        'Study Area Boundary', '139,0,0,255')
    wms_block = _wms_maplayer_xml()

    tree_xml = (
        f'    <layer-tree-layer id="{nod_id}" name="Stations &amp; Halts — {version}" '
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
<qgis projectname="{version} Infrastructure" version="3.44.3-Solothurn">
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
{nod_block}
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
# Plotting Utilities
# =============================================================================

def _get_line_width(num_tracks: int) -> float:
    return 1.0 + (num_tracks - 1) * 0.5


def _add_north_arrow(ax, x=0.05, y=0.95, size=0.05):
    ax.annotate('N', xy=(x, y), xytext=(x, y - size),
                xycoords='axes fraction', ha='center', va='center',
                fontsize=12, fontweight='bold',
                arrowprops=dict(arrowstyle='->', lw=2))


def _add_scale_bar(ax, length_km=10, location=(0.72, 0.05)):
    length_m = length_km * 1000
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x0 = xlim[0] + (xlim[1] - xlim[0]) * location[0]
    y0 = ylim[0] + (ylim[1] - ylim[0]) * location[1]
    ax.plot([x0, x0 + length_m], [y0, y0], 'k-', linewidth=2)
    ax.plot([x0, x0], [y0 - 500, y0 + 500], 'k-', linewidth=1)
    ax.plot([x0 + length_m, x0 + length_m], [y0 - 500, y0 + 500], 'k-', linewidth=1)
    ax.text(x0 + length_m / 2, y0 + 800, f'{length_km} km', ha='center', fontsize=8)


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


def _plot_lakes(ax, boundary: Optional[gpd.GeoDataFrame] = None) -> None:
    """Overlay lakes from swissTLMRegio, optionally clipped to boundary."""
    lakes_path = Path(paths.MAIN) / paths.LAKES_SHP
    if not lakes_path.exists():
        return
    try:
        lakes = gpd.read_file(lakes_path)
        if lakes.crs is None:
            lakes = lakes.set_crs('EPSG:2056')
        elif lakes.crs.to_epsg() != 2056:
            lakes = lakes.to_crs('EPSG:2056')
        if boundary is not None and not boundary.empty:
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
) -> plt.Figure:
    """Infrastructure overview: segments coloured by track count, nodes by class."""
    if title is None:
        title = f"Infrastructure — {network.version}"
    fig, ax = _base_plot(network, title, figsize)
    _plot_lakes(ax, network.boundary)

    segs  = _clip_to_boundary(network.segments, network.boundary)
    nodes = _clip_to_boundary(network.nodes,    network.boundary)

    # Segments by track count
    if len(segs) > 0:
        for n_tracks, color in TRACK_COLORS.items():
            mask = segs['num_tracks'] == n_tracks
            if mask.any():
                segs[mask].plot(
                    ax=ax, color=color,
                    linewidth=_get_line_width(n_tracks), alpha=0.75, zorder=2,
                )
        other = ~segs['num_tracks'].isin(TRACK_COLORS)
        if other.any():
            segs[other].plot(
                ax=ax, color=TRACK_DEFAULT, linewidth=2, alpha=0.7, zorder=2,
            )

    # Nodes by class
    if len(nodes) > 0:
        for cls, color in NODE_COLORS.items():
            mask = nodes['node_class'] == cls
            if mask.any():
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
    legend.append(Line2D([0], [0], color='none', label='─ Tracks ─'))
    for n, color in TRACK_COLORS.items():
        legend.append(Line2D([0], [0], color=color,
                             linewidth=_get_line_width(n), label=f'{n} track(s)'))
    legend.append(Line2D([0], [0], color='none', label=''))
    legend.append(Line2D([0], [0], color='none', label='─ Nodes ─'))
    for cls, color in NODE_COLORS.items():
        marker = 'o' if cls == 'station' else 's'
        size   = 10 if cls == 'station' else 7
        legend.append(Line2D([0], [0], marker=marker, color='w',
                             markerfacecolor=color, markersize=size,
                             markeredgecolor='black' if cls == 'station' else None,
                             label=cls.replace('_', ' ').title()))
    ax.legend(handles=legend, loc='upper right', fontsize=8)

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
) -> plt.Figure:
    """OpenRailwayMap-style gauge map."""
    if title is None:
        title = f"Gauge Map — {network.version}"
    fig, ax = _base_plot(network, title, figsize, extent=extent)
    ax.set_facecolor('#f5f5f5')
    _plot_lakes(ax, network.boundary)

    segs     = _clip_to_boundary(network.segments, network.boundary)

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

    gauges_present = set()
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
    ax.legend(handles=legend, loc='upper right', fontsize=10,
              title='Track Gauge', title_fontsize=11)

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
) -> plt.Figure:
    """OpenRailwayMap-style electrification map (English legend, German-value-aware)."""
    if title is None:
        title = f"Electrification — {network.version}"
    fig, ax = _base_plot(network, title, figsize, extent=extent)
    ax.set_facecolor('#f5f5f5')
    _plot_lakes(ax, network.boundary)

    segs     = _clip_to_boundary(network.segments, network.boundary)

    cats_present = set()
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
    ax.legend(handles=legend, loc='upper right', fontsize=9,
              title='Electrification', title_fontsize=10)

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
) -> plt.Figure:
    """Tunnel / bridge / gallery construct type map."""
    if title is None:
        title = f"Construct Type — {network.version}"
    fig, ax = _base_plot(network, title, figsize, extent=extent)
    ax.set_facecolor('#f5f5f5')
    _plot_lakes(ax, network.boundary)

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
        ax.legend(handles=legend, loc='upper right', fontsize=10,
                  title='Construct Type', title_fontsize=11)

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

_TRACK_SPACING_M = 15   # metres between parallel track centre lines in map coords


def _draw_parallel_tracks(ax, geom, num_tracks: int,
                           color: str = 'black', linewidth: float = 0.7) -> None:
    """Draw num_tracks parallel offset lines for one segment geometry."""
    if num_tracks <= 0 or geom is None or geom.is_empty:
        return
    sub_lines = list(geom.geoms) if geom.geom_type == 'MultiLineString' else [geom]
    offsets = [(i - (num_tracks - 1) / 2) * _TRACK_SPACING_M
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
                    ax.plot(xs, ys, color=color, linewidth=linewidth,
                            solid_capstyle='butt', zorder=3)


def plot_infrastructure_canonical(
    network: NetworkData,
    extent=None,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12),
    show_labels: bool = True,
    is_catchment: bool = False,
) -> plt.Figure:
    """Infrastructure overview: black parallel tracks by count,
    white station circles with CODE labels."""
    if title is None:
        title = f"Infrastructure — {network.version}"
    fig, ax = _base_plot(network, title, figsize, extent=extent)
    _plot_lakes(ax, network.boundary)

    segs  = _clip_to_boundary(network.segments, network.boundary)
    nodes = _clip_to_boundary(network.nodes,    network.boundary)

    # Segments as black parallel lines (one line per track)
    if len(segs) > 0:
        for _, row in segs.iterrows():
            n = int(row.get('num_tracks', 1) or 1)
            n = max(1, min(n, 6))
            _draw_parallel_tracks(ax, row.geometry, n)

    # Stations: white fill, black outline
    if len(nodes) > 0:
        stations = nodes[nodes['node_class'] == 'station']
        if is_catchment:
            if 'transport_mode' in stations.columns:
                stations = stations[stations['transport_mode'].astype(str).str.contains('train', na=False)]
            markersize = 10
        else:
            markersize = 60

        if len(stations) > 0:
            stations.plot(ax=ax, facecolor='white', edgecolor='black',
                          markersize=markersize, marker='o', linewidth=1.2, zorder=5)
            if show_labels:
                for _, row in stations.iterrows():
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
        
        if not is_catchment:
            # Other nodes: small grey circles
            others = nodes[nodes['node_class'] != 'station']
            if len(others) > 0:
                others.plot(ax=ax, color='#888888', markersize=15, marker='o', zorder=4)

    legend = [
        Line2D([0], [0], color='black', linewidth=1.0, label='Single track'),
        Line2D([0], [0], color='black', linewidth=2.0,
               label='Multi-track (parallel lines)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='black', markersize=8, label='Station (CODE label)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#888888',
               markersize=5, label='Junction / other node'),
    ]
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
_PORTAL_BAR_M  = 30                   # perpendicular portal bar length (m)
_BRIDGE_OFFSET_M = 8                  # offset of bridge border lines from centre (m)


def _draw_geom(ax, geom, color, linewidth, linestyle='solid', zorder=3) -> None:
    """Draw a LineString or MultiLineString on ax."""
    if geom is None or geom.is_empty:
        return
    parts = list(geom.geoms) if geom.geom_type == 'MultiLineString' else [geom]
    for part in parts:
        if not part.is_empty and len(part.coords) >= 2:
            xs, ys = zip(*part.coords)
            ax.plot(xs, ys, color=color, linewidth=linewidth, linestyle=linestyle,
                    solid_capstyle='butt', zorder=zorder)


def _draw_tunnel_portals(ax, geom) -> None:
    """Draw perpendicular portal bars at both ends of a tunnel piece."""
    if geom is None or geom.is_empty:
        return
    if geom.geom_type == 'MultiLineString':
        all_coords = [c for sub in geom.geoms for c in sub.coords]
    else:
        all_coords = list(geom.coords)
    if len(all_coords) < 2:
        return

    for pt, ref in ((all_coords[0], all_coords[1]),
                    (all_coords[-1], all_coords[-2])):
        dx, dy = ref[0] - pt[0], ref[1] - pt[1]
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1e-6:
            continue
        px, py = -dy / length, dx / length          # perpendicular unit vector
        half = _PORTAL_BAR_M / 2
        ax.plot(
            [pt[0] - px * half, pt[0] + px * half],
            [pt[1] - py * half, pt[1] + py * half],
            color=_CONSTRUCT_PORTAL_COLOR, linewidth=2.5,
            solid_capstyle='butt', zorder=6,
        )


def plot_construction_components(
    network: NetworkData,
    composition: gpd.GeoDataFrame,
    extent=None,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12),
) -> plt.Figure:
    """Study-area construction components map (OpenRailwayMap-inspired).

    Normal  — solid orange track
    Bridge  — solid orange track + parallel black border lines on each side
    Tunnel  — dashed orange track + perpendicular portal bar at each end
    Gallery — same treatment as tunnel
    """
    if not composition.empty:
        composition = _clip_to_boundary(composition, network.boundary)

    if title is None:
        title = f"Construction Components — {network.version}"
    fig, ax = _base_plot(network, title, figsize, extent=extent)
    _plot_lakes(ax, network.boundary)

    legend_handles: list = []
    legend_seen:    set  = set()

    def _add_legend(label, color, lw, ls='solid'):
        if label not in legend_seen:
            legend_seen.add(label)
            legend_handles.append(
                Line2D([0], [0], color=color, linewidth=lw,
                       linestyle=ls, label=label)
            )

    if not composition.empty and 'construct_type' in composition.columns:
        for _, piece in composition.iterrows():
            geom  = piece.geometry
            if geom is None or geom.is_empty:
                continue
            ctype = str(piece.get('construct_type', 'normal')).lower()

            if ctype == 'normal':
                _draw_geom(ax, geom, _CONSTRUCT_TRACK_COLOR, 1.5, zorder=3)
                _add_legend('Normal track', _CONSTRUCT_TRACK_COLOR, 2)

            elif ctype == 'bridge':
                _draw_geom(ax, geom, _CONSTRUCT_TRACK_COLOR, 1.5, zorder=4)
                for side in ('left', 'right'):
                    try:
                        sub_lines = (list(geom.geoms)
                                     if geom.geom_type == 'MultiLineString'
                                     else [geom])
                        for sub in sub_lines:
                            border = sub.parallel_offset(
                                _BRIDGE_OFFSET_M, side,
                                resolution=8, join_style=2, mitre_limit=5,
                            )
                            if border is not None and not border.is_empty:
                                _draw_geom(ax, border, _CONSTRUCT_BRIDGE_RAIL,
                                           1.5, zorder=5)
                    except Exception:
                        pass
                _add_legend('Bridge (track)', _CONSTRUCT_TRACK_COLOR, 2)
                if 'Bridge (border)' not in legend_seen:
                    legend_seen.add('Bridge (border)')
                    legend_handles.append(
                        Line2D([0], [0], color=_CONSTRUCT_BRIDGE_RAIL,
                               linewidth=1.5, label='Bridge (border rail)')
                    )

            elif ctype in ('tunnel', 'gallery'):
                _draw_geom(ax, geom, _CONSTRUCT_TRACK_COLOR, 1.5,
                           linestyle='dashed', zorder=3)
                _draw_tunnel_portals(ax, geom)
                lbl = 'Tunnel' if ctype == 'tunnel' else 'Gallery'
                _add_legend(lbl, _CONSTRUCT_TRACK_COLOR, 2, ls='dashed')

            else:
                _draw_geom(ax, geom, '#888888', 1.2, zorder=2)

    else:
        if len(network.segments) > 0:
            network.segments.plot(ax=ax, color='#888888', linewidth=1.2, zorder=2)
        print("  Warning: no composition data — falling back to plain segment plot.")

    # Stations overlay
    if len(network.nodes) > 0:
        stations = network.nodes[network.nodes['node_class'] == 'station']
        if len(stations) > 0:
            stations.plot(ax=ax, facecolor='white', edgecolor='black',
                          markersize=60, marker='o', linewidth=1.2, zorder=7)
            for _, row in stations.iterrows():
                code = row.get('CODE', '')
                if pd.notna(code) and str(code).strip():
                    ax.annotate(
                        str(code),
                        xy=(row.geometry.x, row.geometry.y),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=7, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                  edgecolor='none', alpha=0.7),
                        zorder=8,
                    )
    legend_handles.append(
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='black', markersize=8, label='Station')
    )
    ax.legend(handles=legend_handles, loc='upper right', fontsize=8)
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
) -> plt.Figure:
    import matplotlib.cm as cm
    
    if title is None:
        title = f"Owner Map - {network.version}"
    fig, ax = _base_plot(network, title, figsize, extent=extent)
    _plot_lakes(ax, network.boundary)

    segs = _clip_to_boundary(network.segments, network.boundary)
    
    if len(segs) > 0 and 'route_owner' in segs.columns:
        unique_owners = segs['route_owner'].dropna().unique()
        
        cmap = plt.get_cmap('tab20')
        colors = {}
        idx = 0
        for o in unique_owners:
            if 'SBB' in str(o).upper():
                colors[o] = '#E3000F'
            else:
                colors[o] = cmap(idx % 20)
                idx += 1
            
        for owner, color in colors.items():
            mask = segs['route_owner'] == owner
            if mask.any():
                segs[mask].plot(ax=ax, color=color, linewidth=2, alpha=0.8, zorder=2)

        legend = [Line2D([0], [0], color=col, linewidth=2, label=str(own)) for own, col in colors.items()]
        if len(legend) <= 40:
            ax.legend(handles=legend, loc='upper right', fontsize=8, ncol=2)
            
    _add_north_arrow(ax)
    _add_scale_bar(ax)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        print(f"  Saved -> {output_path}")
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
    print("    Infrastructure · Gauge · Electrification")
    print("  Study area (extent = study_area_boundary):")
    print("    Infrastructure · Gauge · Electrification · Construction components")
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
        ('sa_infra',     'Study area — Infrastructure'),
        ('sa_gauge',     'Study area — Gauge'),
        ('sa_elec',      'Study area — Electrification'),
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
            'ca_infra': (plot_infrastructure_canonical, _net_ca, _ca_ext,
                         'ca_infrastructure.pdf'),
            'ca_gauge': (plot_gauge_map,                _net_ca, _ca_ext,
                         'ca_gauge.pdf'),
            'ca_elec':  (plot_electrification_map,      _net_ca, _ca_ext,
                         'ca_electrification.pdf'),
            'sa_infra': (plot_infrastructure_canonical, _net_sa, _sa_ext,
                         'sa_infrastructure.pdf'),
            'sa_gauge': (plot_gauge_map,                _net_sa, _sa_ext,
                         'sa_gauge.pdf'),
            'sa_elec':  (plot_electrification_map,      _net_sa, _sa_ext,
                         'sa_electrification.pdf'),
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
                )
                plt.close(_fig)
            else:
                _fn, _net, _ext, _fname = _plot_dispatch[_pk]
                print(f"  {_plot_label_map[_pk]} ...")
                
                kwargs = {}
                if _pk == 'ca_infra':
                    kwargs['show_labels'] = False
                
                _fig = _fn(_net, extent=_ext, output_path=_plot_dir / _fname, **kwargs)
                plt.close(_fig)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Version : {_chosen}")
    print(f"Nodes   : {G.number_of_nodes()}")
    print(f"Edges   : {G.number_of_edges()}")
    if _plot_set:
        print(f"Plots   : {_plot_dir}")
    print("=" * 60)
