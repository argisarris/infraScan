"""
Version Manager Module

Interactive TUI for creating named infrastructure versions derived from the
macroscopic Base network, or adjusting existing versions.

Workflow
--------
  Phase 0: Select or create a version (folder creation + copy for new versions).
  Phase 1: Node editing loop — remove / adjust / add.
  Phase 2: Segment editing loop — remove / adjust / add.
  Phase 3: Save all three geopackages to disk.

Usage (interactive):
    python infrabuild_version_manager.py
"""

import os
import shutil
import geopandas as gpd
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple
from shapely.geometry import LineString, Point
from shapely.ops import substring, linemerge
import sys

sys.path.insert(0, str(Path(__file__).parent))
import paths


# =============================================================================
# Constants
# =============================================================================

SWISS_CRS = "EPSG:2056"
MERGE_ATTRS = ('num_tracks', 'gauge', 'electrification')

GAUGE_OPTIONS = {
    '1': 1435, '2': 1000, '3': 900, '4': 800, '5': 750, '6': 600,
}

ELECTRIFICATION_OPTIONS = {
    '1': 'AC 15kV 16.7Hz', '2': 'DC 1500V',  '3': 'DC 1000V',
    '4': 'DC 750V',        '5': 'DC 600V',   '6': 'nicht elektrifiziert',
}

NODE_CLASS_OPTIONS = {
    '1': 'station', '2': 'junction', '3': 'abandoned_station',
}

CONSTRUCT_TYPE_OPTIONS = {
    '1': 'bridge', '2': 'normal', '3': 'tunnel',
}


def _prompt_composition_pieces(
    seg_id: str,
    from_name: str,
    to_name: str,
    seg_geom,
    seg_len: float,
) -> gpd.GeoDataFrame:
    """
    Interactively collect one or more composition pieces for a segment and
    return a GeoDataFrame of those pieces (NOT appended to composition yet).
    """
    ct_options = CONSTRUCT_TYPE_OPTIONS

    print(f"\n  Segment length: {seg_len:.0f} m")
    while True:
        multi_raw = input("  Multiple composition pieces? (y/n) [n]: ").strip().lower() or 'n'
        if multi_raw in ('y', 'n'):
            break
        print("  Enter y or n.")
    multi = multi_raw == 'y'

    pieces = []
    remaining = seg_len

    while True:
        print(f"\n  Construct type (remaining: {remaining:.0f} m):")
        for k, v in ct_options.items():
            print(f"    {k}) {v}")
        raw = input("  Select number or type custom value: ").strip()
        ct = ct_options.get(raw, raw) if raw else 'normal'

        if multi:
            try:
                pl = float(input(f"  Piece length (m) [remaining={remaining:.0f}]: ").strip())
            except ValueError:
                print("  Invalid length. Skipping.")
                continue
        else:
            pl = seg_len

        el_raw = input("  Edge level [1]: ").strip() or '1'
        try:
            el = int(el_raw)
        except ValueError:
            el = 1

        uc_raw = input("  Under construction (0/1) [0]: ").strip() or '0'
        uc = 1 if uc_raw == '1' else 0

        tc_raw = input("  Track config (Enter for none): ").strip()
        rt_raw = input("  Railway type  (Enter for none): ").strip()

        pieces.append({
            'segment_id':         seg_id,
            'from_name':          from_name,
            'to_name':            to_name,
            'construct_type':     ct,
            'edge_level':         el,
            'under_construction': uc,
            'track_config':       tc_raw if tc_raw else pd.NA,
            'railway_type':       rt_raw if rt_raw else pd.NA,
            'piece_length_m':     pl,
            'geometry':           seg_geom,
        })
        remaining -= pl
        print(f"  Piece added: {ct}  {pl:.0f} m.")

        if not multi:
            break
        if remaining <= 0:
            print("  Total length reached.")
            break
        add_more = input("  Add another piece? (y/n) [y]: ").strip().lower() or 'y'
        if add_more != 'y':
            break

    return gpd.GeoDataFrame(pieces, crs=SWISS_CRS)


# =============================================================================
# Version Discovery
# =============================================================================

def list_versions(infra_dir: Optional[str] = None) -> List[str]:
    """Return selectable version names (excludes Raw/)."""
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
    if 'Base' in versions:
        versions = ['Base'] + [v for v in versions if v != 'Base']
    return versions


# =============================================================================
# Search / pick helpers
# =============================================================================

def _search_nodes(nodes: gpd.GeoDataFrame, term: str) -> gpd.GeoDataFrame:
    """Return rows whose NAME or CODE contain term (case-insensitive)."""
    term_l = term.lower()
    mask = (
        nodes['NAME'].fillna('').str.lower().str.contains(term_l, regex=False) |
        nodes['CODE'].fillna('').str.lower().str.contains(term_l, regex=False)
    )
    return nodes[mask]


def _search_segments(segments: gpd.GeoDataFrame, term: str) -> gpd.GeoDataFrame:
    """Return rows whose from_name, to_name, or segment_name contain term (case-insensitive)."""
    term_l = term.lower()
    mask = (
        segments['from_name'].fillna('').str.lower().str.contains(term_l, regex=False) |
        segments['to_name'].fillna('').str.lower().str.contains(term_l, regex=False) |
        segments['segment_name'].fillna('').str.lower().str.contains(term_l, regex=False)
    )
    return segments[mask]


def _pick_one(labels: List[str], prompt: str = "Select") -> Optional[int]:
    """
    Display a numbered list and return the 0-based index of the chosen item.
    Returns None if the user presses Enter without a number.
    """
    for i, lbl in enumerate(labels, 1):
        print(f"     {i}) {lbl}")
    while True:
        raw = input(f"   {prompt} (number): ").strip()
        if not raw:
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(labels):
            return int(raw) - 1
        print(f"   Invalid — enter 1–{len(labels)} or press Enter to cancel.")


# =============================================================================
# Phase 0 — Version selection
# =============================================================================

def _run_phase0() -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame,
                            str, str, Path, str]:
    """
    Phase 0 — version selection.

    Returns
    -------
    nodes, segments, composition, version_name, source_version, out_dir, mode
    mode is 'new' or 'adjust'.
    """
    infra_root = Path(paths.MAIN) / paths.NETWORK_INFRASTRUCTURE_DIR

    # Prerequisite: Base version must exist with all three geopackages
    base_dir = infra_root / 'Base'
    required = ['nodes.gpkg', 'segments.gpkg', 'segments_composition.gpkg']
    missing  = [f for f in required if not (base_dir / f).exists()]
    if missing:
        print("\n  ERROR: Base version is incomplete or missing.")
        print(f"  Expected at: {base_dir}")
        for f in missing:
            print(f"    missing: {f}")
        print("  Run infrabuild_network_builder.py to build Base first.")
        raise SystemExit(1)

    # Q1 — what do you want to do?
    print("\n" + "─" * 60)
    print("  What do you want to do?")
    print("    1) Create a new version")
    print("    2) Adjust an existing version")
    while True:
        choice = input("  Select (1/2): ").strip()
        if choice in ('1', '2'):
            break
        print("  Enter 1 or 2.")

    versions = list_versions()
    if not versions:
        print(f"\n  No versions found in {infra_root}.")
        print("  Run infrabuild_network_builder.py to build Base first.")
        raise SystemExit(1)

    if choice == '1':
        mode = 'new'

        # Q2 — choose base version
        print("\n  Choose base version:")
        base_labels = [f"{v}  [base]" if v == 'Base' else v for v in versions]
        idx = _pick_one(base_labels, "Base version")
        if idx is None:
            raise SystemExit(0)
        source_version = versions[idx]

        # Q3 — name for the new version
        while True:
            name = input("\n  Name for the new version (e.g. AK_2035): ").strip()
            if not name:
                print("  Name cannot be empty.")
                continue
            if name in ('Raw', 'Base'):
                print(f"  '{name}' is a reserved name. Choose another.")
                continue
            out_dir = infra_root / name
            if out_dir.exists() and (out_dir / 'nodes.gpkg').exists():
                print(f"  Version '{name}' already exists.")
                overwrite = input("  Overwrite? (y/n) [n]: ").strip().lower() or 'n'
                if overwrite != 'y':
                    continue
            break

        # Create folder and copy geopackages from source version
        source_dir = infra_root / source_version
        out_dir.mkdir(parents=True, exist_ok=True)
        for gpkg in ('nodes.gpkg', 'segments.gpkg', 'segments_composition.gpkg'):
            shutil.copy2(source_dir / gpkg, out_dir / gpkg)
        print(f"  Copied {source_version}/ → {name}/")

    else:  # choice == '2'
        mode = 'adjust'
        # list_versions already excludes Raw
        print("\n  Choose version to adjust:")
        idx = _pick_one(versions, "Version")
        if idx is None:
            raise SystemExit(0)
        name = versions[idx]
        source_version = name
        out_dir = infra_root / name

    # Load the three geopackages
    print(f"\n  Loading '{name}'...")
    nodes       = gpd.read_file(out_dir / 'nodes.gpkg').reset_index(drop=True)
    segments    = gpd.read_file(out_dir / 'segments.gpkg').reset_index(drop=True)
    composition = gpd.read_file(out_dir / 'segments_composition.gpkg').reset_index(drop=True)
    print(f"  {len(nodes)} nodes, {len(segments)} segments, "
          f"{len(composition)} composition pieces loaded.")

    return nodes, segments, composition, name, source_version, out_dir, mode


# =============================================================================
# Node operations
# =============================================================================

def _adjust_node(nodes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Inline CSV-style editing of a single node's attributes."""
    term = input("  Search node (CODE or partial NAME): ").strip()
    if not term:
        return nodes
    hits = _search_nodes(nodes, term)
    if hits.empty:
        print(f"  No nodes matching '{term}'.")
        return nodes

    labels = [f"{r['NAME']}  [{r['CODE']}]" for _, r in hits.iterrows()]
    idx = _pick_one(labels, "Node to adjust")
    if idx is None:
        return nodes

    row_idx = hits.index[idx]
    row = nodes.loc[row_idx]

    COLS = ['NAME', 'CODE', 'E', 'N', 'node_class', 'transport_mode', 'platform_count']
    header  = ", ".join(COLS)
    current = ", ".join(str(row.get(c, '')) for c in COLS)
    print(f"\n  {header}")
    print(f"  {current}")
    updated = input("  Enter updated values (same order; press Enter to keep): ").strip()
    if not updated:
        return nodes

    parts = [p.strip() for p in updated.split(',')]
    if len(parts) != len(COLS):
        print(f"  Expected {len(COLS)} values, got {len(parts)}. No changes made.")
        return nodes

    for col, val in zip(COLS, parts):
        if val == '':
            continue
        if col in ('E', 'N'):
            try:
                nodes.at[row_idx, col] = float(val)
            except ValueError:
                print(f"  Could not parse '{val}' as float for {col}. Skipping.")
        elif col == 'platform_count':
            try:
                nodes.at[row_idx, col] = int(val)
            except ValueError:
                nodes.at[row_idx, col] = val
        else:
            nodes.at[row_idx, col] = val

    # Always regenerate geometry from (possibly updated) E, N
    E = nodes.at[row_idx, 'E']
    N = nodes.at[row_idx, 'N']
    nodes.at[row_idx, 'geometry'] = Point(float(E), float(N))
    print(f"  Node '{nodes.at[row_idx, 'NAME']}' updated.")
    return nodes


def _remove_node(
    nodes: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
    composition: gpd.GeoDataFrame,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Remove a node with optional segment merge."""
    term = input("  Search node (CODE or partial NAME): ").strip()
    if not term:
        return nodes, segments, composition
    hits = _search_nodes(nodes, term)
    if hits.empty:
        print(f"  No nodes matching '{term}'.")
        return nodes, segments, composition

    labels = [f"{r['NAME']}  [{r['CODE']}]" for _, r in hits.iterrows()]
    idx = _pick_one(labels, "Node to remove")
    if idx is None:
        return nodes, segments, composition

    row_idx   = hits.index[idx]
    node_row  = nodes.loc[row_idx]
    node_name = node_row['NAME']

    # Print node attributes
    print(f"\n  Node: {node_name}")
    for col in ['CODE', 'E', 'N', 'node_class', 'transport_mode', 'platform_count']:
        print(f"    {col}: {node_row.get(col, '')}")

    # Connected segments
    conn_mask = (segments['from_name'] == node_name) | (segments['to_name'] == node_name)
    conn_segs = segments[conn_mask]
    print(f"\n  Connected segments ({len(conn_segs)}):")
    for _, s in conn_segs.iterrows():
        print(f"    {s['from_name']} → {s['to_name']}  [{s['segment_id']}]")

    # Merge check: exactly 2 connected segs with identical MERGE_ATTRS
    merged = False
    if len(conn_segs) == 2:
        s0, s1 = conn_segs.iloc[0], conn_segs.iloc[1]
        if all(s0[a] == s1[a] for a in MERGE_ATTRS):
            ans = input(
                "\n  Merge the two adjacent segments into one? (y/n) [n]: "
            ).strip().lower() or 'n'
            if ans == 'y':
                # Surviving outer endpoints
                all_ends = {s0['from_name'], s0['to_name'],
                            s1['from_name'], s1['to_name']}
                surviving = [e for e in all_ends if e != node_name]
                from_end, to_end = surviving[0], surviving[1]

                fe_code_rows = nodes[nodes['NAME'] == from_end]
                te_code_rows = nodes[nodes['NAME'] == to_end]
                fe_code = fe_code_rows.iloc[0]['CODE'] if not fe_code_rows.empty else from_end
                te_code = te_code_rows.iloc[0]['CODE'] if not te_code_rows.empty else to_end
                new_seg_id  = f"c{fe_code}_{te_code}"
                new_geom    = linemerge([s0.geometry, s1.geometry])
                new_length  = new_geom.length
                new_km_start = min(s0['km_start'], s1['km_start'])
                new_km_end   = max(s0['km_end'],   s1['km_end'])

                fn_row = nodes[nodes['NAME'] == from_end]
                tn_row = nodes[nodes['NAME'] == to_end]

                new_row = {
                    'segment_id':      new_seg_id,
                    'segment_name':    new_seg_id,
                    'from_name':       from_end,
                    'to_name':         to_end,
                    'from_N': fn_row.iloc[0]['N'] if not fn_row.empty else pd.NA,
                    'from_E': fn_row.iloc[0]['E'] if not fn_row.empty else pd.NA,
                    'to_N':   tn_row.iloc[0]['N'] if not tn_row.empty else pd.NA,
                    'to_E':   tn_row.iloc[0]['E'] if not tn_row.empty else pd.NA,
                    'length_m':        new_length,
                    'num_tracks':      s0['num_tracks'],
                    'gauge':           s0['gauge'],
                    'electrification': s0['electrification'],
                    'km_start':        new_km_start,
                    'km_end':          new_km_end,
                    'route_number':    s0['route_number'],
                    'route_name':      s0['route_name'],
                    'route_owner':     s0['route_owner'],
                    'geometry':        new_geom,
                }

                # Remove the two old segments
                old_ids = conn_segs['segment_id'].tolist()
                segments = segments[
                    ~segments['segment_id'].isin(old_ids)
                ].reset_index(drop=True)
                segments = pd.concat(
                    [segments, gpd.GeoDataFrame([new_row], crs=SWISS_CRS)],
                    ignore_index=True
                )

                # Replace composition: remove 2 old rows, add 1 normal row
                composition = composition[
                    ~composition['segment_id'].isin(old_ids)
                ].reset_index(drop=True)
                comp_row = gpd.GeoDataFrame([{
                    'segment_id':         new_seg_id,
                    'from_name':          from_end,
                    'to_name':            to_end,
                    'construct_type':     'normal',
                    'edge_level':         1,
                    'under_construction': 0,
                    'track_config':       pd.NA,
                    'railway_type':       pd.NA,
                    'piece_length_m':     new_length,
                    'geometry':           new_geom,
                }], crs=SWISS_CRS)
                composition = pd.concat([composition, comp_row], ignore_index=True)

                # Remove node
                nodes = nodes[nodes.index != row_idx].reset_index(drop=True)
                print(f"  Merged → '{new_seg_id}' ({new_length / 1000:.2f} km).")
                merged = True

    if not merged:
        n_conn = len(conn_segs)
        print(f"\n  How should connected segments be handled?")
        print(f"    1) Remove node only — leave {n_conn} connected segment(s) as-is")
        print(f"    2) Remove node and all {n_conn} connected segment(s)")
        while True:
            c = input("  Select (1/2): ").strip()
            if c in ('1', '2'):
                break
            print("  Enter 1 or 2.")

        # Remove node
        nodes = nodes[nodes.index != row_idx].reset_index(drop=True)

        if c == '2':
            remove_ids = conn_segs['segment_id'].tolist()
            segments = segments[
                ~segments['segment_id'].isin(remove_ids)
            ].reset_index(drop=True)
            composition = composition[
                ~composition['segment_id'].isin(remove_ids)
            ].reset_index(drop=True)
            print(f"  Removed node '{node_name}' and {len(remove_ids)} segment(s).")
        else:
            print(f"  Removed node '{node_name}'. Connected segments left as-is.")

    return nodes, segments, composition


# =============================================================================
# Segment split helper (used by Add Node)
# =============================================================================

def _split_segment_at(
    segments: gpd.GeoDataFrame,
    composition: gpd.GeoDataFrame,
    seg_idx: int,
    split_dist: float,
    new_node_name: str,
    new_node_code: str,
    nodes: gpd.GeoDataFrame,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Split segments.loc[seg_idx] at split_dist metres along its geometry.
    Produces S_A (before split) and S_B (after split).
    Redistributes composition pieces between A and B by cumulative length.
    Returns updated (segments, composition).
    """
    S = segments.loc[seg_idx]
    
    geom = S.geometry
    if geom.geom_type == 'MultiLineString':
        geom = linemerge(geom)
        
    seg_len    = geom.length
    split_dist = min(max(split_dist, 0.0), seg_len)  # clamp to valid range

    geom_A = substring(geom, 0, split_dist)
    geom_B = substring(geom, split_dist, seg_len)

    t      = split_dist / seg_len if seg_len > 0 else 0.0
    km_mid = S['km_start'] + t * (S['km_end'] - S['km_start'])

    split_pt = geom.interpolate(split_dist)

    from_code_rows = nodes[nodes['NAME'] == S['from_name']]
    to_code_rows   = nodes[nodes['NAME'] == S['to_name']]
    from_code = from_code_rows.iloc[0]['CODE'] if not from_code_rows.empty else S['from_name']
    to_code   = to_code_rows.iloc[0]['CODE']   if not to_code_rows.empty   else S['to_name']

    id_A = f"c{from_code}_{new_node_code}"
    id_B = f"c{new_node_code}_{to_code}"

    row_A = {
        'segment_id':      id_A,
        'segment_name':    id_A,
        'from_name':       S['from_name'],
        'to_name':         new_node_name,
        'from_N':          S.get('from_N', pd.NA),
        'from_E':          S.get('from_E', pd.NA),
        'to_N':            split_pt.y,
        'to_E':            split_pt.x,
        'length_m':        geom_A.length,
        'num_tracks':      S['num_tracks'],
        'gauge':           S['gauge'],
        'electrification': S['electrification'],
        'km_start':        S['km_start'],
        'km_end':          km_mid,
        'route_number':    S['route_number'],
        'route_name':      S['route_name'],
        'route_owner':     S['route_owner'],
        'geometry':        geom_A,
    }
    row_B = {
        'segment_id':      id_B,
        'segment_name':    id_B,
        'from_name':       new_node_name,
        'to_name':         S['to_name'],
        'from_N':          split_pt.y,
        'from_E':          split_pt.x,
        'to_N':            S.get('to_N', pd.NA),
        'to_E':            S.get('to_E', pd.NA),
        'length_m':        geom_B.length,
        'num_tracks':      S['num_tracks'],
        'gauge':           S['gauge'],
        'electrification': S['electrification'],
        'km_start':        km_mid,
        'km_end':          S['km_end'],
        'route_number':    S['route_number'],
        'route_name':      S['route_name'],
        'route_owner':     S['route_owner'],
        'geometry':        geom_B,
    }

    # Remove original segment, add A and B
    segments = segments[segments.index != seg_idx].reset_index(drop=True)
    segments = pd.concat(
        [segments, gpd.GeoDataFrame([row_A, row_B], crs=SWISS_CRS)],
        ignore_index=True
    )

    # --- Redistribute composition pieces ---
    old_comp    = composition[composition['segment_id'] == S['segment_id']].copy()
    composition = composition[
        composition['segment_id'] != S['segment_id']
    ].reset_index(drop=True)

    new_comp_rows = []
    cumulative = 0.0
    for _, piece in old_comp.iterrows():
        piece_len   = float(piece['piece_length_m'])
        piece_start = cumulative
        piece_end   = cumulative + piece_len

        base = piece.to_dict()
        base.pop('geometry', None)  # geometry will be set per-segment below

        if piece_end <= split_dist:
            # Entirely in A
            new_comp_rows.append({
                **base,
                'segment_id': id_A,
                'from_name':  S['from_name'],
                'to_name':    new_node_name,
                '_geom':      geom_A,
            })
        elif piece_start >= split_dist:
            # Entirely in B
            new_comp_rows.append({
                **base,
                'segment_id': id_B,
                'from_name':  new_node_name,
                'to_name':    S['to_name'],
                '_geom':      geom_B,
            })
        else:
            # Straddles split — divide into two rows
            len_in_A = split_dist - piece_start
            len_in_B = piece_end  - split_dist
            new_comp_rows.append({
                **base,
                'segment_id':    id_A,
                'from_name':     S['from_name'],
                'to_name':       new_node_name,
                'piece_length_m': len_in_A,
                '_geom':         geom_A,
            })
            new_comp_rows.append({
                **base,
                'segment_id':    id_B,
                'from_name':     new_node_name,
                'to_name':       S['to_name'],
                'piece_length_m': len_in_B,
                '_geom':         geom_B,
            })

        cumulative += piece_len

    if new_comp_rows:
        geoms = [r.pop('_geom') for r in new_comp_rows]
        new_comp_gdf = gpd.GeoDataFrame(new_comp_rows, geometry=geoms, crs=SWISS_CRS)
        composition  = pd.concat([composition, new_comp_gdf], ignore_index=True)

    return segments, composition


def _add_node(
    nodes: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
    composition: gpd.GeoDataFrame,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Add a node by route+km or by coordinates, splitting the nearest segment."""
    print("\n  How do you want to locate the new node?")
    print("    1) By route number and kilometre position")
    print("    2) By exact coordinates (E, N)")
    while True:
        c = input("  Select (1/2): ").strip()
        if c in ('1', '2'):
            break
        print("  Enter 1 or 2.")

    seg_idx    = None
    split_dist = None
    node_E = node_N = None

    if c == '1':
        route = input("  Route number: ").strip()
        route_segs = segments[
            segments['route_number'].fillna('').astype(str) == route
        ]
        if route_segs.empty:
            print(f"  No segments found for route '{route}'.")
            return nodes, segments, composition

        while True:
            try:
                km = float(input("  Kilometre position: ").strip())
            except ValueError:
                print("  Enter a numeric km value.")
                continue
            match = route_segs[
                (route_segs['km_start'].fillna(-1) <= km) &
                (route_segs['km_end'].fillna(-1)   >= km)
            ]
            if match.empty:
                print(f"  No segment covers km {km} on route '{route}'. Try again.")
                continue
            seg_idx = match.index[0]
            S = segments.loc[seg_idx]
            t          = (km - S['km_start']) / (S['km_end'] - S['km_start'])
            split_dist = t * S.geometry.length
            split_pt   = S.geometry.interpolate(split_dist)
            node_E, node_N = split_pt.x, split_pt.y
            break

    else:  # by coordinates
        try:
            node_E = float(input("  Easting  E (m, EPSG:2056): ").strip())
            node_N = float(input("  Northing N (m, EPSG:2056): ").strip())
        except ValueError:
            print("  Invalid coordinates.")
            return nodes, segments, composition
        pt         = Point(node_E, node_N)
        seg_idx    = segments.geometry.distance(pt).idxmin()
        S          = segments.loc[seg_idx]
        split_dist = S.geometry.project(pt)
        split_pt   = S.geometry.interpolate(split_dist)
        node_E, node_N = split_pt.x, split_pt.y

    # Prompt node attributes
    print(f"\n  New node will be at E={node_E:.1f}, N={node_N:.1f}")
    name = input("  NAME: ").strip()
    if not name:
        print("  Name cannot be empty. Cancelled.")
        return nodes, segments, composition
    if name in nodes['NAME'].values:
        print(f"  A node named '{name}' already exists. Cancelled.")
        return nodes, segments, composition

    code = input(f"  CODE [{name[:4].upper()}]: ").strip() or name[:4].upper()

    print("  Node class:")
    for k, v in NODE_CLASS_OPTIONS.items():
        print(f"    {k}) {v}")
    while True:
        nc = input("  Select (1–6) [1]: ").strip() or '1'
        if nc in NODE_CLASS_OPTIONS:
            node_class = NODE_CLASS_OPTIONS[nc]
            break
        print("  Invalid — enter 1–6.")

    # Generate a synthetic Betriebspunkt_Nummer above the BAV range (max + 1,
    # floored at 9_000_000 so synthetic nodes are clearly distinguishable).
    existing_ids = nodes['Betriebspunkt_Nummer'].dropna()
    try:
        max_existing = int(existing_ids.astype(float).max())
    except (ValueError, TypeError):
        max_existing = 0
    synthetic_bpnr = max(max_existing + 1, 9_000_000)
    # Ensure uniqueness in the unlikely case of multiple additions in one session
    while synthetic_bpnr in existing_ids.astype(float).values:
        synthetic_bpnr += 1

    synthetic_node_id = f"synth_{synthetic_bpnr}"

    new_node = {
        'node_id':              synthetic_node_id,
        'Betriebspunkt_Nummer': synthetic_bpnr,
        'NAME':                 name,
        'CODE':                 code,
        'E':                    node_E,
        'N':                    node_N,
        'node_class':           node_class,
        'transport_mode':       None,
        'platform_count':       None,
        'track_count':          None,
        'parent_node':          None,
        'geometry':             Point(node_E, node_N),
    }
    print(f"  Assigned synthetic Betriebspunkt_Nummer: {synthetic_bpnr}")

    # Apply segment split
    segments, composition = _split_segment_at(
        segments, composition, seg_idx, split_dist, name, code, nodes
    )

    # Append new node
    nodes = pd.concat(
        [nodes, gpd.GeoDataFrame([new_node], crs=SWISS_CRS)],
        ignore_index=True
    )
    print(f"  Added node '{name}' ({node_class}) and split segment into _A / _B.")
    return nodes, segments, composition

# =============================================================================
# Segment operations
# =============================================================================

def _remove_segment(
    segments: gpd.GeoDataFrame,
    composition: gpd.GeoDataFrame,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Phase 2 — Remove a segment."""
    term = input("  Search segment (partial from-name, to-name, or segment_name): ").strip()
    if not term:
        return segments, composition
    hits = _search_segments(segments, term)
    if hits.empty:
        print(f"  No segments matching '{term}'.")
        return segments, composition

    labels = [
        f"{r['from_name']} → {r['to_name']}  [{r['segment_id']}]"
        + (f"  ({r['segment_name']})" if pd.notna(r.get('segment_name')) and r.get('segment_name') else "")
        for _, r in hits.iterrows()
    ]
    idx = _pick_one(labels, "Segment to remove")
    if idx is None:
        return segments, composition

    row_idx = hits.index[idx]
    S = segments.loc[row_idx]
    
    ans = input(f"  Remove segment '{S['from_name']} → {S['to_name']}'? (y/n) [n]: ").strip().lower() or 'n'
    if ans == 'y':
        segments = segments[segments.index != row_idx].reset_index(drop=True)
        composition = composition[composition['segment_id'] != S['segment_id']].reset_index(drop=True)
        print(f"  Removed segment '{S['segment_id']}'.")
    return segments, composition


def _adjust_segment(
    segments: gpd.GeoDataFrame,
    composition: gpd.GeoDataFrame,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Phase 2 — Adjust a segment's attributes and optionally redefine its composition."""
    term = input("  Search segment (partial from-name, to-name, or segment_name): ").strip()
    if not term:
        return segments, composition
    hits = _search_segments(segments, term)
    if hits.empty:
        print(f"  No segments matching '{term}'.")
        return segments, composition

    labels = [
        f"{r['from_name']} → {r['to_name']}  [{r['segment_id']}]"
        + (f"  ({r['segment_name']})" if pd.notna(r.get('segment_name')) and r.get('segment_name') else "")
        for _, r in hits.iterrows()
    ]
    idx = _pick_one(labels, "Segment to adjust")
    if idx is None:
        return segments, composition

    row_idx = hits.index[idx]
    row = segments.loc[row_idx]

    COLS = [
        'from_name', 'to_name', 'length_m', 'num_tracks', 'gauge', 'electrification',
        'km_start', 'km_end', 'route_number', 'route_name', 'route_owner'
    ]
    header  = ", ".join(COLS)
    current = ", ".join(str(row.get(c, '')) for c in COLS)
    print(f"\n  {header}")
    print(f"  {current}")

    pieces = composition[composition['segment_id'] == row['segment_id']]
    comp_strs = [f"{p['construct_type']} {float(p['piece_length_m']):.0f}m" for _, p in pieces.iterrows()]
    print(f"  Composition: {len(pieces)} pieces — {' / '.join(comp_strs)}")

    updated = input("  Enter updated values (same order; press Enter to keep): ").strip()
    if updated:
        parts = [p.strip() for p in updated.split(',')]
        if len(parts) != len(COLS):
            print(f"  Expected {len(COLS)} values, got {len(parts)}. No changes made.")
            return segments, composition

        EDITABLE = [
            'num_tracks', 'gauge', 'electrification',
            'km_start', 'km_end', 'route_number', 'route_name', 'route_owner'
        ]
        for col, val in zip(COLS, parts):
            if val == '' or col not in EDITABLE:
                continue
            if col in ('km_start', 'km_end'):
                try:
                    segments.at[row_idx, col] = float(val)
                except ValueError:
                    pass
            elif col in ('num_tracks', 'gauge'):
                try:
                    segments.at[row_idx, col] = int(val)
                except ValueError:
                    segments.at[row_idx, col] = val
            else:
                segments.at[row_idx, col] = val

        print(f"  Segment '{row['segment_id']}' attributes updated.")

    # Composition update
    recomp = input("\n  Redefine composition for this segment? (y/n) [n]: ").strip().lower() or 'n'
    if recomp == 'y':
        seg_geom = segments.loc[row_idx].geometry
        seg_len  = float(segments.loc[row_idx].get('length_m', seg_geom.length))
        # Drop existing composition pieces for this segment
        composition = composition[
            composition['segment_id'] != row['segment_id']
        ].reset_index(drop=True)
        new_comp_gdf = _prompt_composition_pieces(
            row['segment_id'],
            segments.loc[row_idx, 'from_name'],
            segments.loc[row_idx, 'to_name'],
            seg_geom, seg_len,
        )
        composition = pd.concat([composition, new_comp_gdf], ignore_index=True)
        print(f"  Composition updated: {len(new_comp_gdf)} piece(s).")

    return segments, composition


def _add_segment(
    nodes: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
    composition: gpd.GeoDataFrame,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Phase 2 — Add a segment."""
    term_from = input("  FROM node (CODE or partial NAME): ").strip()
    hits_f = _search_nodes(nodes, term_from) if term_from else gpd.GeoDataFrame()
    if hits_f.empty:
        print("  Cancel.")
        return segments, composition
    from_labels = [f"{r['NAME']}  [{r['CODE']}]" for _, r in hits_f.iterrows()]
    idx_f = _pick_one(from_labels, "FROM node")
    if idx_f is None: return segments, composition
    f_node = hits_f.iloc[idx_f]

    term_to = input("  TO node (CODE or partial NAME): ").strip()
    hits_t = _search_nodes(nodes, term_to) if term_to else gpd.GeoDataFrame()
    if hits_t.empty:
        print("  Cancel.")
        return segments, composition
    to_labels = [f"{r['NAME']}  [{r['CODE']}]" for _, r in hits_t.iterrows()]
    idx_t = _pick_one(to_labels, "TO node")
    if idx_t is None: return segments, composition
    t_node = hits_t.iloc[idx_t]

    if f_node['NAME'] == t_node['NAME']:
        print("  FROM and TO node must be different.")
        return segments, composition

    geom = LineString([(f_node['E'], f_node['N']), (t_node['E'], t_node['N'])])
    
    COLS = [
        'num_tracks', 'gauge', 'electrification', 'route_number', 
        'route_name', 'route_owner', 'km_start', 'km_end'
    ]
    print(f"\n  Attributes: {', '.join(COLS)}")
    val_str = input("  Values (comma-separated): ").strip()
    parts = [p.strip() for p in val_str.split(',')]
    if len(parts) != len(COLS):
        print("  Invalid number of parts. Cancelled.")
        return segments, composition
    
    vals = {}
    for col, v in zip(COLS, parts):
        if col in ('num_tracks', 'gauge'):
            vals[col] = int(v) if v.isdigit() else pd.NA
        elif col in ('km_start', 'km_end'):
            try: vals[col] = float(v)
            except ValueError: vals[col] = pd.NA
        else:
            vals[col] = v if v else pd.NA

    seg_id  = f"c{f_node['CODE']}_{t_node['CODE']}"
    seg_len = geom.length

    new_seg = {
        'segment_id':      seg_id,
        'segment_name':    seg_id,
        'from_name':       f_node['NAME'],
        'to_name':         t_node['NAME'],
        'from_N':          f_node['N'],
        'from_E':          f_node['E'],
        'to_N':            t_node['N'],
        'to_E':            t_node['E'],
        'length_m':        seg_len,
        'num_tracks':      vals.get('num_tracks', pd.NA),
        'gauge':           vals.get('gauge', pd.NA),
        'electrification': vals.get('electrification', pd.NA),
        'km_start':        vals.get('km_start', pd.NA),
        'km_end':          vals.get('km_end', pd.NA),
        'route_number':    vals.get('route_number', pd.NA),
        'route_name':      vals.get('route_name', pd.NA),
        'route_owner':     vals.get('route_owner', pd.NA),
        'geometry':        geom,
    }

    print("\n  Now define the segment composition:")
    new_comp_gdf = _prompt_composition_pieces(
        seg_id, f_node['NAME'], t_node['NAME'], geom, seg_len,
    )

    segments    = pd.concat([segments,    gpd.GeoDataFrame([new_seg], crs=SWISS_CRS)], ignore_index=True)
    composition = pd.concat([composition, new_comp_gdf], ignore_index=True)

    print(f"  Added segment '{seg_id}' with {len(new_comp_gdf)} composition piece(s).")
    return segments, composition


# =============================================================================
# Composition operations
# =============================================================================

def _edit_composition(
    segments: gpd.GeoDataFrame,
    composition: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Phase 3 — Interactively add, remove, or edit composition pieces for a segment."""
    term = input("  Search segment (partial from-name, to-name, or segment_name): ").strip()
    if not term:
        return composition
    hits = _search_segments(segments, term)
    if hits.empty:
        print(f"  No segments matching '{term}'.")
        return composition

    labels = [
        f"{r['from_name']} → {r['to_name']}  [{r['segment_id']}]"
        + (f"  ({r['segment_name']})" if pd.notna(r.get('segment_name')) and r.get('segment_name') else "")
        for _, r in hits.iterrows()
    ]
    idx = _pick_one(labels, "Segment to edit composition")
    if idx is None:
        return composition

    seg_row = hits.iloc[idx]
    seg_id  = seg_row['segment_id']
    seg_geom = seg_row.geometry
    seg_len  = float(seg_row.get('length_m', seg_geom.length if seg_geom else 0))

    COMP_COLS = ['construct_type', 'piece_length_m', 'edge_level',
                 'under_construction', 'track_config', 'railway_type']

    while True:
        ct_options = CONSTRUCT_TYPE_OPTIONS
        pieces = composition[composition['segment_id'] == seg_id].copy()
        total_comp = float(pieces['piece_length_m'].sum()) if not pieces.empty else 0.0

        print(f"\n  Segment: {seg_row['from_name']} → {seg_row['to_name']}  "
              f"(segment length: {seg_len:.0f} m)")
        if pieces.empty:
            print("  Composition: (no pieces)")
        else:
            print(f"  Composition ({len(pieces)} pieces, {total_comp:.0f} m total):")
            for i, (_, p) in enumerate(pieces.iterrows(), 1):
                vals = "  ".join(f"{c}={p.get(c, '')}" for c in COMP_COLS)
                print(f"    {i})  {vals}")

        print("\n    1) Add a piece")
        print("    2) Edit a piece")
        print("    r) Replace all pieces (redefine from scratch)")
        print("    d) Done")
        c = input("  Select (1/2/r/d): ").strip().lower()

        if c == 'd':
            break

        elif c == '1':
            print("  Construct type:")
            for k, v in ct_options.items():
                print(f"    {k}) {v}")
            raw = input("  Select number or type custom value: ").strip() or '1'
            ct = ct_options.get(raw, raw)

            try:
                pl = float(input("  Piece length (m): ").strip())
            except ValueError:
                print("  Invalid length. Cancelled.")
                continue

            el_raw = input("  Edge level [1]: ").strip() or '1'
            try:
                el = int(el_raw)
            except ValueError:
                el = 1

            uc_raw = input("  Under construction (0/1) [0]: ").strip() or '0'
            uc = 1 if uc_raw == '1' else 0

            tc_raw = input("  Track config (Enter for none): ").strip()
            rt_raw = input("  Railway type  (Enter for none): ").strip()

            new_piece = gpd.GeoDataFrame([{
                'segment_id':         seg_id,
                'from_name':          seg_row['from_name'],
                'to_name':            seg_row['to_name'],
                'construct_type':     ct,
                'edge_level':         el,
                'under_construction': uc,
                'track_config':       tc_raw if tc_raw else pd.NA,
                'railway_type':       rt_raw if rt_raw else pd.NA,
                'piece_length_m':     pl,
                'geometry':           seg_geom,
            }], crs=SWISS_CRS)
            composition = pd.concat([composition, new_piece], ignore_index=True)
            print(f"  Added piece: {ct}  {pl:.0f} m.")

        elif c == '2':
            if pieces.empty:
                print("  No pieces to edit.")
                continue
            piece_labels = [
                f"{p['construct_type']}  {float(p['piece_length_m']):.0f} m"
                for _, p in pieces.iterrows()
            ]
            pidx = _pick_one(piece_labels, "Piece to edit")
            if pidx is None:
                continue
            edit_loc = pieces.index[pidx]
            p = composition.loc[edit_loc]

            header  = ", ".join(COMP_COLS)
            current = ", ".join(str(p.get(col, '')) for col in COMP_COLS)
            print(f"  {header}")
            print(f"  {current}")
            updated = input("  Enter updated values (same order; Enter to keep): ").strip()
            if not updated:
                continue

            parts = [v.strip() for v in updated.split(',')]
            if len(parts) != len(COMP_COLS):
                print(f"  Expected {len(COMP_COLS)} values. No changes made.")
                continue

            for col, val in zip(COMP_COLS, parts):
                if val == '':
                    continue
                if col == 'piece_length_m':
                    try:
                        composition.at[edit_loc, col] = float(val)
                    except ValueError:
                        print(f"  Could not parse '{val}' as float for {col}. Skipping.")
                elif col in ('edge_level', 'under_construction'):
                    try:
                        composition.at[edit_loc, col] = int(val)
                    except ValueError:
                        composition.at[edit_loc, col] = val
                else:
                    composition.at[edit_loc, col] = val
            print("  Piece updated.")

        elif c == 'r':
            composition = composition[
                composition['segment_id'] != seg_id
            ].reset_index(drop=True)
            new_comp_gdf = _prompt_composition_pieces(
                seg_id, seg_row['from_name'], seg_row['to_name'],
                seg_geom, seg_len,
            )
            composition = pd.concat([composition, new_comp_gdf], ignore_index=True)
            print(f"  Composition replaced: {len(new_comp_gdf)} piece(s).")

        else:
            print("  Invalid choice.")

    return composition


# =============================================================================
# Import operations
# =============================================================================

def _import_nodes(
    current_nodes: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Import and update nodes from another version."""
    infra_root = Path(paths.MAIN) / paths.NETWORK_INFRASTRUCTURE_DIR
    versions = list_versions(infra_root)
    idx = _pick_one(versions, "Version to import nodes from")
    if idx is None:
        return current_nodes
    source_version = versions[idx]
    
    print(f"  Loading nodes from '{source_version}'...")
    source_nodes = gpd.read_file(infra_root / source_version / 'nodes.gpkg').reset_index(drop=True)
    
    diff_items = []  # tuples of (source_idx, status, current_idx_to_drop, description)
    
    # Helper to generate a match key
    def get_node_key(row):
        nid = str(row.get('node_id', ''))
        bpn = str(row.get('Betriebspunkt_Nummer', ''))
        name = str(row.get('NAME', ''))
        if nid != 'None' and nid != 'nan' and nid != '' and bpn != 'None' and bpn != 'nan' and bpn != '':
            return f"{nid}_{bpn}"
        return f"NAME_{name}"
        
    curr_keys = current_nodes.apply(get_node_key, axis=1)
    
    for i, s_row in source_nodes.iterrows():
        s_key = get_node_key(s_row)
        match_idx = current_nodes.index[curr_keys == s_key].tolist()
        
        if not match_idx:
            diff_items.append((i, "New", None, f"New node '{s_row.get('NAME', 'Unknown')}'"))
        else:
            c_idx = match_idx[0]
            c_row = current_nodes.loc[c_idx]
            
            changes = []
            for col in ['E', 'N', 'node_class', 'transport_mode', 'platform_count', 'NAME']:
                s_val = s_row.get(col)
                c_val = c_row.get(col)
                if pd.isna(s_val) and pd.isna(c_val): continue
                if str(s_val) != str(c_val):
                    changes.append(f"{col}: {c_val} -> {s_val}")
            
            if not s_row.geometry.equals(c_row.geometry):
                changes.append("geometry changed")
                
            if changes:
                desc = f"Update node '{s_row.get('NAME', 'Unknown')}' ({', '.join(changes)})"
                diff_items.append((i, "Changed", c_idx, desc))

    if not diff_items:
        print("  No new or modified nodes found in the selected version.")
        return current_nodes
        
    print("\n  Available imports:")
    for j, (_, status, _, desc) in enumerate(diff_items, 1):
        print(f"    {j}) [{status}] {desc}")
        
    ans = input("\n  Enter numbers to import (comma-separated), 'all', or Enter to cancel: ").strip().lower()
    if not ans:
        return current_nodes
        
    selected_indices = []
    if ans == 'all':
        selected_indices = range(len(diff_items))
    else:
        for part in ans.split(','):
            part = part.strip()
            if part.isdigit():
                val = int(part) - 1
                if 0 <= val < len(diff_items):
                    selected_indices.append(val)
                    
    if not selected_indices:
        print("  No valid choices selected.")
        return current_nodes
        
    new_rows = []
    drop_indices = []
    
    for idx_in_diff in selected_indices:
        s_idx, status, c_idx, desc = diff_items[idx_in_diff]
        new_rows.append(source_nodes.loc[s_idx])
        if status == "Changed" and c_idx is not None:
            drop_indices.append(c_idx)
            
    if drop_indices:
        current_nodes = current_nodes.drop(index=drop_indices)
        
    if new_rows:
        current_nodes = pd.concat([current_nodes, gpd.GeoDataFrame(new_rows, crs=SWISS_CRS)], ignore_index=True)
        
    print(f"  Successfully imported {len(selected_indices)} node(s).")
    return current_nodes


def _import_segments(
    current_segs: gpd.GeoDataFrame,
    current_comp: gpd.GeoDataFrame,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Import and update segments (and their composition) from another version."""
    infra_root = Path(paths.MAIN) / paths.NETWORK_INFRASTRUCTURE_DIR
    versions = list_versions(infra_root)
    idx = _pick_one(versions, "Version to import segments from")
    if idx is None:
        return current_segs, current_comp
    source_version = versions[idx]
    
    print(f"  Loading segments and composition from '{source_version}'...")
    source_dir = infra_root / source_version
    source_segs = gpd.read_file(source_dir / 'segments.gpkg').reset_index(drop=True)
    if (source_dir / 'segments_composition.gpkg').exists():
        source_comp = gpd.read_file(source_dir / 'segments_composition.gpkg').reset_index(drop=True)
    else:
        source_comp = gpd.GeoDataFrame(columns=current_comp.columns)
    
    diff_items = []
    
    def get_seg_key(row):
        sid = str(row.get('segment_id', ''))
        sname = str(row.get('segment_name', ''))
        return f"{sid}_{sname}"
        
    curr_keys = current_segs.apply(get_seg_key, axis=1)
    
    for i, s_row in source_segs.iterrows():
        s_key = get_seg_key(s_row)
        match_idx = current_segs.index[curr_keys == s_key].tolist()
        
        if not match_idx:
            desc = f"New segment '{s_row.get('segment_id', '')}' ({s_row.get('from_name', '')} -> {s_row.get('to_name', '')})"
            diff_items.append((i, "New", None, desc))
        else:
            c_idx = match_idx[0]
            c_row = current_segs.loc[c_idx]
            
            changes = []
            for col in ['num_tracks', 'gauge', 'electrification', 'length_m', 'km_start', 'km_end', 'from_name', 'to_name']:
                s_val = s_row.get(col)
                c_val = c_row.get(col)
                if pd.isna(s_val) and pd.isna(c_val): continue
                if str(s_val) != str(c_val):
                    changes.append(f"{col}")
            
            if not s_row.geometry.equals(c_row.geometry):
                changes.append("geometry")
                
            # Check composition difference by row count and total length
            s_c = source_comp[source_comp['segment_id'] == s_row.get('segment_id', '')]
            c_c = current_comp[current_comp['segment_id'] == c_row.get('segment_id', '')]
            if len(s_c) != len(c_c):
                changes.append("composition count")
                
            if changes:
                desc = f"Update segment '{s_row.get('segment_id', '')}' ({', '.join(changes)})"
                diff_items.append((i, "Changed", c_idx, desc))
                
    if not diff_items:
        print("  No new or modified segments found in the selected version.")
        return current_segs, current_comp
        
    print("\n  Available imports:")
    for j, (_, status, _, desc) in enumerate(diff_items, 1):
        print(f"    {j}) [{status}] {desc}")
        
    ans = input("\n  Enter numbers to import (comma-separated), 'all', or Enter to cancel: ").strip().lower()
    if not ans:
        return current_segs, current_comp
        
    selected_indices = []
    if ans == 'all':
        selected_indices = range(len(diff_items))
    else:
        for part in ans.split(','):
            part = part.strip()
            if part.isdigit():
                val = int(part) - 1
                if 0 <= val < len(diff_items):
                    selected_indices.append(val)
                    
    if not selected_indices:
        print("  No valid choices selected.")
        return current_segs, current_comp
        
    new_seg_rows = []
    new_comp_rows = []
    drop_seg_indices = []
    drop_comp_seg_ids = []
    
    for idx_in_diff in selected_indices:
        s_idx, status, c_idx, desc = diff_items[idx_in_diff]
        s_row = source_segs.loc[s_idx]
        s_id = s_row.get('segment_id')
        
        new_seg_rows.append(s_row)
        if s_id:
            s_comp_pieces = source_comp[source_comp['segment_id'] == s_id]
            if not s_comp_pieces.empty:
                new_comp_rows.extend(s_comp_pieces.to_dict('records'))
            
        if status == "Changed" and c_idx is not None:
            drop_seg_indices.append(c_idx)
            drop_comp_seg_ids.append(current_segs.at[c_idx, 'segment_id'])
            
    if drop_seg_indices:
        current_segs = current_segs.drop(index=drop_seg_indices)
    if drop_comp_seg_ids:
        current_comp = current_comp[~current_comp['segment_id'].isin(drop_comp_seg_ids)]
        
    if new_seg_rows:
        current_segs = pd.concat([current_segs, gpd.GeoDataFrame(new_seg_rows, crs=SWISS_CRS)], ignore_index=True)
    if new_comp_rows:
        geoms = [r.pop('geometry', None) for r in new_comp_rows]
        new_comp_gdf = gpd.GeoDataFrame(new_comp_rows, geometry=geoms, crs=SWISS_CRS)
        current_comp = pd.concat([current_comp, new_comp_gdf], ignore_index=True)
        
    print(f"  Successfully imported {len(selected_indices)} segment(s) and their compositions.")
    return current_segs, current_comp


# =============================================================================
# Main
# =============================================================================

def main():
    try:
        nodes, segments, composition, name, source_version, out_dir, mode = _run_phase0()
    except SystemExit:
        return

    phase = 1  # 1 = nodes, 2 = segments, 3 = composition, 4 = save

    while phase <= 4:

        # ── Phase 1: node editing ────────────────────────────────────────────
        if phase == 1:
            while True:
                print("\n" + "─" * 60)
                print("  Phase 1 — Node editing")
                print("    1) Remove a node")
                print("    2) Adjust a node")
                print("    3) Add a node")
                print("    4) Import nodes from another version")
                print("    5) Proceed to segment editing  →")
                c = input("  Select (1-5): ").strip()

                if c == '1':
                    nodes, segments, composition = _remove_node(nodes, segments, composition)
                elif c == '2':
                    nodes = _adjust_node(nodes)
                elif c == '3':
                    nodes, segments, composition = _add_node(nodes, segments, composition)
                elif c == '4':
                    nodes = _import_nodes(nodes)
                elif c == '5':
                    phase = 2
                    break
                else:
                    print("  Invalid choice.")

        # ── Phase 2: segment editing ─────────────────────────────────────────
        elif phase == 2:
            while True:
                print("\n" + "─" * 60)
                print("  Phase 2 — Segment editing")
                print("    1) Remove a segment")
                print("    2) Adjust a segment")
                print("    3) Add a segment")
                print("    4) Import segments from another version")
                print("    5) Proceed to composition editing  →")
                print("    6) ← Back to node editing")
                c = input("  Select (1-6): ").strip()

                if c == '1':
                    segments, composition = _remove_segment(segments, composition)
                elif c == '2':
                    segments, composition = _adjust_segment(segments, composition)
                elif c == '3':
                    segments, composition = _add_segment(nodes, segments, composition)
                elif c == '4':
                    segments, composition = _import_segments(segments, composition)
                elif c == '5':
                    phase = 3
                    break
                elif c == '6':
                    phase = 1
                    break
                else:
                    print("  Invalid choice.")

        # ── Phase 3: composition editing ─────────────────────────────────────
        elif phase == 3:
            while True:
                print("\n" + "─" * 60)
                print("  Phase 3 — Composition editing")
                print("    1) Edit composition for a segment")
                print("    2) Save and close")
                print("    3) ← Back to segment editing")
                c = input("  Select (1-3): ").strip()

                if c == '1':
                    composition = _edit_composition(segments, composition)
                elif c == '2':
                    print("\n  Saving GeoPackages...")
                    nodes.to_file(out_dir / 'nodes.gpkg', driver='GPKG')
                    segments.to_file(out_dir / 'segments.gpkg', driver='GPKG')
                    composition.to_file(out_dir / 'segments_composition.gpkg', driver='GPKG')
                    print(f"  Done. Version '{name}' saved.")
                    phase = 5  # exit outer loop
                    break
                elif c == '3':
                    phase = 2
                    break
                else:
                    print("  Invalid choice.")

        # ── Phase 4: save ────────────────────────────────────────────────────
        elif phase == 4:
            print("\n" + "─" * 60)
            print(f"  Version: {name}")
            print(f"  Based on: {source_version}")
            print(f"  Nodes:    {len(nodes)}")
            print(f"  Segments: {len(segments)}")
            print(f"  Composition pieces: {len(composition)}")
            print(f"  Output: {out_dir}")

            print("\n  1) Save")
            print("  2) Discard changes")
            print("  3) ← Back to composition editing")
            while True:
                ans = input("  Select (1-3): ").strip()
                if ans in ('1', '2', '3'):
                    break
                print("  Enter 1, 2, or 3.")
            if ans == '3':
                phase = 3
                continue
            if ans == '1':
                print("  Saving GeoPackages...")
                nodes.to_file(out_dir / 'nodes.gpkg', driver='GPKG')
                segments.to_file(out_dir / 'segments.gpkg', driver='GPKG')
                composition.to_file(out_dir / 'segments_composition.gpkg', driver='GPKG')
                print(f"  Done. Version '{name}' saved.")
            else:
                print("  Discarded changes.")
            break

if __name__ == '__main__':
    main()