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
MERGE_ATTRS = ('Num_Tracks', 'Gauge', 'Electrification_Class')

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
            'Segment_ID':                   seg_id,
            'From_Name':            from_name,
            'To_Name':              to_name,
            'Engineering_Structure': ct,
            'Edge_Level':           el,
            'Under_Construction':   uc,
            'Piece_Length':         pl,
            'geometry':             seg_geom,
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
    """Return rows whose Name or Code contain term (case-insensitive)."""
    term_l = term.lower()
    mask = (
        nodes['Name'].fillna('').str.lower().str.contains(term_l, regex=False) |
        nodes['Code'].fillna('').str.lower().str.contains(term_l, regex=False)
    )
    return nodes[mask]


def _search_segments(segments: gpd.GeoDataFrame, term: str) -> gpd.GeoDataFrame:
    """Return rows whose From_Name, To_Name, or Segment_ID contain term (case-insensitive)."""
    term_l = term.lower()
    mask = (
        segments['From_Name'].fillna('').str.lower().str.contains(term_l, regex=False) |
        segments['To_Name'].fillna('').str.lower().str.contains(term_l, regex=False) |
        segments['Segment_ID'].fillna('').astype(str).str.lower().str.contains(term_l, regex=False)
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
        idx = _pick_one(versions, "Base version")
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

    labels = [f"{r['Name']}  [{r['Code']}]" for _, r in hits.iterrows()]
    idx = _pick_one(labels, "Node to adjust")
    if idx is None:
        return nodes

    row_idx = hits.index[idx]
    row = nodes.loc[row_idx]

    COLS = ['Name', 'Code', 'E', 'N', 'Node_Class', 'Transport_Mode', 'Platform_Count']
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
        elif col == 'Platform_Count':
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
    print(f"  Node '{nodes.at[row_idx, 'Name']}' updated.")
    return nodes


def _remove_node(
    nodes: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
    composition: gpd.GeoDataFrame,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Remove a node with optional segment merge."""
    term = input("  Search node (Code or partial Name): ").strip()
    if not term:
        return nodes, segments, composition
    hits = _search_nodes(nodes, term)
    if hits.empty:
        print(f"  No nodes matching '{term}'.")
        return nodes, segments, composition

    labels = [f"{r['Name']}  [{r['Code']}]" for _, r in hits.iterrows()]
    idx = _pick_one(labels, "Node to remove")
    if idx is None:
        return nodes, segments, composition

    row_idx   = hits.index[idx]
    node_row  = nodes.loc[row_idx]
    node_name = node_row['Name']

    # Print node attributes
    print(f"\n  Node: {node_name}")
    for col in ['Code', 'E', 'N', 'Node_Class', 'Transport_Mode', 'Platform_Count']:
        print(f"    {col}: {node_row.get(col, '')}")

    # Connected segments
    conn_mask = (segments['From_Name'] == node_name) | (segments['To_Name'] == node_name)
    conn_segs = segments[conn_mask]
    print(f"\n  Connected segments ({len(conn_segs)}):")
    for _, s in conn_segs.iterrows():
        print(f"    {s['From_Name']} → {s['To_Name']}  [{s['Segment_ID']}]")

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
                all_ends = {s0['From_Name'], s0['To_Name'],
                            s1['From_Name'], s1['To_Name']}
                surviving = [e for e in all_ends if e != node_name]
                from_end, to_end = surviving[0], surviving[1]

                fe_code_rows = nodes[nodes['Name'] == from_end]
                te_code_rows = nodes[nodes['Name'] == to_end]
                fe_code = fe_code_rows.iloc[0]['Code'] if not fe_code_rows.empty else from_end
                te_code = te_code_rows.iloc[0]['Code'] if not te_code_rows.empty else to_end
                new_seg_id  = f"c{fe_code}_{te_code}"
                new_geom    = linemerge([s0.geometry, s1.geometry])
                new_length  = new_geom.length
                new_km_start = min(s0['Km_Start'], s1['Km_Start'])
                new_km_end   = max(s0['Km_End'],   s1['Km_End'])

                fn_row = nodes[nodes['Name'] == from_end]
                tn_row = nodes[nodes['Name'] == to_end]

                fe_num = fn_row.iloc[0]['Number'] if not fn_row.empty else pd.NA
                te_num = tn_row.iloc[0]['Number'] if not tn_row.empty else pd.NA
                new_number = f"{int(fe_num)}_{int(te_num)}" if pd.notna(fe_num) and pd.notna(te_num) else pd.NA

                new_row = {
                    'Segment_ID':                  new_seg_id,
                    'Number':              new_number,
                    'Code':                f"{fe_code}_{te_code}",
                    'From_Name':           from_end,
                    'To_Name':             to_end,
                    'From_N': fn_row.iloc[0]['N'] if not fn_row.empty else pd.NA,
                    'From_E': fn_row.iloc[0]['E'] if not fn_row.empty else pd.NA,
                    'To_N':   tn_row.iloc[0]['N'] if not tn_row.empty else pd.NA,
                    'To_E':   tn_row.iloc[0]['E'] if not tn_row.empty else pd.NA,
                    'Length':              new_length,
                    'Num_Tracks':          s0['Num_Tracks'],
                    'Gauge':               s0['Gauge'],
                    'Electrification_Class': s0['Electrification_Class'],
                    'Km_Start':            new_km_start,
                    'Km_End':              new_km_end,
                    'Route_Number':        s0['Route_Number'],
                    'Route_Name':          s0['Route_Name'],
                    'Route_Owner':         s0['Route_Owner'],
                    'geometry':            new_geom,
                }

                # Remove the two old segments
                old_ids = conn_segs['Segment_ID'].tolist()
                segments = segments[
                    ~segments['Segment_ID'].isin(old_ids)
                ].reset_index(drop=True)
                segments = pd.concat(
                    [segments, gpd.GeoDataFrame([new_row], crs=SWISS_CRS)],
                    ignore_index=True
                )

                # Replace composition: remove 2 old rows, add 1 normal row
                composition = composition[
                    ~composition['Segment_ID'].isin(old_ids)
                ].reset_index(drop=True)
                comp_row = gpd.GeoDataFrame([{
                    'Segment_ID':                   new_seg_id,
                    'From_Name':            from_end,
                    'To_Name':              to_end,
                    'Engineering_Structure': 'normal',
                    'Edge_Level':           1,
                    'Under_Construction':   0,
                    'Piece_Length':         new_length,
                    'geometry':             new_geom,
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
            remove_ids = conn_segs['Segment_ID'].tolist()
            segments = segments[
                ~segments['Segment_ID'].isin(remove_ids)
            ].reset_index(drop=True)
            composition = composition[
                ~composition['Segment_ID'].isin(remove_ids)
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
    km_mid = S['Km_Start'] + t * (S['Km_End'] - S['Km_Start'])

    split_pt = geom.interpolate(split_dist)

    from_code_rows = nodes[nodes['Name'] == S['From_Name']]
    to_code_rows   = nodes[nodes['Name'] == S['To_Name']]
    from_code = from_code_rows.iloc[0]['Code'] if not from_code_rows.empty else S['From_Name']
    to_code   = to_code_rows.iloc[0]['Code']   if not to_code_rows.empty   else S['To_Name']

    id_A = f"c{from_code}_{new_node_code}"
    id_B = f"c{new_node_code}_{to_code}"

    row_A = {
        'Segment_ID':                    id_A,
        'From_Name':             S['From_Name'],
        'To_Name':               new_node_name,
        'From_N':                S.get('From_N', pd.NA),
        'From_E':                S.get('From_E', pd.NA),
        'To_N':                  split_pt.y,
        'To_E':                  split_pt.x,
        'Length':                geom_A.length,
        'Num_Tracks':            S['Num_Tracks'],
        'Gauge':                 S['Gauge'],
        'Electrification_Class': S['Electrification_Class'],
        'Km_Start':              S['Km_Start'],
        'Km_End':                km_mid,
        'Route_Number':          S['Route_Number'],
        'Route_Name':            S['Route_Name'],
        'Route_Owner':           S['Route_Owner'],
        'Average_Speed':         S.get('Average_Speed', pd.NA),
        'Predominant_Speed':     S.get('Predominant_Speed', pd.NA),
        'Speed_Coverage_Pct':    S.get('Speed_Coverage_Pct', 0.0),
        'geometry':              geom_A,
    }
    row_B = {
        'Segment_ID':                    id_B,
        'From_Name':             new_node_name,
        'To_Name':               S['To_Name'],
        'From_N':                split_pt.y,
        'From_E':                split_pt.x,
        'To_N':                  S.get('To_N', pd.NA),
        'To_E':                  S.get('To_E', pd.NA),
        'Length':                geom_B.length,
        'Num_Tracks':            S['Num_Tracks'],
        'Gauge':                 S['Gauge'],
        'Electrification_Class': S['Electrification_Class'],
        'Km_Start':              km_mid,
        'Km_End':                S['Km_End'],
        'Route_Number':          S['Route_Number'],
        'Route_Name':            S['Route_Name'],
        'Route_Owner':           S['Route_Owner'],
        'Average_Speed':         S.get('Average_Speed', pd.NA),
        'Predominant_Speed':     S.get('Predominant_Speed', pd.NA),
        'Speed_Coverage_Pct':    S.get('Speed_Coverage_Pct', 0.0),
        'geometry':              geom_B,
    }

    # Remove original segment, add A and B
    segments = segments[segments.index != seg_idx].reset_index(drop=True)
    segments = pd.concat(
        [segments, gpd.GeoDataFrame([row_A, row_B], crs=SWISS_CRS)],
        ignore_index=True
    )

    # --- Redistribute composition pieces ---
    old_comp    = composition[composition['Segment_ID'] == S['Segment_ID']].copy()
    composition = composition[
        composition['Segment_ID'] != S['Segment_ID']
    ].reset_index(drop=True)

    new_comp_rows = []
    cumulative = 0.0
    for _, piece in old_comp.iterrows():
        piece_len   = float(piece['Piece_Length'])
        piece_start = cumulative
        piece_end   = cumulative + piece_len

        base = piece.to_dict()
        base.pop('geometry', None)  # geometry will be set per-segment below

        if piece_end <= split_dist:
            # Entirely in A
            new_comp_rows.append({
                **base,
                'Segment_ID':        id_A,
                'From_Name': S['From_Name'],
                'To_Name':   new_node_name,
                '_geom':     geom_A,
            })
        elif piece_start >= split_dist:
            # Entirely in B
            new_comp_rows.append({
                **base,
                'Segment_ID':        id_B,
                'From_Name': new_node_name,
                'To_Name':   S['To_Name'],
                '_geom':     geom_B,
            })
        else:
            # Straddles split — divide into two rows
            len_in_A = split_dist - piece_start
            len_in_B = piece_end  - split_dist
            new_comp_rows.append({
                **base,
                'Segment_ID':           id_A,
                'From_Name':    S['From_Name'],
                'To_Name':      new_node_name,
                'Piece_Length': len_in_A,
                '_geom':        geom_A,
            })
            new_comp_rows.append({
                **base,
                'Segment_ID':           id_B,
                'From_Name':    new_node_name,
                'To_Name':      S['To_Name'],
                'Piece_Length': len_in_B,
                '_geom':        geom_B,
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
            segments['Route_Number'].fillna('').astype(str) == route
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
                (route_segs['Km_Start'].fillna(-1) <= km) &
                (route_segs['Km_End'].fillna(-1)   >= km)
            ]
            if match.empty:
                print(f"  No segment covers km {km} on route '{route}'. Try again.")
                continue
            seg_idx = match.index[0]
            S = segments.loc[seg_idx]
            t          = (km - S['Km_Start']) / (S['Km_End'] - S['Km_Start'])
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
    name = input("  Name: ").strip()
    if not name:
        print("  Name cannot be empty. Cancelled.")
        return nodes, segments, composition
    if name in nodes['Name'].values:
        print(f"  A node named '{name}' already exists. Cancelled.")
        return nodes, segments, composition

    code = input(f"  Code [{name[:4].upper()}]: ").strip() or name[:4].upper()

    print("  Node class:")
    for k, v in NODE_CLASS_OPTIONS.items():
        print(f"    {k}) {v}")
    while True:
        nc = input("  Select (1–6) [1]: ").strip() or '1'
        if nc in NODE_CLASS_OPTIONS:
            node_class = NODE_CLASS_OPTIONS[nc]
            break
        print("  Invalid — enter 1–6.")

    # Generate a synthetic Number above the BAV range (max + 1,
    # floored at 9_000_000 so synthetic nodes are clearly distinguishable).
    existing_ids = nodes['Number'].dropna()
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
        'Segment_ID':               synthetic_node_id,
        'Number':           synthetic_bpnr,
        'Name':             name,
        'Code':             code,
        'E':                node_E,
        'N':                node_N,
        'Node_Class':       node_class,
        'Transport_Mode':   None,
        'Platform_Count':   None,
        'Track_Count':      None,
        'Parent_Node':      None,
        'geometry':         Point(node_E, node_N),
    }
    print(f"  Assigned synthetic Number: {synthetic_bpnr}")

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
        f"{r['From_Name']} → {r['To_Name']}  [{r['Segment_ID']}]"
        for _, r in hits.iterrows()
    ]
    idx = _pick_one(labels, "Segment to remove")
    if idx is None:
        return segments, composition

    row_idx = hits.index[idx]
    S = segments.loc[row_idx]

    ans = input(f"  Remove segment '{S['From_Name']} → {S['To_Name']}'? (y/n) [n]: ").strip().lower() or 'n'
    if ans == 'y':
        segments = segments[segments.index != row_idx].reset_index(drop=True)
        composition = composition[composition['Segment_ID'] != S['Segment_ID']].reset_index(drop=True)
        print(f"  Removed segment '{S['Segment_ID']}'.")
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
        f"{r['From_Name']} → {r['To_Name']}  [{r['Segment_ID']}]"
        for _, r in hits.iterrows()
    ]
    idx = _pick_one(labels, "Segment to adjust")
    if idx is None:
        return segments, composition

    row_idx = hits.index[idx]
    row = segments.loc[row_idx]

    COLS = [
        'From_Name', 'To_Name', 'Length', 'Num_Tracks', 'Gauge', 'Electrification_Class',
        'Km_Start', 'Km_End', 'Route_Number', 'Route_Name', 'Route_Owner', 'Average_Speed',
    ]
    header  = ", ".join(COLS)
    current = ", ".join(str(row.get(c, '')) for c in COLS)
    print(f"\n  {header}")
    print(f"  {current}")
    print(
        f"  OSM-derived (read-only): Predominant_Speed={row.get('Predominant_Speed', 'N/A')}  "
        f"Speed_Coverage_Pct={row.get('Speed_Coverage_Pct', 'N/A')}"
    )

    pieces = composition[composition['Segment_ID'] == row['Segment_ID']]
    comp_strs = [f"{p['Engineering_Structure']} {float(p['Piece_Length']):.0f}m" for _, p in pieces.iterrows()]
    print(f"  Composition: {len(pieces)} pieces — {' / '.join(comp_strs)}")

    updated = input("  Enter updated values (same order; press Enter to keep): ").strip()
    if updated:
        parts = [p.strip() for p in updated.split(',')]
        if len(parts) != len(COLS):
            print(f"  Expected {len(COLS)} values, got {len(parts)}. No changes made.")
            return segments, composition

        EDITABLE = [
            'Num_Tracks', 'Gauge', 'Electrification_Class',
            'Km_Start', 'Km_End', 'Route_Number', 'Route_Name', 'Route_Owner', 'Average_Speed',
        ]
        for col, val in zip(COLS, parts):
            if val == '' or col not in EDITABLE:
                continue
            if col in ('Km_Start', 'Km_End', 'Average_Speed'):
                try:
                    segments.at[row_idx, col] = float(val)
                except ValueError:
                    pass
            elif col in ('Num_Tracks', 'Gauge'):
                try:
                    segments.at[row_idx, col] = int(val)
                except ValueError:
                    segments.at[row_idx, col] = val
            else:
                segments.at[row_idx, col] = val

        print(f"  Segment '{row['Segment_ID']}' attributes updated.")

    # Composition update
    recomp = input("\n  Redefine composition for this segment? (y/n) [n]: ").strip().lower() or 'n'
    if recomp == 'y':
        seg_geom = segments.loc[row_idx].geometry
        seg_len  = float(segments.loc[row_idx].get('Length', seg_geom.length))
        # Drop existing composition pieces for this segment
        composition = composition[
            composition['Segment_ID'] != row['Segment_ID']
        ].reset_index(drop=True)
        new_comp_gdf = _prompt_composition_pieces(
            row['Segment_ID'],
            segments.loc[row_idx, 'From_Name'],
            segments.loc[row_idx, 'To_Name'],
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
    term_from = input("  FROM node (Code or partial Name): ").strip()
    hits_f = _search_nodes(nodes, term_from) if term_from else gpd.GeoDataFrame()
    if hits_f.empty:
        print("  Cancel.")
        return segments, composition
    from_labels = [f"{r['Name']}  [{r['Code']}]" for _, r in hits_f.iterrows()]
    idx_f = _pick_one(from_labels, "FROM node")
    if idx_f is None: return segments, composition
    f_node = hits_f.iloc[idx_f]

    term_to = input("  TO node (Code or partial Name): ").strip()
    hits_t = _search_nodes(nodes, term_to) if term_to else gpd.GeoDataFrame()
    if hits_t.empty:
        print("  Cancel.")
        return segments, composition
    to_labels = [f"{r['Name']}  [{r['Code']}]" for _, r in hits_t.iterrows()]
    idx_t = _pick_one(to_labels, "TO node")
    if idx_t is None: return segments, composition
    t_node = hits_t.iloc[idx_t]

    if f_node['Name'] == t_node['Name']:
        print("  FROM and TO node must be different.")
        return segments, composition

    geom = LineString([(f_node['E'], f_node['N']), (t_node['E'], t_node['N'])])
    
    COLS = [
        'Num_Tracks', 'Gauge', 'Electrification_Class', 'Route_Number',
        'Route_Name', 'Route_Owner', 'Km_Start', 'Km_End'
    ]
    print(f"\n  Attributes: {', '.join(COLS)}")
    val_str = input("  Values (comma-separated): ").strip()
    parts = [p.strip() for p in val_str.split(',')]
    if len(parts) != len(COLS):
        print("  Invalid number of parts. Cancelled.")
        return segments, composition

    vals = {}
    for col, v in zip(COLS, parts):
        if col in ('Num_Tracks', 'Gauge'):
            vals[col] = int(v) if v.isdigit() else pd.NA
        elif col in ('Km_Start', 'Km_End'):
            try: vals[col] = float(v)
            except ValueError: vals[col] = pd.NA
        else:
            vals[col] = v if v else pd.NA

    spd_raw = input("  Average / predominant speed (km/h, Enter for none): ").strip()
    try:
        avg_speed = float(spd_raw) if spd_raw else pd.NA
    except ValueError:
        avg_speed = pd.NA

    seg_id  = f"c{f_node['Code']}_{t_node['Code']}"
    seg_len = geom.length

    fn_num = f_node.get('Number')
    tn_num = t_node.get('Number')
    seg_number = f"{int(fn_num)}_{int(tn_num)}" if pd.notna(fn_num) and pd.notna(tn_num) else pd.NA

    new_seg = {
        'Segment_ID':                    seg_id,
        'Number':                seg_number,
        'Code':                  f"{f_node['Code']}_{t_node['Code']}",
        'From_Name':             f_node['Name'],
        'To_Name':               t_node['Name'],
        'From_N':                f_node['N'],
        'From_E':                f_node['E'],
        'To_N':                  t_node['N'],
        'To_E':                  t_node['E'],
        'Length':                seg_len,
        'Num_Tracks':            vals.get('Num_Tracks', pd.NA),
        'Gauge':                 vals.get('Gauge', pd.NA),
        'Electrification_Class': vals.get('Electrification_Class', pd.NA),
        'Km_Start':              vals.get('Km_Start', pd.NA),
        'Km_End':                vals.get('Km_End', pd.NA),
        'Route_Number':          vals.get('Route_Number', pd.NA),
        'Route_Name':            vals.get('Route_Name', pd.NA),
        'Route_Owner':           vals.get('Route_Owner', pd.NA),
        'Average_Speed':         avg_speed,
        'Predominant_Speed':     avg_speed,
        'Speed_Coverage_Pct':    1.0 if pd.notna(avg_speed) else 0.0,
        'geometry':              geom,
    }

    print("\n  Now define the segment composition:")
    new_comp_gdf = _prompt_composition_pieces(
        seg_id, f_node['Name'], t_node['Name'], geom, seg_len,
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
    term = input("  Search segment (partial From_Name, To_Name, or ID): ").strip()
    if not term:
        return composition
    hits = _search_segments(segments, term)
    if hits.empty:
        print(f"  No segments matching '{term}'.")
        return composition

    labels = [
        f"{r['From_Name']} → {r['To_Name']}  [{r['Segment_ID']}]"
        for _, r in hits.iterrows()
    ]
    idx = _pick_one(labels, "Segment to edit composition")
    if idx is None:
        return composition

    seg_row = hits.iloc[idx]
    seg_id  = seg_row['Segment_ID']
    seg_geom = seg_row.geometry
    seg_len  = float(seg_row.get('Length', seg_geom.length if seg_geom else 0))

    COMP_COLS = ['Engineering_Structure', 'Piece_Length', 'Edge_Level', 'Under_Construction']

    while True:
        ct_options = CONSTRUCT_TYPE_OPTIONS
        pieces = composition[composition['Segment_ID'] == seg_id].copy()
        total_comp = float(pieces['Piece_Length'].sum()) if not pieces.empty else 0.0

        print(f"\n  Segment: {seg_row['From_Name']} → {seg_row['To_Name']}  "
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

            new_piece = gpd.GeoDataFrame([{
                'Segment_ID':                   seg_id,
                'From_Name':            seg_row['From_Name'],
                'To_Name':              seg_row['To_Name'],
                'Engineering_Structure': ct,
                'Edge_Level':           el,
                'Under_Construction':   uc,
                'Piece_Length':         pl,
                'geometry':             seg_geom,
            }], crs=SWISS_CRS)
            composition = pd.concat([composition, new_piece], ignore_index=True)
            print(f"  Added piece: {ct}  {pl:.0f} m.")

        elif c == '2':
            if pieces.empty:
                print("  No pieces to edit.")
                continue
            piece_labels = [
                f"{p['Engineering_Structure']}  {float(p['Piece_Length']):.0f} m"
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
                if col == 'Piece_Length':
                    try:
                        composition.at[edit_loc, col] = float(val)
                    except ValueError:
                        print(f"  Could not parse '{val}' as float for {col}. Skipping.")
                elif col in ('Edge_Level', 'Under_Construction'):
                    try:
                        composition.at[edit_loc, col] = int(val)
                    except ValueError:
                        composition.at[edit_loc, col] = val
                else:
                    composition.at[edit_loc, col] = val
            print("  Piece updated.")

        elif c == 'r':
            composition = composition[
                composition['Segment_ID'] != seg_id
            ].reset_index(drop=True)
            new_comp_gdf = _prompt_composition_pieces(
                seg_id, seg_row['From_Name'], seg_row['To_Name'],
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
        nid = str(row.get('ID', ''))
        bpn = str(row.get('Number', ''))
        name = str(row.get('Name', ''))
        if nid != 'None' and nid != 'nan' and nid != '' and bpn != 'None' and bpn != 'nan' and bpn != '':
            return f"{nid}_{bpn}"
        return f"Name_{name}"

    curr_keys = current_nodes.apply(get_node_key, axis=1)

    for i, s_row in source_nodes.iterrows():
        s_key = get_node_key(s_row)
        match_idx = current_nodes.index[curr_keys == s_key].tolist()

        if not match_idx:
            diff_items.append((i, "New", None, f"New node '{s_row.get('Name', 'Unknown')}'"))
        else:
            c_idx = match_idx[0]
            c_row = current_nodes.loc[c_idx]

            changes = []
            for col in ['E', 'N', 'Node_Class', 'Transport_Mode', 'Platform_Count', 'Name']:
                s_val = s_row.get(col)
                c_val = c_row.get(col)
                if pd.isna(s_val) and pd.isna(c_val): continue
                if str(s_val) != str(c_val):
                    changes.append(f"{col}: {c_val} -> {s_val}")

            if not s_row.geometry.equals(c_row.geometry):
                changes.append("geometry changed")

            if changes:
                desc = f"Update node '{s_row.get('Name', 'Unknown')}' ({', '.join(changes)})"
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
        sid = str(row.get('ID', ''))
        return sid
        
    curr_keys = current_segs.apply(get_seg_key, axis=1)
    
    for i, s_row in source_segs.iterrows():
        s_key = get_seg_key(s_row)
        match_idx = current_segs.index[curr_keys == s_key].tolist()
        
        if not match_idx:
            desc = f"New segment '{s_row.get('ID', '')}' ({s_row.get('From_Name', '')} -> {s_row.get('To_Name', '')})"
            diff_items.append((i, "New", None, desc))
        else:
            c_idx = match_idx[0]
            c_row = current_segs.loc[c_idx]

            changes = []
            for col in ['Num_Tracks', 'Gauge', 'Electrification_Class', 'Length', 'Km_Start', 'Km_End', 'From_Name', 'To_Name']:
                s_val = s_row.get(col)
                c_val = c_row.get(col)
                if pd.isna(s_val) and pd.isna(c_val): continue
                if str(s_val) != str(c_val):
                    changes.append(f"{col}")
            
            if not s_row.geometry.equals(c_row.geometry):
                changes.append("geometry")
                
            # Check composition difference by row count
            s_c = source_comp[source_comp['Segment_ID'] == s_row.get('ID', '')]
            c_c = current_comp[current_comp['Segment_ID'] == c_row.get('ID', '')]
            if len(s_c) != len(c_c):
                changes.append("composition count")
                
            if changes:
                desc = f"Update segment '{s_row.get('ID', '')}' ({', '.join(changes)})"
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
        s_id = s_row.get('Segment_ID')
        
        new_seg_rows.append(s_row)
        if s_id:
            s_comp_pieces = source_comp[source_comp['Segment_ID'] == s_id]
            if not s_comp_pieces.empty:
                new_comp_rows.extend(s_comp_pieces.to_dict('records'))

        if status == "Changed" and c_idx is not None:
            drop_seg_indices.append(c_idx)
            drop_comp_seg_ids.append(current_segs.at[c_idx, 'ID'])
            
    if drop_seg_indices:
        current_segs = current_segs.drop(index=drop_seg_indices)
    if drop_comp_seg_ids:
        current_comp = current_comp[~current_comp['Segment_ID'].isin(drop_comp_seg_ids)]
        
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