# catchment_build_network.py
#
# CRS used throughout this script: EPSG:2056 (LV95 — Swiss National Grid).
# Stop coordinates are read directly from the stop_E / stop_N columns written
# by catchment_filter_gtfs.py — no reprojection is performed here.
#
# Outputs (written to subfolders defined by the folder constants below)
# -------
#   PT-Feeder (bus / tram / funicular / ship / metro):
#     paths.FEEDER_LINES_DIR / PT_FEEDER_OUTPUT_FOLDER / PT_FEEDER_STOPS_FILE
#     paths.FEEDER_LINES_DIR / PT_FEEDER_OUTPUT_FOLDER / PT_FEEDER_LINES_FILE
#
#   Rail (S-Bahn / regional / inter-regional / long-distance):
#     paths.RAIL_PROCESSED_DIR / RAIL_OUTPUT_FOLDER / RAIL_STOPS_FILE
#     paths.RAIL_PROCESSED_DIR / RAIL_OUTPUT_FOLDER / RAIL_LINES_FILE
#
# Each lines GeoPackage contains one layer per route_type, each with one
# feature per (route_id, direction_id, variant_rank) using structural
# stop-sequence variants classified by trip share, hour spread, and headway CV.
# Each stops GeoPackage contains one layer per dominant mode type.
# Circular lines (first stop == last stop) are stored as a single feature.

import os
import statistics
import time
from collections import defaultdict
from datetime import datetime, timedelta

import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString

import paths
import settings

_start_time = time.time()

# ---------------------------------------------------------------------------
# Folder / file name constants — adjust here to target a different GTFS source
# ---------------------------------------------------------------------------

GTFS_INPUT_FOLDER      = 'GTFS_FP2026_ZH'          # subfolder of paths.GTFS_TRANSIT_DIR

PT_FEEDER_OUTPUT_FOLDER = 'FP2026_ZH_network'       # subfolder of paths.FEEDER_LINES_DIR
PT_FEEDER_STOPS_FILE    = 'pt_feeder_stops.gpkg'
PT_FEEDER_LINES_FILE    = 'pt_feeder_lines.gpkg'

RAIL_OUTPUT_FOLDER      = 'FP2026_ZH_network'       # subfolder of paths.RAIL_PROCESSED_DIR
RAIL_STOPS_FILE         = 'rail_stops.gpkg'
RAIL_LINES_FILE         = 'rail_lines.gpkg'

PT_FEEDER_PROJECT_FILE  = 'pt_feeder_lines.qgz'
RAIL_PROJECT_FILE       = 'rail_lines.qgz'
BUILD_REPORT_FILE       = 'build_report.txt'

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CODEBASE_CRS = 'EPSG:2056'

# Frequency windows (used as validation / fallback for adaptive peak detection)
AM_PEAK_START  = '06:00:00'
AM_PEAK_END    = '09:00:00'
PM_PEAK_START  = '16:00:00'
PM_PEAK_END    = '19:00:00'
OFFPEAK_START  = '09:00:00'
OFFPEAK_END    = '16:00:00'

# Adaptive peak detection — per (route_id, direction_id) window detection
HALFDAY_MIDPOINT              = '12:00:00'  # splits operating day into AM / PM halves
PEAK_BIN_WIDTH_MIN            = 30          # bin width in minutes for departure histogram
PEAK_MIN_OVERLAP              = 0.50        # min overlap fraction with validation window
PEAK_MIN_CLUSTER_BINS         = 2           # cluster must span ≥ this many bins (avoids single-bin artifacts)
PEAK_CLASSIFICATION_THRESHOLD = 0.80        # fraction of deps to classify peak_only / offpeak_only

# Line-level acceptance gate — applied per route_id before variant classification.
# A line must have ≥ LINE_MIN_DEPARTURES_IN_WINDOW departures in at least one
# time window (AM peak, off-peak, PM peak) considering either direction, AND its
# departures (both directions pooled) must span ≥ LINE_MIN_HOUR_SPREAD distinct hours.
LINE_MIN_DEPARTURES_IN_WINDOW = 3     # ≈1 dep/hr in a 3-hour window
LINE_MIN_HOUR_SPREAD          = 2     # min distinct hours across both directions

# Variant classification thresholds
VARIANT_MIN_TRIP_SHARE           = 0.10  # min fraction of direction's trips
VARIANT_MIN_HOUR_SPREAD          = 2     # min distinct hours variant appears in
VARIANT_HEADWAY_CV_MAX           = 0.40  # max coefficient of variation of inter-departure gaps
VARIANT_MIN_DEPARTURES_IN_WINDOW = 3     # min departures in a window to compute frequency
VARIANT_RELAXED_MIN_DEPARTURES   = 2     # relaxed minimum (used when line has ≥2 deps in both AM and PM peaks)

# Short-working suppression: a passing variant is suppressed if its stop sequence
# is a contiguous prefix of a longer passing variant AND at least this fraction of
# its first-stop departures (within 1 minute) overlap with the longer variant.
SHORTWORKING_OVERLAP_MIN = 0.80

# Standard headway snapping: dep/hr values are snapped to the nearest value in
# this list. The list corresponds to headways of 1,2,3,4,5,6,7.5,10,12,15,20,30,60,120,240 min.
STANDARD_FREQUENCIES = [60, 30, 20, 15, 12, 10, 8, 6, 5, 4, 3, 2, 1, 0.5, 0.25]

# Minimum acceptable frequency (dep/hr). Variants whose highest non-NULL
# window frequency is below this threshold are dropped from the network.
MIN_FREQUENCY_DEP_HR = 1

# Minimum number of weekdays (Mon–Fri) a service_id must run on to be
# considered a "typical weekday" service for frequency counting.
# ≥3/5 keeps Mon–Thu (4/5) and full-week (5/5) services while excluding
# MO-only supplements (1/5) and other partial-week exception IDs.
WEEKDAY_MIN_DAYS = 3

# Maximum gap in minutes between two departures on the same route+direction
# that are treated as a single "service slot" (two-terminal branch lines).
TERMINAL_MERGE_MINUTES = 2

# Tier 2 service duration filter: a service_id must be active on at least this
# fraction of available weekdays in the timetable year. This excludes
# construction-period replacements and seasonal-only services that do not
# represent the standard network.
MIN_WEEKDAY_ACTIVE_FRACTION = 0.50

# Mode selection — controls which mode groups are built.
#   'all'         : build both PT-Feeder and Rail (production default)
#   'pt_feeder'   : only PT-Feeder modes (tram, bus, metro, ship, funicular)
#   'rail'        : only Rail modes (S-Bahn, regional, inter-regional, long-distance)
#   list of ints  : only specific route_types, e.g. [900] for tram, [109, 106] for S-Bahn + regional
BUILD_MODES = 'all'

PT_FEEDER_LINE_TYPES = {
    900:  'Tram',
    401:  'Metro',
    700:  'Bus',
    702:  'Express Bus',
    715:  'On-demand Bus',
    1000: 'Ship',
    1400: 'Funicular',
}

RAIL_LINE_TYPES = {
    102: 'Long-Distance Rail',
    103: 'Inter-Regional Rail',
    106: 'Regional Rail',
    109: 'S-Bahn / Suburban Rail',
}

# ---------------------------------------------------------------------------
# Colour scheme (OpenRailwayMap-aligned) for line layers
# ---------------------------------------------------------------------------

# Line colours per route_type
LINE_COLOURS = {
    102:  '#FF0000',  # Long-Distance Rail  — red
    103:  '#FF0000',  # Inter-Regional Rail — red
    106:  '#000000',  # Regional Rail       — black
    109:  '#000000',  # S-Bahn              — black
    401:  '#00246B',  # Metro               — dark blue (ORM)
    900:  '#FF66CC',  # Tram                — pink (ORM)
    700:  '#0000FF',  # Bus                 — blue
    702:  '#0000FF',  # Express Bus         — blue
    715:  '#0000FF',  # On-demand Bus       — blue
    1000: '#0000FF',  # Ship                — blue dashed
    1400: '#000000',  # Funicular           — black dashed
}

LINE_STYLE = {
    1000: 'dashed',
    1400: 'dashed',
}

# PT-feeder stop dominant-mode hierarchy (lower index = higher priority)
PT_FEEDER_STOP_HIERARCHY = [401, 900, 700, 702, 715, 1400, 1000]

# Rail stop style — always white fill, black outline
RAIL_STOP_FILL    = '#FFFFFF'
RAIL_STOP_OUTLINE = '#000000'

# Layer name mapping: route_type → layer name used inside the GeoPackage
LAYER_NAMES = {
    102:  'long_distance_rail',
    103:  'inter_regional_rail',
    106:  'regional_rail',
    109:  'sbahn',
    401:  'metro',
    900:  'tram',
    700:  'bus',
    702:  'express_bus',
    715:  'on_demand_bus',
    1000: 'ship',
    1400: 'funicular',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fpath(filename):
    return os.path.join(paths.GTFS_TRANSIT_DIR, GTFS_INPUT_FOLDER, filename)


def _load(filename):
    p = _fpath(filename)
    if not os.path.isfile(p):
        return None
    print(f"  Loading {filename} ...", end=' ', flush=True)
    df = pd.read_csv(p, dtype=str, low_memory=False)
    print(f"{len(df):,} rows")
    return df


def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def _build_linestring(stop_id_seq, coord_dict):
    """Build a LineString from an ordered sequence of stop_ids using LV95 coords."""
    coords = []
    for sid in stop_id_seq:
        pt = coord_dict.get(sid)
        if pt is not None:
            coords.append((pt.x, pt.y))
    if len(coords) < 2:
        return None
    return LineString(coords)


def _detect_dominant_period(route_trip_ids, sid_active_sets):
    """Identify the dominant calendar period for a route and return its trip_ids.

    Swiss GTFS encodes seasonal / construction timetable changes as separate
    service_ids with non-overlapping active-weekday date sets.  Pooling all
    trips across the year creates phantom variant interleaving (S7's CV issue)
    and spurious construction-only variants (S16's skipped-stop variant).

    Algorithm:
      1. Collect (service_id → active_weekday_set) for all trips in the route.
      2. Cluster service_ids into calendar periods: two service_ids belong to
         the same period if their active date sets overlap (union-find).
      3. For each period, compute the union of active weekdays.
      4. The dominant period is the one covering the most weekdays.
      5. Return the set of trip_ids whose service_id belongs to the dominant
         period.

    If all service_ids share dates (single period), returns all trip_ids unchanged.
    """
    # Map trip_ids → service_ids
    trip_sids = trips_all.loc[
        trips_all['trip_id'].isin(route_trip_ids),
        ['trip_id', 'service_id']
    ]
    if trip_sids.empty:
        return route_trip_ids

    # Unique service_ids with their active date sets
    sids = trip_sids['service_id'].unique()
    sid_dates = {s: sid_active_sets.get(s, set()) for s in sids}

    # Union-find clustering on overlapping date sets
    parent = {s: s for s in sids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    sid_list = list(sids)
    for i in range(len(sid_list)):
        for j in range(i + 1, len(sid_list)):
            if sid_dates[sid_list[i]] & sid_dates[sid_list[j]]:
                union(sid_list[i], sid_list[j])

    # Group service_ids by cluster
    clusters = defaultdict(set)
    for s in sids:
        clusters[find(s)].add(s)

    if len(clusters) <= 1:
        return route_trip_ids  # single period — use everything

    # Find dominant cluster (most union-weekdays)
    best_root, best_days = None, 0
    for root, members in clusters.items():
        union_days = set()
        for s in members:
            union_days |= sid_dates[s]
        if len(union_days) > best_days:
            best_root, best_days = root, len(union_days)

    dominant_sids = clusters[best_root]
    dominant_trips = set(
        trip_sids.loc[trip_sids['service_id'].isin(dominant_sids), 'trip_id']
    )
    return dominant_trips


def _is_circular(stop_id_seq):
    return len(stop_id_seq) >= 3 and stop_id_seq[0] == stop_id_seq[-1]


def _to_minutes_str(t):
    """Convert 'HH:MM:SS' string to fractional minutes past midnight."""
    try:
        h, m, s = str(t).split(':')
        return int(h) * 60 + int(m) + int(s) / 60
    except Exception:
        return None


def _minutes_to_time_str(m):
    """Convert fractional minutes past midnight to 'HH:MM:SS' string."""
    h = int(m // 60)
    mins = int(m % 60)
    secs = int((m % 1) * 60)
    return f"{h:02d}:{mins:02d}:{secs:02d}"


def _detect_peak_windows(route_trip_ids):
    """Detect adaptive AM and PM peak windows for a (route_id, direction_id).

    Pools all weekday departures for the given trip set. Splits at
    HALFDAY_MIDPOINT, bins into PEAK_BIN_WIDTH_MIN slots, finds the densest
    contiguous cluster above the half-day median, then validates against the
    fixed AM/PM windows.

    When pooled density is flat (all bins equal — common when an extended peak
    variant interleaves with a shorter off-peak variant at the same headway),
    falls back to detecting peaks from the longest stop-sequence variant's
    departures, since the extended variant defines the peak service.

    Returns dict with keys:
        am_start, am_end, pm_start, pm_end, op_start, op_end (str 'HH:MM:SS')
        am_adaptive, pm_adaptive (bool — True if adaptively detected)
    """
    fallback = {
        'am_start': AM_PEAK_START, 'am_end': AM_PEAK_END,
        'pm_start': PM_PEAK_START, 'pm_end': PM_PEAK_END,
        'op_start': OFFPEAK_START, 'op_end': OFFPEAK_END,
        'am_adaptive': False, 'pm_adaptive': False,
    }

    # Get weekday trips in this direction
    time_col = 'departure_time' if 'departure_time' in stop_times.columns else 'arrival_time'
    weekday_trips = set(
        trips_all.loc[
            trips_all['trip_id'].isin(route_trip_ids) &
            trips_all['service_id'].isin(_weekday_service_ids),
            'trip_id'
        ]
    )
    if not weekday_trips:
        return fallback

    sub = stop_times[stop_times['trip_id'].isin(weekday_trips)].copy()
    if sub.empty:
        return fallback

    # Deduplicate first-stop departures (same logic as _get_window_departures)
    # so that peak detection and frequency computation see the same counts.
    _first_all = sub.loc[sub.groupby('trip_id')['stop_sequence_int'].idxmin()].copy()
    _first_all = _first_all.merge(
        trips_all[['trip_id', 'route_id', 'direction_id']],
        on='trip_id', how='left'
    )
    _first_all = _dedup_first_stops(_first_all, time_col)

    midpoint = _to_minutes_str(HALFDAY_MIDPOINT)
    day_start = _to_minutes_str(AM_PEAK_START)   # 360 (06:00)
    day_end = _to_minutes_str(PM_PEAK_END)       # 1140 (19:00)

    def _first_stop_minutes(first_stops_df):
        """Get departure minutes from a pre-deduped first-stops DataFrame."""
        mins = [_to_minutes_str(t) for t in first_stops_df[time_col].dropna()]
        return [m for m in mins if m is not None]

    def _find_peak_cluster(deps, half_start, half_end):
        """Find the densest contiguous cluster of bins above the half-day median."""
        if len(deps) < VARIANT_MIN_DEPARTURES_IN_WINDOW:
            return None, None

        bin_starts = []
        b = half_start
        while b < half_end:
            bin_starts.append(b)
            b += PEAK_BIN_WIDTH_MIN

        if not bin_starts:
            return None, None

        bin_counts = [sum(1 for d in deps if bs <= d < bs + PEAK_BIN_WIDTH_MIN)
                      for bs in bin_starts]

        median_count = sorted(bin_counts)[len(bin_counts) // 2]

        # Mark bins as dense (strictly above median)
        dense = [c > median_count for c in bin_counts]

        if not any(dense):
            return None, None  # flat density — no peak detectable

        # Find contiguous runs of dense bins
        runs = []
        i = 0
        while i < len(dense):
            if dense[i]:
                j = i
                total = 0
                while j < len(dense) and dense[j]:
                    total += bin_counts[j]
                    j += 1
                runs.append((i, j - i, total))
                i = j
            else:
                i += 1

        if not runs:
            return None, None

        best = max(runs, key=lambda r: (r[1], r[2]))
        if best[1] < PEAK_MIN_CLUSTER_BINS:
            return None, None  # cluster too narrow — likely a scheduling artifact
        cluster_start = bin_starts[best[0]]
        cluster_end = bin_starts[best[0] + best[1] - 1] + PEAK_BIN_WIDTH_MIN
        return cluster_start, cluster_end

    # --- Primary: pooled density detection ---
    dep_minutes = _first_stop_minutes(_first_all)
    if len(dep_minutes) < VARIANT_MIN_DEPARTURES_IN_WINDOW:
        return fallback

    am_deps = [m for m in dep_minutes if day_start <= m < midpoint]
    pm_deps = [m for m in dep_minutes if midpoint <= m < day_end]

    am_cluster_start, am_cluster_end = _find_peak_cluster(am_deps, day_start, midpoint)
    pm_cluster_start, pm_cluster_end = _find_peak_cluster(pm_deps, midpoint, day_end)

    # --- Fallback: longest-variant detection ---
    # When pooled density is flat (interleaving variants at same headway),
    # detect peaks from the longest stop-sequence variant instead.
    if am_cluster_start is None or pm_cluster_start is None:
        # Build stop sequences per trip to find the longest variant
        seq_per_trip = (
            sub.sort_values('stop_sequence_int')
            .groupby('trip_id')['stop_id']
            .apply(tuple)
        )
        seq_to_trips = defaultdict(set)
        for tid, seq in seq_per_trip.items():
            seq_to_trips[seq].add(tid)

        if len(seq_to_trips) >= 2:
            # Find the longest variant by stop count
            longest_seq = max(seq_to_trips.keys(), key=len)
            longest_trips = seq_to_trips[longest_seq]

            if len(longest_trips) >= VARIANT_MIN_DEPARTURES_IN_WINDOW:
                longest_first = _first_all[_first_all['trip_id'].isin(longest_trips)]
                longest_deps = _first_stop_minutes(longest_first)

                longest_am = [m for m in longest_deps if day_start <= m < midpoint]
                longest_pm = [m for m in longest_deps if midpoint <= m < day_end]

                if am_cluster_start is None:
                    am_cluster_start, am_cluster_end = _find_peak_cluster(
                        longest_am, day_start, midpoint)
                if pm_cluster_start is None:
                    pm_cluster_start, pm_cluster_end = _find_peak_cluster(
                        longest_pm, midpoint, day_end)

    # --- Validate against fixed windows ---
    am_val_start = _to_minutes_str(AM_PEAK_START)
    am_val_end = _to_minutes_str(AM_PEAK_END)
    am_adaptive = False
    if am_cluster_start is not None:
        overlap = max(0, min(am_cluster_end, am_val_end) - max(am_cluster_start, am_val_start))
        duration = am_cluster_end - am_cluster_start
        if duration > 0 and overlap / duration >= PEAK_MIN_OVERLAP:
            am_start = am_cluster_start
            am_end = am_cluster_end
            am_adaptive = True
        else:
            am_start = am_val_start
            am_end = am_val_end
    else:
        am_start = am_val_start
        am_end = am_val_end

    pm_val_start = _to_minutes_str(PM_PEAK_START)
    pm_val_end = _to_minutes_str(PM_PEAK_END)
    pm_adaptive = False
    if pm_cluster_start is not None:
        overlap = max(0, min(pm_cluster_end, pm_val_end) - max(pm_cluster_start, pm_val_start))
        duration = pm_cluster_end - pm_cluster_start
        if duration > 0 and overlap / duration >= PEAK_MIN_OVERLAP:
            pm_start = pm_cluster_start
            pm_end = pm_cluster_end
            pm_adaptive = True
        else:
            pm_start = pm_val_start
            pm_end = pm_val_end
    else:
        pm_start = pm_val_start
        pm_end = pm_val_end

    # Guard: off-peak must have positive duration
    if am_end >= pm_start:
        return fallback

    return {
        'am_start': _minutes_to_time_str(am_start),
        'am_end':   _minutes_to_time_str(am_end),
        'pm_start': _minutes_to_time_str(pm_start),
        'pm_end':   _minutes_to_time_str(pm_end),
        'op_start': _minutes_to_time_str(am_end),
        'op_end':   _minutes_to_time_str(pm_start),
        'am_adaptive': am_adaptive,
        'pm_adaptive': pm_adaptive,
    }


def _dedup_first_stops(first_stops, time_col):
    """Remove overcounted departures from a first-stops DataFrame.

    Two passes, applied in order:

    1. Deduplicate on (departure_time, stop_id) — collapses trips from
       MO-only supplement IDs and seasonal-split FULL-WEEK IDs that encode the
       same physical departure multiple times.  stop_ids are already at
       parent-station level (platform suffixes resolved earlier).

    2. Time-window dedup within the same (route_id, direction_id): any two
       remaining rows that depart within TERMINAL_MERGE_MINUTES of each other
       are collapsed into one.  This handles two-terminal branch lines where
       direction 0 departs from terminus A at :03 and terminus B at :04 —
       both fill the same service slot and a passenger at any intermediate stop
       sees only one tram.  The merge is scoped to the same route+direction so
       two different lines that happen to share a nearby departure time are
       never collapsed.
    """
    if first_stops.empty:
        return first_stops

    df = first_stops.copy()

    # Pass 1 — dedup on (time, stop_id) — already parent-station level
    df = df.drop_duplicates(subset=[time_col, 'stop_id'])

    # Pass 3 — time-window dedup within (route_id, direction_id)
    if 'route_id' not in df.columns or 'direction_id' not in df.columns:
        return df

    def _merge_close(group):
        group = group.sort_values(time_col).reset_index(drop=True)
        keep = [True] * len(group)
        for i in range(1, len(group)):
            try:
                if _to_minutes_str(group.loc[i, time_col]) - _to_minutes_str(group.loc[i - 1, time_col]) <= TERMINAL_MERGE_MINUTES:
                    keep[i] = False
            except Exception:
                pass
        return group[keep]

    df = (
        df.groupby(['route_id', 'direction_id'], group_keys=False)
          .apply(_merge_close)
          .reset_index(drop=True)
    )

    return df


def _get_window_departures(trip_ids, time_col, window_start, window_end):
    """Get deduplicated first-stop departures within a half-open time window [window_start, window_end)
    for a set of trip_ids. Only includes weekday trips. Returns a DataFrame.
    Returns an empty DataFrame if no weekday trips exist (no weekend fallback)."""
    weekday_trips = set(
        trips_all.loc[
            trips_all['trip_id'].isin(trip_ids) &
            trips_all['service_id'].isin(_weekday_service_ids),
            'trip_id'
        ]
    )
    if not weekday_trips:
        return pd.DataFrame()

    sub = stop_times[stop_times['trip_id'].isin(weekday_trips)].copy()
    first_stops = sub.loc[sub.groupby('trip_id')['stop_sequence_int'].idxmin()].copy()

    # Merge route_id and direction_id for dedup pass 3
    first_stops = first_stops.merge(
        trips_all[['trip_id', 'route_id', 'direction_id']],
        on='trip_id', how='left'
    )

    first_stops = _dedup_first_stops(first_stops, time_col)

    in_window = first_stops[
        (first_stops[time_col] >= window_start) & (first_stops[time_col] < window_end)
    ].copy()

    return in_window


def _snap_to_standard_freq(raw_freq):
    """Snap a raw dep/hr value to the nearest entry in STANDARD_FREQUENCIES."""
    if raw_freq is None:
        return None
    return min(STANDARD_FREQUENCIES, key=lambda f: abs(f - raw_freq))


def _median_freq(departures_df, time_col, min_departures=None):
    """Compute dep/hr from median interior inter-departure gap, snapped to the
    nearest standard frequency. Returns None if fewer than *min_departures*
    departures (defaults to VARIANT_MIN_DEPARTURES_IN_WINDOW)."""
    if min_departures is None:
        min_departures = VARIANT_MIN_DEPARTURES_IN_WINDOW
    if len(departures_df) < min_departures:
        return None

    times = departures_df[time_col].apply(_to_minutes_str).dropna().sort_values().tolist()
    if len(times) < min_departures:
        return None

    gaps = [times[i+1] - times[i] for i in range(len(times)-1)]
    # Drop boundary gaps (first and last) if we have enough interior gaps
    if len(gaps) >= 3:
        gaps = gaps[1:-1]
    if not gaps:
        return None

    median_gap = sorted(gaps)[len(gaps) // 2]
    if median_gap <= 0:
        return None
    return _snap_to_standard_freq(60 / median_gap)


def _compute_frequencies(trip_ids, windows=None, min_departures=None):
    """Compute am_peak, pm_peak, offpeak frequencies for a set of trip_ids.

    Parameters
    ----------
    trip_ids : set
        Trip IDs (single direction).
    windows : dict or None
        Detected peak windows from _detect_peak_windows.
        If None, uses the fixed constants.
    min_departures : int or None
        Minimum departures required per window to compute frequency.
        Defaults to VARIANT_MIN_DEPARTURES_IN_WINDOW.

    Returns dict with keys: freq_am_peak_dep_hr, freq_pm_peak_dep_hr, freq_offpeak_dep_hr.
    """
    time_col = 'departure_time' if 'departure_time' in stop_times.columns else 'arrival_time'

    if windows is not None:
        am_s, am_e = windows['am_start'], windows['am_end']
        pm_s, pm_e = windows['pm_start'], windows['pm_end']
        op_s, op_e = windows['op_start'], windows['op_end']
    else:
        am_s, am_e = AM_PEAK_START, AM_PEAK_END
        pm_s, pm_e = PM_PEAK_START, PM_PEAK_END
        op_s, op_e = OFFPEAK_START, OFFPEAK_END

    am_deps    = _get_window_departures(trip_ids, time_col, am_s, am_e)
    pm_deps    = _get_window_departures(trip_ids, time_col, pm_s, pm_e)
    op_deps    = _get_window_departures(trip_ids, time_col, op_s, op_e)

    return {
        'freq_am_peak_dep_hr':  _median_freq(am_deps,  time_col, min_departures=min_departures),
        'freq_pm_peak_dep_hr':  _median_freq(pm_deps,  time_col, min_departures=min_departures),
        'freq_offpeak_dep_hr':  _median_freq(op_deps,  time_col, min_departures=min_departures),
    }


def _compute_directionality(trip_ids_dir0, trip_ids_dir1, windows=None):
    """Return True if peak frequency differs significantly between directions.

    Directional if, in either peak window:
      - Both directions have measurable frequency and max/min ratio >= 1.5, OR
      - One direction has measurable frequency and the other does not
        (asymmetric service — the strongest form of directionality).
    """
    time_col = 'departure_time' if 'departure_time' in stop_times.columns else 'arrival_time'
    if windows is not None:
        peak_pairs = [(windows['am_start'], windows['am_end']),
                      (windows['pm_start'], windows['pm_end'])]
    else:
        peak_pairs = [(AM_PEAK_START, AM_PEAK_END), (PM_PEAK_START, PM_PEAK_END)]
    for ws, we in peak_pairs:
        f0 = _median_freq(_get_window_departures(trip_ids_dir0, time_col, ws, we), time_col)
        f1 = _median_freq(_get_window_departures(trip_ids_dir1, time_col, ws, we), time_col)
        f0_ok = f0 is not None and f0 > 0
        f1_ok = f1 is not None and f1 > 0
        if f0_ok and f1_ok:
            if max(f0, f1) / min(f0, f1) >= 1.5:
                return True
        elif f0_ok or f1_ok:
            # One direction has meaningful service, the other doesn't
            return True
    return False


def _hex_to_rgba(hex_colour, alpha=255):
    """Convert '#RRGGBB' to QGIS RGBA string 'R,G,B,A'."""
    h = hex_colour.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b},{alpha}"


# --- QGIS project (.qgz) generation ---

_SRS_BLOCK = """<spatialrefsys nativeFormat="wkt">
      <proj4>+proj=somerc +lat_0=46.9524055555556 +lon_0=7.43958333333333 +k_0=1 +x_0=2600000 +y_0=1200000 +ellps=bessel +towgs84=674.374,15.056,405.346,0,0,0,0 +units=m +no_defs</proj4>
      <srsid>47</srsid>
      <srid>2056</srid>
      <authid>EPSG:2056</authid>
      <description>CH1903+ / LV95</description>
      <projectionacronym>somerc</projectionacronym>
      <ellipsoidacronym>bessel</ellipsoidacronym>
    </spatialrefsys>"""


def _line_maplayer(layer_id, gpkg_relpath, layer_name, display_name, rgba, line_style, width='0.5'):
    """Return a <maplayer> XML block for a line layer."""
    pen = 'dash' if line_style == 'dashed' else 'solid'
    return f"""  <maplayer geometry="Line" type="vector" hasScaleBasedVisibilityFlag="0">
    <id>{layer_id}</id>
    <datasource>{gpkg_relpath}|layername={layer_name}</datasource>
    <layername>{display_name}</layername>
    <provider encoding="UTF-8">ogr</provider>
    <srs>{_SRS_BLOCK}</srs>
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


def _marker_maplayer(layer_id, gpkg_relpath, layer_name, display_name, fill_rgba, outline_rgba, size='2', outline_width='0.2'):
    """Return a <maplayer> XML block for a point (stops) layer."""
    return f"""  <maplayer geometry="Point" type="vector" hasScaleBasedVisibilityFlag="0">
    <id>{layer_id}</id>
    <datasource>{gpkg_relpath}|layername={layer_name}</datasource>
    <layername>{display_name}</layername>
    <provider encoding="UTF-8">ogr</provider>
    <srs>{_SRS_BLOCK}</srs>
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


_WMS_LAYER_ID   = 'Swisstopo_National_Map__grey__e16b0296_87b7_4e32_b8e8_b46b5990275e'
_WMS_LAYER_NAME = 'Swisstopo National Map (grey)'
_WMS_SOURCE     = ('contextualWMSLegend=0&amp;crs=EPSG:2056&amp;dpiMode=7'
                   '&amp;featureCount=10&amp;format=image/png'
                   '&amp;layers=ch.swisstopo.pixelkarte-grau'
                   '&amp;styles=&amp;url=http://wms.geo.admin.ch/')


def _wms_maplayer():
    """Return a <maplayer> XML block for the Swisstopo WMS basemap."""
    return f"""  <maplayer type="raster" hasScaleBasedVisibilityFlag="0">
    <id>{_WMS_LAYER_ID}</id>
    <datasource>{_WMS_SOURCE}</datasource>
    <layername>{_WMS_LAYER_NAME}</layername>
    <provider encoding="">wms</provider>
    <srs>{_SRS_BLOCK}</srs>
  </maplayer>"""


def _build_qgz(qgz_path, layers):
    """Write a QGIS .qgz project file containing the given layers.

    Parameters
    ----------
    qgz_path : str
        Output path for the .qgz file.
    layers : list of dict
        Each dict has keys:
            layer_id, gpkg_relpath, layer_name, display_name, geom_type,
            colour (hex), line_style ('solid'/'dashed'), fill_colour (hex),
            outline_colour (hex)
        geom_type is 'line' or 'point'.
    """
    import zipfile

    tree_entries = []
    maplayer_blocks = []

    for lyr in layers:
        lid   = lyr['layer_id']
        src   = f"{lyr['gpkg_relpath']}|layername={lyr['layer_name']}"
        dname = lyr['display_name']

        tree_entries.append(
            f'    <layer-tree-layer id="{lid}" name="{dname}" '
            f'checked="Qt::Checked" expanded="1" source="{src}" providerKey="ogr"/>'
        )

        if lyr['geom_type'] == 'line':
            rgba = _hex_to_rgba(lyr['colour'])
            maplayer_blocks.append(
                _line_maplayer(lid, lyr['gpkg_relpath'], lyr['layer_name'],
                               dname, rgba, lyr.get('line_style', 'solid'))
            )
        else:
            fill_rgba    = _hex_to_rgba(lyr['fill_colour'])
            outline_rgba = _hex_to_rgba(lyr['outline_colour'])
            maplayer_blocks.append(
                _marker_maplayer(lid, lyr['gpkg_relpath'], lyr['layer_name'],
                                 dname, fill_rgba, outline_rgba)
            )

    # Swisstopo WMS basemap — bottom of layer tree (rendered first = background)
    tree_entries.append(
        f'    <layer-tree-layer id="{_WMS_LAYER_ID}" name="{_WMS_LAYER_NAME}" '
        f'checked="Qt::Checked" expanded="0" source="{_WMS_SOURCE}" providerKey="wms"/>'
    )
    maplayer_blocks.append(_wms_maplayer())

    tree_xml     = "\n".join(tree_entries)
    layers_xml   = "\n".join(maplayer_blocks)

    qgs = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis projectname="Network Build" version="3.28.0-Firenze">
  <homePath path=""/>
  <title>Network Build</title>
  <autotransaction active="0"/>
  <evaluateDefaultValues active="0"/>
  <trust active="0"/>
  <projectCrs>
    {_SRS_BLOCK}
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
      {_SRS_BLOCK}
    </destinationsrs>
  </mapcanvas>
</qgis>
"""
    with zipfile.ZipFile(qgz_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('project.qgs', qgs)


# ---------------------------------------------------------------------------
# Step 0 — setup
# ---------------------------------------------------------------------------

os.chdir(paths.MAIN)

_pt_out_dir   = os.path.join(paths.FEEDER_LINES_DIR,   PT_FEEDER_OUTPUT_FOLDER)
_rail_out_dir = os.path.join(paths.RAIL_PROCESSED_DIR,  RAIL_OUTPUT_FOLDER)

pt_stops_path  = os.path.join(_pt_out_dir,   PT_FEEDER_STOPS_FILE)
pt_lines_path = os.path.join(_pt_out_dir,   PT_FEEDER_LINES_FILE)
rl_stops_path  = os.path.join(_rail_out_dir,  RAIL_STOPS_FILE)
rl_lines_path = os.path.join(_rail_out_dir,  RAIL_LINES_FILE)

print("=" * 70)
print("catchment_build_network.py")
print(f"  CATCHMENT_METHOD : {settings.CATCHMENT_METHOD}")
print(f"  CATCHMENT_CANTON : {settings.CATCHMENT_CANTON}")
print(f"  GTFS source      : {os.path.join(paths.GTFS_TRANSIT_DIR, GTFS_INPUT_FOLDER)}")
print(f"  PT-Feeder output : {_pt_out_dir}")
print(f"  Rail output      : {_rail_out_dir}")
print(f"  Spatial CRS      : {CODEBASE_CRS}")
print("=" * 70)

_ensure_dir(_pt_out_dir)
_ensure_dir(_rail_out_dir)


# ---------------------------------------------------------------------------
# Step 1 — load filtered GTFS
# ---------------------------------------------------------------------------

print("\n[1] Loading filtered GTFS files ...")

stops          = _load('stops.txt')
routes_all     = _load('routes.txt')
trips_all      = _load('trips.txt')
stop_times     = _load('stop_times.txt')
mode_class     = _load('mode_class.txt')
calendar       = _load('calendar.txt')
calendar_dates = _load('calendar_dates.txt')

if any(x is None for x in [stops, routes_all, trips_all, stop_times, mode_class, calendar]):
    raise FileNotFoundError(
        "One or more required filtered GTFS files are missing from "
        f"{os.path.join(paths.GTFS_TRANSIT_DIR, GTFS_INPUT_FOLDER)}. "
        "Run catchment_filter_gtfs.py first."
    )

# Service IDs that run on at least WEEKDAY_MIN_DAYS out of Mon–Fri
# Derive timetable validity period from calendar.txt date ranges
_cal_starts = pd.to_datetime(calendar['start_date'], format='%Y%m%d', errors='coerce')
_cal_ends   = pd.to_datetime(calendar['end_date'],   format='%Y%m%d', errors='coerce')
TIMETABLE_START = _cal_starts.min().strftime('%Y-%m-%d')
TIMETABLE_END   = _cal_ends.max().strftime('%Y-%m-%d')
print(f"  Timetable period (from calendar.txt): {TIMETABLE_START} to {TIMETABLE_END}")

_weekday_cols = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']
_wd_counts = calendar[_weekday_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1)
_weekday_service_ids = set(calendar.loc[_wd_counts >= WEEKDAY_MIN_DAYS, 'service_id'])
print(f"  Weekday service IDs (≥{WEEKDAY_MIN_DAYS}/5 days): {len(_weekday_service_ids):,}")


# ---------------------------------------------------------------------------
# Step 1.5 — Tier 2 service duration filter (route-level)
#
# Swiss GTFS often encodes a year-round route across many short-lived
# service_ids (seasonal/construction splits).  Evaluating each service_id in
# isolation would incorrectly drop routes like S7 or IC1.
#
# Instead we:
#   1. Compute the set of active weekdays per service_id.
#   2. Map service_ids → route_ids via trips_all.
#   3. Union the active-weekday sets per route_id.
#   4. Retain all service_ids belonging to routes whose union coverage
#      ≥ MIN_WEEKDAY_ACTIVE_FRACTION of available weekdays.
# ---------------------------------------------------------------------------

print(f"\n[1.5] Service duration filter (Tier 2, route-level): "
      f"min {MIN_WEEKDAY_ACTIVE_FRACTION:.0%} of weekdays ...")

def _compute_weekday_active_sets_t2(cal_df, cal_dates_df, tt_start, tt_end):
    """Return (dict: service_id → set of active weekday dates, int: total_available_weekdays)."""
    start = datetime.strptime(tt_start, '%Y-%m-%d')
    end   = datetime.strptime(tt_end,   '%Y-%m-%d')

    # All weekdays in the timetable period
    all_weekdays = []
    d = start
    while d <= end:
        if d.weekday() < 5:  # Mon–Fri
            all_weekdays.append(d)
        d += timedelta(days=1)
    total_weekdays = len(all_weekdays)

    dow_cols = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']

    cal = cal_df.copy()
    for col in dow_cols:
        cal[col] = pd.to_numeric(cal[col], errors='coerce').fillna(0).astype(int)
    cal['_start'] = pd.to_datetime(cal['start_date'], format='%Y%m%d', errors='coerce')
    cal['_end']   = pd.to_datetime(cal['end_date'],   format='%Y%m%d', errors='coerce')

    # Parse calendar_dates exceptions
    if cal_dates_df is not None and not cal_dates_df.empty:
        cde = cal_dates_df.copy()
        cde['_date'] = pd.to_datetime(cde['date'], format='%Y%m%d', errors='coerce')
        cde['_type'] = pd.to_numeric(cde['exception_type'], errors='coerce').fillna(0).astype(int)
        removals  = cde.loc[cde['_type'] == 2].groupby('service_id')['_date'].apply(set).to_dict()
        additions = cde.loc[cde['_type'] == 1].groupby('service_id')['_date'].apply(set).to_dict()
    else:
        removals, additions = {}, {}

    results = {}
    for _, row in cal.iterrows():
        sid = row['service_id']
        s_start, s_end = row['_start'], row['_end']
        if pd.isna(s_start) or pd.isna(s_end):
            results[sid] = set()
            continue
        svc_removals  = removals.get(sid, set())
        svc_additions = additions.get(sid, set())
        active_dates = set()
        for dt in all_weekdays:
            in_base = (s_start <= dt <= s_end) and (row[dow_cols[dt.weekday()]] == 1)
            if dt in svc_removals:
                active = False
            elif dt in svc_additions:
                active = True
            else:
                active = in_base
            if active:
                active_dates.add(dt)
        results[sid] = active_dates

    return results, total_weekdays

_sid_active_sets, _total_weekdays = _compute_weekday_active_sets_t2(
    calendar, calendar_dates, TIMETABLE_START, TIMETABLE_END
)
_min_days = int(_total_weekdays * MIN_WEEKDAY_ACTIVE_FRACTION)

print(f"  Timetable period       : {TIMETABLE_START} to {TIMETABLE_END}")
print(f"  Available weekdays     : {_total_weekdays}")
print(f"  Threshold              : {MIN_WEEKDAY_ACTIVE_FRACTION:.0%} = {_min_days} days")

# Map service_ids → route_ids via trips
_sid_to_routes = trips_all.groupby('service_id')['route_id'].apply(set).to_dict()

# Union active-weekday sets per route_id
_route_active_days = {}  # route_id → set of active weekday dates
for sid, active_dates in _sid_active_sets.items():
    for rid in _sid_to_routes.get(sid, set()):
        if rid not in _route_active_days:
            _route_active_days[rid] = set()
        _route_active_days[rid] |= active_dates

# Determine which routes pass the threshold
_routes_passing = {rid for rid, dates in _route_active_days.items() if len(dates) >= _min_days}
_routes_failing = set(_route_active_days.keys()) - _routes_passing

print(f"  Lines evaluated        : {len(_route_active_days):,}")
print(f"  Lines passing (≥{_min_days}d)  : {len(_routes_passing):,}")
print(f"  Lines failing          : {len(_routes_failing):,}")

# Retain all service_ids belonging to passing routes
_sids_of_passing_routes = set()
for sid, routes in _sid_to_routes.items():
    if routes & _routes_passing:
        _sids_of_passing_routes.add(sid)

# Also retain service_ids not linked to any trip (edge case — keep them)
_all_trip_sids = set(_sid_to_routes.keys())
_orphan_sids = set(_sid_active_sets.keys()) - _all_trip_sids
_t2_kept_sids = _sids_of_passing_routes | _orphan_sids
_t2_dropped_sids = set(_sid_active_sets.keys()) - _t2_kept_sids

print(f"  Service IDs retained   : {len(_t2_kept_sids):,}")
print(f"  Service IDs dropped    : {len(_t2_dropped_sids):,}")

# Restrict _weekday_service_ids to only Tier 2 survivors
_weekday_service_ids = _weekday_service_ids & _t2_kept_sids
print(f"  Weekday service IDs after Tier 2: {len(_weekday_service_ids):,}")

# Filter trips_all and cascade into stop_times
_trips_before = len(trips_all)
trips_all = trips_all[trips_all['service_id'].isin(_t2_kept_sids)].copy()
_retained_trip_ids = set(trips_all['trip_id'])
stop_times = stop_times[stop_times['trip_id'].isin(_retained_trip_ids)].copy()
print(f"  Trips after Tier 2     : {len(trips_all):,} / {_trips_before:,}")
print(f"  stop_times after Tier 2: {len(stop_times):,}")


# ---------------------------------------------------------------------------
# Step 2 — build stop point geometry from LV95 columns
# ---------------------------------------------------------------------------

print("\n[2] Building stop geometry from LV95 columns (stop_E, stop_N) ...")

stops['stop_E_f'] = pd.to_numeric(stops['stop_E'], errors='coerce')
stops['stop_N_f'] = pd.to_numeric(stops['stop_N'], errors='coerce')

stops_geo = gpd.GeoDataFrame(
    stops,
    geometry=gpd.points_from_xy(stops['stop_E_f'], stops['stop_N_f']),
    crs=CODEBASE_CRS
)

# --- Parent-station normalisation ---
# GTFS parent stations (location_type=1) have stop_id = "Parent<numeric_id>".
# Child stops (platforms) reference them via parent_station.
# We remap all stop_times stop_ids to the parent's numeric ID so that each
# physical station is a single node in the network.

_parents = stops_geo[stops_geo['location_type'] == '1'].copy()
_parents['parent_numeric_id'] = _parents['stop_id'].str.replace('Parent', '', n=1)

_children = stops_geo[stops_geo['location_type'] != '1'].copy()
_child_to_parent = dict(zip(_children['stop_id'], _children['parent_station'].str.replace('Parent', '', n=1)))

# Build stop_coord and stop_name keyed by parent numeric ID
stop_coord = {
    row['parent_numeric_id']: row['geometry']
    for _, row in _parents.iterrows()
    if row['geometry'] is not None and not row['geometry'].is_empty
}

stop_name = {
    row['parent_numeric_id']: row['stop_name']
    for _, row in _parents.iterrows()
    if 'stop_name' in _parents.columns
}

print(f"  Parent stations : {len(_parents):,}")
print(f"  Child stops     : {len(_children):,}")
print(f"  stop_coord keys : {len(stop_coord):,}")
print(f"  stop_name keys  : {len(stop_name):,}")

# Remap stop_times.stop_id to parent numeric ID
stop_times['stop_id'] = stop_times['stop_id'].map(_child_to_parent).fillna(stop_times['stop_id'])
_unmapped = (~stop_times['stop_id'].isin(stop_coord)).sum()
if _unmapped > 0:
    print(f"  WARNING: {_unmapped:,} stop_times rows with unmapped stop_id")
else:
    print(f"  All stop_times stop_ids mapped to parent stations")


# ---------------------------------------------------------------------------
# Step 3 — split by mode class
# ---------------------------------------------------------------------------

print("\n[3] Splitting lines by mode class ...")

routes_all['route_type_int'] = pd.to_numeric(routes_all['route_type'], errors='coerce')
routes_all = routes_all.merge(mode_class[['route_id', 'mode_class']], on='route_id', how='left')

if isinstance(BUILD_MODES, list):
    # Explicit route_type list — pick only matching route_types from either group
    _selected = set(BUILD_MODES)
    pt_route_ids   = set(routes_all.loc[
        (routes_all['mode_class'] == 'pt_feeder') & (routes_all['route_type_int'].isin(_selected)), 'route_id'
    ])
    rail_route_ids = set(routes_all.loc[
        (routes_all['mode_class'] == 'rail') & (routes_all['route_type_int'].isin(_selected)), 'route_id'
    ])
    print(f"  BUILD_MODES filter: route_types {sorted(_selected)}")
elif BUILD_MODES == 'pt_feeder':
    pt_route_ids   = set(routes_all.loc[routes_all['mode_class'] == 'pt_feeder', 'route_id'])
    rail_route_ids = set()
    print("  BUILD_MODES filter: pt_feeder only")
elif BUILD_MODES == 'rail':
    pt_route_ids   = set()
    rail_route_ids = set(routes_all.loc[routes_all['mode_class'] == 'rail', 'route_id'])
    print("  BUILD_MODES filter: rail only")
else:  # 'all'
    pt_route_ids   = set(routes_all.loc[routes_all['mode_class'] == 'pt_feeder', 'route_id'])
    rail_route_ids = set(routes_all.loc[routes_all['mode_class'] == 'rail',      'route_id'])

print(f"  PT-Feeder lines  : {len(pt_route_ids):,}")
print(f"  Rail lines       : {len(rail_route_ids):,}")


# ---------------------------------------------------------------------------
# Step 4 — prepare stop_times
# ---------------------------------------------------------------------------

print("\n[4] Preparing stop_times ...")

stop_times['stop_sequence_int'] = pd.to_numeric(stop_times['stop_sequence'], errors='coerce')

# Filter trips to operational window (AM_PEAK_START–PM_PEAK_END, i.e. 06:00–19:00).
# Only trips whose first-stop departure falls within this window are retained.
# This affects variant classification, frequency computation, and trip-share calculations.
_time_col = 'departure_time' if 'departure_time' in stop_times.columns else 'arrival_time'
_st_seq = stop_times.copy()
_st_seq['_seq_int'] = pd.to_numeric(_st_seq['stop_sequence'], errors='coerce')
_first_dep = (
    _st_seq.loc[_st_seq.groupby('trip_id')['_seq_int'].idxmin(), ['trip_id', _time_col]]
    .set_index('trip_id')[_time_col]
)
_in_window = _first_dep[
    (_first_dep >= AM_PEAK_START) & (_first_dep < PM_PEAK_END)
].index
stop_times = stop_times[stop_times['trip_id'].isin(_in_window)].copy()
print(f"  Trips in operational window ({AM_PEAK_START}–{PM_PEAK_END}): {len(_in_window):,}")

trips_slim = trips_all[['trip_id', 'route_id', 'direction_id', 'service_id']].copy()
stop_times_enriched = stop_times.merge(trips_slim, on='trip_id', how='left')
print(f"  stop_times enriched: {len(stop_times_enriched):,} rows")


# ---------------------------------------------------------------------------
# Step 5 — frequency computation (uses _compute_frequencies, called from build_outputs)
# ---------------------------------------------------------------------------

print("\n[5] Frequency computation ready (median inter-departure gap method) ...")


# ---------------------------------------------------------------------------
# Step 6 — variant classification and core builder
# ---------------------------------------------------------------------------

def _classify_variants(dir_group, allowed_trip_ids=None, relax_cv=False):
    """Classify stop-sequence variants for a (route_id, direction_id) group.

    Applies three metrics:
      A. Trip share >= VARIANT_MIN_TRIP_SHARE
      B. Hour spread >= VARIANT_MIN_HOUR_SPREAD
      C. Headway CV <= VARIANT_HEADWAY_CV_MAX (using weekday trips, all-day departures)
         — skipped when relax_cv=True (bidirectional enforcement: opposite
         direction already passed all three criteria).

    Parameters
    ----------
    dir_group : DataFrame
        stop_times_enriched rows for one (route_id, direction_id).
    allowed_trip_ids : set or None
        If given, only these trip_ids are considered (dominant-period filter).
        If None, all weekday trips in dir_group are used.
    relax_cv : bool
        If True, skip metric C (headway CV). Used for bidirectional
        enforcement when the opposite direction already passed.

    Returns a list of (stop_id_tuple, trip_id_set) for structural variants only,
    sorted by trip count descending (variant_rank=1 is most common).
    """
    dir_group = dir_group.copy()

    # Get weekday trip_ids for this direction
    all_trip_ids = set(dir_group['trip_id'].unique())
    weekday_trip_ids = set(
        trips_all.loc[
            trips_all['trip_id'].isin(all_trip_ids) &
            trips_all['service_id'].isin(_weekday_service_ids),
            'trip_id'
        ]
    )
    # Restrict to dominant-period trips if provided
    if allowed_trip_ids is not None:
        weekday_trip_ids = weekday_trip_ids & allowed_trip_ids

    if not weekday_trip_ids:
        return []  # No weekday trips — skip (no weekend fallback)

    total_trips = len(weekday_trip_ids)
    if total_trips == 0:
        return []

    # Build stop sequence per trip (stop_ids are already parent-station level)
    seq_per_trip = (
        dir_group[dir_group['trip_id'].isin(weekday_trip_ids)]
        .sort_values('stop_sequence_int')
        .groupby('trip_id')['stop_id']
        .apply(tuple)
    )

    # Group trips by sequence
    seq_to_trips = defaultdict(set)
    for trip_id, seq in seq_per_trip.items():
        seq_to_trips[seq].add(trip_id)

    time_col_global = 'departure_time' if 'departure_time' in stop_times.columns else 'arrival_time'

    structural = []
    for seq, trip_ids in seq_to_trips.items():
        # Metric A: trip share
        share = len(trip_ids) / total_trips
        if share < VARIANT_MIN_TRIP_SHARE:
            continue

        # Get first-stop departure times for these trips
        sub = stop_times[stop_times['trip_id'].isin(trip_ids)].copy()
        if sub.empty:
            continue
        first_stops = sub.loc[sub.groupby('trip_id')['stop_sequence_int'].idxmin()].copy()

        # Full dedup (pass 1: exact time+stop_id, pass 2: terminal merge
        # within TERMINAL_MERGE_MINUTES per route+direction) — matches the
        # dedup that _compute_frequencies sees via _get_window_departures.
        first_stops = first_stops.merge(
            trips_all[['trip_id', 'route_id', 'direction_id']],
            on='trip_id', how='left'
        )
        first_stops = _dedup_first_stops(first_stops, time_col_global)

        dep_times = first_stops[time_col_global].apply(_to_minutes_str).dropna().sort_values().tolist()

        # Metric B: hour spread
        hours = set(int(t // 60) % 24 for t in dep_times)
        if len(hours) < VARIANT_MIN_HOUR_SPREAD:
            continue

        # Metric C: headway CV computed per window (AM peak, off-peak, PM peak).
        # A variant passes if its intra-window CV is <= VARIANT_HEADWAY_CV_MAX in
        # at least one window that has enough departures to measure. This prevents
        # penalising lines that legitimately run different headways at different
        # times of day (e.g. 7.5 min peak / 15 min off-peak).
        def _window_cv(w_start, w_end):
            w_times = [t for t in dep_times
                       if _to_minutes_str(w_start) <= t < _to_minutes_str(w_end)]
            if len(w_times) < VARIANT_MIN_DEPARTURES_IN_WINDOW + 1:
                return None  # not enough departures to assess
            gaps = [w_times[i+1] - w_times[i] for i in range(len(w_times)-1)]
            # Drop boundary gaps (first and last) if enough interior gaps remain
            if len(gaps) >= 3:
                gaps = gaps[1:-1]
            if not gaps:
                return None
            mean_gap = sum(gaps) / len(gaps)
            if mean_gap <= 0:
                return None
            try:
                std_gap = statistics.stdev(gaps)
            except Exception:
                std_gap = 0.0
            return std_gap / mean_gap

        cv_am = _window_cv(AM_PEAK_START, AM_PEAK_END)
        cv_op = _window_cv(OFFPEAK_START,  OFFPEAK_END)
        cv_pm = _window_cv(PM_PEAK_START,  PM_PEAK_END)

        # At least one window must have a measurable, regular headway
        # (skipped when relax_cv=True — bidirectional enforcement)
        if not relax_cv:
            measurable = [cv for cv in (cv_am, cv_op, cv_pm) if cv is not None]
            if measurable and min(measurable) > VARIANT_HEADWAY_CV_MAX:
                continue

        structural.append((seq, trip_ids, share))

    # Sort by trip count descending
    structural.sort(key=lambda x: len(x[1]), reverse=True)
    return [(seq, trip_ids) for seq, trip_ids, _ in structural]


def _service_period_tag(trip_ids, windows=None):
    """Return 'peak_only', 'offpeak_only', or 'all_day' based on where departures fall.

    Uses threshold-based classification: if ≥ PEAK_CLASSIFICATION_THRESHOLD of
    departures fall in peak (or off-peak) windows, classify accordingly.
    """
    time_col = 'departure_time' if 'departure_time' in stop_times.columns else 'arrival_time'

    if windows is not None:
        am_s, am_e = windows['am_start'], windows['am_end']
        pm_s, pm_e = windows['pm_start'], windows['pm_end']
        op_s, op_e = windows['op_start'], windows['op_end']
    else:
        am_s, am_e = AM_PEAK_START, AM_PEAK_END
        pm_s, pm_e = PM_PEAK_START, PM_PEAK_END
        op_s, op_e = OFFPEAK_START, OFFPEAK_END

    n_am = len(_get_window_departures(trip_ids, time_col, am_s, am_e))
    n_pm = len(_get_window_departures(trip_ids, time_col, pm_s, pm_e))
    n_op = len(_get_window_departures(trip_ids, time_col, op_s, op_e))

    n_peak = n_am + n_pm
    n_total = n_peak + n_op

    if n_total == 0:
        return 'unknown'

    peak_frac = n_peak / n_total
    op_frac   = n_op / n_total

    if peak_frac >= PEAK_CLASSIFICATION_THRESHOLD:
        return 'peak_only'
    elif op_frac >= PEAK_CLASSIFICATION_THRESHOLD:
        return 'offpeak_only'
    else:
        return 'all_day'


def build_outputs(route_ids, mode_label_map, mode_class_tag):
    """
    Build stops and lines dicts of GeoDataFrames, keyed by route_type.

    For each (route_id, direction_id), variants are classified using three metrics.
    Non-structural variants (depot runs, incidental short-workings) are dropped entirely.
    Each structural variant becomes one feature row.
    Directionality is computed across direction 0 and direction 1 for each line.
    """

    ste        = stop_times_enriched[stop_times_enriched['route_id'].isin(route_ids)].copy()
    routes_sub = routes_all[routes_all['route_id'].isin(route_ids)].copy()

    line_records   = {}   # route_type_int → list of dicts
    stop_ids_by_type = {}  # route_type_int → set of stop_id

    # Pre-compute directionality per route_id
    dir_trip_ids = {}  # route_id → {direction_id: set of trip_ids}
    for route_id, rg in ste.groupby('route_id'):
        dir_trip_ids[route_id] = {}
        for dir_id, dg in rg.groupby('direction_id'):
            dir_trip_ids[route_id][dir_id] = set(dg['trip_id'].unique())

    for route_id, rt_group in ste.groupby('route_id'):
        route_row = routes_sub[routes_sub['route_id'] == route_id]
        if route_row.empty:
            continue
        route_row = route_row.iloc[0]

        route_type_int = int(pd.to_numeric(route_row.get('route_type', None), errors='coerce')
                             ) if pd.notna(route_row.get('route_type', None)) else None
        mode_label = mode_label_map.get(route_type_int, 'Unknown') if route_type_int is not None else 'Unknown'
        colour     = LINE_COLOURS.get(route_type_int, '#888888')
        line_style = LINE_STYLE.get(route_type_int, 'solid')
        agency_id  = route_row.get('agency_id', None)
        short_name = route_row.get('route_short_name', None)

        # --- Line-level acceptance gate ---
        # Pool all weekday trips (both directions) and check minimum service.
        _all_line_trips = set(rt_group['trip_id'].unique())
        _weekday_line_trips = set(
            trips_all.loc[
                trips_all['trip_id'].isin(_all_line_trips) &
                trips_all['service_id'].isin(_weekday_service_ids),
                'trip_id'
            ]
        )
        if _weekday_line_trips:
            _time_col_gate = 'departure_time' if 'departure_time' in stop_times.columns else 'arrival_time'
            # Check ≥ LINE_MIN_DEPARTURES_IN_WINDOW in at least one window
            _n_am = len(_get_window_departures(_weekday_line_trips, _time_col_gate, AM_PEAK_START, AM_PEAK_END))
            _n_pm = len(_get_window_departures(_weekday_line_trips, _time_col_gate, PM_PEAK_START, PM_PEAK_END))
            _n_op = len(_get_window_departures(_weekday_line_trips, _time_col_gate, OFFPEAK_START, OFFPEAK_END))
            if max(_n_am, _n_pm, _n_op) < LINE_MIN_DEPARTURES_IN_WINDOW:
                continue  # too few departures in every window

            # Check hour spread (both directions pooled)
            _gate_sub = stop_times[stop_times['trip_id'].isin(_weekday_line_trips)]
            _gate_first = _gate_sub.loc[_gate_sub.groupby('trip_id')['stop_sequence_int'].idxmin(), _time_col_gate]
            _gate_hours = set()
            for _t in _gate_first.dropna():
                try:
                    _h, _, _ = str(_t).split(':')
                    _gate_hours.add(int(_h) % 24)
                except Exception:
                    pass
            if len(_gate_hours) < LINE_MIN_HOUR_SPREAD:
                continue  # departures not spread across enough hours

            # Relaxed frequency threshold: allow 2-departure windows when the
            # line has ≥ VARIANT_RELAXED_MIN_DEPARTURES in both AM and PM peaks
            # (both directions pooled). This keeps marginal but genuine peak
            # services that have sparse but symmetric peak departures.
            _line_min_deps = VARIANT_MIN_DEPARTURES_IN_WINDOW  # default
            if (_n_am >= VARIANT_RELAXED_MIN_DEPARTURES
                    and _n_pm >= VARIANT_RELAXED_MIN_DEPARTURES):
                _line_min_deps = VARIANT_RELAXED_MIN_DEPARTURES
        else:
            continue  # no weekday trips at all

        d_trips = dir_trip_ids.get(route_id, {})
        dir0_trips = d_trips.get('0', set()) | d_trips.get(0, set())
        dir1_trips = d_trips.get('1', set()) | d_trips.get(1, set())

        directions = sorted(rt_group['direction_id'].dropna().unique().tolist())
        if not directions:
            directions = ['0']

        # --- Phase 0: detect dominant calendar period for this route ---
        # Computed once per route_id so both directions share the same period.
        all_route_trip_ids = set(rt_group['trip_id'].unique())
        dominant_trip_ids = _detect_dominant_period(all_route_trip_ids, _sid_active_sets)

        # --- Phase 1: classify variants and apply short-working suppression ---
        def _departure_set(trip_ids_set):
            """Return set of first-stop departure times (minutes) for a trip set."""
            _tc = 'departure_time' if 'departure_time' in stop_times.columns else 'arrival_time'
            _sub = stop_times[stop_times['trip_id'].isin(trip_ids_set)]
            if _sub.empty:
                return set()
            _first = _sub.loc[_sub.groupby('trip_id')['stop_sequence_int'].idxmin(), _tc]
            result = set()
            for t in _first.dropna():
                try:
                    h, m, _ = t.split(':')
                    result.add(int(h) * 60 + int(m))
                except Exception:
                    pass
            return result

        def _overlap_ratio(deps_short, deps_long):
            """Fraction of deps_short that match a departure in deps_long within 1 min."""
            if not deps_short:
                return 0.0
            matched = sum(
                1 for d in deps_short
                if any(abs(d - dl) <= 1 for dl in deps_long)
            )
            return matched / len(deps_short)

        # Collect variants per direction (using dominant-period trips only)
        dir_variants = {}  # direction_id → list of (seq, trip_ids)
        for direction_id in directions:
            dir_group = rt_group[rt_group['direction_id'] == direction_id].copy()
            variants = _classify_variants(dir_group, allowed_trip_ids=dominant_trip_ids)
            if not variants:
                continue
            dir_variants[direction_id] = variants

        # Short-working suppression within each direction (prefix OR suffix match)
        # plus cross-direction propagation: if a short-working is suppressed in one
        # direction, the corresponding short variant in the opposite direction is
        # also suppressed.
        suppressed_stop_counts = set()  # n_stops values suppressed in any direction
        for direction_id, variants in dir_variants.items():
            suppress = set()
            for i, (seq_i, trips_i) in enumerate(variants):
                for j, (seq_j, trips_j) in enumerate(variants):
                    if i == j or len(seq_i) >= len(seq_j):
                        continue
                    # Check if seq_i is a contiguous prefix or suffix of seq_j
                    is_prefix = seq_j[:len(seq_i)] == seq_i
                    is_suffix = seq_j[len(seq_j) - len(seq_i):] == seq_i
                    if is_prefix or is_suffix:
                        deps_i = _departure_set(trips_i)
                        deps_j = _departure_set(trips_j)
                        if _overlap_ratio(deps_i, deps_j) >= SHORTWORKING_OVERLAP_MIN:
                            suppress.add(i)
            for idx in suppress:
                suppressed_stop_counts.add(len(variants[idx][0]))
            dir_variants[direction_id] = [v for k, v in enumerate(variants) if k not in suppress]

        # Cross-direction propagation: if a short-working was suppressed in any
        # direction, suppress variants with the same stop count in other directions
        # when they overlap with a longer variant (relaxed: uses departure overlap
        # against any longer variant, regardless of prefix/suffix match).
        if suppressed_stop_counts:
            for direction_id, variants in dir_variants.items():
                suppress = set()
                for i, (seq_i, trips_i) in enumerate(variants):
                    if len(seq_i) not in suppressed_stop_counts:
                        continue
                    # Check departure overlap against any longer variant
                    for j, (seq_j, trips_j) in enumerate(variants):
                        if i == j or len(seq_i) >= len(seq_j):
                            continue
                        deps_i = _departure_set(trips_i)
                        deps_j = _departure_set(trips_j)
                        if _overlap_ratio(deps_i, deps_j) >= SHORTWORKING_OVERLAP_MIN:
                            suppress.add(i)
                            break
                if suppress:
                    dir_variants[direction_id] = [v for k, v in enumerate(variants) if k not in suppress]

        # --- Bidirectional enforcement (CV relaxation) ---
        # If one direction has accepted variants but the opposite direction has
        # none, re-run _classify_variants for the missing direction with
        # relax_cv=True (only trip share + hour spread required).  This ensures
        # lines are imported in both directions when at least one direction
        # demonstrates regular service.
        # Track which direction triggered and which was inserted so we can
        # withdraw the insertion if the triggering direction is later dropped
        # by frequency filters.
        _bidir_trigger = None   # direction that originally passed
        _bidir_inserted = None  # direction that was force-inserted
        if len(directions) >= 2:
            _has_0 = bool(dir_variants.get(directions[0]))
            _has_1 = bool(dir_variants.get(directions[1]))
            if _has_0 and not _has_1:
                _dg_miss = rt_group[rt_group['direction_id'] == directions[1]].copy()
                _relaxed = _classify_variants(_dg_miss, allowed_trip_ids=dominant_trip_ids,
                                              relax_cv=True)
                if _relaxed:
                    dir_variants[directions[1]] = _relaxed
                    _bidir_trigger = directions[0]
                    _bidir_inserted = directions[1]
            elif _has_1 and not _has_0:
                _dg_miss = rt_group[rt_group['direction_id'] == directions[0]].copy()
                _relaxed = _classify_variants(_dg_miss, allowed_trip_ids=dominant_trip_ids,
                                              relax_cv=True)
                if _relaxed:
                    dir_variants[directions[0]] = _relaxed
                    _bidir_trigger = directions[1]
                    _bidir_inserted = directions[0]

        # --- Phase 2: detect adaptive peak windows per direction ---
        # Uses only the final accepted variants' trip_ids per direction.
        dir_windows = {}  # direction_id → windows dict or None
        for direction_id in directions:
            variants = dir_variants.get(direction_id)
            if variants:
                accepted_trips = set()
                for _seq, _tids in variants:
                    accepted_trips |= _tids
                dir_windows[direction_id] = _detect_peak_windows(accepted_trips)
            else:
                dir_windows[direction_id] = None
            w = dir_windows[direction_id]
            if w and (w['am_adaptive'] or w['pm_adaptive']):
                print(f"    {route_id} dir={direction_id}: "
                      f"AM=[{w['am_start']}–{w['am_end']}) "
                      f"PM=[{w['pm_start']}–{w['pm_end']}) "
                      f"OP=[{w['op_start']}–{w['op_end']})")

        # Directionality: compare dir 0 vs dir 1 (using detected windows)
        freq_directional = False
        if dir0_trips and dir1_trips:
            freq_directional = _compute_directionality(
                dir0_trips, dir1_trips,
                windows=dir_windows.get(directions[0]) if directions else None
            )

        # Track how many features each direction emits (for bidir withdrawal)
        _dir_feature_counts = {d: 0 for d in directions}

        for direction_id in directions:
            variants = dir_variants.get(direction_id)
            if not variants:
                continue
            dir_group = rt_group[rt_group['direction_id'] == direction_id].copy()

            for variant_rank, (seq, variant_trip_ids) in enumerate(variants, start=1):
                # stop_ids are parent-station level — direct lookup in stop_coord
                geom = _build_linestring(seq, stop_coord)
                if geom is None:
                    continue

                circular        = _is_circular(seq)
                _variant_windows = dir_windows.get(direction_id)
                freqs           = _compute_frequencies(variant_trip_ids, windows=_variant_windows,
                                                       min_departures=_line_min_deps)

                # Drop variants with no computable frequency in any window
                freq_vals = [freqs['freq_am_peak_dep_hr'], freqs['freq_pm_peak_dep_hr'], freqs['freq_offpeak_dep_hr']]
                if all(f is None for f in freq_vals):
                    continue

                # Drop variants whose best frequency is below the minimum threshold
                non_null = [f for f in freq_vals if f is not None]
                if max(non_null) < MIN_FREQUENCY_DEP_HR:
                    continue

                service_period  = _service_period_tag(variant_trip_ids, windows=_variant_windows)

                # Trip share: weekday trips only (stop_times already filtered to 06:00–19:00)
                all_dir_trip_ids = set(dir_group['trip_id'].unique())
                weekday_dir_trips = set(
                    trips_all.loc[
                        trips_all['trip_id'].isin(all_dir_trip_ids) &
                        trips_all['service_id'].isin(_weekday_service_ids),
                        'trip_id'
                    ]
                )
                if not weekday_dir_trips:
                    continue  # No weekday trips — skip (no weekend fallback)
                weekday_variant_trips = variant_trip_ids & weekday_dir_trips
                variant_share = len(weekday_variant_trips) / max(len(weekday_dir_trips), 1)

                # Origin and destination from stop names
                origin      = stop_name.get(seq[0],  '') if seq else ''
                destination = stop_name.get(seq[-1], '') if seq else ''
                long_name   = f"{short_name}: {origin} - {destination}" if short_name else f"{origin} - {destination}"

                # Track stop_ids for this route_type (parent-station level)
                stop_ids_by_type.setdefault(route_type_int, set()).update(seq)

                line_records.setdefault(route_type_int, []).append({
                    'route_id':              route_id,
                    'direction_id':          direction_id,
                    'variant_rank':          variant_rank,
                    'variant_trip_share':    round(variant_share, 3),
                    'line_short_name':       short_name,
                    'origin':                origin,
                    'destination':           destination,
                    'line_long_name':        long_name,
                    'line_type':             route_type_int,
                    'mode_label':            mode_label,
                    'mode_class':            mode_class_tag,
                    'agency_id':             agency_id,
                    'is_circular':           circular,
                    'n_stops':               len(seq),
                    'service_period':        service_period,
                    'freq_am_peak_dep_hr':   freqs['freq_am_peak_dep_hr'],
                    'freq_pm_peak_dep_hr':   freqs['freq_pm_peak_dep_hr'],
                    'freq_offpeak_dep_hr':   freqs['freq_offpeak_dep_hr'],
                    'freq_directional':      freq_directional,
                    'colour':                colour,
                    'line_style':            line_style,
                    'geometry':              geom,
                })
                _dir_feature_counts[direction_id] += 1

                if circular:
                    break

        # --- Bidirectional withdrawal ---
        # If bidirectional enforcement inserted a direction but the triggering
        # direction was subsequently dropped (e.g. by frequency filters),
        # remove the inserted direction's features to avoid one-direction orphans.
        if _bidir_inserted is not None:
            trigger_emitted  = _dir_feature_counts.get(_bidir_trigger, 0)
            inserted_emitted = _dir_feature_counts.get(_bidir_inserted, 0)
            if trigger_emitted == 0 and inserted_emitted > 0:
                if route_type_int in line_records:
                    line_records[route_type_int] = [
                        r for r in line_records[route_type_int]
                        if not (r['route_id'] == route_id and r['direction_id'] == _bidir_inserted)
                    ]
                # Also remove stop_ids contributed by the inserted direction's variants
                if route_type_int in stop_ids_by_type:
                    for seq, _ in dir_variants.get(_bidir_inserted, []):
                        stop_ids_by_type[route_type_int] -= set(seq)

    # Build lines GeoDataFrames per route_type
    lines_by_type = {
        rt: gpd.GeoDataFrame(records, crs=CODEBASE_CRS)
        for rt, records in line_records.items()
    }

    # Build stops GeoDataFrames — only from retained structural variant stops
    stops_by_type = _build_stops_by_type(
        stop_ids_by_type, mode_class_tag, mode_label_map
    )

    return stops_by_type, lines_by_type


def _build_stops_by_type(stop_ids_by_type, mode_class_tag, mode_label_map):
    """
    Build per-mode-type stop GeoDataFrames with colour attributes.

    stop_ids are parent-station numeric IDs (e.g. '8502224').  We look them
    up in the parent-station rows of stops_geo (location_type == '1').

    For PT-feeder: each stop is assigned to exactly one layer based on the
    dominant mode (highest-priority route_type serving it per PT_FEEDER_STOP_HIERARCHY).
    For rail: all stops go into a single layer with white fill / black outline.
    """
    # Collect all stop_ids across all types and which types serve each stop
    stop_to_types = {}  # stop_id → set of route_type_int
    for rt, sids in stop_ids_by_type.items():
        for sid in sids:
            stop_to_types.setdefault(sid, set()).add(rt)

    all_stop_ids = set(stop_to_types.keys())

    # Match against parent stations using their numeric ID
    parents = stops_geo[stops_geo['location_type'] == '1'].copy()
    parents['parent_numeric_id'] = parents['stop_id'].str.replace('Parent', '', n=1)
    stops_sub = parents[parents['parent_numeric_id'].isin(all_stop_ids)].copy()

    keep_cols = [c for c in ['parent_numeric_id', 'stop_name', 'stop_lat', 'stop_lon',
                              'geometry'] if c in stops_sub.columns]
    stops_sub = stops_sub[keep_cols].copy()
    stops_sub = stops_sub.rename(columns={'parent_numeric_id': 'stop_id'})

    if mode_class_tag == 'rail':
        stops_sub['fill_colour']    = RAIL_STOP_FILL
        stops_sub['outline_colour'] = RAIL_STOP_OUTLINE
        # Single layer: use the first (and typically only) route_type key
        rt = next(iter(stop_ids_by_type)) if stop_ids_by_type else 109
        return {rt: stops_sub.reset_index(drop=True)}

    # PT-feeder: assign dominant mode per stop
    def _dominant(sid):
        served = stop_to_types.get(sid, set())
        for rt in PT_FEEDER_STOP_HIERARCHY:
            if rt in served:
                return rt
        return next(iter(served)) if served else None

    stops_sub['dominant_rt'] = stops_sub['stop_id'].map(_dominant)
    stops_sub['fill_colour']    = stops_sub['dominant_rt'].map(
        lambda rt: LINE_COLOURS.get(rt, '#888888')
    )
    stops_sub['outline_colour'] = '#000000'

    stops_by_type = {}
    for rt in stops_sub['dominant_rt'].dropna().unique():
        rt_int = int(rt)
        layer  = stops_sub[stops_sub['dominant_rt'] == rt].drop(
            columns=['dominant_rt']
        ).reset_index(drop=True)
        stops_by_type[rt_int] = layer

    return stops_by_type


# ---------------------------------------------------------------------------
# Step 7 — build PT-Feeder outputs
# ---------------------------------------------------------------------------

print("\n[7] Building PT-Feeder stops and lines ...")
pt_stops_by_type, pt_lines_by_type = build_outputs(pt_route_ids, PT_FEEDER_LINE_TYPES, 'pt_feeder')

n_pt_stops  = sum(len(v) for v in pt_stops_by_type.values())
n_pt_lines = sum(len(v) for v in pt_lines_by_type.values())
print(f"  PT-Feeder stops  : {n_pt_stops:,} across {len(pt_stops_by_type)} mode layer(s)")
print(f"  PT-Feeder lines  : {n_pt_lines:,} features across {len(pt_lines_by_type)} mode layer(s)")


# ---------------------------------------------------------------------------
# Step 8 — build Rail outputs
# ---------------------------------------------------------------------------

print("\n[8] Building Rail stops and lines ...")
rail_stops_by_type, rail_lines_by_type = build_outputs(rail_route_ids, RAIL_LINE_TYPES, 'rail')

n_rail_stops  = sum(len(v) for v in rail_stops_by_type.values())
n_rail_lines = sum(len(v) for v in rail_lines_by_type.values())
print(f"  Rail stops  : {n_rail_stops:,} across {len(rail_stops_by_type)} mode layer(s)")
print(f"  Rail lines  : {n_rail_lines:,} features across {len(rail_lines_by_type)} mode layer(s)")


# ---------------------------------------------------------------------------
# Step 9 — write GeoPackages
# ---------------------------------------------------------------------------

print("\n[9] Writing GeoPackages ...")


def _write_layers(by_type, gpkg_path):
    """Write each route_type as a named layer into a single GeoPackage."""
    written = []
    for rt, gdf in by_type.items():
        if gdf.empty:
            continue
        layer_name = LAYER_NAMES.get(rt, f'type_{rt}')
        gdf.to_file(gpkg_path, layer=layer_name, driver='GPKG')
        written.append(layer_name)
        print(f"    Layer '{layer_name}': {len(gdf):,} features")

    if written:
        print(f"  Wrote {gpkg_path} ({len(written)} layer(s))")
    else:
        print(f"  Skipped {gpkg_path} (no data)")
    return written


print(f"  PT-Feeder lines  → {pt_lines_path}")
_write_layers(pt_lines_by_type, pt_lines_path)

print(f"  PT-Feeder stops  → {pt_stops_path}")
_write_layers(pt_stops_by_type,  pt_stops_path)

print(f"  Rail lines   → {rl_lines_path}")
_write_layers(rail_lines_by_type, rl_lines_path)

print(f"  Rail stops  → {rl_stops_path}")
_write_layers(rail_stops_by_type,  rl_stops_path)


# ---------------------------------------------------------------------------
# Step 9b — write QGIS project file (.qgz) with styled layers
# ---------------------------------------------------------------------------

print("\n[9b] Writing QGIS project file ...")

_qgz_layers = []  # layer descriptors for _build_qgz
_layer_counter = 0

def _add_line_layers(by_type, gpkg_filename, label_map):
    """Register line layers for the QGZ project."""
    global _layer_counter
    for rt, gdf in sorted(by_type.items()):
        if gdf.empty:
            continue
        layer_name = LAYER_NAMES.get(rt, f'type_{rt}')
        _layer_counter += 1
        _qgz_layers.append({
            'layer_id':      f'{layer_name}_lines_{_layer_counter:04d}',
            'gpkg_relpath':  f'./{gpkg_filename}',
            'layer_name':    layer_name,
            'display_name':  f'{label_map.get(rt, layer_name)} Lines',
            'geom_type':     'line',
            'colour':        LINE_COLOURS.get(rt, '#888888'),
            'line_style':    LINE_STYLE.get(rt, 'solid'),
        })

def _add_stop_layers(by_type, gpkg_filename, label_map, mode_class_tag):
    """Register stop layers for the QGZ project."""
    global _layer_counter
    for rt, gdf in sorted(by_type.items()):
        if gdf.empty:
            continue
        layer_name = LAYER_NAMES.get(rt, f'type_{rt}')
        if mode_class_tag == 'rail':
            fill = RAIL_STOP_FILL
            outline = RAIL_STOP_OUTLINE
        else:
            fill = LINE_COLOURS.get(rt, '#888888')
            outline = '#000000'
        _layer_counter += 1
        _qgz_layers.append({
            'layer_id':        f'{layer_name}_stops_{_layer_counter:04d}',
            'gpkg_relpath':    f'./{gpkg_filename}',
            'layer_name':      layer_name,
            'display_name':    f'{label_map.get(rt, layer_name)} Stops',
            'geom_type':       'point',
            'fill_colour':     fill,
            'outline_colour':  outline,
        })

# PT-Feeder QGZ: stops below, lines on top
_pt_qgz_layers = []
_add_stop_layers(pt_stops_by_type,    PT_FEEDER_STOPS_FILE,   PT_FEEDER_LINE_TYPES,  'pt_feeder')
_pt_qgz_layers = list(_qgz_layers)  # snapshot
_qgz_layers.clear(); _layer_counter = 0

_add_line_layers(pt_lines_by_type,  PT_FEEDER_LINES_FILE,  PT_FEEDER_LINE_TYPES)
_pt_qgz_layers += list(_qgz_layers)
_qgz_layers.clear(); _layer_counter = 0

_qgz_path_pt = os.path.join(_pt_out_dir, PT_FEEDER_PROJECT_FILE)
_build_qgz(_qgz_path_pt, _pt_qgz_layers)
print(f"  Wrote {_qgz_path_pt} ({len(_pt_qgz_layers)} layer(s))")

# Rail QGZ: stops below, lines on top
_rl_qgz_layers = []
_add_stop_layers(rail_stops_by_type,  RAIL_STOPS_FILE,        RAIL_LINE_TYPES,       'rail')
_rl_qgz_layers = list(_qgz_layers)
_qgz_layers.clear(); _layer_counter = 0

_add_line_layers(rail_lines_by_type, RAIL_LINES_FILE,      RAIL_LINE_TYPES)
_rl_qgz_layers += list(_qgz_layers)
_qgz_layers.clear(); _layer_counter = 0

_qgz_path_rl = os.path.join(_rail_out_dir, RAIL_PROJECT_FILE)
_build_qgz(_qgz_path_rl, _rl_qgz_layers)
print(f"  Wrote {_qgz_path_rl} ({len(_rl_qgz_layers)} layer(s))")


# ---------------------------------------------------------------------------
# Step 10 — validation
# ---------------------------------------------------------------------------

print("\n[10] Validation ...")

tag_pt_stops  = "PASS" if n_pt_stops  > 0 else "FAIL (empty)"
tag_pt_lines = "PASS" if n_pt_lines > 0 else "FAIL (empty)"
tag_rl_stops  = "PASS" if n_rail_stops  > 0 else "FAIL (empty)"
tag_rl_lines = "PASS" if n_rail_lines > 0 else "FAIL (empty)"

# Null geometry check across all layers
def _null_geom_count(by_type):
    return sum(gdf['geometry'].isna().sum() for gdf in by_type.values() if not gdf.empty)

bad_pt_geom = _null_geom_count(pt_lines_by_type)
bad_rl_geom = _null_geom_count(rail_lines_by_type)
tag_pt_geom = "PASS" if bad_pt_geom == 0 else f"FAIL ({bad_pt_geom} null geometries)"
tag_rl_geom = "PASS" if bad_rl_geom == 0 else f"FAIL ({bad_rl_geom} null geometries)"

# Direction_id check
def _bad_dir_count(by_type):
    count = 0
    for gdf in by_type.values():
        if not gdf.empty and 'direction_id' in gdf.columns:
            count += (~gdf['direction_id'].isin(['0', '1'])).sum()
    return count

bad_dir_pt = _bad_dir_count(pt_lines_by_type)
bad_dir_rl = _bad_dir_count(rail_lines_by_type)
tag_dir_pt = "PASS" if bad_dir_pt == 0 else f"WARN ({bad_dir_pt} unexpected direction_id values)"
tag_dir_rl = "PASS" if bad_dir_rl == 0 else f"WARN ({bad_dir_rl} unexpected direction_id values)"

print(f"  PT-Feeder stops non-empty   : {tag_pt_stops}")
print(f"  PT-Feeder lines non-empty   : {tag_pt_lines}")
print(f"  PT-Feeder line geometries   : {tag_pt_geom}")
print(f"  PT-Feeder direction_id valid: {tag_dir_pt}")
print(f"  Rail stops non-empty        : {tag_rl_stops}")
print(f"  Rail lines non-empty        : {tag_rl_lines}")
print(f"  Rail line geometries        : {tag_rl_geom}")
print(f"  Rail direction_id valid     : {tag_dir_rl}")


# ---------------------------------------------------------------------------
# Step 11 — build report
# ---------------------------------------------------------------------------

elapsed = time.time() - _start_time
print(f"\n[11] Writing {BUILD_REPORT_FILE} ...")


def _rt_breakdown(by_type, label_map):
    if not by_type:
        return ["  (no data)"]
    lines = []
    for rt, gdf in sorted(by_type.items()):
        if gdf.empty:
            continue
        label      = label_map.get(rt, 'Unknown')
        layer_name = LAYER_NAMES.get(rt, f'type_{rt}')
        n_lines    = gdf['route_id'].nunique() if 'route_id' in gdf.columns else len(gdf)
        lines.append(f"    {rt:<6}  {label:<25}  {layer_name:<22}  {n_lines:>5,} lines")
    return lines or ["  (no data)"]


report_lines = [
    "=" * 70,
    "NETWORK BUILD REPORT — catchment_build_network.py",
    "=" * 70,
    "",
    "CONFIGURATION",
    f"  CATCHMENT_METHOD : {settings.CATCHMENT_METHOD}",
    f"  CATCHMENT_CANTON : {settings.CATCHMENT_CANTON}",
    f"  GTFS source      : {os.path.join(paths.GTFS_TRANSIT_DIR, GTFS_INPUT_FOLDER)}",
    f"  Spatial CRS      : {CODEBASE_CRS}",
    f"  Weekday filter   : ≥{WEEKDAY_MIN_DAYS}/5 weekdays",
    "",
    "OUTPUT FILES",
    f"  {pt_stops_path}",
    f"  {pt_lines_path}",
    f"  {rl_stops_path}",
    f"  {rl_lines_path}",
    f"  {_qgz_path_pt}  (QGIS project — PT-feeder)",
    f"  {_qgz_path_rl}  (QGIS project — Rail)",
    "",
    "RECORD COUNTS",
    f"  {'Layer':<35}  {'Features':>8}",
    f"  {'-'*35}  {'-'*8}",
    f"  {'pt_feeder_stops (all layers)':<35}  {n_pt_stops:>8,}",
    f"  {'pt_feeder_lines (all layers)':<35}  {n_pt_lines:>8,}",
    f"  {'rail_stops (all layers)':<35}  {n_rail_stops:>8,}",
    f"  {'rail_lines (all layers)':<35}  {n_rail_lines:>8,}",
    "",
    "PT-FEEDER LINE TYPE BREAKDOWN",
    f"  {'rt':<6}  {'Mode':<25}  {'Layer':<22}  {'Lines':>6}",
    f"  {'-'*6}  {'-'*25}  {'-'*22}  {'-'*6}",
] + _rt_breakdown(pt_lines_by_type, PT_FEEDER_LINE_TYPES) + [
    "",
    "RAIL LINE TYPE BREAKDOWN",
    f"  {'rt':<6}  {'Mode':<25}  {'Layer':<22}  {'Lines':>6}",
    f"  {'-'*6}  {'-'*25}  {'-'*22}  {'-'*6}",
] + _rt_breakdown(rail_lines_by_type, RAIL_LINE_TYPES) + [
    "",
    "REPRESENTATIVE TRIP SELECTION",
    "  Structural variants only — classified by trip share, hour spread,",
    "  and headway CV. Non-structural variants (depot runs, incidental",
    "  short-workings) are dropped. Variants with NULL frequency in all",
    "  windows or max frequency < {0} dep/hr are also dropped.".format(MIN_FREQUENCY_DEP_HR),
    "  Each structural variant becomes one feature row. Circular lines",
    "  produce one feature only.",
    "",
    "FREQUENCY",
    f"  AM peak validation window : {AM_PEAK_START} – {AM_PEAK_END}",
    f"  PM peak validation window : {PM_PEAK_START} – {PM_PEAK_END}",
    f"  Off-peak fallback window  : {OFFPEAK_START} – {OFFPEAK_END}",
    "  Peak detection             : adaptive per (route_id, direction_id)",
    f"  Detection bin width        : {PEAK_BIN_WIDTH_MIN} min",
    f"  Min cluster width          : {PEAK_MIN_CLUSTER_BINS} bins ({PEAK_MIN_CLUSTER_BINS * PEAK_BIN_WIDTH_MIN} min)",
    f"  Validation min overlap     : {PEAK_MIN_OVERLAP:.0%}",
    f"  Classification threshold   : {PEAK_CLASSIFICATION_THRESHOLD:.0%}",
    f"  Min deps per window        : {VARIANT_MIN_DEPARTURES_IN_WINDOW} (relaxed to {VARIANT_RELAXED_MIN_DEPARTURES} when ≥{VARIANT_RELAXED_MIN_DEPARTURES} deps in both AM and PM)",
    f"  Weekday filter             : service_ids running on ≥{WEEKDAY_MIN_DAYS} of Mon–Fri",
    "  Method                     : median inter-departure gap (interior gaps, boundary gaps dropped)",
    "  Columns                    : freq_am_peak_dep_hr, freq_pm_peak_dep_hr, freq_offpeak_dep_hr",
    "                               freq_directional (True if peak freq differs ≥1.5× between directions)",
    "  Variants                   : structural only (trip share, hour spread, headway CV filters applied)",
    "  Column                     : service_period (all_day / peak_only / offpeak_only / unknown)",
    "",
    "GEOMETRY",
    "  Straight-line stop-to-stop segments (shapes.txt not used)",
    "",
    "STYLING",
    "  colour / line_style columns written to line layers",
    "  fill_colour / outline_colour columns written to stop layers",
    "  QGIS .qgz project files with pre-configured symbology per output dir",
    "",
    "VALIDATION",
    f"  PT-Feeder stops non-empty   : {tag_pt_stops}",
    f"  PT-Feeder lines non-empty   : {tag_pt_lines}",
    f"  PT-Feeder line geometries   : {tag_pt_geom}",
    f"  PT-Feeder direction_id valid: {tag_dir_pt}",
    f"  Rail stops non-empty        : {tag_rl_stops}",
    f"  Rail lines non-empty        : {tag_rl_lines}",
    f"  Rail line geometries        : {tag_rl_geom}",
    f"  Rail direction_id valid     : {tag_dir_rl}",
    "",
    "RUNTIME",
    f"  Total elapsed                : {elapsed:.1f} seconds",
    "",
    "=" * 70,
]

report_text = "\n".join(report_lines)
report_path = os.path.join(_pt_out_dir, BUILD_REPORT_FILE)
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_text)

print(report_text)
print("\nDone.")
