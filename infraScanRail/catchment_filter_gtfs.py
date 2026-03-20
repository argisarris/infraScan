# catchment_filter_gtfs.py
#
# GTFS stop coordinates are always WGS84 (EPSG:4326) and are reprojected to EPSG:2056
# before spatial filtering.

import os
import shutil
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

import paths
import settings

_start_time = time.time()

# ---------------------------------------------------------------------------
# Input / output folder names — adjust here to filter a different GTFS source
# ---------------------------------------------------------------------------
GTFS_INPUT_FOLDER  = 'GTFS_FP2026_CH_raw'
GTFS_OUTPUT_FOLDER = 'GTFS_FP2026_ZH'

# ---------------------------------------------------------------------------
# Route-type lookups — kept separate so Task 2 can differentiate the two groups
# ---------------------------------------------------------------------------
PT_FEEDER_ROUTE_TYPES = {
    900:  'Tram',
    401:  'Metro',
    700:  'Bus',
    702:  'Express Bus',
    715:  'On-demand Bus',
    1000: 'Ship',
    1400: 'Funicular',
}

RAIL_ROUTE_TYPES = {
    102: 'Long-Distance Rail',
    103: 'Inter-Regional Rail',
    106: 'Regional Rail',
    109: 'S-Bahn / Suburban Rail',
}

# Combined set used for the route-type filter pass
ALL_RETAINED_ROUTE_TYPES = {**PT_FEEDER_ROUTE_TYPES, **RAIL_ROUTE_TYPES}

# Required GTFS files — raise if any are absent
REQUIRED_FILES = [
    'stops.txt',
    'routes.txt',
    'trips.txt',
    'stop_times.txt',
    'calendar.txt',
    'calendar_dates.txt',
    'agency.txt',
]

# Optional GTFS files handled explicitly
OPTIONAL_FILES = ['transfers.txt', 'feed_info.txt']

CODEBASE_CRS = 'EPSG:2056'
GTFS_CRS     = 'EPSG:4326'

# Tier 1 service duration filter: drop service_ids active on fewer than this
# many weekday days across the full timetable year. Removes one-off events,
# single-weekend specials, and very short construction-period services.
MIN_WEEKDAY_ACTIVE_DAYS = 20


# ===========================================================================
# Helpers
# ===========================================================================

def _gtfs_path(filename):
    return os.path.join(paths.GTFS_TRANSIT_DIR, GTFS_INPUT_FOLDER, filename)


def _out_path(filename):
    return os.path.join(paths.GTFS_TRANSIT_DIR, GTFS_OUTPUT_FOLDER, filename)


def _load_required(filename):
    fpath = _gtfs_path(filename)
    if not os.path.isfile(fpath):
        raise FileNotFoundError(
            f"Required GTFS file not found: {fpath}\n"
            f"Please place all required GTFS files in: {os.path.join(paths.GTFS_TRANSIT_DIR, GTFS_INPUT_FOLDER)}"
        )
    print(f"  Loading {filename} ...", end=' ', flush=True)
    df = pd.read_csv(fpath, dtype=str, low_memory=False)
    print(f"{len(df):,} rows")
    return df


def _load_optional(filename):
    """Returns (DataFrame, True) if present, (None, False) if absent."""
    fpath = _gtfs_path(filename)
    if not os.path.isfile(fpath):
        return None, False
    print(f"  Loading {filename} ...", end=' ', flush=True)
    df = pd.read_csv(fpath, dtype=str, low_memory=False)
    print(f"{len(df):,} rows")
    return df, True


def _write(df, filename):
    out = _out_path(filename)
    df.to_csv(out, index=False, encoding='utf-8')
    print(f"  Wrote {filename}: {len(df):,} rows")


# ===========================================================================
# Step 0 — prepare output directory
# ===========================================================================

os.chdir(paths.MAIN)  # All path constants in paths.py are relative to MAIN

print("=" * 70)
print("catchment_filter_gtfs.py")
print(f"  CATCHMENT_METHOD : {settings.CATCHMENT_METHOD}")
print(f"  CATCHMENT_CANTON : {settings.CATCHMENT_CANTON}")
print(f"  GTFS source      : {os.path.join(paths.GTFS_TRANSIT_DIR, GTFS_INPUT_FOLDER)}")
print(f"  GTFS output      : {os.path.join(paths.GTFS_TRANSIT_DIR, GTFS_OUTPUT_FOLDER)}")
print(f"  Spatial CRS      : {CODEBASE_CRS}")
print("=" * 70)

_gtfs_output_dir = os.path.join(paths.GTFS_TRANSIT_DIR, GTFS_OUTPUT_FOLDER)
if os.path.isdir(_gtfs_output_dir):
    print(f"\n[0] Output directory exists — removing for clean overwrite: {_gtfs_output_dir}")
    shutil.rmtree(_gtfs_output_dir)
os.makedirs(_gtfs_output_dir)
print(f"[0] Created output directory: {_gtfs_output_dir}")


# ===========================================================================
# Step 1 — boundary preparation
# ===========================================================================

print(f"\n[1] Loading cantonal boundary from: {paths.CANTON_BOUNDARIES_GPKG}")

# Inspect available layers
available_layers = gpd.list_layers(paths.CANTON_BOUNDARIES_GPKG)
print(f"  Available layers in GeoPackage:")
for _, row in available_layers.iterrows():
    print(f"    - {row['name']}")

# Load first (or only) layer
layer_name = available_layers.iloc[0]['name']
print(f"  Using layer: '{layer_name}'")
cantons_raw = gpd.read_file(paths.CANTON_BOUNDARIES_GPKG, layer=layer_name)

print(f"  Columns: {list(cantons_raw.columns)}")
print(f"  CRS as read from GeoPackage: {cantons_raw.crs}")

canton_col = 'name'
print(f"  Canton name column: '{canton_col}'")
print(f"  Sample canton name values: {sorted(cantons_raw[canton_col].dropna().unique().tolist())}")

# Filter to the target canton
canton_mask = cantons_raw[canton_col].isin(settings.CATCHMENT_CANTON)
canton_rows = cantons_raw[canton_mask]
if canton_rows.empty:
    raise ValueError(
        f"No rows found where column '{canton_col}' matches {settings.CATCHMENT_CANTON}.\n"
        f"Available values: {sorted(cantons_raw[canton_col].dropna().unique().tolist())}\n"
        f"Check CATCHMENT_CANTON in settings.py."
    )

boundary_raw_crs = cantons_raw.crs

# Dissolve to single polygon, reproject to codebase CRS — no buffer
canton_boundary = canton_rows.dissolve().to_crs(CODEBASE_CRS)
canton_polygon = canton_boundary.geometry.iloc[0]
print(f"  Canton boundary dissolved and reprojected to {CODEBASE_CRS}")
print(f"  Boundary envelope: {canton_polygon.bounds}")


# ===========================================================================
# Step 2 — load required GTFS files
# ===========================================================================

print("\n[2] Loading required GTFS files ...")
for f in REQUIRED_FILES:
    if not os.path.isfile(_gtfs_path(f)):
        raise FileNotFoundError(
            f"Required GTFS file not found: {_gtfs_path(f)}\n"
            f"Place all required GTFS files in: {os.path.join(paths.GTFS_TRANSIT_DIR, GTFS_INPUT_FOLDER)}"
        )

stops          = _load_required('stops.txt')
routes_raw     = _load_required('routes.txt')
trips_raw      = _load_required('trips.txt')
stop_times_raw = _load_required('stop_times.txt')
calendar_raw   = _load_required('calendar.txt')
cal_dates_raw  = _load_required('calendar_dates.txt')
agency_raw     = _load_required('agency.txt')

orig_counts = {
    'stops':    len(stops),
    'routes':   len(routes_raw),
    'trips':    len(trips_raw),
    'agencies': len(agency_raw),
}

# Derive timetable validity period from calendar.txt date ranges
_cal_starts = pd.to_datetime(calendar_raw['start_date'], format='%Y%m%d', errors='coerce')
_cal_ends   = pd.to_datetime(calendar_raw['end_date'],   format='%Y%m%d', errors='coerce')
TIMETABLE_START = _cal_starts.min().strftime('%Y-%m-%d')
TIMETABLE_END   = _cal_ends.max().strftime('%Y-%m-%d')
print(f"\n  Timetable period (from calendar.txt): {TIMETABLE_START} to {TIMETABLE_END}")


# ===========================================================================
# Step 2.5 — service duration filter (Tier 1)
#
# Compute effective weekday active days per service_id using calendar.txt +
# calendar_dates.txt, and drop service_ids below MIN_WEEKDAY_ACTIVE_DAYS.
# This removes one-off events, single-weekend specials, and very short
# construction-period services before any downstream processing.
# ===========================================================================

print(f"\n[2.5] Service duration filter (Tier 1): min {MIN_WEEKDAY_ACTIVE_DAYS} weekday active days ...")

def _compute_weekday_active_days(cal_df, cal_dates_df, tt_start, tt_end):
    """Return a Series mapping service_id → number of effective weekday (Mon–Fri)
    active days, accounting for calendar_dates exceptions."""
    start = datetime.strptime(tt_start, '%Y%m%d' if len(tt_start) == 8 else '%Y-%m-%d')
    end   = datetime.strptime(tt_end,   '%Y%m%d' if len(tt_end)   == 8 else '%Y-%m-%d')

    # Build full date range
    all_dates = []
    d = start
    while d <= end:
        all_dates.append(d)
        d += timedelta(days=1)

    # Day-of-week column names in calendar.txt (0=Mon ... 6=Sun)
    dow_cols = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']

    # Parse calendar base schedule
    cal = cal_df.copy()
    for col in dow_cols:
        cal[col] = pd.to_numeric(cal[col], errors='coerce').fillna(0).astype(int)
    cal['_start'] = pd.to_datetime(cal['start_date'], format='%Y%m%d', errors='coerce')
    cal['_end']   = pd.to_datetime(cal['end_date'],   format='%Y%m%d', errors='coerce')

    # Parse calendar_dates exceptions
    cde = cal_dates_df.copy()
    cde['_date'] = pd.to_datetime(cde['date'], format='%Y%m%d', errors='coerce')
    cde['_type'] = pd.to_numeric(cde['exception_type'], errors='coerce').fillna(0).astype(int)

    # Build removal and addition sets per service_id
    removals = cde.loc[cde['_type'] == 2].groupby('service_id')['_date'].apply(set).to_dict()
    additions = cde.loc[cde['_type'] == 1].groupby('service_id')['_date'].apply(set).to_dict()

    results = {}
    for _, row in cal.iterrows():
        sid = row['service_id']
        s_start = row['_start']
        s_end   = row['_end']
        if pd.isna(s_start) or pd.isna(s_end):
            results[sid] = 0
            continue

        svc_removals  = removals.get(sid, set())
        svc_additions = additions.get(sid, set())

        count = 0
        for dt in all_dates:
            if dt.weekday() >= 5:  # Skip Sat/Sun — we only count weekdays
                continue
            # Base schedule: is this date within range and matching day-of-week?
            in_base = (s_start <= dt <= s_end) and (row[dow_cols[dt.weekday()]] == 1)
            # Apply exceptions
            if dt in svc_removals:
                active = False
            elif dt in svc_additions:
                active = True
            else:
                active = in_base
            if active:
                count += 1
        results[sid] = count

    return pd.Series(results, name='weekday_active_days')


_tt_start = TIMETABLE_START.replace('-', '')
_tt_end   = TIMETABLE_END.replace('-', '')

# Use raw calendar/dates (before any filtering)
_weekday_active = _compute_weekday_active_days(calendar_raw, cal_dates_raw, _tt_start, _tt_end)
_n_before = len(calendar_raw)
_dropped_sids = set(_weekday_active[_weekday_active < MIN_WEEKDAY_ACTIVE_DAYS].index)
_kept_sids = set(_weekday_active[_weekday_active >= MIN_WEEKDAY_ACTIVE_DAYS].index)

# Also keep service_ids that appear only in calendar_dates (no calendar.txt row)
# — these are already edge cases; Tier 1 only filters what it can measure.
_cal_only_sids = set(cal_dates_raw['service_id']) - set(calendar_raw['service_id'])

print(f"  Service IDs in calendar.txt: {len(calendar_raw):,}")
print(f"  Service IDs below threshold: {len(_dropped_sids):,}")
print(f"  Service IDs retained       : {len(_kept_sids):,}")
if _cal_only_sids:
    print(f"  Service IDs only in calendar_dates (unmeasured, kept): {len(_cal_only_sids):,}")

# Filter trips to retained service_ids
_retained_service_ids_t1 = _kept_sids | _cal_only_sids
_trips_before_t1 = len(trips_raw)
trips_raw = trips_raw[trips_raw['service_id'].isin(_retained_service_ids_t1)].copy()
print(f"  Trips after Tier 1 filter  : {len(trips_raw):,} / {_trips_before_t1:,}")

# Store stats for the report
_tier1_stats = {
    'total_sids':   len(_weekday_active),
    'dropped_sids': len(_dropped_sids),
    'kept_sids':    len(_kept_sids),
    'min_threshold': MIN_WEEKDAY_ACTIVE_DAYS,
    'trips_before': _trips_before_t1,
    'trips_after':  len(trips_raw),
}


# ===========================================================================
# Step 3 — route type filtering (always before spatial filtering)
# ===========================================================================

print("\n[3] Route-type filtering ...")
routes_raw['route_type_int'] = pd.to_numeric(routes_raw['route_type'], errors='coerce')
routes = routes_raw[routes_raw['route_type_int'].isin(ALL_RETAINED_ROUTE_TYPES)].copy()

# Tag each route with its mode class — carried through for Task 2, not written to GTFS
routes['mode_class'] = routes['route_type_int'].apply(
    lambda rt: 'rail' if int(rt) in RAIL_ROUTE_TYPES else 'pt_feeder'
)
routes = routes.drop(columns=['route_type_int'])

n_pt  = (routes['mode_class'] == 'pt_feeder').sum()
n_rl  = (routes['mode_class'] == 'rail').sum()
print(f"  Routes after route-type filter: {len(routes):,} / {len(routes_raw):,}"
      f"  ({n_pt:,} pt_feeder, {n_rl:,} rail)")

retained_route_ids = set(routes['route_id'])

# Cascade into trips so subsequent steps work on retained routes only
trips_typed = trips_raw[trips_raw['route_id'].isin(retained_route_ids)].copy()
print(f"  Trips after route-type cascade: {len(trips_typed):,} / {len(trips_raw):,}")


# ===========================================================================
# Step 4 — spatial filtering: stops within canton
# ===========================================================================

print(f"\n[4] Spatial filtering of stops to {settings.CATCHMENT_CANTON} ...")

stops['stop_lon_f'] = pd.to_numeric(stops['stop_lon'], errors='coerce')
stops['stop_lat_f'] = pd.to_numeric(stops['stop_lat'], errors='coerce')
stops_geo = gpd.GeoDataFrame(
    stops,
    geometry=[Point(lon, lat) for lon, lat in zip(stops['stop_lon_f'], stops['stop_lat_f'])],
    crs=GTFS_CRS  # GTFS coordinates are always WGS84
)
stops_geo = stops_geo.to_crs(CODEBASE_CRS)

stops_in_canton = stops_geo[stops_geo.geometry.within(canton_polygon)].copy()

# Add Swiss coordinate columns (EPSG:2056 convention: N = northing/y, E = easting/x)
stops_in_canton['stop_N'] = stops_in_canton.geometry.y
stops_in_canton['stop_E'] = stops_in_canton.geometry.x

# Drop helper and geometry columns; retain original stop_lon / stop_lat (WGS84)
stops_filtered = stops_in_canton.drop(
    columns=['stop_lon_f', 'stop_lat_f', 'geometry'], errors='ignore'
)

retained_stop_ids = set(stops_in_canton['stop_id'])
print(f"  Stops within canton: {len(stops_filtered):,} / {len(stops):,}")


# ===========================================================================
# Step 5 — filter stop_times to retained stops
# ===========================================================================

print("\n[5] Filtering stop_times to retained stops ...")
stop_times = stop_times_raw[stop_times_raw['stop_id'].isin(retained_stop_ids)].copy()
print(f"  stop_times rows retained: {len(stop_times):,} / {len(stop_times_raw):,}")


# ===========================================================================
# Step 6 — filter trips: keep any trip with at least one stop within canton,
#           truncating out-of-canton stages (stop_times already clipped in Step 5)
# ===========================================================================

print("\n[6] Filtering trips — truncating cross-boundary trips at canton border ...")

# Trips that have at least one stop within the canton (after route-type filter)
trips_with_canton_stops = set(
    stop_times[stop_times['trip_id'].isin(trips_typed['trip_id'])]['trip_id']
)
trips = trips_typed[trips_typed['trip_id'].isin(trips_with_canton_stops)].copy()

trip_total_counts    = stop_times_raw.groupby('trip_id')['stop_id'].count()
trip_retained_counts = (
    stop_times_raw[stop_times_raw['stop_id'].isin(retained_stop_ids)]
    .groupby('trip_id')['stop_id'].count()
)
fully_within_ids = set(trip_total_counts.index[
    trip_total_counts == trip_retained_counts.reindex(trip_total_counts.index, fill_value=0)
])
n_fully_within = trips_typed['trip_id'].isin(fully_within_ids).sum()
n_truncated = len(trips) - n_fully_within

print(f"  Trips retained (typed set)  : {len(trips_typed):,}")
print(f"  Trips with ≥1 canton stop   : {len(trips):,}")
print(f"    of which fully within     : {n_fully_within:,}")
print(f"    of which truncated        : {n_truncated:,}  (out-of-canton stages dropped)")

# stop_times is already clipped to canton stops from Step 5;
# restrict further to only the retained trip IDs
stop_times = stop_times[stop_times['trip_id'].isin(trips_with_canton_stops)].copy()
print(f"  stop_times after truncation : {len(stop_times):,} rows")


# ===========================================================================
# Step 7 — filter routes to those with at least one retained trip
# ===========================================================================

print("\n[7] Filtering routes to those with retained trips ...")
active_route_ids = set(trips['route_id'])
routes = routes[routes['route_id'].isin(active_route_ids)].copy()
print(f"  Routes retained: {len(routes):,}")


# ===========================================================================
# Step 8 — filter agencies
# ===========================================================================

print("\n[8] Filtering agencies to those operating retained routes ...")
active_agency_ids = set(routes['agency_id'].dropna())
agency = agency_raw[agency_raw['agency_id'].isin(active_agency_ids)].copy()
print(f"  Agencies retained: {len(agency):,} / {len(agency_raw):,}")


# ===========================================================================
# Step 9 — filter calendar and calendar_dates
# ===========================================================================

print("\n[9] Filtering calendar and calendar_dates to retained service IDs ...")
retained_service_ids = set(trips['service_id'])

calendar = calendar_raw[calendar_raw['service_id'].isin(retained_service_ids)].copy()
cal_dates = cal_dates_raw[cal_dates_raw['service_id'].isin(retained_service_ids)].copy()
print(f"  calendar rows retained: {len(calendar):,} / {len(calendar_raw):,}")
print(f"  calendar_dates rows retained: {len(cal_dates):,} / {len(cal_dates_raw):,}")


# ===========================================================================
# Step 10 — optional files
# ===========================================================================

print("\n[10] Processing optional GTFS files ...")
skipped_optional = []

transfers, transfers_present = _load_optional('transfers.txt')
if transfers_present:
    transfers = transfers[
        transfers['from_stop_id'].isin(retained_stop_ids) &
        transfers['to_stop_id'].isin(retained_stop_ids)
    ].copy()
    print(f"    transfers.txt retained: {len(transfers):,} rows")
else:
    skipped_optional.append('transfers.txt')
    print("    transfers.txt: not present — skipped")

feed_info, feed_info_present = _load_optional('feed_info.txt')
if feed_info_present:
    print(f"    feed_info.txt: copied unchanged ({len(feed_info):,} rows)")
else:
    skipped_optional.append('feed_info.txt')
    print("    feed_info.txt: not present — skipped")


# ===========================================================================
# Step 11 — write output
# ===========================================================================

print(f"\n[11] Writing filtered GTFS to: {_gtfs_output_dir}")

# Write mode_class sidecar before stripping the column from routes
mode_class = routes[['route_id', 'mode_class']].copy()
_write(mode_class, 'mode_class.txt')

# Strip mode_class before writing standard GTFS routes.txt
routes_gtfs = routes.drop(columns=['mode_class'])
_write(stops_filtered,  'stops.txt')
_write(routes_gtfs,     'routes.txt')
_write(trips,           'trips.txt')
_write(stop_times,      'stop_times.txt')
_write(calendar,        'calendar.txt')
_write(cal_dates,       'calendar_dates.txt')
_write(agency,          'agency.txt')

if transfers_present:
    _write(transfers, 'transfers.txt')
if feed_info_present:
    _write(feed_info, 'feed_info.txt')


# ===========================================================================
# Step 12 — validation check
# ===========================================================================

print("\n[12] Running self-consistency validation ...")

# Re-load written outputs for a clean check
stops_check      = pd.read_csv(_out_path('stops.txt'),      dtype=str)
routes_check     = pd.read_csv(_out_path('routes.txt'),     dtype=str)
trips_check      = pd.read_csv(_out_path('trips.txt'),      dtype=str)
stop_times_check = pd.read_csv(_out_path('stop_times.txt'), dtype=str)
calendar_check   = pd.read_csv(_out_path('calendar.txt'),   dtype=str)
cal_dates_check  = pd.read_csv(_out_path('calendar_dates.txt'), dtype=str)

valid_stop_ids    = set(stops_check['stop_id'])
valid_route_ids   = set(routes_check['route_id'])
valid_service_ids = set(calendar_check['service_id']) | set(cal_dates_check['service_id'])

# Check 1: stop_times → stops
bad_stop_times = stop_times_check[~stop_times_check['stop_id'].isin(valid_stop_ids)]
n1 = len(bad_stop_times)
tag1 = "PASS" if n1 == 0 else f"FAIL ({n1} violation(s))"
print(f"  stop_times → stops       : {tag1}")

# Check 2: trips → routes
bad_trips_route = trips_check[~trips_check['route_id'].isin(valid_route_ids)]
n2 = len(bad_trips_route)
tag2 = "PASS" if n2 == 0 else f"FAIL ({n2} violation(s))"
print(f"  trips → routes           : {tag2}")

# Check 3: trips → service IDs
bad_trips_service = trips_check[~trips_check['service_id'].isin(valid_service_ids)]
n3 = len(bad_trips_service)
tag3 = "PASS" if n3 == 0 else f"FAIL ({n3} violation(s))"
print(f"  trips → service IDs      : {tag3}")


# ===========================================================================
# Step 13 — filter report
# ===========================================================================

print(f"\n[13] Writing filter_report.txt ...")

# Route-type breakdown of retained routes
routes_check['route_type_int'] = pd.to_numeric(routes_check['route_type'], errors='coerce')
route_type_counts = routes_check['route_type_int'].value_counts().sort_index()

report_lines = [
    "=" * 70,
    "GTFS FILTER REPORT — catchment_filter_gtfs.py",
    "=" * 70,
    "",
    "CONFIGURATION",
    f"  CATCHMENT_METHOD : {settings.CATCHMENT_METHOD}",
    f"  CATCHMENT_CANTON : {settings.CATCHMENT_CANTON}",
    f"  Matched against column '{canton_col}' in CANTON_BOUNDARIES_GPKG",
    "",
    "COORDINATE REFERENCE SYSTEMS",
    f"  CRS of cantonal boundary as read from GeoPackage : {boundary_raw_crs}",
    f"  CRS used for all spatial operations              : {CODEBASE_CRS}",
    f"  (EPSG:2056 confirmed as codebase standard — see catchment_pt.py,",
    f"   data_import.py, generate_infrastructure.py, and others)",
    "",
    "RECORD COUNTS",
    f"  {'File':<20}  {'Original':>10}  {'Retained':>10}",
    f"  {'-'*20}  {'-'*10}  {'-'*10}",
    f"  {'stops.txt':<20}  {orig_counts['stops']:>10,}  {len(stops_filtered):>10,}",
    f"  {'trips.txt':<20}  {orig_counts['trips']:>10,}  {len(trips):>10,}",
    f"  {'routes.txt':<20}  {orig_counts['routes']:>10,}  {len(routes_gtfs):>10,}",
    f"  {'agency.txt':<20}  {orig_counts['agencies']:>10,}  {len(agency):>10,}",
    "",
    "ROUTE-TYPE BREAKDOWN (retained routes)",
    f"  {'route_type':<12}  {'Mode':<20}  {'Class':<12}  {'Count':>6}",
    f"  {'-'*12}  {'-'*20}  {'-'*12}  {'-'*6}",
]
for rt, count in route_type_counts.items():
    rt_int = int(rt) if pd.notna(rt) else None
    label  = ALL_RETAINED_ROUTE_TYPES.get(rt_int, 'Unknown') if rt_int is not None else 'Unknown'
    cls    = 'rail' if rt_int in RAIL_ROUTE_TYPES else 'pt_feeder'
    report_lines.append(f"  {rt_int:<12}  {label:<20}  {cls:<12}  {count:>6,}")

report_lines += [
    "",
    "SERVICE DURATION FILTER (Tier 1)",
    f"  Timetable period           : {TIMETABLE_START} to {TIMETABLE_END}",
    f"  Min weekday active days    : {_tier1_stats['min_threshold']}",
    f"  Service IDs evaluated      : {_tier1_stats['total_sids']:,}",
    f"  Service IDs dropped        : {_tier1_stats['dropped_sids']:,}",
    f"  Service IDs retained       : {_tier1_stats['kept_sids']:,}",
    f"  Trips before Tier 1 filter : {_tier1_stats['trips_before']:,}",
    f"  Trips after Tier 1 filter  : {_tier1_stats['trips_after']:,}",
    "",
    "OPTIONAL FILES",
]
for f in OPTIONAL_FILES:
    status = "skipped (not present)" if f in skipped_optional else "included"
    report_lines.append(f"  {f:<25}  {status}")

report_lines += [
    "",
    "TRIP FILTERING NOTE",
    "  Cross-boundary trips are truncated at the canton border: any trip",
    "  with at least one stop within the canton is retained, but stop_times",
    "  rows for out-of-canton stops are removed. This preserves accurate",
    "  line geometry within the canton. Trips with no canton stops at all",
    "  are dropped entirely.",
    "",
    "CROSS-EVALUATION NOTE",
    "  OD_type = 'pt_catchment_perimeter' and perimeter_demand are retained",
    "  in settings.py for cross-evaluation. Once both the PT-Feeder method",
    "  (this filtered output) and the existing perimeter-based method have",
    "  been run, their results can be compared directly.",
    "",
    "VALIDATION",
    f"  stop_times → stops       : {tag1}",
    f"  trips → routes           : {tag2}",
    f"  trips → service IDs      : {tag3}",
    "",
    "RUNTIME",
    f"  Total elapsed            : {time.time() - _start_time:.1f} seconds",
    "",
    "=" * 70,
]

report_text = "\n".join(report_lines)
with open(_out_path('filter_report.txt'), 'w', encoding='utf-8') as f:
    f.write(report_text)

print(report_text)
print("\nDone.")
