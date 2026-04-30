"""
Manual Services Module

This module provides a CLI interface for manually adding, modifying, and
removing rail services from the network.

Usage:
    from infrastructure.manual_services import add_rail_service
    
    add_rail_service(
        service_name='S99',
        stations=['Zürich HB', 'Zürich Oerlikon', 'Wallisellen', 'Dübendorf'],
        frequency=4,
        travel_times=[6, 3, 4]
    )
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
import json
import warnings
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
import paths


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ServiceDefinition:
    """Definition of a rail service."""
    service_name: str
    stations: List[str]  # Station names or IDs
    frequency: int       # Trains per hour
    travel_times: List[int]  # Minutes between consecutive stations
    direction: str = 'bidirectional'  # 'A', 'B', or 'bidirectional'
    train_type: str = 'standard'
    capacity: int = 500
    via: List[str] = field(default_factory=list)
    version: str = 'current'
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ServiceDefinition':
        return cls(**data)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate service definition.
        
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        if not self.service_name:
            errors.append("Service name is required")
        
        if len(self.stations) < 2:
            errors.append("At least 2 stations are required")
        
        if len(self.travel_times) != len(self.stations) - 1:
            errors.append(f"Travel times count ({len(self.travel_times)}) must be "
                         f"one less than stations count ({len(self.stations)})")
        
        if self.frequency <= 0:
            errors.append("Frequency must be positive")
        
        if any(t <= 0 for t in self.travel_times):
            errors.append("All travel times must be positive")
        
        if self.direction not in ['A', 'B', 'bidirectional']:
            errors.append(f"Invalid direction: {self.direction}")
        
        return len(errors) == 0, errors


@dataclass
class ValidationResult:
    """Result of service validation against network."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    matched_nodes: List[int] = field(default_factory=list)
    unmatched_stations: List[str] = field(default_factory=list)


# =============================================================================
# Node Lookup
# =============================================================================

def load_nodes_for_matching(version: str = 'current') -> Tuple[gpd.GeoDataFrame, Dict]:
    """
    Load nodes and build lookup tables for station matching.
    
    Args:
        version: Network version
        
    Returns:
        Tuple of (nodes GeoDataFrame, lookups dict)
    """
    from .version_manager import VersionManager
    
    vm = VersionManager()
    nodes, _ = vm.build_network(version)
    
    lookups = {
        'by_id': {},
        'by_name': {},
        'by_code': {},
        'by_name_lower': {}
    }
    
    for idx, row in nodes.iterrows():
        node_id = row['Betriebspunkt_Nummer']
        
        lookups['by_id'][node_id] = row
        
        if pd.notna(row.get('NAME')):
            lookups['by_name'][row['NAME']] = node_id
            lookups['by_name_lower'][row['NAME'].lower().strip()] = node_id
        
        if pd.notna(row.get('CODE')):
            lookups['by_code'][row['CODE']] = node_id
    
    return nodes, lookups


def match_station_to_node(
    station: str,
    lookups: Dict
) -> Optional[int]:
    """
    Match station identifier to node ID.
    
    Args:
        station: Station name, code, or ID
        lookups: Lookup tables from load_nodes_for_matching
        
    Returns:
        Node ID or None if not found
    """
    # Try as ID
    try:
        station_id = int(station)
        if station_id in lookups['by_id']:
            return station_id
    except ValueError:
        pass
    
    # Try exact name match
    if station in lookups['by_name']:
        return lookups['by_name'][station]
    
    # Try code match
    if station in lookups['by_code']:
        return lookups['by_code'][station]
    
    # Try lowercase name match
    station_lower = station.lower().strip()
    if station_lower in lookups['by_name_lower']:
        return lookups['by_name_lower'][station_lower]
    
    # Try partial match
    for name, node_id in lookups['by_name_lower'].items():
        if station_lower in name or name in station_lower:
            return node_id
    
    return None


# =============================================================================
# Service Validation
# =============================================================================

def validate_service(
    service: ServiceDefinition,
    nodes: gpd.GeoDataFrame,
    lookups: Dict,
    segments: Optional[gpd.GeoDataFrame] = None
) -> ValidationResult:
    """
    Validate that a service is feasible on the network.
    
    Args:
        service: ServiceDefinition to validate
        nodes: Network nodes
        lookups: Node lookup tables
        segments: Network segments (optional, for connectivity check)
        
    Returns:
        ValidationResult
    """
    result = ValidationResult(is_valid=True)
    
    # Basic validation
    is_valid, errors = service.validate()
    if not is_valid:
        result.is_valid = False
        result.errors.extend(errors)
        return result
    
    # Match stations to nodes
    for station in service.stations:
        node_id = match_station_to_node(station, lookups)
        if node_id is not None:
            result.matched_nodes.append(node_id)
        else:
            result.unmatched_stations.append(station)
    
    # Check if all stations matched
    if result.unmatched_stations:
        result.is_valid = False
        result.errors.append(
            f"Could not match stations: {result.unmatched_stations}"
        )
    
    # Check connectivity (if segments provided)
    if segments is not None and len(result.matched_nodes) >= 2:
        # Build simple connectivity check
        edges = set()
        for _, seg in segments.iterrows():
            edges.add((seg['from_node'], seg['to_node']))
            edges.add((seg['to_node'], seg['from_node']))
        
        for i in range(len(result.matched_nodes) - 1):
            from_node = result.matched_nodes[i]
            to_node = result.matched_nodes[i + 1]
            
            if (from_node, to_node) not in edges:
                result.warnings.append(
                    f"No direct segment between nodes {from_node} and {to_node}"
                )
    
    return result


# =============================================================================
# Service Management Functions
# =============================================================================

def add_rail_service(
    service_name: str,
    stations: List[str],
    frequency: int,
    travel_times: List[int],
    direction: str = 'bidirectional',
    train_type: str = 'standard',
    capacity: int = 500,
    via: Optional[List[str]] = None,
    version: str = 'current',
    validate: bool = True
) -> ServiceDefinition:
    """
    Add a new rail service to the network.
    
    Args:
        service_name: Name of the service (e.g., 'S99', 'IC21')
        stations: List of station names or IDs in sequence
        frequency: Trains per hour
        travel_times: Travel time in minutes between consecutive stations
        direction: 'A', 'B', or 'bidirectional'
        train_type: Train type (e.g., 'Re450', 'IC', 'IR')
        capacity: Passenger capacity per train
        via: Intermediate routing nodes (optional)
        version: Target network version
        validate: Whether to validate against network
        
    Returns:
        ServiceDefinition object
        
    Example:
        add_rail_service(
            service_name='S99',
            stations=['Zürich HB', 'Zürich Oerlikon', 'Wallisellen', 'Dübendorf'],
            frequency=4,
            travel_times=[6, 3, 4]
        )
    """
    service = ServiceDefinition(
        service_name=service_name,
        stations=stations,
        frequency=frequency,
        travel_times=travel_times,
        direction=direction,
        train_type=train_type,
        capacity=capacity,
        via=via or [],
        version=version
    )
    
    if validate:
        nodes, lookups = load_nodes_for_matching(version)
        result = validate_service(service, nodes, lookups)
        
        if not result.is_valid:
            raise ValueError(f"Service validation failed: {result.errors}")
        
        if result.warnings:
            for warning in result.warnings:
                warnings.warn(warning)
    
    # Save to services file
    _save_service(service, version)
    
    print(f"Added service '{service_name}' with {len(stations)} stations")
    return service


def modify_service(
    service_name: str,
    modifications: Dict,
    version: str = 'current'
) -> ServiceDefinition:
    """
    Modify an existing service.
    
    Args:
        service_name: Name of service to modify
        modifications: Dict of fields to update
        version: Network version
        
    Returns:
        Updated ServiceDefinition
    """
    services = _load_services(version)
    
    if service_name not in services:
        raise ValueError(f"Service '{service_name}' not found in version '{version}'")
    
    service_data = services[service_name]
    service_data.update(modifications)
    
    service = ServiceDefinition.from_dict(service_data)
    
    # Validate
    is_valid, errors = service.validate()
    if not is_valid:
        raise ValueError(f"Modified service is invalid: {errors}")
    
    # Save
    _save_service(service, version)
    
    print(f"Modified service '{service_name}'")
    return service


def remove_service(
    service_name: str,
    version: str = 'current'
) -> bool:
    """
    Remove a service from a version.
    
    Args:
        service_name: Name of service to remove
        version: Network version
        
    Returns:
        True if removed, False if not found
    """
    services = _load_services(version)
    
    if service_name not in services:
        warnings.warn(f"Service '{service_name}' not found in version '{version}'")
        return False
    
    del services[service_name]
    
    # Save updated services
    services_path = _get_services_path(version)
    with open(services_path, 'w', encoding='utf-8') as f:
        json.dump(services, f, indent=2)
    
    print(f"Removed service '{service_name}'")
    return True


def list_services(
    version: str = 'current',
    filter_by: Optional[Dict] = None
) -> pd.DataFrame:
    """
    List all services in a version.
    
    Args:
        version: Network version
        filter_by: Optional dict of filters (e.g., {'train_type': 'S-Bahn'})
        
    Returns:
        DataFrame of services
    """
    services = _load_services(version)
    
    if not services:
        return pd.DataFrame()
    
    df = pd.DataFrame([
        {
            'service_name': name,
            'stations_count': len(s.get('stations', [])),
            'frequency': s.get('frequency'),
            'train_type': s.get('train_type'),
            'direction': s.get('direction'),
            'capacity': s.get('capacity'),
            'total_time': sum(s.get('travel_times', []))
        }
        for name, s in services.items()
    ])
    
    if filter_by:
        for key, value in filter_by.items():
            if key in df.columns:
                df = df[df[key] == value]
    
    return df


def get_service(
    service_name: str,
    version: str = 'current'
) -> Optional[ServiceDefinition]:
    """
    Get a specific service definition.
    
    Args:
        service_name: Service name
        version: Network version
        
    Returns:
        ServiceDefinition or None if not found
    """
    services = _load_services(version)
    
    if service_name not in services:
        return None
    
    return ServiceDefinition.from_dict(services[service_name])


# =============================================================================
# Service Edges Generation
# =============================================================================

def generate_service_edges(
    service: ServiceDefinition,
    nodes: gpd.GeoDataFrame,
    lookups: Dict,
    segments: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Generate GeoDataFrame of service edges from a service definition.
    
    Args:
        service: ServiceDefinition
        nodes: Network nodes
        lookups: Node lookups
        segments: Network segments (for geometry)
        
    Returns:
        GeoDataFrame of service edges
    """
    # Match stations to nodes
    node_sequence = []
    for station in service.stations:
        node_id = match_station_to_node(station, lookups)
        if node_id is not None:
            node_sequence.append(node_id)
    
    if len(node_sequence) < 2:
        return gpd.GeoDataFrame()
    
    # Build segment lookup
    segment_lookup = {}
    for _, seg in segments.iterrows():
        key1 = (seg['from_node'], seg['to_node'])
        key2 = (seg['to_node'], seg['from_node'])
        segment_lookup[key1] = seg.geometry
        segment_lookup[key2] = seg.geometry
    
    # Generate edges
    edges = []
    
    directions = ['A', 'B'] if service.direction == 'bidirectional' else [service.direction]
    
    for direction in directions:
        seq = node_sequence if direction == 'A' else list(reversed(node_sequence))
        times = service.travel_times if direction == 'A' else list(reversed(service.travel_times))
        
        for i in range(len(seq) - 1):
            from_node = seq[i]
            to_node = seq[i + 1]
            
            geometry = segment_lookup.get((from_node, to_node))
            
            edges.append({
                'from_node': from_node,
                'to_node': to_node,
                'service_name': service.service_name,
                'direction': direction,
                'travel_time': times[i],
                'frequency': service.frequency,
                'train_type': service.train_type,
                'capacity': service.capacity,
                'geometry': geometry
            })
    
    edges_df = pd.DataFrame(edges)
    edges_gdf = gpd.GeoDataFrame(edges_df, geometry='geometry', crs="EPSG:2056")
    
    return edges_gdf


# =============================================================================
# File Operations
# =============================================================================

def _get_services_path(version: str) -> Path:
    """Get path to services file for a version."""
    base_dir = Path(paths.MAIN) / paths.NETWORK_INFRASTRUCTURE_DIR
    version_dir = base_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)
    return version_dir / "manual_services.json"


def _load_services(version: str) -> Dict:
    """Load services from file."""
    services_path = _get_services_path(version)
    
    if not services_path.exists():
        return {}
    
    with open(services_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _save_service(service: ServiceDefinition, version: str):
    """Save a service to file."""
    services = _load_services(version)
    services[service.service_name] = service.to_dict()
    
    services_path = _get_services_path(version)
    with open(services_path, 'w', encoding='utf-8') as f:
        json.dump(services, f, indent=2, ensure_ascii=False)


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage manual rail services")
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # List services
    list_parser = subparsers.add_parser('list', help='List services')
    list_parser.add_argument('--version', '-v', default='current', help='Network version')
    
    # Add service
    add_parser = subparsers.add_parser('add', help='Add service')
    add_parser.add_argument('name', help='Service name')
    add_parser.add_argument('--stations', '-s', nargs='+', required=True, help='Station names')
    add_parser.add_argument('--frequency', '-f', type=int, required=True, help='Trains per hour')
    add_parser.add_argument('--times', '-t', nargs='+', type=int, required=True, help='Travel times')
    add_parser.add_argument('--direction', '-d', default='bidirectional', 
                           choices=['A', 'B', 'bidirectional'], help='Direction')
    add_parser.add_argument('--version', '-v', default='current', help='Network version')
    
    # Remove service
    remove_parser = subparsers.add_parser('remove', help='Remove service')
    remove_parser.add_argument('name', help='Service name')
    remove_parser.add_argument('--version', '-v', default='current', help='Network version')
    
    # Get service
    get_parser = subparsers.add_parser('get', help='Get service details')
    get_parser.add_argument('name', help='Service name')
    get_parser.add_argument('--version', '-v', default='current', help='Network version')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        df = list_services(version=args.version)
        if len(df) > 0:
            print(df.to_string(index=False))
        else:
            print("No services defined")
    
    elif args.command == 'add':
        service = add_rail_service(
            service_name=args.name,
            stations=args.stations,
            frequency=args.frequency,
            travel_times=args.times,
            direction=args.direction,
            version=args.version,
            validate=True
        )
        print(f"Added: {service}")
    
    elif args.command == 'remove':
        remove_service(args.name, version=args.version)
    
    elif args.command == 'get':
        service = get_service(args.name, version=args.version)
        if service:
            print(json.dumps(service.to_dict(), indent=2))
        else:
            print(f"Service '{args.name}' not found")
    
    else:
        parser.print_help()
