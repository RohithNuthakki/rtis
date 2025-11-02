#!/usr/bin/env python3
"""
Debug and fix HERE API segment ID mismatches

This script analyzes HERE API realtime data and edge mapping files to identify
and fix segment ID mismatches, ensuring proper matching between realtime data
and edge mappings.
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
import glob
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("here_segment_fixer")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Fix HERE API segment ID mismatches')
    parser.add_argument('--data-dir', required=True, help='Directory containing HERE API data files')
    parser.add_argument('--output-dir', help='Output directory for fixed files (default: same as data-dir)')
    parser.add_argument('--realtime-file', help='Specific realtime file to analyze (optional)')
    parser.add_argument('--edge-mapping-file', help='Specific edge mapping file to analyze (optional)')
    parser.add_argument('--fix', action='store_true', help='Apply fixes to create new mapping')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    return parser.parse_args()

def extract_timestamp_from_filename(filename):
    """Extract timestamp from a filename"""
    match = re.search(r'(\d{8}_\d{6})', filename)
    if match:
        return match.group(1)
    return None

def analyze_realtime_segments(realtime_file):
    """Analyze the structure of realtime segments"""
    logger.info(f"Analyzing realtime file: {realtime_file}")
    
    try:
        with open(realtime_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading realtime file: {e}")
        return None
    
    segments_info = []
    results = data.get('results', [])
    
    for idx, segment in enumerate(results):
        segment_info = {
            'index': idx,
            'has_id': 'id' in segment,
            'id_value': segment.get('id'),
            'has_location': 'location' in segment,
            'has_currentFlow': 'currentFlow' in segment,
            'keys': list(segment.keys())
        }
        
        # Check location structure
        if 'location' in segment:
            location = segment['location']
            segment_info['location_keys'] = list(location.keys())
            
            # Check for shape data
            if 'shape' in location:
                shape = location['shape']
                segment_info['shape_keys'] = list(shape.keys())
                
                # Check for links
                if 'links' in shape:
                    links = shape['links']
                    segment_info['num_links'] = len(links)
                    
                    # Check first link structure
                    if links:
                        first_link = links[0]
                        segment_info['link_keys'] = list(first_link.keys())
                        segment_info['link_id'] = first_link.get('linkId')
        
        segments_info.append(segment_info)
    
    return segments_info

def analyze_edge_mapping_segments(edge_mapping_file):
    """Analyze the structure of edge mapping segments"""
    logger.info(f"Analyzing edge mapping file: {edge_mapping_file}")
    
    try:
        with open(edge_mapping_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading edge mapping file: {e}")
        return None
    
    segments_info = []
    segments = data.get('segments', [])
    
    for idx, segment in enumerate(segments):
        segment_info = {
            'index': idx,
            'id': segment.get('id'),
            'has_sumo_edges': 'sumo_edges' in segment,
            'sumo_edges': segment.get('sumo_edges', []),
            'road_name': segment.get('road_name'),
            'keys': list(segment.keys())
        }
        
        segments_info.append(segment_info)
    
    return segments_info

def create_fixed_edge_mapping(realtime_file, edge_mapping_file, output_file):
    """Create a fixed edge mapping that properly matches realtime segments"""
    logger.info("Creating fixed edge mapping...")
    
    # Load both files
    try:
        with open(realtime_file, 'r') as f:
            realtime_data = json.load(f)
        with open(edge_mapping_file, 'r') as f:
            edge_mapping_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading files: {e}")
        return False
    
    # Create new edge mapping with correct segment IDs
    fixed_mapping = {
        'metadata': edge_mapping_data.get('metadata', {}),
        'segments': []
    }
    
    # Get segments from both files
    realtime_segments = realtime_data.get('results', [])
    mapping_segments = edge_mapping_data.get('segments', [])
    
    logger.info(f"Realtime segments: {len(realtime_segments)}")
    logger.info(f"Mapping segments: {len(mapping_segments)}")
    
    # Create mapping by matching segments based on position
    for idx, realtime_segment in enumerate(realtime_segments):
        # Find corresponding mapping segment
        if idx < len(mapping_segments):
            mapping_segment = mapping_segments[idx]
            
            # Create new segment with proper ID
            fixed_segment = mapping_segment.copy()
            
            # Generate ID based on index (since realtime segments don't have IDs)
            fixed_segment['id'] = f"segment_{idx}"
            
            # Add additional info from realtime segment if available
            if 'location' in realtime_segment:
                location = realtime_segment['location']
                if 'shape' in location and 'links' in location['shape']:
                    links = location['shape']['links']
                    if links:
                        # Use first link ID if available
                        link_id = links[0].get('linkId')
                        if link_id:
                            fixed_segment['link_id'] = link_id
            
            fixed_mapping['segments'].append(fixed_segment)
    
    # Save the fixed mapping
    try:
        with open(output_file, 'w') as f:
            json.dump(fixed_mapping, f, indent=2)
        logger.info(f"Fixed edge mapping saved to: {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error saving fixed mapping: {e}")
        return False

def create_segment_id_mapping(realtime_segments_info, mapping_segments_info):
    """Create a mapping between different segment ID formats"""
    id_mapping = {}
    
    # Try to match by index
    for realtime_info in realtime_segments_info:
        idx = realtime_info['index']
        
        # Look for corresponding mapping segment by index
        for mapping_info in mapping_segments_info:
            if mapping_info['index'] == idx:
                # Create ID mapping
                realtime_id = realtime_info.get('id_value') or f"realtime_{idx}"
                mapping_id = mapping_info.get('id')
                
                id_mapping[realtime_id] = mapping_id
                break
    
    return id_mapping

def main():
    """Main execution function"""
    args = parse_arguments()
    
    # Enable debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set output directory
    output_dir = args.output_dir or args.data_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Find files to analyze
    if args.realtime_file:
        realtime_files = [args.realtime_file]
    else:
        realtime_files = glob.glob(os.path.join(args.data_dir, "realtime_*.json"))
    
    if args.edge_mapping_file:
        edge_mapping_files = [args.edge_mapping_file]
    else:
        edge_mapping_files = glob.glob(os.path.join(args.data_dir, "sumo_edge_mapping_*.json"))
    
    if not realtime_files:
        logger.error("No realtime files found")
        return 1
    
    if not edge_mapping_files:
        logger.error("No edge mapping files found")
        return 1
    
    # Use the first files found (or specified files)
    realtime_file = realtime_files[0]
    edge_mapping_file = edge_mapping_files[0]
    
    logger.info(f"Analyzing realtime file: {realtime_file}")
    logger.info(f"Analyzing edge mapping file: {edge_mapping_file}")
    
    # Analyze segment structures
    realtime_segments_info = analyze_realtime_segments(realtime_file)
    mapping_segments_info = analyze_edge_mapping_segments(edge_mapping_file)
    
    if not realtime_segments_info or not mapping_segments_info:
        logger.error("Failed to analyze files")
        return 1
    
    # Create analysis report
    report_file = os.path.join(output_dir, "segment_analysis_report.txt")
    
    with open(report_file, 'w') as f:
        f.write("HERE API Segment Analysis Report\n")
        f.write("===============================\n\n")
        f.write(f"Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Realtime file: {os.path.basename(realtime_file)}\n")
        f.write(f"Edge mapping file: {os.path.basename(edge_mapping_file)}\n\n")
        
        # Realtime segments analysis
        f.write("Realtime Segments Analysis:\n")
        f.write("--------------------------\n")
        f.write(f"Total segments: {len(realtime_segments_info)}\n")
        segments_with_id = sum(1 for s in realtime_segments_info if s['has_id'])
        f.write(f"Segments with ID field: {segments_with_id}\n")
        
        # Sample realtime segments
        f.write("\nSample realtime segments (first 5):\n")
        for info in realtime_segments_info[:5]:
            f.write(f"  Index {info['index']}: ")
            f.write(f"ID={info['id_value']}, ")
            f.write(f"Keys={info['keys']}\n")
            if 'link_id' in info:
                f.write(f"    Link ID: {info['link_id']}\n")
        
        # Edge mapping segments analysis
        f.write("\n\nEdge Mapping Segments Analysis:\n")
        f.write("-------------------------------\n")
        f.write(f"Total segments: {len(mapping_segments_info)}\n")
        segments_with_edges = sum(1 for s in mapping_segments_info if s['sumo_edges'])
        f.write(f"Segments with SUMO edges: {segments_with_edges}\n")
        
        # Sample mapping segments
        f.write("\nSample mapping segments (first 5):\n")
        for info in mapping_segments_info[:5]:
            f.write(f"  Index {info['index']}: ")
            f.write(f"ID={info['id']}, ")
            f.write(f"Road={info['road_name']}, ")
            f.write(f"SUMO edges={info['sumo_edges']}\n")
        
        # ID mapping analysis
        f.write("\n\nSegment ID Mapping:\n")
        f.write("------------------\n")
        id_mapping = create_segment_id_mapping(realtime_segments_info, mapping_segments_info)
        f.write(f"Total mappings: {len(id_mapping)}\n")
        
        # Sample ID mappings
        f.write("\nSample ID mappings (first 5):\n")
        for i, (realtime_id, mapping_id) in enumerate(list(id_mapping.items())[:5]):
            f.write(f"  {realtime_id} -> {mapping_id}\n")
    
    logger.info(f"Analysis report saved to: {report_file}")
    
    # Apply fixes if requested
    if args.fix:
        # Extract timestamp from edge mapping filename
        timestamp = extract_timestamp_from_filename(edge_mapping_file)
        if not timestamp:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create fixed edge mapping
        fixed_mapping_file = os.path.join(output_dir, f"fixed_edge_mapping_{timestamp}.json")
        success = create_fixed_edge_mapping(realtime_file, edge_mapping_file, fixed_mapping_file)
        
        if success:
            logger.info(f"Fixed edge mapping created: {fixed_mapping_file}")
            
            # Create ID mapping file for reference
            id_mapping_file = os.path.join(output_dir, f"segment_id_mapping_{timestamp}.json")
            with open(id_mapping_file, 'w') as f:
                json.dump(id_mapping, f, indent=2)
            logger.info(f"ID mapping saved to: {id_mapping_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())