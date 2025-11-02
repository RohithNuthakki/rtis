#!/usr/bin/env python3
"""
Map HERE API segments to SUMO network edges

This script takes HERE API edge mapping files and matches them to SUMO network edges
based on geographic coordinates, creating filled sumo_edges arrays.
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
import glob

# Setup SUMO environment
if "SUMO_HOME" in os.environ:
    SUMO_HOME = os.environ["SUMO_HOME"]
    tools_path = os.path.join(SUMO_HOME, "tools")
    if tools_path not in sys.path:
        sys.path.append(tools_path)
else:
    print("Error: SUMO_HOME not set. Please set SUMO_HOME environment variable.")
    sys.exit(1)

# Import SUMO libraries
try:
    import sumolib
except ImportError:
    print("Error importing SUMO libraries. Check if SUMO_HOME is correctly set.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("here_to_sumo_mapper")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Map HERE API segments to SUMO edges')
    parser.add_argument('--net-file', required=True, help='SUMO network file (.net.xml)')
    parser.add_argument('--edge-mapping-file', required=True, help='HERE API edge mapping JSON file')
    parser.add_argument('--output-file', help='Output file for updated edge mapping (default: updates input file)')
    parser.add_argument('--max-distance', type=float, default=50.0, help='Maximum distance (meters) for edge matching')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    return parser.parse_args()

def map_segments_to_sumo_edges(edge_mapping_data, net, max_distance=50.0):
    """Map HERE API segments to SUMO network edges based on coordinates"""
    logger.info("Mapping HERE API segments to SUMO edges...")
    
    if not edge_mapping_data or 'segments' not in edge_mapping_data:
        logger.error("Invalid edge mapping data format")
        return False
    
    segments = edge_mapping_data.get('segments', [])
    logger.info(f"Processing {len(segments)} HERE API segments")
    
    # Statistics
    successful_mappings = 0
    failed_mappings = 0
    segments_with_existing_edges = 0
    
    # Process each segment
    for idx, segment in enumerate(segments):
        segment_id = segment.get('id', f'segment_{idx}')
        
        # Check if segment already has SUMO edges mapped
        if 'sumo_edges' in segment and segment['sumo_edges']:
            segments_with_existing_edges += 1
            continue
        
        # Get shape points
        shape_points = segment.get('shape', [])
        if not shape_points or len(shape_points) < 2:
            logger.warning(f"Segment {segment_id} has no valid shape points")
            failed_mappings += 1
            continue
        
        # Get start and end points
        start_point = shape_points[0]
        end_point = shape_points[-1]
        
        matched_edges = []
        
        # Extract coordinates
        if len(start_point) >= 2 and len(end_point) >= 2:
            start_lat, start_lon = start_point[0], start_point[1]
            end_lat, end_lon = end_point[0], end_point[1]
            
            try:
                # Convert to SUMO x,y coordinates
                start_x, start_y = net.convertLonLat2XY(start_lon, start_lat)
                end_x, end_y = net.convertLonLat2XY(end_lon, end_lat)
                
                # Find nearby edges for start and end points
                start_edges = net.getNeighboringEdges(start_x, start_y, max_distance)
                end_edges = net.getNeighboringEdges(end_x, end_y, max_distance)
                
                # If no edges found, try with larger radius
                if not start_edges:
                    start_edges = net.getNeighboringEdges(start_x, start_y, max_distance * 2)
                if not end_edges:
                    end_edges = net.getNeighboringEdges(end_x, end_y, max_distance * 2)
                
                # Sort by distance
                if start_edges:
                    start_edges.sort(key=lambda x: x[1])
                if end_edges:
                    end_edges.sort(key=lambda x: x[1])
                
                # Get closest edges
                matched_edge_ids = set()
                
                if start_edges:
                    closest_start_edge = start_edges[0][0]
                    matched_edge_ids.add(closest_start_edge.getID())
                    
                    # Log details for debugging
                    if idx < 10:  # Log first 10 for debugging
                        logger.debug(f"Segment {segment_id} start: mapped to edge {closest_start_edge.getID()} "
                                   f"at distance {start_edges[0][1]:.1f}m")
                
                if end_edges:
                    closest_end_edge = end_edges[0][0]
                    matched_edge_ids.add(closest_end_edge.getID())
                    
                    # Log details for debugging
                    if idx < 10:  # Log first 10 for debugging
                        logger.debug(f"Segment {segment_id} end: mapped to edge {closest_end_edge.getID()} "
                                   f"at distance {end_edges[0][1]:.1f}m")
                
                # For longer segments, also match intermediate points
                if len(shape_points) > 2 and len(matched_edge_ids) > 0:
                    # Sample additional points along the segment
                    sample_count = min(3, len(shape_points) - 2)
                    step = max(1, (len(shape_points) - 2) // sample_count)
                    
                    for i in range(1, len(shape_points) - 1, step):
                        if len(matched_edge_ids) >= 5:  # Limit to 5 edges per segment
                            break
                        
                        point = shape_points[i]
                        if len(point) >= 2:
                            mid_lat, mid_lon = point[0], point[1]
                            mid_x, mid_y = net.convertLonLat2XY(mid_lon, mid_lat)
                            mid_edges = net.getNeighboringEdges(mid_x, mid_y, max_distance)
                            
                            if mid_edges:
                                mid_edges.sort(key=lambda x: x[1])
                                matched_edge_ids.add(mid_edges[0][0].getID())
                
                # Convert to list
                matched_edges = list(matched_edge_ids)
                
            except Exception as e:
                logger.warning(f"Error mapping segment {segment_id}: {e}")
                failed_mappings += 1
                continue
        
        # Store the mapping
        if matched_edges:
            segment['sumo_edges'] = matched_edges
            successful_mappings += 1
            
            # Log sample mappings
            if successful_mappings <= 5:
                logger.info(f"Segment {segment_id} mapped to edges: {matched_edges}")
        else:
            segment['sumo_edges'] = []
            failed_mappings += 1
            logger.debug(f"No edges found for segment {segment_id}")
    
    # Report statistics
    logger.info(f"Mapping complete:")
    logger.info(f"  Total segments: {len(segments)}")
    logger.info(f"  Segments with existing edges: {segments_with_existing_edges}")
    logger.info(f"  Successfully mapped: {successful_mappings}")
    logger.info(f"  Failed to map: {failed_mappings}")
    logger.info(f"  Success rate: {successful_mappings / (len(segments) - segments_with_existing_edges) * 100:.1f}%" 
               if (len(segments) - segments_with_existing_edges) > 0 else "N/A")
    
    return True

def main():
    """Main execution function"""
    args = parse_arguments()
    
    # Enable debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load SUMO network
    logger.info(f"Loading SUMO network from {args.net_file}")
    try:
        net = sumolib.net.readNet(args.net_file)
    except Exception as e:
        logger.error(f"Error loading SUMO network: {e}")
        return 1
    
    # Load edge mapping file
    logger.info(f"Loading edge mapping file from {args.edge_mapping_file}")
    try:
        with open(args.edge_mapping_file, 'r') as f:
            edge_mapping_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading edge mapping file: {e}")
        return 1
    
    # Map segments to SUMO edges
    success = map_segments_to_sumo_edges(edge_mapping_data, net, args.max_distance)
    
    if not success:
        logger.error("Mapping failed")
        return 1
    
    # Save the updated mapping
    output_file = args.output_file or args.edge_mapping_file
    
    # Create backup of original file if updating in place
    if output_file == args.edge_mapping_file:
        backup_file = args.edge_mapping_file + '.backup'
        logger.info(f"Creating backup: {backup_file}")
        try:
            with open(args.edge_mapping_file, 'r') as f:
                backup_data = f.read()
            with open(backup_file, 'w') as f:
                f.write(backup_data)
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return 1
    
    # Save updated mapping
    logger.info(f"Saving updated mapping to {output_file}")
    try:
        with open(output_file, 'w') as f:
            json.dump(edge_mapping_data, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving output file: {e}")
        return 1
    
    # Create a summary report
    summary_file = output_file.replace('.json', '_summary.txt')
    logger.info(f"Creating summary report: {summary_file}")
    
    try:
        with open(summary_file, 'w') as f:
            f.write("HERE API to SUMO Edge Mapping Summary\n")
            f.write("====================================\n\n")
            f.write(f"Processing time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"SUMO network file: {args.net_file}\n")
            f.write(f"Input edge mapping: {args.edge_mapping_file}\n")
            f.write(f"Output edge mapping: {output_file}\n")
            f.write(f"Maximum matching distance: {args.max_distance} meters\n\n")
            
            # Count statistics
            total_segments = len(edge_mapping_data.get('segments', []))
            segments_with_edges = sum(1 for s in edge_mapping_data.get('segments', []) 
                                    if 'sumo_edges' in s and s['sumo_edges'])
            
            f.write(f"Total segments: {total_segments}\n")
            f.write(f"Segments with SUMO edges: {segments_with_edges}\n")
            f.write(f"Success rate: {segments_with_edges / total_segments * 100:.1f}%\n\n")
            
            # Sample mappings
            f.write("Sample mappings (first 10):\n")
            f.write("--------------------------\n")
            for i, segment in enumerate(edge_mapping_data.get('segments', [])[:10]):
                segment_id = segment.get('id', f'segment_{i}')
                sumo_edges = segment.get('sumo_edges', [])
                road_name = segment.get('road_name', 'Unknown')
                f.write(f"{segment_id} ({road_name}): {sumo_edges}\n")
    except Exception as e:
        logger.error(f"Error creating summary report: {e}")
    
    logger.info("Mapping complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())