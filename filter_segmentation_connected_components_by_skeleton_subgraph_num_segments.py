import numpy as np
import tifffile as tiff
from scipy.ndimage import label
import networkx as nx
from typing import List, Tuple


def read_am(file_path):
    """
    Read metadata and data from Amira (.am) ASCII file as a single string.
    
    """
    with open(file_path, 'r') as file:
        # Read the entire file as a single string
        content = file.read()
    
    # Extract number of vertices, edges, and points from the header
    lines = content.splitlines()
    num_vertices = None
    num_edges = None
    num_points = None
    
    for line in lines:
        if line.startswith("define VERTEX"):
            num_vertices = int(line.split()[2])
        elif line.startswith("define EDGE"):
            num_edges = int(line.split()[2])
        elif line.startswith("define POINT"):
            num_points = int(line.split()[2])
        
    # Locate the first occurrence of "@1"
    first_idx_1 = content.find("@1")
    
    # Locate the second occurrence of "@1"
    second_idx_1 = content.find("@1", first_idx_1 + len("@1"))
    
    # Trim the string after the second occurrence of "@1"
    first_points_string = content[second_idx_1 + len("@1"):].strip()
    
    # Find first_points
    first_points = []
    for index, line in enumerate(first_points_string.splitlines()):
        values = list(map(float, line.split()))
        if len(values) != 3:
            break
        else:
            # Index, x, y, z with 2nd and 4th columns swapped
            first_points.append([index, values[2], values[1], values[0]])
    
    # Find segment_node_pairs
    segment_node_pairs_string = first_points_string.split("@2", 1)[1].strip()
    segment_node_pairs = []
    for index, line in enumerate(segment_node_pairs_string.splitlines(), 1):
        values = list(map(int, line.split()))
        if len(values) != 2:
            break
        else:
            segment_node_pairs.append([values[0], values[1]])
    
    # Find pts_per_segment
    pts_per_segment_string = segment_node_pairs_string.split("@3", 1)[1].strip()
    pts_per_segment = []
    for index, line in enumerate(pts_per_segment_string.splitlines(), 1):
        values = list(map(int, line.split()))
        if len(values) != 1:
            break
        else:
            pts_per_segment.append([values[0]])
            
    # Find all_points 
    all_points_string = pts_per_segment_string.split("@4", 1)[1].strip()
    all_points = []
    for index, line in enumerate(all_points_string.splitlines(), 1):
        values = list(map(float, line.split()))
        if len(values) != 3:
            break
        else:
            # Index, x, y, z with 2nd and 4th columns swapped 
            all_points.append([index, values[2], values[1], values[0]])
    
    # Find thickness
    thickness_string = all_points_string.split("@5", 1)[1].strip()
    thickness = []
    for index, line in enumerate(thickness_string.splitlines(), 1):
        values = list(map(float, line.split()))
        if len(values) != 1:
            break
        else:
            thickness.append([values[0]])
    
    return (num_vertices, num_edges, num_points, 
            first_points, segment_node_pairs, 
            pts_per_segment, all_points, thickness)


def filter_connected_components_by_subgraph(
    tif_volume: np.ndarray,
    labeled_volume: np.ndarray,
    nodes: List[List[float]],
    segments: List[List[int]],
    voxel_size: float,
    threshold_num_segments: int
) -> np.ndarray:
    """
    Filter out small connected components from a 3D segmentation based on graph topology.
    
    Identifies subgraphs with segment count <= threshold and removes corresponding
    3D connected components from the boolean volume, keeping larger subgraphs.
    
    Parameters:
    -----------
    tif_volume : np.ndarray
        Boolean 3D array of the segmentation
    labeled_volume : np.ndarray
        3D array with connected component labels (same shape as tif_volume)
    nodes : List[List[float]]
        List of nodes, each as [id, x, y, z] where coordinates are in physical space
    segments : List[List[int]]
        List of segment pairs [node_id1, node_id2] defining graph edges
    voxel_size : float
        Scaling factor to convert physical coordinates to voxel indices
    threshold_num_segments : int
        Maximum number of segments for a subgraph to be removed; larger subgraphs are kept
    
    Returns:
    --------
    np.ndarray
        Masked boolean volume with small subgraphs removed (keeping large ones)
    """
    
    # Handle edge cases
    if len(segments) == 0 or len(nodes) == 0:
        return tif_volume.copy()
    
    # Step 1: Build graph from segments
    G = nx.Graph()
    G.add_edges_from(segments)
    
    # Step 2: Find connected components (subgraphs)
    connected_subgraphs = list(nx.connected_components(G))
    
    # Step 3: Separate subgraphs into small and large based on segment count
    small_subgraphs = []
    large_subgraphs = []
    
    for subgraph_nodes in connected_subgraphs:
        # Count segments where both endpoints are in this subgraph
        segment_count = sum(1 for seg in segments 
                           if seg[0] in subgraph_nodes and seg[1] in subgraph_nodes)
        
        if segment_count <= threshold_num_segments:
            small_subgraphs.append(subgraph_nodes)
        else:
            large_subgraphs.append(subgraph_nodes)
    
    # If no small subgraphs to remove, return original volume
    if len(small_subgraphs) == 0:
        return tif_volume.copy()
    
    # Step 4: Create node lookup dictionary for fast coordinate access
    # node format: [id, x, y, z]
    node_dict = {int(node[0]): node[1:4] for node in nodes}
    
    # Step 5: Find labels that belong to LARGE subgraphs (these must be kept)
    labels_to_keep = set()
    
    for subgraph_nodes in large_subgraphs:
        for node_id in subgraph_nodes:
            if node_id in node_dict:
                # Get physical coordinates
                x, y, z = node_dict[node_id]
                
                # Convert to voxel coordinates
                vx = int(round(x / voxel_size))
                vy = int(round(y / voxel_size))
                vz = int(round(z / voxel_size))
                
                # Check bounds and sample labeled_volume
                if (0 <= vx < labeled_volume.shape[0] and 
                    0 <= vy < labeled_volume.shape[1] and 
                    0 <= vz < labeled_volume.shape[2]):
                    
                    label = labeled_volume[vx, vy, vz]
                    if label > 0:  # Ignore background label (0)
                        labels_to_keep.add(label)
    
    # Step 6: Find all label IDs from SMALL subgraphs
    labels_from_small = set()
    
    for subgraph_nodes in small_subgraphs:
        for node_id in subgraph_nodes:
            if node_id in node_dict:
                # Get physical coordinates
                x, y, z = node_dict[node_id]
                
                # Convert to voxel coordinates
                # Node coords (X,Y,Z) map to volume axes (0,1,2)
                vx = int(round(x / voxel_size))
                vy = int(round(y / voxel_size))
                vz = int(round(z / voxel_size))
                
                # Check bounds and sample labeled_volume
                if (0 <= vx < labeled_volume.shape[0] and 
                    0 <= vy < labeled_volume.shape[1] and 
                    0 <= vz < labeled_volume.shape[2]):
                    
                    label = labeled_volume[vx, vy, vz]
                    if label > 0:  # Ignore background label (0)
                        labels_from_small.add(label)
    
    # Step 7: Only remove labels that are NOT in large subgraphs
    labels_to_remove = labels_from_small - labels_to_keep
    
    # If no valid labels to remove, return original volume
    if len(labels_to_remove) == 0:
        return tif_volume.copy()
    
    # Step 8: Create mask for labels to remove (vectorized operation)
    labels_array = np.array(list(labels_to_remove))
    mask_to_remove = np.isin(labeled_volume, labels_array)
    
    # Step 9: Apply mask to original volume
    masked_volume = tif_volume.copy()
    masked_volume[mask_to_remove] = False
    
    return masked_volume

def main():
    #-------------------------------------------FILE PATHS----------------------------------------------
    tif_read_path = r"C:\...\Image.tif" # (*.tif) 3D TIF Binary Segmentation
    am_read_path = r"C:\...\CenterlineTree_Skeleton.am"  # (*.am) Avizo ASCII Spatial Graph
    tif_write_path = r"C:\...\Image_Components_Filtered_by_Skeleton.tif" # (*.tif) 3D TIF Binary Segmentation - Components filtered by skeleton sub-graph segment count
    #-------------------------------------------PARAMETERS----------------------------------------------
    threshold_num_semgents = 3 # Keeps only 3D connected components containing skeleton sub-graphs with greater than this number of vessel segments
    voxel_size = 4.512 # isotropic voxel size (usually in microns)
    #-------------------------------------------READ FILES----------------------------------------------
    print("Reading segmentation .tif file...")
    tif_volume = tiff.imread(tif_read_path).astype(bool)
    print("Reading spatial graph .am file...")
    num_vertices, num_edges, num_points, nodes, segments, points_per_segment, points, thickness = read_am(am_read_path)
    #-------------------------------------------CONNECTED COMPONENT MASK----------------------------------------------
    ("Labelling 3D connected components...")
    labeled_volume, num_features = label(tif_volume)
    print(f"Masking out 3D connected components where skeleton subgraphs contain < or = {threshold_num_semgents} segments...")
    masked_volume = filter_connected_components_by_subgraph(tif_volume, labeled_volume, nodes, segments, voxel_size, threshold_num_semgents)
    #-------------------------------------------SAVE MASKED SEGMENTATION----------------------------------------------
    print("Saving masked segmentation .tif file...")
    tiff.imwrite(tif_write_path, masked_volume.astype(np.uint8))
    print("Finished!")


if __name__ == "__main__":
    main()