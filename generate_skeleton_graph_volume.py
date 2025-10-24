import numpy as np
import tifffile as tiff


def create_skeleton_volume_binary_am(volume_shape, points, thickness, voxel_size):
    """
    Creates a binary volume where all voxels within the point spheres are set to 255,
    and all other voxels are set to 0.
    Converts physical (micron) coordinates and radii back into voxel units.
    """

    print("Creating image volume of skeleton graph...")

    skeleton_volume = np.zeros(volume_shape, dtype=np.uint8)

    for point, radius_value in zip(points, thickness):
        idx, z, y, x = point  # coordinates in microns

        # --- Handle nested list/tuple/array for radius ---
        if isinstance(radius_value, (list, tuple, np.ndarray)):
            radius_value = float(radius_value[0])
        else:
            radius_value = float(radius_value)

        # --- Convert from microns → voxel units ---
        z_vox = int(round(z / voxel_size))
        y_vox = int(round(y / voxel_size))
        x_vox = int(round(x / voxel_size))
        radius_vox = int(round(max(radius_value, 0.5) / voxel_size))

        # --- Bounds (clipped to volume) ---
        z_min, z_max = max(0, z_vox - radius_vox), min(volume_shape[0], z_vox + radius_vox + 1)
        y_min, y_max = max(0, y_vox - radius_vox), min(volume_shape[1], y_vox + radius_vox + 1)
        x_min, x_max = max(0, x_vox - radius_vox), min(volume_shape[2], x_vox + radius_vox + 1)

        # --- Generate 3D sphere mask ---
        zz, yy, xx = np.ogrid[z_min:z_max, y_min:y_max, x_min:x_max]
        distance = np.sqrt((zz - z_vox)**2 + (yy - y_vox)**2 + (xx - x_vox)**2)
        sphere_mask = distance <= radius_vox

        # --- Fit mask to region ---
        region_shape = skeleton_volume[z_min:z_max, y_min:y_max, x_min:x_max].shape
        if sphere_mask.shape != region_shape:
            sphere_mask = sphere_mask[:region_shape[0], :region_shape[1], :region_shape[2]]

        # --- Apply mask ---
        skeleton_volume[z_min:z_max, y_min:y_max, x_min:x_max][sphere_mask] = 255

    return skeleton_volume


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

def main():
    #-------------------------------------------FILE PATHS----------------------------------------------
    tif_read_path = r"C:\...\Image.tif" # (*.tif) 3D TIF Binary Segmentation
    am_read_path = r"C:\...\CenterlineTree_Skeleton.am" # (*.am) Avizo ASCII Spatial Graph
    tif_write_path = r"C:\...\Skeleton_Volume.tif"
    #-------------------------------------------PARAMETERS----------------------------------------------
    voxel_size = 4.512 # isotropic voxel size (usually in microns). Make sure this voxel size was applied when loading the segmenation image for input spatial graph (am_read_path) generation.
    #------------------------------------------READ IN FILES--------------------------------------------
    print("Reading segmentation .tif file...")
    tif_volume = tiff.imread(tif_read_path).astype(bool)
    print("Reading spatial graph .am file...")
    num_vertices, num_edges, num_points, nodes, segments, points_per_segment, points, thickness = read_am(am_read_path)
    #------------------------------------------CONSTRUCT SKELETON GRAPH VOLUME--------------------------------------------
    print("Creating skeleton volume rendering...")
    skeleton_volume = create_skeleton_volume_binary_am(tif_volume.shape, points, thickness, voxel_size)
    #------------------------------------------WRITE FILES--------------------------------------------------------
    tiff.imwrite(tif_write_path, skeleton_volume)
    
    
if __name__ == "__main__":
    main()
