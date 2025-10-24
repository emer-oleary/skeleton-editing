import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.ndimage import distance_transform_cdt
from scipy.ndimage import label
import cv2


def adjust_thickness(slope, intercept, thickness):
    """
    Adjust thickness values using provided slope and intercept.
    
    Parameters:
    -----------
    slope : float
        Slope from linear regression (r_eff vs t_cl)
    intercept : float
        Intercept from linear regression
    thickness : array-like
        Thickness values (can be 1D array or list of lists)
    
    Returns:
    --------
    adjusted_thickness : list of lists
        Adjusted thickness values in the same format as input thickness
        (each value wrapped in its own list)
    """
    # Flatten thickness if it's in list of lists format
    thickness_flat = np.array(thickness).flatten()
    
    # Adjust thickness values using the regression parameters
    # Formula: t_adjusted = slope * t_original + intercept
    adjusted_values = slope * thickness_flat + intercept
    
    # Format as list of lists to match input format
    adjusted_thickness = [[float(val)] for val in adjusted_values]
    
    return adjusted_thickness


def measure_effective_radii(
    tif_volume,
    tif_volume_cdm,
    points,
    thickness,
    sample_every_n,
    voxel_size,
    subvolume_edge_size_cdm_factor
):
    """
    Measure effective radii at sampled points along vessel centerlines.
    
    Parameters:
    -----------
    tif_volume : ndarray (bool)
        Binary segmentation volume
    tif_volume_cdm : ndarray (float)
        Chamfer distance map of the segmentation
    points : ndarray
        Array of shape (N, 4) with columns [index, x, y, z] in physical units
    thickness : ndarray
        Per-point thickness values (centerline radius estimates)
    sample_every_n : int
        Sample every nth point
    voxel_size : float
        Voxel size in physical units (microns)
    subvolume_edge_size_cdm_factor : float
        Multiplier for subvolume edge size relative to CDM value
    
    Returns:
    --------
    dict : Dictionary containing results with keys:
        - 't_cl': centerline thickness values at sampled points
        - 'r_eff': effective radius values at sampled points
        - 'sampled_points': sampled point coordinates
        - 'circularity': circularity metric for each measurement
    """
    
    # Results storage
    results = {
        't_cl': [],
        'r_eff': [],
        'sampled_points': [],
        'circularity': [],
        'max_cdm_values': [],
        'min_area': [],
        'perimeter': []
    }
    
    # Sample points
    sampled_indices = np.arange(0, len(points), sample_every_n)
    total_samples = len(sampled_indices)
    
    print(f"Processing {total_samples} sampled points...")
    
    for sample_num, idx in enumerate(sampled_indices, 1):
        # Step 1: Extract coordinates and convert to voxel space
        point_data = points[idx]
        point_id = int(point_data[0])
        x_phys, y_phys, z_phys = point_data[1], point_data[2], point_data[3]
        
        # Convert to voxel coordinates
        x_vox = x_phys / voxel_size
        y_vox = y_phys / voxel_size
        z_vox = z_phys / voxel_size
        
        # Round to nearest integer voxel
        x_int = int(np.round(x_vox))
        y_int = int(np.round(y_vox))
        z_int = int(np.round(z_vox))
        
        # Check if point is within volume bounds
        if not (0 <= x_int < tif_volume.shape[0] and
                0 <= y_int < tif_volume.shape[1] and
                0 <= z_int < tif_volume.shape[2]):
            print(f"{sample_num}/{total_samples}: out of bounds")
            continue
        
        # Check if point is at boundary and skip if so
        at_boundary = (x_int == 0 or y_int == 0 or z_int == 0 or
                      x_int == tif_volume.shape[0] - 1 or
                      y_int == tif_volume.shape[1] - 1 or
                      z_int == tif_volume.shape[2] - 1)
        
        if at_boundary:
            print(f"{sample_num}/{total_samples}: boundary point")
            continue
        
        # Step 2: Get CDM value at this point to determine initial search size
        cdm_value_at_point = tif_volume_cdm[x_int, y_int, z_int]
        # Ensure minimum CDM value is voxel_size
        cdm_value_at_point = max(cdm_value_at_point, voxel_size)
        
        # Calculate initial subvolume edge size (in voxels)
        initial_edge_size = int(np.ceil(cdm_value_at_point * subvolume_edge_size_cdm_factor))
        half_edge = initial_edge_size // 2
        
        # Define bounds for initial CDM search subvolume
        x_min = max(0, x_int - half_edge)
        x_max = min(tif_volume_cdm.shape[0], x_int + half_edge + 1)
        y_min = max(0, y_int - half_edge)
        y_max = min(tif_volume_cdm.shape[1], y_int + half_edge + 1)
        z_min = max(0, z_int - half_edge)
        z_max = min(tif_volume_cdm.shape[2], z_int + half_edge + 1)
        
        # Step 3: Extract CDM subvolume and find maximum CDM value
        cdm_subvolume = tif_volume_cdm[x_min:x_max, y_min:y_max, z_min:z_max]
        max_cdm_value = np.max(cdm_subvolume)
        
        # Step 4: Calculate final subvolume edge size based on max CDM
        final_edge_size = int(np.ceil(max_cdm_value * subvolume_edge_size_cdm_factor))
        half_final = final_edge_size // 2
        
        # Define bounds for final binary volume extraction
        x_min_final = max(0, x_int - half_final)
        x_max_final = min(tif_volume.shape[0], x_int + half_final + 1)
        y_min_final = max(0, y_int - half_final)
        y_max_final = min(tif_volume.shape[1], y_int + half_final + 1)
        z_min_final = max(0, z_int - half_final)
        z_max_final = min(tif_volume.shape[2], z_int + half_final + 1)
        
        # Step 5: Extract binary subvolume
        binary_subvolume = tif_volume[x_min_final:x_max_final, 
                                      y_min_final:y_max_final, 
                                      z_min_final:z_max_final]
        
        # Calculate local coordinates of point within subvolume
        local_x = x_int - x_min_final
        local_y = y_int - y_min_final
        local_z = z_int - z_min_final
        
        # Perform multi-resolution reslicing to find minimum area
        reslice, r_eff, perimeter, area = find_minimum_area_reslice(
            binary_subvolume, local_x, local_y, local_z, max_cdm_value
        )
        r_eff = r_eff*voxel_size
        '''
        # Visualize
        if idx < 3 * sample_every_n:
            visualize_subvolume(binary_subvolume, point_id, 
                              (x_int, y_int, z_int), max_cdm_value)
            visualize_reslice(reslice, point_id, r_eff, perimeter, area, "Optimal")
        '''
        
        # Store results
        results['sampled_points'].append([x_int, y_int, z_int])
        results['t_cl'].append(float(thickness[idx][0]))
        results['max_cdm_values'].append(max_cdm_value)
        results['r_eff'].append(r_eff)
        results['circularity'].append(1.0)  # Placeholder
        results['min_area'].append(area)
        results['perimeter'].append(perimeter)
        
        # Print measurement results
        print(f"{sample_num}/{total_samples}: t_cl={float(thickness[idx][0]):.4f}, r_eff={r_eff:.4f}")
    
    # Convert lists to arrays - ensure 1D arrays for scalar values
    results['t_cl'] = np.array(results['t_cl'], dtype=float).flatten()
    results['r_eff'] = np.array(results['r_eff'], dtype=float).flatten()
    results['circularity'] = np.array(results['circularity'], dtype=float).flatten()
    results['max_cdm_values'] = np.array(results['max_cdm_values'], dtype=float).flatten()
    results['min_area'] = np.array(results['min_area'], dtype=float).flatten()
    results['perimeter'] = np.array(results['perimeter'], dtype=float).flatten()
    results['sampled_points'] = np.array(results['sampled_points'])  # Keep as 2D (N, 3)
    
    return results


def extract_boundary_slice(subvol, local_x, local_y, local_z, global_x, global_y, global_z, volume_shape):
    """Extract a 2D slice at volume boundary."""
    
    # Determine which face we're on
    if global_x == 0:
        reslice = subvol[0, :, :]
    elif global_x == volume_shape[0] - 1:
        reslice = subvol[-1, :, :]
    elif global_y == 0:
        reslice = subvol[:, 0, :]
    elif global_y == volume_shape[1] - 1:
        reslice = subvol[:, -1, :]
    elif global_z == 0:
        reslice = subvol[:, :, 0]
    else:  # global_z == volume_shape[2] - 1
        reslice = subvol[:, :, -1]
    
    # Clean and measure
    reslice_clean, area, perimeter = clean_and_measure_slice(reslice)
    r_eff = calculate_effective_radius(area, perimeter)
    
    return reslice_clean, r_eff, perimeter, area


def find_minimum_area_reslice(subvol, center_x, center_y, center_z, cdm_value):
    """
    Find the reslice orientation that minimizes cross-sectional area.
    Uses coarse-to-fine search: 45° -> 5° -> 1°
    """
    
    # Stage 1: Coarse search every 45 degrees
    angles_coarse = np.arange(0, 360, 45)
    best_area = float('inf')
    best_angles = None
    best_reslice = None
    
    for angle_xy in angles_coarse:
        for angle_xz in angles_coarse:
            for angle_yz in angles_coarse:
                reslice = extract_reslice(subvol, center_x, center_y, center_z,
                                         angle_xy, angle_xz, angle_yz, cdm_value)
                reslice_clean, area, _ = clean_and_measure_slice(reslice)
                
                if area < best_area:
                    best_area = area
                    best_angles = (angle_xy, angle_xz, angle_yz)
                    best_reslice = reslice_clean
    
    # Stage 2: Fine search every 5 degrees around best
    angles_fine = np.arange(-45, 50, 5)
    for d_xy in angles_fine:
        for d_xz in angles_fine:
            for d_yz in angles_fine:
                angle_xy = best_angles[0] + d_xy
                angle_xz = best_angles[1] + d_xz
                angle_yz = best_angles[2] + d_yz
                
                reslice = extract_reslice(subvol, center_x, center_y, center_z,
                                         angle_xy, angle_xz, angle_yz, cdm_value)
                reslice_clean, area, _ = clean_and_measure_slice(reslice)
                
                if area < best_area:
                    best_area = area
                    best_angles = (angle_xy, angle_xz, angle_yz)
                    best_reslice = reslice_clean
    
    # Stage 3: Finest search every 1 degree around best
    angles_finest = np.arange(-5, 6, 1)
    for d_xy in angles_finest:
        for d_xz in angles_finest:
            for d_yz in angles_finest:
                angle_xy = best_angles[0] + d_xy
                angle_xz = best_angles[1] + d_xz
                angle_yz = best_angles[2] + d_yz
                
                reslice = extract_reslice(subvol, center_x, center_y, center_z,
                                         angle_xy, angle_xz, angle_yz, cdm_value)
                reslice_clean, area, perimeter = clean_and_measure_slice(reslice)
                
                if area < best_area:
                    best_area = area
                    best_angles = (angle_xy, angle_xz, angle_yz)
                    best_reslice = reslice_clean
    
    # Calculate effective radius from best reslice
    _, final_area, final_perimeter = clean_and_measure_slice(best_reslice)
    r_eff = calculate_effective_radius(final_area, final_perimeter)
    
    return best_reslice, r_eff, final_perimeter, final_area


def extract_reslice(subvol, center_x, center_y, center_z, angle_xy, angle_xz, angle_yz, cdm_value):
    """
    Extract a 2D reslice at given angles through the center point.
    Similar to adjust_node_centroid logic.
    """
    
    # Create rotation matrices for each axis
    angle_xy_rad = np.radians(angle_xy)
    angle_xz_rad = np.radians(angle_xz)
    angle_yz_rad = np.radians(angle_yz)
    
    # Rotation matrix around Z axis (XY plane)
    Rz = np.array([
        [np.cos(angle_xy_rad), -np.sin(angle_xy_rad), 0],
        [np.sin(angle_xy_rad), np.cos(angle_xy_rad), 0],
        [0, 0, 1]
    ])
    
    # Rotation matrix around Y axis (XZ plane)
    Ry = np.array([
        [np.cos(angle_xz_rad), 0, np.sin(angle_xz_rad)],
        [0, 1, 0],
        [-np.sin(angle_xz_rad), 0, np.cos(angle_xz_rad)]
    ])
    
    # Rotation matrix around X axis (YZ plane)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(angle_yz_rad), -np.sin(angle_yz_rad)],
        [0, np.sin(angle_yz_rad), np.cos(angle_yz_rad)]
    ])
    
    # Combined rotation
    rotation_matrix = Rz @ Ry @ Rx
    
    # Define plane size
    plane_size = int(12 * cdm_value)
    half_size = plane_size // 2
    
    # Create meshgrid for plane coordinates
    y, x = np.meshgrid(np.arange(-half_size, half_size + 1),
                      np.arange(-half_size, half_size + 1))
    z = np.zeros_like(x)
    
    # Create the plane array
    plane = np.zeros((plane_size + 1, plane_size + 1), dtype=bool)
    
    # Stack coordinates and rotate
    coords = np.stack([x, y, z], axis=0).reshape(3, -1)
    rotated_coords = rotation_matrix @ coords
    
    # Translate coordinates to center point
    center_point = np.array([center_x, center_y, center_z])
    coords_3d = rotated_coords.T + center_point
    coords_3d = np.round(coords_3d).astype(int)
    
    # Find valid 3D coordinates
    mask = (coords_3d[:, 0] >= 0) & (coords_3d[:, 0] < subvol.shape[0]) & \
           (coords_3d[:, 1] >= 0) & (coords_3d[:, 1] < subvol.shape[1]) & \
           (coords_3d[:, 2] >= 0) & (coords_3d[:, 2] < subvol.shape[2])
    
    # Get valid coordinates and corresponding plane positions
    valid_coords = coords_3d[mask]
    valid_plane_positions = coords[:2, mask].T  # Use original x,y coordinates
    valid_plane_positions += half_size  # Shift to positive indices
    valid_plane_positions = np.clip(valid_plane_positions, 0, plane_size)
    
    # Sample values and assign to plane
    plane[valid_plane_positions[:, 0], valid_plane_positions[:, 1]] = \
        subvol[valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2]]
    
    return plane


def clean_and_measure_slice(reslice):
    """
    Keep only the central connected component and measure area and perimeter.
    """
    
    if not np.any(reslice):
        return reslice, 0, 0
    
    # Label connected components
    labeled, num_features = label(reslice)
    
    if num_features == 0:
        return reslice, 0, 0
    
    # Find central component
    center = np.array(reslice.shape) // 2
    central_label = labeled[center[0], center[1]]
    
    if central_label == 0:
        # If center is not in a component, find nearest component
        y, x = np.where(labeled > 0)
        if len(y) == 0:
            return reslice, 0, 0
        distances = np.sqrt((y - center[0])**2 + (x - center[1])**2)
        nearest_idx = np.argmin(distances)
        central_label = labeled[y[nearest_idx], x[nearest_idx]]
    
    # Keep only central component
    clean_slice = (labeled == central_label)
    
    # Measure area
    area = np.sum(clean_slice)
    
    # Measure perimeter using OpenCV
    array_uint8 = clean_slice.astype(np.uint8) * 255
    contours, _ = cv2.findContours(array_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if contours:
        perimeter = cv2.arcLength(contours[0], closed=True)
    else:
        perimeter = 0
    
    return clean_slice, area, perimeter


def calculate_effective_radius(area, perimeter):
    """
    Calculate effective radius from area and perimeter.
    r_eff = perimeter / (2 * pi)
    """
    
    if perimeter == 0:
        return 0
    
    r_eff = perimeter / (2 * np.pi)
    return r_eff


def visualize_subvolume(binary_subvolume, point_id, center_coords, max_cdm):
    """
    Create a 3D visualization of the extracted binary subvolume.
    """
    
    if not np.any(binary_subvolume):
        print(f"Warning: Point {point_id} has empty subvolume. Skipping visualization.")
        return
    
    if any(dim == 0 for dim in binary_subvolume.shape):
        print(f"Warning: Point {point_id} has zero-dimension subvolume. Skipping visualization.")
        return
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = np.zeros(binary_subvolume.shape + (4,), dtype=float)
    colors[binary_subvolume] = [0.5, 0.5, 0.5, 0.3]
    
    try:
        ax.voxels(binary_subvolume, facecolors=colors, edgecolor=None)
        
        center_local = np.array(binary_subvolume.shape) // 2
        ax.scatter([center_local[0]], [center_local[1]], [center_local[2]], 
                  c='red', marker='o', s=100, label='Center', zorder=10)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Point {point_id} - Subvolume\nCenter: {center_coords}, Max CDM: {max_cdm:.2f}')
        ax.legend()
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error visualizing point {point_id}: {e}")
        plt.close(fig)


def visualize_reslice(reslice, point_id, r_eff, perimeter, area, slice_type):
    """
    Visualize the 2D reslice that minimizes cross-sectional area.
    """
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(reslice, cmap='gray', interpolation='nearest')
    ax.set_title(f'Point {point_id} - {slice_type} Reslice\n'
                f'r_eff = {r_eff:.2f} voxels, Area = {area:.0f}, Perimeter = {perimeter:.2f}')
    ax.axis('off')
    plt.tight_layout()
    plt.show()

'''
def measure_effective_radii(
    tif_volume,
    tif_volume_cdm,
    points,
    thickness,
    sample_every_n,
    voxel_size,
    subvolume_edge_size_cdm_factor
):
    """
    Measure effective radii at sampled points along vessel centerlines.
    
    Parameters:
    -----------
    tif_volume : ndarray (bool)
        Binary segmentation volume
    tif_volume_cdm : ndarray (float)
        Chamfer distance map of the segmentation
    points : ndarray
        Array of shape (N, 4) with columns [index, x, y, z] in physical units
    thickness : ndarray
        Per-point thickness values (centerline radius estimates)
    sample_every_n : int
        Sample every nth point
    voxel_size : float
        Voxel size in physical units (microns)
    subvolume_edge_size_cdm_factor : float
        Multiplier for subvolume edge size relative to CDM value
    
    Returns:
    --------
    dict : Dictionary containing results with keys:
        - 't_cl': centerline thickness values at sampled points
        - 'r_eff': effective radius values at sampled points
        - 'sampled_points': sampled point coordinates
        - 'circularity': circularity metric for each measurement
    """
    
    # Results storage
    results = {
        't_cl': [],
        'r_eff': [],
        'sampled_points': [],
        'circularity': [],
        'max_cdm_values': []
    }
    
    # Sample points
    sampled_indices = np.arange(0, len(points), sample_every_n)
    
    print(f"Processing {len(sampled_indices)} sampled points...")
    
    for idx in sampled_indices:
        # Step 1: Extract coordinates and convert to voxel space
        point_data = points[idx]
        point_id = int(point_data[0])
        x_phys, y_phys, z_phys = point_data[1], point_data[2], point_data[3]
        
        # Convert to voxel coordinates
        x_vox = x_phys / voxel_size
        y_vox = y_phys / voxel_size
        z_vox = z_phys / voxel_size
        
        # Round to nearest integer voxel
        x_int = int(np.round(x_vox))
        y_int = int(np.round(y_vox))
        z_int = int(np.round(z_vox))
        
        # Check if point is within volume bounds
        if not (0 <= x_int < tif_volume.shape[0] and
                0 <= y_int < tif_volume.shape[1] and
                0 <= z_int < tif_volume.shape[2]):
            print(f"Point {point_id} at ({x_int}, {y_int}, {z_int}) is out of bounds. Skipping.")
            continue
        
        # Step 2: Get CDM value at this point to determine initial search size
        cdm_value_at_point = tif_volume_cdm[x_int, y_int, z_int]
        
        # Calculate initial subvolume edge size (in voxels)
        initial_edge_size = int(np.ceil(cdm_value_at_point * subvolume_edge_size_cdm_factor))
        half_edge = initial_edge_size // 2
        
        # Define bounds for initial CDM search subvolume
        x_min = max(0, x_int - half_edge)
        x_max = min(tif_volume_cdm.shape[0], x_int + half_edge + 1)
        y_min = max(0, y_int - half_edge)
        y_max = min(tif_volume_cdm.shape[1], y_int + half_edge + 1)
        z_min = max(0, z_int - half_edge)
        z_max = min(tif_volume_cdm.shape[2], z_int + half_edge + 1)
        
        # Step 3: Extract CDM subvolume and find maximum CDM value
        cdm_subvolume = tif_volume_cdm[x_min:x_max, y_min:y_max, z_min:z_max]
        max_cdm_value = np.max(cdm_subvolume)
        
        # Step 4: Calculate final subvolume edge size based on max CDM
        final_edge_size = int(np.ceil(max_cdm_value * subvolume_edge_size_cdm_factor))
        half_final = final_edge_size // 2
        
        # Define bounds for final binary volume extraction
        x_min_final = max(0, x_int - half_final)
        x_max_final = min(tif_volume.shape[0], x_int + half_final + 1)
        y_min_final = max(0, y_int - half_final)
        y_max_final = min(tif_volume.shape[1], y_int + half_final + 1)
        z_min_final = max(0, z_int - half_final)
        z_max_final = min(tif_volume.shape[2], z_int + half_final + 1)
        
        # Step 5: Extract binary subvolume
        binary_subvolume = tif_volume[x_min_final:x_max_final, 
                                      y_min_final:y_max_final, 
                                      z_min_final:z_max_final]
        
        # Step 6: Create volume rendering (for first few points as examples)
        if idx < 3 * sample_every_n:  # Visualize first 3 sampled points
            visualize_subvolume(binary_subvolume, point_id, 
                              (x_int, y_int, z_int), max_cdm_value)
        
        # Store intermediate results
        results['sampled_points'].append([x_int, y_int, z_int])
        
        # Store intermediate results
        results['sampled_points'].append([x_int, y_int, z_int])
        
        # Extract scalar thickness value - thickness[idx] is a list with one element
        thickness_value = float(thickness[idx][0])
        
        results['t_cl'].append(thickness_value)
        results['max_cdm_values'].append(max_cdm_value)
        
        # Placeholder for effective radius calculation (next step)
        # TODO: Calculate effective radius from binary_subvolume
        results['r_eff'].append(max_cdm_value)  # Temporary placeholder
        results['circularity'].append(1.0)  # Temporary placeholder
        
        results['t_cl'].append(thickness_value)
        results['max_cdm_values'].append(max_cdm_value)
        
        # Placeholder for effective radius calculation (next step)
        # TODO: Calculate effective radius from binary_subvolume
        results['r_eff'].append(max_cdm_value)  # Temporary placeholder
        results['circularity'].append(1.0)  # Temporary placeholder
    
    # Convert lists to arrays - ensure 1D arrays for scalar values
    results['t_cl'] = np.array(results['t_cl'], dtype=float).flatten()
    results['r_eff'] = np.array(results['r_eff'], dtype=float).flatten()
    results['circularity'] = np.array(results['circularity'], dtype=float).flatten()
    results['max_cdm_values'] = np.array(results['max_cdm_values'], dtype=float).flatten()
    results['sampled_points'] = np.array(results['sampled_points'])  # Keep as 2D (N, 3)
    
    return results
'''





def visualize_subvolume(binary_subvolume, point_id, center_coords, max_cdm):
    """
    Create a 3D visualization of the extracted binary subvolume.
    
    Parameters:
    -----------
    binary_subvolume : ndarray (bool)
        Binary subvolume to visualize
    point_id : int
        Point identifier
    center_coords : tuple
        (x, y, z) coordinates of the center point
    max_cdm : float
        Maximum CDM value found in search
    """
    # Check if subvolume has any True voxels
    if not np.any(binary_subvolume):
        print(f"Warning: Point {point_id} has empty subvolume. Skipping visualization.")
        return
    
    # Check if subvolume has valid dimensions
    if any(dim == 0 for dim in binary_subvolume.shape):
        print(f"Warning: Point {point_id} has zero-dimension subvolume. Skipping visualization.")
        return
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create color array with RGBA values (gray with alpha=0.3)
    colors = np.zeros(binary_subvolume.shape + (4,), dtype=float)
    colors[binary_subvolume] = [0.5, 0.5, 0.5, 0.3]  # Gray with 30% opacity
    
    try:
        # Plot voxels as cubes
        ax.voxels(binary_subvolume, facecolors=colors, edgecolor=None)
        
        # Mark center point as red dot
        center_local = np.array(binary_subvolume.shape) // 2
        ax.scatter([center_local[0]], [center_local[1]], [center_local[2]], 
                  c='red', marker='o', s=100, label='Center', zorder=10)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Point {point_id} - Subvolume\nCenter: {center_coords}, Max CDM: {max_cdm:.2f}')
        ax.legend()
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error visualizing point {point_id}: {e}")
        plt.close(fig)

'''
def plot_r_eff_vs_t_cl(result, voxel_size, min_circularity=0.0):
    """
    Plots measured effective radius (r_eff) vs. input thickness (t_cl).
    
    Parameters
    ----------
    result : dict
        Output dictionary from measure_effective_radii()
    min_circularity : float
        Optional filter to keep only points with circularity >= this threshold
    """

    t_cl = np.asarray(result["t_cl"], float)
    r_eff = np.asarray(result["r_eff"], float)
    circ = np.asarray(result.get("circularity", np.ones_like(t_cl)), float)

    mask = np.isfinite(t_cl) & np.isfinite(r_eff)
    if min_circularity > 0:
        mask &= (circ >= min_circularity)

    t_cl = t_cl[mask]
    r_eff = r_eff[mask]

    if len(t_cl) == 0:
        print("No valid points to plot after filtering.")
        return

    # Linear regression for quick calibration
    slope, intercept, r_value, p_value, std_err = linregress(t_cl, r_eff)
    r2 = r_value**2

    # Figure
    plt.figure(figsize=(6, 6))
    plt.scatter(t_cl, r_eff, s=18, c='royalblue', alpha=0.6, label='Sampled points')
    plt.plot([t_cl.min(), t_cl.max()],
             [t_cl.min(), t_cl.max()],
             'k--', lw=1.2, label='Ideal 1:1 line')
    plt.plot([t_cl.min(), t_cl.max()],
             intercept + slope * np.array([t_cl.min(), t_cl.max()]),
             'r-', lw=1.5, label=f'Fit: r_eff = {slope:.2f}·t_cl + {intercept:.2f}\nR² = {r2:.3f}')

    plt.xlabel("Centerline thickness (microns)")
    plt.ylabel("Measured effective radius (microns)")
    plt.title("Effective Radius vs. Centerline Thickness")
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

    # Print regression summary
    print(f"Linear fit: r_eff = {slope:.3f} * t_cl + {intercept:.3f}  (R² = {r2:.3f})")
    print(f"n = {len(t_cl)} samples,  mean residual = {(r_eff - (slope*t_cl+intercept)).mean():.3f}")
    
    return slope, intercept
'''

def plot_r_eff_vs_t_cl(result, voxel_size, min_circularity, constrain_intercept=True):
    """
    Plots measured effective radius (r_eff) vs. input thickness (t_cl).
    
    Parameters
    ----------
    result : dict
        Output dictionary from measure_effective_radii()
    voxel_size : float
        Voxel size in microns (default: 4.512)
    min_circularity : float
        Optional filter to keep only points with circularity >= this threshold
    constrain_intercept : bool
        If True, constrain intercept to be >= voxel_size/2
    """
    t_cl = np.asarray(result["t_cl"], float)
    r_eff = np.asarray(result["r_eff"], float)
    circ = np.asarray(result.get("circularity", np.ones_like(t_cl)), float)
    mask = np.isfinite(t_cl) & np.isfinite(r_eff)
    if min_circularity > 0:
        mask &= (circ >= min_circularity)
    t_cl = t_cl[mask]
    r_eff = r_eff[mask]
    if len(t_cl) == 0:
        print("No valid points to plot after filtering.")
        return
    
    # Calculate minimum intercept bound
    min_intercept = voxel_size / 2.0
    
    if constrain_intercept:
        # Unconstrained fit first to check if constraint is needed
        slope_unc, intercept_unc, r_value_unc, p_value, std_err = linregress(t_cl, r_eff)
        
        if intercept_unc < min_intercept:
            # Apply constraint: fit with fixed minimum intercept
            # For constrained fit: r_eff = slope * t_cl + min_intercept
            # Solve for slope that minimizes sum of squared residuals
            slope = np.sum((r_eff - min_intercept) * t_cl) / np.sum(t_cl ** 2)
            intercept = min_intercept
            
            # Calculate R² for constrained fit
            ss_res = np.sum((r_eff - (slope * t_cl + intercept)) ** 2)
            ss_tot = np.sum((r_eff - np.mean(r_eff)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            print(f"Intercept constraint applied: {intercept:.3f} >= {min_intercept:.3f}")
        else:
            # Use unconstrained fit
            slope = slope_unc
            intercept = intercept_unc
            r2 = r_value_unc ** 2
            print(f"Unconstrained fit used (intercept {intercept:.3f} > minimum {min_intercept:.3f})")
    else:
        # Unconstrained linear regression
        slope, intercept, r_value, p_value, std_err = linregress(t_cl, r_eff)
        r2 = r_value ** 2
    
    # Figure
    plt.figure(figsize=(6, 6))
    plt.scatter(t_cl, r_eff, s=18, c='royalblue', alpha=0.6, label='Sampled points')
    plt.plot([t_cl.min(), t_cl.max()],
             [t_cl.min(), t_cl.max()],
             'k--', lw=1.2, label='Ideal 1:1 line')
    
    fit_label = f'Fit: r_eff = {slope:.2f}·t_cl + {intercept:.2f}\nR² = {r2:.3f}'
    if constrain_intercept and intercept == min_intercept:
        fit_label += ' (constrained)'
    
    plt.plot([t_cl.min(), t_cl.max()],
             intercept + slope * np.array([t_cl.min(), t_cl.max()]),
             'r-', lw=1.5, label=fit_label)
    plt.xlabel("Centerline thickness (microns)")
    plt.ylabel("Measured effective radius (microns)")
    plt.title("Effective Radius vs. Centerline Thickness")
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()
    
    # Print regression summary
    print(f"Linear fit: r_eff = {slope:.3f} * t_cl + {intercept:.3f}  (R² = {r2:.3f})")
    print(f"n = {len(t_cl)} samples,  mean residual = {(r_eff - (slope*t_cl+intercept)).mean():.3f}")
    
    return slope, intercept



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


def write_to_conventional_am(am_file_write_path, first_points, segments, pts_per_segment, all_points, thickness):
    """
    Write Avizo HxSpatialGraph (.am) with:
      VERTEX coords from first_points
      EDGE connectivity from segments
      NumEdgePoints from pts_per_segment
      EdgePointCoordinates from all_points
      thickness from thickness
    All coordinates are written as z y x to match the original convention.
    """

    # Counts
    num_vertices = len(first_points)
    num_edges = len(segments)
    num_points = len(all_points)

    # Normalize pts_per_segment to flat ints
    def _to_int(x):
        return int(x[0]) if isinstance(x, (list, tuple)) else int(x)
    pts_per_segment_flat = [_to_int(x) for x in pts_per_segment]

    # Basic sanity checks
    if len(pts_per_segment_flat) != num_edges:
        raise ValueError(f"pts_per_segment length {len(pts_per_segment_flat)} != num_edges {num_edges}")
    if sum(pts_per_segment_flat) != num_points:
        raise ValueError(f"Sum(pts_per_segment) {sum(pts_per_segment_flat)} != num_points {num_points}")

    # Normalize thickness to flat floats
    def _to_float(x):
        return float(x[0]) if isinstance(x, (list, tuple)) else float(x)
    thickness_flat = [_to_float(t) for t in thickness]
    if len(thickness_flat) != num_points:
        raise ValueError(f"thickness length {len(thickness_flat)} != num_points {num_points}")

    # Header
    out = []
    out.append("# Avizo 3D ASCII 3.0\n")
    out.append(f"define VERTEX {num_vertices} \ndefine EDGE {num_edges} \ndefine POINT {num_points} \n\n")
    out.append("Parameters {\n    ContentType \"HxSpatialGraph\" \n} \n\n")
    out.append("VERTEX { float[3] VertexCoordinates } @1 \n")
    out.append("EDGE { int[2] EdgeConnectivity } @2 \n")
    out.append("EDGE { int NumEdgePoints } @3 \n")
    out.append("POINT { float[3] EdgePointCoordinates } @4 \n")
    out.append("POINT { float thickness } @5 \n\n")

    # @1 VERTEX coordinates (z y x). Ignore the first_points id column.
    out.append("@1\n")
    for v in first_points:
        # v = [id, x, y, z]
        x, y, z = float(v[1]), float(v[2]), float(v[3])
        out.append(f"{z:.15e} {y:.15e} {x:.15e}\n")

    # @2 EDGE connectivity (assumed already 0-based)
    out.append("\n@2\n")
    for e in segments:
        # e = [u, v]
        out.append(f"{int(e[0])} {int(e[1])}\n")

    # @3 NumEdgePoints
    out.append("\n@3\n")
    for n in pts_per_segment_flat:
        out.append(f"{n}\n")

    # @4 POINT EdgePointCoordinates (z y x). Ignore the all_points id column.
    out.append("\n@4\n")
    for p in all_points:
        # p = [id, x, y, z]
        x, y, z = float(p[1]), float(p[2]), float(p[3])
        out.append(f"{z:.15e} {y:.15e} {x:.15e}\n")

    # @5 POINT thickness
    out.append("\n@5\n")
    for t in thickness_flat:
        out.append(f"{t:.15e}\n")

    # Write file
    with open(am_file_write_path, "w") as f:
        f.write("".join(out))



def main():
    #-------------------------------------------FILE PATHS----------------------------------------------
    tif_read_path = r"C:\...\Image.tif" # (*.tif) 3D TIF Binary Segmentation
    am_read_path = r"C:\...\CenterlineTree_Skeleton.am"  # (*.am) Avizo ASCII Spatial Graph of the above TIF Image 
    am_write_path = r"C:\...\CenterlineTree_Skeleton_AdjustedRadii.am"  # (*.am) Avizo ASCII Spatial Graph - Adjusted point radius estimates with 2D reslice sampling and a constrained linear relationship assumption
    #------------------------------------------READ IN FILES--------------------------------------------
    print("Reading segmentation .tif file...")
    tif_volume = tiff.imread(tif_read_path).astype(bool)
    print("Calculating chamfer distance map...")
    tif_volume_cdm = distance_transform_cdt(tif_volume)
    print("Reading spatial graph .am file...")
    num_vertices, num_edges, num_points, nodes, segments, points_per_segment, points, thickness = read_am(am_read_path)
    #-------------------------------------------PARAMETERS----------------------------------------------
    voxel_size=4.512 # voxel size in microns
    sample_every_n=500 # samples every n points
    subvolume_edge_size_cdm_factor = 6 # 2D Reslice edge size - factor multiple of chamfer distance map. For collapsed vessel cross-sections ~= 5-8. For circular cross-sections ~= 2-5
    #------------------------------------------CALCULATE RESLICE EFFECTIVE RADII--------------------------------------------
    res = measure_effective_radii(
        tif_volume,                 # binary/uint8 segmentation
        tif_volume_cdm,         # float CDM, same shape
        points,                         # [pid, x, y, z]
        thickness,                   # per-point thickness
        sample_every_n,
        voxel_size,
        subvolume_edge_size_cdm_factor         
        )
    t_cl = res["t_cl"] # spatial graph thickness (a.k.a. radius estimate values)
    r_eff = res["r_eff"] # effective radius values from 2D reslice
    #-----------------------------------------------------PLOT--------------------------------------------------------------
    print("Plotting...")
    slope, intercept = plot_r_eff_vs_t_cl(res, voxel_size, min_circularity=0.8)
    print("Note: Check that constrained linear relationship assumption for thickness vs. effective radius is applicable in Figure 1 plot before adjusting thickness.")
    #-------------------------------------------ADJUST SPATIAL GRAPH THICKNESS----------------------------------------------
    print("Adjust spatial graph thickness...")
    #slope = 2.01     # If previously calculated (see plot_r_eff_vs_t_cl above)
    #intercept = 2.26 # If previously calculated (see plot_r_eff_vs_t_cl above)
    adjusted_thickness = adjust_thickness(slope, intercept, thickness)
    #--------------------------------------------WRITE SPATIAL GRAPH WITH ADJUSTED THICKNESS--------------------------------
    print("Writing spatial graph .am file...")
    write_to_conventional_am(am_write_path, nodes,segments, points_per_segment, points, adjusted_thickness)
    print("Done!")
    
if __name__ == "__main__":
    main()