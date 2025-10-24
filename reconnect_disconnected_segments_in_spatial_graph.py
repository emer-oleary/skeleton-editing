import tifffile as tiff
import numpy as np
from collections import defaultdict


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

 
def write_to_conventional_am(am_file_write_path, first_points,  segments, pts_per_segment, all_points, thickness):
    """
    Write Avizo HxSpatialGraph (.am) with:
      VERTEX coords from first_points
      EDGE connectivity from segments
      NumEdgePoints from pts_per_segment
      EdgePointCoordinates from all_points
      thickness from thickness

    Output coordinate order remains z y x to match the original convention.
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
        return float(x[0] if isinstance(x, (list, tuple)) else x)
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

    # @1 VERTEX coordinates (z y x), scaled by voxel_size. Ignore the first_points id column.
    out.append("@1\n")
    for v in first_points:
        x, y, z = float(v[1]), float(v[2]), float(v[3])
        out.append(f"{z:.15e} {y:.15e} {x:.15e}\n")

    # @2 EDGE connectivity (assumed already 0-based)
    out.append("\n@2\n")
    for e in segments:
        out.append(f"{int(e[0])} {int(e[1])}\n")

    # @3 NumEdgePoints
    out.append("\n@3\n")
    for n in pts_per_segment_flat:
        out.append(f"{n}\n")

    # @4 POINT EdgePointCoordinates (z y x), scaled by voxel_size. Ignore the all_points id column.
    out.append("\n@4\n")
    for p in all_points:
        x, y, z = float(p[1]), float(p[2]), float(p[3])
        out.append(f"{z:.15e} {y:.15e} {x:.15e}\n")

    # @5 POINT thickness, scaled by voxel_size
    out.append("\n@5\n")
    for t in thickness_flat:
        out.append(f"{(t):.15e}\n")

    # Write file
    with open(am_file_write_path, "w") as f:
        f.write("".join(out))


def detect_end_points(nodes, segments, volume_shape, eps=1e-6):
    """
    Find endpoint nodes (degree == 1) and filter to those that:
      1) are inside the volume bounds, and
      2) are NOT on any volume boundary face.

    Axis mapping (fixed):
      node (x, y, z) is checked against dims (Z, Y, X) as:
        x -> Z,  y -> Y,  z -> X

    Parameters
    ----------
    nodes : list[[int,float,float,float]]
        [[node_id, x, y, z], ...]
    segments : list[[int,int]]
        [[node_a, node_b], ...] undirected
    volume_shape : tuple(int)
        (Z, Y, X)
    eps : float
        Tolerance for boundary checks.

    Returns
    -------
    list[[int,float,float,float]]
        Filtered endpoints in the same format as `nodes`, sorted by node_id.
    """
    # Build degree per node from segments
    deg = defaultdict(int)
    for a, b in segments:
        a = int(a); b = int(b)
        deg[a] += 1
        deg[b] += 1

    # Candidates are nodes with degree 1
    endpoint_ids = [nid for nid, d in deg.items() if d == 1]
    if not endpoint_ids:
        return []

    # node_id -> (x,y,z)
    id_to_coord = {int(nid): (float(x), float(y), float(z)) for nid, x, y, z in nodes}

    Z, Y, X = map(float, volume_shape)  # tif is (Z, Y, X)

    filtered = []
    for nid in endpoint_ids:
        if nid not in id_to_coord:
            continue
        x, y, z = id_to_coord[nid]

        # New mapping: x→Z, y→Y, z→X
        in_bounds = (
            -eps <= x <= (Z - 1) + eps and
            -eps <= y <= (Y - 1) + eps and
            -eps <= z <= (X - 1) + eps
        )
        if not in_bounds:
            continue

        on_boundary = (
            abs(x - 0.0) <= eps or abs(x - (Z - 1)) <= eps or
            abs(y - 0.0) <= eps or abs(y - (Y - 1)) <= eps or
            abs(z - 0.0) <= eps or abs(z - (X - 1)) <= eps
        )
        if on_boundary:
            continue

        filtered.append([nid, x, y, z])

    filtered.sort(key=lambda r: r[0])
    return filtered


def _point_in_bounds_xyz(x, y, z, X, Y, Z, eps=1e-9):
    return (-eps <= x <= (X-1)+eps and
            -eps <= y <= (Y-1)+eps and
            -eps <= z <= (Z-1)+eps)

def _ray_box_first_hit(origin, direction, X, Y, Z, eps=1e-9):
    """
    Axis-aligned box: [0, X-1] x [0, Y-1] x [0, Z-1]
    Ray: origin + t * direction, t >= 0
    Returns (t_hit, hit_point (3,), face_normal (3,)) or (None, None, None).
    """
    o = np.asarray(origin, dtype=float)
    d = np.asarray(direction, dtype=float)
    # protect div-by-zero
    invd = np.where(np.abs(d) < eps, np.inf, 1.0 / d)

    # slabs
    t1x = (0.0   - o[0]) * invd[0]; t2x = ((X-1) - o[0]) * invd[0]
    tminx, tmaxx = (min(t1x, t2x), max(t1x, t2x))
    t1y = (0.0   - o[1]) * invd[1]; t2y = ((Y-1) - o[1]) * invd[1]
    tminy, tmaxy = (min(t1y, t2y), max(t1y, t2y))
    t1z = (0.0   - o[2]) * invd[2]; t2z = ((Z-1) - o[2]) * invd[2]
    tminz, tmaxz = (min(t1z, t2z), max(t1z, t2z))

    t_enter = max(tminx, tminy, tminz)
    t_exit  = min(tmaxx, tmaxy, tmaxz)
    if t_exit < 0 or t_enter > t_exit:
        return None, None, None

    t_hit = max(t_enter, 0.0)
    hit = o + t_hit * d

    # determine face normal
    tol = 1e-7
    face = np.array([0.0, 0.0, 0.0], dtype=float)
    if abs(t_hit - tminx) < tol:
        face = np.array([-1.0, 0.0, 0.0]) if d[0] > 0 else np.array([+1.0, 0.0, 0.0])
    elif abs(t_hit - tminy) < tol:
        face = np.array([0.0, -1.0, 0.0]) if d[1] > 0 else np.array([0.0, +1.0, 0.0])
    elif abs(t_hit - tminz) < tol:
        face = np.array([0.0, 0.0, -1.0]) if d[2] > 0 else np.array([0.0, 0.0, +1.0])
    else:
        # fallback: choose axis by which plane we are closest to
        # compare distances to planes at hit
        dx = min(abs(hit[0]-0.0), abs(hit[0]-(X-1)))
        dy = min(abs(hit[1]-0.0), abs(hit[1]-(Y-1)))
        dz = min(abs(hit[2]-0.0), abs(hit[2]-(Z-1)))
        if dx <= dy and dx <= dz:
            face = np.array([-1.0, 0.0, 0.0]) if d[0] > 0 else np.array([+1.0, 0.0, 0.0])
        elif dy <= dz:
            face = np.array([0.0, -1.0, 0.0]) if d[1] > 0 else np.array([0.0, +1.0, 0.0])
        else:
            face = np.array([0.0, 0.0, -1.0]) if d[2] > 0 else np.array([0.0, 0.0, +1.0])

    return t_hit, hit, face


def reconnect_end_points(
    end_point_nodes,
    nodes,
    segments,
    points_per_segment,
    points,
    thickness,
    volume_shape,                                   # (Z, Y, X)
    connect_to_boundaries,                          # bool (not used here)
    search_cone_angle,                              # degrees
    search_cone_length_factor,                      # scalar
    reconnection_segment_tortuosity_threshold,      # scalar
    reconnection_segment_radius_ratio_threshold,    # scalar
    *,
    eps=1e-9,
    n_curve_samples=300,
    anchor_max_k=10            #  max number of indices to look for anchor (default 5)
):
    """
    Endpoint reconnection with anchors:
      - Anchor = point up to ±anchor_max_k indices from endpoint (chosen by nearer neighbor direction),
        found by the same logic you’ve been using; anchor thickness clamps ≥ 0.5.
      - Use anchor thickness to set search-cone length (source side).
      - Use anchors as P0 and P3 for the Catmull–Rom segment (P1=src_node, P2=tgt_node).
      - Curve thickness = linear interpolation between the two anchor thicknesses.
      - Existing points between node and anchor get their thickness set to the anchor thickness.
    """

    # ---------- small utils ----------
    def _round2(v): return float(np.round(v, 2))

    def _is_boundary_node(x, y, z, Z, Y, X, tol=1e-6):
        return (abs(x - 0.0) <= tol or abs(x - (Z - 1)) <= tol or
                abs(y - 0.0) <= tol or abs(y - (Y - 1)) <= tol or
                abs(z - 0.0) <= tol or abs(z - (X - 1)) <= tol)

    def resample_by_arclength(polyline_xyz, target_spacing, tol=1e-9):
        pts = np.asarray(polyline_xyz, dtype=float)
        if pts.shape[0] < 2:
            return pts.copy()
        seg = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(s[-1])
        if total <= target_spacing + tol:
            return np.vstack([pts[0], pts[-1]])
        n_out = int(np.floor(total / target_spacing))
        targets = np.linspace(0.0, n_out * target_spacing, n_out + 1)
        out = []
        j = 0
        for st in targets:
            while j + 1 < len(s) and s[j+1] < st:
                j += 1
            if j + 1 >= len(s):
                out.append(pts[-1]); break
            denom = s[j+1] - s[j]
            t = 0.0 if denom <= tol else (st - s[j]) / denom
            out.append(pts[j] * (1 - t) + pts[j+1] * t)
        if np.linalg.norm(out[0] - pts[0]) > tol:
            out[0] = pts[0]
        if np.linalg.norm(out[-1] - pts[-1]) > tol:
            out.append(pts[-1])
        return np.asarray(out)

    def store_xyz_for_points(p, k, n, p0, p1):
        if k == 0:
            return float(p0[0]), float(p0[1]), float(p0[2])
        if k == n - 1:
            return float(p1[0]), float(p1[1]), float(p1[2])
        pr = np.round(p, 2)
        return float(pr[0]), float(pr[1]), float(pr[2])
    '''
    def catmull_rom_segment_centripetal(P0, P1, P2, P3, n_samples=200, alpha=0.5, eps=1e-9):
        def tj(ti, Pi, Pj): return ti + (np.linalg.norm(Pj - Pi) ** alpha)
        t0 = 0.0
        t1 = tj(t0, P0, P1)
        t2 = tj(t1, P1, P2)
        t3 = tj(t2, P2, P3)
        t = np.linspace(t1, t2, int(n_samples))
        A1 = ((t1 - t)/(t1 - t0 + eps))[:,None]*P0 + ((t - t0)/(t1 - t0 + eps))[:,None]*P1
        A2 = ((t2 - t)/(t2 - t1 + eps))[:,None]*P1 + ((t - t1)/(t2 - t1 + eps))[:,None]*P2
        A3 = ((t3 - t)/(t3 - t2 + eps))[:,None]*P2 + ((t - t2)/(t3 - t2 + eps))[:,None]*P3
        B1 = ((t2 - t)/(t2 - t0 + eps))[:,None]*A1 + ((t - t0)/(t2 - t0 + eps))[:,None]*A2
        B2 = ((t3 - t)/(t3 - t1 + eps))[:,None]*A2 + ((t - t1)/(t3 - t1 + eps))[:,None]*A3
        C  = ((t2 - t)/(t2 - t1 + eps))[:,None]*B1 + ((t - t1)/(t2 - t1 + eps))[:,None]*B2
        return C
    
    def cubic_hermite_curve(P0, P1, P2, P3, n_samples=200):
        """
        Cubic Hermite spline passing through P1 and P2.
        Tangents derived from neighboring points P0 and P3.
        """
        # Compute tangents (Catmull-Rom style)
        m1 = 0.5 * (P2 - P0)  # tangent at P1
        m2 = 0.5 * (P3 - P1)  # tangent at P2
        
        t = np.linspace(0, 1, n_samples)
        t2 = t * t
        t3 = t2 * t
        
        # Hermite basis functions
        h00 = 2*t3 - 3*t2 + 1
        h10 = t3 - 2*t2 + t
        h01 = -2*t3 + 3*t2
        h11 = t3 - t2
        
        curve = (h00[:, None] * P1 + 
                 h10[:, None] * m1 + 
                 h01[:, None] * P2 + 
                 h11[:, None] * m2)
        
        return curve
    
    def bezier_from_four_points(P0, P1, P2, P3, n_samples=200, tension=0.5):
        """
        Cubic Bezier passing through P1 and P2.
        Control points calculated from P0 and P3.
        """
        # Calculate control points
        d1 = np.linalg.norm(P1 - P0)
        d2 = np.linalg.norm(P2 - P1)
        d3 = np.linalg.norm(P3 - P2)
        
        # Direction vectors
        v1 = (P2 - P0) / (np.linalg.norm(P2 - P0) + 1e-9)
        v2 = (P3 - P1) / (np.linalg.norm(P3 - P1) + 1e-9)
        
        # Control points at fraction of segment length
        C1 = P1 + tension * d2 * v1
        C2 = P2 - tension * d2 * v2
        
        t = np.linspace(0, 1, n_samples)
        
        # Bezier basis
        b0 = (1-t)**3
        b1 = 3*(1-t)**2*t
        b2 = 3*(1-t)*t**2
        b3 = t**3
        
        curve = (b0[:, None] * P1 + 
                 b1[:, None] * C1 + 
                 b2[:, None] * C2 + 
                 b3[:, None] * P2)
        
        return curve
    
    def piecewise_cubic_smooth(P0, P1, P2, P3, n_samples=200):
        """
        Three cubic segments with C2 continuity at junctions.
        Only returns middle segment (P1 to P2).
        """
        from scipy.interpolate import CubicSpline
        
        # Build spline through all 4 points
        points = np.vstack([P0, P1, P2, P3])
        t_full = np.array([0, 1, 2, 3])
        
        cs = CubicSpline(t_full, points, bc_type='natural')
        
        # Sample only middle segment (t=1 to t=2)
        t_middle = np.linspace(1, 2, n_samples)
        curve = cs(t_middle)
        
        return curve
    '''
    def cubic_hermite_with_endpoint_directions(P0, P1, P2, P3, n_samples=200, tangent_scale=10.0):
        """
        Cubic Hermite spline where:
        - Curve passes through P1 and P2
        - Tangent at P1 follows direction P0→P1
        - Tangent at P2 follows direction P2→P3
        - tangent_scale controls how much the curve follows these directions
        """
        # Tangent at P1 follows P0→P1 direction
        v1 = P1 - P0
        d1 = np.linalg.norm(v1)
        if d1 > 1e-9:
            m1 = tangent_scale * v1  # pointing away from P0
        else:
            m1 = P2 - P1
        
        # Tangent at P2 follows P2→P3 direction
        v2 = P3 - P2
        d2 = np.linalg.norm(v2)
        if d2 > 1e-9:
            m2 = tangent_scale * v2  # pointing toward P3
        else:
            m2 = P2 - P1
        
        t = np.linspace(0, 1, n_samples)
        t2 = t * t
        t3 = t2 * t
        
        # Hermite basis functions
        h00 = 2*t3 - 3*t2 + 1      # P1 weight
        h10 = t3 - 2*t2 + t         # m1 weight
        h01 = -2*t3 + 3*t2          # P2 weight
        h11 = t3 - t2               # m2 weight
        
        curve = (h00[:, None] * P1 + 
                 h10[:, None] * m1 + 
                 h01[:, None] * P2 + 
                 h11[:, None] * m2)
        
        return curve

    # ---------- arrays & clamps ----------
    points_array = np.asarray(points, dtype=float)  # (M,4): [pid,x,y,z]
    if points_array.size == 0:
        return nodes, segments, points_per_segment, points, thickness

    points_xyz = points_array[:, 1:4]               # (M,3)

    # thickness as mutable array (we may edit existing points between node↔anchor)
    thickness_array = np.asarray(
        [float(t[0] if isinstance(t, (list, tuple, np.ndarray)) else t) for t in thickness],
        dtype=float
    )
    thickness_array = np.clip(thickness_array, 0.5, None)  # global clamp

    if thickness_array.shape[0] != points_xyz.shape[0]:
        raise ValueError("thickness length must equal points length")

    Z, Y, X = volume_shape

    # maps for matching node xyz to point idx
    exact_points_index = defaultdict(list)
    rounded_points_index = defaultdict(list)
    for idx, (px, py, pz) in enumerate(points_xyz):
        exact_points_index[(px, py, pz)].append(idx)
        rounded_points_index[(_round2(px), _round2(py), _round2(pz))].append(idx)

    def match_point_index_for_node(node_xyz):
        key_exact = (float(node_xyz[0]), float(node_xyz[1]), float(node_xyz[2]))
        lst = exact_points_index.get(key_exact)
        if lst:
            if len(lst) == 1:
                pi = lst[0]
            else:
                d2 = np.sum((points_xyz[lst] - node_xyz)**2, axis=1)
                pi = lst[int(np.argmin(d2))]
            return pi, points_xyz[pi]
        key_r2 = (_round2(node_xyz[0]), _round2(node_xyz[1]), _round2(node_xyz[2]))
        lst = rounded_points_index.get(key_r2)
        if lst:
            if len(lst) == 1:
                pi = lst[0]
            else:
                d2 = np.sum((points_xyz[lst] - node_xyz)**2, axis=1)
                pi = lst[int(np.argmin(d2))]
            return pi, points_xyz[pi]
        return None, None

    # Filter endpoints: no boundary
    filtered_endpoints = []
    for node_id, x, y, z in end_point_nodes:
        if _is_boundary_node(x, y, z, Z, Y, X):
            continue
        filtered_endpoints.append((int(node_id), np.array([float(x), float(y), float(z)], dtype=float)))

    eps2 = (eps*eps)
    node_info = {}  # node_id -> dict

    def nearest_nonidentical_point_index(node_xyz):
        diffs = points_xyz - node_xyz
        d2 = np.einsum('ij,ij->i', diffs, diffs)
        mask = d2 > eps2
        if not np.any(mask):
            return None
        return int(np.nonzero(mask)[0][np.argmin(d2[mask])])

    def _compute_anchor(node_point_index, node_xyz, adj_point_index, max_k=anchor_max_k):
        """
        Return (anchor_idx, anchor_xyz, anchor_thick) and update thickness_array between node↔anchor to anchor_thick.
        Direction chosen by nearer neighbor to the matched node.
        Accept first candidate within k steps that satisfies the same (≤10 voxels) distance rule used before.
        Promote to the max-thickness index among intermediates.
        """
        if node_point_index is None:
            return None, None, 0.5

        # choose index direction by neighbor proximity
        has_prev = node_point_index - 1 >= 0
        has_next = node_point_index + 1 < points_xyz.shape[0]
        if not has_prev and not has_next:
            dir_sign = 0
        else:
            d_prev = np.inf if not has_prev else float(np.linalg.norm(points_xyz[node_point_index-1] - node_xyz))
            d_next = np.inf if not has_next else float(np.linalg.norm(points_xyz[node_point_index+1] - node_xyz))
            dir_sign = +1 if d_next < d_prev else -1 if has_prev else +1

        # base thickness at matched point
        t0 = float(thickness_array[node_point_index])
        t0 = max(t0, 0.5)

        # search up to max_k
        anchor_idx = node_point_index
        base_xyz = node_xyz
        if dir_sign != 0:
            for k in range(int(max_k), 0, -1):
                cand = node_point_index + dir_sign * k
                if cand < 0 or cand >= points_xyz.shape[0]:
                    continue
                dist = float(np.linalg.norm(points_xyz[cand] - base_xyz))
                if dist <= (2*int(max_k)):  #  voxel distance criterion
                    # promote: pick index with max thickness in [min..max]
                    i0, i1 = (node_point_index, cand) if node_point_index <= cand else (cand, node_point_index)
                    seg_idxs = range(i0, i1 + 1)
                    anchor_idx = max(seg_idxs, key=lambda ix: float(thickness_array[ix]))
                    break

        anchor_xyz = points_xyz[anchor_idx]
        anchor_thick = max(float(thickness_array[anchor_idx]), 0.5)

        # update existing thickness along node↔anchor (inclusive) to anchor_thick
        lo, hi = (node_point_index, anchor_idx) if node_point_index <= anchor_idx else (anchor_idx, node_point_index)
        thickness_array[lo:hi+1] = anchor_thick

        return int(anchor_idx), anchor_xyz, anchor_thick

    # --- Precompute per-endpoint: indices, directions, anchors ---
    for node_id, node_xyz in filtered_endpoints:
        node_point_index, node_matched_xyz = match_point_index_for_node(node_xyz)
        if node_point_index is None:
            continue

        adj_idx = nearest_nonidentical_point_index(node_xyz)
        if adj_idx is None:
            continue

        # direction vector from a nearby index 
        dir_point_idx = None
        direction = -1 if adj_idx < node_point_index else +1
        base_xyz = points_xyz[node_point_index]
        for k in (anchor_max_k, *range(anchor_max_k-1, 0, -1)):  # try K..1
            cand_idx = int(node_point_index + direction * k)
            if 0 <= cand_idx < points_xyz.shape[0]:
                dist = float(np.linalg.norm(points_xyz[cand_idx] - base_xyz))
                if dist <= (2*int(anchor_max_k)):
                    dir_point_idx = cand_idx
                    break
        if dir_point_idx is None:
            continue

        dir_point_xyz = points_xyz[dir_point_idx]
        v_dir = node_xyz - dir_point_xyz
        nrm = np.linalg.norm(v_dir)
        if nrm <= eps:
            continue
        unit_dir = v_dir / nrm

        # compute anchor (limited to anchor_max_k) and update intermediate thicknesses
        anchor_idx, anchor_xyz, anchor_thick = _compute_anchor(
            node_point_index=node_point_index,
            node_xyz=node_xyz,
            adj_point_index=adj_idx,
            max_k=anchor_max_k
        )

        node_info[node_id] = dict(
            node_xyz=node_xyz,
            node_point_index=node_point_index,
            node_matched_xyz=node_matched_xyz,
            adj_point_index=adj_idx,
            dir_point_index=dir_point_idx,
            dir_point_xyz=dir_point_xyz,
            unit_dir=unit_dir,
            anchor_index=anchor_idx,
            anchor_xyz=anchor_xyz,
            anchor_thickness=anchor_thick
        )

    # Rebuild list thickness (reflecting node↔anchor updates)
    thickness = list(thickness_array)

    # Order by anchor thickness (desc) to prioritize stronger branches
    ordered_nodes = sorted(
        [nid for nid in node_info.keys()],
        key=lambda nid: node_info[nid]["anchor_thickness"],
        reverse=True
    )

    # set to keep connected nodes (both ends)
    used = set()

    for src_id in ordered_nodes:
        if src_id in used:
            continue

        src = node_info[src_id]
        src_xyz = src["node_xyz"]
        src_dir = src["unit_dir"]

        # --- NEW: search cone length uses source anchor thickness
        max_dist = float(search_cone_length_factor) * float(src["anchor_thickness"])

        # find candidates not used, closest first (angle + distance)
        best = None  # (dist, tgt_id, tgt_xyz)
        for tgt_id in ordered_nodes:
            if tgt_id == src_id or tgt_id in used:
                continue
            tgt = node_info[tgt_id]
            tgt_xyz = tgt["node_xyz"]

            v = tgt_xyz - src_xyz
            d = float(np.linalg.norm(v))
            if d <= eps or d > max_dist:
                continue
            v_hat = v / d
            ang = float(np.degrees(np.arccos(np.clip(np.dot(src_dir, v_hat), -1.0, 1.0))))
            if ang > float(search_cone_angle):
                continue

            if best is None or d < best[0]:
                best = (d, tgt_id, tgt_xyz)

        if best is None:
            continue

        d, tgt_id, tgt_xyz = best

        # thickness ratio using ANCHORS (not sampled endpoint thickness)
        tA = float(node_info[src_id]["anchor_thickness"])
        tB = float(node_info[tgt_id]["anchor_thickness"])
        thickness_ratio = max(tA, tB) / max(min(tA, tB), 1e-6)
        if thickness_ratio > float(reconnection_segment_radius_ratio_threshold):
            continue

        # distance cap again with avg anchor thickness
        avg_t = 0.5*(tA + tB)
        if d > avg_t * float(search_cone_length_factor):
            continue

        # --- NEW: anchors drive the curve geometry as P0 and P3 ---
        P0 = np.asarray(node_info[src_id]["anchor_xyz"], float)  # src anchor
        P1 = np.asarray(src_xyz, float)                          # src node
        P2 = np.asarray(tgt_xyz, float)                          # tgt node
        P3 = np.asarray(node_info[tgt_id]["anchor_xyz"], float)  # tgt anchor
        '''
        # Catmull Rom Curve
        curve = catmull_rom_segment_centripetal(
            P0, P1, P2, P3,
            n_samples=n_curve_samples, alpha=0.5
        )
        
        # Cubic Slpine Curve
        curve = cubic_hermite_curve(P0, P1, P2, P3, n_samples = 200)
        
        # Bezier 4 Points
        curve = bezier_from_four_points(P0, P1, P2, P3, n_samples=200, tension=0.5)
        
        # Piecewise cubic spline
        curve = piecewise_cubic_smooth(P0, P1, P2, P3, n_samples=200)
        '''
        # Endpoint Directions
        curve = cubic_hermite_with_endpoint_directions(P0, P1, P2, P3, n_samples=200, tangent_scale=10.0)
        
        seglen = np.linalg.norm(curve[1:] - curve[:-1], axis=1)
        curve_len = float(np.sum(seglen))
        if curve_len <= eps:
            continue
        tortuosity = curve_len / d
        if tortuosity > float(reconnection_segment_tortuosity_threshold):
            continue

        # resample and enforce exact node endpoints (trim outside)
        target_spacing = float(np.clip(0.9*min(tA, tB), 1.0, 1.8))
        polyline = resample_by_arclength(curve, target_spacing)
        # enforce exact node endpoints order: src_node → tgt_node
        if np.linalg.norm(polyline[0] - P1) > 1e-9:
            polyline = np.vstack([P1, polyline])
        if np.linalg.norm(polyline[-1] - P2) > 1e-9:
            polyline = np.vstack([polyline, P2])

        # append outputs
        segments.append([int(src_id), int(tgt_id)])
        points_per_segment.append(int(polyline.shape[0]))

        # thickness along curve = linear by arc length from tA to tB (anchors)
        if polyline.shape[0] >= 2:
            seg_arcs = np.linalg.norm(polyline[1:] - polyline[:-1], axis=1)
            s = np.concatenate([[0.0], np.cumsum(seg_arcs)])
            total = s[-1]
            frac = s / total if total > 0 else s
        else:
            frac = np.array([0.0])
        t_line = (1.0 - frac) * tA + frac * tB
        t_line = np.clip(t_line, 0.5, None)

        next_pid = int(max(p[0] for p in points)) + 1 if points else 0
        npl = polyline.shape[0]
        p0, p1 = P1, P2  # node endpoints for storage

        for k in range(npl):
            x, y, z = store_xyz_for_points(polyline[k], k, npl, p0, p1)
            points.append([next_pid, x, y, z])
            thickness.append(float(t_line[k]))
            next_pid += 1

        used.add(src_id)
        used.add(tgt_id)

    return nodes, segments, points_per_segment, points, thickness

'''
def reconnect_end_points(
    end_point_nodes,
    nodes,
    segments,
    points_per_segment,
    points,
    thickness,
    volume_shape,                                   # (Z, Y, X)
    connect_to_boundaries,                          # bool (not used here)
    search_cone_angle,                              # degrees
    search_cone_length_factor,                      # scalar
    reconnection_segment_tortuosity_threshold,      # scalar
    reconnection_segment_radius_ratio_threshold,    # scalar
    *,
    eps=1e-9,
    n_curve_samples=300,
    curvature_boost=1.5,      # extrapolate control points to increase curvature
    catmull_alpha=0.25        # alpha < 0.5 gives more curvature (uniform=0, centripetal=0.5)
):
    """
    Endpoint reconnection with:
      - Adaptive thickness sampling (<=10 voxels at ±5..1 indices) and in-between update.
      - Search-cone direction from the SAME nearby ±k point (<=10 voxels) used by the sampling rule:
          direction = node_xyz - dir_point_xyz, normalized.
          If no ±k within 10 voxels, skip this endpoint.
      - Global min thickness clamp = 0.5.
      - Search endpoints in descending order of sampled thickness.
      - Each endpoint can be connected at most once (both ends removed once connected).
      - Endpoints stored exact xyz; interior points rounded to 2 decimals.
      - Debug prints show initial and sampled thickness (source/target) and ratio.
    """

    # ---------- small utils ----------
    def _round2(v): return float(np.round(v, 2))

    def _is_boundary_node(x, y, z, Z, Y, X, tol=1e-6):
        return (abs(x - 0.0) <= tol or abs(x - (Z - 1)) <= tol or
                abs(y - 0.0) <= tol or abs(y - (Y - 1)) <= tol or
                abs(z - 0.0) <= tol or abs(z - (X - 1)) <= tol)

    def cubic_hermite_curve_3d(p0, m0, p1, m1, n_samples=200):
        t = np.linspace(0.0, 1.0, n_samples)
        t2 = t * t
        t3 = t2 * t
        h00 = 2*t3 - 3*t2 + 1
        h10 = t3 - 2*t2 + t
        h01 = -2*t3 + 3*t2
        h11 = t3 - t2
        return (h00[:, None] * p0 + h10[:, None] * m0 +
                h01[:, None] * p1 + h11[:, None] * m1)

    def resample_by_arclength(polyline_xyz, target_spacing, tol=1e-9):
        pts = np.asarray(polyline_xyz, dtype=float)
        if pts.shape[0] < 2:
            return pts.copy()
        seg = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(s[-1])
        if total <= target_spacing + tol:
            return np.vstack([pts[0], pts[-1]])
        n_out = int(np.floor(total / target_spacing))
        targets = np.linspace(0.0, n_out * target_spacing, n_out + 1)
        out = []
        j = 0
        for st in targets:
            while j + 1 < len(s) and s[j+1] < st:
                j += 1
            if j + 1 >= len(s):
                out.append(pts[-1]); break
            denom = s[j+1] - s[j]
            t = 0.0 if denom <= tol else (st - s[j]) / denom
            out.append(pts[j] * (1 - t) + pts[j+1] * t)
        if np.linalg.norm(out[0] - pts[0]) > tol:
            out[0] = pts[0]
        if np.linalg.norm(out[-1] - pts[-1]) > tol:
            out.append(pts[-1])
        return np.asarray(out)

    def store_xyz_for_points(p, k, n, p0, p1):
        if k == 0:
            return float(p0[0]), float(p0[1]), float(p0[2])
        if k == n - 1:
            return float(p1[0]), float(p1[1]), float(p1[2])
        pr = np.round(p, 2)
        return float(pr[0]), float(pr[1]), float(pr[2])
    
    def _endpoint_anchor_thickness(node_xyz):
        """
        Compute endpoint 'anchor' thickness per spec, without modifying existing thickness list.
        - match node_xyz -> idx
        - choose direction by neighbor distances
        - search k=10..1 with dist <= k * t0 (t0 = thickness[idx] clamped)
        - promote to max-thickness index among intermediates [idx .. cand] (inclusive)
        - return t_anchor (clamped >= 0.5)
        """
        # match endpoint to point index
        match_idx, _ = match_point_index_for_node(node_xyz)
        if match_idx is None:
            return 0.5  # fallback

        # immediate neighbors to choose direction
        has_prev = match_idx - 1 >= 0
        has_next = match_idx + 1 < points_xyz.shape[0]
        if not has_prev and not has_next:
            dir_sign = 0
        else:
            d_prev = np.inf if not has_prev else float(np.linalg.norm(points_xyz[match_idx-1] - node_xyz))
            d_next = np.inf if not has_next else float(np.linalg.norm(points_xyz[match_idx+1] - node_xyz))
            dir_sign = +1 if d_next < d_prev else -1 if has_prev else +1

        # base thickness at matched index (clamped)
        try:
            t0 = float(thickness[match_idx])
        except Exception:
            t0 = 0.5
        t0 = max(t0, 0.5)

        # search k = 10..1 for candidate within distance threshold
        anchor_idx = match_idx
        if dir_sign != 0:
            base_xyz = node_xyz
            for k in range(10, 0, -1):
                cand = match_idx + dir_sign * k
                if cand < 0 or cand >= points_xyz.shape[0]:
                    continue
                dist = float(np.linalg.norm(points_xyz[cand] - base_xyz))
                if dist <= k * t0:
                    # promote to the index with maximum thickness among intermediates (inclusive)
                    i0, i1 = (match_idx, cand) if match_idx <= cand else (cand, match_idx)
                    seg_idxs = range(i0, i1 + 1)
                    # robust unboxing + clamp in case any entries are nested
                    def _t(ix):
                        val = thickness[ix]
                        if isinstance(val, (list, tuple, np.ndarray)):
                            val = float(val[0])
                        else:
                            val = float(val)
                        return max(val, 0.5)
                    anchor_idx = max(seg_idxs, key=_t)
                    break
                
        # return anchor thickness (clamped)
        val = thickness[anchor_idx]
        if isinstance(val, (list, tuple, np.ndarray)):
            val = float(val[0])
        else:
            val = float(val)
        return max(val, 0.5)

    # ---------- arrays & clamps ----------
    points_array = np.asarray(points, dtype=float)  # (M,4): [pid,x,y,z]
    if points_array.size == 0:
        return nodes, segments, points_per_segment, points, thickness

    points_xyz = points_array[:, 1:4]               # (M,3)

    thickness_array = np.asarray(
        [float(t[0] if isinstance(t, (list, tuple, np.ndarray)) else t) for t in thickness],
        dtype=float
    )
    # Global clamp
    thickness_array = np.clip(thickness_array, 0.5, None)

    if thickness_array.shape[0] != points_xyz.shape[0]:
        raise ValueError("thickness length must equal points length")

    Z, Y, X = volume_shape

    # maps for matching node xyz to point idx
    exact_points_index = defaultdict(list)
    rounded_points_index = defaultdict(list)
    for idx, (px, py, pz) in enumerate(points_xyz):
        exact_points_index[(px, py, pz)].append(idx)
        rounded_points_index[(_round2(px), _round2(py), _round2(pz))].append(idx)

    def match_point_index_for_node(node_xyz):
        key_exact = (float(node_xyz[0]), float(node_xyz[1]), float(node_xyz[2]))
        lst = exact_points_index.get(key_exact)
        if lst:
            if len(lst) == 1:
                pi = lst[0]
            else:
                d2 = np.sum((points_xyz[lst] - node_xyz)**2, axis=1)
                pi = lst[int(np.argmin(d2))]
            return pi, points_xyz[pi]
        key_r2 = (_round2(node_xyz[0]), _round2(node_xyz[1]), _round2(node_xyz[2]))
        lst = rounded_points_index.get(key_r2)
        if lst:
            if len(lst) == 1:
                pi = lst[0]
            else:
                d2 = np.sum((points_xyz[lst] - node_xyz)**2, axis=1)
                pi = lst[int(np.argmin(d2))]
            return pi, points_xyz[pi]
        return None, None

    # Filter endpoints: no boundary
    filtered_endpoints = []
    for node_id, x, y, z in end_point_nodes:
        if _is_boundary_node(x, y, z, Z, Y, X):
            continue
        filtered_endpoints.append((int(node_id), np.array([float(x), float(y), float(z)], dtype=float)))

    # Per-endpoint precompute:
    #  - matched point index + initial thickness
    #  - a baseline nearest non-identical point index (to define index direction)
    #  - adaptive sampled thickness (<=10 voxels at ±5..1) and update thickness_array between indices
    #  - search-cone direction from the SAME nearby ±k point (<=10 voxels): dir = normalize(node_xyz - dir_point_xyz)
    eps2 = (eps*eps)
    node_info = {}  # node_id -> dict with all the above

    def nearest_nonidentical_point_index(node_xyz):
        diffs = points_xyz - node_xyz
        d2 = np.einsum('ij,ij->i', diffs, diffs)
        mask = d2 > eps2
        if not np.any(mask):
            return None
        return int(np.nonzero(mask)[0][np.argmin(d2[mask])])

    def adaptive_sample_and_update(node_point_index, adj_point_index,
                                   max_offset=5, max_distance_voxels=10.0):
        """
        Try offsets k ∈ {5,4,3,2,1} along direction determined by adj_point_index vs node_point_index.
        If distance between points_xyz[node_point_index] and points_xyz[node_point_index + dir*k] ≤ max_distance_voxels,
        use that index: set thickness for all indices between node and candidate (inclusive) to that value.
        Returns (sampled_thickness, sampled_index) if found, else (None, None).
        """
        if node_point_index is None or adj_point_index is None:
            return None, None
        direction = -1 if adj_point_index < node_point_index else +1
        base_xyz = points_xyz[node_point_index]
        for k in (5, 4, 3, 2, 1):
            cand_idx = int(node_point_index + direction * k)
            if cand_idx < 0 or cand_idx >= points_xyz.shape[0]:
                continue
            dist = float(np.linalg.norm(points_xyz[cand_idx] - base_xyz))
            if dist <= max_distance_voxels:
                sampled_val = max(float(thickness_array[cand_idx]), 0.5)
                lo, hi = sorted((node_point_index, cand_idx))
                thickness_array[lo:hi+1] = sampled_val
                return sampled_val, cand_idx
        return None, None
    
    def catmull_rom_segment_centripetal(P0, P1, P2, P3, n_samples=200, alpha=0.5, eps=1e-9):
        """
        Return centripetal Catmull–Rom points for the segment between P1 and P2.
        P0,P1,P2,P3 are 3D numpy arrays.
        """
        def tj(ti, Pi, Pj):
            return ti + (np.linalg.norm(Pj - Pi) ** alpha)
    
        t0 = 0.0
        t1 = tj(t0, P0, P1)
        t2 = tj(t1, P1, P2)
        t3 = tj(t2, P2, P3)
    
        t = np.linspace(t1, t2, int(n_samples))
        # Local lerps
        A1 = ((t1 - t)/(t1 - t0 + eps))[:,None]*P0 + ((t - t0)/(t1 - t0 + eps))[:,None]*P1
        A2 = ((t2 - t)/(t2 - t1 + eps))[:,None]*P1 + ((t - t1)/(t2 - t1 + eps))[:,None]*P2
        A3 = ((t3 - t)/(t3 - t2 + eps))[:,None]*P2 + ((t - t2)/(t3 - t2 + eps))[:,None]*P3
    
        B1 = ((t2 - t)/(t2 - t0 + eps))[:,None]*A1 + ((t - t0)/(t2 - t0 + eps))[:,None]*A2
        B2 = ((t3 - t)/(t3 - t1 + eps))[:,None]*A2 + ((t - t1)/(t3 - t1 + eps))[:,None]*A3
    
        C  = ((t2 - t)/(t2 - t1 + eps))[:,None]*B1 + ((t - t1)/(t2 - t1 + eps))[:,None]*B2
        return C

    # Precompute per-endpoint info (includes updating thickness_array)
    for node_id, node_xyz in filtered_endpoints:
        node_point_index, node_matched_xyz = match_point_index_for_node(node_xyz)
        if node_point_index is None:
            continue

        adj_idx = nearest_nonidentical_point_index(node_xyz)
        if adj_idx is None:
            continue

        initial_thick = float(thickness_array[node_point_index])

        # Get sampled thickness AND the sampled index (<=10 voxels at ±5..1)
        sampled_thick, sampled_idx = adaptive_sample_and_update(node_point_index, adj_idx)
        if sampled_thick is None:
            sampled_thick = initial_thick  # unchanged (already ≥ 0.5 due to clamp)

        # Compute search-cone direction from the SAME nearby index (prefer sampled_idx, else try ±4..1 fresh)
        dir_point_idx = None
        if sampled_idx is not None:
            dir_point_idx = sampled_idx
        else:
            # Try the same ±k search purely to find a dir point (no thickness update here)
            direction = -1 if adj_idx < node_point_index else +1
            base_xyz = points_xyz[node_point_index]
            for k in (5, 4, 3, 2, 1):
                cand_idx = int(node_point_index + direction * k)
                if 0 <= cand_idx < points_xyz.shape[0]:
                    dist = float(np.linalg.norm(points_xyz[cand_idx] - base_xyz))
                    if dist <= 10.0:
                        dir_point_idx = cand_idx
                        break

        if dir_point_idx is None:
            # No acceptable nearby point within 10 voxels → skip this endpoint
            continue

        dir_point_xyz = points_xyz[dir_point_idx]
        v_dir = node_xyz - dir_point_xyz
        nrm = np.linalg.norm(v_dir)
        if nrm <= eps:
            continue
        unit_dir = v_dir / nrm

        node_info[node_id] = dict(
            node_xyz=node_xyz,
            node_point_index=node_point_index,
            node_matched_xyz=node_matched_xyz,
            adj_point_index=adj_idx,
            dir_point_index=dir_point_idx,
            dir_point_xyz=dir_point_xyz,
            unit_dir=unit_dir,
            initial_thickness=initial_thick,
            sampled_thickness=sampled_thick
        )

    # Rebuild thickness list from the possibly updated array (so it reflects in-between updates)
    thickness = list(thickness_array)

    # Build order by sampled thickness (desc)
    ordered_nodes = sorted(
        [nid for nid in node_info.keys()],
        key=lambda nid: node_info[nid]["sampled_thickness"],
        reverse=True
    )

    # set to keep connected nodes (both ends)
    used = set()
    
    for src_id in ordered_nodes:
        if src_id in used:
            continue

        src = node_info[src_id]
        src_xyz = src["node_xyz"]
        src_dir = src["unit_dir"]
        src_pi  = src["node_point_index"]
        init_src_t = src["initial_thickness"]
        samp_src_t = src["sampled_thickness"]

        # search cone based on sampled source thickness
        max_dist = float(search_cone_length_factor) * float(samp_src_t)

        # find candidates not used, closest first
        best = None  # (dist, tgt_id, tgt_xyz, init_tgt_t, samp_tgt_t)

        for tgt_id in ordered_nodes:
            if tgt_id == src_id or tgt_id in used:
                continue
            tgt = node_info[tgt_id]
            tgt_xyz = tgt["node_xyz"]
            init_tgt_t = tgt["initial_thickness"]
            samp_tgt_t = tgt["sampled_thickness"]

            # distance + angle checks
            v = tgt_xyz - src_xyz
            d = float(np.linalg.norm(v))
            if d <= eps or d > max_dist:
                continue
            v_hat = v / d
            ang = float(np.degrees(np.arccos(np.clip(np.dot(src_dir, v_hat), -1.0, 1.0))))
            if ang > float(search_cone_angle):
                continue

            if best is None or d < best[0]:
                best = (d, tgt_id, tgt_xyz, init_tgt_t, samp_tgt_t)

        if best is None:
            continue

        d, tgt_id, tgt_xyz, init_tgt_t, samp_tgt_t = best

        thickness_ratio = max(samp_src_t, samp_tgt_t) / max(min(samp_src_t, samp_tgt_t), 1e-6)

        # thresholds
        if thickness_ratio > float(reconnection_segment_radius_ratio_threshold):
            continue

        avg_t = 0.5*(samp_src_t + samp_tgt_t)
        if d > avg_t * float(search_cone_length_factor):
            continue
    
        # Catmull–Rom (centripetal) through [adj_src, src_node, tgt_node, adj_tgt], segment P1→P2
        adj_src_xyz = node_info[src_id]["dir_point_xyz"]
        adj_tgt_xyz = node_info[tgt_id]["dir_point_xyz"]
        
        curve = catmull_rom_segment_centripetal(
            adj_src_xyz, src_xyz, tgt_xyz, adj_tgt_xyz,
            n_samples=n_curve_samples, alpha=0.5
        )
        
        seglen = np.linalg.norm(curve[1:] - curve[:-1], axis=1)
        curve_len = float(np.sum(seglen))
        if curve_len <= eps:
            continue
        tortuosity = curve_len / d
        if tortuosity > float(reconnection_segment_tortuosity_threshold):
            continue

        target_spacing = float(np.clip(0.9*min(samp_src_t, samp_tgt_t), 1.0, 1.8))
        polyline = resample_by_arclength(curve, target_spacing)
        if np.linalg.norm(polyline[0] - src_xyz) > 1e-9:
            polyline = np.vstack([src_xyz, polyline])
        if np.linalg.norm(polyline[-1] - tgt_xyz) > 1e-9:
            polyline = np.vstack([polyline, tgt_xyz])

        # append outputs
        segments.append([int(src_id), int(tgt_id)])
        points_per_segment.append(int(polyline.shape[0]))

        # endpoint anchor thicknesses per spec 
        tA = _endpoint_anchor_thickness(src_xyz)
        tB = _endpoint_anchor_thickness(tgt_xyz)
        
        print(f"[RECONNECT] src_id={src_id} tgt_id={tgt_id}")
        print(f"  tA={tA:.3f}  tB={tB:.3f}  d={d:.2f}")
        
        # thickness along curve (linear by arc length from tA to tB, clamp ≥0.5)
        if polyline.shape[0] >= 2:
            seg_arcs = np.linalg.norm(polyline[1:] - polyline[:-1], axis=1)
            s = np.concatenate([[0.0], np.cumsum(seg_arcs)])
            total = s[-1]
            frac = s / total if total > 0 else s
        else:
            frac = np.array([0.0])
        t_line = (1.0 - frac) * tA + frac * tB
        t_line = np.clip(t_line, 0.5, None)
        
        print(f"  t_line range: [{float(t_line.min()):.3f}, {float(t_line.max()):.3f}]  n={len(t_line)}")

        next_pid = int(max(p[0] for p in points)) + 1 if points else 0
        npl = polyline.shape[0]
        p0, p1 = src_xyz, tgt_xyz
        for k in range(npl):
            x, y, z = store_xyz_for_points(polyline[k], k, npl, p0, p1)
            points.append([next_pid, x, y, z])
            t_val = max(float(t_line[k]), 0.5)  # keep ≥ 0.5
            thickness.append(t_val)
            next_pid += 1

        # mark BOTH endpoints as used so each gets max 1 connection
        used.add(src_id)
        used.add(tgt_id)

    return nodes, segments, points_per_segment, points, thickness
'''

def main():
    #-------------------------------------------FILE PATHS----------------------------------------------
    tif_read_path = r"C:\...\Image.tif" # (*.tif) 3D TIF Binary Segmentation
    am_read_path = r"C:\...\CenterlineTree_Skeleton.am" # (*.am) Avizo ASCII Spatial Graph
    am_write_path = r"C:\...\CenterlineTree_Skeleton_Reconnected.am" # (*.am) Avizo ASCII Spatial Graph - Reconnected some disconnected segment end points with a scaled cone-search aproach
    #-------------------------------------------PARAMETERS----------------------------------------------
    connect_to_boundaries = False # False for whole-organ networks, True for zoom regions / tissue biopsies (Note: True has not been tested yet so use False only)
    search_cone_angle = 50 # reconnection cone angle max. deviation from vessel direction (in degrees)
    search_cone_length_factor = 15 # reconection cone length = multiplication factor for point thickness (a.k.a. radius estimate)
    reconnection_segment_tortuosity_threshold = 1.8 # Max. allowed tortuosity of a reconnection segment (node->node curve length divided by straight-line length)
    reconnection_segment_radius_ratio_threshold = 5 # Max. allowed factor difference in node:node reconnection thickness (a.k.a. radius estimate)
    voxel_size = 4.512 # isotropic voxel size (usually in microns). Make sure this voxel size was applied when loading the segmenation image for input spatial graph (am_read_path) generation.
    #------------------------------------------READ IN FILES--------------------------------------------
    print("Reading segmentation .tif file...")
    tif_volume = tiff.imread(tif_read_path).astype(bool)
    scaled_tif_volume_shape = tuple(np.array(tif_volume.shape) * voxel_size)
    print("Reading spatial graph .am file...")
    num_vertices, num_edges, num_points, nodes, segments, points_per_segment, points, thickness = read_am(am_read_path)
    #-----------------------------------------RECONNECT END POINTS--------------------------------------------
    print("Detecting end points...")
    end_point_nodes = detect_end_points(nodes, segments, scaled_tif_volume_shape)
    print("Connecting end points...")
    connected_nodes, connected_segments, connected_points_per_segment, connected_points, connected_thickness = reconnect_end_points(end_point_nodes, nodes, segments, points_per_segment, points, thickness, scaled_tif_volume_shape, connect_to_boundaries, search_cone_angle, search_cone_length_factor, reconnection_segment_tortuosity_threshold, reconnection_segment_radius_ratio_threshold)
    #---------------------------------------WRITE FILES--------------------------------------------------------
    print("Writing spatial graph .am file...")
    write_to_conventional_am(am_write_path, connected_nodes, connected_segments, connected_points_per_segment, connected_points, connected_thickness)
    print("Done!\n")
    print("Note:")
    print("- Remember to edit the output spatial graph in Avizo to remove intermediate nodes at non-branch or non-end-point locations.")
    print("- ^ Instructions: Open the output spatial graph in Avizo -> Filament tab -> Edit tab -> Indentify Graphs (and select all graphs) -> Select 'Remove intermediate nodes leaving only ending nodes and branching nodes' -> Project tab -> Save spatial graph (File->Save Data as...) as File Type 'Avizo ascii SpatialGraph(*am)'.")
    print("- Once complete, you can calculate and plot the spatial graph statitics (segment radius, segment length, branching angle etc.)")
  
    
if __name__ == "__main__":
    main()
