# 🧠 Skeleton Editing Toolkit

Python scripts for reconstructing, cleaning, and analyzing 3D vessel skeletons from high-resolution micro-CT segmentations and Avizo HxSpatialGraph (`.am`) files. Designed for vascular connectivity, permeability, and flow modeling pipelines.

## ⚙️ Environment Setup

You can use either conda or pip:
```bash
# 🧩 Create environment (recommended)
conda env create -f skeleton_env.yml
conda activate skeleton_env

# 💡 Or install directly
pip install -U numpy scipy tifffile scikit-image pandas matplotlib seaborn networkx opencv-python
```

## 🗂️ Script Overview

### 🧬 `generate_skeleton_graph_volume.py`
Draws a voxelized 3D skeleton from an Avizo `.am` file. Each centerline point becomes a small sphere using its local radius.
* **Input:** `.am` (HxSpatialGraph) + reference `.tif`
* **Output:** binary `.tif` skeleton volume

### 🧩 `filter_segmentation_connected_components_by_skeleton_subgraph_num_segments.py`
Removes small 3D connected components based on their skeleton subgraph size. Keeps only regions with sufficient vessel connectivity.
* **Input:** segmentation `.tif`, skeleton `.am`
* **Output:** filtered segmentation `.tif`

### 🔗 `reconnect_disconnected_segments_in_spatial_graph.py`
Rebuilds missing links between end nodes using a cone-search and local geometry. Ideal for fixing broken or truncated vessels.
* **Input:** skeleton `.am` (+ optional segmentation bounds)
* **Output:** reconnected skeleton `.am`

### 📊 `calculate_vessel_network_properties.py`
Computes network-level metrics: radius, length, tortuosity, branching, intervessel distance, and more.
* **Input:** folder of `.am` files
* **Output:** metric CSVs + summary report

### 🎨 `plot_vessel_network_properties.py`
Creates publication-ready figures: combined violin-box plots, median overlays, and scatter distributions.
* **Input:** metric CSV (e.g., `intervessel_distances.csv`)
* **Output:** `.png` / `.svg` plots

### 🔍 `replace_spatial_graph_point_thickness_with_2D_reslice_eff_radius.py`
Estimates effective vessel radius via 2D orthogonal reslices, fits a linear model (`r_eff = slope·t_cl + intercept`), and writes adjusted radius values back to the `.am`.
* **Input:** segmentation `.tif`, skeleton `.am`
* **Output:** adjusted `.am` + regression plot

## 🧭 Typical Workflow
```
1️⃣  Start from a binary segmentation (.tif)
2️⃣  Generate a skeleton in Avizo  (.am) (Centerline Tree recommended - see note below re parameters)
3️⃣  Reconnect broken vessel segments
4️⃣  Filter small, disconnected components
5️⃣  Compute vessel metrics
6️⃣  Plot and compare algorithms
7️⃣  (Optional) Refit radii using 2D reslices
```

> **Note on Centerline Tree parameters:**  
> You can use Avizo's default parameters or start with optimized values from [Walsh et al. (2024)](https://doi.org/10.1016/j.compbiomed.2024.108140): **Slope = 5.81**, **zeroVal = 2.6**, **Number of parts = -1** (always use -1 for binary segmentations).
> 
> These parameters control branch detection sensitivity. Each branch covers a tube-like volume where voxels within `slope × boundary_distance + zeroVal` are excluded from further endpoint searches, preventing excessive side branches. Lower slope values increase skeleton accuracy but may introduce false branches in collapsed or non-circular vessels.
> 
> For more details on the Centerline Tree algorithm, see [Sato et al. (2000) - TEASAR](https://doi.org/10.1109/PG.2000.852368).

## 🧩 Notes & Tips

* 🧠 **Axis order:** TIFF = `[Z, Y, X]`; Spatial graph = `[x, y, z]`. Index carefully to avoid flipped outputs.
* ⚡ **Distance maps:** Prefer Euclidean EDT for precision; Chamfer CDT for speed.
* 🎯 **Reslice tuning:**
  * Collapsed vessels → `subvolume_edge_size_cdm_factor ≈ 5–8`
  * Round vessels → `≈ 2–5`
* 🪶 **Performance:** Use coarse-to-fine reslice search or local PCA to reduce runtime.

## 🧾 Citation

If you use this toolkit in academic work, please cite your project and link to this repository.

## 🪪 License

MIT License (recommended). Free for academic and non-commercial use.
