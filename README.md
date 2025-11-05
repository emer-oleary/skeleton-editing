# 🩻 Skeleton Graph Editing Toolkit

Python scripts for reconstructing, cleaning, and analyzing 3D blood vessel skeletons from high-resolution CT segmentations (`.tif`) and Avizo SpatialGraph (`.am`) files. Designed for vascular connectivity, permeability, and flow modeling pipelines.

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

### 🔍 `replace_spatial_graph_point_thickness_with_2D_reslice_eff_radius.py`
Estimates effective vessel radius via 2D orthogonal reslices, fits a linear model (`r_eff = slope·t_cl + intercept`), and writes adjusted radius values back to the `.am`.
* **Input:** segmentation `.tif`, skeleton `.am`
* **Output:** adjusted `.am` + regression plot

### 🔗 `reconnect_disconnected_segments_in_spatial_graph.py`
Rebuilds missing links between end nodes using a cone-search and local geometry. Ideal for fixing broken or truncated vessels.
* **Input:** skeleton `.am` (+ optional segmentation bounds)
* **Output:** reconnected skeleton `.am`
  
### 🔧 `reformat_spatial_graph_after_removing_intermediate_nodes.py`
Reformats Avizo spatial graph files after removing intermediate nodes in the Filament editor. Strips out identified-graph data blocks and renumbers data markers to maintain compatibility with analysis scripts.
* **Input:** `.am` file (after Avizo Filament "Remove Intermediate Nodes")
* **Output:** `_reformatted.am` with cleaned structure
* **Use case:** Simplifies skeleton topology by keeping only branch points and endpoints, then ensures the file format matches expected structure for downstream processing.

### 🧬 `generate_skeleton_graph_volume.py`
Draws a voxelized 3D skeleton from an Avizo `.am` file. Each centerline point becomes a small sphere using its local radius.
* **Input:** `.am` (HxSpatialGraph) + reference `.tif`
* **Output:** binary `.tif` skeleton volume

### 🧩 `filter_segmentation_connected_components_by_skeleton_subgraph_num_segments.py`
Removes small 3D connected components based on their skeleton subgraph size. Keeps only regions with sufficient vessel connectivity.
* **Input:** segmentation `.tif`, skeleton `.am`
* **Output:** filtered segmentation `.tif`

### 📊 `calculate_vessel_network_properties.py`
Computes network-level metrics: radius, length, tortuosity, branching, intervessel distance, and more.
* **Input:** folder of `.am` files
* **Output:** metric CSVs + summary report

### 🎨 `plot_vessel_network_properties.py`
Creates publication-ready figures: combined violin-box plots, median overlays, and scatter distributions.
* **Input:** metric CSV (e.g., `intervessel_distances.csv`)
* **Output:** `.png` / `.svg` plots



## 🧭 Typical Workflow

The exact sequence of scripts depends on your goal.  
Below is an example workflow for **cleaning a vessel segmentation** using a reconnected skeleton graph.

---

### 🧪 Example: Cleaning a Vessel Segmentation with a Reconnected Skeleton Graph

1. **Start with a binary vessel segmentation** (`.tif`)  
   Run a skeletonization in **Avizo**, using the **Centerline Tree** module (see *Centerline Tree Parameters* below for slope, zeroVal, and parts guidance).

2. **Adjust the radius values** of the resulting spatial graph using  
   `replace_spatial_graph_point_thickness_with_2D_reslice_eff_radius.py`  
   This step refines each vessel's local radius using 2D reslice measurements.

3. **Reconnect missing or truncated segments** with  
   `reconnect_disconnected_segments_in_spatial_graph.py`  
   This reconstructs vessel continuity by bridging small gaps in the skeleton.

   **Post-processing in Avizo:** Import the reconnected spatial graph (`.am`) into Avizo and open the **Filament** tab. Select **Edit** → **Identify Graphs**, choose **All Graphs**, then click **Remove Intermediate Nodes Leaving Only Ending Nodes and Branching Nodes**. Return to the **Project** tab, select **File** → **Save Data As**, and save as **Avizo ascii SpatialGraph (*.am)**.

4. **Reformat the post-processed spatial graph** using  
   `reformat_spatial_graph_after_removing_intermediate_nodes.py`  
   This cleans the file structure after Avizo's intermediate node removal, ensuring compatibility with downstream analysis scripts.

5. **Filter the segmentation** using  
   `filter_segmentation_connected_components_by_skeleton_subgraph_num_segments.py`  
   This removes small or isolated components whose corresponding skeleton subgraphs have fewer than a specified number of connected segments.

6. **(Optional) Verify the reconnection visually** by generating a voxelized skeleton volume with  
   `generate_skeleton_graph_volume.py`  
   This produces a `.tif` rendering of the reconnected skeleton for overlay or inspection in 3D visualization tools.

7. **(Optional) Quantify and visualize network properties** using  
   `calculate_vessel_network_properties.py` and `plot_vessel_network_properties.py`  
   These scripts compute and plot geometric and topological metrics for the reconstructed vessel network.

---

💡 **In short:**  
> **Segmentation** (`.tif`) → **Centerline Tree** (`.am`) → **Adjust radii** → **Reconnect** → **Avizo post-processing** → **Reformat** → **Filter** → *(Optional)* **Visualize & Analyze**


> **Note on Centerline Tree parameters:**  
> You can use Avizo's default parameters or start with optimized values from [Walsh et al. (2024)](https://doi.org/10.1016/j.compbiomed.2024.108140): **Slope = 5.81**, **zeroVal = 2.6**, **Number of parts = -1** (always use -1 for binary segmentations).
> 
> These parameters control branch detection sensitivity. Each branch covers a tube-like volume where voxels within `slope × boundary_distance + zeroVal` are excluded from further endpoint searches, preventing excessive side branches. Lower slope values increase skeleton accuracy but may introduce false branches in collapsed or non-circular vessels.
> 
> For more details on the Centerline Tree algorithm, see [Sato et al. (2000) - TEASAR](https://doi.org/10.1109/PG.2000.852368).

## 🧩 Notes & Tips

* 🧠 **Axis order:** TIFF = `[Z, Y, X]`; Spatial graph = `[x, y, z]`. Index carefully to avoid flipped outputs.
* 🎯 **Reslice tuning:**
  * Collapsed vessels → `subvolume_edge_size_cdm_factor ≈ 5–8`
  * Round vessels → `≈ 2–5`

## 🧾 Citation

If you use this toolkit in academic work, please cite your project and link to this repository.
