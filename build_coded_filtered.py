import pandas as pd
import pyvista as pv
import numpy as np
import sys

from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray, vtkLine
from vtkmodules.util import numpy_support

# Optional fallback using VTK directly
try:
    import vtk
    HAVE_VTK = True
except Exception:
    HAVE_VTK = False

# ----------------------
# Input files (edit)
# ----------------------
NODES_FILE = "node_regions_clean.csv"
EDGES_FILE = "/Users/vaw1/Downloads/OGB/BALBc_no1_raw/BALBc-no1_iso3um_stitched_segmentation_bulge_size_3.0_edges.csv"
OUT_VTP = "nodes_edges_filtered.vtp"

# ----------------------
# Load data
# ----------------------
nodes = pd.read_csv(NODES_FILE, sep=';')
edges = pd.read_csv(EDGES_FILE, sep=';')

print("Loaded nodes:", len(nodes), "edges:", len(edges))

# ----------------------
# Filter edges (adjust thresholds if desired)
# ----------------------
MIN_LENGTH = 5.0
MIN_VOLUME = 50

edges_filtered = edges[(edges['length'] >= MIN_LENGTH) & (edges['volume'] >= MIN_VOLUME)].copy()
print("Filtered edges:", len(edges_filtered))

# ----------------------
# Keep nodes in edges
# ----------------------
node_ids_in_edges = set(edges_filtered['node1id']).union(edges_filtered['node2id'])
nodes_filtered = nodes[nodes['id'].isin(node_ids_in_edges)].copy()
print("Filtered nodes:", len(nodes_filtered))

# ----------------------
# Reindex nodes (old_id -> new_idx)
# ----------------------
old_ids = nodes_filtered['id'].to_numpy()
old_to_new_idx = {old_id: new_idx for new_idx, old_id in enumerate(old_ids)}
nodes_filtered['new_idx'] = nodes_filtered['id'].map(old_to_new_idx)

# Map edges to new indices and drop missing
edges_filtered['node1_new'] = edges_filtered['node1id'].map(old_to_new_idx)
edges_filtered['node2_new'] = edges_filtered['node2id'].map(old_to_new_idx)
edges_filtered = edges_filtered.dropna(subset=['node1_new','node2_new']).copy()
edges_filtered['node1_new'] = edges_filtered['node1_new'].astype(np.int64)
edges_filtered['node2_new'] = edges_filtered['node2_new'].astype(np.int64)

N_edges = len(edges_filtered)
N_points = len(nodes_filtered)
print("After reindexing: points:", N_points, "edges:", N_edges)

# ----------------------
# Build points array
# ----------------------
points = nodes_filtered[['pos_x','pos_y','pos_z']].to_numpy(dtype=np.float64)
print("points.shape:", points.shape, "dtype:", points.dtype)

# ----------------------
# Build lines in vectorized manner
# ----------------------
e = edges_filtered[['node1_new','node2_new']].to_numpy(dtype=np.int64)
print("e.shape:", e.shape, "dtype:", e.dtype, "min/max:", e.min(), e.max())

# safety checks
if e.shape[0] != N_edges:
    raise RuntimeError("Edge row count mismatch after mapping.")

# Create counts column (value 2 for each edge)
counts = np.full((N_edges, 1), 2, dtype=np.int64)

# 3-column (N,3) array
lines_np = np.hstack((counts, e))   # shape (N_edges, 3)
print("lines_np.shape:", lines_np.shape, "dtype:", lines_np.dtype)
# Flatten to 1D contiguous array
lines_flat = np.ascontiguousarray(lines_np.reshape(-1))
print("lines_flat.shape:", lines_flat.shape, "dtype:", lines_flat.dtype)

# Quick validation: length should be exactly 3 * N_edges
expected_len = 3 * N_edges
print("expected_len (3*N_edges):", expected_len)
if lines_flat.size != expected_len:
    raise RuntimeError(f"Flattened lines length {lines_flat.size} != expected {expected_len}")

# Also ensure all indices fall within [0, N_points-1]
if e.size and ((e < 0).any() or (e >= N_points).any()):
    bad_min = int(e.min()); bad_max = int(e.max())
    raise RuntimeError(f"Edge indices out of bounds: min {bad_min}, max {bad_max}, N_points {N_points}")

# ----------------------
# Try PyVista path (preferred)
# ----------------------
cloud = pv.PolyData(points)
print("Created PolyData: n_points:", cloud.n_points)

# assign point data safely
# region -> simple integer mapping (first region)
all_regions = sorted({r for sublist in nodes_filtered['regions'] for r in eval(sublist)})
region_to_int = {r: i for i, r in enumerate(all_regions)}
region_ids = np.array([region_to_int[eval(r)[0]] for r in nodes_filtered['regions']], dtype=np.int32)

cloud["region_id"] = region_ids
cloud["degree"] = nodes_filtered['degree'].to_numpy(dtype=np.float32)
cloud["isAtSampleBorder"] = nodes_filtered['isAtSampleBorder'].to_numpy(dtype=np.int8)

# Now set lines: must be 1D contiguous int array
lines_flat = lines_flat.astype(np.int64)
try:
    cloud.lines = lines_flat
except Exception as ex:
    print("PyVista failed to set cloud.lines:", repr(ex))
    # fall back to VTK below
else:
    print("Assigned cloud.lines. cloud.n_cells:", cloud.n_cells)
    # sanity check
    if cloud.n_cells != N_edges:
        print("WARNING: cloud.n_cells DOES NOT equal N_edges.")
        print("cloud.n_cells =", cloud.n_cells, "N_edges =", N_edges)
    else:
        print("cloud.n_cells matches N_edges.")
    # assign cell scalars
    edge_lengths = edges_filtered['length'].to_numpy(dtype=np.float32)
    # final check: cell count must match
    if cloud.n_cells == N_edges:
        cloud.cell_data['edge_length'] = edge_lengths
        print("Assigned edge_length to cell_data.")
        cloud.save(OUT_VTP)
        print("Saved (PyVista) ->", OUT_VTP)
        sys.exit(0)
    else:
        print("PyVista path: cell-count mismatch -> falling back to VTK writer.")

# ----------------------
# Fallback: build vtk objects directly (robust)
# ----------------------
if not HAVE_VTK:
    raise RuntimeError("PyVista failed and vtk is not available as a fallback. Install vtk or report diagnostics printed above.")

print("Using VTK fallback (more verbose but robust).")

# Create vtkPoints
vtk_points = vtk.vtkPoints()
vtk_points.SetData(numpy_support.numpy_to_vtk(points.astype(np.float32), deep=True))

# Create vtkCellArray and vtkIdTypeArray for cells
id_array = vtk.vtkIdTypeArray()
id_array.SetNumberOfComponents(1)
id_array.SetNumberOfTuples(lines_flat.size)
# vtkIdTypeArray requires flat sequence of ids; but we must insert as vtkCellArray expects
# We'll use vtkCellArray.InsertNextCell with each line's two point ids
cells = vtk.vtkCellArray()
cells.InitTraversal()

for i in range(N_edges):
    n1 = int(e[i,0])
    n2 = int(e[i,1])
    cells.InsertNextCell(2)
    cells.InsertCellPoint(n1)
    cells.InsertCellPoint(n2)

# Build polydata
poly = vtk.vtkPolyData()
poly.SetPoints(vtk_points)
poly.SetLines(cells)

# Add point data (region_id, degree, isAtSampleBorder)
from vtk.util import numpy_support
poly.GetPointData().AddArray(numpy_support.numpy_to_vtk(region_ids, deep=True, array_type=vtk.VTK_INT))
poly.GetPointData().SetActiveScalars("RegionID")
# But better to name arrays:
arr_region = numpy_support.numpy_to_vtk(region_ids, deep=True)
arr_region.SetName("region_id")
poly.GetPointData().AddArray(arr_region)

arr_deg = numpy_support.numpy_to_vtk(nodes_filtered['degree'].to_numpy(dtype=np.float32), deep=True)
arr_deg.SetName("degree")
poly.GetPointData().AddArray(arr_deg)

arr_border = numpy_support.numpy_to_vtk(nodes_filtered['isAtSampleBorder'].to_numpy(dtype=np.int8), deep=True)
arr_border.SetName("isAtSampleBorder")
poly.GetPointData().AddArray(arr_border)

# Add cell data (edge_length)
arr_edge_len = numpy_support.numpy_to_vtk(edges_filtered['length'].to_numpy(dtype=np.float32), deep=True)
arr_edge_len.SetName("edge_length")
poly.GetCellData().AddArray(arr_edge_len)

# Write out using vtkXMLPolyDataWriter
writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName(OUT_VTP)

# Use binary for smaller size
writer.SetDataModeToBinary()
if vtk.VTK_MAJOR_VERSION >= 9:
    # some vtk versions require SetInputData
    writer.SetInputData(poly)
else:
    writer.SetInput(poly)
writer.Write()
print("Saved (VTK fallback) ->", OUT_VTP)
