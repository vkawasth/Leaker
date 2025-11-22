import pandas as pd
import vtk

nodes_csv = "/Users/vaw1/Downloads/OGB/BALBc_no1_raw/BALBc-no1_iso3um_stitched_segmentation_bulge_size_3.0_nodes.csv"
edges_csv = "/Users/vaw1/Downloads/OGB/BALBc_no1_raw/BALBc-no1_iso3um_stitched_segmentation_bulge_size_3.0_edges.csv"
output_path = "/Users/vaw1/Downloads/OGB/para_BALBc_no1.vtp"


def export_vtp_safe(nodes_csv, edges_csv, output_path, chunk_size=None):
    """
    Export Voreen-style vessel graph to ParaView .vtp
    - Handles real Z coordinates
    - Filters edges referencing missing nodes
    - Ensures CellData arrays match number of lines
    - Optional chunk_size for very large graphs (not implemented fully here)
    """
    
    # Load full CSVs
    nodes = pd.read_csv(nodes_csv, sep=';')
    edges = pd.read_csv(edges_csv, sep=';')

    # Ensure numeric coordinates
    for c in ['pos_x', 'pos_y', 'pos_z']:
        nodes[c] = pd.to_numeric(nodes[c], errors='coerce').fillna(0)

    # Map node ID -> VTK point index
    points = vtk.vtkPoints()
    id_to_vtk = {}
    for idx, row in nodes.iterrows():
        pid = points.InsertNextPoint(row.pos_x, row.pos_y, row.pos_z)
        id_to_vtk[int(row.id)] = pid

    # Filter edges to only include valid node IDs
    edges = edges[
        edges['node1id'].isin(id_to_vtk.keys()) &
        edges['node2id'].isin(id_to_vtk.keys())
    ]

    # Prepare VTK structures
    lines = vtk.vtkCellArray()
    cell_data_arrays = {}
    float_fields = ['length', 'distance', 'curveness', 'volume',
                    'avgCrossSection', 'minRadiusAvg', 'minRadiusStd',
                    'avgRadiusAvg', 'avgRadiusStd', 'maxRadiusAvg', 'maxRadiusStd',
                    'roundnessAvg', 'roundnessStd']

    for f in float_fields:
        arr = vtk.vtkFloatArray()
        arr.SetName(f)
        cell_data_arrays[f] = arr

    border_arr = vtk.vtkIntArray()
    border_arr.SetName('hasNodeAtSampleBorder')

    # Insert lines and cell data safely
    for _, e in edges.iterrows():
        n1 = id_to_vtk[int(e.node1id)]
        n2 = id_to_vtk[int(e.node2id)]

        # Create line (2-point polyline)
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, n1)
        line.GetPointIds().SetId(1, n2)
        lines.InsertNextCell(line)

        # Add cell data AFTER line insertion
        for f in float_fields:
            cell_data_arrays[f].InsertNextValue(float(e[f]))
        border_arr.InsertNextValue(int(e['hasNodeAtSampleBorder']))

    # Create PolyData
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetLines(lines)

    # Attach cell data
    cell_data = poly.GetCellData()
    for arr in cell_data_arrays.values():
        cell_data.AddArray(arr)
    cell_data.AddArray(border_arr)

    # Write VTP
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(poly)
    writer.Write()

    print(f"Saved {output_path} successfully. Nodes: {len(nodes)}, Edges: {len(edges)}")

def export_vtp(nodes_csv, edges_csv, output_path):
    # --- Load CSVs ---
    nodes = pd.read_csv(nodes_csv, sep=';')
    edges = pd.read_csv(edges_csv, sep=';')

    # --- Create VTK structures ---
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()

    # Mapping: node_id → vtk index
    id_to_vtk = {}

    # --- Add Points ---
    for idx, row in nodes.iterrows():
        pid = points.InsertNextPoint(row.pos_x, row.pos_y, row.pos_z)
        id_to_vtk[int(row.id)] = pid

    # --- Build CellData arrays for edges ---
    def vtk_float_array(name):
        arr = vtk.vtkFloatArray()
        arr.SetName(name)
        return arr

    float_fields = {
        'length': vtk_float_array('length'),
        'distance': vtk_float_array('distance'),
        'curveness': vtk_float_array('curveness'),
        'volume': vtk_float_array('volume'),
        'avgCrossSection': vtk_float_array('avgCrossSection'),
        'minRadiusAvg': vtk_float_array('minRadiusAvg'),
        'minRadiusStd': vtk_float_array('minRadiusStd'),
        'avgRadiusAvg': vtk_float_array('avgRadiusAvg'),
        'avgRadiusStd': vtk_float_array('avgRadiusStd'),
        'maxRadiusAvg': vtk_float_array('maxRadiusAvg'),
        'maxRadiusStd': vtk_float_array('maxRadiusStd'),
        'roundnessAvg': vtk_float_array('roundnessAvg'),
        'roundnessStd': vtk_float_array('roundnessStd'),
    }

    # An integer field for border flag
    border_arr = vtk.vtkIntArray()
    border_arr.SetName("hasNodeAtSampleBorder")

    # --- Create Lines (Edges) ---
    for _, e in edges.iterrows():
        if e.node1id not in id_to_vtk or e.node2id not in id_to_vtk:
            continue

        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, id_to_vtk[int(e.node1id)])
        line.GetPointIds().SetId(1, id_to_vtk[int(e.node2id)])
        lines.InsertNextCell(line)

        # Add CellData
        for field, arr in float_fields.items():
            arr.InsertNextValue(float(e[field]))

        border_arr.InsertNextValue(int(e["hasNodeAtSampleBorder"]))

    # --- Create PolyData ---
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetLines(lines)

    # Attach cell data arrays
    cell_data = poly.GetCellData()
    for arr in float_fields.values():
        cell_data.AddArray(arr)
    cell_data.AddArray(border_arr)

    # --- Save to .vtp ---
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(poly)
    writer.Write()

    print(f"Saved: {output_path}")


export_vtp_safe(
    nodes_csv=nodes_csv,
    edges_csv=edges_csv,
    output_path=output_path
)
