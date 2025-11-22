import pandas as pd


NODE = "/Users/vaw1/Downloads/OGB/BALBc_no1_raw/BALBc-no1_iso3um_stitched_segmentation_bulge_size_3.0_nodes.csv"
EDGE = "/Users/vaw1/Downloads/OGB/BALBc_no1_raw/BALBc-no1_iso3um_stitched_segmentation_bulge_size_3.0_edges.csv"

# only load 1000 datasets at a time, on my macbookpro, fulldataset plotting ran all night.
start=0
chunk=10000
nodes_a = pd.read_csv(NODE, sep=";")
nodes = pd.read_csv(NODE, sep=";", skiprows=range(1, start+1), nrows=chunk)
# Load all edges as lookups are going to be performed.
edges_a = pd.read_csv(EDGE, sep=";")
edges = pd.read_csv(EDGE, sep=";", skiprows=range(1, start+1), nrows=2*chunk)

# Remove border touching vessels
edges_chunk = edges[ edges["hasNodeAtSampleBorder"] == 0 ]
# Keep only large vessels
edges_chunk = edges[ edges["avgRadiusAvg"] > 4 ]
# Keep curvy vessels
edges_chunk = edges[ edges["curveness"] > 1.2 ]

# Only draw nodes which are connected
connected_nodes = set(edges['node1id']).union(set(edges['node2id']))
nodes_connected = nodes[nodes['id'].isin(connected_nodes)]

print(nodes.head())
print(edges.head())


import plotly.graph_objects as go

# Map ALL node ID → coordinates 
all_node_pos = {
    int(row.id): (row.pos_x, row.pos_y, row.pos_z)
        for _, row in nodes_a.iterrows()
}

fig = go.Figure()

# Find nodes that do exists in given edges
# no need to be optimistic as there are edges that go beyond our collection.
# This will generate local cluttered graph which we do not want
# We want to overcome local learning coming from GNNs... as 
# deep structures of deepstate has long edges.
# building connectomes...
# give all nodes

def build_edge_coordinates(nodes, edges):
    # Convert node positions into a dictionary for fast lookup
    pos = {int(row.id): (row.pos_x, row.pos_y, row.pos_z) 
           for _, row in nodes.iterrows()}

    edge_x, edge_y, edge_z = [], [], []

    for _, e in edges.iterrows():
        n1 = pos.get(int(e.node1id))
        n2 = pos.get(int(e.node2id))
        if not n1 or not n2:
            #continue
            if not n1:
                n1 = all_node_pos.get(int(e.node1id))
            if not n2:
                n2 = all_node_pos.get(int(e.node2id))

        edge_x += [n1[0], n2[0], None]
        edge_y += [n1[1], n2[1], None]
        edge_z += [n1[2], n2[2], None]

    return edge_x, edge_y, edge_z

# Draw edges which are selected in chunk
# we lookup nodes from all nodes.
:wq
for _, e in edges.iterrows():
    x1, y1, z1 = all_node_pos[e.node1id]
    x2, y2, z2 = all_node_pos[e.node2id]

    fig.add_trace(go.Scatter3d(
        x=[x1, x2], y=[y1, y2], z=[z1, z2],
        mode="lines",
        line=dict(
            width=3,
            color=e.curveness,       # color edges by curvature
            colorscale="Viridis"
        ),
        hoverinfo="text",
        text=f"edge {e.id}<br>curveness={e.curveness:.3f}<br>len={e.length:.2f}"
    ))

# Draw nodes
fig.add_trace(go.Scatter3d(
    x=nodes.pos_x,
    y=nodes.pos_y,
    z=nodes.pos_z,
    mode="markers",
    marker=dict(size=3, color="red"),
    text=[f"node {n}" for n in nodes.id],
    hoverinfo="text"
))

fig.update_layout(
    width=900,
    height=900,
    scene=dict(
        aspectmode="data",
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z"
    ),
    title="3D Vessel Graph (colored by curveness)"
)

fig.show()
