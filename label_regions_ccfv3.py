import csv
import json
import numpy as np
import nrrd

import requests

# Get brain structure
'''
url = "http://api.brain-map.org/api/v2/structure_graph_download/1.json"
output_file = "structure_tree.json"

print("Downloading structure tree...")
r = requests.get(url)

if r.status_code == 200:
    with open(output_file, "wb") as f:
        f.write(r.content)
    print(f"Saved to {output_file}")
else:
    print(f"Error {r.status_code}: Could not download.")
'''

# -------------------------------------------------------
# 1. Load Allen CCFv3 annotation volume & structure tree
# -------------------------------------------------------
ANNOTATION_FILE = "annotation_10.nrrd"       # or annotation_10.nrrd
STRUCTURE_TREE = "structure_tree.json"

print("Loading annotation volume...")
ann, ann_header = nrrd.read(ANNOTATION_FILE)
print(f"Annotation shape: {ann.shape} (Z, Y, X)")

print("Loading structure tree...")
with open(STRUCTURE_TREE, "r") as f:
    tree = json.load(f)

# structure tree is inside tree['msg']
structures = tree["msg"]

# Build dict: structure_id → (name, acronym)
structure_map = {}
for s in structures:
    id = s['id']
    structure_map[s['id']] = {
        'name': s['name'],
        'acronym': s['acronym'],
        'parent': s['parent_structure_id'] if 'parent_structure_id' in s else None
    }

# annotation_10.nrrd generated no mapping of regions as images are of higher resolution.
SCALE_X = 3.0   # divide pos_x by 3
SCALE_Y = 4.3   # divide pos_y by 4.3
SCALE_Z = 3.0   # divide pos_z by 3
OFFSET_X = 0.0  # subtract from pos_x before scaling
OFFSET_Y = 0.0
OFFSET_Z = 0.0
# -------------------------------------------------------
# 2. Map one coordinate to region
# -------------------------------------------------------
def coord_to_region(x, y, z):
    """
    x,y,z must be in CCFv3 voxel space (same resolution as annotation).
    """
    # Scale raw coordinates to voxel indices
    xi = int(round((x - OFFSET_X)/ SCALE_X))
    yi = int(round((y - OFFSET_Y)/ SCALE_Y))
    zi = int(round((z - OFFSET_Z)/ SCALE_Z))


    # CCF is [z, y, x] format
    # Check bounds
    inside_volume = (0 <= xi < ann.shape[2] and
                     0 <= yi < ann.shape[1] and
                     0 <= zi < ann.shape[0])

    if inside_volume:
        region_id = int(ann[zi, yi, xi])
    else:
        region_id = None

    if region_id in structure_map:
        s = structure_map[region_id]
        return (region_id, s['acronym'], s['name'], inside_volume, xi, yi, zi)
    else:
        return (region_id, None, None, inside_volume, xi, yi, zi)


# -------------------------------------------------------
# 3. Stream large node.csv and write output
# -------------------------------------------------------
INPUT_FILE = "/Users/vaw1/Downloads/OGB/BALBc_no1_raw/BALBc-no1_iso3um_stitched_segmentation_bulge_size_3.0_nodes.csv" 
OUTPUT_FILE = "node_regions.csv"

DEBUG_COUNT = 20

print("Processing nodes...")
with open(INPUT_FILE, "r") as csv_in, open(OUTPUT_FILE, "w", newline="") as csv_out:
    reader = csv.DictReader(csv_in, delimiter=';')

    # Preserve original fieldnames and append new region columns
    fieldnames = reader.fieldnames + ["region_id", "region_acronym", "region_name"]
    writer = csv.DictWriter(csv_out, fieldnames=fieldnames, delimiter=';')
    
    writer.writeheader()

    for i, row in enumerate(reader):
        # Convert coordinates to float and scale
        x = float(row["pos_x"])
        y = float(row["pos_y"])
        z = float(row["pos_z"])

        region_id, acronym, name, inside, xi, yi, zi = coord_to_region(x, y, z)

        # Add new region info to row
        row["region_id"] = region_id
        row["region_acronym"] = acronym
        row["region_name"] = name

        writer.writerow(row)
        
        # Debug print for first N rows
        if i < DEBUG_COUNT:
            print(f"Row {i}: pos=({x},{y},{z}) -> voxel=({xi},{yi},{zi}) "
                  f"inside_volume={inside} region_id={region_id} acronym={acronym} name={name}")



print(f"Done. Output saved to {OUTPUT_FILE}")
