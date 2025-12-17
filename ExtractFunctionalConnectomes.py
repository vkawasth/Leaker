import vtk
from vtk.numpy_interface import dataset_adapter as dsa
import numpy as np
import re

# 1. Setup Input/Output
input_obj = self.GetInputDataObject(0, 0)
output_obj = self.GetOutputDataObject(0)
wrapped_input = dsa.WrapDataObject(input_obj)

target_regions = {'HY', 'MB', 'PVR', 'PVZ', 'LZ', 'EP'}

# 2. Access Difference Array
num_pts = input_obj.GetNumberOfPoints()

if 'Prob_Difference' in wrapped_input.PointData.keys():
    prob_diff_array = np.array(wrapped_input.PointData['Prob_Difference'])
    print(f"Data Found. Max Shift: {np.max(np.abs(prob_diff_array)):.12f}")
else:
    print("CRITICAL: Prob_Difference not found.")
    prob_diff_array = np.zeros(num_pts)

# 3. Extraction & String Cleaning
pd = input_obj.GetPointData()
region_names_vtk = pd.GetAbstractArray('region_name')
old_to_new = {}
new_pts = vtk.vtkPoints()
new_pd = output_obj.GetPointData() # Fixed the AttributeError here
new_pd.CopyAllocate(pd)
regional_diffs = {region: [] for region in target_regions}

for i in range(num_pts):
    raw_val = str(region_names_vtk.GetValue(i))
    # Standard cleaning
    clean_name = raw_val.replace("[", "").replace("]", "").replace("'", "").strip().upper()
    
    matched_region = None
    for target in target_regions:
        if target in clean_name:
            matched_region = target
            break
            
    if matched_region:
        new_id = new_pts.InsertNextPoint(input_obj.GetPoint(i))
        new_pd.CopyData(pd, i, new_id)
        old_to_new[i] = new_id
        regional_diffs[matched_region].append(float(prob_diff_array[i]))

# 4. Extract Cells (Lines)
new_cells = vtk.vtkCellArray()
for i in range(input_obj.GetNumberOfCells()):
    pts = input_obj.GetCell(i).GetPointIds()
    if pts.GetNumberOfIds() == 2:
        id0, id1 = pts.GetId(0), pts.GetId(1)
        if id0 in old_to_new and id1 in old_to_new:
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, old_to_new[id0])
            line.GetPointIds().SetId(1, old_to_new[id1])
            new_cells.InsertNextCell(line)

# 5. Finalize Output
output_obj.SetPoints(new_pts)
output_obj.SetLines(new_cells)

# 6. High-Precision Results Summary
print("--- [REGIONAL AROUSAL SUMMARY] ---")
for reg in sorted(target_regions):
    data = regional_diffs[reg]
    count = len(data)
    if count > 0:
        avg = np.mean(data)
        print(f"Region: {reg:4} | Nodes: {count:5} | Avg Shift: {avg:.12f}")
    else:
        print(f"Region: {reg:4} | Nodes:     0 | (Missing from Blowdown)")
