from vtk.numpy_interface import dataset_adapter as dsa
import numpy as np

# 1. Wrap the raw VTP inputs
in0 = dsa.WrapDataObject(inputs[0].VTKObject)
in1 = dsa.WrapDataObject(inputs[1].VTKObject)
output_data = dsa.WrapDataObject(output.VTKObject)

# 2. Use the geometry of the first file as the output template
# ShallowCopy handles the geometry AND the original strings automatically
output.ShallowCopy(inputs[0].VTKObject)

try:
    # 3. Access 'probability' 
    # Use np.asarray to be extra safe with VTK data types
    prob0 = np.asarray(in0.PointData['probability'])
    prob1 = np.asarray(in1.PointData['probability'])
    
    # 4. Access 'local_entropy_bits' 
    ent0 = np.asarray(in0.PointData['local_entropy_bits'])
    ent1 = np.asarray(in1.PointData['local_entropy_bits'])

    # 5. Calculate the absolute differences
    prob_diff = np.abs(prob0 - prob1)
    ent_diff = np.abs(ent0 - ent1)

    # 6. Add numerical results to the output
    # By using the shallow copy above, 'region_name' is already there!
    output_data.PointData.append(prob_diff, 'Prob_Difference')
    output_data.PointData.append(ent_diff, 'Entropy_Difference')
    
    print("SUCCESS: Math complete. Differences added to Point Data.")

except Exception as e:
    print(f"Error: {e}")
