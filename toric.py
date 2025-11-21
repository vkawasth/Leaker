import numpy as np
import matplotlib.pyplot as plt

# Example path (flattened row-major)

path_1 = [(1, 684, 5, 1, 20, 27), (1, 685, 4, 1, 19, 28), (1, 686, 3, 1, 18, 29), (1, 687, 2, 1, 17, 30), (1, 688, 1, 1, 16, 31)]

# Convert to array
path_arr = np.array(path_1, dtype=float)

# Interpolate multiplicatively between consecutive points
curve_points = []
for k in range(len(path_arr)-1):
    start = path_arr[k]
    end = path_arr[k+1]
    ts = np.linspace(0,1,20)  # 20 steps
    for t in ts:
        # toric interpolation (multiplicative)
        point = start * (end/start)**t
        curve_points.append(point)
curve_points = np.array(curve_points)

# Project 6D to 3D for plotting (largest 3 entries)
idx = np.argsort(-path_arr[-1])[:3]  # indices of 3 largest entries
curve_3d = curve_points[:, idx]

# Plot
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(curve_3d[:,0], curve_3d[:,1], curve_3d[:,2], '-o', markersize=3)
ax.set_xlabel('x_i1')
ax.set_ylabel('x_i2')
ax.set_zlabel('x_i3')
ax.set_title('Toric curve along path (3 largest cells)')
plt.show()
