import numpy as np
import open3d as o3d

data_dir = "/root/dataset/uneven2/sequences/00/pc_s_n/ue_003336.npy"
acc_l_dir = "/root/dataset/uneven2/sequences/00/lb_s_pt/ue_003336.npy"
cls_l_dir = "/root/dataset/uneven2/sequences/00/lb_s_n_pl/ue_003336.npy"
data = np.load(data_dir)
acc_l = np.load(acc_l_dir)
cls_l = np.load(cls_l_dir)

pcd = o3d.geometry.PointCloud()
points = data[:, :3]
colors = np.zeros([points.shape[0], 3])
for i in range(points.shape[0]):
    grid_index = (points[i, :2] + 51.2) / 0.8
    if cls_l[int(grid_index[1]), int(grid_index[0])] == 1:
        colors[i, 1] = 1
    elif cls_l[int(grid_index[1]), int(grid_index[0])] == 0:
        colors[i, 2] = 1
    else:
        colors[i, 0] = 1
    '''
    if acc_l[i] == 40 or acc_l[i] == 44 or acc_l[i] == 48 or acc_l[i] == 49 or acc_l[i] == 60:
        colors[i,1]=1
    else:
        colors[i, 0] = 1
    '''
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd])
