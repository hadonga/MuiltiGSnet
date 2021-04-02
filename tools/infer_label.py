import open3d as o3d
import numpy as np

data_dir="/media/lion/ssd1/pillars_data/00/clip_sn_pc/000000.npy"
acc_l_dir="/media/lion/ssd1/pillars_data/00/plpt_cl_s_lb/000000.npy"
cls_l_dir="/media/lion/ssd1/pillars_data/00/pl_cl_s_lb/000000.npy"
data=np.load(data_dir)
acc_l=np.load(acc_l_dir)
cls_l=np.load(cls_l_dir)

pcd=o3d.geometry.PointCloud()
points=data[:,:3]
colors=np.zeros([points.shape[0],3])
for i in range(points.shape[0]):
    grid_index = (points[i,:2] + 51.2) / 0.8
    if cls_l[int(grid_index[1]),int(grid_index[0])]==1:
        colors[i,1]=1
    elif cls_l[int(grid_index[1]),int(grid_index[0])]==0:
        colors[i, 2] = 1
    else :
        colors[i, 0] = 1
    '''
    if acc_l[i] == 40 or acc_l[i] == 44 or acc_l[i] == 48 or acc_l[i] == 49 or acc_l[i] == 60:
        colors[i,1]=1
    else:
        colors[i, 0] = 1
    '''
pcd.points=o3d.utility.Vector3dVector(points)
pcd.colors=o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd])