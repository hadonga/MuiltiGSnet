import numpy as np
import open3d as o3d

data_dir='/media/lion/ssd/dataset/semantic_kitti/dataset/sequences/00/n_pc/000000.npy'
data_s_dir='/media/lion/ssd/dataset/semantic_kitti/dataset/sequences/00/sn_pc/000000.npy'
d_dir='/media/lion/ssd1/pillars_data/00/reduce_velo/000000.npy'
data=np.load(data_s_dir)
print(data.shape)
pcd=o3d.geometry.PointCloud()
pcd.points=o3d.utility.Vector3dVector(data[:,:3])
o3d.visualization.draw_geometries([pcd])