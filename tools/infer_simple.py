import numpy as np
import open3d as o3d

data_dir='/root/dataset/uneven2/sequences/00/pc_s_n/ue_003336.npy'
data_s_dir='/root/dataset/uneven2/sequences/00/pc_s_n/ue_003336.npy'

data_dir="/root/dataset/uneven2/sequences/00/pc_s_n/ue_003336.npy"
acc_l_dir="/root/dataset/uneven2/sequences/00/lb_s_pt/ue_003336.npy"
cls_l_dir="/root/dataset/uneven2/sequences/00/lb_s_n_pl/ue_003336.npy"
data=np.load(data_s_dir)
print(data.shape)
pcd=o3d.geometry.PointCloud()
pcd.points=o3d.utility.Vector3dVector(data[:,:3])
o3d.visualization.draw_geometries([pcd])