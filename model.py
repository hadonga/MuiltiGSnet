"""
PointPillars  from SECOND.
Code written by Alex Lang and Oscar Beijbom, 2018.
Licensed under MIT License [see LICENSE].
"""
'''
2021.1.5 
模型整体结构
把分散的网络结构集合起来，因为之前各种使用的不同网络并没有特别的区别。
修改这部分作为论文的contribution
2021.1.18
增加MLP
2021.1.25 
增加unet++ 测试效果
2021.02.15 增加分离网络
'''

import numpy as np
import sys
import torch
from torch import nn
from torch.nn import functional as F

from torchplus.nn import Empty
from torchplus.tools import change_default_args

from network import TransDSUnet_lite #,AttU_Net, U_Net, Mlp, NestedUNet, New_net, DS_Unet,

# ---------------------------------------------------------------------------- #
#  pointpillar部分加入 model进行更好的管理，并预备按照 mit论文进行修改。 # 为了更好理解而进行少量修改
# ---------------------------------------------------------------------------- #
def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.

    Args:
        actual_num ([type]): [description]
        max_num ([type]): [description]

    Returns:
        [type]: [description]
    """

    actual_num = torch.unsqueeze(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(
        max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator

class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True):
        """
        Pillar Feature Net Layer.
        The Pillar Feature Net could be composed of a series of these layers, but the PointPillars paper results only
        used a single PFNLayer. This layer performs a similar role as second.pytorch.voxelnet.VFELayer.
        :param in_channels: <int>. Number of input channels.
        :param out_channels: <int>. Number of output channels.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param last_layer: <bool>. If last_layer, there is no concatenation of features.
        """
        super().__init__()
        self.name = 'PFNLayer'
        if use_norm:
            BatchNorm1d = change_default_args(eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)
        self.linear = Linear(in_channels, out_channels)
        self.norm = BatchNorm1d(out_channels)

    def forward(self, inputs):
        x = self.linear(inputs)
        x = self.norm(x)
        # x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()  #不知道这个反向变换有什么意义？
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        return x_max

class PillarFeatureNet(nn.Module):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=(64,),
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1)):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """
        super().__init__()
        self.name = 'PillarFeatureNet'
        assert len(num_filters) > 0
        num_input_features += 5
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance

        # Create PillarFeatureNet layers
        num_filters = [num_input_features] + list(num_filters) # [9,64]
        in_filters = num_filters[0]
        out_filters = num_filters[1]
        self.pfn_layers = PFNLayer(in_filters, out_filters, use_norm)
        # self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]

    def forward(self, features, num_points, coors): #features [num_voxels, max_points, ndim]
        # pc->  [voxels, num_points, coors]
        # v是 non-empty pillar number x 35 点 x 4 特征，c是voxel的坐标 voxel_num x3，D 是 pillar内的points数 - voxel_numx1

        #voxels from two batches are concatenated and coord have information corrd [num_voxels, (batch, x,y)]
        # pdb.set_trace()

        # Find distance of x, y, and z from cluster center
        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_points.type_as(features).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean

        # Find distance of x, y, and z from pillar center
        f_center = torch.zeros_like(features[:, :, :2])
        f_center[:, :, 0] = features[:, :, 0] - (coors[:, 3].float().unsqueeze(1) * self.vx + self.x_offset)
        f_center[:, :, 1] = features[:, :, 1] - (coors[:, 2].float().unsqueeze(1) * self.vy + self.y_offset)

        # Combine together feature decorations
        features_ls = [features, f_cluster, f_center]
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)
        features = torch.cat(features_ls, dim=-1)

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        # Forward pass through PFNLayers
        features = self.pfn_layers(features) #here they are considering num of voxels as batch size for linear layer

        return features.squeeze()

class PointPillarsScatter(nn.Module):
    def __init__(self,
                 output_shape,
                 num_input_features=64):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """

        super().__init__()
        self.name = 'PointPillarsScatter'
        self.output_shape = output_shape
        self.ny = output_shape[2]
        self.nx = output_shape[3]
        self.nchannels = num_input_features

    def forward(self, voxel_features, coords, batch_size):

        # batch_canvas will be the final output.
        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(self.nchannels, self.nx * self.ny, dtype=voxel_features.dtype,
                                 device=voxel_features.device)

            # Only include non-empty pillars
            batch_mask = coords[:, 0] == batch_itt
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            voxels = voxels.t()

            # Now scatter the blob back to the canvas.
            canvas[:, indices] = voxels

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(batch_size, self.nchannels, self.ny, self.nx)

        return batch_canvas

# ---------------------------------------------------------------------------- #
# trans_DSUNnet
# ---------------------------------------------------------------------------- #
class Our_trans_DSUNet(nn.Module):
    def __init__(self, cfg):
        super(Our_trans_DSUNet, self).__init__()
        self.cfg = cfg
        self.voxel_feature_extractor = PillarFeatureNet(num_input_features = cfg.input_features,
        # self.voxel_feature_extractor = PillarFeatureNetRadius(num_input_features = cfg.input_features,
                use_norm = cfg.use_norm,
                num_filters=cfg.vfe_filters,
                with_distance=cfg.with_distance,
                voxel_size=cfg.voxel_size,
                pc_range=cfg.pc_range)
            #voxel_feature_extractor:20000X64
        grid_size = (np.asarray(cfg.pc_range[3:]) - np.asarray(cfg.pc_range[:3])) / np.asarray(cfg.voxel_size)
        grid_size = np.round(grid_size).astype(np.int64) #grid_size:128X128X1(100X100X1)
        dense_shape = [1] + grid_size[::-1].tolist() + [cfg.vfe_filters[-1]] #grid_size[::-1] reverses the index from xyz to zyx
        #dense_shape:[1, 1, 128, 128, 64]
        # 得到pseudo image: middle_feature_extractor:1x64x128x128
        self.middle_feature_extractor = PointPillarsScatter(output_shape = dense_shape, num_input_features = cfg.vfe_filters[-1])
        # self.feature_mlp = Mlp()  # bs*128*128*64 -> bs*128*128*3 and bs*128*128*3
        self.encoder_decoder = TransDSUnet_lite(64,3) # bs*128*128*64 -> bs*128*128*3
        # self.encoder_decoder = segnetGndEst(in_channels=64, is_unpooling=True)
        #pred- 16x4x128x128

    def forward(self, voxels, num_points, coors):
        voxel_features = self.voxel_feature_extractor(voxels, num_points, coors)
        spatial_features = self.middle_feature_extractor(voxel_features, coors, self.cfg.batch_size)
        # pred_cls,pred_nor= self.feature_mlp(spatial_features)
        pred = self.encoder_decoder(spatial_features)
        return torch.squeeze(pred)  # gnd_pred : batchsize x 3 x 128 x 128

# ---------------------------------------------------------------------------- #
# DS_Unet : 0.27M
# ---------------------------------------------------------------------------- #
class Our_DS_UNet(nn.Module):
    def __init__(self, cfg):
        super(Our_DS_UNet, self).__init__()
        self.cfg = cfg
        self.voxel_feature_extractor = PillarFeatureNet(num_input_features = cfg.input_features,
        # self.voxel_feature_extractor = PillarFeatureNetRadius(num_input_features = cfg.input_features,
                use_norm = cfg.use_norm,
                num_filters=cfg.vfe_filters,
                with_distance=cfg.with_distance,
                voxel_size=cfg.voxel_size,
                pc_range=cfg.pc_range)
            #voxel_feature_extractor:20000X64
        grid_size = (np.asarray(cfg.pc_range[3:]) - np.asarray(cfg.pc_range[:3])) / np.asarray(cfg.voxel_size)
        grid_size = np.round(grid_size).astype(np.int64) #grid_size:128X128X1(100X100X1)
        dense_shape = [1] + grid_size[::-1].tolist() + [cfg.vfe_filters[-1]] #grid_size[::-1] reverses the index from xyz to zyx
        #dense_shape:[1, 1, 128, 128, 64]
        # 得到pseudo image: middle_feature_extractor:1x64x128x128
        self.middle_feature_extractor = PointPillarsScatter(output_shape = dense_shape, num_input_features = cfg.vfe_filters[-1])

        # self.feature_mlp = Mlp()  # bs*128*128*64 -> bs*128*128*3 and bs*128*128*3
        self.encoder_decoder = DS_Unet(64,3) # bs*128*128*64 -> bs*128*128*3
        # self.encoder_decoder = segnetGndEst(in_channels=64, is_unpooling=True)
        #pred- 16x4x128x128

    def forward(self, voxels, num_points, coors):
        voxel_features = self.voxel_feature_extractor(voxels, num_points, coors)
        spatial_features = self.middle_feature_extractor(voxel_features, coors, self.cfg.batch_size)

        # pred_cls,pred_nor= self.feature_mlp(spatial_features)
        pred = self.encoder_decoder(spatial_features)

        return torch.squeeze(pred)  # gnd_pred : batchsize x 3 x 128 x 128

# ---------------------------------------------------------------------------- #
# New_net : 34.73M
# ---------------------------------------------------------------------------- #
class Our_New_Net(nn.Module):
    def __init__(self, cfg):
        super(Our_New_Net, self).__init__()
        self.cfg = cfg
        self.voxel_feature_extractor = PillarFeatureNet(num_input_features = cfg.input_features,
        # self.voxel_feature_extractor = PillarFeatureNetRadius(num_input_features = cfg.input_features,
                use_norm = cfg.use_norm,
                num_filters=cfg.vfe_filters,
                with_distance=cfg.with_distance,
                voxel_size=cfg.voxel_size,
                pc_range=cfg.pc_range)
            #voxel_feature_extractor:20000X64
        grid_size = (np.asarray(cfg.pc_range[3:]) - np.asarray(cfg.pc_range[:3])) / np.asarray(cfg.voxel_size)
        grid_size = np.round(grid_size).astype(np.int64) #grid_size:128X128X1(100X100X1)
        dense_shape = [1] + grid_size[::-1].tolist() + [cfg.vfe_filters[-1]] #grid_size[::-1] reverses the index from xyz to zyx
        #dense_shape:[1, 1, 128, 128, 64]
        # 得到pseudo image: middle_feature_extractor:1x64x128x128
        self.middle_feature_extractor = PointPillarsScatter(output_shape = dense_shape, num_input_features = cfg.vfe_filters[-1])

        # self.feature_mlp = Mlp()  # bs*128*128*64 -> bs*128*128*3 and bs*128*128*3
        self.encoder_decoder = New_net() # bs*128*128*64 -> bs*128*128*3
        # self.encoder_decoder = segnetGndEst(in_channels=64, is_unpooling=True)
        #pred- 16x4x128x128

    def forward(self, voxels, num_points, coors):
        voxel_features = self.voxel_feature_extractor(voxels, num_points, coors)
        spatial_features = self.middle_feature_extractor(voxel_features, coors, self.cfg.batch_size)

        # pred_cls,pred_nor= self.feature_mlp(spatial_features)
        pred = self.encoder_decoder(spatial_features)

        return torch.squeeze(pred)  # gnd_pred : 3 x128x128

# ---------------------------------------------------------------------------- #
# mlp : 0.4M
# ---------------------------------------------------------------------------- #
class Our_MLP(nn.Module):
    def __init__(self, cfg):
        super(Our_MLP, self).__init__()
        self.cfg = cfg
        self.voxel_feature_extractor = PillarFeatureNet(num_input_features = cfg.input_features,
        # self.voxel_feature_extractor = PillarFeatureNetRadius(num_input_features = cfg.input_features,
                use_norm = cfg.use_norm,
                num_filters=cfg.vfe_filters,
                with_distance=cfg.with_distance,
                voxel_size=cfg.voxel_size,
                pc_range=cfg.pc_range)
            #voxel_feature_extractor:20000X64
        grid_size = (np.asarray(cfg.pc_range[3:]) - np.asarray(cfg.pc_range[:3])) / np.asarray(cfg.voxel_size)
        grid_size = np.round(grid_size).astype(np.int64) #grid_size:128X128X1(100X100X1)
        dense_shape = [1] + grid_size[::-1].tolist() + [cfg.vfe_filters[-1]] #grid_size[::-1] reverses the index from xyz to zyx
        #dense_shape:[1, 1, 128, 128, 64]
        # 得到pseudo image: middle_feature_extractor:1x64x128x128
        self.middle_feature_extractor = PointPillarsScatter(output_shape = dense_shape, num_input_features = cfg.vfe_filters[-1])

        self.feature_mlp = Mlp()  # bs*128*128*64 -> bs*128*128*3 and bs*128*128*3
        # self.encoder_decoder = U_Net() # bs*128*128*64 -> bs*128*128*3
        # self.encoder_decoder = segnetGndEst(in_channels=64, is_unpooling=True)
        #pred- 16x4x128x128

    def forward(self, voxels, num_points, coors):
        voxel_features = self.voxel_feature_extractor(voxels, num_points, coors)
        spatial_features = self.middle_feature_extractor(voxel_features, coors, self.cfg.batch_size)

        pred_cls,pred_nor= self.feature_mlp(spatial_features)
        # pred = self.encoder_decoder(spatial_features)

        return torch.squeeze(pred_cls),torch.squeeze(pred_nor)
        # gnd_pred : 3 x128x128

# ---------------------------------------------------------------------------- #
# Unet : 34.56M
# ---------------------------------------------------------------------------- #
class Our_UNet(nn.Module):
    def __init__(self, cfg):
        super(Our_UNet, self).__init__()
        self.cfg = cfg
        self.voxel_feature_extractor = PillarFeatureNet(num_input_features = cfg.input_features,
        # self.voxel_feature_extractor = PillarFeatureNetRadius(num_input_features = cfg.input_features,
                use_norm = cfg.use_norm,
                num_filters=cfg.vfe_filters,
                with_distance=cfg.with_distance,
                voxel_size=cfg.voxel_size,
                pc_range=cfg.pc_range)
            #voxel_feature_extractor:20000X64
        grid_size = (np.asarray(cfg.pc_range[3:]) - np.asarray(cfg.pc_range[:3])) / np.asarray(cfg.voxel_size)
        grid_size = np.round(grid_size).astype(np.int64) #grid_size:128X128X1(100X100X1)
        dense_shape = [1] + grid_size[::-1].tolist() + [cfg.vfe_filters[-1]] #grid_size[::-1] reverses the index from xyz to zyx
        #dense_shape:[1, 1, 128, 128, 64]
        # 得到pseudo image: middle_feature_extractor:1x64x128x128
        self.middle_feature_extractor = PointPillarsScatter(output_shape = dense_shape, num_input_features = cfg.vfe_filters[-1])

        # self.feature_mlp = Mlp()  # bs*128*128*64 -> bs*128*128*3 and bs*128*128*3
        self.encoder_decoder = U_Net() # bs*128*128*64 -> bs*128*128*3
        # self.encoder_decoder = segnetGndEst(in_channels=64, is_unpooling=True)
        #pred- 16x4x128x128

    def forward(self, voxels, num_points, coors):
        voxel_features = self.voxel_feature_extractor(voxels, num_points, coors)
        spatial_features = self.middle_feature_extractor(voxel_features, coors, self.cfg.batch_size)

        # pred_cls,pred_nor= self.feature_mlp(spatial_features)
        pred = self.encoder_decoder(spatial_features)

        return torch.squeeze(pred)  # gnd_pred : 3 x128x128

# ---------------------------------------------------------------------------- #
# Our_AUNet : 34.91M
# ---------------------------------------------------------------------------- #
class Our_AUNet(nn.Module):
    def __init__(self, cfg):
        super(Our_AUNet, self).__init__()
        self.cfg = cfg
        self.voxel_feature_extractor = PillarFeatureNet(num_input_features=cfg.input_features,
                                                        # self.voxel_feature_extractor = PillarFeatureNetRadius(num_input_features = cfg.input_features,
                                                        use_norm=cfg.use_norm,
                                                        num_filters=cfg.vfe_filters,
                                                        with_distance=cfg.with_distance,
                                                        voxel_size=cfg.voxel_size,
                                                        pc_range=cfg.pc_range)
        # voxel_feature_extractor:20000X64
        grid_size = (np.asarray(cfg.pc_range[3:]) - np.asarray(cfg.pc_range[:3])) / np.asarray(cfg.voxel_size) 
        grid_size = np.round(grid_size).astype(np.int64)  # grid_size:128X128X1(100X100X1)  
        dense_shape = [1] + grid_size[::-1].tolist() + [cfg.vfe_filters[-1]]  # grid_size[::-1] reverses the index from xyz to zyx
        # dense_shape:[1, 1, 128, 128, 64]
        # 得到pseudo image : middle_feature_extractor:1x64x128x128
        self.middle_feature_extractor = PointPillarsScatter(output_shape=dense_shape,
                                                            num_input_features=cfg.vfe_filters[-1])

        self.encoder_decoder = AttU_Net() # output: 3x 128x128
        # pred- 16x4x128x128
    def forward(self, voxels, num_points, coors):
        voxel_features = self.voxel_feature_extractor(voxels, num_points, coors)
        spatial_features = self.middle_feature_extractor(voxel_features, coors, self.cfg.batch_size)
        pred = self.encoder_decoder(spatial_features)
        return torch.squeeze(pred)
        # 3x128x128

# ---------------------------------------------------------------------------- #
# Our_UNet_2plus : 36.67M
# ---------------------------------------------------------------------------- #
class Our_UNet_2plus(nn.Module):
    def __init__(self, cfg):
        super(Our_UNet_2plus, self).__init__()
        self.cfg = cfg
        self.voxel_feature_extractor = PillarFeatureNet(num_input_features = cfg.input_features,
        # self.voxel_feature_extractor = PillarFeatureNetRadius(num_input_features = cfg.input_features,
                use_norm = cfg.use_norm,
                num_filters=cfg.vfe_filters,
                with_distance=cfg.with_distance,
                voxel_size=cfg.voxel_size,
                pc_range=cfg.pc_range)
            #voxel_feature_extractor:20000X64
        grid_size = (np.asarray(cfg.pc_range[3:]) - np.asarray(cfg.pc_range[:3])) / np.asarray(cfg.voxel_size)
        grid_size = np.round(grid_size).astype(np.int64) #grid_size:128X128X1(100X100X1)
        dense_shape = [1] + grid_size[::-1].tolist() + [cfg.vfe_filters[-1]] #grid_size[::-1] reverses the index from xyz to zyx
        #dense_shape:[1, 1, 128, 128, 64]
        # 得到pseudo image: middle_feature_extractor:1x64x128x128
        self.middle_feature_extractor = PointPillarsScatter(output_shape = dense_shape, num_input_features = cfg.vfe_filters[-1])
        # self.feature_mlp = Mlp()  # bs*128*128*64 -> bs*128*128*3 and bs*128*128*3
        self.encoder_decoder = NestedUNet() # bs*128*128*64 -> bs*128*128*3
        # self.encoder_decoder = segnetGndEst(in_channels=64, is_unpooling=True)
        #pred- 16x4x128x128

    def forward(self, voxels, num_points, coors):
        voxel_features = self.voxel_feature_extractor(voxels, num_points, coors)
        spatial_features = self.middle_feature_extractor(voxel_features, coors, self.cfg.batch_size)
        # pred_cls,pred_nor= self.feature_mlp(spatial_features)
        pred = self.encoder_decoder(spatial_features)
        return torch.squeeze(pred)  # gnd_pred : 3 x128x128