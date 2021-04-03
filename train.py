'''说明：
20201230    修改optimizer, 删除argparse部分，设置全部转移到 yaml文件.

20210105    修改 model部分，把Unet和 Attention Unet 合并
            改进 checkpoint储存部分

20210108    修改 worker数量防止core卡死？

20210118    韩国服务器上gpu使用需要先查看。
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 留出前几个GPU跑其他程序, 需要在导入模型前定义
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import yaml
import time
import shutil

from dataset_loader import kitti_loader
# from gdn_dataset import kitti_gnd
from model import Our_trans_DSUNet
from tools.utils import points_to_voxel

# ---------------------------------------------------------------------------- #
# Load config ; declare Meter class, LearningRateSchedule class, checkpointer, etc.
# ---------------------------------------------------------------------------- #

config_file = './config_kittiSem.yaml'

try:
    with open(config_file) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    print("using config file:", config_file)
    print('\n'.join('%s:%s' % item for item in config_dict.items()))


    class ConfigClass:
        def __init__(self, **entries):
            self.__dict__.update(entries)


    cfg = ConfigClass(**config_dict)
except:
    print("Error!!! => no config file found at '{}'".format(config_file))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(epoch_num, state, is_best, path, modelname):
    if not os.path.exists(path):
        os.mkdir(path)
    timenow = getCurrentTime()
    filename = path + modelname + '_epoch_' + str(epoch_num) + '_date' + timenow + '.pth.tar'
    torch.save(state, filename)
    if is_best:
        best_filename = filename[:-8] + '_best.pth.tar'
        shutil.copyfile(filename, best_filename)


def getCurrentTime():
    return time.strftime('_%m%d%H%M', time.localtime(time.time()))


# ---------------------------------------------------------------------------- #
# Setup dataloader, model, loss, optimizer, scheduler, etc
# ---------------------------------------------------------------------------- #

dataset = kitti_loader(data_dir=cfg.root_dir, point_cloud_files= cfg.point_cloud_files,
                       data_type=cfg.data_type, labels_files= cfg.labels_files,
                       train=True, skip_frames=1)
dataloader = DataLoader(dataset, batch_size=cfg.batch_size * cfg.num_gpus, shuffle=True,
                        num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
test_dataset = kitti_loader(data_dir=cfg.root_dir, point_cloud_files= cfg.point_cloud_files,
                            data_type=cfg.data_type, labels_files= cfg.labels_files,
                            train=False)
test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size * cfg.num_gpus, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

# model = Our_AUNet(cfg)
# model = Our_DS_UNet(cfg)
model = Our_trans_DSUNet(cfg)
print("Model has {} paramerters in total".format(sum(x.numel() for x in model.parameters())))

# model = Our_UNet(cfg)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.cuda()

# loss_crs = ClsLoss(ignore_index=0, reduction='mean')
loss_crs = nn.CrossEntropyLoss(ignore_index=0, reduction='mean').cuda()

# lossHuber = nn.SmoothL1Loss(reduction = "mean").cuda()
# loss_spatial = SpatialSmoothLoss().cuda()
# loss_maskedcrs = MaskedCrsLoss().cuda()



# optimizer = torch.optim.SGD(model.parameters(), lr=cfg.base_lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True,
#                                                        threshold=0.0001, threshold_mode='rel', cooldown=1, min_lr=0,
#                                                        eps=1e-08)

# optimizer = torch.optim.SGD(model.parameters(), lr=cfg.base_lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)

# 根据 mit论文 采用 adam 加 cos lr 方法。 但是效果并不好，因为学习率并不是单调递减。
# optimizer = torch.optim.Adam(model.parameters(), lr=cfg.base_lr, weight_decay=cfg.weight_decay)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5 , T_mult=2,eta_min=0.000003)

optimizer = torch.optim.Adam(model.parameters(), lr=cfg.base_lr, weight_decay=cfg.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.35, patience=5, verbose=True,
                                                       threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
                                                       eps=1e-08)

# ---------------------------------------------------------------------------- #
# Train
# ---------------------------------------------------------------------------- #
def train(epoch):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    start = time.time()
    for batch_idx, (data, labels) in tqdm(enumerate(dataloader)):
        data_time.update(time.time() - start)
        batch_size = data.shape[0]  # Batch size
        pts_num = data.shape[1]  # Num of points in PointCloud
        voxels = [];
        coors = [];
        num_points = [];

        data = data.numpy()
        for i in range(batch_size):
            v, c, n = points_to_voxel(data[i], cfg.voxel_size, cfg.pc_range, cfg.max_points_voxel, True, cfg.max_voxels)
            c = torch.from_numpy(c)
            c = F.pad(c, (1, 0), 'constant', i) # 为什么pad填的是i？
            coors.append(c)
            voxels.append(torch.from_numpy(v))
            num_points.append(torch.from_numpy(n))

        coors = torch.cat(coors).float().cuda()
        voxels = torch.cat(voxels).float().cuda()
        num_points = torch.cat(num_points).float().cuda()
        labels = labels.float().cuda()

        optimizer.zero_grad()
        # output_cls,output_nor = model(voxels, num_points, coors)
        output_cls = model(voxels, num_points, coors)
        # output: torch.Size([32, 3, 128, 128])
        # labels: torch.Size([32, 128, 128])

        # loss = loss_crs.FocalLoss(output_cls,labels,gamma=2, alpha=0.5)
        loss = loss_crs(output_cls, labels.long())

        # loss=loss_mlm(output,labels.long())
        # loss=lossHuber(output,labels.long())
        # loss = cfg.alpha * lossHuber(output, labels) + cfg.beta * lossSpatial(output)
        # loss = lossHuber(output, labels)
        # loss = masked_huber_loss(output, labels, mask)

        loss.backward()
        optimizer.step()

        # measure losses and elapsed time
        losses.update(loss.item(), batch_size)
        batch_time.update(time.time() - start)

        start = time.time()
        if batch_idx % cfg.print_freq == 0:
            print('Train Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch, batch_idx, len(dataloader), batch_time=batch_time,
                data_time=data_time, loss=losses))

    return losses.avg

# ---------------------------------------------------------------------------- #
# Validate for one epoch
# ---------------------------------------------------------------------------- #
def validate():
    model.eval()  # switch to evaluate mode
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    with torch.no_grad():
        start = time.time()
        for batch_idx, (data, labels) in enumerate(test_dataloader):
            data_time.update(time.time() - start)
            batch_size = data.shape[0]  # Batch size
            pts_num = data.shape[1]  # Num of points in PointCloud

            voxels = [];
            coors = [];
            num_points = [];

            data = data.numpy()
            for i in range(batch_size):
                v, c, n = points_to_voxel(data[i], cfg.voxel_size, cfg.pc_range, cfg.max_points_voxel, True,
                                          cfg.max_voxels)
                c = torch.from_numpy(c)
                c = F.pad(c, (1, 0), 'constant', i)
                voxels.append(torch.from_numpy(v))
                coors.append(c)
                num_points.append(torch.from_numpy(n))

            voxels = torch.cat(voxels).float().cuda()
            coors = torch.cat(coors).float().cuda()
            num_points = torch.cat(num_points).float().cuda()
            labels = labels.float().cuda()

            output_cls = model(voxels, num_points, coors)

            # loss = loss_crs.FocalLoss(output_cls,labels,gamma=2, alpha=0.5)
            loss = loss_crs(output_cls, labels.long())

            # loss_mbce=loss_maskedcrs(output,labels,masks)
            # loss_spat=loss_spatial(output)
            # # loss= 0.9*loss_mbec+0.1*loss_spat
            # # loss_ce=nn.CrossEntropyLoss(ignore_index=0)(output, labels)
            # loss= loss_mbce
            # loss = cfg.alpha * lossHuber(output, labels) + cfg.beta * lossSpatial(output)
            # loss = lossHuber(output, labels)
            # loss = masked_huber_loss(output, labels, mask)



            # measure elapsed time
            losses.update(loss.item(), batch_size)
            batch_time.update(time.time() - start)

            start = time.time()
            if batch_idx % cfg.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    batch_idx, len(test_dataloader), batch_time=batch_time, loss=losses))
    return losses.avg


lowest_loss = 1.0

def main():
    global lowest_loss
    if cfg.evaluate:
        validate()
        return
    for epoch in range(cfg.epochs):
        loss_tra = train(epoch)
        loss_val = validate()
        scheduler.step(metrics=0)  # adjust_learning_rate

        if (cfg.save_checkpoints):
            # remember best prec@1 and save checkpoint
            is_best = loss_val < lowest_loss
            lowest_loss = min(loss_val, lowest_loss)
            save_checkpoint(epoch, {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'lowest_loss': lowest_loss,
                'optimizer': optimizer.state_dict(),
            }, is_best, cfg.checkpoints_path, cfg.model_name)

if __name__ == '__main__':
    main()
