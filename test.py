import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 留出前几个GPU跑其他程序, 需要在导入模型前定义
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import yaml
import time

# from gdn_dataset import kitti_gnd
from model import Our_AUNet
from tools.utils import points_to_voxel

'''2021.1.27
用于测试模型准确度，交并比，时间。
'''

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

model = Our_AUNet(cfg)
# model= Our_UNet(cfg)
# model= Our_trans_DSUNet(cfg)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.cuda()

pretrain_dir = os.path.join(cfg.checkpoints_path, 'Our_AUNet_kitti_data_3_epoch_8_date_05122317_best.pth.tar')
checkpoint = torch.load(pretrain_dir)
model.load_state_dict(checkpoint['state_dict'])


def test():
    model.eval()
    with torch.no_grad():
        root_dir = os.path.join(cfg.root_dir, '08', 'pc_s_n')
        part_list = os.listdir(root_dir)
        part_list.sort(key=lambda x: str(x[:-4]))
        time_list = []
        acc = []
        TPR = []
        FPR = []
        Iou = []
        Iou_1 = []
        Iou_2 = []
        F1_score = []
        count = 0
        for part in part_list:
            TP = 0
            TN = 0
            FP = 0
            FN = 0
            data_dir = os.path.join(root_dir, part)
            data = np.load(data_dir)
            # data = data[:, :4] # 无normal时

            # prepare label
            label_dir = data_dir.replace('pc_s_n', 'lb_s_pt')
            label = np.load(label_dir)
            for i in range(len(label)):
                if label[i] == 40 or label[i] == 44 or label[i] == 48 or label[i] == 49 or label[i] == 60:
                    label[i] = 1
                else:
                    label[i] = 2

            voxels = []
            coors = []
            num_points = []
            v, c, n = points_to_voxel(data, cfg.voxel_size, cfg.pc_range, cfg.max_points_voxel, True, cfg.max_voxels)
            c = torch.from_numpy(c)
            c = F.pad(c, (1, 0), 'constant', 0)
            voxels.append(torch.from_numpy(v))
            coors.append(c)
            num_points.append(torch.from_numpy(n))

            voxels = torch.cat(voxels).float().cuda()
            coors = torch.cat(coors).float().cuda()
            num_points = torch.cat(num_points).float().cuda()

            start_time = time.time()
            output = model(voxels, num_points, coors)
            end_time = time.time()
            time_list.append(end_time - start_time)
            # print(output.size())
            pred = output[0].argmax(0).cpu().numpy()

            for k in range(len(data)):
                pre_clf = pred[int(np.floor((data[k, 1] + 51.2) / 0.8)), int(np.floor((data[k, 0] + 51.2 / 0.8)))]
                if label[k] == 1:
                    if pre_clf == label[k]:
                        TP += 1
                    else:
                        FN += 1
                else:
                    if pre_clf == label[k]:
                        TN += 1
                    else:
                        FP += 1
            acc_tem = float(TP + TN) / float(TP + TN + FP + FN)
            acc.append(acc_tem)
            TPR_tem = float(TP) / float(TP + FN)
            TPR.append(TPR_tem)
            FPR_tem = float(FP) / float(FP + TN)
            FPR.append(FPR_tem)
            Iou_1_tem = float(TP) / float(TP + FP + FN)
            Iou_1.append(Iou_1_tem)
            Iou_2_tem = float(TN) / float(TN + FP + FN)
            Iou_2.append(Iou_2_tem)
            Iou_tem = (float(TP) / float(TP + FP + FN) + float(TN) / float(TN + FP + FN)) / 2
            Iou.append(Iou_tem)
            F1_score_tem = 2 * float(TP) / (2 * float(TP) + float(FP) + float(FN))
            F1_score.append(F1_score_tem)
            if count % 50 == 0:
                print("- %d frames inference processed." % count)
                print("- ACC:", acc_tem)
                print("- TPR:", TPR_tem)
                print("- FPR:", FPR_tem)
                print("- IOU:", Iou_tem)
                print("- IOU_1:", Iou_1_tem)
                print("- IOU_2:", Iou_2_tem)
                print("- F1_score:", F1_score_tem)
                print("- FPS:", 1 / (end_time - start_time))
            count = count + 1

        acc = np.array(acc)
        TPR = np.array(TPR)
        FPR = np.array(FPR)
        Iou = np.array(Iou)
        Iou_1 = np.array(Iou_1)
        Iou_2 = np.array(Iou_2)
        F1_score = np.array(F1_score)
        time_list = np.array(time_list)

        print("Summary:")

        print('-- ACC:', acc.sum() / len(acc))
        print('-- TPR:', TPR.sum() / len(TPR))
        print('-- FPR:', FPR.sum() / len(FPR))
        print('-- IOU:', Iou.sum() / len(Iou))
        print('-- IOU_1:', Iou_1.sum() / len(Iou_1))
        print('-- IOU_2:', Iou_2.sum() / len(Iou_2))
        print('-- F1_score:', F1_score.sum() / len(F1_score))
        print('-- FPS:', 1 / (time_list.sum() / len(time_list)))

        print("Test Completed.")

        if not os.path.exists(os.path.join(cfg.evaluation_path, cfg.model_name)):
            os.mkdir(os.path.join(cfg.evaluation_path, cfg.model_name))
        np.save(os.path.join(cfg.evaluation_path, cfg.model_name, 'acc'), acc)  # 需要修改
        np.save(os.path.join(cfg.evaluation_path, cfg.model_name, 'TPR'), TPR)
        np.save(os.path.join(cfg.evaluation_path, cfg.model_name, 'FPR'), FPR)
        np.save(os.path.join(cfg.evaluation_path, cfg.model_name, 'Time'), time_list)
        np.save(os.path.join(cfg.evaluation_path, cfg.model_name, 'IOU'), Iou)
        np.save(os.path.join(cfg.evaluation_path, cfg.model_name, 'IOU_1'), Iou_1)
        np.save(os.path.join(cfg.evaluation_path, cfg.model_name, 'IOU_2'), Iou_2)
        np.save(os.path.join(cfg.evaluation_path, cfg.model_name, 'F1_score'), F1_score)


if __name__ == '__main__':
    test()
