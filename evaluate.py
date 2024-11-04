import copy
import sys
sys.path.append('./models')    # 添加目录
import os
import cv2
import numpy as np
import sys
import argparse
from sklearn.metrics import roc_auc_score, roc_curve, ConfusionMatrixDisplay, precision_recall_curve, precision_recall_fscore_support
from tqdm import tqdm
from matplotlib import pyplot as plt
import csv
import shutil

import torch
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models.mvss_res2net import get_mvss_res2net, get_mvss
from torchinfo import summary
from natsort import natsorted
import time
import random
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.metrics import f1_score
from TruForMetrics import computeLocalizationMetrics, computeDetectionMetrics

# 该.py文件更正了之前把mask也resize了的错误

def setDir(filepath):
    '''
    如果文件夹不存在就创建，如果文件存在就清空！
    :param filepath:需要创建的文件夹路径
    :return:
    '''
    if not os.path.exists(filepath):
        # os.mkdir(filepath)
        os.makedirs(filepath, exist_ok=True)
    else:
        shutil.rmtree(filepath, ignore_errors=True)
        # os.mkdir(filepath)
        os.makedirs(filepath, exist_ok=True)

def read_paths(paths_file, subset):
    data = []

    with open(paths_file, 'r') as f:
        lines = f.readlines()
        for l in lines:
            parts = l.rstrip().split(' ')
            input_image_path = parts[0]
            mask_image_path = parts[1]
            if args.with_edge:
                # parts[2] is the path for edges, skipped
                label = int(parts[3])
            else:
                label = int(parts[2])

            if (subset and subset not in input_image_path):
                continue

            data.append((input_image_path, mask_image_path, label))

    return data


def cal_fmeasure(precision, recall):

    fmeasure = [[(2 * p * r) / (p + r + 1e-10)] for p, r in zip(precision, recall)]
    fmeasure=np.array(fmeasure)
    fmeasure=fmeasure[fmeasure[:, 0].argsort()]

    max_fmeasure=fmeasure[-1,0]
    return max_fmeasure


def cal_precision_recall_mae(prediction, gt):
    y_test=gt.flatten()
    y_pred=prediction.flatten()
    precision,recall,thresholds=precision_recall_curve(y_test,y_pred)
    auc_score = roc_auc_score(y_test, y_pred)
    return precision, recall,auc_score


def calculate_pixel_f1_iou(pd, gt):
    # both the predition and groundtruth are empty
    if np.max(pd) == np.max(gt) and np.max(pd) == 0:
        return 1.0, 0.0, 0.0, 1.0, 0.0
    seg_inv, gt_inv = np.logical_not(pd), np.logical_not(gt)
    true_pos = float(np.logical_and(pd, gt).sum())
    true_neg = float(np.logical_and(seg_inv, gt_inv).sum())
    false_pos = np.logical_and(pd, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, gt).sum()
    f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1.)
    false_alarm = false_pos / (true_neg + false_pos + 1.)
    # precision = true_pos / (true_pos + false_pos + 1.)
    # recall = true_pos / (true_pos + false_neg + 1.)
    # cross = np.logical_and(pd, gt)
    # union = np.logical_or(pd, gt)
    # iou = np.sum(cross) / (np.sum(union) + 1.)
    # mcc = (true_pos * true_neg - false_pos * false_neg) / (np.sqrt(
    #     (true_pos + false_pos) * (true_pos + false_neg) * (true_neg + false_pos) * (true_neg + false_neg)) + 1e-6)

    # if f1 == 0:
    #     print(' ')
    return f1, -1, -1, -1, false_alarm


def calculate_img_score(pd, gt):
    seg_inv, gt_inv = np.logical_not(pd), np.logical_not(gt)
    true_pos = float(np.logical_and(pd, gt).sum())
    false_pos = np.logical_and(pd, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, gt).sum()
    true_neg = float(np.logical_and(seg_inv, gt_inv).sum())
    acc = (true_pos + true_neg) / (true_pos + true_neg + false_neg + false_pos + 1e-6)
    sen = true_pos / (true_pos + false_neg + 1e-6)
    spe = true_neg / (true_neg + false_pos + 1e-6)
    f1 = 2 * sen * spe / (sen + spe)
    pc = true_pos / (true_pos + false_pos + 1e-6)
    rc = true_pos / (true_pos + false_neg + 1e-6)
    f1_ori = 2 * pc * rc / (pc + rc + 1e-6)
    false_alarm = false_pos / (true_neg + false_pos + 1e-6)
    return acc, sen, spe, f1, true_pos, true_neg, false_pos, false_neg, f1_ori, false_alarm


def save_cm(y_true, y_pred, save_path):
    plt.figure()
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)


def save_auc(y_true, scores, save_path):
    fpr, tpr, _ = roc_curve(y_true, scores, pos_label=1)

    plt.figure()

    plt.plot(
        fpr,
        tpr,
        label="ROC curve",
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")

    plt.tight_layout()

    plt.savefig(save_path, dpi=300)


class DeepfakeDataset(Dataset):
    def sampling(self, distribution, n_max, balance_data):
        if self.n_c_samples is None:
            self.n_c_samples = n_max

        for label_str in distribution:
            list = distribution[label_str]
            n_list = len(list)


            if balance_data:
                if (n_list >= self.n_c_samples):  # 进行正负样本均衡
                    # undersampling
                    picked = random.sample(list, self.n_c_samples)
                else:
                    # oversampling
                    for _ in range(self.n_c_samples // n_list):
                        for i in list:
                            (input_image_path, mask_image_path, edge_image_path) = i
                            self.input_image_paths.append(input_image_path)
                            self.mask_image_paths.append(mask_image_path)
                            self.edge_image_paths.append(edge_image_path)
                            self.labels.append(int(label_str))

                    picked = random.sample(list, self.n_c_samples % n_list)
            else:
                picked = random.sample(list, n_list)  # 不进行正负样本均衡

            # for picked
            for p in picked:
                (input_image_path, mask_image_path, edge_image_path) = p
                self.input_image_paths.append(input_image_path)
                self.mask_image_paths.append(mask_image_path)
                self.edge_image_paths.append(edge_image_path)
                self.labels.append(int(label_str))

        return
    def __init__(self, paths_file_list, image_size, n_c_samples=None, val=True, manipulate_dataset=False, randomcrop=False):
        self.image_size = image_size
        self.manipulate_dataset = manipulate_dataset
        self.randomcrop = randomcrop

        self.n_c_samples = n_c_samples

        self.val = val

        self.input_image_paths = []
        self.mask_image_paths = []
        self.edge_image_paths = []
        self.labels = []


        for paths_file in paths_file_list:
            distribution = dict()
            n_max = 0
            with open(paths_file, 'r') as f:
                lines = f.readlines()
                for l in lines:
                    if 'MISD_Dataset' in l:
                        parts = l.rstrip().split(',')
                    else:
                        parts = l.rstrip().split(' ')
                    if len(parts) == 4:
                        input_image_path = parts[0]
                        mask_image_path = parts[1]
                        edge_image_path = parts[2]
                        label_str = parts[3]
                    elif len(parts) == 3:
                        input_image_path = parts[0]
                        mask_image_path = parts[1]
                        edge_image_path = parts[1].replace('masks', 'edges')
                        label_str = parts[2]
                    else:
                        raise NotImplementedError('len parts not in 3, 4')

                    if int(label_str) > 1:
                        label_str = '1'
                    # check all paths exists
                    assert os.path.exists(input_image_path)
                    if mask_image_path != 'None':
                        assert os.path.exists(mask_image_path)
                        assert os.path.exists(edge_image_path)
                    assert label_str in '0,1'

                    # add to distribution
                    if (label_str not in distribution):
                        distribution[label_str] = [(input_image_path, mask_image_path,edge_image_path)]
                    else:
                        distribution[label_str].append((input_image_path, mask_image_path,edge_image_path))

                    if (len(distribution[label_str]) > n_max):
                        n_max = len(distribution[label_str])

            self.sampling(distribution, n_max, balance_data=False)

        # ----------
        #  TODO: Transforms for data augmentation (more augmentations should be added)
        # ----------
        # OSN transforms
        self.transform = T.Compose([
            np.float32,
            T.ToTensor(),
            # T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            T.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        ])
        self.transform_mask = T.Compose([
            np.float32,
            T.ToTensor(),
            # T.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        ])
        self.transform_val = A.Compose([
            A.Resize(self.image_size, self.image_size),
            # A.Normalize(always_apply=True, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            # A.Normalize(always_apply=True, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
            # ToTensorV2()
        ])


    def __getitem__(self, item):

        # ----------
        # Read label
        # ----------
        input_label = self.labels[item]

        # ----------
        # Read input image
        # ----------
        input_file_name = self.input_image_paths[item]

        input = cv2.cvtColor(cv2.imread(input_file_name), cv2.COLOR_BGR2RGB)

        height, width, _ = input.shape

        # ----------
        # Read mask
        # ----------
        mask_file_name = self.mask_image_paths[item]
        if ((mask_file_name == "None") and (self.val)):
            mask = np.zeros([height, width, 1], np.uint8)  # a totally black mask for real image
        else:
            mask = cv2.imread(mask_file_name, cv2.IMREAD_GRAYSCALE)
            # mask[mask < 200] = 0  # 确保只有0和255
            # mask[mask >= 200] = 255
            if mask.ndim < 3:
                mask = np.expand_dims(mask, axis=2)

        if (not self.val):
            transformed_train = self.transform_train(image=input, mask=mask)
            input = transformed_train['image']
            mask = transformed_train['mask']
        else:
            transformed_val = self.transform_val(image=input)
            input = transformed_val['image']
            # mask = transformed_val['mask']

        input = input.astype('float')
        mask = mask.astype('float')
        if input.max() > 1.0:
            input = input / 255.0
        if mask.max() > 1.0:
            mask = mask / 255.0
        if input_label > 1:
            input_label = 1

        input = self.transform(input)
        mask = self.transform_mask(mask)
        # edge = edge / 1.0
        # label_numpy = np.array(input_label)
        # input_label = torch.from_numpy(label_numpy)
        # # print(input_label)
        # input_label = F.one_hot(input_label, 4)  # 共有4类;真，cm1,sp2,ip3
        return input, mask, input_label, input_file_name

    def __len__(self):
        return len(self.input_image_paths)



def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--out_dir', type=str, default='/home/zzt/pan2/mvss_eval_pred')
    parser.add_argument('--epoch', type=int, help='for calculating lamda')
    parser.add_argument("--paths_file", type=str, default="/eval_files.txt",
                        help="path to the file with input paths")  # each line of this file should contain "/path/to/image.ext /path/to/mask.ext /path/to/edge.ext 1 (for fake)/0 (for real)"; for real image.ext, set /path/to/mask.ext and /path/to/edge.ext as a string None
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--load_path', type=str, help='path to the pretrained model', default="ckpt/mvssnet.pth")
    parser.add_argument("--image_size", type=int, default=512, help="size of the images for prediction")
    parser.add_argument("--subset", type=str, help="evaluation on certain subset")
    parser.add_argument("--with_edge", type=bool, default=False, help="whether txt file contain edge path")
    parser.add_argument("--eval_all", type=bool, default=False, help="whether txt file contain edge path")
    parser.add_argument("--checkpoints_dir", type=str, help="whether txt file contain edge path")
    parser.add_argument("--fix_lamda", type=float, default=-1, help="fixlamda")
    parser.add_argument("--save_prediction", action='store_true', help="save_prediction")
    parser.add_argument("--workers", type=int, default=8, help="workers")
    args = parser.parse_args()
    return args

txt_paths = [
            # '/home/zzt/pan1/DATASETS/Casiav1/Casiav1plus_with_edge_230322.txt',
            # '/home/zzt/pan1/DATASETS/Casiav1/Casiav1_with_edge_920_230322.txt',
            # '/home/zzt/pan1/DATASETS/Casiav2/Casiav2_with_edge_230322.txt',
            # '/home/zzt/pan1/DATASETS/Columbia/Columbia_with_edge_230322.txt',
            # '/home/zzt/pan1/DATASETS/Columbia/Columbia_with_edge_180_230322.txt',
            # '/home/zzt/pan1/DATASETS/Coverage/Coverage_with_edge_230322.txt',
            # '/home/zzt/pan1/DATASETS/Coverage/Coverage_with_edge_100_230322.txt',
            # '/home/zzt/pan1/DATASETS/NIST16/NIST16_with_edge_230322.txt',
            # '/home/zzt/pan1/DATASETS/NIST16/NIST16_with_edge_testset_231010.txt',
            '/home/zzt/pan1/DATASETS/NIST16/NIST16_with_edge_allfake_230322.txt',
            # '/home/zzt/pan1/DATASETS/IMD2020/IMD2020_with_edge_230322.txt',
            # '/home/zzt/pan1/DATASETS/IIDNET_DATASET/DiverseInpaintingDataset/IIDall_allfake_with_edge_230322.txt',
            # '/home/zzt/pan1/DATASETS/IIDNET_DATASET/DiverseInpaintingDataset/IIDall_allfake_val1100_231114.txt',
            # '/home/zzt/pan1/DATASETS/IIDNET_DATASET/DiverseInpaintingDataset/IIDall10_allfake_with_edge_231223.txt',
            # '/home/zzt/pan1/DATASETS/DEF12K/DEF12k_with_edge_230322.txt',
            # '/home/zzt/pan1/DATASETS/DEF12K/DEF12k_with_edge_val600_231114.txt',
            # '/home/zzt/pan1/DATASETS/test_pics_videos/test_file_paths_mvss.txt',
            # '/home/zzt/pan1/DATASETS/DEF84K/DEF84k_with_edge_val_230322.txt',
            # '/home/zzt/pan1/DATASETS/comofod_small/comofod_with_edge_230322.txt',
            # '/home/zzt/pan1/DATASETS/MISD_Dataset/MISD_with_edge_230322.txt',
            # '/home/zzt/pan2/DATASETS/CocoGlide/CocoGlide_with_edge_230322.txt',
            # '/home/zzt/pan2/DATASETS/OIS/OIS_v01/OIS_list_withEdge_231013.txt',
            # '/home/zzt/pan2/DATASETS/DSO/tifs-database/DSO1_list_withEdge_231013.txt'
            # '/home/zzt/pan1/DATASETS/val10_200_datasets.txt',
            # '/home/zzt/pan1/DATASETS/val6_200_datasets.txt',
            # '/home/zzt/pan1/DATASETS/val6_01_datasets.txt',
            # '/home/zzt/pan1/DATASETS/OSN_Dataset/OSN_CASIA_Facebook_with_edge_240117.txt',
            # '/home/zzt/pan1/DATASETS/OSN_Dataset/OSN_CASIA_Wechat_with_edge_240117.txt',
            # '/home/zzt/pan1/DATASETS/OSN_Dataset/OSN_CASIA_Weibo_with_edge_240117.txt',
            # '/home/zzt/pan1/DATASETS/OSN_Dataset/OSN_CASIA_Whatsapp_with_edge_240117.txt',
            # '/home/zzt/pan1/DATASETS/OSN_Dataset/OSN_CASIA_with_edge_240117.txt',
            # '/home/zzt/pan1/DATASETS/OSN_Dataset/OSN_Columbia_Facebook_with_edge_240117.txt',
            # '/home/zzt/pan1/DATASETS/OSN_Dataset/OSN_Columbia_Wechat_with_edge_240117.txt',
            # '/home/zzt/pan1/DATASETS/OSN_Dataset/OSN_Columbia_Weibo_with_edge_240117.txt',
            # '/home/zzt/pan1/DATASETS/OSN_Dataset/OSN_Columbia_Whatsapp_with_edge_240117.txt',
            # '/home/zzt/pan1/DATASETS/OSN_Dataset/OSN_Columbia_with_edge_240117.txt',
            # '/home/zzt/pan1/DATASETS/OSN_Dataset/OSN_DSO_Facebook_with_edge_240117.txt',
            # '/home/zzt/pan1/DATASETS/OSN_Dataset/OSN_DSO_Wechat_with_edge_240117.txt',
            # '/home/zzt/pan1/DATASETS/OSN_Dataset/OSN_DSO_Weibo_with_edge_240117.txt',
            # '/home/zzt/pan1/DATASETS/OSN_Dataset/OSN_DSO_Whatsapp_with_edge_240117.txt',
            # '/home/zzt/pan1/DATASETS/OSN_Dataset/OSN_DSO_with_edge_240117.txt',
            # '/home/zzt/pan1/DATASETS/OSN_Dataset/OSN_NIST16_Facebook_with_edge_240117.txt',
            # '/home/zzt/pan1/DATASETS/OSN_Dataset/OSN_NIST16_Wechat_with_edge_240117.txt',
            # '/home/zzt/pan1/DATASETS/OSN_Dataset/OSN_NIST16_Weibo_with_edge_240117.txt',
            # '/home/zzt/pan1/DATASETS/OSN_Dataset/OSN_NIST16_Whatsapp_with_edge_240117.txt',
            # '/home/zzt/pan1/DATASETS/OSN_Dataset/OSN_NIST16_with_edge_240117.txt',
            ]

txtPath2nickName = {
    '/home/zzt/pan1/DATASETS/Casiav1/Casiav1plus_with_edge_230322.txt': 'Casiav1plus',
    # '/home/zzt/pan1/DATASETS/Casiav1/Casiav1_with_edge_920_230322.txt': 'Casiav1',
    '/home/zzt/pan1/DATASETS/Casiav2/Casiav2_with_edge_230322.txt': 'Casiav2',
    '/home/zzt/pan1/DATASETS/Columbia/Columbia_with_edge_230322.txt': 'Columbia',
    # '/home/zzt/pan1/DATASETS/Columbia/Columbia_with_edge_180_230322.txt': 'Columbia180',
    '/home/zzt/pan1/DATASETS/Coverage/Coverage_with_edge_230322.txt': 'Coverage',
    # '/home/zzt/pan1/DATASETS/Coverage/Coverage_with_edge_100_230322.txt': 'Coverage100',
    '/home/zzt/pan1/DATASETS/NIST16/NIST16_with_edge_230322.txt': 'NIST16',
    '/home/zzt/pan1/DATASETS/NIST16/NIST16_with_edge_allfake_230322.txt': 'NIST16allfake',
    '/home/zzt/pan1/DATASETS/IMD2020/IMD2020_with_edge_230322.txt': 'IMD2020',
    '/home/zzt/pan1/DATASETS/IIDNET_DATASET/DiverseInpaintingDataset/IIDall_allfake_with_edge_230322.txt': 'IID11k',
    '/home/zzt/pan1/DATASETS/IIDNET_DATASET/DiverseInpaintingDataset/IIDall10_allfake_with_edge_231223.txt': 'IID10k',
    '/home/zzt/pan1/DATASETS/IIDNET_DATASET/DiverseInpaintingDataset/IIDall_allfake_val1100_231114.txt': 'IID_val1100',
    '/home/zzt/pan1/DATASETS/DEF12K/DEF12k_with_edge_230322.txt': 'DEF12K',
    '/home/zzt/pan1/DATASETS/DEF12K/DEF12k_with_edge_val600_231114.txt': 'DEF12K_val600',
    # '/home/zzt/pan1/DATASETS/test_pics_videos/test_file_paths_mvss.txt': 'test_pics_videos',
    # '/home/zzt/pan1/DATASETS/DEF84K/DEF84k_with_edge_val_230322.txt': 'DEF84K',
    '/home/zzt/pan1/DATASETS/comofod_small/comofod_with_edge_230322.txt': 'Comofod',
    '/home/zzt/pan1/DATASETS/MISD_Dataset/MISD_with_edge_230322.txt': 'MISD',
    '/home/zzt/pan2/DATASETS/CocoGlide/CocoGlide_with_edge_230322.txt': 'CocoGlide',
    '/home/zzt/pan2/DATASETS/OIS/OIS_v01/OIS_list_withEdge_231013.txt': 'OIS',
    '/home/zzt/pan2/DATASETS/DSO/tifs-database/DSO1_list_withEdge_231013.txt': 'DSO',
    '/home/zzt/pan1/DATASETS/val10_200_datasets.txt': 'val10_200',
    '/home/zzt/pan1/DATASETS/val6_200_datasets.txt': 'val6_200',
    '/home/zzt/pan1/DATASETS/val6_01_datasets.txt': 'val6_01',
    '/home/zzt/pan1/DATASETS/OSN_Dataset/OSN_CASIA_Facebook_with_edge_240117.txt': 'OSN_CASIA_Facebook',
    '/home/zzt/pan1/DATASETS/OSN_Dataset/OSN_CASIA_Wechat_with_edge_240117.txt': 'OSN_CASIA_Wechat',
    '/home/zzt/pan1/DATASETS/OSN_Dataset/OSN_CASIA_Weibo_with_edge_240117.txt': 'OSN_CASIA_Weibo',
    '/home/zzt/pan1/DATASETS/OSN_Dataset/OSN_CASIA_Whatsapp_with_edge_240117.txt': 'OSN_CASIA_Whatsapp',
    '/home/zzt/pan1/DATASETS/OSN_Dataset/OSN_CASIA_with_edge_240117.txt': 'OSN_CASIA',
    '/home/zzt/pan1/DATASETS/OSN_Dataset/OSN_Columbia_Facebook_with_edge_240117.txt': 'OSN_Columbia_Facebook',
    '/home/zzt/pan1/DATASETS/OSN_Dataset/OSN_Columbia_Wechat_with_edge_240117.txt': 'OSN_Columbia_Wechat',
    '/home/zzt/pan1/DATASETS/OSN_Dataset/OSN_Columbia_Weibo_with_edge_240117.txt': 'OSN_Columbia_Weibo',
    '/home/zzt/pan1/DATASETS/OSN_Dataset/OSN_Columbia_Whatsapp_with_edge_240117.txt': 'OSN_Columbia_Whatsapp',
    '/home/zzt/pan1/DATASETS/OSN_Dataset/OSN_Columbia_with_edge_240117.txt': 'OSN_Columbia',
    '/home/zzt/pan1/DATASETS/OSN_Dataset/OSN_DSO_Facebook_with_edge_240117.txt': 'OSN_DSO_Facebook',
    '/home/zzt/pan1/DATASETS/OSN_Dataset/OSN_DSO_Wechat_with_edge_240117.txt': 'OSN_DSO_Wechat',
    '/home/zzt/pan1/DATASETS/OSN_Dataset/OSN_DSO_Weibo_with_edge_240117.txt': 'OSN_DSO_Weibo',
    '/home/zzt/pan1/DATASETS/OSN_Dataset/OSN_DSO_Whatsapp_with_edge_240117.txt': 'OSN_DSO_Whatsapp',
    '/home/zzt/pan1/DATASETS/OSN_Dataset/OSN_DSO_with_edge_240117.txt': 'OSN_DSO',
    '/home/zzt/pan1/DATASETS/OSN_Dataset/OSN_NIST16_Facebook_with_edge_240117.txt': 'OSN_NIST16_Facebook',
    '/home/zzt/pan1/DATASETS/OSN_Dataset/OSN_NIST16_Wechat_with_edge_240117.txt': 'OSN_NIST16_Wechat',
    '/home/zzt/pan1/DATASETS/OSN_Dataset/OSN_NIST16_Weibo_with_edge_240117.txt': 'OSN_NIST16_Weibo',
    '/home/zzt/pan1/DATASETS/OSN_Dataset/OSN_NIST16_Whatsapp_with_edge_240117.txt': 'OSN_NIST16_Whatsapp',
    '/home/zzt/pan1/DATASETS/OSN_Dataset/OSN_NIST16_with_edge_240117.txt': 'OSN_NIST16',
}

if __name__ == '__main__':

    total_time_start = time.time()
    args = parse_args()

    # load model
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    model = get_mvss(backbone='resnet50',
                         pretrained_base=True,
                         nclass=1,
                         sobel=True,
                         constrain=True,
                         n_input=3,
                         input_image_size=args.image_size,
                         ).to(device)

    # summary(model, input_size=(12, 3, 512, 512))
    checkpoints = os.walk(args.checkpoints_dir)
    for _ in checkpoints:
        checkpoint_names = _
    checkpoint_id = checkpoint_names[2][0][:14]
    print(f'checkpoint_id={checkpoint_id}')
    txt_file_savepath = checkpoint_names[0] + f'{checkpoint_names[2][0][:14]}' + '_evaluate_all_ckpt_result.txt'
    txt_file = open(txt_file_savepath, 'w')
    txt_file.write('checkpoint|dataset|pixel-f1|img-level-acc|sen|spe|f1|combine-f1|f1_pere|falsealarm' + '\n')
    for cnt, checkpoint in enumerate(natsorted(checkpoint_names[2])):
        print(checkpoint)
        # if cnt<10:
        #     continue
        if not checkpoint.split('_')[1].isdigit():
            continue
        args.load_path = args.checkpoints_dir + checkpoint
        args.epoch = cnt

        if os.path.exists(args.load_path):
            model_checkpoint = torch.load(args.load_path, map_location='cpu')
            model.load_state_dict(model_checkpoint, strict=True)
            print("load %s finish" % (os.path.basename(args.load_path)))
        else:
            print("%s not exist" % args.load_path)
            sys.exit()

        # no training
        model.eval()
        if args.eval_all:
            for txt_path in txt_paths:
                args.paths_file = txt_path
                print(txt_path)
                # 准备输出目录
                if args.save_prediction:  # 输出根目录/ckpt总目录名称/当前ckpt名称/测试数据集
                    pred_save_dir = os.path.join(args.out_dir, f'{checkpoint_id}_pred_masks', checkpoint.split('.')[0],
                                                 txtPath2nickName[txt_path])
                    setDir(pred_save_dir)
                dataset = DeepfakeDataset([txt_path],
                                          args.image_size,
                                          val=True,
                                          )
                dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=args.workers, shuffle=False)

                # # read paths for data
                # if not os.path.exists(args.paths_file):
                #     print("%s not exists, quit" % args.paths_file)
                #     sys.exit()
                #
                # if (args.subset):
                #     print("Evaluation on subset {}".format(args.subset))
                #
                # data = read_paths(args.paths_file, args.subset)
                #
                # print("Eval set size is {}!".format(len(data)))
                #
                # # # create/reset output folder
                # # print("Predicted maps will be saved in :%s" % args.out_dir)
                # # os.makedirs(args.out_dir, exist_ok=True)
                # # os.makedirs(os.path.join(args.out_dir, 'masks'), exist_ok=True)
                #
                # # # csv
                # # if (args.subset is None):
                # #     f_csv = open(os.path.join(args.out_dir, 'pred.csv'), 'w')
                # #     writer = csv.writer(f_csv)
                # #
                # #     header = ['Image', 'Score', 'Pred', 'True', 'Correct']
                # #     writer.writerow(header)
                #
                # # transforms
                # # transform = transforms.Compose([
                # #     transforms.ToPILImage(),
                # #     transforms.Resize((args.image_size, args.image_size)),
                # #     transforms.ToTensor()])  # 已从0-255到0.0-1.0
                # # transform by albumentation
                # transform_val = A.Compose([
                #     A.Resize(args.image_size, args.image_size),
                #     A.Normalize(always_apply=True, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                #     # A.Normalize(always_apply=True, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
                #     ToTensorV2()
                # ])
                #
                transform_pil = transforms.Compose([transforms.ToPILImage()])

                # for storting results
                scores, labs = [], []
                f1s = [[], []]
                ious = [[], []]
                mccs = [[], []]
                precisions = [[], []]
                recalls = [[], []]
                f1s_sklearn = []
                precision_sk = []
                recall_sk = []
                # if args.fix_lamda>=0:
                #     print(f'lamda:{args.fix_lamda}')
                # else:
                #     print(f'lamda:{0.9975**(args.epoch*args.epoch)}')
                # crcnn
                f1_scores = []
                aucs = []
                F1_best_list = []
                F1_th_list = []

                with torch.no_grad():
                    for ix, (img, gt, lab, img_path) in enumerate(tqdm(dataloader, mininterval=1)):
                        img_path = img_path[0]
                        # img = cv2.imread(img_path)
                        # ori_size = img.shape
                        #
                        # # resize to fit model
                        # # img = transform(img).to(device).unsqueeze(0)
                        # # apply transform
                        # transformed_val = transform_val(image=img)
                        # img = transformed_val['image'].to(device).unsqueeze(0)
                        img = img.to(device)
                        # img = img.clamp_(-1, 1)
                        gt = gt.squeeze().cpu().numpy()

                        # prediction
                        out_edges, seg, out_cls = model(img)
                        # _, seg = model(img)
                        # lamda = 0.9975**(args.epoch*args.epoch)
                        # if args.fix_lamda >= 0:
                        #     lamda = args.fix_lamda
                        # score = (1-lamda) * g1 + (lamda * g2)
                        score = np.array(torch.sigmoid(out_cls).squeeze().detach().cpu())
                        seg = torch.sigmoid(seg).detach().cpu()
                        # if torch.isnan(seg).any() or torch.isinf(seg).any():
                        #     score = 0.0
                        # else:
                        #     score = torch.max(seg).numpy()
                        # score = out_cls.numpy()

                        # resize to original
                        # seg = torch.sigmoid(seg).detach().cpu()
                        seg = [np.array(transform_pil(seg[i])) for i in range(len(seg))]
                        if len(seg) != 1:
                            print("%s seg size not 1" % img_path)
                            continue
                        else:
                            seg = seg[0].astype(np.uint8)
                        # seg = cv2.resize(seg, (args.image_size, args.image_size))  # the order of size here is important

                        if seg.shape != gt.shape:
                            # gt = cv2.resize(gt, (seg.shape[1], seg.shape[0]))
                            seg = cv2.resize(seg, (gt.shape[1], gt.shape[0]))
                            # print("%s size not match, reshaped" % img_path)

                        # save prediction
                        if args.save_prediction:
                            if 'DiverseInpaintingDataset/TE' in img_path:
                                save_seg_path = os.path.join(pred_save_dir,
                                                             'pred_' + os.path.basename(img_path)[:-4] + '_TE.png')
                            else:
                                save_seg_path = os.path.join(pred_save_dir,
                                                             'pred_' + os.path.basename(img_path)[:-4] + '.png')
                            cv2.imwrite(save_seg_path, seg.astype(np.uint8))

                        # convert from image to floating point
                        seg = seg / 255.0

                        scores.append(score)
                        labs.append(lab)
                        f1 = 0
                        # if mask_path != 'None':  # fake
                        #     gt = cv2.imread(mask_path, 0) / 255.0
                        # else:
                        #     gt = np.zeros((ori_size[0], ori_size[1]))

                        if seg.shape != gt.shape:
                            gt = cv2.resize(gt, (seg.shape[1], seg.shape[0]))
                            print("%s size not match, reshaped" % img_path)


                        # seg_origin = copy.deepcopy(seg)
                        # precision, recall, auc_score = cal_precision_recall_mae(seg, gt)
                        # f1 = cal_fmeasure(precision, recall)
                        # f1_scores.append(f1)
                        # aucs.append(auc_score)
                        # print('-> {:s} | F1 score: {:.3f}  |  AUC: {:.3f}'.format(img_path, f1, auc_score))


                        gt = (gt > args.threshold).astype(np.float64)

                        # #TruFor Metrics
                        # # seg_truFor = np.ones_like(seg) - seg
                        # # seg_truFor = np.concatenate((seg_truFor[np.newaxis, :], seg[np.newaxis, :]), axis=0)
                        # if lab > 0:
                        #     F1_best, F1_th = computeLocalizationMetrics(seg, gt)
                        #     F1_best_list.append(F1_best)
                        #     F1_th_list.append(F1_th)

                        seg = (seg > args.threshold).astype(np.float64)

                        # pixel-level F1
                        f1, precision, recall, iou, mcc = calculate_pixel_f1_iou(seg.flatten(), gt.flatten())
                        # f1s_sklearn.append(f1_score(gt.flatten(), seg.flatten(), average='binary', zero_division=1))
                        # precision_temp, recall_temp, f_score_temp, _ = precision_recall_fscore_support(gt.flatten(), seg.flatten(), average='binary', zero_division=1)
                        # f1s_sklearn.append(f1_score(gt.flatten(), seg.flatten(), average='macro'))
                        # precision_temp, recall_temp, f_score_temp, _ = precision_recall_fscore_support(gt.flatten(),
                        #                                                                                seg.flatten(),
                        #                                                                                average='macro')
                        # precision_sk.append(precision_temp)
                        # recall_sk.append(recall_temp)

                        f1s[lab].append(f1)
                        # ious[lab].append(iou)
                        mccs[lab].append(mcc)
                        # precisions[lab].append(precision)
                        # recalls[lab].append(recall)

                        # write to csv
                        # if (args.subset is None):
                        #     row = [img_path, score, (score > args.threshold).astype(int), lab,
                        #            (score > args.threshold).astype(int) == lab]
                        #     writer.writerow(row)

                # image-level AUC

                y_true = (np.array(labs) > args.threshold).astype(int)
                y_pred = (np.array(scores) > args.threshold).astype(int)
                # print(y_pred)

                # # TruthFor Metrics
                # try:
                #     tf_AUC, tf_bACC = computeDetectionMetrics(scores, y_true)
                # except:
                #     import traceback
                #     traceback.print_exc()
                #     tf_AUC, tf_bACC = -1, -1

                # if (args.subset is None):
                #     # save_path = os.path.join(args.out_dir, 'auc.png')
                #     # save_auc(y_true, scores, save_path)
                #     try:
                #         img_auc = roc_auc_score(y_true, scores)
                #     except:
                #         print("only one class")
                #         img_auc = 0.0
                # else:
                #     img_auc = 0.0

                # meanf1_sk = np.mean(f1s_sklearn)
                # meanprecision_sk = np.mean(precision_sk)
                # meanrecall_sk = np.mean(recall_sk)
                meanf1 = np.mean(f1s[0] + f1s[1])
                # meaniou = np.mean(ious[0] + ious[1])
                meanmcc = np.mean(mccs[0] + mccs[1])
                # meanprecision = np.mean(precisions[0] + precisions[1])
                # meanrecall = np.mean(recalls[0] + recalls[1])
                # meanbestf1_tf = np.nanmean(F1_best_list)
                # meanthf1_tf = np.nanmean(F1_th_list)
                print("pixel-f1: %.4f" % meanf1)
                # print('TruFor skipped:', np.count_nonzero(np.isnan(F1_best_list)))
                # print("meanBestPix_f1_tf: %.4f , meanThPix_f1_tf: %.4f" % (meanbestf1_tf, meanthf1_tf))
                # print("pixel-precision: %.4f" % meanprecision)
                # print("pixel-recall: %.4f" % meanrecall)
                # print("pixel-f1-sklearn: %.4f" % meanf1_sk)
                # print("pixel-precision_sk: %.4f" % meanprecision_sk)
                # print("pixel-recall_sk: %.4f" % meanrecall_sk)
                # print("mean-iou: %.4f" % meaniou)
                print("mean-falsealarm: %.4f" % meanmcc)

                acc, sen, spe, f1_imglevel, tp, tn, fp, fn, f1_pe_re, false_alarm = calculate_img_score(y_pred, y_true)
                print("img level acc: %.4f sen: %.4f  spe: %.4f00  f1: %.4f  f1_pere: %.4f  false_alarm: %.4f"
                      % (acc, sen, spe, f1_imglevel, f1_pe_re, false_alarm))
                print("combine f1: %.4f" % (2 * meanf1 * f1_imglevel / (f1_imglevel + meanf1 + 1e-6)))


                # print('~~~~~~~~~~~~~~~~~~')
                # print('|Dataset : ', args.paths_file, ' |')
                # print('|F1  is %5f |' % np.mean(np.array(f1_scores)))
                # print('|AUC is %5f |' % np.mean(np.array(aucs)))
                # print('~~~~~~~~~~~~~~~~~~')
                #
                print('~~~~~~~~~~~~~~~end~~~~~~~~~~~~~~~~~~~~')

                txt_file.write(
                    checkpoint + '|' + txt_path.split('/')[-1].split('.')[0] + '|' + f'{meanf1}|{acc}|{sen}|{spe}|{f1_imglevel}|{(2 * meanf1 * f1_imglevel / (f1_imglevel + meanf1 + 1e-6))}|{f1_pe_re}|{meanmcc}' + '\n')

                # # confusion matrix
                # save_path = os.path.join(args.out_dir, 'cm' + ('_' + args.subset if args.subset else '') + '.png')
                # save_cm(y_true, y_pred, save_path)

                # if (args.subset is None): f_csv.close()

        else:
            # read paths for data
            if not os.path.exists(args.paths_file):
                print("%s not exists, quit" % args.paths_file)
                sys.exit()

            if (args.subset):
                print("Evaluation on subset {}".format(args.subset))

            data = read_paths(args.paths_file, args.subset)

            print("Eval set size is {}!".format(len(data)))

            # create/reset output folder
            print("Predicted maps will be saved in :%s" % args.out_dir)
            os.makedirs(args.out_dir, exist_ok=True)
            os.makedirs(os.path.join(args.out_dir, 'masks'), exist_ok=True)

            # csv
            if (args.subset is None):
                f_csv = open(os.path.join(args.out_dir, 'pred.csv'), 'w')
                writer = csv.writer(f_csv)

                header = ['Image', 'Score', 'Pred', 'True', 'Correct']
                writer.writerow(header)

            # transforms
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((args.image_size, args.image_size)),
                transforms.ToTensor()])  # 已从0-255到0.0-1.0
            # transform by albumentation
            transform_val = A.Compose([
                A.Resize(args.image_size, args.image_size),
                # A.Normalize(always_apply=True, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.Normalize(always_apply=True, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
                ToTensorV2()
            ])

            transform_pil = transforms.Compose([transforms.ToPILImage()])

            # for storting results
            scores, labs = [], []
            f1s = [[], []]
            ious = [[], []]
            precisions = [[], []]
            recalls = [[], []]
            print(f'lamda:{0.9975 ** (args.epoch * args.epoch)}')
            # crcnn
            f1_scores = []
            aucs = []

            with torch.no_grad():
                for ix, (img_path, mask_path, lab) in enumerate(tqdm(data, mininterval=1)):
                    img = cv2.imread(img_path)
                    ori_size = img.shape

                    # resize to fit model
                    # img = transform(img).to(device).unsqueeze(0)
                    # apply transform
                    transformed_val = transform_val(image=img)
                    img = transformed_val['image'].to(device).unsqueeze(0)

                    # prediction
                    _, seg, g1, g2 = model(img)
                    lamda = 0.9975 ** (args.epoch * args.epoch)
                    score = (1 - lamda) * g1 + (lamda * g2)
                    score = np.array(score.squeeze().detach().cpu())

                    # resize to original
                    seg = torch.sigmoid(seg).detach().cpu()
                    seg = [np.array(transform_pil(seg[i])) for i in range(len(seg))]
                    if len(seg) != 1:
                        print("%s seg size not 1" % img_path)
                        continue
                    else:
                        seg = seg[0].astype(np.uint8)
                    seg = cv2.resize(seg, (ori_size[1], ori_size[0]))  # the order of size here is important

                    # save prediction
                    if args.save_prediciton:
                        save_seg_path = os.path.join(args.out_dir, 'masks',
                                                     'pred_' + os.path.basename(img_path).split('.')[0] + '.png')
                        cv2.imwrite(save_seg_path, seg.astype(np.uint8))

                    # convert from image to floating point
                    seg = seg / 255.0

                    scores.append(score)
                    labs.append(lab)
                    f1 = 0
                    if mask_path != 'None':  # fake
                        gt = cv2.imread(mask_path, 0) / 255.0
                    else:
                        gt = np.zeros((ori_size[0], ori_size[1]))

                    if seg.shape != gt.shape:
                        gt = cv2.resize(gt, (seg.shape[1], seg.shape[0]))
                        print("%s size not match, reshaped" % img_path)
                        continue

                    # seg_origin = copy.deepcopy(seg)
                    # precision, recall, auc_score = cal_precision_recall_mae(seg, gt)
                    # f1 = cal_fmeasure(precision, recall)
                    # f1_scores.append(f1)
                    # aucs.append(auc_score)
                    # print('-> {:s} | F1 score: {:.3f}  |  AUC: {:.3f}'.format(img_path, f1, auc_score))

                    seg = (seg > args.threshold).astype(np.float64)

                    # pixel-level F1
                    f1, precision, recall, iou = calculate_pixel_f1_iou(seg.flatten(), gt.flatten())

                    f1s[lab].append(f1)
                    ious[lab].append(iou)
                    precision[lab].append(precision)
                    recall[lab].append(recall)

                    # write to csv
                    if (args.subset is None):
                        row = [img_path, score, (score > args.threshold).astype(int), lab,
                               (score > args.threshold).astype(int) == lab]
                        writer.writerow(row)

            # image-level AUC

            y_true = (np.array(labs) > args.threshold).astype(int)
            y_pred = (np.array(scores) > args.threshold).astype(int)

            if (args.subset is None):
                save_path = os.path.join(args.out_dir, 'auc.png')
                save_auc(y_true, scores, save_path)
                try:
                    img_auc = roc_auc_score(y_true, scores)
                except:
                    print("only one class")
                    img_auc = 0.0
            else:
                img_auc = 0.0

            meanf1 = np.mean(f1s[0] + f1s[1])
            meaniou = np.mean(ious[0] + ious[1])
            print("pixel-f1: %.4f" % meanf1)
            print("mean-iou: %.4f" % meaniou)

            acc, sen, spe, f1_imglevel, tp, tn, fp, fn = calculate_img_score(y_pred, y_true)
            print("img level acc: %.4f sen: %.4f  spe: %.4f  f1: %.4f au: %.4f"
                  % (acc, sen, spe, f1_imglevel, img_auc))
            print("combine f1: %.4f" % (2 * meanf1 * f1_imglevel / (f1_imglevel + meanf1 + 1e-6)))

            # print('~~~~~~~~~~~~~~~~~~')
            # print('|Dataset : ', args.paths_file, ' |')
            # print('|F1  is %5f |' % np.mean(np.array(f1_scores)))
            # print('|AUC is %5f |' % np.mean(np.array(aucs)))
            # print('~~~~~~~~~~~~~~~~~~')
            #
            print('~~~~~~~~~~~~~~~end~~~~~~~~~~~~~~~~~~~~')
            txt_file.write(checkpoint + '|'+ f'{meanf1}|{meaniou}|{acc}|{sen}|{spe}|{f1_imglevel}|{img_auc}|{(2 * meanf1 * f1_imglevel / (f1_imglevel + meanf1 + 1e-6))}' + '\n')

            # # confusion matrix
            # save_path = os.path.join(args.out_dir, 'cm' + ('_' + args.subset if args.subset else '') + '.png')
            # save_cm(y_true, y_pred, save_path)

            if (args.subset is None): f_csv.close()

    txt_file.close()
    total_time_end = time.time()  # 记录结束时间
    total_time_sum = total_time_end - total_time_start
    print(f'总耗时：{total_time_sum}秒')

