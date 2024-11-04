import argparse
import os
import sys
sys.path.append('./models')
# sys.path.append('./datasets')
# sys.path.append('./datasets/GC')
# sys.path.append('./datasets/segment_anything')
from datetime import datetime
import numpy as np
import random
import torch
# torch.multiprocessing.set_sharing_strategy('file_system')
torch.multiprocessing.set_start_method('spawn', force=True)
import torch.nn as nn
import gc
import time
from torch.autograd import Variable

from torch.utils.data import DataLoader

# tensorboard
from torch.utils.tensorboard import SummaryWriter

from models.mvss_res2net import get_mvss_res2net, get_mvss
from datasets.dataset_ges import *
from focal_loss import MultiCEFocalLoss
import math
from dice_loss import DiceLoss, BinaryDiceLoss
import torchvision.transforms as transforms
from seg_utils.calculate_weights import calculate_weigths_labels
from mypath import Path
from loss import SegmentationLosses
from torchvision.ops.focal_loss import sigmoid_focal_loss
from sklearn.metrics import roc_auc_score, roc_curve, ConfusionMatrixDisplay, precision_recall_curve
from datasets.GC import gc_inpaint
from tqdm import tqdm
from models.OSN.OSN_models import OSN_Model, build_osn_model, add_osn_noise
from datasets.Harmonizer.harmonize_predict import Harmonizer
from datasets.Harmonizer.enhance_predict import Enhancer

device_str = 'cuda:1'
device = torch.device(device_str)
device_OSN = torch.device('cuda:0')

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
    return acc, sen, spe, f1, true_pos, true_neg, false_pos, false_neg, f1_ori


def calculate_pixel_f1_iou(pd, gt):
    # both the predition and groundtruth are empty
    if np.max(pd) == np.max(gt) and np.max(pd) == 0:
        return 1.0, 0.0, 0.0, 1.0, 1.0
    seg_inv, gt_inv = np.logical_not(pd), np.logical_not(gt)
    true_pos = float(np.logical_and(pd, gt).sum())
    true_neg = float(np.logical_and(seg_inv, gt_inv).sum())
    false_pos = np.logical_and(pd, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, gt).sum()
    f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
    precision = true_pos / (true_pos + false_pos + 1e-6)
    recall = true_pos / (true_pos + false_neg + 1e-6)
    cross = np.logical_and(pd, gt)
    union = np.logical_or(pd, gt)
    iou = np.sum(cross) / (np.sum(union) + 1e-6)
    mcc = (true_pos * true_neg - false_pos * false_neg) / (np.sqrt(
        (true_pos + false_pos) * (true_pos + false_neg) * (true_neg + false_pos) * (true_neg + false_neg)) + 1e-6)

    return f1, precision, recall, iou, mcc


class WBCEWithLogitLoss(nn.Module):
    """
    Weighted Binary Cross Entropy.
    `WBCE(p,t)=-β*t*log(p)-(1-t)*log(1-p)`
    To decrease the number of false negatives, set β>1.
    To decrease the number of false positives, set β<1.
    Args:
            @param weight: positive sample weight
        Shapes：
            output: A tensor of shape [N, 1,(d,), h, w] without sigmoid activation function applied
            target: A tensor of shape same with output
    """

    def __init__(self, weight=1.0, ignore_index=None, reduction='mean'):
        super(WBCEWithLogitLoss, self).__init__()
        assert reduction in ['none', 'mean', 'sum']
        self.ignore_index = ignore_index
        weight = float(weight)
        self.weight = weight
        self.reduction = reduction
        self.smooth = 0.01

    def forward(self, output, target):
        assert output.shape[0] == target.shape[0], "output & target batch size don't match"

        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()
            output = output.mul(valid_mask)  # can not use inplace for bp
            target = target.float().mul(valid_mask)

        batch_size = output.size(0)
        output = output.view(batch_size, -1)
        target = target.view(batch_size, -1)

        output = torch.sigmoid(output)
        # avoid `nan` loss
        eps = 1e-6
        output = torch.clamp(output, min=eps, max=1.0 - eps)
        # soft label
        target = torch.clamp(target, min=self.smooth, max=1.0 - self.smooth)

        # loss = self.bce(output, target)
        loss = -self.weight * target.mul(torch.log(output)) - ((1.0 - target).mul(torch.log(1.0 - output)))
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'none':
            loss = loss
        else:
            raise NotImplementedError
        return loss


class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        # pt = torch.sigmoid(predict)
        pt = predict
        # pt = predict.view(-1)
        # target = target.view(-1)
        loss = - ((1 - self.alpha) * ((1 - pt + 1e-5) ** self.gamma) * (target * torch.log(pt + 1e-5)) + self.alpha * (
                (pt + +1e-5) ** self.gamma) * ((1 - target) * torch.log(1 - pt + 1e-5)))

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


class FocalLossV1(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        '''
        Usage is same as nn.BCEWithLogits:
            >>> criteria = FocalLossV1()
            >>> logits = torch.randn(8, 19, 384, 384)
            >>> lbs = torch.randint(0, 2, (8, 19, 384, 384)).float()
            >>> loss = criteria(logits, lbs)
        '''
        probs = torch.sigmoid(logits)
        coeff = torch.abs(label - probs).pow(self.gamma).neg()
        log_probs = torch.where(logits >= 0,
                F.softplus(logits, -1, 50),
                logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0,
                -logits + F.softplus(logits, -1, 50),
                -F.softplus(logits, 1, 50))
        loss = label * self.alpha * log_probs + (1. - label) * (1. - self.alpha) * log_1_probs
        loss = loss * coeff

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


# for dice loss
def dice_loss(out, gt, smooth=1.0):
    gt = gt.view(-1)
    out = out.view(-1)

    intersection = (gt * out).sum()
    dice = (2.0 * intersection + smooth) / (torch.square(gt).sum() + torch.square(out).sum() + smooth)
    # TODO: need to confirm this matches what the paper says, and also the calculation/result is correct

    return 1.0 - dice


# for multiprocessing
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


# for removing damaged images
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def parse_args():
    parser = argparse.ArgumentParser()

    ## job
    parser.add_argument("--id", type=int, help="unique ID from Slurm")
    parser.add_argument("--run_name", type=str, default="MVSS-Net", help="run name")

    ## multiprocessing
    parser.add_argument('--dist_backend', default='nccl', choices=['gloo', 'nccl'], help='multiprocessing backend')
    parser.add_argument('--master_addr', type=str, default="127.0.0.3", help='address')
    parser.add_argument('--master_port', type=int, default=3721, help='address')
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')

    ## dataset
    parser.add_argument("--dataset", type=str, help="train dataset name, 保存和读取计算的class weights")
    parser.add_argument("--use_balanced_weights", type=bool, default=False,
                        help="use_balanced_weights or not,default false")
    parser.add_argument("--paths_file", type=str, default="/dataset/files.txt",
                        help="path to the file with input paths")  # each line of this file should contain "/path/to/image.ext /path/to/mask.ext /path/to/edge.ext 1 (for fake)/0 (for real)"; for real image.ext, set /path/to/mask.ext and /path/to/edge.ext as a string None
    parser.add_argument("--val_paths_file", type=str, help="path to the validation set")
    parser.add_argument("--n_c_samples", type=int, help="samples per classes (None for non-controlled)")
    parser.add_argument("--val_n_c_samples", type=int,
                        help="samples per classes for validation set (None for non-controlled)")

    parser.add_argument("--workers", type=int, default=8, help="number of cpu threads to use during batch generation")

    parser.add_argument("--image_size", type=int, default=512, help="size of the images")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")

    parser.add_argument("--batch_size", type=int, default=16,
                        help="size of the batches")  # no default value given by paper
    parser.add_argument("--manipulate_dataset", action="store_true", help="manipulate dataset or not")
    parser.add_argument("--postprocess_dataset", action="store_true", help="postprocess dataset or not")
    parser.add_argument("--osn_noise", action="store_true", help="add osn noise or not")
    parser.add_argument("--sample_mode", type=str, default='max', help="调整采样方式, max为与数量最多的类别的数量保持一致即过采样少的类别；只用CasiaV2训练时，使用max，否则使用min")
    parser.add_argument("--training_dataset", type=str, default='Casiav2', help="使用的训练数据集,Casiav2或Catnet")

    parser.add_argument("--manipulate_thresh", type=int, default=-1, help="manipulate_thresh, 进行篡改操作的阈值, 0-100")
    parser.add_argument("--cit_thresh", type=float, default=-1., help="cit_thresh, 进行随机颜色光照纹理变化的阈值, 0-1")
    parser.add_argument("--poison_thresh", type=float, default=-1., help="poison_thresh, 进行泊松融合的阈值, 0-1")
    parser.add_argument("--blend_thresh", type=int, default=-1, help="blend_thresh, 进行调和的阈值, 0-100")


    ## model
    parser.add_argument('--load_path', type=str, help='pretrained model or checkpoint for continued training')
    parser.add_argument('--model_select', choices=['mvss', 'mvss_res2net'], default='mvss', help='pretrained model or checkpoint for continued training')

    ## optimizer and scheduler
    parser.add_argument("--optim", choices=['adam', 'sgd'], default='adam', help="optimizer")
    parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--momentum", type=float, default=0.9, help="sgd: momentum of gradient")

    parser.add_argument('--patience', type=int, default=5,
                        help='numbers of epochs to decay for ReduceLROnPlateau scheduler (None to disable)')

    parser.add_argument('--decay_epoch', type=int,
                        help='numbers of epochs to decay for StepLR scheduler (low priority, None to disable)')

    ## training
    parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")

    parser.add_argument("--lr_end", type=float, default=1e-7, help="adam: learning rate")

    parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")

    parser.add_argument("--cond_epoch", type=int, default=0, help="epoch to start training from")

    parser.add_argument("--n_early", type=int, default=10, help="number of epochs for early stopping")

    parser.add_argument("--fix_lamda", type=float, default=-1, help="number of epochs for early stopping")

    parser.add_argument("--fix_seed", action="store_true", help="set to fix seed and use deterministic")

    parser.add_argument("--rand_seed", type=int, default=20230506, help="random seed")

    ## losses 已弃用，在下方直接设置
    parser.add_argument("--lambda_seg", type=float, default=0.16, help="pixel-scale loss weight (alpha)")
    parser.add_argument("--lambda_clf", type=float, default=0.04, help="image-scale loss weight (beta)")

    ## log
    parser.add_argument("--log_interval", type=int, default=100, help="interval between saving image samples")
    parser.add_argument("--checkpoint_interval", type=int, default=2000,
                        help="batch interval between model checkpoints")
    parser.add_argument("--print_interval", type=int, default=10, help="interval print loss")
    parser.add_argument("--warmup_epoch", type=int, default=1, help="number of epochs for warmup")
    args = parser.parse_args()

    return args


def init_env(args, local_rank, global_rank):
    # for debug only
    # torch.autograd.set_detect_anomaly(True)
    if args.load_path != None and args.load_path != 'timm':
        model_state_file = os.path.join(args.load_path,
                                        'last_checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            args.id = args.load_path.split('/')[-1].split('_')[0]

    if (args.id is None):
        args.id = datetime.now().strftime("%Y%m%d%H%M%S")

    # torch.cuda.set_device(local_rank)
    setup_for_distributed(global_rank == 0)

    # finalizing args, print here
    print(args)

    return args


def init_models(args):
    
    if args.model_select == 'mvss':
        print('init model:mvss')
        model = get_mvss(backbone='resnet50',
                            pretrained_base=True,
                            nclass=1,
                            sobel=True,
                            constrain=True,
                            n_input=3,
                            ).to('cuda:1')
    elif args.model_select == 'mvss_res2net':
        print('init model:mvss_res2net')
        model = get_mvss_res2net(backbone='resnet50',
                            pretrained_base=True,
                            nclass=1,
                            sobel=True,
                            constrain=True,
                            n_input=3,
                            ).to('cuda:1')
    else:
        raise NotImplementedError(f'model: {args.model_select} not implemented')
    # model_gc = load_model()
    # model_osn = load_osnModel()
    return model


def init_assist_models():
    print('init model:gc')
    model_gc = load_model()
    print('init model:osn')
    model_osn = load_osnModel()
    print('init model:harmonizer')
    model_harmonizer = Harmonizer()
    print('init model:enhancer')
    model_enhancer = Enhancer()
    return model_gc, model_osn, model_harmonizer, model_enhancer


def load_model():
    # global model_gc
    model_gc = gc_inpaint()
    # model_gc = None
    return model_gc


def load_osnModel():
    # 初始化OSN模型
    OSN = OSN_Model()
    # OSN = build_osn_model()
    # OSN = None
    return OSN


def init_dataset(args, global_rank, world_size, model_gc, model_osn, model_harmonizer, model_enhancer, val=False):
    # return None if no validation set provided
    if (val and args.val_paths_file is None):
        print('No val set!')
        return None, None
    C2_data = '/home/u202220085400001/pan1/DATASETS/Casiav2/Casiav2_with_edge_230322.txt'    # C2
    C2_cm_data = '/home/u202220085400001/pan1/DATASETS/Casiav2/Casiav2_copymove_with_edge_230322.txt'  # C2 copymove
    C2_sp_data = '/home/u202220085400001/pan1/DATASETS/Casiav2/Casiav2_splice_with_edge_230322.txt'  # C2 splice
    C3_data = '/home/u202220085400001/pan1/DATASETS/Coverage/Coverage_with_edge_230322.txt'  # C3
    C4_data = '/home/u202220085400001/pan1/DATASETS/Columbia/Columbia_with_edge_230322.txt'  # C4
    N_data = '/home/u202220085400001/pan1/DATASETS/NIST16/NIST16_with_edge_230322.txt'  # N
    GC_data = '/home/u202220085400001/pan1/DATASETS/IIDNET_DATASET/DiverseInpaintingDataset/IIDGC_with_edge_230322.txt'  # G
    DEF84Ktrain = '/home/u202220085400001/pan1/DATASETS/DEF84K/DEF84k_with_edge_train_230322.txt'
    DEF84Kval = '/home/u202220085400001/pan1/DATASETS/DEF84K/DEF84k_with_edge_val_230322.txt'
    SP_COCO = '/home/u202220085400001/pan2/DATASETS/CAT-Net_datasets/tampCOCO/sp_COCO_list_withEdge.txt'
    CM_COCO = '/home/u202220085400001/pan2/DATASETS/CAT-Net_datasets/tampCOCO/cm_COCO_list_withEdge.txt'
    CM_RAISE = '/home/u202220085400001/pan2/DATASETS/CAT-Net_datasets/tampCOCO/bcm_COCO_list_withEdge.txt'
    CMC_RAISE = '/home/u202220085400001/pan2/DATASETS/CAT-Net_datasets/tampCOCO/bcmc_COCO_list_withEdge.txt'
    compRAISE = '/home/u202220085400001/pan2/DATASETS/RAISE/compRAISE_au_230906.txt'
    rawRAISE = '/home/u202220085400001/pan2/DATASETS/RAISE/rawRAISE_au_230906.txt'
    # target data information
    val10_200_data = '/home/u202220085400001/pan1/DATASETS/val10_200_datasets.txt'
    val6_200_data = '/home/u202220085400001/pan1/DATASETS/val6_200_datasets.txt'
    val6_01_data = '/home/u202220085400001/pan1/DATASETS/val6_01_datasets.txt'
    if args.training_dataset.lower() == 'casiav2':
        src_paths_file_list = [
            C2_data,
            # SP_COCO,
            # CM_COCO,
            # CM_RAISE,
            # CMC_RAISE,
            # compRAISE,
            # rawRAISE,

            # C2_sp_data,
            # src2_data,
            # src3_data,
        ]
    elif args.training_dataset.lower() == 'catnet':
        src_paths_file_list = [
            C2_data,
            SP_COCO,
            CM_COCO,
            CM_RAISE,
            CMC_RAISE,
            compRAISE,
            rawRAISE,

            # C2_sp_data,
            # src2_data,
            # src3_data,
        ]
    else:
        print('training dataset not specified!')
        raise RuntimeError('training dataset not specified!')
    tgt_paths_file_list = [
        val10_200_data,
        # val6_200_data,
        # val6_01_data,
    ]

    # model_gc = load_model()
    # model_osn = load_osnModel()

    dataset = DeepfakeDataset(args,
                              (src_paths_file_list if not val else tgt_paths_file_list),
                              args.image_size,
                              args.id,
                              model_gc if not val else None,
                              model_osn if not val else None,
                              model_harmonizer if not val else None,
                              model_enhancer if not val else None,
                              (args.n_c_samples if not val else args.val_n_c_samples),
                              val,
                              manipulate_dataset=args.manipulate_dataset if not val else False,
                              postprocess_dataset=args.postprocess_dataset if not val else False,
                              osn_noise=args.osn_noise if not val else False,
                              sample_mode=args.sample_mode if not val else 'max',  # max为与数量最多的类别的数量保持一致即过采样少的类别；只用CasiaV2训练时，使用max，否则使用min
                              )

    if (not val):
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=global_rank,
                                                                  shuffle=True)
    else:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=global_rank,
                                                                  shuffle=False)

    local_batch_size = args.batch_size // world_size

    if (not val):
        print('Local batch size is {} ({}//{})!'.format(local_batch_size, args.batch_size, world_size))

    if (not val):
        dataloader = DataLoader(dataset=dataset, batch_size=local_batch_size, num_workers=args.workers,
                                # pin_memory=True,
                                drop_last=True, sampler=sampler, collate_fn=collate_fn)
        print('{} set size is {}!'.format(('Train' if not val else 'Val'), len(dataloader) * args.batch_size))
    else:
        dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=args.workers,
                                # pin_memory=True,
                                drop_last=False, sampler=sampler, collate_fn=collate_fn)
        print('{} set size is {}!'.format(('Train' if not val else 'Val'), len(dataloader)))


    return sampler, dataloader, dataset


def init_optims(args, world_size,
                model):
    # Optimizers
    local_lr = args.lr / world_size

    print('Local learning rate is {} ({}/{})!'.format(local_lr, args.lr, world_size))

    if (args.optim == 'adam'):
        print("Using optimizer adam")
        optimizer = torch.optim.Adam(model.parameters(), lr=local_lr, betas=(args.b1, args.b2))
    elif (args.optim == 'sgd'):
        print("Using optimizer sgd")
        optimizer = torch.optim.SGD(model.parameters(), lr=local_lr, momentum=args.momentum)
    else:
        print("Unrecognized optimizer %s" % args.optim)
        sys.exit()

    return optimizer


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-7):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor
    # def f(x):  # 热重启余弦衰减
    #     """
    #     根据step数返回一个学习率倍率因子，
    #     注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
    #     """
    #     if warmup is True and x <= (warmup_epochs * num_step):
    #         alpha = float(x) / (warmup_epochs * num_step)
    #         # warmup过程中lr倍率因子从warmup_factor -> 1
    #         return warmup_factor * (1 - alpha) + alpha
    #     else:
    #
    #         current_step = (x - warmup_epochs * num_step) % (num_step * epochs)
    #
    #         limit_factor = 0.5 ** (x // (num_step * epochs))
    #         cosine_steps = (epochs - warmup_epochs) * num_step
    #         # warmup后lr倍率因子从1 -> end_factor
    #         return (((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (
    #                     1 - end_factor) + end_factor) * limit_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)



def init_schedulers(args, optimizer, step1batch):
    lr_scheduler = None

    # high priority for ReduceLROnPlateau (validation set required)
    if (args.val_paths_file and args.patience):
        print("Using scheduler ReduceLROnPlateau")
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                                  factor=0.1,
                                                                  patience=args.patience)

    # low priority StepLR
    elif (args.decay_epoch):
        print("Using scheduler StepLR")
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                       step_size=args.decay_epoch,
                                                       gamma=0.5)
    elif (args.warmup_epoch):
        print("Using scheduler LambdaLR")
        lr_scheduler = create_lr_scheduler(optimizer=optimizer,
                                           num_step=(step1batch + 1),
                                           epochs=args.n_epochs,  # 若设为10,args.n_epochs/10=5次热重启
                                           warmup=True,
                                           warmup_epochs=args.warmup_epoch,
                                           warmup_factor=args.lr_end / args.lr,
                                           end_factor=args.lr_end / args.lr)

    else:
        print("No scheduler used")

    return lr_scheduler


# for saving checkpoints
def save_checkpoints(checkpoint_dir, id, epoch, step, get_module,
                     model):
    if (get_module):
        net = model.module
    else:
        net = model

    torch.save(net.state_dict(),
               os.path.join(checkpoint_dir, str(id) + "_" + str(epoch) + '_' + str(step) + '.pth'))


# a single step of prediction and loss calculation (same for both training and validating)
def predict_loss(args, data, model, val_flag,
                 criterion_CE, lamda, weight_mask, weight_edge,
                 Dice_loss_mask, Dice_loss_edge, sigfocal_loss_mask, sigfocal_loss_edge, global_xi, osn_xi_flag=False):
    # load data
    in_imgs, in_masks, in_edges, in_labels = data

    in_imgs = in_imgs.to(device_str, non_blocking=True, dtype=torch.float32)
    in_masks = in_masks.to(device_str, non_blocking=True, dtype=torch.float32)
    in_edges = in_edges.to(device_str, non_blocking=True, dtype=torch.float32)
    in_labels = in_labels.to(device_str, non_blocking=True, dtype=torch.float32)

    if 'det' not in args.model_select:
        in_labels = in_labels.squeeze()

    # in_masks = in_masks.permute(0, 3, 1, 2)
    # in_edges = in_edges.permute(0, 3, 1, 2)

    if not val_flag and osn_xi_flag:
        # 加入模拟OSN噪声
        xi = sampling_xi(in_imgs.size(0), global_xi)
        in_imgs = in_imgs + xi
        in_imgs.clamp_(-1, 1)
    else:
        xi = None


    out_edges, out_masks, out_cls = model(in_imgs)

    out_labels = out_cls

    loss_clf = criterion_CE(out_labels, in_labels.long())
    cls_balance = torch.ones_like(in_labels.to(torch.float))
    if not val_flag:
        if (in_labels == 1).sum():
            cls_balance[in_labels == 1] = 0.5 / ((in_labels == 1).sum().to(torch.float) / in_labels.numel())
            cls_balance[in_labels == 0] = 0.5 / ((in_labels == 0).sum().to(torch.float) / in_labels.numel())
        else:
            print('cls balance is not working!')
    cls_balance = cls_balance.to(device_str)
    loss_clf = torch.mean(loss_clf * cls_balance)

    # Pixel-scale loss
    out_masks_beforesig = out_masks
    out_masks = torch.sigmoid(out_masks)  # FocalLossV1、BCEWithLogitLoss里有sigmoid，外面就不过了
    loss_seg1 = Dice_loss_mask(out_masks, in_masks)  # dice loss
    # loss_seg = PPA_loss_mask(out_masks_beforesig, in_masks, kernel_size=31)  # ppa loss
    loss_seg2 = sigfocal_loss_mask(out_masks_beforesig, in_masks)  # sig_focal loss
    loss_seg = loss_seg1 + loss_seg2*10.
    out_masks = out_masks.squeeze()

    # Edge loss
    # TODO: is it the same as the paper?
    out_edges_beforesig = out_edges
    out_edges = torch.sigmoid(out_edges)  # FocalLossV1、BCEWithLogitLoss里有sigmoid，外面就不过了
    loss_edg1 = Dice_loss_edge(out_edges, in_edges)  # dice loss
    # loss_edg = PPA_loss_edge(out_edges_beforesig, in_edges, kernel_size=9)  # ppa loss
    loss_edg2 = sigfocal_loss_edge(out_edges_beforesig, in_edges)  # sig_focal loss
    loss_edg = loss_edg1 + loss_edg2*10.
    out_edges = out_edges.squeeze()
    in_masks = in_masks.squeeze()
    in_edges = in_edges.squeeze()

    if not val_flag:
        if args.print_interval == 20:
            print(
                'loss_clf:%.6f; loss_edg_dice:%.6f; loss_edg_focal:%.6f; loss_seg_dice:%.6f; loss_seg_focal:%.6f' %(loss_clf.cpu(), loss_edg1.cpu(), loss_edg2.cpu(), loss_seg1.cpu(), loss_seg2.cpu()))
        if args.print_interval > 20:
            args.print_interval = 0
        args.print_interval = args.print_interval + 1

    # Total loss
    alpha = args.lambda_seg
    beta = args.lambda_clf

    weighted_loss_seg = 0.16 * loss_seg
    weighted_loss_clf = 0.04 * loss_clf
    weighted_loss_edg = 0.8 * loss_edg
    # print('seg:', weighted_loss_seg)
    # print('clf:', weighted_loss_clf)
    # print('edg:', weighted_loss_edg)

    loss = weighted_loss_seg + weighted_loss_clf + weighted_loss_edg

    # tensorboard显示(NCHW)，扩充一个维度
    in_masks = in_masks.unsqueeze(dim=-1)
    # print('inmask:', in_masks.shape)
    in_edges = in_edges.unsqueeze(dim=-1)
    out_masks = out_masks.unsqueeze(dim=-1)
    out_edges = out_edges.unsqueeze(dim=-1)

    return loss, weighted_loss_seg, weighted_loss_clf, weighted_loss_edg, in_imgs, in_masks, in_edges, in_labels, out_masks, out_edges, xi, out_labels


def img_trans_cat(input):
    if input.shape[3] == 1:
        input = input.permute(0, 3, 1, 2)
        temp = input
        output = torch.cat((input, temp), dim=1)
        output = torch.cat((output, temp), dim=1)
    elif input.shape[1] == 1:
        temp = input
        output = torch.cat((input, temp), dim=1)
        output = torch.cat((output, temp), dim=1)
    else:
        output = input
        print('error1')
    return output


def sampling_xi(size, global_xi):
    if len(global_xi) < 5:
        idx = range(len(global_xi))
    else:
        idx = np.random.choice(range(len(global_xi)), 5, replace=False)
    rtn = global_xi[idx[0]][0:size]
    rtn.clamp_(-4 / 255, 4 / 255)
    for i in idx[1:]:
        rtn[0:size] += 4 / 255 * global_xi[i][0:size]
        rtn.clamp_(-4 / 255, 4 / 255)
    rtn = Variable(rtn.to('cuda:1'), requires_grad=True)
    return rtn


def train(args, global_rank, sync, get_module,
          model,
          train_sampler, dataloader, train_dataset, val_sampler, val_dataloader,
          optimizer,
          lr_scheduler):

    # # 初始化OSN模型
    # OSN = OSN_Model(batch_size=args.batch_size).to(device_str)

    # whether to use class balanced weights
    if args.use_balanced_weights:
        classes_weights_path_mask = os.path.join(Path.db_root_dir(args.training_dataset),
                                                 args.training_dataset + '_classes_weights_mask.npy')
        classes_weights_path_edge = os.path.join(Path.db_root_dir(args.training_dataset),
                                                 args.training_dataset + '_classes_weights_edge.npy')
        if os.path.isfile(classes_weights_path_mask) and os.path.isfile(classes_weights_path_edge):
            weight_mask = np.load(classes_weights_path_mask)
            weight_edge = np.load(classes_weights_path_edge)
        else:
            weight_mask = calculate_weigths_labels(args.training_dataset, dataloader, num_classes=2, type='mask')
            weight_edge = calculate_weigths_labels(args.training_dataset, dataloader, num_classes=2, type='edge')
        weight_mask = torch.from_numpy(weight_mask.astype(np.float32))
        weight_edge = torch.from_numpy(weight_edge.astype(np.float32))
    else:
        weight_mask = None
        weight_edge = None
    print(f'classes weights mask:{weight_mask}')
    print(f'classes weights edge:{weight_edge}')

    # build Losses
    if 'det' in args.model_select:
        criterion_CE = nn.CrossEntropyLoss(reduction='none').cuda(device)
    else:
        criterion_CE = nn.BCEWithLogitsLoss(reduction='none').cuda(device)
    Dice_loss_mask = SegmentationLosses(weight=weight_mask.cuda(device), cuda=True).build_loss(
        mode='dice')  # mask dice loss
    Dice_loss_edge = SegmentationLosses(weight=weight_edge.cuda(device), cuda=True).build_loss(
        mode='dice')  # edge dice loss
    sigfocal_loss_mask = SegmentationLosses(weight=weight_mask.cuda(device), cuda=True).build_loss(
        mode='balanced_sig_focal')  # mask dice loss
    sigfocal_loss_edge = SegmentationLosses(weight=weight_edge.cuda(device), cuda=True).build_loss(
        mode='balanced_sig_focal')  # edge dice loss

    # tensorboard展示图片去归一化权重
    px_mean = [0.0, 0.0, 0.0]
    px_std = [1.0, 1.0, 1.0]
    mean_list = []
    std_list = []
    for mean, std in zip(px_mean, px_std):
        mean_arr = np.ones((args.batch_size, 1, args.image_size, args.image_size)) * mean
        std_arr = np.ones((args.batch_size, 1, args.image_size, args.image_size)) * std
        if False:
            mean_list.append(torch.tensor(mean_arr, dtype=torch.float16))
            std_list.append(torch.tensor(std_arr, dtype=torch.float16))
        else:
            mean_list.append(torch.tensor(mean_arr, dtype=torch.float32))
            std_list.append(torch.tensor(std_arr, dtype=torch.float32))
    mean_stack = torch.stack(mean_list, dim=1).squeeze().to(device_str)
    std_stack = torch.stack(std_list, dim=1).squeeze().to(device_str)

    # tensorboard
    if global_rank == 0:
        os.makedirs("logs", exist_ok=True)
        writer = SummaryWriter("logs/" + str(args.id) + "_" + args.run_name)
        checkpoint_dir = "/home/u202220085400001/pan1/mvssnet_model/mvss_train_checkpoints/" + str(args.id) + "_" + args.run_name
        os.makedirs(checkpoint_dir, exist_ok=True)

    # for early stopping
    best_val_loss = float('inf')
    best_combine_f1 = float(0)
    n_last_epochs = 0
    early_stopping = False

    # resume
    if args.load_path != None:
        print('Load pretrained model: {}'.format(args.load_path))

        model_state_file = os.path.join(args.load_path,
                                        'last_checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file,
                                    map_location=lambda storage, loc: storage)
            best_combine_f1 = checkpoint['best_combine_f1']
            args.cond_epoch = checkpoint['epoch'] + 1  # 开始epoch
            if 'numpy_random_state' in checkpoint:
                np.random.set_state(checkpoint['numpy_random_state'])
                print("=> loaded numpy_random_state")
            if 'random_random_state' in checkpoint:
                random.setstate(checkpoint['random_random_state'])
                print("=> loaded random_random_state")
            if (not get_module):
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler']),
            print("=> loaded model checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
        else:
            print("No previous checkpoint.")

    for epoch in range(args.cond_epoch, args.n_epochs):

        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(3.0)
        train_dataset.sample_epoch()
        train_sampler.set_epoch(epoch)

        print('Starting Epoch {}'.format(epoch))
        lamda = 0.9975**(epoch * epoch)
        if args.fix_lamda >= 0:
            lamda = args.fix_lamda
        print(f'current lamda:{lamda}')
        # loss sum for epoch
        epoch_total_seg = 0
        epoch_total_clf = 0
        epoch_total_edg = 0

        epoch_total_model = 0

        epoch_val_loss = 0

        # number of steps in one epoch
        # can be replaced by len(dataloader), but kept as warm-up epochs may be added
        epoch_steps = 0
        val_flag = False

        try:
            type(global_xi)
            print("global_xi存在，删除")
            del global_xi
        except NameError:
            print("global_xi不存在")
        global_xi = [torch.zeros([args.batch_size, 3, args.image_size, args.image_size])]

        # ------------------
        #  Train step
        # ------------------
        for step, data in enumerate(dataloader):
            curr_steps = epoch * len(dataloader) + step

            model.train()

            if (sync): optimizer.synchronize()
            optimizer.zero_grad()
            # print(optimizer.param_groups[0]['lr'])

            loss, weighted_loss_seg, weighted_loss_clf, weighted_loss_edg, in_imgs, in_masks, in_edges, in_labels, out_masks, out_edges, xi, score = predict_loss(
                args, data, model, False, criterion_CE, lamda, weight_mask, weight_edge,
                Dice_loss_mask, Dice_loss_edge, sigfocal_loss_mask, sigfocal_loss_edge, global_xi)

            # backward prop
            loss.backward()

            if xi is not None:
                global_xi.append(torch.sign(xi.grad).detach().cpu())

            optimizer.step()

            if (lr_scheduler):
                if (args.warmup_epoch):
                    lr_scheduler.step()  # cos decay

            # log losses for epoch
            epoch_steps += 1

            epoch_total_seg += weighted_loss_seg.item()
            epoch_total_clf += weighted_loss_clf.item()
            epoch_total_edg += weighted_loss_edg.item()
            epoch_total_model += loss.item()

            # --------------
            #  Log Progress (for certain steps)
            # --------------
            if step % args.log_interval == 0 and global_rank == 0:
                print(f"[Epoch {epoch}/{args.n_epochs - 1}] [Batch {step}/{len(dataloader)}] "
                      f"[Total Loss {loss:.3f}]"
                      f"[Pixel-scale Loss {weighted_loss_seg:.3e}]"
                      f"[Edge Loss {weighted_loss_edg:.3e}]"
                      f"[Image-scale Loss {weighted_loss_clf:.3e}]"
                      f"")

                label_text = in_labels.cpu().detach().numpy()
                label_text = 'labels:' + f'{label_text}'
                writer.add_text("input_labels", label_text, curr_steps)
                writer.add_scalar("LearningRate", optimizer.param_groups[0]['lr'], curr_steps)
                writer.add_scalar("Loss/Total Loss", loss, epoch * len(dataloader) + step)
                writer.add_scalar("Loss/Pixel-scale", weighted_loss_seg, curr_steps)
                writer.add_scalar("Loss/Edge", weighted_loss_edg, curr_steps)
                writer.add_scalar("Loss/Image-scale", weighted_loss_clf, curr_steps)

                in_imgs_tb = in_imgs * std_stack + mean_stack  # 去归一化
                writer.add_images('Input Img', in_imgs_tb, epoch * len(dataloader) + step)

                # print(in_masks.shape, out_masks.shape, in_edges.shape, out_edges.shape)
                in_masks = img_trans_cat(in_masks)
                writer.add_images('Input Mask', in_masks, epoch * len(dataloader) + step)
                out_masks = img_trans_cat(out_masks)
                writer.add_images('Output Mask', out_masks, epoch * len(dataloader) + step)
                in_edges = img_trans_cat(in_edges)
                writer.add_images('Input Edge', in_edges, epoch * len(dataloader) + step)
                out_edges = img_trans_cat(out_edges)
                writer.add_images('Output Edge', out_edges, epoch * len(dataloader) + step)


        if epoch%10==0 or epoch==args.n_epochs-1 or epoch in [0]:
            val_flag = True
        if not val_flag:
            print('skip val')
        # ------------------
        #  Validation
        # ------------------
        if (args.val_paths_file and val_sampler and val_dataloader and val_flag):

            val_sampler.set_epoch(epoch)

            model.eval()
            # for storting results
            transform_pil = transforms.Compose([transforms.ToPILImage()])
            scores, labs = [], []
            f1s = [[], []]
            ious = [[], []]
            mccs = [[], []]
            f1_scores = []
            aucs = []
            for step, data in tqdm(enumerate(val_dataloader)):
                with torch.no_grad():
                    _, _, _, lab = data
                    loss, _, _, _, _, gt, _, _, seg, _, _, score = predict_loss(args, data, model, True, criterion_CE, lamda, weight_mask, weight_edge,
                                                                   Dice_loss_mask, Dice_loss_edge, sigfocal_loss_mask, sigfocal_loss_edge, global_xi)

                    epoch_val_loss += loss.item()
                    # calculate f1, iou ...
                    gt = gt.squeeze().cpu().numpy()

                    if 'det' in args.model_select:
                        sm = nn.Softmax(dim=1)
                        score = sm(score)
                        score = np.array(score.squeeze().detach().cpu())[1]
                    else:
                        score = np.array(torch.sigmoid(score).squeeze().detach().cpu())
                    seg = seg.squeeze().unsqueeze(dim=0).unsqueeze(dim=0).detach().cpu()

                    # resize to original
                    seg = [np.array(transform_pil(seg[i])) for i in range(len(seg))]
                    if len(seg) != 1:
                        print("seg size not 1")
                        continue
                    else:
                        seg = seg[0].astype(np.uint8)

                    seg = seg / 255.0

                    scores.append(score)
                    labs.append(lab)
                    f1 = 0

                    if seg.shape != gt.shape:
                        gt = cv2.resize(gt, (seg.shape[1], seg.shape[0]))
                        print("size not match, reshaped")

                    seg = (seg > 0.5).astype(np.float64)

                    # pixel-level F1
                    f1, _, _, iou, mcc = calculate_pixel_f1_iou(seg.flatten(), gt.flatten())

                    f1s[lab].append(f1)
                    ious[lab].append(iou)
                    mccs[lab].append(mcc)

            # image-level AUC

            y_true = (np.array(labs) > 0.5).astype(int)
            y_pred = (np.array(scores) > 0.5).astype(int)
            # print(y_pred)

            try:
                img_auc = roc_auc_score(y_true, scores)
            except:
                print("only one class")
                img_auc = 0.0


            meanf1 = np.mean(f1s[0] + f1s[1])
            meaniou = np.mean(ious[0] + ious[1])
            meanmcc = np.mean(mccs[0] + mccs[1])
            print("pixel level f1: %.4f  mean-iou: %.4f  mean-mcc: %.4f" % (meanf1, meaniou, meanmcc))


            acc, sen, spe, f1_imglevel, tp, tn, fp, fn, f1_pe_re = calculate_img_score(y_pred, y_true)
            print("img level acc: %.4f sen: %.4f  spe: %.4f  f1: %.4f au: %.4f  f1_pere: %.4f"
                  % (acc, sen, spe, f1_imglevel, img_auc, f1_pe_re))
            combine_f1 = (2 * meanf1 * f1_imglevel / (f1_imglevel + meanf1 + 1e-6))
            print("combine f1: %.4f" % (2 * meanf1 * f1_imglevel / (f1_imglevel + meanf1 + 1e-6)))
            print('~~~~~~~~~~~~~~~end~~~~~~~~~~~~~~~~~~~~')


            # early
            if combine_f1 >= best_combine_f1:
                best_combine_f1 = combine_f1
                n_last_epochs = 0
            # if epoch_val_loss <= best_val_loss:
            #     best_val_loss = epoch_val_loss
            #     n_last_epochs = 0
            else:
                n_last_epochs += 1

                if (n_last_epochs >= args.n_early):
                    early_stopping = True

        # ------------------
        #  Step
        # ------------------
        if (lr_scheduler):
            if (args.val_paths_file and args.patience):
                lr_scheduler.step(epoch_val_loss)  # ReduceLROnPlateau
            elif (args.decay_epoch):
                lr_scheduler.step()  # StepLR
            elif (args.warmup_epoch):
                lr_scheduler.step()  # cos decay
            else:
                print("Error in scheduler step")
                sys.exit()

        # --------------
        #  Log Progress (for epoch)
        # --------------
        # loss average for epoch
        if (epoch_steps != 0 and global_rank == 0 and val_flag):
            epoch_avg_seg = epoch_total_seg / epoch_steps
            epoch_avg_edg = epoch_total_edg / epoch_steps
            epoch_avg_clf = epoch_total_clf / epoch_steps
            epoch_avg_model = epoch_total_model / epoch_steps

            if (args.val_paths_file):
                epoch_val_loss_avg = epoch_val_loss / len(val_dataloader)
                best_val_loss_avg = best_val_loss / len(val_dataloader)
            else:
                epoch_val_loss_avg = 0
                best_val_loss_avg = 0

            print(f"[Epoch {epoch}/{args.n_epochs - 1}]"
                  f"[Epoch Total Loss {epoch_avg_model:.3f}]"
                  f"[Epoch Pixel-scale Loss {epoch_avg_seg:.3e}]"
                  f"[Epoch Edge Loss {epoch_avg_edg:.3e}]"
                  f"[Epoch Image-scale Loss {epoch_avg_clf:.3e}]"
                  f"[Epoch Val Loss {epoch_val_loss_avg:.3f} (best Val Loss {best_val_loss_avg:.3f} last for {n_last_epochs:d})]"
                  f"")

            writer.add_scalar("Epoch LearningRate", optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar("Epoch Loss/Total Loss", epoch_avg_model, epoch)
            writer.add_scalar("Epoch Loss/Pixel-scale", epoch_avg_seg, epoch)
            writer.add_scalar("Epoch Loss/Edge", epoch_avg_edg, epoch)
            writer.add_scalar("Epoch Loss/Image-scale", epoch_avg_clf, epoch)
            writer.add_scalar("Epoch Loss/Val", epoch_val_loss_avg, epoch)

            writer.add_images('Epoch Input Img', in_imgs_tb, epoch)

            # val metrics
            writer.add_scalar("Epoch val/pixel f1", meanf1, epoch)
            writer.add_scalar("Epoch val/mean iou", meaniou, epoch)
            writer.add_scalar("Epoch val/mean mcc", meanmcc, epoch)
            writer.add_scalar("Epoch val/image acc", acc, epoch)
            writer.add_scalar("Epoch val/image sen", sen, epoch)
            writer.add_scalar("Epoch val/image spe", spe, epoch)
            writer.add_scalar("Epoch val/image f1", f1_imglevel, epoch)
            writer.add_scalar("Epoch val/image f1_pere", f1_pe_re, epoch)
            writer.add_scalar("Epoch val/image auc", img_auc, epoch)
            writer.add_scalar("Epoch val/combine f1", combine_f1, epoch)

            # save model parameters
            if global_rank == 0:
                if combine_f1 >= best_combine_f1:
                    save_checkpoints(checkpoint_dir, args.id, 'val', 'best', get_module, model)


        # save model parameters
        if global_rank == 0:
            save_checkpoints(checkpoint_dir, args.id, epoch, 'end',  # set step to a string 'end'
                             get_module,
                             model)
            # 保存last-ckpt，方便resume
            torch.save({
                'epoch': epoch,
                'best_combine_f1': best_combine_f1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'numpy_random_state': np.random.get_state(),
                'random_random_state': random.getstate(),
            }, os.path.join(checkpoint_dir, 'last_checkpoint.pth.tar'))

        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(3.0)

        # check early_stopping
        if (early_stopping):
            print('Early stopping')
            break

    print('Finished training')

    if global_rank == 0:
        writer.close()

    pass