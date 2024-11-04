import os
import cv2
import torch
# import test_dataset
import gc_utils as utils
# import network
import argparse
import numpy as np
import re

from mmengine.config import Config
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def build_gc_model():
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    gc_parser = argparse.ArgumentParser()
    # General parameters
    gc_parser.add_argument('--results_path', type=str, default='./results',
                           help='testing samples path that is a folder')
    gc_parser.add_argument('--gan_type', type=str, default='WGAN', help='the type of GAN for training')
    gc_parser.add_argument('--gpu_ids', type=str, default="0", help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    gc_parser.add_argument('--cudnn_benchmark', type=bool, default=True, help='True for unchanged input data type')
    # Training parameters
    gc_parser.add_argument('--epoch', type=int, default=40, help='number of epochs of training')
    gc_parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
    gc_parser.add_argument('--num_workers', type=int, default=8,
                           help='number of cpu threads to use during batch generation')
    # Network parameters
    gc_parser.add_argument('--in_channels', type=int, default=4, help='input RGB image + 1 channel mask')
    gc_parser.add_argument('--out_channels', type=int, default=3, help='output RGB image')
    gc_parser.add_argument('--latent_channels', type=int, default=48, help='latent channels')
    gc_parser.add_argument('--pad_type', type=str, default='zero', help='the padding type')
    gc_parser.add_argument('--activation', type=str, default='elu', help='the activation type')
    gc_parser.add_argument('--norm', type=str, default='none', help='normalization type')
    gc_parser.add_argument('--init_type', type=str, default='xavier', help='the initialization type')
    gc_parser.add_argument('--init_gain', type=float, default=0.02, help='the initialization gain')
    # Dataset parameters
    gc_parser.add_argument('--baseroot', type=str, default='../../inpainting/dataset/Places/img_set')
    gc_parser.add_argument('--baseroot_mask', type=str, default='../../inpainting/dataset/Places/img_set')
    opt = gc_parser.parse_args()

    # Build networks
    generator = utils.create_generator(opt).eval()
    model_name = 'deepfillv2_WGAN_G_epoch40_batchsize4.pth'
    # model_name = os.path.join('pretrained_model', model_name)
    pretrained_dict = torch.load(model_name, map_location='cuda:0')
    generator.load_state_dict(pretrained_dict)
    return generator.to('cuda:0')

def gc_inpainting(img, mask, generator):
    # image
    # img = cv2.imread('./test_data/1.png')
    if mask.ndim >= 3:
        mask = mask[:, :, 0]
    # find the Minimum bounding rectangle in the mask
    '''
    contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cidx, cnt in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(cnt)
        mask[y:y+h, x:x+w] = 255
    '''
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).contiguous()
    mask = torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).contiguous()

    # inference
    img = img.to('cuda:0')
    mask = mask.to('cuda:0')

    # Generator output
    with torch.no_grad():
        first_out, second_out = generator(img, mask)

    # forward propagation
    first_out_wholeimg = img * (1 - mask) + first_out * mask  # in range [0, 1]
    second_out_wholeimg = img * (1 - mask) + second_out * mask  # in range [0, 1]

    # masked_img = img * (1 - mask) + mask
    # mask = torch.cat((mask, mask, mask), 1)
    # Recover normalization: * 255 because last layer is sigmoid activated
    second_out_wholeimg = second_out_wholeimg * 255
    # Process img_copy and do not destroy the data of img
    img_copy = second_out_wholeimg.clone().data.permute(0, 2, 3, 1)[0, :, :, :].cpu().numpy()
    img_copy = np.clip(img_copy, 0, 255)
    img_copy = img_copy.astype(np.uint8)
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
    return img_copy


def replace_cfg_vals(ori_cfg):
    """Replace the string "${key}" with the corresponding value.

    Replace the "${key}" with the value of ori_cfg.key in the config. And
    support replacing the chained ${key}. Such as, replace "${key0.key1}"
    with the value of cfg.key0.key1. Code is modified from `vars.py
    < https://github.com/microsoft/SoftTeacher/blob/main/ssod/utils/vars.py>`_  # noqa: E501

    Args:
        ori_cfg (mmcv.utils.config.Config):
            The origin config with "${key}" generated from a file.

    Returns:
        updated_cfg [mmcv.utils.config.Config]:
            The config with "${key}" replaced by the corresponding value.
    """

    def get_value(cfg, key):
        for k in key.split('.'):
            cfg = cfg[k]
        return cfg

    def replace_value(cfg):
        if isinstance(cfg, dict):
            return {key: replace_value(value) for key, value in cfg.items()}
        elif isinstance(cfg, list):
            return [replace_value(item) for item in cfg]
        elif isinstance(cfg, tuple):
            return tuple([replace_value(item) for item in cfg])
        elif isinstance(cfg, str):
            # the format of string cfg may be:
            # 1) "${key}", which will be replaced with cfg.key directly
            # 2) "xxx${key}xxx" or "xxx${key1}xxx${key2}xxx",
            # which will be replaced with the string of the cfg.key
            keys = pattern_key.findall(cfg)
            values = [get_value(ori_cfg, key[2:-1]) for key in keys]
            if len(keys) == 1 and keys[0] == cfg:
                # the format of string cfg is "${key}"
                cfg = values[0]
            else:
                for key, value in zip(keys, values):
                    # the format of string cfg is
                    # "xxx${key}xxx" or "xxx${key1}xxx${key2}xxx"
                    assert not isinstance(value, (dict, list, tuple)), \
                        f'for the format of string cfg is ' \
                        f"'xxxxx${key}xxxxx' or 'xxx${key}xxx${key}xxx', " \
                        f"the type of the value of '${key}' " \
                        f'can not be dict, list, or tuple' \
                        f'but you input {type(value)} in {cfg}'
                    cfg = cfg.replace(key, str(value))
            return cfg
        else:
            return cfg

    # the pattern of string "${key}"
    pattern_key = re.compile(r'\$\{[a-zA-Z\d_.]*\}')
    # the type of ori_cfg._cfg_dict is mmcv.utils.config.ConfigDict
    updated_cfg = Config(
        replace_value(ori_cfg._cfg_dict), filename=ori_cfg.filename)
    # replace the model with model_wrapper
    if updated_cfg.get('model_wrapper', None) is not None:
        updated_cfg.model = updated_cfg.model_wrapper
        updated_cfg.pop('model_wrapper')
    return updated_cfg


class gc_inpaint():
    def __init__(self):

        # ----------------------------------------
        #        Initialize the parameters
        # ----------------------------------------
        # gc_parser = argparse.ArgumentParser()
        # # General parameters
        # gc_parser.add_argument('--results_path', type=str, default='./results',
        #                        help='testing samples path that is a folder')
        # gc_parser.add_argument('--gan_type', type=str, default='WGAN', help='the type of GAN for training')
        # gc_parser.add_argument('--gpu_ids', type=str, default="0", help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        # gc_parser.add_argument('--cudnn_benchmark', type=bool, default=True, help='True for unchanged input data type')
        # # Training parameters
        # gc_parser.add_argument('--epoch', type=int, default=40, help='number of epochs of training')
        # gc_parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
        # gc_parser.add_argument('--num_workers', type=int, default=8,
        #                        help='number of cpu threads to use during batch generation')
        # # Network parameters
        # gc_parser.add_argument('--in_channels', type=int, default=4, help='input RGB image + 1 channel mask')
        # gc_parser.add_argument('--out_channels', type=int, default=3, help='output RGB image')
        # gc_parser.add_argument('--latent_channels', type=int, default=48, help='latent channels')
        # gc_parser.add_argument('--pad_type', type=str, default='zero', help='the padding type')
        # gc_parser.add_argument('--activation', type=str, default='elu', help='the activation type')
        # gc_parser.add_argument('--norm', type=str, default='none', help='normalization type')
        # gc_parser.add_argument('--init_type', type=str, default='xavier', help='the initialization type')
        # gc_parser.add_argument('--init_gain', type=float, default=0.02, help='the initialization gain')
        # # Dataset parameters
        # gc_parser.add_argument('--baseroot', type=str, default='../../inpainting/dataset/Places/img_set')
        # gc_parser.add_argument('--baseroot_mask', type=str, default='../../inpainting/dataset/Places/img_set')
        # opt_gc = gc_parser.parse_args()
        opt_gc = Config.fromfile('/home/zzt/Python_Projects/MVSS-Net/MVSS-Net-train/datasets/GC/gc_config.py')
        opt_gc = replace_cfg_vals(opt_gc)
        opt_gc = opt_gc.model

        # Build networks
        self.generator = utils.create_generator(opt_gc).eval()
        model_name = 'deepfillv2_WGAN_G_epoch40_batchsize4.pth'
        model_name = os.path.join('./datasets/GC', model_name)
        pretrained_dict = torch.load(model_name, map_location='cpu')
        self.generator.load_state_dict(pretrained_dict)
        self.generator.to('cuda:0')

    def gc_inpaint_one(self, img, mask):
        # image
        # img = cv2.imread('./test_data/1.png')
        if mask.ndim >= 3:
            mask = mask[:, :, 0]


        # find the Minimum bounding rectangle in the mask
        '''
        contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cidx, cnt in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(cnt)
            mask[y:y+h, x:x+w] = 255
        '''
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).contiguous()
        mask = torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).contiguous()

        # inference
        img = img.to('cuda:0')
        mask = mask.to('cuda:0')

        # Generator output
        with torch.no_grad():
            first_out, second_out = self.generator(img, mask)

        # forward propagation
        first_out_wholeimg = img * (1 - mask) + first_out * mask  # in range [0, 1]
        second_out_wholeimg = img * (1 - mask) + second_out * mask  # in range [0, 1]

        # masked_img = img * (1 - mask) + mask
        # mask = torch.cat((mask, mask, mask), 1)
        # Recover normalization: * 255 because last layer is sigmoid activated
        second_out_wholeimg = second_out_wholeimg * 255
        # Process img_copy and do not destroy the data of img
        img_copy = second_out_wholeimg.clone().data.permute(0, 2, 3, 1)[0, :, :, :].cpu().numpy()
        img_copy = np.clip(img_copy, 0, 255)
        img_copy = img_copy.astype(np.uint8)
        # img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
        return img_copy



if __name__ == '__main__':
    img = cv2.imread('/home/zzt/pan1/Codes/DeepFillv2_Pytorch/test_data/2.png')
    mask = cv2.imread('/home/zzt/pan1/Codes/DeepFillv2_Pytorch/test_data_mask/2.png')
    # gc_model = build_gc_model()
    # img_inpainted = gc_inpainting(img, mask, gc_model)
    gc_inpaint_model = gc_inpaint()
    img_inpainted = gc_inpaint_model.gc_inpaint_one(img, mask)
    print('')
