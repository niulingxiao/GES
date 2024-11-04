import os.path
import random
import numpy as np
import cv2
import traceback
# import sys
# sys.path.append('..')
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms as T
# from torchvision.transforms import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
# from gen_copymove_inpainting_mask import gen_mask_imginput
# from make_copymovev7 import gen_cp_ip, make_copymove_tamp, make_splice_tamp, gen_cp_ip_imginput, make_copymove_tamp_real
from PIL import Image
import copy
import gc
import math
from tqdm import tqdm
import json
from segment_anything.utils import rle_to_mask
from RandAugment import RandAugment
# 与v7区别：篡改方式是随机选择，不是顺序概率

def gen_mask_imginput(image):

    # load the original input image and display it to our screen
    # image = cv2.imread(image_path)

    h = image.shape[0]
    w = image.shape[1]
    length = np.random.randint((w // 100 * 5)+1, (w // 100 * 75)+1)  # 5%-75%的width
    # print(length)
    mask = np.zeros((h, w), dtype="uint8")

    y = np.random.randint(0, h)
    x = np.random.randint(0, w)

    y1 = np.clip(y - length // 2, 0, h)
    y2 = np.clip(y + length // 2, 0, h)
    x1 = np.clip(x - length // 2, 0, w)
    x2 = np.clip(x + length // 2, 0, w)

    mask[y1: y2, x1: x2] = 255

    # mask = np.zeros(image.shape[:2], dtype="uint8")
    # cv2.rectangle(mask, (0, 0), (20, 20), 255, -1)
    # cv2.imshow("Rectangular Mask", mask)

    # apply our mask -- notice how only the person in the image is
    # cropped out
    # masked = cv2.bitwise_and(image, image, mask=mask)
    # cv2.imshow("Mask Applied to Image", masked)
    # cv2.waitKey(0)

    # now, let's make a circular mask with a radius of 100 pixels and
    # apply the mask again
    mask_circle = np.zeros(image.shape[:2], dtype="uint8")
    mask_circle = cv2.circle(mask_circle, ((x1 + x2) // 2, (y1 + y2) // 2), length // 2, 255, -1)
    # masked_circle = cv2.bitwise_and(image, image, mask=mask_circle)

    # show the output images
    # cv2.imshow("Circular Mask", mask_circle)
    # cv2.imshow("Mask Applied to Image", masked_circle)
    # cv2.waitKey(0)

    flag = np.random.randint(2)
    # print(flag)
    if flag == 0:
        mask_rect = Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
        # mask_savepath = os.path.join(mask_saveroot, Path(image_name).stem + '_gt.png')
        # mask_rect.save(mask_savepath)
        return mask_rect

    else:
        mask_circ = Image.fromarray(cv2.cvtColor(mask_circle, cv2.COLOR_BGR2RGB))
        # mask_savepath = os.path.join(mask_saveroot, Path(image_name).stem + '_gt.png')
        # mask_circ.save(mask_savepath)
        return mask_circ

def ImagePath2SAMSegMask(img_path, image):
    if 'Casiav2' in img_path:
        json_saveroot = '/home/zzt/pan2/DATASETS/CASIAv2_samseg/au_seg_json'
        json_save_path = os.path.join(json_saveroot, os.path.basename(img_path).split('.')[0] + '_samseg.json')
    elif 'RAISE_RAW' in img_path:
        json_saveroot = '/home/zzt/pan2/DATASETS/RAISE/RAISE_RAW_samseg/au_seg_json'
        json_save_path = os.path.join(json_saveroot, os.path.basename(img_path).split('.')[0] + '_samseg.json')
    elif 'compRAISE' in img_path:
        json_saveroot = '/home/zzt/pan2/DATASETS/RAISE/compRAISE_samseg/au_seg_json'
        json_save_path = os.path.join(json_saveroot, os.path.basename(img_path).split('.')[0] + '_samseg.json')
    else:
        print('make_copymove_tamp sam seg json save path not found:', img_path)
        json_save_path = 'None'
    # image = Image.open(img_path)
    # image = np.asarray(image)
    if json_save_path != 'None':
        with open(json_save_path, "r") as f:
            mask_info = json.load(f)
        # if mask_info['mask_num'] > 6:
        #     rand_mask_index = np.random.randint(mask_info['mask_num']//2)
        # else:
        #     rand_mask_index = np.random.randint(mask_info['mask_num'])
        rand_mask_index = np.random.randint(mask_info['mask_num'])
        # rand_mask_index = 0
        mask = rle_to_mask(mask_info['anno_info'][rand_mask_index]['segmentation'])
        if image.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]))
        mask = mask.astype('uint8') * 255
    else:
        print(f'ImagePath2SAMSegMask failed: {img_path}')
        mask = gen_mask_imginput(image)
        mask = cv2.cvtColor(np.asarray(mask), cv2.COLOR_RGB2GRAY)
    return mask


class DeepfakeDataset(Dataset):
    def sampling(self, distribution, n_max):
        self.input_image_paths = []
        self.mask_image_paths = []
        self.edge_image_paths = []
        self.labels = []
        self.n_c_samples = n_max

        for label_str in distribution:
            list = distribution[label_str]
            n_list = len(list)
            if self.val:
                picked = random.sample(list, n_list)  # 不进行正负样本均衡
            else:
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

                    # CASIAv2数据集，对auth图像的10%进行篡改，为类别均衡，减掉 self.n_c_samples*0.1
                    if not self.val:
                        # picked = random.sample(list, round((self.n_c_samples % n_list)-self.n_c_samples*0.1))
                        picked = random.sample(list, (self.n_c_samples % n_list))
                    else:
                        picked = random.sample(list, (self.n_c_samples % n_list))

            # for picked
            for p in picked:
                (input_image_path, mask_image_path,edge_image_path) = p
                self.input_image_paths.append(input_image_path)
                self.mask_image_paths.append(mask_image_path)
                self.edge_image_paths.append(edge_image_path)
                self.labels.append(int(label_str))
            # print(f'sampled {label_str}:{len(picked)}')

        # if not self.val:
        #     self.input_image_paths = self.input_image_paths[int(n_max*0.1):]
        #     self.mask_image_paths = self.mask_image_paths[int(n_max*0.1):]
        #     self.edge_image_paths = self.edge_image_paths[int(n_max*0.1):]
        #     self.labels = self.labels[int(n_max*0.1):]
        label_0_num = 0
        label_1_num = 0
        for label in self.labels:
            if label == 0:
                label_0_num = label_0_num + 1
            else:
                label_1_num = label_1_num + 1
        print(f'label 0 num {label_0_num}, label 1 num {label_1_num}')
        print(self.input_image_paths[:5], self.input_image_paths[self.n_c_samples: self.n_c_samples + 5])
        print(self.mask_image_paths[:5], self.mask_image_paths[self.n_c_samples: self.n_c_samples + 5])
        print(self.edge_image_paths[:5], self.edge_image_paths[self.n_c_samples: self.n_c_samples + 5])
        print(self.labels[:5], self.labels[self.n_c_samples: self.n_c_samples + 5])
        return

    def __init__(self, paths_file_list, image_size, id, model_gc, model_osn, model_harmonizer, model_enhancer, n_c_samples=None, val=False, manipulate_dataset=False, postprocess_dataset=False, osn_noise=False, sample_mode='max'):
        self.image_size = image_size
        self.manipulate_dataset = manipulate_dataset
        self.postprocess_dataset = postprocess_dataset
        self.osn_noise = osn_noise
        if not val:
            print(f'init train dataset: manipulate_dataset={manipulate_dataset}, postprocess_dataset = {postprocess_dataset}, osn_noise={osn_noise}')
        else:
            print(f'init val dataset: manipulate_dataset={manipulate_dataset}, postprocess_dataset = {postprocess_dataset}, osn_noise={osn_noise}')
        self.model_gc = model_gc
        self.model_osn = model_osn
        self.model_harmonizer = model_harmonizer
        self.model_enhancer = model_enhancer

        self.n_c_samples = n_c_samples
        self.sample_mode = sample_mode
        self.tamper_mode = -1
        self.postprocess_mode = -1

        self.val = val

        self.input_image_paths = []
        self.mask_image_paths = []
        self.edge_image_paths = []
        self.labels = []

        # 函数列表
        self.tamp_function_list = [self.do_cp, self.do_sp, self.do_ip]
        # self.tamp_function_list = [self.do_cp, self.do_empty, self.do_empty]
        self.manipulate_thresh = 90  # 0-100
        self.random_cit_function_list = [self.random_color, self.random_illumination, self.random_textureflatten]
        self.cit_thresh = -1  # 0-1
        self.blend_function_list = [self.poison_blending, self.harmonizer_blending]
        self.poison_thresh = 0.2  # 0-1
        self.blend_thresh = 20  # 0-100
        print(f'manipulate dataset thresholds: manipulate_thresh={self.manipulate_thresh}, cit_thresh={self.cit_thresh}, poison_thresh={self.poison_thresh}, blend_thresh={self.blend_thresh}')

        # for paths_file in paths_file_list:
        #     with open(paths_file, 'r') as f:
        #         lines = f.readlines()
        #         for l in lines:
        #             self.lines.append(l)

        self.distribution = dict()
        self.n_max = 0
        self.n_min = math.inf

        for paths_file in tqdm(paths_file_list):
            with open(paths_file, 'r') as f:
                lines = f.readlines()
                for l in lines:
                    if 'MISD_Dataset' in l:
                        parts = l.rstrip().split(',')
                    else:
                        parts = l.rstrip().split(' ')
                    # parts = l.rstrip().split(' ')
                    if len(parts) == 4:
                        if 'RAISE' in paths_file:
                            input_image_path = os.path.join('/home/zzt/pan2/DATASETS', parts[0])
                            if parts[1] != 'None':
                                mask_image_path = os.path.join('/home/zzt/pan2/DATASETS', parts[1])
                            else:
                                mask_image_path = parts[1]
                            if parts[2] != 'None':
                                edge_image_path = os.path.join('/home/zzt/pan2/DATASETS', parts[2])
                            else:
                                edge_image_path = parts[2]
                            label_str = parts[3]
                        elif 'tampCOCO' in paths_file:
                            input_image_path = os.path.join('/home/zzt/pan2/DATASETS/CAT-Net_datasets/tampCOCO', parts[0])
                            if parts[1] != 'None':
                                mask_image_path = os.path.join('/home/zzt/pan2/DATASETS/CAT-Net_datasets/tampCOCO', parts[1])
                            else:
                                mask_image_path = parts[1]
                            if parts[2] != 'None':
                                edge_image_path = os.path.join('/home/zzt/pan2/DATASETS/CAT-Net_datasets/tampCOCO', parts[2])
                            else:
                                edge_image_path = parts[2]
                            label_str = parts[3]
                        elif 'DATASETS/defacto' in l:
                            input_image_path = parts[0].replace('/media/zzt/新加卷/DATA/DATASETS', '/home/zzt/pan2/DATASETS')
                            mask_image_path = parts[1].replace('/media/zzt/新加卷/DATA/DATASETS', '/home/zzt/pan2/DATASETS')
                            edge_image_path = parts[2].replace('/media/zzt/新加卷/DATA/DATASETS', '/home/zzt/pan2/DATASETS')
                            label_str = parts[3]
                        else:
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
                    if (label_str not in self.distribution):
                        self.distribution[label_str] = [(input_image_path, mask_image_path,edge_image_path)]
                    else:
                        self.distribution[label_str].append((input_image_path, mask_image_path,edge_image_path))

                    # if (len(distribution[label_str]) > n_max):
                    #     n_max = len(distribution[label_str])

        for label_str in ['0', '1']:
            print(f'distribution[{label_str}]:{len(self.distribution[label_str])}')
            if len(self.distribution[label_str]) < self.n_min:
                self.n_min = len(self.distribution[label_str])
            if len(self.distribution[label_str]) > self.n_max:
                self.n_max = len(self.distribution[label_str])
        if self.sample_mode == 'max':
            self.sampling(self.distribution, self.n_max)
            # print(self.input_image_paths[:5], self.input_image_paths[n_max: n_max+5])
            # print(self.mask_image_paths[:5], self.mask_image_paths[n_max: n_max+5])
            # print(self.edge_image_paths[:5], self.edge_image_paths[n_max: n_max+5])
            # print(self.labels[:5], self.labels[n_max: n_max+5])
        elif self.sample_mode == 'min':
            self.sampling(self.distribution, self.n_min)
            # print(self.input_image_paths[:5], self.input_image_paths[n_min: n_min+5])
            # print(self.mask_image_paths[:5], self.mask_image_paths[n_min: n_min+5])
            # print(self.edge_image_paths[:5], self.edge_image_paths[n_min: n_min+5])
            # print(self.labels[:5], self.labels[n_min: n_min+5])
        else:
            raise NotImplementedError('set mode max or min')



        # ----------
        #  TODO: Transforms for data augmentation (more augmentations should be added)
        # ----------
        # # origin
        # self.transform_train = A.Compose([
        #     A.Resize(self.image_size, self.image_size),
        #     # A.RandomBrightnessContrast(
        #     #     brightness_limit=(-0.1, 0.1),
        #     #     contrast_limit=0.1,
        #     #     p=1
        #     # ),
        #     # Rotate
        #     # A.RandomRotate90(p=0.5),
        #     A.HorizontalFlip(p=0.5),
        #     A.VerticalFlip(p=0.5),
        #     # A.OneOf([
        #     # A.ImageCompression(quality_lower=60, quality_upper=100, p=0.2),
        #     # A.GaussianBlur(p=0.2),
        #     # A.MotionBlur(p=0.2),], p=0.5),
        #     # img = (img - mean * max_pixel_value) / (std * max_pixel_value)
        #     # A.Normalize(always_apply=True, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        #     A.Normalize(always_apply=True, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        #     ToTensorV2()
        # ], additional_targets={'edge': 'mask'})
        #
        # self.transform_val = A.Compose([
        #     A.Resize(self.image_size, self.image_size),
        #     # A.Normalize(always_apply=True, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        #     A.Normalize(always_apply=True, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        #     ToTensorV2()
        # ], additional_targets={'edge': 'mask'})

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
        self.transform_train = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.OneOf([
                A.HorizontalFlip(p=1),
                A.VerticalFlip(p=1),
                A.RandomRotate90(p=1),
                A.Transpose(p=1),
            ], p=0.75),
            # A.ImageCompression(quality_lower=50, quality_upper=95, p=0.75),
            # A.OneOf([
            #     A.OneOf([
            #         A.Blur(p=1),
            #         A.GaussianBlur(p=1),
            #         A.MedianBlur(p=1),
            #         A.MotionBlur(p=1),
            #     ], p=1),
            #     A.OneOf([
            #         A.Downscale(p=1),
            #         A.GaussNoise(p=1),
            #         A.ISONoise(p=1),
            #         A.RandomBrightnessContrast(p=1),
            #         A.RandomGamma(p=1),
            #         A.RandomToneCurve(p=1),
            #         A.Sharpen(p=1),
            #     ], p=1),
            #     A.OneOf([
            #         A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=1),
            #         A.GridDistortion(p=1),
            #     ], p=1),
            # ], p=0.25),
        ], additional_targets={'edge': 'mask'})
        self.transform_randaug = T.transforms.Compose([
        # T.transforms.RandomCrop(32, padding=4),
        # T.transforms.RandomHorizontalFlip(),
        # T.transforms.ToTensor(),
        # T.transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),

        ])
        # Add RandAugment with N, M(hyperparameter)
        self.transform_randaug.transforms.insert(0, RandAugment(2, 5))  # N, M

        self.transform_val = A.Compose([
            A.Resize(self.image_size, self.image_size),
        ], additional_targets={'edge': 'mask'})

        assert len(self.input_image_paths) == len(self.mask_image_paths)

    def sample_epoch(self):
        if self.sample_mode == 'max':
            self.sampling(self.distribution, self.n_max)
            # print(self.input_image_paths[:5], self.input_image_paths[n_max: n_max+5])
            # print(self.mask_image_paths[:5], self.mask_image_paths[n_max: n_max+5])
            # print(self.edge_image_paths[:5], self.edge_image_paths[n_max: n_max+5])
            # print(self.labels[:5], self.labels[n_max: n_max+5])
        elif self.sample_mode == 'min':
            self.sampling(self.distribution, self.n_min)
            # print(self.input_image_paths[:5], self.input_image_paths[n_min: n_min+5])
            # print(self.mask_image_paths[:5], self.mask_image_paths[n_min: n_min+5])
            # print(self.edge_image_paths[:5], self.edge_image_paths[n_min: n_min+5])
            # print(self.labels[:5], self.labels[n_min: n_min+5])
        else:
            raise NotImplementedError('set mode max or min')

    def shuffle(self):
        for dataset in self.dataset_list:
            random.shuffle(dataset.tamp_list)

    def do_cp(self, input_img, input_mask, input_edge, input_label, input_file_name):
        try:
            if input_label == 0 or np.asarray(input_mask).max() == 0:  # 真实图像输入
                if random.random() < 0.5:  # 从SAM分割的mask中挑选
                    cp_mask = ImagePath2SAMSegMask(input_file_name, input_img)
                else:  # 随机圆形矩形mask
                    cp_mask = gen_mask_imginput(input_img)
                    cp_mask = cv2.cvtColor(np.asarray(cp_mask), cv2.COLOR_RGB2GRAY)
                img_tamp, img_tamp_mask, img_tamp_edge = self.make_copymove_tamp_real(input_img, cp_mask)
            elif input_label > 0:  # 篡改图像输入
                if random.random() < 0.5:  # 使用篡改区域
                    cp_mask = input_mask
                else:  # 随机圆形矩形mask
                    cp_mask = gen_mask_imginput(input_img)
                    cp_mask = cv2.cvtColor(np.asarray(cp_mask), cv2.COLOR_RGB2GRAY)
                img_tamp, img_tamp_mask, img_tamp_edge = self.make_copymove_tamp(input_img, cp_mask, input_edge)
            else:
                raise Exception(f'cp input_label is not applicable: {input_label}')
        except:
            traceback.print_exc()
            img_tamp, img_tamp_mask, img_tamp_edge = input_img, input_mask, input_edge
            print(f'make_copymove_tamp failed0:{input_file_name}')
        tam_img = img_tamp
        tam_mask = cv2.bitwise_or(input_mask.squeeze(), np.array(img_tamp_mask).squeeze())
        edge_cover = cv2.bitwise_and(input_edge.squeeze(), img_tamp_mask.squeeze(), mask=None)
        edge_cover = input_edge.squeeze() - edge_cover
        tam_edge = cv2.bitwise_or(edge_cover, img_tamp_edge.squeeze())
        if tam_mask.max() > 0:
            tam_label = 1
        else:
            tam_label = 0
        return tam_img, tam_mask, tam_edge, tam_label

    def do_sp(self, input_img, input_mask, input_edge, input_label, input_file_name):
        # 随机取拼接对象
        sp_index = np.random.randint(len(self.input_image_paths))
        sp_image_path = self.input_image_paths[sp_index]
        sp_mask_path = self.mask_image_paths[sp_index]
        try:
            img_tamp, img_tamp_mask, img_tamp_edge, success_flag = self.make_splice_tamp(input_img, input_mask, sp_image_path,
                                                                                    sp_mask_path)
            if success_flag:
                tam_img = np.array(img_tamp)
                tam_mask = cv2.bitwise_or(input_mask.squeeze(), np.array(img_tamp_mask).squeeze())
                edge_cover = cv2.bitwise_and(input_edge.squeeze(), img_tamp_mask.squeeze(), mask=None)
                edge_cover = input_edge.squeeze() - edge_cover
                tam_edge = cv2.bitwise_or(edge_cover, img_tamp_edge.squeeze())
                if np.max(tam_mask) == 0:
                    input_label = 0
                else:
                    input_label = 1
                tam_label = input_label
            else:
                print(f'make_splice_tamp failed0:{sp_image_path}, {sp_mask_path}')
                tam_img = input_img
                tam_mask = input_mask
                tam_edge = input_edge
                tam_label = input_label
        except:
            traceback.print_exc()
            tam_img = input_img
            tam_mask = input_mask
            tam_edge = input_edge
            tam_label = input_label
            print(f'make_splice_tamp failed1:{sp_image_path}, {sp_mask_path}')

        return tam_img, tam_mask, tam_edge, tam_label

    def do_empty(self, input_img, input_mask, input_edge, input_label, input_file_name):
        return input_img, input_mask, input_edge, input_label

    def do_ip(self, input_img, input_mask, input_edge, input_label, input_file_name):
        # 选择 inpaint mask
        select_mask_flag = random.random()
        if input_label == 0 or np.asarray(input_mask).max() == 0:  # 真实图像输入
            if select_mask_flag < 0.4:  # 从SAM分割的mask中挑选
                ip_mask = ImagePath2SAMSegMask(input_file_name, input_img)
            elif select_mask_flag > 0.7:  # 随机不规则mask
                mask_index = np.random.randint(12000)
                mask_index_str = "%05d" % mask_index
                ip_mask = cv2.imread(
                    f'/home/zzt/pan2/DATASETS/NVIDIA_irregular_mask/test_mask/testing_mask_dataset/{mask_index_str}.png', cv2.IMREAD_GRAYSCALE)
            else:  # 随机圆形矩形mask
                ip_mask = gen_mask_imginput(input_img)
                ip_mask = cv2.cvtColor(np.asarray(ip_mask), cv2.COLOR_RGB2GRAY)
        elif input_label > 0:  # 篡改图像输入
            if select_mask_flag < 0.25:  # 使用篡改区域
                ip_mask = np.asarray(input_mask).squeeze()
                if ip_mask.ndim == 3:
                    ip_mask = cv2.cvtColor(np.asarray(ip_mask), cv2.COLOR_RGB2GRAY)
            elif 0.25 <= select_mask_flag < 0.5:  # 从所有图像中挑选
                # 随机取mask
                rand_index = np.random.randint(len(self.input_image_paths))
                rand_image_path = self.input_image_paths[rand_index]
                # ip_image_path_content = cv2.imread(ip_image_path)
                rand_mask_path = self.mask_image_paths[rand_index]
                if rand_mask_path == 'None':  # SAM分割的mask
                    ip_mask = ImagePath2SAMSegMask(rand_image_path, input_img)
                else:  # 随机取的篡改图像mask
                    ip_mask = cv2.imread(rand_mask_path, cv2.IMREAD_GRAYSCALE)
            elif 0.5 <= select_mask_flag > 0.75:  # 随机不规则mask
                mask_index = np.random.randint(12000)
                mask_index_str = "%05d" % mask_index
                ip_mask = cv2.imread(
                    f'/home/zzt/pan2/DATASETS/NVIDIA_irregular_mask/test_mask/testing_mask_dataset/{mask_index_str}.png', cv2.IMREAD_GRAYSCALE)
            else:  # 随机圆形矩形mask
                ip_mask = gen_mask_imginput(input_img)
                ip_mask = cv2.cvtColor(np.asarray(ip_mask), cv2.COLOR_RGB2GRAY)
        else:
            raise Exception(f'ip input_label is not applicable: {input_label}')

        # 进行 inpaint
        img_tamp, img_tamp_mask, img_tamp_edge, img_tamp_label = self.make_ip_imginput(input_img, input_mask, input_edge, ip_mask)
        tam_img = cv2.cvtColor(np.array(img_tamp), cv2.COLOR_BGR2RGB)
        tam_mask = np.array(img_tamp_mask)
        tam_edge = np.array(img_tamp_edge)
        if np.max(tam_mask) == 0:
            tam_label = 0
        else:
            tam_label = 1
        return tam_img, tam_mask, tam_edge, tam_label

    def random_scale(self, image, mask, tamp_pixel_percent):
        if tamp_pixel_percent >= 0.3:
            scale_factor = (random.random() * 0.7) + 0.3  # 0.3到1.0
        else:
            scale_factor = (random.random() * 1.5) + 0.5  # 0.5到2
        data_aug_ind = random.randint(0, 5)
        if data_aug_ind == 0:  # 不resize
            img_resized = image
            mask_resized = mask
        elif data_aug_ind == 1:
            img_resized = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
            mask_resized = cv2.resize(mask, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
        elif data_aug_ind == 2:
            img_resized = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
            mask_resized = cv2.resize(mask, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        elif data_aug_ind == 3:
            img_resized = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
            mask_resized = cv2.resize(mask, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        elif data_aug_ind == 4:
            img_resized = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LANCZOS4)
            mask_resized = cv2.resize(mask, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LANCZOS4)
        elif data_aug_ind == 5:
            img_resized = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
            mask_resized = cv2.resize(mask, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
        else:
            raise Exception('random scale index is not applicable.')
        return img_resized, mask_resized

    def random_scale_with_edge(self, image, mask, edge, tamp_pixel_percent):
        if tamp_pixel_percent >= 0.3:
            scale_factor = (random.random() * 0.7) + 0.3  # 0.3到1.0
        else:
            scale_factor = (random.random() * 1.5) + 0.5  # 0.5到2
        data_aug_ind = random.randint(0, 5)
        if data_aug_ind == 0:  # 不resize
            img_resized = image
            mask_resized = mask
            edge_resized = edge
        elif data_aug_ind == 1:
            img_resized = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
            mask_resized = cv2.resize(mask, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
            edge_resized = cv2.resize(edge, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
        elif data_aug_ind == 2:
            img_resized = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
            mask_resized = cv2.resize(mask, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
            edge_resized = cv2.resize(edge, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        elif data_aug_ind == 3:
            img_resized = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
            mask_resized = cv2.resize(mask, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
            edge_resized = cv2.resize(edge, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        elif data_aug_ind == 4:
            img_resized = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LANCZOS4)
            mask_resized = cv2.resize(mask, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LANCZOS4)
            edge_resized = cv2.resize(edge, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LANCZOS4)
        elif data_aug_ind == 5:
            img_resized = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
            mask_resized = cv2.resize(mask, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
            edge_resized = cv2.resize(edge, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
        else:
            raise Exception('random scale index is not applicable.')
        return img_resized, mask_resized, edge_resized

    def random_scale_for_inpaint(self, image):
        # sizes = [(680, 512), (512, 680), (512, 512), (340, 256), (256, 340), (256, 256)]
        sizes_w_h = [(680, 512), (384, 256)]
        sizes_h_w = [(512, 640), (256, 384)]
        sizes_equal = [(512, 512), (256, 256)]
        # sizes_equal = [(512, 512)]
        h, w, c = image.shape
        if w / h > 1.15:
            size = random.choice(sizes_w_h)
        elif w / h < 0.85:
            size = random.choice(sizes_h_w)
        else:
            size = random.choice(sizes_equal)
        data_aug_ind = random.randint(1, 5)
        if data_aug_ind == 0:  # 不resize
            img_resized = image
        elif data_aug_ind == 1:
            img_resized = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        elif data_aug_ind == 2:
            img_resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        elif data_aug_ind == 3:
            img_resized = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
        elif data_aug_ind == 4:
            img_resized = cv2.resize(image, size, interpolation=cv2.INTER_LANCZOS4)
        elif data_aug_ind == 5:
            img_resized = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)
        else:
            raise Exception('random scale index is not applicable.')
        return img_resized

    def random_rotate(self, image, mask, start_angle, end_angle):
        h, w, c = image.shape
        rotate_angle = random.randint(start_angle, end_angle)
        data_aug_ind = random.randint(0, 1)
        # crop_x1, crop_y1, crop_w, crop_h = find_img_boundary(image, mask)
        if data_aug_ind == 0:  # 不resize
            img_rotated = image
            mask_rotated = mask
        elif data_aug_ind == 1:
            M = cv2.getRotationMatrix2D((w / 2, h / 2), rotate_angle, 1)
            # M = cv2.getRotationMatrix2D((crop_x1 + (crop_w / 2), crop_y1 + (crop_h / 2)), rotate_angle, 1)
            # M = cv2.getRotationMatrix2D((crop_x1 + (crop_w / 2), crop_y1 + (crop_h / 2)), rotate_angle, 1)
            # 计算图像新边界
            new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
            new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))
            # 调整旋转矩阵以考虑平移
            M[0, 2] += (new_w - w) / 2
            M[1, 2] += (new_h - h) / 2
            img_rotated = cv2.warpAffine(image, M, (new_w, new_h))
            mask_rotated = cv2.warpAffine(mask, M, (new_w, new_h))
            # img_rotated = cv2.warpAffine(image, M, (w, h))
            # mask_rotated = cv2.warpAffine(mask, M, (w, h))
        else:
            raise Exception('random rotate index is not applicable.')
        return img_rotated, mask_rotated

    def random_rotate_with_edge(self, image, mask, edge, start_angle, end_angle):
        h, w, c = image.shape
        rotate_angle = random.randint(start_angle, end_angle)
        data_aug_ind = random.randint(0, 1)
        # crop_x1, crop_y1, crop_w, crop_h = find_img_boundary(image, mask)
        if data_aug_ind == 0:  # 不resize
            img_rotated = image
            mask_rotated = mask
            edge_rotated = edge
        elif data_aug_ind == 1:
            M = cv2.getRotationMatrix2D((w / 2, h / 2), rotate_angle, 1)
            # M = cv2.getRotationMatrix2D((crop_x1 + (crop_w / 2), crop_y1 + (crop_h / 2)), rotate_angle, 1)
            # M = cv2.getRotationMatrix2D((crop_x1 + (crop_w / 2), crop_y1 + (crop_h / 2)), rotate_angle, 1)
            # 计算图像新边界
            new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
            new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))
            # 调整旋转矩阵以考虑平移
            M[0, 2] += (new_w - w) / 2
            M[1, 2] += (new_h - h) / 2
            img_rotated = cv2.warpAffine(image, M, (new_w, new_h))
            mask_rotated = cv2.warpAffine(mask, M, (new_w, new_h))
            edge_rotated = cv2.warpAffine(edge, M, (new_w, new_h))
            # img_rotated = cv2.warpAffine(image, M, (w, h))
            # mask_rotated = cv2.warpAffine(mask, M, (w, h))
        else:
            raise Exception('random rotate index is not applicable.')
        return img_rotated, mask_rotated, edge_rotated

    def random_color(self, image, mask):
        return image

    def random_brightness_contrast(self, image, mask):
        """
            random_brightness_contrast, 50% tamp, 50% auth.

            Arguments:
              image (BGR image): (H,W,3), background image to be blended into.
              mask (gray scale image): (H,W), mask for image.

            """
        assert mask.ndim == 2
        img = image.astype(np.float32)
        tam_idx = np.random.randint(6)
        if tam_idx < 3:
            return image
        elif tam_idx == 3:
            aa = random.random() * 1.5 + 0.6  # 对比度 0.6~2.1
            bb = 0
        elif tam_idx == 4:
            aa = 1
            bb = np.random.randint(-80, 80)  # 亮度+-80
        else:
            aa = random.random() * 1.5 + 0.6  # 对比度 0.6~2.1
            bb = np.random.randint(-80, 80)  # 亮度+-80
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        bri_mean = np.mean(img, where=(mask_rgb != 0))
        img1 = aa * (img - bri_mean) + bb + bri_mean
        img1 = np.clip(img1, 0, 255).astype(np.uint8)
        img1 = cv2.bitwise_and(img1, img1, mask=mask)
        return img1

    def find_img_boundary(self, image, mask):
        # 获取图片大小，保证mask和图片大小一致
        # h, w, c = image.shape
        # mask = cv2.resize(mask, (w, h))
        # print("rate", rate)
        # closed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        (cnts, _) = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

        # compute the rotated bounding box of the largest contour
        x1, y1, w, h = cv2.boundingRect(c)

        return x1, y1, w, h

    def random_illumination(self, image, mask):
        # if random.randint(0, 1):
        #     # 膨胀
        #     kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        #     mask = cv2.dilate(mask, kernel, 2)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # a_list = [0.2, 0.4, 0.5, 0.7, 1.0,]
        # rand_a = random.choice(a_list)
        b_list = [0, 0, 0.1, 0.2, 0.3]
        rand_b = random.choice(b_list)
        rand_a = random.random() * 5 + 0.2
        # rand_b = random.random()/4.0
        result = cv2.illuminationChange(image, mask, alpha=rand_a, beta=rand_b)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return result

    def random_textureflatten(self, image, mask):
        # if random.randint(0, 1):
        #     # 膨胀
        #     kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        #     mask = cv2.dilate(mask, kernel, 2)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        low_list = [20, 25, 30, 35]
        high_list = [40, 45, 50, 55, 60]
        kernelsize_list = [3, 5, 7]
        rand_low = random.choice(low_list)
        rand_high = random.choice(high_list)
        rand_kenel = random.choice(kernelsize_list)
        result = cv2.textureFlattening(image, mask, low_threshold=rand_low, high_threshold=rand_high,
                                       kernel_size=rand_kenel)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return result

    def random_color(self, image, mask):
        # if random.randint(0, 1):
        #     # 膨胀
        #     kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        #     mask = cv2.dilate(mask, kernel, 2)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        rand_red = random.random() * 2.0 + 0.5
        rand_green = random.random() * 2.0 + 0.5
        rand_blue = random.random() * 2.0 + 0.5
        result = cv2.colorChange(image, mask, red_mul=rand_red, green_mul=rand_green, blue_mul=rand_blue)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return result

    def poison_blending(self, image, img_tamp, img_tamp_mask_only1):
        """
        Using opencv poison blend function to blend target and obj.

        Arguments:
          image (BGR image): (H,W,3), background image to be blended into.
          img_tamp (RGB image): (H,W,3), foreground image to be blended.
          img_tamp_mask_only1 (gray scale image): (H,W), mask for img_tamp.

        """
        # poisson blending
        try:
            if len(img_tamp_mask_only1.shape) > 2:
                if img_tamp_mask_only1.shape[2] == 3:
                    img_tamp_mask_only1 = cv2.cvtColor(img_tamp_mask_only1, cv2.COLOR_BGR2GRAY)
                else:
                    img_tamp_mask_only1 = np.squeeze(img_tamp_mask_only1)
            img_temp = cv2.cvtColor(img_tamp, cv2.COLOR_RGB2BGR)
            obj = cv2.bitwise_and(img_temp, img_temp, mask=img_tamp_mask_only1)
            dst = image
            mask_inner = img_tamp_mask_only1[1:img_tamp_mask_only1.shape[0] - 2, 1:img_tamp_mask_only1.shape[1] - 2]
            img_tamp_mask_only1 = cv2.copyMakeBorder(mask_inner, 1, 1, 1, 1, cv2.BORDER_ISOLATED | cv2.BORDER_CONSTANT,
                                                     img_tamp_mask_only1, 0)
            roi = cv2.boundingRect(img_tamp_mask_only1)
            center = (int(roi[0] + roi[2] / 2), int(roi[1] + roi[3] / 2))
            # 膨胀
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            img_tamp_mask_only1 = cv2.dilate(img_tamp_mask_only1, kernel, 1)

            blend_type = random.randint(0, 5)
            # blend_type = 3
            if blend_type <= 2:  # 不blend,2
                poisson_blend_img = img_tamp
            elif blend_type == 3:  # 3
                poisson_blend_img = cv2.seamlessClone(obj, dst, img_tamp_mask_only1, center, cv2.NORMAL_CLONE)
                poisson_blend_img = cv2.cvtColor(poisson_blend_img, cv2.COLOR_BGR2RGB)
            elif blend_type == 4:  # 4
                poisson_blend_img = cv2.seamlessClone(obj, dst, img_tamp_mask_only1, center, cv2.MIXED_CLONE)
                poisson_blend_img = cv2.cvtColor(poisson_blend_img, cv2.COLOR_BGR2RGB)
            elif blend_type == 5:  # 5
                poisson_blend_img = cv2.seamlessClone(obj, dst, img_tamp_mask_only1, center, cv2.MONOCHROME_TRANSFER)
                poisson_blend_img = cv2.cvtColor(poisson_blend_img, cv2.COLOR_BGR2RGB)
            else:
                raise Exception('blend index is not applicable.')

            return poisson_blend_img
        except:
            print('poisson blending failed')
            return img_tamp

    def harmonizer_blending(self, image, img_tamp, img_tamp_mask_only1):
        """
        Using harmonizer to blend target and obj.

        Arguments:
          image (BGR image): (H,W,3), background image to be blended into.
          img_tamp (RGB image): (H,W,3), foreground image to be blended.
          img_tamp_mask_only1 (gray scale image): (H,W), mask for img_tamp.

        Output:
          harmonizer_blend_img (RGB image): (H,W,3), blended image.
        """
        pass
        return img_tamp

    def auto_canny(self, image, sigma=0.33):
        # compute the median of the single channel pixel intensities
        v = np.median(image)

        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)

        # return the edged image
        return edged

    def edge2(self, mask):
        # mask[mask < 200] = 0
        # mask[mask >= 200] = 255
        height = mask.shape[0]
        width = mask.shape[1]

        mask_around = np.ones_like(mask, dtype="uint8")
        mask_around = mask_around * 255
        res_pixels = 6
        y1 = res_pixels
        y2 = (height - res_pixels)
        x1 = res_pixels
        x2 = (width - res_pixels)
        mask_around[y1: y2, x1: x2] = 0
        # cv2.imshow('mask_around', mask_around)
        # 膨胀
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        mask = cv2.dilate(mask, kernel, 2)

        # 腐蚀
        kernel_erod = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        mask = cv2.erode(mask.copy(), kernel_erod, 2)
        mask_edge = self.auto_canny(mask)
        # 膨胀
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_edge = cv2.dilate(mask_edge, kernel, 2)
        # 膨胀
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_edge = cv2.dilate(mask_edge, kernel, 2)

        mask_edge2 = np.zeros_like(mask)
        if len(mask.shape) == 3:
            mask_edge2[:, :, 0] = mask_edge
            mask_edge2[:, :, 1] = mask_edge
            mask_edge2[:, :, 2] = mask_edge
        elif len(mask.shape) == 2:
            mask_edge2[:, :] = mask_edge
        else:
            raise NotImplementedError(f'mask shape {len(mask.shape)} not implemented')

        mask_around = cv2.bitwise_and(mask_around, mask)
        mask_edge_final = cv2.bitwise_or(mask_around, mask_edge2)
        # cv2.imshow('mask_final', mask_edge_final)
        # cv2.waitKey()
        return mask_edge_final

    def edge3(self, mask):
        # mask[mask < 200] = 0
        # mask[mask >= 200] = 255
        height = mask.shape[0]
        width = mask.shape[1]
        mask_around = np.ones_like(mask, dtype="uint8")
        mask_around = mask_around * 255
        res_pixels = 5
        y1 = res_pixels
        y2 = (height - res_pixels)
        x1 = res_pixels
        x2 = (width - res_pixels)
        mask_around[y1: y2, x1: x2] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        mask_delate = cv2.dilate(mask, kernel, 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        mask_erode = cv2.erode(mask, kernel, 2)
        mask_edge = mask_delate - mask_erode
        # 膨胀
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_edge = cv2.dilate(mask_edge, kernel, 2)

        # 膨胀
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_edge = cv2.dilate(mask_edge, kernel, 2)

        mask_around = cv2.bitwise_and(mask_around, mask)
        mask_edge_final = cv2.bitwise_or(mask_around, mask_edge)
        # cv2.imshow('mask_final', mask_edge_final)
        # cv2.waitKey()
        return mask_edge_final

    def merge_mask_edge(self, mask, edge):
        height = mask.shape[0]
        width = mask.shape[1]
        mask_around = np.ones_like(mask, dtype="uint8")
        mask_around = mask_around * 255
        res_pixels = 5
        y1 = res_pixels
        y2 = (height - res_pixels)
        x1 = res_pixels
        x2 = (width - res_pixels)
        mask_around[y1: y2, x1: x2] = 0
        mask_around = cv2.bitwise_and(mask_around, mask)
        mask_edge_final = cv2.bitwise_or(mask_around, edge)
        return mask_edge_final

    def make_copymove_tamp(self, image, mask, input_edge):
        if mask.ndim >= 3:
            mask = np.squeeze(mask)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # mask[mask > 127] = 255
        # mask[mask <= 127] = 0
        # 获取图片大小，保证mask和图片大小一致
        h, w, c = image.shape
        mask = cv2.resize(mask, (w, h))
        tamp_pixel_percent = np.count_nonzero(mask) / h / w
        image_masked = cv2.bitwise_and(image, image, mask=mask)
        edge_masked = cv2.bitwise_and(input_edge, input_edge, mask=mask)
        image_resized, mask_resized, edge_resized = self.random_scale_with_edge(image_masked, mask, edge_masked, tamp_pixel_percent)  # 随机缩放

        image_rotated, mask_rotated, edge_rotated = self.random_rotate_with_edge(image_resized, mask_resized, edge_resized, 0, 359)  # 随机旋转
        image_rotated = self.random_brightness_contrast(image_rotated, mask_rotated)  # 随机亮度对比度

        crop_x1, crop_y1, crop_w, crop_h = self.find_img_boundary(image_rotated, mask_rotated)
        image_croped = image_rotated[crop_y1:crop_y1 + crop_h, crop_x1:crop_x1 + crop_w]
        mask_croped = mask_rotated[crop_y1:crop_y1 + crop_h, crop_x1:crop_x1 + crop_w]
        edge_croped = edge_rotated[crop_y1:crop_y1 + crop_h, crop_x1:crop_x1 + crop_w]

        h_croped, w_croped, c_croped = image_croped.shape
        if w - w_croped > 0:
            paste_w = w - w_croped
        else:
            paste_w = w
        if h - h_croped > 0:
            paste_h = h - h_croped
        else:
            paste_h = h
        paste_x1 = random.randint(0, paste_w)
        paste_y1 = random.randint(0, paste_h)
        # image paste
        img1 = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        img2 = Image.fromarray(cv2.cvtColor(image_croped, cv2.COLOR_BGR2RGB))
        mask_croped_img = Image.fromarray(mask_croped)
        img1.paste(img2, (paste_x1, paste_y1), mask=mask_croped_img)
        img_tamp = np.array(img1)
        # mask paste
        mask_temp = np.zeros_like(mask)
        mask_temp_image = Image.fromarray(mask_temp)
        img1 = Image.fromarray(mask)
        img2 = Image.fromarray(mask_croped)
        mask_croped_img = Image.fromarray(mask_croped)
        img1.paste(img2, (paste_x1, paste_y1), mask=mask_croped_img)
        # img_tamp_mask = np.array(img1)
        # img_tamp_mask[img_tamp_mask > 127] = 255
        # img_tamp_mask[img_tamp_mask <= 127] = 0
        mask_temp_image.paste(img2, (paste_x1, paste_y1), mask=mask_croped_img)
        img_tamp_mask_only1 = np.array(mask_temp_image)
        img_tamp_mask_only1[img_tamp_mask_only1 > 127] = 255
        img_tamp_mask_only1[img_tamp_mask_only1 <= 127] = 0
        # edge paste
        edge_temp = np.zeros_like(mask)
        edge_temp_image = Image.fromarray(edge_temp)
        edge_croped_image = Image.fromarray(edge_croped)
        edge_temp_image.paste(edge_croped_image, (paste_x1, paste_y1), mask=edge_croped_image)
        img_tamp_edge_inside = np.array(edge_temp_image)

        # # 膨胀
        # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        # img_tamp_mask_only1 = cv2.dilate(img_tamp_mask_only1, kernel, 1)

        # # edge paste
        # edge = self.edge2(mask)
        # edge_croped = self.edge2(mask_croped)
        # # 去除重叠区域
        # edge_cover = cv2.bitwise_and(edge, img_tamp_mask_only1, mask=None)
        # edge = edge - edge_cover
        # img1 = Image.fromarray(edge)
        # img2 = Image.fromarray(edge_croped)
        # edge_croped_img = Image.fromarray(edge_croped)
        # img1.paste(img2, (paste_x1, paste_y1), mask=edge_croped_img)
        # img_tamp_edge = np.array(img1)

        img_tamp_edge_only1 = self.edge2(img_tamp_mask_only1)
        img_tamp_edge_only1 = cv2.bitwise_or(img_tamp_edge_only1, img_tamp_edge_inside)  # 若copymove区域本来就有篡改边缘则将内部边缘加入
        img_tamp_edge_only1 = self.merge_mask_edge(img_tamp_mask_only1, img_tamp_edge_only1)
        img_tamp_mask_only1[img_tamp_mask_only1 > 127] = 255
        img_tamp_mask_only1[img_tamp_mask_only1 <= 127] = 0
        img_tamp_edge_only1[img_tamp_edge_only1 > 127] = 255
        img_tamp_edge_only1[img_tamp_edge_only1 <= 127] = 0

        # random color, random illumination, random texture
        if random.random() < self.cit_thresh:  # 0.5
            # 随机选择一个、两个或全部函数并执行
            num_choices = random.randint(1, len(self.random_cit_function_list))  # 随机选择的数量
            random_functions = random.sample(self.random_cit_function_list, num_choices)
            # print("随机选择的函数：", [func.__name__ for func in random_functions])
            # 逐个执行被选择的函数
            for func in random_functions:
                # mask_list = [mask, img_tamp_mask, img_tamp_mask_only1]
                # random_mask = random.choice(mask_list)
                img_tamp = func(img_tamp, img_tamp_mask_only1)
        else:
            pass

        # poisson blending, harmonizer blending
        if random.random() < self.poison_thresh:  # 0.5
            # 随机选择一个函数并执行
            random_function = random.choice(self.blend_function_list)
            # print("随机选择的函数：", [func.__name__ for func in random_functions])
            # 执行被选择的函数
            blended_image = random_function(image, img_tamp, img_tamp_mask_only1)
        else:
            blended_image = img_tamp

        # img_tamp_edge = self.merge_mask_edge(img_tamp_mask, img_tamp_edge)
        # img_tamp_mask[img_tamp_mask > 127] = 255
        # img_tamp_mask[img_tamp_mask <= 127] = 0
        # img_tamp_edge[img_tamp_edge > 127] = 255
        # img_tamp_edge[img_tamp_edge <= 127] = 0

        return blended_image, img_tamp_mask_only1, img_tamp_edge_only1

    def make_copymove_tamp_real(self, image, mask):
        if mask.ndim >= 3:
            mask = np.squeeze(mask)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # mask[mask > 127] = 255
        # mask[mask <= 127] = 0
        # 获取图片大小，保证mask和图片大小一致
        h, w, c = image.shape
        mask = cv2.resize(mask, (w, h))
        tamp_pixel_percent = np.count_nonzero(mask) / h / w
        image_masked = cv2.bitwise_and(image, image, mask=mask)
        image_resized, mask_resized = self.random_scale(image_masked, mask, tamp_pixel_percent)  # 随机缩放

        image_rotated, mask_rotated = self.random_rotate(image_resized, mask_resized, 0, 359)  # 随机旋转
        image_rotated = self.random_brightness_contrast(image_rotated, mask_rotated)  # 随机亮度对比度

        crop_x1, crop_y1, crop_w, crop_h = self.find_img_boundary(image_rotated, mask_rotated)
        image_croped = image_rotated[crop_y1:crop_y1 + crop_h, crop_x1:crop_x1 + crop_w]
        mask_croped = mask_rotated[crop_y1:crop_y1 + crop_h, crop_x1:crop_x1 + crop_w]

        h_croped, w_croped, c_croped = image_croped.shape
        if w - w_croped > 0:
            paste_w = w - w_croped
        else:
            paste_w = w
        if h - h_croped > 0:
            paste_h = h - h_croped
        else:
            paste_h = h
        paste_x1 = random.randint(0, paste_w)
        paste_y1 = random.randint(0, paste_h)
        # image paste
        img1 = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        img2 = Image.fromarray(cv2.cvtColor(image_croped, cv2.COLOR_BGR2RGB))
        mask_croped_img = Image.fromarray(mask_croped)
        img1.paste(img2, (paste_x1, paste_y1), mask=mask_croped_img)
        img_tamp = np.array(img1)
        # mask paste
        mask_temp = np.zeros_like(mask)
        mask_temp_image = Image.fromarray(mask_temp)
        img1 = Image.fromarray(mask)
        img2 = Image.fromarray(mask_croped)
        mask_croped_img = Image.fromarray(mask_croped)
        img1.paste(img2, (paste_x1, paste_y1), mask=mask_croped_img)
        img_tamp_mask = np.array(img1)
        # img_tamp_mask[img_tamp_mask > 127] = 255
        # img_tamp_mask[img_tamp_mask <= 127] = 0
        mask_temp_image.paste(img2, (paste_x1, paste_y1), mask=mask_croped_img)
        img_tamp_mask_only1 = np.array(mask_temp_image)
        img_tamp_mask_only1[img_tamp_mask_only1 > 127] = 255
        img_tamp_mask_only1[img_tamp_mask_only1 <= 127] = 0
        # # 膨胀
        # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        # img_tamp_mask_only1 = cv2.dilate(img_tamp_mask_only1, kernel, 1)
        # edge paste
        # edge = edge2(mask)
        # edge_croped = edge2(mask_croped)
        # # 去除重叠区域
        # edge_cover = cv2.bitwise_and(edge, img_tamp_mask_only1, mask=None)
        # # 真实图像输入，不需要去除重叠区域
        # edge = edge - edge_cover
        # img1 = Image.fromarray(edge)
        # img2 = Image.fromarray(edge_croped)
        # edge_croped_img = Image.fromarray(edge_croped)
        # img1.paste(img2, (paste_x1, paste_y1), mask=edge_croped_img)
        img_tamp_edge = self.edge2(img_tamp_mask_only1)

        # random color, random illumination, random texture
        if random.random() < self.cit_thresh:
            # 随机选择一个、两个或全部函数并执行
            num_choices = random.randint(1, len(self.random_cit_function_list))  # 随机选择的数量
            random_functions = random.sample(self.random_cit_function_list, num_choices)
            # print("随机选择的函数：", [func.__name__ for func in random_functions])
            # 逐个执行被选择的函数
            for func in random_functions:
                img_tamp = func(img_tamp, img_tamp_mask_only1)
        else:
            pass

        # poisson blending, harmonizer blending
        if random.random() < self.poison_thresh:
            # 随机选择一个函数并执行
            random_function = random.choice(self.blend_function_list)
            # print("随机选择的函数：", [func.__name__ for func in random_functions])
            # 执行被选择的函数
            blended_image = random_function(image, img_tamp, img_tamp_mask_only1)
        else:
            blended_image = img_tamp

        img_tamp_edge = self.merge_mask_edge(img_tamp_mask_only1, img_tamp_edge)
        img_tamp_mask_only1[img_tamp_mask_only1 > 127] = 255
        img_tamp_mask_only1[img_tamp_mask_only1 <= 127] = 0
        img_tamp_edge[img_tamp_edge > 127] = 255
        img_tamp_edge[img_tamp_edge <= 127] = 0

        return blended_image, img_tamp_mask_only1, img_tamp_edge

    def splice_random_paste(self, image, image_bg, mask, mask_bg):
        h_bg, w_bg, c_bg = image_bg.shape
        if mask.ndim >= 3:
            if mask.shape[2] == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            else:
                mask = np.squeeze(mask)
        if mask_bg.ndim >= 3:
            mask_bg = np.squeeze(mask_bg)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # image_bg = cv2.cvtColor(image_bg, cv2.COLOR_RGB2BGR)
        # image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # mask[mask > 127] = 255
        # mask[mask <= 127] = 0
        # 获取图片大小，保证mask和图片大小一致
        h, w, c = image.shape
        mask = cv2.resize(mask, (w, h))
        tamp_pixel_percent = np.count_nonzero(mask) / h / w
        image_masked = cv2.bitwise_and(image, image, mask=mask)
        image_resized, mask_resized = self.random_scale(image_masked, mask, tamp_pixel_percent)  # 随机缩放

        image_rotated, mask_rotated = self.random_rotate(image_resized, mask_resized, 0, 359)  # 随机旋转
        image_rotated = self.random_brightness_contrast(image_rotated, mask_rotated)  # 随机亮度对比度
        crop_x1, crop_y1, crop_w, crop_h = self.find_img_boundary(image_rotated, mask_rotated)
        image_croped = image_rotated[crop_y1:crop_y1 + crop_h, crop_x1:crop_x1 + crop_w]
        mask_croped = mask_rotated[crop_y1:crop_y1 + crop_h, crop_x1:crop_x1 + crop_w]

        h_croped, w_croped, c_croped = image_croped.shape
        if w_bg - w_croped > 0:
            paste_w = w_bg - w_croped
        else:
            paste_w = w_bg
        if h_bg - h_croped > 0:
            paste_h = h_bg - h_croped
        else:
            paste_h = h_bg
        paste_x1 = random.randint(0, paste_w)
        paste_y1 = random.randint(0, paste_h)
        # image paste
        img1 = Image.fromarray(cv2.cvtColor(image_bg, cv2.COLOR_BGR2RGB))
        img2 = Image.fromarray(cv2.cvtColor(image_croped, cv2.COLOR_BGR2RGB))
        mask_croped_img = Image.fromarray(mask_croped)
        img1.paste(img2, (paste_x1, paste_y1), mask=mask_croped_img)
        img_tamp = np.array(img1)
        # mask paste
        mask_temp = np.zeros_like(mask_bg)
        mask_temp_image = Image.fromarray(mask_temp)
        img1 = Image.fromarray(mask_bg)
        img2 = Image.fromarray(mask_croped)
        mask_croped_img = Image.fromarray(mask_croped)
        img1.paste(img2, (paste_x1, paste_y1), mask=mask_croped_img)
        img_tamp_mask = np.array(img1)
        img_tamp_mask[img_tamp_mask > 127] = 255
        img_tamp_mask[img_tamp_mask <= 127] = 0
        mask_temp_image.paste(img2, (paste_x1, paste_y1), mask=mask_croped_img)
        img_tamp_mask_only1 = np.array(mask_temp_image)
        img_tamp_mask_only1[img_tamp_mask_only1 > 127] = 255
        img_tamp_mask_only1[img_tamp_mask_only1 <= 127] = 0
        # # 膨胀
        # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        # img_tamp_mask_only1 = cv2.dilate(img_tamp_mask_only1, kernel, 1)
        # edge paste
        edge = self.edge3(mask_bg)
        edge_croped = self.edge3(mask_croped)
        # 去除重叠区域
        edge_cover = cv2.bitwise_and(edge, img_tamp_mask_only1, mask=None)
        edge = edge - edge_cover
        # 粘贴edge
        img1 = Image.fromarray(edge)
        img2 = Image.fromarray(edge_croped)
        edge_croped_img = Image.fromarray(edge_croped)
        img1.paste(img2, (paste_x1, paste_y1), mask=edge_croped_img)
        img_tamp_edge = np.array(img1)
        # img_tamp_edge = edge3(img_tamp_mask)
        img_tamp_edge[img_tamp_edge > 1] = 255
        img_tamp_edge[img_tamp_edge <= 1] = 0
        return img_tamp, img_tamp_mask, img_tamp_edge, img_tamp_mask_only1

    def make_splice_tamp(self, image_dst, mask_dst, image_path, mask_path):
        # assert len(image_path_list) == len(mask_path_list)
        # # 随机取拼接对象
        # sp_index = np.random.randint(len(image_path_list)//2, len(image_path_list))
        # image_path = image_path_list[sp_index]
        # mask_path = mask_path_list[sp_index]
        image_ori = cv2.imread(image_path)
        if mask_path == 'None':
            mask_ori = ImagePath2SAMSegMask(image_path, image_ori)
            mask_ori = cv2.cvtColor(mask_ori, cv2.COLOR_GRAY2BGR)
        else:
            mask_ori = cv2.imread(mask_path)

        if mask_ori.max() == 0:  # mask内无物体
            print('splice mask has no object')
            return None, None, None, False
        image_dst = cv2.cvtColor(image_dst, cv2.COLOR_RGB2BGR)
        if mask_ori.shape != image_ori.shape:
            mask_ori = cv2.resize(mask_ori, (image_ori.shape[1], image_ori.shape[0]))
            mask_ori[mask_ori > 127] = 255
            mask_ori[mask_ori <= 127] = 0
            # edge_ori = edge3(mask_ori)
        # mask_ori = cv2.cvtColor(mask_ori, cv2.COLOR_BGR2GRAY)

        # random paste
        img_tamp, img_tamp_mask, img_tamp_edge, img_tamp_mask_only1 = self.splice_random_paste(image_ori, image_dst,
                                                                                          mask_ori, mask_dst)
        # RGB

        # random color, random illumination, random texture
        if random.random() < self.cit_thresh:
            # 随机选择一个、两个或全部函数并执行
            num_choices = random.randint(1, len(self.random_cit_function_list))  # 随机选择的数量
            random_functions = random.sample(self.random_cit_function_list, num_choices)
            # print("随机选择的函数：", [func.__name__ for func in random_functions])
            # 逐个执行被选择的函数
            for func in random_functions:
                mask_list = [mask_dst, img_tamp_mask, img_tamp_mask_only1]
                random_mask = random.choice(mask_list)
                img_tamp = func(img_tamp, random_mask)
        else:
            pass

        # poisson blending, harmonizer blending
        if random.random() < self.poison_thresh:
            # 随机选择一个函数并执行
            random_function = random.choice(self.blend_function_list)
            # print("随机选择的函数：", [func.__name__ for func in random_functions])
            # 执行被选择的函数
            blended_image = random_function(image_dst, img_tamp, img_tamp_mask_only1)
        else:
            blended_image = img_tamp

        img_tamp_edge = self.merge_mask_edge(img_tamp_mask, img_tamp_edge)
        img_tamp_mask[img_tamp_mask > 127] = 255
        img_tamp_mask[img_tamp_mask <= 127] = 0
        img_tamp_edge[img_tamp_edge > 127] = 255
        img_tamp_edge[img_tamp_edge <= 127] = 0
        # print('splice poisson blending succeed')
        return blended_image, img_tamp_mask, img_tamp_edge, True

    def make_ip_imginput(self, img_input, input_mask, input_edge, inpaint_mask):  # 输出BGR
        # inpainting
        origin_img = img_input  # RGB
        # origin_img = cv2.resize(origin_img, (680, 512))
        origin_img = self.random_scale_for_inpaint(origin_img)
        tam_mask = input_mask.squeeze()  # (H,W)
        tam_edge = input_edge.squeeze()
        # tam_mask = cv2.cvtColor(tam_mask, cv2.COLOR_GRAY2RGB)
        tam_mask = cv2.resize(tam_mask, (origin_img.shape[1], origin_img.shape[0]))
        tam_edge = cv2.resize(tam_edge, (origin_img.shape[1], origin_img.shape[0]))
        if tam_mask.max() > 1.0:
            tam_mask[tam_mask > 127] = 255
            tam_mask[tam_mask <= 127] = 0
        else:
            tam_mask[tam_mask > 0.5] = 1.
            tam_mask[tam_mask <= 0.5] = 0.

        inpaint_mask = np.array(inpaint_mask.squeeze())
        inpaint_mask = cv2.resize(inpaint_mask, (origin_img.shape[1], origin_img.shape[0]))
        inpaint_edge = self.edge3(inpaint_mask)

        inpaint_method_idx = np.random.randint(10)
        if inpaint_method_idx < 2:  # 2
            inpaintRadius = np.random.randint(3, 10)
            origin_img = cv2.cvtColor(origin_img, cv2.COLOR_RGB2BGR)
            inp_img = cv2.inpaint(origin_img, inpaint_mask, inpaintRadius, cv2.INPAINT_TELEA)
            tam_mask = cv2.bitwise_or(tam_mask, inpaint_mask)
            edge_cover = cv2.bitwise_and(tam_edge, inpaint_mask, mask=None)
            edge_cover = tam_edge - edge_cover
            tam_edge = cv2.bitwise_or(edge_cover, inpaint_edge)
            return inp_img, tam_mask, tam_edge, 3
        elif inpaint_method_idx >= 8:  # 8
            inpaintRadius = np.random.randint(3, 10)
            origin_img = cv2.cvtColor(origin_img, cv2.COLOR_RGB2BGR)
            inp_img = cv2.inpaint(origin_img, inpaint_mask, inpaintRadius, cv2.INPAINT_NS)
            tam_mask = cv2.bitwise_or(tam_mask, inpaint_mask)
            edge_cover = cv2.bitwise_and(tam_edge, inpaint_mask, mask=None)
            edge_cover = tam_edge - edge_cover
            tam_edge = cv2.bitwise_or(edge_cover, inpaint_edge)
            return inp_img, tam_mask, tam_edge, 3
        else:
            # inp_img = make_inpaint_tamp(origin_img, inpaint_mask)
            # print(origin_img.shape, inpaint_mask.shape)
            inp_img = self.model_gc.gc_inpaint_one(origin_img, inpaint_mask)
            inp_img = cv2.cvtColor(inp_img, cv2.COLOR_RGB2BGR)
            torch.cuda.empty_cache()
            gc.collect()
            # time.sleep(3.0)
            tam_mask = cv2.bitwise_or(tam_mask, inpaint_mask)
            edge_cover = cv2.bitwise_and(tam_edge, inpaint_mask, mask=None)
            edge_cover = tam_edge - edge_cover
            tam_edge = cv2.bitwise_or(edge_cover, inpaint_edge)
            return inp_img, tam_mask, tam_edge, 3

    def __getitem__(self, item):

        if self.manipulate_dataset:
            manipulat_flag = np.random.randint(100)  # [0,100)随机数，控制百分之多少可能性进行篡改
            self.tamper_mode = np.random.randint(7)  # 0:cp, 1:sp, 2:ip, 3:cp+sp, 4:cp+ip, 5:sp+ip, 6:cp+sp+ip
        else:
            manipulat_flag = -1
        if self.postprocess_dataset:
            postprocess_flag = np.random.randint(100)  # [0,100)随机数，控制百分之多少可能性进行后处理
            self.postprocess_mode = np.random.randint(2)  # 0:柏松, 1:调和
        else:
            postprocess_flag = -1
        # ----------
        # Read label
        # ----------
        input_label = int(self.labels[item])

        # ----------
        # Read input image  （H，W，3）RGB
        # ----------
        input_file_name = self.input_image_paths[item]

        # input = cv2.cvtColor(cv2.imread(input_file_name), cv2.COLOR_BGR2RGB)
        input = Image.open(input_file_name)
        if (not self.val):
            if input.size[0] > 1024 or input.size[1] > 1024:  # 大图缩小，加快处理速度
                # input = cv2.resize(input, (1024, 1024))
                input = input.resize((1024, 1024), Image.Resampling.BICUBIC)
        input = np.asarray(input)
        if input.ndim == 2:  # 灰度图转RGB
            input = cv2.cvtColor(input, cv2.COLOR_GRAY2RGB)
        elif input.ndim == 3:  # 保证输入为（H，W，3）三通道RGB图
            if input.shape[2] == 3:
                pass
            elif input.shape[2] == 1:  # 灰度图转RGB
                input = cv2.cvtColor(input.squeeze(), cv2.COLOR_GRAY2RGB)
            elif input.shape[2] == 4:  # RGBA转RGB
                input = cv2.cvtColor(input, cv2.COLOR_RGBA2RGB)
            else:
                raise RuntimeError(f'input image shape: {input.shape}')
        else:
            raise RuntimeError(f'input image shape: {input.shape}')

        height, width, _ = input.shape

        # ----------
        # Read mask （H，W，1）
        # ----------
        mask_file_name = self.mask_image_paths[item]
        if mask_file_name != 'None':
            mask = cv2.imread(mask_file_name, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros([height, width, 1], np.uint8)  # a totally black mask for real image
        if mask.ndim < 3:
            mask = np.expand_dims(mask, axis=2)
        # 纠正label
        if input_label != 0 and mask.max() == 0:  # 如果mask全黑，则设置label为0
            input_label = 0
        if input_label == 0 and mask.max() != 0:
            input_label = 4  # 未知类别

        # ----------
        # Read edge （H，W，1）
        # ----------
        edge_file_name = self.edge_image_paths[item]
        if edge_file_name != 'None':
            edge = cv2.imread(edge_file_name, cv2.IMREAD_GRAYSCALE)
        else:
            edge = np.zeros([height, width, 1], np.uint8)  # a totally black edge mask for real image
        if edge.ndim < 3:
            edge = np.expand_dims(edge, axis=2)

        # 进行篡改
        if not self.val and self.manipulate_dataset:
            if 0 <= manipulat_flag < self.manipulate_thresh:  # 20%可能性进行篡改
                # if self.tamper_mode == 0:  # cp
                #     tam_img, tam_mask, tam_edge, tam_label = self.do_cp(input, mask, edge, input_label, input_file_name)
                # elif self.tamper_mode == 1:  # sp
                #     tam_img, tam_mask, tam_edge, tam_label = self.do_sp(input, mask, edge, input_label)
                # elif self.tamper_mode == 2:  # ip
                #     tam_img, tam_mask, tam_edge, tam_label = self.do_ip(input, mask, edge, input_label)
                # elif self.tamper_mode == 3:  # cp+sp
                #     tam_img, tam_mask, tam_edge, tam_label = self.do_cp_sp(input, mask, edge, input_label)
                # elif self.tamper_mode == 4:  # cp+ip
                #     tam_img, tam_mask, tam_edge, tam_label = self.do_cp_ip(input, mask, edge, input_label)
                # elif self.tamper_mode == 5:  # sp+ip
                #     tam_img, tam_mask, tam_edge, tam_label = self.do_sp_ip(input, mask, edge, input_label)
                # elif self.tamper_mode == 6:  # cp+sp+ip
                #     tam_img, tam_mask, tam_edge, tam_label = self.do_cp_sp_ip(input, mask, edge, input_label)
                # else:
                #     raise NotImplementedError(f'tamper mode not in 0-6: {self.tamper_mode}')
                # 随机选择一个、两个或全部函数并执行
                num_choices = random.randint(1, len(self.tamp_function_list))  # 随机选择的数量
                tamp_random_functions = random.sample(self.tamp_function_list, num_choices)
                # print("随机选择的篡改函数：", [func.__name__ for func in tamp_random_functions])
                # 逐个执行被选择的函数
                for tamp_func in tamp_random_functions:
                    # if tamp_func.__name__
                    tam_img, tam_mask, tam_edge, tam_label = tamp_func(input, mask, edge, input_label, input_file_name)
                    input = np.array(tam_img)
                    mask = np.array(tam_mask)
                    if mask.ndim < 3:
                        mask = np.expand_dims(mask, axis=2)
                    edge = np.array(tam_edge)
                    if edge.ndim < 3:
                        edge = np.expand_dims(edge, axis=2)
                    input_label = int(tam_label)

        # 进行调和
        # 目前在cp,sp,ip三种篡改方式函数里进行融合
        if not self.val and self.manipulate_dataset:
            if 0 <= postprocess_flag < self.blend_thresh:  # 20%可能性进行调和
                if input_label > 0:  # 有篡改图像
                    if random.random() < 0.5:  # Harmonize
                        input = self.model_harmonizer.harmonize_one(input, mask.squeeze())
                    else:  # Enhance
                        input = self.model_enhancer.enhance_one(input)
                else:   # 无篡改图像
                    input = self.model_enhancer.enhance_one(input)


        # print(input_file_name)
        # print(input.shape)
        # print(mask.shape)
        # print(edge.shape)
        # ----------
        # Apply transform (the same for both image and mask)
        # ----------
        seed = np.random.randint(2147483647) # make a seed with numpy generator

        random.seed(seed)
        # image_temp = copy.deepcopy(input)
        input = self.transform_randaug(Image.fromarray(input))
        input = np.asarray(input)
        random.seed(seed)
        mask = self.transform_randaug(Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)))
        mask = cv2.cvtColor(np.asarray(mask), cv2.COLOR_RGB2GRAY)

        if (not self.val):
            # 二值化
            mask[mask > 127] = 255
            mask[mask <= 127] = 0
            edge[edge > 127] = 255
            edge[edge <= 127] = 0
            # 最后检查
            if mask.max() == 0:
                input_label = 0
            else:
                input_label = 1

        if (not self.val):
            if random.random() < 1.1:  # 进行模糊等数据增强
                transformed_train = self.transform_train(image=input, mask=mask, edge=edge)
                input = transformed_train['image']
                mask = transformed_train['mask']
                edge = transformed_train['edge']
            else:  # 不进行模糊等数据增强
                transformed_train = self.transform_val(image=input, mask=mask, edge=edge)
                input = transformed_train['image']
                mask = transformed_train['mask']
                edge = transformed_train['edge']
        else:
            transformed_val = self.transform_val(image=input, mask=mask, edge=edge)
            input = transformed_val['image']
            mask = transformed_val['mask']
            edge = transformed_val['edge']

        edge = edge[0::4, 0::4, :]  # edge mask缩小4倍
        # edge = torch.nn.functional.interpolate(edge, size=self.image_size // 4, mode='nearest')

        input = input.astype('float')
        mask = mask.astype('float')
        edge = edge.astype('float')
        if input.max() > 1.0:
            input = input / 255.0
        if mask.max() > 1.0:
            mask = mask / 255.0
        if edge.max() > 1.0:
            edge = edge / 255.0
        if input_label > 1:  # 二分类
            input_label = 1


        input = self.transform(input)
        mask = self.transform_mask(mask)
        edge = self.transform_mask(edge)

        # mean = torch.Tensor(np.array([0.5, 0.5, 0.5])[:, np.newaxis, np.newaxis])
        # mean = mean.expand(3, self.image_size, self.image_size).to('cuda:0')
        # std = torch.Tensor(np.array([0.5, 0.5, 0.5])[:, np.newaxis, np.newaxis])
        # std = std.expand(3, self.image_size, self.image_size).to('cuda:0')
        if not self.val and self.osn_noise:
            # input = input.unsqueeze(0).to('cuda:0')
            if random.random() < 0.5:
                input = self.model_osn.process(input, True)
            else:
                input = self.model_osn.process(input, False)

            torch.cuda.empty_cache()
            gc.collect()
            input = torch.from_numpy(input.squeeze(0))
        # else:
        #     input.sub_(mean).div_(std)
        #     # input = self.model_osn.process(input, False)
        # input = input.squeeze(0).detach().cpu()


        return input, mask, edge, input_label

    def __len__(self):
        return len(self.input_image_paths)
