import cv2
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
import torch

color_map = []


def decode_segmap(image, nc=4):  # image为(H,W)
  label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=copymove红, 2=splice绿, 3=inpainting蓝
               (128, 0, 0), (0, 128, 0), (0, 0, 128)])
  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
  for l in range(0, nc):
    idx = image == l
    r[idx] = label_colors[l, 0]
    g[idx] = label_colors[l, 1]
    b[idx] = label_colors[l, 2]
  rgb = np.stack([r, g, b], axis=2)
  return rgb


def decode_segmap_train(images, nc=4):  # images为(N,C,H,W)
    images_rgb = []
    # print(images.shape)
    for image in images:
        image = torch.squeeze(image)
        # print(image.shape)
        image_rgb = decode_segmap(image, nc)
        images_rgb.append(image_rgb)
    images_rgb = np.array(images_rgb)
    images_rgb = torch.from_numpy(images_rgb)
    images_rgb = images_rgb.permute(0, 3, 1, 2)
    # print(images_rgb.shape)
    return images_rgb


if __name__ == '__main__':
    pic_folder = '../data/defacto12k/masks_multiclass/'
    picnames = os.listdir(pic_folder)
    random.shuffle(picnames)

    for picname in tqdm(picnames):
        filename = os.path.join(pic_folder, picname)
        mask = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        # print(mask.shape)
        cv2.imshow('gray', mask)
        mask_rgb = decode_segmap(mask)
        mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
        cv2.imshow('rgb', mask_rgb)
        cv2.waitKey()
