import os
import random
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import shutil
import glob


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


# def edge2(mask):
#     height = mask.shape[0]
#     width = mask.shape[1]
#     mask_around = np.ones_like(mask, dtype="uint8")
#     mask_around = mask_around * 255
#     res_pixels = 1
#     y1 = res_pixels
#     y2 = (height - res_pixels)
#     x1 = res_pixels
#     x2 = (width - res_pixels)
#     mask_around[y1: y2, x1: x2] = 0
#     # cv2.imshow('mask_around', mask_around)
#
#     # 腐蚀
#     # kernel_erod = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
#     # mask_eroded = cv2.erode(mask.copy(), kernel_erod, 10)
#     # cv2.imshow('mask_eroded', mask_eroded)
#
#     # 提取边界法1
#     # SE = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#     # img_grad = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, SE)
#     # cv2.imshow('mask_eroded', img_grad)
#
#     # 提取边界法2
#     # mask_edge = edge(mask)
#     # gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
#
#     # 提取边界法3
#     # blurred = cv2.GaussianBlur(mask, (3, 3), 0)
#     mask_edge = auto_canny(mask)
#     # 膨胀
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     # mask_edge = cv2.dilate(mask_edge, kernel, 2)
#     mask_edge2 = np.zeros_like(mask)
#     mask_edge2[:, :, 0] = mask_edge
#     mask_edge2[:, :, 1] = mask_edge
#     mask_edge2[:, :, 2] = mask_edge
#     # 闭操作
#     # kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#     # mask_edge = cv2.morphologyEx(mask_edge, cv2.MORPH_CLOSE, kernel2, iterations=2)
#     # cv2.imshow('mask', mask_edge2)
#     # print(mask_around.shape)
#     # print(mask_edge2.shape)
#     # print(mask.shape)
#     mask_around = cv2.bitwise_and(mask_around, mask)
#     mask_edge_final = cv2.bitwise_or(mask_around, mask_edge2)
#     # cv2.imshow('mask_final', mask_edge_final)
#     # cv2.waitKey()
#     return mask_edge_final
# def edge2(mask):
#     # mask[mask < 200] = 0
#     # mask[mask >= 200] = 255
#     height = mask.shape[0]
#     width = mask.shape[1]
#
#     mask_around = np.ones_like(mask, dtype="uint8")
#     mask_around = mask_around * 255
#     res_pixels = 6
#     y1 = res_pixels
#     y2 = (height - res_pixels)
#     x1 = res_pixels
#     x2 = (width - res_pixels)
#     mask_around[y1: y2, x1: x2] = 0
#     # cv2.imshow('mask_around', mask_around)
#     # 膨胀
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
#     mask = cv2.dilate(mask, kernel, 2)
#
#     # 腐蚀
#     kernel_erod = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
#     mask = cv2.erode(mask.copy(), kernel_erod, 2)
#     # # cv2.imshow('mask_eroded', mask)
#     #
#     # # 提取边界法1
#     # # SE = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#     # # img_grad = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, SE)
#     # # cv2.imshow('mask_eroded', img_grad)
#     #
#     # # 提取边界法2
#     # # mask_edge = edge(mask)
#     # # gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
#     #
#     # # 提取边界法3
#     # # blurred = cv2.GaussianBlur(mask, (3, 3), 0)
#     mask_edge = auto_canny(mask)
#     # 膨胀
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     mask_edge = cv2.dilate(mask_edge, kernel, 2)
#     # # 闭操作
#     # kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#     # mask_edge = cv2.morphologyEx(mask_edge, cv2.MORPH_CLOSE, kernel2, iterations=2)
#     # 膨胀
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     mask_edge = cv2.dilate(mask_edge, kernel, 2)
#     # # 腐蚀
#     # kernel_erod = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     # mask_edge = cv2.erode(mask_edge, kernel_erod, 2)
#
#     mask_edge2 = np.zeros_like(mask)
#     mask_edge2[:, :, 0] = mask_edge
#     mask_edge2[:, :, 1] = mask_edge
#     mask_edge2[:, :, 2] = mask_edge
#     # # 闭操作
#     # kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#     # mask_edge = cv2.morphologyEx(mask_edge, cv2.MORPH_CLOSE, kernel2, iterations=2)
#     # cv2.imshow('mask', mask_edge2)
#     # print(mask_around.shape)
#     # print(mask_edge2.shape)
#     # print(mask.shape)
#     mask_around = cv2.bitwise_and(mask_around, mask)
#     mask_edge_final = cv2.bitwise_or(mask_around, mask_edge2)
#     # cv2.imshow('mask_final', mask_edge_final)
#     # cv2.waitKey()
#     return mask_edge_final
def edge2(mask):
    # mask[mask < 200] = 0
    mask[mask > 0] = 255
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
    # # cv2.imshow('mask_eroded', mask)
    #
    # # 提取边界法1
    # # SE = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # # img_grad = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, SE)
    # # cv2.imshow('mask_eroded', img_grad)
    #
    # # 提取边界法2
    # # mask_edge = edge(mask)
    # # gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    #
    # # 提取边界法3
    # # blurred = cv2.GaussianBlur(mask, (3, 3), 0)
    mask_edge = auto_canny(mask)
    # 膨胀
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_edge = cv2.dilate(mask_edge, kernel, 2)
    # # 闭操作
    # kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # mask_edge = cv2.morphologyEx(mask_edge, cv2.MORPH_CLOSE, kernel2, iterations=2)
    # 膨胀
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_edge = cv2.dilate(mask_edge, kernel, 2)
    # # 腐蚀
    # kernel_erod = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # mask_edge = cv2.erode(mask_edge, kernel_erod, 2)

    mask_edge2 = np.zeros_like(mask)
    if len(mask.shape) == 3:
        mask_edge2[:, :, 0] = mask_edge
        mask_edge2[:, :, 1] = mask_edge
        mask_edge2[:, :, 2] = mask_edge
    elif len(mask.shape) == 2:
        mask_edge2[:, :] = mask_edge
    else:
        raise NotImplementedError(f'mask shape {len(mask.shape)} not implemented')

    # # 闭操作
    # kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # mask_edge = cv2.morphologyEx(mask_edge, cv2.MORPH_CLOSE, kernel2, iterations=2)
    # cv2.imshow('mask', mask_edge2)
    # print(mask_around.shape)
    # print(mask_edge2.shape)
    # print(mask.shape)
    mask_around = cv2.bitwise_and(mask_around, mask)
    mask_edge_final = cv2.bitwise_or(mask_around, mask_edge2)
    # cv2.imshow('mask_final', mask_edge_final)
    # cv2.waitKey()
    return mask_edge_final


def edge3(mask):
    # mask[mask < 200] = 0
    mask[mask > 0] = 255
    height = mask.shape[0]
    width = mask.shape[1]
    mask_around = np.ones_like(mask, dtype="uint8")
    mask_around = mask_around * 255
    res_pixels = 4
    y1 = res_pixels
    y2 = (height - res_pixels)
    x1 = res_pixels
    x2 = (width - res_pixels)
    mask_around[y1: y2, x1: x2] = 0
    # cv2.imshow('mask_around', mask_around)
    # # 膨胀
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    # mask = cv2.dilate(mask, kernel, 2)
    #
    # # 腐蚀
    # kernel_erod = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    # mask = cv2.erode(mask, kernel_erod, 2)
    # # cv2.imshow('mask_eroded', mask)
    #
    # # 提取边界法1
    # # SE = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # # img_grad = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, SE)
    # # cv2.imshow('mask_eroded', img_grad)
    #
    # # 提取边界法2
    # # mask_edge = edge(mask)
    # # gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    #
    # # 提取边界法3
    # # blurred = cv2.GaussianBlur(mask, (3, 3), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    mask_delate = cv2.dilate(mask, kernel, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    mask_erode = cv2.erode(mask, kernel, 2)
    mask_edge = mask_delate - mask_erode
    # 膨胀
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_edge = cv2.dilate(mask_edge, kernel, 2)
    # # 闭操作
    # kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # mask_edge = cv2.morphologyEx(mask_edge, cv2.MORPH_CLOSE, kernel2, iterations=2)
    # # 膨胀
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # mask_edge = cv2.dilate(mask_edge, kernel, 2)
    # # 腐蚀
    # kernel_erod = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # mask_edge = cv2.erode(mask_edge, kernel_erod, 2)

    # # 闭操作
    # kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # mask_edge = cv2.morphologyEx(mask_edge, cv2.MORPH_CLOSE, kernel2, iterations=2)
    # cv2.imshow('mask', mask_edge2)
    # print(mask_around.shape)
    # print(mask_edge2.shape)
    # print(mask.shape)
    mask_around = cv2.bitwise_and(mask_around, mask)
    mask_edge_final = cv2.bitwise_or(mask_around, mask_edge)
    # cv2.imshow('mask_final', mask_edge_final)
    # cv2.waitKey()
    return mask_edge_final


if __name__ == '__main__':
    mask_rootpath = '/home/zzt/pan1/DATASETS/Casiav1/画图用'
    mask_paths = os.listdir(mask_rootpath)
    mask_paths = sorted(mask_paths)

    for cnt, mask_name in enumerate(tqdm(mask_paths)):
        mask_path = os.path.join(mask_rootpath, mask_name)
        mask = cv2.imread(mask_path)
        mask = edge3(mask)
        # cv2.imshow('mask_final', mask)
        # cv2.waitKey()
        save_path = '/home/zzt/pan1/DATASETS/Casiav1/画图用/zebra_mask_edge4.png'
        cv2.imwrite(save_path, mask)
        print('done!')


