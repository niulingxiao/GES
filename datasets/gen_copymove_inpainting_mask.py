# import the necessary packages
import os

import numpy as np
import argparse
import cv2
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# image_root = '/home/zzt/Python_Projects/MVSS-Net/MVSS-Net-train/data/Casiav2/images/Au/'
# images = os.listdir(image_root)
# mask_saveroot = '../masks/'


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
    masked = cv2.bitwise_and(image, image, mask=mask)
    # cv2.imshow("Mask Applied to Image", masked)
    # cv2.waitKey(0)

    # now, let's make a circular mask with a radius of 100 pixels and
    # apply the mask again
    mask_circle = np.zeros(image.shape[:2], dtype="uint8")
    mask_circle = cv2.circle(mask_circle, ((x1 + x2) // 2, (y1 + y2) // 2), length // 2, 255, -1)
    masked_circle = cv2.bitwise_and(image, image, mask=mask_circle)

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


def gen_mask(image_path):

    # load the original input image and display it to our screen
    image = cv2.imread(image_path)

    h = image.shape[0]
    w = image.shape[1]
    length = np.random.randint((w // 100 * 5)+1, (w // 100 * 80)+1)  # 5%-75%的width
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
    masked = cv2.bitwise_and(image, image, mask=mask)
    # cv2.imshow("Mask Applied to Image", masked)
    # cv2.waitKey(0)

    # now, let's make a circular mask with a radius of 100 pixels and
    # apply the mask again
    mask_circle = np.zeros(image.shape[:2], dtype="uint8")
    mask_circle = cv2.circle(mask_circle, ((x1 + x2) // 2, (y1 + y2) // 2), length // 2, 255, -1)
    masked_circle = cv2.bitwise_and(image, image, mask=mask_circle)

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


if __name__ == '__main__':
    for cnt, image_name in enumerate(tqdm(images, mininterval=1)):
        image_path = os.path.join(image_root, image_name)

        # load the original input image and display it to our screen
        image = cv2.imread(image_path)
        # cv2.imshow("Original", image)

        # a mask is the same size as our image, but has only two pixel
        # values, 0 and 255 -- pixels with a value of 0 (background) are
        # ignored in the original image while mask pixels with a value of
        # 255 (foreground) are allowed to be kept
        h = image.shape[0]
        w = image.shape[1]
        length = np.random.randint(w // 10 * 1, w // 10 * 6)
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
        masked = cv2.bitwise_and(image, image, mask=mask)
        # cv2.imshow("Mask Applied to Image", masked)
        # cv2.waitKey(0)

        # now, let's make a circular mask with a radius of 100 pixels and
        # apply the mask again
        mask_circle = np.zeros(image.shape[:2], dtype="uint8")
        mask_circle = cv2.circle(mask_circle, ((x1+x2)//2, (y1+y2)//2), length//2, 255, -1)
        masked_circle = cv2.bitwise_and(image, image, mask=mask_circle)

        # show the output images
        # cv2.imshow("Circular Mask", mask_circle)
        # cv2.imshow("Mask Applied to Image", masked_circle)
        # cv2.waitKey(0)

        flag = np.random.randint(2)
        # print(flag)
        if flag == 0:
            mask_rect = Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
            mask_savepath = os.path.join(mask_saveroot, Path(image_name).stem + '_gt.png')
            mask_rect.save(mask_savepath)

        else:
            mask_circ = Image.fromarray(cv2.cvtColor(mask_circle, cv2.COLOR_BGR2RGB))
            mask_savepath = os.path.join(mask_saveroot, Path(image_name).stem + '_gt.png')
            mask_circ.save(mask_savepath)

