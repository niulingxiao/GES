import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

index = ['real',
    'copymove',
    'splice',
    'inpaint']

color2index = {
    0 : 'real',
    1 : 'copymove',
    2 : 'splice',
    3 : 'inpaint',

}

paths_trainfile = '../data/defacto12k/paths_train_defacto12K_multiclass.txt'
paths_testfile = '../data/defacto12k/paths_test_defacto4K_multiclass.txt'
paths_valfile = '../data/defacto12k/paths_val_defacto4K_multiclass.txt'

mask_copymove_pixelnum = 0
mask_splice_pixelnum = 0
mask_inpaint_pixelnum = 0
mask_real_pixelnum = 0

edge_copymove_pixelnum = 0
edge_splice_pixelnum = 0
edge_inpaint_pixelnum = 0
edge_real_pixelnum = 0

with open(paths_trainfile, 'r') as f:
    lines = f.readlines()
    for l in tqdm(lines, mininterval=1):
        parts = l.rstrip().split(' ')
        input_image_path = parts[0]
        mask_image_path = parts[1]
        edge_image_path = parts[2]
        label_str = parts[3]
        label = int(label_str)

        if os.path.exists(mask_image_path):
            mask = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
            edge = cv2.imread(edge_image_path, cv2.IMREAD_GRAYSCALE)
            mask_pixel_num = len(mask[mask==label])
            edge_pixel_num = len(edge[edge==label])
            mask_real_pixel_num = len(mask[mask==0])
            edge_real_pixel_num = len(edge[edge==0])

            assert label == 1 or 2 or 3
            if label == 1:
                mask_copymove_pixelnum += mask_pixel_num
                edge_copymove_pixelnum += edge_pixel_num
            elif label == 2:
                mask_splice_pixelnum += mask_pixel_num
                edge_splice_pixelnum += edge_pixel_num
            elif label == 3:
                mask_inpaint_pixelnum += mask_pixel_num
                edge_inpaint_pixelnum += edge_pixel_num

            mask_real_pixelnum += mask_real_pixel_num
            edge_real_pixelnum += edge_real_pixel_num

            # print(edge.shape)
            # print(len(edge[edge==label]))
            # cv2.imshow('mask',edge)
            # cv2.waitKey()

print(mask_real_pixelnum,
      mask_copymove_pixelnum,
      mask_splice_pixelnum,
      mask_inpaint_pixelnum,
      edge_real_pixelnum,
      edge_copymove_pixelnum,
      edge_splice_pixelnum,
      edge_inpaint_pixelnum
      )

mask_sum = mask_real_pixelnum + mask_copymove_pixelnum + mask_splice_pixelnum + mask_inpaint_pixelnum
mask_portion_real = mask_real_pixelnum/mask_sum
mask_portion_copymove = mask_copymove_pixelnum/mask_sum
mask_portion_splice = mask_splice_pixelnum/mask_sum
mask_portion_inpaint = mask_inpaint_pixelnum/mask_sum
print('mask')
print(mask_portion_real, mask_portion_copymove, mask_portion_splice, mask_portion_inpaint)

edge_sum = edge_real_pixelnum + edge_copymove_pixelnum + edge_splice_pixelnum + edge_inpaint_pixelnum
edge_portion_real = edge_real_pixelnum/edge_sum
edge_portion_copymove = edge_copymove_pixelnum/edge_sum
edge_portion_splice = edge_splice_pixelnum/edge_sum
edge_portion_inpaint = edge_inpaint_pixelnum/edge_sum
print('edge')
print(edge_portion_real, edge_portion_copymove, edge_portion_splice, edge_portion_inpaint)

mask_data = [mask_real_pixelnum, mask_copymove_pixelnum, mask_splice_pixelnum, mask_inpaint_pixelnum]
edge_data = [edge_real_pixelnum, edge_copymove_pixelnum, edge_splice_pixelnum, edge_inpaint_pixelnum]
plt.bar(range(len(mask_data)), mask_data)
plt.xticks(range(len(mask_data)), index)
for i in range(len(mask_data)):
    plt.text(x=i, y=mask_data[i] + 0.2, s='%d' % mask_data[i])
plt.xlabel("label")
plt.ylabel("pixels")
plt.title("i am title")
plt.show()


