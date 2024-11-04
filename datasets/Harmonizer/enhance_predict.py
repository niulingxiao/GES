import os
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torchvision.transforms.functional as tf

from datasets.Harmonizer.src import model


class Enhancer():
    def __init__(self):
        # pre-defined arguments
        self.gpu = 'cuda:0'

        # create/load the harmonizer model
        print('Create/load Harmonizer...')
        self.enhancer = model.Enhancer()
        self.enhancer = self.enhancer.to(self.gpu)
        self.enhancer.load_state_dict(torch.load('./datasets/Harmonizer/pretrained/enhancer.pth'), strict=True)
        self.enhancer.eval()

    def enhance_one(self, img):
        # load the example
        original = Image.fromarray(img).convert('RGB')

        original = tf.to_tensor(original)[None, ...]
        # NOTE: all pixels in the mask are equal to 1 as the mask is not used in image enhancement
        mask = original * 0 + 1

        original = original.to(self.gpu)
        mask = mask.to(self.gpu)

        # harmonization
        with torch.no_grad():
            arguments = self.enhancer.predict_arguments(original, mask)
            enhanced = self.enhancer.restore_image(original, mask, arguments)[-1]

        # save the result
        enhanced = np.transpose(enhanced[0].cpu().numpy(), (1, 2, 0)) * 255
        enhanced = Image.fromarray(enhanced.astype(np.uint8))
        # enhanced.save(os.path.join(args.example_path, 'enhanced', example))
        enhanced = np.asarray(enhanced)  # RGB

        # print('Finished.')
        return enhanced


