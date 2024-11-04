import os
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torchvision.transforms.functional as tf

from datasets.Harmonizer.src import model

class Harmonizer():
    def __init__(self):
        # pre-defined arguments
        self.gpu = 'cuda:0'

        # create/load the harmonizer model
        print('Create/load Harmonizer...')
        self.harmonizer = model.Harmonizer()
        self.harmonizer = self.harmonizer.to(self.gpu)
        self.harmonizer.load_state_dict(torch.load('./datasets/Harmonizer/pretrained/harmonizer.pth'), strict=True)
        self.harmonizer.eval()
    def harmonize_one(self, img, mask):

        # load the example
        comp = Image.fromarray(img).convert('RGB')
        mask = Image.fromarray(mask).convert('1')
        if comp.size[0] != mask.size[0] or comp.size[1] != mask.size[1]:
            print('The size of the composite image and the mask are inconsistent')
            raise RuntimeError('The size of the composite image and the mask are inconsistent')

        comp = tf.to_tensor(comp)[None, ...]
        mask = tf.to_tensor(mask)[None, ...]

        comp = comp.to(self.gpu)
        mask = mask.to(self.gpu)

        # harmonization
        with torch.no_grad():
            arguments = self.harmonizer.predict_arguments(comp, mask)
            harmonized = self.harmonizer.restore_image(comp, mask, arguments)[-1]

        # save the result
        harmonized = np.transpose(harmonized[0].cpu().numpy(), (1, 2, 0)) * 255
        harmonized = Image.fromarray(harmonized.astype(np.uint8))
        # harmonized.save(os.path.join(args.example_path, 'harmonized', example))
        harmonized = np.asarray(harmonized)  # RGB

        # print('Finished.')
        return harmonized


