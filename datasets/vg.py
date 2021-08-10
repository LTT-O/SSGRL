import os
import sys
import json
import random
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.datasets as datasets


class VG(data.Dataset):

    def __init__(self, mode,
                 image_dir, anno_path, labels_path,
                 input_transform=None):

        assert mode in ('train', 'val')

        self.mode = mode
        self.input_transform = input_transform

        self.img_dir = image_dir
        self.imgName_path = anno_path
        self.img_names = open(self.imgName_path, 'r').readlines()

        # labels : numpy.ndarray, shape->(len(vg), 200)
        # value range->(0 means label don't exist, 1 means label exist)
        self.labels_path = labels_path
        _ = json.load(open(self.labels_path, 'r'))
        self.labels = np.zeros((len(self.img_names), 200)).astype(np.int)
        for i in range(len(self.img_names)):
            self.labels[i][_[self.img_names[i][:-1]]] = 1

    def __getitem__(self, index):
        name = self.img_names[index][:-1]
        input = Image.open(os.path.join(self.img_dir, name)).convert('RGB')
        if self.input_transform:
           input = self.input_transform(input)
        return index, input, self.labels[index]

    def __len__(self):
        return len(self.img_names)

# =============================================================================
# Help Functions
# =============================================================================

# =============================================================================
# Check Code
# =============================================================================
def main():
    pass

if __name__ == "__main__":
    main()