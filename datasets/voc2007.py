import os
import sys
sys.path.append(os.path.join( os.path.dirname(os.path.abspath(__file__)), '..'))


import numpy as np
from PIL import Image
import xml.dom.minidom
from xml.dom.minidom import parse

import torch
import torch.utils.data as data

category_info = {'aeroplane':0, 'bicycle':1, 'bird':2, 'boat':3, 'bottle':4,
                 'bus':5, 'car':6, 'cat':7, 'chair':8, 'cow':9,
                 'diningtable':10, 'dog':11, 'horse':12, 'motorbike':13, 'person':14,
                 'pottedplant':15, 'sheep':16, 'sofa':17, 'train':18, 'tvmonitor':19}

class VOC2007(data.Dataset):

    def __init__(self, mode,
                 img_dir, anno_path, labels_path,
                 input_transform=None):

        assert mode in ('train', 'val')

        self.mode = mode
        self.input_transform = input_transform

        self.img_names  = []
        with open(anno_path, 'r') as f:
             self.img_names = f.readlines()
        self.img_dir = img_dir
        
        self.labels = []
        for name in self.img_names:
            label_file = os.path.join(labels_path,name[:-1]+'.xml')
            label_vector = np.zeros(20)
            DOMTree = xml.dom.minidom.parse(label_file)
            root = DOMTree.documentElement
            objects = root.getElementsByTagName('object')  
            for obj in objects:
                if (obj.getElementsByTagName('difficult')[0].firstChild.data) == '1':
                    continue
                tag = obj.getElementsByTagName('name')[0].firstChild.data.lower()
                label_vector[int(category_info[tag])] = 1.0
            self.labels.append(label_vector)

        # labels : numpy.ndarray, shape->(len(self.img_names), 20)
        # value range->(0 means label don't exist, 1 means label exist)
        self.labels = np.array(self.labels).astype(np.int)
        self.labels[self.labels < 1] = 0

    def __getitem__(self, index):
        name = self.img_names[index][:-1]+'.jpg'
        input = Image.open(os.path.join(self.img_dir, name)).convert('RGB')
        if self.input_transform:
            input = self.input_transform(input)
        return index, input, self.labels[index] 

    def __len__(self):
        return len(self.img_names)


# =============================================================================
# Help Functions
# =============================================================================
def changeLabelProportion(labels, label_proportion):

    # Set Random Seed
    np.random.seed(0)

    mask = np.random.random(labels.shape)
    mask[mask < label_proportion] = 1
    mask[mask < 1] = 0
    label = mask * labels

    assert label.shape == labels.shape

    return label


# =============================================================================
# Check Code
# =============================================================================
def main():
    from config import prefixPathVOC2007

    train_dir, train_anno, train_label = os.path.join(prefixPathVOC2007, 'JPEGImages'), \
                                         os.path.join(prefixPathVOC2007, 'ImageSets/Main/trainval.txt'), \
                                         os.path.join(prefixPathVOC2007, 'Annotations')

    dataset = VOC2007('train',
                       train_dir, train_anno, train_label,
                       label_proportion=0.5)
    print(dataset.labels[0, :20])
    print(dataset.changedLabels[0, :20])

    dataset_ = VOC2007('train',
                       train_dir, train_anno, train_label,
                       label_proportion=0.5)

    print(np.array_equal(dataset.changedLabels, dataset_.changedLabels))

if __name__ == "__main__":
    main()