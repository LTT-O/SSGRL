import os

import PIL
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from config import prefixPathCOCO, prefixPathVG, prefixPathVOC2007, prefixPathVOC2012
from datasets.coco2014 import COCO2014
from datasets.vg import VG
from datasets.voc2007 import VOC2007
from datasets.voc2012 import VOC2012


def get_graph_and_word_file(args):
    def get_graph_file(path, labels):

        graph = np.zeros((labels.shape[1], labels.shape[1]), dtype=np.float)

        for index in range(labels.shape[0]):
            indexs = np.where(labels[index] == 1)[0]
            for i in indexs:
                for j in indexs:
                    graph[i, j] += 1

        for i in range(labels.shape[1]):
            graph[i] /= graph[i, i]

        np.nan_to_num(graph)
        np.save(path, graph)

    # if args.dataset == 'COCO2014':
    #     GraphFilePath, WordFilePath = './data/coco/prob_train.npy', './data/coco/vectors.npy'
    #     print('Counting label co-ourrence...')
    #     get_graph_file(GraphFilePath, labels)
    GraphFilePath, WordFilePath = None, None
    if args.dataset == 'COCO2014':
        GraphFilePath, WordFilePath = './data/coco/prob_train.npy', './data/coco/vectors.npy'

    elif args.dataset == 'VG':
        GraphFilePath, WordFilePath = './data/vg/graph_500_norm.npy', './data/vg/vg_500_vector.npy'

    elif args.dataset == 'VOC2007':
        GraphFilePath, WordFilePath = './data/voc_devkit/VOC2007/prob_trainval.npy', './data/voc_devkit/VOC2007/voc07_vector.npy'
        print('Counting label co-ourrence...')
        # get_graph_file(GraphFilePath, labels)

    elif args.dataset == 'VOC2012':
        GraphFilePath, WordFilePath = './data/voc_devkit/VOC2012/prob_trainval.npy', './data/voc_devkit/VOC2012/voc12_vector.npy'

    return GraphFilePath, WordFilePath


def get_data_path(dataset):
    if dataset == 'COCO2014':
        prefixPath = prefixPathCOCO

        train_dir, train_anno, train_label = os.path.join(prefixPath, 'train2014'), os.path.join(prefixPath,
                                                                                                 'annotations/instances_train2014.json'), './data/coco/train_label_vectors.npy'
        test_dir, test_anno, test_label = os.path.join(prefixPath, 'val2014'), os.path.join(prefixPath,
                                                                                            'annotations/instances_val2014.json'), './data/coco/val_label_vectors.npy'

    elif dataset == 'VG':
        prefixPath = prefixPathVG
        train_dir, train_anno, train_label = os.path.join(prefixPath,
                                                          'VG_100K'), './data/vg/train_list_500.txt', './data/vg/vg_category_500_labels_index.json'
        test_dir, test_anno, test_label = os.path.join(prefixPath,
                                                       'VG_100K'), './data/vg/test_list_500.txt', './data/vg/vg_category_500_labels_index.json'

    elif dataset == 'VOC2007':
        prefixPath = prefixPathVOC2007
        train_dir, train_anno, train_label = os.path.join(prefixPath, 'JPEGImages'), os.path.join(prefixPath,
                                                                                                  'ImageSets/Main/trainval.txt'), os.path.join(
            prefixPath, 'Annotations')
        test_dir, test_anno, test_label = os.path.join(prefixPath, 'JPEGImages'), os.path.join(prefixPath,
                                                                                               'ImageSets/Main/test.txt'), os.path.join(
            prefixPath, 'Annotations')

    elif dataset == 'VOC2012':
        prefixPath = prefixPathVOC2012
        train_dir, train_anno, train_label = os.path.join(prefixPath, 'JPEGImages'), os.path.join(prefixPath,
                                                                                                  'ImageSets/Main/trainval.txt'), os.path.join(
            prefixPath, 'Annotations')
        test_dir, test_anno, test_label = os.path.join(prefixPath, 'JPEGImages'), os.path.join(prefixPath,
                                                                                               'ImageSets/Main/test.txt'), 'None'

    return train_dir, train_anno, train_label, \
           test_dir, test_anno, test_label


def get_data_loader(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    randomCropList = [transforms.RandomCrop(Size) for Size in
                      [640, 576, 512, 448, 384, 320]] if args.scaleSize == 640 else \
        [transforms.RandomCrop(Size) for Size in [512, 448, 384, 320, 256]]
    train_data_transform = transforms.Compose(
        [transforms.Resize((args.scaleSize, args.scaleSize), interpolation=PIL.Image.BICUBIC),
         transforms.RandomChoice(randomCropList),
         transforms.Resize((args.cropSize, args.cropSize), interpolation=PIL.Image.BICUBIC),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         normalize])

    test_data_transform = transforms.Compose(
        [transforms.Resize((args.cropSize, args.cropSize), interpolation=PIL.Image.BICUBIC),
         transforms.ToTensor(),
         normalize])

    train_dir, train_anno, train_label, \
    test_dir, test_anno, test_label = get_data_path(args.dataset)

    if args.dataset == 'COCO2014':
        print("==> Loading COCO2014...")
        train_set = COCO2014('train',
                             train_dir, train_anno, train_label,
                             input_transform=train_data_transform)
        test_set = COCO2014('val',
                            test_dir, test_anno, test_label,
                            input_transform=test_data_transform)

    elif args.dataset == 'VG':
        print("==> Loading VG...")
        train_set = VG('train',
                       train_dir, train_anno, train_label,
                       input_transform=train_data_transform)
        test_set = VG('val',
                      test_dir, test_anno, test_label,
                      input_transform=test_data_transform)

    elif args.dataset == 'VOC2007':
        print("==> Loading VOC2007...")
        train_set = VOC2007('train',
                            train_dir, train_anno, train_label,
                            input_transform=train_data_transform)
        test_set = VOC2007('val',
                           test_dir, test_anno, test_label,
                           input_transform=test_data_transform)

    elif args.dataset == 'VOC2012':
        print("==> Loading VOC2012...")
        train_set = VOC2012(train_dir, train_anno, train_data_transform, train_label)
        test_set = VOC2012(test_dir, test_anno, test_data_transform, test_label)

    train_loader = DataLoader(dataset=train_set,
                              num_workers=args.workers,
                              batch_size=args.batchSize,
                              pin_memory=True,
                              drop_last=True,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_set,
                             num_workers=args.workers,
                             batch_size=args.batchSize,
                             pin_memory=True,
                             drop_last=False,
                             shuffle=False)

    return train_loader, test_loader
