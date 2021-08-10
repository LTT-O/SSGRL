"""
Configuration file!
"""

import logging
import warnings
import argparse

warnings.filterwarnings("ignore")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Dataset Path
# =============================================================================
prefixPathCOCO = '/data1/MS-COCO_2014/'
prefixPathVG = '/data1/VG/'
prefixPathVOC2007 = '/data1/PASCAL/voc2007/VOCdevkit/VOC2007/'
prefixPathVOC2012 = '/data1/PASCAL/voc2012/VOCdevkit/VOC2012/'
# =============================================================================

# ClassNum of Dataset
# =============================================================================
_ClassNum = {'COCO2014': 80,
             'VOC2007': 20,
             'VOC2012': 20,
             'VG': 500,
             }
# =============================================================================


# Argument Parse
# =============================================================================
def str2bool(input):
    if isinstance(input, bool):
        return input

    if input.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif input.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def show_args(args):

    logger.info("==========================================")
    logger.info("==========       CONFIG      =============")
    logger.info("==========================================")

    for arg, content in args.__dict__.items():
        logger.info("{}: {}".format(arg, content))

    logger.info("==========================================")
    logger.info("===========        END        ============")
    logger.info("==========================================")

    logger.info("\n")


def arg_parse():
    parser = argparse.ArgumentParser(description='PyTorch multi label Training')

    parser.add_argument('--post', type=str, default='', help='postname of save model')
    parser.add_argument('--mode', type=str, default='SSGRL', choices=['baseline', 'intra-image', 'inter-image', 'fuse-image'], help='mode of experiment (default: baseline)')
    
    parser.add_argument('--printFreq', type=int, default='500', help='number of print frequency (default: 100)')
    parser.add_argument('--dataset', type=str, default='COCO2014', choices=['COCO2014', 'VG', 'VOC2007', 'VOC2012'], help='dataset for training and testing')
    parser.add_argument('--classNum', type=str, default='80', choices=['80', '500', '20', '20'], help='The label type of the data set')

    parser.add_argument('--timeStep', type=int, default=3, help='time step of model (default: 3)')
    parser.add_argument('--useGatedGNN', type=str2bool, default='False', help='whether to use Gated GNN (default: False)')

    parser.add_argument('--cropSize', type=int, default=576, help='size of crop image')
    parser.add_argument('--scaleSize', type=int, default=640, help='size of rescale image')

    parser.add_argument('--pretrainedModel', type=str, default='./data/checkpoint/resnet101.pth', help='path to pretrained model (default: None)')
    parser.add_argument('--resumeModel', type=str, default='None', help='path to resume model (default: None)')

    parser.add_argument('--evaluate', type=str2bool, default='False', help='whether to evaluate model (default: False)')
    parser.add_argument('--workers', type=int, default=8, help='number of data loading workers (default: 8)')

    parser.add_argument('--epochs', type=int, default=20, help='number of total epochs to run (default: 90)')
    parser.add_argument('--startEpoch', type=int, default=0, help='manual epoch number (default: 0)')
    parser.add_argument('--stepEpoch', type=int, default=10, help='descend the lr in epoch number (default: 30)')

    parser.add_argument('--batchSize', type=int, default=8, help='mini-batch size (default: 8)')
    parser.add_argument('--lr', type=float, default=1e-5, help='initial learning rate (default: 1e-5)')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum (default: 0.9)')
    parser.add_argument('--weightDecay', type=float, default=0, help='weight decay (default: 0)')

    args = parser.parse_args()
    args.classNum = _ClassNum[args.dataset]

    return args
# =============================================================================
