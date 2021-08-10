import sys
import time
import logging
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm

from model.SSGRL import SSGRL
from utils.dataloader import get_graph_and_word_file, get_data_loader
from utils.metrics import AverageMeter, AveragePrecisionMeter, Compute_mAP_VOC2012
from utils.checkpoint import load_pretrained_model, save_code_file, save_checkpoint

from config import arg_parse, logger, show_args

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
logging.basicConfig(level=logging.DEBUG, filename='LTT-SSGRL-1', filemode='a')


def main():
    # Argument Parse
    args = arg_parse()
    # Show Argument
    show_args(args)

    # Create dataloader
    logging.info('==== Data loader ====')
    train_loader, test_loader = get_data_loader(args)
    logging.info('==== Data loader finish ====')

    # ---------------------
    # Load the network
    logger.info("==> Loading the network...")
    GraphFilePath, WordFilePath = get_graph_and_word_file(args)

    model = SSGRL(imageFeatureDim=2048, intermediaDim=1024, outputDim=2048, adjacencyMatrixPath=GraphFilePath,
                  wordFeaturesPath=WordFilePath, classNum=args.classNum, timeStep=args.timeStep)

    if args.pretrainedModel != 'None':
        logger.info("==> Loading pretrained model...")
        model = load_pretrained_model(model, args)
    model.to(device)
    # model = nn.DataParallel(model)
    # model = model.cuda()
    logger.info("==> Done!\n")

    # ---------------------
    loss_fun = nn.BCEWithLogitsLoss(reduce=True, size_average=True)
    loss_fun = loss_fun.to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                 weight_decay=args.weightDecay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepEpoch, gamma=0.1)

    if args.evaluate:
        Validate(test_loader, model, loss_fun, 0, args.classNum, args)
        return

    logger.info('Total: {:.3f} GB'.format(torch.cuda.get_device_properties(0).total_memory / 1024.0 ** 3))
    logging.info("Run Experiment...")

    writer = SummaryWriter('log-1')

    for epoch in range(args.startEpoch, args.startEpoch + args.epochs):
        end = time.time()

        train_loss = train(train_loader, model, loss_fun, optimizer, epoch, args)
        mAP, test_loss = Validate(test_loader, model, loss_fun, epoch, args.classNum, args, writer)
        scheduler.step()

        writer.add_scalar('mAP', mAP, epoch + 1)
        writer.add_scalars('loss', {'Train': train_loss, 'Test': test_loss}, epoch + 1)
        logging.info("Time :{:.3f}".format((time.time() - end) / 3600.0))
        torch.cuda.empty_cache()
    writer.close()


def train(train_loader, model, loss_fun, optimizer, epoch, args):
    model.train()
    model.backbone.eval()
    model.backbone.layer4.train()

    loss = AverageMeter()
    batch_time, data_time = AverageMeter(), AverageMeter()
    logging.info('=============== Train begin ===============')
    end = time.time()
    train_loader = tqdm(train_loader)
    for batchIndex, (sampleIndex, input, targets) in enumerate(train_loader):
        input, targets = input.to(device), targets.to(device)
        # input, targets = input.cuda(), targets.cuda()
        data_time.update(time.time() - end)
        output = model(input)

        loss_ = loss_fun(output, targets)

        loss.update(loss_.item(), input.shape[0])
        loss_.backward()
        optimizer.step()
        optimizer.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()

        if batchIndex % args.printFreq == 0:
            logging.info('[Train] [epoch {0}]: [{1:04d}/{2}]'
                         'Batch Time {batch_time.avg:.3f} Data Time {data_time.avg:.3f}'
                         'Learn Rate {lr:.6f} Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch + 1, batchIndex,
                                                                                           len(train_loader),
                                                                                           batch_time=batch_time,
                                                                                           data_time=data_time,
                                                                                           lr=optimizer.param_groups[0][
                                                                                               'lr'],
                                                                                           loss=loss))
            sys.stdout.flush()
        # break
    return loss.val


def Validate(val_loader, model, loss_fun, epoch, classnum, args, writer):
    model.eval()
    ApMeter = AveragePrecisionMeter()
    Pred, loss, batch_time, data_time = [], AverageMeter(), AverageMeter(), AverageMeter()
    with torch.no_grad():
        val_loader = tqdm(val_loader)
        end = time.time()
        for batchIndex, (sampleIndex, input, targets) in enumerate(val_loader):
            input, targets = input.to(device), targets.to(device)
            # input, targets = input.cuda(), targets.cuda()

            data_time.update(time.time() - end)

            output = model(input)

            loss_ = loss_fun(output, targets)
            loss.update(loss_.item(), input.shape[0])

            ApMeter.add(output, targets)
            Pred.append(torch.cat((output, (targets > 0).float()), dim=1))

            batch_time.update(time.time() - end)
            end = time.time()

            if batchIndex % args.printFreq == 0:
                logger.info('[Test] [Epoch {0}]: [{1:04d}/{2}] '
                            'Batch Time {batch_time.avg:.3f} Data Time {data_time.avg:.3f} '
                            'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                    epoch + 1, batchIndex, len(val_loader),
                    batch_time=batch_time, data_time=data_time,
                    loss=loss))
                sys.stdout.flush()
            # break

        Pred = torch.cat(Pred, 0).cpu().numpy()
        mAP = Compute_mAP_VOC2012(Pred, classnum)

        averageAP = ApMeter.value().mean()
        OP, OR, OF1, CP, CR, CF1 = ApMeter.overall()
        OP_K, OR_K, OF1_K, CP_K, CR_K, CF1_K = ApMeter.overall_topk(5)

        logger.info('[Test] mAP: {mAP:.3f}, averageAP: {averageAP:.3f}\n'
                    '       (Compute with all label) OP: {OP:.3f}, OR: {OR:.3f}, OF1: {OF1:.3f}, CP: {CP:.3f}, CR: {CR:.3f}, CF1:{CF1:.3f}\n'
                    '       (Compute with top-5 label) OP: {OP_K:.3f}, OR: {OR_K:.3f}, OF1: {OF1_K:.3f}, CP: {CP_K:.3f}, CR: {CR_K:.3f}, CF1: {CF1_K:.3f}'.format(
            mAP=mAP, averageAP=averageAP,
            OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1,
            OP_K=OP_K, OR_K=OR_K, OF1_K=OF1_K, CP_K=CP_K, CR_K=CR_K, CF1_K=CF1_K))

        # writer.add_scalar('Test_loss', loss.val, epoch + 1)

        return mAP, loss.val


if __name__ == '__main__':
    main()
