# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import torch
import numpy as np
import cv2
from .evaluation import decode_preds, compute_nme

logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(config, train_loader, model, critertion, optimizer,
          epoch, writer_dict):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    nme_count = 0
    nme_batch_sum = 0

    end = time.time()

    for i, (image, label, size, name) in enumerate(train_loader):
        # measure data time
        data_time.update(time.time()-end)

        # compute the output
        output = model(image)
        labels = label.float().cuda()
        loss = critertion(output, labels)
        '''
        # NME
        score_map = output.data.cpu()
        preds = decode_preds(score_map, meta['center'], meta['scale'], [64, 64])

        nme_batch = compute_nme(preds, meta)
        nme_batch_sum = nme_batch_sum + np.sum(nme_batch)
        nme_count = nme_count + preds.size(0)
        '''
        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), image.size(0))

        batch_time.update(time.time()-end)
        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=image.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            logger.info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

        end = time.time()
    #nme = nme_batch_sum / nme_count
    msg = 'Train Epoch {} time:{:.4f} loss:{:.4f}'\
        .format(epoch, batch_time.avg, losses.avg)
    logger.info(msg)


def validate(config, val_loader, model, criterion, epoch, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()

    predictions = torch.zeros((len(val_loader.dataset), 1, 2))

    model.eval()
    end = time.time()

    with torch.no_grad():
        for i, (image, label, size, name) in enumerate(val_loader):
            data_time.update(time.time() - end)
            output = model(image)
            labels = label.float().cuda()
            #target = target.cuda(non_blocking=True)

            #score_map = output.data.cpu()
            # loss

            loss = criterion(output, labels)
            label1 = output[0,0,:,:]*255
            path = 'output//vis//'+name[0]+'.png'
            label1 = label1.cpu().numpy()
            print(np.max(label1))
            #print(path, type(label1),print(label1.shape))
            cv2.imwrite(path, label1)

            '''
            preds = decode_preds(score_map, meta['center'], meta['scale'], [64, 64])
            # NME
            nme_temp = compute_nme(preds, meta)
            # Failure Rate under different threshold
            failure_008 = (nme_temp > 0.08).sum()
            failure_010 = (nme_temp > 0.10).sum()
            count_failure_008 += failure_008
            count_failure_010 += failure_010

            nme_batch_sum += np.sum(nme_temp)
            nme_count = nme_count + preds.size(0)
            for n in range(score_map.size(0)):
                predictions[meta['index'][n], :, :] = preds[n, :, :]
            '''
            losses.update(loss.item(), image.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    msg = 'Test Epoch {} time:{:.4f} loss:{:.4f} '.format(epoch, batch_time.avg, losses.avg)
    logger.info(msg)

    return losses.avg, predictions


def inference(config, data_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()

    predictions = torch.zeros((len(data_loader.dataset), 1, 2))

    model.eval()
    end = time.time()

    with torch.no_grad():
        for i, (image, label, size, name) in enumerate(data_loader):
            data_time.update(time.time() - end)
            output = model(image)
            label1 = output[0,0,:,:]*255
            path = 'output//vis//'+name[0]+'.png'
            label1 = label1.cpu().numpy()
            print(np.max(label1))
            cv2.imwrite(path, label1)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    msg = 'Test time:{:.4f} '.format(batch_time.avg, losses.avg)
    logger.info(msg)

    return predictions



