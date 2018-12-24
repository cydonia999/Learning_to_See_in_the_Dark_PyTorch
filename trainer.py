import datetime
import math
import os
import shutil
import psutil
import gc
import time

import numpy as np
import scipy.io
import torch
from torch.autograd import Variable

import utils
import tqdm
import copy


class Trainer(object):

    def __init__(self, cmd, cuda, model, criterion, optimizer,
                 train_loader, val_loader, log_file, max_iter,
                 interval_validate=None, lr_scheduler=None,
                 checkpoint_dir=None, result_dir=None, use_camera_wb=False, print_freq=1):
        """
        :param cuda:
        :param model:
        :param optimizer:
        :param train_loader:
        :param val_loader:
        :param log_file: log file name. logs are appended to this file.
        :param max_iter:
        :param interval_validate:
        :param checkpoint_dir:
        :param lr_scheduler:
        """

        self.cmd = cmd
        self.cuda = cuda

        self.model = model
        self.criterion = criterion
        self.optim = optimizer
        self.lr_scheduler = lr_scheduler

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.timestamp_start = datetime.datetime.now()

        if self.cmd == 'train':
            self.interval_validate = len(self.train_loader) if interval_validate is None else interval_validate

        self.epoch = 0
        self.iteration = 0

        self.max_iter = max_iter
        self.best_psnr = 0
        self.print_freq = print_freq

        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_file = log_file
        self.use_camera_wb = use_camera_wb

    def print_log(self, log_str):
        with open(self.log_file, 'a') as f:
            f.write(log_str + '\n')


    def validate(self):
        batch_time = utils.AverageMeter()
        losses = utils.AverageMeter()
        psnr = utils.AverageMeter()
        ssim = utils.AverageMeter()

        training = self.model.training
        self.model.eval()

        end = time.time()
        for batch_idx, (raws, imgs, targets, img_files, img_exposures, lbl_exposures, ratios) in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='{} iteration={} epoch={}'.format('Valid' if self.cmd == 'train' else 'Test',
                                                       self.iteration, self.epoch), ncols=80, leave=False):
            gc.collect()
            if self.cuda:
                raws, targets = raws.cuda(), targets.cuda(async=True)

            with torch.no_grad():
                raws = Variable(raws)
                targets = Variable(targets)
                output = self.model(raws)

                targets = targets[:, :, :output.size(2), :output.size(3)]
                loss = self.criterion(output, targets)
                if np.isnan(float(loss.item())):
                    raise ValueError('loss is nan while validating')
                losses.update(loss.item(), targets.size(0))

            outputs = torch.clamp(output, 0, 1).cpu()
            targets = targets.cpu()

            for output, img, target, img_file, img_exposure, lbl_exposure, ratio in zip(outputs, imgs, targets,
                                                                                        img_files, img_exposures,
                                                                                        lbl_exposures, ratios):
                output = output.numpy().transpose(1, 2, 0) * 255
                target = target.numpy().transpose(1, 2, 0) * 255

                if self.result_dir:
                    if self.cmd == 'test':
                        os.makedirs(self.result_dir, exist_ok=True)
                        fname = os.path.join(self.result_dir, '{}_compare.jpg'.format(os.path.basename(img_file)[:-4]))
                        temp = np.concatenate((target[:, :, :], output[:, :, :]), axis=1)
                        scipy.misc.toimage(temp, high=255, low=0, cmin=0, cmax=255).save(fname)
                        fname = os.path.join(self.result_dir, '{}_single.jpg'.format(os.path.basename(img_file)[:-4]))
                        scipy.misc.toimage(output, high=255, low=0, cmin=0, cmax=255).save(fname)

                # psnr.update(utils.get_psnr(output, target), 1)
                _psnr = utils.get_psnr(output, target)
                print("PSNR", img_file, _psnr)
                psnr.update(_psnr, 1)
                if self.cmd == 'test':
                    _ssim = utils.get_ssim(output, target)
                    print("SSIM", img_file, _ssim)
                    ssim.update(_ssim, 1)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if batch_idx % self.print_freq == 0:
                log_str = '{cmd:}: [{0}/{1}/{loss.count:}]\tepoch: {epoch:}\titer: {iteration:}\t' \
                      'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss: {loss.val:.4f} ({loss.avg:.4f})\tPSNR: {psnr.val:.2f} ({psnr.avg:.2f})\tSSIM: {ssim.val:.4f} ({ssim.avg:.4f})\t'.format(
                    batch_idx, len(self.val_loader), cmd='Valid' if self.cmd == 'train' else 'Test',
                    epoch=self.epoch, iteration=self.iteration,
                    batch_time=batch_time, loss=losses, psnr=psnr, ssim=ssim)
                print(log_str)
                self.print_log(log_str)

        if self.cmd == 'train':
            is_best = psnr.avg > self.best_psnr
            self.best_psnr = max(psnr.avg, self.best_psnr)

            log_str = 'Valid_summary: [{0}/{1}/{psnr.count:}] epoch: {epoch:} iter: {iteration:}\t' \
                  'BestPSNR: {best_psnr:.3f}\t' \
                  'Time: {batch_time.avg:.3f}\tLoss: {loss.avg:.4f}\tPSNR: {psnr.avg:.3f}\t'.format(
                batch_idx, len(self.val_loader), epoch=self.epoch, iteration=self.iteration,
                best_psnr=self.best_psnr, batch_time=batch_time, loss=losses, psnr=psnr)
            print(log_str)
            self.print_log(log_str)

            checkpoint_file = os.path.join(self.checkpoint_dir, 'checkpoint.pth.tar')
            torch.save({
                'epoch': self.epoch,
                'iteration': self.iteration,
                'arch': self.model.__class__.__name__,
                'optim_state_dict': self.optim.state_dict(),
                'model_state_dict': self.model.state_dict(),
                'best_psnr': self.best_psnr,
                'batch_time': batch_time,
                'losses': losses,
                'psnr': psnr,
            }, checkpoint_file)
            if is_best:
                shutil.copy(checkpoint_file, os.path.join(self.checkpoint_dir, 'model_best.pth.tar'))
            if (self.epoch + 1) % 10 == 0: # save each 10 epoch
                shutil.copy(checkpoint_file, os.path.join(self.checkpoint_dir, 'checkpoint-{}.pth.tar'.format(self.epoch)))

            if training:
                self.model.train()

    def train_epoch(self):
        batch_time = utils.AverageMeter()
        data_time = utils.AverageMeter()
        losses = utils.AverageMeter()
        psnr = utils.AverageMeter()
        ssim = utils.AverageMeter()

        self.model.train()
        self.optim.zero_grad()

        end = time.time()
        for batch_idx, (raws, imgs, targets, img_files, img_exposures, lbl_exposures, ratios) in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch={}, iter={}'.format(self.epoch, self.iteration), ncols=80, leave=False):
            iteration = batch_idx + self.epoch * len(self.train_loader)
            data_time.update(time.time() - end)

            gc.collect()

            self.iteration = iteration

            if (self.iteration + 1) % self.interval_validate == 0:
                self.validate()

            if self.cuda:
                raws, targets = raws.cuda(), targets.cuda(async=True)
            raws, targets = Variable(raws), Variable(targets)

            outputs = self.model(raws)
            loss = self.criterion(outputs, targets)
            if np.isnan(float(loss.item())):
                raise ValueError('loss is nan while training')

            # measure accuracy and record loss
            losses.update(loss.item(), targets.size(0))

            outputs = torch.clamp(outputs, 0, 1).data.cpu()
            targets = targets.data.cpu()
            for output, img, target, img_file, img_exposure, lbl_exposure, ratio in zip(outputs, imgs, targets,
                                                                                        img_files, img_exposures,
                                                                                        lbl_exposures, ratios):
                output = output.numpy().transpose(1, 2, 0) * 255
                target = target.numpy().transpose(1, 2, 0) * 255
                psnr.update(utils.get_psnr(output, target), 1)
                if self.result_dir:
                    os.makedirs(self.result_dir + '%04d' % self.epoch, exist_ok=True)
                    fname = self.result_dir + '{:04d}/{:04d}_{}.jpg'.format(self.epoch, batch_idx, os.path.basename(img_file)[:-4])
                    temp = np.concatenate((target[:, :, :], output[:, :, :]), axis=1)
                    scipy.misc.toimage(temp, high=255, low=0, cmin=0, cmax=255).save(fname)

            # backprop
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if self.iteration % self.print_freq == 0:
                log_str = 'Train: [{0}/{1}/{loss.count:}]\tepoch: {epoch:}\titer: {iteration:}\t' \
                      'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                      'Loss: {loss.val:.4f} ({loss.avg:.4f})\tPSNR: {psnr.val:.1f} ({psnr.avg:.1f})\tlr {lr:.6f}'.format(
                    batch_idx, len(self.train_loader), epoch=self.epoch, iteration=self.iteration,
                    lr=self.optim.param_groups[0]['lr'],
                    batch_time=batch_time, data_time=data_time, loss=losses, psnr=psnr)
                print(log_str, flush=True)
                self.print_log(log_str)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()  # update lr

        log_str = 'Train_summary: [{0}/{1}/{loss.count:}]\tepoch: {epoch:}\titer: {iteration:}\t' \
                      'Time: {batch_time.avg:.3f}\tData: {data_time.avg:.3f}\t' \
                      'Loss: {loss.avg:.4f}\tPSNR: {psnr.avg:.1f}\tlr {lr:.6f}'.format(
                    batch_idx, len(self.train_loader), epoch=self.epoch, iteration=self.iteration,
                    lr=self.optim.param_groups[0]['lr'],
                    batch_time=batch_time, data_time=data_time, loss=losses, psnr=psnr)
        print(log_str)
        self.print_log(log_str)


    def train(self):
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
        for epoch in tqdm.trange(self.epoch, max_epoch, desc='Train', ncols=80):
            self.epoch = epoch
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break


class Validator(Trainer):

    def __init__(self, cmd, cuda, model, criterion, val_loader, log_file, result_dir=None, use_camera_wb=False, print_freq=1):
        super(Validator, self).__init__(cmd, cuda=cuda, model=model, criterion=criterion,
                                        val_loader=val_loader, log_file=log_file, print_freq=print_freq,
                                        optimizer=None, train_loader=None, max_iter=None,
                                        interval_validate=None, lr_scheduler=None,
                                        checkpoint_dir=None, result_dir=result_dir, use_camera_wb=use_camera_wb)

    def train(self):
        raise NotImplementedError

