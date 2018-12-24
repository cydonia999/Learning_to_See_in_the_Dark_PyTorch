#!/usr/bin/env python

import collections
import os

import numpy as np
import PIL.Image
import scipy.io
import rawpy
import torch
from torch.utils import data
import torchvision.transforms
import time
import utils

class RAW_Base(data.Dataset):

    def __init__(self, root, image_list_file, patch_size=None, split='train', gt_png=True, use_camera_wb=False,
                 raw_ext='ARW', upper=None):
        """
        :param root: dataset directory
        :param image_list_file: contains image file names under root
        :param patch_size: if None, full images are returned, otherwise patches are returned
        :param split: train or valid
        :param upper: max number of image used for debug
        """
        assert os.path.exists(root), "root: {} not found.".format(root)
        self.root = root
        assert os.path.exists(image_list_file), "image_list_file: {} not found.".format(image_list_file)
        self.image_list_file = image_list_file
        self.patch_size = patch_size
        self.split = split
        self.gt_png = gt_png
        self.use_camera_wb = use_camera_wb
        self.raw_ext = raw_ext

        self.raw_short_read_time = utils.AverageMeter()
        self.raw_short_pack_time = utils.AverageMeter()
        self.raw_short_post_time = utils.AverageMeter()
        self.raw_long_read_time = utils.AverageMeter()
        self.raw_long_pack_time = utils.AverageMeter()
        self.raw_long_post_time = utils.AverageMeter()
        self.png_long_read_time = utils.AverageMeter()

        self.img_info = []
        with open(self.image_list_file, 'r') as f:
            for i, img_pair in enumerate(f):
                img_pair = img_pair.strip()  # ./Sony/short/10003_00_0.04s.ARW ./Sony/long/10003_00_10s.ARW ISO200 F9
                img_file, lbl_file, iso, focus = img_pair.split(' ')
                if self.split == 'test':
                    if os.path.split(img_file)[-1][5:8] != '_00':
                        continue
                img_exposure = float(os.path.split(img_file)[-1][9:-5]) # 0.04
                lbl_exposure = float(os.path.split(lbl_file)[-1][9:-5]) # 10
                ratio = min(lbl_exposure/img_exposure, 300)
                if self.gt_png:
                    lbl_file = lbl_file.replace("long", "gt").replace(self.raw_ext, "png")
                self.img_info.append({
                    'img': img_file,
                    'lbl': lbl_file,
                    'img_exposure': img_exposure,
                    'lbl_exposure': lbl_exposure,
                    'ratio': ratio,
                    'iso': iso,
                    'focus': focus,
                })
                if i % 1000 == 0:
                    print("processing: {} images for {}".format(i, self.split))
                if upper and i == upper - 1:  # for debug purpose
                    break


    def __len__(self):
        return len(self.img_info)

    def __getitem__(self, index):
        info = self.img_info[index]

        img_file = info['img']
        start = time.time()
        raw = rawpy.imread(os.path.join(self.root, img_file))
        self.raw_short_read_time.update(time.time() - start)
        start = time.time()
        input_full = self.pack_raw(raw) * info['ratio']
        self.raw_short_pack_time.update(time.time() - start)

        scale_full = np.zeros((1, 1, 1), dtype=np.float32)
        if self.use_camera_wb:
            start = time.time()
            im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            self.raw_short_post_time.update(time.time() - start)
            scale_full = np.float32(im / 65535.0)

        lbl_file = info['lbl']
        if self.gt_png:
            start = time.time()
            gt_full = np.array(PIL.Image.open(os.path.join(self.root, lbl_file)), dtype=np.float32)
            gt_full = gt_full / 255.0
            self.png_long_read_time.update(time.time() - start)
        else:
            start = time.time()
            lbl_raw = rawpy.imread(os.path.join(self.root, lbl_file))
            self.raw_long_read_time.update(time.time() - start)
            start = time.time()
            im = lbl_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            self.raw_long_post_time.update(time.time() - start)
            gt_full = np.float32(im / 65535.0)

        input_full = input_full.transpose(2, 0, 1)  # C x H x W
        gt_full = gt_full.transpose(2, 0, 1)  # C x H x W


        if self.patch_size:
            # crop
            H, W = input_full.shape[1:3]
            yy, xx = np.random.randint(0, H - self.patch_size),  np.random.randint(0, W - self.patch_size)
            input_patch = input_full[:, yy:yy + self.patch_size, xx:xx + self.patch_size]
            gt_patch = gt_full[:, yy*2:(yy + self.patch_size) * 2, xx*2:(xx + self.patch_size) * 2]

            if np.random.randint(2) == 1:  # random horizontal flip
                input_patch = np.flip(input_patch, axis=2)
                gt_patch = np.flip(gt_patch, axis=2)
            if np.random.randint(2) == 1:  # random vertical flip
                input_patch = np.flip(input_patch, axis=1)
                gt_patch = np.flip(gt_patch, axis=1)
            if np.random.randint(2) == 1:  # random transpose
                input_patch = np.transpose(input_patch, (0, 2, 1))
                gt_patch = np.transpose(gt_patch, (0, 2, 1))

            input_full = input_patch.copy()
            gt_full = gt_patch.copy()

        input_full = np.minimum(input_full, 1.0)

        input_full = torch.from_numpy(input_full).float()
        gt_full = torch.from_numpy(gt_full).float()
        scale_full = torch.from_numpy(scale_full).float()

        return input_full, scale_full, gt_full, img_file, info['img_exposure'], info['lbl_exposure'], info['ratio']

class Sony(RAW_Base):
    def __init__(self, root, image_list_file, split='train', patch_size=None, gt_png=True, use_camera_wb=False,
                 upper=None):
        super(Sony, self).__init__(root, image_list_file, split=split, patch_size=patch_size,
                                   gt_png=gt_png, use_camera_wb=use_camera_wb, raw_ext='ARW', upper=upper)

    def pack_raw(self, raw):
        # pack Bayer image to 4 channels
        im = raw.raw_image_visible.astype(np.float32)
        im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

        im = np.expand_dims(im, axis=2)
        img_shape = im.shape
        H = img_shape[0]
        W = img_shape[1]

        out = np.concatenate((im[0:H:2, 0:W:2, :],
                              im[0:H:2, 1:W:2, :],
                              im[1:H:2, 1:W:2, :],
                              im[1:H:2, 0:W:2, :]), axis=2)
        return out

class Fuji(RAW_Base):
    def __init__(self, root, image_list_file, split='train', patch_size=None, gt_png=True, use_camera_wb=False,
                 upper=None):
        super(Fuji, self).__init__(root, image_list_file, split=split, patch_size=patch_size,
                                   gt_png=gt_png, use_camera_wb=use_camera_wb, raw_ext='RAF', upper=upper)

    def pack_raw(self, raw):
        # pack X-Trans image to 9 channels
        im = raw.raw_image_visible.astype(np.float32)
        im = np.maximum(im - 1024, 0) / (16383 - 1024)  # subtract the black level

        img_shape = im.shape
        H = (img_shape[0] // 6) * 6
        W = (img_shape[1] // 6) * 6

        out = np.zeros((H // 3, W // 3, 9))

        # 0 R
        out[0::2, 0::2, 0] = im[0:H:6, 0:W:6]
        out[0::2, 1::2, 0] = im[0:H:6, 4:W:6]
        out[1::2, 0::2, 0] = im[3:H:6, 1:W:6]
        out[1::2, 1::2, 0] = im[3:H:6, 3:W:6]

        # 1 G
        out[0::2, 0::2, 1] = im[0:H:6, 2:W:6]
        out[0::2, 1::2, 1] = im[0:H:6, 5:W:6]
        out[1::2, 0::2, 1] = im[3:H:6, 2:W:6]
        out[1::2, 1::2, 1] = im[3:H:6, 5:W:6]

        # 1 B
        out[0::2, 0::2, 2] = im[0:H:6, 1:W:6]
        out[0::2, 1::2, 2] = im[0:H:6, 3:W:6]
        out[1::2, 0::2, 2] = im[3:H:6, 0:W:6]
        out[1::2, 1::2, 2] = im[3:H:6, 4:W:6]

        # 4 R
        out[0::2, 0::2, 3] = im[1:H:6, 2:W:6]
        out[0::2, 1::2, 3] = im[2:H:6, 5:W:6]
        out[1::2, 0::2, 3] = im[5:H:6, 2:W:6]
        out[1::2, 1::2, 3] = im[4:H:6, 5:W:6]

        # 5 B
        out[0::2, 0::2, 4] = im[2:H:6, 2:W:6]
        out[0::2, 1::2, 4] = im[1:H:6, 5:W:6]
        out[1::2, 0::2, 4] = im[4:H:6, 2:W:6]
        out[1::2, 1::2, 4] = im[5:H:6, 5:W:6]

        out[:, :, 5] = im[1:H:3, 0:W:3]
        out[:, :, 6] = im[1:H:3, 1:W:3]
        out[:, :, 7] = im[2:H:3, 0:W:3]
        out[:, :, 8] = im[2:H:3, 1:W:3]

        return out
