"""
Bridging Composite and Real: Towards End-to-end Deep Image Matting [IJCV-2021]
Dataset processing.

Copyright (c) 2021, Jizhizi Li (jili8515@uni.sydney.edu.au)
Licensed under the MIT License (see LICENSE for details)
Github repo: https://github.com/JizhiziLi/GFM
Paper link (Arxiv): https://arxiv.org/abs/2010.16188

"""

import torch
import cv2
import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import logging
import pickle
from torchvision import transforms
from torch.autograd import Variable
from skimage.transform import resize

########## Parameters for training
from utils.util import generateRandomPrior

########## Parameters for testing
MAX_SIZE_H = 1600
MAX_SIZE_W = 1600
SHORTER_PATH_LIMITATION = 1080


#########################
## Data transformer
#########################
class MattingTransform(object):
    def __init__(self, out_size, crop_size=[640, 960, 1280]):
        super(MattingTransform, self).__init__()
        self.out_size = out_size
        self.crop_size = crop_size
        
    def __call__(self, *argv):
        ori = argv[0]
        h, w, c = ori.shape
        rand_ind = random.randint(0, len(self.crop_size) - 1)
        crop_size = self.crop_size[rand_ind] if self.crop_size[rand_ind] < min(h, w) else 320
        resize_size = self.out_size
        ### generate crop centered in transition area randomly
        trimap = argv[1]
        trimap_crop = trimap[:h - crop_size, :w - crop_size]
        target = np.where(trimap_crop == 128) if random.random() < 0.5 else np.where(trimap_crop > -100)
        if len(target[0]) == 0:
            target = np.where(trimap_crop > -100)

        rand_ind = np.random.randint(len(target[0]), size=1)[0]
        cropx, cropy = target[1][rand_ind], target[0][rand_ind]
        # # flip the samples randomly
        flip_flag = True if random.random() < 0.5 else False
        # generate samples (crop, flip, resize)
        argv_transform = []
        index = 0
        for item in argv:
            item = item[cropy:cropy + crop_size, cropx:cropx + crop_size]
            if flip_flag:
                item = cv2.flip(item, 1)
            if index == 1:
                item = cv2.resize(item, (resize_size, resize_size), interpolation=cv2.INTER_NEAREST)
            item = cv2.resize(item, (resize_size, resize_size), interpolation=cv2.INTER_LINEAR)
            argv_transform.append(item)
            index += 1

        return argv_transform


def process_fgbg(ori, mask, is_fg, fgbg_path=None):
    if fgbg_path is not None:
        img = np.array(Image.open(fgbg_path))
    else:
        mask_3 = (mask / 255.0)[:, :, np.newaxis].astype(np.float32)
        img = ori * mask_3 if is_fg else ori * (1 - mask_3)
    return img


def resize_img(ori, img):
    img = cv2.resize(img, ori.shape) * 255.0
    return img


def add_guassian_noise(img, fg, bg):
    row, col, ch = img.shape
    mean = 0
    sigma = 10
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy_img = np.uint8(img + gauss)
    noisy_fg = np.uint8(fg + gauss)
    noisy_bg = np.uint8(bg + gauss)
    return noisy_img, noisy_fg, noisy_bg


def generate_composite_rssn(fg, bg, mask, fg_denoise=None, bg_denoise=None):
    ## resize bg accordingly
    h, w, c = fg.shape
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[:, :, 0] = mask / 255.
    bg = resize_img(fg, bg)
    ## use denoise fg/bg randomly
    if fg_denoise is not None and random.random() < 0.5:
        fg = fg_denoise
        bg = resize_img(fg, bg_denoise)
    ## reduce sharpness discrepancy
    if random.random() < 0.5:
        rand_kernel = random.choice([20, 30, 40, 50, 60])
        bg = cv2.blur(bg, (rand_kernel, rand_kernel))
    composite = alpha * fg + (1 - alpha) * bg
    composite = composite.astype(np.uint8)
    ## reduce noise discrepancy
    if random.random() < 0.5:
        composite, fg, bg = add_guassian_noise(composite, fg, bg)
    return composite, fg, bg


def generate_composite_coco(fg, bg, mask):
    h, w, c = fg.shape
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[:, :, 0] = mask / 255.
    bg = resize_img(fg, bg)
    composite = alpha * fg + (1 - alpha) * bg
    composite = composite.astype(np.uint8)
    return composite, fg, bg


def gen_trimap_with_dilate(alpha, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    fg_and_unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
    fg = np.array(np.equal(alpha, 255).astype(np.float32))
    dilate = cv2.dilate(fg_and_unknown, kernel, iterations=1)
    erode = cv2.erode(fg, kernel, iterations=1)
    trimap = erode * 255 + (dilate - erode) * 128
    return trimap.astype(np.uint8)


#########################
## Data Loader
#########################
class GFM_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, mode='train'):

        self.out_size = args.gfm_im_size
        self.mode = mode
        self.transform_gfm = MattingTransform(out_size=self.out_size, crop_size=[640, 960, 1280])
        self.mergePath = args.fg_path
        self.gtPath = args.gt_path
        self.merge_files = os.listdir(args.fg_path)
        self.gt_files = os.listdir(args.gt_path)
        if args.fg_path is not None and args.fg_path != '':
            self.fgPath = args.fg_path
            self.fg_files = os.listdir(args.fg_path)
            self.fg_files.sort()
        else:
            self.fgPath = None

        if args.bg_path is not None and args.bg_path != '':
            self.bgPath = args.bg_path
            self.bg_files = os.listdir(args.bg_path)
            self.bg_files.sort()
        else:
            self.bgPath = None

        if args.fgPath_denoise is not None and args.bgPath_denoise is not None \
                and args.fgPath_denoise != '' and args.bgPath_denoise != '':
            self.bgPath_denoise = args.bgPath_denoise
            self.fgPath_denoise = args.fgPath_denoise
            self.fg_files_denoise = os.listdir(self.fgPath_denoise)
            self.bg_files_denoise = os.listdir(self.bgPath_denoise)
            self.fg_files_denoise.sort()
            self.bg_files_denoise.sort()
        else:
            self.bgPath_denoise = None
            self.fgPath_denoise = None
        self.merge_files.sort()
        self.gt_files.sort()


    def __getitem__(self, index):
        im_index = index % len(self.merge_files)
        im_name = self.merge_files[im_index]
        label_name = self.gt_files[im_index]
        assert label_name.split('.')[0] == im_name.split('.')[0], 'name is not match'
        merge_im = cv2.imread(self.mergePath + im_name)
        label_im = cv2.imread(self.gtPath + label_name, cv2.IMREAD_UNCHANGED)

        if len(label_im.shape) == 2:
            label_im = label_im
        elif label_im.shape[2] == 4:
            label_im = label_im[..., -1]
        else:
            label_im = label_im[:, :, 0]

        if self.fgPath is not None:
            fg_path = self.fgPath + self.fg_files[index]
        else:
            fg_path = None
        if self.bgPath is not None:
            bg_path = self.bgPath + self.bg_files[index]
        else:
            bg_path = None

        fg = process_fgbg(merge_im, label_im, True, fg_path)
        bg = process_fgbg(merge_im, label_im, False, bg_path)

        if self.fgPath_denoise is not None and self.bgPath_denoise is not None:
            fg_denoise = process_fgbg(merge_im, label_im, True, self.fg_files_denoise) if self.RSSN_DENOISE else None
            bg_denoise = process_fgbg(merge_im, label_im, False, self.bg_files_denoise) if self.RSSN_DENOISE else None
            merge_im, fg, bg = generate_composite_rssn(fg, bg, label_im, fg_denoise, bg_denoise)

        # ori, fg, bg = generate_composite_coco(fg, bg, label_im)

        # Generate trimap/dilation/erosion online
        kernel_size_tt = 25
        trimap = gen_trimap_with_dilate(label_im, kernel_size_tt)

        # Data transformation to generate samples
        # crop/flip/resize
        argv = self.transform_gfm(merge_im, trimap, label_im, fg, bg)
        argv_transform = []
        for item in argv:
            if item.ndim < 3:
                item = torch.from_numpy(item.astype(np.float32)[np.newaxis, :, :]) / 255
            else:
                item = torch.from_numpy(item.astype(np.float32)).permute(2, 0, 1) / 255
            argv_transform.append(item)

        [ori, trimap, mask, fg, bg] = argv_transform
        trimap[(trimap != 0) * (trimap != 1)] = 0.5
        prior = generateRandomPrior(argv[2], size=31)
        prior_trimap = prior.copy()
        prior_trimap[prior_trimap == -1] = 1

        # show
        # import matplotlib.pyplot as plt
        # plt.subplot(1, 5, 1)
        # plt.imshow(ori.permute([1, 2, 0]).numpy())
        # plt.subplot(1, 5, 2)
        # plt.imshow(mask[0].numpy(), cmap='gray')
        # plt.subplot(1, 5, 3)
        # plt.imshow(trimap[0].numpy(), cmap='gray')
        # plt.subplot(1, 5, 4)
        # plt.imshow(fg.permute([1, 2, 0]).numpy())
        # plt.subplot(1, 5, 5)
        # plt.imshow(bg.permute([1, 2, 0]).numpy())
        # plt.show()

        return ori, mask, trimap, fg, bg, torch.from_numpy(prior).unsqueeze(0), torch.from_numpy(
            prior_trimap).unsqueeze(0), index

    def __len__(self):
        return len(self.merge_files)
