# data loader
from __future__ import print_function, division
import glob
import os

import cv2
import torch
from skimage import io, transform, color
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

# ==========================dataset load==========================
from datasets.data_util import generateRandomPrior, get_trimap


class RescaleT(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        # #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
        # img = transform.resize(image,(new_h,new_w),mode='constant')
        # lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

        img = transform.resize(image, (self.output_size, self.output_size), mode='constant')
        lbl = transform.resize(label, (self.output_size, self.output_size), mode='constant', order=0,
                               preserve_range=True)

        return {'imidx': imidx, 'image': img, 'label': lbl}


class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        if random.random() >= 0.5:
            image = image[::-1]
            label = label[::-1]

        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        # #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
        img = transform.resize(image, (new_h, new_w), mode='constant')
        lbl = transform.resize(label, (new_h, new_w), mode='constant', order=0, preserve_range=True)

        return {'imidx': imidx, 'image': img, 'label': lbl}


class RandomCrop_u2net(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        if random.random() >= 0.5:
            image = image[::-1]
            label = label[::-1]

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]
        label = label[top: top + new_h, left: left + new_w]

        return {'imidx': imidx, 'image': image, 'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
        tmpLbl = np.zeros(label.shape)

        image = image / np.max(image)
        if (np.max(label) < 1e-6):
            label = label
        else:
            label = label / np.max(label)

        if image.shape[2] == 1:
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
        else:
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
            tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225

        tmpLbl[:, :, 0] = label[:, :, 0]

        # change the r,g,b to b,r,g from [0,255] to [0,1]
        # transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpLbl = label.transpose((2, 0, 1))

        return {'imidx': torch.from_numpy(imidx), 'image': torch.from_numpy(tmpImg), 'label': torch.from_numpy(tmpLbl)}


class ToTensorLab(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, flag=0):
        self.flag = flag

    def __call__(self, sample):

        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        tmpLbl = np.zeros(label.shape)

        if (np.max(label) < 1e-6):
            label = label
        else:
            label = label / np.max(label)

        # change the color space
        if self.flag == 2:  # with rgb and Lab colors
            tmpImg = np.zeros((image.shape[0], image.shape[1], 6))
            tmpImgt = np.zeros((image.shape[0], image.shape[1], 3))
            if image.shape[2] == 1:
                tmpImgt[:, :, 0] = image[:, :, 0]
                tmpImgt[:, :, 1] = image[:, :, 0]
                tmpImgt[:, :, 2] = image[:, :, 0]
            else:
                tmpImgt = image
            tmpImgtl = color.rgb2lab(tmpImgt)

            # nomalize image to range [0,1]
            tmpImg[:, :, 0] = (tmpImgt[:, :, 0] - np.min(tmpImgt[:, :, 0])) / (
                    np.max(tmpImgt[:, :, 0]) - np.min(tmpImgt[:, :, 0]))
            tmpImg[:, :, 1] = (tmpImgt[:, :, 1] - np.min(tmpImgt[:, :, 1])) / (
                    np.max(tmpImgt[:, :, 1]) - np.min(tmpImgt[:, :, 1]))
            tmpImg[:, :, 2] = (tmpImgt[:, :, 2] - np.min(tmpImgt[:, :, 2])) / (
                    np.max(tmpImgt[:, :, 2]) - np.min(tmpImgt[:, :, 2]))
            tmpImg[:, :, 3] = (tmpImgtl[:, :, 0] - np.min(tmpImgtl[:, :, 0])) / (
                    np.max(tmpImgtl[:, :, 0]) - np.min(tmpImgtl[:, :, 0]))
            tmpImg[:, :, 4] = (tmpImgtl[:, :, 1] - np.min(tmpImgtl[:, :, 1])) / (
                    np.max(tmpImgtl[:, :, 1]) - np.min(tmpImgtl[:, :, 1]))
            tmpImg[:, :, 5] = (tmpImgtl[:, :, 2] - np.min(tmpImgtl[:, :, 2])) / (
                    np.max(tmpImgtl[:, :, 2]) - np.min(tmpImgtl[:, :, 2]))

            # tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

            tmpImg[:, :, 0] = (tmpImg[:, :, 0] - np.mean(tmpImg[:, :, 0])) / np.std(tmpImg[:, :, 0])
            tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.mean(tmpImg[:, :, 1])) / np.std(tmpImg[:, :, 1])
            tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.mean(tmpImg[:, :, 2])) / np.std(tmpImg[:, :, 2])
            tmpImg[:, :, 3] = (tmpImg[:, :, 3] - np.mean(tmpImg[:, :, 3])) / np.std(tmpImg[:, :, 3])
            tmpImg[:, :, 4] = (tmpImg[:, :, 4] - np.mean(tmpImg[:, :, 4])) / np.std(tmpImg[:, :, 4])
            tmpImg[:, :, 5] = (tmpImg[:, :, 5] - np.mean(tmpImg[:, :, 5])) / np.std(tmpImg[:, :, 5])

        elif self.flag == 1:  # with Lab color
            tmpImg = np.zeros((image.shape[0], image.shape[1], 3))

            if image.shape[2] == 1:
                tmpImg[:, :, 0] = image[:, :, 0]
                tmpImg[:, :, 1] = image[:, :, 0]
                tmpImg[:, :, 2] = image[:, :, 0]
            else:
                tmpImg = image

            tmpImg = color.rgb2lab(tmpImg)

            # tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

            tmpImg[:, :, 0] = (tmpImg[:, :, 0] - np.min(tmpImg[:, :, 0])) / (
                    np.max(tmpImg[:, :, 0]) - np.min(tmpImg[:, :, 0]))
            tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.min(tmpImg[:, :, 1])) / (
                    np.max(tmpImg[:, :, 1]) - np.min(tmpImg[:, :, 1]))
            tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.min(tmpImg[:, :, 2])) / (
                    np.max(tmpImg[:, :, 2]) - np.min(tmpImg[:, :, 2]))

            tmpImg[:, :, 0] = (tmpImg[:, :, 0] - np.mean(tmpImg[:, :, 0])) / np.std(tmpImg[:, :, 0])
            tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.mean(tmpImg[:, :, 1])) / np.std(tmpImg[:, :, 1])
            tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.mean(tmpImg[:, :, 2])) / np.std(tmpImg[:, :, 2])

        else:  # with rgb color
            tmpImg = image  # / 255
            # tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
            # image = image / np.max(image)
            # if image.shape[2] == 1:
            #     tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            #     tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
            #     tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
            # else:
            #     tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            #     tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
            #     tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225

        tmpLbl[:, :, 0] = label[:, :, 0]

        # change the r,g,b to b,r,g from [0,255] to [0,1]
        # transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
        # imidx = np.ascontiguousarray(imidx)
        # tmpImg = np.ascontiguousarray(tmpImg)
        tmpImg = tmpImg.transpose((2, 0, 1))

        # label = np.ascontiguousarray(label)
        tmpLbl = label.transpose((2, 0, 1))

        # tmpLbl = tmpLbl.copy()

        return {'imidx': torch.from_numpy(imidx.copy()), 'image': torch.from_numpy(tmpImg.copy()),
                'label': torch.from_numpy(tmpLbl.copy())}


class U2Net_Dataset(Dataset):
    def __init__(self, args, mode='train'):

        fg_path = args.fg_path
        label_path = args.gt_path
        img_name_list = os.listdir(fg_path)
        lbl_name_list = os.listdir(label_path)
        img_name_list.sort()
        lbl_name_list.sort()
        transform = transforms.Compose([
            RescaleT(args.scale_size),
            RandomCrop_u2net(args.crop_size),
            ToTensorLab(flag=0)]
        )

        # self.root_dir = root_dir
        # self.image_name_list = glob.glob(image_dir+'*.png')
        # self.label_name_list = glob.glob(label_dir+'*.png')
        self.image_name_list = img_name_list
        self.label_name_list = lbl_name_list
        self.fg_path, self.label_path = fg_path, label_path
        self.transform = transform

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):

        # image = Image.open(self.image_name_list[idx])#io.imread(self.image_name_list[idx])
        # label = Image.open(self.label_name_list[idx])#io.imread(self.label_name_list[idx])

        image = cv2.imread(self.fg_path + self.image_name_list[idx])
        imname = self.image_name_list[idx]
        imidx = np.array([idx])

        if (0 == len(self.label_name_list)):
            label_3 = np.zeros(image.shape)
        else:
            label_3 = cv2.imread(self.label_path + self.label_name_list[idx])

        label = np.zeros(label_3.shape[0:2])
        if (3 == len(label_3.shape)):
            label = label_3[:, :, 0]
        elif (2 == len(label_3.shape)):
            label = label_3

        if (3 == len(image.shape) and 2 == len(label.shape)):
            label = label[:, :, np.newaxis]
        elif (2 == len(image.shape) and 2 == len(label.shape)):
            image = image[:, :, np.newaxis]
            label = label[:, :, np.newaxis]

        sample = {'imidx': imidx, 'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        prior = generateRandomPrior(sample['label'][0].cpu().numpy(), size=31)
        prior_trimap = prior.copy()
        prior_trimap[prior_trimap == -1] = 1

        label_alpha = sample['label'][0]
        trimap = get_trimap(label_alpha.cpu().numpy())

        # show
        # plt.subplot(1, 2, 1)
        # plt.imshow(np.array(sample['image'].permute(1, 2, 0).detach().cpu().numpy()[:,:,[2,1,0]] * 255, dtype='uint8'))
        # plt.subplot(1, 2, 2)
        # plt.imshow(sample['label'][0].detach().cpu().numpy(), cmap='gray')
        # plt.savefig('sss.png')

        return sample['image'], sample['label'][0], torch.from_numpy(trimap), sample['image'], sample['image'], \
               torch.from_numpy(prior), torch.from_numpy(prior_trimap)
