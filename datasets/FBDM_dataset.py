import kornia
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os

import datasets.augmentation as A
import torchvision.transforms as T
from datasets.Base_dataset import Base_Dataset, TRAIN_TYPE, VAL_TYPE, SHOW_TYPE
from utils.util import *


class FBDM_Dataset(Base_Dataset):
    """
    The dataset method for FBDM

    Args:
        forPath (str): The path of the input images.
        labelPath (str): The path of ground truth.
        bgPath (str): The path of background images.
            If the bgPath is None (default), it will not randomly replace the background for input images.
        mode (str): The mode of dataset. [train|val]
            If mode is 'val', getitem func just returns the required tensors.
        out_size (int): The final size of input images. The final size will be (out_size, out_size).
    """

    def __init__(self, args, mode='train'):

        super().__init__(args, mode)

        self.bg_files = []
        if mode == 'val':
            forPath = args.val_img_path
            labelPath = args.val_gt_path
            out_size = args.img_size
            bgPath = None

            self.transform = T.Compose([
                T.ToTensor(),
            ])
        elif mode == 'show':
            forPath = args.show_img_path
            self.transform = T.Compose([
                T.ToTensor(),
            ])
            out_size = args.img_size
            labelPath = args.show_gt_path
            bgPath = None

        elif mode == 'train':
            forPath = args.im_path
            labelPath = args.gt_path
            bgPath = args.bg_path
            out_size = args.img_size

            if bgPath != '' and bgPath is not None:
                self.bg_files = os.listdir(bgPath)
            self.transform_bgm_fg = A.PairCompose([
                A.PairRandomAffineAndResize((out_size, out_size), degrees=(-5, 5), translate=(0.1, 0.1),
                                            scale=(0.8, 1.2),
                                            shear=(-5, 5)),
                A.PairRandomHorizontalFlip(),
                A.PairRandomBoxBlur(0.4, 5),
                A.PairRandomSharpen(0.3),
                A.PairApplyOnlyAtIndices([0], T.ColorJitter(0.3, 0.15, 0.15, 0.05)),
                # A.PairApplyOnlyAtIndices([1], T.ColorJitter(0.15, 0.15, 0.15, 0.05)),
                A.PairApply(T.ToTensor())
            ])

            self.transform_bgm_bg = T.Compose([
                A.RandomAffineAndResize((out_size, out_size), degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.5, 2),
                                        shear=(-5, 5)),
                T.RandomHorizontalFlip(),
                A.RandomBoxBlur(0.4, 5),
                A.RandomSharpen(0.1),
                T.ColorJitter(0.3, 0.15, 0.15, 0.05),
                T.ToTensor()
            ])
        else:
            assert False, 'Not supporting mode: {}' + mode
        self.forthPath = forPath
        self.labelPath = labelPath
        self.bgPath = bgPath
        self.fg_files = os.listdir(forPath)
        self.fg_files.sort()
        self.labelPath_files = os.listdir(labelPath)
        self.labelPath_files.sort()
        # self.trimapPath_files = os.listdir(trimap)
        self.out_size = out_size
        print(self.mode + " dataset's number:{}".format(max(len(self.fg_files), len(self.bg_files))))

    def get_train_data(self, item: int) -> TRAIN_TYPE:
        fg_index = item % len(self.fg_files)
        fg_name = self.fg_files[fg_index]
        label_name = self.labelPath_files[fg_index]
        assert label_name.split('.')[0] == fg_name.split('.')[0], 'name is not match'
        fg_im = cv2.imread(self.forthPath + fg_name)
        label_im = cv2.imread(self.labelPath + label_name)
        if len(label_im.shape) == 2:
            label_im = label_im
        elif label_im.shape[2] == 4:
            label_im = label_im[..., -1]
        else:
            label_im = label_im[:, :, 0]
        im_fg = Image.fromarray(fg_im, mode='RGB')
        im_alpha = Image.fromarray(label_im, mode='L')
        im_fg, im_alpha = self.transform_bgm_fg(im_fg, im_alpha)

        # replace randomly background
        if len(self.bg_files) > 0 and random.random() < 0.3:
            bg_index = item % len(self.bg_files)
            bg_name = self.bg_files[bg_index]
            bg_im = cv2.imread(self.bgPath + bg_name)
            if len(bg_im.shape) == 2:
                bg_im = cv2.cvtColor(bg_im, cv2.COLOR_GRAY2BGR)
            if random.random() < 0.5:
                rand_kernel = random.choice([20, 30, 40, 50, 60])
                bg_im = cv2.blur(bg_im, (rand_kernel, rand_kernel))
            bg_im = Image.fromarray(bg_im, mode='RGB')
            bg_im = self.transform_bgm_bg(bg_im)

            # random adjust alpha
            # if random.random() < 0.3:
            #     im_alpha = im_alpha * random.randint(5, 9) * 0.1
        else:
            bg_im = im_fg

        # shadow
        if random.random() < 0.5:
            aug_shadow = im_alpha.mul(max(0.1, random.random()))
            aug_shadow = T.RandomAffine(degrees=(-5, 5), translate=(0.01, 0.1), scale=(0.95, 1.1), shear=(-5, 5))(
                aug_shadow)
            aug_shadow = kornia.filters.box_blur(aug_shadow.unsqueeze(0), (random.choice(range(20, 40)),) * 2)
            bg_im = bg_im.sub_(aug_shadow[0]).clamp_(0, 1)

        merge_img = im_fg * im_alpha + (1 - im_alpha) * bg_im
        prior = generateRandomPrior(im_alpha[0].cpu().numpy(), size=31)
        prior_trimap = prior.copy()
        prior_trimap[prior_trimap == -1] = 1
        label_alpha = im_alpha[0]
        trimap = get_trimap(label_alpha.cpu().numpy())

        return merge_img, label_alpha.unsqueeze(0), torch.from_numpy(trimap).unsqueeze(0), im_fg, bg_im, \
               torch.from_numpy(prior).unsqueeze(0), torch.from_numpy(prior_trimap).unsqueeze(0), item

    def get_val_data(self, item: int) -> VAL_TYPE:
        fg_index = item % len(self.fg_files)
        fg_name = self.fg_files[fg_index]
        label_name = self.labelPath_files[fg_index]
        assert label_name.split('.')[0] == fg_name.split('.')[0], 'name is not match'
        fg_im = cv2.imread(self.forthPath + fg_name)
        label_im = cv2.imread(self.labelPath + label_name)
        if len(label_im.shape) == 2:
            label_im = label_im
        elif label_im.shape[2] == 4:
            label_im = label_im[..., -1]
        else:
            label_im = label_im[:, :, 0]
        fg_im = scale_img(fg_im, mode='long', skip_small=True, size=self.out_size)
        label_im = scale_img(label_im, mode='long', skip_small=True, size=self.out_size)
        fg_im = padding_to_square(fg_im)
        label_im = padding_to_square(label_im)
        merge_img, label_alpha = fg_im, label_im
        merge_img = self.transform(merge_img)
        merge_gt = self.transform(np.array(label_alpha, dtype='uint8'))
        trimap = torch.from_numpy(get_trimap(merge_gt[0].cpu().numpy()))
        return merge_img, merge_gt, trimap, item

    def get_show_data(self, item: int) -> SHOW_TYPE:
        merge_img, merge_gt, trimap, item = self.get_val_data(item)
        return merge_img, merge_gt, item

    def __len__(self):
        return max(len(self.bg_files), len(self.fg_files))
