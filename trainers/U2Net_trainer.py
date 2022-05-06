import math

import scipy
from kornia.losses import ssim
from scipy.ndimage import grey_dilation, grey_erosion
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from evaluate import *
from utils.loss_util import compute_bfd_loss_prior, muti_bce_loss_fusion, compute_bfd_loss, compute_bfd_loss_seg, \
    compute_bfd_loss_mat, compute_bfd_loss_mat_single, compute_bfd_loss_mat_single2, loss_function_SHM, loss_FBDM_img


def U2Net_Trainer(
        net, image, trimap=None, gt_matte=None,
        mode='modnet', blurer=None, fg=None, bg=None, args=None, epoch=None):
    # forward the model
    with autocast():
        d0, d1, d2, d3, d4, d5, d6 = net(image)
        loss0, loss1, loss2, loss3, loss4, loss5, loss6, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6,
                                                                                     gt_matte, trimap)

    return {'loss': loss,
            'l_0': loss0,
            'l_1': loss1,
            'l_2': loss2,
            'l_3': loss3,
            'l_4': loss4,
            'l_5': loss5}
