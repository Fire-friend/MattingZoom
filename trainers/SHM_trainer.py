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


def SHM_Trainer(
        net, image, trimap=None, gt_matte=None,
        mode='modnet', blurer=None, fg=None, bg=None, args=None, epoch=None):
    # forward the model
    with autocast():
        trimap_pre, alpha_pre = net(image)
        loss, L_alpha, L_composition, L_cross = loss_function_SHM(image,
                                                                  trimap_pre,
                                                                  trimap,
                                                                  alpha_pre,
                                                                  gt_matte)
    return {'loss': loss,
            'l_a': L_alpha,
            'l_c': L_composition,
            'l_cr': L_cross}
