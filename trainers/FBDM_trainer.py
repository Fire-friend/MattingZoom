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


def FBDM_Trainer(
        net, image, trimap=None, gt_matte=None,
        mode='modnet', blurer=None, fg=None, bg=None, args=None, epoch=None):
    # forward the model
    with autocast():
        fg_out_list, bg_out_list, detail_out_fg_list, detail_out_bg_list, prior_fg, prior_bg, out = net(image)
        loss, fg_out_loss, bg_out_loss, detail_out_loss, prior_loss_sum, matte_loss = compute_bfd_loss_mat(
            fg_out_list=fg_out_list, bg_out_list=bg_out_list, detail_out_list_fg=detail_out_fg_list,
            detail_out_list_bg=detail_out_bg_list, out=out,
            gt_matte=gt_matte, image=image, trimap=trimap, fg=fg, bg=bg, prior_fg=prior_fg, prior_bg=prior_bg,
            epoch=epoch)

    return {'loss': loss,
            'l_f': fg_out_loss,
            'l_b': bg_out_loss,
            'l_d': detail_out_loss,
            'l_p': prior_loss_sum,
            'l_m': matte_loss}
