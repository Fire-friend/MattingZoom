import math

import scipy
from kornia.losses import ssim
from scipy.ndimage import grey_dilation, grey_erosion
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from utils.evaluate import *
from utils.loss_util import compute_bfd_loss_prior, muti_bce_loss_fusion, compute_bfd_loss, compute_bfd_loss_seg, \
    compute_bfd_loss_mat, compute_bfd_loss_mat_single, compute_bfd_loss_mat_single2, loss_function_SHM, loss_FBDM_img


def P3M_Trainer(
        net, image, trimap=None, gt_matte=None,
        mode='modnet', blurer=None, fg=None, bg=None, args=None, epoch=None):
    # forward the model
    with autocast():
        predict_global, predict_local, predict_global_side2, predict_global_side1, predict_global_side0, predict_fusion = net(image)
        predict_fusion = predict_fusion.cuda()
        loss_global = get_crossentropy_loss(3, trimap, predict_global)
        loss_global_side2 = get_crossentropy_loss(3, trimap, predict_global_side2)
        loss_global_side1 = get_crossentropy_loss(3, trimap, predict_global_side1)
        loss_global_side0 = get_crossentropy_loss(3, trimap, predict_global_side0)
        loss_global = loss_global_side2 + loss_global_side1 + loss_global_side0 + 3 * loss_global
        loss_local = get_alpha_loss(predict_local, gt_matte, trimap) + get_laplacian_loss(predict_local, gt_matte, trimap)
        loss_fusion_alpha = get_alpha_loss_whole_img(predict_fusion, gt_matte) + get_laplacian_loss_whole_img(
            predict_fusion, gt_matte)
        loss_fusion_comp = get_composition_loss_whole_img_p3m(image, gt_matte, fg, bg, predict_fusion)
        loss = loss_global / 6 + loss_local * 2 + loss_fusion_alpha * 2 + loss_fusion_alpha + loss_fusion_comp

    return {'loss': loss,
            'l_g': loss_global,
            'l_o': loss_local,
            'l_a': loss_fusion_alpha,
            'l_c': loss_fusion_comp}