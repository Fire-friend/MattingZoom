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


def GFM_Trainer(
        net, image, trimap=None, gt_matte=None,
        mode='modnet', blurer=None, fg=None, bg=None, args=None, epoch=None):
    # forward the model
    with autocast():
        # pred_semantic, pred_detail, pred_matte, pred_source0, pred_source1, pred_source2, pred_source3 = net(image)
        pred_semantic, pred_detail, pred_matte = net(image)

        loss_global = get_crossentropy_loss(3, trimap, pred_semantic[:trimap.shape[0]])
        loss_local = get_alpha_loss(pred_detail[:trimap.shape[0]], gt_matte, trimap) + get_laplacian_loss(
            pred_detail[:trimap.shape[0]], gt_matte, trimap)

        loss_fusion_alpha = get_alpha_loss_whole_img(pred_matte[:trimap.shape[0]],
                                                     gt_matte) + get_laplacian_loss_whole_img(
            pred_matte[:trimap.shape[0]], gt_matte)

        loss_fusion_comp = get_composition_loss_whole_img(image[:trimap.shape[0]], gt_matte,
                                                          pred_matte[:trimap.shape[0]])

        loss = 0.25 * loss_global + 0.25 * loss_local + 0.25 * loss_fusion_alpha + 0.25 * loss_fusion_comp

    return {'loss': loss,
            'l_g': loss_global,
            'l_l': loss_local,
            'l_fa': loss_fusion_alpha,
            'l_fc': loss_fusion_comp}
