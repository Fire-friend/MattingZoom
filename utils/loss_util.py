import torch
import torch.nn.functional as F
import kornia
from torch import nn

from utils.evaluate import get_alpha_loss, get_crossentropy_loss


def crop_patch(x, idx, size, padding):
    """
    Crops selected patches from image given indices.

    Inputs:
        x: image (B, C, H, W).
        idx: selection indices Tuple[(P,), (P,), (P,),], where the 3 values are (B, H, W) index.
        size: center size of the patch, also stride of the crop.
        padding: expansion size of the patch.
    Output:
        patch: (P, C, h, w), where h = w = size + 2 * padding.
    """
    if padding != 0:
        x = F.pad(x, (padding,) * 4)

    # Use unfold. Best performance for PyTorch and TorchScript.
    return x.permute(0, 2, 3, 1) \
        .unfold(1, size + 2 * padding, size) \
        .unfold(2, size + 2 * padding, size)[idx[0], idx[1], idx[2]]


def l1_mask_loss(pred, gt, mask):
    # return torch.mean(torch.sum(F.l1_loss(pred * mask, gt * mask, reduction='none'), dim=[1,2,3]) / \
    #        (1e-10 + torch.sum(mask, dim=[1,2,3])))

    # return torch.sum(F.mse_loss(pred * mask, gt * mask, reduction='none')) / (1e-10 + torch.sum(mask))

    return F.mse_loss(pred * mask, gt * mask)


def mse_mask_loss(pred, gt, mask):
    # return F.mse_loss(pred, gt)
    # return torch.sum(F.l1_loss(pred * mask, gt * mask, reduction='none')) / (1e-10 + torch.sum(mask))

    return F.l1_loss(pred * mask, gt * mask)


def compute_bfd_loss_prior(prior, fg_out, bg_out, fgr_out, bgr_out,
                           fusion_fg, fusion_bg, fusion_fgr, fusion_bgr,
                           fg_refine, bg_refine, fgr_refine, bgr_refine,
                           matte_refine, gt_matte, image, trimap, fg, bg, stage='base', prior_trimap=None):
    # out_loss------------------------
    fg_out_loss = 0
    bg_out_loss = 0
    fgr_out_loss = 0
    bgr_out_loss = 0
    # semantic_gt_fg = gt_matte.clone()
    # semantic_gt_bg = 1 - gt_matte.clone()

    for i in range(len(fg_out)):
        with torch.no_grad():
            n, c, h, w = fg_out[i].shape
            # gt_fg = F.interpolate(gt_matte, (h, w), mode='bilinear', align_corners=False)
            gt_matte_scale = F.interpolate(gt_matte, (h, w), mode='bilinear', align_corners=False)
            gt_fg_index = gt_matte_scale != torch.min(gt_matte_scale.reshape(n, -1), dim=1, keepdim=True)[0].unsqueeze(
                2).unsqueeze(3)
            gt_bg_index = gt_matte_scale != torch.max(gt_matte_scale.reshape(n, -1), dim=1, keepdim=True)[0].unsqueeze(
                2).unsqueeze(3)
            scale_img = F.interpolate(image, (h, w), mode='bilinear', align_corners=False)
            # scale_trimap = F.interpolate(trimap, (h, w))
            # gt_detail_index = scale_trimap == 0.5
            if prior_trimap is not None:
                prior_trimap_scale = kornia.resize(prior_trimap, (h, w), interpolation='nearest')
                prior_index = prior_trimap_scale == 1
            else:
                prior_index = 1
        fg_out_loss += l1_mask_loss(fg_out[i], gt_matte_scale, prior_index) + \
                       mse_mask_loss(fg_out[i], gt_matte_scale, ~prior_index)
        bg_out_loss += l1_mask_loss(bg_out[i], (1 - gt_matte_scale), prior_index) + \
                       mse_mask_loss(bg_out[i], (1 - gt_matte_scale), ~prior_index)
        # fgr_out_loss += l1_mask_loss(torch.abs(fgr_out[i] + scale_img).clamp(0, 1) * gt_fg_index,
        #                           kornia.resize(fg, fgr_out[i].shape[2:]) * gt_fg_index, prior_index) + \
        #                 mse_mask_loss(torch.abs(fgr_out[i] + scale_img).clamp(0, 1) * gt_fg_index,
        #                           kornia.resize(fg, fgr_out[i].shape[2:]) * gt_fg_index, ~prior_index)
        # bgr_out_loss += l1_mask_loss(torch.abs(bgr_out[i] + scale_img).clamp(0, 1) * gt_bg_index,
        #                           kornia.resize(bg, bgr_out[i].shape[2:]) * gt_bg_index, prior_index) + \
        #                 mse_mask_loss(torch.abs(bgr_out[i] + scale_img).clamp(0, 1) * gt_bg_index,
        #                           kornia.resize(bg, bgr_out[i].shape[2:]) * gt_bg_index, ~prior_index)

    fg_out_loss /= len(fg_out)
    bg_out_loss /= len(bg_out)
    # fgr_out_loss /= len(fgr_out)
    # bgr_out_loss /= len(bgr_out)
    # out_loss = bgr_out_loss + fg_out_loss + err_out_loss  # + fgr_out_loss + bgr_out_loss

    # fusion_loss-----------------------------
    h, w = fusion_fg.shape[2:]
    fusion_img = kornia.resize(image, (h, w))
    fusion_gt = kornia.resize(gt_matte, (h, w))
    fusion_gt_msk = fusion_gt != torch.min(fusion_gt.reshape(n, -1), dim=1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)
    fusion_bg_msk = fusion_gt != torch.max(fusion_gt.reshape(n, -1), dim=1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)
    if prior_trimap is not None:
        prior_trimap_scale = kornia.resize(prior_trimap, (h, w), interpolation='nearest')
        prior_index = prior_trimap_scale == 1
    else:
        prior_index = 1
    if True:  # stage == 'base':
        # detail_fusion_index = F.interpolate(trimap, (h, w)) == 0.5
        # detail_fusion_loss = F.l1_loss(fusion_detail * detail_fusion_index,
        #                                kornia.resize(gt_matte, (h, w)) * detail_fusion_index)
        fg_fusion_loss = l1_mask_loss(fusion_fg, fusion_gt, prior_index) + \
                         l1_mask_loss(fusion_fg * fusion_img, fusion_gt * fusion_img, prior_index) + \
                         mse_mask_loss(fusion_fg, fusion_gt, ~prior_index) + \
                         mse_mask_loss(fusion_fg * fusion_img, fusion_gt * fusion_img, ~prior_index)
        bg_fusion_loss = l1_mask_loss(fusion_bg, (1 - fusion_gt), prior_index) + \
                         l1_mask_loss(fusion_bg * fusion_img, (1 - fusion_gt) * fusion_img, prior_index) + \
                         mse_mask_loss(fusion_bg, (1 - fusion_gt), ~prior_index) + \
                         mse_mask_loss(fusion_bg * fusion_img, (1 - fusion_gt) * fusion_img, ~prior_index)
        # gt_prior_index = trimap != 0.5
        # fgr_fusion_loss = l1_mask_loss(torch.abs(fusion_fgr + fusion_img).clamp(0, 1) * fusion_gt_msk,
        #                             kornia.resize(fg, (h, w)) * fusion_gt_msk, prior_index) + \
        #                   mse_mask_loss(torch.abs(fusion_fgr + fusion_img).clamp(0, 1) * fusion_gt_msk,
        #                             kornia.resize(fg, (h, w)) * fusion_gt_msk, ~prior_index)
        # bgr_fusion_loss = l1_mask_loss(torch.abs(fusion_bgr + fusion_img).clamp(0, 1) * fusion_bg_msk,
        #                             kornia.resize(bg, (h, w)) * fusion_bg_msk, prior_index) + \
        #                   mse_mask_loss(torch.abs(fusion_bgr + fusion_img).clamp(0, 1) * fusion_bg_msk,
        #                             kornia.resize(bg, (h, w)) * fusion_bg_msk, ~prior_index)
        # prior_loss = F.l1_loss(prior * gt_prior_index, trimap * gt_prior_index)
        fgr_fusion_loss = torch.tensor(0)
        bgr_fusion_loss = torch.tensor(0)
        prior_loss = torch.tensor(0)
        # fusion_loss = bg_fusion_loss + fg_fusion_loss + err_fusion_loss  # + fgr_fusion_loss + bgr_fusion_loss
    else:
        fg_fusion_loss = torch.tensor(0)
        bg_fusion_loss = torch.tensor(0)
        # gt_prior_index = trimap != 0.5
        fgr_fusion_loss = torch.tensor(0)
        bgr_fusion_loss = torch.tensor(0)
        # prior_loss = F.l1_loss(prior * gt_prior_index, trimap * gt_prior_index)
        prior_loss = torch.tensor(0)

    # refine_loss-------------------------
    if prior_trimap is not None:
        prior_index = prior_trimap == 1
    else:
        prior_index = 1
    if stage == 'refine':
        gt_index = gt_matte != 0
        bg_index = gt_matte != 1

        # # hard
        # with torch.no_grad():
        #     err = F.interpolate(torch.abs(fusion_fg - 1 + fusion_bg), scale_factor=4, mode='bilinear')
        #     roi_mask = err > (0.5 * torch.mean(err, dim=[2,3], keepdim=True))
        #
        # def refine_loss(a, b, roi):
        #     loss = torch.sum(F.l1_loss(a, b, reduction='none') * roi) / (torch.sum(roi) + 1e-10) + \
        #            torch.sum(F.l1_loss(kornia.sobel(a), kornia.sobel(b), reduction='none') * roi) / (torch.sum(roi) + 1e-10)
        #     return loss
        #
        # fg_refine_loss = refine_loss(fg_refine, gt_matte, roi_mask)
        # bg_refine_loss = refine_loss(bg_refine, (1-gt_matte), roi_mask)

        fg_refine_loss = l1_mask_loss(fg_refine, gt_matte, prior_index) + \
                         l1_mask_loss(kornia.sobel(fg_refine), kornia.sobel(gt_matte), prior_index) + \
                         mse_mask_loss(fg_refine, gt_matte, ~prior_index) + \
                         mse_mask_loss(kornia.sobel(fg_refine), kornia.sobel(gt_matte), ~prior_index)  # + \
        # mse_mask_loss(fg_refine * (torch.abs(fgr_refine + image).clamp(0, 1)) + (1 - fg_refine) * bg,
        #           image, ~prior_index)+ \
        # l1_mask_loss(fg_refine * (torch.abs(fgr_refine + image).clamp(0, 1)) + (1 - fg_refine) * bg,
        #           image, prior_index)

        bg_refine_loss = l1_mask_loss(bg_refine, 1 - gt_matte, prior_index) + \
                         l1_mask_loss(kornia.sobel(bg_refine), kornia.sobel(1 - gt_matte), prior_index) + \
                         mse_mask_loss(bg_refine, 1 - gt_matte, ~prior_index) + \
                         mse_mask_loss(kornia.sobel(bg_refine), kornia.sobel(1 - gt_matte), ~prior_index)  # + \
        # mse_mask_loss(bg_refine * (torch.abs(bgr_refine + image).clamp(0, 1)) + (1 - bg_refine) * fg,
        #           image, ~prior_index) + \
        # l1_mask_loss(bg_refine * (torch.abs(bgr_refine + image).clamp(0, 1)) + (1 - bg_refine) * fg,
        #    image, prior_index)

        # fgr_refine_loss = l1_mask_loss(torch.abs(fgr_refine + image).clamp(0, 1) * gt_index, fg * gt_index, prior_index) + \
        #                   mse_mask_loss(torch.abs(fgr_refine + image).clamp(0, 1) * gt_index, fg * gt_index, ~prior_index)
        # bgr_refine_loss = l1_mask_loss(torch.abs(bgr_refine + image).clamp(0, 1) * bg_index, bg * bg_index, prior_index) + \
        #                   mse_mask_loss(torch.abs(bgr_refine + image).clamp(0, 1) * bg_index, bg * bg_index, ~prior_index)

        fgr_refine_loss = torch.tensor(0)
        bgr_refine_loss = torch.tensor(0)
        # detail_refine_loss = torch.tensor(0)
    elif stage == 'base':
        fg_refine_loss = torch.tensor(0)
        bg_refine_loss = torch.tensor(0)
        fgr_refine_loss = torch.tensor(0)
        bgr_refine_loss = torch.tensor(0)

    # refine_loss = fg_refine_loss + bg_refine_loss  # + fgr_refine_loss + bgr_refine_loss

    # matte_loss--------------------------
    if stage == 'refine':
        matte_loss = l1_mask_loss(matte_refine, gt_matte, prior_index) + \
                     l1_mask_loss(kornia.sobel(matte_refine), kornia.sobel(gt_matte), prior_index) + \
                     l1_mask_loss(matte_refine * fg + (1 - matte_refine) * bg, image, prior_index) + \
                     mse_mask_loss(matte_refine, gt_matte, ~prior_index) + \
                     mse_mask_loss(kornia.sobel(matte_refine), kornia.sobel(gt_matte), ~prior_index) + \
                     mse_mask_loss(matte_refine * fg + (1 - matte_refine) * bg, image, ~prior_index)
        # matte_loss = torch.tensor(0)
    elif stage == 'base':
        matte_loss = torch.tensor(0)

    # stage loss sum

    # fg_loss
    fg_loss_sum = fg_out_loss + fg_fusion_loss + fg_refine_loss

    # bg_loss
    bg_loss_sum = bg_out_loss + bg_fusion_loss + bg_refine_loss

    # fgr_loss
    fgr_loss_sum = fgr_out_loss + fgr_fusion_loss + fgr_refine_loss

    # bgr_loss
    bgr_loss_sum = bgr_out_loss + bgr_fusion_loss + bgr_refine_loss

    # matte_loss
    matte_loss_sum = matte_loss

    # loss = out_loss + fusion_loss + refine_loss + matte_loss
    loss = prior_loss + fg_loss_sum + bg_loss_sum + matte_loss_sum + fgr_loss_sum + bgr_loss_sum
    return loss, prior_loss, fg_loss_sum, bg_loss_sum, matte_loss_sum, fgr_loss_sum, bgr_loss_sum


def compute_bfd_loss(prior, fg_out, bg_out, fgr_out, bgr_out,
                     fusion_fg, fusion_bg, fusion_fgr, fusion_bgr,
                     fg_refine, bg_refine, fgr_refine, bgr_refine,
                     matte_refine, gt_matte, image, trimap, fg, bg, stage='base'):
    # out_loss------------------------
    fg_out_loss = 0
    bg_out_loss = 0
    fgr_out_loss = 0
    bgr_out_loss = 0
    # semantic_gt_fg = gt_matte.clone()
    # semantic_gt_bg = 1 - gt_matte.clone()

    for i in range(len(fg_out)):
        with torch.no_grad():
            n, c, h, w = fg_out[i].shape
            # gt_fg = F.interpolate(gt_matte, (h, w), mode='bilinear', align_corners=False)
            gt_matte_scale = F.interpolate(gt_matte, (h, w), mode='bilinear', align_corners=False)
            # gt_fg_index = gt_matte_scale != torch.min(gt_matte_scale.reshape(n, -1), dim=1, keepdim=True)[0].unsqueeze(
            #     2).unsqueeze(3)
            # gt_bg_index = gt_matte_scale != torch.max(gt_matte_scale.reshape(n, -1), dim=1, keepdim=True)[0].unsqueeze(
            #     2).unsqueeze(3)
            gt_fg_index = gt_matte_scale != 0
            gt_bg_index = gt_matte_scale != 1
            scale_img = F.interpolate(image, (h, w), mode='bilinear', align_corners=False)
            # scale_trimap = F.interpolate(trimap, (h, w))
            # gt_detail_index = scale_trimap == 0.5
        fg_out_loss += F.l1_loss(torch.sigmoid(fg_out[i]), gt_matte_scale)
        bg_out_loss += F.l1_loss(torch.sigmoid(bg_out[i]), (1 - gt_matte_scale))
        fgr_out_loss += F.l1_loss(torch.abs(fgr_out[i] + scale_img).clamp(0, 1) * gt_fg_index,
                                  kornia.resize(fg, fgr_out[i].shape[2:]) * gt_fg_index)
        bgr_out_loss += F.l1_loss(torch.abs(bgr_out[i] + scale_img).clamp(0, 1) * gt_bg_index,
                                  kornia.resize(bg, bgr_out[i].shape[2:]) * gt_bg_index)

    fg_out_loss /= len(fg_out)
    bg_out_loss /= len(bg_out)
    fgr_out_loss /= len(fgr_out)
    bgr_out_loss /= len(bgr_out)
    # out_loss = bgr_out_loss + fg_out_loss + err_out_loss  # + fgr_out_loss + bgr_out_loss

    # fusion_loss-----------------------------
    h, w = fusion_fg.shape[2:]
    fusion_img = kornia.resize(image, (h, w))
    fusion_gt = kornia.resize(gt_matte, (h, w))
    fusion_gt_msk = fusion_gt != torch.min(fusion_gt.reshape(n, -1), dim=1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)
    fusion_bg_msk = fusion_gt != torch.max(fusion_gt.reshape(n, -1), dim=1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)
    if True:  # stage == 'base':
        # detail_fusion_index = F.interpolate(trimap, (h, w)) == 0.5
        # detail_fusion_loss = F.l1_loss(fusion_detail * detail_fusion_index,
        #                                kornia.resize(gt_matte, (h, w)) * detail_fusion_index)
        fg_fusion_loss = F.l1_loss(fusion_fg, fusion_gt) + \
                         F.l1_loss(fusion_fg * fusion_img, fusion_gt * fusion_img)
        bg_fusion_loss = F.l1_loss(fusion_bg, (1 - fusion_gt)) + \
                         F.l1_loss(fusion_bg * fusion_img, (1 - fusion_gt) * fusion_img)
        # gt_prior_index = trimap != 0.5
        fgr_fusion_loss = F.l1_loss(torch.abs(fusion_fgr + fusion_img).clamp(0, 1) * fusion_gt_msk,
                                    kornia.resize(fg, (h, w)) * fusion_gt_msk)
        bgr_fusion_loss = F.l1_loss(torch.abs(fusion_bgr + fusion_img).clamp(0, 1) * fusion_bg_msk,
                                    kornia.resize(bg, (h, w)) * fusion_bg_msk)
        # prior_loss = F.l1_loss(prior * gt_prior_index, trimap * gt_prior_index)
        prior_loss = torch.tensor(0)
        # fusion_loss = bg_fusion_loss + fg_fusion_loss + err_fusion_loss  # + fgr_fusion_loss + bgr_fusion_loss
    else:
        fg_fusion_loss = torch.tensor(0)
        bg_fusion_loss = torch.tensor(0)
        # gt_prior_index = trimap != 0.5
        fgr_fusion_loss = torch.tensor(0)
        bgr_fusion_loss = torch.tensor(0)
        # prior_loss = F.l1_loss(prior * gt_prior_index, trimap * gt_prior_index)
        prior_loss = torch.tensor(0)

    # refine_loss-------------------------
    if stage == 'refine':
        gt_index = gt_matte != 0
        bg_index = gt_matte != 1
        if prior_trimap is not None:
            prior_index = prior_trimap == 1
        else:
            prior_index = 1

        # # hard
        # with torch.no_grad():
        #     err = F.interpolate(torch.abs(fusion_fg - 1 + fusion_bg), scale_factor=4, mode='bilinear')
        #     roi_mask = err > (0.5 * torch.mean(err, dim=[2,3], keepdim=True))
        #
        # def refine_loss(a, b, roi):
        #     loss = torch.sum(F.l1_loss(a, b, reduction='none') * roi) / (torch.sum(roi) + 1e-10) + \
        #            torch.sum(F.l1_loss(kornia.sobel(a), kornia.sobel(b), reduction='none') * roi) / (torch.sum(roi) + 1e-10)
        #     return loss
        #
        # fg_refine_loss = refine_loss(fg_refine, gt_matte, roi_mask)
        # bg_refine_loss = refine_loss(bg_refine, (1-gt_matte), roi_mask)

        fg_refine_loss = F.l1_loss(fg_refine * prior_index, gt_matte * prior_index) + \
                         F.l1_loss(kornia.sobel(fg_refine), kornia.sobel(gt_matte)) + \
                         F.l1_loss(fg_refine * (torch.abs(fgr_refine + image).clamp(0, 1)) + (1 - fg_refine) * bg,
                                   image)

        bg_refine_loss = F.l1_loss(bg_refine, 1 - gt_matte) + \
                         F.l1_loss(kornia.sobel(bg_refine), kornia.sobel(1 - gt_matte)) + \
                         F.l1_loss(bg_refine * (torch.abs(bgr_refine + image).clamp(0, 1)) + (1 - bg_refine) * fg,
                                   image)

        fgr_refine_loss = F.l1_loss(torch.abs(fgr_refine + image).clamp(0, 1) * gt_index, fg * gt_index)
        bgr_refine_loss = F.l1_loss(torch.abs(bgr_refine + image).clamp(0, 1) * bg_index, bg * bg_index)

        # fg_refine_loss = torch.tensor(0)
        # bg_refine_loss = torch.tensor(0)
        # detail_refine_loss = torch.tensor(0)
    elif stage == 'base':
        fg_refine_loss = torch.tensor(0)
        bg_refine_loss = torch.tensor(0)
        fgr_refine_loss = torch.tensor(0)
        bgr_refine_loss = torch.tensor(0)

    # refine_loss = fg_refine_loss + bg_refine_loss  # + fgr_refine_loss + bgr_refine_loss

    # matte_loss--------------------------
    if stage == 'refine':
        matte_loss = F.l1_loss(matte_refine, gt_matte) + \
                     F.l1_loss(kornia.sobel(matte_refine), kornia.sobel(gt_matte)) + \
                     F.l1_loss(matte_refine * fg + (1 - matte_refine) * bg, image)
        # matte_loss = torch.tensor(0)
    elif stage == 'base':
        matte_loss = torch.tensor(0)

    # stage loss sum

    # fg_loss
    fg_loss_sum = fg_out_loss + fg_fusion_loss + fg_refine_loss

    # bg_loss
    bg_loss_sum = bg_out_loss + bg_fusion_loss + bg_refine_loss

    # fgr_loss
    fgr_loss_sum = fgr_out_loss + fgr_fusion_loss + fgr_refine_loss

    # bgr_loss
    bgr_loss_sum = bgr_out_loss + bgr_fusion_loss + bgr_refine_loss

    # matte_loss
    matte_loss_sum = matte_loss

    # loss = out_loss + fusion_loss + refine_loss + matte_loss
    loss = prior_loss + fg_loss_sum + bg_loss_sum + matte_loss_sum + fgr_loss_sum + bgr_loss_sum
    return loss, prior_loss, fg_loss_sum, bg_loss_sum, matte_loss_sum, fgr_loss_sum, bgr_loss_sum


def compute_bfd_loss_seg(prior, fg_out, bg_out, fgr_out, bgr_out,
                         fusion_fg, fusion_bg, fusion_fgr, fusion_bgr,
                         merge_fgr_refine, merge_alpha,
                         gt_matte, image, trimap, fg, bg, stage='base'):
    # out_loss------------------------
    fg_out_loss = 0
    bg_out_loss = 0
    fgr_out_loss = 0
    bgr_out_loss = 0
    # semantic_gt_fg = gt_matte.clone()
    # semantic_gt_bg = 1 - gt_matte.clone()

    for i in range(len(fg_out)):
        with torch.no_grad():
            n, c, h, w = fg_out[i].shape
            # gt_fg = F.interpolate(gt_matte, (h, w), mode='bilinear', align_corners=False)
            gt_matte_scale = F.interpolate(gt_matte, (h, w), mode='bilinear', align_corners=False)
            # gt_fg_index = gt_matte_scale != torch.min(gt_matte_scale.reshape(n, -1), dim=1, keepdim=True)[0].unsqueeze(
            #     2).unsqueeze(3)
            # gt_bg_index = gt_matte_scale != torch.max(gt_matte_scale.reshape(n, -1), dim=1, keepdim=True)[0].unsqueeze(
            #     2).unsqueeze(3)
            gt_fg_index = gt_matte_scale != 0
            gt_bg_index = gt_matte_scale != 1
            scale_img = F.interpolate(image, (h, w), mode='bilinear', align_corners=False)
            # scale_trimap = F.interpolate(trimap, (h, w))
            # gt_detail_index = scale_trimap == 0.5
        fg_out_loss += F.binary_cross_entropy_with_logits(fg_out[i], gt_matte_scale)
        bg_out_loss += F.binary_cross_entropy_with_logits(bg_out[i], (1 - gt_matte_scale))
        fgr_out_loss += F.l1_loss(torch.abs(fgr_out[i] + scale_img).clamp(0, 1) * gt_fg_index,
                                  kornia.resize(fg, fgr_out[i].shape[2:]) * gt_fg_index)
        bgr_out_loss += F.l1_loss(torch.abs(bgr_out[i] + scale_img).clamp(0, 1) * gt_bg_index,
                                  kornia.resize(bg, bgr_out[i].shape[2:]) * gt_bg_index)

    fg_out_loss /= len(fg_out)
    bg_out_loss /= len(bg_out)
    fgr_out_loss /= len(fgr_out)
    bgr_out_loss /= len(bgr_out)
    # out_loss = bgr_out_loss + fg_out_loss + err_out_loss  # + fgr_out_loss + bgr_out_loss

    # fusion_loss-----------------------------
    h, w = fusion_fg.shape[2:]
    fusion_img = kornia.resize(image, (h, w))
    fusion_gt = kornia.resize(gt_matte, (h, w))
    fusion_gt_msk = fusion_gt != torch.min(fusion_gt.reshape(n, -1), dim=1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)
    fusion_bg_msk = fusion_gt != torch.max(fusion_gt.reshape(n, -1), dim=1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)

    fg_fusion_loss = F.binary_cross_entropy_with_logits(fusion_fg, fusion_gt)
    bg_fusion_loss = F.binary_cross_entropy_with_logits(fusion_bg, (1 - fusion_gt))
    # gt_prior_index = trimap != 0.5
    fgr_fusion_loss = F.l1_loss(torch.abs(fusion_fgr + fusion_img).clamp(0, 1) * fusion_gt_msk,
                                kornia.resize(fg, (h, w)) * fusion_gt_msk)
    bgr_fusion_loss = F.l1_loss(torch.abs(fusion_bgr + fusion_img).clamp(0, 1) * fusion_bg_msk,
                                kornia.resize(bg, (h, w)) * fusion_bg_msk)
    # prior_loss = F.l1_loss(prior * gt_prior_index, trimap * gt_prior_index)
    prior_loss = torch.tensor(0)
    # fusion_loss = bg_fusion_loss + fg_fusion_loss + err_fusion_loss  # + fgr_fusion_loss + bgr_fusion_loss

    # refine_loss = fg_refine_loss + bg_refine_loss  # + fgr_refine_loss + bgr_refine_loss

    # show
    # import matplotlib.pyplot as plt
    # import numpy as np
    # plt.subplot(1, 4, 1)
    # plt.imshow(np.array(torch.sigmoid(merge_alpha[0][0]).detach().cpu().numpy() * 255, dtype='uint8'), cmap='gray')
    # plt.subplot(1, 4, 2)
    # plt.imshow(np.array(gt_matte[0][0].detach().cpu().numpy() * 255, dtype='uint8'), cmap='gray')
    # plt.subplot(1, 4, 3)
    # plt.imshow(np.array(torch.sigmoid(fusion_fg[0][0]).detach().cpu().numpy() * 255, dtype='uint8'), cmap='gray')
    # plt.subplot(1, 4, 4)
    # temp = prior[0][0].detach().cpu().numpy()
    # plt.imshow(np.array((temp - np.min(temp)) / (np.max(temp) - np.min(temp)) * 255, dtype='uint8'), cmap='jet')
    # plt.show()

    # matte_loss--------------------------
    if stage == 'refine':
        gt_index = gt_matte != 0
        matte_loss = F.binary_cross_entropy_with_logits(merge_alpha, gt_matte)
        merge_fgr_loss = F.l1_loss(torch.abs(merge_fgr_refine + image).clamp(0, 1) * gt_index, fg * gt_index)
    elif stage == 'base':
        matte_loss = torch.tensor(0)
        merge_fgr_loss = torch.tensor(0)

    # stage loss sum

    # fg_loss
    fg_loss_sum = fg_out_loss + fg_fusion_loss

    # bg_loss
    bg_loss_sum = bg_out_loss + bg_fusion_loss

    # fgr_loss
    fgr_loss_sum = fgr_out_loss + fgr_fusion_loss

    # bgr_loss
    bgr_loss_sum = bgr_out_loss + bgr_fusion_loss

    # matte_loss
    matte_loss_sum = matte_loss + merge_fgr_loss

    # loss = out_loss + fusion_loss + refine_loss + matte_loss
    loss = prior_loss + fg_loss_sum + bg_loss_sum + matte_loss_sum + fgr_loss_sum + bgr_loss_sum
    return loss, prior_loss, fg_loss_sum, bg_loss_sum, matte_loss_sum, fgr_loss_sum, bgr_loss_sum


def BCE(logits, target):
    logits = logits.clamp(0.01, 0.99)
    # logits: [N, *], target: [N, *]
    loss = - target * torch.log(logits) - \
           (1 - target) * torch.log(1 - logits)
    loss = loss.mean()
    return loss


def compute_bfd_loss_mat(fg_out_list, bg_out_list, detail_out_list_fg, detail_out_list_bg, out,
                         gt_matte, image, trimap, fg, bg, prior_fg, prior_bg, epoch):
    # show
    # import matplotlib.pyplot as plt
    # import numpy as np
    # plt.subplot(1, 3, 1)
    # plt.imshow(np.array(image.permute([0, 2, 3, 1])[0].detach().cpu().numpy() * 255, dtype='uint8'))
    # plt.subplot(1, 3, 2)
    # plt.imshow(np.array(gt_matte.permute([0, 2, 3, 1])[0].detach().cpu().numpy() * 255, dtype='uint8'), cmap='gray')
    # plt.subplot(1, 3, 3)
    # plt.imshow(np.array(out.permute([0, 2, 3, 1])[0].detach().cpu().numpy() * 255, dtype='uint8'), cmap='gray')
    # plt.savefig('./ss.png')
    st = -1
    # out_loss------------------------
    fg_out_loss = 0
    detail_out_loss = 0
    bg_out_loss = 0
    # semantic_loss = 0
    _, s_h, s_w, s_c = gt_matte.shape
    # fg_semantic
    for i in range(len(fg_out_list)):
        with torch.no_grad():
            n, c, h, w = fg_out_list[i].shape
            gt_matte_scale = F.interpolate(gt_matte, (h, w), mode='bilinear', align_corners=True)
            roi_index = (gt_matte_scale > 0) * (gt_matte_scale < 1)
            gt_matte_scale[roi_index] = 1
            roi_index = roi_index.squeeze(1)
            gt_matte_scale = gt_matte_scale.squeeze(1).long()
        fg_out_loss += torch.sum(
            ~roi_index * F.cross_entropy(fg_out_list[i], gt_matte_scale, reduction='none')) / torch.sum(~roi_index)

        # fg_out_loss += torch.sum(~roi_index *
        #     F.binary_cross_entropy_with_logits(fg_out_list[i], gt_matte_scale, reduction='none')) \
        #                / torch.sum(~roi_index)

        # fg_out_loss += F.binary_cross_entropy_with_logits(fg_out_list[i], gt_matte_scale)
        # fg_out_loss += F.l1_loss(fg_out_list[i], gt_matte_scale)
        #     # + F.l1_loss(kornia.sobel(fg_out_list[i]),

        # bg_semantic
        for i in range(len(bg_out_list)):
            with torch.no_grad():
                n, c, h, w = fg_out_list[i].shape
                gt_matte_scale = F.interpolate(gt_matte, (h, w), mode='bilinear', align_corners=True)
                roi_index = (gt_matte_scale > 0) * (gt_matte_scale < 1)
                gt_matte_scale[roi_index] = 1
                roi_index = roi_index.squeeze(1)
                gt_matte_scale = gt_matte_scale.squeeze(1).long()
            bg_out_loss += torch.sum(
                ~roi_index * F.cross_entropy(bg_out_list[i], 1 - gt_matte_scale, reduction='none')) / torch.sum(
                ~roi_index)

            # bg_out_loss += torch.sum(~roi_index *
            #     F.binary_cross_entropy_with_logits(bg_out_list[i], (1 - gt_matte_scale), reduction='none')) \
            #                / torch.sum(~roi_index)

            # bg_out_loss += F.binary_cross_entropy_with_logits(bg_out_list[i], 1-gt_matte_scale)
            # bg_out_loss += F.l1_loss(bg_out_list[i], 1 - gt_matte_scale)

    # detail_loss_fg
    for i in range(len(detail_out_list_fg)):
        with torch.no_grad():
            n, c, h, w = detail_out_list_fg[i].shape
            gt_matte_scale = F.interpolate(gt_matte, (h, w), mode='bilinear', align_corners=True)
            # trimap_scale = F.interpolate(trimap, (h, w))
            uncer = torch.abs(detail_out_list_fg[i] + detail_out_list_bg[i] - 1)
        # detail_out_loss += get_alpha_loss(detail_out_list_fg[i], gt_matte_scale, trimap_scale)
        detail_out_loss += torch.mean(F.l1_loss(detail_out_list_fg[i], gt_matte_scale, reduction='none') * uncer)

    # detail_loss_bg
    for i in range(len(detail_out_list_bg)):
        with torch.no_grad():
            n, c, h, w = detail_out_list_bg[i].shape
            gt_matte_scale = F.interpolate(gt_matte, (h, w), mode='bilinear', align_corners=True)
            # trimap_scale = F.interpolate(trimap, (h, w))
            uncer = torch.abs(detail_out_list_fg[i] + detail_out_list_bg[i] - 1)
        # detail_out_loss += get_alpha_loss(detail_out_list_bg[i], 1 - gt_matte_scale, trimap_scale)
        detail_out_loss += torch.mean(F.l1_loss(detail_out_list_bg[i], 1 - gt_matte_scale, reduction='none') * uncer)

    fg_out_loss /= len(fg_out_list)
    bg_out_loss /= len(bg_out_list)
    detail_out_loss /= (len(detail_out_list_fg) + len(detail_out_list_bg))

    # matte_loss--------------------------
    if epoch > st:
        matte_loss = F.l1_loss(out, gt_matte)  # + F.l1_loss(kornia.sobel(out), kornia.sobel(gt_matte))
    else:
        matte_loss = torch.tensor(0)

    # prior_loss
    # prior_fg_sum = 0
    # prior_bg_sum = 0
    # for i in range(len(prior_fg)):
    #     with torch.no_grad():
    #         n, c, h, w = prior_fg[i].shape
    #         gt_matte_scale = F.interpolate(gt_matte, (h, w), mode='bilinear', align_corners=False)
    #
    #     prior_fg_sum += F.binary_cross_entropy_with_logits(prior_fg[i], gt_matte_scale)
    #     prior_bg_sum += F.binary_cross_entropy_with_logits(prior_bg[i], (1 - gt_matte_scale))
    # prior_fg_sum /= len(prior_fg)
    # prior_bg_sum /= len(prior_bg)

    # prior_loss_sum = 0.5 * prior_fg_sum + 0.5 * prior_bg_sum

    gt_matte_scale = F.interpolate(gt_matte, (prior_fg.shape[2], prior_fg.shape[3]), mode='bilinear',
                                   align_corners=True)
    prior_loss_sum = 0.5 * F.mse_loss(prior_fg, gt_matte_scale) + \
                     0.5 * F.mse_loss(prior_bg, 1 - gt_matte_scale)
    loss = fg_out_loss + bg_out_loss + detail_out_loss + matte_loss + 0.1 * prior_loss_sum
    return loss, fg_out_loss, bg_out_loss, detail_out_loss, prior_loss_sum, matte_loss


def compute_bfd_loss_mat_single2(out_f, out_b, out_d, out,
                                 gt_matte, image, trimap, fg, bg, prior_fg, prior_bg):
    # out_loss------------------------
    fg_out_loss = 0
    bg_out_loss = torch.tensor(0)
    # semantic_loss = 0

    # roi_index = (gt_matte > 0) * (gt_matte < 1)
    # gt_matte[roi_index] = 1
    # roi_index = roi_index.squeeze(1)
    # gt_matte_scale = gt_matte.squeeze(1).long()
    gt_matte_scale = trimap.clone().squeeze(1)
    gt_matte_scale[gt_matte_scale == 0.5] = 2
    fg_out_loss += F.cross_entropy(out_f, gt_matte_scale.long())
    # fg_out_loss += get_crossentropy_loss(3, trimap, out_f)
    # import matplotlib.pyplot as plt
    # plt.imshow(gt_matte_scale[0].detach().cpu().numpy())
    # plt.show()
    # plt.imshow(trimap[0][0].detach().cpu().numpy())
    # plt.show()
    detail_out_loss = get_alpha_loss(out_d, gt_matte, trimap)

    # matte_loss--------------------------
    matte_loss = F.l1_loss(out, gt_matte)  # + F.l1_loss(kornia.sobel(out), kornia.sobel(gt_matte))

    # fg_loss
    fg_loss_sum = fg_out_loss

    # bg_loss
    bg_loss_sum = bg_out_loss

    # detail_loss
    detail_loss_sum = detail_out_loss

    # matte_loss
    matte_loss_sum = matte_loss

    # prior_loss
    # prior_fg_sum = 0
    # prior_bg_sum = 0
    # for i in range(len(prior_fg)):
    #     with torch.no_grad():
    #         n, c, h, w = prior_fg[i].shape
    #         gt_matte_scale = F.interpolate(gt_matte, (h, w), mode='bilinear', align_corners=False)
    #
    #     prior_fg_sum += F.binary_cross_entropy_with_logits(prior_fg[i], gt_matte_scale)
    #     prior_bg_sum += F.binary_cross_entropy_with_logits(prior_bg[i], (1 - gt_matte_scale))
    # prior_fg_sum /= len(prior_fg)
    # prior_bg_sum /= len(prior_bg)

    # prior_loss_sum = 0.5 * prior_fg_sum + 0.5 * prior_bg_sum

    prior_loss_sum = 0.5 * F.binary_cross_entropy_with_logits(prior_fg, gt_matte) + \
                     0.5 * F.binary_cross_entropy_with_logits(prior_bg, 1 - gt_matte)
    loss = fg_loss_sum + bg_loss_sum + detail_loss_sum + matte_loss_sum  # + 0.1 * prior_loss_sum
    return loss, fg_loss_sum, bg_loss_sum, detail_loss_sum, matte_loss_sum, prior_loss_sum


def compute_bfd_loss_mat_single(out_f, out_b, out_d, out,
                                gt_matte, image, trimap, fg, bg, prior_fg, prior_bg):
    # out_loss------------------------
    fg_out_loss = 0
    bg_out_loss = 0
    # semantic_loss = 0

    roi_index = (gt_matte > 0) * (gt_matte < 1)
    gt_matte[roi_index] = 1
    roi_index = roi_index.squeeze(1)
    gt_matte_scale = gt_matte.squeeze(1).long()
    fg_out_loss += torch.sum(
        ~roi_index * F.cross_entropy(out_f, gt_matte_scale, reduction='none')) / torch.sum(~roi_index)
    bg_out_loss += torch.sum(
        ~roi_index * F.cross_entropy(out_b, 1 - gt_matte_scale, reduction='none')) / torch.sum(~roi_index)

    detail_out_loss = get_alpha_loss(out_d, gt_matte, trimap)

    # matte_loss--------------------------
    matte_loss = F.l1_loss(out, gt_matte) + F.l1_loss(kornia.sobel(out), kornia.sobel(gt_matte))

    # fg_loss
    fg_loss_sum = fg_out_loss

    # bg_loss
    bg_loss_sum = bg_out_loss

    # detail_loss
    detail_loss_sum = detail_out_loss

    # matte_loss
    matte_loss_sum = matte_loss

    # prior_loss
    # prior_fg_sum = 0
    # prior_bg_sum = 0
    # for i in range(len(prior_fg)):
    #     with torch.no_grad():
    #         n, c, h, w = prior_fg[i].shape
    #         gt_matte_scale = F.interpolate(gt_matte, (h, w), mode='bilinear', align_corners=False)
    #
    #     prior_fg_sum += F.binary_cross_entropy_with_logits(prior_fg[i], gt_matte_scale)
    #     prior_bg_sum += F.binary_cross_entropy_with_logits(prior_bg[i], (1 - gt_matte_scale))
    # prior_fg_sum /= len(prior_fg)
    # prior_bg_sum /= len(prior_bg)

    # prior_loss_sum = 0.5 * prior_fg_sum + 0.5 * prior_bg_sum

    prior_loss_sum = 0.5 * F.binary_cross_entropy_with_logits(prior_fg, gt_matte) + \
                     0.5 * F.binary_cross_entropy_with_logits(prior_bg, 1 - gt_matte)
    loss = fg_loss_sum + bg_loss_sum + detail_loss_sum  # + matte_loss_sum  # + 0.1 * prior_loss_sum
    return loss, fg_loss_sum, bg_loss_sum, detail_loss_sum, matte_loss_sum, prior_loss_sum


def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v, trimap):
    # focal_bce = BCEFocalLoss(2, 0.5)
    bce = BCE
    loss0 = bce(d0, labels_v)
    loss1 = bce(d1, labels_v)
    loss2 = bce(d2, labels_v)
    loss3 = bce(d3, labels_v)
    loss4 = bce(d4, labels_v)
    loss5 = bce(d5, labels_v)
    loss6 = bce(d6, labels_v)
    # index = trimap == 0.5
    #
    # loss0 = F.l1_loss(d0, labels_v) + F.l1_loss(kornia.sobel(d0), kornia.sobel(labels_v)) + F.l1_loss(d0 * index,
    #                                                                                                   labels_v * index)
    # loss1 = F.l1_loss(d1, labels_v) + F.l1_loss(kornia.sobel(d1), kornia.sobel(labels_v))
    # loss2 = F.l1_loss(d2, labels_v) + F.l1_loss(kornia.sobel(d2), kornia.sobel(labels_v))
    # loss3 = F.l1_loss(d3, labels_v) + F.l1_loss(kornia.sobel(d3), kornia.sobel(labels_v))
    # loss4 = F.l1_loss(d4, labels_v) + F.l1_loss(kornia.sobel(d4), kornia.sobel(labels_v))
    # loss5 = F.l1_loss(d5, labels_v) + F.l1_loss(kornia.sobel(d5), kornia.sobel(labels_v))
    # loss6 = F.l1_loss(d6, labels_v) + F.l1_loss(kornia.sobel(d6), kornia.sobel(labels_v))

    # loss0 = focal_bce(d0, labels_v) + F.l1_loss(kornia.sobel(torch.sigmoid(d0)), kornia.sobel(labels_v))
    # loss1 = focal_bce(d1, labels_v) + F.l1_loss(kornia.sobel(torch.sigmoid(d1)), kornia.sobel(labels_v))
    # loss2 = focal_bce(d2, labels_v) + F.l1_loss(kornia.sobel(torch.sigmoid(d2)), kornia.sobel(labels_v))
    # loss3 = focal_bce(d3, labels_v) + F.l1_loss(kornia.sobel(torch.sigmoid(d3)), kornia.sobel(labels_v))
    # loss4 = focal_bce(d4, labels_v) + F.l1_loss(kornia.sobel(torch.sigmoid(d4)), kornia.sobel(labels_v))
    # loss5 = focal_bce(d5, labels_v) + F.l1_loss(kornia.sobel(torch.sigmoid(d5)), kornia.sobel(labels_v))
    # loss6 = focal_bce(d6, labels_v) + F.l1_loss(kornia.sobel(torch.sigmoid(d6)), kornia.sobel(labels_v))

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    # print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (
    # loss0.data.item(), loss1.data.item(), loss2.data.item(), loss3.data.item(), loss4.data.item(), loss5.data.item(),
    # loss6.data.item()))

    return loss0, loss1, loss2, loss3, loss4, loss5, loss6, loss


class BCEFocalLoss(torch.nn.Module):
    """
    二分类的Focalloss alpha 固定
    """

    def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        pt = torch.sigmoid(_input).clamp(0.01, 0.99)
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


def loss_function_SHM(img, trimap_pre, trimap_gt, alpha_pre, alpha_gt):
    # -------------------------------------
    # classification loss L_t
    # ------------------------
    # Cross Entropy
    # criterion = nn.BCELoss()
    # trimap_pre = trimap_pre.contiguous().view(-1)
    # trimap_gt = trimap_gt.view(-1)
    # L_t = criterion(trimap_pre, trimap_gt)

    criterion = nn.CrossEntropyLoss()
    L_t = criterion(trimap_pre, trimap_gt[:, 0, :, :].long())

    # -------------------------------------
    # prediction loss L_p
    # ------------------------
    eps = 1e-6
    # l_alpha
    L_alpha = torch.sqrt(torch.pow(alpha_pre - alpha_gt, 2.) + eps).mean()

    # L_composition
    fg = torch.cat((alpha_gt, alpha_gt, alpha_gt), 1) * img
    fg_pre = torch.cat((alpha_pre, alpha_pre, alpha_pre), 1) * img

    L_composition = torch.sqrt(torch.pow(fg - fg_pre, 2.) + eps).mean()

    L_p = 0.5 * L_alpha + 0.5 * L_composition

    # train_phase
    loss = L_p + 0.01 * L_t

    return loss, L_alpha, L_composition, L_t


def loss_FBDM_img(I_sigmoid, fusion_sigmoid, gt_matte, Fg):
    f_index = gt_matte == 1
    b_index = gt_matte == 0
    t_index = (gt_matte != 1) * (gt_matte != 0)
    bg = torch.zeros_like(I_sigmoid)-1
    f_loss = torch.sum(F.l1_loss(Fg, I_sigmoid, reduction='none') * f_index) / torch.sum(f_index)
    b_loss = torch.sum(F.l1_loss(bg, I_sigmoid, reduction='none') * b_index) / torch.sum(b_index)
    # f_b_loss = torch.sum(F.l1_loss(f_sigmoid - b_sigmoid, image, reduction='none') * ~t_index) / torch.sum(~t_index)


    t_loss = torch.sum(
        F.l1_loss((I_sigmoid - bg), gt_matte * (Fg - bg), reduction='none') * t_index) / torch.sum(
        t_index)
    # m_loss = F.l1_loss(gt_matte * (f_sigmoid - b_sigmoid), image - b_sigmoid)
    # m_loss = F.l1_loss(gt_matte * f_sigmoid + (1 - gt_matte) * b_sigmoid, image)
    m_loss = F.l1_loss(gt_matte, fusion_sigmoid)
    loss = f_loss + b_loss + t_loss + m_loss
    return loss, f_loss, b_loss, t_loss, m_loss
