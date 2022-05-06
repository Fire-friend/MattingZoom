"""
Bridging Composite and Real: Towards End-to-end Deep Image Matting [IJCV-2021]
Main network file (GFM).

Copyright (c) 2021, Jizhizi Li (jili8515@uni.sydney.edu.au)
Licensed under the MIT License (see LICENSE for details)
Github repo: https://github.com/JizhiziLi/GFM
Paper link (Arxiv): https://arxiv.org/abs/2010.16188

"""
from typing import Tuple

import cv2
import kornia
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt

from utils.util import select_roi, show_tensor
from .att_bloacks import SCse, CBAM, NONLocalBlock2D
from .backbones import SUPPORTED_BACKBONES

from .layers import *


class MaskQuery(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MaskQuery, self).__init__()
        # self.query = BasicConv(in_channel, in_channel, kernel_size=3, stride=1, relu=True, norm=True)
        self.query_fc_in = nn.Linear(in_channel, out_channel)
        # self.query_fc = nn.Linear(out_channel, out_channel)
        self.dropout = nn.Dropout(0.1)
        self.layerNorm = nn.LayerNorm(in_channel)

    def forward(self, z, mask=None):
        # n, c, h, w = z.shape
        # mask = F.interpolate(mask, (h, w), mode='bilinear', align_corners=True)
        # cat_z_mask = torch.cat([z, mask], dim=1)
        # q = self.query(cat_z_mask)
        # q = F.adaptive_avg_pool2d(q, 1)
        # q = q.squeeze(-1).squeeze(-1)
        # q = self.query_fc(q).unsqueeze(1)

        n, c, h, w = z.shape
        if mask is not None:
            mask = F.interpolate(mask, (h, w), mode='bilinear', align_corners=True)
            q = torch.sum(z * mask, dim=[2, 3]) / torch.sum(mask, dim=[2, 3])
        else:
            q = z
            q = F.adaptive_avg_pool2d(q, 1)
            q = q.squeeze(-1).squeeze(-1)
        # q = self.layerNorm(q)
        q = self.query_fc_in(q).unsqueeze(1)
        return q


class FeatureKey(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FeatureKey, self).__init__()
        self.key = BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True, norm=True)
        self.key_fc = nn.Linear(out_channel, out_channel)

    def forward(self, z):
        cat_z_mask = z
        key = self.key(cat_z_mask)
        key = F.adaptive_avg_pool2d(key, 1)
        key = key.squeeze(-1).squeeze(-1)
        key = self.key_fc(key).unsqueeze(-1).unsqueeze(-1)
        return key


class WeightByKeyQuery(nn.Module):
    def __init__(self):
        super(WeightByKeyQuery, self).__init__()

    def forward(self, q, key_list):
        weight_list = [torch.sum(q * key, dim=1, keepdim=True) for key in key_list]
        return F.softmax(torch.cat(weight_list, dim=1), dim=1)


class FusionFeature(nn.Module):
    def __init__(self, out_channel, merge_num, in_channel_list, base_channel, mode='fg'):
        super(FusionFeature, self).__init__()
        self.scale_f = nn.ModuleList(
            [nn.Sequential(
                # SCse(in_channel_list[i]),
                BasicConv(in_channel_list[i], out_channel, 3, 1, relu=True, norm=True),

            ) for i in
                range(merge_num)])

        # self.scale_f_bb = nn.ModuleList(
        #     [nn.Sequential(build_bb(in_channel_list[i], out_channel, out_channel)) for i in
        #      # Fea_att(32)
        #      range(merge_num)])
        #
        # self.scale_f_psp = nn.ModuleList(
        #     [nn.Sequential(PSPModule(in_channel_list[i], out_channel)) for i in
        #      # Fea_att(32)
        #      range(merge_num)])

        # self.scale_f = nn.ModuleList(
        #     [nn.Sequential(BasicConv(int(in_channel_list[i] / 2), out_channel, 1, 1, relu=True, norm=True)) for i in
        #      # Fea_att(32)
        #      range(merge_num)])
        # self.fusion_feature = BasicConv(base_channel * merge_num, base_channel * merge_num, 3, 1, relu=True, norm=True)
        # self.fusion_feature = BasicConv(out_channel * merge_num, out_channel, 1, 1, relu=True, norm=True)
        self.mode = mode
        self.out_channel = out_channel

    def forward(self, weight, feature_list, index, head_index, group_channel_list, prior):
        # n, c, h, w = feature_list[index].shape
        # if self.mode == 'fg':
        #     cat_weight_feature = [
        #         weight[:, i:i + 1:] * F.interpolate(
        #             self.scale_f[i](feature_list[i][:, :int(feature_list[i].shape[1] / 2)]), (h, w),
        #             mode='bilinear',
        #             align_corners=True) for i in range(len(feature_list))]
        # else:
        #     cat_weight_feature = [
        #         weight[:, i:i + 1:] * F.interpolate(
        #             self.scale_f[i](feature_list[i][:, int(feature_list[i].shape[1] / 2):]), (h, w),
        #             mode='bilinear',
        #             align_corners=True) for i in range(len(feature_list))]
        # # cat_weight_feature = [weight[:, i:i + 1:] * feature_list[i] for i in range(len(feature_list))]
        # cat_weight_feature = torch.cat(cat_weight_feature, dim=1)
        # # fusion_feature = self.fusion_feature(cat_weight_feature)
        # return cat_weight_feature

        n, c, h, w = feature_list[index].shape
        fusion_feature = 0
        for i in range(len(feature_list)):
            _, c, h__, w__ = feature_list[i].shape
            prior_sc = F.interpolate(prior, (h__, w__), mode='bilinear')

            # if h__ == 32:
            #     #show
            #     plt.imshow(np.array((weight[:, i:i + 1:] * prior_sc)[0][0].detach().cpu().numpy()*255, dtype='uint8'))
            #     plt.show()

            if i > index:  # up
                fusion_feature += weight[:, i:i + 1:] * F.interpolate(
                    self.scale_f[i](prior_sc * feature_list[i][:,
                                               head_index * group_channel_list[i]:(head_index + 1) * group_channel_list[
                                                   i]]),
                    (h, w),
                    mode='bilinear',
                    align_corners=True)

                # fusion_feature += weight[:, i:i + 1:] * F.adaptive_avg_pool2d(
                #     self.scale_f_psp[i](feature_list[i][:,
                #                         head_index * group_channel_list[i]:(head_index + 1) * group_channel_list[i]]),
                #     (h, w)
                # )
            elif i < index:  # down
                fusion_feature += weight[:, i:i + 1:] * self.scale_f[i](F.interpolate(
                    prior_sc * feature_list[i][:,
                               head_index * group_channel_list[i]:(head_index + 1) * group_channel_list[i]],
                    (h, w),
                    mode='bilinear',
                    align_corners=True))

                # fusion_feature += weight[:, i:i + 1:] * self.scale_f_bb[i](F.adaptive_avg_pool2d(
                #     feature_list[i][:, head_index * group_channel_list[i]:(head_index + 1) * group_channel_list[i]],
                #     (h, w)
                # )
                # )
            else:
                fusion_feature += weight[:, i:i + 1:] * self.scale_f[i](prior_sc * feature_list[i]
                [:, head_index * group_channel_list[i]:(
                                                               head_index + 1) *
                                                       group_channel_list[
                                                           i]])

                # spl = torch.split(feature_list[i][:, head_index * group_channel_list[i]:(head_index + 1) *
                #                                                                         group_channel_list[i]],
                #                   self.out_channel, 1)
                # f = 0
                # for sp in spl:
                #     f += sp
                # fusion_feature += f / len(spl)

                # fusion_feature += weight[:, i:i + 1:] * feature_list[i][:, head_index * group_channel_list[i]:(
                #                                                                               head_index + 1) *
                #                                                                       group_channel_list[
                #                                                                           i]]

                # fusion_feature += PCA(feature_list[i][:, head_index * group_channel_list[i]:(head_index + 1) *group_channel_list[i]], self.out_channel)
        # fusion_feature = self.sa(fusion_feature) * fusion_feature
        return fusion_feature  # cat_weight_feature


class ReEncoder(nn.Module):
    def __init__(self, backbone_output_size_num, backbone_out_channels, re_out_channels, base_channel=32):
        super(ReEncoder, self).__init__()
        self.base_channel = base_channel
        self.backbone_output_size_num = backbone_output_size_num

        # self.scale_z = nn.ModuleList([
        #     BasicConv(in_channel=channel, out_channel=base_channel * backbone_output_size_num, stride=1, kernel_size=1,
        #               norm=True, relu=True) for channel in backbone_out_channels])
        self.num_attention_heads = 1
        self.attention_head_size = base_channel * self.num_attention_heads

        self.Query_fg = nn.ModuleList(
            [MaskQuery(channel, self.attention_head_size) for channel in backbone_out_channels])
        self.Query_bg = nn.ModuleList(
            [MaskQuery(channel, self.attention_head_size) for channel in backbone_out_channels])
        self.Key_fg = nn.ModuleList(
            [MaskQuery(channel, self.attention_head_size) for channel in backbone_out_channels])
        self.Key_bg = nn.ModuleList(
            [MaskQuery(channel, self.attention_head_size) for channel in backbone_out_channels])

        # self.Key = nn.ModuleList([FeatureKey(channel, base_channel) for channel in backbone_out_channels])
        # self.scale_fg = nn.ModuleList(
        #     [BasicConv(in_channel=channel, out_channel=channel, kernel_size=1, stride=1, relu=False, norm=False) for
        #      channel in backbone_out_channels])
        # self.scale_bg = nn.ModuleList(
        #     [BasicConv(in_channel=channel, out_channel=channel, kernel_size=1, stride=1, relu=False, norm=False) for
        #      channel in backbone_out_channels])

        self.se_backbone_out_channels = [int(ch / self.num_attention_heads) for ch in backbone_out_channels]
        re_out_channels = self.se_backbone_out_channels
        # self.fusionFeature_fg = nn.ModuleList(
        #     [FusionFeature(int(channel / self.num_attention_heads), backbone_output_size_num,
        #                    self.se_backbone_out_channels,
        #                    base_channel=base_channel, mode='fg') for channel in backbone_out_channels])
        # self.fusionFeature_bg = nn.ModuleList(
        #     [FusionFeature(int(channel / self.num_attention_heads), backbone_output_size_num,
        #                    self.se_backbone_out_channels,
        #                    base_channel=base_channel, mode='bg') for channel in backbone_out_channels])
        # self.fusionFeature_fgs = nn.ModuleList([self.fusionFeature_fg for i in range(self.num_attention_heads)])
        # self.fusionFeature_bgs = nn.ModuleList([self.fusionFeature_bg for i in range(self.num_attention_heads)])

        self.fusionFeature_fg = nn.ModuleList(
            [FusionFeature(channel, backbone_output_size_num,
                           self.se_backbone_out_channels,
                           base_channel=base_channel, mode='fg') for channel in re_out_channels])
        self.fusionFeature_bg = nn.ModuleList(
            [FusionFeature(channel, backbone_output_size_num,
                           self.se_backbone_out_channels,
                           base_channel=base_channel, mode='bg') for channel in re_out_channels])
        self.fusionFeature_fgs = nn.ModuleList([self.fusionFeature_fg for i in range(self.num_attention_heads)])
        self.fusionFeature_bgs = nn.ModuleList([self.fusionFeature_bg for i in range(self.num_attention_heads)])

        self.scale_fg = nn.ModuleList(
            [BasicConv(ch, ch, kernel_size=1, stride=1, relu=False, norm=False) for ch in backbone_out_channels])
        self.scale_bg = nn.ModuleList(
            [BasicConv(ch, ch, kernel_size=1, stride=1, relu=False, norm=False) for ch in backbone_out_channels])

        # self.WeightByQK = WeightByKeyQuery()
        self.att_dropout = nn.Dropout(0.1)

        # self.prior_fg_fines = nn.ModuleList(
        #     [nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
        #                    BasicConv(in_channel=4, out_channel=1, kernel_size=5, stride=1, relu=False, norm=False)
        #                    ) for i in range(backbone_output_size_num)])
        # self.prior_bg_fines = nn.ModuleList(
        #     [nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
        #                    BasicConv(in_channel=4, out_channel=1, kernel_size=5, stride=1, relu=False, norm=False)
        #                    ) for i in range(backbone_output_size_num)])
        # self.down = nn.UpsamplingBilinear2d(scale_factor=0.5)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads, int(self.attention_head_size / self.num_attention_heads))
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, z_list, prior_fg, prior_bg, x):
        # show_tensor(prior_fg[0][0],mode='gray')
        # show_tensor(prior_bg[0][0], mode='gray')
        # prior_bg = kornia.sobel(x)
        # prior_fg = self.prior_fg(x)
        # plt.subplot(1,2,1)
        # plt.imshow(prior_bg[0][0].detach().cpu().numpy(), cmap='jet')
        # plt.subplot(1, 2, 2)
        # plt.imshow(prior_fg[0][0].detach().cpu().numpy(), cmap='jet')
        # plt.show()

        # for i in range(len(z_list)):
        #     z_list[i] = self.scale_z[i](z_list[i])

        # ss = torch.cat([z.reshape(z.shape[0], -1) for z in z_list], dim=1)

        weight_feature_fg = []
        weight_feature_bg = []

        # weighted encoder
        key_list_fg = []
        key_list_bg = []
        q_list_fg = []
        q_list_bg = []

        # prior_fg_list = []
        # prior_bg_list = []
        # prior_fg_list.append(self.prior_fg_fines[0](torch.cat([prior_fg, x], dim=1)))
        # prior_bg_list.append(self.prior_bg_fines[0](torch.cat([prior_bg, x], dim=1)))
        # x = self.down(x)
        for i in range(len(self.Key_fg)):
            # n, c, h, w = z_list[i].shape

            # scale_fg_prior = F.interpolate(torch.sigmoid(prior_fg), (h, w), mode='bilinear', align_corners=True)
            # scale_bg_prior = F.interpolate(torch.sigmoid(prior_bg), (h, w), mode='bilinear', align_corners=True)
            # prior_fg_sc = self.prior_fg_fines[i](torch.cat([prior_fg, x], dim=1))
            # prior_bg_sc = self.prior_bg_fines[i](torch.cat([prior_bg, x], dim=1))
            # prior_fg = self.down(prior_fg)
            # prior_bg = self.down(prior_bg)
            # x = self.down(x)
            # prior_fg_list.append(prior_fg_sc)
            # prior_bg_list.append(prior_bg_sc)

            prior_fg_sc = prior_fg
            prior_bg_sc = prior_bg

            # show
            # att = torch.abs(prior_fg_sc + prior_bg_sc - 1)
            # show_tensor(att[0][0], mode='jet')
            # roi_index = select_roi(att, mode='hist')
            # show_tensor(roi_index[0][0], mode='gray')

            key_fg = self.Key_fg[i](z_list[i], prior_fg_sc)
            key_bg = self.Key_bg[i](z_list[i], prior_bg_sc)
            query_fg = self.Query_fg[i](z_list[i], prior_fg_sc)
            query_bg = self.Query_bg[i](z_list[i], prior_bg_sc)

            # query_fg = self.Query_fg[i](z_list[i])
            # query_bg = self.Query_bg[i](z_list[i])

            # if i + 1 < len(self.Key_fg):
            # prior_fg_list.append(self.prior_fg_fines[i + 1](torch.cat([prior_fg_list[i], x], dim=1)))
            # prior_bg_list.append(self.prior_bg_fines[i + 1](torch.cat([prior_bg_list[i], x], dim=1)))
            # x = self.down(x)

            if i == 0:
                key_list_fg = key_fg
                key_list_bg = key_bg
                q_list_fg = query_fg
                q_list_bg = query_bg
            else:
                key_list_fg = torch.cat([key_list_fg, key_fg], dim=1)
                key_list_bg = torch.cat([key_list_bg, key_bg], dim=1)
                q_list_fg = torch.cat([q_list_fg, query_fg], dim=1)
                q_list_bg = torch.cat([q_list_bg, query_bg], dim=1)

        q_list_fg = self.transpose_for_scores(q_list_fg)
        q_list_bg = self.transpose_for_scores(q_list_bg)
        key_list_fg = self.transpose_for_scores(key_list_fg)
        key_list_bg = self.transpose_for_scores(key_list_bg)
        att_score_fg = torch.matmul(q_list_fg, key_list_fg.transpose(-1, -2)) / np.sqrt(
            self.attention_head_size / self.num_attention_heads)
        att_score_bg = torch.matmul(q_list_bg, key_list_bg.transpose(-1, -2)) / np.sqrt(
            self.attention_head_size / self.num_attention_heads)
        att_score_fg = self.att_dropout(F.softmax(att_score_fg, dim=-1))
        att_score_bg = self.att_dropout(F.softmax(att_score_bg, dim=-1))

        for i in range(self.backbone_output_size_num):

            # n, c, h, w = z_list[i].shape
            # scale_fg_prior = prior_fg_list[i]
            # scale_bg_prior = prior_bg_list[i]

            temp_fg = []
            temp_bg = []
            for j in range(self.num_attention_heads):
                temp_fg.append(
                    self.fusionFeature_fgs[j][i](
                        att_score_fg[:, j, i, :].unsqueeze(2).unsqueeze(2), z_list, i, j,
                        self.se_backbone_out_channels, prior_fg))
                temp_bg.append(
                    self.fusionFeature_bgs[j][i](
                        att_score_bg[:, j, i, :].unsqueeze(2).unsqueeze(2), z_list, i, j,
                        self.se_backbone_out_channels, prior_bg))
            weight_feature_fg.append(
                (z_list[i] + self.scale_fg[i](torch.cat(temp_fg, dim=1))))
            # scale_fg_prior *  self.scale_fg[i]()torch.sigmoid(scale_fg_prior) *
            weight_feature_bg.append(
                (z_list[i] + self.scale_bg[i](torch.cat(temp_bg, dim=1))))  # z_list[i] +
            # scale_bg_prior *  self.scale_bg[i]() torch.sigmoid(scale_bg_prior) *

        del temp_fg, temp_bg, key_list_fg, key_list_bg, q_list_fg, q_list_bg

        return weight_feature_fg, weight_feature_bg, prior_fg, prior_fg


class PSDB_block(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 predict_channel,
                 skip_channels=None,
                 ):
        super(PSDB_block, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up_ne = nn.UpsamplingNearest2d(scale_factor=2)
        mid_hidd = out_channels  # int((in_channels + skip_channels) / 2)
        self.conv1 = BasicConv(in_channels + skip_channels, mid_hidd, stride=1,
                               kernel_size=3, norm=True, relu=True)
        self.conv2 = BasicConv(mid_hidd, out_channels, stride=1, kernel_size=3, norm=True,
                               relu=True)
        # self.scse = SCse(mid_hidd)
        self.non_local = NONLocalBlock2D(mid_hidd, sub_sample=True, bn_layer=True)
        # self.max_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.out = BasicConv(out_channels, predict_channel, stride=1, kernel_size=1, norm=False, relu=False)
        # self.out_detail = BasicConv(out_channels, predict_channel, stride=1, kernel_size=3, norm=False, relu=False)
        # self.out2 = BasicConv(out_channels, predict_channel, stride=2, kernel_size=3, norm=False, relu=False)
        self.down = nn.UpsamplingBilinear2d(scale_factor=0.5)
        # self.skip_detail = build_bb(skip_channels, int(skip_channels / 2), skip_channels)

    def forward(self, z, skip=None, uncertain_map=None, last_out=None, mode='semantic', input=None):
        z = self.up(z)
        # z = torch.cat([input,z], dim=1)
        if skip is not None:
            z = torch.cat([z, skip], dim=1)
        z = self.conv1(z)
        # z = self.non_local(z)
        # z = self.up(z)
        # z = self.scse(z)
        z_out = self.conv2(z)
        if uncertain_map is not None and last_out is not None:
            uncertain_map = self.up(uncertain_map)
            roi_index = select_roi(uncertain_map, mode='hist')
            semantic_out = self.out(z_out)
            out = self.up(last_out) * ~roi_index + roi_index * semantic_out

            # show
            # show_tensor(torch.argmax(last_out, dim=1)[0])
        else:
            out = self.out(z_out)
            semantic_out = uncertain_map

        # show
        # show_tensor(torch.argmax(out, dim=1)[0])

        return z_out, out, semantic_out


class SUEB_block(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 predict_channel,
                 skip_channels=None):
        super(SUEB_block, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up_ne = nn.UpsamplingNearest2d(scale_factor=2)
        mid_hidd = out_channels  # int((in_channels + skip_channels) / 2)
        self.conv1 = BasicConv(in_channels + skip_channels + 5, mid_hidd, stride=1,
                               kernel_size=3, norm=True, relu=True)
        self.conv2 = BasicConv(mid_hidd * 2, out_channels, stride=1, kernel_size=3, norm=True,
                               relu=True)
        self.conv_x = nn.Sequential(
            BasicConv(3, mid_hidd, stride=1, kernel_size=3, norm=True, relu=True),
            BasicConv(mid_hidd, mid_hidd, stride=1, kernel_size=3, norm=True, relu=True),
            BasicConv(mid_hidd, mid_hidd, stride=1, kernel_size=3, norm=True, relu=True)
        )
        # self.scse = SCse(mid_hidd)
        self.non_local = NONLocalBlock2D(mid_hidd, sub_sample=True, bn_layer=True)
        # self.max_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.out = BasicConv(out_channels, predict_channel, stride=1, kernel_size=1, norm=False, relu=False)
        # self.out_detail = BasicConv(out_channels, predict_channel, stride=1, kernel_size=1, norm=False, relu=False)
        # self.out2 = BasicConv(out_channels, predict_channel, stride=2, kernel_size=3, norm=False, relu=False)

        self.down = nn.UpsamplingBilinear2d(scale_factor=0.5)
        # self.skip_detail = build_bb(skip_channels, int(skip_channels / 2), skip_channels)

    def forward(self, z, x, uncertainty, skip, last_out):
        last_out = self.up(last_out)
        uncertainty = self.up_ne(uncertainty)
        z = self.up(z)
        n, c, h, w = z.shape
        x_scale = F.interpolate(x, (h, w), mode='bilinear', align_corners=True)
        z = torch.cat([x_scale, uncertainty, last_out, z], dim=1)
        if skip is not None:
            z = torch.cat([z, skip], dim=1)
        z = self.conv1(z)
        x_z = self.conv_x(x_scale)  # * uncertainty
        z = torch.cat([z, x_z], dim=1)
        # z = z * (1-uncertainty) + x_z * uncertainty
        # z = self.up(z)
        # z = self.scse(z)
        # z = self.non_local(z)
        z_out = self.conv2(z)
        detail_out = torch.sigmoid(self.out(z_out))

        # show
        # show_tensor(last_out[0][0])
        # show_tensor(uncertainty[0][0], mode='jet')
        # show_tensor(detail_out[0][0])

        return z_out, detail_out


class Decoder_PSDB(nn.Module):
    def __init__(self, in_channels=[32 + 1, 64 + 1, 128 + 1, 256, 160], skip_channels=[160, 160, 160, 160],
                 decoder_out_channels=[16, 32, 64, 128, 256], PSDB_layer=3):
        super(Decoder_PSDB, self).__init__()
        # self.conv_more_fg = build_bb(in_channels[-1], in_channels[-1], decoder_out_channels[-1])
        # self.conv_more_bg = build_bb(in_channels[-1], in_channels[-1], decoder_out_channels[-1])

        # self.conv_more_fg = ASPP(in_channels[-1], [3, 6, 9], decoder_out_channels[-1])
        # self.conv_more_bg = ASPP(in_channels[-1], [3, 6, 9], decoder_out_channels[-1])

        self.conv_more_fg = PSPModule(in_channels[-1], decoder_out_channels[-1], (1, 3, 5))
        self.conv_more_bg = PSPModule(in_channels[-1], decoder_out_channels[-1], (1, 3, 5))
        self.PSDB_layer = PSDB_layer  # [2,3,4]
        self.PSDB_fg = nn.ModuleList(
            [PSDB_block(in_channels=in_channels[-i - 2], skip_channels=skip_channels[-i - 2],
                        out_channels=decoder_out_channels[-i - 2], predict_channel=2) for i in
             range(self.PSDB_layer - 1)])
        self.PSDB_bg = nn.ModuleList(
            [PSDB_block(in_channels=in_channels[-i - 2], skip_channels=skip_channels[-i - 2],
                        out_channels=decoder_out_channels[-i - 2], predict_channel=2) for i in
             range(self.PSDB_layer - 1)])

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, fg_z_list, bg_z_list, z_list, x, mode='train'):
        n, c, h, w = x.shape

        z_fg = self.conv_more_fg(fg_z_list[-1])
        z_bg = self.conv_more_bg(bg_z_list[-1])
        fg_out_list = []
        bg_out_list = []
        att_map = None
        fg_out = None
        bg_out = None

        # PSDB
        for i in range(len(self.PSDB_fg)):
            _, _, hh, ww = fg_z_list[-i - 2].shape
            x_scale = F.interpolate(x, (hh, ww), mode='bilinear')
            de_fea_fg = z_fg
            de_fea_bg = z_bg
            z_fg, fg_out, _ = self.PSDB_fg[i](de_fea_fg,
                                              # skip=fg_z_list[-i - 2],
                                              skip=z_list[-i - 2],
                                              uncertain_map=att_map,
                                              last_out=fg_out,
                                              input=x_scale,
                                              mode='semantic')
            z_bg, bg_out, _ = self.PSDB_bg[i](de_fea_bg,
                                              # skip=bg_z_list[-i - 2],
                                              skip=z_list[-i - 2],
                                              uncertain_map=att_map,
                                              last_out=bg_out,
                                              input=x_scale,
                                              mode='semantic')
            fg_out_list.append(fg_out)
            bg_out_list.append(bg_out)
            with torch.no_grad():
                att_map = torch.abs(
                    torch.max(F.softmax(fg_out, dim=1), dim=1, keepdim=True)[0] +
                    torch.max(F.softmax(bg_out, dim=1), dim=1, keepdim=True)[0] - 2)

        return fg_out_list, bg_out_list, z_fg, z_bg


class Decoder_SUEB(nn.Module):
    def __init__(self, in_channels=[32 + 1, 64 + 1, 128 + 1, 256, 160], skip_channels=[160, 160, 160, 160],
                 decoder_out_channels=[16, 32, 64, 128, 256], PSDB_layer=3):
        super(Decoder_SUEB, self).__init__()
        self.PSDB_layer = PSDB_layer  # [2,3,4]

        self.SUEB_fg = nn.ModuleList(
            [SUEB_block(in_channels=in_channels[-i - 2], skip_channels=skip_channels[-i - 2],
                        out_channels=decoder_out_channels[-i - 2], predict_channel=1) for i in
             range(self.PSDB_layer - 1, len(decoder_out_channels) - 1)])
        self.SUEB_bg = nn.ModuleList(
            [SUEB_block(in_channels=in_channels[-i - 2], skip_channels=skip_channels[-i - 2],
                        out_channels=decoder_out_channels[-i - 2], predict_channel=1) for i in
             range(self.PSDB_layer - 1, len(decoder_out_channels) - 1)])

        # self.predictor_fg = Decoder_SUEB(in_channels=decoder_out_channels[0], predict_channel=1,
        #                                  out_channels=decoder_out_channels[0], skip_channels=0)
        # self.predictor_bg = Decoder_SUEB(in_channels=decoder_out_channels[0], predict_channel=1,
        #                                  out_channels=decoder_out_channels[0], skip_channels=0)

        self.predictor = nn.Sequential(
            BasicConv(2 * decoder_out_channels[0] + 5, decoder_out_channels[0], kernel_size=3, stride=1, relu=True,
                      norm=True),
            BasicConv(decoder_out_channels[0], 1, kernel_size=1, stride=1, relu=False, norm=False),
        )

        # self.fusion = BasicConv((len(decoder_out_channels) - (self.init_layer - 1)) * 2 + 2, 1, kernel_size=3, stride=1,
        #                         relu=False, norm=False)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.fusion_alpha = nn.Parameter(torch.tensor(0.5))
        self.erode_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # 可调整 kernel_size 和 padding

    def forward(self, fg_z_list, bg_z_list, z_list, x, fg_out_list, bg_out_list, z_fg, z_bg):
        detail_out_fg_list = []
        detail_out_bg_list = []
        fg_out = fg_out_list[-1]
        bg_out = bg_out_list[-1]

        with torch.no_grad():
            att_map = torch.abs(
                torch.max(F.softmax(fg_out, dim=1), dim=1, keepdim=True)[0] +
                torch.max(F.softmax(bg_out, dim=1), dim=1, keepdim=True)[0] - 2)
            # att_map = self.erode_pool(att_map)

        with torch.no_grad():
            fg_out = torch.argmax(fg_out, dim=1, keepdim=True).type(fg_out.dtype)
            bg_out = torch.argmax(bg_out, dim=1, keepdim=True).type(bg_out.dtype)

        # SUEB
        for i in range(len(self.SUEB_fg)):
            _, _, hh, ww = fg_z_list[-i - self.PSDB_layer - 1].shape
            de_fea_fg = z_fg
            de_fea_bg = z_bg
            z_fg, fg_out = self.SUEB_fg[i](de_fea_fg,
                                           x,
                                           att_map,
                                           fg_z_list[-i - self.PSDB_layer - 1],
                                           # z_list[-i - self.PSDB_layer - 1],
                                           fg_out)
            z_bg, bg_out = self.SUEB_bg[i](de_fea_bg,
                                           x,
                                           att_map,
                                           bg_z_list[-i - self.PSDB_layer - 1],
                                           # z_list[-i - self.PSDB_layer - 1],
                                           bg_out)
            detail_out_fg_list.append(fg_out)
            detail_out_bg_list.append(bg_out)
            att_map = torch.abs(fg_out - 1 + bg_out)

        x_fg_bg = self.up(torch.cat([z_fg, z_bg, fg_out, bg_out], dim=1))
        x_fg_bg = torch.cat([x_fg_bg, x], dim=1)
        # out = torch.sigmoid(self.predictor(x_fg_bg))
        out = self.predictor(x_fg_bg).clamp(0, 1)

        # show
        # show_tensor(att_map[0][0], mode='jet')
        # show_tensor(out[0][0])

        return fg_out_list, bg_out_list, detail_out_fg_list, detail_out_bg_list, out


class FBDM_Net(nn.Module):
    def __init__(self, args):
        super(FBDM_Net, self).__init__()

        self.backbone_arch = args.backbone
        # self.backbone_pretrained = backbone_pretrained
        self.base_channel = 32
        self.backbone = SUPPORTED_BACKBONES[self.backbone_arch](3)
        # self.prior_fg = AutoPrior()
        # self.prior_bg = AutoPrior()

        self.prior_fg = BasicConv(self.backbone.out_channels[-1], 1, relu=False, norm=False, kernel_size=1,
                                  stride=1)
        self.prior_bg = BasicConv(self.backbone.out_channels[-1], 1, relu=False, norm=False, kernel_size=1,
                                  stride=1)

        re_out_channels = [max(int(ch / 1), 0) for ch in
                           self.backbone.out_channels]  # + [self.backbone.out_channels[-1]]
        # re_out_channels = [32, 64, 64, 128, 256]
        self.reEncoder = ReEncoder(backbone_output_size_num=self.backbone.output_size_num,
                                   backbone_out_channels=self.backbone.out_channels,
                                   re_out_channels=re_out_channels,
                                   base_channel=self.base_channel)
        # if backbone_arch != 'resnet50':
        #     channels = [32, 64, 64, 128, 256]  # 32,
        # else:
        channels = [32, 64, 64, 128, 256]  # 32,
        # channels = [32, 64, 128, 256, 512]

        self.Decoder_PSDB = Decoder_PSDB(
            in_channels=channels[1:] + [re_out_channels[-1]],  # + [self.backbone.out_channels[-1]],
            # + [re_out_channels[-1]],  # + [self.backbone.out_channels[-1]],
            skip_channels=re_out_channels[:self.backbone.output_size_num],  # + [self.backbone.out_channels[-1]],
            # [channel for channel in self.backbone.out_channels],
            decoder_out_channels=channels)

        self.Decoder_SUEB = Decoder_SUEB(
            in_channels=channels[1:] + [re_out_channels[-1]],  # + [self.backbone.out_channels[-1]],
            # + [re_out_channels[-1]],  # + [self.backbone.out_channels[-1]],
            skip_channels=re_out_channels[:self.backbone.output_size_num],  # + [self.backbone.out_channels[-1]],
            # [channel for channel in self.backbone.out_channels],
            decoder_out_channels=channels)

        self.downScale = 2

        # self.grad = GradLayer()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         self._init_conv(m)
        #     elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
        #         self._init_norm(m)
        #
        # if self.backbone_pretrained:
        #     self.backbone.load_pretrained_ckpt(device)

    def forward(self, x):
        # z_list = self.backbone(torch.cat([x, prior_fg_pro, prior_bg_pro], dim=1))
        z_list = self.backbone(x)

        prior_fg = torch.sigmoid(self.prior_fg(z_list[-1]))
        prior_bg = torch.sigmoid(self.prior_bg(z_list[-1]))

        # weighted encoder
        weight_feature_fg, weight_feature_bg, prior_fg_list, prior_bg_list = self.reEncoder(z_list,
                                                                                            prior_fg,
                                                                                            prior_bg,
                                                                                            x)

        fg_out_list, bg_out_list, z_fg, z_bg = self.Decoder_PSDB(weight_feature_fg,
                                                                 weight_feature_bg,
                                                                 z_list,
                                                                 x)
        fg_out_list, bg_out_list, detail_out_fg_list, detail_out_bg_list, out = self.Decoder_SUEB(weight_feature_fg,
                                                                                                  weight_feature_bg,
                                                                                                  z_list,
                                                                                                  x,
                                                                                                  fg_out_list,
                                                                                                  bg_out_list,
                                                                                                  z_fg,
                                                                                                  z_bg)

        return fg_out_list, bg_out_list, detail_out_fg_list, detail_out_bg_list, prior_fg, prior_bg, out
