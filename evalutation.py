import os
import sys
import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from cv2 import resize
from datasets.data_util import *
# from pytorch_toolbelt.inference import tta

from utils.eval import computeAllMatrix
from utils.util import get_yaml_data, set_yaml_to_args, getPackByNameUtil
from tqdm import tqdm

parser = argparse.ArgumentParser()
# parser.add_argument('--input-path', type=str, help='path of input images',
#                     default='/data/wjw/work/matting_set/data/PPM-100/val/image/')
# parser.add_argument('--gt-path', type=str, help='path of output images',
#                     default='/data/wjw/work/matting_tool_study/test_results/')
# parser.add_argument('--output-path', type=str, help='path of output images',
#                     default='/data/wjw/work/matting_tool_study/test_results/')
# parser.add_argument('--ckpt-path', type=str, help='path of pre-trained MODNet',
#                     default='./checkSave/FBDMv2/AM2K/19/checkpoint/model_best')
parser.add_argument('--model', type=str, help='path of pre-trained MODNet',
                    default='FBDMv2')
parser.add_argument('--gpu', type=str, help='path of pre-trained MODNet',
                    default='3')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if __name__ == '__main__':
    # define cmd arguments

    # check input arguments
    # if not os.path.exists(args.input_path):
    #     print('Cannot find input path: {0}'.format(args.input_path))
    #     exit()
    # if not os.path.exists(args.output_path):
    #     print('Cannot find output path: {0}'.format(args.output_path))
    #     exit()
    # if not os.path.exists(args.ckpt_path):
    #     print('Cannot find ckpt path: {0}'.format(args.ckpt_path))
    #     exit()

    yamls_dict = get_yaml_data('./config/' + args.model + '_config.yaml')
    set_yaml_to_args(args, yamls_dict)
    args.ckpt_path = './checkSave/{}/{}/{}/checkpoint/model_best'.format(args.model, args.data_set, args.save_file)
    # get the dataset dynamically
    dataset = getPackByNameUtil(py_name='datasets.' + args.loader_mode + '_dataset',
                                object_name=args.loader_mode + '_Dataset')
    # get the model dynamically
    model = getPackByNameUtil(py_name='models.' + args.model + '_net',
                              object_name=args.model + '_Net')
    val_dataset = dataset(args, mode='val')
    val_loader = DataLoader(val_dataset,
                            shuffle=False,
                            num_workers=1,
                            batch_size=1,
                            pin_memory=True)
    net = model(args).cuda()
    # load pretrained parameters
    if args.ckpt_path != '' and args.ckpt_path is not None:
        print("loading from {}".format(args.ckpt_path))
        saved_state_dict = torch.load(args.ckpt_path)
        new_params = net.state_dict().copy()
        for name, param in new_params.items():
            if (name in saved_state_dict and param.size() == saved_state_dict[name].size()):
                new_params[name].copy_(saved_state_dict[name])
            elif name[7:] in saved_state_dict and param.size() == saved_state_dict[name[7:]].size():
                new_params[name].copy_(saved_state_dict[name[7:]])
            elif 'module.'+name in saved_state_dict and param.size() == saved_state_dict['module.'+name].size():
                new_params[name].copy_(saved_state_dict['module.'+name])
            else:
                print(name[7:])
        net.load_state_dict(new_params)

    net.eval()

    with torch.no_grad():
        error_sad_sum = 0
        error_mad_sum = 0
        error_mse_sum = 0
        error_grad_sum = 0
        sad_fg_sum = 0
        sad_bg_sum = 0
        sad_tran_sum = 0
        index = 0
        val_loop = tqdm(enumerate(val_loader), total=len(val_loader))
        val_loop.set_description('val|')
        for (i, label_data) in val_loop:
            label_img = label_data[0].cuda().float()
            label_alpha = label_data[1].cuda().float()  # .unsqueeze(1)
            trimap = label_data[2].cuda().float().unsqueeze(1)
            out = net(label_img)
            matte = out[-1]
            error_sad, error_mad, error_mse, error_grad, sad_fg, sad_bg, sad_tran = computeAllMatrix(matte,
                                                                                                     label_alpha,
                                                                                                     trimap)
            index += error_sad - error_sad + 1
            error_sad_sum += error_sad
            error_mad_sum += error_mad
            error_mse_sum += error_mse
            error_grad_sum += error_grad
            sad_fg_sum += sad_fg
            sad_bg_sum += sad_bg
            sad_tran_sum += sad_tran

        ave_val_loss = error_mad_sum / index
        ave_error_sad_sum = error_sad_sum / index
        ave_error_mad_sum = error_mad_sum / index
        ave_error_mse_sum = error_mse_sum / index
        ave_error_grad_sum = error_grad_sum / index
        ave_sad_fg_sum = sad_fg_sum / index
        ave_sad_bg_sum = sad_bg_sum / index
        ave_sad_tran_sum = sad_tran_sum / index

        metrix_str = '{:20}\t{:20}\n' \
                     '{:20}\t{:20}\t{:20}\n' \
                     '{:20}\t{:20}\t{:20}\n' \
            .format('Val',
                    'Grad: {:.5f}'.format(ave_error_grad_sum),
                    'Sad: {:.5f}'.format(ave_error_sad_sum),
                    'Mad: {:.5f}'.format(ave_error_mad_sum),
                    'Mse: {:.5f}'.format(ave_error_mse_sum),
                    'Sad_fg: {:.5f}'.format(ave_sad_fg_sum),
                    'Sad_bg: {:.5f}'.format(ave_sad_bg_sum),
                    'Sad_tran: {:.5f}'.format(ave_sad_tran_sum)
                    )
        print(metrix_str)
