import argparse
import os
from util import util
import torch


parser = argparse.ArgumentParser()
parser.add_argument('--chn', type=int, default=64, help='number of filter in the first conv layer')
parser.add_argument('--k', type=int, default=3, help='few shot k')
parser.add_argument('--batch_size', type=int, default=6, help='batch size')
parser.add_argument('--lr_g', type=float, default=2e-4, help='learning rate for generator')
parser.add_argument('--lr_d', type=float, default=2e-4, help='learning rate for discriminator')
parser.add_argument('--training_path', type=str, default='../voxceleb2/train/**/**', help='path for training dataset')
parser.add_argument('--test_path', type=str, default='../voxceleb2/test/**/**', help='path for testing dataset')
parser.add_argument('--img_folder', type=str, default='./images')
parser.add_argument('--checkpoints_folder', type=str, default='./checkpoints')
parser.add_argument('--loss_file', type=str, default='./loss.txt')
parser.add_argument('--error_file', type=str, default='./error.txt')
parser.add_argument('--log_file', type=str, default='./log.txt')

parser.add_argument('--num_workers', type=int, default=24)

# parser.add_argument('--p_loss', type=str, default='dr', choices=['dr', 'hinge'], help='pose embedder loss')

opt = parser.parse_args()

