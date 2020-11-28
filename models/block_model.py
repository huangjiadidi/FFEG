import os
import copy
import random
import numpy as np

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from models.loss import *
from models.network import *

from config.config import opt
from models.pretrained_network import Vgg19


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
VGG = Vgg19().to(device)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.pose_encoder = PoseEncoder()
        # self.generator = BlockGenerator()
        self.generator = ConstGenerator()
        self.pose_dis = PoseDiscriminator()
        self.dis = MultiScaledProjectionDiscriminator()

        self.G_params = list(self.pose_encoder.parameters()) + list(self.generator.parameters())
        self.D_params = list(self.dis.parameters())
        self.PD_params = list(self.pose_dis.parameters())

        self.optim_G = optim.Adam(self.G_params, lr=opt.lr_g)
        self.optim_D = optim.Adam(self.D_params, lr=opt.lr_d)
        self.optim_PD = optim.Adam(self.PD_params, lr=opt.lr_d)


    def optimize_D(self, real, fake, index, target_pose, fine_tune):
        self.optim_D.zero_grad()
        real_score_256, real_score_128, _ = self.dis(real, index, target_pose, fine_tune)
        fake_score_256, fake_score_128, _ = self.dis(fake.detach(), index, target_pose, fine_tune)
        d1_adv_256 = self.get_hinge_d(fake_score_256, real_score_256)
        d1_adv_128 = self.get_hinge_d(fake_score_128, real_score_128)
        loss =  d1_adv_256 + d1_adv_128
        loss.backward()
        self.optim_D.step()

        return d1_adv_256.item(), d1_adv_128.item()
    


    
    def inference(self, source, source_pose, target_pose, k):
        with torch.no_grad():
            imgs = torch.cat([source, source_pose], 1)
            pose_latent = self.pose_encoder(target_pose)
            fake_sample = self.generator(imgs, pose_latent, k)[-1]
            return fake_sample


    def stage2(self, source, target, source_pose, target_pose, other_pose, same_person_pose, index, k, get_image=False, fine_tune=False):
        imgs = torch.cat([source, source_pose], 1)
        
        self.optim_G.zero_grad()
        pose_latent = self.pose_encoder(target_pose)

        fake_sample = self.generator(imgs, pose_latent, k)[-1]

        real_score_256, real_score_128, real_layers = self.dis(target, index, target_pose, fine_tune)
        fake_score_256, fake_score_128, fake_layers = self.dis(fake_sample, index, target_pose, fine_tune)
        
        g_adv_256 = self.get_hinge_g(fake_score_256, real_score_256)
        g_adv_128 = self.get_hinge_g(fake_score_128, real_score_128)

        fm_loss = sum([F.l1_loss(fake_layer, real_layer.detach()) for fake_layer, real_layer in zip(fake_layers, real_layers)])
        perceptual_loss = sum([F.l1_loss(fake_layer, real_layer) for fake_layer, real_layer in zip(VGG(fake_sample), VGG(target))])

        loss = g_adv_256 + g_adv_128 + fm_loss * 10 + perceptual_loss * 20
        loss.backward()
        self.optim_G.step()
        
        
        d1_adv_256, d1_adv_128 = self.optimize_D(target, fake_sample, index, target_pose, fine_tune)
        with torch.no_grad():
            fake_sample2 = self.generator(imgs, pose_latent, k)[-1]
        d2_adv_256, d2_adv_128 = self.optimize_D(target, fake_sample2, index, target_pose, fine_tune)
        

        res = [
            fm_loss.item(),
            perceptual_loss.item(),
            g_adv_256.item(),
            g_adv_128.item(),
            d1_adv_256,
            d1_adv_128,
            d2_adv_256,
            d2_adv_128,
        ]


        if get_image:
            res.append(fake_sample)
            with torch.no_grad():
                pose_latent = self.pose_encoder(other_pose)
                other_fake_sample = self.generator(imgs, pose_latent, k)[-1]
                res.append(other_fake_sample)
        return res
    
    def get_hinge_d(self, f_preds, r_preds):
        return torch.mean(F.relu(1 - r_preds)) + torch.mean(F.relu(1 + f_preds))
    
    def get_hinge_g(self, f_preds, r_preds):
        return -torch.mean(f_preds)


    def get_generator_adv(self, f_preds, r_preds):
        # return -torch.mean(f_preds)

        r_f_diff = r_preds - torch.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - torch.mean(r_preds)

        # return the loss
        return (torch.mean(nn.ReLU()(1 + r_f_diff))
                + torch.mean(nn.ReLU()(1 - f_r_diff)))


    def get_discriminator_adv(self, f_preds, r_preds):
        # return torch.mean(F.relu(1 - r_preds)) + torch.mean(F.relu(1 + f_preds))
        
        # difference between real and fake:
        r_f_diff = r_preds - torch.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - torch.mean(r_preds)

        # return the loss
        return (torch.mean(nn.ReLU()(1 - r_f_diff))
                + torch.mean(nn.ReLU()(1 + f_r_diff)))
            



    def __get_down_sampling(self, img256):
        img128 = F.avg_pool2d(img256, 2)
        img64 = F.avg_pool2d(img128, 2)
        img32 = F.avg_pool2d(img64, 2)
        img16 = F.avg_pool2d(img32, 2)
        img8 = F.avg_pool2d(img16, 2)
        return [img8, img16, img32, img64, img128, img256]
        

    def stage1(self, source, target, source_pose, target_pose, other_pose, same_person_pose, target_mask, index, k, get_image=False):


        floatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        target_mask += ((target_mask == 0).type(floatTensor) * 0.1)
        downsampled_mask = self.__get_down_sampling(target_mask)

        self.optim_G.zero_grad()
        imgs = torch.cat([source, source_pose], 1)
        target_sample = self.__get_down_sampling(target)

        pose_latent = self.pose_encoder(target_pose)
        fake_sample = self.generator(imgs, pose_latent, k)
        
        l1_loss_list = [F.l1_loss(fake * mask, real * mask) for fake, real, mask in zip(fake_sample, target_sample, downsampled_mask)]
        l1_loss_sum = sum([l*(i+1) * 10 for i, l in enumerate(l1_loss_list)])
        
        
        same_latent = self.pose_encoder(same_person_pose)
        other_latent = self.pose_encoder(other_pose)

        diff_video = self.pose_dis(pose_latent, other_latent)
        same_video = self.pose_dis(pose_latent, same_latent)
        pose_gadv = self.get_hinge_g(diff_video, same_video)

        loss = pose_gadv + l1_loss_sum
        loss.backward()
        self.optim_G.step()


        self.optim_PD.zero_grad()
        pose_latent = self.pose_encoder(source[::opt.k, ...])
        diff_video = self.pose_dis(pose_latent.detach(), other_latent.detach())
        same_video = self.pose_dis(pose_latent.detach(), same_latent.detach())
        pose_dadv = self.get_hinge_d(diff_video, same_video)
        pose_dadv.backward()
        self.optim_PD.step()


        res = [pose_gadv.item(), pose_dadv.item(), [i.item() for i in l1_loss_list]]

        if get_image:
            res.append(fake_sample[-1])
            with torch.no_grad():
                others = self.pose_encoder(other_pose)
                res.append(self.generator(imgs, others, k)[-1])

        return res
    




    def forward(self, source, source_pose, target_pose):
        pass
