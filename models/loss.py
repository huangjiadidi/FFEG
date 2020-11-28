import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import opt
from torchvision.transforms import Normalize

from models.pretrained_network import *






class GLoss(nn.Module):
    def __init__(self):
        super().__init__()
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        self.vgg19 = Vgg19().to(device)
        self.arc_face = ArcFace().to(device)
    
    def vgg19_loss(self, fake_img, target_img):
        fake_img = (fake_img + 1) / 2
        target_img = (target_img + 1) / 2
        fake_content = self.vgg19(fake_img)
        target_content = self.vgg19(target_img)
        
        loss = 0
        weight = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        for i in range(len(fake_content)):
            loss += F.l1_loss(fake_content[i], target_content[i]) * weight[i]
        
        return loss * 10
    
    def arcface_loss(self, fake, real):
        fake = F.avg_pool2d(fake[:, :, :224, 16:240], 2)
        real = F.avg_pool2d(real[:, :, :224, 16:240], 2)

        fake_layers = self.arc_face(fake)
        real_layers = self.arc_face(real)
        weight = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        loss = 0
        for i in range(len(fake_layers)):
            loss += F.l1_loss(fake_layers[i], real_layers[i].detach()) * weight[i]
        return loss * 50
    
    def adv_loss(self, f_preds, r_preds):

        # difference between real and fake:
        r_f_diff = r_preds - torch.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - torch.mean(r_preds)

        # return the loss
        return (torch.mean(torch.nn.ReLU()(1 + r_f_diff))
                + torch.mean(torch.nn.ReLU()(1 - f_r_diff)))
    
    
    def rec_loss(self, fake_img, target_img):
        return F.l1_loss(fake_img, target_img) * 10
        

    def fm_loss(self, fake_inter, real_inter):
        loss = 0
        for i in range(len(fake_inter)):
            loss += F.l1_loss(fake_inter[i], real_inter[i].detach())
        return loss
    
    def kl_divergence(self, mu, logvar):
        return (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) * 0.05

    def forward(self, fake_sample, target_sample, fake_inter, real_inter, fake_score1, fake_score2, real_score1, real_score2, mu, logvar):
        # adv_loss1 = self.adv_loss(fake_score1, real_score1)
        # adv_loss2 = self.adv_loss(fake_score2, real_score2)
        rec_loss = self.rec_loss(fake_sample, target_sample)
        # fm_loss = self.fm_loss(fake_inter, real_inter)
        kl_divergence = self.kl_divergence(mu, logvar)
        arcface_loss = self.arcface_loss(fake_sample, target_sample)
        vgg_loss = self.vgg19_loss(fake_sample, target_sample)

        # loss = adv_loss1 + adv_loss2 + rec_loss + fm_loss + kl_divergence + arcface_loss + vgg_loss
        # return loss, [adv_loss1.item(), adv_loss2.item(), rec_loss.item(), fm_loss.item(), kl_divergence.item(), arcface_loss.item(), vgg_loss.item()]

        loss = rec_loss + kl_divergence + arcface_loss + vgg_loss
        return loss, [0, 0, rec_loss.item(), 0, kl_divergence.item(), arcface_loss.item(), vgg_loss.item()]



