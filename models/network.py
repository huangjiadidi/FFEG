import torch
from torch import nn
from config.config import opt
import torch.nn.functional as F

from models.components import *



class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            ResBlockDown(6, 64),
            ResBlockDown(64, 128),
            ResBlockDown(128, 256),
            ResBlockDown(256, 512),
            ResBlockDown(512, 1024),
            ResBlockDown(1024, 224),
        )

    def forward(self, x, k):
        latent = self.encoder(x).view(-1, k, 224*4*4)
        latent = latent.sum(1) / k
        return latent


class PoseEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            ResBlockDown(3, 64),
            ResBlockDown(64, 128),
            ResBlockDown(128, 256),
            ResBlockDown(256, 512),
            ResBlockDown(512, 512),
            ResBlockDown(512, 128),
        )

        self.linear = nn.utils.spectral_norm(nn.Linear(4*4*128, 128))

        # self.sum_pooling = nn.AdaptiveMaxPool2d((1,1))

    def forward(self, x):
        latent = self.encoder(x).view(-1, 4*4*128)
        latent = self.linear(latent)
        return latent
        # latent = self.encoder(x)
        # return self.sum_pooling(latent).view(-1, 128)




class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()

        self.input_block = nn.utils.spectral_norm(nn.Linear(128 + 512, 4*4*512))
        
        self.up1 = ResBlockUp(512, 512)
        self.up2 = ResBlockUp(512, 512)
        self.up3 = ResBlockUp(512, 512)
        self.up4 = ResBlockUp(512, 256)
        self.up5 = ResBlockUp(256, 128)
        self.up6 = ResBlockUp(128, 64)

        self.to_rgb16 = ToRGB(512, 3)
        self.to_rgb32 = ToRGB(512, 3)
        self.to_rgb64 = ToRGB(256, 3)
        self.to_rgb128 = ToRGB(128, 3)
        self.to_rgb256 = ToRGB(64, 3)

    def init_finetune(self, face_emb):
        self.face_emb = nn.Parameter(face_emb)

    def forward(self, imgs, pose_latent, k, finetune=False):
        # face_latent = self.encoder(imgs, k)
        # face_split = torch.chunk(face_latent, 7, 1)
        if not finetune:
            face_latent = self.encoder(imgs, k)
            face_split = torch.chunk(face_latent, 7, 1)
        else:
            face_emb = self.face_emb.expand(pose_latent.size(0), 3584)
            face_split = torch.chunk(face_emb, 7, 1)

        latent = torch.cat([pose_latent, face_split[0]], 1)
        input = self.input_block(latent).view(-1, 512, 4, 4)

        out1 = self.up1(input, pose_latent, face_split[1])
        out2 = self.up2(out1, pose_latent, face_split[2])
        out3 = self.up3(out2, pose_latent, face_split[3])
        out4 = self.up4(out3, pose_latent, face_split[4])
        out5 = self.up5(out4, pose_latent, face_split[5])
        out6 = self.up6(out5, pose_latent, face_split[6])

        rgb16 = self.to_rgb16(out2)
        rgb32 = self.to_rgb32(out3)
        rgb64 = self.to_rgb64(out4)
        rgb128 = self.to_rgb128(out5)
        rgb256 = self.to_rgb256(out6)
        return [rgb16, rgb32, rgb64, rgb128, rgb256]


class PoseDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.dis = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(256, 1024)),
            nn.ReLU(inplace=True),
            nn.utils.spectral_norm(nn.Linear(1024, 1024)),
            nn.ReLU(inplace=True),
            nn.utils.spectral_norm(nn.Linear(1024, 1024)),
            nn.ReLU(inplace=True),
            nn.utils.spectral_norm(nn.Linear(1024, 1)),
        )
    
    def forward(self, pose1, pose2):
        pose = torch.cat([pose1, pose2], 1)
        return self.dis(pose)



class MultiScaledProjectionDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.d256 = nn.ModuleList([
            ResBlockDown(6, 64),
            ResBlockDown(64, 128),
            ResBlockDown(128, 256),
            ResBlockDown(256, 512),
            ResBlockDown(512, 512),
            ResBlockDown(512, 512),
            # ResBlock(512)
        ])

        self.d128 = nn.ModuleList([
            ResBlockDown(6, 64),
            ResBlockDown(64, 128),
            ResBlockDown(128, 256),
            ResBlockDown(256, 512),
            ResBlockDown(512, 512),
            ResBlockDown(512, 512),
            # ResBlock(512)
        ])
        
        self.sum_pooling = nn.AdaptiveAvgPool2d((1,1))

        self.W_i = nn.Parameter(torch.rand(512, 145740))

        self.fine_tune = nn.Parameter(torch.rand(512, 1))
        
        self.w_0 = nn.Parameter(torch.randn(512,1))
        self.b0  = nn.Parameter(torch.randn(1))

        self.w_1 = nn.Parameter(torch.randn(512,1))
        self.b1  = nn.Parameter(torch.randn(1))

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
    
    def forward(self, x, i, pose, fine_tune=False):
        batch_size = x.size(0)

        x = torch.cat([x, pose], 1)

        layer256 = []
        out256 = x
        for layer in self.d256:
            out256 = layer(out256)
            layer256.append(out256)
        out256 = F.relu(out256)
        out256 = self.sum_pooling(out256)
        out256 = out256.view(-1,512,1)
        if not fine_tune:
            out256 = torch.bmm(out256.transpose(1,2), (self.W_i[:,i].unsqueeze(-1)).transpose(0,1) + self.w_0) + self.b0
        else:
            out256 = torch.bmm(out256.transpose(1,2), (self.fine_tune[:,[0]*batch_size].unsqueeze(-1)).transpose(0,1) + self.w_0) + self.b0

        layer128 = []
        out128 = self.downsample(x)
        for layer in self.d128:
            out128 = layer(out128)
            layer128.append(out128)
        out128 = F.relu(out128)
        out128 = self.sum_pooling(out128)
        out128 = out128.view(-1,512,1)
        if not fine_tune:
            out128 = torch.bmm(out128.transpose(1,2), (self.W_i[:,i].unsqueeze(-1)).transpose(0,1) + self.w_1) + self.b1
        else:
            out128 = torch.bmm(out128.transpose(1,2), (self.fine_tune[:,[0]*batch_size].unsqueeze(-1)).transpose(0,1) + self.w_1) + self.b1
        return out256, out128, layer256 + layer128





class ProjectionDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.d256 = nn.ModuleList([
            ResBlockDown(6, 64),
            ResBlockDown(64, 128),
            ResBlockDown(128, 256),
            ResBlockDown(256, 512),
            ResBlockDown(512, 512),
            ResBlockDown(512, 512),
            # ResBlock(512)
        ])

        
        self.sum_pooling = nn.AdaptiveAvgPool2d((1,1))

        self.W_i = nn.Parameter(torch.rand(512, 145740))

        self.fine_tune = nn.Parameter(torch.rand(512, 1))
        
        self.w_0 = nn.Parameter(torch.randn(512,1))
        self.b0  = nn.Parameter(torch.randn(1))


    def forward(self, x, i, pose, fine_tune=False):
        batch_size = x.size(0)

        x = torch.cat([x, pose], 1)

        layer256 = []
        out256 = x
        for layer in self.d256:
            out256 = layer(out256)
            layer256.append(out256)
        out256 = F.relu(out256)
        out256 = self.sum_pooling(out256)
        out256 = out256.view(-1,512,1)
        if not fine_tune:
            out256 = torch.bmm(out256.transpose(1,2), (self.W_i[:,i].unsqueeze(-1)).transpose(0,1) + self.w_0) + self.b0
        else:
            out256 = torch.bmm(out256.transpose(1,2), (self.W_i[:,[0]*batch_size].unsqueeze(-1)).transpose(0,1) + self.w_0) + self.b0

        return out256, layer256




class MultiScaledDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.d256 = nn.ModuleList([
            ResBlockDown(6, 64),
            ResBlockDown(64, 128),
            ResBlockDown(128, 256),
            ResBlockDown(256, 512),
            ResBlockDown(512, 512),
            ResBlockDown(512, 1),
            # ResBlock(512)
        ])

        self.d128 = nn.ModuleList([
            ResBlockDown(6, 64),
            ResBlockDown(64, 128),
            ResBlockDown(128, 256),
            ResBlockDown(256, 512),
            ResBlockDown(512, 512),
            ResBlockDown(512, 1),
            # ResBlock(512)
        ])
        
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
    
    def forward(self, x, i, pose, fine_tune=False):
        batch_size = x.size(0)

        x = torch.cat([x, pose], 1)

        layer256 = []
        out256 = x
        for index, layer in enumerate(self.d256):
            out256 = layer(out256)
            if index != 5:
                layer256.append(out256)
        
        
        layer128 = []
        out128 = self.downsample(x)
        for index, layer in enumerate(self.d128):
            out128 = layer(out128)
            if index != 5:
                layer128.append(out128)
        
        
        return out256, out128, layer256 + layer128
