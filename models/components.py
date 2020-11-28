import torch
import torch.nn as nn

from config.config import opt
import torch.nn.functional as F

class ResBlockDown(nn.Module):
    def __init__(self, in_channel, out_channel, conv_size=3, padding_size=1):
        super(ResBlockDown, self).__init__()
        
        self.relu = nn.ReLU(inplace = False)
        self.avg_pool2d = nn.AvgPool2d(2)
        
        #left
        self.conv_l1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 1,))
        
        #right
        self.conv_r1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, conv_size, padding = padding_size))
        self.conv_r2 = nn.utils.spectral_norm(nn.Conv2d(out_channel, out_channel, conv_size, padding = padding_size))

    def forward(self, x):
        res = x
        
        #left
        out_res = self.conv_l1(res)
        out_res = self.avg_pool2d(out_res)
        
        #right
        out = self.relu(x)
        out = self.conv_r1(out)
        out = self.relu(out)
        out = self.conv_r2(out)
        out = self.avg_pool2d(out)
        
        #merge
        out = out_res + out
        
        return out



class SelfAttention(nn.Module):
    def __init__(self, in_channel):
        super(SelfAttention, self).__init__()
        
        #conv f
        self.conv_f = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel//8, 1))
        #conv_g
        self.conv_g = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel//8, 1))
        #conv_h
        self.conv_h = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, 1))
        
        self.softmax = nn.Softmax(-2) #sum in column j = 1
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        B, C, H, W = x.shape
        f_projection = self.conv_f(x) #BxC'xHxW, C'=C//8
        g_projection = self.conv_g(x) #BxC'xHxW
        h_projection = self.conv_h(x) #BxCxHxW
        
        f_projection = torch.transpose(f_projection.view(B,-1,H*W), 1, 2) #BxNxC', N=H*W
        g_projection = g_projection.view(B,-1,H*W) #BxC'xN
        h_projection = h_projection.view(B,-1,H*W) #BxCxN
        
        attention_map = torch.bmm(f_projection, g_projection) #BxNxN
        attention_map = self.softmax(attention_map) #sum_i_N (A i,j) = 1
        
        #sum_i_N (A i,j) = 1 hence oj = (HxAj) is a weighted sum of input columns
        out = torch.bmm(h_projection, attention_map) #BxCxN
        out = out.view(B,C,H,W)
        
        out = self.gamma*out + x
        return out





def adaIN(feature, mean_style, std_style, eps = 1e-5):
    B,C,H,W = feature.shape
    feature = feature.view(B,C,-1)
            
    std_feat = (torch.std(feature, dim = 2) + eps).view(B,C,1)
    mean_feat = torch.mean(feature, dim = 2).view(B,C,1)
    
    adain = std_style * (feature - mean_feat)/std_feat + mean_style
    
    adain = adain.view(B,C,H,W)
    return adain




class ResBlockUp2(nn.Module):
    def __init__(self, in_channel, out_channel, out_size=None, scale = 2, conv_size=3, padding_size = 1):
        super(ResBlockUp2, self).__init__()
        
        self.in_channel = in_channel
        self.out_channel = out_channel
        
        self.upsample = nn.Upsample(size = out_size, scale_factor=scale)
        self.relu = nn.ReLU(inplace = False)
        
        #left
        self.conv_l1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 1))
        
        #right
        self.conv_r1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, conv_size, padding = padding_size))
        self.conv_r2 = nn.utils.spectral_norm(nn.Conv2d(out_channel, out_channel, conv_size, padding = padding_size))

        self.pose_projection = nn.utils.spectral_norm(nn.Linear(128, in_channel*2, bias=False))
        self.face_projection = nn.utils.spectral_norm(nn.Linear(512, out_channel*2, bias=False))
        
    
    def forward(self, x, pose_latent, face_latent):
        batch_size = x.size(0)
        pose_projection = self.pose_projection(pose_latent).view(batch_size, self.out_channel*2, 1)
        face_projection = self.face_projection(face_latent).view(batch_size, self.in_channel*2, 1)

        mean1 = pose_projection[:, 0:self.out_channel, :]
        std1 = pose_projection[:, self.out_channel:2*self.out_channel, :]
        mean2 = face_projection[:, 0:self.in_channel, :]
        std2 = face_projection[:, self.in_channel:2*self.in_channel, :]
        
        res = x
        
        #left
        out_res = self.upsample(res)
        out_res = self.conv_l1(out_res)
        
        #right
        out = adaIN(x, mean2, std2)
        out = self.relu(out)
        out = self.upsample(out)
        out = self.conv_r1(out)
        out = adaIN(out, mean1, std1)
        out = self.relu(out)
        out = self.conv_r2(out)
        
        out = out + out_res
        
        return out






class ResBlockUp(nn.Module):
    def __init__(self, in_channel, out_channel, out_size=None, scale = 2, conv_size=3, padding_size = 1):
        super(ResBlockUp, self).__init__()
        
        self.in_channel = in_channel
        self.out_channel = out_channel
        
        self.upsample = nn.Upsample(size = out_size, scale_factor=scale)
        self.relu = nn.ReLU(inplace = False)
        
        #left
        self.conv_l1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 1))
        
        #right
        self.conv_r1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, conv_size, padding = padding_size))
        self.conv_r2 = nn.utils.spectral_norm(nn.Conv2d(out_channel, out_channel, conv_size, padding = padding_size))

        # self.pose_projection = nn.utils.spectral_norm(nn.Linear(128, in_channel*2, bias=False))
        # self.face_projection = nn.utils.spectral_norm(nn.Linear(512, out_channel*2, bias=False))
        
        self.pose_projection = nn.utils.spectral_norm(nn.Linear(128, out_channel*2, bias=False))
        self.face_projection = nn.utils.spectral_norm(nn.Linear(512, in_channel*2, bias=False))
    
    def forward(self, x, pose_latent, face_latent):
        batch_size = x.size(0)
        pose_projection = self.pose_projection(pose_latent).view(batch_size, self.out_channel*2, 1)
        face_projection = self.face_projection(face_latent).view(batch_size, self.in_channel*2, 1)

        mean1 = pose_projection[:, 0:self.out_channel, :]
        std1 = pose_projection[:, self.out_channel:2*self.out_channel, :]
        mean2 = face_projection[:, 0:self.in_channel, :]
        std2 = face_projection[:, self.in_channel:2*self.in_channel, :]
        
        res = x
        
        #left
        out_res = self.upsample(res)
        out_res = self.conv_l1(out_res)
        
        #right
        out = adaIN(x, mean2, std2)
        out = self.relu(out)
        out = self.upsample(out)
        out = self.conv_r1(out)
        out = adaIN(out, mean1, std1)
        out = self.relu(out)
        out = self.conv_r2(out)
        
        out = out + out_res
        
        return out



class ResBlock(nn.Module):
    def __init__(self, in_channel):
        super(ResBlock, self).__init__()
        #general
        self.relu = nn.ReLU(inplace = False)
        
        #left
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, 3, padding = 1))
        self.in1 = nn.InstanceNorm2d(in_channel, affine=True)
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, 3, padding = 1))
        self.in2 = nn.InstanceNorm2d(in_channel, affine=True)
        
    def forward(self, x):
        res = x
        
        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.in2(out)
        
        out = out + res
        
        return out





class ToRGB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channels)
        self.colorize = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, (3, 3), padding=1))
        

    def forward(self, x):
        # x = self.conv(x)
        
        # if skip is not None:
        #     skip = self.up(skip)
        #     x = x + skip

        x = self.norm(x)
        x = F.relu(x)
        x = self.colorize(x)
        return torch.tanh(x)









