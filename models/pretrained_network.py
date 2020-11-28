from models.arcface import Backbone
import torch
import torch.nn as nn
from torchvision.models.vgg import vgg19

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class Vgg19(nn.Module):
    def __init__(self):
        super().__init__()
        features = vgg19(pretrained=True).features[:30]
        self.features = nn.ModuleList(list(features)).eval().to(device)


        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)

        for param in self.features.parameters():
            param.requires_grad = False

        self.mean.requires_grad = False
        self.std.requires_grad = False
    

    def forward(self, x):
        x = (x + 1) / 2
        x = (x - self.mean) / self.std
        feature_map_idx = [1, 6, 11, 20, 29]
        
        results = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in feature_map_idx:
                results.append(x)
        return results


class ArcFace(nn.Module):
    def __init__(self):
        super().__init__()
        model = Backbone(50, 0.6, 'ir_se')
        if not torch.cuda.is_available():
            model.load_state_dict(torch.load('./model_ir_se50.pth', map_location='cpu'))
        else:
            model.load_state_dict(torch.load('./model_ir_se50.pth'))
    
        self.input_layer = model.input_layer.eval()
        self.body =  nn.ModuleList(list(model.body)).eval()

        for param in self.input_layer.parameters():
            param.requires_grad = False
        
        for param in self.body.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.input_layer(x)
        res = [out]
        
        for i, layer in enumerate(self.body):
            out = layer(out)
            if i in [2, 6, 20, 23]:
                res.append(out)
        
        return res



