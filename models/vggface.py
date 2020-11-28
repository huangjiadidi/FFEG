import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models import vgg
from torchvision.models.vgg import vgg19


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class VggFace(nn.Module):
    def __init__(self):
        super(VggFace, self).__init__()
        features = vgg_face(pretrained=True).features[:26]
        self.features = nn.ModuleList(list(features)).eval().to(device)

        for param in self.features.parameters():
            param.requires_grad = False
        
        
        self.mean = torch.tensor([131.45376586914062, 103.98748016357422, 91.46234893798828]).view(-1, 1, 1).to(device)

        for param in self.features.parameters():
            param.requires_grad = False

        self.mean.requires_grad = False

    def forward(self, x):
        x = (x + 1) * 127.5
        x = (x - self.mean)
        feature_map_idx = [1, 6, 11, 18, 25]
        results = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in feature_map_idx:
                results.append(x)
        return results



def vgg_face(pretrained=False, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    if torch.cuda.is_available():
        model = vgg.VGG(vgg.make_layers(vgg.cfg['D'], batch_norm=False), num_classes=2622, **kwargs)
    else:
        model = vgg.VGG(vgg.make_layers(vgg.cfgs['D'], batch_norm=False), num_classes=2622, **kwargs)
    if pretrained:
        model.load_state_dict(vgg_face_state_dict())
    return model


def vgg_face_state_dict():
    default = torch.load('./models/vgg_face_dag.pth')
    state_dict = OrderedDict({
        'features.0.weight': default['conv1_1.weight'],
        'features.0.bias': default['conv1_1.bias'],
        'features.2.weight': default['conv1_2.weight'],
        'features.2.bias': default['conv1_2.bias'],
        'features.5.weight': default['conv2_1.weight'],
        'features.5.bias': default['conv2_1.bias'],
        'features.7.weight': default['conv2_2.weight'],
        'features.7.bias': default['conv2_2.bias'],
        'features.10.weight': default['conv3_1.weight'],
        'features.10.bias': default['conv3_1.bias'],
        'features.12.weight': default['conv3_2.weight'],
        'features.12.bias': default['conv3_2.bias'],
        'features.14.weight': default['conv3_3.weight'],
        'features.14.bias': default['conv3_3.bias'],
        'features.17.weight': default['conv4_1.weight'],
        'features.17.bias': default['conv4_1.bias'],
        'features.19.weight': default['conv4_2.weight'],
        'features.19.bias': default['conv4_2.bias'],
        'features.21.weight': default['conv4_3.weight'],
        'features.21.bias': default['conv4_3.bias'],
        'features.24.weight': default['conv5_1.weight'],
        'features.24.bias': default['conv5_1.bias'],
        'features.26.weight': default['conv5_2.weight'],
        'features.26.bias': default['conv5_2.bias'],
        'features.28.weight': default['conv5_3.weight'],
        'features.28.bias': default['conv5_3.bias'],
        'classifier.0.weight': default['fc6.weight'],
        'classifier.0.bias': default['fc6.bias'],
        'classifier.3.weight': default['fc7.weight'],
        'classifier.3.bias': default['fc7.bias'],
        'classifier.6.weight': default['fc8.weight'],
        'classifier.6.bias': default['fc8.bias']
    })
    return state_dict

