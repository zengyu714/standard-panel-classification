"""Init RetinaNet with pretrained ResNet model.

Download pretrained Residual Model params from:
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
"""
import os
import math
import torch
import torch.nn as nn
import torch.nn.init as init

from retina.fpn import FPN18, FPN34, FPN50, FPN101
from retina.retinanet import RetinaNet

# Coarse setting
os.chdir('/home/zengyu/Lab/pytorch/standard-panel-classification/retina')

for base_version in [18, 34, 50, 101]:
    print('Loading pretrained ResNet{} model...'.format(base_version))
    d = torch.load('model/resnet{}.pth'.format(base_version))

    model_name = 'FPN{}'.format(base_version)
    print('Loading into {}...'.format(model_name))
    fpn = eval(model_name)()
    dd = fpn.state_dict()
    for k in d.keys():
        if not k.startswith('fc'):  # skip fc layers
            dd[k] = d[k]

    print('Saving RetinaNet - {}...'.format(model_name))
    net = RetinaNet(model_name)
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.normal(m.weight, mean=0, std=0.01)
            if m.bias is not None:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    pi = 0.01
    init.constant(net.cls_head[-1].bias, -math.log((1 - pi) / pi))

    net.fpn.load_state_dict(dd)
    torch.save(net.state_dict(), 'model/init_{}.pth'.format(model_name))
    print('Done!')
