"""Init RestinaNet50 with pretrained ResNet50 model.

Download pretrained ResNet50 params from:
  https://download.pytorch.org/models/resnet50-19c8e357.pth
"""
import math
import torch
import torch.nn as nn
import torch.nn.init as init

from retina.fpn import FPN50, FPN101
from retina.retinanet import RetinaNet

base_version = 101
print('Loading pretrained ResNet{} model..'.format(base_version))
d = torch.load('../model/resnet{}.pth'.format(base_version))

model_name = 'FPN{}'.format(base_version)
print('Loading into {}..'.format(model_name))
fpn = eval(model_name)()
dd = fpn.state_dict()
for k in d.keys():
    if not k.startswith('fc'):  # skip fc layers
        dd[k] = d[k]

print('Saving RetinaNet..')
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
init.constant(net.cls_head[-1].bias, -math.log((1-pi)/pi))

net.fpn.load_state_dict(dd)
torch.save(net.state_dict(), '../model/init_{}.pth'.format(model_name))
print('Done!')
