# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
from model.faster_rcnn.faster_rcnn import _fasterRCNN
import pdb
import torch.utils.model_zoo as model_zoo


__all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1']


model_urls = {
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
}
    
class squeeze(_fasterRCNN):
  def __init__(self, classes, pretrained=False, class_agnostic=False):
    self.model_path = 'data/pretrained_model/squeezenet1_1-f364aa15.pth'
#    self.model_path = 'data/pretrained_model/squeezenet1_0-a815701f.pth'
    self.dout_base_model = 512
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic

    _fasterRCNN.__init__(self, classes, class_agnostic)
    
  def _init_modules(self):
    squeeze = models.squeezenet1_1()
    resnet = models.resnet18()
    if self.pretrained:
        print("Loading pretrained weights from %s" %(self.model_path))
        state_dict = torch.load(self.model_path)
        squeeze.load_state_dict({k:v for k,v in state_dict.items() if k in squeeze.state_dict()})
        state_dict = torch.load('data/pretrained_model/resnet18-5c106cde.pth')
        resnet.load_state_dict({k:v for k,v in state_dict.items() if k in resnet.state_dict()})
 

    # use last layer
    self.RCNN_base = nn.Sequential(*list(squeeze.features._modules.values()))
       
    #dont't use classifier
    squeeze.classifier = nn.Sequential(*list(squeeze.classifier._modules.values())[:2])
   
    # fix conv layers
    # Fix the layers before Fire5:
    for layer in range(11):
      for p in self.RCNN_base[layer].parameters(): p.requires_grad = False
    
    # top
    num_out = 1000
    self.RCNN_top = nn.Sequential(
#            Fire(512, 64, 256, 256),
#            Fire(512, 64, 256, 256),
            nn.Conv2d(512, 512, kernel_size=(3,3)),
            nn.ReLU(inplace=True),
            *list(squeeze.classifier._modules.values())[:2])

#    self.RCNN_top = resnet.layer4[1:]
        # not using the last maxpool layer
    
    self.RCNN_cls_score = nn.Linear(num_out, self.n_classes)

    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(num_out, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(num_out, 4 * self.n_classes)      
    
#    # not using the last maxpool layer
#    self.RCNN_cls_score = nn.Linear(512, self.n_classes)
#
#    if self.class_agnostic:
#      self.RCNN_bbox_pred = nn.Linear(512, 4)
#    else:
#      self.RCNN_bbox_pred = nn.Linear(512, 4 * self.n_classes)      
      
    
    def set_bn_fix(m):
      classname = m.__class__.__name__
      if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad=False

    self.RCNN_base.apply(set_bn_fix)
    self.RCNN_top.apply(set_bn_fix)
    

  def _head_to_tail(self, pool5):
    
#    pool5_flat = pool5.view(pool5.size(0), -1)
#    print("pool",pool5.shape)
#    fc7 = self.RCNN_top(pool5_flat)
    fc7 = self.RCNN_top(pool5).mean(3).mean(2)

    return fc7

  def train(self, mode=True):
    # Override train so that the training mode is set as we want
    nn.Module.train(self, mode)
    if mode:
      # Set fixed blocks to be in eval mode
#      self.RCNN_base.eval()
#      self.RCNN_base[5].train()
#      self.RCNN_base[6].train()

      def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
          m.eval()

      self.RCNN_base.apply(set_bn_eval)
      self.RCNN_top.apply(set_bn_eval)
  
class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)