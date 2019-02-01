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

    
class alex(_fasterRCNN):
  def __init__(self, classes, pretrained=False, class_agnostic=False):
    self.model_path = 'data/pretrained_model/alexnet-owt-4df8aa71.pth'
    self.dout_base_model = 256
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic

    _fasterRCNN.__init__(self, classes, class_agnostic)
    
  def _init_modules(self):
    alex = models.AlexNet()
    resnet = models.resnet18()
    if self.pretrained:
        print("Loading pretrained weights from %s" %(self.model_path))
        state_dict = torch.load(self.model_path)
        alex.load_state_dict({k:v for k,v in state_dict.items() if k in alex.state_dict()})
        state_dict = torch.load('data/pretrained_model/resnet18-5c106cde.pth')
        resnet.load_state_dict({k:v for k,v in state_dict.items() if k in resnet.state_dict()})
 

    # use last layer
    self.RCNN_base = nn.Sequential(*list(alex.features._modules.values()))
       
    #dont't use classifier
    alex.classifier = nn.Sequential(*list(alex.classifier._modules.values())[:-1])
   
    # fix conv layers
    # Fix the layers before Fire5:
    for layer in range(6):
      for p in self.RCNN_base[layer].parameters(): p.requires_grad = False
    
    # top
#    self.RCNN_top = 
    self.RCNN_top = resnet.layer4#[1:]
    
    # not using the last maxpool layer
    self.RCNN_cls_score = nn.Linear(512, self.n_classes)

    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(512, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(512, 4 * self.n_classes)      
      
    
    def set_bn_fix(m):
      classname = m.__class__.__name__
      if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad=False

    self.RCNN_base.apply(set_bn_fix)
    self.RCNN_top.apply(set_bn_fix)
    

  def _head_to_tail(self, pool5):
    
#    pool5_flat = pool5.view(pool5.size(0), -1)
#    print("pool",pool5.shape)
#    print("poolflat",pool5_flat.shape)
#    fc7 = self.RCNN_top(pool5_flat)
    fc7 = self.RCNN_top(pool5).mean(3).mean(2)

    return fc7

#  def train(self, mode=True):
#    # Override train so that the training mode is set as we want
#    nn.Module.train(self, mode)
#    if mode:
#      # Set fixed blocks to be in eval mode
##      self.RCNN_base.eval()
##      self.RCNN_base[5].train()
##      self.RCNN_base[6].train()
#
#      def set_bn_eval(m):
#        classname = m.__class__.__name__
#        if classname.find('BatchNorm') != -1:
#          m.eval()
#
#      self.RCNN_base.apply(set_bn_eval)
#      self.RCNN_top.apply(set_bn_eval)