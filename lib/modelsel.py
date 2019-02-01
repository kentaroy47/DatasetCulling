# -*- coding: utf-8 -*-
import _init_paths
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from model.faster_rcnn.squeezenet import squeeze
from model.faster_rcnn.squeezenet_mod import squeeze as squeezemod
from model.faster_rcnn.alex import alex

def modelsel(net, classes, class_agnostic):
      # initilize the network here.
    classes = classes
    if net == 'vgg16':
        fasterRCNN = vgg16(classes, pretrained=True, class_agnostic=class_agnostic)
    elif net == 'res101':
        fasterRCNN = resnet(classes, 101, pretrained=True, class_agnostic=class_agnostic)
    elif net == 'res50':
        fasterRCNN = resnet(classes, 50, pretrained=True, class_agnostic=class_agnostic)
    elif net == 'res34':
        fasterRCNN = resnet(classes, 34, pretrained=True, class_agnostic=class_agnostic)
    elif net == 'res18':
        fasterRCNN = resnet(classes, 18, pretrained=True, class_agnostic=class_agnostic)
    elif net == 'res10':
        fasterRCNN = resnet(classes, 10, pretrained=True, class_agnostic=class_agnostic)
    elif net == 'res152':
        fasterRCNN = resnet(classes, 152, pretrained=True, class_agnostic=class_agnostic)
    elif net == 'squeeze':
        fasterRCNN = squeeze(classes, pretrained=True, class_agnostic=class_agnostic)
    elif net == 'squeezemod':
        fasterRCNN = squeezemod(classes, pretrained=True, class_agnostic=class_agnostic)
    elif net == 'alex':
        fasterRCNN = alex(classes, pretrained=True, class_agnostic=class_agnostic)
    else:
        print("network is not defined")

    return fasterRCNN