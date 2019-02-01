# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from model.faster_rcnn.squeezenet import squeeze
from model.faster_rcnn.squeezenet_mod import squeeze as squeezemod
from model.faster_rcnn.alex import alex

import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--net', dest='net',
                    help='vgg16, res101',
                    default='res101', type=str)
  parser.add_argument('--scale', dest='scale',
                      help='optional config file',
                      default=1, type=float)
  parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)
  parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=30, type=int)
  parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=100, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                      help='number of iterations to display',
                      default=10000, type=int)

  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="models",
                      type=str)
  parser.add_argument('--nw', dest='num_workers',
                      help='number of worker to load data',
                      default=1, type=int)
  parser.add_argument('--save', dest='save',
                      help='number of worker to load data',
                      default=5, type=int)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')                      
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')

# config optimization
  parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)
  parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=0.0001, type=float)
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=8, type=int)
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)

# set training session
  parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int)

# resume trained model
  parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=False, type=bool)
  parser.add_argument('--coco', dest='coco',
                      help='resume from coco pretrained model',
                      default=False, type=bool)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=0, type=int)
# log and diaplay
  parser.add_argument('--use_tfb', dest='use_tfboard',
                      help='whether use tensorboard',
                      action='store_true')

  args = parser.parse_args()
  return args


class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data

if __name__ == '__main__':

  args = parse_args()
  cfg.TRAIN.SCALES = (int(args.scale*600),)
  cfg.TEST.SCALES = (int(args.scale*600),)
  print('Called with args:')
  print(args)

  if "pascal_voc" in args.dataset:      
      args.imdb_name = "voc_2007_trainval_"+args.dataset.split("pascal_voc_")[1]
      print(args.imdb_name)
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "pascal_voc_shanghai":
      args.imdb_name = "voc_2007_trainval_shanghai"
      args.imdbval_name = "voc_2007_val_shanghai"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    
  elif args.dataset == "pascal_voc_res101":
      args.imdb_name = "voc_2007_trainval_res101"
      args.imdbval_name = "voc_2007_test_res101"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
  elif args.dataset == "pascal_voc_taipei":
      args.imdb_name = "voc_2007_trainval_taipei"
      args.imdbval_name = "voc_2007_test_res101"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
  elif args.dataset == "pascal_voc_coral":
      args.imdb_name = "voc_2007_trainval_coral"
      args.imdbval_name = "voc_2007_test_coral"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "pascal_voc_coralcoco":
      args.imdb_name = "voc_2007_trainval_coralcoco"
      args.imdbval_name = "voc_2007_test_coral"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "pascal_voc_coralcocodif":
      args.imdb_name = "voc_2007_trainval_coralcocodif"
      args.imdbval_name = "voc_2007_test_coral"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "pascal_voc_jackson":
      args.imdb_name = "voc_2007_trainval_jackson"
      args.imdbval_name = "voc_2007_val_jackson"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "pascal_voc_jackson100":
      args.imdb_name = "voc_2007_trainval_jackson100"
      args.imdbval_name = "voc_2007_test_coral"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "pascal_voc_jackson10":
      args.imdb_name = "voc_2007_trainval_jackson10"
      args.imdbval_name = "voc_2007_test_coral"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "pascal_voc_coralcoco10":
      args.imdb_name = "voc_2007_trainval_coralcoco10"
      args.imdbval_name = "voc_2007_test_coral"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "pascal_voc_coralcoco20":
      args.imdb_name = "voc_2007_trainval_coralcoco20"
      args.imdbval_name = "voc_2007_test_coral"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "pascal_voc_jacksonallobject":
      args.imdb_name = "voc_2007_trainval_jacksonallobject"
      args.imdbval_name = "voc_2007_val_jackson"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "pascal_voc_coralcoco30":
      args.imdb_name = "voc_2007_trainval_coralcoco30"
      args.imdbval_name = "voc_2007_test_coral"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "pascal_voc_coralcoco50":
      args.imdb_name = "voc_2007_trainval_coralcoco50"
      args.imdbval_name = "voc_2007_test_coral"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "pascal_voc_coralcoco100":
      args.imdb_name = "voc_2007_trainval_coralcoco100"
      args.imdbval_name = "voc_2007_test_coral"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "pascal_voc_coralcoco250":
      args.imdb_name = "voc_2007_trainval_coralcoco250"
      args.imdbval_name = "voc_2007_test_coral"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "pascal_voc_coralcoco500":
      args.imdb_name = "voc_2007_trainval_coralcoco500"
      args.imdbval_name = "voc_2007_test_coral"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "pascal_voc_coralcoco1000":
      args.imdb_name = "voc_2007_trainval_coralcoco1000"
      args.imdbval_name = "voc_2007_test_coral"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "pascal_voc_coralcoco4000":
      args.imdb_name = "voc_2007_trainval_coralcoco4000"
      args.imdbval_name = "voc_2007_test_coral"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "pascal_voc_coralcoco8000":
      args.imdb_name = "voc_2007_trainval_coralcoco8000"
      args.imdbval_name = "voc_2007_test_coral"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "pascal_voc_coralpascal":
      args.imdb_name = "voc_2007_trainval_coralpascal"
      args.imdbval_name = "voc_2007_test_coralpascal"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "pascal_voc_0712":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "coco":
      args.imdb_name = "coco_2014_train"
      args.imdbval_name = "coco_2014_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "imagenet":
      args.imdb_name = "imagenet_train"
      args.imdbval_name = "imagenet_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
  elif args.dataset == "vg":
      # train sizes: train, smalltrain, minitrain
      # train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
      args.imdb_name = "vg_150-50-50_minitrain"
      args.imdbval_name = "vg_150-50-50_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)
  
  if args.coco:
      usecaffe=True
  else:
      usecaffe=False
  
  #torch.backends.cudnn.benchmark = True
  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.
  cfg.TRAIN.USE_FLIPPED = True
  cfg.USE_GPU_NMS = args.cuda
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
  train_size = len(roidb)

  print('{:d} roidb entries'.format(len(roidb)))

  output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  sampler_batch = sampler(train_size, args.batch_size)
  
  if args.net =='res10':
      smallstd=True
  else:
      smallstd=False
  print(imdb.num_classes)
  dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           imdb.num_classes, training=True, usecaffe=smallstd)

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                            sampler=sampler_batch, num_workers=args.num_workers)

  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)

  if args.cuda:
    cfg.CUDA = True

  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res34':
    fasterRCNN = resnet(imdb.classes, 34, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res18':
    fasterRCNN = resnet(imdb.classes, 18, pretrained=True, class_agnostic=args.class_agnostic)
    depth = 18
  elif args.net == 'res10':
    fasterRCNN = resnet(imdb.classes, 10, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN = resnet(imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
    depth = 152
  elif args.net == 'squeeze':
    fasterRCNN = squeeze(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'squeezemod':
    fasterRCNN = squeezemod(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'alex':
    fasterRCNN = alex(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()

  lr = cfg.TRAIN.LEARNING_RATE
  lr = args.lr
  #tr_momentum = cfg.TRAIN.MOMENTUM
  #tr_momentum = args.momentum

  params = []
  for key, value in dict(fasterRCNN.named_parameters()).items():
    if value.requires_grad:
      if 'bias' in key:
        params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
      else:
        params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

  if args.optimizer == "adam":
    lr = lr * 0.1
    optimizer = torch.optim.Adam(params)

  elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

  if args.resume:
    print("loading previous models..")
    load_name = os.path.join(output_dir,
                         'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    args.start_epoch = checkpoint['epoch']
    
    if args.coco:
        if args.dataset=="coco":
            print("loading for coco dataset..")
            fasterRCNN.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr = optimizer.param_groups[0]['lr']
#            if 'pooling_mode' in checkpoint.keys():
#              cfg.POOLING_MODE = checkpoint['pooling_mode']
            print("loaded checkpoint %s" % (load_name))

        else:
            print("loading coco..")
            # load only matched weights
              # initilize the network here.
            args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
            args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
            cfg.ANCHOR_SCALES = [4, 8, 16, 32]
            
    #        load_name="models/res101/pascal_voc/faster_rcnn_1_1_3739.pth"
            checkpoint = torch.load(load_name)
            model_coco = resnet(range(81), depth, pretrained=True, class_agnostic=args.class_agnostic)    
            model_coco.create_architecture()
            model_coco.load_state_dict(checkpoint['model'])
            pretrained_dict = model_coco.state_dict()
            model_dict = fasterRCNN.state_dict()
            
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # check size
            for i in pretrained_dict:
                if torch.tensor(pretrained_dict[i]).size == torch.tensor(model_dict[i]).size:
                    model_dict[i]=pretrained_dict[i]
            fasterRCNN.load_state_dict(model_dict)
            optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        # try loading pascal models
        # if fail, try loading coco models and change the cls layer
        try:
            fasterRCNN.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        except:
            if 'res' in args.net:
                model_coco = resnet(range(81), depth, pretrained=True, class_agnostic=args.class_agnostic)    
            elif 'squeeze' in args.net:
                model_coco = squeeze(range(81), pretrained=True, class_agnostic=args.class_agnostic)    
            model_coco.create_architecture()
            model_coco.load_state_dict(checkpoint['model'])
            pretrained_dict = model_coco.state_dict()
            model_dict = fasterRCNN.state_dict()
            
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # check size
            for i in pretrained_dict:
                if torch.tensor(pretrained_dict[i]).size == torch.tensor(model_dict[i]).size:
                    model_dict[i]=pretrained_dict[i]
            fasterRCNN.load_state_dict(model_dict)
            optimizer.load_state_dict(checkpoint['optimizer'])
            
    
    # freeze parameters
    # resnet, 
    #layer=0:rpn
    #layer=5:resnet
    #layer=6:FRCNN-top layer
    #layer=7:class regression
    #layer=8:bbox regression
#    for nlayer, child in enumerate(fasterRCNN.children()):
#        print(child)
#        if nlayer == 5: #or nlayer == 7 or nlayer == 8: #or nlayer == 6: #RPN and resnet
#            for i,param in enumerate(child.parameters()):                   
#                if nlayer ==6 and i > 8:
#                    continue
#                print("disabling training for layer", nlayer)   
#                param.requires_grad = False
    print(fasterRCNN)                              
        
    if not args.dataset == "coco":# and args.optimizer == "adam":
        weight_decay = optimizer.param_groups[0]['weight_decay']
            
        double_bias = True
        bias_decay = True
            
        print(cfg.POOLING_MODE)
        params = []
        for key, value in dict(fasterRCNN.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params':[value],'lr':lr*(double_bias + 1), \
                            'weight_decay': bias_decay and weight_decay or 0}]
                else:
                    params += [{'params':[value],'lr':lr, 'weight_decay': weight_decay}]
#        if args.optimizer == "adam":
        optimizer = torch.optim.Adam(params)
#        else:
#            optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)
    else:
        weight_decay = optimizer.param_groups[0]['weight_decay']
            
        double_bias = True
        bias_decay = True
            
        print(cfg.POOLING_MODE)
        params = []
        for key, value in dict(fasterRCNN.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params':[value],'lr':lr*(double_bias + 1), \
                            'weight_decay': bias_decay and weight_decay or 0}]
                else:
                    params += [{'params':[value],'lr':lr, 'weight_decay': weight_decay}]
    
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)
        
  if args.mGPUs:
    fasterRCNN = nn.DataParallel(fasterRCNN)

  if args.cuda:
    fasterRCNN.cuda()

  iters_per_epoch = int(train_size / args.batch_size)

  if args.use_tfboard:
    from tensorboardX import SummaryWriter
    logger = SummaryWriter("logs")
    
  total_tic = time.time()
  
  for epoch in range(args.start_epoch, args.max_epochs + 1):
    # setting to train mode
    fasterRCNN.train()
    loss_temp = 0
    start = time.time()

    if epoch % (args.lr_decay_step + 1) == 0:
        adjust_learning_rate(optimizer, args.lr_decay_gamma)
        lr *= args.lr_decay_gamma

    data_iter = iter(dataloader)
    for step in range(iters_per_epoch):
      data = next(data_iter)
      im_data.data.resize_(data[0].size()).copy_(data[0])
      im_info.data.resize_(data[1].size()).copy_(data[1])
      gt_boxes.data.resize_(data[2].size()).copy_(data[2])
      num_boxes.data.resize_(data[3].size()).copy_(data[3])

      fasterRCNN.zero_grad()
      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label, _  = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

      loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
           + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
      try:
        loss_temp += loss.item()
      except:
        loss_temp += loss
      # backward
      optimizer.zero_grad()
      loss.backward()
      if args.net == "vgg16":
          clip_gradient(fasterRCNN, 10.)
      optimizer.step()

      if step % args.disp_interval == 0:
        end = time.time()
        if step > 0:
          loss_temp /= (args.disp_interval + 1)

        if args.mGPUs:
          loss_rpn_cls = rpn_loss_cls.mean().item()
          loss_rpn_box = rpn_loss_box.mean().item()
          loss_rcnn_cls = RCNN_loss_cls.mean().item()
          loss_rcnn_box = RCNN_loss_bbox.mean().item()
          fg_cnt = torch.sum(rois_label.data.ne(0))
          bg_cnt = rois_label.data.numel() - fg_cnt
        else:
          loss_rpn_cls = rpn_loss_cls.item()
          loss_rpn_box = rpn_loss_box.item()
          loss_rcnn_cls = RCNN_loss_cls.item()
          loss_rcnn_box = RCNN_loss_bbox.item()
          fg_cnt = torch.sum(rois_label.data.ne(0))
          bg_cnt = rois_label.data.numel() - fg_cnt

        print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                                % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
        print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
        print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                      % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
        if args.use_tfboard:
          info = {
            'loss': loss_temp,
            'loss_rpn_cls': loss_rpn_cls,
            'loss_rpn_box': loss_rpn_box,
            'loss_rcnn_cls': loss_rcnn_cls,
            'loss_rcnn_box': loss_rcnn_box
          }
          logger.add_scalars("logs_s_{}/losses".format(args.session), info, (epoch - 1) * iters_per_epoch + step)

        loss_temp = 0
        start = time.time()

    if not usecaffe:
        save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}.pth'.format(args.session, epoch))
    else:
        save_name = os.path.join(output_dir, 'coco_faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))
    if epoch%int(args.save)==0:
        save_checkpoint({
          'session': args.session,
          'epoch': epoch + 1,
          'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
          'optimizer': optimizer.state_dict(),
          'pooling_mode': cfg.POOLING_MODE,
          'class_agnostic': args.class_agnostic,
        }, save_name)
        print('save model: {}'.format(save_name))

  if args.use_tfboard:
    logger.close()

gputime=time.time()-total_tic
print("epoch time",gputime)