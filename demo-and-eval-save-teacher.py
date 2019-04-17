# --------------------------------------------------------
# Tensorflow Faster R-CNN
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
import cv2
import torch
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as dset
from scipy.misc import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
#from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections_personj, vis_detections_person, vis_detections_car, vis_detections_bike
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from model.faster_rcnn.squeezenet import squeeze
from model.faster_rcnn.alex import alex
import pdb
from util import *

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

def addcars(datas):
    output = datas
    for i, data in enumerate(datas):
        for ii, detect in enumerate(data):
            if i == 6 or i == 20:
                if detect != []:
                    output[7][ii] = np.concatenate((output[7][ii], detect))
        
    return output                

def addcyclists(datas):
    output = datas
    for i, data in enumerate(datas):
        for ii, detect in enumerate(data):
            if i == 14:
                if detect != []:
                    output[2][ii] = np.concatenate((output[2][ii], detect))
        
    return output  

def count(truths):
    out = np.zeros(21)
    for i, truth in enumerate(truths):
        for detect in truth:
            out[i] += np.count_nonzero(detect)
    return out

def flatten(libs):
    flatlist = []
    confidence = []
    idx = []
    for i,lib in enumerate(libs):
        for minilib in lib:
            flatlist.append(minilib)
#            print(minilib)
            confidence.append(minilib[4])
            idx.append(i)
    return flatlist, confidence, idx

def voc_ap(rec, prec, use_07_metric=False):
  """ ap = voc_ap(rec, prec, [use_07_metric])
  Compute VOC AP given precision and recall.
  If use_07_metric is true, uses the
  VOC 07 11 point method (default:False).
  """
  if use_07_metric:
    # 11 point metric
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
      if np.sum(rec >= t) == 0:
        p = 0
      else:
        p = np.max(prec[rec >= t])
      ap = ap + p / 11.
  else:
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
      mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
  return ap

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
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/vgg16.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res101', type=str)
  parser.add_argument('--topx', dest='topx',
                      help='vgg16, res50, res101, res152',
                      default='128')
  parser.add_argument('--scale', dest='scale',
                      help='optional config file',
                      default=1, type=float)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--target', dest='target',
                      help='set target', default="jackson2",)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models',
                      default="models")
  parser.add_argument('--image_dir', dest='image_dir',
                      help='directory to load images for demo',
                      default="images")
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=7, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=14961, type=int)
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')
  parser.add_argument('--webcam_num', dest='webcam_num',
                      help='webcam ID number',
                      default=-1, type=int)
  parser.add_argument('--caffe', dest='caffe',
                      help='use caffe inits',
                      default=False, type=bool)
  parser.add_argument('--show', dest='imshow',
                      help='write out images',
                      default=False, type=bool)
  parser.add_argument('--truth', dest='truth',
                      help='directory to load models')
  parser.add_argument('--coco', dest='coco',
                      help='directory to load models', default=False, type=bool)
  parser.add_argument('--save', dest='save',
                      help='directory to load models', default=False, type=bool)
  parser.add_argument('--writeout', dest='writeout',
                      help='write out images',
                      default=False, type=bool)
  parser.add_argument('--nout', dest='nout',
                      help='write out images',
                      default="None", type=str)
  parser.add_argument('--count', dest='count',
                      help='write out images',
                      default=False, type=bool)
  

  args = parser.parse_args()
  return args



lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

if __name__ == '__main__':

  args = parse_args()
  cfg.TRAIN.SCALES = (int(args.scale*600),)
  cfg.TEST.SCALES = (int(args.scale*600),)
    
#  print('Called with args:')
#  print(args)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  cfg.USE_GPU_NMS = args.cuda

#  print('Using config:')
#  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.

  input_dir = "models/"
  if not os.path.exists(input_dir):
    raise Exception('There is no input directory for loading network from ' + input_dir)
  if os.path.isfile(os.path.join(input_dir,
    'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))):
      load_name = os.path.join(input_dir,
    'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
  else:
      load_name = os.path.join(input_dir,
    'faster_rcnn_{}_{}.pth'.format(args.checksession, args.checkepoch))

  pascal_classes = np.asarray(['__background__',
                           'aeroplane', 'bicycle', 'bear', 'boat',
                           'bottle', 'bus', 'car', 'cat', 'chair',
                           'cow', 'diningtable', 'dog', 'horse',
                           'motorbike', 'person', 'pottedplant',
                           'sheep', 'sofa', 'train', 'truck'])

#  cfg.ANCHOR_SCALES = [4, 8, 16, 32]
#  cfg.ANCHOR_RATIOS = [0.5,1,2]
#  args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50'] 
  print("coco", args.coco)
  cfg.ANCHOR_SCALES = [4, 8, 16, 32]
  cfg.ANCHOR_RATIOS = [0.5,1,2]
  args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  
  if args.coco:
    pascal_classes = np.asarray(['__background__',"person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat",
                               "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse",
                               "sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie",
                               "suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat",
                               "baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass",
                               "cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli",
                               "carrot","hot dog","pizza","donut","cake","chair","sofa","pottedplant","bed",
                               "diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard","cell phone",
                               "microwave","oven","toaster","sink",
                               "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"])
      
  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(pascal_classes, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnet(pascal_classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN = resnet(pascal_classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res34':
    fasterRCNN = resnet(pascal_classes, 34, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res18':
    if args.coco:
        fasterRCNN = resnet(range(81), 18, pretrained=False, class_agnostic=args.class_agnostic)
    else:
        fasterRCNN = resnet(pascal_classes, 18, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'squeeze':
    fasterRCNN = squeeze(pascal_classes, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'alex':
    fasterRCNN = alex(pascal_classes, pretrained=True, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()
  


  fasterRCNN.create_architecture()
#  print(fasterRCNN)
  print("load checkpoint %s" % (load_name))
  if args.cuda > 0:
    checkpoint = torch.load(load_name)
  else:
    checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))

  fasterRCNN.load_state_dict(checkpoint['model'])
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']


  print('load model successfully!')

  # pdb.set_trace()

  print("load checkpoint %s" % (load_name))

  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda > 0:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data, volatile=True)
  im_info = Variable(im_info, volatile=True)
  num_boxes = Variable(num_boxes, volatile=True)
  gt_boxes = Variable(gt_boxes, volatile=True)

  if args.cuda > 0:
    cfg.CUDA = True

  if args.cuda > 0:
    fasterRCNN.cuda()

  fasterRCNN.eval()

  start = time.time()
  max_per_image = 100
  thresh = 0.025
  vis = True

  webcam_num = args.webcam_num
  # Set up webcam or get image directories
  if webcam_num >= 0 :
    cap = cv2.VideoCapture(webcam_num)
    num_images = 0
  else:
    imglist = sorted(os.listdir(args.image_dir), reverse=True)
    num_images = len(imglist)
    num_images2=num_images
  print('Loaded Photo: {} images.'.format(num_images))
  
  #box to save images
  allbox = [[[] for _ in xrange(num_images+1)]
               for _ in xrange(len(pascal_classes))]
  i = -1
  
  if args.save:
      roilist=[]
      base_featlist=[]
  
  while (num_images >= 00):
      i += 1
      total_tic = time.time()
      if webcam_num == -1:
        num_images -= 1

      # Get image from the webcam
      if webcam_num >= 0:
        if not cap.isOpened():
          raise RuntimeError("Webcam could not open. Please check connection.")
        ret, frame = cap.read()
        im_in = np.array(frame)
      # Load the demo image
      else:
        im_file = os.path.join(args.image_dir, imglist[num_images])
#        print(im_file)
        # im = cv2.imread(im_file)
        im_in = np.array(imread(im_file))
      if len(im_in.shape) == 2:
        im_in = im_in[:,:,np.newaxis]
        im_in = np.concatenate((im_in,im_in,im_in), axis=2)
      # rgb -> bgr
#      im = im_in[:,:,::-1]
#      im = im_in[240:400,:480,0:3]
      im = im_in

      blobs, im_scales = _get_image_blob(im)
      assert len(im_scales) == 1, "Only single-image batch implemented"
      im_blob = blobs
      im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

      im_data_pt = torch.from_numpy(im_blob)
      im_data_pt = im_data_pt.permute(0, 3, 1, 2)
      im_info_pt = torch.from_numpy(im_info_np)

      im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
      im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)
      gt_boxes.data.resize_(1, 1, 5).zero_()
      num_boxes.data.resize_(1).zero_()

      # pdb.set_trace()
      det_tic = time.time()

      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
      
#      print("roi", rois.shape)
      
      scores = cls_prob.data
      boxes = rois.data[:, :, 1:5]
      
      if cfg.TEST.BBOX_REG:
          # Apply bounding-box regression deltas
          box_deltas = bbox_pred.data
          if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
          # Optionally normalize targets by a precomputed mean and stdev
            if args.class_agnostic:
                if args.cuda > 0:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                box_deltas = box_deltas.view(1, -1, 4)
            else:
                if args.cuda > 0:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

          pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
          pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
      else:
          # Simply repeat the boxes, once for each class
          pred_boxes = np.tile(boxes, (1, scores.shape[1]))

      pred_boxes /= im_scales[0]

      scores = scores.squeeze()
      pred_boxes = pred_boxes.squeeze()
      
#      print("box:",pred_boxes.shape)
      det_toc = time.time()
      detect_time = det_toc - det_tic
      misc_tic = time.time()
      if vis:
          im2show = np.copy(im)
      for j in xrange(1, len(pascal_classes)):
          inds = torch.nonzero(scores[:,j]>thresh).view(-1)
          # if there is det
          if inds.numel() > 0:
            cls_scores = scores[:,j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            if args.class_agnostic:
              cls_boxes = pred_boxes[inds, :]
            else:
              cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
            
            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
            cls_dets = cls_dets[order]
            keep = nms(cls_dets[order, :], cls_scores[order], cfg.TEST.NMS)
            cls_dets = cls_dets[keep.view(-1).long()]
#            print("class:", pascal_classes[j])
#            print("cls_dets", cls_dets.cpu().numpy())
            a = cls_dets.cpu().numpy()
#            print(a.shape)
            allbox[j][i]=a
        
            if vis:
                # changed threshold to highlight the DSM improvements.
                # plot only person for certain dataset results. This is for image output only!!
              if "tennis" in args.dataset or "coral" in args.dataset or "badminton" in args.dataset:
                  if pascal_classes[j] == "person":
                      im2show = vis_detections_person(im2show, pascal_classes[j], cls_dets.cpu().numpy(), 0.8)
              else:
                  if "car" in pascal_classes[j] or "truck" in pascal_classes[j] or "bus" in pascal_classes[j]:
                      im2show = vis_detections_car(im2show, pascal_classes[j], cls_dets.cpu().numpy(), 0.65)
                  elif pascal_classes[j] == "person":
                      if  "jackson2" not in args.dataset:   
                          im2show = vis_detections_person(im2show, pascal_classes[j], cls_dets.cpu().numpy(), 0.8)
                      else:
                          im2show = vis_detections_personj(im2show, pascal_classes[j], cls_dets.cpu().numpy(), 0.8)
                  elif pascal_classes[j] == "motorbike":
                      im2show = vis_detections_bike(im2show, pascal_classes[j], cls_dets.cpu().numpy(), 0.9)
      else:
                empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))
                allbox[j][i] = empty_array

      misc_toc = time.time()
      nms_time = misc_toc - misc_tic

      if webcam_num == -1:
          sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                           .format(num_images + 1, len(imglist), detect_time, nms_time))
          sys.stdout.flush()

      if vis and webcam_num == -1:
          # cv2.imshow('test', im2show)
          # cv2.waitKey(0)
          im2show = cv2.cvtColor(im2show, cv2.COLOR_BGR2RGB)
          result_path = os.path.join(args.image_dir, imglist[num_images][:-4] + "_det.jpg")
          if args.imshow:
              cv2.imwrite(result_path, im2show)
      else:
#          im2showRGB = cv2.cvtColor(im2show, cv2.COLOR_BGR2RGB)
          if vis:
              cv2.imshow("frame", im2show)
          total_toc = time.time()
          total_time = total_toc - total_tic
          frame_rate = 1 / total_time
#          print('Frame rate:', frame_rate)
          if cv2.waitKey(1) & 0xFF == ord('q'):
              break
  if webcam_num >= 0:
      cap.release()
      cv2.destroyAllWindows()
  
  allbox2 = []
  
  # filter small guess.
  for i, clsbox in enumerate(allbox):
      for ii, boxes in enumerate(clsbox):
          dellist=[]
          for iii, box in enumerate(boxes):
              out = box[0:4]
              if min(out[2]-out[0],out[3]-out[1]) < 20:
                  dellist.append(iii)
          for row in reversed(dellist):
             allbox[i][ii]=np.delete(allbox[i][ii],(row),axis=0) 
          
  coco_classes = coco_class_util()
  
  # write out in pascal voc 2007 order
  if len(allbox)==81:
      for n in pascal_classes:
            for i,cls in enumerate(coco_classes):
                if n == cls:
                    print(cls)
                    print(i)
                    allbox2.append(allbox[i])
      all_boxes = allbox2
  else:
      all_boxes = allbox
  
    
  import pickle
  outfile = 'output/baseline/' + args.topx + '-' + args.target + '-' + args.net + '.pkl'
  with open(outfile, 'wb') as f:
          pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
