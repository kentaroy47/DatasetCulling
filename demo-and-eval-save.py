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
# from model.nms.nms_wrapper import nms
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
  parser.add_argument('--scale', dest='scale',
                      help='optional config file',
                      default=1, type=float)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
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
  im_orig = im[:,:,:].astype(np.float32, copy=True)
#  im_orig -= cfg.PIXEL_MEANS
# changed to use pytorch models
  im_orig /= 255. # Convert range to [0,1]
  pixel_means = [0.485, 0.456, 0.406]
  im_orig -= pixel_means # Minus mean
  pixel_stdens = [0.229, 0.224, 0.225]
  im_orig /= pixel_stdens # divide by stddev
#  im_orig = im
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

  input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
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

      with torch.no_grad():
              im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
              im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
              gt_boxes.resize_(1, 1, 5).zero_()
              num_boxes.resize_(1).zero_()

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
            keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
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
  
  # evaluate mAP
  pascal_classes = np.asarray(['__background__',
                           'aeroplane', 'bicycle', 'bear', 'boat',
                           'bottle', 'bus', 'car', 'cat', 'chair',
                           'cow', 'diningtable', 'dog', 'horse',
                           'motorbike', 'person', 'pottedplant',
                           'sheep', 'sofa', 'train', 'truck'])
      
  
#  if args.coco:
  coco_classes = np.asarray(['__background__',"person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat",
                           "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse",
                           "sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie",
                           "suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat",
                           "baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass",
                           "cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli",
                           "carrot","hot dog","pizza","donut","cake","chair","sofa","pottedplant","bed",
                           "diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard","cell phone",
                           "microwave","oven","toaster","sink",
                           "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"])
  
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
  
  try:
      truth_boxes2 = pickle.load(open(args.truth, "rb"))
      truth_boxes=[]
      if len(truth_boxes2)==81 and sum(x==[] for x in truth_boxes2[21])==len(truth_boxes2[21]):
          truth_boxes=truth_boxes2[0:21]
      elif len(truth_boxes2)==81:      
          for n in pascal_classes:
                for i,cls in enumerate(coco_classes):
                    if n == cls:
                        print(cls)
                        print(i)
                        truth_boxes.append(truth_boxes2[i])
      else:
          truth_boxes=truth_boxes2
  except:
      a=1
      
#  truth_boxes = pickle.load( open("output/res50/voc_2007_test/faster_rcnn_10/detections.pkl", "rb" ) )
  ovthresh = 0.5
  
  TRUTH_THRESHOLD=0.75
  diflist=[]
  
  # filter truth by confidence > 50%
  # sample difficult
  for i in range(len(truth_boxes[:])):
      diflist2=[]
      for ii in range(len(truth_boxes[i][:])):          
          
          poplist = []
          dif=0
          for iii in range(len(truth_boxes[i][ii][:])):
              
              out = truth_boxes[i][ii][iii][0:4]
              
              if truth_boxes[i][ii][iii][4] < TRUTH_THRESHOLD:
                  poplist.append(iii)
              elif min(out[2]-out[0],out[3]-out[1]) < 25:
                  dif+=1
          diflist2.append(dif)
          if len(poplist) is not 0:
              for pop in reversed(poplist):
#                  truth_boxes[i][ii] = np.delete(truth_boxes[i][ii], pop, axis=0)
                  truth_boxes[i][ii] = np.delete(truth_boxes[i][ii], pop, axis=0)  
      diflist.append(diflist2)
  
  # concat cars and cyclists.
  if "tennis" in args.dataset or "badminton" in args.dataset:
      a=1
#      all_boxes = addcars(all_boxes)
#      all_boxes = addcyclists(all_boxes)
    
  aps = []
  for ncls, cls in enumerate(pascal_classes):
      if cls is not "__background__":
#      if cls is "car":
          print("detection for ", cls)
          BBs = all_boxes[ncls][:]
          BBGTs = truth_boxes[ncls][:]
          # flatten
          BB, confidence, idx = flatten(BBs)
          confidence = np.asarray(confidence)
          BB = np.asarray(BB)
          idx = np.asarray(idx)
          
          BBGT_flat, gtconf, gtidx = flatten(BBGTs)
          gtidx = np.asarray(gtidx)
          BBGT_flat = np.asarray(BBGT_flat)
          
          nd = len(BBGT_flat)
#          nd = len(BB)
          tp = np.zeros(nd)
          fp = np.zeros(nd)
          npos = nd-sum(diflist[ncls])
          
          det=[False] * len(BBGTs)

          
          if len(BB) > 0:
                # sort by confidence
                sorted_ind = np.argsort(-confidence)
                sorted_scores = np.sort(-confidence)
                BB = BB[sorted_ind, :]
                idx = [idx[x] for x in sorted_ind]

                
                # go down dets and mark TPs and FPs
                for d in range(nd):
                  if d >= len(BB):
                      continue
                  bb = BB[d,:].astype(float)
                  ovmax = -np.inf
                  index = -np.inf

                  try:
                      BBGT = BBGTs[idx[d]]
                  except:
                      BBGT = []
            
                  if len(BBGT) > 0:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(BBGT[:,0], bb[0])
                    iymin = np.maximum(BBGT[:,1], bb[1])
                    ixmax = np.minimum(BBGT[:,2], bb[2])
                    iymax = np.minimum(BBGT[:,3], bb[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih
            
                    # union
                    uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                           (BBGT[:,2] - BBGT[:,0] + 1.) *
                           (BBGT[:,3] - BBGT[:,1] + 1.) - inters)
            
                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)
            
                  # penaltize by confidence?? not much change
                  if ovmax > ovthresh:
                      tp[d] = 1.
                  else:
                      fp[d] = 1.
#                  if ovmax > ovthresh:
#                        if not det[jmax]:
#                            tp[d] = 1.
#                            det[jmax] = 1
#                        else:
#                            fp[d] = 1.
#                  else:
#                        fp[d] = 1.

        
          # compute precision recall
          fp = np.cumsum(fp)
          tp = np.cumsum(tp)
          rec = tp / float(npos)
          # avoid divide by zero in case the first detection matches a difficult
          # ground truth
          prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
          ap = voc_ap(rec, prec, use_07_metric=True)
#          print("ap:", ap)
          aps.append(ap)
      
  for x in aps:  
      print(x)
  print("mAP", np.mean(aps))

  end = time.time()
  print("test time: %0.4fs" % (end - start))
    
  import pickle
#outfile = 'output/' + target + '-' + args.net + '.pkl'
#with open(outfile, 'wb') as f:
#      pickle.dump(allbox, f, pickle.HIGHEST_PROTOCOL)
  
  counted = count(truth_boxes)
  
  if os.path.isfile("results/results-"+str(args.dataset)+"-"+str(args.checksession)+".txt"):
      mode = "a"
  else:
      mode = "w"
  with open("results/results-"+str(args.dataset)+"-"+str(args.checksession)+".txt", mode) as f: 
    f.write(load_name+"\n")
    f.write(args.image_dir+"\n")
    f.write("Person AP:" + str(aps[15]) + "\n")
    f.write("Car AP:" + str(aps[7]) + "\n")
    f.write("Bike AP:" + str(aps[2]) + "\n")
    f.write("Motorbike AP:" + str(aps[14]) + "\n")
    f.write("Bus AP:" + str(aps[6]) + "\n")
    f.write("Dog AP:" + str(aps[12]) + "\n")
    f.write("Train AP:" + str(aps[19]) + "\n")
    f.write("Truck AP:" + str(aps[20]) + "\n")
    f.write("Bear AP:" + str(aps[3]) + "\n")
    f.write("Person num:" + str(counted[15]) + "\n")
    f.write("Car num:" + str(counted[7]) + "\n")
    if args.count:
        f.write("Person num:" + str(len(truth_boxes[15])) + "\n")
        f.write("Car num:" + str(len(truth_boxes[7])) + "\n")
        f.write("Bike num:" + str(len(truth_boxes[2])) + "\n")
        f.write("Motorbike num:" + str(len(truth_boxes[14])) + "\n")
        f.write("Bus num:" + str(len(truth_boxes[6])) + "\n")
        f.write("Dog num:" + str(len(truth_boxes[12])) + "\n")
        f.write("Train num:" + str(len(truth_boxes[19])) + "\n")
        f.write("Truck num:" + str(len(truth_boxes[20])) + "\n")
        f.write("Bear num:" + str(len(truth_boxes[3])) + "\n")
    
    useful = aps[15],aps[7],aps[2],aps[14],aps[19],aps[6]
    f.write(str(useful) + "\n")
    f.write("mAP:" + str(np.mean(useful)) + "\n")
    
    
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
  
  truth_boxes2 = pickle.load(open(args.truth, "rb"))
  truth_boxes=[]
  if len(truth_boxes2)==81 and sum(x==[] for x in truth_boxes2[21])==len(truth_boxes2[21]):
      truth_boxes=truth_boxes2[0:21]
  elif len(truth_boxes2)==81:     
      for n in pascal_classes:
            for i,cls in enumerate(coco_classes):
                if n == cls:
                    print(cls)
                    print(i)
                    truth_boxes.append(truth_boxes2[i])
  else:
      truth_boxes=truth_boxes2
      
  if args.save:
      outfile = 'output/baseline/' + args.dataset + '-'+ args.net  +'.pkl'
      with open(outfile, 'wb') as f:
          pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
      
#  truth_boxes = pickle.load( open("output/res50/voc_2007_test/faster_rcnn_10/detections.pkl", "rb" ) )
  ovthresh = 0.5
  
  TRUTH_THRESHOLD=0.75
  diflist=[]
  
    # filter truth by confidence > 50%
  # sample difficult
  for i in range(len(truth_boxes[:])):
      diflist2=[]
      for ii in range(len(truth_boxes[i][:])):          
          
          poplist = []
          dif=0
          for iii in range(len(truth_boxes[i][ii][:])):
              
              out = truth_boxes[i][ii][iii][0:4]
              
              if truth_boxes[i][ii][iii][4] < TRUTH_THRESHOLD:
                  poplist.append(iii)
              elif min(out[2]-out[0],out[3]-out[1]) < 30:
                  dif+=1
          diflist2.append(dif)
          if len(poplist) is not 0:
              for pop in reversed(poplist):
#                  truth_boxes[i][ii] = np.delete(truth_boxes[i][ii], pop, axis=0)
                  truth_boxes[i][ii] = np.delete(truth_boxes[i][ii], pop, axis=0)  
      diflist.append(diflist2)
      
  aps = []
  for ncls, cls in enumerate(pascal_classes):
      if cls is not "__background__":
#      if cls is "car":
          print("detection for ", cls)
          BBs = all_boxes[ncls][:]
          BBGTs = truth_boxes[ncls][:]
          # flatten
          BB, confidence, idx = flatten(BBs)
          confidence = np.asarray(confidence)
          BB = np.asarray(BB)
          idx = np.asarray(idx)
          
          BBGT_flat, gtconf, gtidx = flatten(BBGTs)
          gtidx = np.asarray(gtidx)
          BBGT_flat = np.asarray(BBGT_flat)
          
          nd = len(BBGT_flat)
#          nd = len(BB)
          tp = np.zeros(nd)
          fp = np.zeros(nd)
          npos = nd-sum(diflist[ncls])
          
          det=[False] * len(BBGTs)

          
          if len(BB) > 0:
                # sort by confidence
                sorted_ind = np.argsort(-confidence)
                sorted_scores = np.sort(-confidence)
                BB = BB[sorted_ind, :]
                idx = [idx[x] for x in sorted_ind]

                
                # go down dets and mark TPs and FPs
                for d in range(nd):
                  if d >= len(BB):
                      continue
                  bb = BB[d,:].astype(float)
                  ovmax = -np.inf
                  index = -np.inf

                  try:
                      BBGT = BBGTs[idx[d]]
                  except:
                      BBGT = []
            
                  if len(BBGT) > 0:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(BBGT[:,0], bb[0])
                    iymin = np.maximum(BBGT[:,1], bb[1])
                    ixmax = np.minimum(BBGT[:,2], bb[2])
                    iymax = np.minimum(BBGT[:,3], bb[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih
            
                    # union
                    uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                           (BBGT[:,2] - BBGT[:,0] + 1.) *
                           (BBGT[:,3] - BBGT[:,1] + 1.) - inters)
            
                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)
            
                  # penaltize by confidence?? not much change
                  if ovmax > ovthresh:
                      tp[d] = 1.
                  else:
                      fp[d] = 1.
      
          # compute precision recall
          fp = np.cumsum(fp)
          tp = np.cumsum(tp)
          rec = tp / float(npos)
          # avoid divide by zero in case the first detection matches a difficult
          # ground truth
          prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
          ap = voc_ap(rec, prec, use_07_metric=True)
#          print("ap:", ap)
          aps.append(ap)
      
  for x in aps:  
      print(x)
  print("mAP", np.mean(aps))

  end = time.time()
  print("test time: %0.4fs" % (end - start))


import pickle
outfile = 'output/' + str(args.dataset) + '-' + args.net + '.pkl'
if args.writeout:
    with open(outfile, 'wb') as f:
          pickle.dump(allbox, f, pickle.HIGHEST_PROTOCOL)

with open("results/results-"+str(args.dataset)+"-"+str(args.checksession)+".txt", "a") as f: 
    f.write(load_name+"\n")
    try:
        args.nout
        f.write("nout="+str(args.nout)+"\n")
    except NameError:
        a=1
    
    f.write(args.image_dir+"\n")
    f.write("!!Filtered < 30 !!\n")
    f.write("Person AP:" + str(aps[15]) + "\n")
    f.write("Car AP:" + str(aps[7]) + "\n")
    f.write("Bike AP:" + str(aps[2]) + "\n")
    f.write("Motorbike AP:" + str(aps[14]) + "\n")
    f.write("Bus AP:" + str(aps[6]) + "\n")
    f.write("Dog AP:" + str(aps[12]) + "\n")
    f.write("Train AP:" + str(aps[19]) + "\n")
    f.write("Truck AP:" + str(aps[20]) + "\n")
    f.write("Bear AP:" + str(aps[3]) + "\n")
  
  # filter truth by confidence > 50%
  # sample difficult
for i in range(len(truth_boxes[:])):
      diflist2=[]
      for ii in range(len(truth_boxes[i][:])):          
          
          poplist = []
          dif=0
          for iii in range(len(truth_boxes[i][ii][:])):
              
              out = truth_boxes[i][ii][iii][0:4]
              
              if truth_boxes[i][ii][iii][4] < TRUTH_THRESHOLD:
                  poplist.append(iii)
              elif min(out[2]-out[0],out[3]-out[1]) < 40:
                  dif+=1
          diflist2.append(dif)
          if len(poplist) is not 0:
              for pop in reversed(poplist):
#                  truth_boxes[i][ii] = np.delete(truth_boxes[i][ii], pop, axis=0)
                  truth_boxes[i][ii] = np.delete(truth_boxes[i][ii], pop, axis=0)  
      diflist.append(diflist2)
      
aps = []
for ncls, cls in enumerate(pascal_classes):
      if cls is not "__background__":
#      if cls is "car":
          print("detection for ", cls)
          BBs = all_boxes[ncls][:]
          BBGTs = truth_boxes[ncls][:]
          # flatten
          BB, confidence, idx = flatten(BBs)
          confidence = np.asarray(confidence)
          BB = np.asarray(BB)
          idx = np.asarray(idx)
          
          BBGT_flat, gtconf, gtidx = flatten(BBGTs)
          gtidx = np.asarray(gtidx)
          BBGT_flat = np.asarray(BBGT_flat)
          
          nd = len(BBGT_flat)
#          nd = len(BB)
          tp = np.zeros(nd)
          fp = np.zeros(nd)
          npos = nd-sum(diflist[ncls])
          
          det=[False] * len(BBGTs)

          
          if len(BB) > 0:
                # sort by confidence
                sorted_ind = np.argsort(-confidence)
                sorted_scores = np.sort(-confidence)
                BB = BB[sorted_ind, :]
                idx = [idx[x] for x in sorted_ind]

                
                # go down dets and mark TPs and FPs
                for d in range(nd):
                  if d >= len(BB):
                      continue
                  bb = BB[d,:].astype(float)
                  ovmax = -np.inf
                  index = -np.inf

                  try:
                      BBGT = BBGTs[idx[d]]
                  except:
                      BBGT = []
            
                  if len(BBGT) > 0:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(BBGT[:,0], bb[0])
                    iymin = np.maximum(BBGT[:,1], bb[1])
                    ixmax = np.minimum(BBGT[:,2], bb[2])
                    iymax = np.minimum(BBGT[:,3], bb[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih
            
                    # union
                    uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                           (BBGT[:,2] - BBGT[:,0] + 1.) *
                           (BBGT[:,3] - BBGT[:,1] + 1.) - inters)
            
                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)
            
                  # penaltize by confidence?? not much change
                  if ovmax > ovthresh:
                      tp[d] = 1.
                  else:
                      fp[d] = 1.
#                  if ovmax > ovthresh:
#                        if not det[jmax]:
#                            tp[d] = 1.
#                            det[jmax] = 1
#                        else:
#                            fp[d] = 1.
#                  else:
#                        fp[d] = 1.

        
          # compute precision recall
          fp = np.cumsum(fp)
          tp = np.cumsum(tp)
          rec = tp / float(npos)
          # avoid divide by zero in case the first detection matches a difficult
          # ground truth
          prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
          ap = voc_ap(rec, prec, use_07_metric=True)
#          print("ap:", ap)
          aps.append(ap)
      
for x in aps:  
      print(x)

import pickle
outfile = 'output/' + str(args.dataset) + '-' + args.net + '.pkl'
if args.writeout:
    with open(outfile, 'wb') as f:
          pickle.dump(allbox, f, pickle.HIGHEST_PROTOCOL)

with open("results/results-"+str(args.dataset)+"-"+str(args.checksession)+".txt", "a") as f: 
    f.write(load_name+"\n")
    try:
        args.nout
        f.write("nout="+str(args.nout)+"\n")
    except NameError:
        a=1
    
    f.write(args.image_dir+"\n")
    f.write("!!Filtered < 40 !!\n")
    f.write("Person AP:" + str(aps[15]) + "\n")
    f.write("Car AP:" + str(aps[7]) + "\n")
    f.write("Bike AP:" + str(aps[2]) + "\n")
    f.write("Motorbike AP:" + str(aps[14]) + "\n")
    f.write("Bus AP:" + str(aps[6]) + "\n")
    f.write("Dog AP:" + str(aps[12]) + "\n")
    f.write("Train AP:" + str(aps[19]) + "\n")
    f.write("Truck AP:" + str(aps[20]) + "\n")
    f.write("Bear AP:" + str(aps[3]) + "\n")
