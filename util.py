
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 21:57:57 2019

@author: ken
"""

import numpy as np

def filter_jackson(data):
    for i, clsbox in enumerate(data):
      for ii, boxes in enumerate(clsbox):
          dellist=[]
          for iii, box in enumerate(boxes):
              out = box[0:5]
#              print(out)
              if (out[2]-out[0]) > 100 and (out[3]-out[1]) > 100 and i==7:
                  dellist.append(iii)
#                  print(out)
                  
          for row in reversed(dellist):
             data[i][ii]=np.delete(data[i][ii],(row),axis=0) 
    return(data)

def addcars(datas):
    output = datas
    for i, data in enumerate(datas):
        for ii, detect in enumerate(data):
            if i == 6 or i == 20:
                if detect != []:
                    output[7][ii] = np.concatenate((output[7][ii], detect))
        
    return output     

def coco2pascal(data):
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
  
  # filter small guess.
  for i, clsbox in enumerate(data):
      for ii, boxes in enumerate(clsbox):
          dellist=[]
          for iii, box in enumerate(boxes):
              out = box[0:5]
#              print(out)
              if min(out[2]-out[0],out[3]-out[1]) < 20 or out[4] < 0.5:
                  dellist.append(iii)
#                  print(out)
                  
          for row in reversed(dellist):
             data[i][ii]=np.delete(data[i][ii],(row),axis=0) 
  # filter low confidence guess 
    
  # convert to 20 class
  allbox2 = []
  if len(data)==81:
      for n in pascal_classes:
            for i,cls in enumerate(coco_classes):
                if n == cls:
#                    print(cls)
#                    print(i)
                    allbox2.append(data[i])
      data = allbox2
  else:
      data = data       
          
  # modify car class
  data = addcars(data)
  return(data)

def conv20class(data):
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
      
    # convert to 20 class
  allbox2 = []
  if len(data)==81:
      for n in pascal_classes:
            for i,cls in enumerate(coco_classes):
                if n == cls:
#                    print(cls)
#                    print(i)
                    allbox2.append(data[i])
      data = allbox2
  else:
      data = data  
  return(data)
  
def flatten(libs):
    flatlist = []
    confidence = []
    idx = []
    for i,minilib in enumerate(libs):
#        for minilib in lib:
            flatlist.append(minilib[0:4])
#            print(minilib)
            confidence.append(minilib[4])
            idx.append(i)
    return flatlist, confidence, idx
  
def compute_ltrain(teacher, student):
      all_boxes = student
      truth_boxes = teacher
      aps = []
      pascal_classes = np.asarray(['__background__',
                           'aeroplane', 'bicycle', 'bear', 'boat',
                           'bottle', 'bus', 'car', 'cat', 'chair',
                           'cow', 'diningtable', 'dog', 'horse',
                           'motorbike', 'person', 'pottedplant',
                           'sheep', 'sofa', 'train', 'truck'])
    
      if 1 == 1:
          if 1 == 1:
#              print("detection for ", cls)
              BBs = all_boxes
              BBGTs = truth_boxes
              
              # flatten
              BB, confidence, idx = flatten(BBs)
              confidence = np.asarray(confidence)
              BB = np.asarray(BB)
              idx = np.asarray(idx)
              
              BBGT_flat, gtconf, gtidx = flatten(BBGTs)
              gtidx = np.asarray(gtidx)
              BBGT_flat = np.asarray(BBGT_flat)
              
              nd = len(BBGT_flat)
              tp = np.zeros(nd)
              fp = np.zeros(nd)
              npos = nd
              
              if len(BB) > 0:
                    # sort by confidence
                    sorted_ind = np.argsort(-confidence)
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
                        try:
                            ixmin = np.maximum(BBGT[:,0], bb[0])
                        except:
                            BBGT=BBGT.reshape(1,5)
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
                      if ovmax > 0.5:
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
#              print("rec:", rec)
              try:
                  # normalize by number of samples
#                  rec_loss = len(tp)/(max(rec)+0.1)#/counter[ncls]
                    if sum(tp)==0:
                        eps=0.5
                        rec_loss = np.abs(float(npos)/(sum(tp)+eps))
                        prec_loss = (sum(tp)+sum(fp))/(sum(tp)+eps)
                    else:
                        eps=0
                        rec_loss = np.abs(float(npos)/(sum(tp)+eps))
                        prec_loss = (sum(tp)+sum(fp))/(sum(tp)+eps)
              except:
                  rec_loss = 0
              aps.append(rec_loss+prec_loss)
      return(aps)
  
def add_answer(ans, data):
    # ans: list to be added
    # data: data to be added
    # n: 
    ans2 = []
    for i, d in enumerate(data):
        if len(ans)==0:
            ans_sub = []
        else:
            ans_sub = ans[i]
        for ii,dd in enumerate(d):
            ans_sub.append(dd)
        ans2.append(ans_sub)
    return ans2

def add_answer_cut(ans, data):
    # ans: list to be added
    # data: data to be added
    # n: 
    ans2 = []
    for i, d in enumerate(data):
        if len(ans)==0:
            ans_sub = []
        else:
            ans_sub = ans[i]
        for ii,dd in enumerate(d[:-1]):
            ans_sub.append(dd)
        ans2.append(ans_sub)
    return ans2

def add_answer_cut1000(ans, data):
    # ans: list to be added
    # data: data to be added
    # n: 
    ans2 = []
    for i, d in enumerate(data):
        if len(ans)==0:
            ans_sub = []
        else:
            ans_sub = ans[i]
        for ii,dd in enumerate(d[:500]):
            ans_sub.append(dd)
        ans2.append(ans_sub)
    return ans2

def choose_answer_limit(data, cls, indexes):
    # ans: list to be added
    # data: data to be added
    # n: 
    ans2 = []
    for i, d in enumerate(data):
        ans_sub = []
        for ii,index in enumerate(indexes):
            if i in cls:
                ans_sub.append(data[i][index])
            else:
                ans_sub.append(data[0][0])
        ans2.append(ans_sub)
    return ans2

def count_get_image(data, ii, thresh):
    numobj = []
    imglist = []
    for answers in data[ii][:-1]:
        count = 0
        for answer in answers:
            if answer[4]>0.5:
                count += 1
        numobj.append(count)
        
    
    for i, x in enumerate(numobj):
        if x > thresh:
            imglist.append(i)
    return(imglist)
    
def count_image(data, ii):
    numobj = []
    for answers in data[ii][:-1]:
        count = 0
        for answer in answers:
            if answer[4]>0.5:
                count += 1
        numobj.append(count)
        
    return(numobj)
    
def filter_guess(data):
      # filter small guess.
  for i, clsbox in enumerate(data):
      for ii, boxes in enumerate(clsbox):
          dellist=[]
          for iii, box in enumerate(boxes):
              out = box[0:5]
#              print(out)
              if min(out[2]-out[0],out[3]-out[1]) < 20:
                  dellist.append(iii)
#                  print(out)
                  
          for row in reversed(dellist):
             data[i][ii]=np.delete(data[i][ii],(row),axis=0) 
  return(data)
   
def proposed_loss(x, Q):
    eps = 1e-5
    t = 0.5
    out = [-(x*np.log10(x+eps))*Q+(1-x)*np.exp(x)/(np.exp(x)+1)+t]
#    out= [1/(816776*x^4-2E6*x^3+2E6*x^2-524983*x+69975)]
    return(out)
    
def compute_confusion(sdata, Q):
    out = 0
    for data in sdata:
        if data[4] > 0.2:
            out += proposed_loss(data[4], Q)[0]
    return(out)
    
def compute_confusion_top(student, dataset, normalize=True, Q=4):
    ap=[]
    final=[]
    ap.append(0)
    for x in range(1,21): 
        outap=[]
        for i, sdata in enumerate(student[0]):               
           outap.append(compute_confusion(student[x][i], Q))
        if normalize:
            outap/=np.std(outap)+1e-5
            outap-=np.mean(outap)
        else:
            outap=np.asarray(outap)
        ap.append(outap)
        
    if dataset =="coral" or "badminton" in dataset or "tennis" in dataset:
        final=ap[15]
    elif "kentucky" in dataset:
        temp = np.asarray(ap[7])+np.asarray(ap[15])
        final=temp/2
        
    elif "taipei" in dataset:
        temp = ap[7]+ap[15]+ap[6]
        final=temp/3
    else:
        temp = ap[7]+ap[15]
        final=temp/2
    return(final, ap)

def compute_ltrain_top(student, teacher, dataset):
    ap=[]
    final=[]
    ap.append(0)
    for x in range(1,21): 
        outap=[]
        for i, sdata in enumerate(student[0]):               
           outap.append(compute_ltrain(teacher[x][i], student[x][i]))

        outap/=np.std(outap)+1e-5
        outap-=np.mean(outap)
        ap.append(outap)
        
    if dataset =="coral":
        final=ap[7]
    elif "kentucky" in dataset:
        temp = ap[7]+ap[15]
        final=temp
    elif "taipei" in dataset:
        temp = ap[7]+ap[15]+ap[6]
        final=temp
    else:
        temp = ap[7]+ap[15]
        final=temp
    return(final, ap)
    
def count_predictions(student, target, thresh):
    out = 0

    for chunck in student[15]:
        for data in chunck:
            if data[4] > thresh:
                out += 1
    if not "tennis" in target or not "coral" in target:
        for chunck in student[7]:
            for data in chunck:
                if data[4] > thresh:
                    out += 1
    return out

def coco_class_util():
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
    return coco_classes