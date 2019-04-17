#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 08:42:47 2018

@author: kentaroy47
"""

import numpy as np
import copy
import os
import subprocess
import argparse

parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
parser.add_argument('--dataset', dest='dataset',
                      help='training dataset', type=str)
parser.add_argument('--numobj', dest='numobj',
                      help='training dataset', type=str, default=0)
parser.add_argument('--sample', dest='sample',
                      help='training dataset', type=str, default=0)
parser.add_argument('--filterpix', dest='filterpix',
                      help='filter pixel size', type=str, default=40)
parser.add_argument('--topx', dest='topx',
                      help='filter pixel size', type=str, default=64)
parser.add_argument('--net', dest='net',
                      help='target network', type=str, default="res101")
parser.add_argument('--dsm', dest='dsm',
                      help='filter pixel size', type=bool, default=False)
args = parser.parse_args()
target=args.dataset
args.numobj=int(args.numobj)

xmlsource = './'
datasource = 'images/'+target+'_train/*'
resultsdir = 'output/baseline/'+args.topx+'-'+target+'train-res101.pkl'
if not args.dsm:
    valdir = 'output/baseline/'+target+'val-res101.pkl'
else:
    valdir = 'output/baseline/'+target+'dsmval-res101.pkl'
targetdir = 'output/'+target+'-train-labels-res101/'

if args.net=="res18":
    resultsdir = 'output/baseline/'+target+'train-res18.pkl'
    valdir = 'output/baseline/'+target+'val-res18.pkl'
    targetdir = 'output/'+target+'-train-labels-res18/'


if not os.path.isdir(targetdir):
    subprocess.call("mkdir "+targetdir, shell=True)
    
trainfile = 'trainval/trainval_'+target+'.txt'
testfile = 'trainval/test_'+target+'.txt'
filterpix = int(args.filterpix)

train_num = 30000
THRESH = 0.75
OBJECT_ONLY = False
coco=False

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou

pascal_classes = np.asarray(['__background__',
                           'aeroplane', 'bicycle', 'bear', 'boat',
                           'bottle', 'bus', 'car', 'cat', 'chair',
                           'cow', 'diningtable', 'dog', 'horse',
                           'motorbike', 'person', 'pottedplant',
                           'sheep', 'sofa', 'train', 'truck'])

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


            
# get xml files
import glob
import pickle
import xmltodict
import os

#with open(datasource) as f:
files = sorted(glob.glob(datasource))

trainvals = []
outlists = []
MAX_SAMPLE = 100000
for file in files:
     trainvals.append(os.path.basename(file)[:-4])


# make template
with open('./000001.xml') as fd:
    doc = xmltodict.parse(fd.read())
del doc['annotation']['size']
del doc['annotation']['filename']
template = copy.deepcopy(doc['annotation']['object'])
template_all = copy.deepcopy(doc)


results2 = pickle.load(open(resultsdir, "rb"))
results=[]
if len(results2)==81:
    for n in pascal_classes:
        for i,cls in enumerate(coco_classes):
            if n == cls:
                print(cls)
                print(i)
                results.append(results2[i])
else:
    results=results2
    
            
    
classes=pascal_classes
        
box = []
boxsize = []
nbox = []
counter = np.zeros(len(classes))
#trainclass = classes[2],classes[7],classes[14],classes[15]
trainclass = classes

train_num=len(results[0])

for i,file in enumerate(trainvals[:train_num]):
    
    del doc['annotation']['object'] 
    doc['annotation']['object'] = copy.deepcopy(template)
        
    try:
        while len(doc['annotation']['object'])>0:
            temp = doc['annotation']['object'].pop()
    except:
        None
        
    flag = 0
    
    numbox=0
    for ncls, cls in enumerate(classes):
        bboxes=[]
        result = results[ncls][i]
        for out in result:
            if out[4] > THRESH and cls in trainclass: # confident
                a = temp
                a['bndbox']['xmin'] = int(np.floor(out[0]))
                a['bndbox']['ymin'] = int(np.floor(out[1]))
                a['bndbox']['xmax'] = int(np.floor(out[2]))
                a['bndbox']['ymax'] = int(np.floor(out[3]))
                a['name'] = cls
                
#                print("file",file,a)
                # filter small bbox
                if min(out[2]-out[0],out[3]-out[1]) > 20:                    
                    if min(out[2]-out[0],out[3]-out[1]) < filterpix:
                        a['difficult'] = 1
                    if (out[2]-out[0]) > 100 and (out[3]-out[1]) > 100 and cls == 'person' and "jackson" in args.dataset:
                        print("large person")
                        break
                    doc['annotation']['object'].append(copy.deepcopy(a))
                    flag += 1
                    counter[ncls] +=1
                    bboxes.append(out[0:4])
                    numbox+=1
                
                # monitor bbox
                
                box.append(copy.deepcopy(a))
                bsize = min(out[2]-out[0],out[3]-out[1])
                boxsize.append(bsize)                
                       
    nbox.append(flag)
    
    if args.numobj != 0:
        if args.numobj > flag:
            flag = 0
    
    if flag == 0:
#        print("no target was added!")
        a=1
        if not OBJECT_ONLY:
            outlists.append(file)
            write = targetdir + file + '.xml'
            with open(write, "w") as f:
                f.write(xmltodict.unparse(doc, pretty=True))
                
    else:
        outlists.append(file)
        write = targetdir + file + '.xml'
#        print("writing")
        with open(write, "w") as f:
            f.write(xmltodict.unparse(doc, pretty=True))
    #print(xmltodict.unparse(doc, pretty=True))
    

with open(trainfile, "w") as f: 
    if args.sample == 0 or int(args.sample) == 0:
        for trainval in outlists:
            f.write(trainval + '\n')
    else:
        for trainval in outlists[::int(args.sample)]:
            f.write(trainval + '\n')
import os
if args.numobj != 0:
    if os.path.isfile("num_train"+trainfile):
        with open("num_train"+trainfile, "a") as f: 
            f.write("number of objects: "+str(args.numobj) + '\n')
            f.write("training samples: "+str(len(outlists)) + '\n')
    else:
        with open("num_train"+trainfile, "w") as f: 
            f.write("number of objects: "+str(args.numobj) + '\n')
            f.write("training samples: "+str(len(outlists)) + '\n')
    
    
import subprocess

command = "cp -f "+trainfile+" data/VOCdevkit"+target+"/VOC2007/ImageSets/Main/"
subprocess.call(command, shell=True)


print("wrote out training data")

vals2 = pickle.load(open(valdir, "rb"))
vals=[]
if len(vals2)==81 and args.dsm:
    for i in range(21):
        vals.append(vals2[i])
elif len(vals2)==81:
    for n in pascal_classes:
        for i,cls in enumerate(coco_classes):
            if n == cls:
                print(cls)
                print(i)
                vals.append(vals2[i])                
else:
     vals = vals2

#with open(datasource) as f:
datasource = '/data2/lost+found/img/'+target+'_val/*'
files = sorted(glob.glob(datasource))
targetdir = 'output/'+target+'-test-labels-res101/'
subprocess.call("mkdir "+targetdir, shell=True)

trainvals = []
outlists = []
MAX_SAMPLE = 100000
for file in files:
     trainvals.append(os.path.basename(file)[:-4])
     
# copy results here.
results = vals

train_num=len(results[0])

for i,file in enumerate(trainvals[:train_num]):
    
    del doc['annotation']['object'] 
    doc['annotation']['object'] = copy.deepcopy(template)
        
    try:
        while len(doc['annotation']['object'])>0:
            temp = doc['annotation']['object'].pop()
    except:
        None
        
    flag = 0
    
    numbox=0
    for ncls, cls in enumerate(classes):
        bboxes=[]
        result = results[ncls][i]
        for out in result:
            if out[4] > THRESH and cls in trainclass: # confident
                a = temp
                a['bndbox']['xmin'] = int(np.floor(out[0]))
                a['bndbox']['ymin'] = int(np.floor(out[1]))
                a['bndbox']['xmax'] = int(np.floor(out[2]))
                a['bndbox']['ymax'] = int(np.floor(out[3]))
                a['name'] = cls

                # filter small and error bbox
                if min(out[2]-out[0],out[3]-out[1]) > 20:                    
                    if min(out[2]-out[0],out[3]-out[1]) < filterpix:
                        a['difficult'] = 1
                    if (out[2]-out[0]) > 100 and (out[3]-out[1]) > 100 and cls == 'person' and "jackson" in args.dataset:
                        break
                    doc['annotation']['object'].append(copy.deepcopy(a))
                    flag += 1
                    counter[ncls] +=1
                    bboxes.append(out[0:4])
                    numbox+=1                
                
                # monitor bbox
                
                box.append(copy.deepcopy(a))
                bsize = min(out[2]-out[0],out[3]-out[1])
                boxsize.append(bsize)                
                       
    nbox.append(flag)
    
    if args.numobj != 0:
        if args.numobj > flag:
            flag = 0
    
    if flag == 0:
#        print("no target was added!")
        a=1
        if not OBJECT_ONLY:
            outlists.append(file)
            write = targetdir + file + '.xml'
            with open(write, "w") as f:
                f.write(xmltodict.unparse(doc, pretty=True))
                
    else:
        outlists.append(file)
        write = targetdir + file + '.xml'
#        print("writing")
        with open(write, "w") as f:
            f.write(xmltodict.unparse(doc, pretty=True))
    #print(xmltodict.unparse(doc, pretty=True))
    

with open(testfile, "w") as f: 
    if args.sample == 0 or int(args.sample) == 0:
        for trainval in outlists:
            f.write(trainval + '\n')
    else:
        for trainval in outlists[::int(args.sample)]:
            f.write(trainval + '\n')
import os
if args.numobj != 0:
    if os.path.isfile("num_train"+testfile):
        with open("num_train"+testfile, "a") as f: 
            f.write("number of objects: "+str(args.numobj) + '\n')
            f.write("training samples: "+str(len(outlists)) + '\n')
    else:
        with open("num_train"+testfile, "w") as f: 
            f.write("number of objects: "+str(args.numobj) + '\n')
            f.write("training samples: "+str(len(outlists)) + '\n')

command = "cp -f "+testfile+" data/VOCdevkit"+target+"/VOC2007/ImageSets/Main/test_"+target+".txt"
subprocess.call(command, shell=True)
