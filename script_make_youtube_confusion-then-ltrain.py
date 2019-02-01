#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 15:33:23 2018

@author: ken

v1.. just count the number of objects not regarding the class.
v2.. pick the image that has most object in class, respectively.
"""

import subprocess
import pickle
import glob
import numpy as np
from util import *
import argparse

parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
parser.add_argument('--topx', dest='topx',
                      default=512)
parser.add_argument('--dataset', dest='dataset',
                      default="kentucky")
parser.add_argument('--subsample', dest='subsample',
                      default=1, type=int)
args = parser.parse_args()
topx = int(args.topx)

# get keys over threshold
def pick_keys(count_data, threshold):
    keylist = []
    for i, data in enumerate(count_data):
        if data > threshold:
            keylist.append(i)
    return(keylist)

# prepare lists
answerlist = []
finalimagelist = []
predictionlist = []
countperson = []
countcar = []
finalltrain = []
finalconfusion = []

# loop through the dataset
for x in [1]:
    for y in [1]:
        imagelist = []

        
        dataset = args.dataset
        # read pickle
        with open("output/pascal_voc_"+dataset+"-res18.pkl", "rb") as f:
           student = pickle.load(f)       
       # obtain predictions of teacher
        with open("output/baseline/"+dataset+"train-res101.pkl", "rb") as f:
           teacher = pickle.load(f)
       # get image list
        temp = sorted(glob.glob("/data2/lost+found/img/"+dataset+"_train/*"))
        for x in temp:
            if "val" not in x:
                imagelist.append(x)
        
       # convert teacher's predictions to pascal format.
       # filter small -> class==20 -> make car class.
        teacher2 = coco2pascal(teacher)
        student2 = coco2pascal(student) # for filtering
        
        # count objects
        countperson.extend(count_image(student2, 15))
        countcar.extend(count_image(student2, 7))
        
        # for coral
        if dataset == "coral":
           teacher2[7] = teacher2[15]
           student2[7] = student2[15]

        # derive ltrain
               # get list of ltrain.
        ltrain = []
        confusion = []
        for tdata, sdata, tdata2, sdata2 in zip(teacher2[7], student2[7], teacher2[15], student2[15]):
           # compute confusion
           aps = (compute_confusion(sdata))
           aps2 = (compute_confusion(sdata2))
           if aps < 1:
               val1 = 1
           else:
               val1 = 1/aps
           if aps2 < 1:
               val2 = 1
           else:
               val2 = 1/aps2
           ltrainval = 1/((val1+val2)/2)
           confusion.append(ltrainval)
           
           # compute ltrain
           aps = (compute_ltrain(tdata, sdata))
           aps2 = (compute_ltrain(tdata2, sdata2))
           if aps[0] < 1:
               val1 = 1
           else:
               val1 = 1/aps[0]
           if aps2[0] < 1:
               val2 = 1
           else:
               val2 = 1/aps2[0]
           ltrainval = 1/((val1+val2)/2)
           ltrain.append(ltrainval)
           
       # concat ltrain in long list
       # ltrain can be a bit long.. handle exceptions
        if len(ltrain)!=len(imagelist):
           finalltrain.extend(ltrain[:-1])
        else:
           finalltrain.extend(ltrain)

        if len(confusion)!=len(imagelist):
           finalconfusion.extend(ltrain[:-1])
        else:
           finalconfusion.extend(ltrain)
       
        # make a image list as well
        finalimagelist.extend(imagelist)
       
       # make prediction list
        teacher = conv20class(teacher)
        if len(teacher[7])!=len(imagelist):
           predictionlist = add_answer_cut(predictionlist, teacher)
        else:
           predictionlist = add_answer(predictionlist, teacher)

# confirm length
print("prediction length:", len(predictionlist[1]))
print("image length:", len(finalimagelist))

# pick in confusion first
keylist = sorted(range(len(finalconfusion)), key=lambda i: finalconfusion[i], reverse=True)[:int(topx*4)]
# mask ltrain
for i, data in enumerate(finalltrain):
    if i not in keylist:
        finalltrain[i] = 0

# pick in ltrain second
top_key = sorted(range(len(finalltrain)), key=lambda i: finalltrain[i], reverse=True)[:int(topx)]

# make image list
finalimagelist2 = []
finalpredictions = []
for key in top_key:
    finalimagelist2.append(finalimagelist[key])
finalimagelist = finalimagelist2

# make predictions
finalpredictions = choose_answer_limit(predictionlist, range(0,21), top_key)

# copy images to a folder and keep the answers in a list.
dest = "/data2/lost+found/img/"+str(topx)+"-confusion-ltrain-"+dataset+"_train"
import os
if not os.path.isdir(dest):
    subprocess.call("mkdir "+dest, shell=True)
#    if not os.path.isfile(dest+"/*.jpg"):
print("number of images..", len(finalimagelist2))
for i, image in enumerate(finalimagelist2):
    num = '%04d' % i
    subprocess.call("cp "+image+" "+dest+"/ltrain"+str(num)+".jpg", shell=True)
    
# save the final list of answers in pickle file
with open("output/baseline/"+str(topx)+"-confusion-ltrain-"+dataset+"train-res101.pkl", 'wb') as f:
    pickle.dump(finalpredictions, f, pickle.HIGHEST_PROTOCOL)

print("==== conversion completed! ====")

import subprocess
command='python script_makevoc.py --dataset '+str(topx)+'-confusion-ltrain-'+dataset
subprocess.call(command, shell=True)
print("==== dataset make completed! ====")