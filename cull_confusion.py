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
                      default=64)
parser.add_argument('--dataset', dest='dataset',
                      default="kentucky")
parser.add_argument('--subsample', dest='subsample',
                      default=1, type=int)
parser.add_argument('--Q', dest='Q',
                      default=4, type=int)
args = parser.parse_args()

# the we reduce the images until topx*4
topx = int(args.topx)*4

Q = int(args.Q)


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

# loop through the dataset
for x in [1]:
    for y in [1]:
        imagelist = []

        
        dataset = args.dataset
        # read pickle
        with open("output/"+dataset+"-res18.pkl", "rb") as f:
           student = pickle.load(f)       
       # obtain predictions of teacher
       # get image list
        temp = sorted(glob.glob("images/"+dataset+"_train/*"))
        for x in temp:
            if "val" not in x:
                imagelist.append(x)
        
       # convert teacher's predictions to pascal format.
       # filter small -> class==20 -> make car class.
        student2 = coco2pascal(student) # for filtering
        
        # count objects
        countperson.extend(count_image(student2, 15))
        countcar.extend(count_image(student2, 7))
        
        # for coral
        if dataset == "coral":
           student2[7] = student2[15]
           
        # derive ltrain
        # get list of ltrain.
        ltrain = []
        ltrain, outap = compute_confusion_top(student2, dataset, Q)
       
       # concat ltrain in long list
       # ltrain can be a bit long.. handle exceptions
        if len(ltrain)!=len(imagelist):
           finalltrain.extend(ltrain[:-1])
        else:
           finalltrain.extend(ltrain)

       
        # make a image list as well
        finalimagelist.extend(imagelist)
       
# confirm length
print("image length:", len(finalimagelist))

# pick in ltrain
top_key = sorted(range(len(finalltrain)), key=lambda i: finalltrain[i], reverse=True)[:int(topx)]

# make image list
finalimagelist2 = []
finalpredictions = []
for key in top_key:
    finalimagelist2.append(finalimagelist[key])

# make new dataset.
subprocess.call("mkdir images/confusion-"+dataset, shell=True)
subprocess.call("rm images/confusion-"+dataset+"/*", shell = True)

for i, image in enumerate(finalimagelist):
    if i in top_key:
        subprocess.call("cp "+image+" images/confusion-"+dataset, shell =True)
        
print("finished culling dataset by confusion")
