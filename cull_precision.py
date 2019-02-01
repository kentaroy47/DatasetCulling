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
topx = (args.topx)

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
       # obtain predictions of teacher
        with open("output/"+dataset+"-confusion-res18.pkl", "rb") as f:
           student = pickle.load(f)
        with open("output/"+dataset+"-res101-"+topx+".pkl", "rb") as f:
           teacher = pickle.load(f)
       # get image list
        temp = sorted(glob.glob("images/confusion-"+dataset+"/*"))
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
        
        # derive ltrain
               # get list of ltrain.
        ltrain = []
        confusion = []
        
        # compute confusion and precision
        ltrain, monap = compute_ltrain_top(student2, teacher2, dataset)
                   
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

# dump top_key for confirmation
subprocess.call("mkdir keys", shell=True)
with open("keys/"+str(topx)+"-ltrain-"+dataset+".pkl", 'wb') as f:
    pickle.dump(top_key, f, pickle.HIGHEST_PROTOCOL)

# copy images to a folder and keep the answers in a list.
dest = "images/"+str(topx)+"-"+dataset+"_train"
import os
if not os.path.isdir(dest):
    subprocess.call("mkdir "+dest, shell=True)
#    if not os.path.isfile(dest+"/*.jpg"):
print("number of images..", len(finalimagelist2))
for i, image in enumerate(finalimagelist2):
    num = '%04d' % i
    subprocess.call("cp "+image+" "+dest+"/ltrain"+str(num)+".jpg", shell=True)
    
# save the final list of answers in pickle file
with open("output/baseline/"+str(topx)+"-"+dataset+"train-res101.pkl", 'wb') as f:
    pickle.dump(finalpredictions, f, pickle.HIGHEST_PROTOCOL)

print("==== conversion completed! ====")

