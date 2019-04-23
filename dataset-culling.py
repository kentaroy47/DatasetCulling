#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 21:36:21 2019

@author: ken
"""

# -*- coding: utf-8 -*-

import subprocess
import os
import argparse
import time
parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
parser.add_argument('--topx', dest='topx',
                      default=211)
parser.add_argument('--dataset', dest='dataset',
                      default="jackson2")
parser.add_argument('--nodatasetculling', action='store_false')
parser.add_argument('--notrain', action='store_false')
parser.add_argument('--notest', action='store_false')
args = parser.parse_args()
topx = str(args.topx)

# print stuff
start =time.time()
print("Dataset culling is :", args.nodatasetculling)
print("Culling the dataset to:", args.topx)
print("Target dataset:", args.dataset)

# target dataset
target = args.dataset

print("make dirs")
subprocess.call("mkdir trainval",shell=True)
subprocess.call("mkdir data",shell=True)
subprocess.call("mkdir models",shell=True)
subprocess.call("mkdir trainval",shell=True)
subprocess.call("mkdir images",shell=True)

# make dataset.
# make student predictions
print("making predictions using student model..")
if not (os.path.isfile("output/"+target+"-res18.pkl")):
    com = "python demo-and-eval-save-student.py --vis --topx "+topx+" --target "+target+" --net res18 --image_dir images/"+target+"_train --coco True --cuda --dataset pascal_voc --checksession 500 --checkepoch 40 --checkpoint 625"
    subprocess.call(com, shell=True)

# culling by confusion.
print("culling by confusion..")
com = "python cull_confusion.py --dataset "+target+" --topx "+topx
subprocess.call(com, shell=True)

# make teacher predictions.
print("making teacher predictions..")
if not (os.path.isfile("output/"+target+"-res101-"+topx+".pkl")):
    com = "python demo-and-eval-save-teacher.py --topx "+topx+" --target "+target+" --net res101 --image_dir images/confusion-"+target+" --coco True --cuda --dataset pascal_voc --checksession 1 --checkepoch 10 --checkpoint 9771"
    subprocess.call(com, shell=True)

# culling by precision.
print("culling by precision..")
com = "python cull_precision.py --dataset "+target+" --topx "+topx
subprocess.call(com, shell=True)

# compile difficult dataset.
print("compile difficult dataset..")
com = "python script_makevoc.py --dataset "+target+" --topx "+topx
subprocess.call(com, shell=True)

# train DSM model.
com = "python trainval_net_ds_savemod.py --dataset pascal_voc_"+topx+"-"+target+" --cuda --epoch 20 --net res18 --r True --checksession 500 --checkepoch 40 --checkpoint 625"
if args.notrain:
	subprocess.call(com, shell=True)

# test DSM model.
com = "python demo-and-eval-save.py --dataset pascal_voc_"+topx+"-"+target+" --cuda --net res18 --checksession 1 --checkepoch 20 --image_dir images/jackson2_val --truth output/baseline/jackson2val-res101.pkl"
if args.notest:
	subprocess.call(com, shell=True)

end = time.time()
print("total time taken:", end-start)

