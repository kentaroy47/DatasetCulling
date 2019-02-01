# -*- coding: utf-8 -*-

"""
script for setting up pascal-like datasets
copyright:kentaroy47
10/2/2018
"""

import os
import subprocess
import argparse

parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
parser.add_argument('--dataset', dest='dataset',
                      help='training dataset', type=str)
parser.add_argument('--numobj', dest='numobj',
                      help='training dataset', type=str, default='0')
parser.add_argument('--sample', dest='sample',
                      help='training dataset', type=str, default='0')
parser.add_argument('--filterpix', dest='filterpix',
                      help='filter pixel size', type=str, default=20)
parser.add_argument('--net', dest='net',
                      help='target network', type=str, default="res101")
parser.add_argument('--dsm', dest='dsm',
                      help='filter pixel size', type=bool, default=False)
args = parser.parse_args()
target=args.dataset
filterpix = str(args.filterpix)

# make dir in datasets
datasetname = "VOCdevkit"+target
command="mkdir data/"+datasetname
subprocess.call(command, shell=True)
command="mkdir data/"+datasetname+"/VOC2007/"
subprocess.call(command, shell=True)
command="mkdir data/"+datasetname+"/VOC2007/JPEGImages"
subprocess.call(command, shell=True)
command="mkdir data/"+datasetname+"/VOC2007/Annotations"
subprocess.call(command, shell=True)
command="mkdir data/"+datasetname+"/VOC2007/ImageSets"
subprocess.call(command, shell=True)
command="mkdir data/"+datasetname+"/VOC2007/ImageSets/Main"
subprocess.call(command, shell=True)

# generate xml
args.numobj = str(args.numobj)
if args.dsm:
    command="python xml_makelabels_domain.py --dataset "+target+" --numobj "+args.numobj + \
    " --sample "+args.sample+" --filterpix "+filterpix+" --net "+args.net+" --dsm "+str(args.dsm)
else:
    command="python xml_makelabels_domain.py --dataset "+target+" --numobj "+args.numobj + \
    " --sample "+args.sample+" --filterpix "+filterpix+" --net "+args.net
print("com:", command)

subprocess.call(command, shell=True)

# make link
command="cp -rf output/"+target+"-train-labels-"+args.net+"/* data/"+datasetname+"/VOC2007/Annotations/"
subprocess.call(command, shell=True)
command="cp -rf output/"+target+"-test-labels-"+args.net+"/* data/"+datasetname+"/VOC2007/Annotations/"
subprocess.call(command, shell=True)
command="cp -rf /data2/lost+found/img/"+target+"_train/* data/"+datasetname+"/VOC2007/JPEGImages/"
subprocess.call(command, shell=True)
command="cp -rf /data2/lost+found/img/"+target+"_val/* data/"+datasetname+"/VOC2007/JPEGImages/"
subprocess.call(command, shell=True)

# copy text file to Main
command="cp trainval/trainval_"+target+".txt data/"+datasetname+"/VOC2007/ImageSets/Main/"
subprocess.call(command, shell=True)

# make models and copy
command="mkdir models/res18/pascal_voc_"+target
subprocess.call(command, shell=True)
command="cp models/faster_rcnn_500_40_625.pth models/res18/pascal_voc_"+target
subprocess.call(command, shell=True)
#command="mkdir models/squeeze/pascal_voc_"+target
#subprocess.call(command, shell=True)
#command="cp models/squeeze/faster_rcnn_500_40_625.pth models/squeeze/pascal_voc_"+target
#subprocess.call(command, shell=True)