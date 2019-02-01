# -*- coding: utf-8 -*-

import subprocess
import os

# how much to cull the dataset size to.
topx = str(256)
# target dataset
target = "jackson2"

# make dataset.
    # make student predictions
if not (os.path.isfile("output/"+target+"-res18.pkl")):
    com = "python demo-and-eval-save-student.py --target "+target+" --net res18 --image_dir images/"+target+"_train --coco True --cuda --dataset pascal_voc --checksession 500 --checkepoch 40 --checkpoint 625"
    subprocess.call(com, shell=True)

    # culling by confusion.
com = "python cull_confusion.py --dataset "+target+" --topx "+topx
subprocess.call(com, shell=True)

    # make teacher predictions.
    
    
    # culling by precision.

# compile difficult dataset.

# train model.

# test model.