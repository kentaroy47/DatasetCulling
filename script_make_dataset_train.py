# -*- coding: utf-8 -*-

import subprocess
import os

# how much to cull the dataset size to.
topx = str(64)
# target dataset
target = "jackson2"

# make dataset.
    # make student predictions
if not (os.path.isfile("output/"+target+"-res18-"+topx+".pkl")):
    com = "python demo-and-eval-save-student.py --show True --vis --topx "+topx+" --target "+target+" --net res18 --image_dir images/"+target+"_train --coco True --cuda --dataset pascal_voc --checksession 500 --checkepoch 40 --checkpoint 625"
    subprocess.call(com, shell=True)

    # culling by confusion.
com = "python cull_confusion.py --dataset "+target+" --topx "+topx
subprocess.call(com, shell=True)

    # make teacher predictions.
if not (os.path.isfile("output/"+target+"-res101-"+topx+".pkl")):
    com = "python demo-and-eval-save-teacher.py --topx "+topx+" --target "+target+" --net res101 --image_dir images/confusion-"+target+" --coco True --cuda --dataset pascal_voc --checksession 1 --checkepoch 10 --checkpoint 9771"
    subprocess.call(com, shell=True)
    
    # culling by precision.
com = "python cull_precision.py --dataset "+target+" --topx "+topx
subprocess.call(com, shell=True)

# compile difficult dataset.
com = "python make_dataset.py --dataset "+target+" --topx "+topx
subprocess.call(com, shell=True)

# train DSM model.

# test DSM model.