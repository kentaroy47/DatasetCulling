# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.pascal_voc import pascal_voc
from datasets.coco import coco
from datasets.imagenet import imagenet
from datasets.vg import vg

import numpy as np


#__sets[name] = (lambda split=split, year=year: pascal_voc(split, 2007))

# Set up voc_<year>_<split>
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval_taipei', 'trainval_shanghai',  'trainval_shanghai80', 'val_shanghai', 'val_jackson', 'trainval_jackson80', 'trainval_jackson100', 'trainval_jackson', 'trainval_jacksonallobject', 'trainval', 'test', 'trainval_res101', 'trainval_coral', 'trainval_coralpascal', 'trainval_coralcoco', 'trainval_coralcocob', 'trainvalb', 'trainval_coralcoco10', 'trainval_coral80', 'trainval_coralcoco30','trainval_coralcoco50', 'trainval_coralcoco100', 'trainval_coralcoco250', 'trainval_coralcoco500', 'trainval_coralcoco1000', 'trainval_coralcoco4000', 'trainval_coralcoco8000', 'trainval_coralcocodif']:
    name = 'voc_{}_{}'.format(year, split)
    if 'trainval' in split:
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))
    elif 'val' in split:
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))
#        print("split:",split)
#    elif split =='trainval_res101':
#        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year, res101=True))
#    elif split =='trainval_coral':
#        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year, coral=True))
#    elif split =='trainval_shanghai':
#        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year, shanghai=True))
#    elif split =='trainval_shanghai80':
#        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year, shanghai=True, cococlass=True))
#    elif split =='val_shanghai':
#        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year, shanghai=True))
#    elif split =='trainval_taipei':
#        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year, taipei=True))
#    elif split =='trainval_coralpascal':
#        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year, coralpascal=True))
#    elif split =='trainval_coralcoco':
#        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year, coralcoco=True))
#    elif split =='trainval_jackson':
#        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year, jackson=True))
#    elif split =='trainval_jacksonallobject':
#        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year, jackson=True))
#    elif split =='val_jackson':
#        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year, jackson=True))
#    elif split =='trainval_jackson80':
#        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year, jackson=True, cococlass=True))
#    elif split =='trainval_jackson100':
#        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year, jackson=True))
#    elif split =='trainval_coralcoco10':
#        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year, coralcoco=True))
#    elif split =='trainval_coral80':
#        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year, coral=True, cococlass=True))
#    elif split =='trainval_coralcoco30':
#        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year, coralcoco=True))
#    elif split =='trainval_coralcoco50':
#        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year, coralcoco=True))
#    elif split =='trainval_coralcoco100':
#        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year, coralcoco=True))
#    elif split =='trainval_coralcoco250':
#        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year, coralcoco=True))
#    elif split =='trainval_coralcoco500':
#        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year, coralcoco=True, datanum=500))
#    elif split =='trainval_coralcoco1000':
#        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year, coralcoco=True, datanum=1000))
#    elif split =='trainval_coralcoco4000':
#        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year, coralcoco=True))
#    elif split =='trainval_coralcoco8000':
#        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year, coralcoco=True))
#    elif split =='trainval_coralcocodif':
#        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year, coralcoco=True, dif=True))  
#    elif split =='trainval_coralcocob':
#        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year, coralcoco=True, binary=True))
#    else:
#        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))
        

# Set up coco_2014_<split>
for year in ['2014']:
  for split in ['train', 'val']:#, 'minival', 'valminusminival', 'trainval']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2014_cap_<split>
for year in ['2014']:
  for split in ['train', 'val']:#, 'capval', 'valminuscapval', 'trainval']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
  for split in ['test', 'test-dev']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up vg_<split>
# for version in ['1600-400-20']:
#     for split in ['minitrain', 'train', 'minival', 'val', 'test']:
#         name = 'vg_{}_{}'.format(version,split)
#         __sets[name] = (lambda split=split, version=version: vg(version, split))
for version in ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']:
    for split in ['minitrain', 'smalltrain', 'train', 'minival', 'smallval', 'val', 'test']:
        name = 'vg_{}_{}'.format(version,split)
        __sets[name] = (lambda split=split, version=version: vg(version, split))
        
# set up image net.
for split in ['train', 'val', 'val1', 'val2', 'test']:
    name = 'imagenet_{}'.format(split)
    devkit_path = 'data/imagenet/ILSVRC/devkit'
    data_path = 'data/imagenet/ILSVRC'
    __sets[name] = (lambda split=split, devkit_path=devkit_path, data_path=data_path: imagenet(split,devkit_path,data_path))

def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    print('Unknown dataset: {}'.format(name))
    split=name[9:]
    print(split)
    year="2007"
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
