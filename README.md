# What is this?
This is the implementation of the paper:

**Dataset Culling: Towards efficient training of distillation based domain specific models.**

Kentaro Yoshioka, Edward Lee, Simon Wong, Mark Horowitz (Stanford)

https://arxiv.org/abs/1902.00173

## What is Dataset Culling?
By training a domain specific model (say, a specific model for each traffic camera) we can get high accuracy even with a small model.
However, **training of such models can be quite expensive.**

So, we aim to speed up the training of domain-specific models **50x** by Dataset Culling!

The idea behind this is simple: for training with domain-specific data, lots of easy data do not contribute to training. 
We simply *cull* those *easy-to-classify* data out in the proposed pipeline, gaining significant training speedups without accuracy loss.

Interestingly, for some data, we even find some accuracy improvements by Dataset Culling.

## Results of Dataset Culling
![Results](https://github.com/kentaroy47/DatasetCulling/blob/master/figure-of-dataset.png)
Here are results of three scenes (more results in paper).

Training a domain specific model with Dataset Culling, we can get upto 18x computation efficiency compared to large teacher models.

Moreover, the accuracy is almost the same.

## what does this repo do?
This repo lets you try the pipeline with some domain specific data (traffic camera from YoutubeLive) and some pretrained models (COCO trained res18, res101 based FR-CNNs.)

![Pipeline](https://github.com/kentaroy47/DatasetCulling/blob/master/fig1.jpg)
Dataset Culling conducts three operations in its pipeline.  
1. Reduction of the dataset size (~50x) by confidence loss.
2. Reduction of the dataset size (~6x) by precision loss.
3. Optimization of the image resolution (optResolution).

FYI: It might be interesting to look at my previous project, **training domain specific models** if you have interests in domain specific models itself.

Github: https://github.com/kentaroy47/training-domain-specific-models

arXiv: https://arxiv.org/abs/1811.02689

# Setting up Dataset Culling enviroment

## Requirements
Python3
Pytorch 0.4.0 (major code fix required for pytorch 1.0..)
GPU enviroment recommended.

**The faster R-CNN implementation is largely based on jwyang's repo.**
Take a look at his readme if you get stuck during install..!

https://github.com/jwyang/faster-rcnn.pytorch

Don't hesitate to post issues or PRs if you find bugs. Thx.

## Instalation
1. Clone this repo.

```
git clone https://github.com/kentaroy47/DatasetCulling.git
cd DatasetCulling
```

2. Download pretrained student and teacher models.

```
mkdir models/
cd models

# resnet101 teacher. Note that this is in caffe normalization..
Download.. https://www.dropbox.com/s/dpq6qv0efspelr3/faster_rcnn_1_10_9771.pth?dl=0

# resnet18 student
Download.. https://drive.google.com/file/d/1KvrBMDYD5QtccjWbeKsLDZj6gBYRwVum/view?usp=sharing
cd ..
```

3. Download example Dataset
I provide an short video from youtubelive to get started.
If you are further interested, try downloading more videos from youtube and apply DatasetCulling!

```
Download..  https://drive.google.com/file/d/1TnNcOpLqJzBwqYfRs7Oh8WRHOlk6I5ET/view?usp=sharing

# extract in the repo dir.
tar -zxvf images.tar.gz

```

4. Apply Dataset Culling and train student models!
Everything is in the script. The dataset will be constructed inside directory.

```py
# Construct dataset with Dataset Culling. This takes about ten minutes.
python dataset-culling.py

# Train wihout Dataset Culling. This takes two hours!!
python dataset-culling.py --datasetculling False

```

5. Eval results. 


### Repo Progress
Upload initial commits (1/30/2019)
Update readme (2/10/2019)
Update initial models and dataset (2/10/2019)
Enable optResolution in pipeline (TBD. next week maybe?)
