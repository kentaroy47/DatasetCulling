# What is this?
This is the implementation of the paper:

**Dataset Culling: Towards efficient training of distillation based domain specific models.**

Kentaro Yoshioka, Edward Lee, Simon Wong, Mark Horowitz (Stanford)

https://arxiv.org/abs/1902.00173

We aim to speed up the training of domain-specific models **50x** by Dataset Culling.
The idea behind this is simple: for training with domain-specific data, lots of easy data do not contribute to training. 
We simply *cull* those *easy-to-classify* data out in the proposed pipeline, gaining significant training speedups without accuracy loss.
Interestingly, for some data, we even find some accuracy improvements by Dataset Culling.

## Results..
![Results](https://github.com/kentaroy47/DatasetCulling/figure-of-dataset.png)


Dataset Culling conducts three operations in its pipeline.  
1. Reduction of the dataset size (~50x) by confidence loss.
2. Reduction of the dataset size (~6x) by precision loss.
3. Optimization of the image resolution (optResolution).



## Requirements
Python3
Pytorch 0.4.0 (major code fix required for pytorch 1.0..)
GPU enviroment recommended.

**The faster R-CNN implementation is largely based on jwyang's repo.**
Take a look at his readme if you get stuck during install..!
https://github.com/jwyang/faster-rcnn.pytorch

FYI: It might be interesting to look at my previous project, **training domain specific models** if you have interests in domain specific models itself.
Github: https://github.com/kentaroy47/training-domain-specific-models
arXiv: https://arxiv.org/abs/1811.02689

Don't hesitate to post issues or PRs if you find bugs. Thx.

## Instalation
1. Clone this repo.


## Progress
Upload initial commits (1/30/2019)
Update readme (2/10/2019)
Update initial models and dataset (2/10/2019)
Enable optResolution in pipeline (TBD. next week maybe?)

```py
python dataset-culling.py
```
