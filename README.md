### Repo Progress
~~Upload initial commits~~ (1/30/2019)

~~Update readme~~ (2/10/2019)

~~Update initial models and dataset~~ (2/10/2019)

~~Support Pytorch1.0~~ (2/17/2019)

- Enable optResolution in pipeline (working..)

- Clean up install files. 

- Fixed the download link for the models.

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

We can get compared to large teacher models: 

1) upto 18x computation efficiency!

2) Similar or better detection accuracy!

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
`Python3`

`Pytorch 0.4.0` 

If you want to use `Pytorch 1.0`, plz go to the `Pytorch 1.0 branch`.

GPU enviroment required. can add CPU options but not scheduled..

Don't hesitate to post issues or PRs if you find bugs. Thx.

## Instalation
1. Clone this repo and pip dependencies.

```
git clone https://github.com/kentaroy47/DatasetCulling.git
cd DatasetCulling
pip install -r requirements.txt
```

2. Download pretrained student and teacher models + image files to get started.

```
wget https://www.dropbox.com/s/ew47jhdu67bdocf/files.tar.gz

# extract in the repo dir.
tar -zxvf files.tar.gz

```

3. Compile Cython scripts.
```
cd lib
sh make.sh
```

As pointed out by [ruotianluo/pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn), choose the right `-arch` in `make.sh` file, to compile the cuda code:

  | GPU model  | Architecture |
  | ------------- | ------------- |
  | TitanX (Maxwell/Pascal) | sm_52 |
  | GTX 960M | sm_50 |
  | GTX 1080 (Ti) | sm_61 |
  | Grid K520 (AWS g2.2xlarge) | sm_30 |
  | Tesla K80 (AWS p2.xlarge) | sm_37 |

**The faster R-CNN implementation is largely based on jwyang's repo and require complie of cython scripts.**

Cython parts must be compiled using lib/make.sh.
Please look at jwyang's readme for the details.

https://github.com/jwyang/faster-rcnn.pytorch

4. Apply Dataset Culling and train student models!
Everything is in the script.
 
The dataset will be constructed inside directory.

```sh
# Construct dataset with Dataset Culling. This takes about 15 minutes with 1080Ti.
# The training is done with horizontal flipped data-argumentation.
python dataset-culling.py

# change the number of training sample like this. default is 256.
python dataset-culling.py --topx 64

# Train wihout Dataset Culling. This will take about >3 hours with 1080Ti.
python dataset-culling.py --topx 3600

```

5. Eval results. 
The test is also done in dataset-culling.

You can just do test by..

```sh
python dataset-culling.py --notrain
```

### Try cleaning up data/cache or output/ folder if training does not start.

# Contarct
Ken Yoshioka (kyoshioka47@gmail.com)


