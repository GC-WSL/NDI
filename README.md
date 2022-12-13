# NDI
The official code of "Absolute Wrong Makes Better: Boosting Weakly Supervised Object Detection via Negative Deterministic Information".

For more details, please refer to [http://arxiv.org/abs/2204.10068].

## Get Started
### Installation
```Shell
sh install.sh
```
### Data Preparation
Download the VOC2007/VOC2012 dataset and put them into the `./data` directory. For example:
```Shell
  ./data/VOC2007/                           
  ./data/VOC2007/Annotations
  ./data/VOC2007/JPEGImages
  ./data/VOC2007/ImageSets    
```
### Training
```Shell
CUDA_VISIBLE_DEVICES=0 python tools/train_net_step.py --dataset voc2007 --cfg configs/NDI-WSOD.py --bs 1 --nw 4 --iter_size 4
```
### Testing
```Shell
CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --dataset voc2007test --cfg configs/NDI-WSOD.py --load_ckpt $model_path
```
## Acknowledgment
Our detection code is built upon [PCL](https://github.com/ppengtang/pcl.pytorch) and [IM-CFB](https://github.com/BlazersForever/IM-CFB). We are very grateful to all the contributors to these codebases.
## TODO List:
-Adopt the two-stage training strategy for more precise detection results.

-NDI-based Weakly Supervised Semantic Segmentation (NDI-WSSS) is coming.
