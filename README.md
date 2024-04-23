# MS-FAST: Multi-Scale Frequency Aware Spiking Transformer
Our codes are based on the official imagenet example by PyTorch, pytorch-image-models by Ross Wightman and SpikingJelly by Wei Fang.
<p align="center">
<img src="/MS-FAST_Code/imgs/model1.png" width="">
</p>

## Introduction
Evaluated across multiple datasets, MS-FAST achieves SOTA Top-1 classification accuracy, compresses model size without compromising performance.

## Requirements
Ltimm==0.5.4
cupy==10.3.1
pytorch==1.10.0+cu111
spikingjelly==0.0.0.0.12
pyyaml

## Data Preparation
data prepare: Tiny_Imagenet with the following folder structure, you can extract Tiny_Imagenet by this [script](http://cs231n.stanford.edu/tiny-imagenet-200.zip)。
```
│Tiny_Imagenet/
├──train/
│  ├── n01443537
│  │   ├── n01443537_0.JPEG
│  │   ├── n01443537_1.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01443537
│  │   ├── val_68.JPEG
│  │   ├── val_611.JPEG
│  │   ├── ......
│  ├── ......
```
### Training  on Tiny-ImageNet
Setting hyper-parameters in Tiny_Imagenet.yml

```
cd Tiny_Imagenet
python -m torch.distributed.launch --nproc_per_node=4 train.py
```

### Testing Tiny-ImageNet Val data 
```
cd Tiny_ImageNet
python test.py
```

### Training  on cifar10
Setting hyper-parameters in cifar10.yml
```
cd cifar10
python train.py
```
### Training  on cifar100
Setting hyper-parameters in cifar100.yml
```
cd cifar100
python train.py
```
### Training  on cifar10DVS
```
cd cifar10dvs
python train.py
```
### Training  on DVS128-GESTURE
```
cd DVS128-GESTURE
python train.py
```
### Training  on UCF101-DVS
```
cd UCF101-DVS
python train.py
```