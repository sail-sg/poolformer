# PoolFormer: [MetaFormer Is Actually What You Need for Vision](https://arxiv.org/abs/2111.11418) (CVPR 2022 Oral)

<p align="center">
<a href="https://arxiv.org/abs/2111.11418" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2111.11418-b31b1b.svg?style=flat" /></a>
<a href="https://huggingface.co/spaces/akhaliq/poolformer" alt="Hugging Face Spaces">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue" /></a>
<a href="https://colab.research.google.com/github/sail-sg/poolformer/blob/main/misc/poolformer_demo.ipynb" alt="Colab">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" /></a>
</p>


---
:fire: :fire: Our follow-up work "[MetaFormer Baselines for Vision](https://arxiv.org/abs/2210.13452)" (code: [metaformer](https://github.com/sail-sg/metaformer)) introduces more MetaFormer baselines including
+ **IdentityFormer** with token mixer of identity mapping surprisingly achieve >80% accuracy.
+ **RandFormer** achieves >81% accuracy by random token mixing, demonstrating MetaForemr works well with arbitrary token mixers.
+ **ConvFormer** with token mixer of separable convolution significantly outperforms ConvNeXt by large margin.
+ **CAFormer** with token mixers of separable convolutions and vanilla self-attention sets new record on ImageNet-1K.

---


This is a PyTorch implementation of **PoolFormer** proposed by our paper "[MetaFormer Is Actually What You Need for Vision](https://arxiv.org/abs/2111.11418)" (CVPR 2022 Oral).


**Note**: Instead of designing complicated token mixer to achieve SOTA performance, the target of this work is to demonstrate the competence of Transformer models largely stem from the general architecture MetaFormer. Pooling/PoolFormer are just the tools to support our claim. 

![MetaFormer](https://user-images.githubusercontent.com/49296856/177275244-13412754-3d49-43ef-a8bd-17c0874c02c1.png)
Figure 1: **MetaFormer and performance of MetaFormer-based models on ImageNet-1K validation set.** 
We argue that the competence of Transformer/MLP-like models primarily stem from the general architecture MetaFormer instead of the equipped specific token mixers.
To demonstrate this, we exploit an embarrassingly simple non-parametric operator, pooling, to conduct extremely basic token mixing. 
Surprisingly, the resulted model PoolFormer consistently outperforms the DeiT and ResMLP as shown in (b), which well supports that MetaFormer is actually what we need to achieve competitive performance. RSB-ResNet in (b) means the results are from “ResNet Strikes Back” where ResNet is trained with improved training procedure for 300 epochs.


<p align="center">
  <img src="https://user-images.githubusercontent.com/49296856/205430159-54bba545-520e-4ab8-8a77-278d90b54ec4.png" alt="PoolFormer"/>
</p>

Figure 2: (a) **The overall framework of PoolFormer.** (b) **The architecture of PoolFormer block.** Compared with Transformer block, it replaces attention with an extremely simple non-parametric operator, pooling, to conduct only basic token mixing.

## Bibtex
```
@inproceedings{yu2022metaformer,
  title={Metaformer is actually what you need for vision},
  author={Yu, Weihao and Luo, Mi and Zhou, Pan and Si, Chenyang and Zhou, Yichen and Wang, Xinchao and Feng, Jiashi and Yan, Shuicheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10819--10829},
  year={2022}
}
```

**Detection and instance segmentation on COCO** configs and trained models are [here](detection/).

**Semantic segmentation on ADE20K** configs and trained models are [here](segmentation/).

The code to visualize Grad-CAM activation maps of PoolFomer, DeiT, ResMLP, ResNet and Swin are [here](misc/cam_image.py).

The code to measure MACs are [here](misc/mac_count_with_fvcore.py).

## Image Classification
### 1. Requirements

torch>=1.7.0; torchvision>=0.8.0; pyyaml; [apex-amp](https://github.com/NVIDIA/apex) (if you want to use fp16); [timm](https://github.com/rwightman/pytorch-image-models) (`pip install git+https://github.com/rwightman/pytorch-image-models.git@9d6aad44f8fd32e89e5cca503efe3ada5071cc2a`)

data prepare: ImageNet with the following folder structure, you can extract ImageNet by this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).

```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```



### 2. PoolFormer Models

| Model    |  #Params | Image resolution | #MACs* | Top1 Acc| Download | 
| :---     |   :---:    |  :---: |  :---: |  :---:  |  :---:  |
| poolformer_s12  |    12M     |   224  |  1.8G |  77.2  | [here](https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_s12.pth.tar) |
| poolformer_s24 |   21M     |   224 | 3.4G | 80.3  | [here](https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_s24.pth.tar) |
| poolformer_s36  |   31M     |   224 | 5.0G | 81.4  | [here](https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_s36.pth.tar) |
| poolformer_m36 |   56M     |   224 | 8.8G | 82.1  | [here](https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_m36.pth.tar) |
| poolformer_m48  |   73M     |   224 | 11.6G | 82.5  | [here](https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_m48.pth.tar) | 


All the pretrained models can also be downloaded by [BaiDu Yun](https://pan.baidu.com/s/1HSaJtxgCkUlawurQLq87wQ) (password: esac). * For convenient comparison with future models, we update the numbers of MACs counted by [fvcore](https://github.com/facebookresearch/fvcore) library ([example code](misc/mac_count_with_fvcore.py)) which are also reported in the [new arXiv version](https://arxiv.org/abs/2111.11418).


#### Web Demo

Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the Web Demo: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/poolformer)



#### Usage
We also provide a Colab notebook which run the steps to perform inference with poolformer: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sail-sg/poolformer/blob/main/misc/poolformer_demo.ipynb)


### 3. Validation

To evaluate our PoolFormer models, run:

```bash
MODEL=poolformer_s12 # poolformer_{s12, s24, s36, m36, m48}
python3 validate.py /path/to/imagenet  --model $MODEL -b 128 \
  --pretrained # or --checkpoint /path/to/checkpoint 
```



### 4. Train
We show how to train PoolFormers on 8 GPUs. The relation between learning rate and batch size is lr=bs/1024*1e-3.
For convenience, assuming the batch size is 1024, then the learning rate is set as 1e-3 (for batch size of 1024, setting the learning rate as 2e-3 sometimes sees better performance). 


```bash
MODEL=poolformer_s12 # poolformer_{s12, s24, s36, m36, m48}
DROP_PATH=0.1 # drop path rates [0.1, 0.1, 0.2, 0.3, 0.4] responding to model [s12, s24, s36, m36, m48]
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./distributed_train.sh 8 /path/to/imagenet \
  --model $MODEL -b 128 --lr 1e-3 --drop-path $DROP_PATH --apex-amp
```

### 5. Visualization
![gradcam](https://user-images.githubusercontent.com/15921929/201674709-024a5356-42f2-433d-89e7-801c23646211.png)

The code to visualize Grad-CAM activation maps of PoolFomer, DeiT, ResMLP, ResNet and Swin are [here](misc/cam_image.py).


## Acknowledgment
Our implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.

[pytorch-image-models](https://github.com/rwightman/pytorch-image-models), [mmdetection](https://github.com/open-mmlab/mmdetection), [mmsegmentation](https://github.com/open-mmlab/mmsegmentation).


Besides, Weihao Yu would like to thank TPU Research Cloud (TRC) program for the support of partial computational resources.
