# Applying PoolFormer to Semantic Segmentation

Our semantic segmentation implementation is based on [MMSegmentation v0.19.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.19.0) and [PVT segmentation](https://github.com/whai362/PVT/tree/v2/segmentation). Thank the authors for their wonderful works.

For details see [MetaFormer is Actually What You Need for Vision](https://arxiv.org/abs/2111.11418). 

## Note
Please note that we just simply follow the hyper-parameters of PVT which may not be the optimal ones for PoolFormer. 
Feel free to tune the hyper-parameters to get better performance. 


## Bibtex
```
@article{yu2021metaformer,
  title={MetaFormer is Actually What You Need for Vision},
  author={Yu, Weihao and Luo, Mi and Zhou, Pan and Si, Chenyang and Zhou, Yichen and Wang, Xinchao and Feng, Jiashi and Yan, Shuicheng},
  journal={arXiv preprint arXiv:2111.11418},
  year={2021}
}
```

## Usage

Install MMSegmentation v0.19.0. `Dockerfile_mmdetseg' is the docker file that I use to set up the environment for detection and segmentation. You can also refer to it.


## Data preparation

Prepare ADE20K according to the [guidelines](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/dataset_prepare.md#prepare-datasets) in MMSegmentation.


## Results and models

| Method | Backbone | Pretrain | Iters | mIoU | Config | Download |
| --- | --- | --- |:---:|:---:| --- | --- |
| Semantic FPN | PoolFormer-S12   | ImageNet-1K |  40K  |     37.2    | [config](configs/sem_fpn/PoolFormer/fpn_poolformer_s12_ade20k_40k.py) | [log](https://drive.google.com/file/d/12_fdrElU0yeMImJRcHhhYekB28lu-12v/view?usp=sharing) & [model](https://drive.google.com/file/d/1BcqU1yU2IPkI7RtWEmIw-R8tqMGY_XBt/view?usp=sharing) |
| Semantic FPN | PoolFormer-S24  | ImageNet-1K |  40K  |     40.3    | [config](configs/sem_fpn/PoolFormer/fpn_poolformer_s24_ade20k_40k.py) | [log](https://drive.google.com/file/d/1_NpbNM6sToh6pWVQRbdZW6ToeX6BU2Bl/view?usp=sharing) & [model](https://drive.google.com/file/d/1DO329W8eDrfgycHi7YagFWz7IyAb07Wl/view?usp=sharing) |
| Semantic FPN | PoolFormer-S36 | ImageNet-1K |  40K  |     42.0    | [config](configs/sem_fpn/PoolFormer/fpn_poolformer_s36_ade20k_40k.py) | [log](https://drive.google.com/file/d/1aK1y9CKDRsJsL6OGmOWNmZMh41_qA1Z9/view?usp=sharing) & [model](https://drive.google.com/file/d/1Rd6XxBXLEYWH-70IMvF6UiVymaWA3gik/view?usp=sharing) |
| Semantic FPN | PoolFormer-M36  | ImageNet-1K |  40K  |     42.4    | [config](configs/sem_fpn/PoolFormer/fpn_poolformer_m36_ade20k_40k.py) | [log](https://drive.google.com/file/d/1tsaDngVwrIiIvWdU4W_EGAks_EXWQhzD/view?usp=sharing) & [model](https://drive.google.com/file/d/1Xgk7FI3FpOW2__UQhnGf7UGUHA3UzTRq/view?usp=sharing) |
| Semantic FPN | PoolFormer-M48  | ImageNet-1K |  40K  |     42.7    | [config](configs/sem_fpn/PoolFormer/fpn_poolformer_m48_ade20k_40k.py) | [log](https://drive.google.com/file/d/1_LI7xA0B7ladlytlBrGDjHMpYDAtSHh3/view?usp=sharing) & [model](https://drive.google.com/file/d/1KjeR_4Ue0QyslDimp3OYkRqeqNAwoAez/view?usp=sharing) |


All the models can also be downloaded by [BaiDu Yun](https://pan.baidu.com/s/1HSaJtxgCkUlawurQLq87wQ) (password: esac).

## Evaluation
To evaluate PoolFormer-S12 + Semantic FPN on a single node with 8 GPUs run:
```
dist_test.sh configs/sem_fpn/PoolFormer/fpn_poolformer_s12_ade20k_40k.py /path/to/checkpoint_file 8 --out results.pkl --eval mIoU
```


## Training
To train PoolFormer-S12 + Semantic FPN on a single node with 8 GPUs run:

```
dist_train.sh configs/sem_fpn/PoolFormer/fpn_poolformer_s12_ade20k_40k.py 8
```