# Applying PoolFormer to Object Detection

For details see [MetaFormer is Actually What You Need for Vision](https://arxiv.org/abs/2111.11418) (CVPR 2020 Oral). 

## Note
Please note that we just simply follow the hyper-parameters of [PVT](https://github.com/whai362/PVT/tree/v2/detection) which may not be the optimal ones for PoolFormer. 
Feel free to tune the hyper-parameters to get better performance. 


## Environement Setup

Install [MMDetection v2.19.0](https://github.com/open-mmlab/mmdetection/tree/v2.19.0) from souce cocde,

or

```
pip install mmdet==2.19.0 --user
```

Apex (optional):
```
git clone https://github.com/NVIDIA/apex
cd apex
python setup.py install --cpp_ext --cuda_ext --user
```

If you would like to disable apex, modify the type of runner as `EpochBasedRunner` and comment out the following code block in the configuration files:
```
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
```

Note: Since we write [PoolFormer backbone code](../models/poolformer.py) of detection and segmentation in a same file which requires to install both [MMDetection v2.19.0](https://github.com/open-mmlab/mmdetection/tree/v2.19.0) and [MMSegmentation v0.19.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.19.0). Please continue to install MMSegmentation or modify the backone code.


[Dockerfile_mmdetseg](Dockerfile_mmdetseg) is the docker file that I use to set up the environment for detection and segmentation. You can also refer to it.

## Data preparation

Prepare COCO according to the guidelines in [MMDetection v2.19.0](https://github.com/open-mmlab/mmdetection/tree/v2.19.0).


## Results and models on COCO


| Method     | Backbone | Pretrain    | Lr schd | Aug | box AP | mask AP | Config                                               | Download |
|------------|----------|-------------|:-------:|:---:|:------:|:-------:|------------------------------------------------------|----------|
| RetinaNet  | PoolFormer-S12 | ImageNet-1K |    1x   |  No |  36.2  |    -    | [config](configs/retinanet_poolformer_s12_fpn_1x_coco.py) | [log](https://drive.google.com/file/d/1wdpzEmthjj8WJ99SnCLb32sF38FBbod7/view?usp=sharing) & [model](https://drive.google.com/file/d/1GKx4jbxdO4ClagPXXt7CoomrV4pOpqul/view?usp=sharing) |
| RetinaNet  | PoolFormer-S24 | ImageNet-1K |    1x   |  No |  38.9  |    -    | [config](configs/retinanet_poolformer_s24_fpn_1x_coco.py) | [log](https://drive.google.com/file/d/1eNlNM1HDBLWejhMgMETvkPxLvUcP0OZ9/view?usp=sharing) & [model](https://drive.google.com/file/d/1EjsWpdopem-xeLndPQnQcHp8aoEUHQXR/view?usp=sharing) |
| RetinaNet  | PoolFormer-S36 | ImageNet-1K |    1x   |  No |  39.5  |    -    | [config](configs/retinanet_poolformer_s36_fpn_1x_coco.py) | [log](https://drive.google.com/file/d/1qk-dSgfgYqFbo4zPu3Z3WdV7Kzm28_Xf/view?usp=sharing) & [model](https://drive.google.com/file/d/1EgJDCg7LXXnHdGdJaHyEnoBPm-fNG2bt/view?usp=sharing) |
| Mask R-CNN | PoolFormer-S12 | ImageNet-1K |    1x   |  No |  37.3 |   34.6  | [config](configs/mask_rcnn_poolformer_s12_fpn_1x_coco.py) | [log](https://drive.google.com/file/d/1UfIP32QmT7MxBL_AQ3z1h7L21xYlB6aJ/view?usp=sharing) & [model](https://drive.google.com/file/d/1-GSkqaS3SovfCVDsH8CzS1DikPX3cFTY/view?usp=sharing) |
| Mask R-CNN | PoolFormer-S24 | ImageNet-1K |    1x   |  No |  40.1  |   37.0  | [config](configs/mask_rcnn_poolformer_s24_fpn_1x_coco.py) | [log](https://drive.google.com/file/d/1yz6NPJ63ZlN02Oj2TY6KnjxK2Xg03BBa/view?usp=sharing) & [model](https://drive.google.com/file/d/10Br62EU-VErQq6rP67sf4qXJIBLOnmLT/view?usp=sharing) |
| Mask R-CNN | PoolFormer-S36 | ImageNet-1K |    1x   |  No |  41.0  |   37.7  | [config](configs/mask_rcnn_poolformer_s36_fpn_1x_coco.py) | [log](https://drive.google.com/file/d/1oac1AVJ9skQZp0yXjTYY9_IhM8AxHVjT/view?usp=sharing) & [model](https://drive.google.com/file/d/1LyJxcO0fw2hwZg9Z--Zbjbw3W7U4JyqT/view?usp=sharing) |


All the models can also be downloaded by [BaiDu Yun](https://pan.baidu.com/s/1HSaJtxgCkUlawurQLq87wQ) (password: esac).


## Evaluation
To evaluate PoolFormer-S12 + RetinaNet on COCO val2017 on a single node with 8 GPUs run:
```
FORK_LAST3=1 dist_test.sh configs/retinanet_poolformer_s12_fpn_1x_coco.py /path/to/checkpoint_file 8 --out results.pkl --eval bbox
```
To evaluate PoolFormer-S12 + Mask R-CNN on COCO val2017, run:
```
dist_test.sh configs/mask_rcnn_poolformer_s12_fpn_1x_coco.py /path/to/checkpoint_file 8 --out results.pkl --eval bbox segm
```


## Training
To train PoolFormer-S12 + RetinaNet on COCO train2017 on a single node with 8 GPUs for 12 epochs run:

```
FORK_LAST3=1 dist_train.sh configs/retinanet_poolformer_s12_fpn_1x_coco.py 8
```

To train PoolFormer-S12 + Mask R-CNN on COCO train2017:
```
dist_train.sh configs/mask_rcnn_poolformer_s12_fpn_1x_coco.py 8
```

## Bibtex
```
@article{yu2021metaformer,
  title={MetaFormer is Actually What You Need for Vision},
  author={Yu, Weihao and Luo, Mi and Zhou, Pan and Si, Chenyang and Zhou, Yichen and Wang, Xinchao and Feng, Jiashi and Yan, Shuicheng},
  journal={arXiv preprint arXiv:2111.11418},
  year={2021}
}
```

## Acknowledgment
Our implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.

[mmdetection](https://github.com/open-mmlab/mmdetection), [PVT detection](https://github.com/whai362/PVT/tree/v2/detection).