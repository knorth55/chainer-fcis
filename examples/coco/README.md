# COCO instance segmentation

![Example](../../static/coco_example.png)

## Inference

Trained model can be dowloaded [here](https://drive.google.com/open?id=0B5DV6gwLHtyJZTR0NFllNGlwS3M).

```bash
# Pretrained model will be downloaded automatically
# or run below.
# python download_models.py

python demo.py
```

## Training

### Caution

COCO training requires very long time.
If you only have 1 GPU, I recommend to try VOC training first.

### Requirements 
- [Cython](http://cython.org/)
- [pycocotools](https://github.com/cocodataset/cocoapi)
- [OpenMPI](https://www.open-mpi.org/)
- [nccl](https://developer.nvidia.com/nccl)
- [ChainerMN](https://github.com/chainer/chainermn)

```bash
# Download datasets manually in ~/data/datasets/coco
# or run below.
# python download_datasets.py --all

python train.py
```

## Evaluation

### Inference

```bash
# Download datasets manually in ~/data/datasets/coco
# or run below.
# python download_datasets.py --val

python evaluate.py
```

**FCIS ResNet101**

| Implementation | mAP/iou@[0.5:0.95] | mAP/iou@0.5 | mAP/iou@[0.5:0.95] \(small) | mAP/iou@[0.5:0.95] \(medium) | mAP/iou@[0.5:0.95] \(large) |
|:--------------:|:------------------:|:-----------:|:---------------------------:|:---------------------------:|:--------------------------:|
| [Original](https://github.com/msracver/FCIS) | 0.292 | 0.495 | 0.071 | 0.313 | 0.500|
| Ours | 0.259 | 0.444 | 0.058 | 0.271 | 0.466 |


## Dataset Download

- [COCO](http://cocodataset.org/)

```bash
# Dataset will be downloaded to ~/data/datasets/coco
python download_datasets.py --all
```
