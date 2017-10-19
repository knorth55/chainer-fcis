# COCO instance segmentation


## Inference

Pretrained model can be dowloaded [here](https://drive.google.com/open?id=0B5DV6gwLHtyJZTR0NFllNGlwS3M).

```bash
# Pretrained model will be downloaded automatically
# or run below.
# python download_models.py
python demo.py
```

## Evaluation

```bash
# Download datasets manually
# or run below.
# python download_datasets.py --val
python evaluate.py --data-dir /your/coco/dataset/dir
```

**FCIS ResNet101**

| Implementation | mAP/iou@[0.5:0.95] | mAP/iou@0.5 | mAP/iou@[0.5:0.95] \(small) | mAP/iou@[0.5:0.95] \(medium) | mAP/iou@[0.5:0.95] \(large) |
|:--------------:|:------------------:|:-----------:|:---------------------------:|:---------------------------:|:--------------------------:|
| [Original](https://github.com/msracver/FCIS) | 0.292 | 0.495 | 0.071 | 0.313 | 0.500|
| Ours | 0.259 | 0.443 | 0.057 | 0.270 | 0.465 |

## Training

Not Implemented yet
