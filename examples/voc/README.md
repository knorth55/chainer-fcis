# Pascal VOC & SBD instance segmentation

![Example](../../static/voc_example.png)

## Inference

Trained model can be dowloaded [here](https://drive.google.com/open?id=1PscvchtzYsT_xsNX8EsmY1j0Kju6j0r0).

```bash
python demo.py
```

## Training

```bash
# Download dataset manually in ~/data/datasets/VOC
# or run below.
# python download_datasets.py --sbd

python train.py
```

## Evaluation

```bash
# Download dataset manually in ~/data/datasets/VOC
# or run below.
# python download_datasets.py --sbd

python evaluate.py
```

### FCIS ResNet101

This is evaluation of model trained by our repo.

| Implementation | Sampling Strategy | mAP@0.5 | mAP@0.7 |
|:--------------:|:-----------------:|:-------:|:-------:|
| [Original](https://github.com/msracver/FCIS) | Random | 0.646 | 0.499 |
| Ours | Random | 0.630 | 0.488 |

**Detailed Evaluation**

| Item | Ours AP@0.5 | Original AP@0.5 | Ours AP@0.7 | Original AP@0.7 |
|:------------:|:-----:|:-----:|:-----:|:-----:|
| **mean**     | 0.630 | 0.646 | 0.488 | 0.499 |
| aeroplane    | 0.771 | 0.783 | 0.642 | 0.649 |
| bicycle      | 0.698 | 0.684 | 0.446 | 0.455 |
| bird         | 0.686 | 0.698 | 0.588 | 0.570 |
| boat         | 0.459 | 0.486 | 0.346 | 0.338 |
| bottle       | 0.416 | 0.444 | 0.305 | 0.333 |
| bus          | 0.784 | 0.790 | 0.721 | 0.754 |
| car          | 0.667 | 0.677 | 0.518 | 0.547 |
| cat          | 0.836 | 0.840 | 0.748 | 0.745 |
| chair        | 0.398 | 0.420 | 0.231 | 0.246 |
| cow          | 0.631 | 0.657 | 0.479 | 0.534 |
| dining table | 0.331 | 0.384 | 0.179 | 0.245 |
| dog          | 0.809 | 0.812 | 0.710 | 0.703 |
| horse        | 0.708 | 0.720 | 0.517 | 0.478 |
| motorbike    | 0.719 | 0.738 | 0.546 | 0.530 |
| person       | 0.704 | 0.754 | 0.473 | 0.543 |
| potted plant | 0.407 | 0.415 | 0.242 | 0.264 |
| sheep        | 0.651 | 0.704 | 0.498 | 0.533 |
| sofa         | 0.491 | 0.491 | 0.368 | 0.324 |
| train        | 0.781 | 0.775 | 0.649 | 0.660 |
| tv/monitor   | 0.642 | 0.639 | 0.553 | 0.536 |

## Dataset Download

- [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/)
- [SBD](http://home.bharathh.info/pubs/codes/SBD/download.html)

```bash
# Dataset will be downloaded to ~/data/datasets/VOC
python download_datasets.py --voc
python download_datasets.py --sbd
```
