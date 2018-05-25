# Pascal VOC & SBD instance segmentation

![Example](../../static/voc_example.png)

## Inference

Trained model can be dowloaded [here](https://drive.google.com/open?id=1PscvchtzYsT_xsNX8EsmY1j0Kju6j0r0).

```bash
python demo.py
```

## Single GPU Training

```bash
# Download dataset manually in ~/data/datasets/VOC
# or run below.
# python download_datasets.py --sbd

python train.py
```

## Multi GPU Training

### Requirements

- [OpenMPI](https://www.open-mpi.org/)
- [nccl](https://developer.nvidia.com/nccl)
- [ChainerMN](https://github.com/chainer/chainermn)

```bash
mpiexec -n 4 python train_multigpu.py
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
| [Original](https://github.com/msracver/FCIS) | Random | **0.657** | **0.520** |
| Ours | Random | 0.632 | 0.492 |

**Detailed Evaluation**

| Item | Ours AP@0.5 | Original AP@0.5 | Ours AP@0.7 | Original AP@0.7 |
|:------------:|:-----:|:-----:|:-----:|:-----:|
| **mean**     | 0.632 | **0.657** | 0.492 | **0.520** |
| aeroplane    | 0.769 | 0.798 | 0.618 | 0.672 |
| bicycle      | 0.685 | 0.702 | 0.442 | 0.454 |
| bird         | 0.687 | 0.719 | 0.584 | 0.609 |
| boat         | 0.462 | 0.503 | 0.324 | 0.366 |
| bottle       | 0.420 | 0.478 | 0.316 | 0.374 |
| bus          | 0.789 | 0.794 | 0.709 | 0.717 |
| car          | 0.659 | 0.682 | 0.543 | 0.572 |
| cat          | 0.841 | 0.855 | 0.767 | 0.775 |
| chair        | 0.384 | 0.422 | 0.229 | 0.254 |
| cow          | 0.639 | 0.628 | 0.477 | 0.502 |
| dining table | 0.343 | 0.395 | 0.193 | 0.229 |
| dog          | 0.801 | 0.818 | 0.707 | 0.733 |
| horse        | 0.723 | 0.727 | 0.499 | 0.545 |
| motorbike    | 0.702 | 0.760 | 0.556 | 0.566 |
| person       | 0.718 | 0.757 | 0.508 | 0.557 |
| potted plant | 0.410 | 0.442 | 0.274 | 0.288 |
| sheep        | 0.683 | 0.705 | 0.534 | 0.536 |
| sofa         | 0.502 | 0.512 | 0.344 | 0.392 |
| train        | 0.784 | 0.786 | 0.670 | 0.684 |
| tv/monitor   | 0.642 | 0.659 | 0.549 | 0.567 |

## Dataset Download

- [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/)
- [SBD](http://home.bharathh.info/pubs/codes/SBD/download.html)

```bash
# Dataset will be downloaded to ~/data/datasets/VOC
python download_datasets.py --voc
python download_datasets.py --sbd
```
