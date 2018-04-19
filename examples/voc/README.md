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
| [Original](https://github.com/msracver/FCIS) | Random | 0.646 | 0.499 |
| Ours | Random | 0.632 | 0.492 |

**Detailed Evaluation**

| Item | Ours AP@0.5 | Original AP@0.5 | Ours AP@0.7 | Original AP@0.7 |
|:------------:|:-----:|:-----:|:-----:|:-----:|
| **mean**     | 0.632 | 0.646 | 0.492 | 0.499 |
| aeroplane    | 0.769 | 0.783 | 0.618 | 0.649 |
| bicycle      | 0.685 | 0.684 | 0.442 | 0.455 |
| bird         | 0.687 | 0.698 | 0.584 | 0.570 |
| boat         | 0.462 | 0.486 | 0.324 | 0.338 |
| bottle       | 0.420 | 0.444 | 0.316 | 0.333 |
| bus          | 0.789 | 0.790 | 0.709 | 0.754 |
| car          | 0.659 | 0.677 | 0.543 | 0.547 |
| cat          | 0.841 | 0.840 | 0.767 | 0.745 |
| chair        | 0.384 | 0.420 | 0.229 | 0.246 |
| cow          | 0.639 | 0.657 | 0.477 | 0.534 |
| dining table | 0.343 | 0.384 | 0.193 | 0.245 |
| dog          | 0.801 | 0.812 | 0.707 | 0.703 |
| horse        | 0.723 | 0.720 | 0.499 | 0.478 |
| motorbike    | 0.702 | 0.738 | 0.556 | 0.530 |
| person       | 0.718 | 0.754 | 0.508 | 0.543 |
| potted plant | 0.410 | 0.415 | 0.274 | 0.264 |
| sheep        | 0.683 | 0.704 | 0.534 | 0.533 |
| sofa         | 0.502 | 0.491 | 0.344 | 0.324 |
| train        | 0.784 | 0.775 | 0.670 | 0.660 |
| tv/monitor   | 0.642 | 0.639 | 0.549 | 0.536 |

## Dataset Download

- [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/)
- [SBD](http://home.bharathh.info/pubs/codes/SBD/download.html)

```bash
# Dataset will be downloaded to ~/data/datasets/VOC
python download_datasets.py --voc
python download_datasets.py --sbd
```
