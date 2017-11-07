# Pascal VOC & SBD instance segmentation

## Training

```bash
# Download dataset in ~/data/datasets/VOC
python train.py --gpu 0
```

## Dataset

- [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/)
- [SBD](http://home.bharathh.info/pubs/codes/SBD/download.html)

```bash
# Dataset will be downloaded to ~/data/datasets/VOC
python download_datasets.py --voc --all
python download_datasets.py --sbd --all
```
