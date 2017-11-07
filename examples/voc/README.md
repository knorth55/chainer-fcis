# Pascal VOC & SBD instance segmentation

## Inference

Trained model can be dowloaded [here](https://drive.google.com/open?id=1wIb2eHEIoBvaOR5OfxX7CsoJxGTIH97T).

```bash
# Pretrained model will be downloaded automatically
# or run below.
# python download_models.py

python demo.py
```

## Training

```bash
# Download dataset manually in ~/data/datasets/VOC
# or run below.
# python download_datasets.py --sbd

python train.py --gpu 0
```

## Dataset

- [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/)
- [SBD](http://home.bharathh.info/pubs/codes/SBD/download.html)

```bash
# Dataset will be downloaded to ~/data/datasets/VOC
python download_datasets.py --voc
python download_datasets.py --sbd
```
