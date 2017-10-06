chainer-fcis - FCIS 
===================
![Example](static/example.png)

This is [Chainer](https://github.com/chainer/chainer) implementation of [Fully Convolutional Instance-aware Semantic Segmentation](https://arxiv.org/abs/1611.07709).

Original repository is [msracver/FCIS](https://github.com/msracver/FCIS).

Requirement
-----------

- Chainer
- ChainerCV
- NumPy
- OpenCV2

Installation
------------

# Attention

Only GPU implementation, No CPU implementation yet.


```bash
git clone https://github.com/knorth55/chainer-fcis.git 
cd chainer-fcis
pip install -e .
```

Inference
---------

Inference can be done as below;

```bash
cd examples/coco/
python demo.py
```

Above is our implementation output, and below is original.

<img src="static/output.png" width="60%" >
<img src="static/original_output.png" width="60%" >

Training
--------

I'm going to implement it soon.

LICENSE
-------
[MIT LICENSE](LICENSE)


Powered by [DL HACKS](http://deeplearning.jp/hacks/)
