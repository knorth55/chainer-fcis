# Original work by Yasunori Kudo (@yasunorikudo)
# https://github.com/yasunorikudo/chainer-ResNet
#
# Modified by Shingo Kitagawa (@knorth55)

import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L


class BottleNeckA(chainer.Chain):

    eps = 1e-5

    def __init__(self, in_size, out_size, ch, stride=2):
        super(BottleNeckA, self).__init__()
        initialW = initializers.HeNormal()

        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_size, ch, 1, stride, 0, initialW=initialW, nobias=True)
            self.bn1 = L.BatchNormalization(ch, eps=self.eps)
            self.conv2 = L.Convolution2D(
                ch, ch, 3, 1, 1, initialW=initialW, nobias=True)
            self.bn2 = L.BatchNormalization(ch, eps=self.eps)
            self.conv3 = L.Convolution2D(
                ch, out_size, 1, 1, 0, initialW=initialW, nobias=True)
            self.bn3 = L.BatchNormalization(out_size, eps=self.eps)

            self.conv4 = L.Convolution2D(
                in_size, out_size, 1, stride, 0,
                initialW=initialW, nobias=True)
            self.bn4 = L.BatchNormalization(out_size)

    def __call__(self, x):
        h1 = F.relu(self.bn1(self.conv1(x)))
        h1 = F.relu(self.bn2(self.conv2(h1)))
        h1 = self.bn3(self.conv3(h1))
        h2 = self.bn4(self.conv4(x))

        return F.relu(h1 + h2)


class DilatedBottleNeckA(chainer.Chain):

    eps = 1e-5

    def __init__(self, in_size, out_size, ch, stride=1):
        super(DilatedBottleNeckA, self).__init__()
        initialW = initializers.HeNormal()

        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_size, ch, 1, stride, 0, initialW=initialW, nobias=True)
            self.bn1 = L.BatchNormalization(ch, eps=self.eps)
            self.conv2 = L.DilatedConvolution2D(
                ch, ch, 3, 1, 2, dilate=2,
                initialW=initialW, nobias=True)
            self.bn2 = L.BatchNormalization(ch, eps=self.eps)
            self.conv3 = L.Convolution2D(
                ch, out_size, 1, 1, 0, initialW=initialW, nobias=True)
            self.bn3 = L.BatchNormalization(out_size, eps=self.eps)

            self.conv4 = L.Convolution2D(
                in_size, out_size, 1, stride, 0,
                initialW=initialW, nobias=True)
            self.bn4 = L.BatchNormalization(out_size)

    def __call__(self, x):
        h1 = F.relu(self.bn1(self.conv1(x)))
        h1 = F.relu(self.bn2(self.conv2(h1)))
        h1 = self.bn3(self.conv3(h1))
        h2 = self.bn4(self.conv4(x))

        return F.relu(h1 + h2)


class BottleNeckB(chainer.Chain):

    eps = 1e-5

    def __init__(self, in_size, ch):
        super(BottleNeckB, self).__init__()
        initialW = initializers.HeNormal()

        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_size, ch, 1, 1, 0, initialW=initialW, nobias=True)
            self.bn1 = L.BatchNormalization(ch, eps=self.eps)
            self.conv2 = L.Convolution2D(
                ch, ch, 3, 1, 1, initialW=initialW, nobias=True)
            self.bn2 = L.BatchNormalization(ch, eps=self.eps)
            self.conv3 = L.Convolution2D(
                ch, in_size, 1, 1, 0, initialW=initialW, nobias=True)
            self.bn3 = L.BatchNormalization(in_size, eps=self.eps)

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))

        return F.relu(h + x)


class DilatedBottleNeckB(chainer.Chain):

    eps = 1e-5

    def __init__(self, in_size, ch):
        super(DilatedBottleNeckB, self).__init__()
        initialW = initializers.HeNormal()

        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_size, ch, 1, 1, 0, initialW=initialW, nobias=True)
            self.bn1 = L.BatchNormalization(ch, eps=self.eps)
            self.conv2 = L.DilatedConvolution2D(
                ch, ch, 3, 1, 2, dilate=2,
                initialW=initialW, nobias=True)
            self.bn2 = L.BatchNormalization(ch, eps=self.eps)
            self.conv3 = L.Convolution2D(
                ch, in_size, 1, 1, 0, initialW=initialW, nobias=True)
            self.bn3 = L.BatchNormalization(in_size, eps=self.eps)

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))

        return F.relu(h + x)


class ResNet101C1(chainer.Chain):

    eps = 1e-5

    def __init__(self):
        super(ResNet101C1, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                3, 64, 7, 2, 3, initialW=initializers.HeNormal(), nobias=True)
            self.bn1 = L.BatchNormalization(64, eps=self.eps)

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2, pad=0)
        return h


class ResNet101C2(chainer.Chain):

    layer = 3

    def __init__(self):
        super(ResNet101C2, self).__init__()
        with self.init_scope():
            self.res2_a = BottleNeckA(64, 256, 64, stride=1)
            for i in range(1, self.layer):
                self.add_link('res2_b{}'.format(i), BottleNeckB(256, 64))

    def __call__(self, x):
        h = self.res2_a(x)
        for i in range(1, self.layer):
            h = self['res2_b{}'.format(i)](h)
        return h


class ResNet101C3(chainer.Chain):

    layer = 4

    def __init__(self):
        super(ResNet101C3, self).__init__()
        with self.init_scope():
            self.res3_a = BottleNeckA(256, 512, 128, stride=2)
            for i in range(1, self.layer):
                self.add_link('res3_b{}'.format(i), BottleNeckB(512, 128))

    def __call__(self, x):
        h = self.res3_a(x)
        for i in range(1, self.layer):
            h = self['res3_b{}'.format(i)](h)
        return h


class ResNet101C4(chainer.Chain):

    layer = 23

    def __init__(self):
        super(ResNet101C4, self).__init__()
        with self.init_scope():
            self.res4_a = BottleNeckA(512, 1024, 256, stride=2)
            for i in range(1, self.layer):
                self.add_link('res4_b{}'.format(i), BottleNeckB(1024, 256))

    def __call__(self, x):
        h = self.res4_a(x)
        for i in range(1, self.layer):
            h = self['res4_b{}'.format(i)](h)
        return h


class ResNet101C5(chainer.Chain):

    layer = 3

    def __init__(self):
        super(ResNet101C5, self).__init__()
        with self.init_scope():
            self.res5_a = DilatedBottleNeckA(1024, 2048, 512, stride=1)
            for i in range(1, self.layer):
                self.add_link('res5_b{}'.format(i),
                              DilatedBottleNeckB(2048, 512))

    def __call__(self, x):
        h = self.res5_a(x)
        for i in range(1, self.layer):
            h = self['res5_b{}'.format(i)](h)
        return h
