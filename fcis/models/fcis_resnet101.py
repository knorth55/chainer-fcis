import chainer
import chainer.functions as F
import chainer.links as L
from chainercv.links.model.faster_rcnn.region_proposal_network import \
    RegionProposalNetwork
from fcis.models import ResNet101C1
from fcis.models import ResNet101C2
from fcis.models import ResNet101C3
from fcis.models import ResNet101C4
from fcis.models import ResNet101C5


class FCISResnet101(chainer.Chain):

    def __init__(
            self, in_channels=512, mid_channels=512, ratios=[0.5, 1, 2],
            anchor_scales=[8, 16, 32], feat_stride=16,
            proposal_creator_params={},
    ):
        super(FCISResnet101, self).__init__()
        with self.init_scope():
            self.res1 = ResNet101C1()
            self.res2 = ResNet101C2()
            self.res3 = ResNet101C3()
            self.res4 = ResNet101C4()

            self.rpn = RegionProposalNetwork(
                1024, 512,
                ratios=ratios,
                anchor_scales=anchor_scales,
                feat_stride=self.feat_stride,
                proposal_creator_params=proposal_creator_params
            )

            self.res5 = ResNet101C5()
            self.conv_new_1 = L.Convolution2D(
                2048, 1024, 1, 1, 0)

    def __call__(self, x, scale=1.0):
        img_size = x.shape[2:]

        # Feature Extractor
        h = self.res1(x)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.extractor(x)

        # RPN
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(
            h, img_size, scale)

        h = self.res5(h)
        h = F.relu(self.conv_new_1(h))
