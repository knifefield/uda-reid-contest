from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision

from .circle import Circle
from mmt.utils.weight_init import weights_init_classifier
from .non_local import NONLocalBlock2D

__all__ = ['MyResNet', 'resnet_att']


class MyResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, circle=1):
        super(MyResNet, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        self.circle = circle
        self.cut_at_pooling = cut_at_pooling
        self.non_local = NONLocalBlock2D(1024)
        # Construct base (pretrained) resnet
        if depth not in MyResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        resnet = MyResNet.__factory[depth](pretrained=pretrained)
        layer3_head_ = [resnet.layer3[i] for i in range(3)]
        layer3_head = nn.Sequential(*layer3_head_)
        layer3_tail_ = [resnet.layer3[i] for i in range(3, 6)]
        layer3_tail = nn.Sequential(*layer3_tail_)
        resnet.layer3 = nn.Sequential(layer3_head, self.non_local, layer3_tail)
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)
        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.maxpool,  # no relu
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = resnet.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
                self.feat_bn = nn.BatchNorm1d(self.num_features)
            self.feat_bn.bias.requires_grad_(False)
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                if self.circle > 0:
                    self.classifier = Circle(self.num_features, self.num_classes, 0.25, 256)
                    self.classifier.apply(weights_init_classifier)
                else:
                    self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
                    init.normal_(self.classifier.weight, std=0.001)
        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

        if not pretrained:
            self.reset_params()

    def forward(self, x, targets=None):
        x = self.base(x)

        x_ap = self.gap(x)
        x_mp = self.gmp(x)
        x = x_ap + x_mp
        x = x.view(x.size(0), -1)

        if self.has_embedding:
            bn_x = self.feat_bn(self.feat(x))
        else:
            bn_x = self.feat_bn(x)

        if self.training is False:
            bn_x = F.normalize(bn_x)
            return bn_x

        if self.norm:
            bn_x = F.normalize(bn_x)
        elif self.has_embedding:
            bn_x = F.relu(bn_x)

        if self.dropout > 0:
            bn_x = self.drop(bn_x)

        if self.num_classes > 0:
            if self.circle > 0:
                bn_x = F.normalize(bn_x)
                prob = self.classifier(bn_x, targets)
            else:
                prob = self.classifier(bn_x)
        else:
            return x, bn_x

        return x, prob

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        resnet = MyResNet.__factory[self.depth](pretrained=self.pretrained)
        self.base[0].load_state_dict(resnet.conv1.state_dict())
        self.base[1].load_state_dict(resnet.bn1.state_dict())
        self.base[2].load_state_dict(resnet.maxpool.state_dict())
        self.base[3].load_state_dict(resnet.layer1.state_dict())
        self.base[4].load_state_dict(resnet.layer2.state_dict())
        self.base[5].load_state_dict(resnet.layer3.state_dict())
        self.base[6].load_state_dict(resnet.layer4.state_dict())


def resnet_att(**kwargs):
    return MyResNet(50, **kwargs)
