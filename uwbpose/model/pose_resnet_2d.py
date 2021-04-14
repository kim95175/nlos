# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------
# SSH

import torch
import torch.nn as nn
import torch.nn.functional as F

BN_MOMENTUM = 0.1
INPUT_D = 1


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PoseResNet(nn.Module):

    def __init__(self, block, layers, **kwargs):
        self.inplanes = 64
        self.deconv_with_bias = False
        print("---------------flatten pose net---------------")
        super(PoseResNet, self).__init__()
        # ResNet
        print(INPUT_D, self.inplanes)
        self.conv1 = nn.Sequential(
            #nn.Conv2d(INPUT_D, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(INPUT_D, self.inplanes, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.inplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inplanes, self.inplanes, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.inplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)  # Bottleneck 3
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # Bottleneck 4
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # Bottleneck 6
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # Bottlenect 3

        # used for deconv layers   num_deconv_layers,  num_deconv_filters, num_deconv_kernels
        self.deconv_layers = self._make_deconv_layer(
            3,  # NUM_DECONV_LAYERS
            [256, 256, 256],  # NUM_DECONV_FILTERS
            [4, 4, 4],  # NUM_DECONV_KERNERLS
        )

        self.final_layer = nn.Conv2d(
            in_channels=256,  # NUM_DECONV_FILTERS[-1]
            out_channels=13,  # NUM_JOINTS,
            kernel_size=1,  # FINAL_CONV_KERNEL
            stride=1,
            padding=0  # if FINAL_CONV_KERNEL = 3 else 1
        )

        dummy = torch.zeros(64, 1, 126, 126)
        self.check_dim(dummy)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        elif deconv_kernel == 5:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = F.interpolate(x, scale_factor=40, mode='bilinear', align_corners=False)
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        return x

    def check_dim(self, x):
        # x = F.interpolate(x, scale_factor=40, mode='bilinear', align_corners=False)
        print("raw_x.shape ", x.shape)
        x = self.conv1(x)
        print("conv1",x.shape)
        #x = self.maxpool(x)
        #print(x.shape)
        x = self.layer1(x)
        print("layer1 ",x.shape)
        x = self.layer2(x)
        print("layer2 ",x.shape)
        x = self.layer3(x)
        print("layer3 ",x.shape)
        x = self.layer4(x)
        print("layer4 ",x.shape)
        x = self.deconv_layers(x)
        print("before_final",x.shape)
        x = self.final_layer(x)
        print(x.shape)
        return x

    def init_weights(self, pretrained=''):
        pass


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def get_2d_pose_net(num_layer, input_depth, **kwargs):
    global INPUT_D
    INPUT_D = input_depth
    num_layers = num_layer
    block_class, layers = resnet_spec[num_layers]

    # model = PoseResNet(block_class, layers, cfg, **kwargs)

    model = PoseResNet(block_class, layers, **kwargs)

    # if is_train and cfg.MODEL.INIT_WEIGHTS:
    #    model.init_weights(cfg.MODEL.PRETRAINED)
    # model.init_weights('models/imagenet/resnet50-19c8e357.pth')
    return model
