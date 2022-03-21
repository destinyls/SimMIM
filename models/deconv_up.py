import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

BN_MOMENTUM = 0.1

class DeconvUp(nn.Module):
    def __init__(self, input_channels):
        super(DeconvUp, self).__init__()
        # used for deconv layers
        self.inplanes = input_channels
        self.deconv_with_bias = False
        self.out_channels = 256
        # used for deconv layers

        self.deconv_layers_1 = self._make_deconv_layer(256, 4)
        self.deconv_layers_2 = self._make_deconv_layer(256, 4)
        self.deconv_layers_3 = self._make_deconv_layer(256, 4)
        self.init_weights()

    def forward(self, x):
        up_level1 = self.deconv_layers_1(x)
        up_level2 = self.deconv_layers_2(up_level1)
        up_level3 = self.deconv_layers_3(up_level2)

        return up_level3

    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_filters, num_kernels):
        layers = []
        kernel, padding, output_padding = self._get_deconv_cfg(num_kernels)
        planes = num_filters
        
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

    def init_weights(self, pretrained=True):
        if pretrained:
            # print('=> init resnet deconv weights from normal distribution')
            self.init_deconv(self.deconv_layers_1)
            self.init_deconv(self.deconv_layers_2)
            self.init_deconv(self.deconv_layers_3)
        else:
            print('=> imagenet pretrained model dose not exist')
            print('=> please download it first')
            raise ValueError('imagenet pretrained model does not exist')

    def init_deconv(self, layer):
        for _, m in layer.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                # print('=> init {}.bias as 0'.format(name))
                nn.init.normal_(m.weight, std=0.001)
                if self.deconv_with_bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # print('=> init {}.weight as 1'.format(name))
                # print('=> init {}.bias as 0'.format(name))
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)