import torch
from torch import nn
import math
import logging

class Output(nn.Module):
    def __init__(self):
        super(Output, self).__init__()
        self.scope = 256 # what value????
        self.activation = nn.Sigmoid()

        self.convolution12 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1)
        self.convolution13 = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=1, stride=1)
        self.convolution14 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)

    def forward(self, h5):
        # logging.info("Output Layer\n")
        # logging.info("\nInput h5 dimensions: [1, channels, height, width] = {}\n".format(h5.size()))

        score_map = self.activation(self.convolution12(h5))
        # logging.info("\nScore map dimensions: {}".format(score_map.size()))
        # logging.info("\nScore map: {}\n".format(score_map))

        distance_map = self.scope * self.activation(self.convolution13(h5))
        # logging.info("\nDistance map dimensions: {}\n".format(distance_map.size()))

        angle_map = math.pi * (self.activation(self.convolution14(h5)) - 0.5)
        # logging.info("\nAngle map dimensions: {}\n".format(angle_map.size()))

        geometry_map = torch.cat((distance_map, angle_map), 1)
        # geometry_map = self.activation(self.convolution13(h5))
        # logging.info("\nGeometry map dimensions: {}".format(geometry_map.size()))
        # logging.info("\nGeometry map: {}\n".format(geometry_map))

        # logging.info("\nRange of values in geometry map: ({}, {})\n".format(torch.min(geometry_map), torch.max(geometry_map)))

        return (score_map, geometry_map)
