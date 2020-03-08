import torch
from torch import nn
import logging

class QUADbox(nn.Module):
    def __init__(self):
        super(QUADbox, self).__init__()
        self.scope = 512
        self.activation = nn.Sigmoid()

    def Generate_Map(self, h, in_channels, out_channels, stride=1):
        convolution = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
        return self.activation(convolution(h))

    def forward(self, h5):
        logging.info("\nInput h5 dimensions: [1, channels, height, width] = {}\n".format(h5.size()))

        score_map = self.Generate_Map(h5, in_channels=32, out_channels=1)
        logging.info("\nScore map dimensions: [1, channels, height, width] = {}\n".format(score_map.size()))
        logging.info("Score map: {}".format(score_map))

        geometry_map = self.scope * self.Generate_Map(h5, in_channels=32, out_channels=8)
        logging.info("\nGeometry map dimensions: [1, channels, height, width] = {}\n".format(geometry_map.size()))
        logging.info("Geometry map: {}".format(geometry_map))

        return (score_map, geometry_map)

class Output(nn.Module):
    def __init__(self, trained=False):
        super(Output, self).__init__()
        self.quad = QUADbox()

    def forward(self, merged_features):
        logging.info("Output Layer\n")
        score_map, geometry_map = self.quad.forward(merged_features)
        return (score_map, geometry_map)
