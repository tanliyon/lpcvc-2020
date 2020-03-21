import torch
from torch import nn
import logging

# class Merge(nn.Module):
#     def __init__(self):
#         super(Merge, self).__init__()
#         self.activation = nn.ReLU(inplace=True)
#
#     def Merge_Base(self, f1, f2, in_channels, out_channels, stride=1):
#         unpool = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         output = unpool(f1)
#         output = torch.cat((output, f2), 1)
#         convolution1 = nn.Conv2d(in_channels=output.size()[1], out_channels=out_channels, kernel_size=1, stride=stride)
#         convolution2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
#         normalization = nn.BatchNorm2d(num_features=out_channels)
#         output = normalization(convolution1(output))
#         return self.activation(normalization(convolution2(output)))
#
#     def Merge_Final(self, h, in_channels, out_channels, stride=1):
#         convolution = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
#         normalization = nn.BatchNorm2d(num_features=out_channels)
#         return self.activation(normalization(convolution(h)))
#
#     def forward(self, h1, f2, f3, f4):
#         logging.info("\nInput h1 dimensions before first merge: [1, channels, height, width] = {}\n".format(h1.size()))
#
#         h2 = self.Merge_Base(h1, f2, in_channels=384, out_channels=128)
#         logging.info("\nh2 dimensions after first merge: [1, channels, height, width] = {}\n".format(h2.size()))
#
#         h3 = self.Merge_Base(h2, f3, in_channels=128, out_channels= 64)
#         logging.info("\nh3 dimensions after second merge: [1, channels, height, width] = {}\n".format(h3.size()))
#
#         h4 = self.Merge_Base(h3, f4, in_channels= 64, out_channels= 32)
#         logging.info("\nh4 dimensions after third merge: [1, channels, height, width] = {}\n".format(h4.size()))
#
#         h5 = self.Merge_Final(h4, in_channels= 32, out_channels= 32)
#         logging.info("\nh5 dimensions after final convolution: [1, channels, height, width] = {}\n".format(h5.size()))
#
#         return h5

class Unet(nn.Module):
    def __init__(self, trained=False):
        super(Unet, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.unpool1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.convolution9_1 = nn.Conv2d(in_channels=640, out_channels=128, kernel_size=1, stride=1)
        self.convolution9_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.normalization9 = nn.BatchNorm2d(num_features=128)

        self.convolution10_1 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1)
        self.convolution10_2 = nn.Conv2d(in_channels= 64, out_channels=64, kernel_size=3, padding=1)
        self.normalization10 = nn.BatchNorm2d(num_features=64)

        self.convolution11_1 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, stride=1)
        self.convolution11_2 = nn.Conv2d(in_channels= 32, out_channels=32, kernel_size=3, padding=1)
        self.normalization11 = nn.BatchNorm2d(num_features=32)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features):
        logging.info("Feature Merging Layer\n")
        h1, f2, f3, f4 = features
        logging.info("\nInput h1 dimensions: [1, channels, height, width] = {}\n".format(h1.size()))

        h2 = self.unpool1(h1)
        h2 = torch.cat((h2, f2), 1)
        logging.info(h2.size())
        h2 = self.normalization9(self.convolution9_1(h2))
        h2 = self.activation(self.normalization9(self.convolution9_2(h2)))
        logging.info("\nInput h2 dimensions: [1, channels, height, width] = {}\n".format(h2.size()))

        h3 = self.unpool1(h2)
        h3 = torch.cat((h3, f3), 1)
        logging.info(h3.size())
        h3 = self.normalization10(self.convolution10_1(h3))
        h3 = self.activation(self.normalization10(self.convolution10_2(h3)))
        logging.info("\nInput h3 dimensions: [1, channels, height, width] = {}\n".format(h3.size()))

        h4 = self.unpool1(h3)
        h4 = torch.cat((h4, f4), 1)
        logging.info(h4.size())
        h4 = self.normalization11(self.convolution11_1(h4))
        h4 = self.activation(self.normalization11(self.convolution11_2(h4)))
        logging.info("\nInput h4 dimensions: [1, channels, height, width] = {}\n".format(h4.size()))

        h5 = self.activation(self.normalization11(self.convolution11_2(h4)))
        logging.info("\nInput h5 dimensions: [1, channels, height, width] = {}\n".format(h5.size()))

        return h5

        # h = self.merge(features[0], features[1], features[2], features[3])
        # return h
