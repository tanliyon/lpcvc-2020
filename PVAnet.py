import torch
from torch import nn
import logging

class CRelu(nn.Module):
    def __init__(self):
        super(CRelu, self).__init__()
        self.activation = nn.ReLU(inplace=True)

    def CRelu_Base(self, x, in_channels, out_channels, stride=2):
        convolution1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=stride, padding=3, bias=False)
        normalization = nn.BatchNorm2d(num_features=out_channels * 2)
        output = convolution1(x)
        output = torch.cat((output, -output), 1)
        output = self.activation(normalization(output))

        pool = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
        output = pool(output)
        return output

    def CRelu_Residual(self, x, in_channels, mid_channels, out_channels, stride=1, projection=False):
        copy_input = x

        convolution1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=stride)
        normalization1 = nn.BatchNorm2d(num_features=mid_channels)
        output = self.activation(normalization1(convolution1(x)))

        convolution2 = nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, padding=1)
        normalization2 = nn.BatchNorm2d(num_features=mid_channels * 2)
        output = convolution2(output)
        output = torch.cat((output, -output), 1)
        output = self.activation(normalization2(output))

        convolution3 = nn.Conv2d(in_channels=mid_channels * 2, out_channels=out_channels, kernel_size=1)
        output = self.activation(convolution3(output))

        if projection:
            projectionConvolution = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
            copy_input = projectionConvolution(copy_input)

        return output + copy_input

    def forward(self, x):
        logging.info("\nInput x dimensions before crelu: [1, channels, height, width] = {}\n".format(x.size()))

        x1 = self.CRelu_Base(x, in_channels=x.size()[1], out_channels=16, stride=2)
        logging.info("\nx1 dimensions after crelu base: [1, channels, height, width] = {}\n".format(x1.size()))

        f4 = self.CRelu_Residual(x1, in_channels= 32, mid_channels=24, out_channels= 64, stride=1, projection=True)
        f4 = self.CRelu_Residual(f4, in_channels= 64, mid_channels=24, out_channels= 64)
        f4 = self.CRelu_Residual(f4, in_channels= 64, mid_channels=24, out_channels= 64)
        logging.info("\nf4 dimensions after crelu residual1: [1, channels, height, width] = {}\n".format(f4.size()))

        f3 = self.CRelu_Residual(f4, in_channels= 64, mid_channels=48, out_channels=128, stride=2, projection=True)
        f3 = self.CRelu_Residual(f3, in_channels=128, mid_channels=48, out_channels=128)
        f3 = self.CRelu_Residual(f3, in_channels=128, mid_channels=48, out_channels=128)
        f3 = self.CRelu_Residual(f3, in_channels=128, mid_channels=48, out_channels=128)
        logging.info("\nf3 dimensions after crelu residual2: [1, channels, height, width] = {}\n".format(f3.size()))

        return (f4, f3)

class Inception(nn.Module):
    def __init__(self):
        super(Inception, self).__init__()
        self.activation = nn.ReLU(inplace=True)

    def _Conv1x1(self, x, in_channels, out_channels=64, stride=1):
        convolution = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
        normalization = nn.BatchNorm2d(num_features=out_channels)
        return self.activation(normalization(convolution(x)))

    def _Conv3x3(self, x, in_channels, out_channels, stride=1):
        convolution = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1)
        return convolution(x)

    def Inception_Residual(self, x, in_channels, mid3x3_channels, mid5x5_channels, out_channels, pool_channels=0, stride=1, projection=False):
        copy_input = x

        x1 = self._Conv1x1(x, in_channels, stride=stride)

        normalization2 = nn.BatchNorm2d(num_features=mid3x3_channels[1])
        x2 = self._Conv1x1( x, in_channels=in_channels, out_channels=mid3x3_channels[0], stride=stride)
        x2 = self._Conv3x3(x2, in_channels=mid3x3_channels[0], out_channels=mid3x3_channels[1])
        x2 = self.activation(normalization2(x2))

        normalization3 = nn.BatchNorm2d(num_features=mid5x5_channels[2])
        x3 = self._Conv1x1( x, in_channels=in_channels, out_channels=mid5x5_channels[0], stride=stride)
        x3 = self._Conv3x3(x3, in_channels=mid5x5_channels[0], out_channels=mid5x5_channels[1])
        x3 = self._Conv3x3(x3, in_channels=mid5x5_channels[1], out_channels=mid5x5_channels[2])
        x3 = self.activation(normalization3(x3))

        output = torch.cat((x1, x2, x3), 1)

        if pool_channels > 0:
            pool = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
            x4 = self._Conv1x1(pool(x), in_channels=in_channels, out_channels=pool_channels)
            output = torch.cat((output, x4), 1)

        finalConvolution = nn.Conv2d(in_channels=64+mid3x3_channels[1]+mid5x5_channels[2]+pool_channels, out_channels=out_channels, kernel_size=1)
        output = self.activation(finalConvolution(output))

        if projection:
            projectionConvolution = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
            copy_input = projectionConvolution(copy_input)

        return output + copy_input

    def forward(self, f3):
        f2 = self.Inception_Residual(f3, in_channels=128, mid3x3_channels=[48, 128], mid5x5_channels=[24, 48, 48], out_channels=256, pool_channels=128, stride=2, projection=True)
        f2 = self.Inception_Residual(f2, in_channels=256, mid3x3_channels=[64, 128], mid5x5_channels=[24, 48, 48], out_channels=256)
        f2 = self.Inception_Residual(f2, in_channels=256, mid3x3_channels=[64, 128], mid5x5_channels=[24, 48, 48], out_channels=256)
        f2 = self.Inception_Residual(f2, in_channels=256, mid3x3_channels=[64, 128], mid5x5_channels=[24, 48, 48], out_channels=256)
        logging.info("\nf2 dimensions after inception residual1: [1, channels, height, width] = {}\n".format(f2.size()))

        f1 = self.Inception_Residual(f2, in_channels=256, mid3x3_channels=[96, 192], mid5x5_channels=[32, 64, 64], out_channels=384, pool_channels=128, stride=2, projection=True)
        f1 = self.Inception_Residual(f1, in_channels=384, mid3x3_channels=[96, 192], mid5x5_channels=[32, 64, 64], out_channels=384)
        f1 = self.Inception_Residual(f1, in_channels=384, mid3x3_channels=[96, 192], mid5x5_channels=[32, 64, 64], out_channels=384)
        f1 = self.Inception_Residual(f1, in_channels=384, mid3x3_channels=[96, 192], mid5x5_channels=[32, 64, 64], out_channels=384)
        logging.info("\nf1 dimensions after inception residual2: [1, channels, height, width] = {}\n".format(f1.size()))

        return (f2, self.activation(f1))

def PVAnet(inputs):
    crelu = CRelu()
    inception = Inception()
    logging.info("Feature Extraction Layer\n")
    f4, f3 = crelu.forward(inputs.unsqueeze(0))
    f2, f1 = inception.forward(f3)
    return (f1, f2, f3, f4)
