import torch
from torch import nn
import logging

# class CRelu(nn.Module):
#     def __init__(self):
#         super(CRelu, self).__init__()
#         self.activation = nn.ReLU(inplace=True)
#
#     def CRelu_Base(self, x, in_channels, out_channels, stride=2):
#         convolution1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=stride, padding=3, bias=False)
#         normalization = nn.BatchNorm2d(num_features=out_channels * 2)
#         output = convolution1(x)
#         output = torch.cat((output, -output), 1)
#         output = self.activation(normalization(output))
#
#         pool = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
#         output = pool(output)
#         return output
#
#     def CRelu_Residual(self, x, in_channels, mid_channels, out_channels, stride=1, projection=False):
#         copy_input = x
#
#         convolution1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=stride)
#         normalization1 = nn.BatchNorm2d(num_features=mid_channels)
#         output = self.activation(normalization1(convolution1(x)))
#
#         convolution2 = nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, padding=1)
#         normalization2 = nn.BatchNorm2d(num_features=mid_channels * 2)
#         output = convolution2(output)
#         output = torch.cat((output, -output), 1)
#         output = self.activation(normalization2(output))
#
#         convolution3 = nn.Conv2d(in_channels=mid_channels * 2, out_channels=out_channels, kernel_size=1)
#         output = self.activation(convolution3(output))
#
#         if projection:
#             projectionConvolution = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
#             copy_input = projectionConvolution(copy_input)
#
#         return output + copy_input
#
#     def forward(self, x):
#         logging.info("\nInput x dimensions before crelu: [1, channels, height, width] = {}\n".format(x.size()))
#
#         x1 = self.CRelu_Base(x, in_channels=x.size()[1], out_channels=16, stride=2)
#         logging.info("\nx1 dimensions after crelu base: [1, channels, height, width] = {}\n".format(x1.size()))
#
#         f4 = self.CRelu_Residual(x1, in_channels= 32, mid_channels=24, out_channels= 64, stride=1, projection=True)
#         f4 = self.CRelu_Residual(f4, in_channels= 64, mid_channels=24, out_channels= 64)
#         f4 = self.CRelu_Residual(f4, in_channels= 64, mid_channels=24, out_channels= 64)
#         logging.info("\nf4 dimensions after crelu residual1: [1, channels, height, width] = {}\n".format(f4.size()))
#
#         f3 = self.CRelu_Residual(f4, in_channels= 64, mid_channels=48, out_channels=128, stride=2, projection=True)
#         f3 = self.CRelu_Residual(f3, in_channels=128, mid_channels=48, out_channels=128)
#         f3 = self.CRelu_Residual(f3, in_channels=128, mid_channels=48, out_channels=128)
#         f3 = self.CRelu_Residual(f3, in_channels=128, mid_channels=48, out_channels=128)
#         logging.info("\nf3 dimensions after crelu residual2: [1, channels, height, width] = {}\n".format(f3.size()))
#
#         return (f4, f3)
#
# class Inception(nn.Module):
#     def __init__(self):
#         super(Inception, self).__init__()
#         self.activation = nn.ReLU(inplace=True)
#
#     def _Conv1x1(self, x, in_channels, out_channels=64, stride=1):
#         convolution = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
#         normalization = nn.BatchNorm2d(num_features=out_channels)
#         return self.activation(normalization(convolution(x)))
#
#     def _Conv3x3(self, x, in_channels, out_channels, stride=1):
#         convolution = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1)
#         return convolution(x)
#
#     def Inception_Residual(self, x, in_channels, mid3x3_channels, mid5x5_channels, out_channels, pool_channels=0, stride=1, projection=False):
#         copy_input = x
#
#         x1 = self._Conv1x1(x, in_channels, stride=stride)
#
#         normalization2 = nn.BatchNorm2d(num_features=mid3x3_channels[1])
#         x2 = self._Conv1x1( x, in_channels=in_channels, out_channels=mid3x3_channels[0], stride=stride)
#         x2 = self._Conv3x3(x2, in_channels=mid3x3_channels[0], out_channels=mid3x3_channels[1])
#         x2 = self.activation(normalization2(x2))
#
#         normalization3 = nn.BatchNorm2d(num_features=mid5x5_channels[2])
#         x3 = self._Conv1x1( x, in_channels=in_channels, out_channels=mid5x5_channels[0], stride=stride)
#         x3 = self._Conv3x3(x3, in_channels=mid5x5_channels[0], out_channels=mid5x5_channels[1])
#         x3 = self._Conv3x3(x3, in_channels=mid5x5_channels[1], out_channels=mid5x5_channels[2])
#         x3 = self.activation(normalization3(x3))
#
#         output = torch.cat((x1, x2, x3), 1)
#
#         if pool_channels > 0:
#             pool = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
#             x4 = self._Conv1x1(pool(x), in_channels=in_channels, out_channels=pool_channels)
#             output = torch.cat((output, x4), 1)
#
#         finalConvolution = nn.Conv2d(in_channels=64+mid3x3_channels[1]+mid5x5_channels[2]+pool_channels, out_channels=out_channels, kernel_size=1)
#         output = self.activation(finalConvolution(output))
#
#         if projection:
#             projectionConvolution = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
#             copy_input = projectionConvolution(copy_input)
#
#         return output + copy_input
#
#     def forward(self, f3):
#         f2 = self.Inception_Residual(f3, in_channels=128, mid3x3_channels=[48, 128], mid5x5_channels=[24, 48, 48], out_channels=256, pool_channels=128, stride=2, projection=True)
#         f2 = self.Inception_Residual(f2, in_channels=256, mid3x3_channels=[64, 128], mid5x5_channels=[24, 48, 48], out_channels=256)
#         f2 = self.Inception_Residual(f2, in_channels=256, mid3x3_channels=[64, 128], mid5x5_channels=[24, 48, 48], out_channels=256)
#         f2 = self.Inception_Residual(f2, in_channels=256, mid3x3_channels=[64, 128], mid5x5_channels=[24, 48, 48], out_channels=256)
#         logging.info("\nf2 dimensions after inception residual1: [1, channels, height, width] = {}\n".format(f2.size()))
#
#         f1 = self.Inception_Residual(f2, in_channels=256, mid3x3_channels=[96, 192], mid5x5_channels=[32, 64, 64], out_channels=384, pool_channels=128, stride=2, projection=True)
#         f1 = self.Inception_Residual(f1, in_channels=384, mid3x3_channels=[96, 192], mid5x5_channels=[32, 64, 64], out_channels=384)
#         f1 = self.Inception_Residual(f1, in_channels=384, mid3x3_channels=[96, 192], mid5x5_channels=[32, 64, 64], out_channels=384)
#         f1 = self.Inception_Residual(f1, in_channels=384, mid3x3_channels=[96, 192], mid5x5_channels=[32, 64, 64], out_channels=384)
#         logging.info("\nf1 dimensions after inception residual2: [1, channels, height, width] = {}\n".format(f1.size()))
#
#         return (f2, self.activation(f1))

class PVAnet(nn.Module):
    def __init__(self, trained=False):
        super(PVAnet, self).__init__()
        self.activation = nn.ReLU(inplace=True)

        # CRelu Base
        self.convolution1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=7, stride=2, padding=3, bias=False)
        self.normalization1 = nn.BatchNorm2d(num_features=16 * 2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # CRelu Residual Layer 1
        self.convolution2_1 = nn.Conv2d(in_channels= 32, out_channels=24, kernel_size=1, stride=1)
        self.convolution2_2 = nn.Conv2d(in_channels= 64, out_channels=24, kernel_size=1, stride=1)
        self.normalization2_1 = nn.BatchNorm2d(num_features=24)

        self.convolution3_1 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, padding=1)
        self.normalization3_1 = nn.BatchNorm2d(num_features=24 * 2)

        self.convolution4_1 = nn.Conv2d(in_channels=24 * 2, out_channels= 64, kernel_size=1)

        self.projectionConvolution1 = nn.Conv2d(in_channels=32, out_channels= 64, kernel_size=1, stride=1)

        # CRelu Residual Layer 2
        self.convolution2_3 = nn.Conv2d(in_channels= 64, out_channels=48, kernel_size=1, stride=2)
        self.convolution2_4 = nn.Conv2d(in_channels=128, out_channels=48, kernel_size=1, stride=1)
        self.normalization2_3 = nn.BatchNorm2d(num_features=48)

        self.convolution3_2 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, padding=1)
        self.normalization3_2 = nn.BatchNorm2d(num_features=48 * 2)

        self.convolution4_2 = nn.Conv2d(in_channels=48 * 2, out_channels=128, kernel_size=1)

        self.projectionConvolution2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2)

        # Inception Layer 1
        self.convolution5_1 = nn.Conv2d(in_channels=128, out_channels= 64, kernel_size=1, stride=2)
        self.convolution5_2 = nn.Conv2d(in_channels=128, out_channels= 48, kernel_size=1, stride=2)
        self.convolution5_3 = nn.Conv2d(in_channels=128, out_channels= 24, kernel_size=1, stride=2)
        self.convolution5_4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1)
        self.convolution5_5 = nn.Conv2d(in_channels=256, out_channels= 64, kernel_size=1, stride=1)
        self.convolution5_6 = nn.Conv2d(in_channels=256, out_channels= 64, kernel_size=1, stride=1)
        self.convolution5_7 = nn.Conv2d(in_channels=256, out_channels= 24, kernel_size=1, stride=1)
        self.normalization5_1 = nn.BatchNorm2d(num_features= 64)
        self.normalization5_2 = nn.BatchNorm2d(num_features= 48)
        self.normalization5_3 = nn.BatchNorm2d(num_features= 24)
        self.normalization5_4 = nn.BatchNorm2d(num_features=128)

        self.convolution6_1 = nn.Conv2d(in_channels=48, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.convolution6_2 = nn.Conv2d(in_channels=24, out_channels= 48, kernel_size=3, stride=1, padding=1)
        self.convolution6_3 = nn.Conv2d(in_channels=48, out_channels= 48, kernel_size=3, stride=1, padding=1)
        self.convolution6_4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.finalConvolution1 = nn.Conv2d(in_channels=64+128+48+128, out_channels=256, kernel_size=1)
        self.finalConvolution2 = nn.Conv2d(in_channels=64+128+48, out_channels=256, kernel_size=1)

        self.projectionConvolution3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2)

        # Inception Layer 2
        self.convolution7_1 = nn.Conv2d(in_channels=256, out_channels= 64, kernel_size=1, stride=2)
        self.convolution7_2 = nn.Conv2d(in_channels=256, out_channels= 96, kernel_size=1, stride=2)
        self.convolution7_3 = nn.Conv2d(in_channels=256, out_channels= 32, kernel_size=1, stride=2)
        self.convolution7_4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1)
        self.convolution7_5 = nn.Conv2d(in_channels=384, out_channels= 64, kernel_size=1, stride=1)
        self.convolution7_6 = nn.Conv2d(in_channels=384, out_channels= 96, kernel_size=1, stride=1)
        self.convolution7_7 = nn.Conv2d(in_channels=384, out_channels= 32, kernel_size=1, stride=1)
        self.normalization7_1 = nn.BatchNorm2d(num_features= 96)
        self.normalization7_2 = nn.BatchNorm2d(num_features= 32)

        self.convolution8_1 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.convolution8_2 = nn.Conv2d(in_channels=32, out_channels= 64, kernel_size=3, stride=1, padding=1)
        self.convolution8_3 = nn.Conv2d(in_channels=64, out_channels= 64, kernel_size=3, stride=1, padding=1)
        self.normalization8_1 = nn.BatchNorm2d(num_features=192)

        self.finalConvolution3 = nn.Conv2d(in_channels=64+192+64+128, out_channels=384, kernel_size=1)
        self.finalConvolution4 = nn.Conv2d(in_channels=64+192+64, out_channels=384, kernel_size=1)

        self.projectionConvolution4 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=1, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, projection=False):
        # logging.info("Feature Extraction Layer\n")
        # logging.info("\nx dimensions: [1, channels, height, width] = {}\n".format(x.size()))

        # CRelu Base Layer
        #x1 = self.convolution1(x.unsqueeze(0))
        x1 = self.convolution1(x)
        x1 = torch.cat((x1, -x1), 1)
        x1 = self.activation(self.normalization1(x1))
        x1 = self.pool1(x1)
        # logging.info("\nx1 dimensions: [1, channels, height, width] = {}\n".format(x1.size()))

        # CRelu Residual Layer 1
        x2 = self.activation(self.normalization2_1(self.convolution2_1(x1)))
        x2 = self.convolution3_1(x2)
        x2 = torch.cat((x2, -x2), 1)
        x2 = self.activation(self.normalization3_1(x2))
        x2 = self.activation(self.convolution4_1(x2))
        x2 = x2 + self.projectionConvolution1(x1)

        x3 = self.activation(self.normalization2_1(self.convolution2_2(x2)))
        x3 = self.convolution3_1(x3)
        x3 = torch.cat((x3, -x3), 1)
        x3 = self.activation(self.normalization3_1(x3))
        x3 = self.activation(self.convolution4_1(x3))
        x3 = x3 + x2

        f4 = self.activation(self.normalization2_1(self.convolution2_2(x3)))
        f4 = self.convolution3_1(f4)
        f4 = torch.cat((f4, -f4), 1)
        f4 = self.activation(self.normalization3_1(f4))
        f4 = self.activation(self.convolution4_1(f4))
        f4 = f4 + x3
        # logging.info("\nf4 dimensions: [1, channels, height, width] = {}\n".format(f4.size()))

        # CRelu Residual Layer 2
        x2 = self.activation(self.normalization2_3(self.convolution2_3(f4)))
        x2 = self.convolution3_2(x2)
        x2 = torch.cat((x2, -x2), 1)
        x2 = self.activation(self.normalization3_2(x2))
        x2 = self.activation(self.convolution4_2(x2))
        x2 = x2 + self.projectionConvolution2(f4)

        x3 = self.activation(self.normalization2_3(self.convolution2_4(x2)))
        x3 = self.convolution3_2(x3)
        x3 = torch.cat((x3, -x3), 1)
        x3 = self.activation(self.normalization3_2(x3))
        x3 = self.activation(self.convolution4_2(x3))
        x3 = x3 + x2

        x4 = self.activation(self.normalization2_3(self.convolution2_4(x3)))
        x4 = self.convolution3_2(x4)
        x4 = torch.cat((x4, -x4), 1)
        x4 = self.activation(self.normalization3_2(x4))
        x4 = self.activation(self.convolution4_2(x4))
        x4 = x4 + x3

        f3 = self.activation(self.normalization2_3(self.convolution2_4(x4)))
        f3 = self.convolution3_2(f3)
        f3 = torch.cat((f3, -f3), 1)
        f3 = self.activation(self.normalization3_2(f3))
        f3 = self.activation(self.convolution4_2(f3))
        f3 = f3 + x4
        # logging.info("\nf3 dimensions: [1, channels, height, width] = {}\n".format(f3.size()))

        # Inception Layer 1
        x1 = self.activation(self.normalization5_1(self.convolution5_1(f3)))

        x2 = self.activation(self.normalization5_2(self.convolution5_2(f3)))
        x2 = self.activation(self.normalization5_4(self.convolution6_1(x2)))

        x3 = self.activation(self.normalization5_3(self.convolution5_3(f3)))
        x3 = self.activation(self.normalization5_2(self.convolution6_3(self.convolution6_2(x3))))

        x4 = self.activation(self.normalization5_4(self.convolution5_4(self.pool1(f3))))

        x5 = torch.cat((x1, x2, x3, x4), 1)
        x5 = self.activation(self.finalConvolution1(x5))
        x5 = x5 + self.projectionConvolution3(f3)

        x1 = self.activation(self.normalization5_1(self.convolution5_5(x5)))

        x2 = self.activation(self.normalization5_1(self.convolution5_6(x5)))
        x2 = self.activation(self.normalization5_4(self.convolution6_4(x2)))

        x3 = self.activation(self.normalization5_3(self.convolution5_7(x5)))
        x3 = self.activation(self.normalization5_2(self.convolution6_3(self.convolution6_2(x3))))

        x4 = torch.cat((x1, x2, x3), 1)
        x4 = self.activation(self.finalConvolution2(x4))
        x4 = x4 + x5

        x1 = self.activation(self.normalization5_1(self.convolution5_5(x4)))

        x2 = self.activation(self.normalization5_1(self.convolution5_6(x4)))
        x2 = self.activation(self.normalization5_4(self.convolution6_4(x2)))

        x3 = self.activation(self.normalization5_3(self.convolution5_7(x4)))
        x3 = self.activation(self.normalization5_2(self.convolution6_3(self.convolution6_2(x3))))

        x5 = torch.cat((x1, x2, x3), 1)
        x5 = self.activation(self.finalConvolution2(x5))
        x5 = x5 + x4

        x1 = self.activation(self.normalization5_1(self.convolution5_5(x5)))

        x2 = self.activation(self.normalization5_1(self.convolution5_6(x5)))
        x2 = self.activation(self.normalization5_4(self.convolution6_4(x2)))

        x3 = self.activation(self.normalization5_3(self.convolution5_7(x5)))
        x3 = self.activation(self.normalization5_2(self.convolution6_3(self.convolution6_2(x3))))

        f2 = torch.cat((x1, x2, x3), 1)
        f2 = self.activation(self.finalConvolution2(f2))
        f2 = f2 + x5
        # logging.info("\nf2 dimensions: [1, channels, height, width] = {}\n".format(f2.size()))

        # Inception Layer 2
        x1 = self.activation(self.normalization5_1(self.convolution7_1(f2)))

        x2 = self.activation(self.normalization7_1(self.convolution7_2(f2)))
        x2 = self.activation(self.normalization8_1(self.convolution8_1(x2)))

        x3 = self.activation(self.normalization7_2(self.convolution7_3(f2)))
        x3 = self.activation(self.normalization5_1(self.convolution8_3(self.convolution8_2(x3))))

        x4 = self.activation(self.normalization5_4(self.convolution7_4(self.pool1(f2))))

        x5 = torch.cat((x1, x2, x3, x4), 1)
        x5 = self.activation(self.finalConvolution3(x5))
        x5 = x5 + self.projectionConvolution4(f2)

        x1 = self.activation(self.normalization5_1(self.convolution7_5(x5)))

        x2 = self.activation(self.normalization7_1(self.convolution7_6(x5)))
        x2 = self.activation(self.normalization8_1(self.convolution8_1(x2)))

        x3 = self.activation(self.normalization7_2(self.convolution7_7(x5)))
        x3 = self.activation(self.normalization5_1(self.convolution8_3(self.convolution8_2(x3))))

        x4 = torch.cat((x1, x2, x3), 1)
        x4 = self.activation(self.finalConvolution4(x4))
        x4 = x4 + x5

        x1 = self.activation(self.normalization5_1(self.convolution7_5(x4)))

        x2 = self.activation(self.normalization7_1(self.convolution7_6(x4)))
        x2 = self.activation(self.normalization8_1(self.convolution8_1(x2)))

        x3 = self.activation(self.normalization7_2(self.convolution7_7(x4)))
        x3 = self.activation(self.normalization5_1(self.convolution8_3(self.convolution8_2(x3))))

        x5 = torch.cat((x1, x2, x3), 1)
        x5 = self.activation(self.finalConvolution4(x5))
        x5 = x5 + x4

        x1 = self.activation(self.normalization5_1(self.convolution7_5(x5)))

        x2 = self.activation(self.normalization7_1(self.convolution7_6(x5)))
        x2 = self.activation(self.normalization8_1(self.convolution8_1(x2)))

        x3 = self.activation(self.normalization7_2(self.convolution7_7(x5)))
        x3 = self.activation(self.normalization5_1(self.convolution8_3(self.convolution8_2(x3))))

        f1 = torch.cat((x1, x2, x3), 1)
        f1 = self.activation(self.finalConvolution4(f1))
        f1 = f1 + x5
        # logging.info("\nf1 dimensions: [1, channels, height, width] = {}\n".format(f1.size()))

        return (f1, f2, f3, f4)
        # logging.info("Feature Extraction Layer\n")
        # f4, f3 = self.crelu(inputs.unsqueeze(0))
        # f2, f1 = self.inception(f3)
        # return (f1, f2, f3, f4)
