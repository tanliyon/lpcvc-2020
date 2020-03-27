import torch
from torchvision import  models
import torch.nn as nn
from data_manager.data_manager import textDataset
from utils.config import CONFIG
import PIL.Image as Image
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy
from matplotlib import pyplot as plt


class CNN(nn.Module):
    def __init__(self, imgH, nc, nh):
        super(CNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        model = models.mobilenet_v2()
        model.load_state_dict(torch.load("output/mobile_net_v2.pth"), strict=False)
        modules = model.features
        model = nn.Sequential(*modules)
        # self.inception = models.inception_v3(pretrained=True)
        self.cnn = nn.Sequential(*modules)


        self.rnn = nn.Sequential(
            BidirectionalLSTM(1280, nh, nh),
            BidirectionalLSTM(nh, nh, nh))

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features calculate
        encoder_outputs = self.rnn(conv)  # seq * batch * n_classes// 25 × batchsize × 256（隐藏节点个数）

        return encoder_outputs

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output



# # transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
# transform = transforms.Compose([
#                                 transforms.Resize((32, 280), interpolation=Image.NEAREST), transforms.ToTensor()])
# dataset = textDataset("gt_new.txt", "/home/damini/Documents/icdar_2015/train", "data", "train", transform)
# val_dataset = textDataset("gt_new.txt", "/home/damini/Documents/icdar_2015/val", "data", "val", transform)
# train_loader = DataLoader(dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
# # 71, 16, 256
# model_try = CNN(32, 1, 265)
# out = model_try(next(iter(val_loader))[0])
# print("done")
# # mobile.load_state_dict(torch.load("output/mobile_net_v2.pth"), strict=False)
# # # torch.save(mobile.state_dict(), "output/mobile_net_v2.pth")
# # print("done")
