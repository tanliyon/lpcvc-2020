from torch import nn
from Models.PVAnet import *
from Models.Unet import *
from Models.QUADbox import *

class EAST(nn.Module):
    def __init__(self):
        super(EAST, self).__init__()

    def forward(self, inputs):
        return Output(Unet(PVAnet(inputs)))
