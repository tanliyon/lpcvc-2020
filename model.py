from torch import nn
from PVAnet import *
from Unet import *
from QUADbox import *

class EAST(nn.Module):
    def __init__(self):
        super(EAST, self).__init__()

    def forward(self, inputs):
        return Output(Unet(PVAnet(inputs)))
