from torch import nn
from Models.PVAnet import *
from Models.Unet import *
from Models.Output import *

class EAST(nn.Module):
    def __init__(self):
        super(EAST, self).__init__()
        self.extractor = PVAnet()
        self.merger = Unet()
        self.output = Output()

    def forward(self, inputs):
        return self.output(self.merger(self.extractor(inputs)))
