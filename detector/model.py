from torch import nn
from detector.Models.PVAnet import *
from detector.Models.Unet import *
from detector.Models.Output import *

class EAST(nn.Module):
    def __init__(self):
        super(EAST, self).__init__()
        self.extractor = PVAnet()
        self.merger = Unet()
        self.output = Output()

    def forward(self, inputs):
        return self.output(self.merger(self.extractor(inputs)))
