import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms
import numpy as np
from PIL import Image
from PVAnet import *
from Unet import *

import logging

def main():
    logging.basicConfig(filename="UnetTest.log", level=logging.INFO)
    logging.info("\nStarted")
    batch_size = 2
    training_path = "./Samples/"
    transform = transforms.Compose([
                    transforms.Resize((192, 192)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
    training_data = torchvision.datasets.ImageFolder(root=training_path, transform=transform)
    training_loader = data.DataLoader(training_data, batch_size=batch_size, shuffle=True,  num_workers=4)

    for i, train in enumerate(training_data, 0):
        logging.info("Image #{}\n".format(i))
        inputs, labels = train
        features = PVAnet(inputs)
        output = Unet(features)
        logging.info("-------------------------------------------------")

    logging.info("Finished\n")

if __name__ == '__main__':
    main()
