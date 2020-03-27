import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms
import numpy as np
from PIL import Image
from model import *
from detect import *
from Models.PVAnet import *
from Models.Unet import *
from Models.QUADbox import *

import logging

def main():
    logging.basicConfig(filename="EAST.log", level=logging.INFO)
    logging.info("\nStarted")
    batch_size = 2
    training_path = "./Samples/"
    transform = transforms.Compose([
                    transforms.Grayscale(num_output_channels=1),
                    transforms.Resize((126, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])
    training_data = torchvision.datasets.ImageFolder(root=training_path, transform=transform)
    training_loader = data.DataLoader(training_data, batch_size=batch_size, shuffle=True,  num_workers=4)

    device = torch.device("cpu")
    model = EAST()
    model.to(device)
    optimizer = optim.Adam(list(model.parameters()), lr=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2 // 2], gamma=0.1)

    for i, train in enumerate(training_data, 0):
        logging.info("Image #{}\n".format(i))
        inputs, labels = train
        # features = extractor(inputs)
        # h = merger(features)
        # score_map, geometry_map = output(h)
        score_map, geometry_map = model(inputs)
        # box = detect(score_map, geometry_map)
        # plot_img = plot_boxes(torchvision.transforms.ToPILImage()(inputs), box)
        # plot_img.show()
        logging.info("-------------------------------------------------")

    logging.info("Finished\n")

if __name__ == '__main__':
    main()
