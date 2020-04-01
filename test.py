import os
import torch
import torchvision
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
from model import *
from detect import *
from error import *

import logging
import random

def main(image_directory_path, model_path):
    logging.basicConfig(filename="test_1.log", level=logging.INFO)
    logging.info("\nStarted")
    # transform = transforms.Compose([
    #                 transforms.Grayscale(num_output_channels=1),
    #                 transforms.Resize((126, 224)),
    #                 transforms.ToTensor(),
    #                 transforms.Normalize((0.5,), (0.5,))
    #             ])
    # training_data = torchvision.datasets.ImageFolder(root=training_path, transform=transform)
    # training_loader = data.DataLoader(training_data, batch_size=batch_size, shuffle=True,  num_workers=4)
    does_directory_exist(image_directory_path)
    image_paths = os.listdir(image_directory_path)
    image_paths = random.sample(image_paths, len(image_paths))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    model.load_state_dict(torch.load(model_path))

    for i, image_path in enumerate(image_paths):
        model.eval()
        logging.info("Image #{} path {}\n".format(i, image_path))
        image = Image.open(image_directory_path + image_path)
        image = image.convert("L")
        check_number_of_channels(image)
        boxes = detect(image, model, device)
        plot_image = plot_boxes(image, boxes)
        plot_image.show()
        # score_map, geometry_map = model(image.to(device))
        # box = detect(score_map, geometry_map)
        # plot_img = plot_boxes(torchvision.transforms.ToPILImage()(inputs), box)
        # plot_img.show()
        logging.info("-------------------------------------------------")
        break

    logging.info("Finished\n")

if __name__ == '__main__':
    image_directory_path = "./Dataset/Test/TestImages/"
    model_path = "./Dataset/Train/TrainEpoch/model_epoch_400.pth"
    main(image_directory_path, model_path)
