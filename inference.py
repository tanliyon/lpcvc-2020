import torch
import torchvision
from torchvision import transforms
from model import *
from detect import *

def detection(frames_path):
    transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
                ])
    frames_data = torchvision.datasets.ImageFolder(root=frames_path, transform=transform)

    model = EAST()

    frames = []
    boxes = []

    for i, frame in enumerate(frames_data, 0):
        input, label = frame
        score_map, geometry_map = model(input)
        box = detect(score_map, geometry_map)
        frames.append(input)
        boxes.append(box)
        # plot_img = plot_boxes(torchvision.transforms.ToPILImage()(input), boxes)
        # plot_img.show()

    return frames, boxes

if __name__ == '__main__':
    detection("./Samples/")
