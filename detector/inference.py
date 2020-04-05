import torch
import torchvision
import numpy as np
from torchvision import transforms
from detector.model import *
from detector.detect import *

MODEL_PATH = "./detector.pth"

def detection(frames_path):
    transform = transforms.Compose([
                    transforms.Grayscale(num_output_channels=1),
                    transforms.Resize((126, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5,), std=(0.5,))
                ])
    frames_data = torchvision.datasets.ImageFolder(root=frames_path, transform=transform)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    model = EAST()
    #model.load_state_dict(checkpoint["model_state_dict"])
    # model = model.to(device)

    frames = []
    boxes = []

    for i, frame in enumerate(frames_data):
        input, label = frame
        with torch.no_grad():
            score_map, geometry_map = model(input.to(device))
        box = get_boxes(score_map.squeeze(0).cpu().numpy(), geometry_map.squeeze(0).cpu().numpy())
        # box = detect(score_map, geometry_map, device)
        
        if type(box).__module__ != np.__name__ or len(box) == 0:
            continue
        
        frames.append(input)
        boxes.append(box)
        # plot_img = plot_boxes(torchvision.transforms.ToPILImage()(input), box)
        # plot_img.show()

    return frames, boxes

if __name__ == '__main__':
    detection("./Samples/")
