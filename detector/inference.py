import torch
import torchvision
from torchvision import transforms
from detector.model import *
from detector.detect import *
MODEL_PATH = "detector.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def detection(frames_path):
    transform = transforms.Compose([
                    transforms.Grayscale(num_output_channels=1),
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5,), std=(0.5,))
                ])
    frames_data = torchvision.datasets.ImageFolder(root=frames_path, transform=transform)
    
    model = EAST()
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    frames = []
    boxes = []

    for i, frame in enumerate(frames_data, 0):
        inp, _ = frame
        inp = inp.to(device)
        score_map, geometry_map = model(inp)
        box = detect(score_map, geometry_map)
        frames.append(inp)
        boxes.append(box)
        # plot_img = plot_boxes(torchvision.transforms.ToPILImage()(input), boxes)
        # plot_img.show()

    return frames, boxes

if __name__ == '__main__':
    detection("./Samples/")
