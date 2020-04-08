import torch
import torchvision
from torchvision import transforms
from model import *
from detect import *

# MODEL_PATH = "detector.pth"
MODEL_PATH = "./Dataset/Train/TrainEpoch/model_epoch_600.pth"

def detection(frames_path):
    transform = transforms.Compose([
                    transforms.Grayscale(num_output_channels=1),
                    transforms.Resize((126, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5,), std=(0.5,))
                ])
    original_data = torchvision.datasets.ImageFolder(root=frames_path)
    frames_data = torchvision.datasets.ImageFolder(root=frames_path, transform=transform)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    model = EAST()
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    frames = []
    boxes = []

    for i, frame in enumerate(frames_data, 0):
        input, label = frame
        with torch.no_grad():
            score_map, geometry_map = model(input.to(device))
        box = get_boxes(score_map.squeeze(0).cpu().numpy(), geometry_map.squeeze(0).cpu().numpy())

        if not box.all():
            continue

        # box = detect(score_map, geometry_map)
        frames.append(input)
        boxes.append(box)

        image, _ = original_data[i]
        old_height, old_width = image.height, image.width
        new_height, new_width = 126, 224
        ratio_height, ratio_width = new_height / old_height, new_width / old_width

        resized_image = image.resize((new_width, new_height), Image.BILINEAR)
        plot_img = plot_boxes(resized_image, box)
        #plot_img = plot_boxes(torchvision.transforms.ToPILImage()(input), box)
        plot_img.show()
    print(boxes)

    return frames, boxes

if __name__ == '__main__':
    detection("./Samples/")
