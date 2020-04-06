import torch
import torchvision
import numpy as np
from torchvision import transforms
from detector.model import *
from detector.detect import *

def init_EAST():
    detector = EAST()
    model_weights = torch.load('detector.pth', map_location='cpu')
    detector.load_state_dict(model_weights['model_state_dict'])
    return detector.eval()

def detect(detector, img):
    with torch.no_grad():
        score_map, geometry_map = detector(img.unqueeze(0) if len(img.shape) == 3 else img)
    
    boxes = []
    for i in range(score_map.shape[0]):
        boxes.append(get_boxes(score_map[i].numpy(), geometry_map[i].numpy()))
    
    return boxes

if __name__ == '__main__':
    detection("./Samples/")
