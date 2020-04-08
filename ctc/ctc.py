# Implemented based on: https://github.com/BelBES/crnn-pytorch
import os
import string
import torch
import numpy as np
from torchvision import transforms
from ctc.model import CRNN

def crop(img, box, transform):
    tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y = box
    top = min(tl_y, tr_y)
    left = min(tl_x, bl_x)
    height = max(bl_y, br_y) - min(tl_y, tr_y)
    width = max(tr_x, br_x) - min(tl_x, bl_x)
    return transform(transforms.functional.crop(img, top, left, height, width))

def init_CTC():
    OCR = CRNN(pretrained=False)
    OCR.decode = True
    OCR.load_state_dict(torch.load('ctc.pth', map_location='cpu'))
    return OCR.eval()

def CTC_OCR(OCR, img, boxes):
    ctc_transform = transforms.Compose([
        transforms.Resize((50, 600)),
        transforms.ToTensor()
    ])
    
    to_pil = transforms.ToPILImage()
    pil_img = []
    
    for im in img:
        pil_img.append(to_pil(im))
    
    words = []
    count = []
    
    for i, box in enumerate(boxes):
        if type(box).__module__ != np.__name__ or len(box) == 0:
        	continue	
        count.append(len(box))
        for b in box:
            words.append(crop(pil_img[i], b, ctc_transform))
    
    if not words:
    	return ([], [])
    	
    words = torch.stack(words)
    with torch.no_grad():
        predictions = OCR(words.unsqueeze(0) if len(words.shape) == 3 else words)
    
    return predictions, count
