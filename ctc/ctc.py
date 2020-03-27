# Implemented based on: https://github.com/BelBES/crnn-pytorch
import os
import string
import torch
import numpy as np
import argparse
import torch.nn as nn
import torch.optim as optim
from matplotlib.path import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from model import CRNN

# Model's weights file relative path
MODEL_PATH = "ctc.pth"

def ctc_recognition(frames, bboxes):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	transform = transforms.compose([
		transforms.Resize((50, 600)),
		transforms.ToTensor()])
	load_path = os.path.join(os.getcwd(), MODEL_PATH)

	net = CRNN()
	net.decode = True
	net.to(device)
	net.load_state_dict(torch.load(load_path))
	preds = []

	for frame in frames:
		words = []
		for box in bboxes:
			tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y = box
			top = min(tl_y, tr_y)
			left = min(tl_x, bl_x)
			height = max(bl_y, br_y) - min(tl_y, tr_y)
			width = max(tr_x, br_x) - min(tl_x, bl_x)
			words.append(transforms.functional.crop(frame, top, left, height, width))
		transform(words)
		words.to(device)
		preds.append(net(words))

	return preds
