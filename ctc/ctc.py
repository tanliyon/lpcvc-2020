# Implemented based on: https://github.com/BelBES/crnn-pytorch
import os
import string
import torch
import numpy as np
import argparse
import torch.nn as nn
import torch.optim as optim
# from matplotlib.path import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from ctc.model import CRNN

# Model's weights file relative path
MODEL_PATH = "ctc.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
	transforms.Resize((50, 600)),
	transforms.Grayscale(),
	transforms.ToTensor()])
to_pil = transforms.ToPILImage()
load_path = os.path.join(os.getcwd(), MODEL_PATH)

def ctc_recognition(frames, bboxes):
	net = CRNN(pretrained=False)
	net.decode = True
	net = net.to(device)
	net.load_state_dict(torch.load(load_path, map_location=device))
	preds = []

	for frame, frame_bboxes in zip(frames, bboxes):
		words = []
		pred_frame = []
		frame = to_pil(frame)
		for box in frame_bboxes:
			tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y = box
			top = min(tl_y, tr_y)
			left = min(tl_x, bl_x)
			height = max(bl_y, br_y) - min(tl_y, tr_y)
			width = max(tr_x, br_x) - min(tl_x, bl_x)
			words.append(transform(transforms.functional.crop(frame, top, left, height, width)))

		words = torch.stack(words)
		words = words.to(device)
		preds.append(net(words))

	return preds

if __name__ == "__main__":
	from TextLocalizationSet import collate
	from TextLocalizationSet import TextLocalizationSet

	infer_set = TextLocalizationSet(train=True, transform=None)
	infer_loader = DataLoader(infer_set, batch_size=4,
                             shuffle=True, num_workers=1,
                             collate_fn=collate)

	for i, data in enumerate(infer_loader):
		img, bbox, trans = data

		bbox_data = []
		for k in range(len(bbox)):
			bbox_frame = []
			for j in range(len(bbox[k])):
				tl, tr, br, bl = bbox[k][j]
				tl_x, tl_y = tl
				tr_x, tr_y = tr
				br_x, br_y = br
				bl_x, bl_y = bl
				bbox_frame.append([tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y])
			bbox_data.append(bbox_frame)

		print(ctc_recognition(img, bbox_data))
		print(trans)

		if i==3:
			break
