# Implemented based on: https://github.com/BelBES/crnn-pytorch
import os
import torch
import numpy as np
import mpipe
from torchvision import transforms
from ctc.model import CRNN

class CTC(mpipe.OrderedWorker):
	def __init__(self):
		MODEL_PATH = "ctc.pth"
		self.transform = transforms.Compose([
							transforms.Resize((50, 600)),
							transforms.ToTensor(),
							transforms.Normalize(mean=(0.5,), std=(0.5,))
							])
		self.to_pil = transforms.ToPILImage()
		self.net = CRNN(pretrained=False)
		self.net.decode = True
		self.net.eval()
		self.net.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
	
	def doTask(self, detection):
		frame, bboxes = detection
		words = []
		
		frame = frame[0]
		frame = self.to_pil(frame)
		bboxes = bboxes[0]
		
		if type(bboxes).__module__ != np.__name__ or len(bboxes) == 0:
			print("CTC done")
			return []
		
		for box in bboxes:
			words.append(self.crop(frame, box))
		words = torch.stack(words)
		
		print("CTC done")
		with torch.no_grad():
			return self.net(words)
		
	def crop(self, frame, box):
		tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y = box
		top = min(tl_y, tr_y)
		left = min(tl_x, bl_x)
		height = max(bl_y, br_y) - min(tl_y, tr_y)
		width = max(tr_x, br_x) - min(tl_x, bl_x)
		return self.transform(transforms.functional.crop(frame, top, left, height, width))

	
