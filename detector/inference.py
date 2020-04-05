import torch
import torchvision
import mpipe
from detector.model import *
from detector.detect import *

class Detector(mpipe.OrderedWorker):
	def __init__(self):
		MODEL_PATH = "detector.pth"
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		checkpoint = torch.load(MODEL_PATH, map_location=self.device)
		
		self.model = EAST()
		self.model.eval()
		self.model.load_state_dict(checkpoint["model_state_dict"])
		self.model.to(self.device)
		
	def doTask(self, frame):
		print("Detect")
		boxes = []
		
		with torch.no_grad():
			score_map, geometry_map = self.model(frame[0].to(self.device))
			
		print("Detect2")
		box = get_boxes(score_map.squeeze(0).cpu().numpy(), geometry_map.squeeze(0).cpu().numpy())
		# box = detect(score_map, geometry_map)
		boxes.append(box)
		
		print("Detector done")
		return (frame, boxes)

