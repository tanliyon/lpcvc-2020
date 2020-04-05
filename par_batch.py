import torch.multiprocessing as mp
import time
import torch
import torchvision
from torchvision import transforms
from detector.model import EAST
from ctc.model import CRNN
from detector.detect import get_boxes

def list_2_lists(lst, count):
    idx = 0
    for c in count:
        yield lst[idx : idx + c]
        idx += c

def crop(img, box):
	transform = transforms.Compose([
		transforms.Resize((50, 600)),
		transforms.ToTensor(),
		transforms.Normalize(mean=(0.5,), std=(0.5,))
		])
		
	tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y = box
	top = min(tl_y, tr_y)
	left = min(tl_x, bl_x)
	height = max(bl_y, br_y) - min(tl_y, tr_y)
	width = max(tr_x, br_x) - min(tl_x, bl_x)
	return transform(transforms.functional.crop(img, top, left, height, width))

def detect(model, dataset, data_conn, signal):
	for i, frame in enumerate(dataset):
		img, _ = frame
		with torch.no_grad():
			score_map, geometry_map = model(img)
		boxes = []
		#print(score_map[0].shape)
		for i in range(score_map.shape[0]):
                    boxes.append(get_boxes(score_map[i].cpu().numpy(), geometry_map[i].cpu().numpy()))
		data_conn.send((img, boxes))
	print("Detection done")
	data_conn.close()
	signal.wait()

def recognize(model, text_q, data_conn, signal):
	while True:
		if data_conn.poll(40):
			img, boxes = data_conn.recv()
		else:
			break
		
		to_pil = transforms.ToPILImage()
		pil_img = []
		
		for im in img:
                    pil_img.append(to_pil(im))
		
		words = []
		count = []
	
		for i, box in enumerate(boxes):
			count.append(len(box))
			for b in box:
                            words.append(crop(pil_img[i], b))
	
		words = torch.stack(words)
		with torch.no_grad():
			predictions = model(words)
		
		text_q.put(list(list_2_lists(predictions, count)))
                    
	print("Recognition done")
	signal.set()

if __name__ == '__main__':
	mp.set_start_method('spawn', force=True)
	transform = transforms.Compose([
		transforms.Resize((126, 224)),
		transforms.Grayscale(num_output_channels=1),
		transforms.ToTensor(),
		transforms.Normalize(mean=(0.5,), std=(0.5,))
	])

	detector = EAST()
	model_wgt = torch.load('detector.pth', map_location='cpu')
	detector.load_state_dict(model_wgt["model_state_dict"])
	detector.eval()
	detector.share_memory()
	
	ctc = CRNN(pretrained=False)
	ctc.decode = True
	ctc.load_state_dict(torch.load('ctc.pth', map_location='cpu'))
	ctc.eval()
	ctc.share_memory()
	
	parent_d_conn, child_d_conn = mp.Pipe(duplex=False)
	signal = mp.Event()
	text_q = mp.Queue()
	
	frames_data = torchvision.datasets.ImageFolder(root='./frames', transform=transform)
	frames_loader = torch.utils.data.DataLoader(frames_data, batch_size=32, shuffle=False)
	
	recog_p = mp.Process(target=recognize, args=(ctc, text_q, parent_d_conn, signal))
	detect_p = mp.Process(target=detect, args=(detector, frames_loader, child_d_conn, signal))
	
	detect_p.start()
	recog_p.start()
	
	recog_p.join()
	detect_p.join()
	
	text_list = []
	text_q.put(None)
	for i in iter(text_q.get, None):
		text_list.append(i)
	print(text_list)
		
		
	
	
