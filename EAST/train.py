import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
from dataset import custom_dataset
from east_model import EAST as east
from loss import Loss
import os
import time
import sys
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, interval):
	file_num = len(os.listdir(train_img_path))
	trainset = custom_dataset(train_img_path, train_gt_path)
	train_loader = data.DataLoader(trainset, batch_size=batch_size, \
                                   shuffle=True, num_workers=num_workers, drop_last=True)
	writer = SummaryWriter()
	criterion = Loss()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = east()
	data_parallel = False
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
		data_parallel = True
	try:
		model.to(device)
	except RuntimeError as e:
		if 'out of memory' in str(e):
			print('| WARNING: ran out of memory, retrying batch', sys.stdout)
			sys.stdout.flush()
			for p in model.parameters():
				if p.grad is not None:
					del p.grad  # free some memory
			torch.cuda.empty_cache()
			model.to(device)
		else:
			raise e
	model.load_state_dict(torch.load("/home/damini/PycharmProjects/low_power_attention_model/lpcvc-2020/EAST/pths/model_epoch_75.pth"))
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[epoch_iter//2], gamma=0.1)
	for epoch in range(76, epoch_iter):
		model.train()
		scheduler.step()
		epoch_loss = 0
		epoch_time = time.time()
		for i, (img, gt_score, gt_geo, ignored_map) in enumerate(train_loader):
			start_time = time.time()
			img, gt_score, gt_geo, ignored_map = img.to(device), gt_score.to(device), gt_geo.to(device), ignored_map.to(device)
			try:
				pred_score, pred_geo = model(img.squeeze(0))
			except RuntimeError as e:
				if 'out of memory' in str(e):
					print('| WARNING: ran out of memory, retrying batch', sys.stdout)
					sys.stdout.flush()
					for p in model.parameters():
						if p.grad is not None:
							del p.grad  # free some memory
					torch.cuda.empty_cache()
					pred_score, pred_geo = model(img.squeeze(0))
				else:
					raise e
			loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map, i)

			epoch_loss += loss.item()
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			print('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format(\
              epoch+1, epoch_iter, i+1, int(file_num/batch_size), time.time()-start_time, loss.item()))

		print('epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(epoch_loss/int(file_num/batch_size), time.time()-epoch_time))
		print(time.asctime(time.localtime(time.time())))
		print('='*50)
		if (epoch + 1) % interval == 0:
			state_dict = model.module.state_dict() if data_parallel else model.state_dict()
			torch.save(state_dict, os.path.join(pths_path, 'model_epoch_{}.pth'.format(epoch+1)))
		writer.add_scalar('Loss/epoch_loss', epoch_loss/int(file_num/batch_size), epoch)


if __name__ == '__main__':
	train_img_path = os.path.abspath('/home/damini/Documents/ICDAR_detection_2015/train_img')
	train_gt_path  = os.path.abspath('/home/damini/Documents/ICDAR_detection_2015/train_gt')
	pths_path      = './pths'
	batch_size     = 8
	lr             = 1e-3
	num_workers    = 4
	epoch_iter     = 1000
	save_interval  = 5
	train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, save_interval)	
	
