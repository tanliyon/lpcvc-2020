import time
import torch
import torchvision
from torch import nn, optim
from model import *
from loss import *
from dataset import *

class Train:
    def __init__(self, image_directory_path, annotation_directory_path, epoch_directory_path):
        self.image_directory_path = image_directory_path
        self.annotation_directory_path = annotation_directory_path
        self.epoch_directory_path = epoch_directory_path

    def train(self, new_length, learning_rate, epochs, batch_size, num_workers, save_interval, shuffle=True, drop_last=False):
        train_dataset = TextLocalizationSet(self.image_directory_path, self.annotation_directory_path, new_length)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last, collate_fn=collate)

        # criterion = Loss()
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # model = EAST()
        # data_parallel = False
        # if torch.cuda.device_count() > 1:
        # 	model = nn.DataParallel(model)
        # 	data_parallel = True
        # model.to(device)
        # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[epochs // 2], gamma=0.1)
        #
        # for epoch in range(epochs):
        #     model.train()
        #     scheduler.step()
        #     epoch_loss = 0
        #     epoch_time = time.time()
        #
        #     for i, (name, image, quad_coordinates, text_tags) in enumerate(train_loader):
        #     	start_time = time.time()
        #     	img, gt_score, gt_geo, ignored_map = img.to(device), gt_score.to(device), gt_geo.to(device), ignored_map.to(device)
        #     	pred_score, pred_geo = model(img)
        #     	loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)
        #         epoch_loss += loss.item()
    	# 		optimizer.zero_grad()
    	# 		loss.backward()
    	# 		optimizer.step()
        #
        #     if (epoch + 1) % save_interval == 0:
    	# 		state_dict = model.module.state_dict() if data_parallel else model.state_dict()
    	# 		torch.save(state_dict, os.path.join(self.epoch_directory_path, 'model_epoch_{}.pth'.format(epoch + 1)))

        for i, (name, image, quad_coordinates, text_tags) in enumerate(train_loader):
            break

"""
Testing and sample usage of functions
"""
if __name__ == '__main__':
    imagePath = "./Dataset/Train/TrainImages/"
    annotationPath = "./Dataset/Train/TrainTruth/"
    trainingPath = "./Dataset/Train/TrainEpoch/"
    train = Train(imagePath, annotationPath, trainingPath)
    train.train(new_length=256, learning_rate=1e-3, epochs=2, batch_size=2, num_workers=4, save_interval=5)
