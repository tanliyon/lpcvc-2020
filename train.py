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

        criterion = Loss()
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")

        # extractor = PVAnet()
        # merger = Unet()
        # output = Output()
        # model = EAST(extractor, merger, output)
        # model = EAST()
        # data_parallel = False
        # if torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)
        #     data_parallel = True
        # model.to(device)
        # for param in model.state_dict():
        #     print(param, "\t", model.state_dict()[param].size())
        # parameters = list(model.parameters)
        optimizer = torch.optim.Adam(list(extractor.parameters()) + list(merger.parameters()) + list(output.parameters() ), lr=learning_rate)
        # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[epochs // 2], gamma=0.1)
        #
        # for epoch in range(epochs):
        #     model.train()
        #     scheduler.step()
        #     epoch_loss = 0
        #     epoch_score_loss = 0
        #     epoch_geometry_loss = 0
        #     epoch_time = time.time()
        #
        #     for i, (name, image, _, _, _, score_map, training_mask, geometry_map) in enumerate(train_loader):
        #         start_time = time.time()
        #         image, score_map, training_mask, geometry_map = image.to(device), score_map.to(device), training_mask.to(device), geometry_map.to(device)
        #         predicted_score_map, predicted_geometry_map = model(image)
        #         loss = criterion(score_map, predicted_score_map, geometry_map, predicted_geometry_map)
        #         epoch_loss += loss.item()
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
        #
        #     if (epoch + 1) % save_interval == 0:
        #         state_dict = model.module.state_dict() if data_parallel else model.state_dict()
        #         torch.save(state_dict, os.path.join(self.epoch_directory_path, 'model_epoch_{}.pth'.format(epoch + 1)))

"""
Testing and sample usage of functions
"""
if __name__ == '__main__':
    imagePath = "./Dataset/Train/TrainImages/"
    annotationPath = "./Dataset/Train/TrainTruth/"
    trainingPath = "./Dataset/Train/TrainEpoch/"
    train = Train(imagePath, annotationPath, trainingPath)
    # batch = 24, epochs = 600
    train.train(new_length=256, learning_rate=1e-3, epochs=1, batch_size=2, num_workers=4, save_interval=5)
