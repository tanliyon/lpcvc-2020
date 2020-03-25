import time
import torch
import torchvision
from torchvision import transforms
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
        train_dataset = TextLocalizationSet(self.image_directory_path, self.annotation_directory_path, (126, 224))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last, collate_fn=collate)

        criterion = Loss()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = EAST()

        data_parallel = False
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            data_parallel = True

        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            epoch_score_loss = 0
            epoch_geometry_loss = 0
            epoch_time = time.time()
            for i, (name, path, image, coordinates, transcriptions, score_map, training_mask, geometry_map) in enumerate(train_loader):
                start_time = time.time()
                optimizer.zero_grad()
                for j in range(len(image)):
                    img, score, mask, geometry = image[j].to(device), score_map[j].to(device), training_mask[j].to(device), geometry_map[j].to(device)
                    predicted_score_map, predicted_geometry_map = model(img)
                    loss = criterion(score, predicted_score_map, geometry, predicted_geometry_map)
                    epoch_loss += loss.item()
                    print("Epoch {}, Batch {}, Item {}, Batch Loss {:.6f}".format(epoch + 1, i + 1, j + 1, loss.item()))
                    loss.backward()
                optimizer.step()

            scheduler.step()
            if (epoch + 1) % save_interval == 0:
                state_dict = model.module.state_dict() if data_parallel else model.state_dict()
                torch.save(state_dict, os.path.join(self.epoch_directory_path, 'model_epoch_{}.pth'.format(epoch + 1)))

"""
Testing and sample usage of functions
"""
if __name__ == '__main__':
    image_directory_path = "./Dataset/Train/TrainImages/"
    annotation_directory_path = "./Dataset/Train/TrainTruth/"
    training_path = "./Dataset/Train/TrainEpoch/"
    # train_dataset = TextLocalizationSet(image_directory_path, annotation_directory_path, (126, 224))
    # train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=0, drop_last=False, collate_fn=collate)
    # for i, (name, path, image, coordinates, transcriptions, score_map, training_mask, geometry_map) in enumerate(train_loader):
    #     print("Here")
    train = Train(image_directory_path, annotation_directory_path, training_path)
    # batch = 24, epochs = 600
    train.train(new_length=256, learning_rate=1e-3, epochs=2, batch_size=24, num_workers=2, save_interval=5)
