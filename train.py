import time
import torch
import torchvision
from torchvision import transforms
from torch import nn, optim
from model import *
from loss import *
from dataset import *
from error import *
import logging

class Train:
    def __init__(self, image_directory_path, annotation_directory_path, epoch_directory_path, validate_image_path, validate_annotation_path):
        self.image_directory_path = image_directory_path
        self.annotation_directory_path = annotation_directory_path
        self.epoch_directory_path = epoch_directory_path
        self.validate_image_path = validate_image_path
        self.validate_annotation_path = validate_annotation_path
        self.model_path = "/detector.pth"

    def train(self, new_dimensions, learning_rate, epochs, batch_size, num_workers, save_interval, validate_interval, shuffle=True, drop_last=False):
        logging.basicConfig(filename="training_10.log", level=logging.INFO)
        logging.info("\nStarted")

        train_dataset = TextLocalizationSet(self.image_directory_path, self.annotation_directory_path, new_dimensions)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last, collate_fn=collate)

        validate_dataset = TextLocalizationSet(self.validate_image_path, self.validate_annotation_path, new_dimensions)
        validate_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last, collate_fn=collate)

        criterion = Loss()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = EAST()

        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        model.apply(weights_init)

        data_parallel = False
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            data_parallel = True

        try:
            model.to(device)
        except RuntimeError as exception:
            if out_of_memory(exception, model):
                model.to(device)

        checkpoint = torch.load(training_path + self.model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs//2], gamma=0.1)

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            epoch_time = time.time()
            for i, (path, image, score_map, training_mask, geometry_map) in enumerate(train_loader):
                batch_loss = 0
                start_time = time.time()
                optimizer.zero_grad()
                for j in range(len(image)):
                    img, score, mask, geometry = image[j].to(device), score_map[j].to(device), training_mask[j].to(device), geometry_map[j].to(device)
                    predicted_score_map, predicted_geometry_map = model(img)
                    loss = criterion(score, predicted_score_map.squeeze(0), geometry, predicted_geometry_map.squeeze(0), mask)
                    batch_loss += loss.item()
                    loss.backward()
                optimizer.step()
                epoch_loss += batch_loss
                logging.info("Batch {}, Batch Loss {:.6f}\n".format(i + 1, batch_loss))

            logging.info("\n\nEpoch {}, Time {:.6f}, Epoch Loss {:.6f}\n\n".format(epoch + 1, time.time() - epoch_time, epoch_loss))
            scheduler.step()

            if (epoch + 1) % save_interval == 0:
                model_state_dict = model.module.state_dict() if data_parallel else model.state_dict()
                optimizer_state_dict = optimizer.state_dict()
                torch.save({"epoch": epoch + 1,
                            "model_state_dict": model_state_dict,
                            "optimizer_state_dict": optimizer_state_dict,
                            "epoch_loss": epoch_loss}, os.path.join(self.epoch_directory_path, 'model_epoch_{}.pth'.format(epoch + 1)))

            if (epoch + 1) % validate_interval == 0:
                model.eval()
                validation_loss = 0
                for i, (path, image, score_map, training_mask, geometry_map) in enumerate(validate_loader):
                    with torch.no_grad():
                        for j in range(len(image)):
                            predicted_score_map, predicted_geometry_map = model(image[j].to(device))
                            loss = criterion(score_map[j], predicted_score_map.squeeze(0).cpu(), geometry_map[j], predicted_geometry_map.squeeze(0).cpu(), training_mask[j])
                            validation_loss += loss.item()
                logging.info("\n\nEpoch {}, Validation Loss {:.6f}\n\n".format(epoch + 1, validation_loss))

        logging.info("\nEnded")

"""
Testing and sample usage of functions
"""
if __name__ == '__main__':
    image_directory_path = "./Dataset/Train/TrainImagesNew/"
    annotation_directory_path = "./Dataset/Train/TrainTruthNew/"
    training_path = "./Dataset/Train/TrainEpoch/"
    validate_image_path = "./Dataset/Test/TestImages/"
    validate_annotation_path = "./Dataset/Test/TestTruth/"
    # image_directory_path = "./Samples/TrainImages/"
    # annotation_directory_path = "./Samples/TrainTruth/"
    # training_path = "./Samples/TrainModel/"
    train = Train(image_directory_path, annotation_directory_path, training_path, validate_image_path, validate_annotation_path)
    train.train(new_dimensions=(126, 224), learning_rate=1e-3, epochs=1000, batch_size=8, num_workers=4, save_interval=10, validate_interval=5, shuffle=True)
