# Implemented based on: https://github.com/BelBES/crnn-pytorch

import os
import string
import torch
import numpy as np
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import argparse

from model import CRNN

parser = argparse.ArgumentParser(description='Train the CTC model.')
parser.add_argument('--epoch', default=50,
                    type=int, help='Number of epochs to train the model')
parser.add_argument('--learning_rate', default=0.01,
                    type=int, help='Base learning rate to start training with')
parser.add_argument('--weight_decay', default=0.001,
                    type=int, help='Base learning rate to start training with')
parser.add_argument('--save_dir', default='trained_model',
                    type=str, help='Name of directory to save trained model')
parser.add_argument('--load_path', default=None,
                    type=str, help='Relative path to load trained model')
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#TODO: Handle input of varying sizes. For now the height needs to be 32
transform = transforms.Compose(
    [transforms.ToTensor(),
    ])

net = CRNN()
net.to(device)
criterion = nn.CTCLoss()
optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

if args.load_path:
    load_path = os.path.join(os.getcwd(), args.load_path)

# TODO: Create directory to save_path if it does not exist
# TODO: Check for valid path
save_path = os.path.join(os.getcwd(), args.save_path)

"""
if args.load_model:
    net.load_state_dict(torch.load(args.load_path))

for i, data in enumerate(trainloader):
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = data

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # print statistics
    running_loss += loss.item()
    if i % 2000 == 1999:    # print every 2000 mini-batches
        print('[%d, %5d] loss: %.3f' %
            (epoch + 1, i + 1, running_loss / 2000))
        running_loss = 0.0

    # Save model
    torch.save(model.state_dict(), args.save_dir)
"""

# Debug
x = torch.rand(8, 3, 32, 500)
x = x.to(device)
model = CRNN()
model.decode = True
model = model.to(device)
y = model(x)
print(f"String output: {y}")


# Notes:
# Number of prediction = width of conv output
