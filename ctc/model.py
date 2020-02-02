# Implemented based on: https://github.com/BelBES/crnn-pytorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import numpy as np
import string

class CRNN(nn.Module):
    def __init__(self, backend='resnet18', rnn_hidden_size=128, rnn_num_layers=2, rnn_dropout=0):
        super(CRNN, self).__init__()

        # '-' is used as a special character to indicate duplicate character
        # Ex: hhheel-looo -> hel-lo -> hello
        self.outputs = string.digits + string.ascii_lowercase + '-'
        self.num_classes = len(self.outputs)
        self.decode = False

        # Create the convolution layers with pretrained weights from resnet18 
        # except final avgpool and fc layers
        self.feature_extractor = getattr(models, backend)(pretrained=True)
        self.cnn = nn.Sequential(
            self.feature_extractor.conv1,
            self.feature_extractor.bn1,
            self.feature_extractor.relu,
            self.feature_extractor.maxpool,
            self.feature_extractor.layer1,
            self.feature_extractor.layer2,
            self.feature_extractor.layer3,
            self.feature_extractor.layer4
        )

        # Create the rnn model
        # Set input size = last batch normalization's number of features
        self.rnn_input_size = self.cnn[-1][-1].bn2.num_features 
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.rnn = nn.GRU(self.rnn_input_size,
                          rnn_hidden_size, rnn_num_layers,
                          dropout=rnn_dropout, bidirectional=True)

        # Input to linear layer is (num_layers * num_directions)
        self.linear = nn.Linear(rnn_hidden_size * 2, self.num_classes)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
    	# Initialize the first hidden layer as 0
        hidden = self.init_hidden(x.size(0), x.is_cuda)
        features = self.cnn(x)
        features = self.features_to_sequence(features)
        seq, hidden = self.rnn(features, hidden)
        seq = self.linear(seq)
        seq = self.softmax(seq)
        if self.decode:
            seq = self.decode_seq(seq)

        return seq

    def init_hidden(self, batch_size, gpu=False):
    	# Initialize initial weight with dimension:
    	# (num_layers * num_directions, batch, hidden_size)
        h0 = Variable(torch.zeros(self.rnn_num_layers * 2,
                                  batch_size,
                                  self.rnn_hidden_size))
        if gpu:
            h0 = h0.cuda()
        return h0

    def features_to_sequence(self, features):
    	# Debug
        print(f"Conv output size: {features.size()}")

        batch, channel, height, width = features.size()
        assert height == 1, "the height of features must be 1"

        # Make width the number of input sequence to rnn
        # Change b, c, h, w -> w, b, h, c
        features = features.permute(3, 0, 2, 1)
        # Remove height with size=1
        features = features.squeeze(2)
        return features

    def pred_to_string(self, pred):
        seq = []
        out = []

        # Find the index where softmax is the maximum
        for i in range(pred.shape[0]):
            label = np.argmax(pred[i])
            seq.append(label - 1)

        # Debug
        print(f"Prediction index: {seq}")

        # Construct the final string
        for i in range(len(seq)):
            if len(out) == 0:
            	# Don't add to out if prediction is '-'
                if seq[i] != -1:
                    out.append(seq[i])
            else:
            	# Don't add to out if prediciton is '-' or the characters are repeated
                if seq[i] != -1 and seq[i] != seq[i - 1]:
                    out.append(seq[i])

        # Translate index -> string based on self.outputs
        out = ''.join(self.outputs[i] for i in out)
        return out

    def decode_seq(self, pred):
    	# Swap w, b, h -> b, w, h
        pred = pred.permute(1, 0, 2).cpu().data.numpy()
        seq = []

        for i in range(pred.shape[0]):
            seq.append(self.pred_to_string(pred[i]))
        return seq