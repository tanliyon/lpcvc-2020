import torch.nn as nn

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.FeatureExtraction = ResNet_FeatureExtractor(3, 512)
        self.FeatureExtraction_output = 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

        self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, 256, 256),
                BidirectionalLSTM(256, 256, 256))
        self.SequenceModeling_output = 256

        self.Prediction = Attention(self.SequenceModeling_output, 256, 64)#64 TOTAL ALPHABETS


    def forward(self, input, text, is_train=True):

        # input = self.Transformation(input)

        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)
        contextual_feature = self.SequenceModeling(visual_feature)
        # prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=len(text[0]))
        prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=25)

        return prediction