import torch.nn as nn
import torchvision
import torch
from collections import OrderedDict
from .seresnet import se_resnet101, se_resnet50
from .resnet import resnet18, resnet34

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5
model_feature_size = {"vgg" : 512 * 4 * 13,"resnet18" : 512,"resnet34" : 512,"se_resnet101" : 2048, "se_resnet50" : 2048}

class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding

class WeightedGate(nn.Module):
    def __init__(self, total_size):
        super(WeightedGate, self).__init__()
        self.weighted = nn.Linear(total_size, 2)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, results):
        results = self.weighted(results)
        return self.softmax(results)

class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, encoder_dim, decoder_dim, vocab_size, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()
        
        self.LSTM_dim = 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 16))

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim * 2,  attention_dim)  # attention network

        self.dropout = nn.Dropout(p=self.dropout)

        self.decode_step1 = nn.LSTMCell(encoder_dim, self.LSTM_dim, bias=True)  # decoding LSTMCell
        self.decode_step2 = nn.LSTMCell( 2 * self.LSTM_dim, decoder_dim, bias=True)  # decoding LSTMCell

        self.decode_step1_r = nn.LSTMCell(encoder_dim, self.LSTM_dim, bias=True)  # decoding LSTMCell
        self.decode_step2_r = nn.LSTMCell( 2 * self.LSTM_dim, decoder_dim, bias=True)  # decoding LSTMCell

        self.fc = nn.Linear( encoder_dim + decoder_dim * 2, vocab_size)  # linear layer to find scores over vocabulary
        self.weighted_gate = WeightedGate( encoder_dim + 2 * decoder_dim)

    def init_hidden_state(self, batch_size, dim):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        h = torch.zeros((batch_size, dim)).cuda()
        c = torch.zeros((batch_size, dim)).cuda()
        return h, c

    def forward(self, encoder_out):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """
        # encoder out shape: (b, c, h, w)
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(1) # w
        vocab_size = self.vocab_size

        # Flatten image
        num_pixels = encoder_out.size(2) * encoder_out.size(3)
        features_holistic = encoder_out.view( encoder_out.size(0), encoder_out.size(1) ,num_pixels).permute(0,2,1)  # (batch_size,num_pixels, c)
        features = self.avgpool(encoder_out) #(b, c, 1, w)
        features = features.squeeze(2).permute(0,2,1) #(b, c, w) -> (b, w, c)
        slide_length = features.size(1)

        # Initialize LSTM state
        h1, c1 = self.init_hidden_state(features.shape[0], self.LSTM_dim)  # (batch_size, decoder_dim)
        h1_r, c1_r = self.init_hidden_state(features.shape[0], self.LSTM_dim)  # (batch_size, decoder_dim)

        h2, c2 = self.init_hidden_state(features.shape[0], self.decoder_dim)  # (batch_size, decoder_dim)
        h2_r, c2_r = self.init_hidden_state(features.shape[0], self.LSTM_dim)  # (batch_size, decoder_dim)


        # Create tensors to hold word predicion scores and alphas
        # predictions = torch.zeros(batch_size, num_pixels, vocab_size).to(device)
        predictions = torch.zeros(slide_length, batch_size, vocab_size).to(device)
        # alphas = torch.zeros(batch_size, num_pixels, num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding

        h1s = []
        h1rs = []

        h2s = []
        h2rs = []
        for t in range(slide_length):
 
            h1, c1 = self.decode_step1(
                features[:, t, :], (h1, c1))  # (batch_size_t, decoder_dim)
            h1s.append(h1)

            h1_r, c1_r = self.decode_step1_r(
                features[:, slide_length - t - 1, :], (h1_r, c1_r))  # (batch_size_t, decoder_dim)
            h1rs.append(h1_r)

        for t in range(slide_length):
            h2, c2 = self.decode_step2(
                torch.cat( [h1s[t], h1rs[slide_length - t - 1]], dim=1), (h2, c2))  # (batch_size_t, decoder_dim)
            h2s.append(h2)

            h2_r, c2_r = self.decode_step2_r(
                torch.cat([h1s[slide_length - t - 1], h1rs[t]], dim=1), (h2_r, c2_r))  # (batch_size_t, decoder_dim)
            h2rs.append(h2_r)


        for t in range(slide_length):
            attention_weighted_encoding = self.attention(features_holistic,  torch.cat([h2s[t], h2rs[slide_length - t - 1]], dim=1))

            # when t is zero, get end of predictions, which are zeros
            weight = self.weighted_gate(torch.cat([h2s[t], h2rs[slide_length - t - 1], attention_weighted_encoding], dim=1))
            preds = self.fc(self.dropout(torch.cat([weight[:, 0].unsqueeze(1) * h2s[t], 
                                                    weight[:, 0].unsqueeze(1) * h2rs[slide_length - t - 1], 
                                                     weight[:, 1].unsqueeze(1) * attention_weighted_encoding], dim=1)))
            predictions[t, :, :] = preds
        
        return predictions

class CRNN(nn.Module):
    def __init__(self, n_class, n_hidden, model_name):
        super(CRNN, self).__init__()
        assert model_name in model_feature_size.keys()
        self.model_name = model_name
        self.avgpool = nn.AdaptiveAvgPool2d((1, 16))
        
        if self.model_name == "se_resnet101":
            models = se_resnet101(pretrained='imagenet')
            self.features = torch.nn.Sequential(models.layer0, models.layer1, \
                models.layer2, models.layer3, models.layer4)

        if self.model_name == "se_resnet50":
            models = se_resnet50(pretrained='imagenet')
            self.features = torch.nn.Sequential(models.layer0, models.layer1, \
                models.layer2, models.layer3, models.layer4)

        if self.model_name == "resnet18":
            models = resnet18(pretrained=True)
            self.features = torch.nn.Sequential(OrderedDict(list(models.named_children())[:-2]))

        if self.model_name == "resnet34":
            models = resnet34(pretrained=True)
            self.features = torch.nn.Sequential(OrderedDict(list(models.named_children())[:-2]))

        self.decoder = DecoderWithAttention(attention_dim, model_feature_size[self.model_name], decoder_dim, n_class)
        
    def forward(self, img):
        features = self.features(img)
        output = self.decoder(features)
        return output
