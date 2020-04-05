from OCR.utils.config import CONFIG
import torchvision.transforms as transforms
from OCR.utils.util import *
#from torch.utils.data import Dataset, DataLoader
#import torch.utils.data
import OCR.models.crnn_lang_chenjun as crnn

SOS_token = 0
EOS_TOKEN = 1
BLANK = 2
to_pil = transforms.ToPILImage()

def OCR_att(frames, bboxes):
    max_length = 15
    converter = strLabelConverterForAttention(CONFIG.ALPHABETS, ":")
    encoder = crnn.CNN(32,1,256)
    number_classes = len(CONFIG.ALPHABETS.split(":")) + 3
    decoder = crnn.decoderV2(256, number_classes, dropout_p=0.1)
    encoder.load_state_dict(torch.load("./encoder_2.pth", map_location='cpu'))
    decoder.load_state_dict(torch.load("./decoder_2.pth", map_location='cpu'))
    
    to_bw = transforms.Compose([transforms.Resize((32, 280), interpolation=Image.NEAREST), transforms.ToTensor()])
    for e, d in zip(encoder.parameters(), decoder.parameters()):
        e.requires_grad = False
        d.requires_grad = False
    encoder.eval()
    decoder.eval()
    word_list = []
    
    for p, (frame, frame_bboxes) in enumerate(zip(frames, bboxes)):
        words = []
        frame = to_pil(frame)
        
        for box in frame_bboxes:
            tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y = box
            top = min(tl_y, tr_y)
            left = min(tl_x, bl_x)
            height = max(bl_y, br_y) - min(tl_y, tr_y)
            width = max(tr_x, br_x) - min(tl_x, bl_x)
            words.append(to_bw(transforms.functional.crop(frame, top, left, height, width)))

        words = torch.stack(words)
        decoded_word = []
        decoded_words = []
        
        for word in words:
            encoder_outputs = encoder(word.unsqueeze(0) if len(word.shape) == 3 else word)
            decoder_input = torch.zeros(1).long()
            decoder_input = decoder_input
            decoder_hidden = decoder.initHidden(1)
            decoder_attentions = torch.zeros(max_length, 71)
            for di in range(max_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
                probs = torch.exp(decoder_output)
                decoder_attentions[di-1] = decoder_attention.data
                topv, topi = decoder_output.data.topk(1)
                ni = topi.squeeze(1)
                decoder_input = ni
                if ni == EOS_TOKEN:
                    break
                else:
                    decoded_word.append(converter.decode(ni))
            decoded_words.append("".join(decoded_word).replace("$", ""))
        word_list.append(decoded_words)

    return word_list




