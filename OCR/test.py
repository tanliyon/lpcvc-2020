from torchvision import transforms
from OCR.utils.config import CONFIG
from OCR.utils.util import *
import OCR.models.crnn_lang_chenjun as crnn


def crop(img, box, transform):
    tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y = box
    top = min(tl_y, tr_y)
    left = min(tl_x, bl_x)
    height = max(bl_y, br_y) - min(tl_y, tr_y)
    width = max(tr_x, br_x) - min(tl_x, bl_x)
    return transform(transforms.functional.crop(img, top, left, height, width))

def init_attn():
    encoder = crnn.CNN(32,1,256)
    number_classes = len(CONFIG.ALPHABETS.split(":")) + 3
    decoder = crnn.decoderV2(256, number_classes, dropout_p=0.1)
    encoder.load_state_dict(torch.load("./encoder.pth", map_location='cpu'))
    decoder.load_state_dict(torch.load("./decoder.pth", map_location='cpu'))
    return encoder, decoder

def attn_OCR(encoder, decoder, img, boxes):
    attn_transform = transforms.Compose([
        transforms.Resize((32, 280)),
        transforms.ToTensor()
    ])
    
    EOS_TOKEN = 1
    max_length = 15
    to_pil = transforms.ToPILImage()
    converter = strLabelConverterForAttention(CONFIG.ALPHABETS, ":")
    pil_img = []
    
    for im in img:
        pil_img.append(to_pil(im))
    
    words = []
    count = []
    
    for i, box in enumerate(boxes):
        count.append(len(box))
        for b in box:
            words.append(crop(pil_img[i], b, attn_transform))
    
    decoded_words = []
    
    for word in words:
        decoded_word = []
        with torch.no_grad():
            encoder_outputs = encoder(word.unsqueeze(0) if len(word.shape) == 3 else word)
            
        decoder_input = torch.zeros(1).long()
        decoder_input = decoder_input
        decoder_hidden = decoder.initHidden(1)
        decoder_attentions = torch.zeros(max_length, 71)
        
        for di in range(max_length):
            with torch.no_grad():
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
    
    return decoded_words, count




