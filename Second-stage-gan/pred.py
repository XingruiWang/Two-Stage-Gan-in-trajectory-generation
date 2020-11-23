#coding:utf-8
import os
import time
import torch.utils.data as Data
import torch.nn as nn
import random
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence
from model import Encoder, Decoder
from dataset import *
from utils import *
from vis import *
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
torch.backends.cudnn.benchmark = True

torch.manual_seed(0)
torch.cuda.manual_seed(0)

checkpoint = './checkpoint_big/checkpoint_best.pth'
data_path = '/data/lzt/project/waysguider_3/dataset/'
output_path = './pred_vis/'

def predict(checkpoint ,data_path, output_path):
    """
    prediction
    """
    max_len = 8
    batch_size = 32

    checkpoint = torch.load(checkpoint)
    decoder = checkpoint['decoder']
    encoder = checkpoint['encoder']
    
    decoder.eval()
    encoder.eval()

    dataset = GuiderDataset(data_path,0.2,max_len=max_len)
    pred_loader = Data.DataLoader(dataset.test_set(), batch_size=batch_size, shuffle=False) # pred 1 seq each time

    with torch.no_grad():
        for i, data in tqdm(enumerate(pred_loader)):
            # Move to device, if available
            imgs = data['image'].to(device)  # (b,c,w,h)
            enter = data['enter'].to(device)  # (b,2)
            esc = data['esc'].to(device) # (b,2)
            length = torch.full((batch_size,1),max_len,dtype=torch.long)

            encoder_out = encoder(imgs)
            
            batch_size = encoder_out.size(0)
            encoder_dim = encoder_out.size(-1)

            # Flatten image
            encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (b, num_pixels, encoder_dim)
            num_pixels = encoder_out.size(1) # for attention. not useful at the moment

            # Initialize LSTM state
            h, c, h_inv, c_inv = decoder.init_hidden_state(encoder_out, enter, esc)
            
            # Create tensors to hold two coordination predictions
            predictions_ord = torch.zeros((batch_size,max_len,2)).to(device)  # (b,max_len,2)
            predictions_inv = torch.zeros((batch_size,max_len,2)).to(device)  # (b,max_len,2)

            predictions_ord[:,0,:] = enter
            predictions_inv[:,max_len-1,:] = esc

            for t in range(max_len):
                h_a = torch.cat([h,h_inv],dim=1)
                h_b = torch.cat([h_inv,h],dim=1)
                c_a = torch.cat([c,c_inv],dim=1)
                c_b = torch.cat([c_inv,c],dim=1)
                
                #attention_weighted_encoding, alpha = decoder.attention(encoder_out,h_a)
                #attention_weighted_encoding_inv, alpha_inv = decoder.attention(encoder_out,h_b)
                #gate = decoder.sigmoid(decoder.f_beta(h_a))
                #gate_inv = decoder.sigmoid(decoder.f_beta(h_b))
                #attention_weighted_encoding = gate * attention_weighted_encoding
                #attention_weighted_encoding_inv = gate_inv * attention_weighted_encoding_inv
                
                # weight is attention (differ from var weights below)
                weight = F.softmax(decoder.attention(h_a)) # weight for each input pixels
                weight_inv = F.softmax(decoder.attention(h_b)) # (batch_size_t,n_pixels)
                
                h, c = decoder.decoder(
                    torch.cat([decoder.position_embedding(predictions_ord[:,t,:]),encoder_out[:,t,:] * weight],dim=1),
                    (h_a, c_a))  # (batch_size_t, decoder_dim)
                
                h_inv, c_inv = decoder.decoder_inv(
                    torch.cat([decoder.position_embedding(predictions_inv[:,max_len-1-t,:]),encoder_out[:,t,:] * weight_inv],dim=1),
                    (h_b, c_b))
                
                h = decoder.trans_h(h)
                c = decoder.trans_c(c)
                h_inv = decoder.trans_h(h_inv)
                c_inv = decoder.trans_c(c_inv)
                
                preds = decoder.fc(decoder.dropout(h))  # (batch_size_t, 2)
                preds_inv = decoder.fc(decoder.dropout(h_inv))
                if t < max_len - 1:
                    predictions_ord[:, t + 1, :] = preds # (b,max_len,2)
                    predictions_inv[:, max_len-2-t,:] = preds_inv
            
            ## weight scheme 1
            #first_part = [1]*int(max_len/2)
            #second_part = [0]*(max_len - int(max_len/2))
            #weights = np.array(first_part + second_part)
            #weights_inv = np.array(second_part + first_part)
            
            ## weight scheme 2
            weights = np.array([_ for _ in range(max_len)])
            weights = np.exp(-weights)
            weights_inv = weights[::-1]
            weights = np.vstack([weights,weights_inv])
            weights /= weights.sum(axis=0)
            weights_inv = weights[1,:]
            weights = weights[0,:]
            
            weights = torch.tensor(weights,dtype=torch.float).to(device).unsqueeze(0).unsqueeze(2)
            weights_inv = torch.tensor(weights_inv, dtype=torch.float).to(device).unsqueeze(0).unsqueeze(2)
            predictions = (predictions_ord * weights + predictions_inv * weights_inv)
            
            output_dir = output_path + 'batch-{}/'.format(i)
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            visualize(output_dir,imgs, predictions, predictions_ord, predictions_inv, data['seq'], enter, esc, length, 'pred')
            

if __name__ == '__main__':
    predict(checkpoint ,data_path, output_path)