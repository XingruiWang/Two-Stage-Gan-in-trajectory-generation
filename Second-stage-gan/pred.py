#coding:utf-8
import os
import time
import torch.utils.data as Data
import torch.nn as nn
import random
from torch.nn.utils.rnn import pack_padded_sequence
from model_org import Encoder, Decoder
from dataset import *
from utils import *
from vis import *
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
torch.backends.cudnn.benchmark = True

torch.manual_seed(0)
torch.cuda.manual_seed(0)

checkpoint = '/home/wxr/projects/waysguider3/waysguider/checkpoint_2_stage/checkpoint_best_wrong.pth'
data_path = '/home/wxr/projects/waysguider3/waysguider/dataset/'
output_path = '/home/wxr/projects/waysguider3/waysguider/pred_vis/'

def vis_att(attention, name, i, img, dir = 'attention_vis'):
    attention = attention.cpu().numpy()[0]
    attention = attention.reshape(16,16)
    attention = attention/np.max(attention)*255
    attention = cv.resize(attention, (64,64), interpolation=cv.INTER_CUBIC)
    attention = attention.astype('int32')
    path = 'attention_vis/{}'.format(i)
    if not os.path.exists(path):
        os.mkdir(path)
    print(os.path.join(path, '{}_{}.png'.format(img, name)))
    cv.imwrite(os.path.join(path, '{}_{}.png'.format(img, name)) , attention)


    #print(np.max(attention), attention.shape)

class traj_loss(nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(traj_loss, self).__init__()
        self.criterion = nn.MSELoss(size_average,reduce,reduction)


    def forward(self, pred, target, real_length):
        weight_loss = self._weigthed_loss(pred, target)
        #cos_loss = self._cos_loss(pred, target, real_length)
        return weight_loss

    def turn_d(self, a, b, c):
        v1 = b-a
        v2 = c-a
        m1 = torch.sqrt(torch.sum(v1 * v1, dim = 1))
        m2 = torch.sqrt(torch.sum(v2 * v2, dim = 1))
        dot = torch.sum(v1 * v2, dim = 1)
        cos = dot / (m1 * m2)
        #print('cos = {}'.format(cos))
        return cos

    def _cos_loss(self, pred, target, real_length):
        #real_length,_ = real_length.squeeze(1).sort(dim=0, descending=True)
        batch, length, _ = pred.size()
        cos1 = self.turn_d(target[:,0,:], target[:,1,:], pred[:,0,:])
        loss = self.criterion(cos1, torch.ones(batch).to(device))
        #print('real length = {}'.format(real_length.sort()))
        for n in range(1,length):
            real_batch = sum([l > n for l in real_length])
            #print('real_batch = {}, n = {}'.format(real_batch, n))
            loss = loss + self.criterion(self.turn_d(target[:,n,:], target[:,n+1,:], pred[:,n,:])[:real_batch], torch.ones(real_batch).to(device))

        print('turn_loss = {}'.format(loss))
        return loss

    def _weigthed_loss(self, pred, target):
        """
        w = e ^ (1 - index / length)
        loss = sum( w * point-wise-loss )
        """
        target = target[:,1:,:] # 我调的时候传入的target也包含了sequence的第一个点，所以在这去掉了
        length = pred.size(1)

        weight = torch.tensor([(length-i)/length for i in range(length)])
        weight = torch.exp(weight)

        loss = self.criterion(pred[:,0,:], target[:,0,:])*weight[0]
        for n in range(1, length):
            loss = loss + self.criterion(pred[:,n,:], target[:,n,:])*weight[n]
        return loss

def predict(checkpoint ,data_path, output_path):
    """
    prediction
    """
    max_len = 8
    batch_size = 64

    checkpoint = torch.load(checkpoint)
    decoder = Decoder(128).to(device)
    decoder_state = decoder.state_dict()
    update_state = checkpoint['decoder'].state_dict()
    decoder_state.update(update_state)
    for k, _ in update_state.items():
        print('=> loading {} from pretrained model'.format(k))
    decoder.load_state_dict(update_state)
    encoder = checkpoint['encoder']

    decoder.eval()
    encoder.eval()

    dataset = GuiderDataset(data_path,0.2,max_len=max_len)
    pred_loader = Data.DataLoader(dataset.test_set(), batch_size=batch_size, shuffle=False) # pred 1 seq each time
    criterion = traj_loss().to(device)
    with torch.no_grad():
        for i, data in tqdm(enumerate(pred_loader)):
            # Move to device, if available
            imgs = data['image'].to(device)  # (b,c,w,h)
            enter = data['enter'].to(device)  # (b,2)
            esc = data['esc'].to(device) # (b,4)
            #img_map = data['map'].to(device)
            seq = data['seq'].to(device)
            length = torch.full((batch_size,1),max_len,dtype=torch.long)

            encoder_out = encoder(imgs)

            batch_size = encoder_out.size(0)
            encoder_dim = encoder_out.size(-1)

            # Flatten image
            encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (b, num_pixels, encoder_dim)
            num_pixels = encoder_out.size(1) # for attention. not useful at the moment

            # Initialize LSTM state
            h, c = decoder.init_hidden_state(encoder_out, enter, esc)  # (batch_size, decoder_dim)

            # Create tensors to hold two coordination predictions
            predictions = torch.zeros((batch_size,max_len,2)).to(device)  # (b,max_len,2)
            #predictions[:,:3,:] = seq[:,:3,:]

            predictions[:,0,:] = enter


            for t in range(max_len):
                attention_weighted_encoding, alpha = decoder.attention(encoder_out,h)
                gate = decoder.sigmoid(decoder.f_beta(h))
                #vis_att(attention_weighted_encoding, str(t), i, data['name'][0])
                attention_weighted_encoding = gate * attention_weighted_encoding


                h, c = decoder.decoder(
                    torch.cat([decoder.position_embedding(predictions[:,t,:]),attention_weighted_encoding],dim=1),
                    (h, c))  # (batch_size_t, decoder_dim)
                preds = decoder.fc(decoder.dropout(h))  # (batch_size_t, 2)
                if t < max_len - 1:
                    predictions[:, t + 1, :] = preds # (b,max_len,2)

            output_dir = output_path + 'batch-{}/'.format(i)
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)

            visualize(output_dir,imgs, predictions, enter, esc, length, 'pred', seq)

if __name__ == '__main__':
    predict(checkpoint ,data_path, output_path)
