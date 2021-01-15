#coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
from torchvision.models.vgg import VGG
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generator
class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=64, fine_tune=True):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size
        
        # use resnet as pretrain model
        #resnet = torchvision.models.resnet50(pretrained=True)

        # Remove linear and pool layers (since we're not doing classification)
        #modules = list(resnet.children())[:-2]
        #self.encodenet = nn.Sequential(*modules)
        
        # use hrnet pretrained result
        #self.encodenet = torch.load('model_best.pth')
        
        # use fcn8s as pretrained result
        vgg_model = VGGNet(pretrained=False, requires_grad=True, remove_fc=True)
        fcn_model = FCN8s(pretrained_net=vgg_model, n_class=2)
        #vgg_model.cuda()
        #fcn_model.cuda()
        pretrained_dict = torch.load('roadnetwork_best.pth')
        pretrained_dict = {k[7:]:v for k, v in pretrained_dict.items()} # used to seperate 'module.'
        self.encodenet = fcn_model
        self.encodenet.load_state_dict(pretrained_dict)
        self.conv = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0, dilation=1)
        
        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        if fine_tune:
            self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.encodenet(images)['x1']  # (batch_size, 128, image_size, image_size)
        out = self.conv(out) # (channel 32 to 1)
        out = self.adaptive_pool(out)  # (batch_size, encode_dim, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 1)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.encodenet.parameters():
            p.requires_grad = fine_tune
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        #for c in list(self.encodenet.children()):
        #    for p in c.parameters():
        #        p.requires_grad = fine_tune

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
        self.decoder_att = nn.Linear(decoder_dim*2, attention_dim)  # linear layer to transform decoder's output
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
        batch_size = encoder_out.shape[0]
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=2).view(batch_size,-1)  # (batch_size, num_pixels)

        return attention_weighted_encoding, alpha


class Decoder(nn.Module):
    """Decoder with Attention module"""
    def __init__(self,decoder_dim, encoder_size=4096,encoder_dim=1, attention_dim=16, dropout=0.5):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        """
        super().__init__()
        self.encoder_dim = encoder_dim # 1
        self.encoder_size = encoder_size # 16*16
        #self.condition_dim = condition_dim # enter point coordination
        self.decoder_dim = decoder_dim

        self.attention = nn.Linear(decoder_dim*2,encoder_size) # map coordinates to weights of pixels

        self.dropout = nn.Dropout(p=dropout)
        # 这里有个问题，位置信息是否需要先做embedding？ 如果使用attention和图片输入，应该就需要
        self.position_embedding = nn.Linear(2,encoder_size) # 将坐标的信息转化为和图片对齐
        self.decoder = nn.LSTMCell(encoder_size*2, decoder_dim*2, bias=True) # 2 indicates (X,Y)
        self.decoder_inv = nn.LSTMCell(encoder_size*2, decoder_dim*2, bias=True) # 2 indicates (X,Y)
        self.trans_h = nn.Linear(decoder_dim*2, decoder_dim)
        self.trans_c = nn.Linear(decoder_dim*2, decoder_dim)

        self.init_condition = nn.Linear(4, encoder_size)
        self.init_h = nn.Linear(encoder_size, decoder_dim)
        self.init_c = nn.Linear(encoder_size , decoder_dim)
        #self.f_beta = nn.Linear(decoder_dim*2, encoder_size)
        self.sigmoid= nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, 2) # regression question
        
        self.init_weights()

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out, enter, esc):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param enter: enter point
        :param esc: escape point
        :return: hidden state, cell state
        """

        flat_encoder_out = encoder_out.view(encoder_out.shape[0],-1) # (batch_size, encoder_size/n_pixels)
        condition = torch.cat([enter,esc],dim=1) # (batch_size,4)
        condition_inv = torch.cat([esc,enter],dim=1)
        condition_embedding = self.init_condition(condition)  # (batch_size, encoder_size)
        condition_embedding_inv = self.init_condition(condition_inv)

        h = self.init_h(flat_encoder_out * condition_embedding) # (batch_size, decoder_dim)
        h_inv = self.init_h(flat_encoder_out * condition_embedding_inv)
        c = self.init_c(flat_encoder_out * condition_embedding)
        c_inv = self.init_c(flat_encoder_out * condition_embedding_inv)
        return h, c, h_inv, c_inv

    def forward(self, encoder_out, enter, esc, sequence, sequence_inv, seq_len):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param enter: enter point (b,2)
        :param esc: escape point (b,2)
        :param sequence: coodination sequence (batch_size, max_seq_len, 2)
        :param seq_len: sequence length (batch_size)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1) # for attention. not useful at the moment

        # Sort input data by decreasing lengths; why? apparent below
        seq_len, sort_ind = seq_len.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind,:,:]
        sequence = sequence[sort_ind,:,:]
        sequence_inv = sequence_inv[sort_ind,:,:]

        # position embedding to match encoder_size
        sequence =  self.position_embedding(sequence) # (b,max_len,encoder_size)
        sequence_inv = self.position_embedding(sequence_inv)

        # Initialize LSTM state
        h, c, h_inv, c_inv = self.init_hidden_state(encoder_out, enter, esc)  # (batch_size, decoder_dim)

        # Create tensors to hold two coordination predictions
        predictions = torch.zeros((batch_size, sequence.shape[1], 2)).to(device)  # (b,max_len(2~t),e_size)
        predictions_inv = torch.zeros((batch_size, sequence.shape[1], 2)).to(device)  # (b,max_len,e_size)
        predictions_assemble = torch.zeros((batch_size,sequence.shape[1]+1,2)).to(device)
        
        #alphas = torch.zeros((batch_size, sequence.shape[1], num_pixels)).to(device)
        #alphas_inv = torch.zeros((batch_size, sequence.shape[1], num_pixels)).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new coordinate in the decoder with the previous word and the attention weighted encoding
        for t in range(sequence.shape[1]):
            h_a = torch.cat([h,h_inv],dim=1)
            h_b = torch.cat([h_inv,h],dim=1)
            c_a = torch.cat([c,c_inv],dim=1)
            c_b = torch.cat([c_inv,c],dim=1)
            
            batch_size_t = sum([l > t for l in seq_len])
            #attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
            #                                                    h_a[:batch_size_t]) #(b,e_size) (b,e_size)
            #attention_weighted_encoding_inv, alpha_inv = self.attention(encoder_out[:batch_size_t],
            #                                                    h_b[:batch_size_t])
            #gate = self.sigmoid(self.f_beta(h_a[:batch_size_t]))            
            #gate_inv = self.sigmoid(self.f_beta(h_b[:batch_size_t]))

            #attention_weighted_encoding = gate * attention_weighted_encoding
            #attention_weighted_encoding_inv = gate_inv * attention_weighted_encoding_inv
            
            weight = F.softmax(self.attention(h_a[:batch_size_t])) # weight for each input pixels
            weight_inv = F.softmax(self.attention(h_b[:batch_size_t])) # (batch_size_t,n_pixels)
            
            h, c = self.decoder(
                torch.cat([sequence[:batch_size_t,t,:],encoder_out[:batch_size_t,t,:] * weight],dim=1),
                (h_a[:batch_size_t,:], c_a[:batch_size_t,:]))  # (batch_size_t, decoder_dim)
            h_inv, c_inv = self.decoder_inv(
                torch.cat([sequence_inv[:batch_size_t,t,:],encoder_out[:batch_size_t,t,:] * weight_inv],dim=1),
                (h_b[:batch_size_t,:], c_b[:batch_size_t,:]))

            h = self.trans_h(h)
            c = self.trans_c(c)
            h_inv = self.trans_h(h_inv)
            c_inv = self.trans_c(c_inv)
            
            preds = self.fc(self.dropout(h))  # (batch_size_t, 2)
            preds_inv = self.fc(self.dropout(h_inv))
            predictions[:batch_size_t, t, :] = preds # (b,max_len,2)
            predictions_inv[:batch_size_t, t,:] = preds_inv # used to train model
            predictions_assemble[:batch_size_t, sequence.shape[1]-1-t,:] # used to store a visualizable result
            #alphas[:batch_size_t,t,:] = alpha # this is used to visualize(not implemented yet), and add regularization
            #alphas_inv[:batch_size_t,t,:] = alpha_inv
            
            # visualizable result
            predictions_assemble = move_forward(predictions_assemble, seq_len, device)
            predictions_assemble[:,1:,:] += predictions.data
            predictions_assemble[:,1:-1,:] /= 2
            # a weights to be implemented
            #weights = torch.range(0.1,1,seq_len) ??
            #predictions = (predictions + predictions_inv) / 2.
        return predictions, predictions_inv, predictions_assemble, sort_ind, 
#alphas, alphas_inv


# pretrain utils
class FCN8s(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)

        score = self.relu(self.deconv1(x5))               # size=(N, 512, x.H/16, x.W/16)
        score = self.bn1(score + x4)                      # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.relu(self.deconv2(score))            # size=(N, 256, x.H/8, x.W/8)
        score = self.bn2(score + x3)                      # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score1 = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score2 = self.bn4(self.relu(self.deconv4(score1)))  # size=(N, 64, x.H/2, x.W/2)
        score3 = self.bn5(self.relu(self.deconv5(score2)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score3)                    # size=(N, n_class, x.H/1, x.W/1)

        return {'x4':score1, 'x2':score2, 'x1':score3, 'x0':score}

class VGGNet(VGG):
    def __init__(self, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True, show_params=False):
        super().__init__(make_layers(cfg[model]))
        self.ranges = ranges[model]

        if pretrained:
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc:  # delete redundant fully-connected layer params, can save memory
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}

        # get the output of each maxpooling layer (5 maxpool in VGG net)
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output["x%d"%(idx+1)] = x

        return output
    
ranges = {
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

# cropped version from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)