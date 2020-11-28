#coding:utf-8
import torch
import torch.nn as nn
import torchvision
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generator
class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=16, fine_tune=True):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        #resnet = torchvision.models.resnet50(pretrained=True)

        # Remove linear and pool layers (since we're not doing classification)
        #modules = list(resnet.children())[:-2]
        #self.encodenet = nn.Sequential(*modules)

        # use hrnet pretrained result
        self.encodenet = torch.load('model_best.pth')

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
        out = self.encodenet(images)  # (batch_size, 1, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, encode_dim, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 1)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.encodenet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.encodenet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

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
        batch_size = encoder_out.shape[0]
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, num_pixels)

        return attention_weighted_encoding, alpha


class Decoder(nn.Module):
    """Decoder with Attention module"""
    def __init__(self,decoder_dim, condition_dim=256, encoder_size=256,encoder_dim=1, position_dim=128, attention_dim=4 , dropout=0.5):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        """
        super().__init__()
        self.encoder_dim = encoder_dim # 1
        self.encoder_size = encoder_size # 16*16
        self.position_embedding_dim = position_dim  # 位置embedding？
        self.condition_dim = condition_dim # enter point coordination / esc direction
        self.decoder_dim = decoder_dim
        self.attention_dim = attention_dim

        self.attention = Attention(encoder_dim,decoder_dim,attention_dim)

        self.dropout = nn.Dropout(p=dropout)
        # 这里有个问题，位置信息是否需要先做embedding？
        position_embedding = [nn.Linear(2,self.position_embedding_dim),nn.Linear(self.position_embedding_dim,self.position_embedding_dim)]
        self.position_embedding = nn.Sequential(*position_embedding)
        self.decoder = nn.LSTMCell(self.position_embedding_dim + self.encoder_size, decoder_dim,bias=True) # 2 indicates (X,Y)
        self.init_condition = nn.Linear(6, condition_dim)
        self.init_h = nn.Linear(encoder_size + condition_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_size + condition_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_size)
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

        flat_encoder_out = encoder_out.view(encoder_out.shape[0],-1) # (batch_size, 16*16)
        condition = torch.cat([enter,esc],dim=1) # (batch_size,6)
        condition_embedding = self.init_condition(condition)  # (batch_size,condition_dim)

        h = self.init_h(torch.cat([flat_encoder_out,condition_embedding],dim=1)) # (batch_size, decoder_dim)
        c = self.init_c(torch.cat([flat_encoder_out,condition_embedding],dim=1))
        return h, c

    def forward(self, encoder_out, enter, esc, sequence, seq_len, teach_force_rate):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param enter: enter point (b,2)
        :param esc: escape point (b,2)
        :param sequence: coodination sequence (batch_size, max_seq_len, 2)
        :param seq_len: sequence length (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1) # for attention. not useful at the moment
        #print(encoder_out.size(), num_pixels)

        # Sort input data by decreasing lengths; why? apparent below
        seq_len, sort_ind = seq_len.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind,:,:]
        sequence = sequence[sort_ind,:,:]

        # position embedding ???
        preds = sequence[:,0,:]
        sequence =  self.position_embedding(sequence) # (b,max_len,position_embedding_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out, enter, esc)  # (batch_size, decoder_dim)

        # Create tensors to hold two coordination predictions
        predictions = torch.zeros((batch_size, sequence.shape[1],2)).to(device)  # (b,max_len,2)
        alphas = torch.zeros((batch_size, sequence.shape[1], num_pixels)).to(device)
        decision = random.random()
        use_teacher_forcing = decision <= teach_force_rate
        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new coordinate in the decoder with the previous word and the attention weighted encoding
        for t in range(sequence.shape[1]):
            batch_size_t = sum([l > t for l in seq_len])
            #local_encoder_out = encoder_out[:]
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t]) #(b,e_size) (b,e_size)
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding
            if use_teacher_forcing:
                h, c = self.decoder(
                    torch.cat([sequence[:batch_size_t,t,:],attention_weighted_encoding],dim=1),
                    (h[:batch_size_t,:], c[:batch_size_t,:]))  # (batch_size_t, decoder_dim)
            else:
                h, c = self.decoder(
                    torch.cat([self.position_embedding(preds[:batch_size_t,:]),attention_weighted_encoding],dim=1),
                    (h[:batch_size_t,:], c[:batch_size_t,:]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, 2)
            predictions[:batch_size_t, t, :] = preds # (b,max_len,2)
            alphas[:batch_size_t,t,:] = alpha # this is used to visualize(not implemented yet), and add regularization
            decision = random.random()
            use_teacher_forcing = decision <= teach_force_rate
        return predictions, sort_ind, alphas

# descriminator
class Descriminator(nn.Module):
    def __init__(self):
        super().__init__()
        pass

# gan utils
