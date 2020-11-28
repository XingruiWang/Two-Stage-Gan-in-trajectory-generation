import time
import torch.utils.data as Data
import torch.nn as nn
import random
from torch.nn.utils.rnn import pack_padded_sequence
from model import Encoder, Decoder
from dataset import *
from utils import *
from vis import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
torch.backends.cudnn.benchmark = True

torch.manual_seed(0)
torch.cuda.manual_seed(0)

data_path = '/data/wangxingrui'
decoder_dim = 128
dropout = 0.5

start_epoch = 1
epochs = 3000  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement
batch_size = 32
workers = 1  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_loss = 10.  # best loss score right now
print_freq = 1  # print training/validation stats every __ batches
fine_tune_encoder = False # fine-tune encoder?
checkpoint = './checkpoint/checkpoint_best.pth'# path to checkpoint, None if none
save_path = './checkpoint/' # checkpoint save path
vis_dir = './vis/' # store visualized result
teach_force_rate = 1.0
max_len = 8 # the longest sequence

def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch, teach_rate = 1.0):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    encoder.train()
    decoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()
    teach_force_rate = teach_rate
    for i,data in enumerate(train_loader):
        img_name = data['name']
        imgs = data['image'].to(device) # (b,c,w,h)
        seq = data['seq'].to(device) # (b,max_len,2)
        enter = data['enter'].to(device) # (b,2)
        esc = data['esc'].to(device) # (b,4) one-hot indicate four direction
        length = data['len'] # (b,1) it seem to be a 1D CPU int64 tensor when use pack_padded_sequence below

        skip = [95,114,115,118,121, 123,212,214,221, 247,258,259, 262,265]
        if i in skip:
           continue



        data_time.update(time.time() - start)

        # Forward prop.
        imgs = encoder(imgs) # encoder_out
        pred, sort_ind, alphas = decoder(imgs, enter, esc, seq[:,:-1,:], length-1, teach_force_rate)
        # pred (b,max_len,2)

        targets_with_start = seq[sort_ind,:,:] # to the sorted version
        length = length[sort_ind,:] - 1
        # Remove timesteps that we didn't decode at, or are pads
        #pred = pack_padded_sequence(pred, length.squeeze(1), batch_first=True)
        #targets = pack_padded_sequence(targets, length.squeeze(1), batch_first=True)

        # Calculate loss
        #loss = criterion(pred, targets_with_start, length)
        loss = criterion(pred, targets_with_start[:,1:,:])
        #loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        decoder_optimizer.zero_grad()

        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        losses.update(loss.item(), length.sum().item())
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}] [{1}/{2}]\n'
                  'Batch Time {batch_time.val:.3f}s (Average:{batch_time.avg:.3f}s)\n'
                  'Data Load Time {data_time.val:.3f}s (Average:{data_time.avg:.3f}s)\n'
                  'Loss {loss.val:.4f} (Average:{loss.avg:.4f})\n'
                  'Teach_force_rate{rate:.3f}\n'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses, rate = teach_force_rate))
    return losses.avg
class traj_loss(nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(traj_loss, self).__init__()
        self.criterion = nn.MSELoss(size_average,reduce,reduction)


    def forward(self, pred, target, real_length):
        weight_loss = self._weigthed_loss(pred, target)
        cos_loss = self._cos_loss(pred, target, real_length)
        return weight_loss + 0.04 * cos_loss

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

    '''
    def function(self, pred, target, criterion):
        batch, _, _ = pred.size()
        d = target[:,1:,:]-target[:,:-1,:]#(b,m 2)
        ab_square = d*d
        add = tensor.ones(batch, 2,1)
        d_square = torch.matmul(ab_square, add) #(b, max_len, 1)

        pred_d = target[:,1:,:]-pred

        pred_ab_dot = pred_d*d
        pred_dot = torch.matmul(pred_ab_dot, add) #(b, max_len, 1)

        v_square = pred_dot**2 / d_square

        pred_square = torch.matmul(pred_d*pred_d, add) #(b, max_len, 1)

        h_square = pred_square - v_square
        print(h_square)

        if target.requires_grad:
            if reduction != 'none':
                h_square = torch.mean(h_square) if reduction == 'mean' else torch.sum(h_square)
        return h_square
    '''

def validate(val_loader, encoder, decoder, criterion, teach_force_rate):

    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()

    start = time.time()

    # explicitly disable gradient calculation to avoid CUDA memory error
    with torch.no_grad():
        # Batches
        for i, data in enumerate(val_loader):
            # Move to device, if available
            if i == 23:
                continue
            imgs = data['image'].to(device)  # (b,c,w,h)
            seq = data['seq'].to(device)  # (b,max_len,2)
            enter = data['enter'].to(device)  # (b,2)
            esc = data['esc'].to(device)  # (b,4)
            length = data['len']  # (b,1)  it seem to be a 1D CPU int64 tensor

            # Forward prop.
            if encoder is not None:
                imgs_encode = encoder(imgs)
            pred, sort_ind, alphas = decoder(imgs_encode, enter, esc, seq[:,:-1,:], length - 1, teach_force_rate)

            targets_with_start = seq[sort_ind,:,:]
            length = length[sort_ind,:] - 1

            pred_cal = pred.clone()
            #pred_cal = pack_padded_sequence(pred_cal, length.squeeze(1), batch_first=True)
            #targets = pack_padded_sequence(targets, length.squeeze(1), batch_first=True)

            # Calculate loss
            # loss = criterion(pred_cal, targets_with_start, length)
            loss = criterion(pred, targets_with_start[:,1:,:])
            #print(pred.size(1), targets_with_start.size(1))

            # Keep track of metrics
            losses.update(loss.item(),length.sum().item())
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\n'
                      'Batch Time {batch_time.val:.3f}s (Average:{batch_time.avg:.3f}s)\n'
                      'Loss {loss.val:.4f} (Average:{loss.avg:.4f})\n'.format(i, len(val_loader), batch_time=batch_time,loss=losses))

    return losses.avg, imgs[sort_ind,:,:,:], pred, enter[sort_ind,:], esc[sort_ind,:], length[sort_ind,:], targets_with_start

def main():
    global epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, best_loss, save_path, vis_dir, decoder_dim,teach_force_rate

    if checkpoint is None:
        decoder = Decoder(decoder_dim)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                         lr=decoder_lr)
        encoder = Encoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                         lr=encoder_lr) if fine_tune_encoder else None

    else:
        print('Resume at checkpoint %s'%(checkpoint))
        checkpoint = torch.load(checkpoint)
        decoder = Decoder(decoder_dim)
        encoder = Encoder()
        start_epoch = 0#checkpoint['epoch'] + 1
        epochs_since_improvement = 0
        best_loss = 1
        decoder_pretrain = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        # encoder.load_state_dict(encoder_pretrain.state_dict())
        decoder.load_state_dict(decoder_pretrain.state_dict())
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(
                params=filter(lambda p: p.requires_grad, encoder.parameters()),
                lr=encoder_lr)

    decoder = decoder.to(device)
    encoder = encoder.to(device)

    #criterion = traj_loss().to(device)
    criterion = nn.MSELoss(reduction = 'mean')

    dataset = GuiderDataset(data_path,0.2,max_len=max_len)
    train_loader = Data.DataLoader(dataset.train_set(), batch_size=batch_size, shuffle=False)
    val_loader = Data.DataLoader(dataset.test_set(), batch_size=batch_size, shuffle=False)
    # loss, imgs, pred, enter, esc, length, target = validate(val_loader=val_loader,
    #                             encoder=encoder, decoder=decoder, criterion=criterion,teach_force_rate = teach_force_rate)
    # visualize the last batch of validate epoch
    # visualize(vis_dir, imgs, pred, enter, esc, length, 'before', target)
    for epoch in range(start_epoch, start_epoch + epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement > 0 and epochs_since_improvement % 50 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        train_loss = train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch, teach_rate = teach_force_rate)

        # One epoch's validation, return the average loss of each batch in this epoch
        loss, imgs, pred, enter, esc, length, target = validate(val_loader=val_loader,
                                    encoder=encoder, decoder=decoder, criterion=criterion,teach_force_rate = teach_force_rate)
        # visualize the last batch of validate epoch
        visualize(vis_dir, imgs, pred, enter, esc, length, epoch, target)

        # Check if there was an improvement
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0
        if loss < 0.011:
            teach_force_rate *= 0.9
        # Save checkpoint
        save_checkpoint(save_path, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, loss, is_best)

if __name__ == '__main__':
    main()
