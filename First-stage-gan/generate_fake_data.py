import torch
import torch.nn as nn
import torchvision.utils as vutils
import numpy as np
import argparse
import os
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--large_num', required=True, help='num of generate data')
parser.add_argument('--model_path', required=True, help='Generator_path')
parser.add_argument('--output_path',required=True,help = 'path of output')
opt = parser.parse_args()
large_num = int(opt.large_num)
ngf=64
nc=1
nz=128
model_path=opt.model_path
output = opt.output_path
if not os.path.exists(output):
    os.mkdir(output)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
        # inputs is Z, going into a convolution
        nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
        nn.BatchNorm2d(ngf * 4),
        nn.ReLU(True),
        # state size. (ngf*8) x 4 x 4
        nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf * 2),
        nn.ReLU(True),
        # state size. (ngf*4) x 8 x 8
        nn.ConvTranspose2d(ngf * 2, ngf , 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf),
        nn.ReLU(True),
        # state size. (ngf) x 32 x 32
        nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
        nn.Tanh()
        )

    def forward(self, inputs):
        outputs = self.main(inputs)
        outputs = outputs.reshape((-1,3,32,32))
        return outputs

netG = Generator()
netG = torch.load(model_path)

device = torch.device("cuda" )
netG.to(device)



for i in tqdm(range(large_num)):
    noise = torch.randn(1,128,1,1,device=device)
    fake_imgs = netG(noise)
    fake_cpu = fake_imgs.cpu()
    fake_cpu = ((fake_cpu/2)+0.5)*795
    path = os.path.join(output, str(i)+'.npy')
    np.save(path,fake_cpu[0].detach().numpy().transpose(1,2,0).astype(np.int))