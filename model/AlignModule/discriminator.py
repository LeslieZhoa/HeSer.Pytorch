from torch import nn
from model.AlignModule.lib import blocks
from torch.nn.utils import spectral_norm
import math
import torch
# heavily copy from https://github.com/shrubb/latent-pose-reenactment

# class Discriminator(nn.Module):
#     def __init__(self, args):
#         super().__init__()

#         def get_down_block(in_channels, out_channels, padding):
#             return blocks.ResBlock(in_channels, out_channels, padding, upsample=False, downsample=True,
#                                    norm_layer='none')

#         def get_res_block(in_channels, out_channels, padding):
#             return blocks.ResBlock(in_channels, out_channels, padding, upsample=False, downsample=False,
#                                    norm_layer='none')

#         if args.padding == 'zero':
#             padding = nn.ZeroPad2d
#         elif args.padding == 'reflection':
#             padding = nn.ReflectionPad2d

#         self.out_channels = args.embed_channels

#         self.down_block = nn.Sequential(
#             # padding(1),
#             spectral_norm(
#                 nn.Conv2d(args.in_channels, args.num_channels, 3, 1, 1),
#                 eps=1e-4),
#             nn.ReLU(),
#             # padding(1),
#             spectral_norm(
#                 nn.Conv2d(args.num_channels, args.num_channels, 3, 1, 1),
#                 eps=1e-4),
#             nn.AvgPool2d(2))
#         self.skip = nn.Sequential(
#             spectral_norm(
#                 nn.Conv2d(args.in_channels, args.num_channels, 1),
#                 eps=1e-4),
#             nn.AvgPool2d(2))

#         self.blocks = nn.ModuleList()
#         num_down_blocks = min(int(math.log(args.output_image_size, 2)) - 2, args.dis_num_blocks)
#         in_channels = args.num_channels
#         for i in range(1, num_down_blocks):
#             out_channels = min(in_channels * 2, args.max_num_channels)
#             if i == args.dis_num_blocks - 1: out_channels = self.out_channels
#             self.blocks.append(get_down_block(in_channels, out_channels, padding))
#             in_channels = out_channels
#         for i in range(num_down_blocks, args.dis_num_blocks):
#             if i == args.dis_num_blocks - 1: out_channels = self.out_channels
#             self.blocks.append(get_res_block(in_channels, out_channels, padding))

#         self.linear = spectral_norm(nn.Linear(self.out_channels, 1), eps=1e-4)


#     def forward(self, input):
        
#         feats = []

#         out = self.down_block(input)
#         out = out + self.skip(input)
#         feats.append(out)
#         for block in self.blocks:
#             out = block(out)
#             feats.append(out)
#         out = torch.relu(out)
#         out = out.view(out.shape[0], self.out_channels, -1).sum(2)
#         out_linear = self.linear(out)[:, 0]
#         return out_linear,feats

from torch.nn.utils import spectral_norm

# PyTorch implementation by vinesmsuic
# Referenced from official tensorflow implementation: https://github.com/SystemErrorWang/White-box-Cartoonization/blob/master/train_code/network.py
# slim.convolution2d uses constant padding (zeros).
# Paper used spectral_norm

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,activate=True):
        super().__init__()
        self.sn_conv = spectral_norm(nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride, 
                padding,
                padding_mode="zeros" # Author's code used slim.convolution2d, which is using SAME padding (zero padding in pytorch) 
            ))
        self.activate = activate
        if self.activate:
            self.LReLU = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.sn_conv(x)
        if self.activate:
            x = self.LReLU(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128,256]):
        super().__init__()
        
        self.model = nn.Sequential(
            #k3n32s2
            Block(in_channels, features[0], kernel_size=3, stride=2, padding=1),
            #k3n32s1
            Block(features[0], features[0], kernel_size=3, stride=1, padding=1),

            #k3n64s2
            Block(features[0], features[1], kernel_size=3, stride=2, padding=1),
            #k3n64s1
            Block(features[1], features[1], kernel_size=3, stride=1, padding=1),

            #k3n128s2
            Block(features[1], features[2], kernel_size=3, stride=2, padding=1),
            #k3n128s1
            Block(features[2], features[2], kernel_size=3, stride=1, padding=1),

            #k3n256s2
            Block(features[2], features[3], kernel_size=3, stride=2, padding=1),
            #k3n256s1
            Block(features[3], features[3], kernel_size=3, stride=1, padding=1),


            #k1n1s1
            Block(features[3], out_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        x = self.model(x)

        return x
    