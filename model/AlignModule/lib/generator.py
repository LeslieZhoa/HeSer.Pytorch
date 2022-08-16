import torch
from torch import nn
from torch.nn.utils import spectral_norm
from model.AlignModule.lib import blocks

import math

# heavily copy from https://github.com/shrubb/latent-pose-reenactment

class Constant(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.constant = nn.Parameter(torch.ones(1, *shape))

    def forward(self, batch_size):
        return self.constant.expand((batch_size,) + self.constant.shape[1:])


class Generator(nn.Module):
    def __init__(self, args):
        super().__init__()

        def get_res_block(in_channels, out_channels, padding, norm_layer):
            return blocks.ResBlock(in_channels, out_channels, padding, upsample=False, downsample=False,
                                   norm_layer=norm_layer)

        def get_up_block(in_channels, out_channels, padding, norm_layer):
            return blocks.ResBlock(in_channels, out_channels, padding, upsample=True, downsample=False,
                                   norm_layer=norm_layer)

        if args.padding == 'zero':
            padding = nn.ZeroPad2d
        elif args.padding == 'reflection':
            padding = nn.ReflectionPad2d
        else:
            raise Exception('Incorrect `padding` argument, required `zero` or `reflection`')

        assert math.log2(args.output_image_size / args.gen_constant_input_size).is_integer(), \
            "`gen_constant_input_size` must be `image_size` divided by a power of 2"
        num_upsample_blocks = int(math.log2(args.output_image_size / args.gen_constant_input_size))
        out_channels_block_nonclamped = args.num_channels * (2 ** num_upsample_blocks)
        out_channels_block = min(out_channels_block_nonclamped, args.max_num_channels)

        self.constant = Constant(out_channels_block, args.gen_constant_input_size, args.gen_constant_input_size)
       

        # Decoder
        layers = []
        for i in range(args.gen_num_residual_blocks):
            layers.append(get_res_block(out_channels_block, out_channels_block, padding, 'ada' + args.norm_layer))
        
        for _ in range(num_upsample_blocks):
            in_channels_block = out_channels_block
            out_channels_block_nonclamped //= 2
            out_channels_block = min(out_channels_block_nonclamped, args.max_num_channels)
            layers.append(get_up_block(in_channels_block, out_channels_block, padding, 'ada' + args.norm_layer))

        layers.extend([
            blocks.AdaptiveNorm2d(out_channels_block, args.norm_layer),
            nn.ReLU(True),
            # padding(1),
            spectral_norm(
                nn.Conv2d(out_channels_block, args.out_channels, 3, 1, 1),
                eps=1e-4),
            nn.Tanh()
        ])
        self.decoder_blocks = nn.Sequential(*layers)

        self.adains = [module for module in self.modules() if module.__class__.__name__ == 'AdaptiveNorm2d']


        joint_embedding_size = args.identity_embedding_size + args.pose_embedding_size + args.por_embedding_size + args.exp_embedding_size
        self.affine_params_projector = nn.Sequential(
            spectral_norm(nn.Linear(joint_embedding_size, max(joint_embedding_size, 512))),
            nn.ReLU(True),
            spectral_norm(nn.Linear(max(joint_embedding_size, 512), self.get_num_affine_params()))
        )


    def get_num_affine_params(self):
        return sum(2*module.num_features for module in self.adains)

    def assign_affine_params(self, affine_params):
        for m in self.modules():
            if m.__class__.__name__ == "AdaptiveNorm2d":
                new_bias = affine_params[:, :m.num_features]
                new_weight = affine_params[:, m.num_features:2 * m.num_features]

                if m.bias is None: # to keep m.bias being `nn.Parameter`
                    m.bias = new_bias.contiguous()
                else:
                    m.bias.copy_(new_bias)

                if m.weight is None: # to keep m.weight being `nn.Parameter`
                    m.weight = new_weight.contiguous()
                else:
                    m.weight.copy_(new_weight)

                if affine_params.size(1) > 2 * m.num_features:
                    affine_params = affine_params[:, 2 * m.num_features:]

    def assign_embeddings(self, por,id,pose,exp):
        
        joint_embedding = torch.cat((por,id,pose,exp), dim=1)

        affine_params = self.affine_params_projector(joint_embedding)
        self.assign_affine_params(affine_params)

    
    def forward(self, por,id,pose,exp):
        self.assign_embeddings(por,id,pose,exp)

        batch_size = len(por)
        output = self.decoder_blocks(self.constant(batch_size))
        
        return output
