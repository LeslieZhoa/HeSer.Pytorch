from torch import nn
import torchvision
class ExpEncoder(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.exp_encoder = torchvision.models.mobilenet_v2(num_classes=args.exp_embedding_size)

    def forward(self,x):
        return self.exp_encoder(x)