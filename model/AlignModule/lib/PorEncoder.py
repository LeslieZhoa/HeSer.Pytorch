from torch import nn
import torchvision
class PorEncoder(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.por_encoder = torchvision.models.resnext50_32x4d(num_classes=args.por_embedding_size)

    def forward(self,x):

        batch = x.shape[0]
        if len(x.shape) > 4:
            x = x.view(-1,*x.shape[2:])
        feat = self.por_encoder(x)
        if feat.shape[0] != batch:
            feat = feat.view(batch,feat.shape[0]//batch,-1)
            feat = feat.mean(1)
        
        return feat
       