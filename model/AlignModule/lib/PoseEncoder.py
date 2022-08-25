from torch import nn
import torchvision

class PoseEncoder(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.pose_encoder = torchvision.models.mobilenet_v2(num_classes=args.pose_embedding_size)

    def forward(self,x):
        
        return self.pose_encoder(x)