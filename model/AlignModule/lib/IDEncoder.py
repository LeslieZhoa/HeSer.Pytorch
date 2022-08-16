import torch
from torch import nn
from model.AlignModule.module import Backbone


class IDEncoder(nn.Module):
    def __init__(self,model_path):
        super(IDEncoder, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(model_path))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        for module in [self.facenet, self.face_pool]:
            for param in module.parameters():
                param.requires_grad = False


    def forward(self, x):
        batch = x.shape[0]
        if len(x.shape) > 4:
            x = x.view(-1,*x.shape[2:])
        feat = self.facenet(self.face_pool(x))
        if feat.shape[0] != batch:
            feat = feat.view(batch,feat.shape[0]//batch,-1)
            feat = feat.mean(1)
        
        return feat



