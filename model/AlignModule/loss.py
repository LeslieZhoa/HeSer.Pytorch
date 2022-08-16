import torch
from torch import nn
from collections import OrderedDict
import torchvision
import torch.nn.functional as F
def compute_dis_loss(fake_pred,real_pred,D_loss):
    d_real = torch.relu(1. - real_pred).mean() 
    d_fake = torch.relu(1. + fake_pred).mean() 
    D_loss['d_real'] = d_real 
    D_loss['d_fale'] = d_fake 
    return d_real + d_fake 

def compute_gan_loss(fake_pred):
    return -fake_pred.mean()

def compute_id_loss(fake_id_f,real_id_f):
    return 1.0 - torch.cosine_similarity(fake_id_f,real_id_f, dim = 1)


# Perceptual loss that uses a pretrained VGG network

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(-1)
        
class PerceptualLoss(nn.Module):
    def __init__(self,model_path, normalize_grad=False):
        super().__init__()
       
        self.normalize_grad = normalize_grad

       
        vgg_weights = torch.load(model_path)

        map = {'classifier.6.weight': u'classifier.7.weight', 'classifier.6.bias': u'classifier.7.bias'}
        vgg_weights = OrderedDict([(map[k] if k in map else k, v) for k, v in vgg_weights.items()])

        model = torchvision.models.vgg19()
        model.classifier = nn.Sequential(Flatten(), *model.classifier._modules.values())

        model.load_state_dict(vgg_weights)

        model = model.features

        mean = torch.tensor([103.939, 116.779, 123.680]) / 255.
        std = torch.tensor([1., 1., 1.]) / 255.

        num_layers = 30

        self.register_buffer('mean', mean[None, :, None, None])
        self.register_buffer('std' ,  std[None, :, None, None])

        layers_avg_pooling = []

        for weights in model.parameters():
            weights.requires_grad = False

        for module in model.modules():
            if module.__class__.__name__ == 'Sequential':
                continue
            elif module.__class__.__name__ == 'MaxPool2d':
                layers_avg_pooling.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
            else:
                layers_avg_pooling.append(module)

            if len(layers_avg_pooling) >= num_layers:
                break

        layers_avg_pooling = nn.Sequential(*layers_avg_pooling)

        self.model = layers_avg_pooling

    def normalize_inputs(self, x):
        return (x - self.mean) / self.std

    def forward(self, input, target):
        input = (input + 1) / 2
        target = (target.detach() + 1) / 2

        loss = 0

        features_input = self.normalize_inputs(input)
        features_target = self.normalize_inputs(target)

        for layer in self.model:
            features_input = layer(features_input)
            features_target = layer(features_target)

            if layer.__class__.__name__ == 'ReLU':
                if self.normalize_grad:
                    pass
                else:
                    loss = loss + F.l1_loss(features_input, features_target)

        return loss
