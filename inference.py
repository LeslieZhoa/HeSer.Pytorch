from model.AlignModule.lib import *
from model.BlendModule.generator import Generator as Decoder
from model.AlignModule.config import Params as AlignParams
from model.BlendModule.config import Params as BlendParams 
from trainer.AlignTrainer import AlignTrainer
from model.third.model import BiSeNet
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torch
import cv2
import numpy as np
import pdb

class Infer:
    def __init__(self,align_path,blend_path,parsing_path):
        align_params = AlignParams()
        blend_params = BlendParams()
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
      
        self.parsing = BiSeNet(n_classes=19).to(self.device)
        self.Epor = PorEncoder(align_params).to(self.device)
        self.Eid = IDEncoder(align_params.id_model).to(self.device)
        self.Epose = PoseEncoder(align_params).to(self.device)
        self.Eexp = ExpEncoder(align_params).to(self.device)
        self.netG = Generator(align_params).to(self.device)
        self.decoder = Decoder(blend_params).to(self.device)
        
        self.loadModel(align_path,blend_path,parsing_path)
        self.eval_model(self.Epor,self.Eid,self.Epose,self.Eexp,self.netG,self.decoder,self.parsing)
        self.mean =torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

    def run(self,tgt_img_path,src_img_paths):
        
        tgt_img = cv2.imread(tgt_img_path)
        tgt_inp = self.preprocess(tgt_img)

        src_img = cv2.imread(src_img_paths[0])
           
        src_inp = self.preprocess_multi(src_img_paths)

        gen = self.forward(src_inp,tgt_inp) 
        gen = self.postprocess(gen[0])
        cat_img = np.concatenate([cv2.resize(src_img,[512,512]),
                            gen,cv2.resize(tgt_img,[512,512])],1)
        return cat_img
    
    def forward(self,xs,xt):
        with torch.no_grad():
            por_f = self.Epor(xs)
            id_f = self.Eid(AlignTrainer.process_id_input(xs,crop=True))

            pose_f = self.Epose(F.adaptive_avg_pool2d(xt,256))
            exp_f = self.Eexp(AlignTrainer.process_id_input(xt,crop=True,size=256))

            xg = self.netG(por_f,id_f,pose_f,exp_f)

            M_a = self.parsing(self.preprocess_parsing(xg))
            M_t = self.parsing(self.preprocess_parsing(xt))
            
            M_a = self.postprocess_parsing(M_a)
            M_t = self.postprocess_parsing(M_t)
            xg_gray = TF.rgb_to_grayscale(xg,num_output_channels=1)
            fake = self.decoder(xg,xg_gray,xt,M_a,M_t,xt,train=False)
       
        return fake

    def preprocess(self,x):
        if isinstance(x,str):
            x = cv2.imread(x)
        x = cv2.resize(x,[512,512])
        x = (x[...,::-1].transpose(2,0,1)[np.newaxis,:] / 255 - 0.5) * 2
        return torch.from_numpy(x.astype(np.float32)).to(self.device)

    def preprocess_multi(self,xs):
        x_list = []
        for x in xs:
            x = cv2.imread(x)
            x = cv2.resize(x,[512,512])
            x_list.append((x[...,::-1].transpose(2,0,1)[np.newaxis,:] / 255 - 0.5) * 2)
        x_list = np.concatenate(x_list,0)
        return torch.from_numpy(x_list.astype(np.float32)).to(self.device).unsqueeze(0)

    def postprocess(self,x):
        return (x.permute(1,2,0).cpu().numpy()[...,::-1] + 1) * 127.5

    def preprocess_parsing(self,x):
        
        return ((x+1)/2.0 - self.mean.view(1,-1,1,1).to(self.device)) / \
                self.std.view(1,-1,1,1).to(self.device)

    def postprocess_parsing(self,x):
        return torch.argmax(x[0],1).unsqueeze(1).float()
        


    def loadModel(self,align_path,blend_path,parsing_path):
        ckpt = torch.load(align_path, map_location=lambda storage, loc: storage)
        self.netG.load_state_dict(ckpt['G'],strict=False)
        self.Eexp.load_state_dict(ckpt['Eexp'],strict=False)
        self.Eid.load_state_dict(ckpt['Eid'],strict=False)
        self.Epor.load_state_dict(ckpt['Epor'],strict=False)

        ckpt = torch.load(blend_path, map_location=lambda storage, loc: storage)
        self.decoder.load_state_dict(ckpt['G'],strict=False)

        self.parsing.load_state_dict(torch.load(parsing_path))

    
    def eval_model(self,*args):
        for arg in args:
            arg.eval()

if __name__ == "__main__":
    model = Infer('checkpoint/Aligner/323-00000000.pth',
                'checkpoint/Blender/073-00000000.pth',
                'pretrained_models/parsing.pth')

    src_path_list = ['dataset/select-align/img/id00061/2XrRfyv-EmE-0001/2122.png',
                    'dataset/select-align/img/id00061/2XrRfyv-EmE-0001/2125.png',
                    'dataset/select-align/img/id00061/2XrRfyv-EmE-0001/2130.png',
                    'dataset/select-align/img/id00061/2XrRfyv-EmE-0001/2135.png',
                    'dataset/select-align/img/id00061/2XrRfyv-EmE-0001/2140.png']
    tgt_path = 'dataset/select-align/img/id00061/4kSyBHethpE-0002/2055.png'
    oup = model.run(tgt_path,src_path_list)

    cv2.imwrite('2.png',oup)