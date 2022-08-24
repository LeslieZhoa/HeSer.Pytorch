#! /usr/bin/python 
# -*- encoding: utf-8 -*-
'''
@author zhaoxiang
@date 20220812
'''

import os 

from torchvision import transforms 
import PIL.Image as Image
from dataloader.DataLoader import DatasetBase
import random
import torchvision.transforms.functional as F


class BlendData(DatasetBase):
    def __init__(self, slice_id=0, slice_count=1,dist=False, **kwargs):
        super().__init__(slice_id, slice_count,dist, **kwargs)


        self.transform = transforms.Compose([
            transforms.Resize((kwargs['size'], kwargs['size'])),
            transforms.ToTensor()
        ])
        self.color_fn = transforms.Compose([transforms.ColorJitter(0.5, 0.5, 0.5, 0.1)])
        
        
        self.norm = transforms.Compose([transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])

        self.gray = transforms.Compose([transforms.Grayscale(num_output_channels=1)])

        # source root
        root = kwargs['root']
        self.paths = [os.path.join(root,f) for f in os.listdir(root)]
        self.length = len(self.paths)
        random.shuffle(self.paths)
        self.eval = kwargs['eval']
        
       

    def __getitem__(self,i):
        
        idx = i % self.length
        id_path = self.paths[idx]
        video_paths = [os.path.join(id_path,f) for f in os.listdir(id_path)]
        vIdx = random.randint(0, len(video_paths) - 1)
        video_path = video_paths[vIdx]
        img_paths = [os.path.join(video_path,f) for f in os.listdir(video_path)]
        img_idx = random.randint(0, len(img_paths) - 1)
        img_path = img_paths[img_idx]
        mask_path = img_path.replace('img','mask')

        idx = (i + random.randint(0,self.length-1)) % self.length
        id_path = self.paths[idx]
        video_paths = [os.path.join(id_path,f) for f in os.listdir(id_path)]
        vIdx = random.randint(0, len(video_paths) - 1)
        video_path = video_paths[vIdx]
        img_paths = [os.path.join(video_path,f) for f in os.listdir(video_path)]
        img_idx = random.randint(0, len(img_paths) - 1)
        ex_img_path = img_paths[img_idx]
        ex_mask_path = ex_img_path.replace('img','mask')

        
        with Image.open(img_path) as img:
            gt = self.transform(img.convert('RGB'))
            I_a = self.transform(self.color_fn(img.convert('RGB')))
        gt = self.norm(gt)
        I_a = self.norm(I_a)
        with Image.open(mask_path) as img:
            M_a = self.transform(img.convert('L')) * 255

       
        I_gray = self.gray(I_a)
        if random.random() > 0.3:
            I_t = F.hflip(gt)
            M_t = F.hflip(M_a)
        
        else:
            I_t = gt 
            M_t = M_a 

        with Image.open(ex_img_path) as img:
            hat_t = self.transform(img.convert('RGB'))
        hat_t = self.norm(hat_t)
        with Image.open(ex_mask_path) as img:
            M_hat = self.transform(img.convert('L')) * 255

        return I_a,I_gray,I_t,hat_t,M_a,M_t,M_hat,gt

    def __len__(self):
        if self.eval:
            return 1000
        else:
            # return self.length
            return max(self.length,1000)

