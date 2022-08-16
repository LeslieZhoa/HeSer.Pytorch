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
import math
import numpy as np 
import torch


class AlignData(DatasetBase):
    def __init__(self, slice_id=0, slice_count=1,dist=False, **kwargs):
        super().__init__(slice_id, slice_count,dist, **kwargs)


        # self.transform = transforms.Compose([
        #     transforms.Resize((kwargs['size'], kwargs['size'])),
        #      transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # ])

        # # source root
        # root = kwargs['root']
        # self.paths = [os.path.join(root,f) for f in os.listdir(root)]
        # self.length = len(self.paths)
        # random.shuffle(self.paths)

       

    def __getitem__(self,i):
        # idx = i % self.length
        # img_path = self.paths[idx]

        # with Image.open(img_path) as img:
        #     Img = self.transform(img)
        
        img = torch.from_numpy(np.random.uniform(-1,1,size=(3,512,512)).astype(np.float32))
        return img.unsqueeze(0),img,img


    def __len__(self):
        # return self.length
        return 4

