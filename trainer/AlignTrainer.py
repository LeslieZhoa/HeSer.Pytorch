#! /usr/bin/python 
# -*- encoding: utf-8 -*-
'''
@author zhaoxiang
@date 20220812
'''
import torch 

from trainer.ModelTrainer import ModelTrainer
from model.AlignModule.lib import *
from model.AlignModule.discriminator import Discriminator
from itertools import chain
from utils.utils import *
import torch.nn.functional as F
from model.AlignModule.loss import *
import random
import torch.distributed as dist

class AlignTrainer(ModelTrainer):

    def __init__(self, args):
        super().__init__(args)
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
      
        self.Epor = PorEncoder(args).to(self.device)
        self.Eid = IDEncoder(args.id_model).to(self.device)
        self.Epose = PoseEncoder(args).to(self.device)
        self.Eexp = ExpEncoder(args).to(self.device)
        self.netG = Generator(args).to(self.device)

        self.netD = Discriminator(args).to(self.device)


        self.optimG,self.optimD = self.create_optimizer() 

        if args.pretrain_path is not None:
            self.loadParameters(args.pretrain_path)

        if args.dist:
            self.netG,self.netG_module = self.use_ddp(self.netG)
            self.Eexp,self.Eexp_module = self.use_ddp(self.Eexp)
            self.Epor,self.Epor_module = self.use_ddp(self.Epor)
            self.Epose,self.Epose_module = self.use_ddp(self.Epose)
            self.netD,self.netD_module = self.use_ddp(self.netD)
        else:
            self.netG_module = self.netG 
            self.Eexp_module = self.Eexp
            self.Epor_module = self.Epor
            self.Epose_module = self.Epose
            self.netD_module = self.netD
        
        if self.args.per_loss:
            self.perLoss = PerceptualLoss(args.per_model).to(self.device)
            self.perLoss.eval()
        
        if self.args.rec_loss:
            self.L1Loss = torch.nn.L1Loss()
        self.Eid.eval()

    def create_optimizer(self):
        g_optim = torch.optim.Adam(
                    chain(self.Epor.parameters(),self.Eexp.parameters(),
                    self.Epose.parameters(),self.netG.parameters()),
                    lr=self.args.g_lr,
                    betas=(self.args.beta1, self.args.beta2),
                    )
        d_optim = torch.optim.Adam(
                    self.netD.parameters(),
                    lr=self.args.d_lr,
                    betas=(self.args.beta1, self.args.beta2),
                    )
        
        return  g_optim,d_optim

    
    def run_single_step(self, data, steps):
        self.netG.train()
        self.Epor.train() 
        self.Epose.train() 
        self.Eexp.train() 
        super().run_single_step(data, steps)
        

    def run_discriminator_one_step(self, data,step):
        
        D_losses = {}
        requires_grad(self.netG, False)
        requires_grad(self.Epor, False)
        requires_grad(self.Epose, False)
        requires_grad(self.Eexp, False)
        requires_grad(self.netD, True)

        xs,xt,crop_xt,gt = data 
        xg = self.forward(xs,crop_xt,xt)
        fake_pred,fake_f = self.netD(xg)
        real_pred,real_f = self.netD(gt)
        d_loss = compute_dis_loss(fake_pred, real_pred,D_losses)
        D_losses['d'] = d_loss
        
        self.optimD.zero_grad()
        d_loss.backward()
        self.optimD.step()
        
        self.d_losses = D_losses


    def run_generator_one_step(self, data,step):
        
        
        requires_grad(self.netG, True)
        requires_grad(self.Epor, True)
        requires_grad(self.Epose, True)
        requires_grad(self.Eexp, True)
        requires_grad(self.netD, False)
        
        xs,xt,crop_xt,gt = data 
        G_losses,loss,xg = self.compute_g_loss(xs,crop_xt,xt,gt)
        self.optimG.zero_grad()
        loss.mean().backward()
        self.optimG.step()
        
        self.g_losses = G_losses
        
        self.generator = [xs[:,0].detach() if len(xs.shape)>4 else xs.detach(),xt.detach(),xg.detach(),gt.detach()]
        
    
    def evalution(self,test_loader,steps,epoch):
        
        loss_dict = {}
        index = random.randint(0,len(test_loader)-1)
        counter = 0
        with torch.no_grad():
            for i,data in enumerate(test_loader):
                
                data = self.process_input(data)
                xs,xt,crop_xt,gt = data 
                G_losses,losses,xg = self.compute_g_loss(xs,crop_xt,xt,gt)
                for k,v in G_losses.items():
                    loss_dict[k] = loss_dict.get(k,0) + v.detach()
                if i == index and self.args.rank == 0 :
                    
                    show_data = [xs[:,0],xt,xg,gt]
                    self.val_vis.display_current_results(self.select_img(show_data),steps)
                counter += 1
        
       
        for key,val in loss_dict.items():
            loss_dict[key] /= counter

        if self.args.dist:
            # if self.args.rank == 0 :
            dist_losses = loss_dict.copy()
            for key,val in loss_dict.items():
                
                dist.reduce(dist_losses[key],0)
                value = dist_losses[key].item()
                loss_dict[key] = value / self.args.world_size

        if self.args.rank == 0 :
            self.val_vis.plot_current_errors(loss_dict,steps)
            self.val_vis.print_current_errors(epoch+1,0,loss_dict,0)

        return loss_dict
       

    def forward(self,xs,crop_xt,xt):
       
        por_f = self.Epor(xs)
        id_f = self.Eid(self.process_id_input(xs,crop=True))

        pose_f = self.Epose(xt)
        exp_f = self.Eexp(self.process_id_input(crop_xt,crop=True,size=256))

        xg = self.netG(por_f,id_f,pose_f,exp_f)
       
        return xg

    def compute_g_loss(self,xs,crop_xt,xt,gt):
        G_losses = {}
        loss = 0
        xg = self.forward(xs,crop_xt,xt)
        fake_pred,fake_f = self.netD(xg)
        gan_loss = compute_gan_loss(fake_pred) * self.args.lambda_gan
        G_losses['g_losses'] = gan_loss
        loss += gan_loss
        
        if self.args.rec_loss:
            rec_loss = self.L1Loss(xg,gt) * self.args.lambda_rec 
            G_losses['rec_loss'] = rec_loss
            loss += rec_loss
        
        if self.args.id_loss:
            fake_id_f = self.Eid(self.process_id_input(xg,crop=True))
            real_id_f = self.Eid(self.process_id_input(gt,crop=True))
            id_loss = compute_id_loss(fake_id_f,real_id_f).mean() * self.args.lambda_id 
            G_losses['id_loss'] = id_loss 
            loss += id_loss 

        if self.args.per_loss:
            per_loss = self.perLoss(xg,gt) * self.args.lambda_per 
            G_losses['per_loss'] = per_loss
            loss += per_loss 

        return G_losses,loss,xg

    @staticmethod
    def process_id_input(x,crop=False,size=112):
        c,h,w = x.shape[-3:]
        batch = x.shape[0]
        scale = 0.4 / 1.8
        if crop:
            crop_x = x[...,int(h*scale):int(-h*scale),int(w*scale):int(-w*scale)]
        else:
            crop_x = x
        if len(x.shape) > 4:
            resize_x = F.adaptive_avg_pool2d(crop_x.view(-1,*crop_x.shape[-3:]),size)
            resize_x = resize_x.view(batch,-1,c,size,size)
        else:
            resize_x = F.adaptive_avg_pool2d(crop_x,size)
        return resize_x
    def get_latest_losses(self):
        return {**self.g_losses,**self.d_losses}

    def get_latest_generated(self):
        return self.generator

    def loadParameters(self,path):
        ckpt = torch.load(path, map_location=lambda storage, loc: storage)
        self.netG.load_state_dict(ckpt['G'],strict=False)
        self.Eexp.load_state_dict(ckpt['Eexp'],strict=False)
        self.Eid.load_state_dict(ckpt['Eid'],strict=False)
        self.Epor.load_state_dict(ckpt['Epor'],strict=False)
        self.Epose.load_state_dict(ckpt['Epose'],strict=False)
        self.netD.load_state_dict(ckpt['D'],strict=False)
        self.optimG.load_state_dict(ckpt['g_optim'])
        self.optimD.load_state_dict(ckpt['d_optim'])

    def saveParameters(self,path):
        torch.save(
                    {
                        "G": self.netG_module.state_dict(),
                        'D':self.netD_module.state_dict(),
                        "Eexp": self.Eexp_module.state_dict(),
                        "Eid":self.Eid.state_dict(),
                        'Epor':self.Epor_module.state_dict(),
                        'Epose':self.Epose_module.state_dict(),
                        "g_optim": self.optimG.state_dict(),
                        "d_optim": self.optimD.state_dict(),
                        "args": self.args,
                    },
                    path
                )

    def get_lr(self):
        return self.optimG.state_dict()['param_groups'][0]['lr']

    
    def select_img(self, data, name='fake', axis=2):
        data = [F.adaptive_avg_pool2d(x,self.args.output_image_size) for x in data]
        return super().select_img(data, name, axis)
    


    
            


        


    
    

    
