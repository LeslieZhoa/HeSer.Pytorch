#! /usr/bin/python 
# -*- encoding: utf-8 -*-
'''
@author zhaoxiang
@date 20220812
'''
import torch 

from trainer.ModelTrainer import ModelTrainer
from model.BlendModule.generator import Generator
from model.AlignModule.discriminator import Discriminator
from utils.utils import *
from model.AlignModule.loss import *
import torch.nn.functional as F
import random
import torch.distributed as dist

class BlendTrainer(ModelTrainer):

    def __init__(self, args):
        super().__init__(args)
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
      
        self.netG = Generator(args).to(self.device)

        self.netD = Discriminator(args).to(self.device)


        self.optimG,self.optimD = self.create_optimizer() 

        if args.pretrain_path is not None:
            self.loadParameters(args.pretrain_path)

        if args.dist:
            self.netG,self.netG_module = self.use_ddp(self.netG)
            self.netD,self.netD_module = self.use_ddp(self.netD)
        else:
            self.netG_module = self.netG 
            self.netD_module = self.netD
        
        if self.args.per_loss:
            self.perLoss = PerceptualLoss(args.per_model).to(self.device)
            self.perLoss.eval()
        
        if self.args.rec_loss:
            self.L1Loss = torch.nn.L1Loss()
        

    def create_optimizer(self):
        g_optim = torch.optim.Adam(
                    self.netG.parameters(),
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
        super().run_single_step(data, steps)
        

    def run_discriminator_one_step(self, data,step):
        
        D_losses = {}
        requires_grad(self.netG, False)
        requires_grad(self.netD, True)

        I_a,I_gray,I_t,hat_t,M_a,M_t,M_hat,gt = data 
        fake,M_Ah,M_Ai = self.netG(I_a,I_gray,I_t,M_a,M_t,gt,train=True)
        fake_pred,fake_f = self.netD(torch.cat([fake,M_Ah,M_Ai],1))
        real_pred,real_f = self.netD(torch.cat([gt,M_Ah,M_Ai],1))
        d_loss = compute_dis_loss(fake_pred, real_pred,D_losses)
        D_losses['d'] = d_loss
        
        self.optimD.zero_grad()
        d_loss.backward()
        self.optimD.step()
        
        self.d_losses = D_losses


    def run_generator_one_step(self, data,step):
        
        
        requires_grad(self.netG, True)
        requires_grad(self.netD, False)
        
        I_a,I_gray,I_t,hat_t,M_a,M_t,M_hat,gt = data 
        G_losses,loss,xg = self.compute_g_loss(I_a,I_gray,I_t,M_a,M_t,gt)
        self.optimG.zero_grad()
        loss.mean().backward()
        self.optimG.step()

        g_losses,loss,fake_nopair,label_nopair = self.compute_cycle_g_loss(I_a,I_gray,I_t,hat_t,M_a,M_t,M_hat)
        self.optimG.zero_grad()
        loss.mean().backward()
        self.optimG.step()
        
        self.g_losses = {**G_losses,**g_losses}
        
        self.generator = [I_a.detach(),fake_nopair.detach(),
                        label_nopair.detach(),xg.detach(),gt.detach()]
        
    
    def evalution(self,test_loader,steps,epoch):
        
        loss_dict = {}
        index = random.randint(0,len(test_loader)-1)
        counter = 0
        with torch.no_grad():
            for i,data in enumerate(test_loader):
                
                data = self.process_input(data)
                I_a,I_gray,I_t,hat_t,M_a,M_t,M_hat,gt = data 
                G_losses,losses,xg = self.compute_g_loss(I_a,I_gray,I_t,M_a,M_t,gt)
                for k,v in G_losses.items():
                    loss_dict[k] = loss_dict.get(k,0) + v.detach()
                if i == index and self.args.rank == 0 :
                    
                    show_data = [I_a,xg,gt]
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
       

    def compute_g_loss(self,I_a,I_gray,I_t,M_a,M_t,gt):
        G_losses = {}
        loss = 0
        fake,M_Ah,M_Ai = self.netG(I_a,I_gray,I_t,M_a,M_t,gt,train=True)
        fake_pred,fake_f = self.netD(torch.cat([fake,M_Ah,M_Ai],1))
        gan_loss = compute_gan_loss(fake_pred) * self.args.lambda_gan
        G_losses['g_losses'] = gan_loss
        loss += gan_loss
        
        if self.args.rec_loss:
            rec_loss = self.L1Loss(fake,gt) * self.args.lambda_rec 
            G_losses['rec_loss'] = rec_loss
            loss += rec_loss
        

        if self.args.per_loss:
            per_loss = self.perLoss(fake,gt) * self.args.lambda_per 
            G_losses['per_loss'] = per_loss
            loss += per_loss 

        return G_losses,loss,fake

    
    def compute_cycle_g_loss(self,I_a,I_gray,I_t,hat_t,M_a,M_t,M_hat):
        G_losses = {}
        loss = 0
        fake_pair,label_pair = self.netG(I_a,I_gray,I_t,M_a,M_t,cycle=True)
        fake_nopair,label_nopair = self.netG(I_a,I_gray,hat_t,M_a,M_hat,cycle=True)
        
        loss = self.L1Loss(fake_pair,label_pair) + \
            self.L1Loss(fake_nopair,label_nopair)
        G_losses['cycle'] = loss
        fake_nopair = F.interpolate(fake_nopair, size=I_t.shape[-2:],mode='bilinear')
        label_nopair = F.interpolate(label_nopair, size=I_t.shape[-2:],mode='bilinear')
        return G_losses,loss,fake_nopair,label_nopair

    
    def get_latest_losses(self):
        return {**self.g_losses,**self.d_losses}

    def get_latest_generated(self):
        return self.generator

    def loadParameters(self,path):
        ckpt = torch.load(path, map_location=lambda storage, loc: storage)
        self.netG.load_state_dict(ckpt['G'],strict=False)
        self.netD.load_state_dict(ckpt['D'],strict=False)
        self.optimG.load_state_dict(ckpt['g_optim'])
        self.optimD.load_state_dict(ckpt['d_optim'])

    def saveParameters(self,path):
        torch.save(
                    {
                        "G": self.netG_module.state_dict(),
                        "D": self.netD_module.state_dict(),
                        "g_optim": self.optimG.state_dict(),
                        "d_optim": self.optimD.state_dict(),
                        "args": self.args,
                    },
                    path
                )

    def get_lr(self):
        return self.optimG.state_dict()['param_groups'][0]['lr']
    


    
            


        


    
    

    
