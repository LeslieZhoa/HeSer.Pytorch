
'''
@author zhaoxiang
@date 20220812
'''
import torch 
from dataloader.AlignLoader import AlignData
from dataloader.BlendLoader import BlendData


def requires_grad(model, flag=True):
    if model is None:
        return 
    for p in model.parameters():
        p.requires_grad = flag
def need_grad(x):
    x = x.detach()
    x.requires_grad_()
    return x

def init_weights(m,init_type='normal', gain=0.02):
        
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            torch.nn.init.normal_(m.weight.data, 1.0, gain)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
            torch.nn.init.normal_(m.weight.data, 0.0, gain)
        elif init_type == 'xavier':
            torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
        elif init_type == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(m.weight.data, gain=1.0)
        elif init_type == 'kaiming':
            torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
            torch.nn.init.orthogonal_(m.weight.data, gain=gain)
        elif init_type == 'none':  # uses pytorch's default init method
            m.reset_parameters()
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)
def setup_seed(seed):
     torch.manual_seed(seed)
     if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def get_data_loader(args):
    if args.model == 'align':
        train_data = AlignData(dist=args.dist,
            size=args.size,
            root=args.train_root,
            frame_num=args.frame_num,
            skip_frame=args.skip_frame,
            use_pixelwise_augs=args.use_pixelwise_augs, 
            use_affine_scale=args.use_affine_scale,
            use_affine_shift=args.use_affine_shift,
            eval=False)
        
        test_data = AlignData(dist=args.dist,
            size=args.size,
            root=args.val_root,
            frame_num=args.frame_num,
            skip_frame=args.skip_frame,
            use_pixelwise_augs=False, 
            use_affine_scale=False,
            use_affine_shift=False,
            eval=True)

    elif args.model == 'blend':
        train_data = BlendData(dist=args.dist,
            size=args.size,
            root=args.train_root,eval=False)
        
        test_data = BlendData(dist=args.dist,
            size=args.size,
            root=args.val_root,eval=True)
    

    train_loader = torch.utils.data.DataLoader(
                        train_data,
                        batch_size=args.batch_size,
                        num_workers=args.nDataLoaderThread,
                        pin_memory=False,
                        drop_last=True
                    )
    test_loader = None if test_data is None else \
        torch.utils.data.DataLoader(
                        test_data,
                        batch_size=args.batch_size,
                        num_workers=args.nDataLoaderThread,
                        pin_memory=False,
                        drop_last=True
                    )
    return train_loader,test_loader,len(train_data) 



def merge_args(args,params):
   for k,v in vars(params).items():
      setattr(args,k,v)
   return args

def convert_img(img,unit=False):
   
    img = (img + 1) * 0.5
    if unit:
        return torch.clamp(img*255+0.5,0,255)
    
    return torch.clamp(img,0,1)