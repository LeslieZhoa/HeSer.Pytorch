'''
@author zhaoxiang
@date 20220823
'''
class Params:
    def __init__(self):
       
        self.name = 'Blender'
        self.pretrain_path = 'checkpoint/Blender/025-00000000.pth'
        self.size = 512

        self.train_root = 'dataset/process/img'
        self.val_root = 'dataset/process/img'
        
        self.f_in_channels = 512
        self.f_inter_channels = 256
        self.temperature = 0.001
        self.dilate_kernel = 17
        self.decoder_ic = 12

        # discriminator
        self.embed_channels = 512
        self.padding = 'zero'
        self.in_channels = 5
        self.out_channels = 3
        self.num_channels = 64
        self.max_num_channels = 512
        self.output_image_size = 512
        self.dis_num_blocks = 7

        self.per_model = 'pretrained_models/vgg19-d01eb7cb.pth'
        
        # loss
        self.rec_loss = True 
        self.per_loss = True 
        self.lambda_gan = 0.2
        self.lambda_rec = 1.0
        self.lambda_per = 0.0005

        self.g_lr = 1e-4
        self.d_lr = 4e-4
        self.beta1 = 0.9
        self.beta2 = 0.999