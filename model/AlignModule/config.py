class Params:
    def __init__(self):
       
        self.name = 'Aligner'
        self.pretrain_path = 'checkpoint/Aligner/038-00000200.pth'
        self.size = 512

        self.train_root = 'dataset/process/img'
        self.val_root = 'dataset/process/img'
        self.use_pixelwise_augs = True 
        self.use_affine_scale = True
        self.use_affine_shift = True
        self.frame_num = 5
        self.skip_frame = 5

        self.identity_embedding_size = 512
        self.pose_embedding_size = 256
        self.por_embedding_size = 512
        self.exp_embedding_size = 256
        self.embed_channels = 512
        self.padding = 'zero'
        self.in_channels = 3
        self.out_channels = 3
        self.num_channels = 64
        self.max_num_channels = 512
        self.norm_layer = 'in'
        self.gen_constant_input_size = 4
        self.gen_num_residual_blocks = 2
        self.output_image_size = 512

        self.dis_num_blocks = 7

        self.id_model = 'pretrained_models/model_ir_se50.pth'
        self.per_model = 'pretrained_models/vgg19-d01eb7cb.pth'

        self.rec_loss = True 
        self.id_loss = True 
        self.per_loss = True 
        self.lambda_gan = 0.2
        self.lambda_rec = 1.0
        self.lambda_id = 0.2
        self.lambda_per = 0.0005

        self.g_lr = 1e-4
        self.d_lr = 4e-4
        self.beta1 = 0.9
        self.beta2 = 0.999