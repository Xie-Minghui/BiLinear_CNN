class Config:

    def __init__(self,
        ):
        self.path_data = './data/'
        self.epochs = 128
        self.batch_size = 64

        self.path_continue = './vgg_16_epoch_5_acc_0.5633413876423887.ckpt'
        self.path_fc = './vgg_16_epoch_5_acc_0.5895754228512254.ckpt'
        # self.path_model = './vgg_16_epoch_.ckpt'
        self.use_pretrained = False
        
        if self.use_pretrained:
            self.epoch_continue = 5
        else:
            self.epoch_continue = 0

        self.train_all = True

        if self.train_all:
            self.lr = 1e-2
        else:
            self.lr = 1.0

config = Config()