class Config:

    def __init__(self,
        ):
        self.path_data = './data/'
        self.epochs = 128
        self.batch_size = 32

        self.path_continue = './vgg_16_epoch_5_acc_0.5633413876423887.ckpt'
        self.path_fc = './vgg_16_all_epoch_62_acc_0.6769071453227476.ckpt'
        # self.path_fc = './vgg_16_epoch_16_acc_0.6161546427338627.ckpt'
        # self.path_model = './vgg_16_epoch_.ckpt'
        self.use_pretrained = False
        
        if self.use_pretrained:
            self.epoch_continue = 5
        else:
            self.epoch_continue = 0

        self.train_all = True

        if self.train_all:
            self.lr = 1e-5
        else:
            self.lr = 1.0

config = Config()