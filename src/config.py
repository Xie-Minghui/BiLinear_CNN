class Config:

    def __init__(self,
        ):
        self.path_data = './data/'
        self.epochs = 128
        self.lr = 1.0
        self.batch_size = 128

        self.path_continue = './vgg_16_epoch_5_acc_0.5633413876423887.ckpt'
        self.path_fc = './'
        self.path_model = './vgg_16_epoch_.ckpt'
        self.use_pretrained = True
        self.epoch_continue = 5

config = Config()