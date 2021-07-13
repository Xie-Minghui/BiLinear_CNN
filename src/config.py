class Config:

    def __init__(self,
        ):
        self.path_data = './data/'
        self.epochs = 128
        self.lr = 5e-2
        self.batch_size = 8

        self.path_fc = './'
        self.path_model = './vgg_16_epoch_.ckpt'

config = Config()