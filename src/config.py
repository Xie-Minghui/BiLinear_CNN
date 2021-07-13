class Config:

    def __init__(self,
           embedding_dim = 300
        ):
        self.path_data = './data/'
        self.epochs = 20
        self.lr = 1e-4

        self.embedding_dim = embedding_dim
        self.vocab_size = 1000
        self.num_filter = 32
        self.num_rel = 2
        self.batch_size = 4

config = Config()