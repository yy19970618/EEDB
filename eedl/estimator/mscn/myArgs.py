class myArgs():
    def __init__(self, **kwargs):
        self.bs = 1024
        self.lr = 0.001
        self.epochs = 30
        self.num_samples = 600
        self.hid_units = 4
        self.train_num = 100000

        # overwrite parameters from user
        self.__dict__.update(kwargs)