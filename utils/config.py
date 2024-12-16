from pprint import pprint


# Default Configs for training
# NOTE that, config items could be overwriten
# by passing argument through command line.
# e.g. --voc-data-dir='./data/'

class Config:
    # data
    # rootpath = './data/data64/'
    rootpath = 'E:/DATA/2007BP/part1_128/'
    testpath = './data/test/seismic2.npy'
    use_augment = True
    gpu_num = 1
    use_cuda = True
    num_workers = 1

    # param for optimizer
    weight_decay = 0.0
    lr_decay = 0.1  # 1e-3 -> 1e-4
    lr = 1e-3  # 1e-3
    lr_step = 2*500  #128:7  56

    # training
    iternum = 3600000*3
    train_batch_size = 60
    test_batch_size = 1

    test_num = 1000
    # model
    test_only = False
    load_net = './checkpoints/Unet128_sigb2_0.009973560646176338.pth'
    # detect
    savepath = "./result/"


    def _parse(self):
        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items()
                if not k.startswith('_')}


opt = Config()
