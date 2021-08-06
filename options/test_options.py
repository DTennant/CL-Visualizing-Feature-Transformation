import os
from .base_options import BaseOptions


class TestOptions(BaseOptions):

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        parser.add_argument('--ckpt', type=str, default=None,
                            help='the checkpoint to test')
        parser.add_argument('--aug_linear', type=str, default='NULL',
                            choices=['NULL', 'RA'],
                            help='linear evaluation augmentation')
        parser.add_argument('--crop', type=float, default=0.2,
                            help='crop threshold for RandomResizedCrop')
        parser.add_argument('--n_class', type=int, default=1000,
                            help='number of classes for linear probing')

        
        parser.add_argument('--use_ema', action='store_true')

        parser.set_defaults(epochs=60)
        parser.set_defaults(learning_rate=30)
        parser.set_defaults(lr_decay_epochs='60,80')
        parser.set_defaults(lr_decay_rate=0.1)
        parser.set_defaults(weight_decay=0)

        return parser

    def modify_options(self, opt):
        opt = self.override_options(opt)

        iterations = opt.lr_decay_epochs.split(',')
        opt.lr_decay_epochs = list([])
        for it in iterations:
            opt.lr_decay_epochs.append(int(it))

        # set up saving name
        if opt.ckpt:
            opt.model_name = opt.ckpt.split('/')[-2]
        else:
            print('warning: no pre-trained model!')
            opt.model_name = 'Scratch'
        opt.model_name = '{}_linear_{}_{}_lr{}'.format(
            opt.model_name, opt.aug_linear, opt.crop, opt.learning_rate)
        if opt.cosine:
            opt.model_name = '{}_cosine'.format(opt.model_name)
        opt.model_name = '{}_lrdecay{}'.format(opt.model_name, opt.lr_decay_epochs)

        # create folders
        opt.model_folder = os.path.join(opt.model_path, opt.model_name)
        if not os.path.isdir(opt.model_folder):
            os.makedirs(opt.model_folder)
        opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
        if not os.path.isdir(opt.tb_folder):
            os.makedirs(opt.tb_folder)

        return opt
