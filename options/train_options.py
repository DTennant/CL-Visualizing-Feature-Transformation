import os
import math
from .base_options import BaseOptions


class TrainOptions(BaseOptions):

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        parser.add_argument('--aug', default='A', type=str,
                            help='data augmentation for training')
        parser.add_argument('--beta', type=float, default=0.5,
                            help='balance between Jigsaw and InsDis')
        parser.add_argument('--warm', action='store_true',
                            help='add warm-up setting')
        parser.add_argument('--amp', action='store_true',
                            help='using mixed precision')
        parser.add_argument('--opt_level', type=str, default='O2',
                            choices=['O1', 'O2'])

        parser.add_argument('--input_res', type=int, default=224)
        parser.add_argument('--feature_mix', action='store_true', help='use f mixup or not')
        parser.add_argument('--mix_ratio', type=float, default=1.0, help='the ratio of f mixed sample for contrast')
        parser.add_argument('--mix_alpha', type=float, default=0.2, help='the alpha for beta distribution')
        parser.add_argument('--time_mix', type=int, default=2, help='the time to do mixup')
        parser.add_argument('--only_mix', action='store_true', help='only using mixuped feature')


        parser.add_argument('--mixnorm', action='store_true')
        parser.add_argument('--mixnorm_target', type=str, choices=['pos', 'neg', 'posneg'], default='neg')
        parser.add_argument('--postmix_norm', action='store_true')
        parser.add_argument('--expolation_mask', action='store_true')
        parser.add_argument('--mask_distribution', type=str, choices=['uniform', 'beta'], default='beta')
        parser.add_argument('--dim_mask', type=str, choices=['pos', 'neg', 'both', 'none'], default='none')
        parser.add_argument('--beta_alpha', type=float, default=2.0)
        parser.add_argument('--sep_alpha', action='store_true')
        parser.add_argument('--pos_alpha', type=float, default=2.0)
        parser.add_argument('--neg_alpha', type=float, default=1.6)
        
        parser.add_argument('--norm_target', type=str, choices=['pos', 'neg'])

        parser.add_argument('--custom_head', type=str, help='using custom head')

        parser.add_argument('--exp_iter', type=int, default=-1, help='when not -1, will be written into the dir name')
        
        parser.add_argument('--aug_linear', type=str, default='NULL')
        parser.add_argument('--crop', type=float, default=0.2,
                            help='crop threshold for RandomResizedCrop')
        
        parser.add_argument('--mix_start_epoch', type=int, default=0)


        parser.add_argument('--save_score', action='store_true')

        return parser

    def modify_options(self, opt):
        opt = self.override_options(opt)
        
        if opt.custom_head != None:
            opt.head = opt.custom_head

        iterations = opt.lr_decay_epochs.split(',')
        opt.lr_decay_epochs = list([])
        for it in iterations:
            opt.lr_decay_epochs.append(int(it))

        # set up saving name
        opt.model_name = '{}_{}_{}_Jig_{}_{}_aug_{}_{}_{}'.format(
            opt.method, opt.arch, opt.modal, opt.jigsaw, opt.mem,
            opt.aug, opt.head, opt.nce_t
        )
        if opt.amp:
            opt.model_name = '{}_amp_{}'.format(opt.model_name, opt.opt_level)
        if opt.cosine:
            opt.model_name = '{}_cosine'.format(opt.model_name)


        # warm-up for large-batch training, e.g. 1024 with multiple nodes
        if opt.batch_size > 256:
            opt.warm = True
        if opt.warm:
            opt.model_name = '{}_warm'.format(opt.model_name)
            opt.warmup_from = 0.01
            if opt.epochs > 500:
                opt.warm_epochs = 10
            else:
                opt.warm_epochs = 5
            if opt.cosine:
                eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
                opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                            1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
            else:
                opt.warmup_to = opt.learning_rate

        opt.model_name = '{}_alpha{}'.format(opt.model_name, opt.alpha)
        opt.model_name = '{}_ncet{}'.format(opt.model_name, opt.nce_t)
        opt.model_name = '{}_ncek{}'.format(opt.model_name, opt.nce_k)
        # opt.model_name = '{}_head{}'.format(opt.model_name, opt.head)
        # opt.model_name = '{}_mixup{}_ratio{}'.format(opt.model_name, opt.feature_mix, opt.mix_ratio)
        if opt.mixnorm:
            if opt.expolation_mask:
                assert opt.mixnorm_target in ['pos', 'posneg']
            opt.model_name = '{}_mixnorm_target{}_postnorm{}_distri{}'.format(opt.model_name, opt.mixnorm_target, 
                                                                         opt.postmix_norm, opt.mask_distribution)
            if opt.mixnorm_target in ['pos', 'posneg']:
                opt.model_name = '{}_expolation{}'.format(opt.model_name, opt.expolation_mask)
            if opt.mask_distribution == 'beta' and not opt.sep_alpha:
                opt.model_name = '{}_alpha{}'.format(opt.model_name, opt.beta_alpha)
            if opt.sep_alpha and opt.mask_distribution == 'beta':
                opt.model_name = '{}_pos{}_neg{}'.format(opt.model_name, opt.pos_alpha, opt.neg_alpha)

        else:
            opt.model_name = '{}_baseline'.format(opt.model_name)
        

        opt.model_name = '{}_epoch{}'.format(opt.model_name, opt.epochs)
        opt.model_name = '{}_bs{}'.format(opt.model_name, opt.batch_size)
        opt.model_name = '{}_lr{}'.format(opt.model_name, opt.learning_rate)
        

        if opt.exp_iter != -1:
            opt.model_name = '{}_iter{}'.format(opt.model_name, opt.exp_iter)

        # create folders
        opt.model_folder = os.path.join(opt.model_path, opt.model_name)
        if not os.path.isdir(opt.model_folder):
            os.makedirs(opt.model_folder)
        opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
        if not os.path.isdir(opt.tb_folder):
            os.makedirs(opt.tb_folder)

        return opt
