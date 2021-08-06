from .mem_bank import RGBMem, CMCMem
from .mem_moco import RGBMoCo, CMCMoCo
from .mem_moco import RGBMoCo_dropmix_norm


def build_mem(opt, n_data):
    if opt.mem == 'bank':
        mem_func = RGBMem if opt.modal == 'RGB' else CMCMem
        memory = mem_func(opt.feat_dim, n_data,
                          opt.nce_k, opt.nce_t, opt.nce_m)

    elif opt.mem == 'moco':
        mem_func = RGBMoCo if opt.modal == 'RGB' else CMCMoCo

        if opt.mixnorm:
            mem_func = RGBMoCo_dropmix_norm
            memory = mem_func(opt.feat_dim, opt.nce_k, opt.nce_t, opt.mixnorm_target, 
                            opt.postmix_norm, opt.expolation_mask, opt.dim_mask, opt.mask_distribution, 
                            opt.beta_alpha, opt.norm_target, opt.pos_alpha, opt.neg_alpha, opt.sep_alpha)
        else:
            memory = mem_func(opt.feat_dim, opt.nce_k, opt.nce_t)
    else:
        raise NotImplementedError(
            'mem not suported: {}'.format(opt.mem))

    return memory
