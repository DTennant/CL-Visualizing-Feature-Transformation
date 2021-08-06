import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F


class BaseMoCo(nn.Module):
    """base class for MoCo-style memory cache"""
    def __init__(self, K=65536, T=0.07):
        super(BaseMoCo, self).__init__()
        self.K = K
        self.T = T
        self.index = 0

    def _update_pointer(self, bsz):
        self.index = (self.index + bsz) % self.K

    def _update_memory(self, k, queue):
        """
        Args:
          k: key feature
          queue: memory buffer
        """
        with torch.no_grad():
            num_neg = k.shape[0]
            out_ids = torch.arange(num_neg).cuda()
            out_ids = torch.fmod(out_ids + self.index, self.K).long()
            queue.index_copy_(0, out_ids, k)

    # def _compute_logit(self, q, k, queue):
    #     """
    #     Args:
    #       q: query/anchor feature
    #       k: key feature
    #       queue: memory buffer
    #     """
    #     # pos logit
    #     bsz = q.shape[0]
    #     pos = torch.bmm(q.view(bsz, 1, -1), k.view(bsz, -1, 1))
    #     pos = pos.view(bsz, 1)

    #     # neg logit
    #     neg = torch.mm(queue, q.transpose(1, 0))
    #     neg = neg.transpose(0, 1)

    #     out = torch.cat((pos, neg), dim=1)
    #     out = torch.div(out, self.T)
    #     out = out.squeeze().contiguous()

    #     return out
    
    def _compute_logit(self, q, k, queue):
        """
        Args:
          q: query/anchor feature
          k: key feature
          queue: memory buffer
        """

        bsz = q.shape[0]
        pos = torch.bmm(q.view(bsz, 1, -1), k.view(bsz, -1, 1))
        pos = pos.view(bsz, 1)

        neg = torch.mm(queue, q.transpose(1, 0))
        neg = neg.transpose(0, 1)


        score_neg_mean = torch.mean(neg, axis=1, keepdim=True).half().cpu().detach().numpy()
        score_neg_var = torch.var(neg, axis=1, keepdim=True).half().cpu().detach().numpy()
        score_pos = pos.half().cpu().detach().numpy()
        exp_neg_mean = torch.mean(torch.exp(neg / self.T), axis=1, keepdim=True).cpu().detach().numpy()
        exp_pos_mean = torch.exp(pos / self.T).cpu().detach().numpy()
        score = np.concatenate((score_neg_mean, score_neg_var, score_pos, exp_neg_mean, exp_pos_mean), axis=1)
        
        out = torch.cat((pos, neg), dim=1)
        out = torch.div(out, self.T)
        out = out.squeeze().contiguous()

        return out, score



class RGBMoCo(BaseMoCo):
    """Single Modal (e.g., RGB) MoCo-style cache"""
    def __init__(self, n_dim, K=65536, T=0.07):
        super(RGBMoCo, self).__init__(K, T)
        # create memory queue
        self.register_buffer('memory', torch.randn(K, n_dim))
        self.memory = F.normalize(self.memory)

    def forward(self, q, k, q_jig=None, all_k=None):
        """
        Args:
          q: query on current node
          k: key on current node
          q_jig: jigsaw query
          all_k: gather of feats across nodes; otherwise use q
        """
        bsz = q.size(0)
        k = k.detach()

        # compute logit
        queue = self.memory.clone().detach()
        logits, score = self._compute_logit(q, k, queue)
        if q_jig is not None:
            logits_jig = self._compute_logit(q_jig, k, queue)

        # set label
        labels = torch.zeros(bsz, dtype=torch.long).cuda()

        # update memory
        all_k = all_k if all_k is not None else k
        self._update_memory(all_k, self.memory)
        self._update_pointer(all_k.size(0))

        if q_jig is not None:
            return logits, logits_jig, labels
        else:
            return logits, labels

class RGBMoCo_dropmix_norm(BaseMoCo):
    def __init__(self, n_dim, K=65536, T=0.07, 
                    mix_target='pos', postmix_norm=False, expolation_mask=False,
                    dim_mask='both', mask_distribution='uniform', alpha=2.0, norm_target='pos',
                    pos_alpha=2.0, neg_alpha=1.6, sep_alpha=False, mix_jig=False):
        super(RGBMoCo_dropmix_norm, self).__init__(K, T)
        # create memory queue
        self.register_buffer('memory', torch.randn(K, n_dim))
        self.memory = F.normalize(self.memory)

        assert mix_target in ['pos', 'neg', 'posneg']
        self.mix_target = mix_target
        self.postmix_norm = postmix_norm
        self.expolation_mask = expolation_mask

        assert mask_distribution in ['uniform', 'beta']
        self.mask_distribution = mask_distribution
        assert dim_mask in ['pos', 'neg', 'both', 'none']
        self.dim_mask = dim_mask

        self.alpha = alpha
        self.pos_alpha = pos_alpha
        self.neg_alpha = neg_alpha
        self.sep_alpha = sep_alpha

        if self.expolation_mask:
            assert self.mix_target in ['pos', 'posneg']
        
        self.mix_jig = mix_jig
        
        self.norm_target = 'pos'


    def forward(self, q, k, q_jig=None, all_k=None, mix_now=True):
        bsz = q.size(0)
        k = k.detach()
        ori_k = k.clone().detach()

        # compute logit
        queue = self.memory.clone().detach()
        ori_queue = self.memory.clone().detach()

        """
            Mixing targets
        """
        if mix_now:
            if self.mix_target == 'pos':
                mask_shape = q.shape
                if self.mask_distribution == 'uniform':
                    mask = torch.rand(size=mask_shape).cuda()
                elif self.mask_distribution == 'beta':
                    mask = np.random.beta(self.alpha, self.alpha, size=mask_shape)
                
                if self.expolation_mask:
                    mask += 1
                if isinstance(mask, np.ndarray):
                    mask = torch.from_numpy(mask).float().cuda()

                q_mix = mask * q + (1 - mask) * k
                k_mix = mask * k + (1 - mask) * q
                q, k = q_mix, k_mix

            elif self.mix_target == 'posneg':
                pos_mask_shape = q.shape
                neg_mask_shape = queue.shape
                if self.mask_distribution == 'uniform':
                    pos_mask = torch.rand(size=pos_mask_shape).cuda()
                    neg_mask = torch.rand(size=neg_mask_shape).cuda()
                elif self.mask_distribution == 'beta':
                    pos_mask = np.random.beta(self.alpha, self.alpha, size=pos_mask_shape)
                    neg_mask = np.random.beta(self.alpha, self.alpha, size=neg_mask_shape)

                    if self.sep_alpha:
                        pos_mask = np.random.beta(self.pos_alpha, self.pos_alpha, size=pos_mask_shape)
                        neg_mask = np.random.beta(self.neg_alpha, self.neg_alpha, size=neg_mask_shape)
                        
                        if self.dim_mask == 'none':
                            pos_mask = np.random.beta(self.pos_alpha, self.pos_alpha)
                            neg_mask = np.random.beta(self.neg_alpha, self.neg_alpha)
                        elif self.dim_mask == 'pos':
                            neg_mask = np.random.beta(self.neg_alpha, self.neg_alpha)
                        elif self.dim_mask == 'neg':
                            pos_mask = np.random.beta(self.pos_alpha, self.pos_alpha)
                        elif self.dim_mask == 'both':
                            pass

                if self.expolation_mask:
                    pos_mask += 1
                if isinstance(pos_mask, np.ndarray):
                    pos_mask = torch.from_numpy(pos_mask).float().cuda()
                if isinstance(neg_mask, np.ndarray):
                    neg_mask = torch.from_numpy(neg_mask).float().cuda()
                q_mix = pos_mask * q + (1 - pos_mask) * k
                k_mix = pos_mask * k + (1 - pos_mask) * q
                q, k = q_mix, k_mix

                indices = torch.randperm(queue.shape[0]).cuda()
                queue = neg_mask * queue + (1 - neg_mask) * queue[indices]

            else:
                mask_shape = queue.shape
                if self.mask_distribution == 'uniform':
                    mask = torch.rand(size=mask_shape).cuda()
                elif self.mask_distribution == 'beta':
                    mask = np.random.beta(self.alpha, self.alpha, size=mask_shape)

                if isinstance(mask, np.ndarray):
                    mask = torch.from_numpy(mask).float().cuda()

                indices = torch.randperm(queue.shape[0]).cuda()
                queue = mask * queue + (1 - mask) * queue[indices]

            if self.postmix_norm:
                if self.norm_target == 'pos':
                    q, k = F.normalize(q, dim=1), F.normalize(k, dim=1)
                elif self.norm_target == 'neg':
                    queue = F.normalize(queue, dim=1)
                else:
                    q, k = F.normalize(q, dim=1), F.normalize(k, dim=1)
                    queue = F.normalize(queue, dim=1)
        else:
            print('not mixing')

        logits, score = self._compute_logit(q, k, queue)
        # logits = self._compute_logit(q, k, queue)
        
        if q_jig is not None:
            
            if 'pos' in self.mix_target and self.mix_jig:
                mask_shape = q_jig.shape
                if self.mask_distribution == 'uniform':
                    mask = torch.rand(size=mask_shape).cuda()
                elif self.mask_distribution == 'beta':
                    mask = np.random.beta(self.alpha, self.alpha, size=mask_shape)
                
                if self.expolation_mask:
                    mask += 1
                if isinstance(mask, np.ndarray):
                    mask = torch.from_numpy(mask).float().cuda()

                k = ori_k
                q_mix = mask * q_jig + (1 - mask) * k
                k_mix = mask * k + (1 - mask) * q_jig
                q_jig, k = q_mix, k_mix
            
            logits_jig = self._compute_logit(q_jig, ori_k, ori_queue)

        # set label
        labels = torch.zeros(bsz, dtype=torch.long).cuda()

        # update memory
        all_k = all_k if all_k is not None else k
        all_k = all_k.float()
        # print(f'self.memory.dtype: {self.memory.dtype}')
        # print(f'all_k.dtype: {all_k.dtype}')
        self._update_memory(all_k, self.memory)
        self._update_pointer(all_k.size(0))

        if q_jig is not None:
            return logits, logits_jig, labels
        else:
            return [logits, labels], score





class CMCMoCo(BaseMoCo):
    """MoCo-style memory for two modalities, e.g. in CMC"""
    def __init__(self, n_dim, K=65536, T=0.07):
        super(CMCMoCo, self).__init__(K, T)
        # create memory queue
        self.register_buffer('memory_1', torch.randn(K, n_dim))
        self.register_buffer('memory_2', torch.randn(K, n_dim))
        self.memory_1 = F.normalize(self.memory_1)
        self.memory_2 = F.normalize(self.memory_2)

    def forward(self, q1, k1, q2, k2,
                q1_jig=None, q2_jig=None,
                all_k1=None, all_k2=None):
        """
        Args:
          q1: q of modal 1
          k1: k of modal 1
          q2: q of modal 2
          k2: k of modal 2
          q1_jig: q jig of modal 1
          q2_jig: q jig of modal 2
          all_k1: gather of k1 across nodes; otherwise use k1
          all_k2: gather of k2 across nodes; otherwise use k2
        """
        bsz = q1.size(0)
        k1 = k1.detach()
        k2 = k2.detach()

        # compute logit
        queue1 = self.memory_1.clone().detach()
        queue2 = self.memory_2.clone().detach()
        logits1 = self._compute_logit(q1, k2, queue2)
        logits2 = self._compute_logit(q2, k1, queue1)
        if (q1_jig is not None) and (q2_jig is not None):
            logits1_jig = self._compute_logit(q1_jig, k2, queue2)
            logits2_jig = self._compute_logit(q2_jig, k1, queue1)

        # set label
        labels = torch.zeros(bsz, dtype=torch.long).cuda()

        # update memory
        all_k1 = all_k1 if all_k1 is not None else k1
        all_k2 = all_k2 if all_k2 is not None else k2
        assert all_k1.size(0) == all_k2.size(0)
        self._update_memory(all_k1, self.memory_1)
        self._update_memory(all_k2, self.memory_2)
        self._update_pointer(all_k1.size(0))

        if (q1_jig is not None) and (q2_jig is not None):
            return logits1, logits2, logits1_jig, logits2_jig, labels
        else:
            return logits1, logits2, labels





