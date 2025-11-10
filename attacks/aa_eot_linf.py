import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange
from autoattack import AutoAttack

class AutoAttackLinf:
    def __init__(self, get_logit, attack_steps=200, eps=0.5, step_size=0.007, eot=20):
        self.eps = eps
        self.get_logit = get_logit
        self.attack_steps = attack_steps
        self.eot = eot
        self.attack_version = 'rand'


    def __call__(self, x, y):
        x_adv = self.forward(x, y)
        return x_adv

    def forward(self, x, y):
        if self.attack_version == 'standard':
            attack_list = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
        elif self.attack_version == 'rand':
            attack_list = ['apgd-ce', 'apgd-dlr']
        else:
            raise NotImplementedError(f'Unknown attack version: {self.attack_version}!')
        
        # 修改这里：如果要使用自定义攻击列表，必须设置version='custom'
        if self.attack_version in ['standard', 'rand']:
            # 对于预定义版本，不要指定attacks_to_run
            adversary_sde = AutoAttack(self.get_logit, norm='Linf', eps=self.eps,
                                    version=self.attack_version, device=x.device)
        else:
            # 只有自定义版本才使用attacks_to_run
            adversary_sde = AutoAttack(self.get_logit, norm='Linf', eps=self.eps,
                                    version='custom', attacks_to_run=attack_list, device=x.device)
        
        x_adv_sde = adversary_sde.run_standard_evaluation(x, y, bs=x.shape[0])
        return x_adv_sde