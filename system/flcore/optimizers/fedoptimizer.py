import random
import torch
from torch.optim import Optimizer

class pFedMeOptimizer(Optimizer):
    def __init__(self, params, local_model=None, lr=0.01, lambdaa=0.1, mu=0.001):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, lambdaa=lambdaa, mu=mu)
        super(pFedMeOptimizer, self).__init__(params, defaults)
        self.weight_update = local_model.copy()
    def step(self):
        group = None
        for group in self.param_groups:
            for p, localweight in zip(group['params'], self.weight_update):
                localweight = localweight.to(p)
                # approximate local model
                p.data = p.data - group['lr'] * (p.grad.data + group['lambdaa'] * (p.data - localweight.data) + group['mu'] * p.data)

        return group['params']

