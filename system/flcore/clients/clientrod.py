import copy
import torch
import torch.nn as nn
import numpy as np
from flcore.clients.clientbase import Client
import torch.nn.functional as F

class RodEvalModel(nn.Module):
    def __init__(self, glob_m, pers_pred):
        super(RodEvalModel, self).__init__()
        self.glob_m = glob_m
        self.pers_pred = pers_pred
    def forward(self, x):
        rep = self.glob_m.base(x)
        out = self.glob_m.predictor(rep)
        out += self.pers_pred(rep)
        return out

class clientRoD(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        
        self.pred = copy.deepcopy(self.model.predictor)
        self.opt_pred = torch.optim.SGD(self.pred.parameters(), lr=self.learning_rate)

        self.sample_per_class = torch.zeros(self.num_classes)
        trainloader = self.load_train_data()
        for x, y in trainloader:
            for yy in y:
                self.sample_per_class[yy.item()] += 1
        self.sample_per_class = self.sample_per_class / torch.sum(self.sample_per_class)


    def train(self):
        trainloader = self.load_train_data()

        # self.model.to(self.device)
        self.model.train()

        max_local_steps = self.local_steps

        for step in range(max_local_steps):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                rep = self.model.base(x)
                out_g = self.model.predictor(rep)
                loss_bsm = balanced_softmax_loss(y, out_g, self.sample_per_class)
                loss_bsm.backward()
                self.optimizer.step()
                
                self.opt_pred.zero_grad()
                out_p = self.pred(rep.detach())
                loss = self.criterion(out_g.detach() + out_p, y)
                loss.backward()
                self.opt_pred.step()

        # self.model.cpu()

    # comment for testing on new clients
    def get_eval_model(self, temp_model=None):
        # temp_model is the current round global model (after aggregation)
        return RodEvalModel(temp_model, self.pred)

# https://github.com/jiawei-ren/BalancedMetaSoftmax-Classification
def balanced_softmax_loss(labels, logits, sample_per_class, reduction="mean"):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss
