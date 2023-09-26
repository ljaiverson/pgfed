import torch
import numpy as np
import copy
import torch.nn as nn
from flcore.clients.clientbase import Client
from utils.tensor_utils import model_dot_product

class clientPGFed(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        self.lambdaa = args.lambdaa # /ita_2 in paper, learning rate for a_i
        self.latest_grad = copy.deepcopy(self.model)
        self.prev_loss_minuses = {}
        self.prev_mean_grad = None
        self.prev_convex_comb_grad = None
        self.a_i = None

    def train(self):
        trainloader = self.load_train_data()
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
                output = self.model(x)
                loss = self.criterion(output, y)
                loss.backward()

                if self.prev_convex_comb_grad is not None:
                    for p_m, p_prev_conv in zip(self.model.parameters(), self.prev_convex_comb_grad.parameters()):
                        p_m.grad.data += p_prev_conv.data
                    dot_prod = model_dot_product(self.model, self.prev_mean_grad, requires_grad=False)
                    self.update_a_i(dot_prod)
                self.optimizer.step()
        
        # get loss_minus and latest_grad
        self.loss_minus = 0.0
        test_num = 0
        self.optimizer.zero_grad()
        for i, (x, y) in enumerate(trainloader):
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            test_num += y.shape[0]
            output = self.model(x)
            loss = self.criterion(output, y)
            self.loss_minus += (loss * y.shape[0]).item()
            loss.backward()

        self.loss_minus /= test_num
        for p_l, p in zip(self.latest_grad.parameters(), self.model.parameters()):
            p_l.data = p.grad.data.clone() / len(trainloader)
        self.loss_minus -= model_dot_product(self.latest_grad, self.model, requires_grad=False)

    def get_eval_model(self, temp_model=None):
        model = self.model if temp_model is None else temp_model
        return model

    def update_a_i(self, dot_prod):
        for clt_j, mu_loss_minus in self.prev_loss_minuses.items():
            self.a_i[clt_j] -= self.lambdaa * (mu_loss_minus + dot_prod)
            self.a_i[clt_j] = max(self.a_i[clt_j], 0.0)
    
    def set_model(self, old_m, new_m, momentum=0.0):
        for p_old, p_new in zip(old_m.parameters(), new_m.parameters()):
            p_old.data = (1 - momentum) * p_new.data.clone() + momentum * p_old.data.clone()

    def set_prev_mean_grad(self, mean_grad):
        if self.prev_mean_grad is None:
            self.prev_mean_grad = copy.deepcopy(mean_grad)
        else:
            self.set_model(self.prev_mean_grad, mean_grad)
        
    def set_prev_convex_comb_grad(self, convex_comb_grad, momentum=0.0):
        if self.prev_convex_comb_grad is None:
            self.prev_convex_comb_grad = copy.deepcopy(convex_comb_grad)
        else:
            self.set_model(self.prev_convex_comb_grad, convex_comb_grad, momentum=momentum)