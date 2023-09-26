import copy
import torch
import torch.nn as nn
import numpy as np
from flcore.clients.clientbase import Client

class clientAPFL(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        self.alpha = args.alpha
        self.model_local = copy.deepcopy(self.model)
        self.optimizer_local = torch.optim.SGD(self.model_local.parameters(), lr=self.learning_rate)
        self.model_per = copy.deepcopy(self.model)
        self.optimizer_per = torch.optim.SGD(self.model_per.parameters(), lr=self.learning_rate)

    def set_parameters(self, model):
        for new_param, old_param, param_l, param_p in zip(model.parameters(), self.model.parameters(),
            self.model_local.parameters(), self.model_per.parameters()):
            old_param.data = new_param.data.clone()
            param_p.data = self.alpha * param_l.data + (1 - self.alpha) * new_param.data

    def train(self):
        trainloader = self.load_train_data()
        self.model.train()

        max_local_steps = self.local_steps

        # self.model_per: personalized model (v_bar), self.model: global_model (w)
        for step in range(max_local_steps):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                # update global model (self.model)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()

                # update local model (self.model_local) grad_(v_bar) = 
                self.optimizer_per.zero_grad()
                output_per = self.model_per(x)
                loss_per = self.criterion(output_per, y)
                loss_per.backward() # update (by gradient) model_local before updating model_per (by interpolation)

                # update model_local by gradient (gradient is alpha * grad(model_per))
                # see https://github.com/lgcollins/FedRep/blob/main/models/Update.py#L410 and the algorithm in paper
                self.optimizer_local.zero_grad()
                for p_l, p_p in zip(self.model_local.parameters(), self.model_per.parameters()):
                    if p_l.grad is None:
                        p_l.grad = self.alpha * p_p.grad.data.clone()
                    else:
                        p_l.grad.data = self.alpha * p_p.grad.data.clone()
                self.optimizer_local.step()
                
                # update model_per by interpolation
                for p_p, p_g, p_l in zip(self.model_per.parameters(), self.model.parameters(), self.model_local.parameters()):
                    p_p.data = self.alpha * p_l.data + (1 - self.alpha) * p_g.data


