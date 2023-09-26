import copy
import torch
import torch.nn as nn
import numpy as np
from flcore.clients.clientbase import Client
from utils.tensor_utils import l2_squared_diff, model_dot_product

class clientDyn(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)

        self.alpha = args.alpha

        self.global_model_vector = None
        self.old_grad = copy.deepcopy(self.model)
        for p in self.old_grad.parameters():
            p.requires_grad = False
            p.data.zero_()
        
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
                output = self.model(x)
                loss = self.criterion(output, y)

                if self.untrained_global_model != None:
                    loss += self.alpha/2 * l2_squared_diff(self.model, self.untrained_global_model)
                    loss -= model_dot_product(self.model, self.old_grad)

                loss.backward()
                self.optimizer.step()

        if self.untrained_global_model != None:
            for p_old_grad, p_cur, p_broadcast in zip(self.old_grad.parameters(), self.model.parameters(), self.untrained_global_model.parameters()):
                p_old_grad.data -= self.alpha * (p_cur.data - p_broadcast.data)

        # self.model.cpu()
            
    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()
        self.untrained_global_model = copy.deepcopy(model)
        for p in self.untrained_global_model.parameters():
            p.requires_grad = False
