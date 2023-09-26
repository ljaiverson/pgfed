import copy
import torch
import torch.nn as nn
import numpy as np
from flcore.clients.clientbase import Client

class clientPer(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)

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
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()

    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()
