import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn import metrics
import copy
from flcore.clients.clientbase import Client

class clientRep(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.base.parameters(), lr=self.learning_rate)
        self.poptimizer = torch.optim.SGD(self.model.predictor.parameters(), lr=self.learning_rate)

        self.plocal_steps = args.plocal_steps

    def train(self):
        trainloader = self.load_train_data()

        # self.model.to(self.device)
        self.model.train()

        for param in self.model.base.parameters():
            param.requires_grad = False
        for param in self.model.predictor.parameters():
            param.requires_grad = True

        for step in range(self.plocal_steps):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                self.poptimizer.zero_grad()
                output = self.model(x)
                loss = self.criterion(output, y)
                loss.backward()
                self.poptimizer.step()
                
        max_local_steps = self.local_steps

        for param in self.model.base.parameters():
            param.requires_grad = True
        for param in self.model.predictor.parameters():
            param.requires_grad = False

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
            
    def set_parameters(self, base):
        for new_param, old_param in zip(base.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()
