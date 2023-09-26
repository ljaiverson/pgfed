import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn import metrics
import torch
import copy
import torch.nn as nn
from flcore.clients.clientbase import Client

class clientPerAvg(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.alpha = args.alpha
        self.beta = args.beta
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer1 = torch.optim.SGD(self.model.parameters(), lr=self.alpha)
        self.optimizer2 = torch.optim.SGD(self.model.parameters(), lr=self.beta)

    def train(self):
        trainloader = self.load_train_data(self.batch_size*2)
        self.model.train()

        max_local_steps = self.local_steps

        for step in range(max_local_steps):  # local update
            for X, Y in trainloader:
                temp_model = copy.deepcopy(list(self.model.parameters()))

                # step 1
                if type(X) == type([]):
                    x = [None, None]
                    x[0] = X[0][:self.batch_size].to(self.device)
                    x[1] = X[1][:self.batch_size]
                else:
                    x = X[:self.batch_size].to(self.device)
                y = Y[:self.batch_size].to(self.device)
                self.optimizer1.zero_grad()
                output = self.model(x)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer1.step()

                # step 2
                if type(X) == type([]):
                    x = [None, None]
                    x[0] = X[0][self.batch_size:].to(self.device)
                    x[1] = X[1][self.batch_size:]
                else:
                    x = X[self.batch_size:].to(self.device)
                y = Y[self.batch_size:].to(self.device)
                self.optimizer2.zero_grad()
                output = self.model(x)
                loss = self.criterion(output, y)
                loss.backward()

                # restore the model parameters to the one before first update
                for old_param, new_param in zip(self.model.parameters(), temp_model):
                    old_param.data = new_param.data.clone()

                self.optimizer2.step()

        # self.model.cpu()

    def train_one_step(self):
        trainloader = self.load_train_data(self.batch_size)
        iter_trainloader = iter(trainloader)
        self.model.train()
        (x, y) = next(iter_trainloader)
        if type(x) == type([]):
            x[0] = x[0].to(self.device)
        else:
            x = x.to(self.device)
        y = y.to(self.device)
        self.optimizer2.zero_grad()
        output = self.model(x)
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer2.step()

    # comment for testing on new clients
    def test_metrics(self, temp_model=None):
        temp_model = copy.deepcopy(self.model)
        self.train_one_step()
        return_val = super().test_metrics(temp_model)
        self.clone_model(temp_model, self.model)
        return return_val
