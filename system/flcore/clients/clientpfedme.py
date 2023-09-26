import numpy as np
import copy
import torch
import torch.nn as nn
from flcore.optimizers.fedoptimizer import pFedMeOptimizer
from flcore.clients.clientbase import Client


class clientpFedMe(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.lambdaa = args.lambdaa
        self.K = args.K
        self.personalized_learning_rate = args.p_learning_rate

        # these parameters are for personalized federated learing.
        self.local_params = copy.deepcopy(list(self.model.parameters()))
        self.personalized_params = copy.deepcopy(list(self.model.parameters()))
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        trainloader = self.load_train_data()
        self.model.train()

        max_local_steps = self.local_steps

        self.optimizer = pFedMeOptimizer(self.model.parameters(),
                                         local_model=self.local_params,
                                         lr=self.personalized_learning_rate,
                                         lambdaa=self.lambdaa)
        for step in range(max_local_steps):  # local update
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                # K is number of personalized steps
                for i in range(self.K):
                    self.optimizer.zero_grad()
                    output = self.model(x)
                    loss = self.criterion(output, y)
                    loss.backward()
                    # finding aproximate theta
                    self.personalized_params = self.optimizer.step()

                # update local weight after finding aproximate theta
                for new_param, localweight in zip(self.personalized_params, self.local_params):
                    localweight = localweight.to(self.device)
                    localweight.data = localweight.data - self.lambdaa * self.learning_rate * (localweight.data - new_param.data)

        # self.model.cpu()

        self.update_parameters(self.model, self.local_params)


    # comment for testing on new clients
    def get_eval_model(self, temp_model=None):
        self.update_parameters(self.model, self.personalized_params)
        return self.model

    def set_parameters(self, model):
        for new_param, old_param, local_param in zip(model.parameters(), self.model.parameters(), self.local_params):
            old_param.data = new_param.data.clone()
            local_param.data = new_param.data.clone()
