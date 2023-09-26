import torch
import copy
import random
import os
import logging
import numpy as np
from flcore.clients.clientfomo import clientFomo
from flcore.servers.serverbase import Server


class FedFomo(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.message_hp = f"{args.algorithm}, lr:{args.local_learning_rate:.5f}"
        clientObj = clientFomo
        self.message_hp_dash = self.message_hp.replace(", ", "-")
        self.hist_result_fn = os.path.join(args.hist_dir, f"{self.actual_dataset}-{self.message_hp_dash}-{args.goal}-{self.times}.h5")

        self.set_clients(args, clientObj)

        self.P = torch.diag(torch.ones(self.num_clients, device=self.device))
        self.uploaded_models = [self.global_model]
        self.uploaded_ids = []
        self.M = min(args.M, self.join_clients)
            
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

    def train(self):
        for i in range(self.global_rounds):
            self.selected_clients = self.select_clients()
            self.send_models()
            print(f"\n------------- Round number: [{i+1:3d}/{self.global_rounds}]-------------")
            print(f"==> Training for {len(self.selected_clients)} clients...", flush=True)

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            # self.aggregate_parameters()

            if i%self.eval_gap == 0:
                print("==> Evaluating personalized model")
                self.evaluate()
                if i == 80:
                    self.check_early_stopping()
        print(f"==> Best mean personalized accuracy: {self.best_mean_test_acc*100:.2f}%", flush=True)


        self.save_results(fn=self.hist_result_fn)
        message_res = f"\ttest_acc:{self.best_mean_test_acc:.6f}"
        logging.info(self.message_hp + message_res)
        # self.save_global_model()


    def send_models(self):
        assert (len(self.selected_clients) > 0)
        for client in self.selected_clients:

            if len(self.uploaded_ids) > 0:
                M_ = min(self.M, len(self.uploaded_models)) # if clients dropped
                indices = torch.topk(self.P[client.id][self.uploaded_ids], M_).indices.tolist()

                uploaded_ids = []
                uploaded_models = []
                for i in indices:
                    uploaded_ids.append(self.uploaded_ids[i])
                    uploaded_models.append(self.uploaded_models[i])

                client.receive_models(uploaded_ids, uploaded_models)

    def prepare_global_model(self):
        self.global_model = copy.deepcopy(self.clients[0].model)
        for p in self.global_model.parameters():
            p.data.zero_()
        for c in self.clients:
            self.add_parameters(c.train_samples, c.model)
        return
    
    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(self.selected_clients, self.join_clients)

        self.uploaded_ids = []
        self.uploaded_weights = []
        tot_samples = 0
        self.uploaded_models = []
        for client in active_clients:
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            tot_samples += client.train_samples
            self.uploaded_models.append(copy.deepcopy(client.model))
            self.P[client.id] += client.weight_vector
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
            