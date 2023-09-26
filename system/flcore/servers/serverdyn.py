import copy
import torch
from flcore.clients.clientdyn import clientDyn
from flcore.servers.serverbase import Server
import os
import logging

class FedDyn(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.message_hp = f"{args.algorithm}, lr:{args.local_learning_rate:.5f}, alpha:{args.alpha:.5f}"
        clientObj = clientDyn
        self.message_hp_dash = self.message_hp.replace(", ", "-")
        self.hist_result_fn = os.path.join(args.hist_dir, f"{self.actual_dataset}-{self.message_hp_dash}-{args.goal}-{self.times}.h5")

        self.set_clients(args, clientObj)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

        self.alpha = args.alpha
        
        self.server_state = copy.deepcopy(args.model)
        for param in self.server_state.parameters():
            param.data.zero_()

    def train(self):
        for i in range(self.global_rounds):
            self.selected_clients = self.select_clients()
            self.send_models()
            print(f"\n------------- Round number: [{i+1:3d}/{self.global_rounds}]-------------")
            print(f"==> Training for {len(self.selected_clients)} clients...", flush=True)
            for client in self.selected_clients:
                client.train()

            self.receive_models()
            self.update_server_state()
            self.aggregate_parameters()

            if i%self.eval_gap == 0:
                print("==> Evaluating global models...", flush=True)
                self.send_models(mode="all")
                # self.evaluate(mode="global")
                self.evaluate()
                if i == 80:
                    self.check_early_stopping()

        print(f"==> Best mean global accuracy: {self.best_mean_test_acc*100:.2f}%", flush=True)

        self.save_results(fn=self.hist_result_fn)
        message_res = f"\ttest_acc:{self.best_mean_test_acc:.6f}"
        logging.info(self.message_hp + message_res)
        # self.save_global_model()

    def add_parameters(self, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() / self.join_clients

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for client_model in self.uploaded_models:
            self.add_parameters(client_model)

        for server_param, state_param in zip(self.global_model.parameters(), self.server_state.parameters()):
            server_param.data -= (1/self.alpha) * state_param.data

    def update_server_state(self):
        assert (len(self.uploaded_models) > 0)

        model_delta = copy.deepcopy(self.uploaded_models[0])
        for param in model_delta.parameters():
            param.data.zero_()

        for client_model in self.uploaded_models:
            for server_param, client_param, delta_param in zip(self.global_model.parameters(), client_model.parameters(), model_delta.parameters()):
                delta_param.data += (client_param - server_param) / self.num_clients

        for state_param, delta_param in zip(self.server_state.parameters(), model_delta.parameters()):
            state_param.data -= self.alpha * delta_param
