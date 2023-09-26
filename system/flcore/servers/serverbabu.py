from flcore.clients.clientbabu import clientBABU
from flcore.servers.serverbase import Server
import torch
import os
import sys
import logging


class FedBABU(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.message_hp = f"{args.algorithm}, lr:{args.local_learning_rate:.5f}, alpha:{args.alpha:.5f}"
        self.message_hp_dash = self.message_hp.replace(", ", "-")
        self.hist_result_fn = os.path.join(args.hist_dir, f"{self.actual_dataset}-{self.message_hp_dash}-{args.goal}-{self.times}.h5")

        self.set_clients(args, clientBABU)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()

    def train(self):
        for i in range(self.global_rounds):
            self.selected_clients = self.select_clients()
            self.send_models()
            print(f"\n------------- Round number: [{i+1:3d}/{self.global_rounds}]-------------")
            print(f"==> Training for {len(self.selected_clients)} clients...", flush=True)
            for client in self.selected_clients:
                client.train()
            
            self.receive_models()
            self.aggregate_parameters()
            if i%self.eval_gap == 0:
                print("==> Evaluating global models...", flush=True)
                self.send_models(mode="all")
                self.evaluate(mode="global")
                if i > 40:
                    self.check_early_stopping()
            
        print("\n--------------------- Fine-tuning ----------------------")
        self.send_fine_tune_models(mode="all")
        for client in self.clients:
            client.fine_tune()
        print("------------- Evaluating fine-tuned models -------------")
        self.evaluate(mode="personalized")
        print(f"==> Mean personalized accuracy: {self.rs_test_acc[-1]*100:.2f}", flush=True)
        message_res = f"\ttest_acc:{self.rs_test_acc[-1]:.6f}"
        logging.info(self.message_hp + message_res)

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            # self.uploaded_models are a list of client.model.base's
            self.add_parameters(w, client_model)
        # after self.aggregate_parameters(), the self.global_model are still a model with base and predictor

    def send_fine_tune_models(self, mode="selected"):
        if mode == "selected":
            assert (len(self.selected_clients) > 0)
            for client in self.selected_clients:
                client.set_fine_tune_parameters(self.global_model)
        elif mode == "all":
            for client in self.clients:
                client.set_fine_tune_parameters(self.global_model)
        else:
            raise NotImplementedError

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_weights = []
        tot_samples = 0
        self.uploaded_ids = []
        self.uploaded_models = []
        for client in self.selected_clients:
            self.uploaded_weights.append(client.train_samples)
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_models.append(client.model.base)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = os.path.join("models", self.dataset)
            model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        return torch.load(model_path)
