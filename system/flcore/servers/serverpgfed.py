import copy
from flcore.clients.clientpgfed import clientPGFed
from flcore.servers.serverbase import Server
import numpy as np
import torch
import h5py
import os
import logging


class PGFed(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.message_hp = f"{args.algorithm}, lr:{args.local_learning_rate:.5f}, mu:{args.mu:.5f}, lambda:{args.lambdaa:.5f}"
        if self.algorithm == "PGFedMo":
            self.momentum = args.beta
            self.message_hp += f", beta:{args.beta:.5f}" # momentum
        else:
            self.momentum = 0.0
        clientObj = clientPGFed
        self.message_hp_dash = self.message_hp.replace(", ", "-")
        self.hist_result_fn = os.path.join(args.hist_dir, f"{self.actual_dataset}-{self.message_hp_dash}-{args.goal}-{self.times}.h5")

        self.set_clients(args, clientObj)

        self.mu = args.mu
        self.alpha_mat = (torch.ones((self.num_clients, self.num_clients)) / self.join_clients).to(self.device)
        self.uploaded_grads = {}
        self.loss_minuses = {}
        self.mean_grad = None
        self.convex_comb_grad = None
        self.best_global_mean_test_acc = 0.0
        self.rs_global_test_acc = []
        self.rs_global_test_auc = []
        self.rs_global_test_loss = []
        self.last_ckpt_fn = os.path.join(self.ckpt_dir, f"{self.actual_dataset}-{self.message_hp_dash}-{args.goal}-{self.times}.pt")

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

    def train(self):
        early_stop = False
        for i in range(self.global_rounds):
            self.selected_clients = self.select_clients()
            print(f"\n------------- Round number: [{i+1:3d}/{self.global_rounds}]-------------")
            print(f"==> Training for {len(self.selected_clients)} clients...", flush=True)
            self.send_models()
            for client in self.selected_clients:
                client.train()

            self.receive_models()
            self.aggregate_parameters()

            if i%self.eval_gap == 0:
                print("==> Evaluating personalized models...", flush=True)
                self.evaluate()
                if i >= 40 and self.check_early_stopping():
                    early_stop = True
                    print("==> Performance is too low. Excecuting early stop.")
                    break

        print(f"==> Best mean personalized accuracy: {self.best_mean_test_acc*100:.2f}%", flush=True)
        if not early_stop:
            self.save_results(fn=self.hist_result_fn)
            # message_res = f"\tglobal_test_acc:{self.best_global_mean_test_acc:.6f}\ttest_acc:{self.best_mean_test_acc:.6f}"
            message_res = f"\ttest_acc:{self.best_mean_test_acc:.6f}"
            logging.info(self.message_hp + message_res)
            # state = {
            #         "model": self.global_model.cpu().state_dict(),
            #         # "best_global_acc": self.best_global_mean_test_acc,
            #         "best_personalized_acc": self.best_mean_test_acc,
            #         "alpha_mat": self.alpha_mat.cpu()
            #     }
            # state.update({f"client{c.id}": c.model.cpu().state_dict() for c in self.clients})
            # self.save_global_model(model_path=self.last_ckpt_fn, state=state)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)
        self.uploaded_ids = []
        self.uploaded_grads = {}
        self.loss_minuses = {}
        self.uploaded_models = []
        self.uploaded_weights = []
        tot_samples = 0
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            self.alpha_mat[client.id] = client.a_i
            self.uploaded_grads[client.id] = client.latest_grad
            self.loss_minuses[client.id] = client.loss_minus * self.mu

            self.uploaded_weights.append(client.train_samples)
            tot_samples += client.train_samples
            self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_grads) > 0)
        self.model_weighted_sum(self.global_model, self.uploaded_models, self.uploaded_weights)
        w = self.mu/self.join_clients
        weights = [w for _ in range(self.join_clients)]
        self.mean_grad = copy.deepcopy(list(self.uploaded_grads.values())[0])
        self.model_weighted_sum(self.mean_grad, list(self.uploaded_grads.values()), weights)

    def model_weighted_sum(self, model, models, weights):
        for p_m in model.parameters():
            p_m.data.zero_()
        for w, m_i in zip(weights, models):
            for p_m, p_i in zip(model.parameters(), m_i.parameters()):
                p_m.data += p_i.data.clone() * w

    def send_models(self, mode="selected"):
        assert (len(self.selected_clients) > 0)
        for client in self.selected_clients:
            client.a_i = self.alpha_mat[client.id]
            client.set_parameters(self.global_model)
        if len(self.uploaded_grads) == 0:
            return
        self.convex_comb_grad = copy.deepcopy(list(self.uploaded_grads.values())[0])
        for client in self.selected_clients:
            client.set_prev_mean_grad(self.mean_grad)
            mu_a_i = self.alpha_mat[client.id] * self.mu
            grads, weights = [], []
            for clt_idx, grad in self.uploaded_grads.items():
                weights.append(mu_a_i[clt_idx])
                grads.append(grad)
            self.model_weighted_sum(self.convex_comb_grad, grads, weights)
            client.set_prev_convex_comb_grad(self.convex_comb_grad, momentum=self.momentum)
            client.prev_loss_minuses = copy.deepcopy(self.loss_minuses)
