from flcore.clients.clientper import clientPer
from flcore.servers.serverbase import Server
import copy
import os
import logging

class FedPer(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.message_hp = f"{args.algorithm}, lr:{args.local_learning_rate:.5f}"
        clientObj = clientPer
        self.message_hp_dash = self.message_hp.replace(", ", "-")
        self.hist_result_fn = os.path.join(args.hist_dir, f"{self.actual_dataset}-{self.message_hp_dash}-{args.goal}-{self.times}.h5")

        self.set_clients(args, clientObj)

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
                print("==> Evaluating personalized models...", flush=True)
                self.send_models(mode="all")
                self.evaluate(self.global_model)

        print(f"==> Best mean personalized accuracy: {self.best_mean_test_acc*100:.2f}%", flush=True)

        self.save_results(fn=self.hist_result_fn)
        message_res = f"\ttest_acc:{self.best_mean_test_acc:.6f}"
        logging.info(self.message_hp + message_res)


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

    def prepare_global_model(self):
        temp_model = copy.deepcopy(self.global_model) # base
        self.global_model = copy.deepcopy(self.clients[0].model)
        for p_t, p_g in zip(temp_model.parameters(), self.global_model.base.parameters()):
            p_g.data = p_t.data.clone()
        for p in self.global_model.predictor.parameters():
            p.data.zero_()
        for c in self.clients:
            for p_g, p_c in zip(self.global_model.predictor.parameters(), c.model.predictor.parameters()):
                p_g.data += p_c.data * c.train_samples
        return