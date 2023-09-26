from flcore.clients.clientapfl import clientAPFL
from flcore.servers.serverbase import Server
import logging
import os


class APFL(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.message_hp = f"{args.algorithm}, lr:{args.local_learning_rate:.5f}, alpha:{args.alpha:.2f}"
        clientObj = clientAPFL
        self.message_hp_dash = self.message_hp.replace(", ", "-")
        self.hist_result_fn = os.path.join(args.hist_dir, f"{self.actual_dataset}-{self.message_hp_dash}-{args.goal}-{self.times}.h5")

        self.set_clients(args, clientObj)

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
            self.aggregate_parameters()

            if i % self.eval_gap == 0:
                print("==> Evaluating personalized models...", flush=True)
                self.send_models(mode="all")
                self.evaluate()
                if i == 80:
                    self.check_early_stopping()

        print(f"\n==> Best mean personalized accuracy: {self.best_mean_test_acc*100:.2f}%", flush=True)

        self.save_results(fn=self.hist_result_fn)
        message_res = f"\ttest_acc:{self.best_mean_test_acc:.6f}"
        logging.info(self.message_hp + message_res)
