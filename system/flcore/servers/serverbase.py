import torch
import os
import numpy as np
import h5py
import copy
import time
import sys
import random
import logging

from utils.data_utils import read_client_data


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.device = args.device
        self.dataset = args.dataset
        self.global_rounds = args.global_rounds
        self.local_steps = args.local_steps
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.join_clients = int(self.num_clients * self.join_ratio)
        self.algorithm = args.algorithm
        self.goal = args.goal
        self.top_cnt = 100
        self.best_mean_test_acc = -1.0
        self.clients = []
        self.selected_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_test_loss = []
        self.rs_train_loss = []
        self.clients_test_accs = []
        self.domain_mean_test_accs = []

        self.times = times
        self.eval_gap = args.eval_gap

        self.set_seed(self.times)
        self.set_path(args)

        # preprocess dataset name
        if self.dataset.startswith("cifar"):
            dir_alpha = 0.3
        elif self.dataset == "organamnist25":
            dir_alpha = 1.0
        elif self.dataset.startswith("organamnist"):
            dir_alpha = 0.3
        elif self.dataset.startswith("organamnist"):
            if self.num_clients == 20:
                dir_alpha = 0.3
            else:
                dir_alpha = 1.0
        else:
            dir_alpha = float("nan")

        self.actual_dataset = f"{self.dataset}-{self.num_clients}clients_alpha{dir_alpha:.1f}"
        logger_fn = os.path.join(args.log_dir, f"{args.algorithm}-{self.actual_dataset}.log")
        self.set_logger(save=True, fn=logger_fn)

    def set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def set_logger(self, save=False, fn=None):
        if save:
            fn = "testlog.log" if fn == None else fn
            logging.basicConfig(
                filename=fn,
                filemode="a",
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                level=logging.DEBUG
            )
        else:
            logging.basicConfig(
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                level=logging.DEBUG
            )

    def set_path(self, args):
        self.hist_dir = args.hist_dir
        self.log_dir = args.log_dir
        self.ckpt_dir = args.ckpt_dir
        if not os.path.exists(args.hist_dir):
            os.makedirs(args.hist_dir)
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)

    def set_clients(self, args, clientObj):
        self.new_clients = None
        for i in range(self.num_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data))
            self.clients.append(client)

    def select_clients(self):
        selected_clients = list(np.random.choice(self.clients, self.join_clients, replace=False))
        return selected_clients

    def send_models(self, mode="selected"):
        if mode == "selected":
            assert (len(self.selected_clients) > 0)
            for client in self.selected_clients:
                client.set_parameters(self.global_model)
        elif mode == "all":
            for client in self.clients:
                client.set_parameters(self.global_model)
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
            self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def prepare_global_model(self):
        pass

    def reset_records(self):
        self.best_mean_test_acc = 0.0
        self.clients_test_accs = []
        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_test_loss = []

    def train_new_clients(self, epochs=20):
        self.global_model = self.global_model.to(self.device)
        self.clients = self.new_clients
        self.reset_records()
        for c in self.clients:
            c.model = copy.deepcopy(self.global_model)
        self.evaluate()
        for epoch_idx in range(epochs):
            for c in self.clients:
                c.standard_train()
            print(f"==> New clients epoch: [{epoch_idx+1:2d}/{epochs}] | Evaluating local models...", flush=True)
            self.evaluate()
        print(f"==> Best mean global accuracy: {self.best_mean_test_acc*100:.2f}%", flush=True)
        self.save_results(fn=self.hist_result_fn)
        message_res = f"\tnew_clients_test_acc:{self.best_mean_test_acc:.6f}"
        logging.info(self.message_hp + message_res)

    def save_global_model(self, model_path=None, state=None):
        if model_path is None:
            model_path = os.path.join("models", self.dataset)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        if state is None:
            torch.save({"global_model": self.global_model.cpu().state_dict()}, model_path)
        else:
            torch.save(state, model_path)

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = os.path.join("models", self.dataset)
            model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)
        
    def save_results(self, fn=None):
        if fn is None:
            algo = self.dataset + "_" + self.algorithm
            result_path = self.hist_dir

        if (len(self.rs_test_acc)):
            if fn is None:
                algo = algo + "_" + self.goal + "_" + str(self.times+1)
                file_path = os.path.join(result_path, "{}.h5".format(algo))
            else:
                file_path = fn
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_test_loss', data=self.rs_test_loss)
                hf.create_dataset('clients_test_accs', data=self.clients_test_accs)
                # hf.create_dataset('rs_train_loss', data=self.rs_train_loss)


    def test_metrics(self, temp_model=None):
        """ A personalized evaluation scheme (test_acc's do not average based on num_samples) """
        test_accs, test_aucs, test_losses, test_nums = [], [], [], []
        for c in self.clients:
            test_acc, test_auc, test_loss, test_num = c.test_metrics(temp_model)  # test_acc, test_num, test_auc
            test_accs.append(test_acc)
            test_aucs.append(test_auc)
            test_losses.append(test_loss)
            test_nums.append(test_num)
        ids = [c.id for c in self.clients]
        return ids, test_accs, test_aucs, test_losses, test_nums

    # evaluate selected clients
    def evaluate(self, temp_model=None, mode="personalized"):
        ids, test_accs, test_aucs, test_losses, test_nums = self.test_metrics(temp_model)
        self.clients_test_accs.append(copy.deepcopy(test_accs))
        if mode == "personalized":
            mean_test_acc, mean_test_auc, mean_test_loss = np.mean(test_accs), np.mean(test_aucs), np.mean(test_losses)
        elif mode == "global":
            mean_test_acc, mean_test_auc, mean_test_loss = np.average(test_accs, weights=test_nums), np.average(test_aucs, weights=test_nums), np.average(test_losses, weights=test_nums)
        else:
            raise NotImplementedError
        # compute domain means for
        if self.dataset.startswith("Office-home") and (mean_test_acc > self.best_mean_test_acc):
            self.best_mean_test_acc = mean_test_acc
            self.domain_mean_test_accs = np.mean(np.array(test_accs).reshape(4, -1), axis=1)
        self.best_mean_test_acc = max(mean_test_acc, self.best_mean_test_acc)
        self.rs_test_acc.append(mean_test_acc)
        self.rs_test_auc.append(mean_test_auc)
        self.rs_test_loss.append(mean_test_loss)
        print(f"==> test_loss: {mean_test_loss:.5f} | mean_test_accs: {mean_test_acc*100:.2f}% | best_acc: {self.best_mean_test_acc*100:.2f}%\n")

    def check_early_stopping(self, thresh=0.0):
        # Early stopping
        if thresh == 0.0:
            if (self.dataset == "cifar100"):
                thresh = 0.2
            elif (self.dataset == "cifar10"):
                thresh = 0.6
            elif (self.dataset.startswith("organamnist")):
                thresh = 0.8
            else:
                thresh = 0.23
        return (self.rs_test_acc[-1] < thresh) and (self.rs_test_acc[-2] < thresh) and (self.rs_test_acc[-3] < thresh) and (self.rs_test_acc[-4] < thresh)

