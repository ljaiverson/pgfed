import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision

from flcore.trainmodel.models import *

warnings.simplefilter("ignore")
torch.manual_seed(0)

def run(args):
    model_str = args.model
    for i in range(args.prev, args.times):
        print(f"\n============= Running time: [{i+1}th/{args.times}] =============", flush=True)
        print("Creating server and clients ...")

        # Generate args.model
        if model_str == "cnn":
            if args.dataset == "mnist" or args.dataset.startswith("organamnist"):
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
            elif args.dataset.upper() == "CIFAR10" or args.dataset.upper() == "CIFAR100" or args.dataset.startswith("Office-home"):
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            else:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)

        else:
            raise NotImplementedError

        # select algorithm
        if args.algorithm.startswith("Local"):
            from flcore.servers.serverlocal import Local
            server = Local(args, i)

        elif args.algorithm.startswith("FedAvg"):
            from flcore.servers.serveravg import FedAvg
            server = FedAvg(args, i)

        elif args.algorithm.startswith("FedDyn"):
            from flcore.servers.serverdyn import FedDyn
            server = FedDyn(args, i)

        elif args.algorithm.startswith("pFedMe"):
            from flcore.servers.serverpfedme import pFedMe
            server = pFedMe(args, i)

        elif args.algorithm.startswith("FedFomo"):
            from flcore.servers.serverfomo import FedFomo
            server = FedFomo(args, i)

        elif args.algorithm.startswith("APFL"):
            from flcore.servers.serverapfl import APFL
            server = APFL(args, i)

        elif args.algorithm.startswith("FedRep"):
            from flcore.servers.serverrep import FedRep
            args.predictor = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = LocalModel(args.model, args.predictor)
            server = FedRep(args, i)

        elif args.algorithm.startswith("LGFedAvg"):
            from flcore.servers.serverlgfedavg import LGFedAvg
            args.predictor = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = LocalModel(args.model, args.predictor)
            server = LGFedAvg(args, i)

        elif args.algorithm.startswith("FedPer"):
            from flcore.servers.serverper import FedPer
            args.predictor = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = LocalModel(args.model, args.predictor)
            server = FedPer(args, i)

        elif args.algorithm.startswith("PerAvg"):
            from flcore.servers.serverperavg import PerAvg
            server = PerAvg(args, i)

        elif args.algorithm.startswith("FedRoD"):
            from flcore.servers.serverrod import FedRoD
            args.predictor = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = LocalModel(args.model, args.predictor)
            server = FedRoD(args, i)

        elif args.algorithm.startswith("FedBABU"):
            args.predictor = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = LocalModel(args.model, args.predictor)
            from flcore.servers.serverbabu import FedBABU
            server = FedBABU(args, i)

        elif args.algorithm.startswith("PGFed"):
            from flcore.servers.serverpgfed import PGFed
            server = PGFed(args, i)
            
        else:
            raise NotImplementedError



        server.train()
        if args.dataset.startswith("Office-home") and args.times != 1:
            import logging
            m = server.domain_mean_test_accs
            logging.info(f"domains means and average:\t{m[0]:.6f}\t{m[1]:.6f}\t{m[2]:.6f}\t{m[3]:.6f}\t{server.best_mean_test_acc:.6f}")



        # # comment the above block and uncomment the following block for fine-tuning on new clients
        # if len(server.clients) == 100:
        #     old_clients_num = 80
        #     server.new_clients = server.clients[old_clients_num:]
        #     server.clients = server.clients[:old_clients_num]
        #     server.num_clients = old_clients_num
        #     server.join_clients = int(old_clients_num * server.join_ratio)
        # if not args.algorithm.startswith("Local"):
        #     server.train()
        #     server.prepare_global_model()
        # n_epochs = 20
        # print(f"\n\n==> Training for new clients for {n_epochs} epochs")
        # server.train_new_clients(epochs=n_epochs)

def get_args():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, default="cnn", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="cifar10",
                        choices=["cifar10", "cifar100", "organaminist25", "organaminist50", "organaminist100", "Office-home20"])
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="cnn")
    parser.add_argument('-p', "--predictor", type=str, default="cnn")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-gr', "--global_rounds", type=int, default=3)
    parser.add_argument('-ls', "--local_steps", type=int, default=5)
    parser.add_argument('-algo', "--algorithm", type=str, default="PGFed",
                        choices=["Local", "FedAvg", "FedDyn", "pFedMe", "FedFomo", "APFL", "FedRep",
                                 "LGFedAvg", "FedPer", "PerAvg", "FedRoD", "FedBABU", "PGFed"])
    parser.add_argument('-jr', "--join_ratio", type=float, default=0.25,
                        help="Ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=25,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")

    # FL algorithms (multiple algs)
    parser.add_argument('-bt', "--beta", type=float, default=0.0,
                        help="PGFed momentum, average moving parameter for pFedMe, Second learning rate of Per-FedAvg")
    parser.add_argument('-lam', "--lambdaa", type=float, default=1.0,
                        help="PGFed learning rate for a_i, Regularization weight for pFedMe")
    parser.add_argument('-mu', "--mu", type=float, default=0,
                        help="PGFed weight for aux risk, pFedMe weight")
    parser.add_argument('-K', "--K", type=int, default=5,
                        help="Number of personalized training steps for pFedMe")
    parser.add_argument('-lrp', "--p_learning_rate", type=float, default=0.01,
                        help="pFedMe personalized learning rate to caculate theta aproximately using K steps")
    # FedFomo
    parser.add_argument('-M', "--M", type=int, default=8,
                        help="Server only sends M client models to one client at each round")
    # APFL
    parser.add_argument('-al', "--alpha", type=float, default=0.5)
    # FedRep
    parser.add_argument('-pls', "--plocal_steps", type=int, default=5)
    # FedBABU
    parser.add_argument('-fts', "--fine_tuning_steps", type=int, default=1)
    # save directories
    parser.add_argument("--hist_dir", type=str, default="../results/", help="dir path for output hist file")
    parser.add_argument("--log_dir", type=str, default="../logs/", help="dir path for log (main results) file")
    parser.add_argument("--ckpt_dir", type=str, default="../checkpoints/", help="dir path for checkpoints")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    total_start = time.time()
    args = get_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"
    print("=" * 50)

    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local steps: {}".format(args.local_steps))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Global rounds: {}".format(args.global_rounds))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Local model: {}".format(args.model))
    print("Using device: {}".format(args.device))

    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("=" * 50)

    run(args)
