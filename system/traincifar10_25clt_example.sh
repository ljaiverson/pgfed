# The following commands provides an example on how to coduct training with different FL/pFL algorithms.
# These commands assumes the dataset (in this case, cifar10 with 25 clients) has already been generated.
# These commands train the model for 2 global rounds (-gr flag).
# In each round 25% of the clients will be selected (-jr flag).
# Each selected client will train the model for 2 epochs or local steps (-ls flag)

# Local
python main.py -data cifar10 -nc 25 -jr 0.25 -gr 2 -ls 2 -algo Local

# FedAvg
python main.py -data cifar10 -nc 25 -jr 0.25 -gr 2 -ls 2 -algo FedAvg

# FedDyn
python main.py -data cifar10 -nc 25 -jr 0.25 -gr 2 -ls 2 -algo FedDyn -al 0.1

# pFedMe
python main.py -data cifar10 -nc 25 -jr 0.25 -gr 2 -ls 2 -algo pFedMe -bt 1.0 -lrp 0.01

# FedFomo
python main.py -data cifar10 -nc 25 -jr 0.25 -gr 2 -ls 2 -algo FedFomo

# APFL
python main.py -data cifar10 -nc 25 -jr 0.25 -gr 2 -ls 2 -algo APFL -al 0.5

# FedRep
python main.py -data cifar10 -nc 25 -jr 0.25 -gr 2 -ls 2 -algo FedRep -pls 1

# LGFedAvg
python main.py -data cifar10 -nc 25 -jr 0.25 -gr 2 -ls 2 -algo LGFedAvg

# FedPer
python main.py -data cifar10 -nc 25 -jr 0.25 -gr 2 -ls 2 -algo FedPer

# Per-FedAvg
python main.py -data cifar10 -nc 25 -jr 0.25 -gr 2 -ls 2 -algo PerAvg -al 0.005 -bt 0.005

# FedRoD
python main.py -data cifar10 -nc 25 -jr 0.25 -gr 2 -ls 2 -algo FedRoD

# FedBABU
python main.py -data cifar10 -nc 25 -jr 0.25 -gr 2 -ls 2 -algo FedBABU -al 0.001 -bt 0.01

# PGFed
python main.py -data cifar10 -nc 25 -jr 0.25 -gr 2 -ls 2 -algo PGFed -mu 0.1 -lam 0.01 -bt 0.0

# PGFedMo
python main.py -data cifar10 -nc 25 -jr 0.25 -gr 2 -ls 2 -algo PGFed -mu 0.1 -lam 0.01 -bt 0.5

