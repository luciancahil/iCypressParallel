import logging
import os

import torch
from torch_geometric import seed_everything
from deepsnap.dataset import GraphDataset
from graphgym.cmd_args import parse_args
from graphgym.config import cfg, dump_cfg, load_cfg, set_run_dir, set_out_dir
from graphgym.loader import create_dataset, create_loader
from graphgym.logger import create_logger, setup_printing
from graphgym.model_builder import create_model
from graphgym.optimizer import create_optimizer, create_scheduler
from graphgym.register import train_dict
from graphgym.train import train
from graphgym.utils.agg_runs import agg_runs
from graphgym.utils.comp_budget import params_count
from graphgym.utils.device import auto_select_device
from graphgym.models.gnn import GNNStackStage
from CytokinesDataSet import CytokinesDataSet
from graphgym.models.layer import GeneralMultiLayer, Linear, GeneralConv
from graphgym.models.gnn import GNNStackStage
import numpy as np

from Visualization import Visualize

# Load cmd line args
args = parse_args()
# Load config file
load_cfg(cfg, args)
set_out_dir(cfg.out_dir, args.cfg_file)
# Set Pytorch environment
torch.set_num_threads(cfg.num_threads)
dump_cfg(cfg)
# Repeat for different random seeds
set_run_dir(cfg.out_dir)
setup_printing()
# Set configurations for each run
cfg.seed = cfg.seed + 1
seed_everything(cfg.seed)
auto_select_device()
# Set machine learning pipeline
datasets = create_dataset()
loaders = create_loader(datasets)
loggers = create_logger()
model = torch.load(args.model_path)


# Print model info
logging.info(model)
logging.info(cfg)
cfg.params = params_count(model)
logging.info('Num parameters: %s', cfg.params)
# run model on data
for loader in loaders:
    for batch in loader:
        batch.edge_weights = model.edge_weights.repeat(len(batch.G))
        output = model(batch)
        print("Output")
        print(output)

"""
    for data in loaders:
        print
        inputs, labels = data  # Adjust this line if your data structure is different

        output = model(inputs)

        print(output)"""