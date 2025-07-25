import logging
import os
from deepsnap.batch import Batch

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
import time
from torch.utils.data import DataLoader

from Visualization import Visualize


#TODO: Fix old genes_to_tissues
#TODO; Add edge option to hyperparameters
if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()
    # Load config file
    load_cfg(cfg, args)
    set_out_dir(cfg.out_dir, args.cfg_file)
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    print("dumping")
    dump_cfg(cfg)
    # Repeat for different random seeds
    datasets = None
    for i in range(5):
        set_run_dir(cfg.out_dir)
        setup_printing()
        # Set configurations for each run
        cfg.seed = cfg.seed + 1
        seed_everything(cfg.seed)
        auto_select_device()
        # Set machine learning pipeline
        # wonderful! I'm going to have to do this myself.
        if datasets != None and cfg.dataset.task == 'node':
            # manually change the labels
            num_nodes = len(datasets[1][0].node_feature)
            node_labels = [0]*num_nodes

            for dataset in datasets:
                for j in range(len(dataset[0].node_label)):
                    node_labels[dataset[0].node_label_index[j]] = dataset[0].node_label[j]

            start = int(num_nodes - num_nodes * (i+1) / 5)
            end = int(num_nodes - num_nodes * (i) / 5)
            node_labels = torch.tensor(node_labels)
            # get indicies
            test_indicies = [k for k in range(start, end)]
            train_indicies = [k for k in range(num_nodes) if k not in test_indicies]

            # get labels
            test_labels = [node_labels[k] for k in test_indicies]
            train_labels = [node_labels[k] for k in train_indicies]

            # convert to tensors
            train_labels = torch.tensor(train_labels)
            train_indicies = torch.tensor(train_indicies)
            test_labels = torch.tensor(test_labels)
            test_indicies = torch.tensor(test_indicies)

            # write to dataset
            datasets[0][0].node_label = train_labels
            datasets[0][0].node_label_index = train_indicies

            datasets[1][0].node_label = test_labels
            datasets[1][0].node_label_index = test_indicies

            print(datasets[0][0].node_label)
            print(datasets[0][0].node_label_index)
            print(datasets[1][0].node_label)
            print(datasets[1][0].node_label_index)
        else:
            datasets = create_dataset()

        #
        if(cfg.dataset.task != 'node'):
            if(len(datasets[0]) % cfg.train.batch_size == 1):
                graph_list = list(datasets[0])
                if graph_list:
                    datasets[0] = GraphDataset(
                                    graph_list[:-1],
                                    task=cfg.dataset.task,
                                    edge_train_mode=cfg.dataset.edge_train_mode,
                                    edge_message_ratio=cfg.dataset.edge_message_ratio,
                                    edge_negative_sampling_ratio=cfg.dataset.edge_negative_sampling_ratio,
                                    resample_disjoint=cfg.dataset.resample_disjoint,
                                    minimum_node_per_graph=0)

                else:
                    print("The dataset would be empty; no new dataset created.")


        loaders = create_loader(datasets)
        loggers = create_logger()
        model = create_model()

        total = 0
        

        # Add edge_weights attribute to the datasets so that they can be accessed in batches
        num_edges = len(datasets[0][0].edge_index[0])
        edge_weights = torch.nn.Parameter(torch.zeros(num_edges))
        name = cfg.dataset.name.split(",")[1]
        visual_path = os.path.join("Visuals", name + "_edges.pt")
        torch.save(datasets[0][0].edge_index, visual_path)
        for loader in loaders:
            for dataset in loader.dataset:
                dataset.edge_weights = edge_weights


        #add edge weights to the set of parameters
        newParam = list()
        for param in model.parameters():
            newParam.append(param)
        
        # Uncomment to add edge weights
        # newParam.append(edge_weights)

        optimizer = create_optimizer(newParam)
        scheduler = create_scheduler(optimizer)
        # Print model info
        logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info('Num parameters: %s', cfg.params)
        # Start training

        print("Device: " + str(cfg.device))
        print("Cuda available: " + str(torch.cuda.is_available()))
        if cfg.train.mode == 'standard':
            train(loggers, loaders, model, optimizer, scheduler)
        else:
            train_dict[cfg.train.mode](loggers, loaders, model, optimizer,
                                        scheduler)
        
        loaders = [None]*2
        loaders[0] = DataLoader(datasets[0],
                        collate_fn=Batch.collate(),
                        batch_size=1,
                        shuffle=True,
                        num_workers=cfg.num_workers,
                        pin_memory=False)
        
        loaders[1] = DataLoader(datasets[1],
                        collate_fn=Batch.collate(),
                        batch_size=1,
                        shuffle=True,
                        num_workers=cfg.num_workers,
                        pin_memory=False)

    
    print(args.get_edge_weights)
    print(type(args.get_edge_weights))
    if(args.get_edge_weights == '1'):
        print("Hello!")

        for batch in loaders[0]:
            model(batch)
        
        node_name_location = os.path.join("datasets", name, "processed", "nodeNames.pt")
        node_to_num = torch.load(node_name_location)
        num_to_node = dict()
        weight_path = os.path.join("EdgeWeights", "{}_edge_weights.csv".format(name))
        file = open(weight_path, mode='w')
        
        for key in node_to_num.keys():
            num_to_node[node_to_num[key]] = key


        edge_list =  model._modules['mp']._modules["block0"]._modules['f']._modules['0']._modules['layer'].edge_weights[0]



        # write top row. First elemetn is source, next is dest
        file.write("Patient Name")
        for k in range(len(edge_list[0])):
            source = num_to_node[edge_list[0][k].item()]
            dest = num_to_node[edge_list[1][k].item()]

            file.write(",{}-{}".format(source, dest))


        file.write("\n")
        number = 0

        for loader in loaders:
            for batch in loader:
                if(number % 100 == 0):
                    print(str(number))
                
                number += 1
                model(batch)
                # get the message passing layers
                message_passing_layers = model._modules['mp']
                # get the last message_passing layer
                num_mp_layers = len([c for c in message_passing_layers.children()])
                for k in range(num_mp_layers):
                    block_name = "block{}".format(k)
                    gan = message_passing_layers._modules[block_name]._modules['f']._modules['0']._modules['layer']
                    weight_list = gan.edge_weights[1].squeeze().tolist()
                    weight_list = str(weight_list)
                    file.write("{}:{},{}\n".format(batch.name[0], block_name, weight_list.replace("[","").replace("]","").replace(" ","")))

                file.write("\n")
        print("Edge Weights Saved to {}".format(weight_path))

# Aggregate results from different seeds
    agg_runs(cfg.out_dir, cfg.metric_best)
    # When being launched in batch mode, mark a yaml as done
    if args.mark_done:
        os.rename(args.cfg_file, f'{args.cfg_file}_done')
    
    name = cfg.dataset.name.split(",")[1]
    visual_path = os.path.join("Visuals", name + "_visuals.pt")

    if(cfg.dataset.task == 'node'):
        # visualization for node classification
        torch.save(edge_weights, visual_path)
    else: #visualization for graph classification
        last_layers_pooled = []
        truths = []
        for loader in loaders:
            for batch in loader:
                last_layer_pooled, truth = model.get_last_hidden_layer_pooled(batch) # first one gives me the vector output of the neural network. 
                last_layers_pooled += last_layer_pooled
                truths.append(truth)
        last_layer_tensor = torch.stack(last_layers_pooled)
        truths_tensor = torch.cat(truths)
        numpy_matrix = last_layer_tensor.numpy()
        numpy_truth = truths_tensor.numpy()
        # save data
        visualization_data = {}
        visualization_data['latent'] = numpy_matrix
        visualization_data['truth'] = numpy_truth
        visual_path = os.path.join("Visuals", name + "_visuals.pt")
        gene_weight_path = os.path.join("Visuals", name + "_genes.pt")

        pre_message_layers = []

        for child in model.children(): # We are at the network level.
            if(isinstance(child, GeneralMultiLayer)): 
                for grandchild in child.children(): # we are at the MultiLayer object
                    for object in grandchild.children(): # we are at the GeneralLayer object
                        if(isinstance(object, Linear)):
                            for layer in object.children(): # we are at the Linear object
                                pre_message_layers.append(layer)
                                colorWeights = layer.weight
                        
        
        first_layer = pre_message_layers[0]
        first_weights = first_layer.weight.detach().numpy()
        gene_weight_sum = [0] * len(first_weights[0])

        for neuron in first_weights:
            for index, value in enumerate(neuron):
                gene_weight_sum[index] += value
        
        gene_weight_sum = [value / len(first_weights) for value in gene_weight_sum]
        geneList = torch.load((os.path.join("datasets", name, "raw", "geneList.pt")))


        gene_weight_dict = dict()

        for i, weight in enumerate(gene_weight_sum):
            gene_weight_dict[geneList[i][0] + "@" + geneList[i][1]] = weight
        
        
        sorted_gene_list = sorted(gene_weight_dict.items(), key=lambda x:-abs(x[1]))
        visualization_data['genes']  = sorted_gene_list

        
        graph_visuals = {'colours':colorWeights, 'graph': datasets[0].graphs[0].G, 'name': name, 'edge_weights': edge_weights}
        visualization_data['graph_data'] = graph_visuals
        torch.save(visualization_data, visual_path)
        print("Visualization data stored at: " + visual_path)
        Visualize.save_PCA(numpy_matrix, numpy_truth, name)
        Visualize.save_TSNE(numpy_matrix, numpy_truth, name)

        print(args.save)
        print(args.save == '1')
        if(args.save == '1'):
            now = time.time() # unix time stamp, to save anohter if need be
            model_path = os.path.join("models", name + "_" + str(now) +"_model.pt")
            print("Model saved at " + model_path)
            model.edge_weights = edge_weights
            torch.save(model, model_path)

        print("Experiment name: " + name)


    

"""
    correlations = []
    for loader in loaders:
        for batch in loader:
            correlation = model.get_correlations(batch) # first one gives me the vector output of the neural network. 
            correlations += correlation
    for child in model.children(): # We are at the network level.
        if(isinstance(child, GeneralMultiLayer)): 
            for grandchild in child.children(): # we are at the MultiLayer object
                for object in grandchild.children(): # we are at the GeneralLayer object
                    if(isinstance(object, Linear)):
                        for layer in object.children(): # we are at the Linear object
                            colorWeights = layer.weight
    Visualize.visualize_graph(colorWeights, datasets[0].graphs[0].G, name, edge_weights)


    for child in model.children(): # We are at the network level.
        if(isinstance(child, GeneralMultiLayer)): 
            for grandchild in child.children(): # we are at the MultiLayer object
                for object in grandchild.children(): # we are at the GeneralLayer object
                    if(isinstance(object, Linear)):
                        for layer in object.children(): # we are at the Linear object
                            colorWeights = layer.weight
    Visualize.visualize_graph(colorWeights, datasets[0].graphs[0].G, name, edge_weights)

"""
