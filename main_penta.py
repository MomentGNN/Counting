## ---- imports -----

import argparse
import utils_parsing as parse
import os

import random
import pickle

import torch
import numpy as np

from torch_geometric.data import DataLoader

from utils import process_arguments
from utils_data_prep import separate_data, separate_data_given_split

from train_test_funcs import train, setup_optimization

from models_graph_classification import GNNSubstructures

from MomentGNN import  add_attributes_penta



## ---- main function -----

def main(args):
    
    
    ## ----------------------------------- argument processing
    
    args, loss_fn, prediction_fn, perf_opt = process_arguments(args)
    # evaluator = Evaluator(args['dataset_name']) if args['dataset'] == 'ogb' else None
    evaluator = None

    
    ## ----------------------------------- infrastructure

    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed(args['seed'])
    torch.cuda.manual_seed_all(args['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args['seed'])
    os.environ['PYTHONHASHSEED'] = str(args['seed'])
    random.seed(args['seed'])
    print('[info] Setting all random seeds {}'.format(args['seed']))
    
    torch.set_num_threads(args['num_threads'])
    if args['GPU']:
        device = torch.device("cuda:"+str(args['device_idx']) if torch.cuda.is_available() else "cpu")
        print('[info] Training will be performed on {}'.format(torch.cuda.get_device_name(args['device_idx'])))
    else:
        device = torch.device("cpu")
        print('[info] Training will be performed on cpu')


        
        
    ## ----------------------------------- datasets: prepare and preprocess (count or load subgraph counts)
    
    path = os.path.join('./datasets', 'ZINC')


    # exampleObj = graphs_ptg

    # fileObj = open('ZINC.obj', 'wb')
    # pickle.dump(exampleObj,fileObj)
    # fileObj.close()  

    fileObj = open('ZINC.obj', 'rb')
    graphs_ptg = pickle.load(fileObj)
    fileObj.close()
    num_classes = 1
    


    pentagons = torch.zeros(len(graphs_ptg))
    hexagons = torch.zeros(len(graphs_ptg))
    

    for i in range(len(graphs_ptg)):
        graph_t = graphs_ptg[i]
        # dense_adj = to_dense_adj(graph_t.edge_index).squeeze()
        x_count = graph_t.identifiers

        pentagons[i] = sum(x_count[:,2])/5
        hexagons[i] = sum(x_count[:,3])/6

        graphs_ptg[i].y = torch.tensor([[pentagons[i]]])
        x_char = add_attributes_penta(graph_t.edge_index)
        # graphs_ptg[i].identifiers = x_char


        graphs_ptg[i].x = x_char
        graphs_ptg[i].edge_features = None

    
    
    ## ----------------------------------- node and edge feature dimensions

    if graphs_ptg[0].x.dim()==1:
        num_features = 1
    else:
        num_features = graphs_ptg[0].num_features
        


    d_in_node_encoder = None
        # d_in_edge_encoder = [num_edge_features]

        
    
    ## ----------------------------------- encode ids and degrees (and possibly edge features)


    
    d_id =[1]*graphs_ptg[0].identifiers.shape[1]

        
    d_degree =  []
    


    print("Training starting now...")
    train_losses_folds = []; train_accs_folds = []
    test_losses_folds = []; test_accs_folds = []
    val_losses_folds = []; val_accs_folds = []


    fold_idxs = [-1] if args['onesplit'] else args['fold_idx']
    for fold_idx in fold_idxs:
        
        print('############# FOLD NUMBER {:01d} #############'.format(fold_idx))
            

        # split data into training/validation/test
        if args['split'] == 'random':  # use a random split
            dataset_train, dataset_test = separate_data(graphs_ptg, args['split_seed'], fold_idx)
            dataset_val = None
        elif args['split'] == 'given':  # use a precomputed split
            dataset_train, dataset_test, dataset_val = separate_data_given_split(graphs_ptg, path, fold_idx)

        # instantiate data loaders
        loader_train = DataLoader(dataset_train,
                                  batch_size=args['batch_size'], 
                                  shuffle=args['shuffle'],
                                  worker_init_fn=random.seed(args['seed']), 
                                  num_workers=args['num_workers'])
        loader_test = DataLoader(dataset_test,
                                 batch_size=args['batch_size'], 
                                 shuffle=False, 
                                 worker_init_fn=random.seed(args['seed']), 
                                 num_workers=args['num_workers'])
        if dataset_val is not None:
            loader_val = DataLoader(dataset_val,
                                    batch_size=args['batch_size'], 
                                    shuffle=False,
                                    worker_init_fn=random.seed(args['seed']), 
                                    num_workers=args['num_workers'])
        else:
            loader_val = None
             
        # instantiate model
        Model = GNNSubstructures
            
        model = Model(
            in_features=num_features, 
            out_features=num_classes, 
            d_in_id=d_id,
            d_in_node_encoder=d_in_node_encoder, 
            d_degree=d_degree,
            **args)
        model = model.to(device)
        print("Instantiated model:\n{}".format(model))
        
        # count model params
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("[info] Total number of parameters is: {}".format(params))


            
        # optimizer and lr scheduler
        optimizer, scheduler = setup_optimization(model, **args)
        
        
        start_epoch = 0
            
        # train (!)
        metrics = train(
            loader_train,
            loader_test, 
            model,
            optimizer,
            loss_fn,
            loader_val=loader_val,
            prediction_fn=prediction_fn,
            evaluator=evaluator,
            scheduler=scheduler,
            min_lr=args['min_lr'],
            fold_idx=fold_idx,
            start_epoch=start_epoch, 
            n_epochs=args['num_epochs'],
            eval_freq=args['eval_frequency'])

        # log results of training
        train_losses, train_accs, test_losses, test_accs, val_losses, val_accs = metrics
        train_losses_folds.append(train_losses)
        train_accs_folds.append(train_accs)
        test_losses_folds.append(test_losses)
        test_accs_folds.append(test_accs)
        val_losses_folds.append(val_losses)
        val_accs_folds.append(val_accs)
        best_idx = perf_opt(val_accs) if loader_val is not None else perf_opt(test_accs)
        print("Training complete!")
        print("\tbest train accuracy {:.4f}\n\tbest test accuracy {:.4f}".format(train_accs[best_idx], test_accs[best_idx]))
            
       

            
    # log metrics 

    train_accs_folds = np.array(train_accs_folds)
    test_accs_folds = np.array(test_accs_folds)
    train_losses_folds = np.array(train_losses_folds)
    test_losses_folds = np.array(test_losses_folds)

    train_accs_mean = np.mean(train_accs_folds, 0)
    train_accs_std = np.std(train_accs_folds, 0)
    test_accs_mean = np.mean(test_accs_folds, 0)
    test_accs_std = np.std(test_accs_folds, 0)
    

    
    if val_losses_folds[0] is not None:
        val_accs_folds = np.array(val_accs_folds)
        val_losses_folds = np.array(val_losses_folds)
        val_accs_mean = np.mean(val_accs_folds, 0)
        val_accs_std = np.std(val_accs_folds, 0)
    
    best_index = perf_opt(test_accs_mean) if val_losses_folds[0] is None else perf_opt(val_accs_mean)
    
    
            
    print("Best train mean: {:.4f} +/- {:.4f}".format(train_accs_mean[best_index], train_accs_std[best_index]))
    print("Best test mean: {:.4f} +/- {:.4f}".format(test_accs_mean[best_index], test_accs_std[best_index]))
    
    scores = dict()
    scores['best_train_mean'] = train_accs_mean[best_index]
    scores['best_train_std'] = train_accs_std[best_index]
    scores['last_train_std'] = train_accs_std[-1]
    scores['last_train_mean'] = train_accs_mean[-1]
    scores['best_test_mean'] = test_accs_mean[best_index]
    scores['best_test_std'] = test_accs_std[best_index]
    scores['last_test_std'] = test_accs_std[-1]
    scores['last_test_mean'] = test_accs_mean[-1]
    if val_losses_folds[0] is not None:
        scores['best_validation_std'] = val_accs_std[best_index]
        scores['best_validation_mean'] = val_accs_mean[best_index]
        scores['last_validation_std'] = val_accs_std[-1]
        scores['last_validation_mean'] = val_accs_mean[-1]
      
    
    return scores



if __name__ == '__main__':   
   
    parser = argparse.ArgumentParser()
    
    # set seeds to ensure reproducibility
    parser.add_argument('--seed', type=int, default=0)
    
    # this specifies the folds for cross-validation
    parser.add_argument('--fold_idx', type=parse.str2list2int, default=[0,1,2,3,4,5,6,7,8,9])
    parser.add_argument('--onesplit', type=parse.str2bool, default=True)

    # set multiprocessing to true in order to do the precomputation in parallel
    parser.add_argument('--multiprocessing', type=parse.str2bool, default=False)
    parser.add_argument('--num_processes', type=int, default=64)
    
    ###### data loader parameters
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_threads', type=int, default=1)

    parser.add_argument('--split', type=str, default='given')

    
    ###### encoding args: different ways to encode discrete data

  

    parser.add_argument('--input_node_encoder', type=str, default='None')
    # parser.add_argument('--edge_encoder', type=str, default='None')
    
    
    # sum or concatenate embeddings when multiple discrete features available
    parser.add_argument('--multi_embedding_aggr', type=str, default='sum')
    
    # only used for the GIN variant: creates a dummy variable for self loops (e.g. edge features or edge counts)
    parser.add_argument('--extend_dims', type=parse.str2bool, default=True)
    
    ###### model to be used and architecture parameters, in particular
    # - d_h: is the dimension for internal mlps, set to None to
    #   make it equal to d_out
    # - final_projection: is for jumping knowledge, specifying
    #   which layer is accounted for in the last model stage, if
    #   the list has only one element, that that value gets applied
    #   to all the layers
    # - jk_mlp: set it to True to use an MLP after each jk layer, otherwise a linear layer will be used
    
    # parser.add_argument('--model_name', type=str, default='GSN_sparse')
    parser.add_argument('--random_features',  type=parse.str2bool, default=False)
    parser.add_argument('--num_mlp_layers', type=int, default=2)
    parser.add_argument('--d_h', type=int, default=None)
    parser.add_argument('--activation_mlp', type=str, default='relu')
    parser.add_argument('--bn_mlp', type=parse.str2bool, default=True)
    
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--d_msg', type=int, default=None)
    parser.add_argument('--d_out', type=int, default=16)
    parser.add_argument('--bn', type=parse.str2bool, default=True)
    parser.add_argument('--dropout_features', type=float, default=0)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--aggr', type=str, default='add')
    parser.add_argument('--flow', type=str, default='source_to_target')
    
    parser.add_argument('--final_projection', type=parse.str2list2bool, default=[True])
    parser.add_argument('--jk_mlp', type=parse.str2bool, default=False)
    parser.add_argument('--residual', type=parse.str2bool, default=False)
    
    parser.add_argument('--readout', type=str, default='sum')
    
    ###### architecture variations:
    # - msg_kind: gin  
    #             general (general formulation with MLPs - eq 3,4 of the main paper)
    # - inject*: passes the relevant variable to deeper layers akin to skip connections.
    #            If set to False, then the variable is used only as input to the first layer
    parser.add_argument('--msg_kind', type=str, default='general')
    parser.add_argument('--inject_ids', type=parse.str2bool, default=False)
    
    ###### optimisation parameters
    parser.add_argument('--shuffle', type=parse.str2bool, default=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--eval_frequency', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--regularization', type=float, default=0)
    parser.add_argument('--scheduler', type=str, default='ReduceLROnPlateau')
    parser.add_argument('--scheduler_mode', type=str, default='min')
    parser.add_argument('--min_lr', type=float, default=0.0)
    parser.add_argument('--decay_steps', type=int, default=50)
    parser.add_argument('--decay_rate', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=20)
    
        
    ###### training parameters: task, loss, metric
    parser.add_argument('--regression', type=parse.str2bool, default=True)
    parser.add_argument('--loss_fn', type=str, default='L1Loss')
    parser.add_argument('--prediction_fn', type=str, default='L1Loss')
    
    
    ######  general (mode, gpu, logging)
    
    parser.add_argument('--GPU', type=parse.str2bool, default=False)
    parser.add_argument('--device_idx', type=int, default=0)
    

    args = parser.parse_args()
    print(args)
    main(vars(args))
