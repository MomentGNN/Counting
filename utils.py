
from utils_graph_learning import multi_class_accuracy
import torch.nn as nn
import numpy as np



def process_arguments(args):

    # ###### choose the substructures: usually loaded from networkx,
    # ###### except for 'all_simple_graphs' where they need to be precomputed,
    # ###### or when a custom edge list is provided in the input by the user
    
    # define if degree is going to be used as a feature and when (for each layer or only at initialization)

    # args['degree_as_tag'] = [args['degree_as_tag']] + [False for _ in range(args['num_layers']-1)]
        
    # define if existing features are going to be retained when the degree is used as a feature
    # args['retain_features'] = [args['retain_features']] + [True for _ in range(args['num_layers']-1)]
        
    # replicate d_out dimensions if the rest are not defined (msg function, mlp hidden dimension, encoders, etc.)
    # and repeat hyperparams for every layer
    if args['d_msg'] == -1:
        args['d_msg'] = [None for _ in range(args['num_layers'])]
    elif args['d_msg'] is None:
        args['d_msg'] = [args['d_out'] for _ in range(args['num_layers'])]
    else:
        args['d_msg'] = [args['d_msg'] for _ in range(args['num_layers'])]    
    
    if args['d_h'] is None:
        args['d_h'] = [[args['d_out']] * (args['num_mlp_layers'] - 1) for _ in range(args['num_layers'])]
    else:
        args['d_h'] = [[args['d_h']] * (args['num_mlp_layers'] - 1) for _ in range(args['num_layers'])]
        
    args['d_out_edge_encoder'] = [args['d_out'] for _ in range(args['num_layers'])]

        

    args['d_out_node_encoder'] = args['d_out']

    
    args['d_out_id_embedding'] = args['d_out']


    args['d_out_degree_embedding'] = args['d_out']

    

    # repeat hyperparams for every layer
    args['d_out'] = [args['d_out'] for _ in range(args['num_layers'])]
    
    args['train_eps'] = [False for _ in range(args['num_layers'])]
    
    if len(args['final_projection']) == 1:
        args['final_projection'] = [args['final_projection'][0] for _ in range(args['num_layers'])] + [True]
        
    args['bn'] = [args['bn'] for _ in range(args['num_layers'])]
    args['dropout_features'] = [args['dropout_features'] for _ in range(args['num_layers'])] + [args['dropout_features']]
    
    # loss function & metrics
    if args['loss_fn'] == 'CrossEntropyLoss':
        assert args['regression'] is False, "Can't use Cross-Entropy loss in regression."
        loss_fn = nn.CrossEntropyLoss()
    elif args['loss_fn'] == 'BCEWithLogitsLoss':
        assert args['regression'] is False, "Can't use binary Cross-Entropy loss in regression."
        loss_fn = nn.BCEWithLogitsLoss()
    elif args['loss_fn'] == 'MSELoss':
        loss_fn = nn.MSELoss()
    elif args['loss_fn'] == 'L1Loss':
        loss_fn = nn.L1Loss()
    else:
        raise NotImplementedError
        
    if args['prediction_fn'] == 'multi_class_accuracy':
        assert args['regression'] is False, "Can't use Classification Accuracy metric in regression."
        prediction_fn = multi_class_accuracy
    elif args['prediction_fn'] == 'MSELoss':
        prediction_fn = nn.MSELoss(reduction='sum')
    elif args['prediction_fn'] == 'L1Loss':
        prediction_fn = nn.L1Loss(reduction='sum')
    elif args['prediction_fn'] == 'None':
        prediction_fn = None
    else:
        raise NotImplementedError
        
    if args['regression']:
        perf_opt = np.argmin
    else:
        perf_opt = np.argmax 
        
    return args, loss_fn, prediction_fn, perf_opt


