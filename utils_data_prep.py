import os
import numpy as np

from sklearn.model_selection import StratifiedKFold





def separate_data(graph_list, seed, fold_idx):
    
    ### Code obtained from here: https://github.com/weihua916/powerful-gnns
    
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)

    if hasattr(graph_list[0], 'label'):
        labels = [graph.label for graph in graph_list]
    elif hasattr(graph_list[0], 'y'):
        labels = [graph.y for graph in graph_list]
    else:
        raise NotImplementedError
        
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_graph_list, test_graph_list

def separate_data_given_split(graph_list, path, fold_idx):
    
    ### Splits data based on pre-computed splits
    
    assert -1 <= fold_idx and fold_idx < 10, "Parameter fold_idx must be from -1 to 9, with -1 referring to the special model selection split."

    train_filename = os.path.join(path, '10fold_idx', 'train_idx-{}.txt'.format(fold_idx+1))
    test_filename = os.path.join(path, '10fold_idx', 'test_idx-{}.txt'.format(fold_idx+1))
    val_filename = os.path.join(path, '10fold_idx', 'val_idx-{}.txt'.format(fold_idx+1))
    train_idx = np.loadtxt(train_filename, dtype=int)
    test_idx = np.loadtxt(test_filename, dtype=int)
        
    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]
    val_graph_list = None                           
    
    if os.path.exists(val_filename):
        val_idx = np.loadtxt(val_filename, dtype=int)
        val_graph_list = [graph_list[i] for i in val_idx]

    return train_graph_list, test_graph_list, val_graph_list
