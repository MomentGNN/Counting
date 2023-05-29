import torch
import pickle
from torch_geometric.data import DataLoader, Data
from random import sample
from MomentGNN import add_attributes2

def ZINC_gen(K, batch_size):

    fileObj = open('ZINC910.obj', 'rb')
    graphs_list = pickle.load(fileObj)
    fileObj.close()

    if K == 9:
        nine_zero_list = graphs_list[4]
        nine_count_list = graphs_list[3]
        nine_zero_list_short = sample(nine_zero_list,len(nine_count_list))

        # shuffle everything

        nine_zero_list_short = sample(nine_zero_list_short,len(nine_zero_list_short))
        nine_count_list = sample(nine_count_list,len(nine_count_list))

        # split in training and testing sets

        nine_zero_training = nine_zero_list_short[:len(nine_zero_list_short)*7//10]
        nine_zero_val = nine_zero_list_short[len(nine_zero_list_short)*7//10:len(nine_zero_list_short)*8//10]
        nine_zero_testing = nine_zero_list_short[len(nine_zero_list_short)*8//10:]

        nine_count_training = nine_count_list[:len(nine_count_list)*7//10]
        nine_count_val = nine_count_list[len(nine_count_list)*7//10:len(nine_count_list)*8//10]
        nine_count_testing = nine_count_list[len(nine_count_list)*8//10:]

        nine_train_ind = nine_zero_training + nine_count_training
        nine_val_ind = nine_zero_val + nine_count_val
        nine_test_ind = nine_zero_testing + nine_count_testing

        # shuffle again

        nine_train_ind = sample(nine_train_ind,len(nine_train_ind))
        nine_test_ind = sample(nine_test_ind,len(nine_test_ind))


        ZINC_dataset9_train = []
        ZINC_dataset9_val = []
        ZINC_dataset9_test = []
        list_n = []
        for i, k in enumerate(nine_train_ind):
            ZINC_dataset9_train.append(Data(x = add_attributes2(graphs_list[0][k], K), edge_index = graphs_list[0][k],y = torch.tensor([graphs_list[1][k]])))
            # list_n.append(ZINC_dataset9_train[i].x.shape[0])
        # print(sum(list_n)/len(list_n))
        for i, k in enumerate(nine_test_ind):
            ZINC_dataset9_test.append(Data(x = add_attributes2(graphs_list[0][k], K), edge_index = graphs_list[0][k],y = torch.tensor([graphs_list[1][k]])))
        for i, k in enumerate(nine_val_ind):
            ZINC_dataset9_val.append(Data(x = add_attributes2(graphs_list[0][k], K), edge_index = graphs_list[0][k],y = torch.tensor([graphs_list[1][k]])))





        # instantiate data loaders


        train_loader = DataLoader(ZINC_dataset9_train, batch_size, shuffle=True)
        test_loader = DataLoader(ZINC_dataset9_test, batch_size, shuffle=False)
        gener_val_loader = DataLoader(ZINC_dataset9_val, batch_size, shuffle=False)
        num_input_features = ZINC_dataset9_train[0].x.shape[1]
    elif K == 10:
        deca_zero_list = graphs_list[6]
        deca_count_list = graphs_list[5]
        deca_zero_list_short = sample(deca_zero_list,len(deca_count_list))

        # shuffle everything

        deca_zero_list_short = sample(deca_zero_list_short,len(deca_zero_list_short))
        deca_count_list = sample(deca_count_list,len(deca_count_list))


        deca_zero_training = deca_zero_list_short[:len(deca_zero_list_short)*7//10]
        deca_zero_val = deca_zero_list_short[len(deca_zero_list_short)*7//10:len(deca_zero_list_short)*8//10]
        deca_zero_testing = deca_zero_list_short[len(deca_zero_list_short)*8//10:]

        deca_count_training = deca_count_list[:len(deca_count_list)*7//10]
        deca_count_val = deca_count_list[len(deca_count_list)*7//10:len(deca_count_list)*8//10]
        deca_count_testing = deca_count_list[len(deca_count_list)*8//10:]

        deca_train_ind = deca_zero_training + deca_count_training
        deca_val_ind = deca_zero_val + deca_count_val
        deca_test_ind = deca_zero_testing + deca_count_testing



        # shuffle again


        deca_train_ind = sample(deca_train_ind,len(deca_train_ind))
        deca_test_ind = sample(deca_test_ind,len(deca_test_ind))

        ZINC_dataset10_train = []
        ZINC_dataset10_test = []
        ZINC_dataset10_val = []
        list_n = []

        for i, k in enumerate(deca_train_ind):
            ZINC_dataset10_train.append(Data(x = add_attributes2(graphs_list[0][k], K), edge_index = graphs_list[0][k],y = torch.tensor([graphs_list[2][k]])))
        for i, k in enumerate(deca_test_ind):
            ZINC_dataset10_test.append(Data(x = add_attributes2(graphs_list[0][k], K), edge_index = graphs_list[0][k],y = torch.tensor([graphs_list[2][k]])))
        for i, k in enumerate(deca_val_ind):
            ZINC_dataset10_val.append(Data(x = add_attributes2(graphs_list[0][k], K), edge_index = graphs_list[0][k],y = torch.tensor([graphs_list[2][k]])))

        num_input_features = ZINC_dataset10_train[0].x.shape[1]
        # instantiate data loaders


        train_loader = DataLoader(ZINC_dataset10_train, batch_size, shuffle=True)
        test_loader = DataLoader(ZINC_dataset10_test, batch_size, shuffle=False)
        gener_val_loader = DataLoader(ZINC_dataset10_val, batch_size, shuffle=False)


    return train_loader, test_loader, gener_val_loader, num_input_features