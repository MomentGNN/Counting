#!/usr/bin/env python
# coding: utf-8

import os
import torch
import torch.nn.functional as F
import argparse
import numpy as np
import time
import yaml
from models.gin import GIN
from easydict import EasyDict as edict
from datasets_generation.ZINCgen import ZINC_gen


# Change the following to point to the the folder where the datasets are stored
if os.path.isdir('/datasets2/'):
    rootdir = '/datasets2/CYCLE_DETECTION/'
else:
    rootdir = './data/datasets_kcycle_nsamples=10000/'
yaml_file = './config_cycles.yaml'
# yaml_file = './benchmark/kernel/config4cycles.yaml'
torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--k', type=int, default=10,
                    help="Length of the cycles to detect")
parser.add_argument('--n', type=int, default=24, help='Average number of nodes in the graphs')
parser.add_argument('--save-model', action='store_true',
                    help='Save the model once training is done')
parser.add_argument('--gpu', type=int, default=0, help='Id of gpu device. By default use cpu')
parser.add_argument('--lr', type=float, default=0.01, help="Initial learning rate")
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--clip', type=float, default=10, help="Gradient clipping")
parser.add_argument('--generalization', action='store_true', default=True,
                    help='Evaluate out of distribution accuracy')
args = parser.parse_args()

# Log parameters
test_every_epoch = 5
print_every_epoch = 1
log_interval = 20




# Handle the device
use_cuda = args.gpu is not None and torch.cuda.is_available()
if use_cuda:
    device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(args.gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] =     str(args.gpu)
else:
    device = "cpu"
args.device = device
args.kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
print('Device used:', device)



# Load the config file of the model
with open(yaml_file) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    config['map_x_to_u'] = False        # Not used here
    config = edict(config)
    print(config)


model_name = 'GIN'
args.name = 'MomentGNN_' + str(args.k)

# Create a folder for the saved models
if not os.path.isdir('./saved_models/' + args.name) and args.generalization:
    os.mkdir('./saved_models/' + args.name)



# if args.n is None:
#     args.n = n_gener[args.k]['train']

if config.num_layers == -1:
    config.num_layers = args.k



def train(epoch):
    """ Train for one epoch. """
    model.train()
    lr_scheduler(args.lr, epoch, optimizer)
    loss_all = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
    return loss_all / len(train_loader.dataset)


def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


def lr_scheduler(lr, epoch, optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * (0.995 ** (epoch / 5))






# Load the data
batch_size = args.batch_size


train_loader, test_loader, gener_val_loader, config.num_input_features = ZINC_gen(args.k, batch_size)




# Load model

transform=None
transform_val = None
transform_test = None
# config.num_input_features = 1



config.use_batch_norm = True
model = GIN(config.num_input_features, config.num_classes, config.num_layers,
            config.hidden, config.hidden_final, config.dropout_prob, config.use_batch_norm)


model = model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.5, weight_decay=args.weight_decay)


print("Starting to train")
start = time.time()
best_epoch = -1
best_generalization_acc = 0
best_gen_train_acc = 0
for epoch in range(args.epochs):
    epoch_start = time.time()
    tr_loss = train(epoch)
    if epoch % print_every_epoch == 0:
        acc_train = test(train_loader)
        current_lr = optimizer.param_groups[0]["lr"]
        duration = time.time() - epoch_start
        print(f'Time:{duration:2.2f} | {epoch:5d} | Loss: {tr_loss:2.5f} | Train Acc: {acc_train:2.5f} | LR: {current_lr:.6f}')
        if epoch % test_every_epoch == 0:
            acc_test = test(test_loader)
            print(f'Test accuracy: {acc_test:2.5f}')
            acc_generalization = test(gener_val_loader)
            print("Validation generalization accuracy", acc_generalization)
            if acc_generalization >= best_generalization_acc:
                if acc_generalization == best_generalization_acc:
                    if acc_train >= best_gen_train_acc:
                        best_generalization_acc = acc_generalization
                        best_gen_train_acc = acc_train
                else:
                    best_generalization_acc = acc_generalization
                    best_gen_train_acc = acc_train
                print(f"New best generalization error + accuracy > 90% at epoch {epoch}")
                # Remove existing models
                folder = f'./saved_models/{args.name}/'
                files_in_folder = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
                for file in files_in_folder:
                    try:
                        os.remove(folder + file)
                    except:
                        print("Could not remove file", file)
                # Save new model
                torch.save(model, f'./saved_models/{args.name}/epoch{epoch}.pkl')
                print(f"Model saved at epoch {epoch}.")
                best_epoch = epoch


cur_lr = optimizer.param_groups[0]["lr"]
print(f'{epoch:2.5f} | Loss: {tr_loss:2.5f} | Train Acc: {acc_train:2.5f} | LR: {cur_lr:.6f} | Test Acc: {acc_test:2.5f}')
print(f'Elapsed time: {(time.time() - start) / 60:.1f} minutes')
print('done!')

final_acc = test(test_loader)
print(f"Final accuracy: {final_acc}")
print("Done.")

new_n = 24
gener_test_loader = test_loader
model = torch.load(f"./saved_models/{args.name}/epoch{best_epoch}.pkl", map_location=device)
model.eval()
acc_test_generalization = test(gener_test_loader)
print(f"Generalization accuracy on {args.k} cycles with {new_n} nodes", acc_test_generalization)
