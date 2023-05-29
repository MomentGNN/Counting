import torch


from utils_misc import isnotebook
if isnotebook():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

# import wandb


def setup_optimization(model, **args):
    
    #-------------- instantiate optimizer and scheduler
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['regularization'])
    if args['scheduler'] == 'ReduceLROnPlateau':
        # print("Instantiating ReduceLROnPlateau scheduler.")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                           factor=args['decay_rate'], 
                                                           patience=args['patience'],
                                                           verbose=True)
    elif args['scheduler'] == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args['decay_steps'], gamma=args['decay_rate'])
    elif args['scheduler'] == 'None':
        scheduler = None
    else:
        raise NotImplementedError('Scheduler {} is not currently supported.'.format(args['scheduler']))
    return optimizer, scheduler


def train(
       loader_train,
       loader_test,
       model,
       optim,
       loss_fn,
       start_epoch,
       n_epochs,
       eval_freq,
       loader_val=None,
       n_iters=None,
       prediction_fn=None,
       evaluator=None,
       scheduler=None,
       min_lr=0.0,
       patience=None,
       fold_idx=None,
       n_iters_test = None):
    
    train_losses = []; train_accs = []
    test_losses = []; test_accs = []
    val_losses = []; val_accs = []
    fold_idx = '' if fold_idx is None else fold_idx
    
    device = next(model.parameters()).device
    for epoch in range(start_epoch, n_epochs):
        
        model.train()
        
        data_iterator = iter(loader_train)
        n_iters = len(loader_train) if n_iters is None else n_iters
        
        for iteration in tqdm(range(n_iters)):
            try:
                data = next(data_iterator)
            except StopIteration:
                
                data_iterator = iter(loader_train)
                data = next(data_iterator)
            
            optim.zero_grad()
            data = data.to(device)
            y_hat = model(data)
            ## ignore nan targets (unlabeled) when computing training loss (relevant for OGB).
            if evaluator is not None:
                if evaluator.name == 'ogbg-ppa':
                    loss = loss_fn(y_hat, data.y.view(-1,))
                else:
                    data.y = data.y.view(-1,1)
                    y_hat = y_hat.view(-1,1)
                    is_labeled = (data.y == data.y)
                    loss = loss_fn(y_hat[is_labeled], data.y[is_labeled])
            else:
                loss = loss_fn(y_hat, data.y)                
            loss.backward()
             
            optim.step()
            

        if scheduler and 'ReduceLROnPlateau' not in str(scheduler.__class__):
            scheduler.step()
    
        if epoch % eval_freq == 0:
            
            log = 'Epoch: {0:03d}, Train: {1:.4f}, Test: {2:.4f}, lr: {3:.8f}'
            with torch.no_grad():

                #-------------- evaluate differently for OGB (evaluation is done by a given evaluator)
                train_loss, train_acc =\
                            test(loader_train, model, loss_fn, device, prediction_fn)
                test_loss, test_acc =\
                            test(loader_test, model, loss_fn, device, prediction_fn)
                    
                    
                train_losses.append(train_loss); train_accs.append(train_acc); 
                test_losses.append(test_loss); test_accs.append(test_acc)
                
                if loader_val is not None:
                    log = 'Epoch: {0:03d}, Train: {1:.4f}, Test: {2:.4f}, Val: {3:.4f}, Val Loss: {4:4f}, lr: {5:.8f}'
                    val_loss, val_acc =\
                            test(loader_val, model, loss_fn, device, prediction_fn)
                    val_losses.append(val_loss); val_accs.append(val_acc)
                    log_args = [epoch, train_acc, test_acc, val_acc, val_loss]
                else:
                    log_args = [epoch, train_acc, test_acc]
                
                if scheduler and 'ReduceLROnPlateau' in str(scheduler.__class__):
                    ref_metric = val_loss if loader_val is not None else test_loss
                    scheduler.step(ref_metric)
                    
                log_args += [optim.param_groups[0]['lr']]
                print(log.format(*log_args))
                
                
        
        current_lr = optim.param_groups[0]['lr']
        if current_lr < min_lr:
            break
    
    val_losses = None if len(val_losses) == 0 else val_losses
    val_accs = None if len(val_accs) == 0 else val_accs
    return train_losses, train_accs, test_losses, test_accs, val_losses, val_accs
                
    
def test(loader, model, loss_fn, device, prediction_fn=None, n_iters=None):

    model.eval()
    losses = []; accs = []
    
    with torch.no_grad():
        
        data_iterator = iter(loader)
        n_iters = len(loader) if n_iters is None else n_iters
        for iteration in range(n_iters):
            try:
                data = next(data_iterator)
            except StopIteration:
                data_iterator = iter(loader)
                data = next(data_iterator)
        
            data = data.to(device)
            y_hat = model(data)
            
            loss = loss_fn(y_hat, data.y)
            
            losses.append(loss.item() * data.num_graphs)

            if prediction_fn:
                acc = prediction_fn(y_hat, data.y).item()
                accs.append(acc)

    avg_losses =  sum(losses)/len(loader.dataset)
    avg_accs = sum(accs)/len(loader.dataset)
    return avg_losses, avg_accs

