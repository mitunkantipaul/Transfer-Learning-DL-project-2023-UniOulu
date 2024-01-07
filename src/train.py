import torch
import numpy as np
from torch import optim
from .models import get_optimizer, get_model

# evaluation function
def eval(model, data_loader):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()
    model.eval()
    correct = 0.0
    num_images = 0.0
    for batch_idx, (images, labels) in enumerate(data_loader):
        if use_cuda:
            images = images.cuda()
            labels = labels.cuda()
        outs = model(images) 
        preds = outs.argmax(dim=1)
        correct += preds.eq(labels).sum()
        num_images += len(labels)

    acc = correct / num_images
    return acc


# training function
def train(model, train_loader, valid_loader, epoches, criterion, optimizer_args, sheduler=False):

    metrics = {
        'train_acc':[],
        'val_acc':[],
        'train_loss':[],
    }

    optimizer = get_optimizer(model, **optimizer_args)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()
    # Set up learning rate scheduler
    # StepLR: Decays the learning rate by gamma every step_size epochs
    if sheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    for epoch in range(epoches):
        model.train() 
        running_loss = 0.0
        correct = 0.0 # used to accumulate number of correctly recognized images
        num_images = 0.0 # used to accumulate number of images
        for batch_idx, (images, labels) in enumerate(train_loader):
            if use_cuda:
                images = images.cuda()
                labels = labels.cuda()
             
            outs = model(images)
            loss = criterion(outs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outs.data, 1)
            correct += (predicted == labels).sum().item()
            num_images += labels.size(0)
            running_loss += loss.item()
        if sheduler:
            # Step the learning rate scheduler at the end of each epoch
            scheduler.step()
            
            
        acc = correct / num_images
        acc_eval = eval(model, valid_loader)
     
        metrics['train_acc'].append(acc.cpu().numpy() if isinstance(acc, torch.Tensor) and acc.is_cuda else acc)
        metrics['val_acc'].append(acc_eval.cpu().numpy() if isinstance(acc_eval, torch.Tensor) and acc_eval.is_cuda else acc_eval)
        metrics['train_loss'].append(running_loss)
        print('epoch: %d, lr: %f, accuracy: %f, loss: %f, valid accuracy: %f' % (epoch, optimizer.param_groups[0]['lr'], acc, loss.item(), acc_eval))

    return model, acc,  acc_eval, metrics