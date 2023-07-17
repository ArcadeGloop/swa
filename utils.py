import os
import torch
import torchvision.transforms as transforms

def get_transforms_for(data=None):
    
    if data=='CIFAR10':
          transform_train = transforms.Compose([
              transforms.RandomCrop(32, padding=4),
              transforms.RandomHorizontalFlip(),
              transforms.ToTensor(),
              transforms.Normalize((0.49139968 ,0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),
          ])
          transform_test = transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize((0.49139968 ,0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),
          ])
          
    if data=='CIFAR100':
          transform_train = transforms.Compose([
              transforms.RandomCrop(32, padding=4),
              transforms.RandomHorizontalFlip(),
              transforms.ToTensor(),
              transforms.Normalize((0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047)),
          ])
          transform_test = transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize((0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047)),
          ])
          
    return transform_train, transform_test
          

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(dir, epoch, **kwargs): # add train accuracy here
    state = {
        'epoch': epoch,
        
    }
    state.update(kwargs)
    filepath = os.path.join(dir, 'checkpoint-%d.pt' % epoch)
    torch.save(state, filepath)

# training_results,test_results,swa_results,
# 'training_loss':training_results['loss'] ,  
# 'training_accuracy': training_results['accuracy'],
# 'test_loss': test_results['loss'],
# 'test_accuracy': test_results['accuracy'],
# 'swa_test_loss': test_results['loss'],
# 'swa_test_accuracy': test_results['accuracy'],

def train_epoch(loader, model, criterion, optimizer):
    loss_sum = 0.0
    correct = 0.0

    model.train()

    for i, (input, target) in enumerate(loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        loss = criterion(output, target_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.data * input.size(0)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target_var.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / len(loader.dataset),
        'accuracy': correct / len(loader.dataset) * 100.0,
    }


def eval(loader, model, criterion):
    loss_sum = 0.0
    correct = 0.0

    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
    
            output = model(input_var)
            loss = criterion(output, target_var)
    
            loss_sum += loss.data * input.size(0)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target_var.data.view_as(pred)).sum().item()
    
        return {
            'loss': loss_sum / len(loader.dataset),
            'accuracy': correct / len(loader.dataset) * 100.0,
        }


def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha


def weighted_moving_average(net1, net2, weight, weight_sum):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= weight_sum
        param1.data += param2.data * weight
        param1.data/=(weight_sum+weight)


def exponential_smoothing(net1, net2,smoothing_factor=0.1): 
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data= smoothing_factor*param2.data+(1-smoothing_factor)*param1.data

    

def get_average_function(function_type):
    
    if function_type == 'weighted_moving_average':
        return weighted_moving_average
    
    if function_type =='exponential_smoothing':
        return exponential_smoothing


def get_weight(base_weight,type_of_weight,list_of_scores,minmax_scaled=False):    
    if type_of_weight=='loss':
        base_weight=1/base_weight
    
    if minmax_scaled:
        if list_of_scores is not None:
            maximum=max(list_of_scores)
            minimum=min(list_of_scores)
            return (base_weight-minimum)/(maximum-minimum)            
    
    return base_weight
    




    # if minmax_scaled:
    #     maximum=max(list_of_scores)
    #     minimum=min(list_of_scores)
    #     return (weight-minimum)/(maximum-minimum)
    
    # return weight
    
    
    
    
    # weight=utils.get_weight(weight,args.type_of_weight,list_of_scores, minmax_scaled=args.scale_weights)
