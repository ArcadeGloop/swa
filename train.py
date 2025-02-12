import argparse
import os
import sys
import time
import torch
import torch.nn.functional as F
import torchvision
import models
import utils
import tabulate



parser = argparse.ArgumentParser(description='SGD/SWA training')
parser.add_argument('--dir', type=str, default='training_dir', required=True, help='training directory (default: None)')

parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default=None, required=True, metavar='MODEL',
                    help='model name (default: None)')

parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to resume training from (default: None)')

parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--save_freq', type=int, default=1, metavar='N', help='save frequency (default: 1)')
parser.add_argument('--eval_freq', type=int, default=3, metavar='N', help='evaluation frequency (default: 3)')
parser.add_argument('--lr_init', type=float, default=0.1, metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay (default: 1e-4)')

parser.add_argument('--swa', action='store_true', help='swa usage flag (default: off)')
parser.add_argument('--swa_start', type=float, default=161, metavar='N', help='SWA start epoch number (default: 161)')
parser.add_argument('--swa_lr', type=float, default=0.05, metavar='LR', help='SWA LR (default: 0.05)')
parser.add_argument('--swa_c_epochs', type=int, default=1, metavar='N',
                    help='SWA model collection frequency/cycle length in epochs (default: 1)')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

# our additions
parser.add_argument('--val_size', type=float, default=0.2, help='validation set size (default: 0.2)')
parser.add_argument('--weight_from_data', type=str, default='validation', help='whether to use validation or train weights for our swa (default: validation)')
parser.add_argument('--type_of_weight', type=str, default='accuracy', help='type of weight to use, loss and accuracy (default: accuracy)')
parser.add_argument('--type_of_average', type=str, default='weighted_moving_average', help='type of averaging to use (default: weighted_moving_average)')
parser.add_argument('--scale_weights', type=bool, default=False, help='whether to use MinMax scaling (default: False)')
parser.add_argument('--smoothing_factor', type=float, default=0.1, help='smoothing factor to be used for exponential smoothing (default: 0.1)')
parser.add_argument('--eval_base_model', type=bool, default=False, help='whether to evaluation before swa starts (default: False)')


args = parser.parse_args()


if args.weight_from_data=='validation':
    args.eval_freq=1


print('Preparing directory %s' % args.dir)
os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print('Using model %s' % args.model)
model_cfg = getattr(models, args.model)


print('Loading dataset %s from %s' % (args.dataset, args.data_path))
ds = getattr(torchvision.datasets, args.dataset)
path = os.path.join(args.data_path, args.dataset.lower())
train_transform, test_transform=utils.get_transforms_for(args.dataset) # our addition
train_set = ds(path, train=True, download=True, transform=train_transform)
num_classes = max(train_set.targets) + 1
generator = torch.Generator().manual_seed(args.seed)
train_set, valid_set = torch.utils.data.random_split(train_set, [1-args.val_size, args.val_size]) # our addition
test_set = ds(path, train=False, download=True, transform=test_transform)
loaders = {
    'train': torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    ),
    'validation': torch.utils.data.DataLoader( # our addition
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    ),
    'test': torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
}

print('Preparing model')
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
model.cuda()


if args.swa:
    print('SWA training')
    
    # original model
    swa_model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
    swa_model.cuda()
    
    # our model with new averaging
    our_swa_model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
    our_swa_model.cuda()
    # weight_function=get_weight_function(args.weight_config)

    
    # copy weights of first model to have same initial start
    utils.moving_average(our_swa_model,swa_model) 
    swa_n = 0
    
else:
    print('SGD training')


def schedule(epoch):
    t = (epoch) / (args.swa_start if args.swa else args.epochs)
    lr_ratio = args.swa_lr / args.lr_init if args.swa else 0.01
    if t <= 0.5: # for half of the training cycle, we dont change the default learning rate
        factor = 1.0
    elif t <= 0.9: # then until there's 10% of the iterations left, we linearly decay the LR per cycle 
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else: # in SWA mode: last 10% of training time will use the swa learning rate, in SGD: 1% of the initial learning rate
        factor = lr_ratio
    return args.lr_init * factor



criterion = F.cross_entropy

optimizer = torch.optim.SGD(model.parameters(),
                            lr=args.lr_init,
                            momentum=args.momentum,
                            weight_decay=args.wd)                                

start_epoch = 0
if args.resume is not None:
    print('Resume training from %s' % args.resume)
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    
    if args.swa: 
    
        # original swa
        swa_state_dict = checkpoint['swa_state_dict']
        if swa_state_dict is not None:
            swa_model.load_state_dict(swa_state_dict)
        
        # our swa
        our_swa_state_dict = checkpoint['our_swa_state_dict']
        if our_swa_state_dict is not None:
            our_swa_model.load_state_dict(our_swa_state_dict)
            
        swa_n_ckpt = checkpoint['swa_n']
        if swa_n_ckpt is not None:
            swa_n = swa_n_ckpt

columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'vl_loss', 'vl_acc', 'time']
if args.swa:
    columns = columns[:-1] + ['swa_vl_loss', 'swa_vl_acc','our_swa_vl_loss', 'our_swa_vl_acc'] + columns[-1:]
    swa_res = {'loss': None, 'accuracy': None}
    our_swa_res = {'loss': None, 'accuracy': None}

# 'our_swa_te_loss','our_swa_te_acc'

train_res = {'loss': None, 'accuracy': None}
val_res = {'loss': None, 'accuracy': None}
swa_res = {'loss': None, 'accuracy': None}
our_swa_res = {'loss': None, 'accuracy': None}

utils.save_checkpoint(
    args.dir,
    start_epoch,
    
    # our additions
    train_res=train_res,
    val_res=val_res,
    swa_res=swa_res if args.swa else None,
    our_swa_res=our_swa_res if args.swa else None,
    our_swa_state_dict=our_swa_model.state_dict() if args.swa else None,
    
    state_dict=model.state_dict(),   
    swa_state_dict=swa_model.state_dict() if args.swa else None,
    swa_n=swa_n if args.swa else None,
    optimizer=optimizer.state_dict()
)



# _______________ TRAINING STARTS HERE _______________

weight_sum=0
swa_first_iter=True
# list_of_scores=[]
results={}
minimum_weight=0


for epoch in range(start_epoch, args.epochs):
    time_ep = time.time()

    lr = schedule(epoch)
    utils.adjust_learning_rate(optimizer, lr)
    train_res = utils.train_epoch(loaders['train'], model, criterion, optimizer)
    results.update({'train':train_res})    

    
    if args.eval_base_model and (epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1):
        val_res = utils.eval(loaders['validation'], model, criterion)
    else:
        val_res = {'loss': None, 'accuracy': None}
    
    # when args.swa_c_epochs==1 (the default), the third condition is always true
    # compute average when in swa mode
    if args.swa and (epoch + 1) >= args.swa_start and (epoch + 1 - args.swa_start) % args.swa_c_epochs == 0:
        
        # check loss and accuracy on the validation set when evaluation frequency is reached and for the final epoch
        if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
            val_res = utils.eval(loaders['validation'], model, criterion)
        else:
            val_res = {'loss': None, 'accuracy': None}
        
        results.update({'validation':train_res})    

        # original swa
        utils.moving_average(swa_model, model, 1.0 / (swa_n + 1))
        
        # our swa
        base_weight=results[args.weight_from_data][args.type_of_weight]

        weight=utils.get_weight(base_weight, args.type_of_weight, minimum_weight, scale=args.scale_weights)
      
        
                
        if swa_first_iter:
            # copy the weights on first iteration
            utils.moving_average(our_swa_model, model, 1.0 / (swa_n + 1))
            swa_first_iter=False
            minimum_weight=weight*0.9
            weight_sum+=weight
            
        else: # our swa weighted average, iteration 1+
             
            if (args.type_of_average=='weighted_moving_average'):
                utils.weighted_moving_average(our_swa_model, model, weight,weight_sum)
            if (args.type_of_average=='exponential_smoothing'):
                utils.exponential_smoothing(our_swa_model, model, args.smoothing_factor)


        swa_n += 1
        
        if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
            utils.bn_update(loaders['train'], swa_model)
            swa_res = utils.eval(loaders['validation'], swa_model, criterion)
            
            # our swa
            utils.bn_update(loaders['train'], our_swa_model)
            our_swa_res = utils.eval(loaders['validation'], our_swa_model, criterion)

        else:
            swa_res = {'loss': None, 'accuracy': None}
            our_swa_res = {'loss': None, 'accuracy': None}

    if (epoch + 1) % args.save_freq == 0:
        utils.save_checkpoint(
            args.dir,
            epoch + 1,
            
            # our additions
            train_res=train_res,
            val_res=val_res,
            swa_res=swa_res if args.swa else None,
            our_swa_res=our_swa_res if args.swa else None,
            our_swa_state_dict=our_swa_model.state_dict() if args.swa else None,
            
            state_dict=model.state_dict(),
            swa_state_dict=swa_model.state_dict() if args.swa else None,
            swa_n=swa_n if args.swa else None,
            optimizer=optimizer.state_dict()
        )

    # print progress
    time_ep = time.time() - time_ep
    values = [epoch + 1, lr, train_res['loss'], train_res['accuracy'], val_res['loss'], val_res['accuracy'], time_ep]
    if args.swa:
        values = values[:-1] + [swa_res['loss'], swa_res['accuracy'], our_swa_res['loss'], our_swa_res['accuracy'] ] + values[-1:]
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if epoch % 25 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)


if args.swa:
    # final test set performance
    utils.bn_update(loaders['train'], swa_model)
    swa_test_res = utils.eval(loaders['test'], swa_model, criterion)
    
    
    # our swa
    utils.bn_update(loaders['train'], our_swa_model)
    our_swa_test_res = utils.eval(loaders['test'], our_swa_model, criterion)



# pring test results

print('__________________________')
print(f'the test accuracy of the original SWA is: {swa_test_res["accuracy"]:.4f}')
print(f'the test accuracy of our SWA is: {our_swa_test_res["accuracy"]:.4f}')



if args.epochs % args.save_freq != 0:
    utils.save_checkpoint(
        args.dir,
        args.epochs,
        
        # our additions
        train_res=train_res,
        val_res=val_res,
        swa_res=swa_res if args.swa else None,
        our_swa_res=our_swa_res if args.swa else None,
        our_swa_state_dict=our_swa_model.state_dict() if args.swa else None,
        
        # adding test results
        swa_test_res = swa_test_res,
        our_swa_test_res=our_swa_test_res,
        
        state_dict=model.state_dict(),
        swa_state_dict=swa_model.state_dict() if args.swa else None,
        swa_n=swa_n if args.swa else None,
        optimizer=optimizer.state_dict()
    )




