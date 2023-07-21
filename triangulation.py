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

# Notes



# TODO
# if best model doesnt improve for X epochs, reset learning rate cycle
# start swa at first when loss decrease slows down compared to how it started. 
# then average only the best performing models. not just consecutive models. 
# could add online bayesian optimisation for hyperparam tuning, 
# each time for increasing number of epochs.


# _________ Triangulation _________
# 1. use the same LR until a plateau is reached. uing exponential smoothing with alpha, another hyper param
# 2. then start SWA for C epochs. C is a hyper parameter
# 3. after C epochs continue training the SWA model
# 4. reduce learning rate by D, by multiplying LR. another hyper parameter.



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
parser.add_argument('--lr_init', type=float, default=0.1, metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay (default: 1e-4)')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

# our additions
parser.add_argument('--val_size', type=float, default=0.2, help='validation set size (default: 0.2)')
parser.add_argument('--swa_duration', type=int, default=5, help='duration of SWA (default: 5)')
parser.add_argument('--device', type=str, default='cuda', help='device to train on, cuda or cpu (default: cuda)')
parser.add_argument('--sgd_duration', type=int, default=10, help='duration of SGD (default: 10)')
parser.add_argument('--decrease', type=float, default=0.9, help='multiply learning rate by this value after SWA (default: 0.9)')


# parser.add_argument('--trigger', type=float, default=-0.02, help='smoothed average of loss difference to trigger SWA start (-0.02)')
# parser.add_argument('--difference_init', type=float, default=1.0, help='W0 for exponenential smoothing (default: 1.0)')
# parser.add_argument('--alpha', type=float, default=0.3, help='smoothing factor to be used for exponential smoothing (default: 0.3)')




args = parser.parse_args()


device=torch.device(args.device)



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
model.to(device)


# swa model 
swa_model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
swa_model.to(device)


print('SWAT-SGD training')
# print(f'SWA will be triggered when loss difference reaches: {args.trigger}')
print(f'SGD will run for {args.sgd_duration} epochs')
print(f'SWA will run for {args.swa_duration} epochs')
print(f'learning rate will decrease according to original schedule')

# print(f'learning rate will decrease by: {args.decrease}')


# train_val_loss_diff=args.difference_init


def schedule(epoch):
    t = (epoch) / (args.epochs)
    lr_ratio = 0.01
    # if t <= 0.5: # for half of the training cycle, we dont change the default learning rate
    #     factor = 1.0
    if t <= 0.9: # then until there's 10% of the iterations left, we linearly decay the LR per cycle 
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else: # in SWA mode: last 10% of training time will use the swa learning rate, in SGD: 1% of the initial learning rate
        factor = lr_ratio
    return args.lr_init * factor

criterion = F.cross_entropy
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=args.lr_init,
    momentum=args.momentum,
    weight_decay=args.wd
)

start_epoch = 0
if args.resume is not None:
    print('Resume training from %s' % args.resume)
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch']
    # train_val_loss_diff=train_val_loss_diff,
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
 
columns = ['ep', 'lr', 'swa_n', 'swa_mode', 'tr_loss', 'tr_acc', 'vl_loss', 'vl_acc',  'time'] # 'loss_diff',


train_res = {'loss': None, 'accuracy': None}
val_res = {'loss': None, 'accuracy': None}




utils.save_checkpoint(
    args.dir,
    start_epoch,
    
    # our additions
    train_res=train_res,
    val_res=val_res,
    # train_val_loss_diff=train_val_loss_diff,
    
    state_dict=model.state_dict(),   
    optimizer=optimizer.state_dict()
)



# _______________ TRAINING STARTS HERE _______________



lr=args.lr_init

swa_mode=False

swa_n=0


for epoch in range(start_epoch, args.epochs):
    time_ep = time.time()

    train_res = utils.train_epoch(loaders['train'], model, criterion, optimizer)
    
    # check loss and accuracy on the validation set 
    utils.bn_update(loaders['train'], model)
    val_res = utils.eval(loaders['validation'], model, criterion)
    
 
    # train_val_loss_diff=train_val_loss_diff*(1-args.alpha) + args.alpha*(train_res['loss']-val_res['loss'])
    
    
    # if train_val_loss_diff < args.trigger and not swa_mode:
    if (epoch+1)%10==0 and not swa_mode:
        swa_mode=True
    
    if swa_mode:
        
        utils.moving_average(swa_model, model, 1.0 / (swa_n + 1))
        swa_n+=1
        

    if (epoch + 1) % args.save_freq == 0:
        utils.save_checkpoint(
            args.dir,
            epoch + 1,
            
            # our additions
            train_res=train_res,
            val_res=val_res,
            # train_val_loss_diff=train_val_loss_diff,
            lr=optimizer.param_groups[0]['lr'],

            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict()
        )

    # print progress
    time_ep = time.time() - time_ep
    values = [epoch + 1, lr, swa_n, swa_mode, train_res['loss'], train_res['accuracy'], val_res['loss'], val_res['accuracy'], time_ep]  # train_val_loss_diff
    
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if epoch % 25 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)
    
          
    if swa_n >= args.swa_duration:
        
        # stop swa
        swa_mode=False
        swa_n=0
        # train_val_loss_diff=args.difference_init/(epoch+1)

        #copy swa model weights to the model we train
        # continue training the swa model
        utils.moving_average(model, swa_model, 1.0) 
        
        # reduce learning rate
        lr = schedule(epoch)
        # lr = optimizer.param_groups[0]['lr']*args.decrease
        # lr = args.lr_init/(epoch+1)
        utils.adjust_learning_rate(optimizer, lr)

    



# final test set performance
utils.bn_update(loaders['train'], model)
test_res = utils.eval(loaders['test'], model, criterion)


# pring test results

print('__________________________')
# print(f'the test accuracy of the original SWA is: {test_res["accuracy"]:.4f}')
print(f'the test accuracy is: {test_res["accuracy"]:.4f}')



if args.epochs % args.save_freq != 0:
    utils.save_checkpoint(
        args.dir,
        args.epochs,
        
        # our additions
        train_res=train_res,
        val_res=val_res,        
        # train_val_loss_diff=train_val_loss_diff,
        lr=optimizer.param_groups[0]['lr'],

        
        state_dict=model.state_dict(),
        optimizer=optimizer.state_dict()
    )




