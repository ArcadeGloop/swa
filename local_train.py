import argparse
import os
import sys
import time
import torch
import torch.nn.functional as F
import torchvision

# import models

# from models import mobilenetv3

from torchvision import models

import utils
import tabulate


dir(models)

mobile_net=models.mobilenet_v3_small(weights=None)

eff_net=models.efficientnet_v2_s(weights=None)



# parser = argparse.ArgumentParser(description='SGD/SWA training')
# parser.add_argument('--dir', type=str, default='training_dir', required=True, help='training directory (default: None)')

# parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
# parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
#                     help='path to datasets location (default: None)')
# parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
# parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
# parser.add_argument('--model', type=str, default=None, required=True, metavar='MODEL',
#                     help='model name (default: None)')

# parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
#                     help='checkpoint to resume training from (default: None)')

# parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')
# parser.add_argument('--save_freq', type=int, default=25, metavar='N', help='save frequency (default: 25)')
# parser.add_argument('--eval_freq', type=int, default=5, metavar='N', help='evaluation frequency (default: 5)')
# parser.add_argument('--lr_init', type=float, default=0.1, metavar='LR', help='initial learning rate (default: 0.01)')
# parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
# parser.add_argument('--wd', type=float, default=1e-4, help='weight decay (default: 1e-4)')

# parser.add_argument('--swa', action='store_true', help='swa usage flag (default: off)')
# parser.add_argument('--swa_start', type=float, default=161, metavar='N', help='SWA start epoch number (default: 161)')
# parser.add_argument('--swa_lr', type=float, default=0.05, metavar='LR', help='SWA LR (default: 0.05)')
# parser.add_argument('--swa_c_epochs', type=int, default=1, metavar='N',
#                     help='SWA model collection frequency/cycle length in epochs (default: 1)')

# parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

# args = parser.parse_args()

train_dir='training_dir'
data='CIFAR10' # or 'CIFAR100'
data_path='data'
model_name='mobilenet_v3_small' # 'efficientnet_v2_s
batch_size=128
swa=True




print('Preparing directory %s' % train_dir)
os.makedirs(train_dir, exist_ok=True)
with open(os.path.join(train_dir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

torch.backends.cudnn.benchmark = True
torch.manual_seed(1)
torch.cuda.manual_seed(1)


transform_train, transform_test=utils.get_transforms_for(data)

print('Loading dataset %s from %s' % (data,data_path))
ds = getattr(torchvision.datasets, data)
path = os.path.join(data_path, data.lower())
train_set = ds(path, train=True, download=True, transform=transform_train)
test_set = ds(path, train=False, download=True, transform=transform_test)
loaders = {
    'train': torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    ),
    'test': torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
}
num_classes = max(train_set.targets) + 1


print('Using model %s' % model_name)
model_cfg = getattr(models, model_name)

dir(model_cfg)

print('Preparing model')
model = model_cfg( num_classes=num_classes)
# model.cuda()


if swa:
    print('SWA training')
    swa_model = model_cfg(num_classes=num_classes)
    # swa_model.cuda()
    swa_n = 0
else:
    print('SGD training')


swa_start=5
epochs=20
lr_init=0.1
swa_lr=0.01
momentum=0.9
weight_decay=5e-4

def schedule(epoch):
    t = (epoch) / (swa_start if swa else epochs)
    lr_ratio = swa_lr / lr_init if swa else 0.01
    if t <= 0.5: # for half of the training cycle, we dont change the default learning rate
        factor = 1.0
    elif t <= 0.9: # then until there's 10% of the iterations left, we linearly decay the LR per cycle 
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else: # in SWA mode: last 10% of training time will use the swa learning rate, in SGD: 1% of the initial learning rate
        factor = lr_ratio
    return lr_init * factor


criterion = F.cross_entropy
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=lr_init,
    momentum=momentum,
    weight_decay=weight_decay
)

resume=None

start_epoch = 0

# if resume is not None:
#     print('Resume training from %s' % args.resume)
#     checkpoint = torch.load(args.resume)
#     start_epoch = checkpoint['epoch']
#     model.load_state_dict(checkpoint['state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer'])
#     if args.swa:
#         swa_state_dict = checkpoint['swa_state_dict']
#         if swa_state_dict is not None:
#             swa_model.load_state_dict(swa_state_dict)
#         swa_n_ckpt = checkpoint['swa_n']
#         if swa_n_ckpt is not None:
#             swa_n = swa_n_ckpt

columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc', 'time']
if swa:
    columns = columns[:-1] + ['swa_te_loss', 'swa_te_acc'] + columns[-1:]
    swa_res = {'loss': None, 'accuracy': None}

utils.save_checkpoint(
    train_dir,
    start_epoch,
    state_dict=model.state_dict(),
    swa_state_dict=swa_model.state_dict() if swa else None,
    swa_n=swa_n if swa else None,
    optimizer=optimizer.state_dict()
)

eval_freq=3
swa_c_epochs=3
save_freq=8

for epoch in range(start_epoch, epochs):
    time_ep = time.time()

    lr = schedule(epoch)
    utils.adjust_learning_rate(optimizer, lr)
    train_res = utils.train_epoch(loaders['train'], model, criterion, optimizer)

    # check loss and accuracy on the test set when evaluation frequency is reached and for the final epoch
    if epoch == 0 or epoch % eval_freq == eval_freq - 1 or epoch == epochs - 1:
        test_res = utils.eval(loaders['test'], model, criterion)
    else:
        test_res = {'loss': None, 'accuracy': None}

    # when args.swa_c_epochs==1 (the default), the third condition is always true
    # compute moving average when in swa mode
    if swa and (epoch + 1) >= swa_start and (epoch + 1 - swa_start) % swa_c_epochs == 0:
        utils.moving_average(swa_model, model, 1.0 / (swa_n + 1))
        swa_n += 1
        if epoch == 0 or epoch % eval_freq == eval_freq - 1 or epoch == epochs - 1:
            utils.bn_update(loaders['train'], swa_model)
            swa_res = utils.eval(loaders['test'], swa_model, criterion)
        else:
            swa_res = {'loss': None, 'accuracy': None}

    if (epoch + 1) % save_freq == 0:
        utils.save_checkpoint(
            train_dir,
            epoch + 1,
            state_dict=model.state_dict(),
            swa_state_dict=swa_model.state_dict() if swa else None,
            swa_n=swa_n if swa else None,
            optimizer=optimizer.state_dict()
        )

    # print progress
    time_ep = time.time() - time_ep
    values = [epoch + 1, lr, train_res['loss'], train_res['accuracy'], test_res['loss'], test_res['accuracy'], time_ep]
    if swa:
        values = values[:-1] + [swa_res['loss'], swa_res['accuracy']] + values[-1:]
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if epoch % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)

if epochs % save_freq != 0:
    utils.save_checkpoint(
        train_dir,
        epochs,
        state_dict=model.state_dict(),
        swa_state_dict=swa_model.state_dict() if swa else None,
        swa_n=swa_n if swa else None,
        optimizer=optimizer.state_dict()
    )
