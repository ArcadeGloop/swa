# %% _____________________________ description

"""
will visualize SWA training vs regular SGD with momentum


"""


# %% _____________________________ imports

import seaborn as sns
import matplotlib.pyplot
import os
import torch
import pandas as pd


# %% _____________________________ functions



# %% _____________________________ 


rootdir=r'C:\Users\alexy\Downloads\for visualization\trainingcheckpoints_weightedaverage_validation_accuracy_notscaled\training_dir'


file_paths=[]

# get file paths
for subdir, dirs, files in os.walk(rootdir): 
    for file in files:
        
        
        file_path=os.path.join(subdir, file)
        file_paths.append(file_path)



# checkpoint = torch.load(file_paths[40])

# checkpoint.keys()


# checkpoint['train_res']

# train_val_loss_diff=train_val_loss_diff,
# model.load_state_dict(checkpoint['state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer'])


experiment_results=[]

for file_path in file_paths:
    
    if '.sh' in file_path:
        continue
    
    experiment=file_path.split("\\")[-3]
    checkpoint = torch.load(file_path)

    
    experiment_results.append({'experiment': experiment, 
                               'epoch':checkpoint['epoch'] ,
                               'train_loss': checkpoint['train_res']['loss'],
                               'train_accuracy': checkpoint['train_res']['accuracy'],
                               'validation_loss': checkpoint['val_res']['loss'],
                               'validation_accuracy':checkpoint['val_res']['accuracy'] })




experiment_results_df=pd.DataFrame(experiment_results)


experiment_results_df['validation_accuracy'].plot()


sns.plot(experiment_results_df, x='epoch',)





epochs=100
lr_init=0.1
lr_final=0.001
decay_function='linear' # or "exponential"

decay_start=0.1
decay_end=0.9

slope=(lr_init-lr_final)/(decay_start-decay_end)  # 

intercept= lr_init -slope*decay_start




def schedule(epoch):
    t = (epoch) / (epochs) 
    if t <= decay_start: # for half of the training cycle, we dont change the default learning rate
        lr = lr_init
    elif t <= decay_end: # then until there's 10% of the iterations left, we linearly decay the LR per cycle 
        lr = t*slope+intercept
    else: # in SWA mode: last 10% of training time will use the swa learning rate, in SGD: 1% of the initial learning rate
        lr = lr_final
    return lr


import numpy as np 
lrs=[]

for epoch in np.arange(100):
    lrs.append(schedule(epoch))

sns.lineplot(lrs)
