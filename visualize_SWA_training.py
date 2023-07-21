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
