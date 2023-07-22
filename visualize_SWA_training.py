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


# rootdir=r'C:\Users\alexy\Downloads\for visualization\weightedaverage_validation_accuracy_notscaled\training_dir'
rootdir=r'C:\Users\alexy\Downloads\for visualization\sgd'


file_paths=[]

# get file paths
for subdir, dirs, files in os.walk(rootdir): 
    for file in files:
        
        
        file_path=os.path.join(subdir, file)
        file_paths.append(file_path)



checkpoint = torch.load(file_paths[40])

checkpoint.keys()


# checkpoint['train_res']

# train_val_loss_diff=train_val_loss_diff,
# model.load_state_dict(checkpoint['state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer'])


experiment_results=[]

 # without SWA results
for file_path in file_paths[2:]:
    
    if  '.sh' in file_path or '-0' in file_path:
        continue
    
    experiment=file_path.split("\\")[-3]
    checkpoint = torch.load(file_path)

    
    experiment_results.append({'experiment': experiment, 
                                'epoch':checkpoint['epoch'] ,
                                'train_loss': float(checkpoint['train_res']['loss']),
                                'train_accuracy': checkpoint['train_res']['accuracy']})
                                




# with swa results
# for file_path in file_paths:
    
#     if  '.sh' in file_path or '-150' in file_path:
#         continue
    
#     experiment=file_path.split("\\")[-3]
#     checkpoint = torch.load(file_path)

    
#     experiment_results.append({'experiment': experiment, 
#                                'epoch':checkpoint['epoch'] ,
#                                'train_loss': float(checkpoint['train_res']['loss']),
#                                'train_accuracy': checkpoint['train_res']['accuracy'],
#                                'validation_loss': float(checkpoint['val_res']['loss']),
#                                'validation_accuracy':checkpoint['val_res']['accuracy'],
#                                'original_swa_loss': float(checkpoint['swa_res']['loss']),
#                                'original_swa_accuracy': checkpoint['swa_res']['accuracy'],
#                                'our_swa_loss':float(checkpoint['our_swa_res']['loss']) ,
#                                'our_swa_accuracy':  checkpoint['our_swa_res']['accuracy']})




experiment_results_df=pd.DataFrame(experiment_results)

experiment_results_df.columns



melted_results_df=pd.melt(experiment_results_df, id_vars=['experiment','epoch'], value_vars=[ 'train_accuracy',
        'train_loss'])



# melted_results_df=pd.melt(experiment_results_df, id_vars=['experiment','epoch'], value_vars=['train_loss', 'train_accuracy',
#        'validation_loss', 'validation_accuracy', 'original_swa_loss',
#        'original_swa_accuracy', 'our_swa_loss', 'our_swa_accuracy'])



sns.lineplot(melted_results_df[melted_results_df['variable'].str.contains('loss')], x='epoch',y='value', hue='variable')

sns.lineplot(melted_results_df, x='epoch',y='value', hue='variable')



