# %% _____________________________ description

"""
will visualize SWA training vs regular SGD with momentum


"""


# %% _____________________________ imports

import seaborn as sns
import matplotlib.pyplot as plt
import os
import torch
import pandas as pd
import numpy as np


# %% _____________________________ functions



# %% _____________________________ 


# rootdir=r'C:\Users\alexy\Downloads\for visualization\weightedaverage_validation_accuracy_notscaled\training_dir'
# rootdir=r'C:\Users\alexy\Downloads\for visualization\sgd'


rootdir='C:/Users/alexy/Downloads/for visualization/iterative_swa_from_same_sgd'
experiment='iterative_swa'



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
for file_path in file_paths:
    
    if  '.sh' in file_path or '-9' in file_path or '-0' in file_path:
        continue
    
    checkpoint = torch.load(file_path)

    
    experiment_results.append({'experiment': experiment, 
                               # 'learning_rate':checkpoint['lr'],
                                'epoch':checkpoint['epoch'] ,
                                'train_loss': float(checkpoint['train_res']['loss']),
                                'train_accuracy': checkpoint['train_res']['accuracy'],
                                'validation_loss': float(checkpoint['val_res']['loss']),
                                'validation_accuracy':checkpoint['val_res']['accuracy']})




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

# experiment_results_df.columns


melted_results_df=pd.melt(experiment_results_df, id_vars=['experiment','epoch'], value_vars=[ 'train_accuracy',
        'train_loss','validation_accuracy', 'validation_loss'])




# melted_results_df=pd.melt(experiment_results_df, id_vars=['experiment','epoch'], value_vars=[ 'train_accuracy',
#         'train_loss'])




# melted_results_df=pd.melt(experiment_results_df, id_vars=['experiment','epoch'], value_vars=['train_loss', 'train_accuracy',
#        'validation_loss', 'validation_accuracy', 'original_swa_loss',
#        'original_swa_accuracy', 'our_swa_loss', 'our_swa_accuracy'])


# sns.set_theme()
# sns.lineplot(melted_results_df[melted_results_df['variable'].str.contains('accuracy')], x='epoch',y='value', hue='variable')

# # sns.lineplot(melted_results_df, x='epoch',y='value', hue='variable')

# for x in np.arange(15,150,10):
#     plt.plot([x, x], [50, 100], 'r--', lw=2)


# %% 


rootdir='C:/Users/alexy/Downloads/for visualization/sgd'
experiment='sgd'



file_paths=[]

# get file paths
for subdir, dirs, files in os.walk(rootdir): 
    for file in files:
        
        
        file_path=os.path.join(subdir, file)
        file_paths.append(file_path)




experiment_results=[]

# without SWA results
for file_path in file_paths:
    
    if  '.sh' in file_path or '-9' in file_path or '-0' in file_path:
        continue
    
    checkpoint = torch.load(file_path)

    
    experiment_results.append({'experiment': experiment, 
                               # 'learning_rate':checkpoint['lr'],
                                'epoch':checkpoint['epoch'] ,
                                'train_loss': float(checkpoint['train_res']['loss']),
                                'train_accuracy': checkpoint['train_res']['accuracy'],
                                'validation_loss': float(checkpoint['val_res']['loss']),
                                'validation_accuracy':checkpoint['val_res']['accuracy']})




experiment_results_df_2=pd.DataFrame(experiment_results)

# temp=experiment_results_df_2[experiment_results_df_2['epoch'].isin(np.arange(11))]
# temp['experiment']=['iterative_swa']*temp.shape[0]

# experiment_results_df=pd.concat([experiment_results_df,temp],axis=0)

melted_results_df_2=pd.melt(experiment_results_df_2, id_vars=['experiment','epoch'], value_vars=[ 'train_accuracy',
        'train_loss','validation_accuracy', 'validation_loss'])



melted_results_df_3=pd.concat([melted_results_df,melted_results_df_2],axis=0)
melted_results_df_3['legend']=melted_results_df_3['experiment']+ '_' +melted_results_df_3['variable']



sns.set_theme()
sns.lineplot(melted_results_df_3[melted_results_df['variable'].str.contains('accuracy')], x='epoch',y='value', hue='legend')


for x in np.arange(15,150,10):
    plt.plot([x, x], [50, 100], 'r--', lw=2)



