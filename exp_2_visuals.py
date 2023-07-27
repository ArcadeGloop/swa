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


# %% _____________________________ version 2



rootdir=r'C:\Users\alexy\Downloads\for visualization\experiment_2'


file_paths=[]

# get file paths
for subdir, dirs, files in os.walk(rootdir): 
    for file in files:
        
        
        file_path=os.path.join(subdir, file)
        file_paths.append(file_path)



checkpoint = torch.load(file_paths[40])

checkpoint.keys()


experiment_results=[]


# without SWA results
for file_path in file_paths:
    
    if  '.sh' in file_path  or '-0.' in file_path or'-150' in file_path:
        continue
    
    checkpoint = torch.load(file_path)
 
    experiment_results.append({'experiment': file_path, 
                               # 'learning_rate':checkpoint['lr'],
                                'epoch':checkpoint['epoch'] ,
                                'train_loss': float(checkpoint['train_res']['loss']),
                                'train_accuracy': checkpoint['train_res']['accuracy'],
                                'validation_loss': float(checkpoint['val_res']['loss']),
                                'validation_accuracy':checkpoint['val_res']['accuracy']})




experiment_results_df=pd.DataFrame(experiment_results)

# %% _____________________________ resctucture data


experiment_results_df['experiment']=experiment_results_df['experiment'].str.split('\\').str[-2]

experiment_results_df['experiment'].unique()


melted_results_df=pd.melt(experiment_results_df, id_vars=['experiment','epoch'], value_vars=[ 'train_accuracy',
        'train_loss','validation_accuracy', 'validation_loss'])



melted_results_df['legend']=melted_results_df['experiment']+ '_' +melted_results_df['variable']


# %% plot it

sns.set_theme()
fig=sns.lineplot(melted_results_df[melted_results_df['variable'].str.contains('validation_accuracy')], x='epoch',y='value', hue='experiment')


for x in np.arange(15,150,10):
    plt.plot([x, x], [50, 100], 'r--', lw=2)


x_y_label_size=22

plt.xlabel("epoch",size=x_y_label_size)
plt.xticks(np.arange(15,151,5))
plt.yticks(np.arange(70,101,5), size=x_y_label_size-2)
plt.xlim([15,151])
plt.ylim([79,100])
plt.tick_params('x',labelsize=x_y_label_size-5,labelrotation=0 )
plt.setp(fig.get_legend().get_texts(), fontsize=x_y_label_size-2) 
plt.setp(fig.get_legend().get_title(), fontsize=x_y_label_size-2) 
plt.ylabel("Accuracy",size=x_y_label_size)

