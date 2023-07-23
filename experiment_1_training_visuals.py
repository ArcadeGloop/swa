# %% _____________________________ description

"""


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


rootdir=r'C:\Users\alexy\Downloads\for visualization\experiment_1'


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
                                    'epoch':checkpoint['epoch'],
                                    'swa_validation_loss': float(checkpoint['swa_res']['loss']),
                                    'swa_validation_accuracy':checkpoint['swa_res']['accuracy'],
                                    'our_swa_validation_loss': float(checkpoint['our_swa_res']['loss']),
                                    'our_swa_validation_accuracy':checkpoint['our_swa_res']['accuracy']})
    


experiment_results_df=pd.DataFrame(experiment_results)

experiment_results_df['experiment']=experiment_results_df['experiment'].str.split('\\').str[-3]

experiment_results_df['experiment'].unique()

experiment_results_df['experiment']=experiment_results_df['experiment'].map({'trainingcheckpoints_exponential_01':'exponential_smoothing_alpha_0.1',
                                                                             'trainingcheckpoints_exponential_05':'exponential_smoothing_alpha_0.5',
                                                                             'trainingcheckpoints_exponential_09':'exponential_smoothing_alpha_0.9',
                          'trainingcheckpoints_weightedaverage_validation_accuracy_notscaled':'unscaled_accuracy_weighted_average',
                          'trainingcheckpoints_weightedaverage_validation_accuracy_scaled':'scaled_accuracy_weighted_average',
                          'trainingcheckpoints_weightedaverage_validation_loss_notscaled':'unscaled_loss_weighted_average',
                          'trainingcheckpoints_weightedaverage_validation_loss_scaled':'scaled_loss_weighted_average'})

# experiment_results_df.columns


melted_results_df=pd.melt(experiment_results_df,
                          id_vars=['experiment','epoch'],
                          value_vars=[ 'swa_validation_loss',
                                      'swa_validation_accuracy',
                                      'our_swa_validation_loss',
                                      'our_swa_validation_accuracy'])


melted_results_df['legend']=melted_results_df['experiment']+ '_' +melted_results_df['variable']


# %% plot 1
plot_df=melted_results_df[melted_results_df['variable'].str.contains('accuracy')]
plot_df=plot_df[plot_df['legend'].str.contains('our')]


plot_df_2=melted_results_df[melted_results_df['variable'].str.contains('accuracy')]
plot_df_2=plot_df_2[~plot_df_2['legend'].str.contains('our')]

plot_df_2['legend']=['original_swa']*plot_df_2.shape[0]

plot_df=pd.concat([plot_df,plot_df_2],axis=0)

plot_df['legend'].unique()

plot_df['legend']=plot_df['legend'].map({'exponential_smoothing_alpha_0.1_our_swa_validation_accuracy': 'exponential_smoothing_alpha_0.1',
                       'unscaled_accuracy_weighted_average_our_swa_validation_accuracy': 'unscaled_accuracy_weighted_average',
                       'scaled_accuracy_weighted_average_our_swa_validation_accuracy': 'scaled_accuracy_weighted_average',
                       'unscaled_loss_weighted_average_our_swa_validation_accuracy': 'unscaled_loss_weighted_average',
                       'original_swa':'original_swa',
                       'exponential_smoothing_alpha_0.5_our_swa_validation_accuracy': 'exponential_smoothing_alpha_0.5',
                       'exponential_smoothing_alpha_0.9_our_swa_validation_accuracy':'exponential_smoothing_alpha_0.9',
                       'scaled_loss_weighted_average_our_swa_validation_accuracy':'scaled_loss_weighted_average'})

# plot_df['legend'].unique()

# %%
sns.set_theme()
fig=sns.lineplot(plot_df, x='epoch',y='value', hue='legend')


x_y_label_size=22

plt.xlabel("epoch",size=x_y_label_size)
plt.xticks(np.arange(150,201,1))
plt.yticks(np.arange(84,92,1), size=x_y_label_size-2)
plt.ylim([86,91.5])
plt.xlim([151,200])
plt.tick_params('x',labelsize=x_y_label_size-5,labelrotation=90 )
plt.setp(fig.get_legend().get_texts(), fontsize=x_y_label_size-2) 
plt.setp(fig.get_legend().get_title(), fontsize=x_y_label_size-2) 
plt.ylabel("Validation Accuracy",size=x_y_label_size)


# %%

a=plot_df[(plot_df['epoch']==200) & (plot_df['variable']=='our_swa_validation_accuracy')]


b=plot_df[(plot_df['epoch']==200) & (plot_df['variable']=='swa_validation_accuracy')]

b['value'].mean()

b['value'].std()
