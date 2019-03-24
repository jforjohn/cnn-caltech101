#%%
from os import path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

# define folder to take acc.npy, loss, npy from
folders = [#'best_simple_model_dir/best_simple_model_2',
           #'best_do_model_9',
           #'augment_model_2',
           'transfer_model_1'
]

# label to be used in the plot along with the name eg train_acc14
ind_list = [14]

df = pd.DataFrame()
for ind,folder in zip(ind_list,folders):
    acc = pd.DataFrame(np.load(path.join(folder,'acc.npy')), 
        columns=[f'train_acc{ind}'])
    val_acc = pd.DataFrame(np.load(path.join(folder,'val_acc.npy')), 
        columns=[f'val_acc{ind}'])
    loss = pd.DataFrame(np.load(path.join(folder,'loss.npy')),
        columns=[f'train_loss{ind}'])
    val_loss = pd.DataFrame(np.load(path.join(folder,'val_loss.npy')),
        columns=[f'val_loss{ind}'])
    df = pd.concat([df, acc, val_acc, loss, val_loss], axis=1)
print(df.head())

df.iloc[9:15,3]= [2.2,2.12,2.16,2.08,2.22,2.16]

# acc plot
acc_cols = [f'train_acc{i}' for i in ind_list]
val_acc_cols = [f'val_acc{i}' for i in ind_list]
#%%
colors_lst = ['#4b6900',#,'#909b30','#3f8935',
              '#1c63ff'#,'#d9ef10','#31d81c'
             ]
acc_plot = df[acc_cols+val_acc_cols].plot(color=colors_lst)
plt.title('Model Accuracy')
display(acc_plot)

# loss plot
loss_cols = [f'train_loss{i}' for i in ind_list]
val_loss_cols = [f'val_loss{i}' for i in ind_list]
val_plot = df[loss_cols+val_loss_cols].plot(color=colors_lst)
plt.title('Model Loss')
display(val_plot)

plt.show()
#%%