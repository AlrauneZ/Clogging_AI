#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

params = ['CN','SC','HC','VF']
iterations = [500,1000,2000]

### Plot Specifications
plt.close('all')
textsize = 9
title_text = ['Coordination number','Surface coverage','Conductivity','Void fraction']

###############################################################################
### Data Files & Setting
###############################################################################

data_name = '../data/SI/DT_Training_{}_{}_It{}.csv'#.format(cond,param,n_it)
data_namef = '../data/SI/DT_Training_{}_{}_full.csv'#.format(cond,param,n_it)

cond = 'unfav' #'fav' 3

###############################################################################
### Plot
###############################################################################

fig = plt.figure(figsize = [7.5,5])
print("########################################")
print("Decision Tree - Hyper parameter testing: \n{}orable conditions".format(cond))
for ip,param in enumerate(params):
    print("\n########################################\nParameter {}".format(param))

    for it,n_it in enumerate(iterations):
        data_load = np.loadtxt(data_name.format(cond,param,n_it),delimiter = ',')
        range_max_depth = data_load[1:,0] #range(2,20)
        range_min_samples_split = data_load[0,1:] #range(2,10)
        ind = np.unravel_index(np.argmax(data_load[1:,1:], axis=None), data_load[1:,1:].shape)
        print("\nIterations: {}".format(n_it))
        print('Maximum score {:.2f} for max_depth = {} and min_samples_split = {}'.format(data_load[ind[0]+1,ind[1]+1],range_max_depth[ind[0]],range_min_samples_split[ind[1]]))

        ax = fig.add_subplot(len(iterations)+1,len(params), it*len(params) + ip + 1)
       
        ax.imshow(data_load[1:,1:].T-1,cmap='hot')
        if it == 0:
            ax.set_title(title_text[ip],fontsize=textsize)
          
        if ip == 0:
            ax.set_ylabel(r"$mss$ - N = {}".format(n_it),fontsize=textsize)    
            ax.set_yticks(range_min_samples_split[::3]-2)
            ax.set_yticklabels(range_min_samples_split[::3],fontsize=textsize-1)    
        else:
            ax.set_yticks([])

        ax.set_xticks(range_max_depth[::4]-2)
        ax.set_xticklabels(range_max_depth[::4],fontsize=textsize-1)    

    data_full = np.loadtxt(data_namef.format(cond,param),delimiter = ',')
    range_max_depth = data_full[1:,0] #range(2,20)
    range_min_samples_split = data_full[0,1:] #range(2,10)

    ind = np.unravel_index(np.argmax(data_full[1:,1:], axis=None), data_full[1:,1:].shape)
    print("\nFull")
    print('Maximum score {:.2f} for max_depth = {} and min_samples_split = {}'.format(data_full[ind[0]+1,ind[1]+1],range_max_depth[ind[0]],range_min_samples_split[ind[1]]))

    ax = fig.add_subplot(len(iterations)+1,len(params), len(iterations)*len(params)+ ip+1)       
    ax.imshow(data_full[1:,1:].T-1,cmap='hot')
    ax.set_xticks(range_max_depth[::4]-2)
    ax.set_xticklabels(range_max_depth[::4],fontsize=textsize-1)    

    if ip == 0:
        ax.set_ylabel(r"$mss$ - full",fontsize=textsize)    
        ax.set_yticks(range_min_samples_split[::3]-2)
        ax.set_yticklabels(range_min_samples_split[::3],fontsize=textsize-1)    
    else:
        ax.set_yticks([])
    ax.set_xlabel(r"$D$ - max_depth",fontsize=textsize)    


fig.text(0.007,0.023,'{}orable \nconditions'.format(cond),fontsize=textsize, bbox=dict(facecolor='w', alpha=0.2,boxstyle='round'))#transform=ax1.transAxes)
plt.tight_layout()

plt.savefig('../results/FigS04_DT_Hyper_{}.pdf'.format(cond))   

