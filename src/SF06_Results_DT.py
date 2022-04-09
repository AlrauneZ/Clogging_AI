#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

conditions = ['fav','unfav']
params = ['CN','SC','HC','VF']
iterations = [500,1000,2000]

### Plot Specifications
plt.close('all')
textsize = 8
title_text = ['Coordination number','Surface coverage','Conductivity','Void fraction']

###############################################################################
### Data Files & Setting
###############################################################################

data_name = '../results/DT_Training_{}_{}_It{}.csv'#.format(cond,param,n_it)
data_namef = '../results/DT_Training_{}_{}_full.csv'#.format(cond,param,n_it)

cond = 'unfav'

###############################################################################
### Plot
###############################################################################

fig = plt.figure(figsize = [7.5,5])
print("########################################")
print("Decision Tree - Hyper parameter testing: \n{}orable conditions".format(cond))
for ip,param in enumerate(params):
    print("\n########################################\nParameter {}".format(param))

    data_full = np.loadtxt(data_namef.format(cond,param),delimiter = ',')
    range_max_depth = data_full[1:,0] #range(2,20)
    range_min_samples_split = data_full[0,1:] #range(2,10)

    ind = np.unravel_index(np.argmax(data_full[1:,1:], axis=None), data_full[1:,1:].shape)
    print("\nFull".format(param))
    print('Maximum score {:.2f} for max_depth = {} and min_samples_split = {}'.format(data_full[ind[0]+1,ind[1]+1],range_max_depth[ind[0]],range_min_samples_split[ind[1]]))

    ax = fig.add_subplot(len(params), len(iterations)+1, (ip+1)*(len(iterations)+1))       
    ax.imshow(data_full[1:,1:].T-1,cmap='hot')#,vmin=-5,vmax=0)
    ax.set_xticks(range_max_depth[::4]-2)
    ax.set_xticklabels(range_max_depth[::4],fontsize=textsize)    
    ax.set_yticks([])
    # ax.set_yticks(range_min_samples_split[::3]-2)
    # ax.set_yticklabels(range_min_samples_split[::3],fontsize=textsize)    

    if ip == 0:
        ax.set_title('full',fontsize=textsize)
    elif ip == len(param)+1:
        ax.set_xlabel(r"$D$ - max_depth",fontsize=textsize)    

    for it,n_it in enumerate(iterations):
        data_load = np.loadtxt(data_name.format(cond,param,n_it),delimiter = ',')
        range_max_depth = data_load[1:,0] #range(2,20)
        range_min_samples_split = data_load[0,1:] #range(2,10)
        ind = np.unravel_index(np.argmax(data_load[1:,1:], axis=None), data_load[1:,1:].shape)
        print("\nIterations: {}".format(n_it))
        print('Maximum score {:.2f} for max_depth = {} and min_samples_split = {}'.format(data_load[ind[0]+1,ind[1]+1],range_max_depth[ind[0]],range_min_samples_split[ind[1]]))

        ax = fig.add_subplot(len(params), len(iterations)+1, ip*(len(iterations)+1) + it + 1)
       
        ax.imshow(data_load[1:,1:].T-1,cmap='hot')#,vmin=-5,vmax=0)
        ax.tick_params(axis="both",which="major",labelsize=textsize-1)   
        if it == 0:
            ax.set_ylabel(r"$mss$ - {}".format(param),fontsize=textsize)    
            ax.set_yticks(range_min_samples_split[::3]-2)
            ax.set_yticklabels(range_min_samples_split[::3],fontsize=textsize)    
        else:
            ax.set_yticks([])
          
        if ip == 0:
            ax.set_title(n_it,fontsize=textsize)
        elif ip == len(param)+1:
            ax.set_xlabel(r"$D$ - max_depth",fontsize=textsize)    

        ax.set_xticks(range_max_depth[::4]-2)
        ax.set_xticklabels(range_max_depth[::4],fontsize=textsize)    

# fig.colorbar()
fig.text(0.035,0.025,'DT, {}orable conditions'.format(cond),fontsize=textsize, bbox=dict(facecolor='w', alpha=0.2,boxstyle='round'))#transform=ax1.transAxes)
plt.tight_layout()

plt.savefig('../results/Fig07_DT_Hyper_Py_{}.pdf'.format(cond))   

