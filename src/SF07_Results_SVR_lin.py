#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

conditions = ['fav','unfav']
params = ['CN','SC','HC','VF']
iterations = [500,1500,3000]

### Plot Specifications
plt.close('all')
textsize = 9
title_text = ['Coordination number','Surface coverage','Conductivity','Void fraction']

###############################################################################
### Data Files & Setting
###############################################################################

data_name = '../data/SI/SVR_lin_Training_{}_{}_It{}.csv'#.format(cond,param,n_it)
data_namef = '../data/SI/SVR_lin_Training_{}_{}_full.csv'#.format(cond,param,n_it)

cond = 'fav'

###############################################################################
### Plot
###############################################################################

fig = plt.figure(figsize = [7.5,7.5])
for ip,param in enumerate(params):

    for it,n_it in enumerate(iterations):
        data_load = np.loadtxt(data_name.format(cond,param,n_it),delimiter = ',')

        ax = fig.add_subplot(len(iterations)+1,len(params), it*len(params) + ip + 1)        
        ax.imshow(data_load[1:,1:]-1,cmap='hot')

        if it == 0:
            ax.set_title(title_text[ip],fontsize=textsize)

        if ip == 0:
            ax.set_ylabel(r"$C$ - N={}".format(n_it),fontsize=textsize)    
            ax.set_yticks([0,1,2,3,4,5])
            ax.set_yticklabels([r'$0.001$',r'$0.01$',r'$0.1$',r'$1$',r'$10$',r'$100$'],fontsize=textsize-1)    
        else:
            ax.set_yticks([])
          

        ax.set_xticks([1,3,5])
        ax.set_xticklabels([r'$0.01$',r'$1$',r'$100$'],fontsize=textsize-1)    

    data_full = np.loadtxt(data_namef.format(cond,param),delimiter = ',')
    ax = fig.add_subplot(len(iterations)+1,len(params), len(iterations)*len(params)+ ip+1)       
    ax.imshow(data_full[1:,1:]-1,cmap='hot')#,vmin=-5,vmax=0)
    ax.set_xticks([1,3,5])
    ax.set_xticklabels([r'$0.01$',r'$1$',r'$100$'],fontsize=textsize-1)    
    ax.set_yticks([])

    if ip == 0:
        ax.set_ylabel(r"$C$ - full",fontsize=textsize)    
        ax.set_yticks([0,1,2,3,4,5])
        ax.set_yticklabels([r'$0.001$',r'$0.01$',r'$0.1$',r'$1$',r'$10$',r'$100$'],fontsize=textsize-1)    
    else:
        ax.set_yticks([])

    ax.set_xlabel(r"$\gamma$",fontsize=textsize)    

fig.text(0.01,0.02,'{}orable \nconditions'.format(cond),fontsize=textsize, bbox=dict(facecolor='w', alpha=0.2,boxstyle='round'))#transform=ax1.transAxes)
plt.tight_layout()

plt.savefig('../results/FigS07_SVR_Hyper_{}.pdf'.format(cond))   
