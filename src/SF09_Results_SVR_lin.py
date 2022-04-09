#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

conditions = ['fav','unfav']
params = ['CN','SC','HC','VF']
iterations = [500,1500,3000]

### Plot Specifications
plt.close('all')
textsize = 8
title_text = ['Coordination number','Surface coverage','Conductivity','Void fraction']

###############################################################################
### Data Files & Setting
###############################################################################

data_name = '../results/SVR_lin_Training_{}_{}_It{}.csv'#.format(cond,param,n_it)
data_namef = '../results/SVR_lin_Training_{}_{}_full.csv'#.format(cond,param,n_it)

cond = 'fav'

###############################################################################
### Plot
###############################################################################

fig = plt.figure(figsize = [7.5,7.5])
for ip,param in enumerate(params):

    data_full = np.loadtxt(data_namef.format(cond,param),delimiter = ',')
    ax = fig.add_subplot(len(params), len(iterations)+1, (ip+1)*(len(iterations)+1))       
    ax.imshow(data_full[1:,1:]-1,cmap='hot')#,vmin=-5,vmax=0)
    ax.set_xticks([1,3,5])
    ax.set_xticklabels([r'$0.01$',r'$1$',r'$100$'],fontsize=textsize)    
    ax.set_yticks([])
    # ax.set_yticklabels([r'$10^{-3}$',r'$0.1$',r'$10$'],fontsize=textsize)    
    if ip == 0:
        ax.set_title('full',fontsize=textsize)
    elif ip == len(param)+1:
        ax.set_xlabel(r"$\gamma$",fontsize=textsize)    

    for it,n_it in enumerate(iterations):
        data_load = np.loadtxt(data_name.format(cond,param,n_it),delimiter = ',')

        ax = fig.add_subplot(len(params), len(iterations)+1, ip*(len(iterations)+1) + it + 1)
       
        ax.imshow(data_load[1:,1:]-1,cmap='hot')#,vmin=-5,vmax=0)
        ax.tick_params(axis="both",which="major",labelsize=textsize-1)   
        if it == 0:
            ax.set_ylabel(r"$C$ - {}".format(param),fontsize=textsize)    
            ax.set_yticks([0,1,2,3,4,5])
            ax.set_yticklabels([r'$0.001$',r'$0.01$',r'$0.1$',r'$1$',r'$10$',r'$100$'],fontsize=textsize)    
        else:
            ax.set_yticks([])
          
        if ip == 0:
            ax.set_title(n_it,fontsize=textsize)
        elif ip == len(param)+1:
            ax.set_xlabel(r"$\gamma$",fontsize=textsize)    

        ax.set_xticks([1,3,5])
        ax.set_xticklabels([r'$0.01$',r'$1$',r'$100$'],fontsize=textsize)    

# fig.colorbar()
fig.text(0.035,0.025,'SVR lin, {}orable conditions'.format(cond),fontsize=textsize, bbox=dict(facecolor='w', alpha=0.2,boxstyle='round'))#transform=ax1.transAxes)
plt.tight_layout()

plt.savefig('../results/Fig06_SVR_Hyper_Py_{}.pdf'.format(cond))   

