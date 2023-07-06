#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

conditions = ['fav','unfav']
params = ['CN','SC','HC','VF']
iterations = [100,200,300,400]
n_estimators = [100,200,300]

### Plot Specifications
plt.close('all')
textsize = 9
title_text = ['Coordination number','Surface coverage','Conductivity','Void fraction']

###############################################################################
### Plot
###############################################################################

fig = plt.figure(figsize = [7.5,3])
for ic, cond in enumerate(conditions):

    for ip,param in enumerate(params):
             
        ax = fig.add_subplot(len(conditions),len(params), ic*len(params) + ip + 1)
        
        
        for it,n_it in enumerate(iterations):
            data = np.loadtxt('../results/SI/RF_Training_{}_It{}.csv'.format(cond,n_it),delimiter = ',')
            ax.plot(data[:,0],data[:,1+ip],marker = 'o',ls = '-',lw=2,markersize = 4)

        data = np.loadtxt('../results/SI/RF_Training_{}_full.csv'.format(cond),delimiter = ',')
        ax.plot(data[:,0],data[:,1+ip],marker = 'o',ls = '-',lw=2,markersize = 4)
             
        ax.grid(True)
        ax.tick_params(axis="both",which="major",labelsize=textsize-1)   
        # ax.set_xticks(data[::2,0])

        if ic == 0:
            ax.set_title(title_text[ip],fontsize=textsize)
            if ip == 0:        
                ax.set_ylabel('NSE - fav',fontsize=textsize)
        elif ic == 1:
            ax.set_xlabel(r'$D_{RF}$',fontsize=textsize)
            if ip == 0:        
                ax.set_ylabel('NSE - unfav',fontsize=textsize)

plt.tight_layout()
fig.legend(iterations+['full'], ncol = 7,bbox_to_anchor=[0.80, 0.105],fontsize=textsize)#,&nbsp)
fig.subplots_adjust(bottom=0.24)# Adjusting the sub-plots

plt.savefig('../results/FigS03_RF_Hyper.pdf')   

