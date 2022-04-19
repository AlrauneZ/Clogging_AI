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
### Restructure Data
###############################################################################

data = np.zeros((len(params),len(n_estimators)+1,len(iterations)+1))

for ic, cond in enumerate(conditions):
    for it,n_it in enumerate(iterations):
        data_load = np.transpose(np.loadtxt('../results/extra/RF_Training_{}_It{}.csv'.format(cond,n_it),delimiter = ','))
        data[:,1:,it+1]=data_load[1:,:]

    for ip,param in enumerate(params):
        data[ip,1:,0] = n_estimators
        data[ip,0,1:] = iterations
        np.savetxt('../results/RF_Training_{}_{}.csv'.format(cond,param),data[ip,:,:].T,delimiter = ',') 

###############################################################################
### Plot
###############################################################################

fig = plt.figure(figsize = [7.5,3])
for ic, cond in enumerate(conditions):
    data_full = np.loadtxt('../results/RF_Training_{}_full.csv'.format(cond),delimiter = ',')     

    for ip,param in enumerate(params):
        data = np.loadtxt('../results/RF_Training_{}_{}.csv'.format(cond,param),delimiter = ',')     
        x = data[1:,0]    
        score = data[1:,1:]
        n_iterations = len(x)
        
        ax = fig.add_subplot(2,4, ic*4 + ip + 1)
        ax.plot(x,score,marker = 'o',ls = '-',lw=2,markersize = 4)
        ax.grid(True)
        ax.tick_params(axis="both",which="major",labelsize=textsize-1)   

        if ic == 0:
            ax.set_title(title_text[ip],fontsize=textsize)
            if ip == 0:        
                ax.set_ylabel('$R^2$ - fav',fontsize=textsize)
        elif ic == 1:
            ax.set_xlabel('Number of iterations',fontsize=textsize)
            if ip == 0:        
                ax.set_ylabel('$R^2$ - unfav',fontsize=textsize)

plt.tight_layout()
fig.legend([r'$100$',r'$200$',r'$300$',r'$400$'], ncol = len(iterations),bbox_to_anchor=[0.77, 0.105],fontsize=textsize)#,&nbsp)
fig.subplots_adjust(bottom=0.24)# Adjusting the sub-plots

plt.savefig('../results/FigS05_RF_Hyper_Py.pdf')   

