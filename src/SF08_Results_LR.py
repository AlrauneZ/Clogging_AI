#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

conditions = ['fav','unfav']
params = ['CN','SC','HC','VF']
iterations = [500,1000,1500,2000,3000]
alphas = [0.0001,0.0010,0.0100,0.1000,1.0000,10.0000,100.0000]
#n_it = 25

### Plot Specifications
plt.close('all')
textsize = 8
title_text = ['Coordination number','Surface coverage','Conductivity','Void fraction']

###############################################################################
### Restructure Data
###############################################################################

# data = np.zeros((len(params),len(alphas)+1,len(iterations)+1))
# for ic, cond in enumerate(conditions):
#     for it,n_it in enumerate(iterations):
#         data_load = np.transpose(np.loadtxt('../results/extra/LR_Training_{}_It{}.csv'.format(cond,n_it),delimiter = ','))
#         data[:,1:,it+1]=data_load[1:,:]

#     for ip,param in enumerate(params):
#         data[ip,1:,0] = alphas
#         data[ip,0,1:] = iterations
#         np.savetxt('../results/LR_Training_{}_{}.csv'.format(cond,param),data[ip,:,:].T,delimiter = ',') 

###############################################################################
### Plot
###############################################################################

fig = plt.figure(figsize = [7.5,3])
for ic, cond in enumerate(conditions):
    data_full = np.loadtxt('../results/LR_Training_{}_full.csv'.format(cond),delimiter = ',')     

    for ip,param in enumerate(params):
        data = np.loadtxt('../results/LR_Training_{}_{}.csv'.format(cond,param),delimiter = ',')     
        x = data[1:,0]    
        score = data[1:,1:]
        n_iterations = len(x)
        
        ax = fig.add_subplot(2,4, ic*4 + ip + 1)

        ax.plot(x,score,marker = 'o',ls = ':',lw=1,markersize = 3)
#        ax.set_xlim([0.5,n_iterations+0.5])
        # ax.set_xticks(np.arange(1,n_iterations+1,4))

        # ax.scatter(3100*np.ones(len(data_full[:,0])),data_full[:,ip+1],marker='s')


        ax.grid(True)
        ax.tick_params(axis="both",which="major",labelsize=textsize-1)   

        if ic == 0:
            ax.set_title(title_text[ip],fontsize=textsize)
            # R2_max = [[-0.2,0.6],[-5,0],[-2.5,0],[-2.5,0]]
            if ip == 0:        
                ax.set_ylabel('$R^2$ - fav. cond.',fontsize=textsize)
        elif ic == 1:
            ax.set_xlabel('Number of iterations',fontsize=textsize)
            # R2_max = [[-3,0.6],[-3,0],[0.4,1],[-2,1]]
            if ip == 0:        
                ax.set_ylabel('$R^2$ - unfav. cond.',fontsize=textsize)
        # ax.set_ylim(R2_max[ip])    
        # ax.set_ylim([-4,1])    

        # for it,n_it in enumerate(iterations):
        #     ax.scatter(3100,data_full[it,ip+1],marker='^',color='C{}'.format(it))

# fig.text(0.035,0.025,'ANN, {}orable conditions'.format(condition),fontsize=textsize, bbox=dict(facecolor='w', alpha=0.2,boxstyle='round'))#transform=ax1.transAxes)
plt.tight_layout()
fig.legend([r'$10^{-4}$',r'$10^{-3}$',r'$10^{-2}$',r'$0.1$',r'$1$',r'$10$',r'$100$' ], ncol = 7,bbox_to_anchor=[0.85, 0.09],fontsize=textsize)#,&nbsp)
fig.subplots_adjust(bottom=0.2)# Adjusting the sub-plots

plt.savefig('../results/Fig05_LR_Hyper_Py.pdf')   

