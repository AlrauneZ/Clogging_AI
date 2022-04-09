#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

conditions = ['fav','unfav']
params = ['CN','SC','HC','VF']
n_it = 25

### Plot Specifications
plt.close('all')
textsize = 8
title_text = ['Coordination number','Surface coverage','Conductivity','Void fraction']

###############################################################################
### Plot
###############################################################################
fig = plt.figure(figsize = [7.5,3])

for ic, cond in enumerate(conditions):
    for ip,param in enumerate(params):

        # data1 = np.loadtxt('../data/ANN_{}_{}.csv'.format(cond,param),delimiter = ',')      
        # score = data1[:,np.arange(2,data1.shape[1],3)]
        # n_iterations = data1.shape[0]
        # x = np.arange(1,n_iterations+1)

        # data_save = np.zeros((score.shape[0]+1,score.shape[1]+1))         
        # data_save[0,1:] = [2,3,4,5,6,7,8]
        # data_save[1:,0] = np.arange(1,score.shape[0]+1)
        # data_save[1:,1:] = score
        # np.savetxt('../data/ANN_{}_{}_R2.csv'.format(cond,param),data_save,delimiter = ',')

        # data = np.loadtxt('../data/ANN_{}_{}_R2.csv'.format(cond,param),delimiter = ',')
        # x = data[1:,0]
        # score = data[1:,1:]
        # n_iterations = len(x)

        data = np.transpose(np.loadtxt('../results/ANN_Training_{}_{}_It{}.csv'.format(cond,param,n_it),delimiter = ','))
        x = data[1:,0]
        score = data[1:,1:]
        n_iterations = len(x)
        
        ax = fig.add_subplot(2,4, ic*4 + ip + 1)

        ax.plot(x,score,marker = 'o',ls = ':',lw=0.5,markersize = 2)
        ax.set_xlim([0.5,n_iterations+0.5])
        ax.set_xticks(np.arange(1,n_iterations+1,4))
        ax.grid(True)
        ax.tick_params(axis="both",which="major",labelsize=textsize-1)
    

        if ic == 0:
            ax.set_title(title_text[ip],fontsize=textsize)
            R2_max = [[-0.2,0.6],[-5,0],[-2.5,0],[-2.5,0]]
            if ip == 0:        
                ax.set_ylabel('$R^2$ - fav. cond.',fontsize=textsize)
        elif ic == 1:
            ax.set_xlabel('Number of iterations',fontsize=textsize)
            R2_max = [[-3,0.6],[-3,0],[0.4,1],[-2,1]]
            if ip == 0:        
                ax.set_ylabel('$R^2$ - unfav. cond.',fontsize=textsize)
        # ax.set_ylim(R2_max[ip])    
        ax.set_ylim([-200,1])    

# fig.text(0.035,0.025,'ANN, {}orable conditions'.format(condition),fontsize=textsize, bbox=dict(facecolor='w', alpha=0.2,boxstyle='round'))#transform=ax1.transAxes)
plt.tight_layout()
fig.legend(['2','3','4','5','6','7','8   Neurons' ], ncol = 7,bbox_to_anchor=[0.85, 0.09],fontsize=textsize)#,&nbsp)
fig.subplots_adjust(bottom=0.2)# Adjusting the sub-plots

# plt.savefig('../results/Fig01_ANN_Hyper.png',dpi=300)   
plt.savefig('../results/Fig01_ANN_Hyper_Py.pdf')   

