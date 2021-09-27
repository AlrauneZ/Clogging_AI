#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

condition = 'fav' #'fav' # 
params = ['CN','SC','HC','VF']

### Plot Specifications
plt.close('all')
textsize = 8
title_text = ['Coordination number','Surface coverage','Conductivity','Void fraction']
ylabel_text = ['MSE','MAE','$R^2$']
if condition == 'fav':
    MSE_max = [[0.2,0.55],[0.017,0.025],[0.0035,0.0065],[0.0035,0.0065 ]]
    R2_max = [[-0.2,0.6],[-5,0],[-2.5,0],[-2.5,0]]
elif condition == 'unfav':
    MSE_max = [[0.2,0.95],[0.008,0.018],[0.003,0.025],[0,0.006]]
    R2_max = [[-3,0.6],[-3,0],[0.4,1],[-2,1]]

###############################################################################
### Plot
###############################################################################
fig = plt.figure(figsize = [7.5,5])
for ip,param in enumerate(params):
    data1 = np.loadtxt('../data/ANN_{}_{}.csv'.format(condition,param),delimiter = ',')
    x = np.arange(1,data1.shape[0]+1)

    for i in range(3):
        ax = fig.add_subplot(3,4, 4*i + ip + 1)
        score = data1[:,np.arange(i,data1.shape[1],3)]
        ax.plot(x,score,marker = 'o',ls = ':',lw=0.5,markersize = 2)
        ax.set_xlim([0.5,data1.shape[0]+0.5])

        ax.set_xticks(np.arange(1,data1.shape[0]+1,4))
        ax.grid(True)
        ax.tick_params(axis="both",which="major",labelsize=textsize-1)

        if i == 0:
            ax.set_title(title_text[ip],fontsize=textsize)
            ax.set_ylim(MSE_max[ip])
        elif i == 2:
            ax.set_ylim(R2_max[ip])
            ax.set_xlabel('Number of iterations',fontsize=textsize)
        if ip == 0:        
            ax.set_ylabel(ylabel_text[i],fontsize=textsize)

plt.tight_layout()
fig.legend(['2','3','4','5','6','7','8   Neurons' ], ncol = 7,bbox_to_anchor=[0.9, 0.06],fontsize=textsize)#,&nbsp)
fig.subplots_adjust(bottom=0.13)# Adjusting the sub-plots
fig.text(0.035,0.025,'ANN, {}orable conditions'.format(condition),fontsize=textsize, bbox=dict(facecolor='w', alpha=0.2,boxstyle='round'))#transform=ax1.transAxes)
# plt.savefig('../results/Fig01_ANN_{}.png'.format(condition),dpi=300)   
plt.savefig('../results/Fig01_ANN_{}.pdf'.format(condition))   