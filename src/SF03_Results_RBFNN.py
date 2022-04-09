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
val_hyper = ['0.1','0.5','1','2','3','4','5' ]
if condition == 'fav':
    i_hyper = np.array([0,3,6]) #np.array([0,3,6,12,18,24,30])
    MSE_max = [[0,21],[0,8],[0,1],[0,2.2]]
    MAE_max = [[0.5,2],[0,1.5],[0.25,0.45],[0,0.6]]
    R2_max = [[-65,0],[-1000,5],[-210,5],[-1000,5]]
elif condition == 'unfav':
    i_hyper = np.array([0,3,6,12,18]) #np.array([0,3,6,12,18,24,30])
    MSE_max = [[0.3,0.7],[0,0.21],[0.07,0.32],[0.002,0.02]]
    MAE_max = [[0.3,1.2],[0.08,0.3],[0.2,0.38],[0.04,0.13]]
    R2_max = [[-2,0],[-50,0],[-6,0],[-10,0]]

###############################################################################
### Plot
###############################################################################
fig = plt.figure(figsize = [7.5,5])
for ip,param in enumerate(params):
    data1 = np.loadtxt('../data/RBFNN_{}_{}.csv'.format(condition,param),delimiter = ',')
    x = np.arange(1,data1.shape[0]+1)

    for i in range(3):
        ax = fig.add_subplot(3,4, 4*i + ip + 1)
        score = data1[:,i_hyper+i]
        ax.plot(x,score,marker = 'o',ls = ':',lw=0.5,markersize = 2)
        ax.set_xlim([0.5,data1.shape[0]+0.5])

        ax.set_xticks(np.arange(1,data1.shape[0]+1,4))
        ax.grid(True)
        ax.tick_params(axis="both",which="major",labelsize=textsize-1)

        if i == 0:
            ax.set_title(title_text[ip],fontsize=textsize)
            ax.set_ylim(MSE_max[ip])
        elif i == 1:
            ax.set_ylim(MAE_max[ip])
        elif i == 2:
            ax.set_ylim(R2_max[ip])
            ax.set_xlabel('Number of iterations',fontsize=textsize)
        if ip == 0:        
            ax.set_ylabel(ylabel_text[i],fontsize=textsize)

plt.tight_layout()
fig.legend(val_hyper[0:len(i_hyper)], ncol = len(i_hyper),bbox_to_anchor=[0.85, 0.06],fontsize=textsize)#,&nbsp)
fig.subplots_adjust(bottom=0.13)# Adjusting the sub-plots
fig.text(0.035,0.025,'RBFNN, {}orable conditions'.format(condition),fontsize=textsize, bbox=dict(facecolor='w', alpha=0.2,boxstyle='round'))#transform=ax1.transAxes)

# plt.savefig('../results/Fig_RBFNN_{}.png'.format(condition),dpi=300)   
plt.savefig('../results/Fig_RBFNN_{}.pdf'.format(condition))   
