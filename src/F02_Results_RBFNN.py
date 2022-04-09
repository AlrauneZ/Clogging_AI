#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

conditions = ['fav','unfav']
params = ['CN','SC','HC','VF']

### Plot Specifications
plt.close('all')
textsize = 8
title_text = ['Coordination number','Surface coverage','Conductivity','Void fraction']

ylabel_text = ['MSE','MAE','$R^2$']
val_hyper = ['0.1','0.5','1','2','3','4','5' ]
i_hyper = np.array([0,3,6,12,18]) #np.array([0,3,6,12,18,24,30])

###############################################################################
### Plot
###############################################################################
fig = plt.figure(figsize = [7.5,3])

for ic, cond in enumerate(conditions):
    for ip,param in enumerate(params):
        data1 = np.loadtxt('../data/RBFNN_{}_{}.csv'.format(cond,param),delimiter = ',')
        x = np.arange(1,data1.shape[0]+1)
        ax = fig.add_subplot(2,4, 4*ic + ip + 1)

        score = data1[:,i_hyper+2]
        ax.plot(x,score,marker = 'o',ls = ':',lw=0.5,markersize = 2)

        ax.set_xlim([0.5,data1.shape[0]+0.5])
        ax.set_xticks(np.arange(1,data1.shape[0]+1,4))
        ax.grid(True)
        ax.tick_params(axis="both",which="major",labelsize=textsize-1)

        if ic == 0:
            ax.set_title(title_text[ip],fontsize=textsize)
            R2_max = [[-10,1],[-20,1],[-40,1],[-10,1]]
#            R2_max = [[-65,0],[-1000,5],[-210,5],[-1000,5]]
            if ip == 0:        
                ax.set_ylabel('$R^2$ - fav. cond.',fontsize=textsize)
        elif ic == 1:
            ax.set_xlabel('Number of iterations',fontsize=textsize)
            R2_max = [[-2,1],[-10,1],[-5,1],[-10,1]]
            if ip == 0:        
                ax.set_ylabel('$R^2$ - unfav. cond.',fontsize=textsize)

        ax.set_ylim(R2_max[ip])

plt.tight_layout()
fig.legend(val_hyper[0:len(i_hyper)], ncol = len(i_hyper),bbox_to_anchor=[0.75, 0.09],fontsize=textsize)#,&nbsp)
fig.subplots_adjust(bottom=0.2)# Adjusting the sub-plots

# plt.savefig('../results/Fig02_RBFNN_Hyper.png',dpi=300)   
plt.savefig('../results/Fig02_RBFNN_Hyper.pdf')   
