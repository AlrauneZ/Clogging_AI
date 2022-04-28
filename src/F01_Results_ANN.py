#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

conditions = ['fav','unfav']
params = ['CN','SC','HC','VF']
n_it = 25
step_it = 5
file_data = '../data/ANN_{}_{}_R2.csv'#.format(cond,param)

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

        data = np.loadtxt(file_data.format(cond,param),delimiter = ',').T
        x = data[1:,0]
        score = data[1:,step_it::step_it]
    
        n_iterations = len(data[0,step_it::step_it])
        
        ax = fig.add_subplot(2,4, ic*4 + ip + 1)

        ax.plot(x,score,marker = 'o',ls = '-',lw=1,markersize = 4)
        ax.grid(True)
        ax.tick_params(axis="both",which="major",labelsize=textsize-1)
    
        if ic == 0:
            ax.set_title(title_text[ip],fontsize=textsize)
            R2_max = [[-0.2,0.6],[-5,0],[-2.5,0],[-2.5,0]]
        elif ic == 1:
            ax.set_xlabel('Number of Neurons',fontsize=textsize)
            R2_max = [[-3,0.6],[-3,0],[0.4,1],[-2,1]]
        if ip == 0:        
            ax.set_ylabel(r'$R^2$',fontsize=textsize)

fig.text(0.01,0.54,'fav. cond.',fontsize=textsize, bbox=dict(facecolor='w', alpha=0.5,boxstyle='round'))#,transform=ax.transAxes)
fig.text(0.01,0.04,'unfav. cond.',fontsize=textsize, bbox=dict(facecolor='w', alpha=0.5,boxstyle='round'))#,transform=ax.transAxes)


plt.tight_layout()
fig.legend(np.array(data[0,step_it::step_it],dtype=int), ncol = n_iterations,bbox_to_anchor=[0.75, 0.09],fontsize=textsize)
fig.subplots_adjust(bottom=0.2)# Adjusting the sub-plots

# plt.savefig('../results/Fig01_ANN_Hyper.png',dpi=300)   
plt.savefig('../results/Fig01_ANN_Hyper.pdf')   

