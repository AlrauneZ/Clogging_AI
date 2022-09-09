#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

conditions = ['fav','unfav']
params = ['CN','SC','HC','VF']
iterations = [500,1000,2000]
hyper = 'mss' # 'D' #
data_name = '../results/SI/DT_Training_{}_{}_It{}.csv'#.format(cond,param,n_it)
data_namef = '../results/SI/DT_Training_{}_{}_full.csv'#.format(cond,param,n_it)

### Plot Specifications
plt.close('all')
textsize = 8
title_text = ['Coordination number','Surface coverage','Conductivity','Void fraction']

### identified hyperparameters during Training
max_depth = [[2,8,9,8],[2,3,4,4]]
min_samples_split = [[5,6,2,7],[5,5,4,3]]
# max_depth = [[2,8,9,8],[2,7,5,4]]
# min_samples_split = [[5,6,2,7],[3,2,8,4]]

max_depth[0][2]
###############################################################################
### Plot
###############################################################################
fig = plt.figure(figsize = [7.5,3])

for ic, cond in enumerate(conditions):
    for ip,param in enumerate(params):

        ax = fig.add_subplot(2,4, ic*4 + ip + 1)
        
        for it,n_it in enumerate(iterations):

            data_load = np.loadtxt(data_name.format(cond,param,n_it),delimiter = ',')

            if hyper == 'D':
                # ###range: max_depth
                x = data_load[1:,0] #range(2,20)
                score = data_load[1:,min_samples_split[ic][ip]-1]
                
            elif hyper == 'mss':
                ### range: min_samples_split
                x = data_load[0,1:] #range(2,10)
                score = data_load[max_depth[ic][ip]-1,1:]
                
            ax.plot(x,score,marker = 'o',ls = '-',lw=1,markersize = 4)
          
        data_load = np.loadtxt(data_namef.format(cond,param),delimiter = ',')
       
        if hyper == 'D':
            # ###range: max_depth
            x = data_load[1:,0] #range(2,20)
            score = data_load[1:,min_samples_split[ic][ip]-1]
            hyp2, hyp2_val = 'mss',min_samples_split[ic][ip]
        elif hyper == 'mss':
            ### range: min_samples_split
            x = data_load[0,1:] #range(2,10)
            score = data_load[max_depth[ic][ip]-1,1:]
            hyp2,hyp2_val = 'D',max_depth[ic][ip]

        ax.plot(x,score,marker = 'o',ls = '-',lw=1,markersize = 4)
        ax.grid(True)
        ax.tick_params(axis="both",which="major",labelsize=textsize-1)
    
        if ic == 0:
            ax.set_title(title_text[ip],fontsize=textsize)
        elif ic == 1:
            ax.set_xlabel('{}'.format(hyper),fontsize=textsize)
        if ip == 0:        
            ax.set_ylabel(r'$R^2$',fontsize=textsize)
        
        ax.text(0.05,0.6,'{} = {}'.format(hyp2,hyp2_val),fontsize=textsize, bbox=dict(facecolor='w', alpha=0.5,boxstyle='round'),transform=ax.transAxes)

fig.text(0.01,0.54,'fav. cond.',fontsize=textsize, bbox=dict(facecolor='w', alpha=0.5,boxstyle='round'))#,transform=ax.transAxes)
fig.text(0.01,0.04,'unfav. cond.',fontsize=textsize, bbox=dict(facecolor='w', alpha=0.5,boxstyle='round'))#,transform=ax.transAxes)


plt.tight_layout()
fig.legend(iterations+['full'], ncol = 7,bbox_to_anchor=[0.75, 0.10],fontsize=textsize)#,&nbsp)
fig.subplots_adjust(bottom=0.2)# Adjusting the sub-plots

# plt.savefig('../results/Fig01_ANN_Hyper.png',dpi=300)   
# plt.savefig('../results/Fig02_DT_Hyper_mms.pdf')   
plt.savefig('../results/Fig02_DT_Hyper_{}.pdf'.format(hyper))   
