#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

conditions = ['fav','unfav']
params = ['CN','SC','HC','VF']
iterations = [50,100,200]

### Plot Specifications
plt.close('all')
textsize = 9
title_text = ['Coordination number','Surface coverage','Conductivity','Void fraction']

name_file  = '../results/ANN/ANN_L_{}_{}_It{}.csv'

###############################################################################
### Plot
###############################################################################

fig = plt.figure(figsize = [7.5,3])
for ic, cond in enumerate(conditions):
    print('#################\n Condition {} \n#################\n'.format(cond))

    for ip,param in enumerate(params):
        print('\nParameter {} \n###############\n'.format(param))
             
        ax = fig.add_subplot(len(conditions),len(params), ic*len(params) + ip + 1)
            
        for it,n_it in enumerate(iterations):
            print('######################\n Iteration number {} '.format(n_it))
           
            data = np.loadtxt(name_file.format(param,cond,n_it),delimiter = ',')
            ax.plot(data[:,0],data[:,1],ls = '-',lw=2)#,marker = 'o',markersize = 4)

            rmax = np.max(data[:,1])
            irmax = np.argmax(data[:,1])
            print(" R2 max for {} = {} (i = {})".format(param,rmax,irmax))

 
        ax.grid(True)
        ax.tick_params(axis="both",which="major",labelsize=textsize-1)   
        ax.set_xlim([0,data[-1,0]+1])      
        if ic == 0:
            ax.set_title(title_text[ip],fontsize=textsize)
            if ip == 0:        
                ax.set_ylabel('NSE',fontsize=textsize)
                ax.set_ylim([0,1])
            elif ip == 1 :
                ax.set_ylim([-3,-1])
            elif ip ==2:        
                ax.set_ylim([-3,1])
            elif ip == 3:        
                ax.set_ylim([-2,1])

        elif ic == 1:
            ax.set_xlabel('Number of Neurons',fontsize=textsize)
            ax.set_ylim([0,1])
            if ip == 0:        
                ax.set_ylabel('NSE',fontsize=textsize)

fig.text(0.01,0.54,'fav. cond.',fontsize=textsize, bbox=dict(facecolor='w', alpha=0.5,boxstyle='round'))#,transform=ax.transAxes)
fig.text(0.01,0.04,'unfav. cond.',fontsize=textsize, bbox=dict(facecolor='w', alpha=0.5,boxstyle='round'))#,transform=ax.transAxes)

plt.tight_layout()
fig.legend(iterations, ncol = 3,bbox_to_anchor=[0.70, 0.105],fontsize=textsize)#,&nbsp)
fig.subplots_adjust(bottom=0.24)# Adjusting the sub-plots
plt.savefig('../results/Fig01_ANN_Hyper.pdf')   

