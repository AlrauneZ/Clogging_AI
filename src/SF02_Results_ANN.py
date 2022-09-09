#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

conditions = ['fav','unfav']
params = ['CN','SC','HC','VF']

### Plot Specifications
plt.close('all')
textsize = 9
title_text = ['Coordination number','Surface coverage','Conductivity','Void fraction']

n_it = 100
name_file  = '../results/ANN/ANN_Lfull_{}_{}_{}.csv'
nn_fav = np.array([[219,57, 194, 146],[184,339,156,156]])

###############################################################################
### Plot
###############################################################################

fig = plt.figure(figsize = [7.5,3])
for ic, cond in enumerate(conditions):
    for ip,param in enumerate(params):
        print("#################### \n Parameter {} - {}".format(param,cond))
        ax = fig.add_subplot(len(conditions),len(params), ic*len(params) + ip + 1)
             
        data = np.loadtxt(name_file.format(param,cond,n_it),delimiter = ',')

        ax.plot(data[:,0],data[:,1],marker = 'o',ls = '-',lw=2,markersize = 4,c = 'C{}'.format(ip))
        ax.fill_between(data[:,0],data[:,1],data[:,1]+data[:,2], alpha=.5, ls = '--',lw=1,color = 'C{}'.format(ip))
        ax.fill_between(data[:,0],data[:,1],data[:,1]-data[:,2], alpha=.5, ls = '--',lw=1,color = 'C{}'.format(ip))


        # if n_it == 200:
        rmax = np.max(data[:,1])
        irmax = np.argmax(data[:,1])
        print("\n R2 max = {} at i = {}".format(rmax,irmax))
        print(" sigma = {:.4f} at i = {}".format(data[irmax,2],irmax))

        sigmin = np.min(data[:,2])                
        isigmin = np.argmin(data[:,2])
        
        print("\n sigma min = {:.4f} at i = {}".format(sigmin,isigmin))
        print(" R2 at sigma min = {:.4f} at i = {}".format(data[isigmin,1],isigmin))

        print("\n R2 at training min = {:.4f} at i = {}".format(data[nn_fav[ic,ip],1],nn_fav[ic,ip]))
        print(" sigma = {:.4f} at i = {}".format(data[nn_fav[ic,ip],2],nn_fav[ic,ip]))
        print(" ")
             
        ax.grid(True)
        ax.tick_params(axis="both",which="major",labelsize=textsize-1)   
        ax.set_ylim([0,1.03])
        
        if ic == 0:
            ax.set_title(title_text[ip],fontsize=textsize)
            if ip == 0:        
                ax.set_ylabel('$R^2$ - fav',fontsize=textsize)
        elif ic == 1:
            ax.set_xlabel('Number of Neurons',fontsize=textsize)
            if ip == 0:        
                ax.set_ylabel('$R^2$ - unfav',fontsize=textsize)
        

plt.tight_layout()
plt.savefig('../results/FigS02_ANN_Hyper_Full.pdf')   

