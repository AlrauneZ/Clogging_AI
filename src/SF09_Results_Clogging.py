#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

### Plot Specifications
plt.close('all')
textsize = 9

### Import the dataset
file_data = '../data/SI/{}_{}_clogging.csv'#.format(alg,cond)

### Plot
fig = plt.figure(figsize = [7.5,3.5])
for ic, cond in enumerate(['fav','unfav']):
    for ia, alg in enumerate(['ANN','DT']):
        data = np.loadtxt(file_data.format(alg,cond),delimiter = ',')#,skiprows=1)
        ax = fig.add_subplot(2,2,2*ic +ia + 1)

        if ia == 0:
            iterations = data[1::3,0]
            ax.set_ylabel('$R^2$ ',fontsize=textsize)
            # ax.set_ylim([0.91,1.005])
            ax.set_xticks(data[0,1:])
        elif ia == 1:
            iterations = data[1:,0]
            # ax.set_ylim([0.77,1.01])
            ax.set_xticks(data[0,1::2])
        ax.set_ylim([0.77,1.01])
        ax.grid(True)
        ax.tick_params(axis="both",which="major",labelsize=textsize)

        for inn, n_it in enumerate(iterations):
            ax.plot(data[0,1:],data[inn+1,1:],marker = 'o',ls = '-',lw=1,markersize = 4,label = '{:.0f}'.format(n_it))

        if ic ==0:
            ax.set_title('{}'.format(alg),fontsize=textsize)#, bbox=dict(facecolor='w', alpha=0.5,boxstyle='round'),transform=ax.transAxes) 
            # ax.text(0.07,0.84,'{}'.format(alg),fontsize=textsize, bbox=dict(facecolor='w', alpha=0.5,boxstyle='round'),transform=ax.transAxes) 
        elif ic == 1: 
            if alg == 'ANN': 
                ax.set_xlabel('Number of Neurons',fontsize=textsize)
            else:
                ax.set_xlabel('Maximum depth',fontsize=textsize)
            ax.legend(loc = 'lower right',fontsize=textsize,ncol = 1)


fig.text(0.01,0.54,'fav. cond.',fontsize=textsize, bbox=dict(facecolor='w', alpha=0.5,boxstyle='round'))#,transform=ax.transAxes)
fig.text(0.01,0.04,'unfav. cond.',fontsize=textsize, bbox=dict(facecolor='w', alpha=0.5,boxstyle='round'))#,transform=ax.transAxes)

plt.tight_layout()
plt.savefig('../results/FigS09_Hyper_Clogging.pdf')   

