#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

### Plot Specifications
plt.close('all')
textsize = 9
frac = 0.993

### Import the dataset
file_data = '../results/SI/Clogging_Training_{}_{}.csv'#.format(alg,cond)

### Plot
fig = plt.figure(figsize = [7.5,3.5])
for ia, alg in enumerate(['ANN','DT']):
# for ia, alg in enumerate(['ANN']):
    for ic, cond in enumerate(['fav','unfav']):

        data = np.loadtxt(file_data.format(alg,cond),delimiter = ',')#,skiprows=1)
        ax = fig.add_subplot(2,2,2*ic +ia + 1)

        if alg == 'ANN':
            iterations = data[0,1:]
            ax.set_ylabel('$R^2$ ',fontsize=textsize)
            
            ax.set_xlim([1.8,20.2])
            ax.set_xticks(data[1:20:2,0])
        elif alg == 'DT':
            iterations = data[0,1:]
            ax.set_xlim([1.9,10.1])
            ax.set_xticks(data[1::2,0])
        # ax.set_ylim([0.89,1.01])
        ax.set_ylim([0.9,1.0])
        ax.grid(True)
        ax.tick_params(axis="both",which="major",labelsize=textsize)

        for inn, n_it in enumerate(iterations):
            ax.plot(data[1:,0],data[1:,inn+1],marker = 'o',ls = '-',lw=1,markersize = 4,label = '{:.0f}'.format(n_it))

        mean = np.mean(data[1:,1:],axis = 1)
        # ax.plot(data[1:,0],mean,marker = 'o',ls = '-',lw=1,markersize = 4,label = '{:.0f}'.format(n_it+1))
        rmax = np.max(mean)
        irmax = np.argmax(mean)+2
        print("\n ###################################", alg)
        print("mean iterations: ")
        print("R2 max = {} at i = {}".format(rmax,irmax))
        irmax_rel = np.array(np.where(mean-frac*rmax>0))[0,0]
        rmax_rel = mean[irmax_rel]
        print(" {:.0f}% R2 max = {} at i = {}".format(100*frac,rmax_rel,irmax_rel+2))

        if ic ==0:
            ax.set_title('{}'.format(alg),fontsize=textsize)
        elif ic == 1: 
            if alg == 'ANN': 
                ax.set_xlabel('Number of neurons',fontsize=textsize)
                ax.legend(loc = 'lower center',fontsize=textsize,ncol = 4)
            else:
                ax.set_xlabel('Maximum depth',fontsize=textsize)
                ax.legend(loc = 'center right',fontsize=textsize,ncol = 2)


fig.text(0.01,0.52,'fav. cond.',fontsize=textsize, bbox=dict(facecolor='w', alpha=0.5,boxstyle='round'))#,transform=ax.transAxes)
fig.text(0.01,0.05,'unfav. cond.',fontsize=textsize, bbox=dict(facecolor='w', alpha=0.5,boxstyle='round'))#,transform=ax.transAxes)

plt.tight_layout()
plt.savefig('../results/FigS09_Hyper_Clogging.pdf')   

