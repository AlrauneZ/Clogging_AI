#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


### Plot Specifications
plt.close('all')
textsize = 8

### Import the dataset
file_data = '../data/{}_{}_clogging.csv'#.format(alg,cond)

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


# ax2 = fig.add_subplot(1,2,2)
# for inn, n_it in enumerate(data_unfav[1::2,0]):
#     ax2.plot(data_unfav[0,1:],data_unfav[inn+1,1:],marker = 'o',ls = '-',lw=1,markersize = 4)
# ax2.grid(True)
# ax2.tick_params(axis="both",which="major",labelsize=textsize)
# ax2.set_xlabel('Number of Neurons',fontsize=textsize)
# ax2.set_ylabel('$R^2$ - unfav',fontsize=textsize)
# ax2.legend(data_unfav[1::2,0],loc = 'lower left',fontsize=textsize,ncol = 1)
# ax2.set_ylim([0.91,1.002])
# ax2.text(0.54,0.04,'unfavorable conditions',fontsize=textsize, bbox=dict(facecolor='w', alpha=0.5,boxstyle='round'),transform=ax2.transAxes)

plt.tight_layout()
plt.savefig('../results/Fig04_Clogging_all.pdf')   

# ### Plot
# fig = plt.figure(figsize = [7.5,2.8])
# ax1 = fig.add_subplot(1,2,1)
# for inn, n_it in enumerate(data_fav[1:10:2,0]):
#     ax1.plot(data_fav[0,1:],data_fav[inn+1,1:],marker = 'o',ls = '-',lw=1,markersize = 4)

# ax1.grid(True)
# ax1.tick_params(axis="both",which="major",labelsize=textsize)
# ax1.set_xlabel('Number of Neurons',fontsize=textsize)
# ax1.set_ylabel('$R^2$ - fav',fontsize=textsize)
# ax1.set_ylim([0.91,1.002])
# ax1.legend(data_fav[1::2,0],loc = 'upper left',fontsize=textsize,ncol = 1)
# ax1.text(0.05,0.04,'favorable conditions',fontsize=textsize, bbox=dict(facecolor='w', alpha=0.5,boxstyle='round'),transform=ax1.transAxes)


# ax2 = fig.add_subplot(1,2,2)
# for inn, n_it in enumerate(data_unfav[1::2,0]):
#     ax2.plot(data_unfav[0,1:],data_unfav[inn+1,1:],marker = 'o',ls = '-',lw=1,markersize = 4)
# ax2.grid(True)
# ax2.tick_params(axis="both",which="major",labelsize=textsize)
# ax2.set_xlabel('Number of Neurons',fontsize=textsize)
# ax2.set_ylabel('$R^2$ - unfav',fontsize=textsize)
# ax2.legend(data_unfav[1::2,0],loc = 'lower left',fontsize=textsize,ncol = 1)
# ax2.set_ylim([0.91,1.002])
# ax2.text(0.54,0.04,'unfavorable conditions',fontsize=textsize, bbox=dict(facecolor='w', alpha=0.5,boxstyle='round'),transform=ax2.transAxes)

# plt.tight_layout()
# plt.savefig('../results/Fig04_Clogging_new.pdf')   


# ### Plot
# fig = plt.figure(figsize = [7.5,2.8])
# ax1 = fig.add_subplot(1,2,1)
# for inn, neurons in enumerate(data_fav[0,1:]):
#     ax1.plot(data_fav[1:,0],data_fav[1:,inn+1],marker = 'o',ls = '-',lw=1,markersize = 4)

# ax1.grid(True)
# ax1.tick_params(axis="both",which="major",labelsize=textsize)
# ax1.set_xlabel('Number of iterations',fontsize=textsize)
# ax1.set_ylabel('$R^2$ - fav',fontsize=textsize)
# ax1.set_ylim([0.91,1.002])
# ax1.legend(data_fav[0,1:],loc = 'upper right',fontsize=textsize)#,&nbsp) #,ncol = 6
# ax1.text(0.6,0.04,'favorable conditions',fontsize=textsize, bbox=dict(facecolor='w', alpha=0.5,boxstyle='round'),transform=ax1.transAxes)


# ax2 = fig.add_subplot(1,2,2)
# for inn, neurons in enumerate(data_unfav[0,1:]):
#     ax2.plot(data_unfav[1:,0],data_unfav[1:,inn+1],marker = 'o',ls = '-',lw=1,markersize = 4)
# ax2.grid(True)
# ax2.tick_params(axis="both",which="major",labelsize=textsize)
# ax2.set_xlabel('Number of iterations',fontsize=textsize)
# ax2.set_ylabel('$R^2$ - unfav',fontsize=textsize)
# #ax2.legend(['2','3','4','5','6' ],loc = 'lower right',fontsize=textsize)#,&nbsp) #,ncol = 6
# ax2.set_ylim([0.91,1.002])
# ax2.text(0.55,0.04,'unfavorable conditions',fontsize=textsize, bbox=dict(facecolor='w', alpha=0.5,boxstyle='round'),transform=ax2.transAxes)

# plt.tight_layout()
# plt.savefig('../results/Fig04_Clogging.pdf')   


