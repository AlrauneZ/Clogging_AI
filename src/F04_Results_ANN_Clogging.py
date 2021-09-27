#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

condition = 'fav' #'fav' # 
params = ['CN','SC','HC','VF']

### Plot Specifications
plt.close('all')
textsize = 8

### Import the dataset
data1 = np.loadtxt('../data/ANN_fav_clogging.csv',delimiter = ',');
p1 = data1[:,0:5]

data2 = np.loadtxt('../data/ANN_unfav_clogging.csv',delimiter = ',');
p2 = data2[:,0:5]

### Plot
fig = plt.figure(figsize = [7.5,2.8])
ax1 = fig.add_subplot(1,2,1)
ax1.plot(np.arange(1,p1.shape[0]+1),p1,marker = 'o',ls = ':',lw=0.5,markersize = 3)
ax1.grid(True)
ax1.tick_params(axis="both",which="major",labelsize=textsize)
ax1.set_xlabel('Number of iterations',fontsize=textsize)
ax1.set_ylabel('Rate',fontsize=textsize)
ax1.set_xlim([0.5,p1.shape[0]+0.5])
ax1.set_ylim([0.91,1.002])
ax1.legend(['2','3','4','5','6' ],ncol = 1,loc = 'upper right',fontsize=textsize)#,&nbsp) #,ncol = 6
ax1.text(0.55,0.04,'favorable conditions',fontsize=textsize, bbox=dict(facecolor='w', alpha=0.5,boxstyle='round'),transform=ax1.transAxes)

#title('Performance of predicting clogging under the favorable condition')
#set(gca,'FontSize',18,'FontName','Times New Roman');
#legend('2neurons','3neurons','4neurons','5neurons','6neurons','FontSize',18,'FontName','Times New Roman')

ax2 = fig.add_subplot(1,2,2)
ax2.plot(np.arange(1,p2.shape[0]+1),p2,marker = 'o',ls = ':',lw=0.5,markersize = 3)
ax2.grid(True)
ax2.tick_params(axis="both",which="major",labelsize=textsize)
ax2.set_xlabel('Number of iterations',fontsize=textsize)
#ax2.legend(['2','3','4','5','6' ],loc = 'lower right',fontsize=textsize)#,&nbsp) #,ncol = 6
ax2.set_xlim([.5,p2.shape[0]+0.5])
ax2.set_ylim([0.91,1.002])
ax2.text(0.5,0.04,'unfavorable conditions',fontsize=textsize, bbox=dict(facecolor='w', alpha=0.5,boxstyle='round'),transform=ax2.transAxes)

plt.tight_layout()
# plt.savefig('../results/Fig01_ANN_{}.png'.format(condition),dpi=300)   
plt.savefig('../results/Fig04_Clogging.pdf')   

#title('Performance of predicting clogging under the unfavorable condition')

