#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

condition = 'unfav' #'fav' # 
params = ['CN','SC','HC','VF']

### Plot Specifications
plt.close('all')
textsize = 8
title_text = ['Coordination number','Surface coverage','Conductivity','Void fraction']
score_text = ['MSE','MAE','$R^2$']
method_names =['DT','RF','ANN','RBFNN','LR O','LR Ridge','LR Lasso','SVR lin','SVR poly','SRV rbf']
y_pos = np.arange(len(method_names))
# method_names =['DT','RF','RBFNN','ANN','Srbf','Slin','Lasso','Ridge','OLS','Spoly']

### Results
results = np.loadtxt('../data/Results_Testing_{}.csv'.format(condition),delimiter = ',',skiprows=1,usecols=y_pos+1)

###############################################################################
### Plot
###############################################################################
fig = plt.figure(figsize = [7.5,5])
for ip,param in enumerate(params):
    for i in range(3):
        ax = fig.add_subplot(3,4, 4*i + ip + 1)

        ### keeping the order of methods
        score = results[3*ip + i,:]
        names = method_names

        ### sorting the methods results according performance
        isort =  np.argsort(results[3*ip + i,:])
        score = results[3*ip + i,isort]
        names = np.array(method_names)[isort]

        ax.barh(y_pos,score,color = 'C{}'.format(ip)) #        ax.barh(y_pos,score) 
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names,fontsize=textsize)
        
        if i==2:
        # if i!=2:
            ax.invert_yaxis()  # labels read top-to-bottom

        ax.tick_params(axis="both",which="major",labelsize=textsize-1)
        ax.set_xlabel(score_text[i],fontsize=textsize)

        # ax.set_xlim([0.5,data1.shape[0]+0.5])
        # ax.set_xticks(np.arange(1,data1.shape[0]+1,4))
        # ax.grid(True)

        if i == 0:
            ax.set_title(title_text[ip],fontsize=textsize)
            # ax.set_ylim(MSE_max[ip])
        # elif i == 2:
            # ax.set_ylim(R2_max[ip])
 
        # if ip == 0:        
        #     ax.set_ylabel(ylabel_text[i],fontsize=textsize)

plt.tight_layout()
fig.text(0.035,0.025,'{}orable conditions'.format(condition),fontsize=textsize, bbox=dict(facecolor='w', alpha=0.2,boxstyle='round'))#transform=ax1.transAxes)

# plt.savefig('../results/Fig01_ANN_{}.png'.format(condition),dpi=300)   
# plt.savefig('../results/Fig03_Results_Testing_{}.pdf'.format(condition))   
