#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

#condition = 'unfav' #'fav' # 
conditions = ['fav','unfav']
params = ['CN','SC','HC','VF']

file_results = '../data/Results_Testing_{}.csv'#.format(condition)

title_text = ['Coordination number','Surface coverage','Conductivity','Void fraction']
method_names =['ANN','DT','RF','LR','SVR']
colors=['C3','C0','C2','C1','C5']
y_pos = np.arange(len(method_names))


###############################################################################
### Plot
###############################################################################
### Plot Specifications
plt.close('all')
textsize = 8

fig = plt.figure(figsize = [7.5,3.5])
for ic,cond in enumerate(conditions):
    ### Load Results
    results = np.loadtxt(file_results.format(cond),delimiter = ',',skiprows=5,usecols=y_pos+1)

    for ip,param in enumerate(params):
        ax = fig.add_subplot(len(conditions),len(params), len(params)*ic + ip + 1)

        # ### keeping the order of methods
        # r2_score = results[ip,:]
        # names = method_names

        ### sorting the methods results according performance
        isort =  np.argsort(results[ip,:])
        r2_score = results[ip,isort]
        names = np.array(method_names)[isort]
        colors_sort = np.array(colors)[isort]
        
        ax.barh(y_pos,r2_score,color = colors_sort)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names,fontsize=textsize)
        ax.invert_yaxis()  # labels read top-to-bottom
        
        ax.tick_params(axis="both",which="major",labelsize=textsize-1)
        ax.set_xlabel('MSE',fontsize=textsize)

        if ic == 0:
            ax.set_title(title_text[ip],fontsize=textsize)
plt.tight_layout()
fig.text(0.02,0.5,'favorable',fontsize=textsize, bbox=dict(facecolor='w', alpha=0.2,boxstyle='round'))#transform=ax1.transAxes)
fig.text(0.02,0.025,'unfavorable',fontsize=textsize, bbox=dict(facecolor='w', alpha=0.2,boxstyle='round'))#transform=ax1.transAxes)

plt.savefig('../results/FigS08_Results_Testing_MSE.pdf')   
