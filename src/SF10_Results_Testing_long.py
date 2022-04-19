#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

condition =  'unfav' #'fav' #
params = ['CN','SC','HC','VF']

### Plot Specifications
file_results = '../data/Results_Testing_{}.csv'#.format(condition)

title_text = ['Coordination number','Surface coverage','Conductivity','Void fraction']
score_text = ['$R^2$','MSE','MAE']
method_names =['ANN','RBFNN','DT','RF','LR','SVR']
colors=['C3','C1','C0','C2','C4','C5']
y_pos = np.arange(len(method_names))


### Results
results = np.loadtxt(file_results.format(condition),delimiter = ',',skiprows=1,usecols=y_pos+1)

###############################################################################
### Plot
###############################################################################
plt.close('all')
textsize = 9
fig = plt.figure(figsize = [7.5,5])
for ip,param in enumerate(params):
    for isc,score_name in enumerate(score_text):
        ax = fig.add_subplot(len(score_text),len(params),len(params)*isc + ip + 1)

        # ### keeping the order of methods
        # score = results[isc*len(params) + ip,:]
        # names = method_names

        ### sorting the methods results according performance
        isort =  np.argsort(results[isc*len(params) + ip,:])
        score = results[isc*len(params) + ip,isort]
        names = np.array(method_names)[isort]
        colors_sort = np.array(colors)[isort]

        ax.barh(y_pos,score,color = colors_sort) #        ax.barh(y_pos,score) 
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names,fontsize=textsize)
        
        if isc!=0:
            ax.invert_yaxis()  # labels read top-to-bottom

        ax.tick_params(axis="both",which="major",labelsize=textsize-1)
        ax.set_xlabel(score_name,fontsize=textsize)

        if isc == 0:
            ax.set_title(title_text[ip],fontsize=textsize)

plt.tight_layout()
fig.text(0.02,0.025,'{}orable'.format(condition),fontsize=textsize, bbox=dict(facecolor='w', alpha=0.2,boxstyle='round'))#transform=ax1.transAxes)

# plt.savefig('../results/Fig01_ANN_{}.png'.format(condition),dpi=300)   
plt.savefig('../results/FigS08_Results_Testing_{}.pdf'.format(condition))   
