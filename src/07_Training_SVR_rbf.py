import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler

###############################################################################
### Load data and set algorithm parameters
###############################################################################

iterations = 500 #
condition = 'unfav' #'fav' #

### hypyerparameter testing range
range_C = [0.001, 0.01, 0.1, 1, 10, 100]
range_gamma = [0.001, 0.01, 0.1, 1, 10, 100]
n_splits = 10

### data specifics
name_output = ['Coordination number','Surface coverage','Conductivity','Void fraction']
name_output_short = ['CN','SC','HC','VF']
n_training = 73

## Load simulation results from Lattice Boltzman Method (physical transport simualation) 
file_LBM = "../data/LBM_Results.xlsx"
xl = pd.ExcelFile(file_LBM)
data_LBM = pd.read_excel(xl,skiprows = [1],sheet_name=condition)
data_training = data_LBM[0:n_training]
mm = MinMaxScaler()
mm_data = mm.fit_transform(np.array(data_training))
input_data = mm_data[:,0:4]
output_data = mm_data[:,4:8]

np.set_printoptions(suppress = True)
np.random.seed(12345678+iterations)
full = True #False #

##############################################################################
## Run Training Procedure
##############################################################################

print("\nTraining procedure for support vector regression algorithm with the rbf kernel under {}orable conditions, {} iterations".format(condition,iterations))

for io,no in enumerate(name_output):
    print("\nTraining for output parameter: {}".format(no))

    ###############################################################################
    ### Run Training Procedure with Cross Validation
    ###############################################################################

    results = np.zeros((len(range_C)+1,len(range_gamma)+1))
    results[0,1:] = range_gamma
    results[1:,0] = range_C
    
    for ic,C in enumerate(range_C):
        for ig,gamma in enumerate(range_gamma):
            svr_rbf = SVR(kernel = 'rbf', gamma = gamma, C = C)
            rkf = RepeatedKFold(n_splits = n_splits, n_repeats = iterations)
            scores = cross_val_score(svr_rbf, input_data, output_data[:,io], cv = rkf)
            score = np.mean(scores)
            print("For C {} and gamma {}, the average score is {:.3f}".format(C, gamma,score))
            results[ic+1,ig+1] = score

    ###############################################################################
    ### Save Scores
    ###############################################################################

    # ind = np.unravel_index(np.argmax(results, axis=None), results.shape)
    # print('Maximum score {:.2f} for C = {} and gamma = {}'.format(results[ind],range_C[ind[0]],range_gamma[ind[1]]))
    ind = np.unravel_index(np.argmax(results[1:,1:], axis=None), results[1:,1:].shape)
    print('\nMaximum score {:.2f} for C = {} and gamma = {}\n'.format(results[ind[0]+1,ind[1]+1],range_C[ind[0]],range_gamma[ind[1]]))
    np.savetxt('../results/SVR_lin_Training_{}_{}_It{}.csv'.format(condition,name_output_short[io],iterations),results,fmt = '%.4f',delimiter = ',')

if full:
    ###############################################################################
    ### Run Training Procedure with full training Data set
    ###############################################################################
    for io,no in enumerate(name_output):
   
        results_all = np.zeros((len(range_C)+1,len(range_gamma)+1))
        results_all[0,1:] = range_gamma
        results_all[1:,0] = range_C
    
        for ic,C in enumerate(range_C):
            for ig,gamma in enumerate(range_gamma):
                
                svr_rbf = SVR(kernel = 'rbf', gamma = gamma, C = C)
                svr_rbf.fit(input_data,output_data[:,io])
                r2_training = svr_rbf.score(input_data, output_data[:,io])
                results_all[ic+1,ig+1] =  r2_training
          
        ind = np.unravel_index(np.argmax(results_all[1:,1:], axis=None), results_all[1:,1:].shape)
        print('\nMaximum score {:.2f} for C = {} and gamma = {}\n'.format(results_all[ind[0]+1,ind[1]+1],range_C[ind[0]],range_gamma[ind[1]]))
        np.savetxt('../results/SVR_lin_Training_{}_{}_full.csv'.format(condition,name_output_short[io]),results_all,fmt = '%.4f',delimiter = ',')

print("\nSVR_rbf Training finished.")
