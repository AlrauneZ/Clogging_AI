import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold, cross_val_score

###############################################################################
### Load data and set algorithm parameters
###############################################################################

iterations = 400
condition =  'unfav' #'fav' #

### hypyerparameter testing range
range_n_estimators = np.arange(100,400,100)
max_features = 4
random_state = 42
n_splits = 10

### data specifics
n_training = 73
name_output = ['Coordination number','Surface coverage','Conductivity','Void fraction']
name_output_short = ['CN','SC','HC','VF']

### Load simulation results from Lattice Boltzman Method (physical transport simualation) 
file_LBM = "../data/LBM_Results.xlsx"
xl = pd.ExcelFile(file_LBM)
data_LBM = pd.read_excel(xl,skiprows = [1],sheet_name=condition)
data_training = data_LBM[0:n_training]
input_data = np.array(data_training)[:,0:4]
output_data = np.array(data_training)[:,4:8]

### local settings
np.set_printoptions(suppress = True)
np.random.seed(12345678+iterations)
full = False

###############################################################################
### Run Training Procedure
###############################################################################
results_CV = np.zeros((len(name_output)+1,len(range_n_estimators)))
results_CV[0,:] = range_n_estimators 
results_all = np.zeros((len(name_output)+1,len(range_n_estimators)))
results_all[0,:] = range_n_estimators 

print("\nTraining procedure for random forest tree algorithm \n under {}orable conditions \n with {} Iterations".format(condition,iterations))
for io,no in enumerate(name_output):
    results = np.zeros((len(range_n_estimators)))
    print("\nTraining for output parameter: {}".format(no))
    for ine, n_estimators  in enumerate(range_n_estimators):
       
        forest = RandomForestRegressor(n_estimators = n_estimators, max_features = max_features, random_state = random_state)
        rkf = RepeatedKFold(n_splits = n_splits, n_repeats = iterations)
        scores = cross_val_score(forest, input_data, output_data[:,io], cv = rkf)
        score = np.mean(scores)
        print("For n_estimator {}, the average score is {:.3f}".format(n_estimators,score))
        results[ine] = score

    ###############################################################################
    ### Save Scores
    ###############################################################################

    ind = np.unravel_index(np.argmax(results, axis=None), results.shape)
    print('Maximum score {:.2f} for n_estimator = {}'.format(results[ind],range_n_estimators[ind[0]]))
    results_CV[io+1,:] = results

print('Save Cross Validation Training Results to file. \n')
np.savetxt('../results/RF_Training_{}_It{}.csv'.format(condition,iterations),results_CV.T,fmt = '%.4f',delimiter = ',',header=" , ".join(['n_estimator'] + name_output_short))

###############################################################################
### Run Training Procedure with full training Data set
###############################################################################
if full:
    print("\nTraining for Random Forst algorithm \n under {}orable conditions".format(condition))
    for io,no in enumerate(name_output):
        print("\nOutput Parameter: {}".format(name_output[io]))
        for ine, n_estimators  in enumerate(range_n_estimators):
            print('\nSelected Hyperparameters: \n n_estimators  =  {}'.format(n_estimators))
            forest = RandomForestRegressor(n_estimators = n_estimators, max_features = max_features, random_state = random_state)
    
            forest.fit(input_data,output_data[:,io])
            r2_training = forest.score(input_data, output_data[:,io])
            print("\nTraining set score (R2): {:.4f}".format(r2_training))
            results_all[io+1,ine] =  r2_training
            
    print('Save Training full set results to file. \n')
    np.savetxt('../results/RF_Training_{}_full.csv'.format(condition),results_all.T,fmt = '%.4f',delimiter = ',',header=" , ".join(['n_estimator'] + name_output_short))

print("RF Training finished.")

