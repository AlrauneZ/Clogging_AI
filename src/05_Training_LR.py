import numpy as np
import random
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

###############################################################################
### Load data and set algorithm parameters
###############################################################################

iterations = 2000
random.seed(12345678+iterations)

condition = 'fav' #'unfav' #

range_alpha = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

test_sample_length = 0.9
n_splits = 10
name_output = ['Coordination number','Surface coverage','Conductivity','Void fraction']
name_output_short = ['CN','SC','HC','VF']

data_LBM = np.loadtxt("../data/LBM_results_{}.csv".format(condition), delimiter = ',',skiprows = 1)
n_test_samples = int(np.round(test_sample_length*len(data_LBM)))

input_data = data_LBM[:n_test_samples,0:4]
output_data = data_LBM[:n_test_samples,4:8]

input_data_testing = data_LBM[n_test_samples:,0:4]
output_data_testing = data_LBM[n_test_samples:,4:8]


np.set_printoptions(suppress = True)
results = np.zeros((len(range_alpha)))
results_CV = np.zeros((len(name_output)+1,len(range_alpha)))
results_CV[0,:] = range_alpha 
results_all = np.zeros((len(name_output)+1,len(range_alpha)))
results_all[0,:] = range_alpha 

###############################################################################
### Run Training Procedure with Cross Validation
###############################################################################
                      
print("\nTraining procedure for linear regression algorithm under {}orable conditions".format(condition))
for io in range(0,4):
#for io in range(0,output_data.shape[1]):
    print("\nTraining for output parameter: {}".format(name_output[io]))
    for ia,alpha in enumerate(range_alpha):
        ridge = Ridge(alpha = alpha)
        rkf = RepeatedKFold(n_splits = n_splits, n_repeats = iterations)
        scores = cross_val_score(ridge, input_data, output_data[:,io], cv = rkf)
        score = np.mean(scores)
        print("For alpha {}, the average score is {:.3f}".format(alpha,score))
        results[ia] = score

    ###############################################################################
    ### Save Scores
    ###############################################################################

    ind = np.unravel_index(np.argmax(results, axis=None), results.shape)
    print('Maximum score {:.2f} for alpha = {}'.format(results[ind],range_alpha[ind[0]]))
    results_CV[io+1,:] = results
#    np.savetxt('../results/LR_Training_{}_{}_It{}.csv'.format(condition,name_output_short[io],iterations),results,fmt = '%.4f',delimiter = ',')

print('Save Cross Validation Training Results to file. \n')
np.savetxt('../results/LR_Training_{}_It{}.csv'.format(condition,iterations),results_CV.T,fmt = '%.4f',delimiter = ',',header=" , ".join(['alpha'] + name_output_short))

###############################################################################
### Run Training Procedure with full training Ddata set
###############################################################################

print("\nTraining for Linear regression algorithm \n under {}orable conditions".format(condition))
for io in range(0,4):
    print("\nOutput Parameter: {}".format(name_output[io]))
    for ia,alpha in enumerate(range_alpha):
        print('\nSelected Hyperparameters: \n alpha  =  {}'.format(alpha))
        ridge = Ridge(alpha = alpha)
        ridge.fit(input_data,output_data[:,io])
        r2_training = ridge.score(input_data, output_data[:,io])
        print("\nTraining set score (R2): {:.4f}".format(r2_training))
        results_all[io+1,ia] =  r2_training
        
        # y_pred = ridge.predict(input_data_testing)
        # print('\nAI predicted values: \n {}'.format(y_pred))
        # print('LBM simulation values \n {}'.format(output_data_testing[:,io]))

        # print("\nR2 for the test set: {:.4f}".format(r2_score(output_data_testing[:,io], y_pred)))
        # print("MAE for the test set: {:.4f}".format(mean_absolute_error(output_data_testing[:,io], y_pred)))
        # print("MSE for the test set: {:.4f}".format(mean_squared_error(output_data_testing[:,io], y_pred)))

print('Save Training full set results to file. \n')
np.savetxt('../results/LR_Training_{}_full.csv'.format(condition),results_all.T,fmt = '%.4f',delimiter = ',',header=" , ".join(['alpha'] + name_output_short))

print("LR Ridge Training finished.")