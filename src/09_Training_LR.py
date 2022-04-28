import numpy as np
from sklearn.linear_model import Ridge
# from sklearn.linear_model import Lasso
from sklearn.model_selection import RepeatedKFold, cross_val_score

###############################################################################
### Load data and set algorithm parameters
###############################################################################

iterations = 3000
condition = 'unfav' #'fav' #

range_alpha = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

test_sample_length = 0.9
n_splits = 10
name_output = ['Coordination number','Surface coverage','Conductivity','Void fraction']
name_output_short = ['CN','SC','HC','VF']

data_LBM = np.loadtxt("../data/LBM_results_{}.csv".format(condition), delimiter = ',',skiprows = 1)
n_test_samples = int(np.round(test_sample_length*len(data_LBM)))

input_data = data_LBM[:n_test_samples,0:4]
output_data = data_LBM[:n_test_samples,4:8]

np.set_printoptions(suppress = True)
results = np.zeros((len(range_alpha)))

###############################################################################
### Run Training Procedure
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

#    for ia,alpha in enumerate(range_alpha):
#        lasso = Lasso(alpha = alpha, max_iter = 100000)
#        rkf = RepeatedKFold(n_splits = n_splits, n_repeats = iterations)
#        scores = cross_val_score(lasso, input_data, output_data[:,io], cv = rkf)
#        score = np.mean(scores_1)
#        print("For alpha {}, the average score is {:.3f}".format(alpha,score))
#        results[imd,imss] = score

    ###############################################################################
    ### Save Scores
    ###############################################################################

    ind = np.unravel_index(np.argmax(results, axis=None), results.shape)
    print(ind)
    print('Maximum score {:.2f} for alpha = {}'.format(results[ind],range_alpha[ind[0]]))
    np.savetxt('../results/LR_Training_{}_{}_It{}.csv'.format(condition,name_output_short[io],iterations),results,fmt = '%.4f',delimiter = ',')

print("LR Training finished.")
