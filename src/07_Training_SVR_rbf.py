import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler

###############################################################################
### Load data and set algorithm parameters
###############################################################################

iterations = 3000
condition = 'unfav' #'fav' #

range_C = [0.001, 0.01, 0.1, 1, 10, 100]
range_gamma = [0.001, 0.01, 0.1, 1, 10, 100]

test_sample_length = 0.9
n_splits = 10
name_output = ['Coordination number','Surface coverage','Conductivity','Void fraction']
name_output_short = ['CN','SC','HC','VF']

data_LBM = np.loadtxt("../data/LBM_results_{}.csv".format(condition), delimiter = ',',skiprows = 1)
n_test_samples = int(np.round(test_sample_length*len(data_LBM)))

mm = MinMaxScaler()
mm_data = mm.fit_transform(data_LBM)

input_data = mm_data[:n_test_samples,0:4]
output_data = data_LBM[:n_test_samples,4:8]

np.set_printoptions(suppress = True)
results = np.zeros((len(range_C),len(range_gamma)))

###############################################################################
### Run Training Procedure
###############################################################################

print("\nTraining procedure for support vector regression algorithm with the rbf kernel under {}orable conditions".format(condition))

for io in range(0,4):
#for io in range(0,output_data.shape[1]):
    print("\nTraining for output parameter: {}".format(name_output[io]))
    for ic,C in enumerate(range_C):
        for ig,gamma in enumerate(range_gamma):
            svr_rbf = SVR(kernel = 'rbf', gamma = gamma, C = C)
            rkf = RepeatedKFold(n_splits = n_splits, n_repeats = iterations)
            scores = cross_val_score(svr_rbf, input_data, output_data[:,io], cv = rkf)
            score = np.mean(scores)
            print("For C {} and gamma {}, the average score is {:.3f}".format(C, gamma,score))
            results[ic,ig] = score

    ###############################################################################
    ### Save Scores
    ###############################################################################

    ind = np.unravel_index(np.argmax(results, axis=None), results.shape)
    print(ind)
    print('Maximum score {:.2f} for C = {} and gamma = {}'.format(results[ind],range_C[ind[0]],range_gamma[ind[1]]))
    np.savetxt('../results/SVR_lin_Training_{}_{}_It{}.csv'.format(condition,name_output_short[io],iterations),results,fmt = '%.4f',delimiter = ',')

print("SVR_rbf Training finished.")
