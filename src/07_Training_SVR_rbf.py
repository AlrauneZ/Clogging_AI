import numpy as np
import random
from sklearn.svm import SVR
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler

###############################################################################
### Load data and set algorithm parameters
###############################################################################

iterations = 100
condition = 'fav' #'unfav' #
random.seed(12345678+iterations)

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

###############################################################################
### Run Training Procedure
###############################################################################

print("\nTraining procedure for support vector regression algorithm with the rbf kernel under {}orable conditions".format(condition))

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

    # ###############################################################################
    # ### Run Training Procedure with full training Data set
    # ###############################################################################
    # results_all = np.zeros((len(range_C)+1,len(range_gamma)+1))
    # results_all[0,1:] = range_gamma
    # results_all[1:,0] = range_C

    # for ic,C in enumerate(range_C):
    #     for ig,gamma in enumerate(range_gamma):
            
    #         svr_rbf = SVR(kernel = 'rbf', gamma = gamma, C = C)
    #         svr_rbf.fit(input_data,output_data[:,io])
    #         r2_training = svr_rbf.score(input_data, output_data[:,io])
    #         results_all[ic+1,ig+1] =  r2_training
      
    # ind = np.unravel_index(np.argmax(results_all[1:,1:], axis=None), results_all[1:,1:].shape)
    # print('\nMaximum score {:.2f} for C = {} and gamma = {}\n'.format(results_all[ind[0]+1,ind[1]+1],range_C[ind[0]],range_gamma[ind[1]]))
    # np.savetxt('../results/SVR_lin_Training_{}_{}_full.csv'.format(condition,name_output_short[io]),results_all,fmt = '%.4f',delimiter = ',')

print("\nSVR_rbf Training finished.")
