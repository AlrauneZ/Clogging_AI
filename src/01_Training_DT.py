import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RepeatedKFold, cross_val_score

###############################################################################
### Load data and set algorithm parameters
###############################################################################

iterations = 2000
condition =  'fav' #'unfav' #

range_max_depth = range(2,20)
range_min_samples_split = range(2,10)

test_sample_length = 0.9
n_splits = 10
name_output = ['Coordination number','Surface coverage','Conductivity','Void fraction']
name_output_short = ['CN','SC','HC','VF']

data_LBM = np.loadtxt("../data/LBM_results_{}.csv".format(condition), delimiter = ',',skiprows = 1)
n_test_samples = int(np.round(test_sample_length*len(data_LBM)))

input_data = data_LBM[:n_test_samples,0:4]
output_data = data_LBM[:n_test_samples,4:8]

np.set_printoptions(suppress = True)

###############################################################################
### Run Training Procedure
###############################################################################

print("\nTraining procedure for decision tree algorithm under {}orable conditions".format(condition))
for io,no in enumerate(name_output):
    results = np.zeros((len(range_max_depth)+1,len(range_min_samples_split)+1))
    results[0,1:] = range_min_samples_split
    results[1:,0] = range_max_depth

    print("\nTraining for output parameter: {}".format(no))
    for imd,max_depth in enumerate(range_max_depth):
        for imss,min_samples_split in enumerate(range_min_samples_split):
            tree = DecisionTreeRegressor(max_depth = max_depth, min_samples_split = min_samples_split, random_state = 42)
            rkf_1 = RepeatedKFold(n_splits = n_splits, n_repeats = iterations)  
            scores_1 = cross_val_score(tree, input_data, output_data[:,io], cv = rkf_1)
            score_1 = np.mean(scores_1)
            # print("For max_depth {} and min_samples_split {}, the average score is {:.3f}".format(max_depth, min_samples_split,score_1))
            results[imd+1,imss+1] = score_1

    ###############################################################################
    ### Save Scores
    ###############################################################################

    ind = np.unravel_index(np.argmax(results[1:,1:], axis=None), results[1:,1:].shape)
    print('\nMaximum score {:.2f} for max_depth = {} and min_samples_split = {}'.format(results[ind[0]+1,ind[1]+1],range_max_depth[ind[0]],range_min_samples_split[ind[1]]))
    np.savetxt('../results/DT_Training_{}_{}_It{}.csv'.format(condition,name_output_short[io],iterations),results,fmt = '%.4f',delimiter = ',')

    # ###############################################################################
    # ### Run Training Procedure with full training Data set
    # ###############################################################################
    # results_all = np.zeros((len(range_max_depth)+1,len(range_min_samples_split)+1))
    # results_all[0,1:] = range_min_samples_split
    # results_all[1:,0] = range_max_depth

    # for imd,max_depth in enumerate(range_max_depth):
    #     for imss,min_samples_split in enumerate(range_min_samples_split):

    #         tree = DecisionTreeRegressor(max_depth = max_depth, min_samples_split = min_samples_split, random_state = 42)            
    #         tree.fit(input_data,output_data[:,io])
    #         r2_training = tree.score(input_data, output_data[:,io])
    #         results_all[imd+1,imss+1] =  r2_training
      
    # ind = np.unravel_index(np.argmax(results[1:,1:], axis=None), results[1:,1:].shape)
    # print('\nMaximum score {:.2f} for max_depth = {} and min_samples_split = {}'.format(results_all[ind[0]+1,ind[1]+1],range_max_depth[ind[0]],range_min_samples_split[ind[1]]))
    # np.savetxt('../results/DT_Training_{}_{}_full.csv'.format(condition,name_output_short[io]),results_all,fmt = '%.4f',delimiter = ',')

print("DC Training finished.")

