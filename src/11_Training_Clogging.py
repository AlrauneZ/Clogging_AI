import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RepeatedKFold, cross_val_score

###############################################################################
### Load data and set algorithm parameters
###############################################################################

condition =  'fav' #'unfav' #
algorithm =  'ALL'  # #'DT' #'ANN' #,
#full =  True #False #

### data specifics
name_output = 'Clogging'
n_splits = 10
n_training = 73

### Load simulation results from Lattice Boltzman Method (physical transport simualation) 
file_LBM = "../data/LBM_Results.xlsx"
xl = pd.ExcelFile(file_LBM)
data_LBM = pd.read_excel(xl,skiprows = [1],sheet_name=condition)
data_training = data_LBM[0:n_training]
input_data = np.array(data_training)[:,0:4]
output_data = np.array(data_training)[:,8]
output_data = np.where(output_data==1,output_data,-1) 

### local settings
np.set_printoptions(suppress = True)
random_state = 42

################################################
### Range of hyperparameters during Training ###
################################################

### Decision Tree
range_iterations = [20,50,100,200]
range_max_depth = range(2,11)
min_samples_split = 2

### Artificial Neural Network
range_neurons = range(2,20)

###############################################################################
### Run Training Procedure for Decision Tree
###############################################################################
if algorithm in ['DT','ALL']:
    print("\nTraining procedure for decision tree algorithm under {}orable conditions".format(condition))
    
    results = np.zeros((len(range_max_depth)+1,len(range_iterations)+1))
    results[0,1:] = range_iterations
    results[1:,0] = range_max_depth
        
    for it, iterations in enumerate(range_iterations):
        np.random.seed(12345678+iterations)
        for imd,max_depth in enumerate(range_max_depth):
            tree = DecisionTreeClassifier(max_depth = max_depth, min_samples_split = min_samples_split, random_state = random_state)
            rkf = RepeatedKFold(n_splits = n_splits, n_repeats = iterations)  
            scores = cross_val_score(tree, input_data, output_data, cv = rkf)
            score = np.mean(scores)
            results[imd+1,it+1] = score
    
            # if full:
            #     ### Run Training Procedure with full training Data set
            #     tree.fit(input_data,output_data)
            #     r2_training = tree.score(input_data, output_data)
            #     results[imd+1,-1] =  r2_training
    
    ###############################################################################
    ### Save Scores
    ###############################################################################
    
    # ind = np.unravel_index(np.argmax(results[1:,1:], axis=None), results[1:,1:].shape)
    # print('\nMaximum score {:.2f} for max_depth = {} and iterations = {}'.format(results[ind[0]+1,ind[1]+1],range_max_depth[ind[0]],range_min_samples_split[ind[1]]))
    # np.savetxt('../results/Clogging_Training_DT_{}_It{}.csv'.format(condition,iterations),results,fmt = '%.4f',delimiter = ',')
    np.savetxt('../results/Clogging_Training_DT_{}.csv'.format(condition),results.T,fmt = '%.4f',delimiter = ',')
          
    # if full:
    #     ind = np.unravel_index(np.argmax(results[1:,1:], axis=None), results[1:,1:].shape)
    #     print('\nMaximum score {:.2f} for max_depth = {} and min_samples_split = {}'.format(results_all[ind[0]+1,ind[1]+1],range_max_depth[ind[0]],range_min_samples_split[ind[1]]))
    #     np.savetxt('../results/Clogging_Training_DT_{}_full.csv'.format(condition),results_all,fmt = '%.4f',delimiter = ',')
    
    print("DC Training finished.")
 
###############################################################################
### Run Training Procedure for Artificial Neural Network
###############################################################################

if algorithm in ['ANN','ALL']:
    print("\nTraining procedure for Artificial Neural Network algorithm under {}orable conditions".format(condition))
    
    results = np.zeros((len(range_neurons),2))
    results[:,0] = range_neurons
    
    # results_all = np.zeros((len(range_neurons),2))
    # results_all[:,0] = range_neurons

    print("\n################################################")
    print("Artificial Neural Network algorithm")
    print("################################################")
               
    for il , neurons in enumerate(range_neurons):
        ann = MLPClassifier(hidden_layer_sizes=(neurons,),activation='logistic')#,random_state=random_state)#,max_iter=500)

        rkf = RepeatedKFold(n_splits = n_splits, n_repeats = iterations)  
        scores = cross_val_score(ann, input_data, output_data, cv = rkf)
        score = np.mean(scores)
        results[il,1] = score

        # if full:
        #     ### Run Training Procedure with full training Data set
        #     ann.fit(input_data,output_data)
        #     r2_training = ann.score(input_data, output_data)
        #     results_all[il,1] =  r2_training

    ###############################################################################
    ### Save Scores
    ###############################################################################
    ind = np.argmax(results[:,1])
    print('\nMaximum score {:.2f} for neurons = {}'.format(results[ind,1],results[ind,0]))
    np.savetxt('../results/Clogging_Training_ANN_{}_It{}.csv'.format(condition,iterations),results,fmt = '%.4f',delimiter = ',')
          
    # if full:
    #     ind = np.argmax(results_all[:,1])
    #     print('\nMaximum score {:.2f} for neurons = {}'.format(results_all[ind,1],results[ind,0]))
    #     np.savetxt('../results/Clogging_Training_ANN_{}_full.csv'.format(condition),results_all,fmt = '%.4f',delimiter = ',')
    
    print("ANN Training finished.")

