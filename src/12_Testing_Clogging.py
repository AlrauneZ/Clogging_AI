import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys

sys.stdout = open('../results/Clogging_Test_results.txt', 'w')

###############################################################################
### Load data and set algorithm parameters
###############################################################################

conditions = ['fav','unfav']
n_test_samples = 73
np.set_printoptions(suppress = True)

##################################################
### Identified hyperparameters during Training ###
##################################################

### Decision Tree
max_depth = [6,3]

### Artificial Neural Network
nns = [5,4]

file_LBM = "../data/LBM_Results.xlsx"
xl = pd.ExcelFile(file_LBM)

for ic,cond in enumerate(conditions):
    print("\n################################################")
    print("Performance Testing for Clogging \nunder {}orable conditions".format(cond))
    print("################################################")

    # test_results = np.zeros((3,2))

    data = pd.read_excel(xl,skiprows = [1],sheet_name=cond)
    data_LBM = np.array(data)
    
    input_data_training = data_LBM[:n_test_samples,0:4]
    output_data_training = data_LBM[:n_test_samples,8]
    
    input_data_testing = data_LBM[n_test_samples:,0:4]
    output_data_testing = data_LBM[n_test_samples:,8]

    ###############################################################################
    ### Run Testing Procedure and Print Output
    ###############################################################################
    
    print("\n################################################")
    print("Decision Tree algorithm")
    print("################################################")
               
    print('\nSelected Hyperparameters: \n max_depth  =  {}'.format(max_depth[ic]))
    tree = DecisionTreeClassifier(max_depth = max_depth[ic], min_samples_split = 2, random_state = 42)
    tree.fit(input_data_training,output_data_training)
    r2_training = tree.score(input_data_training, output_data_training)
    y_pred = tree.predict(input_data_testing)
    
    print('\nAI predicted values: \n {}'.format(y_pred))
    print('LBM simulation values \n {}'.format(output_data_testing))
    
    print("\nTraining data set score (R2): {:.4f}".format(r2_training))
    print("\nTest data set:")
    print("R2 = {:.4f}".format(r2_score(output_data_testing, y_pred)))
    print("MSE = {:.4f}".format(mean_squared_error(output_data_testing, y_pred)))
    print("MAE = {:.4f}".format(mean_absolute_error(output_data_testing, y_pred)))

    # test_results[0,0] = r2_score(output_data_testing, y_pred)
    # test_results[1,0] = mean_squared_error(output_data_testing, y_pred)
    # test_results[2,0] = mean_absolute_error(output_data_testing, y_pred)

    print("\n################################################")
    print("Random Forest")
    print("################################################")

    forest = RandomForestClassifier()#
    forest.fit(input_data_training,output_data_training)
    r2_training = forest.score(input_data_training, output_data_training)
    y_pred = forest.predict(input_data_testing)
        
    print('\nAI predicted values: \n {}'.format(y_pred))
    print('LBM simulation values \n {}'.format(output_data_testing))
        
    print("\nTraining data set score (R2): {:.4f}".format(r2_training))
    print("\nTest data set:")
    print("R2 = {:.4f}".format(r2_score(output_data_testing, y_pred)))
    print("MSE = {:.4f}".format(mean_squared_error(output_data_testing, y_pred)))
    print("MAE = {:.4f}".format(mean_absolute_error(output_data_testing, y_pred)))

    # test_results[0,1] = r2_score(output_data_testing, y_pred)
    # test_results[1,1] = mean_squared_error(output_data_testing, y_pred)
    # test_results[2,1] = mean_absolute_error(output_data_testing, y_pred)
 
    print("\n################################################")
    print("Artificial Neural Network algorithm")
    print("################################################")
               
    ann = MLPClassifier(hidden_layer_sizes=(nns[ic],),max_iter=500,solver = "adam", learning_rate_init = 0.01)#, activation,learning_rate , alpha, batch_size , power_t, max_iter, shuffle, random_state, tol, verbose, warm_start, momentum, nesterovs_momentum, early_stopping, validation_fraction, beta_1, beta_2, epsilon, n_iter_no_change, max_fun)
    ann.fit(input_data_training,output_data_training)
    r2_training = ann.score(input_data_training, output_data_training)
    y_pred = ann.predict(input_data_testing)
    
    print('\nAI predicted values: \n {}'.format(y_pred))
    print('LBM simulation values \n {}'.format(output_data_testing))
    
    print("\nTraining data set score (R2): {:.4f}".format(r2_training))
    print("\nTest data set:")
    print("R2 = {:.4f}".format(r2_score(output_data_testing, y_pred)))
    print("MSE = {:.4f}".format(mean_squared_error(output_data_testing, y_pred)))
    print("MAE = {:.4f}".format(mean_absolute_error(output_data_testing, y_pred)))

    # test_results[0,4] = r2_score(output_data_testing, y_pred)
    # test_results[1,4] = mean_squared_error(output_data_testing, y_pred)
    # test_results[2,4] = mean_absolute_error(output_data_testing, y_pred)

    # np.savetxt('../results/Clogging_Test_metrics_{}.csv'.format(cond),test_results,fmt = '%.4f',delimiter=',',header = 'DT , RF, LR, SVR, ANN ')    
