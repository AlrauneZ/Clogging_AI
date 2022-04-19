import numpy as np
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
# from sklearn.linear_model import Ridge, RidgeClassifier
# from sklearn.svm import SVR,SVC
# from sklearn.neural_network import MLPRegressor,MLPClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
sys.stdout = open('../results/Clogging_Test_results.txt', 'w')

###############################################################################
### Load data and set algorithm parameters
###############################################################################


test_sample_length = 0.9
conditions = ['fav','unfav']
np.set_printoptions(suppress = True)

##################################################
### Identified hyperparameters during Training ###
##################################################

### Decision Tree
max_depth = [5,3]
min_samples_split = [2,2]

### Random Forest
#n_estimators,max_features, random_state =  100, 4, 42

### Artificial Neural Network
#layers = [2,2]

for ic,cond in enumerate(conditions):
    print("\n################################################")
    print("Performance Testing for Clogging \nunder {}orable conditions".format(cond))
    print("################################################")

    # test_results = np.zeros((3,2))

    data_LBM = np.loadtxt("../data/LBM_results_{}.csv".format(cond), delimiter = ',',skiprows = 1)
    n_test_samples = int(np.round(test_sample_length*len(data_LBM)))
    
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
               
    print('\nSelected Hyperparameters: \n max_depth  =  {}\n min_samples_split = {}'.format(max_depth[ic], min_samples_split[ic]))
    tree = DecisionTreeClassifier(max_depth = max_depth[ic], min_samples_split = min_samples_split[ic], random_state = 42)
    # tree = DecisionTreeRegressor(max_depth = max_depth[ic], min_samples_split = min_samples_split[ic], random_state = 42)
    tree.fit(input_data_training,output_data_training)
    r2_training = tree.score(input_data_training, output_data_training)
    y_pred = tree.predict(input_data_testing)
    
    print('\nAI predicted values: \n {}'.format(y_pred))
    print('LBM simulation values \n {}'.format(output_data_testing))
    
    print("\nTraining data set score (R2): {:.4f}".format(r2_training))
    print("\nTest data set:")
    #print("Test set score: {:.4f}".format(tree.score(input_data_testing, output_data_testing[:,io])))
    print("R2 = {:.4f}".format(r2_score(output_data_testing, y_pred)))
    print("MSE = {:.4f}".format(mean_squared_error(output_data_testing, y_pred)))
    print("MAE = {:.4f}".format(mean_absolute_error(output_data_testing, y_pred)))

    # test_results[0,0] = r2_score(output_data_testing, y_pred)
    # test_results[1,0] = mean_squared_error(output_data_testing, y_pred)
    # test_results[2,0] = mean_absolute_error(output_data_testing, y_pred)

    print("\n################################################")
    print("Random Forest")
    print("################################################")

    # print('\nSelected Hyperparameters: \n n_estimators  =  {}\n max_features = {}\n random state = {}'.format(n_estimators, max_features,random_state))
    # forest = RandomForestRegressor(n_estimators = n_estimators, max_features = max_features, random_state = random_state)
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

    # print("\n################################################")
    # print("Linear Regression")
    # print("################################################")

    # print('\nSelected Hyperparameters: \n alpha  =  {}'.format(alpha[ic]))
    # # ridge = Ridge(alpha = alpha[ic])
    # ridge = RidgeClassifier(alpha = alpha[ic])
    # ridge.fit(input_data_training,output_data_training)
    # r2_training = ridge.score(input_data_training, output_data_training)
    # y_pred = ridge.predict(input_data_testing)

    # print('\nAI predicted values: \n {}'.format(y_pred))
    # print('LBM simulation values \n {}'.format(output_data_testing))

    # print("\nTraining data set score (R2): {:.4f}".format(r2_training))
    # print("\nTest data set:")
    # print("R2 = {:.4f}".format(r2_score(output_data_testing, y_pred)))
    # print("MSE = {:.4f}".format(mean_squared_error(output_data_testing, y_pred)))
    # print("MAE = {:.4f}".format(mean_absolute_error(output_data_testing, y_pred)))

    # test_results[0,2] = r2_score(output_data_testing, y_pred)
    # test_results[1,2] = mean_squared_error(output_data_testing, y_pred)
    # test_results[2,2] = mean_absolute_error(output_data_testing, y_pred)
        
    # print("\n################################################")
    # print("Support Vector Regression")
    # print("################################################")

    # print('\nSelected Hyperparameters: \n C  =  {}\n gamma = {}'.format(C[ic], gamma[ic]))
    # # svr_rbf = SVR(kernel = 'rbf', gamma = gamma[ic], C = C[ic])
    # svr_rbf = SVC(kernel = 'rbf', gamma = gamma[ic], C = C[ic])
    # svr_rbf.fit(input_data_training,output_data_training)
    # r2_training = svr_rbf.score(input_data_training, output_data_training)
    # y_pred = svr_rbf.predict(input_data_testing)

    # print('\nAI predicted values: \n {}'.format(y_pred))
    # print('LBM simulation values \n {}'.format(output_data_testing))

    # print("\nTraining data set score (R2): {:.4f}".format(r2_training))
    # print("\nTest data set:")
    # print("R2 = {:.4f}".format(r2_score(output_data_testing, y_pred)))
    # print("MSE = {:.4f}".format(mean_squared_error(output_data_testing, y_pred)))
    # print("MAE = {:.4f}".format(mean_absolute_error(output_data_testing, y_pred)))

    # test_results[0,3] = r2_score(output_data_testing, y_pred)
    # test_results[1,3] = mean_squared_error(output_data_testing, y_pred)
    # test_results[2,3] = mean_absolute_error(output_data_testing, y_pred)
 
    # print("\n################################################")
    # print("Artificial Neural Network algorithm")
    # print("################################################")
               
    # # print('\nSelected Hyperparameters: \n max_depth  =  {}\n min_samples_split = {}'.format(max_depth[ic], min_samples_split[ic]))

    # # ann = MLPRegressor(hidden_layer_sizes=(nn,),max_iter=500)#, activation, solver, alpha, batch_size, learning_rate, learning_rate_init, power_t, max_iter, shuffle, random_state, tol, verbose, warm_start, momentum, nesterovs_momentum, early_stopping, validation_fraction, beta_1, beta_2, epsilon, n_iter_no_change, max_fun)
    # ann = MLPClassifier(hidden_layer_sizes=(layers[ic],),activation='logistic',random_state=random_state)#,max_iter=500)
    # ann.fit(input_data_training,output_data_training)
    # r2_training = ann.score(input_data_training, output_data_training)
    # y_pred = ann.predict(input_data_testing)
    
    # print('\nAI predicted values: \n {}'.format(y_pred))
    # print('LBM simulation values \n {}'.format(output_data_testing))
    
    # print("\nTraining data set score (R2): {:.4f}".format(r2_training))
    # print("\nTest data set:")
    # #print("Test set score: {:.4f}".format(tree.score(input_data_testing, output_data_testing[:,io])))
    # print("R2 = {:.4f}".format(r2_score(output_data_testing, y_pred)))
    # print("MSE = {:.4f}".format(mean_squared_error(output_data_testing, y_pred)))
    # print("MAE = {:.4f}".format(mean_absolute_error(output_data_testing, y_pred)))

    # test_results[0,4] = r2_score(output_data_testing, y_pred)
    # test_results[1,4] = mean_squared_error(output_data_testing, y_pred)
    # test_results[2,4] = mean_absolute_error(output_data_testing, y_pred)

    # np.savetxt('../results/Clogging_Test_metrics_{}.csv'.format(cond),test_results,fmt = '%.4f',delimiter=',',header = 'DT , RF, LR, SVR, ANN ')    
