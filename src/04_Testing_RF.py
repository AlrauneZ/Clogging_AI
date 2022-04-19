import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
sys.stdout = open('../results/RF_Test_results.txt', 'w')

###############################################################################
### Load data and set algorithm parameters
###############################################################################


n_estimators = 100
max_features = 4
random_state = 42

test_sample_length = 0.9
conditions = ['fav','unfav']
name_output = ['Coordination number','Surface coverage','Conductivity','Void fraction']
np.set_printoptions(suppress = True)
file_LBM = "../data/LBM_results_{}.csv"
# test_results = np.zeros((12,2))

for ic,cond in enumerate(conditions):

    data_LBM = np.loadtxt(file_LBM.format(cond), delimiter = ',',skiprows = 1)
    n_test_samples = int(np.round(test_sample_length*len(data_LBM)))
    
    input_data_training = data_LBM[:n_test_samples,0:4]
    output_data_training = data_LBM[:n_test_samples,4:8]
    
    input_data_testing = data_LBM[n_test_samples:,0:4]
    output_data_testing = data_LBM[n_test_samples:,4:8]


    ###############################################################################
    ### Run Testing Procedure and Print Output
    ###############################################################################
    
    print("###################################################")
    print("Performance Testing for Random forest algorithm \n under {}orable conditions".format(cond))
    print("###################################################")
    
    for io,param in enumerate(name_output):
        print("\n#####################################\nOutput Parameter: {}".format(name_output[io]))
        print('\nSelected Hyperparameters: \n n_estimators  =  {}\n max_features = {}\n random state = {}'.format(n_estimators, max_features,random_state))
        forest = RandomForestRegressor(n_estimators = n_estimators, max_features = max_features, random_state = random_state)
        forest.fit(input_data_training,output_data_training[:,io])
        r2_training = forest.score(input_data_training, output_data_training[:,io])
        y_pred = forest.predict(input_data_testing)
        
        print('\nAI predicted values: \n {}'.format(y_pred))
        print('LBM simulation values \n {}'.format(output_data_testing[:,io]))
        
        print("\nTraining data set score (R2): {:.4f}".format(r2_training))
        print("\nTest data set:")
        print("R2 = {:.4f}".format(r2_score(output_data_testing[:,io], y_pred)))
        print("MSE = {:.4f}".format(mean_squared_error(output_data_testing[:,io], y_pred)))
        print("MAE = {:.4f}".format(mean_absolute_error(output_data_testing[:,io], y_pred)))

#         test_results[io,ic] = r2_score(output_data_testing[:,io], y_pred)
#         test_results[4+io,ic] = mean_squared_error(output_data_testing[:,io], y_pred)
#         test_results[8+io,ic] = mean_absolute_error(output_data_testing[:,io], y_pred)
        
# np.savetxt('../results/Test_results_RF.csv',test_results,fmt = '%.4f',delimiter=',')    
