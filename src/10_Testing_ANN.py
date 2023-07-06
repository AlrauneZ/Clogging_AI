import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor #MLPClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import sys
sys.stdout = open('../results/ANN_Test_results.txt', 'w')

###############################################################################
### Load data and set algorithm parameters
###############################################################################

conditions = ['fav','unfav']

name_output = ['Coordination number','Surface coverage','Conductivity','Void fraction']
n_test_samples = 73
np.set_printoptions(suppress = True)
# test_results = np.zeros((12,2))

### identified hyperparameters during Training
nns = np.array([[219,57,194,146], [184,339,156,156]])

file_LBM = "../data/LBM_Results.xlsx"
xl = pd.ExcelFile(file_LBM)

for ic,cond in enumerate(conditions):

    data = pd.read_excel(xl,skiprows = [1],sheet_name=cond)
    data_LBM = MinMaxScaler().fit_transform(np.array(data))
   
    input_data_training = data_LBM[:n_test_samples,0:4]
    output_data_training = data_LBM[:n_test_samples,4:8]
    
    input_data_testing = data_LBM[n_test_samples:,0:4]
    output_data_testing = data_LBM[n_test_samples:,4:8]   


    ###############################################################################
    ### Run Testing Procedure and Print Output
    ###############################################################################
    
    print("####################################################")
    print("Performance Testing for Artificial Neural Network algorithm \n under {}orable conditions".format(cond))
    print("####################################################")
    
    for io,param in enumerate(name_output):
        nn = nns[ic,io]           
        np.random.seed(12345678+io+10*ic)
    
        print("\n#####################################\nOutput Parameter: {}".format(param))
        print('\nSelected Hyperparameters: \n Number of neurons  =  {}'.format(nn))
        # print(' seed:  =  {}'.format(seed))

        ann = MLPRegressor(hidden_layer_sizes=(nn,),max_iter=500,solver = "adam", learning_rate_init = 0.01)#, activation,learning_rate , alpha, batch_size , power_t, max_iter, shuffle, random_state, tol, verbose, warm_start, momentum, nesterovs_momentum, early_stopping, validation_fraction, beta_1, beta_2, epsilon, n_iter_no_change, max_fun)
        ann.fit(input_data_training,output_data_training[:,io])
            
        r2_training = ann.score(input_data_training, output_data_training[:,io])
        y_pred = ann.predict(input_data_testing)
        r2_testing = r2_score(output_data_testing[:,io], y_pred)
       
        print('\nAI predicted values: \n {}'.format(y_pred))
        print('LBM simulation values \n {}'.format(output_data_testing[:,io]))
        
        print("\nTraining data set score (NSE=R2): {:.4f}".format(r2_training))
        print("\nTest data set:")    
        print("NSE = {:.4f}".format(r2_testing))
        print("MSE = {:.4f}".format(mean_squared_error(output_data_testing[:,io], y_pred)))
        print("MAE = {:.4f}".format(mean_absolute_error(output_data_testing[:,io], y_pred)))

#         test_results[io,ic] = r2_score(output_data_testing[:,io], y_pred)
#         test_results[4+io,ic] = mean_squared_error(output_data_testing[:,io], y_pred)
#         test_results[8+io,ic] = mean_absolute_error(output_data_testing[:,io], y_pred)
        
# np.savetxt('../results/Test_results_ANN.csv',test_results,fmt = '%.4f',delimiter=',')    
