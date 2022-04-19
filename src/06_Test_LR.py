import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
sys.stdout = open('../results/LR_Test_results.txt', 'w')

###############################################################################
### Load data and set algorithm parameters
###############################################################################

conditions = ['fav','unfav']

test_sample_length = 0.9
name_output = ['Coordination number','Surface coverage','Conductivity','Void fraction']
file_LBM = "../data/LBM_results_{}.csv"
np.set_printoptions(suppress = True)
# test_results = np.zeros((12,2))

### identified hyperparameters during Training
alpha_fav = [0.1,1,0.01,10]
alpha_unfav = [0.01,100,0.01,0.01]

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
    
    print("####################################################")
    print("Performance Testing for Linear Regression algorithm \n under {}orable conditions".format(cond))
    print("####################################################")
    
    for io,param in enumerate(name_output):
            
        if cond == 'fav':        
            alpha = alpha_fav[io]
        elif cond == 'unfav':
            alpha = alpha_unfav[io]
    
        print("\n#####################################\nOutput Parameter: {}".format(param))
        print('\nSelected Hyperparameters: \n alpha  =  {}'.format(alpha))
        ridge = Ridge(alpha = alpha)
        ridge.fit(input_data_training,output_data_training[:,io])
        
        r2_training = ridge.score(input_data_training, output_data_training[:,io])
        y_pred = ridge.predict(input_data_testing)
        
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
        
# np.savetxt('../results/Test_results_LR.csv',test_results,fmt = '%.4f',delimiter=',')    
