import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import sys
sys.stdout = open('../results/SVR_Test_results.txt', 'w')

###############################################################################
### Load data and set algorithm parameters
###############################################################################

conditions = ['fav','unfav']

name_output = ['Coordination number','Surface coverage','Conductivity','Void fraction']
n_test_samples = 73
np.set_printoptions(suppress = True)
# test_results = np.zeros((12,2))

### identified hyperparameters during Training
C_fav = [10,100,10,100]
gamma_fav = [1,1,1,1]

C_unfav = [100,100,100,100]
gamma_unfav = [0.1,0.1,1,0.1]

file_LBM = "../data/LBM_Results.xlsx"
xl = pd.ExcelFile(file_LBM)

for ic,cond in enumerate(conditions):

    data = pd.read_excel(xl,skiprows = [1],sheet_name=cond)
    data_LBM = np.array(data)

    mm = MinMaxScaler()
    mm_data = mm.fit_transform(data_LBM)

    input_data_training = mm_data[:n_test_samples,0:4]
    output_data_training = data_LBM[:n_test_samples,4:8]

    input_data_testing = mm_data[n_test_samples:,0:4]
    output_data_testing = data_LBM[n_test_samples:,4:8]
  
    ###############################################################################
    ### Run Testing Procedure and Print Output
    ###############################################################################

    print("####################################################")
    print("\nPerformance Testing for Support Vector Regression \n under {}orable conditions".format(cond))
    print("####################################################")
    
    for io,param in enumerate(name_output):
            
        if cond == 'fav':        
            C, gamma = C_fav[io], gamma_fav[io]
        elif cond == 'unfav':
            C, gamma = C_unfav[io], gamma_unfav[io]
    
        print("\n#####################################\nOutput Parameter: {}".format(param))
        print('\nSelected Hyperparameters: \n C  =  {}\n gamma = {}'.format(C, gamma))
        svr_rbf = SVR(kernel = 'rbf', gamma = gamma, C = C)
        svr_rbf.fit(input_data_training,output_data_training[:,io])
        r2_training = svr_rbf.score(input_data_training, output_data_training[:,io])
        y_pred = svr_rbf.predict(input_data_testing)
        
        print('\nAI predicted values: \n {}'.format(y_pred))
        print('LBM simulation values \n {}'.format(output_data_testing[:,io]))
        
        print("\nTraining set score (R2): {:.4f}".format(r2_training))
        print("\nTest data set:")
        print("\nNSE = {:.4f}".format(r2_score(output_data_testing[:,io], y_pred)))
        print("MSE = {:.4f}".format(mean_squared_error(output_data_testing[:,io], y_pred)))
        print("MAE = {:.4f}".format(mean_absolute_error(output_data_testing[:,io], y_pred)))

#         test_results[io,ic] = r2_score(output_data_testing[:,io], y_pred)
#         test_results[4+io,ic] = mean_squared_error(output_data_testing[:,io], y_pred)
#         test_results[8+io,ic] = mean_absolute_error(output_data_testing[:,io], y_pred)
        
# np.savetxt('../results/Test_results_SVR.csv',test_results,fmt = '%.4f',delimiter=',')    
