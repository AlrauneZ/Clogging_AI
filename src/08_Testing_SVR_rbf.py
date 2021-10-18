import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

###############################################################################
### Load data and set algorithm parameters
###############################################################################

condition = 'unfav' #'fav' #
param = 'CN' # 'SC' # 'HC' # 'VF'

### identified hyperparameters during Training
if param == 'CN':
    id_output_param = 0
    if condition == 'fav':
        C = 10
        gamma = 1
    elif condition == 'unfav':
        C = 100
        gamma = 1
elif param == 'SC':
    id_output_param = 1
    if condition == 'fav':
        C = 0.1
        gamma = 1
    elif condition == 'unfav':
        C = 1
        gamma = 100
if param == 'HC':
    id_output_param = 2
    if condition == 'fav':
        C = 100
        gamma = 0.1
    elif condition == 'unfav':
        C = 10
        gamma = 1
if param == 'VF':
    id_output_param = 3
    if condition == 'fav':
        C = 0.01
        gamma = 0.001
    elif condition == 'unfav':
        C = 10
        gamma = 0.01


test_sample_length = 0.9
name_output = ['Coordination number','Surface coverage','Conductivity','Void fraction']

data_LBM = np.loadtxt("../data/LBM_results_{}.csv".format(condition), delimiter = ',',skiprows = 1)
n_test_samples = int(np.round(test_sample_length*len(data_LBM)))

mm = MinMaxScaler()
mm_data = mm.fit_transform(data_LBM)

input_data_training = mm_data[:n_test_samples,0:4]
output_data_training = data_LBM[:n_test_samples,4:8]

input_data_testing = mm_data[n_test_samples:,0:4]
output_data_testing = data_LBM[n_test_samples:,4:8]

np.set_printoptions(suppress = True)

###############################################################################
### Run Testing Procedure and Print Output
###############################################################################

print("\nPerformance Testing for Support vector regression algorithm \n under {}orable conditions".format(condition))
print("\nOutput Parameter: {}".format(name_output[id_output_param]))
print('\nSelected Hyperparameters: \n C  =  {}\n gamma = {}'.format(C, gamma))
svr_rbf = SVR(kernel = 'rbf', gamma = gamma, C = C)
svr_rbf.fit(input_data_training,output_data_training[:,id_output_param])
y_pred = svr_rbf.predict(input_data_testing)

print('\nAI predicted values: \n {}'.format(y_pred))
print('LBM simulation values \n {}'.format(output_data_testing[:,id_output_param]))

print("\nTraining set score (R2): {:.4f}".format(svr_rbf.score(input_data_training, output_data_training[:,id_output_param])))
#print("Test set score: {:.4f}".format(svr_rbf.score(input_data_testing, output_data_testing[:,id_output_param])))
print("\nR2 for the test set: {:.4f}".format(r2_score(output_data_testing[:,id_output_param], y_pred)))
print("MAE for the test set: {:.4f}".format(mean_absolute_error(output_data_testing[:,id_output_param], y_pred)))
print("MSE for the test set: {:.4f}".format(mean_squared_error(output_data_testing[:,id_output_param], y_pred)))
