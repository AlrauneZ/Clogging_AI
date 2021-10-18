import numpy as np
from sklearn.linear_model import Ridge
# from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

###############################################################################
### Load data and set algorithm parameters
###############################################################################

condition = 'unfav' #'fav' #
param = 'CN' # 'SC' # 'HC' # 'VF'

### identified hyperparameters during Training
if param == 'CN':
    id_output_param = 0
    if condition == 'fav':
        alpha = 0.1
    #   alpha = 0.01
    elif condition == 'unfav':
        alpha = 0.01
    #   alpha = 0.0001
elif param == 'SC':
    id_output_param = 1
    if condition == 'fav':
        alpha = 100
    #   alpha = 0.1
    elif condition == 'unfav':
        alpha = 100
    #   alpha = 100
if param == 'HC':
    id_output_param = 2
    if condition == 'fav':
        alpha = 0.01
    #   alpha = 0.0001
    elif condition == 'unfav':
        alpha = 0.001
    #   alpha = 0.0001
if param == 'VF':
    id_output_param = 3
    if condition == 'fav':
        alpha = 1
    #   alpha = 0.001
    elif condition == 'unfav':
        alpha = 0.001
    #   alpha = 0.001


test_sample_length = 0.9
name_output = ['Coordination number','Surface coverage','Conductivity','Void fraction']

data_LBM = np.loadtxt("../data/LBM_results_{}.csv".format(condition), delimiter = ',',skiprows = 1)
n_test_samples = int(np.round(test_sample_length*len(data_LBM)))

input_data_training = data_LBM[:n_test_samples,0:4]
output_data_training = data_LBM[:n_test_samples,4:8]

input_data_testing = data_LBM[n_test_samples:,0:4]
output_data_testing = data_LBM[n_test_samples:,4:8]

np.set_printoptions(suppress = True)

###############################################################################
### Run Testing Procedure and Print Output
###############################################################################

print("\nPerformance Testing for Linear regression algorithm \n under {}orable conditions".format(condition))
print("\nOutput Parameter: {}".format(name_output[id_output_param]))
print('\nSelected Hyperparameters: \n alpha  =  {}'.format(alpha))
ridge = Ridge(alpha = alpha)
# lasso = Lasso(alpha = alpha)
ridge.fit(input_data_training,output_data_training[:,id_output_param])
# lasso.fit(input_data_training,output_data_training[:,id_output_param])
y_pred = ridge.predict(input_data_testing)
# y_pred = lasso.predict(input_data_testing)

print('\nAI predicted values: \n {}'.format(y_pred))
print('LBM simulation values \n {}'.format(output_data_testing[:,id_output_param]))

print("\nTraining set score (R2): {:.4f}".format(ridge.score(input_data_training, output_data_training[:,id_output_param])))
#print("Test set score: {:.4f}".format(ridge.score(input_data_testing, output_data_testing[:,id_output_param])))

# print("\nTraining set score (R2): {:.4f}".format(lasso.score(input_data_training, output_data_training[:,id_output_param])))
# print("Test set score: {:.4f}".format(lasso.score(input_data_testing, output_data_testing[:,id_output_param])))

print("\nR2 for the test set: {:.4f}".format(r2_score(output_data_testing[:,id_output_param], y_pred)))
print("MAE for the test set: {:.4f}".format(mean_absolute_error(output_data_testing[:,id_output_param], y_pred)))
print("MSE for the test set: {:.4f}".format(mean_squared_error(output_data_testing[:,id_output_param], y_pred)))
