import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

###############################################################################
### Load data and set algorithm parameters
###############################################################################

condition = 'unfav' #'fav' #
param = 'CN' # 'SC' # 'HC' # 'VF'

n_estimators = 100
max_features = 4
random_state = 42

if param == 'CN':
    id_output_param = 0
elif param == 'SC':
    id_output_param = 1
elif param == 'HC':
    id_output_param = 2
elif param == 'VF':
    id_output_param = 3

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

print("\nPerformance Testing for Random forest algorithm \n under {}orable conditions".format(condition))
print("\nOutput Parameter: {}".format(name_output[id_output_param]))
print('\nSelected Hyperparameters: \n n_estimators  =  {}\n max_features = {}\n random state = {}'.format(n_estimators, max_features,random_state))
forest = RandomForestRegressor(n_estimators = n_estimators, max_features = max_features, random_state = random_state)
forest.fit(input_data_training,output_data_training[:,id_output_param])
r2_training = forest.score(input_data_training, output_data_training[:,id_output_param])
y_pred = forest.predict(input_data_testing)

print('\nAI predicted values: \n {}'.format(y_pred))
print('LBM simulation values \n {}'.format(output_data_testing[:,id_output_param]))

print("\nTraining set score (R2): {:.4f}".format(r2_training))
#print("Test set score: {:.4f}".format(forest.score(input_data_testing, output_data_testing[:,id_output_param])))
print("\nR2 for the test set: {:.4f}".format(r2_score(output_data_testing[:,id_output_param], y_pred)))
print("MAE for the test set: {:.4f}".format(mean_absolute_error(output_data_testing[:,id_output_param], y_pred)))
print("MSE for the test set: {:.4f}".format(mean_squared_error(output_data_testing[:,id_output_param], y_pred)))
