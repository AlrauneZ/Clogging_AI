import numpy as np
from sklearn.tree import DecisionTreeRegressor
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
        max_depth = 2
        min_samples_split = 8 
    elif condition == 'unfav':
        max_depth = 2
        min_samples_split = 3 
elif param == 'SC':
    id_output_param = 1
    if condition == 'fav':        
        max_depth = 15
        min_samples_split = 6
    elif condition == 'unfav':
        max_depth = 4
        min_samples_split = 2
if param == 'HC':
    id_output_param = 2
    if condition == 'fav':        
        max_depth = 9
        min_samples_split = 3
    elif condition == 'unfav':
        max_depth = 10
        min_samples_split = 9
if param == 'VF':
    id_output_param = 3
    if condition == 'fav':        
        max_depth = 17
        min_samples_split = 7
    elif condition == 'unfav':
        max_depth = 4
        min_samples_split = 4 

# id_output_param = 0
# max_depth = 2
# min_samples_split = 8

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

print("\nPerformance Testing for Decision Tree algorithm \n under {}orable conditions".format(condition))
print("\nOutput Parameter: {}".format(name_output[id_output_param])) 
print('\nSelected Hyperparameters: \n max_depth  =  {}\n min_samples_split = {}'.format(max_depth, min_samples_split))
tree = DecisionTreeRegressor(max_depth = max_depth, min_samples_split = min_samples_split, random_state = 42)
tree.fit(input_data_training,output_data_training[:,id_output_param])
y_pred = tree.predict(input_data_testing)

print('\nAI predicted values: \n {}'.format(y_pred))
print('LBM simulation values \n {}'.format(output_data_testing[:,id_output_param]))

print("\nTraining set score (R2): {:.4f}".format(tree.score(input_data_training, output_data_training[:,id_output_param])))
#print("Test set score: {:.4f}".format(tree.score(input_data_testing, output_data_testing[:,id_output_param])))
print("\nR2 for the test set: {:.4f}".format(r2_score(output_data_testing[:,id_output_param], y_pred)))
print("MAE for the test set: {:.4f}".format(mean_absolute_error(output_data_testing[:,id_output_param], y_pred)))
print("MSE for the test set: {:.4f}".format(mean_squared_error(output_data_testing[:,id_output_param], y_pred)))
