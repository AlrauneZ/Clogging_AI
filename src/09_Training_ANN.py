import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor #MLPClassifier
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler

# ###############################################################################
# ### Load data and set algorithm parameters
# ###############################################################################

iterations =50 #
condition = 'fav' #'unfav' #
### Index of input parameter to start training:
io = 3

### data specifics
name_output = ['Coordination number','Surface coverage','Conductivity','Void fraction']
name_output_short = ['CN','SC','HC','VF']
n_training = 73
n_splits = 10

## Load simulation results from Lattice Boltzman Method (physical transport simualation) 
file_LBM = "../data/LBM_Results.xlsx"
xl = pd.ExcelFile(file_LBM)
data_LBM = pd.read_excel(xl,skiprows = [1],sheet_name=condition)
data_training = np.array(data_LBM[0:n_training])
data_training = MinMaxScaler().fit_transform(data_training)
input_data = data_training[:,0:4]
output_data = data_training[:,4:8]

if condition == 'fav' :
    nn_max = [300,200,200,400]
else:
    nn_max = [300,400,200,200]
range_neurons = np.arange(2,nn_max[io])

np.set_printoptions(suppress = True)
np.random.seed(12345678+iterations)


#name_file = '../results/ANN_Training_{}_{}_It{}.csv'.format(condition,name_output_short[io],iterations)
name_file = '../results/ANN_L_{}_{}_It{}.csv'.format(name_output_short[io],condition,iterations)

###############################################################################
### Run Training Procedure with Cross Validation
###############################################################################

print("\nTraining procedure with cross validation \n for ANN under {}orable conditions, {} iterations".format(condition,iterations))
no = name_output[io]

print("\nTraining for output parameter: {}".format(no))
results = np.zeros((2,len(range_neurons)))
results[0,:] = range_neurons

for i0, nn in enumerate(range_neurons):
    ann = MLPRegressor(hidden_layer_sizes=(nn,),max_iter=500,solver = "adam", learning_rate_init = 0.01)#, activation,learning_rate , alpha, batch_size , power_t, max_iter, shuffle, random_state, tol, verbose, warm_start, momentum, nesterovs_momentum, early_stopping, validation_fraction, beta_1, beta_2, epsilon, n_iter_no_change, max_fun)
    rkf = RepeatedKFold(n_splits = n_splits, n_repeats = iterations)
    scores = cross_val_score(ann, input_data, output_data[:,io], cv = rkf)
    score = np.mean(scores)
    print("For N= {} number of neurons, the average score is {:.3f}".format(nn,score))
    results[1,i0] = score

###############################################################################
### Save Scores
###############################################################################

print('Save Cross Validation Training Results to file. \n')
np.savetxt(name_file,results.T,fmt = '%.4f',delimiter = ',',header="n_neurons,{}".format(name_output_short[io]))
print("ANN Training finished for {}.".format(no))
