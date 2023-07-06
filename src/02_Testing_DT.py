import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
sys.stdout = open('../results/DT_Test_results.txt', 'w')

###############################################################################
### Load data and set algorithm parameters
###############################################################################

conditions = ['fav','unfav']
name_output = ['Coordination number','Surface coverage','Conductivity','Void fraction']
n_test_samples = 73
np.set_printoptions(suppress = True)

### identified hyperparameters during Training
max_depth_fav = [2,8,9,8]
min_samples_split_fav = [5,6,2,7]
max_depth_unfav = [2,3,4,4]
min_samples_split_unfav = [5,5,4,3]

file_LBM = "../data/LBM_Results.xlsx"
xl = pd.ExcelFile(file_LBM)

for ic,cond in enumerate(conditions):

    data_LBM = pd.read_excel(xl,skiprows = [1],sheet_name=cond)
    data_LBM = np.array(data_LBM)
    
    input_data_training = data_LBM[:n_test_samples,0:4]
    output_data_training = data_LBM[:n_test_samples,4:8]
    
    input_data_testing = data_LBM[n_test_samples:,0:4]
    output_data_testing = data_LBM[n_test_samples:,4:8]

    ###############################################################################
    ### Run Testing Procedure and Print Output
    ###############################################################################
    
    print("################################################")
    print("Performance Testing for Decision Tree algorithm \nunder {}orable conditions".format(cond))
    print("################################################")
    
    for io,param in enumerate(name_output):
            
        if cond == 'fav':        
            max_depth,min_samples_split = max_depth_fav[io], min_samples_split_fav[io]
        elif cond == 'unfav':
            max_depth,min_samples_split = max_depth_unfav[io], min_samples_split_unfav[io]
    
        print("\n#####################################\nOutput Parameter: {}".format(param)) 
        print('\nSelected Hyperparameters: \n max_depth  =  {}\n min_samples_split = {}'.format(max_depth, min_samples_split))
        tree = DecisionTreeRegressor(max_depth = max_depth, min_samples_split = min_samples_split, random_state = 42)
        tree.fit(input_data_training,output_data_training[:,io])
        y_pred = tree.predict(input_data_testing)
        
        print('\nAI predicted values: \n {}'.format(y_pred))
        print('LBM simulation values \n {}'.format(output_data_testing[:,io]))
        
        print("\nTraining data set score (NSE): {:.4f}".format(tree.score(input_data_training, output_data_training[:,io])))
        print("\nTest data set:")
        #print("Test set score: {:.4f}".format(tree.score(input_data_testing, output_data_testing[:,io])))
        print("NSE = {:.4f}".format(r2_score(output_data_testing[:,io], y_pred)))
        print("MSE = {:.4f}".format(mean_squared_error(output_data_testing[:,io], y_pred)))
        print("MAE = {:.4f}".format(mean_absolute_error(output_data_testing[:,io], y_pred)))

#         test_results[io,ic] = r2_score(output_data_testing[:,io], y_pred)
#         test_results[4+io,ic] = mean_squared_error(output_data_testing[:,io], y_pred)
#         test_results[8+io,ic] = mean_absolute_error(output_data_testing[:,io], y_pred)
        
# np.savetxt('../results/Test_results_DT.csv',test_results,fmt = '%.4f',delimiter=',')    
