import numpy as np
import pandas as pd

file_names = ["train-100-10.csv","train-100-100.csv","train-1000-100.csv",
              "train-50(1000)-100.csv","train-100(1000)-100.csv",
              "train-150(1000)-100.csv"]

df_results = pd.DataFrame({"Dataset":file_names,"Lambda":0,"Avg Min MSE":0.0})

#iterates through the 6 dataset files
for file in file_names:
#    loads the dataset csv file
    df_raw = pd.read_csv(f"HW1_dataset//{file}")
    
    #removes extraneous unnamed rows 11 and 12 for train-100-10
    if file == "train-100-10.csv":
        df_raw = df_raw.drop(["Unnamed: 11", "Unnamed: 12"], axis=1)

    #creates dicionary containing dataframe folds 1 - 10
    og_fold_dict =  {}
    a = 0
    b = int(len(df_raw)/10)
    i = 1
    while b <= len(df_raw):
        og_fold_dict[f"df_fold{i}"] = df_raw.iloc[a:b]
        a = b
        b = b + int(len(df_raw)/10) #creates 10 folds of equal size based on size of df
        i += 1

    #creates a copy of the fold_dict to be used to create for the i loop
    fold_dict = og_fold_dict.copy()

#%%
    '''
    mse_calculator calculates the mse for training and test data for a single
    value of lambda, given a lambda, training matrix, and test matrix as input.
    
    It returns the list [lambda, test mse]
    
    '''
    def mse_calculator(lmbda, df_train, df_test):
    
        #create relevant matrices for TRAIN
        df_train_x = df_train.drop(["y"], axis=1)
        df_train_x.insert(0,"x0", 1)
        df_train_y = df_train[["y"]]
        df_train_xT = df_train_x.T
        df_train_xTx = df_train_xT.dot(df_train_x)
        
        #create lambda*I matrix
        lam = lmbda
        ID_matrix = np.eye(len(df_train_xTx.columns))
        lmbda_ID = lam * ID_matrix
        
        #create relevant matrices for TEST
        df_test_x = df_test.drop(["y"], axis=1)
        df_test_x.insert(0,"x0", 1)
        df_test_y = df_test[["y"]]
        
        #3. calculating (xT*x + lambda*I) for TRAIN
        df_train_xTx_Lmbda = df_train_xTx + lmbda_ID
        
        #calculating inverse of (xT*x + lambda*I) for TRAIN
        df_train_inverse = pd.DataFrame(np.linalg.inv(df_train_xTx_Lmbda))
        
        #rename columns of df_inverse to match with df_y rows for TRAIN
        for x in range(len(df_train_inverse.columns)):
            df_train_inverse.rename({x:f"x{x}"}, axis = 1, inplace = True)
        
        #x-dagger = (xT*x +lambda*I)^-1*xT for TRAIN
        df_train_dagger = df_train_inverse.dot(df_train_xT)
    
        #final calculation to get w for TRAIN
        df_w = df_train_dagger.dot(df_train_y)
    
        #rename rows of df_w_train to match with df_x_train columns
        for x in range(len(df_w.index)):
            df_w.rename({x:f"x{x}"}, axis = 0, inplace = True)
    
        #4. calculating MSE  
        #calculating 1/N*||xw - y||^2 for TEST
        df_test_xw = df_test_x.dot(df_w)
        df_test_mse = ((df_test_xw - df_test_y)**2)/len(df_test_x)
        
        #summing all values in matrix df_mse to get ||xw - y||^2 for TEST
        mse_test = df_test_mse.sum()  
        
        return [lam, mse_test.values[0]]
    
    #dataframe to store avg MSE for each ith test fold
    fold_mse = pd.DataFrame({"Test Fold":[1,2,3,4,5,6,7,8,9,10], "test MSE": 0.0})
    
    #df to graph with MSE's of lambdas 0-150
    lambdas = [x for x in range(151)]
    avg_lmbda_mse = pd.DataFrame({"Lambdas":lambdas, 'Avg Fold MSE': 0.0})
    
    for lam in range(151):
        #i loop that creates the test and training matrices for each i [1-10]
        for i in range(1, 11):
            df_test = fold_dict.pop(f"df_fold{i}") #ith fold for test df
            df_train = pd.DataFrame()
            for key in fold_dict: #concat remaining folds to form train df
                df_train = pd.concat([df_train,fold_dict[key]])
            fold_dict = og_fold_dict.copy() #reset fold df
            
            #runs mse_calculator function
            mse_val = mse_calculator(lam, df_train, df_test)
            
            #adds mse for ith fold test to fold_mse df
            fold_mse.loc[fold_mse["Test Fold"] == i, "test MSE"] = mse_val[1]       
        
        #averages the MSEs for folds 1-10 for lambda i
        avg_mse = fold_mse["test MSE"].mean()
        
        #adds average mse to avg_lmbda_mse df 
        avg_lmbda_mse.loc[avg_lmbda_mse['Lambdas'] == lam, 'Avg Fold MSE'] = avg_mse
    
    
    min_avg_mse = avg_lmbda_mse["Avg Fold MSE"].min()
    min_lmbda = avg_lmbda_mse.loc[avg_lmbda_mse["Avg Fold MSE"].idxmin(),"Lambdas"]
    
    df_results.loc[df_results["Dataset"] == file, "Lambda"] = min_lmbda
    df_results.loc[df_results["Dataset"] == file, "Avg Min MSE"] = min_avg_mse

print(df_results)
df_results.to_csv("Prob2_avg_lmbda_mse.csv", index = False)  

'''
***PERONAL NOTE***
fix the idx min for this file and all of 1a and 1b
- it is currently grabbing the index when it should be grabbing lambda, meaning
that the min lambdas for the graphs starting with lambda =1 are off by 1.
'''
        
    


