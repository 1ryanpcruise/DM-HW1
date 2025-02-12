import pandas as pd
import numpy as np
import random as rand
import matplotlib.pyplot as plt

'''
mse_calculator calculates the mse for training and test data for a single
value of lambda, given a lambda, training matrix, and test matrix as input.

It returns the list [lambda, train mse,  test mse]

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
    #calculating 1/N*||xw - y||^2 for TRAIN
    df_train_xw = df_train_x.dot(df_w)
    df_mse_train = ((df_train_xw - df_train_y)**2)/len(df_train_x)
    
    #calculating 1/N*||xw - y||^2 for TEST
    df_test_xw = df_test_x.dot(df_w)
    df_test_mse = ((df_test_xw - df_test_y)**2)/len(df_test_x)
    
    #summing all values in matrix df_mse to get ||xw - y||^2 for TRAIN
    mse_train = df_mse_train.sum()
    
    #summing all values in matrix df_mse to get ||xw - y||^2 for TEST
    mse_test = df_test_mse.sum()  
    
    return [lam, mse_train.values[0], mse_test.values[0]]


#import csv file for train-100-10
df_raw_train = pd.read_csv("HW1_dataset//train-1000-100.csv")

#import csv file for test-100-10
df_raw_test = pd.read_csv("HW1_dataset//test-1000-100.csv")

#number of times process is repeated before averaging
trials = 50

#list of the number of observances for each subset
subset = [50,75,125,150,225,275,300,325,350,425,475,500,550,600,675,725,800,850,950]

#df to track MSE across number of trials
df_trial = pd.DataFrame({"trials":[x for x in range(1,trials+1)]})
for x in subset:
    df_trial[f"train {x}"] = 0.0
    df_trial[f"test {x}"] = 0.0

#df to store final avg mse's to graph
df_lmbda = pd.DataFrame({"subsets":subset, "train MSE":0.0, "test MSE":0.0})

#%%
#iterate through different subsets
for sub in subset:
    for x in range(1,trials+1): # num of times process is repeated
        df_subset = df_raw_train.sample(sub) #rand sample of subset
        mse = mse_calculator(150, df_subset, df_raw_test) #calculates mse for each subset
        
        #store trial value in df_trial for train and test
        df_trial.loc[df_trial["trials"] == x, f"train {sub}"] = mse[1]
        df_trial.loc[df_trial["trials"] == x, f"test {sub}"] = mse[2]
        
    #average trial MSE's for train and test
    avg_subset_mse_train = df_trial[f"train {sub}"].mean() 
    avg_subset_mse_test = df_trial[f"test {sub}"].mean() 
    
    #stor avg'd MSE's to df_lmbda for graphing
    df_lmbda.loc[df_lmbda["subsets"] == sub, "train MSE"] = avg_subset_mse_train
    df_lmbda.loc[df_lmbda["subsets"] == sub, "test MSE"] = avg_subset_mse_test
    
#sort subsets ascending for graphing
df_lmbda = df_lmbda.sort_values(by="subsets")

#exports df_lmbda to csv
df_lmbda.to_csv("prob3_lmbda_1.csv", index = False)

print(df_trial)
print(df_lmbda)
   
#create scatter plot of lmbda_1 graph
plt.plot(df_lmbda["subsets"], df_lmbda["train MSE"], label = "Train MSE")
plt.plot(df_lmbda["subsets"], df_lmbda["test MSE"], label = "Test MSE")
plt.legend()
#plt.ylim(2,6)
plt.xlabel("Subsets")
plt.ylabel("MSE")
plt.title("Learning Curve for Lambda = 150")

plt.show()