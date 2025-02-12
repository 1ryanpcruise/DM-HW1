import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#import csv file for train-100-10
df_raw_train = pd.read_csv("HW1_dataset//train-50(1000)-100.csv")

#import csv file for test-100-10
df_raw_test = pd.read_csv("HW1_dataset//test-1000-100.csv")

#df to graph with MSE's of lambdas 0-150
lambdas = [x for x in range(1,151)]
mse_graph = pd.DataFrame(lambdas)
mse_graph.rename({0:"Lambdas"}, axis = 1, inplace=True)
mse_graph['MSE_train'] = 0.0
mse_graph['MSE_train'].astype(float)
mse_graph['MSE_test'] = 0.0
mse_graph['MSE_test'].astype(float)

'''
The code below follows the following general steps:
1. Create dataframes that do not need to run within the lambda loop
2. Create the Lambda ID matrix (lambda*I)
2. Calculate w = (xT*x +lambda*I)^-1*xT*y
4. Calculate y-hat, or XW, for TRAIN and TEST data
5. Calculate the MSE for TRAIN and TEST
6. Graph (lambda, MSE) for TRAIN and TEST

'''
#1a. create relevant matrices for TRAIN
#I'm not sure why, but only the train-100-10.csv file created 2 unnamed columns
df_x_train = df_raw_train.drop(["y"], axis=1)
df_x_train.insert(0,"x0", 1)
df_y_train = df_raw_train[["y"]]
df_xT_train = df_x_train.T
df_xTx_train = df_xT_train.dot(df_x_train)

#1b. create relevant matrices for TEST
df_x_test = df_raw_test.drop(["y"], axis=1)
df_x_test.insert(0,"x0", 1)
df_y_test = df_raw_test[["y"]]

for lam in range(1,151):
    #2. creating lambda ID matrix
    lmbda = lam
    ID_matrix = np.eye(len(df_xTx_train.columns))
    lmbda_ID = lmbda * ID_matrix

    #3. calculating (xT*x + lambda*I) for TRAIN
    df_xTxLmbda_train = df_xTx_train + lmbda_ID
    
    #calculating inverse of (xT*x + lambda*I) for TRAIN
    df_inverse_train = pd.DataFrame(np.linalg.inv(df_xTxLmbda_train))
    
    #rename columns of df_inverse to match with df_y rows for TRAIN
    for x in range(len(df_inverse_train.columns)):
        df_inverse_train.rename({x:f"x{x}"}, axis = 1, inplace = True)
    
    #x-dagger = (xT*x +lambda*I)^-1*xT for TRAIN
    df_dagger_train = df_inverse_train.dot(df_xT_train)

    #final calculation to get w for TRAIN
    df_w = df_dagger_train.dot(df_y_train)

    #rename rows of df_w_train to match with df_x_train columns
    for x in range(len(df_w.index)):
        df_w.rename({x:f"x{x}"}, axis = 0, inplace = True)

    #4. calculating MSE
    #calculating 1/N*||xw - y||^2 for TRAIN
    df_xw_train = df_x_train.dot(df_w)
    df_mse_train = ((df_xw_train - df_y_train)**2)/len(df_x_train)
    
    #calculating 1/N*||xw - y||^2 for TEST
    df_xw_test = df_x_test.dot(df_w)
    df_mse_test = ((df_xw_test - df_y_test)**2)/len(df_x_test)

    #summing all values in matrix df_mse to get ||xw - y||^2 for TRAIN
    mse_train = df_mse_train.sum()
    
    #summing all values in matrix df_mse to get ||xw - y||^2 for TEST
    mse_test = df_mse_test.sum()

    # add MSE_lambda to mse_graph for TRAIN
    mse_graph.loc[mse_graph["Lambdas"] == lam, "MSE_train"] = mse_train.iloc[0]
    
    # add MSE_lambda to mse_graph for TEST
    mse_graph.loc[mse_graph["Lambdas"] == lam, "MSE_test"] = mse_test.iloc[0]
    
    
#mse_graph.to_csv("mse_test.csv", index = False)

print(mse_graph)

#gets the (x,y) coordinate of the minimum test mse
min_x = mse_graph.loc[mse_graph["MSE_test"].idxmin(),"Lambdas"]
min_y = mse_graph["MSE_test"].min()

#create scatter plot of mse_graph
plt.plot(mse_graph['Lambdas'], mse_graph["MSE_train"], label = "Train MSE")
plt.plot(mse_graph['Lambdas'], mse_graph["MSE_test"], label = "Test MSE")
plt.plot(min_x, min_y, marker="o", color='red', label=f"min ({min_x}, {round(min_y,2)})")
plt.legend()
plt.xlabel("Lambda")
plt.ylabel("MSE")
plt.title("MSE graph for train-50(1000)-100 and test-1000-100 \n Lambda 1-150")

plt.show()