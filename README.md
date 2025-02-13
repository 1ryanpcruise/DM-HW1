# DM-HW1
Homework 1 for Data Mining Spring 2025\
Author: Ryan Cruise, SJ - 2/12/2025\
<br/>
**Overview**\
You can find my written responses in the "HW1_responses" word document. "HW1_responses" also has details on how the graphs correspond to each question and dataset. All of my graphs are in the "graphs" subfolder. If you have issues loading the
.csv files, make sure you are in the correct directory to access the files; all of the pd.read_csv() functions assume
that the working directory is the directory that contains the "HW1_dataset" folder.\
<br/>

**Homework 1(a)**\
I created one python file for each of the six datasets (see below). Each file trains on the training dataset and 
tests on the testing dataset. For example, hw_1a_100_10.py trains the model on train-100-10.csv and tests the model
using test-100-10.csv. The corresponding graphs can be found in the "graphs" subfolder.\
<br/>
Python files for **Homework 1(a):** 
- hw1_1a_100_10.py
- hw1_1a_100_100.py
- hw1_1a_1000_100.py
- hw1_1a_50_1000_100.py
- hw1_1a_100_1000_100.py
- hw1_1a_150_1000_100.py
<br/>

Programming flow for **Homework 1(a)**
- All of the 6 python files for 1(a) follow these steps:
1. load train and test data
2. create df to store MSE's for each lambda
3. calculate all of the matrices for test and train that do not use lambda
4. begin lambda loop (0-150) and calculate matrices for train and test that use lambda
5. store values in the df from step 2
6. find the minimum MSE and its corresponding lambda
7. graph the df from step 2

**Homework 1(b)**\
Similarly to homework 1(a), I created one python file for each dataset. Each file trains on the dataset I created per the
instructions and tests on test-1000-100.csv. The corresponding graphs can be found in the "graphs" subfolder.\
<br/>
Python files for **Homework 1(b):**
- hw1_1b_50_1000_100.py
- hw1_1b_100_100.py
- hw1_1b_100_1000_100.py
<br/>

Programming flow for **Homework 1(b)**
- All of the 3 python files for 1(b) follow these steps:
- the only difference between the code of 1(a) and 1(b) is that lambda starts at 1 instead of 0.
1. load train and test data
2. create df to store MSE's for each lambda
3. calculate all of the matrices for test and train that do not use lambda
4. begin lambda loop (1-150) and calculate matrices for train and test that use lambda
5. store values in the df from step 2
6. find the minimum MSE and its corresponding lambda
7. graph the df from step 2

**Homework 2(a)**\
I only have one python file for homework #2 that does the calculations for all of the files. This python file exports a csv
of the dataframe that contains all of the MSE values for the 10-fold CV for each dataset. Because this file runs for all six datasets, it slow in comparison to the other files. On my Microsoft Surface Pro 9, it took about 3.5 - 4 minutes to run.\
<br/>
Python file for **Homework 2(a):**
- hw1_prob2.py
<br/>

Programming flow for **Homework 2(a)**
- The python file for question 2 follows the following steps:
1. create an empty dataframe that will store the final MSEs for each of the 6 datasets
2. begin the outermost loop that iterates through the list of the filenames of each of the 6 datasets
3. create a dictionary that contains all of the folds for each dataset
4. create a function that calculates the MSE given the lambda, training df, and test df and returns the lambda and test MSE
5. create dataframes to store test MSE per fold and the average MSE for each lambda
6. begin lambda loop (0-159)
7. begin fold loop (1-10)
8. for each fold, the ith fold is stored as the test df and the remaining folds are used as the training df
9. run the mse calculator given the lambda, the remaining training folds, and the ith test fold
10. calculate and store the mean test MSE for the 10 runs (1 run for each fold)
11. calculate and store the final minimum test MSE given the 150 lambda trials for each dataset
12. print to terminal and export results as csv

**Homework 3**\
I created one python file for Lambda = 1, Lambda = 25, and Lambda = 150. The corresponding graphs can be found in the "graphs" subfolder.\
<br/>
Python files for **Homework 3:**
- hw1_prob3_lam1
- hw1_prob3_lam25
- hw1_prob3_lam150

Programming flow for **Homework 3**
- All of the 3 python files for 3 follow these steps:
- the only difference between the code of these files is the value of lambda, which I just hardcoded
1. create a function that calculates the test and train MSE (the same function from problem 2)
2. load train and test data
3. create a value that will determine the number of repititions (trials) that the code will run
4. create an integer list of all the sizes of subset dataframes
5. create a df that stores the MSEs for each repition (trial) before averaging
6. create a df that stores the final average MSEs for train and test
7. begin subset loop that iterates through each subset size from step 4
8. begin repition (trial) loop determined by step 3
9. create a sample df based on the current subset
10. calculate train and test MSE (lambda is hardcoded here)
11. average the train and test MSEs for all repitions (trials)
12. store final averaged MSEs tor train and test for each subset
13. grapgh 


