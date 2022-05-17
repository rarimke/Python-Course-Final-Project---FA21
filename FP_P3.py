#File: FP_P3.py
#Author: Rilee Ann Rimke
#Date: Dec 5, 2021

#Program that imports data from the breast_cancer_wisconsin dataset, and uses the k-means clustering algorithm
# and calculates the error rate of the predicted clusters from the algorithm.

#referenced (phase 2): https://stackoverflow.com/a/64342136
#https://www.kite.com/python/answers/how-to-merge-two-pandas-series-into-a-dataframe-in-python
#https://stackoverflow.com/a/35387129  
#https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.subtract.html
#https://www.kite.com/python/answers/how-to-sum-rows-of-a-pandas-dataframe-in-python
#https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html
#https://www.geeksforgeeks.org/how-to-create-an-empty-dataframe-and-append-rows-columns-to-it-in-pandas/

#references (phase 3): https://stackoverflow.com/questions/35277075/python-pandas-counting-the-occurrences-of-a-specific-value
#https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.lt.html#pandas.DataFrame.lt

import pandas as pd
from random import randrange

def ErrorRate(df, m):

    class_2_count = (df['Class'] == 2).sum()
    class_4_count = (df['Class'] == 4).sum()
    predicted_2_count = (df["Predicted_Class"] == 2).sum()
    predicted_4_count = (df["Predicted_Class"] == 4).sum()
    #if class is 4 but predicted class was 2, then predicted < class and vice versa 
    #j is df['predicted'] == 2 & df['class'] == 4 
    #k is df['predicted'] == 4 & df['class'] == 2 
    error_2_numerator = (df["Predicted_Class"].lt(other = df['Class'], axis = 0)).sum()
    error_4_numerator = (df["Predicted_Class"].gt(other = df['Class'], axis = 0)).sum()
    #print("error_2_numerator: ", error_2_numerator) #for debugging
    #print("error_4_numerator: ", error_4_numerator) #for debugging
    error_2 = (error_2_numerator/predicted_2_count) * 100
    error_4 = (error_4_numerator/predicted_4_count) * 100
    error_t = ((error_2_numerator + error_4_numerator)/ m) * 100
    #print("error_2: ", error_2) #for debugging
    #print("error_4: ", error_4) #for debugging
    #print("error_t: ", error_t) #for debugging
    print("============================================================")
    #print("DATA: ", m)
    #print("Class 2: ", class_2_count)
    #print("Class 4: ", class_4_count)
    print("\nTOTAL ERROR RATE:", round(error_t, 2), "%")
    print("Class 2: ", round(error_2, 2), "%")
    print("Class 4: ", round(error_4, 2), "%")
    #Swap predicted class if total error rate > 50% (used label_swap.py from Canvas)
    if error_t > 50:
        #swap the 2s and 4s in the predicted class
        predicted = df["Predicted_Class"]
        predicted = predicted.replace({2:4, 4:2})
        df["Predicted_Class"] = predicted
        #calculate all the error rates again
        predicted_2_count = (df["Predicted_Class"] == 2).sum()
        predicted_4_count = (df["Predicted_Class"] == 4).sum()
        error_2_numerator = (df["Predicted_Class"].lt(other = df['Class'], axis = 0)).sum()
        error_4_numerator = (df["Predicted_Class"].gt(other = df['Class'], axis = 0)).sum()
        error_2 = (error_2_numerator/predicted_2_count) * 100
        error_4 = (error_4_numerator/predicted_4_count) * 100
        error_t = ((error_2_numerator + error_4_numerator)/ m) * 100
        print("\nTOTAL ERROR RATE (after flipping predicted class):", round(error_t, 2), "%")
        print("Class 2: ", round(error_2, 2), "%")
        print("Class 4: ", round(error_4, 2), "%")
    

def get_mus(mu_2, mu_4, df2):
    #Assignment step - calculate Euclidean distances
    dist_2 = (((df2 - mu_2.values) ** 2).sum(axis=1))** (1/2)
    dist_4 = (((df2 - mu_4.values) ** 2).sum(axis=1))** (1/2)
    boolean = dist_2 < dist_4  #if true then class 2, else class 4
    class_col = boolean.replace([True, False], [2, 4])
    df3 = df2.assign(class_ = class_col.values)
   
    #Recalculation Step -> update the means now that data is sorted into clusters
    #getting the rows of the classes for the new means
    new_mu_2 = df3.loc[df3['class_'] == 2]
    new_mu_4 = df3.loc[df3['class_'] == 4]
   
    new_mu_2 = new_mu_2.drop(["class_"], axis = 1) # axis = 1 means to drop cols
    new_mu_4 = new_mu_4.drop(["class_"], axis = 1) # axis = 1 means to drop cols
    
    #getting the dimensions
    x1, y1 = new_mu_2.shape
    x2, y2 = new_mu_4.shape
    #getting the means, actually the new_mu values
    new_mu_2 = ((new_mu_2).sum(axis = 0)) / x1
    new_mu_4 = ((new_mu_4).sum(axis = 0)) / x2
    
   # print("class_col in getmus: ", class_col.head(21)) #for debugging
    
    return new_mu_2, new_mu_4, class_col

def main():
 
    #import data from .csv to a Pandas DataFrame, replace '?' with NaN
    df = pd.read_csv('breast_cancer_wisconsin.csv', na_values = ['?'])
    #get number of rows and cols of the dataset
    m, n = df.shape #699 rows, 11 columns -> col 1 - col 9 are a2-a10
    #use the mean to impute (replace) the missing values
    df["a7"] = df["a7"].fillna(df["a7"].mean())  #.round(decimals = 1)  
    #create a dataframe with columns removed from the data to use for the calculations
    df2 = df.drop(["scn", "class"], axis = 1) # axis = 1 means to drop cols
    #print(df.head(10))#for debugging
    #print(df2.head(10))#for debugging
    
#Initialization Step
    #get random numbers to use for initialization step of k-means algorithm, K = 2
    r1 = randrange(0, m-1)
    r2 = randrange(0, m-1)
    #initialize mu_2 and mu_4
    mu_2 = df2.loc[r1:r1, :]
    mu_4 = df2.loc[r2:r2, :]
    #print("mu2\n", mu_2)#for debugging
    #print("mu4\n",mu_4)#for debugging

#Assignment step - calculate Euclidean distances
    dist_2 = (((df2 - mu_2.values) ** 2).sum(axis=1))** (1/2)
    dist_4 = (((df2 - mu_4.values) ** 2).sum(axis=1))** (1/2)
    #print("dist2\n", dist_2)#for debugging
    #print("dist4\n",dist_4)#for debugging
    
    #assign data points to clusters
    boolean = dist_2 < dist_4  #if true then class 2, else class 4
    #print(boolean.dtypes)#for debugging
    class_col = boolean.replace([True, False], [2, 4])
    #print("boolean",boolean)#for debugging
    #print("class columns", class_col)#for debugging
    #print(class_col.shape)#for debugging
    
    #creating a data frame with the "class" column -> to be able to sort the data clusters
    df3 = df2.assign( class_ = class_col.values)
    #print(df3.head(10)) #for debugging

#Recalculation Step -> update the means now that data is sorted into clusters
    #getting the rows of the classes for the new means
    new_mu_2 = df3.loc[df3['class_'] == 2] #creates an object
    new_mu_4 = df3.loc[df3['class_'] == 4]
    #print("new_mu_2 type: ", new_mu_2.dtypes)#print statements for debugging
    #print("New-mu2\n", new_mu_2)
    #print("New-mu4\n",new_mu_4)
    new_mu_2 = new_mu_2.drop(["class_"], axis = 1) # axis = 1 means to drop cols
    new_mu_4 = new_mu_4.drop(["class_"], axis = 1) # axis = 1 means to drop cols
    #print("New-mu2\n", new_mu_2) #for debugging
    #print("New-mu4\n",new_mu_4)
    
    #getting the dimensions
    x1, y1 = new_mu_2.shape
    x2, y2 = new_mu_4.shape
    
    #getting the means, actually the new_mu values
    new_mu_2 = ((new_mu_2).sum(axis = 0)) / x1
    new_mu_4 = ((new_mu_4).sum(axis = 0)) / x2
    #print("New-mu2\n", new_mu_2) #for debugging
    #print("New-mu4\n",new_mu_4)
    #print("class_col before getmus: ", class_col.head(21)) #for debugging
#Repeat Assignment Step and Recalculation Step until:
 #all data points do not change their cluster(the new mean is equal to the previously calculated mean)
 #or Assignment Step and Recalculation Step iterated 1500 times -> change to 20, 1500 way too many
    i = 0
    while True or i < 20:
        #get boolean values, checks if previous mean is equal to the new mean
        mu_2_result = mu_2.equals(other = new_mu_2) 
        mu_4_result = mu_4.equals(other = new_mu_4)
        
        if mu_2_result and mu_4_result: #check if the means are equal, if they are equal then stop
            print("------------------------Final Means------------------------")
            print("mu_2:", new_mu_2.values)
            print("mu_4:", new_mu_4.values)
            print("---------------------Cluster Assignment---------------------")
            data_df = pd.DataFrame()
            data_df["ID"] = df["scn"]
            data_df["Class"] = df["class"]
            data_df["Predicted_Class"] = new_class_col
            print(data_df.head(21))
            #get number of rows and cols of the dataset
            o, p = data_df.shape      
            ErrorRate(data_df, o)
            return False
        else:
        #try again (find new means)
            mu_2 = new_mu_2
            mu_4 = new_mu_4
            new_mu_2, new_mu_4, new_class_col = get_mus(mu_2, mu_4, df2)
            i += 1
            #print("i: ", i) #for debugging
    

main()

