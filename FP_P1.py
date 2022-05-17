#File: FP_P1.py
#Author: Rilee Ann Rimke
#Date: Nov 6, 2021

#Program that draws histograms and computes descriptive statistics for
#the columns of data in the breast_cancer_wisconsin dataset 

#referenced: A11_Q1, A10_Q2 (Course Assignments)
    
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def make_graphs(df, c, col):

    fig = plt.figure()
    sp = fig.add_subplot(1,1,1)
    sp.set_title("Histogram of " + col)
    sp.set_xlabel("Value of the attribute")
    sp.set_ylabel("Number of data points")
    sp.hist(df, bins=10, color = c, edgecolor = 'black', linewidth = 1.2, alpha = 0.5)
    plt.xticks(np.arange(2, 12, 2)) #to get the ticks we want for the x-axis
    
    plt.draw()
    
def get_stats(df, m):
    
    #claculate/get mean, median, variance, standard deviation
    avg = df.sum() / m
    median = df.median()
    var_df = ((df - avg) **2).sum() / (m-1)
    stan_dev = var_df ** (1/2)
    
    #print the results
    print("Mean: ", round(avg, 1))
    print("Median: ", round(int(median), 0))
    print("Variance: ", round(var_df, 1))
    print("Standard Deviation: ", round(stan_dev, 1))
    

def main():
 
    #import data from .csv to a Pandas DataFrame, replace '?' with NaN
    df = pd.read_csv('breast_cancer_wisconsin-2.csv', na_values = ['?'])
   #get number of rows and cols
    m, n = df.shape
    #use the mean to impute (replace) the missing values
    df["a7"] = df["a7"].fillna(df["a7"].mean())  #.round(decimals = 1)  
    
    #plot the data using a histogram
    make_graphs(df.loc[:, "a2"], c = "slateblue", col = "A2")
    make_graphs(df.loc[:, "a3"], c = "teal", col = "A3")
    make_graphs(df.loc[:, "a4"], c = "burlywood", col = "A4")
    make_graphs(df.loc[:, "a5"], c = "mediumseagreen", col = "A5")
    make_graphs(df.loc[:, "a6"], c = "coral", col = "A6")
    make_graphs(df.loc[:, "a7"], c = "grey", col = "A7")
    make_graphs(df.loc[:, "a8"], c = "plum", col = "A8")
    make_graphs(df.loc[:, "a9"], c = "darkorchid", col = "A9")
    make_graphs(df.loc[:, "a10"], c = "olive", col = "A10")
   
   
    #calculate the descriptive statistics of the dataset
    print("\nAttribute 2 -----------")
    get_stats(df.loc[:, "a2"], m)
    
    print("\nAttribute 3 -----------")
    get_stats(df.loc[:, "a3"], m)
    
    print("\nAttribute 4 -----------")
    get_stats(df.loc[:, "a4"], m)
    
    print("\nAttribute 5 -----------")
    get_stats(df.loc[:, "a5"], m)
    
    print("\nAttribute 6 -----------")
    get_stats(df.loc[:, "a6"], m)
    
    print("\nAttribute 7 -----------")
    get_stats(df.loc[:, "a7"], m)
    
    print("\nAttribute 8 -----------")
    get_stats(df.loc[:, "a8"], m)
    
    print("\nAttribute 9 -----------")
    get_stats(df.loc[:, "a9"], m)
    
    print("\nAttribute 10 -----------")
    get_stats(df.loc[:, "a10"], m)
  
    
main()
