# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:46:53 2020

@author: Santosh Sah
"""

from KNearestNeighborsClassificationUtils import (importKNearestNeighborsClassificationDataset, saveTrainingAndTestingDataset)

def preprocess():
    
    X_train, X_test, y_train, y_test = importKNearestNeighborsClassificationDataset("K_Nearest_Neighbors_Classification_Social_Network_Ads.csv")
    
    saveTrainingAndTestingDataset(X_train, X_test, y_train, y_test)
    

if __name__ == "__main__":
    preprocess()