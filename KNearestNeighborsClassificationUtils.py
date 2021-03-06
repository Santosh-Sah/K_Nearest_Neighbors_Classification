# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:09:12 2020

@author: Santosh Sah
"""

"""
importing the libraries
"""

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

"""
Import dataset and read specific column. Split the dataset in training and testing set.
"""
def importKNearestNeighborsClassificationDataset(kNearestNeighborsClassificationDatasetFileName):
    
    kNearestNeighborsClassificationDataset = pd.read_csv(kNearestNeighborsClassificationDatasetFileName)
    X = kNearestNeighborsClassificationDataset.iloc[:, [2, 3]].values
    y = kNearestNeighborsClassificationDataset.iloc[:, 4].values
    
    #spliting the dataset into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    return X_train, X_test, y_train, y_test

"""
Save standard scalar object as a pickel file. This standard scalar object must be used to standardized the dataset for training, testing and new dataset.
To use this standard scalar object we need to read it and then use it.
"""
def saveKNearestNeighborsClassificationStandardScaler(kNearestNeighborsClassificationStandardScalar):
    
    #Write KNearestNeighborsClassificationStandardScaler in a picke file
    with open("KNearestNeighborsClassificationStandardScaler.pkl",'wb') as KNearestNeighborsClassificationStandardScaler_Pickle:
        pickle.dump(kNearestNeighborsClassificationStandardScalar, KNearestNeighborsClassificationStandardScaler_Pickle, protocol = 2)

"""
Save training and testing dataset
"""
def saveTrainingAndTestingDataset(X_train, X_test, y_train, y_test):
    
    #Write X_train in a picke file
    with open("X_train.pkl",'wb') as X_train_Pickle:
        pickle.dump(X_train, X_train_Pickle, protocol = 2)
    
    #Write X_test in a picke file
    with open("X_test.pkl",'wb') as X_test_Pickle:
        pickle.dump(X_test, X_test_Pickle, protocol = 2)
    
    #Write y_train in a picke file
    with open("y_train.pkl",'wb') as y_train_Pickle:
        pickle.dump(y_train, y_train_Pickle, protocol = 2)
    
    #Write y_test in a picke file
    with open("y_test.pkl",'wb') as y_test_Pickle:
        pickle.dump(y_test, y_test_Pickle, protocol = 2)

"""
Save SupportVectorMachineModel as a pickle file.
"""
def saveKNearestNeighborsClassificationModel(kNearestNeighborsClassificationModel):
    
    #Write KNearestNeighborsClassificationModel as a picke file
    with open("KNearestNeighborsClassificationModel.pkl",'wb') as KNearestNeighborsClassificationModel_Pickle:
        pickle.dump(kNearestNeighborsClassificationModel, KNearestNeighborsClassificationModel_Pickle, protocol = 2)

"""
read KNearestNeighborsClassificationStandardScalar from pickel file
"""
def readKNearestNeighborsClassificationStandardScaler():
    
    #load KNearestNeighborsClassificationStandardScaler object
    with open("KNearestNeighborsClassificationStandardScaler.pkl","rb") as KNearestNeighborsClassificationStandardScaler:
        kNearestNeighborsClassificationStandardScalar = pickle.load(KNearestNeighborsClassificationStandardScaler)
    
    return kNearestNeighborsClassificationStandardScalar

"""
read KNearestNeighborsClassificationModel from pickle file
"""
def readKNearestNeighborsClassificationModel():
    
    #load KNearestNeighborsClassificationModel model
    with open("KNearestNeighborsClassificationModel.pkl","rb") as KNearestNeighborsClassificationModel:
        kNearestNeighborsClassificationModel = pickle.load(KNearestNeighborsClassificationModel)
    
    return kNearestNeighborsClassificationModel

"""
read X_train from pickle file
"""
def readKNearestNeighborsClassificationXTrain():
    
    #load X_train
    with open("X_train.pkl","rb") as X_train_pickle:
        X_train = pickle.load(X_train_pickle)
    
    return X_train

"""
read X_test from pickle file
"""
def readKNearestNeighborsClassificationXTest():
    
    #load X_test
    with open("X_test.pkl","rb") as X_test_pickle:
        X_test = pickle.load(X_test_pickle)
    
    return X_test

"""
read y_train from pickle file
"""
def readKNearestNeighborsClassificationYTrain():
    
    #load y_train
    with open("y_train.pkl","rb") as y_train_pickle:
        y_train = pickle.load(y_train_pickle)
    
    return y_train

"""
read y_test from pickle file
"""
def readKNearestNeighborsClassificationYTest():
    
    #load y_test
    with open("y_test.pkl","rb") as y_test_pickle:
        y_test = pickle.load(y_test_pickle)
    
    return y_test

"""
save y_pred as a pickle file
"""

def saveKNearestNeighborsClassificationYPred(y_pred):
    
    #Write y_red in a picke file
    with open("y_pred.pkl",'wb') as y_pred_Pickle:
        pickle.dump(y_pred, y_pred_Pickle, protocol = 2)

"""
read y_predt from pickle file
"""
def readKNearestNeighborsClassificationYPred():
    
    #load y_test
    with open("y_pred.pkl","rb") as y_pred_pickle:
        y_pred = pickle.load(y_pred_pickle)
    
    return y_pred