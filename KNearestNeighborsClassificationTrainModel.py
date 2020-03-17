# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:02:07 2020

@author: Santosh Sah
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from KNearestNeighborsClassificationUtils import (saveKNearestNeighborsClassificationModel, readKNearestNeighborsClassificationXTrain, 
                                                  readKNearestNeighborsClassificationYTrain, saveKNearestNeighborsClassificationStandardScaler)

"""
Train KNearestNeighborsClassification model 
"""
def trainKNearestNeighborsClassificationModel():
    
    kNearestNeighborsClassificationStandardScalar = StandardScaler()
    
    X_train = readKNearestNeighborsClassificationXTrain()
    y_train = readKNearestNeighborsClassificationYTrain()
    
    kNearestNeighborsClassificationStandardScalar.fit(X_train)
    saveKNearestNeighborsClassificationStandardScaler(kNearestNeighborsClassificationStandardScalar)
    
    X_train = kNearestNeighborsClassificationStandardScalar.transform(X_train)
    
    kNearestNeighborsClassification = KNeighborsClassifier(n_neighbors = 5, metric = "minkowski", p = 2)
    kNearestNeighborsClassification.fit(X_train, y_train)
    
    saveKNearestNeighborsClassificationModel(kNearestNeighborsClassification)

if __name__ == "__main__":
    trainKNearestNeighborsClassificationModel()