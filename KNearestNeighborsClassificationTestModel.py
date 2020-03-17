# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 11:01:45 2020

@author: Santosh Sah
"""

from KNearestNeighborsClassificationUtils import (readKNearestNeighborsClassificationXTest, readKNearestNeighborsClassificationModel,
                                     saveKNearestNeighborsClassificationYPred, readKNearestNeighborsClassificationStandardScaler)

"""
test the model on testing dataset
"""
def testKNearestNeighborsClassificationModel():
    
    X_test = readKNearestNeighborsClassificationXTest()
    kNearestNeighborsClassificationStandardScaler = readKNearestNeighborsClassificationStandardScaler()
    X_test = kNearestNeighborsClassificationStandardScaler.transform(X_test)
    
    kNearestNeighborsClassificationModel = readKNearestNeighborsClassificationModel()
    
    y_pred = kNearestNeighborsClassificationModel.predict(X_test)
    saveKNearestNeighborsClassificationYPred(y_pred)
    
    print(y_pred)
    
if __name__ == "__main__":
    testKNearestNeighborsClassificationModel()