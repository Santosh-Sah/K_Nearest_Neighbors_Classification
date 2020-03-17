# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:48:29 2020

@author: Santosh Sah
"""

import pandas as pd
from KNearestNeighborsClassificationUtils import readKNearestNeighborsClassificationModel, readKNearestNeighborsClassificationStandardScaler

def predict():
    
    kNearestNeighborsClassification = readKNearestNeighborsClassificationModel()
    kNearestNeighborsClassificationStandardScaler = readKNearestNeighborsClassificationStandardScaler()
    
    inputValue = [[26, 1000]]
    inputValueDataframe = pd.DataFrame(kNearestNeighborsClassificationStandardScaler.transform(inputValue))
    
    predictedValue = kNearestNeighborsClassification.predict(inputValueDataframe.values)
    
    print(predictedValue)

if __name__ == "__main__":
    predict()