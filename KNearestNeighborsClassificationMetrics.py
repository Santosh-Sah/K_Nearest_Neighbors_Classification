# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:39:09 2020

@author: Santosh Sah
"""

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from KNearestNeighborsClassificationUtils import (readKNearestNeighborsClassificationYTest, readKNearestNeighborsClassificationYPred)

"""

calculating KNearestNeighborsClassification confussion matrix

"""
def testKNearestNeighborsClassificationConfussionMatrix():
    
    y_test = readKNearestNeighborsClassificationYTest()
    y_pred = readKNearestNeighborsClassificationYPred()
    
    kNearestNeighborsClassificationConfussionMatrix = confusion_matrix(y_test, y_pred)
    print(kNearestNeighborsClassificationConfussionMatrix)
    
    """
    Below is the confussion matrix
    [[55  3]
    [ 1 21]]
    
    """
"""
calculating accuracy score

"""

def testKNearestNeighborsClassificationAccuracy():
    
    y_test = readKNearestNeighborsClassificationYTest()
    y_pred = readKNearestNeighborsClassificationYPred()
    
    kNearestNeighborsClassificationConfussionAccuracy = accuracy_score(y_test, y_pred)
    
    print(kNearestNeighborsClassificationConfussionAccuracy) #.9125%

"""
calculating classification report

"""

def testKNearestNeighborsClassificationClassificationReport():
    
    y_test = readKNearestNeighborsClassificationYTest()
    y_pred = readKNearestNeighborsClassificationYPred()
    
    kNearestNeighborsClassificationConfussionClassificationReport = classification_report(y_test, y_pred)
    
    print(kNearestNeighborsClassificationConfussionClassificationReport)
    
    """
              precision    recall  f1-score   support

          0       0.98      0.95      0.96        58
          1       0.88      0.95      0.91        22

avg / total       0.95      0.95      0.95        80
    """
    
if __name__ == "__main__":
    #testKNearestNeighborsClassificationConfussionMatrix()
    #testKNearestNeighborsClassificationAccuracy()
    testKNearestNeighborsClassificationClassificationReport()