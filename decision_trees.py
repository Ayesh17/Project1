#!/usr/bin/env python
# coding: utf-8

import numpy as np

from binary_decision_tree import DecisionTreeClass
from random_forest import RandomForestClass
from real_value_decision_tree import DecisionTreeClassRealClass


def DT_train_binary(X, Y, max_depth):
    decisionTree = DecisionTreeClass(max_depth)
    decisionTree.fit(X, Y)
    decisionTree.print_tree()
    return decisionTree

def DT_test_binary(X, Y, DT):
    Y_pred = DT.predict(X)
    correct = 0
    wrong = 0

    for i in range (len(Y)):
        if Y_pred[i] == Y[i]:
            correct +=1
        else:
            wrong +=1
    accuracy = correct / (correct+wrong)
    return accuracy

#X must be a 2D array
def DT_make_prediction(X, DT):
    Y_pred = DT.predict(X)
    return Y_pred

def DT_train_real(X, Y, max_depth):
    decisionTreeReal = DecisionTreeClassRealClass(max_depth)
    decisionTreeReal.fit(X, Y)
    decisionTreeReal.print_tree()
    return decisionTreeReal

def DT_test_real(X, Y, DT):
    Y_pred = DT.predict(X)
    correct = 0
    wrong = 0

    for i in range (len(Y)):
        if Y_pred[i] == Y[i]:
            correct +=1
        else:
            wrong +=1
    accuracy = correct / (correct+wrong)
    return accuracy

def RF_build_random_forest(X,Y, max_depth, num_of_trees):
    classifier = RandomForestClass(num_of_trees, max_depth)
    classifier.fit(X, Y)
    return classifier


def RF_test_random_forest(X,Y,RF):
    Y_pred = RF.predict(X,Y)
    correct = 0
    wrong = 0

    for i in range(len(Y)):
        if Y_pred[i] == Y[i]:
            correct += 1
        else:
            wrong += 1
    accuracy = correct / (correct + wrong)
    return accuracy





