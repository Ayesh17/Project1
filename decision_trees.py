#!/usr/bin/env python
# coding: utf-8

# # Import tools
# In[1]:


import numpy as np

# # Get the data

# In[9]:
from binary_decision_tree import DecisionTree
from random_forest import RandomForest
from real_value_decision_tree import DecisionTreeReal


def build_nparray(data):
    header = np.array(data[0])
    samples = []
    for i in range(1, len(data)):
        sample = []
        for j in range(len(data[i]) - 1):
            sample.append(float(data[i][j]))
        samples.append(sample)
    samples_arr = np.array(samples)
 #sas
    labels = []
    for i in range(1, len(data)):
        labels.append(int(data[i][len(data[0])-1]))
    labels_arr = np.array(labels)

    return samples_arr, labels_arr


file_name = "cat_dog_data.csv"
data = np.genfromtxt(file_name, dtype=str, delimiter=',')
a_samples, a_labels = build_nparray(data)


# # Node class

# In[3]:



class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        ''' constructor '''

        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain

        # for leaf node
        self.value = value


# # Tree class

# In[10]:


class DecisionTreeClassifier():
    def __init__(self, max_depth, min_samples_split =2):
        ''' constructor '''

        # initialize the root of the tree
        self.root = None  # to traverse through the tree

        # stopping conditions
        self.min_samples_split = min_samples_split  # stopping conditions min and max. If the no. of samples are less than min samples, we dont split that node any further
        self.max_depth = max_depth

    def build_tree(self, dataset, curr_depth=0):
        ''' recursive function to build the tree '''

        X, Y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = np.shape(X)

        # split until stopping conditions are met
        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            # find the best split
            best_split = self.get_best_split(dataset, num_samples, num_features)
            # check if information gain is positive
            if best_split["info_gain"] > 0:
                # recur left
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth + 1)
                # recur right
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth + 1)
                # return decision node
                return Node(best_split["feature_index"], best_split["threshold"],
                            left_subtree, right_subtree, best_split["info_gain"])

        # compute leaf node
        leaf_value = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)

    def get_best_split(self, dataset, num_samples, num_features):
        ''' function to find the best split '''

        # dictionary to store the best split
        best_split = {}
        max_info_gain = -float("inf")

        # loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                # check if childs are not null
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_info_gain = self.information_gain(y, left_y, right_y, "gini")
                    # update the best split if needed
                    if curr_info_gain > max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain

        # return best split
        return best_split

    def split(self, dataset, feature_index, threshold):
        ''' function to split the data '''

        dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])
        return dataset_left, dataset_right

    def information_gain(self, parent, l_child, r_child, mode="entropy"):
        ''' function to compute information gain '''

        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        gain = self.entropy(parent) - (weight_l * self.entropy(l_child) + weight_r * self.entropy(r_child))
        return gain

    def entropy(self, y):
        ''' function to compute entropy '''

        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy


    def calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''

        Y = list(Y)
        return max(Y, key=Y.count)

    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''

        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("Feature " + str((tree.feature_index)+1), "=", int(tree.threshold))
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)

    def fit(self, X, Y ):
        ''' function to train the tree '''
        y_arr = np.empty([len(Y), 1])
        for i in range(len(y_arr)):
            y_arr[i][0] = Y[i]
        dataset = np.concatenate((X, y_arr), axis=1)
        self.root = self.build_tree(dataset)

    def predict(self, X):
        ''' function to predict new dataset '''

        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions

    def make_prediction(self, x, tree):
        ''' function to predict a single data point '''

        if tree.value != None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)


# # Train-Test split

# In[11]:




X_train = a_samples
Y_train = a_labels
# Fit the model

# In[12]:

def DT_train_binary(X, Y, max_depth):
    decisionTree = DecisionTree(max_depth)
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
    decisionTreeReal = DecisionTreeReal(max_depth)
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
    classifier = RandomForest(num_of_trees,max_depth)
    classifier.fit(X, Y)
    return classifier


def RF_test_random_forest(X,Y,RF):
    Y_pred = RF.predict(X)
    print(Y_pred[0])
    correct = 0
    wrong = 0

    for i in range(len(Y)):
        if Y_pred[i] == Y[i]:
            correct += 1
        else:
            wrong += 1
    accuracy = correct / (correct + wrong)
    return accuracy





