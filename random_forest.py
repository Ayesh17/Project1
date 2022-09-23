#imports
import numpy as np
from binary_decision_tree import DecisionTreeClass

#get bootsrap smaples and labels
def bootstrap_sample(samples, labels):
    num_total_samples = samples.shape[0]
    num_train_samples = int(num_total_samples/10)
    ids = np.random.choice(num_total_samples, num_train_samples, replace=True)
    return samples[ids], labels[ids]

#get the most common label(0,1)
def most_common_label(labels):
    zeros_count = 0
    ones_count = 0
    for i in range(len(labels)):
        if labels[i] == 0:
             zeros_count += 1
        else:
            ones_count += 1
    if zeros_count > ones_count:
        most_common = 0
    else:
        most_common = 1
    return most_common

#Random Forest class
class RandomForestClass:

    def __init__(self, num_trees, max_depth, min_samples_split=2):
        self.num_trees = num_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.trees = []

    #fit
    def fit(self, samples, labels):
        self.trees = []
        for _ in range(self.num_trees):
            tree = DecisionTreeClass(min_samples_split=self.min_samples_split, max_depth=self.max_depth)
            samples_, labels_ = bootstrap_sample(samples, labels)
            tree.fit(samples_, labels_)
            self.trees.append(tree)

    #prediction
    def predict(self, samples, labels):
        tree_preds = np.array([tree.predict(samples) for tree in self.trees])
        for tree in range(len(tree_preds)):
            correct =0
            wrong =0
            for pred in range(len(tree_preds[0])):
                if tree_preds[tree][pred] == labels[pred]:
                    correct +=1
                else:
                    wrong+=1
            accuracy = correct / (correct+wrong)
            print(f"DT {tree} : {accuracy}")

        tree_preds = np.swapaxes(tree_preds, 0, 1)
        labels_pred = [most_common_label(tree_pred) for tree_pred in tree_preds]

        return np.array(labels_pred)