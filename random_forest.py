from collections import Counter

import numpy as np

from binary_decision_tree import DecisionTree


def bootstrap_sample(samples, labels):
    n_samples = samples.shape[0]
    idxs = np.random.choice(n_samples, n_samples, replace=True)
    return samples[idxs], labels[idxs]

def most_common_label(labels):
    counter = Counter(labels)
    most_common = counter.most_common(1)[0][0]
    return most_common


class RandomForest:

    def __init__(self, n_trees=10, min_samples_split=2,
                 max_depth=100, n_feats=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []

    def fit(self, samples, labels):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(min_samples_split=self.min_samples_split,
                                max_depth=self.max_depth, n_feats=self.n_feats)
            samples_, labels_ = bootstrap_sample(samples, labels)
            tree.fit(samples_, labels_)
            self.trees.append(tree)

    def predict(self, samples):
        tree_preds = np.array([tree.predict(samples) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        labels_pred = [most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(labels_pred)