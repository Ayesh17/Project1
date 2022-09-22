import numpy as np

def entropy( labels):
    hist = np.bincount( labels)
    ps = hist / len( labels)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

class Node:

    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:

    def __init__(self, max_depth, min_samples_split=2, n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self, samples, labels):
        self.n_feats = samples.shape[1] if not self.n_feats else min(self.n_feats, samples.shape[1])
        self.root = self._grow_tree(samples, labels)

    def predict(self, samples):
        return np.array([self._traverse_tree(samples, self.root) for samples in samples])

    def _grow_tree(self, samples, labels, depth=0):
        n_samples, n_features = samples.shape
        n_labels = len(np.unique( labels))

        # stopping criteria
        if (depth >= self.max_depth
                or n_labels == 1
                or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label( labels)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

        # greedily select the best split according to information gain
        best_feat, best_thresh = self._best_criteria(samples, labels, feat_idxs)

        # grow the children that result from the split
        left_idxs, right_idxs = self._split(samples[:, best_feat], best_thresh)
        left = self._grow_tree(samples[left_idxs, :], labels[left_idxs], depth + 1)
        right = self._grow_tree(samples[right_idxs, :], labels[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_criteria(self, samples, labels, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            samples_column = samples[:, feat_idx]
            thresholds = np.unique(samples_column)
            for threshold in thresholds:
                gain = self._information_gain( labels, samples_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def _information_gain(self, labels, samples_column, split_thresh):
        # parent loss
        parent_entropy = entropy( labels)

        # generate split
        left_idxs, right_idxs = self._split(samples_column, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # compute the weighted avg. of the loss for the children
        n = len( labels)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy( labels[left_idxs]), entropy(labels[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # information gain is difference in loss before vs. after split
        ig = parent_entropy - child_entropy
        return ig

    def _split(self, samples_column, split_thresh):
        left_idxs = np.argwhere(samples_column <= split_thresh).flatten()
        right_idxs = np.argwhere(samples_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _traverse_tree(self, samples, node):
        if node.is_leaf_node():
            return node.value

        if samples[node.feature] <= node.threshold:
            return self._traverse_tree(samples, node.left)
        return self._traverse_tree(samples, node.right)

    #     def _most_common_label(self, labels):
    #         counter = Counter(labels)
    #         most_common = counter.most_common(1)[0][0]
    #         return most_common

    def _most_common_label(self, labels):
        # counts = {"ones":[], "zeros":[]}
        zeros_count = 0
        ones_count = 0
        for i in range(len(labels)):
            if labels[i] == 0:
                zeros_count += 1
            else:
                ones_count += 1
        # counts["ones"].append(ones_count)
        # counts["zeros"].append(zeros_count)
        # print(counts)
        most_common = 0
        if zeros_count > ones_count:
            most_common = 0
        else:
            most_common = 1
        return most_common

    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''

        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("Feature " + str(tree.feature+1), "=", int(tree.threshold), "?")
            print("%s(yes)  left:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%s(no)  right:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)

def accuracy( labels_true, labels_pred):
  accuracy = np.sum( labels_true == labels_pred)/len( labels_true)
  return accuracy