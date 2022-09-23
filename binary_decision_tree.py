import numpy as np

#get Entropy
def entropy(labels):
    hist = np.bincount(labels)
    ps = hist / len(labels)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

#Node class
class NodeClass:

    #for decision node
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    #for leaf node
    def is_leaf_node(self):
        return self.value is not None


#Decision Tree class
class DecisionTreeClass:

    def __init__(self, max_depth, min_samples_split=2):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    #fit
    def fit(self, samples, labels):
        self.root = self.build_tree(samples, labels)

    #predict
    def predict(self, samples):
        return np.array([self.traverse_tree(sample, self.root) for sample in samples])

    #build thr tree
    def build_tree(self, samples, labels, depth=0):
        num_samples, num_features = samples.shape
        num_labels = len(np.unique(labels))

        # stopping conditions
        if (depth >= self.max_depth
                or num_labels == 1
                or num_samples < self.min_samples_split):
            leaf_value = self.most_common_label(labels)
            return NodeClass(value=leaf_value)

        feature_ids = np.random.choice(num_features, num_features, replace=False)

        #select the best split based on information gain
        best_feature, best_threshold= self.best_split(samples, labels, feature_ids)

        # grow the children that result from the split
        left_ids, right_ids = self.split(samples[:, best_feature], best_threshold)
        left_tree = self.build_tree(samples[left_ids, :], labels[left_ids], depth + 1)
        right_tree = self.build_tree(samples[right_ids, :], labels[right_ids], depth + 1)

        return NodeClass(best_feature, best_threshold, left_tree, right_tree)

    #get the best split
    def best_split(self, samples, labels, feat_ids):
        max_gain = np.NINF
        split_id, split_threshold= None, None
        for feat_id in feat_ids:
            samples_col = samples[:, feat_id]
            thresholds = np.unique(samples_col)
            for threshold in thresholds:
                cur_gain = self.information_gain(labels, samples_col, threshold)

                if cur_gain >= max_gain:
                    max_gain = cur_gain
                    split_id = feat_id
                    split_threshold= threshold

        return split_id, split_threshold

    #get the information gain
    def information_gain(self, labels, samples_column, split_threshold):
        # parent's entropy'
        parent_entropy = entropy(labels)

        # split
        left_ids, right_ids = self.split(samples_column, split_threshold)

        if len(left_ids) == 0 or len(right_ids) == 0:
            return 0

        # child's entropy
        num = len(labels)
        num_left, num_right = len(left_ids), len(right_ids)
        entropy_left, entropy_right = entropy(labels[left_ids]), entropy(labels[right_ids])
        child_entropy = (num_left / num) * entropy_left + (num_right / num) * entropy_right

        #information gain
        info_gain = parent_entropy - child_entropy
        return info_gain

    #split
    def split(self, samples_column, split_threshold):
        left_ids = np.argwhere(samples_column <= split_threshold).flatten()
        right_ids = np.argwhere(samples_column > split_threshold).flatten()
        return left_ids, right_ids

    #traverse through the tree
    def traverse_tree(self, samples, node):
        if node.is_leaf_node():
            return node.value

        if samples[node.feature] <= node.threshold:
            return self.traverse_tree(samples, node.left)
        return self.traverse_tree(samples, node.right)

    #get most common label(0,1)
    def most_common_label(self, labels):
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

    #print the tree
    def print_tree(self, tree=None, indent=" "):

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
