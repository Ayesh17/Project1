from binary_decision_tree import DecisionTree

class DecisionTreeReal(DecisionTree):

    def print_tree(self, tree=None, indent=" "):

        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("Feature " + str(tree.feature+1), "<=", tree.threshold, "?")
            print("%s(yes)  left:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%s(no)  right:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
