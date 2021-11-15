import decision_tree

print('\nzeros/ones split with some intersection')
zeros = [0, 1, 2, 3, 5]
ones = [4, 6, 6, 7, 8, 9, 11, 13]
m = decision_tree.BinaryTreeModel(zeros, ones)
print('[tree]\n' + str(m.root))
m.print_misclassified(zeros, ones)
print('[evaluation] train set')
print(m.evaluate(zeros, ones))

print('\nhypothetical out of sample test set')
zeros_test = [0.5, 1.5, 2.5, 3.9]
ones_test = [6.5, 11.5, 13.5]
m.print_misclassified(zeros_test, ones_test)
print('[evaluation] hypothetical out of sample test set')
print(m.evaluate(zeros_test, ones_test))

print('\nwith min_samples_split: 3 (less overfit)')
m = decision_tree.BinaryTreeModel(zeros, ones, min_samples_split = 3)
print('[tree]\n' + str(m.root))
m.print_misclassified(zeros, ones)
print('[evaluation] train set')
print(m.evaluate(zeros, ones))

print('\nhypothetical out of sample test set (less overfit)')
zeros_test = [0.5, 1.5, 2.5, 3.9]
ones_test = [6.5, 11.5, 13.5]
m.print_misclassified(zeros_test, ones_test)
print('[evaluation] hypothetical out of sample test set')
print(m.evaluate(zeros_test, ones_test))

print('\ndeep tree without prune')
zeros = [0, 1, 2, 3, 4, 5, 5, 5, 5]
ones = [4, 6]
m = decision_tree.BinaryTreeModel(zeros, ones, min_samples_split = 3, pruned = False)
print('[tree]\n' + str(m.root))
m.print_misclassified(zeros, ones)
print('[evaluation] train set')
print(m.evaluate(zeros, ones))

print('\ndeep tree pruned')
zeros = [0, 1, 2, 3, 4, 5, 5, 5, 5]
ones = [4, 6]
m = decision_tree.BinaryTreeModel(zeros, ones, min_samples_split = 3)
print('[tree]\n' + str(m.root))
m.print_misclassified(zeros, ones)
print('[evaluation] train set')
print(m.evaluate(zeros, ones))

print('\na class skewed around single value')
zeros = [0, 0, 0, 0, 0, 0, 0, 1]
ones = [3, 4, 5]
m = decision_tree.BinaryTreeModel(zeros, ones)
print('[tree]\n' + str(m.root))
m.print_misclassified(zeros, ones)
print('[evaluation] train set')
print(m.evaluate(zeros, ones))

print('\nzero / one ranges interchange')
zeros = [0, 1, 10, 11, 20, 21]
ones = [5, 6, 15]
m = decision_tree.BinaryTreeModel(zeros, ones)
print('[tree]\n' + str(m.root))
m.print_misclassified(zeros, ones)
print('[evaluation] train set')
print(m.evaluate(zeros, ones))
