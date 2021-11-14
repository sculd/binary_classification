import decision_tree

zeros = [0, 1, 2, 3, 5]
ones = [4, 6, 6, 7, 8, 9, 11, 13]
m = decision_tree.BinaryTreeModel(zeros, ones)
print('[tree]\n' + str(m.root))
m.print_misclassified(zeros, ones)

print('[evaluation] train set')
print(m.evaluate(zeros, ones))

# hypothetical out of sample test set
zeros_test = [0.5, 1.5, 2.5]
ones_test = [4.1, 6.5, 11.5, 13.5]
m.print_misclassified(zeros_test, ones_test)
print('[evaluation] hypothetical out of sample test set')
print(m.evaluate(zeros_test, ones_test))

