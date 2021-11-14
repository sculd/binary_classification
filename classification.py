import decision_tree

def learn(zeros, ones):
    """
    :param zeroes: labeled data (represented as scalar features) from class 0.
    :param ones: labeled data (represented as scalar features) from class 1.
    :return: a model.
    """
    return decision_tree.BinaryTreeModel(zeros, ones)


