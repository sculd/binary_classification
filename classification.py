import decision_tree

def learn(zeros, ones):
    """
    :param zeros: labeled data (represented as sorted scalar features) from class 0 (positives).
    :param ones: labeled data (represented as sorted scalar features) from class 1 (negatives).
    :return: a model.
    """
    return decision_tree.BinaryTreeModel(zeros, ones)


