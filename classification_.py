from bisect import bisect_left

_DEFAULT_THRESHOLD = 0.0
_DEFAULT_BUFFER = 1.0


def learn(zeroes, ones):
    """
    Find a threshold that divides data into "left" and "right".

    Two parts such that you minimize the classification error on the training set.
    The error function = # of miss classifications. The suggested model outputs the binary evaluation instead of p.
    It is assumed that the threshold divides to put class 0 to "left", 1 to "right".
    It is expected that if the feature value ties with the threshold, the it is classified as 0.
    The threshold 0 is assumed when there is no training data.

    time complexity:
    O(m+n)

    space complexity:
    O(const)

    :param ones: labeled data (represented as scalar features) from class 1.
    :param zeroes: labeled data (represented as scalar features) from class 0.
    :return: the optimal threshold.
    """
    if not zeroes and not ones:
        return _DEFAULT_THRESHOLD

    if not zeroes:
        return ones[0] - _DEFAULT_BUFFER

    v = min(ones[0] if ones else zeroes[0], zeroes[0]) - _DEFAULT_BUFFER

    # the threshold would be decided within the intersection (if any) of ones and zeros.
    # zeroes: z0 z1 z2 ...     za zb zc zm
    # ones:               o0 o1 ...       ...    on
    # o0 <= threshold < zm

    threshold = v
    right = zeroes[-1]
    zeroes_i, ones_i = 0, 0
    misses = len(zeroes)
    min_misses = misses

    while v < right:
        if ones_i == len(ones):
            v = right
            misses -= len(zeroes) - zeroes_i

        elif ones[ones_i] <= zeroes[zeroes_i]:
            v = ones[ones_i]
            ones_i += 1
            misses += 1

        else:
            v = zeroes[zeroes_i]
            zeroes_i += 1
            misses -= 1

        # prefer choosing larger threshold in case of tie to avoid overfit.
        if misses <= min_misses:
            threshold = v
            min_misses = misses

    # the result overfits to fall exactly on a class "0" value.
    next_i = bisect_left(ones, threshold)
    buffer = (ones[next_i] - threshold) / 2. if next_i < len(ones) else _DEFAULT_BUFFER

    return threshold + buffer
