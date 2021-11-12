from bisect import bisect_left

_DEFAULT_THRESHOLD = 0.0
_DEFAULT_BUFFER = 1.0


class Binary():
    def __init__(self):
        self.threshold = threshold
        self.buffer = buffer

    def eval(self, zeroes, ones):
        pass

    def train(self, zeroes, ones):
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
