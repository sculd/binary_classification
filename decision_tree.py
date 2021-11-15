import math
from collections import Counter


def _calculate_entropy(cnt_0, cnt_1):
    if cnt_0 + cnt_1 == 0: return 0
    p1, p2 = 1. * cnt_0 / (cnt_0 + cnt_1), 1. * cnt_1 / (cnt_0 + cnt_1)
    return 0 if p1 == 1 or p2 == 1 else -p1 * math.log2(p1) - p2 * math.log2(p2)


class DecisionTreeNode:
    def __init__(self, data):
        self.left, self.right = None, None
        self.pivot = None

        self.data = data
        cnts = Counter(map(lambda p: p[1], data))
        cnt_0, cnt_1 = cnts.get(0, 0), cnts.get(1, 0)
        if cnt_0 >= cnt_1:
            self.label = 0
        else:
            self.label = 1

        self.entropy = _calculate_entropy(cnt_0, cnt_1)
        self._create_next_generation()

    def _indented_value(self, indent):
        i = ' '.join(['' for _ in range(indent)])
        l = '\n' + self.left._indented_value(indent+4) if self.left else ''
        r = '\n' + self.right._indented_value(indent+4) if self.right else ''
        return '{i}label: {t}, pivot: {p}, data: {d}{l}{r}'.format(t=self.label, p=self.pivot, d=','.join(map(lambda p: str(p), self.data)), i=i, l=l, r=r)

    def __str__(self):
        return self._indented_value(0)

    def _create_next_generation(self):
        if len(self.data) < 3:
            return

        cnts = Counter(map(lambda p: p[1], self.data))
        cnt_0_left, cnt_1_left = 0, 0
        cnt_0_right, cnt_1_right = cnts.get(0, 0), cnts.get(1, 0)
        cnt_total = cnts.get(0, 0) + cnts.get(1, 0)
        entropy_removed_max = 0
        for d in self.data:
            if d[1] == 0:
                cnt_0_left += 1
                cnt_0_right -= 1
            else:
                cnt_1_left += 1
                cnt_1_right -= 1
            entropy_left = _calculate_entropy(cnt_0_left, cnt_1_left)
            entropy_right = _calculate_entropy(cnt_0_right, cnt_1_right)
            entropy_removed = self.entropy - ((cnt_0_left + cnt_1_left) * entropy_left + (cnt_0_right + cnt_1_right) * entropy_right)  * 1. / cnt_total
            if entropy_removed > entropy_removed_max:
                entropy_removed_max = entropy_removed
                self.pivot = d[0]

        if entropy_removed_max > 0:
            self.left = DecisionTreeNode([d for d in self.data if d[0] <= self.pivot])
            self.right = DecisionTreeNode([d for d in self.data if d[0] > self.pivot])

    def predict(self, feature):
        while True:
            if self.pivot is None:
                return self.label

            if feature <= self.pivot:
                return self.left.predict(feature)
            else:
                return self.right.predict(feature)


class BinaryTreeModel:
    def __init__(self, zeros, ones):
        data = self._combine_zeros_ones(zeros, ones)
        self.root = DecisionTreeNode(data)

    def _combine_zeros_ones(self, zeros, ones):
        data = []
        i, j, m, n = 0, 0, len(zeros), len(ones)
        while i < m or j < n:
            if i == m:
                data.append((ones[j], 1))
                j += 1
            elif j == n:
                data.append((zeros[i], 0))
                i += 1
            elif zeros[i] >= ones[j]:
                data.append((ones[j], 1))
                j += 1
            else:
                data.append((zeros[i], 0))
                i += 1

        return data

    def predict(self, feature):
        return self.root.predict(feature)

    def evaluate(self, zeros, ones):
        data = self._combine_zeros_ones(zeros, ones)

        corrects = 0
        true_positives, positives = 0, 0
        true_negatives, negatives = 0, 0

        for d in data:
            if self.predict(d[0]) == d[1]:
                corrects += 1
                if d[1] == 0:
                    true_positives += 1
                else:
                    true_negatives += 1

            if d[1] == 0:
                positives += 1
            else:
                negatives += 1

        false_positives = negatives - true_negatives
        false_negatives = positives - true_positives

        return {
            'accuracy': round(1. * (true_positives + true_negatives) / len(data), 2),
            'precision': round(1. * true_positives / (true_positives + false_positives), 2),
            'recall': round(1. * true_positives / positives, 2),
            'negative predictive value': round(1. * true_negatives / (true_negatives + false_negatives), 2),
            'true negative rate': round(1. * true_negatives / negatives, 2),
        }

    def print_misclassified(self, zeros, ones):
        print('[misclassified samples]')
        for v in list(map(lambda v: (v, 0), zeros)) + list(map(lambda v: (v, 1), ones)):
            l, p = v[1], self.predict(v[0])
            if p != l:
                print('feature: {v}, label {l}, predicted: {p}'.format(v=v[0], l=l, p=p))

