import unittest

from classification import learn


class TestClassification(unittest.TestCase):
    def test_empty_inputs(self):
        self.assertEqual(learn([], []), 0)
        self.assertEqual(learn([0, 1, 2], []), 2.0 + 1.0)
        self.assertEqual(learn([], [3, 4, 5]), 3.0 - 1.0)

    def test_no_intersection(self):
        self.assertEqual(learn([0, 1, 2], [3, 4, 5]), 2 + 0.5)
        self.assertEqual(learn([0, 1, 2], [4, 5, 6]), 2 + 1.0)

    def test_threshold_on_left(self):
        self.assertEqual(learn([1, 3], [0, 0, 0, 0, 0, 1, 2]), 0.0 - 0.5)
        self.assertEqual(learn([1, 2], [0, 0, 0, 0, 0, 3, 4]), 0.0 - 0.5)

    def test_threshold_on_right(self):
        self.assertEqual(learn([0, 2, 3], [1]), 3.0 + 1.0)
        self.assertEqual(learn([0, 1, 3, 3, 3, 3, 3], [0, 1, 2]), 3.0 + 1.0)
        self.assertEqual(learn([0, 3, 3, 3, 3, 3], [0, 2]), 3.0 + 1.0)

    def test_threshold_on_first_zeroes(self):
        self.assertEqual(learn([0], [1, 2, 3]), 0.0 + 0.5)
        self.assertEqual(learn([1], [0, 2, 3]), 1.0 + 0.5)
        self.assertEqual(learn([2], [1, 3, 4]), 2.0 + 0.5)
        self.assertEqual(learn([0, 1, 2], [3, 4, 5, 6, 7]), 2.0 + 0.5)

    def test_threshold_on_second_zeroes(self):
        self.assertEqual(learn([0, 1, 3, 6], [3, 7, 8, 8]), 6.0 + 0.5)
        self.assertEqual(learn([0, 1, 3, 4, 6], [3, 4, 7, 8, 8]), 6.0 + 0.5)
        self.assertEqual(learn([0, 1, 3, 3, 3, 6], [0, 7, 8, 8]), 6.0 + 0.5)

    def test_threshold_on_ones_first_and_first_zeroes(self):
        self.assertEqual(learn([1, 3], [0, 2, 2, 2, 2, 3, 4]), 1.0 + 0.5)

    def test_threshold_on_ones_first_and_second_zeroes(self):
        self.assertEqual(learn([0, 1, 3, 3, 3, 3, 3, 3, 6], [0, 1, 1, 1, 7, 8, 8]), 6.0 + 0.5)


if __name__ == '__main__':
    unittest.main()
