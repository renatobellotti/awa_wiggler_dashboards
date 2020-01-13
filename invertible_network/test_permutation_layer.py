import unittest
import numpy as np
import tensorflow as tf
from permutation_layer import PermutationLayer


class PermutationLayerTest(unittest.TestCase):
    def setUp(self):
        # reproducibility
        np.random.seed(287999829)

        self.shape = (5, 6)
        self.x_test = np.arange(self.shape[0]*self.shape[1]).reshape(self.shape)

    def testForward(self):
        # build the simplest network consisting of a PermutationLayer only
        l = tf.keras.layers.Input((self.shape[1],))
        l1 = PermutationLayer()
        m = tf.keras.models.Sequential([l, l1])

        # permute the columns of x_test
        permuted = m(self.x_test)

        # check shapes
        self.assertTrue(permuted.shape == self.x_test.shape)

        # check if all obtained columns are columns of x_test
        true_columns = np.hsplit(self.x_test, self.shape[1])
        permuted_columns = np.hsplit(permuted, self.shape[1])

        for col in permuted_columns:
            equalities = [np.isclose(col, other).all() for other in true_columns]
            num_true = equalities.count(True)
            self.assertTrue(num_true == 1)

        # check if forward and backward passes lead to the same result
        inverse_permuted = l1.inverse(permuted)
        self.assertTrue(tf.reduce_all(tf.math.equal(self.x_test, inverse_permuted)))

if __name__ == '__main__':
    unittest.main()
