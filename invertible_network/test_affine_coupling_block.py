import unittest
import numpy as np
import tensorflow as tf
from affine_coupling_block import AffineCouplingBlock

# Needed to get O(1e-16) error when comparing inverse(forward(x)) ?= x.
# Whithout this line, the default datatype to use in layers is float32.
# This leads probably to problems with np.exp(), because the error
# is then O(1e-7)!
tf.keras.backend.set_floatx('float64')


class AffineCouplingBlockTest(unittest.TestCase):

    def setUp(self):
        # ensure reproducibility for all tests
        np.random.seed(3279479)

        self.batch_size = 11
        self.num_features = 24
        self.X = np.random.uniform(0., 1.,
                                   size=(self.batch_size, self.num_features))
        #self.X = self.X.astype('float32')

        config = {
            'subnet_type': 'identity'
        }
        
        self.clamp_exp = 2.

        input_layer = tf.keras.layers.Input(shape=self.num_features)
        self.l = AffineCouplingBlock(config, config, config, config, clamp_exp=self.clamp_exp)
        self.model = tf.keras.models.Sequential([input_layer, self.l])

    def testConsistency(self):
        '''Test if inversion(forwar(x)) == x'''
        prediction = self.l(self.X)
        inverse = self.l.inverse(prediction)

        self.assertTrue(np.isclose(inverse, self.X).all())

    def testPredictionForIdentityCoefficients(self):
        '''Test if the prediction is correct for s_i = t_i = Id'''
        # prediction of the custom layer
        y_pred = self.l(self.X)

        # true prediction assuming s1 = s2 = t1 = t2 = Id
        x1, x2 = np.split(self.X, 2, axis=1)
        y1 = x1 * np.exp(2./np.pi * self.clamp_exp * np.arctan(x2)) + x2
        y2 = x2 * np.exp(2./np.pi * self.clamp_exp * np.arctan(y1)) + y1
        y_true = np.hstack([y1, y2])
        
        #raise RuntimeError(y_pred - y_true)

        # compare
        self.assertTrue(np.isclose(y_pred, y_true).all())

    def testPredictionForScalingCoefficients(self):
        '''Test if the prediction is correct for s_i = t_i = 2*Id'''
        # build the layer to test
        config = {
            'subnet_type': 'scaling',
            'scaling_factor': 2.
        }

        input_layer = tf.keras.layers.Input(shape=self.num_features)
        l = AffineCouplingBlock(config, config, config, config)
        model = tf.keras.models.Sequential([input_layer, l])

        # prediction of the custom layer
        y_pred = l(self.X)

        # true prediction assuming s1 = s2 = t1 = t2 = Id
        x1, x2 = np.split(self.X, 2, axis=1)
        y1 = x1 * np.exp(2./np.pi * self.clamp_exp * np.arctan(2*x2)) + 2*x2
        y2 = x2 * np.exp(2./np.pi * self.clamp_exp * np.arctan(2*y1)) + 2*y1
        y_true = np.hstack([y1, y2])

        # compare
        self.assertTrue(np.isclose(y_pred, y_true).all())


if __name__ == '__main__':
    unittest.main()
