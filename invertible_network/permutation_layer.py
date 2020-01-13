import tensorflow as tf
import numpy as np

class PermutationLayer(tf.keras.layers.Layer):
    def __init__(self, permuted_indices=None, inverse_indices=None, **kwargs):
        '''Init.'''
        super(PermutationLayer, self).__init__(**kwargs)

        self._permuted_indices = permuted_indices
        self._inverse_indices = inverse_indices

    def build(self, input_shape):
        if self._permuted_indices is None:
            self._permuted_indices = np.random.permutation(input_shape[1])
        # for details, see:
        # https://stackoverflow.com/questions/11649577/how-to-invert-a-permutation-array-in-numpy
        if self._inverse_indices is None:
            self._inverse_indices = np.argsort(self._permuted_indices)

    def call(self, inputs):
        # see:
        # https://stackoverflow.com/questions/41187181/how-to-index-a-list-with-a-tensorflow-tensor
        return tf.gather(inputs, self._permuted_indices, axis=1)

    def inverse(self, inputs):
        return tf.gather(inputs, self._inverse_indices, axis=1)

    # code adapted from:
    # https://stackoverflow.com/questions/58678836/notimplementederror-layers-with-arguments-in-init-must-override-get-conf
    def get_config(self):
        config = super(PermutationLayer, self).get_config().copy()

        config.update({
            'permuted_indices': self._permuted_indices,
            'inverse_indices': self._inverse_indices,
        })

        return config
