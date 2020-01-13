import tensorflow as tf

class AffineCouplingBlock(tf.keras.layers.Layer):
    '''
    A coupling block as described by Ardizzone et al. (2019) in equ. (5)

    Note: For this network, the shape of the input and of the output layers have to match!
    '''
    def __init__(self, s1_config, s2_config, t1_config, t2_config, clamp_exp=2., share_s_and_t=False, **kwargs):
        '''
        Init.

        :param s1_config, s2_config, t1_config, t2_config: each is a dict that
            contains a configuration of a subnet.

            So far, only dense layers are supported. Format:

            {
                'subnet_type': 'dense',
                'units': list(int)          # number of units in each layer
                'activations': list(string) # activations of each layer
            }

            **Note:**

            The last dense layer of the subnet must have the same number of units
            as input_shape[1] of the build() function!
        :param clamp_exp: the argument of exp() will be in [-clamp_exp, +clamp_exp]
        '''
        super(AffineCouplingBlock, self).__init__(**kwargs)

        self._s1_config = s1_config
        self._s2_config = s2_config
        self._t1_config = t1_config
        self._t2_config = t2_config
        
        self._share_s_and_t = share_s_and_t
        
        # constant needed for clamping
        # atan() has range (-pi/2, +pi/2) and 2/pi ≃ 0.636619772
        self._clamp_exp = clamp_exp
        self._clamp_scale = clamp_exp * 0.636619772

        self._s1 = None
        self._s2 = None
        self._t1 = None
        self._t2 = None

    def build(self, input_shape):
        # We split the input of this layer into two halves
        # along the feature axis.
        # Each coefficient function (s_i, t_i) is fed only
        # one half of the tensor.
        shape = (input_shape[0], input_shape[1] // 2)
        
        if self._share_s_and_t:
            self._s1 = self._build_subnet_from_config(self._s1_config, shape)
            self._s2 = self._build_subnet_from_config(self._s2_config, shape)
            self._t1 = self._s1
            self._t2 = self._s2
        else:
            self._s1 = self._build_subnet_from_config(self._s1_config, shape)
            self._s2 = self._build_subnet_from_config(self._s2_config, shape)
            self._t1 = self._build_subnet_from_config(self._t1_config, shape)
            self._t2 = self._build_subnet_from_config(self._t2_config, shape)

    def call(self, inputs):
        '''
        The following formulae are implemented:

        y1 = x1 .* exp(s2(x2)) + t2(x2)
        y2 = x2 .* exp(s1(y1)) + t1(y1)
        
        Additionally, the arguments of the exponential functions
        are clamped to the range +/- clamp_exp (see constructor).

        :param inputs: Tensor representing x
        :returns: Tensor y
        '''
        # split the inputs in 2 halves
        x1, x2 = tf.split(inputs, 2, axis=1)

        y1 = x1 * tf.math.exp(self._clamp_scale * tf.math.atan(self._s2(x2))) + self._t2(x2)
        y2 = x2 * tf.math.exp(self._clamp_scale * tf.math.atan(self._s1(y1))) + self._t1(y1)

        return tf.concat([y1, y2], axis=1)

    def inverse(self, inputs):
        '''
        The inverse of ```call()```.

        The following formulae are implemented:

        x2 = (y2 - t1(y1)) .* exp(-s1(y1))
        x1 = (y1 - t2(x2)) .* exp(-s2(x2))
        
        Additionally, the arguments of the exponential functions
        are clamped to the range +/- clamp_exp (see constructor).

        :param inputs: Tensor representing y
        :returns: Tensor x
        '''
        y1, y2 = tf.split(inputs, 2, axis=1)

        x2 = (y2 - self._t1(y1)) * tf.math.exp(- self._clamp_scale * tf.math.atan(self._s1(y1)))
        x1 = (y1 - self._t2(x2)) * tf.math.exp(- self._clamp_scale * tf.math.atan(self._s2(x2)))

        return tf.concat([x1, x2], axis=1)

    # code adapted from:
    # https://stackoverflow.com/questions/58678836/notimplementederror-layers-with-arguments-in-init-must-override-get-conf
    def get_config(self):
        config = super(AffineCouplingBlock, self).get_config().copy()
        
        config.update({
            's1_config': self._s1_config,
            's2_config': self._s2_config,
            't1_config': self._t1_config,
            't2_config': self._t2_config,
            'clamp_exp': self._clamp_exp
        })
        
        return config

    def _build_dense_subnet_from_config(self, config, input_shape):
        '''Build a dense coefficient subnet from the given configuration.'''
        units = config['units']
        activations = config['activations']

        layers = [tf.keras.layers.Input(shape=input_shape[1])]
        for i, activation in enumerate(activations):
            layers.append(tf.keras.layers.Dense(units=units[i], activation=activation))
        return tf.keras.Sequential(layers)

    def _build_identity_layer_from_config(self, config, input_shape):
        '''For testing.'''
        # dummy layer used for testing
        class IdentityLayer(tf.keras.layers.Layer):
            def call(self, inputs):
                return inputs

        return IdentityLayer()

    def _build_scaling_layer_from_config(self, config, input_shape):
        '''For testing.'''
        # dummy layer used for testing
        class ScalingLayer(tf.keras.layers.Layer):

            def __init__(self, scaling_factor, **kwargs):
                super(ScalingLayer, self).__init__(**kwargs)
                self._s = scaling_factor

            def call(self, inputs):
                return self._s * inputs

        return ScalingLayer(config['scaling_factor'])

    def _build_subnet_from_config(self, config, input_shape):
        '''Choose the correct factory method and call it.'''
        if config['subnet_type'] == 'dense':
            return self._build_dense_subnet_from_config(config, input_shape)
        elif config['subnet_type'] == 'identity':
            return self._build_identity_layer_from_config(config, input_shape)
        elif config['subnet_type'] == 'scaling':
            return self._build_scaling_layer_from_config(config, input_shape)
        else:
            raise NotImplementedError('AffineCOuplingBlock: Unsupported subnet type')
