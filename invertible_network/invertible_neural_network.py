import math
import json
import logging
from timeit import default_timer # needed to tell user how long a single epoch takes to train
import numpy as np
import tensorflow as tf
from mllib.model import KerasSurrogate
from .affine_coupling_block import AffineCouplingBlock
from .permutation_layer import PermutationLayer
from .ardizzone_et_al_helpers import prepare_X, prepare_y, apply_zero_padding, mmd2

def build_invertible_neural_network(x_dim, y_dim, z_dim, nominal_dim,
                                    number_of_blocks,
                                    coefficient_net_config_s,
                                    coefficient_net_config_t,
                                    share_s_and_t=False):
        # input layer
        input_layer = tf.keras.layers.Input(shape=nominal_dim)
        l = input_layer

        # blocks consisting of an AffineCouplingBlock
        # followed by a PermutationLayer
        for i in range(number_of_blocks):
            # build an affine coupling block
            s1_config = coefficient_net_config_s
            s2_config = coefficient_net_config_s
            t1_config = coefficient_net_config_t
            t2_config = coefficient_net_config_t
            l = AffineCouplingBlock(s1_config, s2_config, t1_config, t2_config, share_s_and_t)(l)

            # add a permutation layer between the affine coupling layers
            if i != (number_of_blocks - 1):
                l = PermutationLayer()(l)

        return tf.keras.Model(input_layer, l)


def inverse_pass(model, dimensions, inputs):
    num_layers = len(model.layers)
    x_dim = dimensions['x_dim']

    out = inputs
    for i in range(num_layers):
        l = model.get_layer(index=num_layers - 1 - i)
        if type(l) is not tf.keras.layers.InputLayer:
            out = l.inverse(out)

    return out

@tf.function
def train_one_epoch(model, dimensions, dataset, optimizer, batch_size, number_of_samples, loss_x, loss_y, loss_z, loss_reconstruction, loss_weights, dtype, writer):
    '''
    :param loss_weights: factors to multiply lx, ly, lz by
    :param writer: file_writer for logging to tensorboard
    '''
    # scale of noise to add before reconstruction pass
    y_noise_scale = 0.1
    
    x_dim = dimensions['x_dim']
    y_dim = dimensions['y_dim']
    z_dim = dimensions['z_dim']
    nominal_dim = dimensions['nominal_dim']
    artificial_yz_dim = nominal_dim - y_dim - z_dim
    artificial_x_dim = nominal_dim - x_dim

    # shape of the padding (= artificial "features")
    artificial_yz_batch_shape = (batch_size, artificial_yz_dim)
    artificial_x_batch_shape = (batch_size, artificial_x_dim)
    
    ignore_dims_loss = tf.keras.losses.MeanSquaredError()
    
    total_loss = tf.constant(0., dtype=dtype)
    num_processed_batches = tf.constant(0, dtype='int64')
    num_batches = number_of_samples // batch_size
    
    tf.print('Total number of batches:', num_batches)
    
    # drop_remainder=True needed if batch_size does not divide dataset size evenly
    for X_train_batch, label_train_batch in dataset.shuffle(number_of_samples).batch(batch_size, drop_remainder=True):
        num_processed_batches += 1
            
        with tf.GradientTape() as tape:
            ################
            # Forward pass
            ################
            # get predictions for y and z
            prediction = model(X_train_batch)

            y_pred = prediction[:, :y_dim]
            z_pred = prediction[:, y_dim:y_dim+z_dim]

            # get true values of y and z
            y_true = label_train_batch[:, :y_dim]
            z_true = label_train_batch[:, y_dim:y_dim+z_dim]

            # losses from forward pass
            ly = loss_y(y_true, y_pred)
            lz = loss_z(label_train_batch[:, :y_dim+z_dim], prediction[:, :y_dim+z_dim])

            ################
            # Backward pass
            ################
            # perform backward pass
            X_pred = inverse_pass(model, dimensions, label_train_batch)
            lx = loss_x(X_train_batch[:, :x_dim], X_pred[:, :x_dim])
            
            ################################################
            # make sure artificial dimensions are not used
            ################################################
            l_artificial_dims = ignore_dims_loss(tf.zeros(artificial_yz_batch_shape),
                                                 label_train_batch[:, y_dim+z_dim:])

            l_artificial_dims += ignore_dims_loss(tf.zeros(artificial_x_batch_shape),
                                                 X_train_batch[:, x_dim:])
            
            ##########################
            # reconstruction loss
            ##########################
            input_for_reconstruction = prediction

            # add noise to the y and z components
            perturbed_yz = prediction[:, :y_dim + z_dim] + y_noise_scale * tf.random.normal((input_for_reconstruction.shape[0], y_dim + z_dim),
                                                                                           mean=0.,
                                                                                           stddev=1.0,
                                                                                           dtype=tf.dtypes.float64)
            zero_padding = tf.zeros((prediction.shape[0], nominal_dim - y_dim - z_dim), dtype=tf.dtypes.float64)
            input_for_reconstruction = tf.concat([perturbed_yz, zero_padding], axis=1)

            X_reconstructed = inverse_pass(model, dimensions, input_for_reconstruction)
            l_reconstruction = loss_reconstruction(X_train_batch, X_reconstructed)

            #######################
            # calculate total loss
            #######################
            total_loss = (loss_weights[0] * lx
                          + loss_weights[1] * ly
                          + loss_weights[2] * lz
                          + loss_weights[3] * l_artificial_dims
                          + loss_weights[4] * l_reconstruction)

        # calculate the gradients
        grads = tape.gradient(total_loss, model.trainable_variables)

        # perform optimization step
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if num_processed_batches % 1000 == 0:
            tf.print('Processed batch', num_processed_batches)

            with writer.as_default():
                tf.summary.scalar('lx', data=lx, step=num_processed_batches)
                tf.summary.scalar('ly', data=ly, step=num_processed_batches)
                tf.summary.scalar('lz', data=lz, step=num_processed_batches)
                tf.summary.scalar('lartificial', data=l_artificial_dims, step=num_processed_batches)
                tf.summary.scalar('l_reconstruction', data=l_reconstruction, step=num_processed_batches)
                tf.summary.scalar('l_tot', data=total_loss, step=num_processed_batches)


class InvertibleNetworkSurrogate(KerasSurrogate):

    @staticmethod
    def from_config(x_dim, y_dim, z_dim, nominal_dim,
                 number_of_blocks,
                 coefficient_network_units,
                 coefficient_network_activations,
                 share_s_and_t,
                 preprocessor_x,
                 preprocessor_y,
                 name,
                 version):
        '''
        :param number_of_blocks: how many blocks consisting of
                                 AffineCouplingLayers and PermutationLayers
                                 to use
        :param coefficient_network_units: list(int) number of units in each
                                          layer of the coefficient networks
        :param coefficient_network_activations: list(str) activations after
                                                each layer of the
                                                coefficient networks
        '''
        coefficient_config = {
            'subnet_type': 'dense',
            'units': coefficient_network_units,
            'activations': coefficient_network_activations
        }

        model = build_invertible_neural_network(x_dim,
                                    y_dim,
                                    z_dim,
                                    nominal_dim,
                                    number_of_blocks,
                                    coefficient_config,
                                    coefficient_config,
                                    share_s_and_t)

        surr = InvertibleNetworkSurrogate(model, preprocessor_x, preprocessor_y, name, version)

        surr._x_dim = x_dim
        surr._y_dim = y_dim
        surr._z_dim = z_dim
        surr._nominal_dim = nominal_dim

        return surr

    def _fit_model(self, X, y, **kwargs):
        '''
        parameters = {
            'optimizer': Tensorflow optimizer object,
            'batch_size': int,
            'loss_weight_x': float,
            'loss_weight_y': float,
            'loss_weight_z': float,
            'loss_weight_artificial': float, # weight for loss to make
                                             # artificial dimensions
                                             # close to zero
            'loss_weight_reconstruction': float,
            'tensorboard_dir': str,          # tensorboard logdir
            'epochs': int,
        }
        '''
        # build datastructures for training
        dimensions = {
            'x_dim': self._x_dim,
            'y_dim': self._y_dim,
            'z_dim':self._z_dim,
            'nominal_dim': self._nominal_dim,
        }

        loss_weights = np.array([
            kwargs['loss_weight_x'],
            kwargs['loss_weight_y'],
            kwargs['loss_weight_z'],
            kwargs['loss_weight_artificial'],
            kwargs['loss_weight_reconstruction'],
        ])
        # normalise loss weights
        loss_weights /= np.sum(loss_weights)

        # losses
        def loss_x(x_true, x_pred):
            return mmd2(x_true, x_pred)

        loss_y = tf.keras.losses.MeanSquaredError()

        def loss_z(z_true, z):
            return mmd2(z_true, z)

        loss_reconstruction = tf.keras.losses.MeanSquaredError()

        # build the training dataset
        X_padded, y_padded = apply_zero_padding(X, y,
                                               self._x_dim,
                                               self._y_dim,
                                               self._z_dim,
                                               self._nominal_dim)
        dataset = tf.data.Dataset.from_tensor_slices((X_padded, y_padded))

        # organise logging to tensorboard
        writer = tf.summary.create_file_writer(kwargs['tensorboard_dir'])

        # train
        for i in range(kwargs['epochs']):
            logging.info(f'Start epoch {i}...')

            start = default_timer()

            train_one_epoch(self.model,
                            dimensions,
                            dataset,
                            kwargs['optimizer'],
                            kwargs['batch_size'],
                            X.shape[0], # number of samples
                            loss_x, loss_y, loss_z, loss_reconstruction,
                            loss_weights,
                            'float64',
                            writer)

            end = default_timer()

            # get duration [s]
            duration = end - start
            logging.info(f"Epoch {i+1}/{kwargs['epochs']} needed {int(duration//60)}:{int(math.ceil(duration) % 60)}.")

    def _predict_model(self, X, **kwargs):
        X_padded = prepare_X(X, self._x_dim, self._nominal_dim)
        prediction = self.model.predict(X_padded, **kwargs)

        # remove latent space and padding dimensions
        return prediction[:, :self._y_dim]

    def sample(self, y):
        '''Generate samples of the X space corresponding to the given y space configurations.'''
        yz_padded = prepare_y(y, self._y_dim, self._z_dim, self._nominal_dim)
        dimensions = {
            'x_dim': self._x_dim,
            'y_dim': self._y_dim,
            'z_dim':self._z_dim,
            'nominal_dim': self._nominal_dim,
        }

        X_sampled = inverse_pass(self.model, dimensions, yz_padded)

        return X_sampled[:, :self._x_dim]

    def _save_model(self, model_dir):
        # save neural network
        model_path = '{}/model.hdf5'.format(model_dir) 
        self.model.save(model_path)

        # save the dimensions
        dimensions = {
            'x_dim': self._x_dim,
            'y_dim': self._y_dim,
            'z_dim':self._z_dim,
            'nominal_dim': self._nominal_dim,
        }
        with open('{}/dimensions.json'.format(model_dir), 'w') as file:
            json.dump(dimensions, file, indent=4)

    @classmethod
    def _load_model(cls, model_dir):
        '''
        :param identifier: [models_dir, model_name]
        '''
        custom_objects = {
            'AffineCouplingBlock': AffineCouplingBlock,
            'PermutationLayer': PermutationLayer
        }
        model_path = '{}/model.hdf5'.format(model_dir)
        model = tf.keras.models.load_model(model_path,
                                        custom_objects=custom_objects,
                                        compile=False)

        return model

    @classmethod
    def _build_surrogate(cls, model, preprocessor_x, preprocessor_y, name, version, model_dir):
        surr = InvertibleNetworkSurrogate(model,
                                          preprocessor_x,
                                          preprocessor_y,
                                          name,
                                          version)
        # load the dimensions
        json_path = '{}/dimensions.json'.format(model_dir)
        
        with open(json_path, 'r') as file:
            dimensions = json.load(file)

        surr._x_dim = dimensions['x_dim']
        surr._y_dim = dimensions['y_dim']
        surr._z_dim = dimensions['z_dim']
        surr._nominal_dim = dimensions['nominal_dim']

        return surr
