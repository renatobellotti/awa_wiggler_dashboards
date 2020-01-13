#!/usr/bin/env python
import numpy as np
import tensorflow as tf

def generate_gaussian_mixture_data(n_samples_per_centre):
    X = []
    y = []
    
    # taken from FrEIA source code (by Ardizzone)
    centres = [
         (-2.4142, 1.),
         (-1., 2.4142),
         (1.,  2.4142),
         (2.4142,  1.),
         (2.4142, -1.),
         (1., -2.4142),
         (-1., -2.4142),
         (-2.4142, -1.)
    ]

    cov = np.eye(2)

    for i in range(8):
        xi = centres[i][0]
        yi = centres[i][1]

        samples = np.random.normal(size=(n_samples_per_centre, 2), scale=0.2)
        samples[:, 0] += xi
        samples[:, 1] += yi
        X.append(samples)
        y.append(np.ones(n_samples_per_centre) * i)

    X = np.vstack(X)
    y = np.concatenate(y)

    return X, y

def prepare_X(X, x_dim, nominal_dim, zeros_noise_scale=5e-2):
    '''Prepare the DVARs so they can be fed to the forward pass.'''
    X_noise = zeros_noise_scale * tf.random.normal((X.shape[0], nominal_dim - x_dim),
                                                    mean=0.,
                                                    stddev=1.0,
                                                    dtype=tf.dtypes.float64)

    return tf.concat([X, X_noise], axis=1)

def sample_z_space(n, z_dim):
    '''
    :param n: number of samples to sample
    :param z_dim: dimension of the latent space
    '''
    return tf.random.normal((n, z_dim),
                            mean=0.,
                            stddev=1.0,
                            dtype=tf.dtypes.float64)

def prepare_y(y, y_dim, z_dim, nominal_dim, zeros_noise_scale=5e-2):
    '''Call this function to prepare QOIs so they can be fed to the inverse pass.'''
    # sample z values
    z = sample_z_space(y.shape[0], z_dim)

    # pad y, z with noise
    yz_noise = zeros_noise_scale * tf.random.normal((y.shape[0], nominal_dim - y_dim - z_dim),
                                                    mean=0.,
                                                    stddev=1.0,
                                                    dtype=tf.dtypes.float64)

    return tf.concat([y, z, yz_noise], axis=1)

def apply_zero_padding(X, y, x_dim, y_dim, z_dim, nominal_dim):
    '''
    :param X: numpy array representing the "hidden" parameters
    :param y: numpy array representing the measurable quantities
    :param x_dim: dimension of the hidden quantities
    :param y_dim: dimension of the measurable quantities (number of "target features" in usual ML terms)
    :param z_dim: dimension of the latent space (hyperparameter)
    :param nominal_dim: nominal dimension (hyperparameter)
    '''
    zeros_noise_scale = 5e-2
    
    # pad input of forward pass with zeros
    X_padded = prepare_X(X, x_dim, nominal_dim, zeros_noise_scale)
    
    ########################################
    # build target matrix
    ########################################
    yz_padded = prepare_y(y, y_dim, z_dim, nominal_dim, zeros_noise_scale)

    return X_padded, yz_padded


def mmd2(x, y, h=0.2):
    xx = tf.linalg.matmul(x, tf.transpose(x))
    yy = tf.linalg.matmul(y, tf.transpose(y))
    xy = tf.linalg.matmul(x, tf.transpose(y))
    
    rx = tf.linalg.tensor_diag_part(xx)
    rx = tf.tile(rx, [xx.shape[0]])
    rx = tf.reshape(rx, (xx.shape[0], xx.shape[0]))
    ry = tf.linalg.tensor_diag_part(yy)
    ry = tf.tile(ry, [yy.shape[0]])
    ry = tf.reshape(ry, (yy.shape[0], yy.shape[0]))
    
    dxx = tf.transpose(rx) + rx - 2. * xx
    dyy = tf.transpose(ry) + ry - 2. * yy
    dxy = tf.transpose(rx) + ry - 2. * xy
    
    k_xx = 1. / (1. + dxx / h**2)
    k_yy = 1. / (1. + dyy / h**2)
    k_xy = 1. / (1. + dxy / h**2)
    
    return tf.reduce_mean(k_xx + k_yy - 2. * k_xy)
