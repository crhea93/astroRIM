"""
This file contains the steps required to calculate a the gradient of the loglikelihood function.
"""
import tensorflow as tf
import numpy as np

def calc_grad_standard(Y, A, C_N, x):
    r"""
    Calculate gradient of log likelihood function.

    .. math::
        \nabla \mathfrak{L}_t = (\mathbf{y}-A\mathbf{x})^TC_N^{-1}A

    Args:
        Y: True "unconvolved" model
        A: Convolution matrix
        C_N: Noise vector (1D)
        x: Current solution calculated from RIM
    
    Return:
        asinh of the gradient 
    """
    x = tf.cast(x, tf.float32)
    A = tf.cast(A, tf.float32)
    Y = tf.cast(Y, tf.float32)
    C_N = tf.linalg.diag(C_N)  # Diagonalize the noise vector
    C_N = tf.cast(C_N, tf.float32)
    x_max = tf.reduce_max(x)  # Calculate maximum value of x
    x_max = tf.maximum(x_max, tf.constant(1e-4, dtype=tf.float32))  # Make sure no weird division
    x_norm = tf.divide(x, x_max)   # Normalize x -- we do this to get the correct normalization so that A*x is correctly scaled
    conv_sol = tf.einsum('...ij,...j->...i', A, x_norm)  # Calculate A*x
    C_N_inv = tf.linalg.inv(C_N)  # Invert C_N
    residual_init = Y - conv_sol  # Returns a bn vector
    residual = tf.einsum("...i, ...ij -> ...j", residual_init, C_N_inv)  # Multiply (Y - conv_sol).T*C_N_inv -> returns a bn vector
    grad = tf.einsum("...i, ...ij -> ...j", residual, A)  # Multiply residual by A

    return tf.math.asinh(grad)
