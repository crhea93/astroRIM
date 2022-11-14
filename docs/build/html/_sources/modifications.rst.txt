Files to Modify 
===============

On this page, you will find a description of the two files that users can modify to change the RIM's architecture or likelihood function.

`rim_model.py`
--------------
The `rim_model.py` file contains the definition of the RIM architecture. Regardless of the dimensions of your RIM, you will need to modify two functions
within the `RIM_model` class: the `__init__` and the `call`. 

`__init__`
^^^^^^^^^^
The default functionality is the following:

.. code-block:: python

    def __init__(self, conv_filters, kernel_size, rnn_units):
        """
        Initiation of class

        Args:
            conv_filters: Number of convolutional layers (Int)
            kernel_size: Size of convoluitional kernel (Int)
            rnn_units: List of units in GRUs ([Int])

        """
        super().__init__(self)
        # Define Layers of RIM
        self.conv1d_1 = tf.keras.layers.Conv1D(filters=conv_filters, kernel_size=kernel_size, strides=1,
                                               padding='same', activation='tanh')
        self.gru1 = tf.keras.layers.GRU(rnn_units[0], activation='tanh', recurrent_activation='sigmoid',
                                        return_sequences=True, return_state=True)
        self.conv1d_2 = tf.keras.layers.Conv1DTranspose(filters=conv_filters, kernel_size=kernel_size, strides=1,
                                                        padding='same', activation='tanh')
        self.gru2 = tf.keras.layers.GRU(rnn_units[1], activation='tanh', recurrent_activation='sigmoid',
                                        return_sequences=True, return_state=True)
        self.conv1d_3 = tf.keras.layers.Conv1D(filters=1, kernel_size=kernel_size, strides=1,
                                                        padding='same', activation='linear')
        self.rnn_units = rnn_units  # DO NOT CHANGE THIS LINE

Here is where we define the architecture of our network. In this example, we have 3 convolutional layers punctuated by gated recurrent units (GRUs). 
We can see that the convolutional layers all have the same number of filters and kernel size (this could be changed if you wish). Additionally, each layer 
has a stride of 1 with the same padding and a tanh activation function. The GRUs have a tanh activation function and a sigmoid recurrent function. The important
piece here is that we read in a vector describing the number of units for each GRU.

Users can modify any of the defined values above. Additionally, they can add (or delete) layers.

In order to make sure that the network operates in the correct sequence, we must update the `call` function.

`call`
^^^^^^

The `call` function currently ressembles the following:

.. code-block:: python

    def call(self, sol, log_L, hidden_states=[None], return_state=True, training=False):
        """
        A single run through the recurrent inference machine. This is the standard architecture.


        conv2d -> gru -> conv2d_T -> gru -> conv2d


        When we go from a convolutional layer to a GRU, we need to first flatten the activation map. We then use the
        expand_dims function to make sure the input of the GRU has the correct format (batch length, feature dims, 1).

        Args:

            sol: Solution at time step t (x_t)
            log_L: Gradient of log likelihood
            hidden_states: List of hidden state vectors
            return_state: Return the hidden state boolean
            training: Boolean determining whether or not the layer acts in training or inference mode

        Return:
            x: Value of delta x 
            states1: State vector one (optional)
            states2: State vector two (optional)
        """
        [states1, states2] = hidden_states
        sol = tf.expand_dims(sol, -1)  # Don't change
        log_L = tf.expand_dims(log_L, -1)  # Don't change
        x = tf.concat([sol, log_L], 2)  # Pass previous solution and gradient of log likelihood at previous step; don't change
        x = self.conv1d_1(x, training=training)
        if states1 is None:
            states1 = self.gru1.get_initial_state(x)
        x, states1 = self.gru1(x, initial_state=states1, training=training)
        x = self.conv1d_2(x, training=training)
        if states2 is None:
            states2 = self.gru2.get_initial_state(x)
        x, states2 = self.gru2(x, initial_state=states2, training=training)
        x = self.conv1d_3(x, training=training)
        x = tf.squeeze(x, [-1])  # Don't change
        hidden_states = [states1, states2]
        if return_state:
            return x, hidden_states
        else:
        return x
It is here that you would change the structure depending on whatever architecture you wish to employ.


`RIM_physical`
--------------
In this function we define the gradient of the log likelihood. The standard implementation is as follows:

.. code-block:: python 

    def calc_grad_standard(Y, A, C_N, x):
    """
    Calculate gradient of log likelihood function

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
    conv_sol = tf.einsum('bij,bk->bi', A, x_norm)  # Calculate A*x
    C_N_inv = tf.linalg.inv(C_N)  # Invert C_N
    residual_init = Y - conv_sol  # Returns a bn vector
    residual = tf.einsum("...i, ...ij -> ...j", residual_init, C_N_inv)  # Multiply (Y - conv_sol).T*C_N_inv -> returns a bn vector
    grad = tf.einsum("...i, ...ij -> ...j", residual, A)  # Multiply residual by A

    return tf.math.asinh(grad)
