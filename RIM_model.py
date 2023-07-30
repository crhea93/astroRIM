import tensorflow as tf
import numpy as np
import time


class RIM_Model_1D(tf.keras.Model):
    """
    Subclass model to create recurrent inference machine architecture for a 1 dimensional version
    """
    def __init__(self, conv_filters, kernel_size, rnn_units):
        """
        Initiation of class

        Args:
            conv_filters: Number of convolutional layers (Int)
            kernel_size: Size of convoluitional kernel (Int)
            rnn_units: List of units in GRUs ([Int])

        """
        self.rnn_units = rnn_units
        super().__init__(self)
        # Define Layers of RIM
        self.conv1d_1 = tf.keras.layers.Conv1D(filters=conv_filters, kernel_size=kernel_size, strides=1,
                                               padding='same', activation='tanh')
        self.gru1 = tf.keras.layers.GRU(self.rnn_units[0], activation='tanh', recurrent_activation='sigmoid',
                                        return_sequences=True, return_state=True)
        self.conv1d_2 = tf.keras.layers.Conv1DTranspose(filters=conv_filters, kernel_size=kernel_size, strides=1,
                                                        padding='same', activation='tanh')
        self.gru2 = tf.keras.layers.GRU(self.rnn_units[1], activation='tanh', recurrent_activation='sigmoid',
                                        return_sequences=True, return_state=True)
        self.conv1d_3 = tf.keras.layers.Conv1D(filters=1, kernel_size=kernel_size, strides=1,
                                                        padding='same', activation='linear')


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
        #print(sol.shape, log_L.shape)
        sol = tf.expand_dims(sol, -1)
        #sol = tf.cast(sol, dtype=tf.float16)
        log_L = tf.expand_dims(log_L, -1)
        x = tf.concat([sol, log_L], 2)  # Pass previous solution and gradient of log likelihood at previous step
        #x = tf.expand_dims(x, -1)
        x = self.conv1d_1(x, training=training)
        if states1 is None:
            states1 = self.gru1.get_initial_state(x)
        x, states1 = self.gru1(x, initial_state=states1, training=training)
        x = self.conv1d_2(x, training=training)
        if states2 is None:
            states2 = self.gru2.get_initial_state(x)
        x, states2 = self.gru2(x, initial_state=states2, training=training)
        x = self.conv1d_3(x, training=training)
        x = tf.squeeze(x, [-1])
        #x = tf.cast(x, dtype=tf.float16)
        if return_state:
            return x, [states1, states2]
        else:
            return x



class RIM_Model_2D(tf.keras.Model):
    """
    Subclass model to create recurrent inference machine architecture
    """
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
        conv_filters = 1
        kernel_size = (1, 1)
        self.conv2d_1 = tf.keras.layers.Conv2D(filters=conv_filters, kernel_size=kernel_size, strides=(1,1),
                                               padding='same', activation='tanh')
        self.gru_setup1 = tf.keras.layers.Flatten()
        self.gru_setup2 = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))
        self.gru1 = tf.keras.layers.GRU(rnn_units[0], activation='tanh', recurrent_activation='sigmoid',
                                        return_sequences=True, return_state=True)
        self.conv2d_2 = tf.keras.layers.Conv2DTranspose(filters=conv_filters, kernel_size=kernel_size, strides=(1, 1),
                                                        padding='same', activation='tanh')
        self.gru2 = tf.keras.layers.GRU(rnn_units[1], activation='tanh', recurrent_activation='sigmoid',
                                        return_sequences=True, return_state=True)
        self.conv2d_3 = tf.keras.layers.Conv2D(filters=1, kernel_size=kernel_size, strides=(1, 1),
                                               padding='same', activation='linear')
        self.rnn_units = rnn_units  # DO NOT CHANGE THIS LINE

    def call(self,sol, log_L, hidden_states, return_state=True, training=False):
        """
        A single run through the recurrent inference machine
        conv2d -> gru -> conv2d_T -> gru -> conv2d
        When we go from a convolutional layer to a GRU, we need to first flatten the activation map. We then use the
        expand_dims function to make sure the input of the GRU has the correct format (batch length, feature dims, 1).
        Args:
            sol: solution at time step t (x_t)
            log_L: Gradient of log likelihood
            hidden_states: List of hidden state vectors
            return_state: Return the hidden state boolean
            training: Boolean determining whether or not the layer acts in training or inference mode
        """
        [states1, states2] = hidden_states
        self.conv_setup1 = tf.keras.layers.Reshape((sol[0].shape[0], sol[0].shape[0], self.rnn_units[0]))
        self.conv_setup2 = tf.keras.layers.Reshape((sol[0].shape[0], sol[0].shape[0], self.rnn_units[1]))
        sol = tf.expand_dims(sol, -1)  # Don't change
        log_L = tf.expand_dims(log_L, -1)  # Don't change
        x = tf.concat([sol, log_L], 3)  # Pass previous solution and gradient of log likelihood at previous step; don't change
        x = self.conv2d_1(x, training=training)
        if states1 is None:
            states1 = self.gru1.get_initial_state(x)
        x = self.gru_setup1(x)
        x = self.gru_setup2(x)
        x, states1 = self.gru1(x, initial_state=states1, training=training)
        x = self.conv_setup1(x)
        x = self.conv2d_2(x, training=training)
        if states2 is None:
            states2 = self.gru2.get_initial_state(x)
        x = self.gru_setup1(x)
        x = self.gru_setup2(x)
        x, states2 = self.gru2(x, initial_state=states2, training=training)
        x = self.conv_setup2(x)
        x = self.conv2d_3(x, training=training)
        x = tf.squeeze(x, [-1])  # Don't change
        hidden_states = [states1, states2]
        if return_state:
            return x, hidden_states
        else:
            return x
