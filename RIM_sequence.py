import numpy as np
import tensorflow as tf
import time
from datetime import datetime
import sys
import os
LOGFLOOR = tf.constant(1e-8, dtype=tf.float32)
os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

class RIM(tf.keras.Model):
    """
    Subclass model to create recurrent inference machine

    If the problem is 2D, we assume the input is square.

    """
    def __init__(self, rim_model, gradient, input_size, dimensions=1, t_steps=10, learning_rate=0.01, decay=0.5, 
                patience=10, learning_rate_function='step', epochs_drop=10
                ):
        """
        Initialize Recurrent Inference Machine

        Args:
            rim_model: Instance of RIM Model class
            gradient: Instance of gradient of likelihood function 
            input_size: Size of input vector
            dimensions: Number of dimensions of the problem (default 1)
            t_steps: Number of time steps in RIM (default 10)
            learning_rate: Initial learning rate value (default 1e-2)
            decay: Decay rate (default 0.5)
            patience: Number of epochs to wait before decaying (default 10)
            learning_rate_function: Function to use to update learning rate (default 'step'; other options are 'exponential' and 'linear')
            epochs_drop: Number of epochs before decaying the learning rate (only if learning_rate_function='step'; defaut 10)

        Returns:
            Instance of RIM ready to be fit
        
        """
        super().__init__(self)
        # Define Optimization Function, Loss Function, and Metrics
        self.learning_rate_function = self.assign_learning_rate_function(learning_rate_function)  # Set function for learning rate: options ['drop', 'exponential', 'linear']
        self.learning_rate = learning_rate#tf.cast(learning_rate, tf.float32)  # Set learning rate
        self.learning_rate_init = learning_rate  # Set initial learning rate
        self.decay = decay#tf.cast(decay, tf.float32)  # Set decay value
        self.epochs_drop = epochs_drop  # Set number of epochs to wait until dropping the learning rate
        self.patience = tf.cast(patience, tf.int16)  # Set patience for early stopping
        self.optimizer = tf.keras.optimizers.Adamax(learning_rate=self.learning_rate, clipnorm=5)  # Set adamax optimzer with clipnorm
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.train_acc_metric = tf.keras.metrics.MeanSquaredError()
        self.val_acc_metric = tf.keras.metrics.MeanSquaredError()
        self.size_ = input_size  # Size of input for either spectrum or 2D image
        self.dimensions = dimensions  # Number of dimensions in problem
        self.t_steps = t_steps  # Number of time steps (number of times the RIM is run)
        self.model = rim_model#RIM_Model_1D(self.conv_filters, self.kernel_size, self.rnn_units1, self.rnn_units2)  # Initialize Model
        self.batch_size = 1  # Intialize batch size -- will be overwritten in fit function
        self.calc_grad = gradient

        # Setup log file
        self.log = open('log_%s.txt' % (datetime.now().strftime("%m%d%Y%H:%M:%S")), 'w+')


    def init_states(self, batch_size):
        """
        Initialize hidden state1 and hidden state2.

        Args:
            batch_size: Number of spectra in batch
        
        """
        #h_1 = None
        #h_2 = None
        if self.dimensions == 1:
            # Create the number of hidden states equivalent to the number of GRUs
            hidden_vectors = [tf.zeros(shape=(batch_size, val)) for val in self.model.rnn_units]
            #h_1 = tf.zeros(shape=(batch_size, self.rnn_units1))
            #h_2 = tf.zeros(shape=(batch_size, self.rnn_units2))
        elif self.dimensions == 2:
            #h_1 = tf.zeros(shape=(batch_size, self.rnn_units1, self.rnn_units1))
            #h_2 = tf.zeros(shape=(batch_size, self.rnn_units2, self.rnn_units2))
            hidden_vectors = [tf.zeros(shape=(batch_size, self.model.rnn_units[val], self.model.rnn_units[val])) for val in len(self.model.rnn_units)]
        else:
            print('Please enter a valid dimension size (1 or 2)')
        return hidden_vectors

    def init_sol(self, batch_size):
        """
        Initialize solution

        Args:
            batch_size: Number of spectra in batch
        
        """
        y_init = None
        if self.dimensions == 1:
            y_init = tf.ones(shape=(batch_size, self.size_))
        elif self.dimensions == 2:
            y_init = tf.ones(shape=(batch_size, self.size_, self.size_))
        else:
            print('Please enter a valid dimension size (1 or 2)')
        return y_init


    @tf.function
    def learning_rate_decay(self, epoch):
        r"""
        Update the learrning rate at the end of each epoch based on the decay equation
        
        .. math::
            \alpha=\frac{\alpha_0}{1+\eta*\epsilon}

        Args:
            epoch: Current epoch of training (epsilon)

        Return:
            Updated learning rate

        """
        decay_term = (1 + self.decay * epoch)
        self.learning_rate = self.learning_rate/decay_term


    @tf.function
    def exp_decay(self, epoch):
        r"""
        Update the learrning rate at the end of epoch  drop based on the decay equation

        .. math::
            \alpha = \frac{\alpha_0}{1+\eta*floor(\epsilon/\text{epoch drop})}

        Args:
            epoch: Current epoch of training (epsilon)

        Return:
            Updated learning rate
        
        """
        decay_term = (1 + self.decay * tf.math.floor((epoch)/self.epochs_drop))
        self.learning_rate = self.learning_rate/decay_term
    

    #@tf.function
    def step_decay(self, epoch):
        r"""
        Update the learning rate at the end of a given epoch 

        .. math::
            \alpha = \alpha_0 * \eta**floor(\epsilon/\text{epoch drop})

        Args: 
            epoch: Current epoch of training (epsilon)

        Return:
            Updated learning rate

        """
    
        self.learning_rate = self.learning_rate_init * tf.math.pow(self.decay, tf.math.floor(epoch/self.epochs_drop))

    def assign_learning_rate_function(self,learning_rate_function):
        """
        Assign learning rate function based off user's input

        Args:
            learning_rate_function: Selected learning rate function
        """
        lrf = None  # Set learning rate function
        if learning_rate_function == 'step':
            lrf = self.step_decay
        elif learning_rate_function == 'exponental':
            lrf = self.exp_decay
        elif learning_rate_function == 'linear':
            lrf = self.learning_rate_decay
        else:
            print("Please enter one of the following options for the learning_rate_function: ['step', 'exponential', 'linear']")
            print('Terminating program')
            sys.exit()
        return lrf

    @tf.function
    def mean_mse(self, y_true, x_sol):
        """
        Calculate mean squared error (mse) over all time steps

        Args:
            y_true: True solution
            x_sol: Final updated solution from RIM

        Return: 
            mse value
        """
        x_sol = tf.cast(x_sol, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        return tf.math.reduce_mean(tf.math.square(y_true-x_sol))

    @tf.function
    def train_step(self, step, x, y, model, A, C, batch_size):
        """
        Test network on training data

        Args:
            step: Current time step
            x: Batched true spectra for training set
            y: Batched observed spectra for training set
            model: RIM_model instance
            A: Batched response matrix for training set
            C: Batched covariance matrix for training set
            batch_size: Number of spectra in a batch

        Return:
            train_loss_fun: Loss function vald for training set batch

        """
        x = tf.cast(x, dtype=tf.float32)
        # Initialize States
        hidden_states = self.init_states(batch_size)
        # Initialize Solution
        xi_t = self.init_sol(batch_size)
        train_loss_value = 0  # Initialize training loss
        # start the scope of gradient
        with tf.GradientTape(persistent=True) as tape:
            # We run this for the number of time steps in the RIM
            for t_step in range(self.t_steps):
                # Calculate the gradient of the log likelihood function
                with tape.stop_recording():  # Dont record tape becuase it uneccessarily slows things down and isnt need here!
                    log_L = self.calc_grad(y, A, C, xi_t)
                logits, hidden_states = model(xi_t, log_L, hidden_states=hidden_states, training=True, return_state=True)  # forward pass
                xi_t = xi_t + logits  # Update Solution where x_t = del_x + x_(t-1). del_x == logits; x_(t-1) == sol_t
                train_loss_value += self.mean_mse(x, xi_t)  # compute loss on updated solutions (i.e. at each time step) in neural network space
            train_loss_value /= self.t_steps  # Have to normalize
            # compute gradient
        grads = tape.gradient(train_loss_value, model.trainable_weights)
        # Clip by norm each layer
        # update weights
        self.optimizer.apply_gradients(zip(grads, model.trainable_weights))
        # update metrics
        self.train_acc_metric.update_state(x, xi_t)
        # Done looping through time.
        del tape
        return train_loss_value


    @tf.function
    def valid_step(self, step, x, y, model, A, C, batch_size, return_state=True):
        """
        Test network on validation data  (no training)

        Args:
            step: Current time step
            x: Batched true spectra for validation set
            y: Batched observed spectra for validation set
            model: RIM_model instance
            A: Batched response matrix for validation set
            C: Batched noise covariance matrix for validation set
            batch_size: Number of spectra in a batch

        Return:
            val_loss_fun: Loss function vald for validation set batch
            sol_t: Updated solution given the validation step

        """
        # Initialize States
        hidden_states = self.init_states(batch_size)
        # Initialize Solution
        sol_t = self.init_sol(batch_size)
        # forward pass, no backprop, inference mode  # We run this for the number of time steps in the RIM
        for t_step in range(self.t_steps):
            log_L = self.calc_grad(y, A, C, sol_t)
            val_logits, hidden_states = model(sol_t, log_L, hidden_states=hidden_states, training=False)
            sol_t = sol_t + val_logits
        # Compute the loss value
        val_loss_value = self.mean_mse(x, sol_t)
        # Update val metrics
        self.val_acc_metric.update_state(x, sol_t)
        return val_loss_value, sol_t

    @tf.function
    def test_step(self, step, y, model, A, C, batch_size, return_state=True):
        """
        Test network on test data  (no training)

        Args:
            step: Current time step
            y: Batched observed spectra for test set
            model: RIM_model instance
            A: Batched response matrix for test set
            C: Batched noise covariance matrix for test set
            batch_size: Number of spectra in a batch

        Return:
            solution_step: Updated solution given the test step

        """
        solution_step = []  # Solution at each time step
        # Initialize States
        hidden_states = self.init_states(batch_size)
        # Initialize Solution
        sol_t = self.init_sol(batch_size)
        # forward pass, no backprop, inference mode # We run this for the number of time steps in the RIM
        for t_step in range(self.t_steps):
            log_L = self.calc_grad(y, A, C, sol_t)
            val_logits, hidden_states = model(sol_t, log_L, hidden_states=hidden_states, training=False)
            sol_t = sol_t + val_logits
            solution_step.append(sol_t)
        return solution_step

    def fit(self, batch_size, epochs, train_dataset, val_dataset):
        """
        A full training and validation algorithm

        Call with the following ::

            ysol_valid, training_loss, valid_loss, learning_rates = model.fit(batch_size, epochs, train_dataset, val_dataset)

        Args:
            batch_size: number of batches (int)
            epochs: number of epochs (int)
            train_dataset: batched training set (X_train, Y_train, A_train, C_train)
            val_dataset: batched validation set (X_valid, Y_valid, A_valid, C_valid)
        
        Return:
            ysol: Solution vectors for validation set (batch, timesteps, values)
            training_loss_values: Vector of training loss values
            valid_loss_values: Vector of validation loss values
            learning_rate_values: Vector of learning rate values

        """
        #self.log.write('Starting log at %s\n'%((time.strftime("%H:%M:%S", start_total))))
        start_total = time.time()  # Training start time
        wait = 0  # Set wait time for early stopping
        best = 0  # Best fit for early stopping
        # custom training loop
        self.batch_size = batch_size
        global train_loss_value, val_loss_value
        # Define some output stuffs
        template_train = 'Training epoch: {0}::  Completion: {1:.2f}%  ETA {2:}  loss: {3:.3E}  MSE: {4:.3E}\n'
        template_valid= 'Validation epoch: {0}::  Completion: {1:.2f}%  ETA: {2:}  train_loss: {3:.3E}  train_MSE: {4:.3E}  val_loss: {5:.3E} val_MSE: {6:.3E}\n'
        start_time = time.time()
        time_since_prev = start_time  # Initialize the time since the previous call finished
        max_batch_step = len(train_dataset)
        training_loss_values = []  # List of trianing loss values -- one for each epoch
        valid_loss_values = []  # List of trianing loss values -- one for each epoch
        learning_rate_values = []  # List of learning rate value -- one for each epoch
        learning_rate_values.append(self.learning_rate)  # Initialize
        for epoch in range(epochs):  # Step through algorithm for each epoch
            start_epoch = time.time()  # Training start time
            t = time.time()
            # Iterate over the batches of the train dataset.
            for train_batch_step, (x_batch_train, y_batch_train, a_train_batch, c_batch_train) in enumerate(train_dataset):
                x_batch_train = tf.cast(x_batch_train, dtype=tf.float32)
                y_batch_train = tf.cast(y_batch_train, dtype=tf.float32)
                c_batch_train = tf.cast(c_batch_train, dtype=tf.float32)
                train_batch_step = tf.convert_to_tensor(train_batch_step, dtype=tf.float32)
                train_loss_value = self.train_step(train_batch_step, x_batch_train,
                                                             y_batch_train,
                                                             self.model, a_train_batch, c_batch_train, batch_size=batch_size)
                # Output!
                current_percent = np.round(100*(train_batch_step/max_batch_step), 2)
                current_step = float(train_batch_step/max_batch_step)
                if int(train_batch_step) % 10 == 0:
                    ETA = time.gmtime(int(max_batch_step-train_batch_step)*(time.time()-time_since_prev))
                    print(template_train.format(
                        epoch + 1, current_percent, time.strftime("%H:%M:%S", ETA),
                        train_loss_value, float(self.train_acc_metric.result())
                    ), end="\r")
                time_since_prev = time.time()  # Update time of previous call
            # Evaluation on validation set -- Run a validation loop at the end of each epoch.
            for val_batch_step, (x_batch_val, y_batch_val, a_batch_val, c_batch_val) in enumerate(val_dataset):
                x_batch_val = tf.cast(x_batch_val, dtype=tf.float32)
                y_batch_val = tf.cast(y_batch_val, dtype=tf.float32)
                c_batch_val = tf.cast(c_batch_val, dtype=tf.float32)
                val_batch_step = tf.convert_to_tensor(val_batch_step, dtype=tf.float32)
                val_loss_value, ysol = self.valid_step(val_batch_step, x_batch_val, y_batch_val, self.model, a_batch_val, c_batch_val, batch_size=batch_size)
                # More output!
                current_percent = np.round(100*(val_batch_step/max_batch_step), 2)
                if int(val_batch_step) % 10 == 0:
                    ETA = time.gmtime(int(max_batch_step-val_batch_step)*(time.time()-time_since_prev))
                    print(template_valid.format(
                        epoch + 1, current_percent, time.strftime("%H:%M:%S", ETA),
                        train_loss_value, float(self.train_acc_metric.result()),
                        val_loss_value, float(self.val_acc_metric.result())
                    ), end="\r")
                time_since_prev = time.time()  # Update time of previous call
            # End of Epoch training/validation -- now several print statements and calculations
            print(template_valid.format(
                epoch + 1, 100, '0',
                train_loss_value, float(self.train_acc_metric.result()),
                val_loss_value, float(self.val_acc_metric.result())
            ))
            self.log.write(template_valid.format(
                epoch + 1, 100, '0',
                train_loss_value, float(self.train_acc_metric.result()),
                val_loss_value, float(self.val_acc_metric.result())
            ))
            # Reset metrics at the end of each epoch
            train_acc = self.train_acc_metric.result()
            val_acc = self.val_acc_metric.result()
            self.train_acc_metric.reset_states()
            self.val_acc_metric.reset_states()
            print("Training MSE: %.4f" % (float(train_acc),))
            print("Validation MSE: %.4f" % (float(val_acc),))
            total_time = time.gmtime(float(time.time() - start_epoch))
            print("Time taken on epoch: %s seconds \n\n" % (time.strftime("%H:%M:%S", total_time)))
            training_loss_values.append(train_loss_value)
            valid_loss_values.append(val_loss_value)
            self.model.save_weights('./tmp/weights%i'%epoch)  # Save weights for epoch in temporary folder
            # Check for early stopping
            wait += 1  # Update waiting integer
            if val_loss_value > best:
                best = val_loss_value  # Update best fit
                wait = 0  # Reset wait 
            if wait >= self.patience:  # If the validation loss has not decreased in X (defined by self.patience) steps
                pass # break
            # Update learning rate
            self.learning_rate_function(epoch)
            self.optimizer.learning_rate = self.learning_rate  # Reinitialize optimizer with new learning rate
            learning_rate_values.append(self.learning_rate.numpy())  # Add new value to list of learning rate values
        # End time steps
        return ysol, training_loss_values, valid_loss_values, learning_rate_values

    def call(self, test_dataset, training=False):
        """
        A single run through the recurrent inference machine for prediction purposes

        Args:
            test_dataset: batch test set (X_test, Y_test, A_test)

        Return:
            solutions: Solution vector of the form (test_number, timesteps, vector)
        """
        solutions = []  # List containing all the solutions at each timestep
        for test_batch_step, (y_batch_test, a_batch_test, c_batch_test) in enumerate(test_dataset):
            test_batch_step = tf.convert_to_tensor(test_batch_step, dtype=tf.int32)
            sol_t = self.test_step(test_batch_step, y_batch_test, self.model, a_batch_test, c_batch_test, batch_size=self.batch_size)
            solutions.append(sol_t)
        return solutions

    def predict(self, test_dataset, training=False):
        """
        A single run through the recurrent inference machine for prediction purposes

        Args:
            test_dataset: batch test set (X_test, Y_test, A_test)

        """
        solution_steps = []
        y = test_dataset[0]
        A = test_dataset[1]
        C = test_dataset[2]
        hidden_states = self.init_states(1)
        # Initialize Solution
        sol_t = self.init_sol(1)
        for t_step in range(self.t_steps):
            log_L = self.calc_grad(y, A, C, sol_t)
            val_logits, hidden_states = self.model(sol_t, log_L, hidden_states=hidden_states, training=False)
            sol_t = sol_t + val_logits
            solution_steps.append(sol_t)
        return solution_steps
