"""
Standard runfile for RIM on Synthetic Chandra Data

The file works in the following manner:
 - Read in data & extract relavent regions
 - Create training, validation, and test sets
 - Load model and define parameters
 - Fit model on training and validation set
 - Plot results on validation set
 - Run network on test set & record losses/deconvolved spectra
 - Save model weights & test results
 
The observed spectra and true spectra must have the following naming convention:
    `spectra_name.pkl`
    `true_name.pkl`
    
where `name` is a parameter provided in the input file. The rmf must have a similar naming structure (i.e.`rmf_name.pkl`).

There is a standard internal structure for these files as well. Therefore I suggest using the provided code (`create_syn_data.py`) to create
synthetic data and rmfs. If you change the internal structure, you will need to update `RIM_data_generator.py` to reflect the changes.
    
#>> python run_RIM.py nodes conv_filter kernel_size epochs t_steps batch_size learning_rate decay name outputPath dataPath rmfPath

>> python run_RIM.py example.i

"""
import sys
sys.path.insert(0, '/home/carterrhea/Documents/astroRIM')
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import tensorflow as tf
from read_input import read_input_file
from RIM_data_generator import CustomDataGen
from RIM_sequence import RIM
from RIM_model import RIM_Model_1D  # Import name  of architecture to use
from RIM_physical import calc_grad_standard  # Import name of gradient log likelihood
from astropy.io import fits
#nodes, conv_filters, kernel_size, epochs, t_steps, batch_size, learning_rate, decay, name, outputPath, dataPath, rmfPath = sys.argv[1:]
#nodes = int(nodes); conv_filters = int(conv_filters); kernel_size = int(kernel_size); epochs = int(epochs); 
#t_steps = int(t_steps); batch_size = int(batch_size); learning_rate = float(learning_rate); decay = float(decay)
inputFile = read_input_file(sys.argv[1])  # Read input file 
nodes = inputFile['nodes']; conv_filters = inputFile['conv_filters']; kernel_size = inputFile['kernel_size']
learning_rate_function = inputFile['learning_rate_function']; learning_rate = inputFile['learning_rate']
t_steps = inputFile['t_steps']; batch_size = inputFile['batch_size'] 
epochs=inputFile['epochs']; epochs_drop = inputFile['epochs_drop']
dataPath = inputFile['dataPath']; rmfPath = inputFile['rmfPath']
name = inputFile['name']; outputPath = inputFile['outputPath']
rmfName = inputFile['rmfName']
if not os.path.exists(outputPath):
    os.mkdir(outputPath)
# Load model and define hyper parameters
# Initiate RIM architecture to use. We are using the standard RIM archtecture defined in `rim_model.py` as `RIM_Model_1D`.
rim_architecture = RIM_Model_1D(conv_filters=conv_filters, kernel_size=kernel_size, rnn_units=[nodes, nodes])
# Load model and define hyper parameters
dimensions = 1  # Dimensions of the problem
model = RIM(rim_model=rim_architecture, gradient=calc_grad_standard, input_size=140, dimensions=dimensions, t_steps=t_steps, 
            learning_rate=learning_rate, learning_rate_function=learning_rate_function, epochs_drop=epochs_drop, outputPath=outputPath)

#train_dataset = CustomDataGen(X_path=dataPath, A_path=rmfPath, dataType='training', batch_size=batch_size, numData=5000, dataName=name, outputName=outputPath)
#valid_dataset = CustomDataGen(X_path=dataPath, A_path=rmfPath, dataType='validation', batch_size=batch_size, numData=5000, dataName=name, outputName=outputPath)
#responses_data = pickle.load(open(rmfPath+'/rmfs_original.pkl', 'rb'))
spectra_data = pickle.load(open(dataPath+'/spectra_%s.pkl'%name, 'rb'))
true_spectra_data = pickle.load(open(dataPath+'/true_%s.pkl'%name, 'rb'))
# Pull out spectra                                                                                                                                                                           \
                                                                                                                                                                                              
min_ = 35  # Min spectral Channel                                                                                                                                                            \
                                                                                                                                                                                              
max_ = 175  # Max spectral Channel                                                                                                                                                           \
                                                                                                                                                                                              
spectra_x = [data[1][0][0][min_:max_] for data in spectra_data.items()]
spectra_y = [data[1][0][1][min_:max_] for data in spectra_data.items()]
spectra_y_maxes = [np.max(data[1][0][1][min_:max_]) for data in spectra_data.items()]
noise = [data[1][0][2][min_:max_] for ct,data in enumerate(spectra_data.items())]
true_spectra_y = [data for data in true_spectra_data.items()]
true_spectra_y = [data[1][1][min_:max_]  for data in true_spectra_y]
spectra_response = [data[1][1] for data in spectra_data.items()]
#responses = [responses_data[val][:1024, :1024][min_:max_, min_:max_] for val in spectra_response]
response = fits.open(rmfPath+rmfName)[0].data[min_:max_, min_:max_]
# Create training and validation sets                                                                                                                                                        \
                                                                                                                                                                                              
train_percentage = 0.7
valid_percentage = 0.9
test_percentage = 1.0
len_X = len(true_spectra_y)

X_train = true_spectra_y[:int(train_percentage*len_X)]
Y_train = spectra_y[:int(train_percentage*len_X)]
A_train = np.array([responses] * int(train_percentage*len_X))
#C_train = [np.diag(val) for val in noise[:int(train_percentage*len_X)]]                                                                                                                     \
                                                                                                                                                                                              
C_train = noise[:int(train_percentage*len_X)]

X_valid = true_spectra_y[int(train_percentage*len_X):int(valid_percentage*len_X)]
Y_valid = spectra_y[int(train_percentage*len_X):int(valid_percentage*len_X)]
A_valid = np.array([responses] * (int(valid_percentage*len_X) - int(train_percentage*len_X)))
#C_valid = [np.diag(val) for val in noise[int(train_percentage*len_X):int(valid_percentage*len_X)]]                                                                                          \
                                                                                                                                                                                              
C_valid = noise[int(train_percentage*len_X):int(valid_percentage*len_X)]

X_test = true_spectra_y[int(valid_percentage * len_X):]
Y_test = spectra_y[int(valid_percentage * len_X):]
A_test = np.array([responses] * (int(test_percentage*len_X) - int(valid_percentage*len_X)))
                                                                                                                                                          
C_test = noise[int(valid_percentage*len_X):]

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train, A_train, C_train))
train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
train_dataset = train_dataset.prefetch(3)
# Prepare the validation dataset                                                                                                                                                             
valid_dataset = tf.data.Dataset.from_tensor_slices((X_valid, Y_valid, A_valid, C_valid))
valid_dataset = valid_dataset.batch(batch_size, drop_remainder=True)
valid_dataset = valid_dataset.prefetch(3)

test_dataset = tf.data.Dataset.from_tensor_slices((Y_test, A_test, C_test))
test_dataset = test_dataset.batch(batch_size, drop_remainder=True)

# Fit model
ysol_valid, training_loss, validation_loss, learning_rates = model.fit(batch_size, epochs, train_dataset, valid_dataset)
pickle.dump(training_loss, open('%s/training_loss_%in_%ie_%its_%ib_%s.pkl' % (outputPath, nodes, epochs, t_steps, batch_size, name), 'wb'))
pickle.dump(validation_loss, open('%s/validation_loss_%in_%ie_%its_%ib_%s.pkl' % (outputPath, nodes, epochs, t_steps, batch_size ,name), 'wb'))
pickle.dump(learning_rates, open('%s/learning_rate_%in_%ie_%its_%ib_%s.pkl' % (outputPath, nodes, epochs, t_steps, batch_size, name), 'wb'))

# Plot the training vs validation loss
plt.plot(np.linspace(0, len(training_loss), len(training_loss)), training_loss, label='training')
plt.plot(np.linspace(0, len(validation_loss), len(validation_loss)), validation_loss, label='validation')
plt.xlabel('Epoch')
plt.ylabel('Loss Function Value')
plt.legend()
plt.savefig('%s/train_vs_valid_%in_%ie_%its_%ib_%.2Elr_%s.png' % (outputPath, nodes, epochs, t_steps, batch_size, learning_rate, name))
plt.clf()

# Plot the learning rate
plt.plot(np.linspace(0, len(learning_rates[1:]), len(learning_rates[1:])), learning_rates[1:], label='learning rate')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.legend()
plt.yscale('log')
plt.savefig('%s/learning_rates%in_%ie_%its_%s.png' % (outputPath, nodes, epochs, t_steps, name))
plt.clf()


# Run trained network on test set
print('Test data')
#test_dataset = CustomDataGen(X_path=dataPath, A_path=rmfPath, dataType='test', batch_size=batch_size, numData=1000, dataName=name, outputName=outputPath)
ysol = model(test_dataset)
# Save model weights
model.save_weights('%s/weights_%in_%ie_%its_%ib_%.2Elr_%s/weights' % (outputPath, nodes, epochs, t_steps, batch_size, learning_rate, name))
# Save test set results
print('Saving test results')
pickle.dump(ysol, open('%s/ysol_test_%in_%ie_%its_%ib_%.2Elr_%s.pkl' % (outputPath, nodes, epochs, t_steps, batch_size, learning_rate, name), 'wb'))
