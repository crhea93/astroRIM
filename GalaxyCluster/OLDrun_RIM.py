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
from read_input import read_input_file
from RIM_data_generator import CustomDataGen
from RIM_sequence import RIM
from RIM_model import RIM_Model_1D  # Import name  of architecture to use
from RIM_physical import calc_grad_standard  # Import name of gradient log likelihood

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
if not os.path.exists(outputPath):
    os.mkdir(outputPath)
# Load model and define hyper parameters
# Initiate RIM architecture to use. We are using the standard RIM archtecture defined in `rim_model.py` as `RIM_Model_1D`.
rim_architecture = RIM_Model_1D(conv_filters=conv_filters, kernel_size=kernel_size, rnn_units=[nodes, nodes])
# Load model and define hyper parameters
dimensions = 1  # Dimensions of the problem
model = RIM(rim_model=rim_architecture, gradient=calc_grad_standard, input_size=140, dimensions=dimensions, t_steps=t_steps, 
            learning_rate=learning_rate, learning_rate_function=learning_rate_function, epochs_drop=epochs_drop, outputPath=outputPath)

train_dataset = CustomDataGen(X_path=dataPath, A_path=rmfPath, dataType='training', batch_size=batch_size, numData=5000, dataName=name, outputName=outputPath)
valid_dataset = CustomDataGen(X_path=dataPath, A_path=rmfPath, dataType='validation', batch_size=batch_size, numData=5000, dataName=name, outputName=outputPath)

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
test_dataset = CustomDataGen(X_path=dataPath, A_path=rmfPath, dataType='test', batch_size=batch_size, numData=1000, dataName=name, outputName=outputPath)
ysol = model(test_dataset)
# Save model weights
model.save_weights('%s/weights_%in_%ie_%its_%ib_%.2Elr_%s/weights' % (outputPath, nodes, epochs, t_steps, batch_size, learning_rate, name))
# Save test set results
print('Saving test results')
pickle.dump(ysol, open('%s/ysol_test_%in_%ie_%its_%ib_%.2Elr_%s.pkl' % (outputPath, nodes, epochs, t_steps, batch_size, learning_rate, name), 'wb'))
