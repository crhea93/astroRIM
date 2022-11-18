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
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from RIM_sequence import RIM
import sys
from RIM_data_generator import CustomDataGen

nodes, conv_filters, kernel_size, epochs, t_steps, batch_size, learning_rate, name, output_name = sys.argv[1:]
nodes = int(nodes); conv_filters = int(conv_filters); kernel_size = int(kernel_size); epochs = int(epochs); 
t_steps = int(t_steps); batch_size = int(batch_size); learning_rate = float(learning_rate); 
# Read in spectra and responses
responses_data = pickle.load(open('rmfs.pkl', 'rb'))
spectra_data = pickle.load(open('spectra_%s.pkl'%name, 'rb'))
true_spectra_data = pickle.load(open('true_spectra_%s.pkl'%name, 'rb'))
# Pull out spectra
min_ = 35  # Min spectral Channel
max_ = 175  # Max spectral Channel
spectra_x = [data[1][0][0][min_:max_] for data in spectra_data.items()]
spectra_y = [data[1][0][1][min_:max_]/np.max(data[1][0][1][min_:max_]) for data in spectra_data.items()]
spectra_y_maxes = [np.max(data[1][0][1][min_:max_]) for data in spectra_data.items()]
noise = [data[1][0][2][min_:max_]/spectra_y_maxes[ct] for ct,data in enumerate(spectra_data.items())]
true_spectra_y = [data for data in true_spectra_data.items()]
true_spectra_y = [data[1][1][min_:max_]/np.max(data[1][1][min_:max_])  for data in true_spectra_y]
spectra_response = [data[1][1] for data in spectra_data.items()]
responses = [responses_data[val][:1024, :1024][min_:max_, min_:max_] for val in spectra_response]

# Create training and validation sets                                                                                                   
train_percentage = 0.7
valid_percentage = 0.9
test_percentage = 1.0
len_X = len(true_spectra_y)

X_train = true_spectra_y[:int(train_percentage*len_X)]
Y_train = spectra_y[:int(train_percentage*len_X)]
A_train = responses[:int(train_percentage*len_X)]
#C_train = [np.diag(val) for val in noise[:int(train_percentage*len_X)]]
C_train = noise[:int(train_percentage*len_X)]

X_valid = true_spectra_y[int(train_percentage*len_X):int(valid_percentage*len_X)]
Y_valid = spectra_y[int(train_percentage*len_X):int(valid_percentage*len_X)]
A_valid = responses[int(train_percentage*len_X):int(valid_percentage*len_X)]
#C_valid = [np.diag(val) for val in noise[int(train_percentage*len_X):int(valid_percentage*len_X)]]
C_valid = noise[int(train_percentage*len_X):int(valid_percentage*len_X)]

X_test = true_spectra_y[int(valid_percentage * len_X):]
Y_test = spectra_y[int(valid_percentage * len_X):]
A_test = responses[int(valid_percentage * len_X):]
#C_test = [np.diag(val) for val in noise[int(valid_percentage*len_X):]]
C_test = noise[int(valid_percentage*len_X):]

# Plot data for verification
fig = plt.figure(figsize=(16,8))
for i in range(10):
    test_spec = spectra_y[i]
    plt.plot(np.linspace(0,len(test_spec), len(test_spec)), test_spec)
plt.savefig('Outputs/test_convolved_%s.png'%name)
plt.clf()

fig = plt.figure(figsize=(16,8))
for i in range(10):
    test_spec = true_spectra_y[i]
    plt.plot(np.linspace(0,len(test_spec), len(test_spec)), test_spec)
plt.savefig('Outputs/test_true_%s.png'%name)
plt.clf()

# Load model and define hyper parameters
n = len(Y_train[0])
model = RIM(rnn_units1=nodes, rnn_units2=nodes, conv_filters=conv_filters, kernel_size=kernel_size, input_size=n, dimensions=1,
            t_steps=t_steps, learning_rate=learning_rate)

train_dataset = CustomDataGen(X=X_train, Y=Y_train, A=A_train, C=C_train, ids=np.arange(0, int(train_percentage*len_X)), batch_size=batch_size)
valid_dataset = CustomDataGen(X=X_valid, Y=Y_valid, A=A_valid, C=C_valid, ids=np.arange(int(train_percentage*len_X), int(valid_percentage*len_X)), batch_size=batch_size)

# Fit model
ysol_valid, training_loss, validation_loss, learning_rates = model.fit(batch_size, epochs, train_dataset, valid_dataset)
pickle.dump(training_loss, open('Outputs/training_loss_%in_%ie_%its_%ib_%s.pkl' % (nodes, epochs, t_steps, batch_size, name), 'wb'))
pickle.dump(validation_loss, open('Outputs/validation_loss_%in_%ie_%its_%ib_%s.pkl' % (nodes, epochs, t_steps, batch_size ,name), 'wb'))
pickle.dump(learning_rates, open('Outputs/learning_rate_%in_%ie_%its_%ib_%s.pkl' % (nodes, epochs, t_steps, batch_size, name), 'wb'))

# Plot the training vs validation loss
plt.plot(np.linspace(0, len(training_loss), len(training_loss)), training_loss, label='training')
plt.plot(np.linspace(0, len(validation_loss), len(validation_loss)), validation_loss, label='validation')
plt.legend()
plt.savefig('Outputs/train_vs_valid_%in_%ie_%its_%ib_%.2Elr_%s.png' % (nodes, epochs, t_steps, batch_size, learning_rate, name))
plt.clf()

# Plot the learning rate
plt.plot(np.linspace(0, len(learning_rates[1:]), len(learning_rates[1:])), learning_rates[1:], label='learning rate')
plt.legend()
plt.yscale('log')
plt.savefig('Outputs/learning_rates%in_%ie_%its_%s.png' % (nodes, epochs, t_steps, name))
plt.clf()


# Run trained network on test set
#test_dataset = tf.data.Dataset.from_tensor_slices((Y_test, A_test, C_test))
#test_dataset = test_dataset.batch(batch_size, drop_remainder=True)
test_dataset = CustomDataGen(X=X_test, Y=Y_test, A=A_test, C=C_test, ids=np.arange(int(valid_percentage*len_X), len_X), batch_size=batch_size)
print('Test data')
ysol = model(test_dataset)
# Save model weights
model.save_weights('Outputs/weights_%in_%ie_%its_%ib_%.2Elr_%s/weights' % (nodes, epochs, t_steps, batch_size, learning_rate, name))
# Save test set results
pickle.dump(ysol, open('Outputs/ysol_test_%in_%ie_%its_%ib_%.2Elr_%s.pkl' % (nodes, epochs, t_steps, batch_size, learning_rate, name), 'wb'))
print('Saving test results')
ysol_list = []
for val in ysol:
    ysol = [val.numpy() for val in val]
    ysol_list.append(ysol)

fig = plt.figure(figsize=(16,8))
plt.clf()
plt.plot(np.linspace(-1,1,n), Y_test[-1]/np.max(Y_test[-1]), label='Convolved', color='C1')
plt.plot(np.linspace(-1,1,n), X_test[-1], label='True',  color='C0', linewidth=4)
plt.plot(np.linspace(-1,1,n), ysol_list[-1][-1][-1].reshape(n), label='Predicted', linestyle='dashed', color='C2', linewidth=3)
plt.legend(prop={'size': 20})
plt.ylabel('Normalized y-axis', fontsize=20)
plt.xlabel('X-axis', fontsize=20)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.title('RIM Solution for Random Test')
plt.savefig('Outputs/Test_Example_%in_%ie_%its_%ib_%.2Elr_%s.png' % (nodes, epochs, t_steps, batch_size, learning_rate, name))


plt.clf()
nrows = 3
ncols = 3
fig, axs = plt.subplots(nrows, ncols, figsize=(24,18))
plt.ylabel('Normalized Photon Flux')
plt.xlabel('Energy (keV)')
for i in range(nrows):
    for j in range(ncols):
        test_index = int(np.random.uniform(0,992))
        ysol_plot = ysol_list[test_index][-1]  # Last solution at last time step
        axs[i,j].plot(spectra_x[-1], Y_test[test_index]/np.max(Y_test[test_index]), label='Observed')
        axs[i,j].plot(spectra_x[-1], ysol_plot/np.max(ysol_plot), label='Predicted', linewidth=2, color='C2')
        axs[i,j].plot(spectra_x[-1], X_test[test_index][-1][min_:max_]/np.max(X_test[test_index][-1][min_:max_]), label='True', linestyle='--', color='C3')
        axs[i,j].legend(loc=1, prop={'size': 20})
        axs[i,j].title.set_text('Test Spectrum %i'%test_index)
        
plt.savefig('Outputs/Random_Test_Example_%in_%ie_%its_%ib_%.2Elr_%s.png' % (nodes, epochs, t_steps, batch_size, learning_rate, name))

