nodes = 256  # Nodes in GRUs
conv_filters = 4  # Number of Convolutional Filters per Convolutional Layer
kernel_size = 3  # Convolutional Kernel Size 
epochs = 10  # Training Epochs 
t_steps = 5  # Time steps in RIM 
batch_size = 256  # Training batch size  
learning_rate = 0.002  # Initial learning rate 
learning_rate_function  = step  # Type of learning rate function (options are: step, exponential, or linear)
decay = 0.85  # Decay rate 
epochs_drop = 2  # Number of epochs before changing learning rate
name = new  # Appendix of name of data (i.e. name=mini if data is called 'spectra_mini.pkl')
outputPath = /home/carterrhea/Downloads/Outputs  # Output data path 
dataPath = /home/carterrhea/Documents/RIM_data  # Full path to spectra pickle files (i.e. `spectra_name.pkl` and `true_name.pkl`)
rmfPath = /home/carterrhea/Documents/RIM_data  # Full path to rmf pickle file (i.e. `rmf_name.pkl`)
