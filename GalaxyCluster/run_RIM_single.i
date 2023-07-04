nodes = 1000  # Nodes in GRUs
conv_filters = 8  # Number of Convolutional Filters per Convolutional Layer
kernel_size = 3  # Convolutional Kernel Size 
epochs = 30  # Training Epochs 
t_steps = 4  # Time steps in RIM 
batch_size = 32  # Training batch size  
learning_rate = 0.002  # Initial learning rate 
learning_rate_function  = step  # Type of learning rate function (options are: step, exponential, or linear)
decay = 0.9  # Decay rate 
epochs_drop = 8  # Number of epochs before changing learning rate
name = 5286  # Appendix of name of data (i.e. name=mini if data is called 'spectra_mini.pkl')
outputPath = /home/crhea/home/astroRIM/GalaxyCluster  # Output data path 
dataPath = /home/crhea/home  # Full path to spectra pickle files (i.e. `spectra_name.pkl` and `true_name.pkl`)
rmfPath = /home/crhea/home  # Full path to rmf pickle file (i.e. `rmf_name.pkl`)
rmfName = 5286_rmf.img
