import tensorflow as tf
import numpy as np

class CustomDataGen(tf.keras.utils.Sequence):
    """
    Custom data generator to read in data from pickle file 
    """
    
    def __init__(self, X, Y, A, C, ids,
                 batch_size,
                ):
        
        self.X = X
        self.Y = Y
        self.A = A
        self.C = C
        self.ids = ids
        self.batch_size = batch_size
        
        self.n = len(self.ids)
    
    def __getitem__(self, idx):
        
        #batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        batch_x = self.X[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_y = self.X[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_a = self.A[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_c = self.C[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        #X, y = self.__get_data(batches)        
        return batch_x, batch_y, batch_a, batch_c
    
    def __len__(self):
        return self.n // self.batch_size
