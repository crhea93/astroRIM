import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import pickle
import os 

def max_calc(array):
    max_val = np.max(array)
    if max_val < 1e-16:
        max_val = 1
    return max_val



class CustomDataGenClusters(tf.keras.utils.Sequence):
    """
    Custom data generator to read in data from pickle file 
    """
    
    def __init__(self, X_path, A_path, clusterIDs, batch_size, dataType, numData
                ):
        
        self.X_path = X_path
        self.A_path = A_path
        self.clusterIDs = clusterIDs
        if type(self.clusterIDs[0]) != str():  # Make sure the IDs are in string form
            self.clusterIDs = [str(clusterID) for clusterID in self.clusterIDs]
        self.batch_size = batch_size
        self.dataType = dataType.lower()  # Either training, validation, or test
        self.numData = numData  # The number of sampels in the generator
        self.X_data = [] 
        self.Y_data = []
        self.A_data = [] 
        #self.collectDataFiles()  # Get data files
        
        
    def __len__(self):
        return int(self.numData) // self.batch_size

    
    

    def __get_data__(self, clusterID):
        """_summary_
        Given batch of data we will randomly load in spectra and their corresponding response matrices
        Args:
            batch (int): Batch size
        """
        
        min_ = 35
        max_ = 175
        trueData = pickle.load(open(os.path.join(self.X_path, 'true_%s.pkl'%clusterID), 'rb'))
        spectraData = pickle.load(open(os.path.join(self.X_path, 'spectra_%s.pkl'%clusterID), 'rb'))
        responseData = pickle.load(open(os.path.join(self.A_path, 'rmfs_%s.pkl'%clusterID), 'rb'))
        self.n = len(responseData)
        lowerBound, upperBound = self.__bounds__(len(trueData))
        specNums = np.random.randint(self.lowerBound, self.upperBound, self.batch_size)  # Obtain random indices matching batch size
        X = [trueData[specNum][1][min_:max_]/max_calc(trueData[specNum][1][min_:max_]) for specNum in specNums]  # Get true data
        Y = [spectraData[specNum][0][1][min_:max_]/max_calc(spectraData[specNum][0][1][min_:max_]) for specNum in specNums]  # Get observation data
        respNums = [spectraData[specNum][1] for specNum in specNums] # Corresponding response matrix IDs
        try:
            A = [responseData[respNum][min_:max_, min_:max_] for respNum in respNums]  # Get response data
        except KeyError:
            print(clusterID)
            print(len(responseData))
            print(respNums)
            print(self.dataType)
            print(self.lowerBound)
            print(self.upperBound)
        C = [spectraData[specNum][0][2][min_:max_]/max_calc(spectraData[specNum][0][2][min_:max_]) for specNum in specNums]  # Get noise data
        return X, Y, A, C
        
               
    def collectDataFiles(self):
        """
        Given path to data collect filenames
        """
        # Collect true spectra file names and collect observed spectra file names
        for clusterID in self.clusterIDs:
            self.Y_data.append(os.path.join(self.X_path, 'true_%s.pkl'%clusterID))
            self.X_data.append(os.path.join(self.X_path, 'spectra_%s.pkl'%clusterID))
            self.A_data.append(os.path.join(self.A_path, 'rmfs_%s.pkl'%clusterID))

    
    def __getitem__(self, idx):

        cluster = random.choice(self.clusterIDs)
        x, y, a, c = self.__get_data__(cluster)
        return np.array(x), np.array(y), np.array(a), np.array(c)
    
    
    def __bounds__(self, dataLength):
        """
        Determine the bounds for the indices based on which data type (i.e. training, validation, or test).
        The default is 70%, 20%, 10%, respectively.
        """
        if self.dataType == 'training' or self.dataType=='train':
            lowerBound = 0
            upperBound = int(0.7*dataLength)
        elif self.dataType == 'validation' or self.dataType=='valid':
            lowerBound = int(0.7*dataLength)
            upperBound = int(0.9*dataLength)
        elif self.dataType == 'test':
            lowerBound = int(0.9*dataLength)
            upperBound = dataLength
        else:
            print("Please set the dataType to either 'training', 'validation', or 'test'.")
            print("Exiting")
            exit()
        if lowerBound > 0:
            lowerBound -= 1
        upperBound -= 1
        self.lowerBound = lowerBound
        self.upperBound = upperBound
        return lowerBound, upperBound




######################################################
######################################################
######################################################

class CustomDataGen(tf.keras.utils.Sequence):
    """
    Custom data generator to read in data from pickle file 
    """
    
    def __init__(self, X_path, A_path, dataType, numData, dataName, outputName,
                 batch_size,
                ):
        
        self.X_path = X_path
        self.A_path = A_path
        self.batch_size = batch_size
        self.dataType = dataType.lower()  # Either training, validation, or test
        self.numData = numData  # The number of sampels in the generator
        self.dataName = dataName  # Name of data i.e. ('mini' if data is called 'true_mini.pkl')
        self.outputName = outputName
        self.X_data = [] 
        self.Y_data = []
        self.A_data = [] 
    
        
    def __len__(self):
        return int(self.numData) // self.batch_size

    
    
    def __get_data__(self):
        """_summary_
        Given batch of data we will randomly load in spectra and their corresponding response matrices
        Args:
        """
        
        min_ = 35
        max_ = 175
        trueData = pickle.load(open(os.path.join(self.X_path, 'true_%s.pkl'%self.dataName), 'rb'))
        spectraData = pickle.load(open(os.path.join(self.X_path, 'spectra_%s.pkl'%self.dataName), 'rb'))
        responseData = pickle.load(open(os.path.join(self.A_path, 'rmfs_%s.pkl'%self.dataName), 'rb'))
        self.n = len(responseData)
        #print(self.n)
        lowerBound, upperBound = self.__bounds__(self.n)
        specNums = np.random.randint(self.lowerBound, self.upperBound, self.batch_size)  # Obtain random indices matching batch size
        #print(self.dataType)
        #print(specNums)
        #X = [trueData[specNum][1][min_:max_]/max_calc(trueData[specNum][1][min_:max_]) for specNum in specNums]  # Get true data
        #Y = [spectraData[specNum][0][1][min_:max_]/max_calc(spectraData[specNum][0][1][min_:max_]) for specNum in specNums]  # Get observation data
        X = [trueData[specNum][1][min_:max_] for specNum in specNums]  # Get true data
        Y = [spectraData[specNum][0][1][min_:max_] for specNum in specNums]  # Get observation data
        respNums = [spectraData[specNum][1] for specNum in specNums] # Corresponding response matrix IDs
        A = [responseData[respNum][min_:max_, min_:max_] for respNum in respNums]  # Get response data
        #C = [spectraData[specNum][0][2][min_:max_]/max_calc(spectraData[specNum][0][2][min_:max_]) for specNum in specNums]  # Get noise data
        C = [spectraData[specNum][0][2][min_:max_] for specNum in specNums]  # Get noise data
        
        return X, Y, A, C
        
               
    def collectDataFiles(self):
        """
        Given path to data collect filenames
        """
        # Collect true spectra file names and collect observed spectra file names
        self.Y_data.append(os.path.join(self.X_path, 'true_%s.pkl'%self.dataName))
        self.X_data.append(os.path.join(self.X_path, 'spectra_%s.pkl'%self.dataName))
        self.A_data.append(os.path.join(self.A_path, 'rmfs_%s.pkl'%self.dataName))

    
    def __getitem__(self, idx):

        x, y, a, c = self.__get_data__()
        if self.dataType == 'training' or self.dataType=='train':
            return np.array(x), np.array(y), np.array(a), np.array(c)
        elif self.dataType == 'validation' or self.dataType=='valid':
            return np.array(x), np.array(y), np.array(a), np.array(c)
        elif self.dataType == 'test':
            return  np.array(y), np.array(a), np.array(c)
    
    
    def __bounds__(self, dataLength):
        """
        Determine the bounds for the indices based on which data type (i.e. training, validation, or test).
        The default is 70%, 20%, 10%, respectively.
        """
        if self.dataType == 'training' or self.dataType=='train':
            lowerBound = 0
            upperBound = int(0.7*dataLength)
        elif self.dataType == 'validation' or self.dataType=='valid':
            lowerBound = int(0.7*dataLength)
            upperBound = int(0.9*dataLength)
        elif self.dataType == 'test':
            lowerBound = int(0.9*dataLength)
            upperBound = dataLength
        else:
            print("Please set the dataType to either 'training', 'validation', or 'test'.")
            print("Exiting")
            exit()
        if lowerBound > 0:
            lowerBound -= 1
        upperBound -= 1
        self.lowerBound = lowerBound
        self.upperBound = upperBound
        return lowerBound, upperBound
    
    def plot_data(self):
        # Plot data for verification
        fig = plt.figure(figsize=(16,8))
        for i in range(10):
            test_spec = self.Y_data[i]
            plt.plot(np.linspace(0,len(test_spec), len(test_spec)), test_spec)
        plt.savefig('%s/test_convolved_%s.png'%(self.outputName, self.dataName))
        plt.clf()

        fig = plt.figure(figsize=(16,8))
        for i in range(10):
            test_spec = self.X_data[i]
            plt.plot(np.linspace(0,len(test_spec), len(test_spec)), test_spec)
        plt.savefig('%s/test_true_%s.png'%(self.outputName, self.dataName))
        plt.clf()
        return None
