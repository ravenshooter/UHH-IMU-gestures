import numpy as np
import pandas as pd


class DataSet(object):
    """
    Helper class to wrap around on sequence of measurements. 
    One dataset contains all recordings of one gesture performed by person.
    """
        
    
    def __init__(self, fused, gyro, acc, targets,means, stds, gestures):
        self.fused = fused
        self.gyro = gyro
        self.acc = acc 
        self.targets = targets
        self.means = means
        self.stds = stds
        self.gestures = gestures
        



       
    def plot(self, targetNr=2,normalized = -1):
        """
        Conveniece method to display a dataset. Shows fused signal, gyro and accelerometer. 
        Displays manual marker and engery based target.

        Requieres matplotlib
        """

        import matplotlib.pyplot as plt


        fig = plt.figure(figsize=(10,10))
        plt.clf()
        plt.subplot(411)
        plt.title('Fused')
        labels = ['X', 'Y', 'Z']
        for i in range(3):
            plt.plot(self.fused[:,i],label=labels[i])
        #plt.plot(self.targets[:,targetNr],label='Target')
        plt.ylim(-1.5,1.5)
        plt.legend()
        
        plt.subplot(412)
        plt.title('Gyro')
        labels = ['X', 'Y', 'Z']
        for i in range(3):
            plt.plot(self.gyro[:,i],label=labels[i])
        #plt.plot(self.targets[:,targetNr],label='Target')
        plt.legend()
        
        plt.subplot(413)
        plt.title('Acc')
        labels = ['X', 'Y', 'Z']
        for i in range(3):
            plt.plot(self.acc[:,i],label=labels[i])
        #plt.plot(self.targets[:,targetNr],label='Target')
        plt.legend()
        
        plt.subplot(414)
        plt.title('Marker and Target')
        labels = ['Marker', 'Target']
        plt.plot(self.targets[:,0], label=labels[0])
        plt.plot(self.targets[:,2], label=labels[1])
        plt.ylim(-0.5,1.5)
        plt.legend()
        plt.tight_layout()
        plt.show()
        return fig
        
    def getData(self):
        return  np.concatenate((self.fused,self.gyro,self.acc,self.targets),1)
    
    def getFused(self):
        return self.fused
    
    def getAcc(self):
        return self.acc
    
    def getGyro(self):
        return self.gyro 
    
    def getDataForTraining(self, classNrs, targetNr=2):
        """
        Method to return data in a format ready to be used 
        """

        sensorData = np.concatenate([self.fused, self.gyro, self.acc], axis=1)


        # Drop the first 1/4 timesteps of each target signal. 
        i = 0
        target = np.copy(self.targets)
        while i < len(self.targets):
            tLen = 0
            while i < len(self.targets) and self.targets[i,2] == 1: #energy based target found
                tLen = tLen + 1 #store length of target
                i = i+1
            if tLen != 0:
                dropArea = np.min([tLen-5,tLen*(1/4)]) #calculate area to drop
                target[i-tLen:i-tLen+dropArea,2]=0 #drop calculated area at beginning of signal
            i = i+1   
        
        # Create one hot encoede table
        labels = np.zeros((len(sensorData),len(classNrs)))
        i = 0
        for classNr in classNrs:
            labels[:,i] = target[:,targetNr].T * self.gestures[classNr]
            i = i+1
            
        
        return (sensorData, labels)
    
        
        
  
def createDataSetFromFile(fileName):
    """
    Reads a npz file and converts it into an instance of the DataSet class

    :param fileName: The actual name of a file in the dataSets folder.
    """
    data = np.load('dataSets/'+fileName)
    fused = data['fused']
    gyro = data['gyro']
    acc = data['acc']
    targets = data['targets']
    means = data['means']
    stds = data['stds']
    gestures = data['gestures']
    return DataSet(fused, gyro, acc, targets, means, stds, gestures)


def concatDS(dataSets, usedGestures):
    """
    Concatenates sequences from all different gestures. 

    Returns an array containing sensor measurements and an array containing one hot encoded targets 
    """
    inputs = np.concatenate([dataSet.getDataForTraining(usedGestures,2)[0] for dataSet in dataSets], axis=0)
    labels = np.concatenate([dataSet.getDataForTraining(usedGestures,2)[1] for dataSet in dataSets], axis=0)
    return (inputs, labels)


def createData(dataSetName, inputGestures, usedGestures=None):
    """
    Returns a tuple containing a 2d array of sensor values (timesteps x channels) and a 2d array containing targets (timesteps x class)

    :param dataSetName: Name of the dataset to be loaded.
    :param inputGestures: List of ids of gestures to be loaded.
    :param usedGestures: List of ids of gestures that should be present as targets. If a gesture is in inputGestures but not in usedGestures it's sensor values will be loaded but no target value will be set.
     
    """

    if not usedGestures:
        usedGestures = inputGestures

    dataSets= []
    for gesture in inputGestures:
        fullName = dataSetName + '_' +str(gesture) + '_' + 'fullSet.npz'
        dataSets.append(createDataSetFromFile(fullName))
    resultInputs,resultTargets = concatDS(dataSets, inputGestures)

    # select all gestures that are being used for this run
    inds = np.where(np.in1d(inputGestures, usedGestures))[0]
    
    return (resultInputs,resultTargets[:,inds])

    



def loadAllData(inputGestures, inputFiles, gestureNames):
    """
    Convenience method to load all data.
    Data is returned as a list of tuples containing data and labels
    """
     
    #read datasets and add them to dataStep
    dataStep = []

    for fileName in inputFiles:
        ind, t  = createData(fileName, inputGestures)
        dataStep.append((ind,t))

    return dataStep


def loadAllDataPD(inputGestures, inputFiles, gestureNames):
    """
    Loads all data and returns a big multi level pandas dataframe.
    Frame has inputs and targets as main columns, rows are dataset name and timestep
    """

    data = loadAllData(inputGestures, inputFiles, gestureNames)

    columns = ['person','timestep','fused_X', 'fused_Y', 'fused_Z', 'gyro_X', 'gyro_Y', 'gyro_Z','acc_X', 'acc_Y', 'acc_Z',] + gestureNames
    pd_data = pd.DataFrame(columns=columns)
    for filename, (inputs, targets) in zip(inputFiles, data):
        data_targets = pd.DataFrame(np.append(inputs, targets, axis=1), columns=columns[2:])
        data_targets['timestep'] = data_targets.index.values
        data_targets['person'] = filename
        pd_data = pd_data.append(data_targets)
    pd_data = pd_data.set_index(['person','timestep'])

    pd_data.columns = pd.MultiIndex.from_tuples([('targets' if col in gestureNames else 'inputs', col) for col in pd_data.columns])

    return pd_data




if __name__ == "__main__":

    #===========================================================================
    # Decide which gesture data shall be used for training
    #===========================================================================
    inputGestures = [0,1,2,3,4,5,6,7,8,9]


    #===========================================================================
    # Pick datasets to load
    #===========================================================================
    inputFiles = ['ni','j','na','l', 's']


    #===========================================================================
    # Add labels for gestures
    #===========================================================================
    gestureNames = ['left','right','forward','backward','bounce up','bounce down','turn left','turn right','shake lr','shake ud']
        

    # load all data as list of numpy arrays
    data = loadAllData(inputGestures, inputFiles, gestureNames)

    # load all data as pd dataframe
    pd_data = loadAllDataPD(inputGestures, inputFiles, gestureNames)
