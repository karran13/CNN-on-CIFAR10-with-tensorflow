'''
Created on Dec 30, 2017

@author: Binki
'''
import numpy as np

class dataHelper(object):
    '''
    classdocs
    '''


    def __init__(self, path):
        '''
        Constructor
        '''
        self.path=path
        
        
    def unpickle(self,file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict


    def initialize_main_dict(self):
        self.data_dict = self.unpickle(self.path)

# for key,val in data_dict.items():
#     print("{} = {}".format(key, val))

    


    def returnDataset(self):
        data = 'data'
        data_key = data.encode(encoding='utf_8', errors='strict')
        self.data_set_train = self.data_dict[data_key]
        
        #reshape data to required dimensions
        
        self.data_set_train = np.array(self.data_set_train)
        self.data_set_train = np.reshape(self.data_set_train, [-1, 3, 32, 32])
        self.data_set_train = self.data_set_train.transpose([0, 2, 3, 1])

        return self.data_set_train


    def returnLabelsNumClasses(self):
        labels = 'labels'
        label_key = labels.encode(encoding='utf_8', errors='strict')
        self.data_set_labels = np.array(self.data_dict[label_key])
        self.num_classes = np.max(self.data_set_labels) + 1
        
        #reshape to one hot form
        
        self.data_set_labels = np.eye(self.num_classes, dtype=float)[self.data_set_labels]
        return self.data_set_labels,self.num_classes
    
    