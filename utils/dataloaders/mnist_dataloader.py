from pathlib import Path
import tensorflow.keras as keras
import tensorflow as tf

from numpy import load
import numpy as np

import sys
sys.path.append('../..')
import utils.dataloaders.data_augmentor_mnist as DataAugmentor

class MNISTDataloader(tf.keras.utils.Sequence):
    
    def __init__(self,  path_to_data: Path, batch_size, dataset_split="Train", data_augmentation = "False"):
        
        if dataset_split == "Train":
            self.x = load(f"{path_to_data}/train_samples.npy")
            self.y = load(f"{path_to_data}/train_labels.npy")
        elif dataset_split == "Validation":
            self.x = load(f"{path_to_data}/val_samples.npy")
            self.y = load(f"{path_to_data}/val_labels.npy")
        else:
            self.x = load(f"{path_to_data}/test_samples.npy")
            self.y = load(f"{path_to_data}/test_labels.npy")

        self.batch_size = batch_size
        
        self.data_augmentor = None
        if data_augmentation == "True":
            print("Using data augmentation")
            self.data_augmentor = DataAugmentor.DataAugmentorMNIST()
        
    def on_epoch_end(self):
        #TODO Shuffle maybe
        pass
    
    def count(self):
        return len(self.y)
    
    def __getitem__(self, index):
        batch_samples = self.x[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.y[index * self.batch_size:(index + 1) * self.batch_size]
        
        if self.data_augmentor is not None:
            batch_samples = self.data_augmentor.augment_data(batch_samples)
        
        batch_samples = np.expand_dims(batch_samples, axis=-1) # <--- add batch axis
        batch_samples = batch_samples.astype('float32') / 255
        
        return batch_samples, keras.utils.to_categorical(batch_labels,10)
    
    
    def __len__(self):
        return len(self.y) // self.batch_size    
