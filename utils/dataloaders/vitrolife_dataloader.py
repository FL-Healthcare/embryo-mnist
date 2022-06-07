from pathlib import Path 
import numpy as np
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import sys
from threading import Thread
from PIL import Image

sys.path.append('../..')

import utils.dataloaders.data_augmentor as DataAugmentor
import utils.dataloaders.dynamic_aggresive_augmentor as DataAugmentorDynamic

    
class VitroLifeDataloader(tf.keras.utils.Sequence):

    def __init__(self,  data_folder: Path, batch_size, dataset_split = "Train", data_augmentation = "False", balancing_strategy="None", clinic_ID = None, dynamic_augment=False):
        self.dataset_split = dataset_split
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.dynamic_augment = dynamic_augment

        metadata_file_path = os.path.join(data_folder,"metadata.csv")
        meta_data = pd.read_csv(metadata_file_path)
        
        # Filtre either train/validation/test
        if self.dataset_split == "Train":
            self.meta_data = meta_data.loc[meta_data['Testset'] == 0]
            meta_data_train, _ = train_test_split(self.meta_data, train_size=0.8279, random_state=42)
            self.meta_data = meta_data_train
        elif  self.dataset_split == "Validation":  
            self.meta_data = meta_data.loc[meta_data['Testset'] == 0]
            _, meta_data_validation = train_test_split(self.meta_data, train_size=0.8279, random_state=42)
            self.meta_data = meta_data_validation
        else:
            self.meta_data = meta_data.loc[meta_data['Testset'] == 1]
        
        # If clinic is provided, filter nonly clinic ID
        # Only for federated 
        if clinic_ID is not None:
            self.meta_data =self.meta_data.loc[self.meta_data['LabID'] == clinic_ID]
           
        self.sample_IDS = self.meta_data["SampleID"].tolist()
        self.sample_labels = self.meta_data["Label"].tolist()          
    
        
        # some clients do not hold an entire batch for testing
        if len(self.sample_IDS) < self.batch_size:
            self.batch_size = len(self.sample_IDS)
            
        # oversample if needed    
        if balancing_strategy == "OverSampling":
            print("Using oversampler strategy in dataloader")
            over_sampled_samples, over_sampled_labels = self.over_sample(self.meta_data)
            self.sample_IDS = over_sampled_samples
            self.sample_labels = over_sampled_labels
            
        
        self.data_augmentor = None
        if data_augmentation == "True":
            print("Using data augmentation")
            if self.dynamic_augment == False:
                self.data_augmentor = DataAugmentor.DataAugmentor()
            else:
                self.data_augmentor = DataAugmentorDynamic.DataAugmentor()
    
    
    def set_aggresive_augment_lvl(self, lvl):
        self.data_augmentor.set_augment_lvl(lvl)
    
    
    def over_sample(self, meta_data):
        meta_data_pos = meta_data.loc[meta_data['Label'] == 1] # Prenancy
        meta_data_neg = meta_data.loc[meta_data['Label'] == 0] # Non Prenancy
                
        pos_samples = meta_data_pos['SampleID'].tolist()
        neg_samples = meta_data_neg['SampleID'].tolist()
        
        pos_labels = meta_data_pos['Label'].tolist()
        neg_labels = meta_data_neg['Label'].tolist()
        
        # valid indexes of positive samples
        ids = np.arange(len(pos_samples))
        
        # select as many indexes in pos, as indexes in neg  # Oversample
        choices = np.random.choice(ids, len(neg_samples))
        
        over_sampled_pos = np.array(pos_samples)[choices]
        over_sampled_pos_label = np.array(pos_labels)[choices]
        
        # Combine oversampled pos + neg
        total_samples = np.concatenate([over_sampled_pos, np.array(neg_samples)], axis=0)
        total_labels = np.concatenate([over_sampled_pos_label, np.array(neg_labels)], axis=0)
        
        # shuffle
        order = np.arange(len(total_labels))
        np.random.shuffle(order)
        
        total_samples = total_samples[order]
        total_labels = total_labels[order]
        
        return total_samples, total_labels
            
        
    def get_ground_truths(self):
        return self.sample_labels[0:self.__len__()*self.batch_size]
    
    
    def count(self):
        return len(self.sample_labels)
    
    
    def on_epoch_end(self):
        temp = list(zip(self.sample_IDS, self.sample_labels))
        random.shuffle(temp)
        self.sample_IDS, self.sample_labels = zip(*temp)
    
    
    def __getitem__(self, index):
        batch_samples_names = self.sample_IDS[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.sample_labels[index * self.batch_size:(index + 1) * self.batch_size]
        
        batch_samples = [None]*self.batch_size
        workers = [None]*self.batch_size
        
        # start workers as the batch size
        for index, sample in enumerate(batch_samples_names):
            # prepare filename to load
            filename = os.path.join(self.data_folder,"{:05d}.npz".format(sample))
            
            workers[index] = Thread(target=self.async_image_worker, args=(filename, self.data_augmentor, batch_samples, index))
            workers[index].start()
            

        # await all workers to finish
        for worker in workers:
            worker.join()
        
        batch_samples = np.array(batch_samples)
        batch_samples = np.expand_dims(batch_samples, axis=-1) 
        
        batch_labels = np.array(batch_labels)
        
        return batch_samples, batch_labels
    
    
    def __len__(self):
        return len(self.sample_labels) // self.batch_size    

    
    def async_image_worker(self, filename, augmentor, batch_result, index):
        data = np.load(filename)
        
        raw_img = load_image(data, augmentor)  
        batch_result[index] = raw_img
    
    
def load_image(data, augmentor):
    images = data['images'] # dimensions: frame (time),height,width,channel (focal)
    
    if augmentor is not None:
        img_raw = augmentor.augment_data(data)
        img = Image.fromarray(img_raw)
        newsize = (250,250)
        img = img.resize(newsize)
        img_raw= np.asarray(img)
        img_raw = img_raw.astype('float32')/255
        return img_raw
    else:   
         # Select default (time, focal)
        focal = 1
        frame = 0

        img_raw = images[frame,:,:,focal] # Take center focal image from frameID 0 (use this for testing/evaluation)
        img = Image.fromarray(img_raw)
        newsize = (250,250)
        img = img.resize(newsize)
        img_raw= np.asarray(img)
        img_raw = img_raw.astype('float32')/255
        return img_raw
