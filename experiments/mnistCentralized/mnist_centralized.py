import os
import sys
import numpy as np
import logging
import argparse
from datetime import date
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
import random
from os import path
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from numpy import load

from tensorflow.compat.v1.keras.backend import set_session

import shutil
sys.path.append('../..')

import utils.models.mnist_model as Models
import utils.models.model_saver as ModelSaverCallback

import utils.dataloaders.mnist_dataloader as Dataloader

import wandb
from wandb.keras import WandbCallback

            

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.basicConfig(filename='mnist_centralized.log', level=logging.DEBUG)


def parse_args():
    logging.debug("Starting")
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--ExperimentName", type=str,
                help="Number of rounds to run")
    parser.add_argument("-d", "--DataAugmentation", type=str,
                help="True/False")
    parser.add_argument("-b", "--BatchSize", type=int,
                    help="Batch size")
    parser.add_argument("-e", "--NumberOfEpochs", type=int,
                help="Number of epochs")
    
    parsed_args = parser.parse_args()

    return parsed_args


def set_GPU_limits():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory 
    config.log_device_placement = True # to log device placement (on which device the operation ran)

    sess = tf.compat.v1.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

    
def init_wandb(experiment_name, username):
    #init wandb
    project_name = "MNIST"
    wandb.init(project=f"{project_name}", entity=username)
    wandb.run.name = experiment_name

    
def load_model():
    model_generator =  Models.MNISTModel()
    mnist_model = model_generator.get_mnist_model()
    return mnist_model


def load_data_from_disk():
    base_path = "/mnt/data/FederatedLearning/experiments/mnistCentralized"
    train_x = load(f"{base_path}/data/train_x.npy")
    train_y = load(f"{base_path}/data/train_y.npy")
    test_x = load(f"{base_path}/data/test_x.npy")
    test_y = load(f"{base_path}/data/test_y.npy")
    
    return (train_x, train_y), (test_x, test_y)

def get_data(batch_size, data_augmentation ="False"):
    (train_x, train_y), (test_x, test_y) = load_data_from_disk()
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.16666666666, random_state=42)
    
    train_gen = None
    if data_augmentation=="True":
        print("Using dataaugmentation")
        train_gen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range = 0.02, # Randomly zoom image 
            width_shift_range=0.05,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.05,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip image
    else:
        train_gen = ImageDataGenerator()
        
    val_gen = ImageDataGenerator()
    test_gen = ImageDataGenerator()
    
    # Add axis for greyscale
    train_x = np.expand_dims(train_x, axis=-1)
    val_x = np.expand_dims(val_x, axis=-1)
    test_x = np.expand_dims(test_x, axis=-1)
    
    #Norminalize to 0-1
    train_x = train_x.astype('float32') / 255
    val_x = val_x.astype('float32') / 255
    test_x = test_x.astype('float32') / 255
    
    # One hot encode
    train_y =  keras.utils.to_categorical(train_y,10)
    val_y =  keras.utils.to_categorical(val_y,10)
    test_y =  keras.utils.to_categorical(test_y,10)    
    
    # Create generators
    train_generator = train_gen.flow(train_x, train_y, batch_size=int(batch_size))
    valid_generator = val_gen.flow(val_x, val_y, batch_size=int(batch_size))
    test_generator = test_gen.flow(test_x, test_y, batch_size=int(batch_size))
    
    return train_generator, valid_generator, test_generator, (val_x, val_y), (test_x, test_y)


def train_model(model, experiment_name, train_generator, val_generator, number_of_epochs, data_augmentation="False"):
    model_save_callback = ModelSaverCallback.ModelSaver("./models", experiment_name)
    model.fit(train_generator, validation_data=val_generator, callbacks=[WandbCallback(), model_save_callback], epochs=number_of_epochs)   
            
        
        
def plot_confusion_and_roc(model, raw):
    classes = ['0','1','2','3','4','5','6','7','8','9']
    
    ground_truths =np.argmax(raw[1], axis=1)

    predicted_probabilities = model.predict(raw[0])
    predicted_arg_max = np.argmax(predicted_probabilities, axis=1)

    wandb.log({"roc" : wandb.plot.roc_curve(ground_truths, predicted_probabilities, labels=classes)})
    wandb.sklearn.plot_confusion_matrix(ground_truths, predicted_arg_max, labels=classes)
    
    
def eval_model(model, generator):
    # Evalute metrics
    result = model.evaluate(generator)
    # [loss, tp, fp, tn, fn, accuracy, presision, recall, auc, prc]
    selected_results = [result[5], result[6],result[7],result[8]]
    selecetd_metrics =  ['accuracy', 'presision', 'recall', 'auc']

    # Log selected metrics
    for index, metric in enumerate(selecetd_metrics):
          wandb.log({f"{metric}_test": selected_results[index]})
    
    
if __name__ == "__main__":
    parsed_args = parse_args()
    print(f"-n {parsed_args.ExperimentName} -b {parsed_args.BatchSize} -d {parsed_args.DataAugmentation} -e {parsed_args.NumberOfEpochs}")
    
    # Limit GPU
    set_GPU_limits()
    
    #init wandb
    init_wandb(parsed_args.ExperimentName, "anton-peter")

    # load model
    mnist_model = load_model()
    
    # load mnist data into mem
    train_generator, valid_generator, test_generator, val_raw, test_raw = get_data(parsed_args.BatchSize, parsed_args.DataAugmentation)

    # training
    train_model(mnist_model, parsed_args.ExperimentName, train_generator, valid_generator, parsed_args.NumberOfEpochs, parsed_args.DataAugmentation)    
    
    # Eval model on validation set
    #eval_model(mnist_model, valid_generator)
        
    # plots ROC & confusion matrix
    plot_confusion_and_roc(mnist_model,  val_raw)
    
    # gracefully finish logging wandb
    wandb.finish()
    
    
    
