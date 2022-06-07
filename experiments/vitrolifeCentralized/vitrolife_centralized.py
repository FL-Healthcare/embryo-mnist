import os
import shutil
import sys
import numpy as np
import logging
import argparse
from datetime import date

import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session

sys.path.append('../..')

import utils.models.model_saver as ModelSaverCallback
import utils.models.vitrolife_model as Models
import utils.dataloaders.vitrolife_dataloader as Dataloader

import wandb
from wandb.keras import WandbCallback

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.basicConfig(filename='vitrolife.log', level=logging.DEBUG)


def parse_args():
    logging.debug("Starting")
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--ExperimentName", type=str,
                help="Number of rounds to run")
    parser.add_argument("-b", "--BatchSize", type=int,
                    help="Batch size")
    parser.add_argument("-bs", "--BalancingStrategy", type=str,
                help="ClassWeights/OverSampling/None")
    parser.add_argument("-d", "--DataAugmentation", type=str,
                help="True/False")
    parser.add_argument("-e", "--NumberOfEpochs", type=int,
                help="Number of epochs")
    parser.add_argument("-l", "--LogsPerEpoch", type=int,
                help="Number of logs per epoch")
    
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
    project_name = "Embryo-Final"
    wandb.init(project=f"{project_name}", entity=username)
    wandb.run.name = experiment_name

    
def load_model():
    model_generator =  Models.VitroLifeModel()
    vitrolife_model = model_generator.get_low_GPU_mem_model_batchnorm()
    return vitrolife_model


def get_dataloaders(batch_size, balancing_strategy, data_augmentation, path_to_data = "/mnt/data/vitroLifeDataset"):
    
    training_vitrolife_dataloader = Dataloader.VitroLifeDataloader(path_to_data, batch_size, dataset_split="Train", data_augmentation=data_augmentation, balancing_strategy = balancing_strategy)
    validation_vitrolife_dataloader = Dataloader.VitroLifeDataloader(path_to_data, batch_size, dataset_split="Validation")
    testing_vitrolife_dataloader = Dataloader.VitroLifeDataloader(path_to_data, batch_size, dataset_split="Test")
    
    return (training_vitrolife_dataloader, validation_vitrolife_dataloader, testing_vitrolife_dataloader)


def train_model(model, training_loader, validation_loader, batch_size, number_of_epochs, logs_per_epoch, balancing_strategy, experiment_name):
    model_save_callback = ModelSaverCallback.ModelSaver("./models", experiment_name)
    steps_per_epoch = training_loader.__len__()/logs_per_epoch
    if balancing_strategy == "ClassWeights":
        print("Using Class-Weighted training")
        count_pregnancy = len(training_loader.meta_data.loc[training_loader.meta_data['Label'] == 1]) # Prenancy
        count_no_pregnancy = len(training_loader.meta_data.loc[training_loader.meta_data['Label'] == 0]) # Non Prenancy
        
        neg =  count_no_pregnancy
        pos =  count_pregnancy
        total = neg + pos
        
        weight_for_0 = (1 / neg) * (total / 2.0)
        weight_for_1 = (1 / pos) * (total / 2.0)
        
        class_weight_addon = {0: weight_for_0, 1: weight_for_1}
        model.fit(training_loader, validation_data=validation_loader, callbacks=[WandbCallback(), model_save_callback], epochs=number_of_epochs*logs_per_epoch, steps_per_epoch=steps_per_epoch, class_weight=class_weight_addon)
    else:   
        model.fit(training_loader, validation_data=validation_loader, callbacks=[WandbCallback(), model_save_callback], epochs=number_of_epochs*logs_per_epoch, steps_per_epoch=steps_per_epoch)
        
        
def plot_confusion_and_roc_binary(model, batchsize):
    classes = ['No Pregnancy', 'Pregnancy']
    
    _, validation_vitrolife_dataloader,  _ = get_dataloaders(batchsize, balancing_strategy="None", data_augmentation="False")
    
    ground_truths = validation_vitrolife_dataloader.get_ground_truths()
    predicted_probabilities = model.predict(validation_vitrolife_dataloader)

    predicted_arg_max = []
    predicted_probabilities_transformed = []

    for proba in predicted_probabilities:
        predicted_probabilities_transformed.append([1-proba, proba])
        if proba > 0.5:
            predicted_arg_max.append(1)
        else:
            predicted_arg_max.append(0)


    wandb.log({"roc" : wandb.plot.roc_curve(ground_truths, predicted_probabilities_transformed, labels=classes)})
    wandb.sklearn.plot_confusion_matrix(ground_truths, predicted_arg_max, labels=classes)
    
    
def eval_model(model, batch_size):
    _, validation_vitrolife_dataloader, _ = get_dataloaders(batch_size, balancing_strategy="None", data_augmentation="False")
   
    # Evalute metrics
    result = model.evaluate(validation_vitrolife_dataloader)
    # [loss, tp, fp, tn, fn, accuracy, presision, recall, auc, prc]
    selected_results = [result[5], result[6],result[7],result[8]]
    selecetd_metrics =  ['accuracy', 'presision', 'recall', 'auc']

    # Log selected metrics
    for index, metric in enumerate(selecetd_metrics):
          wandb.log({f"{metric}_test": selected_results[index]})
    
    
if __name__ == "__main__":
    parsed_args = parse_args()
    print(f"-n {parsed_args.ExperimentName} -b {parsed_args.BatchSize} -bs {parsed_args.BalancingStrategy} -d {parsed_args.DataAugmentation} -e {parsed_args.NumberOfEpochs} -l {parsed_args.LogsPerEpoch}")
    
    # Limit GPU
    set_GPU_limits()
    
    #init wandb
    init_wandb(parsed_args.ExperimentName, "anton-peter")
    
    # load model
    vitrolife_model = load_model()
    
    # Data loaders
    training_vitrolife_dataloader, validation_vitrolife_dataloader, testing_vitrolife_dataloader = get_dataloaders(parsed_args.BatchSize, parsed_args.BalancingStrategy, parsed_args.DataAugmentation)
    
    # training
    train_model(vitrolife_model, training_vitrolife_dataloader,validation_vitrolife_dataloader, parsed_args.BatchSize, parsed_args.NumberOfEpochs, parsed_args.LogsPerEpoch, parsed_args.BalancingStrategy, parsed_args.ExperimentName)          
              
    # Eval model on val set
    # eval_model(vitrolife_model, parsed_args.BatchSize)
        
    # plots ROC & confusion matrix on val set
    plot_confusion_and_roc_binary(vitrolife_model, parsed_args.BatchSize)
    
    # gracefully finish logging wandb
    wandb.finish()