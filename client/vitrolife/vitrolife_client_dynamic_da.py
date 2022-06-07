import flwr as fl
from tensorflow.keras import backend as k
import gc
import collections
import math
import shutil
import os
from scipy.stats import linregress

from tensorflow import keras

from numba import cuda

import numpy as np
import tensorflow as tf

from datetime import date

from tensorflow.compat.v1.keras.backend import set_session
import utils.dataloaders.auto_flush as AutoFlush
import utils.models.memory_monitor as Memory

import sys
sys.path.append('../..')

import utils.dataloaders.vitrolife_dataloader as DataLoader
import utils.models.vitrolife_model as Model

from numpy import save

import wandb
from wandb.keras import WandbCallback

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

class VitrolifeClient(fl.client.NumPyClient):
    def __init__(self, client_id, number_of_rounds, number_of_epochs, experiment_name,  balancing_strategy, use_data_augmentor, batch_size):
        self.client_id = client_id
        self.number_of_rounds = number_of_rounds
        self.number_of_epochs = number_of_epochs
        self.balancing_strategy = balancing_strategy
        self.use_data_augmentor = use_data_augmentor
        self.batch_size = batch_size
        
        self.rolling_loss = collections.deque([], maxlen=10)
        
        self.current_round = 0
        self.global_epoch = 0
            
        self.experiment_name = experiment_name
        self.init_wanbd()
        self.init_model_folder()
      
        # Used during training and eval
        self.training_generator = DataLoader.VitroLifeDataloader("/mnt/data/vitroLifeDataset", self.batch_size , dataset_split="Train", data_augmentation = self.use_data_augmentor, balancing_strategy=self.balancing_strategy, clinic_ID = self.client_id,  dynamic_augment=True)
        self.validation_generator = DataLoader.VitroLifeDataloader("/mnt/data/vitroLifeDataset", self.batch_size , dataset_split="Validation", clinic_ID = self.client_id)
        
        self.gpu_selection_based_on_data_size()
        #self.set_GPU_limits()
        
        self.model_gen = Model.VitroLifeModel()
        self.model = self.model_gen.get_low_GPU_mem_model_batchnorm_HIGH_LEARNING()
        
      
    
    def gpu_selection_based_on_data_size(self):
        # Each model takes around 2.5Gb memory
        # Clients with low number of samples, should use CPU, instead of GPU
        
        minimum_sample_count_CPU_threshold = 0
        # DIRTY Fix to make maximum number of GPU clients spawn
        if self.balancing_strategy == "OverSampling":
            minimum_sample_count_CPU_threshold = 650
        else:
            #minimum_sample_count_CPU_threshold = 500
            # Batch 32
            #minimum_sample_count_CPU_threshold = 650
            minimum_sample_count_CPU_threshold = 500
        if self.training_generator.count()  >= minimum_sample_count_CPU_threshold:
            print(f"Client {self.client_id} is using GPU")
            self.is_GPU_client = True
            self.set_GPU_limits()
        else:
            print(f"Client {self.client_id} is using CPU")
            self.is_GPU_client = False
            self.disable_gpu()


    def init_wanbd(self):
        project_name = "Embryo-QFed"
        wandb.init(project=f"{project_name}", group=f"{self.experiment_name}", entity="anton-peter")
        wandb.run.name = f"Client-{self.client_id+1}"

        
    def _is_last_round(self):
        if self.current_round == self.number_of_rounds:
            return True
        else:
            return False
        
    
    def disable_gpu(self):
        try:
            # Disable all GPUS
            tf.config.set_visible_devices([], 'GPU')
            visible_devices = tf.config.get_visible_devices()
            for device in visible_devices:
                assert device.device_type != 'GPU'
        except:
            # Invalid device or cannot modify virtual devices once initialized.
            pass
    
    
    def set_GPU_limits(self):
        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
          # Invalid device or cannot modify virtual devices once initialized.
          pass
    
        
    def get_parameters(self):
        return self.model.get_weights()

    
    def fit(self, parameters, config):
        self.current_round += 1
        self.model.set_weights(parameters)
        
        history = None
        if self.balancing_strategy == "ClassWeights":
            print("Using Class-Weighted custom training")
            self.custom_train(self.model, self.number_of_epochs, self.training_generator,  self.validation_generator)
        else:   
            print("Starting custom training loop")
            #self.model.fit(self.training_generator, validation_data=self.validation_generator, callbacks=[WandbCallback(), AutoFlush.FlushCallback()], epochs=self.number_of_epochs) 
            self.custom_train(self.model, self.number_of_epochs, self.training_generator,  self.validation_generator)

        # not used, since we use wanbb
        results = {
            "loss":42,
            "accuracy": 42,
        }

        return self.model.get_weights(), self.training_generator.count(), results

    
    def update_data_augmentation(self, loss):
        self.rolling_loss.append(loss)
        
        if self.global_epoch >=20:
            x_axis = [i for i in range(len(self.rolling_loss))]
            y_loss = [0]*len(self.rolling_loss)
            for i, loss in enumerate(self.rolling_loss):
                y_loss[i] = loss

            line_regress_result = linregress(np.array(x_axis), np.array(y_loss))
            slope = line_regress_result.slope
            if slope > 0:
                print(f"cleint: {self.client_id} is overfitting - Apply Aggresive")
                self.training_generator.set_aggresive_augment_lvl("Aggresive")
            else:
                self.training_generator.set_aggresive_augment_lvl("Normal")
    
    
    def evaluate(self, parameters, config):      
        self.model.set_weights(parameters)
         
        # Log fresh aggregated weights from round
        # result = [loss, tp, fp, tn, fn, accuracy, presision, recall, auc, prc]
        result = self.custom_eval(self.model, self.validation_generator, None, wandb_log=False)
        
        self.update_data_augmentation(result[0])
        
        selected_results = [result[0], result[5], result[6],result[7],result[8]]
        selecetd_metrics =  ['R_loss', 'R_accuracy', 'R_presision', 'R_recall', 'R_auc']
        
        # Log selected metrics under axis Round
        for index, metric in enumerate(selecetd_metrics):
            wandb.log({f"{metric}": selected_results[index], 'Round' : self.current_round}, step=self.global_epoch)

        # Let the second client save fresh aggregated model every round
        if self.client_id == 1:
            # save model
            self.model.save(f"./models/{self.experiment_name}/R-{self.current_round}")
            
        # Generate prediction for confusion and ROC    
        if self._is_last_round():
            print(f"Eval final model - client {self.client_id}")
                              
            # Predict and save files for ROC and Confustion matrix
            predicted_probabilities = self.custom_predict(self.model, self.validation_generator)
            fullpath = "/mnt/data/FederatedLearning/client/vitrolife/clientLogs"
            save(f"{fullpath}/predicted_{self.client_id}.npy",predicted_probabilities)
            save(f"{fullpath}/ground_{self.client_id}.npy", self.validation_generator.get_ground_truths())
            wandb.finish()
        
        # Return args not used, since all happends on wandb
        return (0.4, self.validation_generator.count(), {"accuracy": 0.42})

    
    def start(self):
        fl.client.start_numpy_client("localhost:8080", client=self)        
        
        
    # Taking from https://stackoverflow.com/questions/65487717/how-to-make-use-of-class-weights-to-calculated-custom-loss-fuction-while-using-c        
    def weigted_loss(self, labels, logits, weights):
        loss_func = keras.losses.BinaryCrossentropy(from_logits=False)
        loss = loss_func(labels, logits)
        class_weights = tf.gather(weights, labels)
        return tf.reduce_mean(class_weights * loss)    
    
        
    def custom_train(self, model, epochs, train_gen, val_gen,):
        loss_fn = None
        class_wts = None
        if self.balancing_strategy == "ClassWeights":          
            count_pregnancy = len(self.training_generator.meta_data.loc[self.training_generator.meta_data['Label'] == 1]) # Prenancy
            count_no_pregnancy = len(self.training_generator.meta_data.loc[self.training_generator.meta_data['Label'] == 0]) # Non Prenancy

            neg =  count_no_pregnancy
            pos =  count_pregnancy
            total = neg + pos

            weight_for_0 = (1 / neg) * (total / 2.0)
            weight_for_1 = (1 / pos) * (total / 2.0)
            
            class_wts = tf.constant([weight_for_0, weight_for_1])
            loss_fn = self.weigted_loss
        else:   
            loss_fn = keras.losses.BinaryCrossentropy(from_logits=False)
      
        loss_metric = keras.metrics.Mean()
        METRICS = [
            keras.metrics.TruePositives(name='tp'),
            keras.metrics.FalsePositives(name='fp'),
            keras.metrics.TrueNegatives(name='tn'),
            keras.metrics.FalseNegatives(name='fn'), 
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
            keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
        ]
        
        for epoch in range(epochs):
            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(train_gen):
                # Open a GradientTape to record the operations run
                # during the forward pass, which enables auto-differentiation.
                with tf.GradientTape() as tape:

                    # Run the forward pass of the layer.
                    # The operations that the layer applies
                    # to its inputs are going to be recorded
                    # on the GradientTape.
                    logits = model(x_batch_train, training=True)  # Logits for this minibatch
                    y_batch_train = np.expand_dims(y_batch_train, axis=-1) 
                    # Compute the loss value for this minibatch.
                    loss_value = 0
                    if self.balancing_strategy == "ClassWeights":
                        loss_value = loss_fn(y_batch_train, logits, class_wts)
                    else:
                        loss_value = loss_fn(y_batch_train, logits)

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                grads = tape.gradient(loss_value, model.trainable_weights)

                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                self.model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
                
                loss_metric.update_state(loss_value)
                # update metrics 
                for metric in METRICS:
                    metric.update_state(y_batch_train, logits)
                 
                
            # epoch end - Log training metrics
            metrics_to_log = []   
            metrics_to_log.append(loss_metric.result())
            for metric in METRICS:
                metrics_to_log.append(metric.result())
            
            self.log_custom_metrics(metrics_to_log, "", self.global_epoch)
         
            # epoch end - Log Val metrics
            self.custom_eval(model, val_gen, self.global_epoch)
            
            self.global_epoch += 1
            
        
    def custom_eval(self, model, generator, epoch, wandb_log=True):    
        loss_fn = keras.losses.BinaryCrossentropy(from_logits=False)
        loss_metric = keras.metrics.Mean()
        
        METRICS = [
            keras.metrics.TruePositives(name='tp'),
            keras.metrics.FalsePositives(name='fp'),
            keras.metrics.TrueNegatives(name='tn'),
            keras.metrics.FalseNegatives(name='fn'), 
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
            keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
        ]

        for step, (x_batch_val, y_batch_val) in enumerate(generator):
            val_logits = model(x_batch_val, training=False)
            y_batch_val = np.expand_dims(y_batch_val, axis=-1) 
            # Update val metrics

            loss_value = loss_fn(y_batch_val, val_logits)

            loss_metric.update_state(loss_value)
            # update metrics 
            for metric in METRICS:
                metric.update_state(y_batch_val, val_logits)

        metrics_to_log = []   
        metrics_to_log.append(loss_metric.result())
        for metric in METRICS:
            metrics_to_log.append(metric.result())
           
        if wandb_log:
            self.log_custom_metrics(metrics_to_log, "val_", epoch)
            
        return metrics_to_log
        
        
        
    def custom_predict(self, model, generator):
        predictions = []
        for step, (predict_x, _ ) in enumerate(generator):
            predict_logits = model(predict_x, training=False)
            predictions.extend(predict_logits)
            
        return np.array(predictions)

    
    def init_model_folder(self):
        if self.client_id == 0:
             # Check whether the specified path exists or not
            model_paths = f"./models/{self.experiment_name}"
            isExist = os.path.exists(model_paths)

            #delete old experiment folder   
            if isExist:
                shutil.rmtree(model_paths)
                os.makedirs(model_paths)
                print("Cleaning old model folder")
                print("New model directory is created!")
            else:   
                # Create a new directory because it does not exist 
                os.makedirs(model_paths)
                print("New model directory is created")  
        
        
    def log_custom_metrics(self, metrics, dataset_split, epoch):
        selecetd_metrics = ['loss', 'tp', 'fp', 'tn', 'fn', 'accuracy', 'presision', 'recall', 'auc', 'prc']
        
        # Log selected metrics
        for index, metric in enumerate(selecetd_metrics):
              wandb.log({f"{dataset_split}{metric}": metrics[index], 'epoch': epoch}, step=self.global_epoch)