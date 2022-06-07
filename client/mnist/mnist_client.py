import flwr as fl

import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
import utils.models.model_saver as ModelSaverCallback

import utils.dataloaders.mnist_dataloader as DataLoader
import utils.models.mnist_model as Model
import utils.dataloaders.auto_flush as AutoFlush

from datetime import date
import time
from numpy import save
import os
import shutil

import wandb
from wandb.keras import WandbCallback

class MNISTClient(fl.client.NumPyClient):
    def __init__(self, client_id, number_of_rounds, number_of_epochs, experiment_name, group_name, batch_size, data_augmentation):
        self.client_id = client_id
        self.number_of_rounds = number_of_rounds
        self.number_of_epochs = number_of_epochs
        self.batch_size = batch_size
        
        self.current_round = 0
        self.global_epoch = 0

        self.experiment_name = experiment_name
        self.group_name = group_name
        self.init_wanbd()
        
        self.init_model_folder()
        
        # Make some clients use CPU
        if self.client_id < 9:
            self.set_GPU_limits()
        else:    
            self.disable_gpu()

        model_gen = Model.MNISTModel()
        self.model = model_gen.get_mnist_model()
        self.train_loader, self.validation_loader, self.test_loader = self._get_generators(data_augmentation)
    
    def _get_generators(self, data_augmentation):
        base_full_path = f"/mnt/data/FederatedLearning/data/MNIST/partitions/Client-{self.client_id+1}"
        
        train_loader = DataLoader.MNISTDataloader(base_full_path, self.batch_size, dataset_split="Train", data_augmentation = data_augmentation)
        validation_loader = DataLoader.MNISTDataloader(base_full_path, self.batch_size, dataset_split="Validation")
        test_loader = DataLoader.MNISTDataloader(base_full_path, self.batch_size, dataset_split="Test")
        
        return (train_loader, validation_loader, test_loader)
    
    
    def init_wanbd(self):
        project_name = "MNIST"
        wandb.init(project=f"{project_name}", group=f"{self.group_name}", entity="anton-peter")
        wandb.run.name = f"{self.experiment_name}-Client-{self.client_id+1}"
        #time.sleep(4)
        
     
    def _is_last_round(self):
        if self.current_round == self.number_of_rounds:
            return True
        else:
            return False

        
    def set_GPU_limits(self):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True  # dynamically grow the memory 
        config.log_device_placement = True # to log device placement (on which device the operation ran)

        sess = tf.compat.v1.Session(config=config)
        set_session(sess)  # set this TensorFlow session as the default session for Keras


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

        
    def get_parameters(self):
        return self.model.get_weights()

    
    def fit(self, parameters, config):
        self.current_round += 1
        self.model.set_weights(parameters)
    
        history = self.model.fit(self.train_loader, validation_data=self.validation_loader, callbacks=[WandbCallback(), AutoFlush.FlushCallback()], epochs=self.number_of_epochs, verbose=0)

        #history = self.model.fit(self.train_loader, validation_data=self.validation_loader, callbacks=[AutoFlush.FlushCallback()], epochs=self.number_of_epochs, verbose=0)
        self.global_epoch += self.number_of_epochs
        
        
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
        }
        
        return self.model.get_weights(), self.train_loader.__len__()*self.batch_size, results

    
    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)

        # Log fresh aggregated weights from round
        result = self.model.evaluate(self.validation_loader, callbacks=[AutoFlush.FlushCallback()], verbose=0)
        # [loss, tp, fp, tn, fn, accuracy, presision, recall, auc, prc]
        selected_results = [result[0], result[5], result[6],result[7],result[8]]
        selecetd_metrics =  ['R_loss', 'R_accuracy', 'R_presision', 'R_recall', 'R_auc']

        # Log selected metrics
        for index, metric in enumerate(selecetd_metrics):
            wandb.log({f"{metric}": selected_results[index], 'Round' : self.current_round}, step=self.global_epoch)

        # Let the first client save the final model every round
        if self.client_id == 0:
            # save model
            self.model.save(f"./models/{self.experiment_name}/R-{self.current_round}")
        
        # Log ROC and confusion matrix stats
        if self._is_last_round():
            print("Last round client")
            
            predicted_probabilities = self.model.predict(self.validation_loader)
            fullpath = "/mnt/data/FederatedLearning/client/mnist/clientLogs"

            save(f"{fullpath}/predicted_{self.client_id}.npy",predicted_probabilities)
            save(f"{fullpath}/ground_{self.client_id}.npy", self.validation_loader.y.flatten()[0:self.validation_loader.__len__()*self.batch_size])
            wandb.finish()
        
        return (0.4, self.validation_loader.__len__()*self.batch_size, {"accuracy": 0.42})

    
    def init_model_folder(self):
        if self.client_id == 0:
             # Check whether the specified path exists or not
            model_paths = f"./models/{self.experiment_name }"
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
    
    
    def start(self):
        fl.client.start_numpy_client("localhost:8080", client=self)    

        