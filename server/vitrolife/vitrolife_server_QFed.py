import flwr as fl
import tensorflow as tf

import time
from numpy import load

import wandb
from datetime import date

import sys

sys.path.append('../..')

import utils.dataloaders.vitrolife_dataloader as DataLoader
import utils.models.vitrolife_model as Model

class VitrolifeAggregationServer:
    def __init__(self, number_of_rounds, number_of_clients, fraction_of_clients, experiment_name):
        self.current_round = 0
        
        print("Init server")
        print(f"Number of rounds {number_of_rounds}, fraction fit/eval {fraction_of_clients}, experiment name {experiment_name}")
        self.number_of_rounds = number_of_rounds
        self.fraction_of_clients = fraction_of_clients
        
        self.experiment_name = experiment_name
        self.set_GPU_limits()
        #self.init_wanbd_server()
        
        self.number_of_clients = number_of_clients
        self.fraction_of_clients = fraction_of_clients
        
        self.validation_gen = DataLoader.VitroLifeDataloader("/mnt/data/vitroLifeDataset", 16 , dataset_split="Validation")
        
        self.model_gen = Model.VitroLifeModel()
        self.model = self.model_gen.get_low_GPU_mem_model_batchnorm_HIGH_LEARNING()
        
        
        self.start_time = time.time()
        
     
    def set_GPU_limits(self):
        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
          # Invalid device or cannot modify virtual devices once initialized.
          pass
        

    def init_wanbd_server(self):
        project_name = "Embryo-QFed"
        wandb.init(project=f"{project_name}", group=f"{self.experiment_name}", entity="anton-peter")
        wandb.run.name = "Aggregation-Server"
    
    def get_eval_fn(self):
        """Return an evaluation function for server-side evaluation."""

        # Load data and model here to avoid the overhead of doing it in `evaluate` itself

        # The `evaluate` function will be called after every round
        def evaluate(weights):
            self.model.set_weights(weights)  # Update model with the latest parameters
            results = self.model.evaluate(self.validation_gen)
            metrics = {"auc": results[8]}
            return results[0], metrics

        return evaluate


        
    def start(self):
        # Create strategy'
        
        strategy_QFed = fl.server.strategy.QFedAvg(
            q_param = 0.4,
            qffl_learning_rate= 0.2,
            fraction_fit= self.fraction_of_clients,
            fraction_eval = self.fraction_of_clients,
            min_fit_clients = int(self.fraction_of_clients * self.number_of_clients),
            min_eval_clients = int(self.fraction_of_clients * self.number_of_clients),
            min_available_clients=self.number_of_clients,
            eval_fn=self.get_eval_fn()
        )
        
        
        
        strategy_fedAVG = fl.server.strategy.FedAvg(
            fraction_fit= self.fraction_of_clients,
            fraction_eval = self.fraction_of_clients,
            min_fit_clients = int(self.fraction_of_clients * self.number_of_clients),
            min_eval_clients = int(self.fraction_of_clients * self.number_of_clients),
            min_available_clients=self.number_of_clients,
            eval_fn =  None,
            on_fit_config_fn = None,
            on_evaluate_config_fn = None,
            accept_failures = True,
            initial_parameters = None,
            evaluate_metrics_aggregation_fn = None,
        )
        
        fl.server.start_server(server_address="localhost:8080", config={"num_rounds": self.number_of_rounds}, strategy=strategy_QFed)