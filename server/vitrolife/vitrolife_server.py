import flwr as fl

import time
from numpy import load

import wandb
from datetime import date

class VitrolifeAggregationServer:
    def __init__(self, number_of_rounds, number_of_clients, fraction_of_clients, experiment_name):
        self.current_round = 0
        
        print("Init server")
        print(f"Number of rounds {number_of_rounds}, fraction fit/eval {fraction_of_clients}, experiment name {experiment_name}")
        self.number_of_rounds = number_of_rounds
        self.fraction_of_clients = fraction_of_clients
        
        self.experiment_name = experiment_name
        self.init_wanbd_server()
        
        self.number_of_clients = number_of_clients
        self.fraction_of_clients = fraction_of_clients
        
        self.start_time = time.time()
        

    def init_wanbd_server(self):
        project_name = "Embryo-QFed"
        wandb.init(project=f"{project_name}", group=f"{self.experiment_name}", entity="anton-peter")
        wandb.run.name = "Aggregation-Server"
    
    
    def get_eval_metrics_aggregation_fn(self):
         
        def eval_metrics_aggregate(result):
            self.current_round +=1
            
            current_time = time.time()
            print(f"Current time {current_time-self.start_time}")
            self.start_time = time.time()
            
            print(f"Server Metric aggregation {self.current_round }=={self.number_of_rounds}")
            if(self.current_round == self.number_of_rounds):
                print("Server Last round metrics aggregation")
                fullpath = "../../client/vitrolife/clientLogs"

                ground_truths = []
                predicted = []
                
                ground_truths_matrix = []
                predicted_matrix = []

                for client_id in range(self.number_of_clients):
                    current_predicted = load(f"{fullpath}/predicted_{client_id}.npy")
                    current_ground = load(f"{fullpath}/ground_{client_id}.npy")
                    print("Loading done")
                    
                    # Be aware of max sampling warning
                    current_ground = current_ground.tolist()
                    current_predicted = current_predicted.tolist()
                    
                    ground_truths_matrix.extend(current_ground)
                    predicted_matrix.extend(current_predicted)
                    
                    ground_truths.extend(current_ground)
                    predicted.extend(current_predicted)               
                
                predicted_arg_max = []
                predicted_probabilities_transformed = []
                
                # For binary classification
                for proba in predicted:
                    proba = proba[0]
                    predicted_probabilities_transformed.append([1-proba, proba])
                    if proba > 0.5:
                        predicted_arg_max.append(1)
                    else:
                        predicted_arg_max.append(0)

                predicted_arg_max = [proba.index(max(proba)) for proba in predicted_matrix]
                classes = ['No Pregnancy', 'Pregnancy']       
            
                wandb.log({"roc" : wandb.plot.roc_curve(ground_truths, predicted_probabilities_transformed, labels=classes)})
                wandb.sklearn.plot_confusion_matrix(ground_truths, predicted_arg_max, labels=classes)
                wandb.finish()               
            
            return {"test": 42}
        
        return eval_metrics_aggregate


        
    def start(self):
        # Create strategy
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
            evaluate_metrics_aggregation_fn = self.get_eval_metrics_aggregation_fn(),
        )
        
        fl.server.start_server(server_address="localhost:8080", config={"num_rounds": self.number_of_rounds}, strategy=strategy_fedAVG)