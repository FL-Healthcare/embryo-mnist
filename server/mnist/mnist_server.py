import flwr as fl

import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session

from numpy import load
import random
import wandb



class MNISTAggregationServer:
    def __init__(self, number_of_rounds, number_of_clients, fraction_of_clients, experiment_name, group_name):
        self.current_round = 0
        self.number_of_rounds = number_of_rounds
        
        self.experiment_name = experiment_name
        self.group_name = group_name
        self.init_wanbd_server()
        self.set_GPU_limits()

        self.number_of_clients = number_of_clients
        self.fraction_of_clients = fraction_of_clients
        

    def init_wanbd_server(self):
        project_name = "MNIST"
        wandb.init(project=f"{project_name}", group=f"{self.group_name}", entity="anton-peter")
        wandb.run.name = "Aggregation-Server"
        
    
    def get_eval_metrics_aggregation_fn(self):
         
        def eval_metrics_aggregate(result):
            self.current_round +=1
            print(f"Server Metric aggregation {self.current_round }=={self.number_of_rounds}")
            if(self.current_round == self.number_of_rounds):
                print("Server Last round metrics aggregation")
                fullpath = "../../client/mnist/clientLogs"

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
                    
                    # TODO Interval is hardcoded
                    ground_truths.extend(current_ground)
                    predicted.extend(current_predicted)               


                predicted_arg_max = [proba.index(max(proba)) for proba in predicted_matrix]
                classes = ['0','1','2','3','4','5','6','7','8','9']
                
                combined = list(zip(ground_truths, predicted))
                sample_size = 5000
                sampled_with_out_repition = random.sample(combined, sample_size)
                ground_truths, predicted = zip(*sampled_with_out_repition)
                
                wandb.log({"roc" : wandb.plot.roc_curve(ground_truths, predicted, labels=classes)})
                wandb.sklearn.plot_confusion_matrix(ground_truths_matrix, predicted_arg_max, labels=classes)
                wandb.finish()
            
            return {"test": 42}
        
        return eval_metrics_aggregate

   
    def set_GPU_limits(self):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True  # dynamically grow the memory 
        config.log_device_placement = True # to log device placement (on which device the operation ran)

        sess = tf.compat.v1.Session(config=config)
        set_session(sess)  # set this TensorFlow session as the default session for Keras

        
    def start(self):
        # Create strategy
        strategy_fedAVG = fl.server.strategy.FedAvg(
            fraction_fit=self.fraction_of_clients,
            fraction_eval=self.fraction_of_clients,
            min_fit_clients=int(self.fraction_of_clients*self.number_of_clients),
            min_eval_clients =int(self.fraction_of_clients*self.number_of_clients),
            min_available_clients=self.number_of_clients,
            eval_fn =  None,
            on_fit_config_fn = None,
            on_evaluate_config_fn = None,
            accept_failures = True,
            initial_parameters = None,
            evaluate_metrics_aggregation_fn = self.get_eval_metrics_aggregation_fn(),
        )
        
        fl.server.start_server(server_address="localhost:8080", config={"num_rounds": self.number_of_rounds}, strategy=strategy_fedAVG)
        