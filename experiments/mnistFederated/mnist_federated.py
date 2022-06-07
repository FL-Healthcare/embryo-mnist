import os
import time
from multiprocessing import Process
import sys


import logging
import argparse

sys.path.append('../..')

import client.mnist.mnist_client as Client
import server.mnist.mnist_server as Server


# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.basicConfig(filename='mnist_federated.log', level=logging.DEBUG)


def start_client(client_id, number_of_rounds, number_of_epochs, experiment_name, group_name, batch_size, data_augmentation):
    client = Client.MNISTClient(client_id, number_of_rounds, number_of_epochs, experiment_name, group_name, batch_size, data_augmentation)
    client.start()


def start_server(number_of_rounds, number_of_clients, fraction_of_client, experiment_name, group_name):
    server = Server.MNISTAggregationServer(number_of_rounds, number_of_clients, fraction_of_client, experiment_name, group_name)
    server.start()


def simulation(number_of_clients, fraction_of_client, number_of_rounds, number_of_epochs, experiment_name, group_name, batch_size, data_augmentation):
    """Start a FL simulation."""

    # This will hold all the processes which we are going to create
    processes = []

    # Start the server
    server_process = Process(
        target=start_server, args=(number_of_rounds, number_of_clients, fraction_of_client, experiment_name, group_name)
    )
    server_process.start()
    processes.append(server_process)

    # Optionally block the script here for a second or two so the server has time to start
    time.sleep(10)

    # Start all the clients
    for client_id in range(number_of_clients):
        client_process = Process(target=start_client, args=(client_id, number_of_rounds,number_of_epochs, experiment_name, group_name, batch_size, data_augmentation))
        client_process.start()
        processes.append(client_process)

    # Block until all processes are finished
    for p in processes:
        p.join()

if __name__ == "__main__":
    logging.debug("Starting")
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--ExperimentName", type=str,
                help="Number of rounds to run")
    parser.add_argument("-g", "--GroupName", type=str,
                help="Group name for wandb")
    parser.add_argument("-c", "--NumberOfClients", type=int,
                    help="Number of clients to simulate")
    parser.add_argument("-f", "--FractionOfClients", type=float,
                    help="Fraction of clients per round")
    parser.add_argument("-e", "--NumberOfEpochs", type=int,
                help="Number of epochs per round")
    parser.add_argument("-b", "--BatchSize", type=int,
                help="Size of each batch")
    parser.add_argument("-r", "--NumberOfRounds", type=int,
                help="Number of rounds to run")
    parser.add_argument("-d", "--DataAugmentation", type=str,
                help="True/False")
    parsed_args = parser.parse_args()
    

    print(f"-n {parsed_args.ExperimentName} -g {parsed_args.GroupName} -c {parsed_args.NumberOfClients} -f {parsed_args.FractionOfClients} -r {parsed_args.NumberOfRounds} -b {parsed_args.BatchSize} -e {parsed_args.NumberOfEpochs} -d {parsed_args.DataAugmentation}")
    simulation(parsed_args.NumberOfClients, parsed_args.FractionOfClients, parsed_args.NumberOfRounds, parsed_args.NumberOfEpochs, parsed_args.ExperimentName, parsed_args.GroupName, parsed_args.BatchSize, parsed_args.DataAugmentation)
