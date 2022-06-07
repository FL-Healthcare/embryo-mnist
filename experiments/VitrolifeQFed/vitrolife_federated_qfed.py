#!/usr/bin/env python3
import os
import time
from multiprocessing import Process
import sys

import logging
import argparse

sys.path.append('../..')

import client.vitrolife.vitrolife_client as Client
import server.vitrolife.vitrolife_server_QFed as Server

logging.basicConfig(filename='vitrolife_federated.log', level=logging.DEBUG)


def start_client(client_id, number_of_rounds, number_of_epochs, experiment_name, balancing_strategy, use_data_augmentor, batch_size):
    client = Client.VitrolifeClient(client_id, number_of_rounds, number_of_epochs, experiment_name, balancing_strategy, use_data_augmentor, batch_size)
    client.start()


def start_server(number_of_rounds, number_of_clients, fraction_of_client, experiment_name):
    server = Server.VitrolifeAggregationServer(number_of_rounds, number_of_clients, fraction_of_client, experiment_name)
    server.start()


def simulation(experiment_name, number_of_clients, fraction_of_client, number_of_rounds, balancing_strategy, use_data_augmentor, batch_size, number_of_epochs):
    """Start a FL simulation."""
    
    # Start all the clients
    #client_with_big_data = [7,1,17,5,8,18,4,12,2,3,20,14,6]
    
    # This will hold all the processes which we are going to create
    processes = []

    # Start the server
    server_process = Process(
        #target=start_server, args=(number_of_rounds, len(client_with_big_data), fraction_of_client, experiment_name)
        target=start_server, args=(number_of_rounds, number_of_clients, fraction_of_client, experiment_name)
    )
    server_process.start()
    processes.append(server_process)

    # Optionally block the script here for a second or two so the server has time to start
    time.sleep(10)

    #for client_id in client_with_big_data:
    for client_id in range(number_of_clients):
        client_process = Process(target=start_client, args=(client_id, number_of_rounds,number_of_epochs, experiment_name, balancing_strategy, use_data_augmentor, batch_size))
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
    parser.add_argument("-c", "--NumberOfClients", type=int,
                    help="Number of clients to simulate")
    parser.add_argument("-f", "--FractionOfClients", type=float,
                    help="Fraction of clients per round")
    parser.add_argument("-e", "--NumberOfEpochs", type=int,
                help="Number of epochs per round")
    parser.add_argument("-r", "--NumberOfRounds", type=int,
                help="Number of rounds to run")
    parser.add_argument("-b", "--BatchSize", type=int,
                    help="Batch size")
    parser.add_argument("-d", "--DataAugmentation", type=str,
                help="True/False")
    parser.add_argument("-bs", "--BalancingStrategy", type=str,
                help="ClassWeights/OverSampling/None")
    parsed_args = parser.parse_args()
    

    print(f"-n {parsed_args.ExperimentName} -c {parsed_args.NumberOfClients} -f {parsed_args.FractionOfClients} -r {parsed_args.NumberOfRounds} -bs {parsed_args.BalancingStrategy} -d {parsed_args.DataAugmentation} -b {parsed_args.BatchSize} -e {parsed_args.NumberOfEpochs}")

    simulation(parsed_args.ExperimentName, parsed_args.NumberOfClients, parsed_args.FractionOfClients, parsed_args.NumberOfRounds,parsed_args.BalancingStrategy, parsed_args.DataAugmentation, parsed_args.BatchSize, parsed_args.NumberOfEpochs)
