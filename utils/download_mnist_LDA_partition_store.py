from typing import Callable, Dict, List, Optional, Tuple
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split

from keras.datasets import mnist
from numpy import save
from numpy import asarray
from numpy import load

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from flwr.common.parameter import weights_to_parameters
from flwr.common.typing import Parameters, Scalar, Weights
from flwr.dataset.utils.common import (
    XY,
    create_lda_partitions,
    shuffle,
    sort_by_label,
    split_array_at_indices,
)


def gen_MNIST_partitions(
    path_original_dataset,
    dataset_name,
    num_total_clients,
    lda_concentration,
):
    """Defines root path for partitions and calls functions to create them."""
    
    fed_dir = f"{path_original_dataset}/{dataset_name}/partitions"
    print(fed_dir)
    
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    print(f"total : {len(train_y)+len(test_y)}")
    
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.16666666666, random_state=42)
    

    print(f"test : {len(test_y)}")
    print(f"val : {len(val_y)}")
    print(f"train : {len(train_y)}")
    
    # partion LDA train
    dist = partition_MNIST_and_save(
        dataset= (train_x, train_y),
        fed_dir=fed_dir,
        dirichlet_dist=None,
        num_partitions=num_total_clients,
        concentration=lda_concentration,
        partition_type = "train",
    )
    
    # Use dist distribution 'dist' from train generation
    # partion LDA val
    partition_MNIST_and_save(
        dataset=(val_x, val_y),
        fed_dir=fed_dir,
        dirichlet_dist=dist,
        num_partitions=num_total_clients,
        concentration=lda_concentration,
        partition_type = "val",
    )
    
    # test
    # partion LDA test
    partition_MNIST_and_save(
        dataset=(test_x, test_y),
        fed_dir=fed_dir,
        dirichlet_dist=dist,
        num_partitions=num_total_clients,
        concentration=lda_concentration,
        partition_type = "test",
    )
    
    return fed_dir


def partition_MNIST_and_save(
    dataset,
    fed_dir,
    dirichlet_dist = None,
    num_partitions = 500,
    concentration = 0.1,
    partition_type = "train",
):
    """Creates and saves partitions for CIFAR10.
    Args:
        dataset (XY): Original complete dataset.
        fed_dir (Path): Root directory where to save partitions.
        dirichlet_dist (Optional[npt.NDArray[np.float32]], optional):
            Pre-defined distributions to be used for sampling if exist. Defaults to None.
        num_partitions (int, optional): Number of partitions. Defaults to 500.
        concentration (float, optional): Alpha value for Dirichlet. Defaults to 0.1.
    Returns:
        np.ndarray: Generated dirichlet distributions.
    """
    # Create partitions
    clients_partitions, dist = create_lda_partitions(
        dataset=dataset,
        dirichlet_dist=dirichlet_dist,
        num_partitions=num_partitions,
        concentration=concentration,
        accept_imbalanced=True
    )
    print(clients_partitions[0][0].shape)
    
    # Save partions
    save_partitions(list_partitions=clients_partitions, fed_dir=fed_dir, partition_type=partition_type)

    return dist



def save_partitions(
    list_partitions, fed_dir, partition_type = "train"):
    """Saves partitions to individual files.
    Args:
        list_partitions (List[XY]): List of partitions to be saves
        fed_dir (Path): Root directory where to save partitions.
        partition_type (str, optional): Partition type ("train" or "test"). Defaults to "train".
    """
    for idx, partition in enumerate(list_partitions):
        path_dir = f"{fed_dir}/Client-{idx+1}"
        os.makedirs(path_dir, exist_ok=True)
        save(f"{path_dir}/{partition_type}_samples.npy", partition[0])
        save(f"{path_dir}/{partition_type}_labels.npy", partition[1])
       
        

        

if __name__ == "__main__":
    # Load a partitioned dataset and show distribution of examples
    gen_MNIST_partitions("../data", "MNIST", 23, 1000)