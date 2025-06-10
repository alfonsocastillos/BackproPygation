from array import array
import numpy as np

from custom_modules import mnist_utils

def get_training_data(use_mnist: bool, data_dir: str='dataset') -> tuple:
    '''
    Gets the training data from the specified directory.

    Args:
        use_mnist: True if the training data is provided by the MNIST dataset.
        data_dir: The directory containing the training data. Default: 'dataset'.

    Returns:
        The training images and labels in the form of a tuple.
    '''
    if use_mnist:
        return mnist_utils.get_training_data(data_dir)
    else:
        return ()

def shuffle(data: tuple[list, array]) -> None | tuple:
    '''
    Shuffles the given lists, maintaining their original relation.

    Args:
        args: Any number of lists to shuffle.

    Returns:
        A tuple containing the shuffled lists. None if no input arguments are given.        
    '''
    size: int = len(data[0])
    permutation: np.ndarray[np.long] = np.random.permutation(size)
    shuffled: tuple = (
        [data[0][i] for i in permutation],
        np.array(data[1], dtype=np.uint8)[permutation]
    )    
    return shuffled