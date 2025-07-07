from array import array
import numpy as np
from matplotlib import pyplot

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
    

def get_testing_data(use_mnist: bool, data_dir: str='dataset') -> tuple:
    '''
    Gets the testing data from the specified directory.

    Args:
        use_mnist: True if the testing data is provided by the MNIST dataset.
        data_dir: The directory containing the testing data. Default: 'dataset'.

    Returns:
        The testing images and labels in the form of a tuple.
    '''
    if use_mnist:
        data: tuple = mnist_utils.get_testing_data(data_dir)
        out: tuple = (data[0], np.array(data[1], dtype=np.uint8)) 
        return out
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

def get_random(use_mnist: bool, data_dir: str='dataset', label=None) -> tuple:
    data: tuple = shuffle(get_testing_data(use_mnist, data_dir))
    return_index: int = 0
    if label != None:        
        for index in data[1]:
            if data[1][index] == label:
                return_index = index
                break
    return (data[0][return_index], data[1][return_index])

def plot_image(image: list) -> None:
    pyplot.subplot()
    pyplot.imshow(np.reshape(image, (28, 28)), cmap=pyplot.get_cmap('gray'))
    pyplot.show()

def plot_accuracy(label_success: dict) -> None:
    pyplot.bar(range(len(label_success)), list(label_success.values()), align='center')
    pyplot.xticks(range(len(label_success)), list(label_success.keys()))    
    pyplot.show()