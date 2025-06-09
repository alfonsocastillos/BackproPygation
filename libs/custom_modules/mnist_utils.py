from mnist import MNIST

def get_training_data(data_dir: str) -> tuple:
    '''
    Gets the training data from the MNIST dataset directory.

    Args:
        data_dir: The path to the dataset directory.
    
    Returns:
        A tuple containing two lists: 
            1. A list of training images, each being a list of unsign bytes.
            2. An array of unsign integer labels for each training image.
    '''
    mdata = MNIST(data_dir)
    return mdata.load_training()

def get_test_data(data_dir: str) -> tuple:
    '''
    Gets the test data from the MNIST dataset directory.

    Args:
        data_dir: The path to the dataset directory.
    
    Returns:
        A tuple containing two lists: 
            1. A list of test images, each being a list of unsign bytes.
            2. An array of unsign integer labels for each test image.
    '''

    mdata = MNIST(data_dir)
    return mdata.load_testing()