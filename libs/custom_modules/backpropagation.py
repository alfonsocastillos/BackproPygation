import numpy as np
import logging

from custom_modules import utils

class Backpropagation():
    __batch_size: int
    __input_layer: int
    __hidden_layers: int
    __output_layer: int
    __data_dir: str
    __images: list
    __labels: np.ndarray
    __neurons: list[np.ndarray]
    __weights: list[np.ndarray]
    __biases: list[np.ndarray]
    __logger: logging.Logger = logging.getLogger(__name__)

    def __init__(self, batch_size: int=10, hidden_layers: int=1, output_layer: int=10, data_dir: str='dataset') -> None:
        '''
        Constructor for the Backpropagation class.

        Args:
            batch_size (optional): The number of training examples used before modifying weights. Default: 10.
            hidden_layers (optional): The number of hidden layers inside the ANN. Default: 1.
            data_dir (optional): The path to the directory containing the training and testing data. Default='dataset'.        
        '''
        self.__batch_size = batch_size
        self.__hidden_layers = hidden_layers
        self.__output_layer = output_layer
        self.__data_dir = data_dir

    def train(self) -> None:
        '''
        Trains the ANN based on the provided parameters supplied on the constructor.
        '''
        self.__logger.info(f'Training began with batch size of {self.__batch_size}')

        # Get shuffled training data from data directory.
        training_data: tuple = utils.shuffle(utils.get_training_data(True, self.__data_dir))
        self.__images = training_data[0]
        self.__labels = training_data[1]

        # Check for discrepancies in sizes.
        if len(self.__images) != self.__labels.size or len(self.__images) < 1:
            self.__logger.error('Discrepancy with the dataset size.')
            self.__logger.error(f'Number of images {len(self.__images)}, Number of labels {self.__labels.size()}')
            raise Exception('Data discrepancy while training neural network.')
        self.__input_layer = len(self.__images[0])
        
        # Initialize neurons, weights and biases.
        self.__initialize_network()
               
    def test(self) -> None:
        pass

    def __forward(self, image: list, label: int) -> None:
        pass

    def __backward(self) -> None:
        pass

    def __rand_layer_size(self) -> int:
        '''
        Determines a random number of neurons that a hidden layer should have. This number is always between the number
        of neurons present in the input layer and the output layer and never more than twice the neurons in the input layer.

        Returns:
            An integer with the calculated number of neurons for the given hidden layer.
        '''
        lower_lim, upper_lim = self.__input_layer, self.__output_layer
        if upper_lim < lower_lim:
            lower_lim, upper_lim = upper_lim, lower_lim
        size: int = np.random.randint(lower_lim, upper_lim)
        size = min(size, self.__input_layer * 2)
        return size

    def __initialize_network(self) -> None:
        '''
        Initializes weights, biases and neurons for this neural network.
        '''
        self.__neurons = []
        self.__weights = []
        self.__biases = []

        # Establish input layer size.
        self.__neurons.append(np.empty(len(self.__images[0])))
    
        for _ in range(self.__hidden_layers):

            # Create next hidden layer size.
            self.__neurons.append(np.empty(self.__rand_layer_size()))
            this_layer_size: int = self.__neurons[-2].size
            next_layer_size: int = self.__neurons[-1].size

            # Create random set of weights and biases between this layer and the next one.
            self.__weights.append(
                np.random.rand( 
                    next_layer_size,
                    this_layer_size
                )
            )
            self.__biases.append(np.random.rand(this_layer_size))

        # Establish output layer size.
        self.__neurons.append(np.empty(self.__output_layer))

        # Create random set of weights and biases between this layer and the output.
        last_layer_size: int = self.__neurons[-2].size
        self.__weights.append(
            np.random.rand(
                self.__output_layer,
                last_layer_size
            )
        )
        self.__biases.append(np.random.rand(self.__output_layer))
