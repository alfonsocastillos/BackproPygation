import numpy as np
import logging
from typing import Callable

from custom_modules import utils
from custom_modules import math_utils

logging.basicConfig()

class Backpropagation():
    __input_layer: int
    __hidden_layers: int
    __output_layer: int
    __out_pattern: dict
    __images: list
    __labels: np.ndarray
    __neurons: list[np.ndarray]
    __weights: list[np.ndarray]
    __biases: list[np.ndarray]
    __cost_gradient: list[dict[str, np.ndarray]]
    __act_function: Callable[[int | float | np.ndarray], float | np.ndarray]
    __act_deriv: Callable[[int | float | np.ndarray], float | np.ndarray]
    __logger: logging.Logger = logging.getLogger(__name__)
    __logger.setLevel(logging.INFO)    
    
    def __init__(self, out_pattern: dict, hidden_layers: int=1, output_layer: int=10, act_function: str="sigmoid",) -> None:
        """
        Constructor for the Backpropagation class.

        Args:            
            out_pattern: The output pattern associated with each label.
            hidden_layers (optional): The number of hidden layers inside the ANN. Default: 1.
            output_layer (optional): The number of neurons that form the output layer. Default: 10.                        
            act_function (optional): The activation function for the neurons. Allowed values: sigmoid, tanh, ReLU. Default: sigmoid.
        """
        self.__hidden_layers = hidden_layers
        self.__output_layer = output_layer
        self.__out_pattern = out_pattern
        match act_function.lower():
            case "sigmoid":
                self.__act_function = math_utils.sigmoid
                self.__act_deriv = math_utils.d_sigmoid
            case "tanh":
                self.__act_function = math_utils.tanh
                self.__act_deriv = math_utils.d_tanh
            case "relu":
                self.__act_function = math_utils.relu
                self.__act_deriv = math_utils.d_relu
            case _:
                self.__logger.error("Unknown activation function.")
                self.__logger.error("Activation must be one of: sigmoid, tanh or ReLU.")
                raise Exception("Unknown activation function.")

    def train(self, data_dir: str="dataset", batch_size: int=10, learn_rate: float=0.1) -> None:
        """
        Trains the ANN based on the provided parameters supplied on the constructor.

        Args:
            batch_size (optional): The number of training examples used before modifying weights. Default: 10. For a stocastic learning, specify a batch size of 1.
            data_dir (optional): The path to the directory containing the training and testing data. Default: "dataset".
            learn_rate (optional): The learning rate, i.e. how much weights and biases are modified each batch. Must be between 0 and 1.
        """
        if batch_size < 1 or type(batch_size) != int:
            self.__logger.error("Incorrect batch size provided.")
            raise Exception("Incorrect batch size for training. Must be an integer greater than 0.")
        if not 0 < learn_rate <= 1:
            self.__logger.error("Learning rate not within the valid range.")
            raise Exception("Learning rate not within valid range. Must be within 0 and 1.")

        self.__logger.info(f"Training began with batch size of {batch_size}")

        # Get shuffled training data from data directory.
        training_data: tuple = utils.shuffle(utils.get_training_data(True, data_dir))
        self.__images = training_data[0]
        self.__labels = training_data[1]

        # Check for discrepancies in sizes.
        if len(self.__images) != self.__labels.size or len(self.__images) < 1:
            self.__logger.error("Discrepancy with the dataset size.")
            self.__logger.error(f"Number of images {len(self.__images)}, Number of labels {self.__labels.size()}")
            raise Exception("Data discrepancy while training neural network.")
        self.__input_layer = len(self.__images[0])
        
        # Initialize neurons, weights and biases.
        self.__initialize_network()

        # Train neural network.
        while len(self.__images) > 0:            

            # Reset Cost gradient with each batch.
            for layer in range(len(self.__weights)):
                self.__cost_gradient[layer] = {
                    "dw": np.zeros(self.__weights[layer].shape),
                    "db": np.zeros(self.__biases[layer].shape)
                }

            # Compute forward and back propagation for each training example in the batch.
            for _ in range(batch_size):
                label: int = int(self.__labels[-1])
                self.__labels = self.__labels[:-1]
                self.__forward(self.__images.pop())
                self.__back(label)
            
            # At the end of the batch, update weights and biases.
            for layer in range(len(self.__cost_gradient)):
                self.__weights[layer] = self.__weights[layer] - learn_rate * self.__cost_gradient[layer]["dw"] / batch_size
                self.__biases[layer] = self.__biases[layer] - learn_rate * self.__cost_gradient[layer]["db"] / batch_size
                                       
    def test(self, data_dir: str="dataset") -> float:
        """
        Test the accuracy of the Neural Network with a set of testing data.

        Args:
            data_dir: the directory where the testing data is contained. 

        Return: 
            Success rate for this ANN.
        """

        # No need to shuffle testing data.
        test_data: tuple = utils.get_testing_data(True, data_dir)
        self.__images = test_data[0]
        self.__labels = test_data[1]
        total_examples: int = len(self.__images)
        overall_success: int = 0
        label_success: dict = {success: [0, 0] for success in self.__out_pattern}

        # Check for discrepancies in sizes.
        if len(self.__images) != self.__labels.size or len(self.__images) < 1:
            self.__logger.error("Discrepancy with the dataset size.")
            self.__logger.error(f"Number of images {len(self.__images)}, Number of labels {self.__labels.size()}")
            raise Exception("Data discrepancy while testing neural network.")
        
        while len(self.__images) > 0:
            label: int = int(self.__labels[-1])
            label_success[label][1] += 1
            self.__labels = self.__labels[:-1]
            self.__forward(self.__images.pop())
            prediction: int = np.argmax(self.__neurons[-1])
            if prediction == label:
                overall_success += 1
                label_success[label][0] += 1
        rate = overall_success / total_examples
        self.__logger.info(f"Success rate: {rate * 100}%")
        self.__logger.info(f"Per label success: {label_success}")
        for label, metrics in label_success.items():
            label_success[label] = metrics[0] / metrics[1]
        label_success["Overall"] = rate
        utils.plot_accuracy(label_success)
        return rate

    def test_single(self, label=None) -> None:
        """
        Test a single random example.

        Args:
            label: (optional) The desired element type to test.
        """
        image, label = utils.get_random(True, "dataset", label)
        self.__logger.info(f"Testing label: {label}")    
        self.__forward(image)
        prediction: int = np.argmax(self.__neurons[-1])
        self.__logger.info(f"ANN guess: {prediction}")
        utils.plot_image(image)

    def __forward(self, image: list) -> None:
        """
        Compute the output for the ANN for a given training example.

        Args:
            image: the training example given as a list of values.
        """
        # Normalize values between 0 and 1
        self.__neurons[0] = np.array(image) / 255

        # Compute neurons value for every layer (including output).
        for layer in range(1, self.__hidden_layers + 2):
            self.__neurons[layer] = self.__act_function(
                np.add(
                    np.matmul(self.__weights[layer - 1], self.__neurons[layer - 1]),
                    self.__biases[layer - 1]
                )
            )

    def __back(self, label: int) -> None:
        """
        Compute the back propagation for the given training example.

        Args:
            label: the expected output for the training example.
        """
        dC_da: np.ndarray = 2 * self.__neurons[-1] - self.__out_pattern[label]       
        for layer in range(len(self.__weights), 0, -1):
            dC_da = self.__calc_layer_cost(layer, dC_da)
            

    def __calc_layer_cost(self, layer: int, dC_da: np.ndarray) -> np.ndarray:
        """
        Computes the influence that each weight and bias has on the Cost function and stores those values.

        Args:
            layer: the current layer index.
            dC_da: a matrix describing the influence of each output neuron for this layer on the Cost function.

        Returns: 
            A matrix (dC/da) describing the influence of each input neuron for this layer on the Cost function.
        """
        # Compute z for this layer.
        z: np.ndarray = np.add(
            np.matmul(self.__weights[layer - 1], self.__neurons[layer - 1]),
            self.__biases[layer - 1]
        )

        # Compute dC/db and dC/dw.
        dC_db: np.ndarray = np.multiply(               
            self.__act_deriv(z),
            dC_da
        )
        self.__cost_gradient[layer - 1]["db"] += dC_db
        
        dC_dw: np.ndarray = np.matmul(            
            dC_db[:,np.newaxis],
            self.__neurons[layer - 1][np.newaxis]
        )
        self.__cost_gradient[layer - 1]["dw"] += dC_dw

        # Compute dC_da for this layer.
        dC_da = np.matmul(
            dC_db[np.newaxis],
            self.__weights[layer - 1]
        )

        return np.transpose(dC_da)[:,0]

    def __rand_layer_size(self) -> int:
        """
        Determines a random number of neurons that a hidden layer should have. This number is always between the number
        of neurons present in the input layer and the output layer and never more than twice the neurons in the input layer.

        Returns:
            An integer with the calculated number of neurons for the given hidden layer.
        """
        lower_lim, upper_lim = self.__input_layer, self.__output_layer
        if upper_lim < lower_lim:
            lower_lim, upper_lim = upper_lim, lower_lim
        size: int = np.random.randint(lower_lim, upper_lim)
        size = min(size, self.__input_layer * 2)
        return size

    def __initialize_network(self) -> None:
        """
        Initializes weights, biases and neurons for this neural network.
        """
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
                np.random.uniform(-1, 1, (next_layer_size, this_layer_size))
            )
            self.__biases.append(np.random.uniform(-1, 1, next_layer_size))

        # Establish output layer size.
        self.__neurons.append(np.empty(self.__output_layer))

        # Create random set of weights and biases between this layer and the output.
        last_layer_size: int = self.__neurons[-2].size
        self.__weights.append(
            np.random.uniform(-1, 1, (self.__output_layer, last_layer_size))
        )
        self.__biases.append(np.random.uniform(-1, 1, self.__output_layer))

        self.__cost_gradient = [None for _ in range(len(self.__weights))]
