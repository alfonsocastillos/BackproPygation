import logging 
from custom_modules.backpropagation import Backpropagation

logging.basicConfig()
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def main():
    bp = Backpropagation(
        {
            0: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            1: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            2: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            3: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            4: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            5: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            6: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            7: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            8: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            9: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        }, 
        hidden_layers=3
    )
    bp.train()
    logger.info('Finished training.')
    bp.test()
    logger.info('Displaying Neural Network accuracy.')
    
if __name__ == '__main__':
    try:
        main()
    except Exception as ex:
        logger.error('There was an error while executing this program', exc_info=True)