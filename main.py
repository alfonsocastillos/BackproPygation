from logging import Logger, getLogger

logger: Logger = getLogger(__name__)

def main():
    pass
    
if __name__ == '__main__':
    try:
        main()
    except Exception as ex:
        logger.error('There was an error while executing this program', exc_info=True)