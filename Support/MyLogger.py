import logging
import logging.handlers

def get_logger():
    mylogger = logging.getLogger("my")
    mylogger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler('my.log', mode='a')
    mylogger.addHandler(file_handler)
    return mylogger

MY_LOGGER = get_logger()

def debuglog(message):
    MY_LOGGER.debug(message)