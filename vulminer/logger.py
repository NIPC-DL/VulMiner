import os
import logging

if not os.path.exists('.log'):
    os.mkdir('.log')
    open('.log/vm.log', 'a').close()

LOG_FILE = '.log/vm.log'
LOG_LEVEL = logging.DEBUG
FMT = logging.Formatter('%(asctime)s %(filename)s %(lineno)d %(message)s')

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(LOG_LEVEL)
file_handler.setFormatter(FMT)
logger.addHandler(file_handler)

cmd_handler = logging.StreamHandler()
cmd_handler.setLevel(LOG_LEVEL)
cmd_handler.setFormatter(FMT)
logger.addHandler(cmd_handler)
