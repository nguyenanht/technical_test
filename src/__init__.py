import logging

from src.config import logging_config
from src.config import config

VERSION_PATH = config.PACKAGE_ROOT / '__version__'

# Configure logger for use in package
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging_config.get_console_handler())
logger.propagate = False


with open(VERSION_PATH, 'r') as version_file:
    __version__ = version_file.read().strip()
