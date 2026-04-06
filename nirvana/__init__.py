
# Set version
from .version import version
__version__ = version

# Initialize the logger
import logging
from .logger import get_logger
log = get_logger(level=logging.DEBUG)
