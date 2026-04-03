
# Set version
from .version import version
__version__ = version

def short_warning(message, category, filename, lineno, file=None, line=None):
    """
    Return the format for a short warning message.
    """
    return ' %s: %s\n' % (category.__name__, message)

import warnings
warnings.formatwarning = short_warning

