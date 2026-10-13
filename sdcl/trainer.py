"""Compatibility import for locally saved historical detector checkpoints."""

import sys
from sdcl.training import trainer as _implementation

sys.modules[__name__] = _implementation
