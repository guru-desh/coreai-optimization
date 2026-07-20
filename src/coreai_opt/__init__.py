# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""coreai_opt - A library for PyTorch model compression and optimizations.

For deployment via Core AI on Apple Silicon.
"""

from . import palettization, pruning, quantization
from ._about import __version__
from .common import CoreAIExportError, CoreMLExportError, ExportBackend

__all__ = [
    "CoreAIExportError",
    "CoreMLExportError",
    "ExportBackend",
    "__version__",
]
