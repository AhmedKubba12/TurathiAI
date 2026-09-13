"""TurathiAI — diffusion-based text-to-image adaptation for Emirati residential
cultural-heritage architectural design.

Public API is intentionally small; the modules are designed to be used as CLIs
(``python -m turathiai.<module>``) or imported piecewise.
"""
from .config import Config, load_config

__version__ = "1.0.0"
__all__ = ["Config", "load_config", "__version__"]
