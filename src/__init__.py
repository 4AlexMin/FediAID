"""Top-level package for DOSN-AIGT detector."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("dosn_aigt_detector")
except PackageNotFoundError:
    __version__ = "0.0.0"
