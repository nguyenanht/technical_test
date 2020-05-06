from api.config import PACKAGE_ROOT

with open(PACKAGE_ROOT / '__version__') as version_file:
    __version__ = version_file.read().strip()
