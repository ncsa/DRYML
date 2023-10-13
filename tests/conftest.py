from fixtures import create_name, create_temp_file, \
    create_temp_named_file, create_temp_dir, ray_server, \
    get_ray

from dryml.core2.util import get_relevant_context
from os.path import realpath, basename
from icecream import install, IceCreamDebugger
from inspect import getframeinfo
install()

# Install our own context getter for icecream 2.1.3
def dryml_get_context(self, callFrame, callNode):
    frameInfo = getframeinfo(callFrame)
    lineNumber = callNode.lineno
    parentFunction = get_relevant_context(callFrame)

    filepath = (realpath if self.contextAbsPath else basename)(frameInfo.filename)
    return filepath, lineNumber, parentFunction

IceCreamDebugger._getContext = dryml_get_context

ic.configureOutput(
    includeContext=True,
    outputFunction=lambda s: print(s),
    prefix="Debug | ")

__all__ = [
    create_name,
    create_temp_file,
    create_temp_dir,
    create_temp_named_file,
    ray_server,
    get_ray,
]
