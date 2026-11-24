# coding=utf-8
__version__ = "6.2.0"

import platform

from zuper_commons.logs import ZLogger

logger = ZLogger("gym-duckietown")
import os

import pyglet

on_mac = "Darwin" in platform.system()
print(pyglet.options)
if on_mac:
    # EGL headless mode does not exist on macOS. Also skip pyglet's shadow window: it is
    # created on import and crashes when there is no screen (headless Mac over SSH).
    # The simulator creates its own context, see graphics.create_shadow_context().
    pyglet.options["headless"] = False
    pyglet.options["shadow_window"] = False
else:
    pyglet.options["headless"] = True

# pyglet checks glGetError after every GL call when debug_gl is on, which costs ~1.3 ms per
# rendered frame. It only takes effect if set before pyglet.gl is imported, so set it here
# (the simulator imports pyglet.gl). DUCKIETOWN_DEBUG_GL=1 turns the checks back on.
pyglet.options["debug_gl"] = os.environ.get("DUCKIETOWN_DEBUG_GL", "0") == "1"

path = os.path.dirname(os.path.dirname(__file__))
logger.debug(f"gym-duckietown version {__version__} path {path}\n")
