# coding=utf-8
"""
Offscreen OpenGL context for macOS using CGL.

pyglet 1.5 can only get a GL context on macOS through a Cocoa window, which needs a
GUI login session with at least one screen. On a headless Mac reached over SSH neither
exists. The simulator only renders into its own frame buffers, so a bare CGL context
(no drawable) is enough. This module creates one and wraps it in a pyglet Context so
pyglet's texture/buffer bookkeeping keeps working.
"""
import ctypes

from pyglet import gl
from pyglet.gl import base

from . import logger

_cgl = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/OpenGL.framework/OpenGL")
_cgl.CGLErrorString.restype = ctypes.c_char_p

kCGLPFAColorSize = 8
kCGLPFAAlphaSize = 11
kCGLPFADepthSize = 12
kCGLPFAAccelerated = 73
kCGLPFAAllowOfflineRenderers = 96
kCGLPFAOpenGLProfile = 99
# The simulator uses the fixed-function pipeline, so it needs a legacy (2.1) context
kCGLOGLPVersion_Legacy = 0x1000


def _check(err: int, what: str):
    if err:
        raise RuntimeError(f"{what} failed: {_cgl.CGLErrorString(err).decode()}")


def _create_cgl_context() -> ctypes.c_void_p:
    base_attrs = [
        kCGLPFAOpenGLProfile, kCGLOGLPVersion_Legacy,
        kCGLPFAColorSize, 24, kCGLPFAAlphaSize, 8, kCGLPFADepthSize, 24,
        kCGLPFAAllowOfflineRenderers,
    ]
    # Prefer the GPU; fall back to Apple's software renderer if that is all we get
    for attrs in (base_attrs + [kCGLPFAAccelerated, 0], base_attrs + [0]):
        attrs = (ctypes.c_int * len(attrs))(*attrs)
        pix, npix = ctypes.c_void_p(), ctypes.c_int()
        err = _cgl.CGLChoosePixelFormat(attrs, ctypes.byref(pix), ctypes.byref(npix))
        if not err and pix:
            break
    else:
        _check(err, "CGLChoosePixelFormat")

    ctx = ctypes.c_void_p()
    err = _cgl.CGLCreateContext(pix, None, ctypes.byref(ctx))
    _cgl.CGLDestroyPixelFormat(pix)
    _check(err, "CGLCreateContext")
    return ctx


class _CGLContext(base.Context):
    """pyglet Context backed by a drawable-less CGL context."""

    def __init__(self, cgl_ctx):
        super().__init__(config=None)
        self._cgl_ctx = cgl_ctx
        # No drawable: satisfy Context.set_current(), which requires a canvas
        self.canvas = True

    def set_current(self):
        _check(_cgl.CGLSetCurrentContext(self._cgl_ctx), "CGLSetCurrentContext")
        super().set_current()


class OffscreenContext:
    """Stands in for the invisible shadow window: the simulator only calls switch_to()."""

    def __init__(self):
        self.context = _CGLContext(_create_cgl_context())

    def switch_to(self):
        if gl.current_context is not self.context:
            self.context.set_current()


_shared = None


def get_offscreen_context() -> OffscreenContext:
    # One context per process, shared by all simulators, since textures are cached
    # globally (graphics.Texture.tex_cache) and must live in the same GL object space
    global _shared
    if _shared is None:
        _shared = OffscreenContext()
        _shared.switch_to()
        info = gl.gl_info
        logger.info(f"CGL offscreen context: {info.get_renderer()} ({info.get_version()})")
    return _shared
