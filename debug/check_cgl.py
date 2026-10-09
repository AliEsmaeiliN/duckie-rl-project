#!/usr/bin/env python3
"""
Probe whether macOS can give this process an offscreen OpenGL context via CGL,
without a window, a screen, or a GUI login session (e.g. over plain SSH).

Usage (on the Mac):  python debug/check_cgl.py
"""
import ctypes
import sys

if sys.platform != "darwin":
    sys.exit("This probe is macOS-only.")

_cgl = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/OpenGL.framework/OpenGL")

kCGLPFAColorSize = 8
kCGLPFAAlphaSize = 11
kCGLPFADepthSize = 12
kCGLPFAAccelerated = 73
kCGLPFAAllowOfflineRenderers = 96
kCGLPFAOpenGLProfile = 99
kCGLOGLPVersion_Legacy = 0x1000

GL_VENDOR, GL_RENDERER, GL_VERSION = 0x1F00, 0x1F01, 0x1F02
GL_FRAMEBUFFER, GL_RENDERBUFFER = 0x8D40, 0x8D41
GL_RGBA8, GL_COLOR_ATTACHMENT0, GL_FRAMEBUFFER_COMPLETE = 0x8058, 0x8CE0, 0x8CD5
GL_COLOR_BUFFER_BIT, GL_RGBA, GL_UNSIGNED_BYTE = 0x4000, 0x1908, 0x1401

_cgl.CGLErrorString.restype = ctypes.c_char_p
_cgl.glGetString.restype = ctypes.c_char_p
_cgl.glCheckFramebufferStatus.restype = ctypes.c_uint
_cgl.glClearColor.argtypes = [ctypes.c_float] * 4


def try_context(accelerated: bool):
    attrs = [
        kCGLPFAOpenGLProfile, kCGLOGLPVersion_Legacy,
        kCGLPFAColorSize, 24, kCGLPFAAlphaSize, 8, kCGLPFADepthSize, 24,
        kCGLPFAAllowOfflineRenderers,
    ]
    if accelerated:
        attrs.append(kCGLPFAAccelerated)
    attrs.append(0)
    attrs = (ctypes.c_int * len(attrs))(*attrs)

    pix, npix = ctypes.c_void_p(), ctypes.c_int()
    err = _cgl.CGLChoosePixelFormat(attrs, ctypes.byref(pix), ctypes.byref(npix))
    if err or not pix:
        return f"CGLChoosePixelFormat failed: {_cgl.CGLErrorString(err).decode()}"

    ctx = ctypes.c_void_p()
    err = _cgl.CGLCreateContext(pix, None, ctypes.byref(ctx))
    _cgl.CGLDestroyPixelFormat(pix)
    if err:
        return f"CGLCreateContext failed: {_cgl.CGLErrorString(err).decode()}"
    _cgl.CGLSetCurrentContext(ctx)

    info = {k: _cgl.glGetString(v).decode() for k, v in
            [("vendor", GL_VENDOR), ("renderer", GL_RENDERER), ("version", GL_VERSION)]}

    # Render into an FBO and read it back, exactly like the simulator does.
    w = h = 64
    fbo, rb = ctypes.c_uint(), ctypes.c_uint()
    _cgl.glGenFramebuffers(1, ctypes.byref(fbo))
    _cgl.glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    _cgl.glGenRenderbuffers(1, ctypes.byref(rb))
    _cgl.glBindRenderbuffer(GL_RENDERBUFFER, rb)
    _cgl.glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, w, h)
    _cgl.glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb)
    status = _cgl.glCheckFramebufferStatus(GL_FRAMEBUFFER)
    info["fbo_complete"] = status == GL_FRAMEBUFFER_COMPLETE

    _cgl.glViewport(0, 0, w, h)
    _cgl.glClearColor(1.0, 0.0, 0.0, 1.0)
    _cgl.glClear(GL_COLOR_BUFFER_BIT)
    buf = (ctypes.c_ubyte * (w * h * 4))()
    _cgl.glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, buf)
    info["readback_rgba"] = tuple(buf[:4])
    info["readback_ok"] = tuple(buf[:4]) == (255, 0, 0, 255)

    _cgl.CGLSetCurrentContext(None)
    _cgl.CGLDestroyContext(ctx)
    return info


if __name__ == "__main__":
    for accel in (True, False):
        print(f"--- accelerated={accel} ---")
        print(try_context(accel))
