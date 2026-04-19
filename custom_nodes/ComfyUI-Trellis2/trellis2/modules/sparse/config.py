from typing import *

CONV = 'flex_gemm'
DEBUG = False


def _detect_attn_backend():
    import os
    env = os.environ.get('SPARSE_ATTN_BACKEND') or os.environ.get('ATTN_BACKEND')
    if env in ['xformers', 'flash_attn', 'flash_attn_3', 'sdpa']:
        return env
    try:
        import flash_attn  # noqa: F401
        return 'flash_attn'
    except ImportError:
        pass
    try:
        import xformers.ops  # noqa: F401
        return 'xformers'
    except ImportError:
        pass
    return 'sdpa'


ATTN = _detect_attn_backend()


def __from_env():
    global CONV, DEBUG
    import os
    env_sparse_conv_backend = os.environ.get('SPARSE_CONV_BACKEND')
    env_sparse_debug = os.environ.get('SPARSE_DEBUG')
    if env_sparse_conv_backend is not None and env_sparse_conv_backend in ['none', 'spconv', 'torchsparse', 'flex_gemm']:
        CONV = env_sparse_conv_backend
    if env_sparse_debug is not None:
        DEBUG = env_sparse_debug == '1'
    print(f"[SPARSE] Conv backend: {CONV}; Attention backend: {ATTN}")


__from_env()


def set_conv_backend(backend: Literal['none', 'spconv', 'torchsparse', 'flex_gemm']):
    global CONV
    CONV = backend

def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug

def set_attn_backend(backend: Literal['xformers', 'flash_attn', 'sdpa']):
    global ATTN
    ATTN = backend
