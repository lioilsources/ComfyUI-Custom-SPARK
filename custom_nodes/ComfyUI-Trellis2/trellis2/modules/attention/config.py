from typing import *

DEBUG = False


def _detect_backend():
    import os
    env = os.environ.get('ATTN_BACKEND')
    if env in ['xformers', 'flash_attn', 'flash_attn_3', 'sdpa', 'naive']:
        return env
    try:
        import flash_attn_interface  # noqa: F401
        return 'flash_attn_3'
    except ImportError:
        pass
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


BACKEND = _detect_backend()


def __from_env():
    global DEBUG
    import os
    env_attn_debug = os.environ.get('ATTN_DEBUG')
    if env_attn_debug is not None:
        DEBUG = env_attn_debug == '1'
    print(f"[ATTENTION] Using backend: {BACKEND}")


__from_env()


def set_backend(backend: Literal['xformers', 'flash_attn']):
    global BACKEND
    BACKEND = backend

def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug
