"""Cooperative cancellation shared by the server, tools and reasoners."""
from __future__ import annotations

import threading
from contextlib import contextmanager

_state = threading.local()


class ExecutionCancelled(RuntimeError):
    pass


def current_cancel_event():
    return getattr(_state, "cancel_event", None)


def check_cancelled():
    event = current_cancel_event()
    if event is not None and event.is_set():
        raise ExecutionCancelled("Inference cancelled")


@contextmanager
def cancellation_scope(event=None):
    previous = current_cancel_event()
    _state.cancel_event = event if event is not None else previous
    try:
        check_cancelled()
        yield
    finally:
        _state.cancel_event = previous


@contextmanager
def inference_lock(lock):
    while not lock.acquire(timeout=0.1):
        check_cancelled()
    try:
        check_cancelled()
        yield
        check_cancelled()
    finally:
        lock.release()
