"""Signals."""

from __future__ import annotations

import os
import signal
import time
from threading import Event

from transformers import TrainerCallback


_INTERRUPT_REQUESTED = Event()


def signal_requested() -> bool:
    return _INTERRUPT_REQUESTED.is_set()


def install_signal_handlers() -> None:
    def _request(signum, frame) -> None:
        del signum, frame
        _INTERRUPT_REQUESTED.set()

    signal.signal(signal.SIGUSR1, _request)
    signal.signal(signal.SIGTERM, _request)


class SignalCheckpointCallback(TrainerCallback):

    def __init__(self, save_reserve_seconds: float = 600.0) -> None:
        self.save_reserve_seconds = save_reserve_seconds
        self._step_started: float | None = None
        self._longest_step = 0.0

    def _allocation_ending(self):
        return False

    def on_step_begin(self, args, state, control, **kwargs):
        self._step_started = time.monotonic()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self._step_started is not None:
            self._longest_step = max(self._longest_step, time.monotonic() - self._step_started)
        if not signal_requested() and self._allocation_ending():
            print(f"Allocation ends before another {self._longest_step:.0f}s step; "
                  "saving checkpoint for runtime recovery", flush=True)

            _INTERRUPT_REQUESTED.set()
        if signal_requested():
            control.should_save = True
            control.should_training_stop = True
        return control
