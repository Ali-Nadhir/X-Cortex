"""Local inference resource measurements; never sent to a tracing service."""
from __future__ import annotations

import threading
import time
from typing import Any


class ResourceMonitor:
    def __init__(self, interval: float = 0.1):
        import psutil

        self.process = psutil.Process()
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True, name="xcortex-resource-sampler")
        self.peak_rss = 0
        self.peak_private = 0
        self.samples = 0
        self.initial: dict[str, Any] = {}
        self.final: dict[str, Any] = {}

    def _sample(self) -> dict[str, Any]:
        memory = self.process.memory_info()._asdict()
        self.peak_rss = max(self.peak_rss, memory.get("rss", 0))
        self.peak_private = max(self.peak_private, memory.get("private", memory.get("pagefile", 0)))
        self.samples += 1
        return memory

    def _loop(self):
        while not self.stop_event.wait(self.interval):
            self._sample()

    def start(self):
        self.initial = self._sample()
        self.thread.start()
        return self

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        self.thread.join(timeout=2)
        self.final = self._sample()
        return {
            "method": "process RSS/private-byte sampling", "sample_interval_seconds": self.interval,
            "samples": self.samples, "sampled_peak_rss_bytes": self.peak_rss,
            "sampled_peak_private_bytes": self.peak_private,
            "os_lifetime_peak_rss_bytes": self.final.get("peak_wset"),
            "initial": self.initial, "final": self.final,
            "page_fault_delta": self.final.get("num_page_faults", 0) - self.initial.get("num_page_faults", 0),
        }
