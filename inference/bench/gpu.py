"""GPU telemetry via nvidia-smi: memory, temperature, power.

Used to record peak VRAM during a benchmark and to protect the laptop:
runs wait for the GPU to cool down first and abort above a hard limit.
"""

from __future__ import annotations

import math
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Self

QUERY = "memory.used,memory.total,temperature.gpu,power.draw,utilization.gpu"
SMI_CMD = ["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"]

TEMP_LIMIT_C = 85.0  # project rule: keep below ~85C on long runs
COOLDOWN_TARGET_C = 65.0


@dataclass(frozen=True)
class GpuSample:
    mem_used_mib: float
    mem_total_mib: float
    temp_c: float
    power_w: float
    util_pct: float


def _num(text: str) -> float:
    try:
        return float(text)
    except ValueError:  # "[N/A]" on some laptop GPUs
        return float("nan")


def parse_smi(line: str) -> GpuSample:
    parts = [p.strip() for p in line.strip().splitlines()[0].split(",")]
    if len(parts) != 5:
        raise ValueError(f"unexpected nvidia-smi output: {line!r}")
    return GpuSample(*(_num(p) for p in parts))


def read_gpu(run: Callable[[], str] | None = None) -> GpuSample:
    if run is None:

        def run() -> str:
            return subprocess.run(SMI_CMD, capture_output=True, text=True, check=True, timeout=10).stdout

    return parse_smi(run())


@dataclass
class GpuStats:
    samples: int = 0
    peak_mem_mib: float = 0.0
    peak_temp_c: float = 0.0
    mean_power_w: float = 0.0
    over_temp: bool = False
    _power_sum: float = field(default=0.0, repr=False)
    _power_n: int = field(default=0, repr=False)

    def add(self, s: GpuSample) -> None:
        self.samples += 1
        self.peak_mem_mib = max(self.peak_mem_mib, s.mem_used_mib)
        if not math.isnan(s.temp_c):
            self.peak_temp_c = max(self.peak_temp_c, s.temp_c)
            if s.temp_c >= TEMP_LIMIT_C:
                self.over_temp = True
        if not math.isnan(s.power_w):
            self._power_sum += s.power_w
            self._power_n += 1
            self.mean_power_w = self._power_sum / self._power_n


class GpuSampler:
    """Polls the GPU in a background thread while a benchmark runs."""

    def __init__(
        self,
        interval_s: float = 0.5,
        reader: Callable[[], GpuSample] = read_gpu,
        on_over_temp: Callable[[], None] | None = None,
    ) -> None:
        self.interval_s = interval_s
        self.reader = reader
        self.on_over_temp = on_over_temp
        self.stats = GpuStats()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        fired = False
        while not self._stop.is_set():
            try:
                sample = self.reader()
            except Exception:  # noqa: BLE001 - telemetry must never kill a run
                sample = None
            if sample is not None:
                self.stats.add(sample)
            if self.stats.over_temp and not fired and self.on_over_temp:
                fired = True
                self.on_over_temp()
            self._stop.wait(self.interval_s)

    def __enter__(self) -> Self:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)


def wait_for_cooldown(
    target_c: float = COOLDOWN_TARGET_C,
    timeout_s: float = 300.0,
    reader: Callable[[], GpuSample] = read_gpu,
    sleep: Callable[[float], None] = time.sleep,
) -> float:
    """Block until the GPU is at or below `target_c`; returns the last temperature."""
    waited = 0.0
    temp = reader().temp_c
    while not math.isnan(temp) and temp > target_c and waited < timeout_s:
        sleep(5.0)
        waited += 5.0
        temp = reader().temp_c
    return temp
