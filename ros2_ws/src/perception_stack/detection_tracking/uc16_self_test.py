"""UC-16: Startup self-test (ES-10.1).

On launch the application checks the camera, LiDAR, gimbal link, storage and
compute in parallel, each with its own timeout, and the whole run must finish
within 30 s. A check that raises, returns False or times out fails. Start is
blocked until every check passes, and the error names the failed checks so
the operator knows what to fix.
"""
from __future__ import annotations

import shutil
import threading
import time
from dataclasses import dataclass
from typing import Callable

CheckFn = Callable[[], "bool | tuple[bool, str]"]


@dataclass(frozen=True)
class Check:
    name: str
    fn: CheckFn
    timeout_s: float = 5.0


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str
    duration_s: float


@dataclass(frozen=True)
class SelfTestReport:
    results: tuple[CheckResult, ...]
    duration_s: float

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failed_names(self) -> list[str]:
        return [r.name for r in self.results if not r.passed]


class StartBlocked(RuntimeError):
    def __init__(self, failed: list[str]):
        super().__init__("Start blocked; failed self-test checks: " + ", ".join(failed))
        self.failed = failed


REQUIRED_CHECKS = ("camera", "lidar", "gimbal_link", "storage", "compute")


def run_self_test(checks: list[Check], budget_s: float = 30.0) -> SelfTestReport:
    names = {c.name for c in checks}
    missing = [n for n in REQUIRED_CHECKS if n not in names]
    if missing:
        raise ValueError(f"self-test is missing required checks: {missing}")

    started = time.monotonic()
    outcomes: dict[str, tuple[bool, str, float]] = {}
    threads = []
    for check in checks:
        # Daemon threads: a hung driver call can't keep the app from exiting.
        thread = threading.Thread(target=_run_one, args=(check, outcomes), daemon=True,
                                  name=f"selftest-{check.name}")
        thread.start()
        threads.append((check, thread))

    results: list[CheckResult] = []
    for check, thread in threads:
        remaining = budget_s - (time.monotonic() - started)
        thread.join(timeout=max(0.0, min(check.timeout_s, remaining)))
        if check.name in outcomes:
            ok, detail, duration = outcomes[check.name]
        else:
            ok, detail, duration = False, f"timed out after {check.timeout_s:.1f} s", check.timeout_s
        results.append(CheckResult(check.name, ok, detail, duration))
    return SelfTestReport(tuple(results), time.monotonic() - started)


def _run_one(check: Check, outcomes: dict) -> None:
    t0 = time.monotonic()
    try:
        out = check.fn()
        ok, detail = (out if isinstance(out, tuple) else (bool(out), "ok" if out else "failed"))
        outcomes[check.name] = (bool(ok), detail, time.monotonic() - t0)
    except Exception as exc:  # a crashing check is a failed check
        outcomes[check.name] = (False, f"error: {exc}", time.monotonic() - t0)


class SessionStartGate:
    """Start is allowed only after a passing self-test."""

    def __init__(self):
        self.report: SelfTestReport | None = None

    def record(self, report: SelfTestReport) -> None:
        self.report = report

    def start(self) -> None:
        if self.report is None:
            raise StartBlocked(["self-test not run"])
        if not self.report.passed:
            raise StartBlocked(self.report.failed_names)


def storage_check(path: str, min_free_gb: float = 100.0) -> Check:
    """Default storage check: enough free space for a recording session."""
    def fn():
        free_gb = shutil.disk_usage(path).free / 1e9
        return free_gb >= min_free_gb, f"{free_gb:.0f} GB free (need {min_free_gb:.0f} GB)"
    return Check("storage", fn, timeout_s=2.0)
