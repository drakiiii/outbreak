"""Simulation controller: a pausable, inspectable driver around the engine.

:class:`Simulation` wraps an :class:`~outbreak.model.EpidemicModel` with an
explicit state machine (idle / running / paused / finished), a complete
step-by-step history, and (de)serialisation so a run can be saved, reloaded and
resumed exactly. Because the engine advances one step at a time, pausing is
simply "stop calling :meth:`step`"; all state and history remain available for
inspection.

For stochastic studies, :func:`run_ensemble` runs many independent realisations
and returns their histories for aggregation in :mod:`outbreak.metrics`.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Callable, List, Optional

import numpy as np

from .agents import AgentModel
from .config import ScenarioConfig
from .metrics import EpidemicSummary, history_to_columns, summarize
from .model import EpidemicModel, StepRecord


# Subclassing both ``str`` and ``Enum`` makes each member compare/serialise as
# its string value (e.g. ``RunState.IDLE == "idle"``), while still giving the
# named members of a state machine. ``.value`` yields the plain string.
class RunState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"


def build_engine(config: ScenarioConfig, rng: Optional[np.random.Generator] = None):
    """Construct the engine selected by ``config.simulation.engine``.

    Both engines satisfy the same interface the controller relies on, so the
    rest of the package is agnostic to which one is running.
    """
    # Dispatch on a config string to pick the concrete engine class. Both expose
    # the same step()/get_state()/set_state() interface, so callers don't branch.
    if config.simulation.engine == "agent":
        return AgentModel(config, rng=rng)
    return EpidemicModel(config, rng=rng)


class Simulation:
    """Drive and observe a single epidemic realisation."""

    def __init__(self, config: ScenarioConfig, rng: Optional[np.random.Generator] = None):
        self.config = config.validate()  # validate() returns a checked config
        self.model = build_engine(self.config, rng=rng)
        self.history: List[StepRecord] = []  # one StepRecord appended per step()
        self.state: RunState = RunState.IDLE
        self._n_steps = self.config.simulation.n_steps  # total steps for a full run

    # ------------------------------------------------------------- properties
    @property
    def current_day(self) -> float:
        return self.model.current_day()

    @property
    def progress(self) -> float:
        """Fraction of the configured duration completed, in [0, 1]."""
        # model.t is the current step index; clamp to 1.0 and guard against /0.
        return min(1.0, self.model.t / self._n_steps) if self._n_steps else 1.0

    @property
    def is_finished(self) -> bool:
        return self.state == RunState.FINISHED

    @property
    def r0(self) -> float:
        return self.model.config.disease.r0

    # ------------------------------------------------------------- transitions
    def step(self) -> Optional[StepRecord]:
        """Advance one step. Returns the new record, or ``None`` if finished."""
        # Already at the end: mark finished and signal "nothing produced".
        if self.model.t >= self._n_steps:
            self.state = RunState.FINISHED
            return None
        self.state = RunState.RUNNING
        record = self.model.step()  # engine advances t and returns the new state
        self.history.append(record)
        # If that step reached the configured horizon, transition to FINISHED.
        if self.model.t >= self._n_steps:
            self.state = RunState.FINISHED
        return record

    def run(
        self,
        steps: Optional[int] = None,
        on_step: Optional[Callable[[StepRecord], None]] = None,
        stop_when: Optional[Callable[["Simulation"], bool]] = None,
    ) -> List[StepRecord]:
        """Run up to ``steps`` steps (or to completion if ``None``).

        ``on_step`` is called after each step (useful for live UI updates);
        ``stop_when`` is polled after each step and, if it returns True, pauses
        the run early (useful for "run until Rt < 1", "run to day 100", etc.).
        """
        produced: List[StepRecord] = []
        # How many steps to attempt: the explicit count, or "remaining to horizon".
        budget = steps if steps is not None else self._n_steps - self.model.t
        for _ in range(max(0, budget)):  # max(0, ...) guards against a negative budget
            record = self.step()
            if record is None:  # step() returned None => already finished, stop looping
                break
            produced.append(record)
            if on_step is not None:
                on_step(record)  # per-step callback, e.g. push an update to a UI
            # stop_when is polled after each step; truthy result pauses the run early.
            if stop_when is not None and stop_when(self):
                self.state = RunState.PAUSED
                break
        return produced

    def run_to_end(self, on_step: Optional[Callable[[StepRecord], None]] = None) -> List[StepRecord]:
        return self.run(steps=None, on_step=on_step)

    def pause(self) -> None:
        # No-op unless currently running; pausing is just "stop calling step()".
        if self.state == RunState.RUNNING:
            self.state = RunState.PAUSED

    def resume(self, **kwargs) -> List[StepRecord]:
        # Resuming a finished run does nothing; otherwise continue from current t.
        if self.state == RunState.FINISHED:
            return []
        return self.run(**kwargs)

    def reset(self, rng: Optional[np.random.Generator] = None) -> None:
        """Restart from t=0 with the same configuration."""
        # Rebuild a fresh engine (t back to 0) and clear history/state.
        self.model = build_engine(self.config, rng=rng)
        self.history = []
        self.state = RunState.IDLE

    # --------------------------------------------------------------- analysis
    def summary(self) -> Optional[EpidemicSummary]:
        return summarize(self.history, self.r0, self.config.reporting)

    def to_columns(self) -> dict:
        return history_to_columns(self.history, self.config.reporting)

    def record_at_day(self, day: float) -> Optional[StepRecord]:
        """Return the recorded state nearest to ``day`` (for timeline scrubbing)."""
        if not self.history:
            return None
        days = np.array([r.day for r in self.history])
        # argmin of the absolute differences = index of the closest recorded day.
        return self.history[int(np.argmin(np.abs(days - day)))]

    # ----------------------------------------------------------- (de)serialise
    def to_dict(self) -> dict:
        """Full snapshot: configuration + engine state + run state.

        History is intentionally omitted to keep snapshots compact; it can be
        replayed deterministically, or exported separately via :meth:`to_columns`.
        """
        return {
            "config": self.config.to_dict(),
            "engine_state": self.model.get_state(),  # everything needed to resume
            "run_state": self.state.value,           # store the plain string, not the Enum
            "n_steps": self._n_steps,
        }

    def save(self, path: str) -> None:
        # default=_json_default tells json how to encode NumPy scalars/arrays.
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=_json_default)

    @classmethod
    def from_dict(cls, data: dict) -> "Simulation":
        # Rebuild the config first, construct a fresh Simulation, then overlay the
        # saved engine and run state so the run resumes exactly where it left off.
        config = ScenarioConfig.from_dict(data["config"])
        sim = cls(config)
        sim.model.set_state(data["engine_state"])
        # RunState(...) parses the stored string back into the enum member.
        sim.state = RunState(data.get("run_state", RunState.PAUSED.value))
        sim._n_steps = data.get("n_steps", config.simulation.n_steps)
        return sim

    @classmethod
    def load(cls, path: str) -> "Simulation":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


def run_ensemble(
    config: ScenarioConfig,
    n_runs: int = 50,
    base_seed: Optional[int] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> List[List[StepRecord]]:
    """Run ``n_runs`` independent stochastic realisations to completion.

    Each run gets its own deterministic seed derived from ``base_seed`` so the
    whole ensemble is reproducible. Returns a list of histories suitable for
    :func:`outbreak.metrics.aggregate_ensemble`.
    """
    # SeedSequence + spawn() derives n_runs statistically-independent child seeds
    # from one base seed. This is the recommended way to get reproducible yet
    # non-correlated RNG streams for parallel/independent realisations.
    seed_seq = np.random.SeedSequence(base_seed)
    child_seeds = seed_seq.spawn(n_runs)
    histories: List[List[StepRecord]] = []
    for i, child in enumerate(child_seeds):
        rng = np.random.default_rng(child)  # one fresh Generator per run
        sim = Simulation(config, rng=rng)
        sim.run_to_end()
        histories.append(sim.history)
        if progress is not None:
            progress(i + 1, n_runs)  # report (completed, total) to an optional callback
    return histories


# json.dump calls this for any value it can't serialise natively. NumPy scalar
# and array types aren't JSON-native, so convert them to plain Python types.
def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # ndarray -> nested lists
    if isinstance(obj, (bytes, bytearray)):
        return list(obj)
    # Re-raise as TypeError so json reports an unencodable object as usual.
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")
