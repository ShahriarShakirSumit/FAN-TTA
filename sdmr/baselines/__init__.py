"""
Baseline adaptation strategies for comparison with SDMR.

Implements the baselines from the paper:
1. NoAdapt      — static model, no adaptation (lower bound)
2. PeriodicFT   — periodic fine-tuning at fixed intervals
3. FullFTAlways — fine-tune on every frame (upper bound on adaptation cost)
4. BNAdapt      — batch-norm adaptation only (TENT-like)
5. OracleRouter — oracle that knows when shifts occur (upper bound on performance)
6. RandomRouter — random adaptation actions (sanity check)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
import random

from ..adaptation import AdaptationAction, RouterDecision


class BaselineStrategy:
    """Base class for baseline adaptation strategies."""

    def __init__(self, name: str):
        self.name = name
        self._step = 0

    def decide(self, **kwargs) -> RouterDecision:
        self._step += 1
        return self._decide_impl(**kwargs)

    def _decide_impl(self, **kwargs) -> RouterDecision:
        raise NotImplementedError

    def reset(self) -> None:
        self._step = 0


class NoAdapt(BaselineStrategy):
    """No adaptation baseline — static frozen model."""

    def __init__(self):
        super().__init__("no_adapt")

    def _decide_impl(self, **kwargs) -> RouterDecision:
        return RouterDecision(
            action=AdaptationAction.NONE,
            confidence=1.0,
            reasoning="NoAdapt: static model, no changes",
        )


class PeriodicFT(BaselineStrategy):
    """
    Periodic fine-tuning at fixed intervals.

    Args:
        period: Fine-tune every N frames.
        action: Which adaptation action to use.
        budget: Number of gradient steps per adaptation.
    """

    def __init__(
        self,
        period: int = 50,
        action: AdaptationAction = AdaptationAction.HEAD_ONLY,
        budget: int = 10,
    ):
        super().__init__(f"periodic_ft_{period}")
        self.period = period
        self.action = action
        self.budget = budget

    def _decide_impl(self, **kwargs) -> RouterDecision:
        if self._step % self.period == 0:
            return RouterDecision(
                action=self.action,
                confidence=1.0,
                adaptation_budget=self.budget,
                reasoning=f"PeriodicFT: step {self._step} triggers {self.action.label}",
            )
        return RouterDecision(
            action=AdaptationAction.NONE,
            confidence=1.0,
            reasoning=f"PeriodicFT: waiting ({self._step % self.period}/{self.period})",
        )


class FullFTAlways(BaselineStrategy):
    """Fine-tune on every single frame (maximum adaptation cost)."""

    def __init__(
        self,
        action: AdaptationAction = AdaptationAction.BOUNDED_FULL,
        budget: int = 5,
    ):
        super().__init__("full_ft_always")
        self.action = action
        self.budget = budget

    def _decide_impl(self, **kwargs) -> RouterDecision:
        return RouterDecision(
            action=self.action,
            confidence=1.0,
            adaptation_budget=self.budget,
            reasoning="FullFTAlways: adapting every frame",
        )


class BNAdapt(BaselineStrategy):
    """
    Batch-norm only adaptation (TENT/TTT-style).

    Refreshes BN running stats on every frame using test-time data.
    """

    def __init__(self, refresh_interval: int = 1, buffer_size: int = 16):
        super().__init__("bn_adapt")
        self.refresh_interval = refresh_interval
        self.buffer_size = buffer_size

    def _decide_impl(self, **kwargs) -> RouterDecision:
        if self._step % self.refresh_interval == 0:
            return RouterDecision(
                action=AdaptationAction.BN_REFRESH,
                confidence=1.0,
                adaptation_budget=self.buffer_size,
                reasoning="BNAdapt: refreshing batch-norm stats",
            )
        return RouterDecision(
            action=AdaptationAction.NONE,
            confidence=1.0,
            reasoning="BNAdapt: waiting",
        )


class OracleRouter(BaselineStrategy):
    """
    Oracle baseline that knows the exact shift schedule.

    Used as upper bound on adaptation performance.

    Args:
        shift_schedule: List of (onset_frame, end_frame, severity) tuples.
        action_map: Maps severity ranges to actions.
    """

    def __init__(
        self,
        shift_schedule: list[tuple[int, int, float]],
        budget: int = 20,
    ):
        super().__init__("oracle")
        self.shift_schedule = shift_schedule
        self.budget = budget
        self._shift_onsets = {onset for onset, _, _ in shift_schedule}

    def _decide_impl(self, **kwargs) -> RouterDecision:
        # Check if current step is a shift onset
        if self._step in self._shift_onsets:
            # Find severity
            for onset, end, severity in self.shift_schedule:
                if onset == self._step:
                    # Oracle picks optimal action based on severity
                    if severity < 0.3:
                        action = AdaptationAction.BN_REFRESH
                    elif severity < 0.6:
                        action = AdaptationAction.HEAD_ONLY
                    elif severity < 0.8:
                        action = AdaptationAction.BOUNDED_FULL
                    else:
                        action = AdaptationAction.RESET_AND_FT

                    return RouterDecision(
                        action=action,
                        confidence=1.0,
                        severity=severity,
                        adaptation_budget=self.budget,
                        reasoning=f"Oracle: shift at frame {self._step}, sev={severity:.2f}",
                    )

        return RouterDecision(
            action=AdaptationAction.NONE,
            confidence=1.0,
            reasoning="Oracle: no shift at this frame",
        )


class RandomRouter(BaselineStrategy):
    """
    Random adaptation actions (sanity check baseline).

    Args:
        action_prob: Probability of taking any action per frame.
        seed: Random seed.
    """

    def __init__(self, action_prob: float = 0.1, seed: int = 42):
        super().__init__("random")
        self.action_prob = action_prob
        self.rng = random.Random(seed)

    def _decide_impl(self, **kwargs) -> RouterDecision:
        if self.rng.random() < self.action_prob:
            action = AdaptationAction(
                self.rng.randint(1, len(AdaptationAction) - 1)
            )
            return RouterDecision(
                action=action,
                confidence=0.5,
                adaptation_budget=self.rng.randint(1, 20),
                reasoning=f"Random: selected {action.label}",
            )
        return RouterDecision(
            action=AdaptationAction.NONE,
            confidence=0.5,
            reasoning="Random: no action this frame",
        )


class ThresholdRouter(BaselineStrategy):
    """
    Simple threshold-based baseline.

    Adapts whenever a single scalar signal exceeds a threshold.
    This is the "naive" approach that SDMR aims to beat.

    Args:
        signal_key: Which signal to threshold on.
        threshold: Detection threshold.
        action: Action to take when triggered.
    """

    def __init__(
        self,
        signal_key: str = "confidence_mean",
        threshold: float = 0.3,
        comparison: str = "below",  # "below" or "above"
        action: AdaptationAction = AdaptationAction.BOUNDED_FULL,
        budget: int = 10,
        cooldown: int = 20,
    ):
        super().__init__(f"threshold_{signal_key}")
        self.signal_key = signal_key
        self.threshold = threshold
        self.comparison = comparison
        self.action = action
        self.budget = budget
        self.cooldown = cooldown
        self._last_action_step = -999

    def _decide_impl(self, **kwargs) -> RouterDecision:
        signal_snapshot = kwargs.get("signal_snapshot", {})
        value = signal_snapshot.get(self.signal_key, 0.0)

        triggered = (
            (self.comparison == "below" and value < self.threshold) or
            (self.comparison == "above" and value > self.threshold)
        )

        in_cooldown = (self._step - self._last_action_step) < self.cooldown

        if triggered and not in_cooldown:
            self._last_action_step = self._step
            return RouterDecision(
                action=self.action,
                confidence=0.7,
                adaptation_budget=self.budget,
                reasoning=f"Threshold: {self.signal_key}={value:.3f} "
                          f"{'<' if self.comparison == 'below' else '>'} "
                          f"{self.threshold}",
                signal_snapshot=signal_snapshot,
            )

        return RouterDecision(
            action=AdaptationAction.NONE,
            confidence=0.7,
            reasoning=f"Threshold: {self.signal_key}={value:.3f}, no trigger",
            signal_snapshot=signal_snapshot,
        )


# Factory function for creating baselines from config
def create_baseline(name: str, **kwargs) -> BaselineStrategy:
    """Create a baseline strategy by name."""
    import inspect

    registry = {
        "no_adapt": NoAdapt,
        "periodic_ft": PeriodicFT,
        "full_ft_always": FullFTAlways,
        "bn_adapt": BNAdapt,
        "oracle": OracleRouter,
        "random": RandomRouter,
        "threshold": ThresholdRouter,
    }
    if name not in registry:
        raise ValueError(
            f"Unknown baseline: {name}. Available: {list(registry.keys())}"
        )
    cls = registry[name]
    # Filter kwargs to only those accepted by the class constructor
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )
    if has_var_keyword:
        filtered = kwargs
    else:
        filtered = {k: v for k, v in kwargs.items() if k in valid_params}
    return cls(**filtered)
