"""Tests for adaptation router and baselines."""

import numpy as np
import pytest

from sdmr.adaptation import (
    RuleBasedRouter,
    CostAwareRouter,
    AdaptationAction,
    RouterDecision,
)
from sdmr.baselines import (
    NoAdapt,
    PeriodicFT,
    FullFTAlways,
    BNAdapt,
    OracleRouter,
    RandomRouter,
    ThresholdRouter,
    create_baseline,
)


class TestAdaptationAction:
    def test_ordering(self):
        assert AdaptationAction.NONE < AdaptationAction.BN_REFRESH
        assert AdaptationAction.BN_REFRESH < AdaptationAction.HEAD_ONLY
        assert AdaptationAction.HEAD_ONLY < AdaptationAction.BOUNDED_FULL
        assert AdaptationAction.BOUNDED_FULL < AdaptationAction.RESET_AND_FT

    def test_cost_ordering(self):
        costs = [a.relative_cost for a in AdaptationAction]
        assert costs == sorted(costs)

    def test_labels(self):
        assert AdaptationAction.NONE.label == "none"
        assert AdaptationAction.BN_REFRESH.label == "bn_refresh"


class TestRuleBasedRouter:
    def test_no_shift_no_action(self):
        router = RuleBasedRouter()
        decision = router.decide(
            shift_detected=False,
            shift_severity=0.0,
            shift_persistence=0.0,
            input_drift_score=0.0,
            output_drift_score=0.0,
            feature_drift_score=0.0,
        )
        assert decision.action == AdaptationAction.NONE

    def test_mild_input_shift(self):
        router = RuleBasedRouter()
        decision = router.decide(
            shift_detected=True,
            shift_severity=0.2,
            shift_persistence=0.1,
            input_drift_score=0.5,
            output_drift_score=0.1,
            feature_drift_score=0.1,
        )
        assert decision.action in (AdaptationAction.BN_REFRESH, AdaptationAction.HEAD_ONLY)

    def test_severe_shift(self):
        router = RuleBasedRouter()
        decision = router.decide(
            shift_detected=True,
            shift_severity=0.9,
            shift_persistence=0.9,
            input_drift_score=0.8,
            output_drift_score=0.8,
            feature_drift_score=0.8,
        )
        assert decision.action in (
            AdaptationAction.BOUNDED_FULL,
            AdaptationAction.RESET_AND_FT,
        )

    def test_cooldown(self):
        router = RuleBasedRouter(cooldown_frames=5)
        # First detection triggers action
        d1 = router.decide(
            shift_detected=True, shift_severity=0.5, shift_persistence=0.5,
            input_drift_score=0.5, output_drift_score=0.5, feature_drift_score=0.5,
        )
        assert d1.action != AdaptationAction.NONE

        # Immediately after should be in cooldown
        d2 = router.decide(
            shift_detected=False, shift_severity=0.0, shift_persistence=0.0,
            input_drift_score=0.0, output_drift_score=0.0, feature_drift_score=0.0,
        )
        assert d2.action == AdaptationAction.NONE

    def test_action_distribution(self):
        router = RuleBasedRouter(cooldown_frames=0)
        for _ in range(10):
            router.decide(
                shift_detected=True, shift_severity=0.5, shift_persistence=0.3,
                input_drift_score=0.5, output_drift_score=0.1, feature_drift_score=0.1,
            )
        dist = router.get_action_distribution()
        assert sum(dist.values()) == 10


class TestCostAwareRouter:
    def test_budget_constraint(self):
        base = RuleBasedRouter(cooldown_frames=0)
        router = CostAwareRouter(
            base, budget_per_window=1.0, window_size=100
        )
        # Force many expensive actions
        high_cost_count = 0
        for _ in range(20):
            d = router.decide(
                shift_detected=True, shift_severity=0.9, shift_persistence=0.9,
                input_drift_score=0.8, output_drift_score=0.8, feature_drift_score=0.8,
            )
            if d.action.relative_cost > 0.5:
                high_cost_count += 1
        # Budget should prevent all from being expensive
        assert high_cost_count < 20


class TestBaselines:
    def test_no_adapt(self):
        b = NoAdapt()
        for _ in range(100):
            d = b.decide()
        assert d.action == AdaptationAction.NONE

    def test_periodic_ft(self):
        b = PeriodicFT(period=10)
        actions = []
        for _ in range(25):
            d = b.decide()
            actions.append(d.action)
        # Should trigger at steps 10, 20
        assert actions[9] != AdaptationAction.NONE  # step 10
        assert actions[19] != AdaptationAction.NONE  # step 20

    def test_bn_adapt(self):
        b = BNAdapt()
        d = b.decide()
        assert d.action == AdaptationAction.BN_REFRESH

    def test_oracle(self):
        schedule = [(10, 20, 0.5), (50, 60, 0.9)]
        b = OracleRouter(shift_schedule=schedule)
        # Advance to step 10
        for _ in range(9):
            b.decide()
        d = b.decide()  # step 10
        assert d.action != AdaptationAction.NONE

    def test_random(self):
        b = RandomRouter(action_prob=1.0, seed=42)
        d = b.decide()
        assert d.action != AdaptationAction.NONE

    def test_threshold(self):
        b = ThresholdRouter(
            signal_key="conf_mean",
            threshold=0.5,
            comparison="below",
            cooldown=0,
        )
        d = b.decide(signal_snapshot={"conf_mean": 0.3})
        assert d.action != AdaptationAction.NONE

    def test_create_baseline_factory(self):
        b = create_baseline("no_adapt")
        assert isinstance(b, NoAdapt)
        b = create_baseline("periodic_ft", period=20)
        assert isinstance(b, PeriodicFT)

    def test_create_baseline_invalid(self):
        with pytest.raises(ValueError):
            create_baseline("nonexistent_baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
