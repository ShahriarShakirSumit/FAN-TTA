"""
Tests for the theoretical foundations module.

Tests cover:
1. Problem formulation (loss computation, budget, objective)
2. Regret analysis (bounds, empirical regret, oracle)
3. Convergence guarantees (parameter bounds, forgetting, BN stability)
"""

import numpy as np
import pytest

from sdmr.theory.problem_formulation import (
    StreamingTTAProblem,
    AdaptationPolicy,
    ShiftDetector,
    RegretMetric,
)
from sdmr.theory.regret_analysis import (
    RouterRegretBound,
    compute_empirical_regret,
    oracle_hindsight_cost,
)
from sdmr.theory.convergence import (
    BoundedAdaptationConvergence,
    bn_refresh_stability_bound,
    adaptation_forgetting_bound,
)


# ──────────────────────────────────────────────────────────────────────
# Problem Formulation
# ──────────────────────────────────────────────────────────────────────

class TestStreamingTTAProblem:
    def test_default_construction(self):
        prob = StreamingTTAProblem()
        assert prob.num_frames == 1000
        assert prob.num_actions == 5
        assert len(prob.action_costs) == 5
        assert np.isclose(prob.action_costs[0], 0.0)  # NONE is free

    def test_action_labels(self):
        prob = StreamingTTAProblem()
        labels = prob.action_labels
        assert len(labels) == 5
        assert labels[0] == "none"

    def test_detection_loss_perfect(self):
        prob = StreamingTTAProblem()
        # Perfect detection: one box, matches GT exactly
        preds = {
            'boxes': np.array([[10, 20, 50, 60]]),
            'scores': np.array([0.9]),
            'classes': np.array([0]),
        }
        gt = {
            'boxes': np.array([[10, 20, 50, 60]]),
            'labels': np.array([0]),
        }
        loss = prob.compute_detection_loss(preds, gt)
        assert loss < 0.1  # nearly zero loss

    def test_detection_loss_no_predictions(self):
        prob = StreamingTTAProblem()
        preds = {'boxes': np.array([]).reshape(0, 4), 'scores': np.array([])}
        gt = {'boxes': np.array([[10, 20, 50, 60]]), 'labels': np.array([0])}
        loss = prob.compute_detection_loss(preds, gt)
        assert loss == 1.0  # missed all

    def test_detection_loss_no_gt(self):
        prob = StreamingTTAProblem()
        preds = {
            'boxes': np.array([[10, 20, 50, 60]]),
            'scores': np.array([0.9]),
        }
        gt = {'boxes': np.array([]).reshape(0, 4), 'labels': np.array([])}
        loss = prob.compute_detection_loss(preds, gt)
        assert loss == 1.0  # false positives

    def test_detection_loss_empty_both(self):
        prob = StreamingTTAProblem()
        preds = {'boxes': np.array([]).reshape(0, 4), 'scores': np.array([])}
        gt = {'boxes': np.array([]).reshape(0, 4)}
        loss = prob.compute_detection_loss(preds, gt)
        assert loss == 0.0  # no objects, no error

    def test_cumulative_cost(self):
        prob = StreamingTTAProblem()
        actions = np.array([0, 0, 1, 0, 2])  # NONE, NONE, BN, NONE, HEAD
        cost = prob.compute_cumulative_cost(actions)
        expected = 0.0 + 0.0 + 0.05 + 0.0 + 0.3
        assert np.isclose(cost, expected)

    def test_budget_feasibility(self):
        prob = StreamingTTAProblem(total_budget=1.0)
        feasible = np.array([0, 0, 0, 1, 0])  # cost = 0.05
        assert prob.is_budget_feasible(feasible)
        infeasible = np.array([4, 4, 4, 4, 4])  # cost = 5.0
        assert not prob.is_budget_feasible(infeasible)

    def test_objective(self):
        prob = StreamingTTAProblem(total_budget=100.0)
        losses = np.array([0.2, 0.3, 0.1, 0.4, 0.5])
        actions = np.array([0, 0, 0, 0, 0])
        obj = prob.compute_objective(losses, actions, lambda_cost=0.0)
        assert np.isclose(obj, np.mean(losses))

    def test_objective_with_budget_violation(self):
        prob = StreamingTTAProblem(total_budget=0.01)  # very tight budget
        losses = np.array([0.5, 0.5])
        actions = np.array([3, 3])  # BOUNDED_FULL, cost 0.7 each
        obj = prob.compute_objective(losses, actions, lambda_cost=10.0)
        assert obj > np.mean(losses)  # penalty added

    def test_iou_computation(self):
        box1 = np.array([0, 0, 10, 10])
        boxes = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]])
        ious = StreamingTTAProblem._compute_iou_vector(box1, boxes)
        assert np.isclose(ious[0], 1.0)   # identical
        assert ious[1] > 0 and ious[1] < 1.0  # partial overlap
        assert np.isclose(ious[2], 0.0)   # no overlap

    def test_custom_construction(self):
        prob = StreamingTTAProblem(
            num_frames=500,
            num_actions=3,
            action_costs=np.array([0.0, 0.1, 0.5]),
            total_budget=25.0,
        )
        assert prob.num_actions == 3
        assert prob.total_budget == 25.0


class TestShiftDetector:
    def test_delay_bound(self):
        det = ShiftDetector(signal_variance=1.0, shift_magnitude=0.5, false_alarm_rate=0.05)
        delay = det.theoretical_delay_bound
        # σ² · log(1/α) / Δ² = 1.0 * log(20) / 0.25 ≈ 12.0
        assert delay > 0
        assert delay > 10  # should be around 12

    def test_larger_shift_lower_delay(self):
        small = ShiftDetector(shift_magnitude=0.1)
        large = ShiftDetector(shift_magnitude=1.0)
        assert large.theoretical_delay_bound < small.theoretical_delay_bound

    def test_sample_complexity(self):
        det = ShiftDetector(signal_variance=1.0, shift_magnitude=0.5)
        n = det.compute_sample_complexity(confidence=0.95)
        assert isinstance(n, int)
        assert n > 0

    def test_expected_delay(self):
        det = ShiftDetector()
        assert det.expected_detection_delay == det.theoretical_delay_bound


class TestRegretMetric:
    def test_zero_regret(self):
        losses = np.array([0.1, 0.2, 0.3])
        metric = RegretMetric(
            policy_losses=losses,
            oracle_losses=losses.copy(),
        )
        assert np.isclose(metric.cumulative_regret, 0.0)

    def test_positive_regret(self):
        policy = np.array([0.5, 0.6, 0.7])
        oracle = np.array([0.1, 0.2, 0.3])
        metric = RegretMetric(policy_losses=policy, oracle_losses=oracle)
        assert metric.cumulative_regret > 0
        assert np.isclose(metric.cumulative_regret, 1.2)

    def test_regret_trajectory(self):
        policy = np.array([0.5, 0.6, 0.7])
        oracle = np.array([0.1, 0.2, 0.3])
        metric = RegretMetric(policy_losses=policy, oracle_losses=oracle)
        traj = metric.regret_trajectory
        assert len(traj) == 3
        assert traj[-1] == pytest.approx(1.2)
        # Trajectory should be monotonically increasing (since regret > 0)
        assert all(traj[i] <= traj[i + 1] for i in range(len(traj) - 1))

    def test_sublinear_regret(self):
        # Construct sqrt-growth regret
        T = 200
        t = np.arange(1, T + 1, dtype=float)
        policy = 0.5 + 0.1 / np.sqrt(t)  # decreasing gap
        oracle = np.full(T, 0.5)
        metric = RegretMetric(policy_losses=policy, oracle_losses=oracle)
        assert metric.is_sublinear()

    def test_average_regret(self):
        metric = RegretMetric(
            policy_losses=np.array([1.0, 1.0]),
            oracle_losses=np.array([0.0, 0.0]),
        )
        assert np.isclose(metric.average_regret, 1.0)

    def test_empty_regret(self):
        metric = RegretMetric()
        assert metric.cumulative_regret == 0.0

    def test_cost_regret(self):
        metric = RegretMetric(
            policy_losses=np.array([0.5]),
            oracle_losses=np.array([0.5]),
            policy_costs=np.array([0.7]),
            oracle_costs=np.array([0.1]),
        )
        assert np.isclose(metric.cost_regret, 0.6)


# ──────────────────────────────────────────────────────────────────────
# Regret Analysis
# ──────────────────────────────────────────────────────────────────────

class TestRouterRegretBound:
    def test_static_bound(self):
        bound = RouterRegretBound(num_frames=1000, num_actions=5)
        static = bound.static_regret_bound()
        # √(2 · 1000 · log(5)) ≈ √(2 · 1000 · 1.609) ≈ 56.7
        assert 50 < static < 70

    def test_dynamic_bound_geq_static(self):
        bound = RouterRegretBound(num_frames=1000, num_actions=5, num_shifts=3)
        assert bound.dynamic_regret_bound() >= bound.static_regret_bound()

    def test_more_shifts_higher_regret(self):
        b1 = RouterRegretBound(num_shifts=1)
        b3 = RouterRegretBound(num_shifts=3)
        b10 = RouterRegretBound(num_shifts=10)
        assert b1.dynamic_regret_bound() < b3.dynamic_regret_bound()
        assert b3.dynamic_regret_bound() < b10.dynamic_regret_bound()

    def test_sublinear_rate(self):
        bound = RouterRegretBound(num_frames=10000, num_shifts=3)
        rate = bound.normalized_dynamic_regret_rate()
        assert rate < 0.1  # should be small for large T

    def test_rate_decreases_with_T(self):
        rate_100 = RouterRegretBound(num_frames=100).normalized_dynamic_regret_rate()
        rate_10000 = RouterRegretBound(num_frames=10000).normalized_dynamic_regret_rate()
        assert rate_10000 < rate_100

    def test_detection_delay(self):
        bound = RouterRegretBound(
            signal_variance=1.0,
            min_shift_magnitude=0.5,
            false_alarm_rate=0.05,
        )
        assert bound.max_detection_delay > 0

    def test_validate_empirical_within_bound(self):
        bound = RouterRegretBound(num_frames=100, num_actions=5)
        result = bound.validate_empirical_regret(5.0)
        assert 'within_static_bound' in result
        assert 'within_dynamic_bound' in result
        assert result['within_static_bound']  # 5.0 should be within bound

    def test_validate_empirical_exceeds_bound(self):
        bound = RouterRegretBound(num_frames=10, num_actions=2)
        result = bound.validate_empirical_regret(1000.0)
        assert not result['within_static_bound']

    def test_cost_regret_bound(self):
        bound = RouterRegretBound()
        cost_bound = bound.cost_regret_bound()
        assert cost_bound > 0

    def test_per_segment_bound(self):
        bound = RouterRegretBound(num_actions=5)
        seg_bound = bound.per_segment_bound(100)
        assert seg_bound > 0
        # Shorter segments → lower regret
        assert bound.per_segment_bound(10) < bound.per_segment_bound(100)

    def test_to_latex(self):
        bound = RouterRegretBound()
        latex = bound.to_latex()
        assert r"\begin{theorem}" in latex
        assert r"R_T" in latex


class TestComputeEmpiricalRegret:
    def test_basic(self):
        policy = np.array([0.5, 0.6, 0.7])
        oracle = np.array([0.1, 0.2, 0.3])
        metric = compute_empirical_regret(policy, oracle)
        assert isinstance(metric, RegretMetric)
        assert metric.cumulative_regret > 0

    def test_with_costs(self):
        metric = compute_empirical_regret(
            np.array([0.5, 0.5]),
            np.array([0.5, 0.5]),
            np.array([0.7, 0.3]),
            np.array([0.0, 0.0]),
        )
        assert metric.cumulative_regret == 0.0  # same losses
        assert metric.cost_regret == 1.0  # policy used more budget


class TestOracleHindsightCost:
    def test_best_fixed_action(self):
        # 5 frames, 3 actions
        losses = np.array([
            [0.5, 0.3, 0.8],
            [0.5, 0.2, 0.7],
            [0.5, 0.4, 0.6],
            [0.5, 0.1, 0.9],
            [0.5, 0.3, 0.5],
        ])
        costs = np.array([0.0, 0.1, 0.5])
        budget = 100.0  # large budget, no constraint

        actions, total_loss = oracle_hindsight_cost(losses, costs, budget)
        assert len(actions) == 5
        # Action 1 has lowest avg loss (1.3/5 = 0.26)
        assert actions[0] == 1
        assert all(a == 1 for a in actions)

    def test_budget_constrained(self):
        losses = np.array([
            [0.5, 0.1],
            [0.5, 0.1],
        ])
        costs = np.array([0.0, 0.5])
        budget = 0.1  # too tight for action 1

        actions, _ = oracle_hindsight_cost(losses, costs, budget)
        # Should fall back to free action
        assert all(a == 0 for a in actions)


# ──────────────────────────────────────────────────────────────────────
# Convergence Analysis
# ──────────────────────────────────────────────────────────────────────

class TestBoundedAdaptationConvergence:
    def test_parameter_deviation(self):
        conv = BoundedAdaptationConvergence(
            learning_rate=1e-4,
            max_gradient_norm=10.0,
            num_adaptation_steps=10,
        )
        dev = conv.parameter_deviation_bound()
        # η · T · G = 1e-4 · 10 · 10 = 0.01
        assert np.isclose(dev, 0.01)

    def test_deviation_with_weight_decay(self):
        conv = BoundedAdaptationConvergence(
            learning_rate=1e-4,
            max_gradient_norm=10.0,
            num_adaptation_steps=100,
            weight_decay=0.01,
        )
        dev_wd = conv.parameter_deviation_bound()
        # With weight decay, deviation should be bounded
        conv_no_wd = BoundedAdaptationConvergence(
            learning_rate=1e-4,
            max_gradient_norm=10.0,
            num_adaptation_steps=100,
            weight_decay=0.0,
        )
        dev_no_wd = conv_no_wd.parameter_deviation_bound()
        assert dev_wd < dev_no_wd

    def test_source_forgetting(self):
        conv = BoundedAdaptationConvergence(
            learning_rate=1e-4,
            max_gradient_norm=10.0,
            num_adaptation_steps=10,
            loss_lipschitz=5.0,
        )
        forget = conv.source_forgetting_bound()
        # β · η · T · G = 5 · 0.01 = 0.05
        assert np.isclose(forget, 0.05)

    def test_target_improvement(self):
        conv = BoundedAdaptationConvergence(distribution_gap=0.3)
        improvement = conv.target_improvement_bound()
        assert improvement >= 0
        assert improvement <= conv.distribution_gap

    def test_optimal_tradeoff(self):
        conv = BoundedAdaptationConvergence()
        result = conv.optimal_adaptation_intensity()
        assert 'optimal_intensity' in result
        assert 'optimal_total_loss' in result
        assert 'optimal_steps' in result
        assert result['optimal_steps'] >= 1

    def test_safe_learning_rate(self):
        conv = BoundedAdaptationConvergence()
        safe_lr = conv.safe_learning_rate_bound()
        assert safe_lr > 0
        assert safe_lr < 1.0  # should be small

    def test_validate_hyperparameters(self):
        conv = BoundedAdaptationConvergence(learning_rate=1e-5)
        result = conv.validate_hyperparameters()
        assert 'lr_is_safe' in result
        assert 'forgetting_acceptable' in result

    def test_to_latex(self):
        conv = BoundedAdaptationConvergence()
        latex = conv.to_latex()
        assert r"\begin{theorem}" in latex
        assert r"\theta" in latex

    def test_more_steps_more_deviation(self):
        conv = BoundedAdaptationConvergence()
        dev_10 = conv.parameter_deviation_bound(total_steps=10)
        dev_100 = conv.parameter_deviation_bound(total_steps=100)
        assert dev_100 > dev_10


class TestBNRefreshStability:
    def test_convergence(self):
        result = bn_refresh_stability_bound(
            momentum=0.1, num_frames=100, distribution_shift_magnitude=0.5
        )
        assert result['convergence_factor'] < 0.01  # (0.9)^100 ≈ 2.66e-5
        assert result['remaining_deviation'] < 0.01

    def test_higher_momentum_faster(self):
        slow = bn_refresh_stability_bound(momentum=0.01, num_frames=50)
        fast = bn_refresh_stability_bound(momentum=0.2, num_frames=50)
        assert fast['convergence_factor'] < slow['convergence_factor']

    def test_half_life(self):
        result = bn_refresh_stability_bound(momentum=0.1)
        assert result['half_life'] > 0
        # For m=0.1: half_life = log(2)/log(1/0.9) ≈ 6.6
        assert 5 < result['half_life'] < 8

    def test_frames_to_converge(self):
        result = bn_refresh_stability_bound(
            momentum=0.1, distribution_shift_magnitude=1.0
        )
        assert result['frames_to_converge'] > 0

    def test_zero_shift(self):
        result = bn_refresh_stability_bound(distribution_shift_magnitude=0.01)
        # Very small shift → fast convergence
        assert result['frames_to_converge'] == 0 or result['remaining_deviation'] < 0.01


class TestAdaptationForgettingBound:
    def test_single_event(self):
        result = adaptation_forgetting_bound(
            [{'steps': 10, 'learning_rate': 1e-4}],
            loss_lipschitz=5.0,
            max_gradient_norm=10.0,
        )
        assert result['total_forgetting_bound'] > 0
        bound = 5.0 * 1e-4 * 10 * 10.0  # β · η · T · G = 0.05
        assert np.isclose(result['total_forgetting_bound'], bound)
        assert result['recommendation'] == 'safe'

    def test_multiple_events(self):
        events = [
            {'steps': 10, 'learning_rate': 1e-4},
            {'steps': 20, 'learning_rate': 1e-3},
            {'steps': 5, 'learning_rate': 1e-4},
        ]
        result = adaptation_forgetting_bound(events, loss_lipschitz=5.0)
        assert result['num_events'] == 3
        assert result['total_forgetting_bound'] > result['max_single_event_forgetting']

    def test_dangerous_forgetting(self):
        # Large learning rate, many steps → danger
        result = adaptation_forgetting_bound(
            [{'steps': 100, 'learning_rate': 0.01}],
            loss_lipschitz=5.0,
            max_gradient_norm=10.0,
        )
        assert result['total_forgetting_bound'] > 0.3
        assert result['recommendation'] == 'danger: consider reducing lr or steps'

    def test_empty_events(self):
        result = adaptation_forgetting_bound([])
        assert result['total_forgetting_bound'] == 0.0
        assert result['num_events'] == 0
