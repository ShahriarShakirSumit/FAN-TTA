"""
Formal Problem Formulation for Signal-Driven Model Refresh (SDMR).

Defines the mathematical framework for streaming test-time adaptation
for object detection, as presented in Section 3 of the NeurIPS paper.

─────────────────────────────────────────────────────────────────────────
PROBLEM DEFINITION (Streaming TTA for Object Detection)
─────────────────────────────────────────────────────────────────────────

Given:
    - Source-trained detector f_θ₀ : X → Y where θ₀ = argmin_{θ} E_{(x,y)~P_S}[L(f_θ(x), y)]
    - A non-stationary test stream S = {x_t}_{t=1}^{T} where x_t ~ P_t
    - P_t may differ from P_S (distribution shift) and P_t ≠ P_{t'} in general
    - No access to ground-truth labels at test time
    - A finite set of adaptation actions A = {a_0, a_1, ..., a_K}
      with costs c(a_k) ∈ [0, 1]

Goal:
    Find an adaptation policy π : H_t → A that maps the observation
    history H_t = {(x_1, ŷ_1, s_1), ..., (x_t, ŷ_t, s_t)} to an action,
    minimizing the cumulative streaming loss while respecting a
    computational budget constraint:

    min_π  Σ_{t=1}^{T} L_det(f_{θ_t}(x_t), y_t)
    s.t.   Σ_{t=1}^{T} c(π(H_t)) ≤ B

    where:
    - L_det is the detection loss (1 - mAP@50 at frame t)
    - s_t = φ(x_t, ŷ_t, θ_t) is the diagnostic signal vector
    - θ_t is updated as θ_t = Update(θ_{t-1}, a_t, x_t) where a_t = π(H_t)
    - B is the total compute budget over T frames

Key Challenge:
    The policy π must decide WHAT to do (which action) and WHEN to do it
    (at which frames), using ONLY unlabeled test data and model signals,
    under a non-stationary distribution that may shift abruptly or gradually.

Innovation:
    SDMR decomposes π into:
    1. Signal extraction: s_t = φ(x_t, ŷ_t, θ_t) — multi-modal diagnosis
    2. Shift detection: d_t = Detect(s_{1:t}) ∈ {0, 1} — change point detection
    3. Shift characterization: (τ_t, σ_t) = Characterize(s_{1:t}) — type & severity
    4. Action selection: a_t = Route(τ_t, σ_t, b_remaining) — cost-aware routing

─────────────────────────────────────────────────────────────────────────
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import IntEnum


@dataclass
class StreamingTTAProblem:
    """
    Formal specification of the streaming TTA problem for object detection.

    This class encodes the mathematical problem formulation that goes into
    Section 3 of the NeurIPS paper. It defines the spaces, objective, and
    constraints.

    Paper notation mapping:
        T         → num_frames
        X         → input_space (images)
        Y         → output_space (detections: boxes, classes, scores)
        P_S       → source_distribution
        P_t       → target_distribution_at_t (non-stationary)
        A         → action_space (K+1 actions)
        c(a_k)    → action_costs[k]
        B         → total_budget
        π         → adaptation_policy
        φ         → signal_extractor
        L_det     → detection_loss_fn
    """

    # Problem dimensions
    num_frames: int = 1000                  # T: stream length
    num_actions: int = 5                    # |A|: number of available actions
    signal_dim: int = 48                    # d: dimension of signal vector s_t

    # Action costs c(a_k) ∈ [0, 1], ordered by cost
    action_costs: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.05, 0.3, 0.7, 1.0])
    )

    # Budget constraint
    total_budget: float = 50.0              # B: total compute budget
    budget_per_frame: float = 0.05          # B/T: average per-frame budget

    # Distribution shift parameters
    num_shift_types: int = 4                # number of distinct shift categories
    max_severity: float = 1.0               # maximum shift severity

    # Detection-specific parameters
    num_classes: int = 1                    # number of object classes (person)
    iou_threshold: float = 0.5             # IoU threshold for mAP

    def __post_init__(self):
        """Validate problem specification."""
        assert len(self.action_costs) == self.num_actions, \
            f"action_costs length ({len(self.action_costs)}) != num_actions ({self.num_actions})"
        assert np.all(self.action_costs >= 0) and np.all(self.action_costs <= 1), \
            "action_costs must be in [0, 1]"
        assert self.total_budget > 0, "total_budget must be positive"
        assert np.isclose(self.action_costs[0], 0.0), \
            "First action (NONE) must have zero cost"

    @property
    def action_labels(self) -> list[str]:
        """Human-readable action labels."""
        return ["none", "bn_refresh", "head_only", "bounded_full", "reset_and_ft"]

    def compute_detection_loss(
        self,
        predictions: dict,
        ground_truth: dict,
    ) -> float:
        """
        Compute L_det(f_θ(x_t), y_t) = 1 - mAP@IoU.

        This is the per-frame detection loss used in the objective.

        Args:
            predictions: dict with 'boxes' (N,4), 'scores' (N,), 'classes' (N,)
            ground_truth: dict with 'boxes' (M,4), 'labels' (M,)

        Returns:
            loss ∈ [0, 1] where 0 = perfect detection
        """
        pred_boxes = predictions.get('boxes', np.array([]))
        pred_scores = predictions.get('scores', np.array([]))
        gt_boxes = ground_truth.get('boxes', np.array([]))

        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            return 0.0  # no objects, no error
        if len(gt_boxes) == 0:
            return 1.0 if len(pred_boxes) > 0 else 0.0
        if len(pred_boxes) == 0:
            return 1.0  # missed all objects

        # Compute AP via precision-recall
        ap = self._compute_ap(pred_boxes, pred_scores, gt_boxes)
        return 1.0 - ap

    def _compute_ap(
        self,
        pred_boxes: np.ndarray,
        pred_scores: np.ndarray,
        gt_boxes: np.ndarray,
    ) -> float:
        """Compute Average Precision at self.iou_threshold."""
        if len(pred_scores) == 0:
            return 0.0

        # Sort by confidence
        sorted_idx = np.argsort(-pred_scores)
        pred_boxes = pred_boxes[sorted_idx]

        n_gt = len(gt_boxes)
        tp = np.zeros(len(pred_boxes))
        fp = np.zeros(len(pred_boxes))
        matched = np.zeros(n_gt, dtype=bool)

        for i, pbox in enumerate(pred_boxes):
            if n_gt == 0:
                fp[i] = 1
                continue

            ious = self._compute_iou_vector(pbox, gt_boxes)
            best_gt = np.argmax(ious)

            if ious[best_gt] >= self.iou_threshold and not matched[best_gt]:
                tp[i] = 1
                matched[best_gt] = True
            else:
                fp[i] = 1

        # Cumulative precision-recall
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recall = cum_tp / n_gt
        precision = cum_tp / (cum_tp + cum_fp)

        # AP via 11-point interpolation
        ap = 0.0
        for r_thresh in np.linspace(0, 1, 11):
            prec_at_r = precision[recall >= r_thresh]
            ap += np.max(prec_at_r) / 11.0 if len(prec_at_r) > 0 else 0.0

        return ap

    @staticmethod
    def _compute_iou_vector(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Compute IoU of one box against a set of boxes."""
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        area_box = (box[2] - box[0]) * (box[3] - box[1])
        area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area_box + area_boxes - inter

        return inter / np.maximum(union, 1e-10)

    def compute_cumulative_cost(self, actions: np.ndarray) -> float:
        """
        Compute Σ_{t=1}^{T} c(a_t) for a sequence of actions.

        Args:
            actions: array of action indices, shape (T,)

        Returns:
            total cost
        """
        return float(np.sum(self.action_costs[actions]))

    def is_budget_feasible(self, actions: np.ndarray) -> bool:
        """Check if an action sequence respects the budget constraint."""
        return self.compute_cumulative_cost(actions) <= self.total_budget

    def compute_objective(
        self,
        losses: np.ndarray,
        actions: np.ndarray,
        lambda_cost: float = 0.0,
    ) -> float:
        """
        Compute the Lagrangian objective:
            J(π) = (1/T) Σ L_det(t) + λ · max(0, Σ c(a_t) - B)

        Args:
            losses: per-frame detection losses, shape (T,)
            actions: per-frame action indices, shape (T,)
            lambda_cost: Lagrange multiplier for budget constraint

        Returns:
            objective value
        """
        avg_loss = np.mean(losses)
        cost_violation = max(0, self.compute_cumulative_cost(actions) - self.total_budget)
        return avg_loss + lambda_cost * cost_violation


@dataclass
class AdaptationPolicy:
    """
    Abstract adaptation policy π : H_t → A.

    In the paper, this is instantiated as:
    - π_rule: Rule-based router (interpretable, no training needed)
    - π_learn: Learned router (XGBoost on signal features)
    - π_cost: Cost-aware wrapper (budget-constrained)
    """

    name: str = "abstract_policy"

    def select_action(
        self,
        signal_vector: np.ndarray,
        shift_detected: bool,
        shift_severity: float,
        budget_remaining: float,
        step: int,
    ) -> int:
        """
        Select an action given the current state.

        Args:
            signal_vector: s_t ∈ R^d, the diagnostic signal vector
            shift_detected: d_t ∈ {0, 1}, whether change point detected
            shift_severity: σ_t ∈ [0, 1], estimated severity
            budget_remaining: B - Σ_{i<t} c(a_i)
            step: current time step t

        Returns:
            action index a_t ∈ {0, 1, ..., K}
        """
        raise NotImplementedError


@dataclass
class ShiftDetector:
    """
    Formal specification of the change point detection sub-problem.

    Given signal stream s_{1:t}, detect shift onset d_t ∈ {0, 1}.

    Theorem (Detection Delay Bound):
        Under a shift with magnitude Δ in signal space, the multi-signal
        CUSUM detector identifies the shift within
            delay ≤ O(σ² · log(1/α) / Δ²)
        frames with probability ≥ 1 - α, where σ² is the signal variance
        under the null (no-shift) hypothesis.

    This is proved formally in regret_analysis.py.
    """

    # Detector parameters
    signal_variance: float = 1.0            # σ²: variance under null
    shift_magnitude: float = 0.5            # Δ: minimum detectable shift
    false_alarm_rate: float = 0.05          # α: desired false alarm rate

    @property
    def theoretical_delay_bound(self) -> float:
        """
        Compute the theoretical detection delay bound.

        delay ≤ σ² · log(1/α) / Δ²

        This appears as Proposition 1 in the paper.
        """
        return (
            self.signal_variance
            * np.log(1.0 / self.false_alarm_rate)
            / max(self.shift_magnitude ** 2, 1e-10)
        )

    @property
    def expected_detection_delay(self) -> float:
        """
        Expected detection delay (ADD - Average Detection Delay).

        E[delay] = σ² · log(1/α) / Δ² · (1 + o(1))

        Follows from the Wald identity for sequential tests.
        """
        return self.theoretical_delay_bound

    def compute_sample_complexity(self, confidence: float = 0.95) -> int:
        """
        Minimum samples needed to detect a shift of magnitude Δ
        with probability ≥ confidence.

        n ≥ 2σ² · log(2/δ) / Δ²  where δ = 1 - confidence

        This follows from Hoeffding's inequality applied to the
        cumulative sum test statistic.
        """
        delta = 1.0 - confidence
        n = 2.0 * self.signal_variance * np.log(2.0 / delta) / max(self.shift_magnitude ** 2, 1e-10)
        return int(np.ceil(n))


@dataclass
class RegretMetric:
    """
    Regret metric comparing policy π against the oracle policy π*.

    Definition (Regret):
        R_T(π) = Σ_{t=1}^{T} L_det(f_{θ_t^π}(x_t), y_t) - Σ_{t=1}^{T} L_det(f_{θ_t^{π*}}(x_t), y_t)

    where π* is the oracle policy that knows the true shift schedule.

    The regret captures how much worse the online policy performs compared
    to the best possible policy with full knowledge.
    """

    policy_losses: np.ndarray = field(default_factory=lambda: np.array([]))  # L_det under π
    oracle_losses: np.ndarray = field(default_factory=lambda: np.array([]))  # L_det under π*
    policy_costs: np.ndarray = field(default_factory=lambda: np.array([]))   # c(a_t) under π
    oracle_costs: np.ndarray = field(default_factory=lambda: np.array([]))   # c(a_t) under π*

    @property
    def cumulative_regret(self) -> float:
        """R_T(π) = Σ L_π(t) - Σ L_π*(t)"""
        if len(self.policy_losses) == 0:
            return 0.0
        return float(np.sum(self.policy_losses) - np.sum(self.oracle_losses))

    @property
    def average_regret(self) -> float:
        """R_T(π) / T"""
        T = len(self.policy_losses)
        return self.cumulative_regret / max(T, 1)

    @property
    def regret_trajectory(self) -> np.ndarray:
        """Cumulative regret over time: R_t for t = 1, ..., T."""
        if len(self.policy_losses) == 0:
            return np.array([])
        per_step = self.policy_losses - self.oracle_losses
        return np.cumsum(per_step)

    @property
    def cost_regret(self) -> float:
        """Additional cost incurred compared to oracle."""
        if len(self.policy_costs) == 0:
            return 0.0
        return float(np.sum(self.policy_costs) - np.sum(self.oracle_costs))

    @property
    def normalized_regret(self) -> float:
        """
        Regret normalized by T, suitable for comparing across stream lengths.
        Should approach 0 as T → ∞ for a sublinear-regret policy.
        """
        return self.average_regret

    def is_sublinear(self, tolerance: float = 0.01) -> bool:
        """
        Check if regret growth is sublinear in T.

        Fits R_t ∝ t^α and checks α < 1.
        A good policy should have α ≈ 0.5 (√T regret).
        """
        traj = self.regret_trajectory
        if len(traj) < 10:
            return True  # too short to tell

        t = np.arange(1, len(traj) + 1, dtype=float)
        # Fit log(R_t) = α·log(t) + c in least squares
        # Only use positive regret values
        positive_mask = traj > 0
        if np.sum(positive_mask) < 5:
            return True

        log_t = np.log(t[positive_mask])
        log_r = np.log(traj[positive_mask])

        # Linear regression: log_r = α * log_t + c
        A = np.column_stack([log_t, np.ones_like(log_t)])
        result = np.linalg.lstsq(A, log_r, rcond=None)
        alpha = result[0][0]

        return alpha < (1.0 - tolerance)
