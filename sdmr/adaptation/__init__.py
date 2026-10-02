"""
SDMR Adaptation Router — the core contribution.

Implements the Signal-Driven Model Refresh policy that:
1. Fuses diagnosis signals (input, output, feature) into a state vector
2. Decides WHEN to adapt (shift detection → trigger)
3. Decides HOW to adapt (which adaptation action to take)
4. Decides HOW MUCH to adapt (adaptation intensity / budget)

The router is the "brain" of SDMR: it replaces naive periodic or
threshold-based adaptation with a learned/rule-based policy that
considers the shift type, severity, and model state.

Adaptation actions (ordered by cost):
1. "none"         — no action, model is fine
2. "bn_refresh"   — refresh batch-norm running stats only (cheapest)
3. "head_only"    — fine-tune detection head only
4. "bounded_full" — update all params with bounded step size
5. "reset_and_ft" — reset to source checkpoint + fine-tune (most expensive)

Architecture:
    DiagnosisSignals → [TemporalBuffer → ChangePointDetector] → Router → Action
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from enum import IntEnum


class AdaptationAction(IntEnum):
    """Adaptation actions ordered by computational cost."""
    NONE = 0
    BN_REFRESH = 1
    HEAD_ONLY = 2
    BOUNDED_FULL = 3
    RESET_AND_FT = 4

    @property
    def label(self) -> str:
        return self.name.lower()

    @property
    def relative_cost(self) -> float:
        """Approximate relative cost (0-1 scale)."""
        return {
            0: 0.0,
            1: 0.05,
            2: 0.3,
            3: 0.7,
            4: 1.0,
        }[self.value]


@dataclass
class RouterDecision:
    """Output of the adaptation router."""
    action: AdaptationAction = AdaptationAction.NONE
    confidence: float = 0.0
    severity: float = 0.0
    shift_type_guess: str = "unknown"  # "input", "semantic", "geometric", "mixed"
    adaptation_budget: int = 1         # num gradient steps (if applicable)
    learning_rate_scale: float = 1.0   # LR multiplier
    reasoning: str = ""                # human-readable explanation
    signal_snapshot: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "action": self.action.label,
            "action_cost": self.action.relative_cost,
            "confidence": self.confidence,
            "severity": self.severity,
            "shift_type": self.shift_type_guess,
            "budget": self.adaptation_budget,
            "lr_scale": self.learning_rate_scale,
            "reasoning": self.reasoning,
        }


class RuleBasedRouter:
    """
    Rule-based adaptation router using interpretable signal thresholds.

    This is the baseline router; a learned router (via XGBoost or small MLP)
    can be trained to replace it. The rule-based version is:
    - Fully interpretable (paper contribution: ablation-friendly)
    - No training required
    - Competitive with learned policies for well-tuned thresholds

    Decision logic:
    1. Check if change-point detector fired → no shift = no action
    2. Classify shift type from signal profiles
    3. Map (shift_type, severity) → action
    """

    def __init__(
        self,
        # Signal thresholds for shift classification
        input_drift_threshold: float = 0.3,
        output_drift_threshold: float = 0.3,
        feature_drift_threshold: float = 0.3,
        # Severity thresholds for action escalation
        mild_severity: float = 0.3,
        moderate_severity: float = 0.6,
        severe_threshold: float = 0.8,
        # Budget settings
        base_budget: int = 5,
        max_budget: int = 50,
        # Cooldown
        cooldown_frames: int = 10,
        # Cost awareness
        cost_penalty: float = 0.1,
    ):
        self.input_thresh = input_drift_threshold
        self.output_thresh = output_drift_threshold
        self.feature_thresh = feature_drift_threshold
        self.mild = mild_severity
        self.moderate = moderate_severity
        self.severe = severe_threshold
        self.base_budget = base_budget
        self.max_budget = max_budget
        self.cooldown_frames = cooldown_frames
        self.cost_penalty = cost_penalty

        self._last_action_frame = -999
        self._step = 0
        self._action_history: list[AdaptationAction] = []

    def decide(
        self,
        shift_detected: bool,
        shift_severity: float,
        shift_persistence: float,
        input_drift_score: float,
        output_drift_score: float,
        feature_drift_score: float,
        signal_snapshot: Optional[dict[str, float]] = None,
    ) -> RouterDecision:
        """
        Make an adaptation decision based on current diagnosis signals.

        Args:
            shift_detected: Whether the change-point detector fired.
            shift_severity: Severity from change-point detector [0,1].
            shift_persistence: Persistence score [0,1].
            input_drift_score: Aggregated input-space drift magnitude.
            output_drift_score: Aggregated output-space drift magnitude.
            feature_drift_score: Aggregated feature-space drift magnitude.
            signal_snapshot: Full signal dict for logging.

        Returns:
            RouterDecision with action, budget, and reasoning.
        """
        self._step += 1
        snapshot = signal_snapshot or {}

        # Check cooldown
        frames_since_last = self._step - self._last_action_frame
        if frames_since_last < self.cooldown_frames and not shift_detected:
            return RouterDecision(
                action=AdaptationAction.NONE,
                confidence=0.9,
                reasoning="In cooldown period",
                signal_snapshot=snapshot,
            )

        # Also detect shift from drift scores (soft detection)
        # This supplements the formal change-point detector
        drift_detected = (
            input_drift_score > self.input_thresh
            or output_drift_score > self.output_thresh
            or feature_drift_score > self.feature_thresh
        )
        effective_shift = shift_detected or drift_detected

        # No shift detected by any method → no action
        if not effective_shift:
            return RouterDecision(
                action=AdaptationAction.NONE,
                confidence=0.8,
                reasoning="No shift detected",
                signal_snapshot=snapshot,
            )

        # If only drift-detected (no formal change-point), use drift-based severity
        if not shift_detected and drift_detected:
            # Normalize z-score drift to [0, 1] severity using tanh-like mapping
            # z=0.5 → sev≈0.15, z=1.0 → sev≈0.3, z=2.0 → sev≈0.54, z=5.0 → sev≈0.81
            max_drift = max(input_drift_score, output_drift_score, feature_drift_score)
            shift_severity = min(1.0, max_drift / (max_drift + 3.0))
            shift_persistence = 0.3  # assume low persistence for drift-only

        # Classify shift type
        shift_type = self._classify_shift(
            input_drift_score, output_drift_score, feature_drift_score
        )

        # Determine action based on shift type and severity
        action, reasoning = self._select_action(
            shift_type, shift_severity, shift_persistence,
            input_drift_score, output_drift_score, feature_drift_score,
        )

        # Determine adaptation budget
        budget = self._compute_budget(shift_severity, action)

        # Learning rate scaling based on severity
        lr_scale = self._compute_lr_scale(shift_severity, action)

        self._last_action_frame = self._step
        self._action_history.append(action)

        return RouterDecision(
            action=action,
            confidence=min(1.0, shift_severity + 0.3),
            severity=shift_severity,
            shift_type_guess=shift_type,
            adaptation_budget=budget,
            learning_rate_scale=lr_scale,
            reasoning=reasoning,
            signal_snapshot=snapshot,
        )

    def _classify_shift(
        self,
        input_score: float,
        output_score: float,
        feature_score: float,
    ) -> str:
        """Classify the shift type from signal profiles."""
        # Dominant signal determines type
        scores = {
            "input": input_score,
            "semantic": output_score,
            "feature": feature_score,
        }

        # Check if multiple signals are high → mixed
        high_signals = sum(
            1 for v in scores.values() if v > self.input_thresh
        )
        if high_signals >= 2:
            return "mixed"

        dominant = max(scores, key=scores.get)  # type: ignore
        if scores[dominant] < self.input_thresh:
            return "unknown"
        return dominant

    def _select_action(
        self,
        shift_type: str,
        severity: float,
        persistence: float,
        input_drift: float,
        output_drift: float,
        feature_drift: float,
    ) -> tuple[AdaptationAction, str]:
        """Map (shift_type, severity) to an adaptation action."""

        # Pure input shift (brightness, contrast, etc.)
        # → BN refresh is usually sufficient and cheapest
        if shift_type == "input" and severity < self.moderate:
            return (
                AdaptationAction.BN_REFRESH,
                f"Input-only shift (sev={severity:.2f}): BN refresh sufficient"
            )

        # Moderate input shift
        if shift_type == "input" and severity < self.severe:
            return (
                AdaptationAction.HEAD_ONLY,
                f"Moderate input shift (sev={severity:.2f}): head fine-tune"
            )

        # Semantic/output shift → need at least head update
        if shift_type == "semantic" and severity < self.moderate:
            return (
                AdaptationAction.HEAD_ONLY,
                f"Semantic shift detected (sev={severity:.2f}): head fine-tune"
            )

        # Feature-level shift → deeper adaptation needed
        if shift_type == "feature" and severity < self.severe:
            return (
                AdaptationAction.BOUNDED_FULL,
                f"Feature-level shift (sev={severity:.2f}): bounded full update"
            )

        # Severe shift of any type → bounded full
        if severity >= self.severe and persistence < 0.8:
            return (
                AdaptationAction.BOUNDED_FULL,
                f"Severe shift (sev={severity:.2f}, type={shift_type}): bounded full"
            )

        # Severe + persistent shift → reset and retrain
        if severity >= self.severe and persistence >= 0.8:
            return (
                AdaptationAction.RESET_AND_FT,
                f"Severe persistent shift (sev={severity:.2f}): reset + fine-tune"
            )

        # Mixed shift → bounded full as safe default
        if shift_type == "mixed":
            return (
                AdaptationAction.BOUNDED_FULL,
                f"Mixed shift (sev={severity:.2f}): bounded full update"
            )

        # Default: head-only as conservative choice
        return (
            AdaptationAction.HEAD_ONLY,
            f"Default action for {shift_type} shift (sev={severity:.2f})"
        )

    def _compute_budget(
        self, severity: float, action: AdaptationAction
    ) -> int:
        """Compute adaptation budget (gradient steps) based on severity."""
        if action == AdaptationAction.NONE:
            return 0
        if action == AdaptationAction.BN_REFRESH:
            return max(5, int(self.base_budget * (1 + severity)))

        # Scale budget with severity
        budget = int(self.base_budget * (1 + severity * 3))
        budget = min(budget, self.max_budget)
        return budget

    def _compute_lr_scale(
        self, severity: float, action: AdaptationAction
    ) -> float:
        """Compute LR scale — lower for mild shifts, higher for severe."""
        if action in (AdaptationAction.NONE, AdaptationAction.BN_REFRESH):
            return 0.0

        # Linearly scale LR with severity
        scale = 0.1 + 0.9 * severity
        # Cap for safety
        return min(scale, 2.0)

    def get_action_distribution(self) -> dict[str, int]:
        """Return histogram of actions taken so far."""
        dist: dict[str, int] = {}
        for a in AdaptationAction:
            dist[a.label] = sum(1 for h in self._action_history if h == a)
        return dist

    def reset(self) -> None:
        self._last_action_frame = -999
        self._step = 0
        self._action_history.clear()


class LearnedRouter:
    """
    Learned adaptation router using gradient-boosted trees (XGBoost).

    Trained from logged signal-action-outcome tuples to learn the
    mapping: signal_vector → best_action.

    This provides the "upper bound" on routing quality when enough
    training data is available, and serves as ablation vs rule-based.
    """

    def __init__(
        self,
        signal_names: list[str],
        n_actions: int = 5,
        model_path: Optional[str] = None,
    ):
        self.signal_names = signal_names
        self.n_actions = n_actions
        self.model = None
        self._trained = False

        if model_path:
            self.load(model_path)

    def train(
        self,
        signal_matrix: np.ndarray,
        action_labels: np.ndarray,
        outcome_rewards: np.ndarray,
    ) -> dict[str, float]:
        """
        Train the router from logged data.

        Args:
            signal_matrix: (N, D) signal vectors
            action_labels: (N,) int action indices
            outcome_rewards: (N,) reward values (higher = better outcome)

        Returns:
            Training metrics dict.
        """
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost required for LearnedRouter. pip install xgboost")

        # Weight samples by outcome reward
        sample_weights = np.maximum(outcome_rewards, 0.1)

        dtrain = xgb.DMatrix(
            signal_matrix,
            label=action_labels,
            weight=sample_weights,
        )

        params = {
            "objective": "multi:softprob",
            "num_class": self.n_actions,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "mlogloss",
            "seed": 42,
        }

        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=200,
            verbose_eval=False,
        )
        self._trained = True

        # Compute training accuracy
        preds = self.model.predict(dtrain)
        pred_labels = np.argmax(preds, axis=1)
        accuracy = float(np.mean(pred_labels == action_labels))

        return {"accuracy": accuracy, "n_samples": len(action_labels)}

    def decide(
        self,
        signal_vector: np.ndarray,
        signal_snapshot: Optional[dict[str, float]] = None,
    ) -> RouterDecision:
        """
        Make a decision using the learned model.

        Args:
            signal_vector: 1D array of diagnosis signals.
            signal_snapshot: Optional full signal dict for logging.
        """
        if not self._trained or self.model is None:
            return RouterDecision(
                action=AdaptationAction.NONE,
                confidence=0.0,
                reasoning="Router not trained yet",
            )

        try:
            import xgboost as xgb
        except ImportError:
            return RouterDecision(
                action=AdaptationAction.NONE,
                reasoning="XGBoost not available",
            )

        dtest = xgb.DMatrix(signal_vector.reshape(1, -1))
        probs = self.model.predict(dtest)[0]
        action_idx = int(np.argmax(probs))
        confidence = float(probs[action_idx])

        action = AdaptationAction(action_idx)

        return RouterDecision(
            action=action,
            confidence=confidence,
            severity=0.0,  # not directly available from learned model
            reasoning=f"Learned model: p={confidence:.2f} for {action.label}",
            signal_snapshot=signal_snapshot or {},
        )

    def save(self, path: str) -> None:
        if self.model is not None:
            self.model.save_model(path)

    def load(self, path: str) -> None:
        try:
            import xgboost as xgb
            self.model = xgb.Booster()
            self.model.load_model(path)
            self._trained = True
        except Exception:
            self._trained = False


class CostAwareRouter:
    """
    Wrapper that adds cost-awareness to any base router.

    Applies a cost penalty to more expensive actions and
    tracks cumulative adaptation budget.

    This implements the "cost-aware" ablation variant from
    the SDMR paper: does explicit cost-benefit matter?
    """

    def __init__(
        self,
        base_router: RuleBasedRouter,
        budget_per_window: float = 10.0,
        cost_weight: float = 0.3,
        window_size: int = 100,
    ):
        self.base_router = base_router
        self.budget_per_window = budget_per_window
        self.cost_weight = cost_weight
        self.window_size = window_size

        self._window_cost = 0.0
        self._step_in_window = 0

    def decide(
        self,
        shift_detected: bool,
        shift_severity: float,
        shift_persistence: float,
        input_drift_score: float,
        output_drift_score: float,
        feature_drift_score: float,
        signal_snapshot: Optional[dict[str, float]] = None,
    ) -> RouterDecision:
        """Make cost-aware decision."""
        self._step_in_window += 1

        # Reset budget window
        if self._step_in_window > self.window_size:
            self._step_in_window = 1
            self._window_cost = 0.0

        # Get base decision
        decision = self.base_router.decide(
            shift_detected=shift_detected,
            shift_severity=shift_severity,
            shift_persistence=shift_persistence,
            input_drift_score=input_drift_score,
            output_drift_score=output_drift_score,
            feature_drift_score=feature_drift_score,
            signal_snapshot=signal_snapshot,
        )

        # Check budget
        action_cost = decision.action.relative_cost
        remaining_budget = self.budget_per_window - self._window_cost

        if action_cost > remaining_budget and decision.action != AdaptationAction.NONE:
            # Downgrade action to fit budget
            decision = self._downgrade_action(decision, remaining_budget)

        # Track cost
        self._window_cost += decision.action.relative_cost
        return decision

    def _downgrade_action(
        self, decision: RouterDecision, budget: float
    ) -> RouterDecision:
        """Downgrade action to fit within budget."""
        # Try actions from current down to none
        for action in reversed(list(AdaptationAction)):
            if action.relative_cost <= budget:
                return RouterDecision(
                    action=action,
                    confidence=decision.confidence * 0.8,
                    severity=decision.severity,
                    shift_type_guess=decision.shift_type_guess,
                    adaptation_budget=max(1, decision.adaptation_budget // 2),
                    learning_rate_scale=decision.learning_rate_scale,
                    reasoning=f"Downgraded from {decision.action.label} "
                              f"to {action.label} (budget constraint)",
                    signal_snapshot=decision.signal_snapshot,
                )

        return RouterDecision(
            action=AdaptationAction.NONE,
            confidence=1.0,
            reasoning="Budget exhausted",
            signal_snapshot=decision.signal_snapshot,
        )

    def reset(self) -> None:
        self.base_router.reset()
        self._window_cost = 0.0
        self._step_in_window = 0
