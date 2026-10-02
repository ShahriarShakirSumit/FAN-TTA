"""
Convergence Guarantees for SDMR Bounded Adaptation.

Provides formal analysis of:
1. How bounded adaptation preserves source model performance
2. How BN refresh maintains feature alignment
3. The forgetting-adaptation tradeoff

─────────────────────────────────────────────────────────────────────────
MAIN THEORETICAL RESULTS
─────────────────────────────────────────────────────────────────────────

Theorem 4 (Bounded Adaptation Stability):
    Under bounded adaptation with learning rate η and T adaptation steps,
    the parameter deviation from source is bounded:
        ‖θ_T - θ_0‖ ≤ η · T · G_max
    
    where G_max is the maximum gradient norm. The in-distribution loss
    increase is bounded by:
        L_S(θ_T) - L_S(θ_0) ≤ β · η · T · G_max
    
    where β is the Lipschitz constant of the source loss landscape.

Theorem 5 (BN Refresh Stability):
    BN refresh with exponential moving average (momentum m) satisfies:
        ‖μ_t - μ_S‖ ≤ (1-m)^t · ‖μ_0 - μ_S‖ + m · Δ_dist
    
    where Δ_dist is the distribution shift magnitude. Under stationary
    target distribution, BN stats converge to the target distribution
    moments at rate O((1-m)^t).

Proposition 2 (Forgetting-Adaptation Tradeoff):
    For any adaptation policy with cumulative adaptation intensity I_T,
    the source performance degradation and target improvement satisfy:
        ΔL_S ≤ β · I_T     (forgetting)
        ΔL_T ≤ L_gap - γ · I_T  (adaptation gain)
    
    The optimal intensity is I* = L_gap / (β + γ), achieving:
        L_total* = L_gap · β / (β + γ)

─────────────────────────────────────────────────────────────────────────
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class BoundedAdaptationConvergence:
    """
    Convergence analysis for bounded model adaptation.

    Provides computable bounds on parameter deviation, forgetting,
    and adaptation effectiveness.
    """

    # Adaptation parameters
    learning_rate: float = 1e-4         # η: adaptation learning rate
    max_gradient_norm: float = 10.0     # G_max: max gradient norm (clipped)
    num_adaptation_steps: int = 10      # number of gradient steps per adaptation
    weight_decay: float = 0.0           # regularization strength

    # Source model properties
    loss_lipschitz: float = 5.0         # β: Lipschitz constant of L_S w.r.t. θ
    loss_smoothness: float = 10.0       # L: smoothness constant (Hessian bound)

    # Distribution shift
    distribution_gap: float = 0.3       # L_gap: loss gap between source and target

    def parameter_deviation_bound(self, total_steps: int = None) -> float:
        """
        Theorem 4, Part 1: Parameter deviation bound.

        ‖θ_T - θ_0‖ ≤ η · T_total · G_max

        where T_total is the total number of gradient steps taken.

        With weight decay λ, the bound tightens to:
            ‖θ_T - θ_0‖ ≤ (η · G_max / λ) · (1 - (1-λ·η)^T)
        """
        T = total_steps or self.num_adaptation_steps
        eta = self.learning_rate
        G = self.max_gradient_norm

        if self.weight_decay > 0:
            lam = self.weight_decay
            return (eta * G / lam) * (1 - (1 - lam * eta) ** T)
        else:
            return eta * T * G

    def source_forgetting_bound(self, total_steps: int = None) -> float:
        """
        Theorem 4, Part 2: Source performance degradation bound.

        L_S(θ_T) - L_S(θ_0) ≤ β · ‖θ_T - θ_0‖ ≤ β · η · T · G_max

        By Lipschitz continuity of the source loss.
        """
        param_dev = self.parameter_deviation_bound(total_steps)
        return self.loss_lipschitz * param_dev

    def target_improvement_bound(self, total_steps: int = None) -> float:
        """
        Target loss improvement bound (best case).

        Under convexity assumptions and step size η ≤ 1/L:
            L_T(θ_T) ≤ L_T(θ_0) - (η/2) · T · ‖∇L_T‖² + (L·η²·T·G²)/2

        Simplified: improvement ≈ L_gap · (1 - exp(-γ·T·η))
        where γ depends on the gradient signal quality.
        """
        T = total_steps or self.num_adaptation_steps
        # Approximate improvement rate
        gamma = 0.1  # conservative adaptation rate
        improvement = self.distribution_gap * (1 - np.exp(-gamma * T * self.learning_rate * 1000))
        return min(improvement, self.distribution_gap)

    def optimal_adaptation_intensity(self) -> dict:
        """
        Proposition 2: Optimal forgetting-adaptation tradeoff.

        Finds I* = argmin_I [β·I + max(0, L_gap - γ·I)]

        The optimal balance point minimizes total loss across both
        source and target distributions.
        """
        beta = self.loss_lipschitz  # forgetting rate
        gamma = 1.0                 # adaptation gain rate (normalized)
        L_gap = self.distribution_gap

        I_star = L_gap / (beta + gamma)
        L_total_star = L_gap * beta / (beta + gamma)

        # Convert intensity to concrete steps
        steps_star = I_star / (self.learning_rate * self.max_gradient_norm)

        return {
            'optimal_intensity': I_star,
            'optimal_total_loss': L_total_star,
            'optimal_steps': max(1, int(np.ceil(steps_star))),
            'forgetting_at_optimum': beta * I_star,
            'remaining_gap': max(0, L_gap - gamma * I_star),
        }

    def safe_learning_rate_bound(self) -> float:
        """
        Maximum learning rate that guarantees stable adaptation.

        For convex losses: η ≤ 1/L (smoothness)
        For bounded gradients: η ≤ ε / (T · G_max · β)
          where ε is the tolerable source performance drop.
        """
        # Smoothness-based bound
        smooth_bound = 1.0 / self.loss_smoothness

        # Forgetting-based bound (tolerate 5% source performance drop)
        eps = 0.05
        T = self.num_adaptation_steps
        forget_bound = eps / (T * self.max_gradient_norm * self.loss_lipschitz)

        return min(smooth_bound, forget_bound)

    def validate_hyperparameters(self) -> dict:
        """
        Check if current hyperparameters satisfy theoretical requirements.
        """
        safe_lr = self.safe_learning_rate_bound()
        forgetting = self.source_forgetting_bound()
        param_dev = self.parameter_deviation_bound()

        return {
            'learning_rate': self.learning_rate,
            'safe_lr_bound': safe_lr,
            'lr_is_safe': self.learning_rate <= safe_lr,
            'parameter_deviation': param_dev,
            'source_forgetting': forgetting,
            'forgetting_acceptable': forgetting < 0.1,  # < 10% degradation
            'optimal_tradeoff': self.optimal_adaptation_intensity(),
        }

    def to_latex(self) -> str:
        """Generate LaTeX for the convergence theorem."""
        return (
            r"\begin{theorem}[Bounded Adaptation Stability]" "\n"
            r"Under gradient-bounded adaptation with learning rate $\eta$ "
            r"and gradient norm bound $G_{\max}$, after $T$ adaptation steps:" "\n"
            r"\begin{enumerate}" "\n"
            r"\item Parameter deviation: $\|\theta_T - \theta_0\| \leq \eta \cdot T \cdot G_{\max}$" "\n"
            r"\item Source forgetting: $\mathcal{L}_S(\theta_T) - \mathcal{L}_S(\theta_0) "
            r"\leq \beta \cdot \eta \cdot T \cdot G_{\max}$" "\n"
            r"\item Safe learning rate: $\eta \leq \min\left(\frac{1}{L}, "
            r"\frac{\epsilon}{\beta \cdot T \cdot G_{\max}}\right)$" "\n"
            r"\end{enumerate}" "\n"
            r"\end{theorem}"
        )


def bn_refresh_stability_bound(
    momentum: float = 0.1,
    num_frames: int = 32,
    distribution_shift_magnitude: float = 0.5,
    initial_bn_deviation: float = 0.0,
) -> dict:
    """
    Theorem 5: BN Refresh Stability Analysis.

    For BN with exponential moving average (EMA momentum m):
        μ_t = (1-m) · μ_{t-1} + m · μ_batch

    Under a target distribution with mean μ_T:
        ‖μ_t - μ_T‖ ≤ (1-m)^t · ‖μ_0 - μ_T‖

    Convergence to target stats is exponential with rate (1-m).

    The tradeoff:
    - High m → fast convergence to target but noisy estimates
    - Low m → smooth estimates but slow adaptation
    - Optimal m depends on batch size and shift magnitude

    Args:
        momentum: BN EMA momentum m
        num_frames: number of frames for BN refresh
        distribution_shift_magnitude: ‖μ_S - μ_T‖
        initial_bn_deviation: current deviation from source stats

    Returns:
        dict with convergence analysis
    """
    m = momentum
    t = num_frames
    delta = distribution_shift_magnitude

    # Convergence of BN stats to target
    convergence_factor = (1 - m) ** t
    remaining_deviation = convergence_factor * (initial_bn_deviation + delta)

    # Estimation noise (due to finite batch size)
    # Var[μ_batch] ≈ σ² / batch_size, accumulated through EMA
    # variance of EMA: Var[μ_t] ≈ m / (2 - m) · σ²/B
    effective_variance_factor = m / (2 - m)

    # Number of frames to reach within ε of target
    epsilon = 0.05
    if convergence_factor > 0 and delta > epsilon:
        frames_to_converge = int(np.ceil(
            np.log(epsilon / max(delta, 1e-10)) / np.log(max(1 - m, 1e-10))
        ))
    else:
        frames_to_converge = 0

    return {
        'convergence_factor': convergence_factor,
        'remaining_deviation': remaining_deviation,
        'effective_variance_factor': effective_variance_factor,
        'frames_to_converge': frames_to_converge,
        'exponential_rate': -np.log(max(1 - m, 1e-10)),
        'half_life': np.log(2) / max(-np.log(max(1 - m, 1e-10)), 1e-10),
    }


def adaptation_forgetting_bound(
    adaptation_events: list[dict],
    loss_lipschitz: float = 5.0,
    learning_rate: float = 1e-4,
    max_gradient_norm: float = 10.0,
) -> dict:
    """
    Proposition 2: Cumulative forgetting from multiple adaptation events.

    Given a sequence of adaptation events, compute the total source
    performance degradation bound.

    The forgetting is additive across events but can be partially
    recovered by model reset:
        ΔL_S ≤ Σ_i β · η_i · T_i · G_max  (without reset)
        ΔL_S ≤ max_i β · η_i · T_i · G_max  (with periodic reset)

    Args:
        adaptation_events: list of dicts with 'steps' and 'learning_rate'
        loss_lipschitz: Lipschitz constant β
        learning_rate: default learning rate
        max_gradient_norm: gradient clipping norm

    Returns:
        dict with forgetting analysis
    """
    total_forgetting = 0.0
    max_single_event = 0.0
    event_details = []

    for event in adaptation_events:
        steps = event.get('steps', 1)
        lr = event.get('learning_rate', learning_rate)
        G = event.get('gradient_norm', max_gradient_norm)

        event_forgetting = loss_lipschitz * lr * steps * G
        total_forgetting += event_forgetting
        max_single_event = max(max_single_event, event_forgetting)

        event_details.append({
            'steps': steps,
            'learning_rate': lr,
            'forgetting_bound': event_forgetting,
        })

    return {
        'total_forgetting_bound': total_forgetting,
        'max_single_event_forgetting': max_single_event,
        'num_events': len(adaptation_events),
        'average_forgetting_per_event': total_forgetting / max(len(adaptation_events), 1),
        'event_details': event_details,
        'recommendation': (
            'safe' if total_forgetting < 0.1 else
            'caution' if total_forgetting < 0.3 else
            'danger: consider reducing lr or steps'
        ),
    }
