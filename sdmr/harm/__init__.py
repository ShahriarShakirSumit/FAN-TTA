"""
Harm prediction module for SDMR.

Predicts the expected performance harm (AP drop) if no adaptation
is taken at the current frame. This is a key contribution:

  h(t) = E[L(θ_t, x_{t:t+H}) - L(θ*, x_{t:t+H})]

where L is detection loss, θ_t is current model, θ* is oracle-adapted,
and H is the prediction horizon.

The harm predictor enables:
1. Cost-benefit routing: only adapt when harm > cost
2. Budget allocation: prioritize high-harm frames
3. Regret minimization: explicit regret estimation
"""

from .harm_predictor import (
    HarmPredictor,
    HarmEstimate,
    HistogramHarmPredictor,
    GradientHarmPredictor,
    EnsembleHarmPredictor,
)

__all__ = [
    "HarmPredictor",
    "HarmEstimate",
    "HistogramHarmPredictor",
    "GradientHarmPredictor",
    "EnsembleHarmPredictor",
]
