"""
SDMR Theoretical Foundations.

This module provides the formal mathematical framework for the SDMR paper,
including problem formulation, regret analysis, and convergence guarantees.

Modules:
- problem_formulation: Formal definition of streaming TTA for detection
- regret_analysis: Regret bounds for the adaptation router
- convergence: Convergence guarantees for bounded adaptation
"""

from .problem_formulation import (
    StreamingTTAProblem,
    AdaptationPolicy,
    ShiftDetector,
    RegretMetric,
)
from .regret_analysis import (
    RouterRegretBound,
    compute_empirical_regret,
    oracle_hindsight_cost,
)
from .convergence import (
    BoundedAdaptationConvergence,
    bn_refresh_stability_bound,
    adaptation_forgetting_bound,
)

__all__ = [
    "StreamingTTAProblem",
    "AdaptationPolicy",
    "ShiftDetector",
    "RegretMetric",
    "RouterRegretBound",
    "compute_empirical_regret",
    "oracle_hindsight_cost",
    "BoundedAdaptationConvergence",
    "bn_refresh_stability_bound",
    "adaptation_forgetting_bound",
]
