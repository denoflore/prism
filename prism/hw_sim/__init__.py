"""
prism.hw_sim — Hardware simulation models for the PRISM photonic accelerator.

Provides physics-based models of the photonic components used in PRISM:
  - MRRModel: Lorentzian micro-ring resonator with electro-optic tuning
"""

from prism.hw_sim.mrr_model import MRRModel

__all__ = ["MRRModel"]
