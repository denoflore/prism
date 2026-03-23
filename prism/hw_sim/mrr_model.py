"""
PRISM: Micro-Ring Resonator (MRR) Transmission Model

Physics-based model of an add-drop MRR with electro-optic (Pockels) tuning.

An MRR acts as a wavelength-selective filter. Near resonance, its transmission
follows a Lorentzian lineshape. In PRISM, each WDM channel passes through an
MRR whose resonance is voltage-tuned to encode a MAC weight:
  - On-resonance: light is dropped (low through-port, high drop-port)
  - Off-resonance: light passes through (high through-port, low drop-port)
  - Balanced weight: w = T_through - T_drop, giving w in [-1, +1]

The electro-optic (EO) tuning uses the Pockels effect in TFLN (thin-film
lithium niobate) to shift the MRR resonance proportional to applied voltage:
  delta_lambda = -(lambda^2 * r33 * n_eff^2 * Gamma) / (4 * pi * R * n_g)
  per volt applied.

Default parameters match the X-cut TFLN MRR design in the PRISM paper:
  Q_loaded=10000, ER=20dB, lambda_res=1550nm, r33=30.9pm/V,
  n_eff=2.138, radius=20um.

Usage:
    from prism.hw_sim.mrr_model import MRRModel

    mrr = MRRModel()
    T_thru, T_drop = mrr.add_drop_transmission(detuning_nm=0.05)
    weight = mrr.balanced_weight(detuning_nm=0.05)
    detuning = mrr.voltage_to_detuning(voltage=1.0)
"""

import torch
import math
from dataclasses import dataclass, field


@dataclass
class MRRModel:
    """
    Lorentzian micro-ring resonator model with electro-optic tuning.

    The MRR is modeled in the add-drop configuration with two bus waveguides.
    Transmission is parameterized by loaded Q factor and extinction ratio (ER),
    which together determine the coupling regime and loss.

    Attributes:
        Q_loaded: Loaded quality factor (determines linewidth).
        ER_dB: Extinction ratio in dB (through-port on/off contrast).
        lambda_res: Resonance wavelength in nm.
        r33: Electro-optic coefficient in pm/V (LiNbO3 r33).
        n_eff: Effective refractive index of the waveguide mode.
        n_g: Group index (for resonance shift calculation).
        Gamma_EO: Electro-optic overlap integral (0 to 1).
        radius_um: Ring radius in micrometers.
    """
    Q_loaded: float = 10_000      # design value; FDTD measured Q_L=12,500+/-1,800 (slab100/gap100)
    ER_dB: float = 20.0           # FDTD extinction ratio (mean D_max=0.317 -> ~20 dB)
    lambda_res: float = 1550.0    # nm, C-band telecom
    r33: float = 30.9             # pm/V, X-cut LiNbO3 EO coefficient (Hu et al., Nat. Commun. 2025)
    n_eff: float = 2.138          # Lumerical MODE, TFLN rib WG 600nm x 1.4um
    n_g: float = 2.30             # FDTD ring FSR -> n_g = lambda^2 / (FSR * 2*pi*R)
    Gamma_EO: float = 0.70        # EO overlap integral, lateral S-G electrode, d_eff=2.5um
    radius_um: float = 20.0       # ring radius, circumference=125.7um, FSR=8.29nm

    # Derived (computed in __post_init__)
    FWHM: float = field(init=False)
    d_lambda_per_volt: float = field(init=False)

    def __post_init__(self):
        """Compute derived quantities from primary parameters."""
        # FWHM (full width at half maximum) of the Lorentzian
        # FWHM = lambda_res / Q_loaded
        self.FWHM = self.lambda_res / self.Q_loaded  # in nm

        # Electro-optic tuning rate: resonance shift per volt
        # From Pockels effect in a ring resonator:
        #   d_lambda/dV = -lambda_res * n_eff^2 * r33 * Gamma / (2 * n_g * d_eff)
        # With lateral electrode gap d_eff ~ 2 * radius for rough estimate,
        # or using the empirical value from FDTD:
        #   d_lambda/dV = lambda_res * Delta_n_per_V / n_g
        # where Delta_n_per_V = -(n_eff^3 * r33 * Gamma) / (2 * d_eff)
        # Using d_eff = 2.5 um (lateral electrode effective gap):
        d_eff_nm = 2500.0  # nm (2.5 um converted)
        Delta_n_per_V = (self.n_eff ** 3 * self.r33 * 1e-3 * self.Gamma_EO
                         / (2.0 * d_eff_nm))  # r33 in pm/V -> nm/V via 1e-3
        self.d_lambda_per_volt = self.lambda_res * Delta_n_per_V / self.n_g  # nm/V

    def lorentzian_transmission(
        self,
        detuning_nm: torch.Tensor | float,
        Q: float | None = None,
        ER_dB: float | None = None,
    ) -> torch.Tensor:
        """
        Through-port transmission of an add-drop MRR (Lorentzian dip).

        The through-port transmission near resonance follows:
          T_through(delta) = 1 - (1 - 1/ER) / (1 + (2*delta/FWHM)^2)

        where delta is the wavelength detuning from resonance, FWHM = lambda/Q,
        and ER is the linear extinction ratio.

        At resonance (delta=0): T = 1/ER (the extinction floor).
        Far from resonance: T -> 1 (full transmission).

        Args:
            detuning_nm: Wavelength detuning from resonance in nm.
                         Can be a float or torch.Tensor.
            Q: Override loaded Q factor. Uses self.Q_loaded if None.
            ER_dB: Override extinction ratio in dB. Uses self.ER_dB if None.

        Returns:
            T_through: Through-port power transmission, same shape as detuning_nm.
        """
        Q = Q if Q is not None else self.Q_loaded
        ER_dB = ER_dB if ER_dB is not None else self.ER_dB

        if not isinstance(detuning_nm, torch.Tensor):
            detuning_nm = torch.tensor(detuning_nm, dtype=torch.float64)

        FWHM = self.lambda_res / Q
        ER_linear = 10.0 ** (ER_dB / 10.0)  # convert dB to linear

        # Lorentzian through-port: dip at resonance
        x = 2.0 * detuning_nm / FWHM  # normalized detuning
        T_through = 1.0 - (1.0 - 1.0 / ER_linear) / (1.0 + x ** 2)

        return T_through

    def add_drop_transmission(
        self,
        detuning_nm: torch.Tensor | float,
        Q: float | None = None,
        ER_dB: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Through-port and drop-port transmission of an add-drop MRR.

        The add-drop MRR has two bus waveguides. Power conservation requires:
          T_through + T_drop + T_loss = 1

        For a lossless (or near-lossless) critically-coupled ring:
          T_drop(delta) = (1 - 1/ER) / (1 + (2*delta/FWHM)^2)
          T_through(delta) = 1 - T_drop(delta)

        The drop port is complementary to the through port:
          - At resonance: T_drop is maximum, T_through is minimum
          - Off resonance: T_drop -> 0, T_through -> 1

        Args:
            detuning_nm: Wavelength detuning from resonance in nm.
            Q: Override loaded Q factor.
            ER_dB: Override extinction ratio in dB.

        Returns:
            (T_through, T_drop): Tuple of power transmissions.
        """
        T_through = self.lorentzian_transmission(detuning_nm, Q, ER_dB)

        if not isinstance(detuning_nm, torch.Tensor):
            detuning_nm = torch.tensor(detuning_nm, dtype=torch.float64)

        ER_dB_val = ER_dB if ER_dB is not None else self.ER_dB
        ER_linear = 10.0 ** (ER_dB_val / 10.0)
        Q_val = Q if Q is not None else self.Q_loaded
        FWHM = self.lambda_res / Q_val
        x = 2.0 * detuning_nm / FWHM

        # Drop port: Lorentzian peak at resonance
        T_drop = (1.0 - 1.0 / ER_linear) / (1.0 + x ** 2)

        return T_through, T_drop

    def balanced_weight(
        self,
        detuning_nm: torch.Tensor | float,
        Q: float | None = None,
        ER_dB: float | None = None,
    ) -> torch.Tensor:
        """
        Balanced MRR weight: w = T_through - T_drop, ranging in [-1, +1].

        This is the key operation for PRISM's MAC computation:
          - At resonance (delta=0): w = 1/ER - (1-1/ER) ~ -1 (for high ER)
          - Far off resonance: w = 1 - 0 = +1
          - At some intermediate detuning: w = 0

        By voltage-tuning the MRR resonance relative to a fixed WDM channel,
        PRISM programs arbitrary weights in [-1, +1].

        Args:
            detuning_nm: Wavelength detuning from resonance in nm.
            Q: Override loaded Q factor.
            ER_dB: Override extinction ratio in dB.

        Returns:
            w: Balanced weight in [-1, +1], same shape as detuning_nm.
        """
        T_through, T_drop = self.add_drop_transmission(detuning_nm, Q, ER_dB)
        return T_through - T_drop

    def voltage_to_detuning(
        self,
        voltage: torch.Tensor | float,
        r33: float | None = None,
        n_eff: float | None = None,
        wavelength: float | None = None,
    ) -> torch.Tensor:
        """
        Convert applied voltage to resonance wavelength detuning via Pockels effect.

        The electro-optic effect in TFLN shifts the MRR resonance:
          delta_lambda = d_lambda/dV * V

        where d_lambda/dV depends on the material EO coefficient (r33),
        effective index, overlap integral, and electrode geometry.

        Args:
            voltage: Applied voltage in volts. Can be float or Tensor.
            r33: Override EO coefficient in pm/V. Uses self.r33 if None.
            n_eff: Override effective index. Uses self.n_eff if None.
            wavelength: Override wavelength in nm. Uses self.lambda_res if None.

        Returns:
            detuning_nm: Resonance shift in nm (positive = blue shift for +V).
        """
        if not isinstance(voltage, torch.Tensor):
            voltage = torch.tensor(voltage, dtype=torch.float64)

        # Use overrides or defaults
        r33_val = r33 if r33 is not None else self.r33
        n_eff_val = n_eff if n_eff is not None else self.n_eff
        lam = wavelength if wavelength is not None else self.lambda_res

        # Recompute tuning rate if parameters are overridden
        if r33 is not None or n_eff is not None or wavelength is not None:
            d_eff_nm = 2500.0
            Delta_n_per_V = (n_eff_val ** 3 * r33_val * 1e-3 * self.Gamma_EO
                             / (2.0 * d_eff_nm))
            rate = lam * Delta_n_per_V / self.n_g
        else:
            rate = self.d_lambda_per_volt

        detuning_nm = rate * voltage
        return detuning_nm

    def __repr__(self) -> str:
        return (
            f"MRRModel(Q={self.Q_loaded:.0f}, ER={self.ER_dB:.0f}dB, "
            f"lambda={self.lambda_res:.1f}nm, r33={self.r33:.1f}pm/V, "
            f"FWHM={self.FWHM*1e3:.1f}pm, "
            f"d_lambda/dV={self.d_lambda_per_volt*1e3:.2f}pm/V)"
        )


def demo():
    """Quick demonstration of MRR model functionality."""
    mrr = MRRModel()
    print(f"MRR Model: {mrr}")
    print()

    # Sweep detuning
    detuning = torch.linspace(-0.5, 0.5, 11)
    T_thru, T_drop = mrr.add_drop_transmission(detuning)
    w = mrr.balanced_weight(detuning)

    print(f"  {'Detuning (nm)':>14s}  {'T_through':>10s}  {'T_drop':>10s}  {'Weight':>10s}")
    print(f"  {'-'*14}  {'-'*10}  {'-'*10}  {'-'*10}")
    for i in range(len(detuning)):
        print(f"  {detuning[i].item():>14.3f}  {T_thru[i].item():>10.4f}  "
              f"{T_drop[i].item():>10.4f}  {w[i].item():>10.4f}")

    print()

    # Voltage to detuning
    voltages = torch.tensor([0.0, 1.0, 2.0, 5.0, 10.0])
    shifts = mrr.voltage_to_detuning(voltages)
    print(f"  {'Voltage (V)':>12s}  {'Shift (pm)':>12s}")
    print(f"  {'-'*12}  {'-'*12}")
    for i in range(len(voltages)):
        print(f"  {voltages[i].item():>12.1f}  {shifts[i].item()*1e3:>12.2f}")


if __name__ == "__main__":
    demo()
