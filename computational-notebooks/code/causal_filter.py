"""Surviving closure causal-filter functions, retained without numerical changes."""
from dataclasses import dataclass
import math
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp

@dataclass(frozen=True)
class FilterParameters:
    center_local_momega: float
    kappa_total_per_M: float
    kappa_in_fraction: float = 0.5
    kappa_detector_fraction: float = 0.5

    @property
    def kappa_in(self) -> float:
        return self.kappa_total_per_M * self.kappa_in_fraction

    @property
    def kappa_detector(self) -> float:
        return self.kappa_total_per_M * self.kappa_detector_fraction

def causal_filter_contraction(
    times_M: np.ndarray,
    input_envelope: np.ndarray,
    *,
    eta: float,
    filter_parameters: FilterParameters,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], np.ndarray]:
    """Passive two-port cavity with detector output plus cavity memory.

    Returns global cumulative energy, observer-accessible cumulative energy,
    diagnostics, and the complex cavity amplitude.
    """
    times = np.asarray(times_M, dtype=float)
    envelope = np.asarray(input_envelope, dtype=complex)
    real_drive = CubicSpline(times, envelope.real)
    imag_drive = CubicSpline(times, envelope.imag)
    kappa = filter_parameters.kappa_total_per_M
    kappa_in = filter_parameters.kappa_in
    kappa_det = filter_parameters.kappa_detector
    if kappa <= 0 or kappa_in < 0 or kappa_det < 0:
        raise ValueError("filter rates must be nonnegative")
    if kappa_in + kappa_det > kappa * (1 + 1e-12):
        raise ValueError("port rates cannot exceed total damping")
    root_eta = math.sqrt(max(0.0, eta))
    root_kappa_in = math.sqrt(kappa_in)

    def rhs(time: float, state: np.ndarray) -> np.ndarray:
        drive = complex(float(real_drive(time)), float(imag_drive(time)))
        cavity = complex(state[0], state[1])
        admitted_drive = root_eta * drive
        derivative = -(
            kappa / 2 + 1j * filter_parameters.center_local_momega
        ) * cavity + root_kappa_in * admitted_drive
        detector_rate = kappa_det * abs(cavity) ** 2
        reflected = admitted_drive - root_kappa_in * cavity
        reflected_rate = abs(reflected) ** 2
        global_rate = abs(drive) ** 2
        return np.array(
            [
                derivative.real,
                derivative.imag,
                detector_rate,
                reflected_rate,
                global_rate,
            ],
            dtype=float,
        )

    spacing = float(np.min(np.diff(times)))
    solution = solve_ivp(
        rhs,
        (float(times[0]), float(times[-1])),
        np.zeros(5, dtype=float),
        t_eval=times,
        method="DOP853",
        rtol=2e-11,
        atol=2e-13,
        max_step=min(0.025, spacing / 2),
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    cavity = solution.y[0] + 1j * solution.y[1]
    detector_energy = solution.y[2]
    reflected_energy = solution.y[3]
    global_energy = solution.y[4]
    cavity_energy = np.abs(cavity) ** 2
    accessible = detector_energy + cavity_energy
    balance = eta * global_energy - accessible - reflected_energy
    diagnostics = {
        "maximum_energy_balance_residual": float(np.max(np.abs(balance))),
        "maximum_nesting_violation": float(np.max(accessible - global_energy)),
        "minimum_accessible": float(np.min(accessible)),
        "terminal_global": float(global_energy[-1]),
        "terminal_accessible": float(accessible[-1]),
        "terminal_detector_output": float(detector_energy[-1]),
        "terminal_cavity_memory": float(cavity_energy[-1]),
        "terminal_reflected": float(reflected_energy[-1]),
        "terminal_balance_residual": float(balance[-1]),
    }
    return global_energy, accessible, diagnostics, cavity

def channel_for_packets(
    times_M: np.ndarray,
    envelopes: np.ndarray,
    *,
    eta: float,
    filter_parameters: FilterParameters,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]], np.ndarray]:
    global_curves = []
    observer_curves = []
    diagnostics = []
    cavities = []
    for index in range(envelopes.shape[1]):
        global_curve, observer_curve, diagnostic, cavity = causal_filter_contraction(
            times_M,
            envelopes[:, index],
            eta=eta,
            filter_parameters=filter_parameters,
        )
        global_curves.append(global_curve)
        observer_curves.append(observer_curve)
        diagnostics.append(diagnostic)
        cavities.append(cavity)
    return (
        np.column_stack(global_curves),
        np.column_stack(observer_curves),
        diagnostics,
        np.column_stack(cavities),
    )

