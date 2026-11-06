"""Original exponential cavity recurrence and subset contractions, extracted from independent_reproduction.py."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def mask_weights(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    output = np.zeros(32, dtype=float)
    for mask in range(32):
        probability = 1.0
        for carrier in range(5):
            if mask & (1 << carrier):
                probability *= probabilities[carrier]
            else:
                probability *= 1 - probabilities[carrier]
        output[mask] = probability
    return output


def hp_value(table: np.ndarray, d_probabilities: np.ndarray, alpha: float) -> float:
    return float(mask_weights(d_probabilities) @ table @ mask_weights(np.full(5, alpha)))


def hp_fixed_early(table: np.ndarray, d_probabilities: np.ndarray, e_mask: int) -> float:
    return float(mask_weights(d_probabilities) @ table[:, e_mask])


def root_bisection(table: np.ndarray, d_probabilities: np.ndarray) -> float | None:
    low, high = 0.0, 1.0
    f_low = hp_value(table, d_probabilities, low) - 0.5
    f_high = hp_value(table, d_probabilities, high) - 0.5
    if f_low * f_high > 0:
        return None
    for _ in range(80):
        midpoint = (low + high) / 2
        value = hp_value(table, d_probabilities, midpoint) - 0.5
        if value == 0:
            return midpoint
        if f_low * value <= 0:
            high = midpoint
            f_high = value
        else:
            low = midpoint
            f_low = value
    return (low + high) / 2


def qec_value(probabilities: np.ndarray) -> float:
    weights = mask_weights(probabilities)
    return float(sum(weights[mask] for mask in range(32) if int(mask).bit_count() >= 3))


def exponential_cavity(
    times: np.ndarray,
    drive: np.ndarray,
    *,
    eta: float,
    center: float,
    kappa: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Piecewise-midpoint exponential cavity propagation."""
    times = np.asarray(times, dtype=float)
    drive = np.asarray(drive, dtype=complex)
    kappa_in = kappa / 2
    kappa_det = kappa / 2
    decay = kappa / 2 + 1j * center
    cavity = np.zeros(len(times), dtype=complex)
    detector = np.zeros(len(times), dtype=float)
    reflected = np.zeros(len(times), dtype=float)
    global_energy = np.zeros(len(times), dtype=float)
    root_eta = math.sqrt(eta)
    root_in = math.sqrt(kappa_in)
    for index in range(len(times) - 1):
        dt = float(times[index + 1] - times[index])
        propagator = np.exp(-decay * dt)
        midpoint_drive = root_eta * (drive[index] + drive[index + 1]) / 2
        cavity[index + 1] = (
            propagator * cavity[index]
            + root_in * midpoint_drive * (1 - propagator) / decay
        )
        detector[index + 1] = detector[index] + (
            kappa_det * dt * (abs(cavity[index]) ** 2 + abs(cavity[index + 1]) ** 2) / 2
        )
        reflected_left = root_eta * drive[index] - root_in * cavity[index]
        reflected_right = root_eta * drive[index + 1] - root_in * cavity[index + 1]
        reflected[index + 1] = reflected[index] + dt * (
            abs(reflected_left) ** 2 + abs(reflected_right) ** 2
        ) / 2
        global_energy[index + 1] = global_energy[index] + dt * (
            abs(drive[index]) ** 2 + abs(drive[index + 1]) ** 2
        ) / 2
    accessible = detector + np.abs(cavity) ** 2
    balance = eta * global_energy - accessible - reflected
    return global_energy, accessible, float(np.max(np.abs(balance)))


