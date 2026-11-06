"""Recorded collection and delivery to a finite Hayden--Preskill receiver.

The collection instrument records a carrier's acceptance time without resolving
its logical rail. Its CDF agrees with the controlled catcher's fixed-time
survival probability. A count gate stops collection, packs the accepted qubits
into a source register, and transfers that register to designated storage.
Early radiation stays in its existing register; diary decoding is a separate
readout task. All times are measured in units of M.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss


DEFAULT_CUTOFF = 57.15
DEFAULT_DELAYS = (0., 2., 5., 10., 20.)
MODES = ("common_constant", "finite_chain_reset")
ROUND_OFF = 2e-11


@dataclass(frozen=True)
class RecordedCollection:
    """Piecewise-linear acceptance CDFs on an explicitly supplied time grid."""

    times: np.ndarray
    cdf: np.ndarray
    detector_output: np.ndarray
    acceptance_beta: np.ndarray
    roundoff_adjustment: float

    @property
    def carriers(self):
        return self.cdf.shape[1]


def recorded_cdf(run, epsilon=1e-6):
    """Construct and validate the recorded instrument's acceptance CDF.

    This declares a new time-recorded instrument. The coherent catcher itself
    need not supply an acceptance-time record. With detector-port dose d,
    beta=1-[epsilon/(epsilon+d)]**2 gives density beta*d_dot and cumulative
    acceptance m=d**2/(epsilon+d). Between supplied samples, this module uses
    linear interpolation of m, thereby specifying its numerical time law.
    """
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    times = np.asarray(run["times"], dtype=float)
    observed = np.asarray(run["observer_carriers"], dtype=float)
    cavities = np.asarray(run["cavities"])
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all():
        raise ValueError("times must contain at least two finite samples")
    if np.any(np.diff(times) <= 0):
        raise ValueError("times must be strictly increasing")
    if observed.ndim != 2 or observed.shape[0] != len(times):
        raise ValueError("observer_carriers must have time and carrier axes")
    if observed.shape[1] < 1 or cavities.shape != observed.shape:
        raise ValueError("cavities and observer_carriers must have equal shapes")
    if not np.isfinite(observed).all() or not np.isfinite(cavities).all():
        raise ValueError("the detector samples must be finite")
    raw = observed - abs(cavities)**2
    if raw.min() < -ROUND_OFF or raw.max() > 1 + ROUND_OFF:
        raise ValueError("detector-port dose must lie in [0,1]")
    if np.max(abs(raw[0])) > ROUND_OFF:
        raise ValueError("the supplied collection record must start empty")
    if np.diff(raw, axis=0).min() < -ROUND_OFF:
        raise ValueError("detector-port dose must be nondecreasing")
    # Remove only validation-bounded floating-point roundoff, recording its size.
    dose = np.maximum.accumulate(np.clip(raw, 0., 1.), axis=0)
    dose[0] = 0.
    correction = float(np.max(abs(dose - raw)))
    if correction > ROUND_OFF:
        raise ValueError("roundoff repair would exceed the declared tolerance")
    cdf = dose**2 / (epsilon + dose)
    beta = 1 - (epsilon / (epsilon + dose))**2
    return RecordedCollection(times.copy(), cdf, dose, beta, correction)


def cdf_at(record, times):
    """Evaluate the declared linear CDF, without extrapolating its time range."""
    values = np.asarray(times, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("query times must be finite")
    if np.any(values < record.times[0]) or np.any(values > record.times[-1]):
        raise ValueError("query time lies outside the supplied collection grid")
    return np.stack([np.interp(values, record.times, record.cdf[:, j])
                     for j in range(record.carriers)], axis=-1)


def count_tail(probabilities, gate):
    """Probability of at least gate acceptances for independent carriers."""
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must be a finite vector in [0,1]")
    if not isinstance(gate, (int, np.integer)) or not 1 <= gate <= len(p):
        raise ValueError("gate must be between one and the number of carriers")
    law = np.array([1.])
    for probability in p:
        law = np.convolve(law, [1-probability, probability])
    return float(law[gate:].sum())


def gate_density(probabilities, densities, gate):
    """Joint density of gate time and its accepted subset.

    f_S(t)=sum_{j in S} F'_j(t) prod_{i in S except j} F_i(t)
                                  prod_{k outside S} [1-F_k(t)].
    Columns use the ordinary little-endian carrier-mask order. For n carriers,
    each within-bin density is a polynomial of degree at most n-1.
    """
    p = np.asarray(probabilities, dtype=float)
    f = np.asarray(densities, dtype=float)
    if p.ndim != 2 or p.shape != f.shape or not np.isfinite(p).all() or not np.isfinite(f).all():
        raise ValueError("CDF and density samples need equal finite 2D shapes")
    if np.any((p < -ROUND_OFF) | (p > 1+ROUND_OFF)) or np.any(f < -ROUND_OFF):
        raise ValueError("invalid CDF or density values")
    n = p.shape[1]
    if not isinstance(gate, (int, np.integer)) or not 1 <= gate <= n:
        raise ValueError("invalid count gate")
    out = np.zeros((len(p), 1 << n))
    for subset in combinations(range(n), gate):
        mask = sum(1 << j for j in subset)
        outside = [j for j in range(n) if j not in subset]
        for last in subset:
            term = f[:, last].copy()
            for j in subset:
                if j != last:
                    term *= p[:, j]
            for j in outside:
                term *= 1-p[:, j]
            out[:, mask] += term
    return out


def finite_chain_floor(age, omega=.025, largest_node=62, cutoff=DEFAULT_CUTOFF):
    """h=1 finite-chain probability certificate with preselected isolation.

    Interaction ends after min(age,cutoff); captured information is held
    losslessly until readout. The cutoff is frozen at 57.15M, the maximum of
    the certificate on the declared 0.05M selection grid. It is not reselected
    for a source, arrival branch, or loading delay.
    """
    if not np.isfinite(omega) or omega <= 0 or largest_node < 1:
        raise ValueError("positive omega and at least one receiving site required")
    if not np.isfinite(cutoff) or cutoff < 0:
        raise ValueError("cutoff must be finite and nonnegative")
    age = np.asarray(age, dtype=float)
    if not np.isfinite(age).all():
        raise ValueError("age must be finite")
    x = omega * np.minimum(np.maximum(age, 0.), cutoff)
    error = 2*np.sqrt(largest_node+1)*np.tanh(x)**(largest_node+2)
    return np.maximum(0., 1-np.cosh(x)**(-4)-error)


def capture_probability(age, mode, rate=.025, omega=.025,
                        largest_node=62, cutoff=DEFAULT_CUTOFF):
    if mode not in MODES:
        raise ValueError("unknown receiving mode")
    if mode == "finite_chain_reset":
        return finite_chain_floor(age, omega, largest_node, cutoff)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("the common rate must be finite and positive")
    age = np.asarray(age, dtype=float)
    if not np.isfinite(age).all():
        raise ValueError("age must be finite")
    return -np.expm1(-rate*np.maximum(age, 0.))


def gate_weights(record, gate, deadline=120., loading_delay=0.,
                 mode="finite_chain_reset", order=8, rate=.025, omega=.025,
                 largest_node=62, cutoff=DEFAULT_CUTOFF):
    """Integrate the capture-weighted gate law over the recorded instrument."""
    if not np.isfinite(deadline) or not record.times[0] <= deadline <= record.times[-1]:
        raise ValueError("deadline must lie within the supplied collection grid")
    if not np.isfinite(loading_delay) or loading_delay < 0:
        raise ValueError("loading_delay must be finite and nonnegative")
    if not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError("quadrature order must be a positive integer")
    if not isinstance(gate, (int, np.integer)) or not 1 <= gate <= record.carriers:
        raise ValueError("invalid count gate")
    if mode not in MODES:
        raise ValueError("unknown receiving mode")
    ready = deadline-loading_delay
    zeros = np.zeros(1 << record.carriers)
    if ready <= record.times[0]:
        return dict(weights=zeros.copy(), gate_law=zeros.copy(), gate_probability=0.,
                    gate_mass_residual=0., captured_probability=0.)
    boundaries = record.times[(record.times >= record.times[0]) & (record.times < ready)]
    extra = [ready]
    if mode == "finite_chain_reset" and record.times[0] < ready-cutoff < ready:
        extra.append(ready-cutoff)
    boundaries = np.unique(np.r_[boundaries, extra])
    lo, hi = boundaries[:-1], boundaries[1:]
    gx, gw = leggauss(order)
    nodes = ((lo+hi)[:, None]/2 + (hi-lo)[:, None]*gx/2).ravel()
    quadrature = ((hi-lo)[:, None]*gw/2).ravel()
    bins = np.searchsorted(record.times, nodes, side="right")-1
    bins = np.minimum(bins, len(record.times)-2)
    slopes = np.diff(record.cdf, axis=0)/np.diff(record.times)[:, None]
    p = record.cdf[bins]+(nodes-record.times[bins])[:, None]*slopes[bins]
    density = gate_density(p, slopes[bins], gate)
    probability = capture_probability(ready-nodes, mode, rate, omega, largest_node, cutoff)
    law = quadrature @ density
    weights = (quadrature*probability) @ density
    expected_gate = count_tail(cdf_at(record, ready), gate)
    residual = abs(float(law.sum())-expected_gate)
    if residual > 2e-10:
        raise ArithmeticError("gate density does not integrate to its count-tail probability")
    if weights.min() < -ROUND_OFF or weights.sum() > expected_gate+ROUND_OFF:
        raise ArithmeticError("invalid capture-weighted gate law")
    return dict(weights=weights, gate_law=law, gate_probability=expected_gate,
                gate_mass_residual=residual, captured_probability=float(weights.sum()))


def statewise_scores(source, weights):
    """Keep sourcewise scores until after averaging the recorded gate law."""
    tables = np.asarray(source, dtype=float)
    if tables.ndim == 3:
        if tables.shape[1] != tables.shape[2]:
            raise ValueError("late and early mask axes must have equal lengths")
        tables = tables[:, :, -1]
    if tables.ndim != 2 or not np.isfinite(tables).all():
        raise ValueError("source must contain finite information tables")
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (tables.shape[1],) or not np.isfinite(weights).all():
        raise ValueError("weights must match the late-mask axis")
    if weights.min() < -ROUND_OFF or weights.sum() > 1+ROUND_OFF:
        raise ValueError("weights must be a subprobability law")
    if tables.min() < -ROUND_OFF or tables.max() > 1+ROUND_OFF:
        raise ValueError("normalized source information must lie in [0,1]")
    if np.max(abs(tables[:, 0])) > ROUND_OFF:
        raise ValueError("the early-only register must be reference-independent")
    tables = np.clip(tables, 0., 1.)
    information = tables @ weights
    fidelity = .25+(np.exp2(2*tables-2)-.25) @ weights
    return information, fidelity


def evaluate(source, runs, deadline=120., loading_delays=DEFAULT_DELAYS,
             order=8, check_order=4, rate=.025, omega=.025,
             largest_node=62, cutoff=DEFAULT_CUTOFF, epsilon=1e-6):
    """Evaluate every grid, count gate, loading delay and receiving mode."""
    summaries, detail = [], {}
    for grid, run in runs.items():
        record = recorded_cdf(run, epsilon)
        for gate in range(1, record.carriers+1):
            for delay in loading_delays:
                for mode in MODES:
                    args = dict(deadline=deadline, loading_delay=delay, mode=mode,
                                rate=rate, omega=omega, largest_node=largest_node, cutoff=cutoff)
                    result = gate_weights(record, gate, order=order, **args)
                    check = gate_weights(record, gate, order=check_order, **args)
                    information, fidelity = statewise_scores(source, result["weights"])
                    q_error = float(np.max(abs(result["weights"]-check["weights"])))
                    score_check = statewise_scores(source, check["weights"])[1]
                    limiting = int(np.argmin(fidelity))
                    key = f"{grid}__gate{gate}__load{float(delay):g}__{mode}"
                    detail[key] = dict(weights=result["weights"], gate_law=result["gate_law"],
                                       information=information, fidelity_lower=fidelity)
                    summaries.append(dict(grid=str(grid), receiving_mode=mode, gate_count=gate,
                        loading_delay_M=float(delay), deadline_M=float(deadline),
                        receiving_site_levels=2**gate+1, receiving_data_levels=2**gate,
                        largest_receiving_node=largest_node if mode == "finite_chain_reset" else None,
                        omega_per_M=omega if mode == "finite_chain_reset" else None,
                        common_rate_per_M=rate if mode == "common_constant" else None,
                        acquisition_cutoff_M=cutoff if mode == "finite_chain_reset" else None,
                        gate_probability=result["gate_probability"],
                        captured_probability_certificate=result["captured_probability"],
                        minimum_retained_information_lower=float(information.min()),
                        worst_source_fidelity_lower=float(fidelity[limiting]),
                        limiting_source_index=limiting,
                        gate_mass_residual=result["gate_mass_residual"],
                        quadrature_weight_difference=q_error,
                        quadrature_fidelity_difference=float(np.max(abs(fidelity-score_check))),
                        cdf_roundoff_adjustment=record.roundoff_adjustment))
    return summaries, detail


def study(source, runs, total=120., loading_times=DEFAULT_DELAYS, **kwargs):
    """Return summary rows and sourcewise results without writing any files."""
    if not runs:
        raise ValueError("at least one supplied detector grid is required")
    summaries, detail = evaluate(source, runs, deadline=total,
                                 loading_delays=loading_times, **kwargs)
    records = {str(grid): recorded_cdf(run, kwargs.get("epsilon", 1e-6))
               for grid, run in runs.items()}
    metadata = dict(
        instrument="Rail-independent, time-recorded collection with m=d^2/(epsilon+d)",
        interpolation="Piecewise-linear CDF on each supplied detector time grid",
        source_policy="The same gate and receiving parameters apply to every frozen source",
        score="All-trial deterministic Bell-fidelity lower certificate; source minimum taken after integration",
        constant_mode="Common absolute-time constant hazard; exact recorded-cohort information identity",
        chain_mode="h=1 finite Jacobi chain, clock reset at loading completion, then source isolation and lossless hold",
        resource_scope="Matched chain length and coupling scale; routing, packing, time records and early-memory storage are additional declared resources",
        readout_scope="Diary decoding occurs after designated-memory acquisition; no efficient decoder circuit is asserted",
        rows=len(summaries), source_count=len(source),
        parameters=dict(total=total, loading_times=list(loading_times),
                        epsilon=kwargs.get("epsilon", 1e-6),
                        rate=kwargs.get("rate", .025), omega=kwargs.get("omega", .025),
                        largest_node=kwargs.get("largest_node", 62), h=1,
                        cutoff=kwargs.get("cutoff", DEFAULT_CUTOFF),
                        quadrature_order=kwargs.get("order", 8),
                        check_quadrature_order=kwargs.get("check_order", 4)),
        maximum_gate_mass_residual=max(row["gate_mass_residual"] for row in summaries),
        maximum_quadrature_weight_difference=max(row["quadrature_weight_difference"] for row in summaries),
        maximum_quadrature_fidelity_difference=max(row["quadrature_fidelity_difference"] for row in summaries))
    return dict(rows=summaries, detail=detail, records=records, metadata=metadata)


def write_results(output, result):
    """Write the returned study, retaining every source's terminal certificate."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summaries = result["rows"]
    with (output/"summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    arrays = {f"{key}__{name}": value for key, row in result["detail"].items()
              for name, value in row.items()}
    np.savez_compressed(output/"statewise_scores.npz", **arrays)
    for grid, record in result["records"].items():
        np.savez_compressed(output/f"{grid}_recorded_cdf.npz", times=record.times,
                            cdf=record.cdf, detector_output=record.detector_output,
                            acceptance_beta=record.acceptance_beta)
    (output/"metadata.json").write_text(json.dumps(result["metadata"], indent=2)+"\n",
                                       encoding="utf-8")


def save(output, source, runs, **kwargs):
    """Evaluate and write a study; equivalent to study then write_results."""
    total = kwargs.pop("deadline", 120.)
    loading_times = kwargs.pop("loading_delays", DEFAULT_DELAYS)
    result = study(source, runs, total=total, loading_times=loading_times, **kwargs)
    write_results(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = np.load(args.root/"inputs/hp_joint_information.npy")
    runs = {name: dict(np.load(args.root/"reference/upstream"/f"{name}.npz"))
            for name in ("baseline", "grid_refined", "physical_refined")}
    from datetime import datetime
    output = args.output or args.root/"outputs/runs"/datetime.now().strftime("progressive_%Y%m%d_%H%M%S")
    output = output.resolve()
    for protected in (args.root/"reference", args.root/"inputs"):
        if output.is_relative_to(protected.resolve()):
            parser.error("Choose an output directory outside reference/ and inputs/.")
    if output.exists():
        parser.error("Choose a new output directory; existing runs are preserved.")
    result = save(output, source, runs)
    print(json.dumps(result["metadata"], indent=2))
