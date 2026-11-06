"""Reproduce the classical-record access certificate and its scope controls.

Run with Python 3.10+, NumPy and SciPy. No manuscript inputs are changed.
The construction is an application of coupling/linear-identifiability methods,
not a new probability bound or a theorem of quantum objectivity.
"""
from fractions import Fraction
from itertools import combinations, product
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


PAIRS = [(a, b) for a in range(4) for b in range(4) if a != b]
OBSERVATIONS = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2), (3, 2)]
OBS_INDEX = {value: index for index, value in enumerate(OBSERVATIONS)}
READOUTS = np.asarray(list(product((0, 1), repeat=6)), dtype=int)


def mask_law(support):
    return np.asarray([1 / len(support) if pair in support else 0 for pair in PAIRS])


def local_observation(site, pointer):
    return site, pointer if site < 2 else 2


def entropy(probabilities):
    p = np.asarray(probabilities).ravel()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def mutual_information(joint):
    return entropy(joint.sum(axis=1)) + entropy(joint.sum(axis=0)) - entropy(joint)


def local_information(w, observer):
    joint = np.zeros((2, 6))
    for mass, pair in zip(w, PAIRS):
        for pointer in (0, 1):
            observation = local_observation(pair[observer], pointer)
            joint[pointer, OBS_INDEX[observation]] += mass / 2
    return mutual_information(joint)


def union_information(w):
    # Both carrier identities and both recorded symbols are accessible together.
    joint = np.zeros((2, 48))
    for index, (mass, pair) in enumerate(zip(w, PAIRS)):
        for pointer in (0, 1):
            symbol = pointer if any(site < 2 for site in pair) else 2
            joint[pointer, 4 * index + symbol] += mass / 2
    return mutual_information(joint)


def local_decoder_optimum(w):
    # All 64 deterministic maps per reader are considered independently.
    # Randomized local strategies/shared randomness are convex combinations,
    # so cannot exceed this optimum for a linear success score.
    scores = np.zeros((64, 64))
    individual = [np.zeros(64), np.zeros(64)]
    for mass, (a, b) in zip(w, PAIRS):
        for pointer in (0, 1):
            ia = OBS_INDEX[local_observation(a, pointer)]
            ib = OBS_INDEX[local_observation(b, pointer)]
            correct_a = READOUTS[:, ia] == pointer
            correct_b = READOUTS[:, ib] == pointer
            scores += mass / 2 * np.outer(correct_a, correct_b)
            individual[0] += mass / 2 * correct_a
            individual[1] += mass / 2 * correct_b
    return float(scores.max()), [float(values.max()) for values in individual]


def partial_information_plot():
    values = []
    for size in range(5):
        branch = []
        for subset in combinations(range(4), size):
            branch.append(Fraction(int(any(site < 2 for site in subset)), 1))
        values.append(str(sum(branch, Fraction(0)) / len(branch)))
    return values


def coherence_control():
    # A distinct source/control: the qubit repetition isometry |x> -> |xx>.
    # Basis-state readout tests classical records; Bell fidelity tests coherence.
    ghz = np.zeros(8)
    ghz[[0, 7]] = 1 / np.sqrt(2)
    coherent = np.outer(ghz, ghz)
    dephased = np.diag(np.diag(coherent))
    decode = np.zeros((8, 8))
    for basis in range(8):
        reference, first, second = (basis >> 2) & 1, (basis >> 1) & 1, basis & 1
        output = (reference << 2) | (first << 1) | (second ^ first)
        decode[output, basis] = 1
    bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
    results = []
    single_marginals = []
    for state in (coherent, dephased):
        shaped = state.reshape(2, 2, 2, 2, 2, 2)
        single_marginals.append(np.trace(shaped, axis1=2, axis2=5).reshape(4, 4))
        decoded = (decode @ state @ decode.T).reshape(2, 2, 2, 2, 2, 2)
        retained = np.trace(decoded, axis1=2, axis2=5).reshape(4, 4)
        results.append(float(bell @ retained @ bell))
    residual = float(np.max(np.abs(single_marginals[0] - single_marginals[1])))
    assert residual < 1e-14
    np.testing.assert_allclose(results, [1, .5], atol=1e-14)
    return {
        "full_register_Bell_fidelity_before_after_dephasing": results,
        "single_record_reference_state_maximum_difference": residual,
        "pointer_readout_accuracy_before_after": [1.0, 1.0],
        "scope": "Distinct coherent/dephased sources; their full quantum mutual information differs."
    }


def calculate():
    law_a = mask_law([(0, 1), (1, 0), (2, 3), (3, 2)])
    law_b = mask_law([(0, 2), (2, 0), (1, 3), (3, 1)])
    rows = [np.ones(12)]
    rhs = [1.0]
    for observer in (0, 1):
        for site in range(4):
            rows.append(np.asarray([int(pair[observer] == site) for pair in PAIRS]))
            rhs.append(.25)
    matrix, target = np.asarray(rows), np.asarray(rhs)
    both_good = np.asarray([int(a < 2 and b < 2) for a, b in PAIRS])
    both_correct = .5 + .5 * both_good
    lower = linprog(both_good, A_eq=matrix, b_eq=target, bounds=(0, None), method="highs")
    upper = linprog(-both_good, A_eq=matrix, b_eq=target, bounds=(0, None), method="highs")
    assert lower.success and upper.success
    np.testing.assert_allclose([lower.fun, -upper.fun], [0, .5], atol=1e-12)
    assert np.linalg.matrix_rank(matrix) == 7
    assert np.linalg.matrix_rank(np.vstack([matrix, both_good])) == 8
    np.testing.assert_allclose(matrix @ (law_a - law_b), 0, atol=1e-14)
    assert abs(both_good @ (law_a - law_b) - .5) < 1e-14
    certificate = np.zeros(10)
    certificate[0], certificate[-1] = .5, .5
    augmented = np.vstack([matrix, both_good])
    np.testing.assert_allclose(certificate @ augmented, both_correct, atol=1e-14)

    protocols = {}
    for name, w in [("A", law_a), ("B", law_b)]:
        q = float(both_good @ w)
        local_optimum, marginal_accuracy = local_decoder_optimum(w)
        information = [local_information(w, observer) for observer in (0, 1)]
        np.testing.assert_allclose(matrix @ w, target, atol=1e-14)
        np.testing.assert_allclose(local_optimum, .5 + .5 * q, atol=1e-14)
        np.testing.assert_allclose(marginal_accuracy, [.75, .75], atol=1e-14)
        np.testing.assert_allclose(information, [.5, .5], atol=1e-14)
        np.testing.assert_allclose(union_information(w), 1 - q, atol=1e-14)
        protocols[name] = {
            "ordered_pairs_one_based": [list(np.asarray(pair) + 1) for pair, mass in zip(PAIRS, w) if mass],
            "nonzero_assignment_probability": .25,
            "probability_both_readable": q,
            "local_mutual_information_bits": information,
            "optimal_individual_correctness": marginal_accuracy,
            "optimal_both_correct_no_communication": local_optimum,
            "optimal_both_correct_communication_allowed": 1 - .5 * q,
            "pooled_mutual_information_bits": union_information(w),
        }

    # Check the entire one-parameter witness family, not only its endpoints.
    for t in np.linspace(0, 1, 11):
        w = t * law_a + (1 - t) * law_b
        optimum, _ = local_decoder_optimum(w)
        np.testing.assert_allclose(optimum, .5 + .25 * t, atol=1e-14)

    # Removing source independence invalidates inference even while each
    # local carrier identity remains uniform. The two conditional supports
    # are disjoint, so a public mask reveals X exactly in this counterexample.
    leaking_mask = np.vstack([law_a / 2, law_b / 2])
    np.testing.assert_allclose(mutual_information(leaking_mask), 1, atol=1e-14)
    for pointer_law in (law_a, law_b):
        np.testing.assert_allclose(matrix @ pointer_law, target, atol=1e-14)

    pip = partial_information_plot()
    assert pip == ["0", "1/2", "5/6", "1", "1"]
    result = {
        "status": "Exact conditional construction; executable verification; no hardware experiment.",
        "source": "Uniform classical X; sites 1,2 contain X; sites 3,4 blank; source-independent masks.",
        "protocols": protocols,
        "uniform_fragment_source_information_plot_bits": pip,
        "compatible_probability_both_readable_interval": [float(lower.fun), float(-upper.fun)],
        "compatible_local_joint_correctness_interval": [.5 + .5 * lower.fun, .5 - .5 * upper.fun],
        "calibration_matrix_rank_before_after_joint_statistic": [7, 8],
        "decoder_enumeration": {"maps_per_observer": 64, "map_pairs": 4096, "mixture_cases_checked": 11},
        "source_independence_counterexample_public_mask_information_bits": mutual_information(leaking_mask),
        "coherence_control": coherence_control(),
        "limitations": [
            "Does not alter intrinsic source redundancy or establish objectivity of arbitrary quantum states.",
            "The conventional uniform-fragment source curve is fixed; protocol-weighted pooled information changes.",
            "Flag/source independence and communication rights are essential.",
            "Noisy records require recalibration and a separate decoder optimization.",
            "Fréchet bounds and good-record/bad-record sources have established antecedents."
        ]
    }
    return result
