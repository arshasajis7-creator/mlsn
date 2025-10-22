"""Heuristic optimizer for RCWA mask geometries.

This module provides a lightweight stochastic search routine that tweaks
configuration parameters such as material tables, layer thicknesses, and mask
hole placement.  The physics model is intentionally simplified so it can run
quickly inside the desktop tool without requiring an RCWA solve for each
candidate.  The optimizer favours configurations whose frequency sweep centres
on the desired target, distributes holes across the unit cell, and keeps layer
thicknesses within the requested ranges.

The goal is to offer users a starting point that is usually better than manual
guesswork; after optimisation the configuration can still be refined and fed to
the full RCWA adapter for validation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Iterable, List, Sequence, Tuple

from ..models.configuration import Configuration, MaskHole, MaskSpec


@dataclass
class OptimizationSettings:
    """Tunable parameters for :func:`optimize_configuration`.

    The defaults are deliberately conservative so that the generated geometry is
    physically plausible for the baseline datasets bundled with the project.
    All distances are expressed in metres to match the configuration schema.
    """

    target_frequency_GHz: float = 10.0
    desired_hole_count: int = 4
    iterations: int = 60
    random_seed: int | None = None
    hole_count_range: Tuple[int, int] = (1, 9)
    hole_radius_range_m: Tuple[float, float] = (0.0005, 0.0045)
    square_aspect_limits: Tuple[float, float] = (0.5, 2.0)
    cell_margin_fraction: float = 0.8
    layer1_thickness_range_m: Tuple[float, float] = (0.001, 0.008)
    mask_thickness_range_m: Tuple[float, float] = (0.0005, 0.004)
    layer3_thickness_range_m: Tuple[float, float] = (0.002, 0.015)
    material_choices: Sequence[str] = (
        "m1.csv",
        "m2.csv",
        "m3.csv",
        "metal.csv",
        "mhole.csv",
    )
    shapes: Sequence[str] = ("circle", "square")


@dataclass
class OptimizationResult:
    configuration: Configuration
    score: float
    history: List[str]


def optimize_configuration(
    base_config: Configuration, settings: OptimizationSettings
) -> OptimizationResult:
    """Return an optimised configuration along with the scoring history."""

    rng = random.Random(settings.random_seed)
    best_config = base_config
    best_score = _score_configuration(base_config, settings)
    history = [f"Baseline score: {best_score:.4f}"]

    for iteration in range(settings.iterations):
        candidate = _sample_configuration(base_config, settings, rng)
        score = _score_configuration(candidate, settings)
        history.append(f"Iter {iteration + 1:03d}: score={score:.4f}")
        if score > best_score:
            best_score = score
            best_config = candidate
            history.append("  ↳ improved best configuration")

    return OptimizationResult(configuration=best_config, score=best_score, history=history)


def _sample_configuration(
    base_config: Configuration, settings: OptimizationSettings, rng: random.Random
) -> Configuration:
    """Generate a candidate configuration by perturbing *base_config*."""

    freq = base_config.freq
    center_freq = 0.5 * (freq.start + freq.stop)
    span = max(freq.stop - freq.start, 1e-3)

    # Slightly shift the frequency window towards the target.
    delta = settings.target_frequency_GHz - center_freq
    window_shift = 0.25 * delta
    new_start = max(freq.start + window_shift, 0.1)
    new_stop = max(new_start + span, new_start + 0.5)

    freq_spec = replace(freq, start=new_start, stop=new_stop)

    # Sample layer thicknesses within requested ranges.
    layer_top = replace(
        base_config.layer_top,
        thickness_m=_uniform(settings.layer1_thickness_range_m, rng),
        material_csv=rng.choice(tuple(settings.material_choices)),
    )
    mask = _sample_mask(base_config.mask, base_config, settings, rng)
    layer_bottom = replace(
        base_config.layer_bottom,
        thickness_m=_uniform(settings.layer3_thickness_range_m, rng),
        material_csv=rng.choice(tuple(settings.material_choices)),
    )

    polarization = rng.choice(["TE", "TM"])

    harmonics_x = rng.randrange(7, 21, 2)
    harmonics_y = rng.randrange(7, 21, 2)

    return replace(
        base_config,
        freq=freq_spec,
        layer_top=layer_top,
        mask=mask,
        layer_bottom=layer_bottom,
        polarization=polarization,
        n_harmonics=[harmonics_x, harmonics_y],
    )


def _sample_mask(
    mask: MaskSpec, base_config: Configuration, settings: OptimizationSettings, rng: random.Random
) -> MaskSpec:
    cell = base_config.cell
    half_x = 0.5 * cell.Lx_m * settings.cell_margin_fraction
    half_y = 0.5 * cell.Ly_m * settings.cell_margin_fraction

    min_holes, max_holes = settings.hole_count_range
    hole_count = rng.randint(min_holes, max_holes)

    holes: List[MaskHole] = []
    for _ in range(hole_count):
        shape = rng.choice(tuple(settings.shapes))
        x = rng.uniform(-half_x, half_x)
        y = rng.uniform(-half_y, half_y)

        if shape == "square":
            size1 = _uniform(settings.hole_radius_range_m, rng) * 2.0
            aspect = rng.uniform(*settings.square_aspect_limits)
            size2 = max(size1 * aspect, 1e-4)
        else:
            size1 = _uniform(settings.hole_radius_range_m, rng) * 2.0
            size2 = None

        holes.append(
            MaskHole(
                shape=shape,
                x_m=x,
                y_m=y,
                size1=size1,
                size2=size2,
            )
        )

    return replace(
        mask,
        thickness_m=_uniform(settings.mask_thickness_range_m, rng),
        solid_csv=rng.choice(tuple(settings.material_choices)),
        hole_csv=rng.choice(tuple(settings.material_choices)),
        holes=holes,
    )


def _score_configuration(config: Configuration, settings: OptimizationSettings) -> float:
    freq = config.freq
    center = 0.5 * (freq.start + freq.stop)
    span = max(freq.stop - freq.start, 1e-9)

    # Reward proximity to the target frequency.
    score = -abs(settings.target_frequency_GHz - center)

    # Encourage moderate sweep width.
    score -= 0.05 * abs(span - 5.0)

    # Prefer having the desired number of holes.
    hole_count = len(config.mask.holes)
    score -= 0.1 * abs(settings.desired_hole_count - hole_count)

    # Evaluate how well holes fill the unit cell.
    if hole_count:
        fill_score = _hole_distribution_score(config.mask.holes, config.cell.Lx_m, config.cell.Ly_m)
        score += 0.5 * fill_score

    # Penalise extreme thickness choices.
    score -= _range_penalty(config.layer_top.thickness_m, settings.layer1_thickness_range_m)
    score -= _range_penalty(config.layer_bottom.thickness_m, settings.layer3_thickness_range_m)
    score -= _range_penalty(config.mask.thickness_m, settings.mask_thickness_range_m)

    return score


def _hole_distribution_score(holes: Iterable[MaskHole], cell_x: float, cell_y: float) -> float:
    hole_list = list(holes)
    if not hole_list:
        return 0.0

    half_x = cell_x / 2.0 or 1e-9
    half_y = cell_y / 2.0 or 1e-9
    coverage = 0.0
    balance_penalty = 0.0
    for hole in hole_list:
        radius_x = hole.size1 / 2.0
        radius_y = (hole.size2 or hole.size1) / 2.0
        area = math.pi * radius_x * radius_y if hole.shape == "circle" else hole.size1 * (hole.size2 or hole.size1)
        coverage += area
        balance_penalty += (abs(hole.x_m) / half_x + abs(hole.y_m) / half_y) / 2.0

    norm_area = cell_x * cell_y
    coverage_ratio = min(coverage / max(norm_area, 1e-9), 1.5)
    balance = max(1.0 - balance_penalty / max(len(hole_list), 1), -1.0)
    return 0.7 * coverage_ratio + 0.3 * balance


def _range_penalty(value: float, allowed: Tuple[float, float]) -> float:
    low, high = allowed
    if value < low:
        return (low - value) / max(high - low, 1e-9)
    if value > high:
        return (value - high) / max(high - low, 1e-9)
    return 0.0


def _uniform(boundaries: Tuple[float, float], rng: random.Random) -> float:
    low, high = boundaries
    return rng.uniform(low, high)

