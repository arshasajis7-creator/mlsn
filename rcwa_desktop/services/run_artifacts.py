"""Utility helpers for exporting per-run artefacts (SVG plots and geometry layouts)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List

from ..models.configuration import Configuration


def write_reflection_loss_svg(freq: Iterable[float], rl: Iterable[float], destination: Path) -> None:
    freq_list = list(freq)
    rl_list = list(rl)
    if not freq_list or not rl_list:
        return

    width, height = 840, 480
    margin_left, margin_right = 80, 30
    margin_top, margin_bottom = 40, 70

    min_x = min(freq_list)
    max_x = max(freq_list)
    min_y = min(rl_list)
    max_y = max(rl_list)

    if math.isclose(max_x, min_x):
        max_x = min_x + 1.0
    if math.isclose(max_y, min_y):
        max_y = min_y + 1.0

    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    def map_x(value: float) -> float:
        return margin_left + (value - min_x) / (max_x - min_x) * plot_width

    def map_y(value: float) -> float:
        return margin_top + plot_height - (value - min_y) / (max_y - min_y) * plot_height

    polyline_points = " ".join(f"{map_x(x):.2f},{map_y(y):.2f}" for x, y in zip(freq_list, rl_list))

    grid_lines: List[str] = []
    tick_count = 4
    for i in range(tick_count + 1):
        gx = margin_left + plot_width * i / tick_count
        grid_lines.append(
            f'<line x1="{gx:.2f}" y1="{margin_top:.2f}" x2="{gx:.2f}" y2="{(margin_top + plot_height):.2f}" '
            'stroke="#d0d0d0" stroke-width="1" />'
        )
        tick_freq = min_x + (max_x - min_x) * i / tick_count
        grid_lines.append(
            f'<text x="{gx:.2f}" y="{(margin_top + plot_height + 25):.2f}" font-size="14" text-anchor="middle" '
            f'fill="#444">{tick_freq:.2f}</text>'
        )

        gy = margin_top + plot_height * i / tick_count
        grid_lines.append(
            f'<line x1="{margin_left:.2f}" y1="{gy:.2f}" x2="{(margin_left + plot_width):.2f}" y2="{gy:.2f}" '
            'stroke="#e0e0e0" stroke-width="1" />'
        )
        tick_rl = max_y - (max_y - min_y) * i / tick_count
        grid_lines.append(
            f'<text x="{(margin_left - 10):.2f}" y="{gy + 5:.2f}" font-size="14" text-anchor="end" '
            f'fill="#444">{tick_rl:.2f}</text>'
        )

    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>
  {' '.join(grid_lines)}
  <rect x="{margin_left}" y="{margin_top}" width="{plot_width}" height="{plot_height}"
        fill="none" stroke="#222" stroke-width="2"/>
  <polyline points="{polyline_points}" fill="none" stroke="#1976d2" stroke-width="3"/>
  <text x="{width / 2:.2f}" y="{margin_top / 2:.2f}" font-size="18" text-anchor="middle" fill="#222">
    Reflection Loss vs Frequency
  </text>
  <text x="{width / 2:.2f}" y="{height - margin_bottom / 3:.2f}" font-size="16" text-anchor="middle" fill="#222">
    Frequency (GHz) [{min_x:.2f} - {max_x:.2f}]
  </text>
  <text x="{margin_left / 2:.2f}" y="{height / 2:.2f}" font-size="16" text-anchor="middle" fill="#222"
        transform="rotate(-90 {margin_left / 2:.2f},{height / 2:.2f})">
    Reflection Loss (dB) [{min_y:.2f} - {max_y:.2f}]
  </text>
</svg>
"""
    destination.write_text(svg, encoding="utf-8")


def write_mask_svg(config: Configuration, destination: Path) -> None:
    cell = config.cell
    mask = getattr(config, "mask", None)
    if mask is None:
        return

    Lx = cell.Lx_m or 1.0
    Ly = cell.Ly_m or 1.0

    width, height = 720, 720
    margin = 60
    inner_width = width - 2 * margin
    inner_height = height - 2 * margin

    def map_x(value: float) -> float:
        return margin + (value + Lx / 2.0) / Lx * inner_width

    def map_y(value: float) -> float:
        return margin + (Ly / 2.0 - value) / Ly * inner_height

    elements: List[str] = [
        f'<rect x="{margin}" y="{margin}" width="{inner_width}" height="{inner_height}" '
        'fill="#f5f5f5" stroke="#333" stroke-width="2"/>'
    ]

    for hole in mask.holes:
        cx = map_x(hole.x_m)
        cy = map_y(hole.y_m)
        rotation = getattr(hole, "rotation_deg", 0.0) or 0.0
        transform = f' transform="rotate({rotation:.2f} {cx:.2f} {cy:.2f})"' if rotation else ""

        fill_style = 'fill="#1976d2" fill-opacity="0.55" stroke="#0d47a1" stroke-width="1.5"'

        if hole.shape in {"square", "rectangle"}:
            width_m = hole.size1
            height_m = hole.size2 if hole.size2 is not None else hole.size1
            width_px = (width_m / Lx) * inner_width
            height_px = (height_m / Ly) * inner_height
            x = cx - width_px / 2.0
            y = cy - height_px / 2.0
            elements.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{width_px:.2f}" height="{height_px:.2f}" '
                f'{fill_style}{transform}/>'
            )
        else:
            if hole.shape == "ellipse":
                axis_a = hole.size1 / 2.0
                axis_b = (hole.size2 if hole.size2 is not None else hole.size1) / 2.0
                rx = (axis_a / Lx) * inner_width
                ry = (axis_b / Ly) * inner_height
            else:  # circle (default)
                radius = hole.size1 / 2.0
                rx = (radius / Lx) * inner_width
                ry = (radius / Ly) * inner_height
            elements.append(
                f'<ellipse cx="{cx:.2f}" cy="{cy:.2f}" rx="{rx:.2f}" ry="{ry:.2f}" '
                f'{fill_style}{transform}/>'
            )

    title = (
        f"L2 Mask Layout - Cell {Lx * 1e3:.1f}x{Ly * 1e3:.1f} mm "
        f"({len(mask.holes)} holes)"
    )

    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>
  <text x="{width / 2:.2f}" y="{margin / 2:.2f}" font-size="18" text-anchor="middle" fill="#222">{title}</text>
  {" ".join(elements)}
</svg>
"""
    destination.write_text(svg, encoding="utf-8")
