# RCWA Desktop UI (Stage 3)

Standalone PyQt6 application for configuring and launching RCWA simulations.

## Features

- General tab: frequency sweep, cell dimensions, output prefix with validation.
- Materials tab: select CSV datasets and thickness (mm) for L1, mask (solid/hole) and L3.
- Geometry tab: editable table of mask holes (circle or square) with coordinates in millimeters.
- Execution tab: run the existing `adapter_step1.py`, view stdout/stderr logs, and visualize Reflection Loss (dB) vs frequency immediately inside the UI.
- Load / save configuration JSON compatible with the RCWA adapter.

## Requirements

```bash
pip install PyQt6 matplotlib
```

## Running

```bash
python main.py
```

This is the first fully functional milestone. Future enhancements can add batch sweeps, job history, and advanced geometry primitives.
