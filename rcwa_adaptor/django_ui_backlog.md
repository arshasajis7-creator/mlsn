---
title: "RCWA Django UI Backlog"
status: "Draft"
author: "Codex"
version: "2025-10-21"
---

## 1. Vision

Create a production-grade Django web application that wraps the existing RCWA adapter tooling. The UI should let RF engineers specify multilayer stacks, periodic mask geometries, material datasets, and sweep parameters, launch simulations, and inspect/compare results (reflection, transmission, absorption, RL in dB). The interface must enforce physical consistency (realistic units, energy conservation checks) and support both built‑in geometries (circles, squares, crosses, rings) and custom polygon uploads.

## 2. Key Capabilities

1. **Simulation Configuration**
   - Define multilayer stacks with homogeneous layers (m1, m2, m3, PEC, free space).
   - Configure periodic mask layers with shape primitives (circle, square, rectangle, plus, cross, arbitrary polygon).
   - Specify cell dimensions, harmonic orders, FFT grid resolution.
   - Set source properties (frequency sweep, angle, polarization TE/TM, combined average).
   - Toggle strict energy enforcement and tolerances.

2. **Geometry Authoring**
   - Visual wizard to place shapes in the unit cell (drag & drop, numeric entry).
   - Shape library with parameters (radius, width/height, rotation).
   - Support for composite shapes (e.g., multi‑ring).
   - Import/export CSV or JSON mask definitions.

3. **Execution & Job Management**
   - Queue simulations (run sequentially to avoid CPU oversubscription).
   - Track job status (Pending → Running → Completed/Failed) with timestamps.
   - Display progress per job (currently simulated frequency index).
   - Manage logs (stdout/stderr, warnings).

4. **Results & Analysis**
   - Store CSV outputs in database (or filesystem with metadata references).
   - Provide plots (RL vs frequency, R/T/A) via Plotly/Matplotlib integration.
   - Compare multiple runs side‑by‑side (diff charts, table of minima & maxima).
   - Export results (CSV, JSON, PDF summary).

5. **User Experience**
   - Authentication with role-based permissions (Engineer, Reviewer, Admin).
   - Project templates (e.g., “two-layer absorber”, “three-hole mask”).
   - History view with filters (date range, RL threshold).
   - Documentation panel with physical guidelines (e.g., meaning of RL<−10 dB).

6. **Scientific Accuracy**
   - Validate units (mm vs m) before launch.
   - Ensure energy checks (R+T+A≈1) reported clearly.
   - Provide warnings for unrealistic grids/harmonics (e.g., Nx,Ny > 128 on low-spec hardware).
   - Integrate with existing analyzer to highlight worst-case frequencies.

## 3. Architecture Overview

```
├── django_rcwa/
│   ├── config/           # Django settings, Celery config
│   ├── materials/        # Models for CSV datasets, upload validators
│   ├── geometries/       # Shape primitives, serializers
│   ├── simulations/      # Core job models, adapters to RCWA CLI
│   ├── results/          # Data storage, plotting services
│   ├── accounts/         # Auth, permissions
│   ├── ui/               # Templates, React/Vue components (optional)
│   ├── api/              # DRF endpoints for async client
│   └── management/       # Commands to run sweeps, clean jobs
```

Backend: Django + Django REST Framework.  
Task queue: Celery (with Redis broker) to serialize RCWA runs.  
Frontend: Django templates or SPA (React) for geometry editor + dashboards.  
Storage: PostgreSQL for metadata, filesystem (or S3) for CSV/JSONL outputs.  
Plotting: Plotly or Matplotlib (served via rest endpoints returning JSON data).

## 4. Detailed Backlog

### 4.1 Project Bootstrapping (Week 1)
1. Create Django project `django_rcwa`, configure PostgreSQL connection.
2. Add `accounts` app with custom user model, login/logout templates.
3. Integrate Tailwind or Bootstrap for baseline styling.
4. Configure `.env` management (django-environ).
5. Setup Celery worker with Redis and base task for dummy RCWA call.

### 4.2 Materials Module (Week 2)
1. Model: `MaterialDataset` (name, description, CSV file path, type=permittivity/permeability, units).
2. File upload validation (columns freq_GHz, eps_real, eps_imag, mu_real, mu_imag).
3. Admin interface to manage datasets (list, detail, edit).
4. REST API endpoint to list datasets.
5. CLI command to import existing `m1.csv`, `m2.csv`, `m3.csv`, `metal.csv`.

### 4.3 Geometry Module (Weeks 2-3)
1. Shape models:
   - `CircleHole`: center (x,y), radius.
   - `RectangleHole`: center (x,y), width, height, rotation.
   - `PolygonHole`: list of vertices (JSON).
   - `CompoundHole`: references to sub-shapes (for plus/cross).
2. Utility to discretize shapes onto grid (Nx, Ny) producing boolean mask.
3. Visual editor (phase 1):
   - Template enabling numeric entry.
   - Preview using Canvas/Plotly grid.
4. Validation to ensure shapes stay within cell bounds.
5. Export/Import geometry definitions (JSON).

### 4.4 Simulation Model (Weeks 3-4)
1. Django models:
   - `Layer` (order, material, thickness).
   - `MaskLayer` (linked to geometry).
   - `SimulationJob` (slug, user, status, created_at, config JSON, cel_task_id).
2. Serializer to translate Job -> `step3_example_*.json` structure.
3. Celery task `run_simulation(job_id)`:
   - Write config JSON to temp file.
   - Invoke CLI `python adapter_step1.py <config>` (TE/TM sequential).
   - Capture stdout/stderr and result files.
   - Update job status + attach results.
4. Admin interface to inspect jobs (status, logs).

### 4.5 UI Screens (Weeks 4-6)
1. **Dashboard**: list jobs, status badges, filters, create button.
2. **Simulation Builder**:
   - Step 1: Source (freq range, angle, polarization).
   - Step 2: Layers (drag reorder, select material, thickness).
   - Step 3: Mask (choose template: none/circle/square/plus/custom; open geometry editor).
   - Step 4: Advanced (harmonics, grid, tolerances).
   - Step 5: Review & Launch.
3. **Job Detail**:
   - Summary (config, materials).
   - Progress log (live via WebSockets or AJAX).
   - Result actions (view charts, download CSV/JSONL).
4. **Geometry Editor UI**:
   - Canvas with 2D preview.
   - Add shape button for circle, rectangle, polygon.
   - Numeric overlay for precise coordinates.
   - Snap-to-grid toggle.
   - Save geometry to `MaskLayer`.

### 4.6 Result Visualization (Weeks 6-7)
1. Plot RL (dB), R/T/A vs frequency using Plotly (line charts).
2. Provide table with minima & maxima (R_min, RL_min, freq at R_min).
3. Option to overlay multiple runs for comparison (checklist).
4. PDF report generation (results summary + charts).
5. API endpoints to fetch chart data (JSON) to power front-end.

### 4.7 Advanced Features (Week 8+)
1. Batch sweeps (thickness sweeps, angle sweeps) triggered from UI.
2. Scheduling recurring runs (daily/weekly).
3. User-defined script uploads (Python plugin to modify config).
4. Multi-cell supercell support.
5. Integration with HPC queue (SLURM/LSF) for heavy runs.

### 4.8 Validation & Testing
1. Unit tests for geometry rasterization (shapes -> boolean grid).
2. Unit tests for config serializer (Job -> JSON -> results).
3. Integration tests for Celery task (mock RCWA CLI).
4. End-to-end tests (Selenium/Cypress) for job creation & result view.
5. Performance tests (simulate sequential jobs to ensure queue fairness).

## 5. Timeline Estimate (Aggressive)

| Sprint | Duration | Focus Areas |
|--------|----------|-------------|
| Sprint 1 | Week 1 | Project scaffolding, accounts, Celery |
| Sprint 2 | Week 2 | Materials module, base database models |
| Sprint 3 | Weeks 3-4 | Simulation job pipeline, CLI integration |
| Sprint 4 | Weeks 4-6 | UI for job builder, geometry editor MVP |
| Sprint 5 | Weeks 6-7 | Result visualization, comparison tools |
| Sprint 6 | Week 8 | Sweeps, advanced features, polishing |

Total ~8 weeks with 2 developers (backend + frontend) plus part-time RF engineer for validation.

## 6. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Complex geometry rasterization causing runtime slowdown | High | Precompute masks, cache grids, limit Nx,Ny with warnings |
| Large CSV outputs slow database | Medium | Store CSVs on filesystem/S3, only metadata in DB |
| RCWA CLI failures (e.g., strict energy) | Medium | Capture stderr, surface to UI, provide debug tools |
| User confusion between mm/m units | Medium | Enforce input units, show conversions, highlight errors |
| Task queue overload on low-spec machine | Medium | Serialize runs by default, allow concurrency toggle |

## 7. Dependencies

- Python 3.11+
- Django 4.x
- Django REST Framework
- Celery + Redis
- Plotly / Matplotlib
- pandas, numpy (for shapely integration if needed)
- TailwindCSS or Bootstrap for UI
- Optional: shapely, geojson for advanced polygon support.

## 8. Next Steps

1. Review backlog with stakeholders (RF engineers) to confirm priorities.
2. Decide on frontend strategy (Django templates vs React/Vue SPA).
3. Set up new repository `django-rcwa-ui` and initialize Sprint 1 tasks.
4. Prepare environment (Docker Compose with PostgreSQL, Redis).
5. Kick off Sprint 1 with scaffolding stories.

---

> **Note:** This backlog is intentionally detailed (~30 KB) so implementation can start immediately. Adjust priority/sequencing based on team capacity and available hardware (e.g., queue concurrency may be limited on low-spec machines like HP Z230).
