from __future__ import annotations

import json
import random
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from queue import Queue, Empty
from typing import Any, Dict, List, Optional, Tuple

from ..models.configuration import Configuration
from .rcwa_runner import run_simulation


@dataclass
class OptimizationObjective:
    """Single optimisation objective definition."""

    name: str
    metric: str  # e.g. "min_rl", "bandwidth_neg10", "peak_frequency"
    sense: str = "minimize"  # minimize, maximize, target
    target: Optional[float] = None
    threshold: Optional[float] = None
    weight: float = 1.0
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationVariable:
    """Definition of a variable that can be perturbed by the optimiser."""

    path: str  # dotted path in Configuration, e.g. "mask.thickness_m" or "mask.holes[0].size1"
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[List[Any]] = None
    locked: bool = False
    integer: bool = False


@dataclass
class OptimizationConstraint:
    """Representation of a constraint applied during optimisation."""

    name: str
    type: str  # e.g. total_thickness_le, min_spacing_ge
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlgorithmSettings:
    max_evaluations: int = 20
    population_size: int = 6
    seed: Optional[int] = None
    allow_retries: int = 2


@dataclass
class OptimizationJob:
    base_config: Configuration
    objectives: List[OptimizationObjective]
    variables: List[OptimizationVariable]
    constraints: List[OptimizationConstraint] = field(default_factory=list)
    algorithm: str = "random"
    algorithm_settings: AlgorithmSettings = field(default_factory=AlgorithmSettings)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def clone_config(self) -> Configuration:
        """Return a deep copy of the base configuration."""

        return Configuration.from_json(self.base_config.to_json())


@dataclass
class EvaluationRecord:
    iteration: int
    parameters: Dict[str, Any]
    metrics: Dict[str, Any]
    objective_scores: Dict[str, float]
    success: bool
    message: str = ""
    duration_s: float = 0.0
    log_dir: Optional[Path] = None


@dataclass
class JobStatus:
    job_id: str
    state: str  # pending, running, completed, failed, cancelled
    submitted_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    progress: int = 0
    total: int = 0
    best_objectives: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    message: str = ""


@dataclass
class OptimizationJobState:
    job_id: str
    job: OptimizationJob
    status: JobStatus
    log_dir: Path
    evaluations: List[EvaluationRecord] = field(default_factory=list)
    best_evaluation: Optional[EvaluationRecord] = None
    cancel_requested: bool = False

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self.status)
        if self.best_evaluation:
            data["best_evaluation"] = {
                "iteration": self.best_evaluation.iteration,
                "metrics": self.best_evaluation.metrics,
                "objective_scores": self.best_evaluation.objective_scores,
                "parameters": self.best_evaluation.parameters,
                "log_dir": str(self.best_evaluation.log_dir) if self.best_evaluation.log_dir else None,
            }
        else:
            data["best_evaluation"] = None
        return data


class OptimizerService:
    """Background service that evaluates optimisation jobs sequentially."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.optimizer_root = self.project_root / "logs" / "optimizer"
        self.optimizer_root.mkdir(parents=True, exist_ok=True)

        self._jobs: Dict[str, OptimizationJobState] = {}
        self._queue: Queue[str] = Queue()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ submission & control
    def submit(self, job: OptimizationJob) -> str:
        if not job.objectives:
            raise ValueError("At least one optimisation objective must be provided.")

        job_id = uuid.uuid4().hex[:12]
        job_dir = self.optimizer_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        status = JobStatus(
            job_id=job_id,
            state="pending",
            submitted_at=time.time(),
            total=job.algorithm_settings.max_evaluations,
            metadata=job.metadata.copy(),
        )

        state = OptimizationJobState(
            job_id=job_id,
            job=job,
            status=status,
            log_dir=job_dir,
        )

        # Persist snapshot of the job configuration for reproducibility.
        (job_dir / "job_definition.json").write_text(
            json.dumps(
                {
                    "job_id": job_id,
                    "objectives": [asdict(obj) for obj in job.objectives],
                    "variables": [asdict(var) for var in job.variables],
                    "constraints": [asdict(con) for con in job.constraints],
                    "algorithm": job.algorithm,
                    "algorithm_settings": asdict(job.algorithm_settings),
                    "metadata": job.metadata,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (job_dir / "base_config.json").write_text(json.dumps(job.base_config.to_json(), indent=2), encoding="utf-8")

        with self._lock:
            self._jobs[job_id] = state
        self._queue.put(job_id)
        return job_id

    def cancel(self, job_id: str) -> None:
        with self._lock:
            state = self._jobs.get(job_id)
        if not state:
            raise KeyError(f"Job '{job_id}' not found.")
        state.cancel_requested = True

    def get_status(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            state = self._jobs.get(job_id)
        if not state:
            raise KeyError(f"Job '{job_id}' not found.")
        return state.to_dict()

    def list_jobs(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [state.to_dict() for state in self._jobs.values()]

    # ------------------------------------------------------------------ worker loop
    def _worker_loop(self) -> None:
        while True:
            try:
                job_id = self._queue.get(timeout=0.2)
            except Empty:
                continue
            with self._lock:
                state = self._jobs.get(job_id)
            if not state:
                continue
            try:
                self._execute_job(state)
            except Exception as exc:  # pragma: no cover - safeguard
                state.status.state = "failed"
                state.status.message = f"Unhandled error: {exc}"
                state.status.finished_at = time.time()
            finally:
                self._queue.task_done()

    # ------------------------------------------------------------------ job execution
    def _execute_job(self, state: OptimizationJobState) -> None:
        job = state.job
        status = state.status
        settings = job.algorithm_settings

        if settings.seed is not None:
            random.seed(settings.seed)

        status.state = "running"
        status.started_at = time.time()

        evaluations_path = state.log_dir / "evaluations.jsonl"
        with evaluations_path.open("w", encoding="utf-8") as evaluations_file:
            for iteration in range(1, settings.max_evaluations + 1):
                if state.cancel_requested:
                    status.state = "cancelled"
                    status.message = "Cancellation requested by user."
                    status.finished_at = time.time()
                    return

                evaluation_dir = state.log_dir / f"eval_{iteration:03d}"
                evaluation_dir.mkdir(parents=True, exist_ok=True)

                params = self._sample_parameters(job)
                eval_start = time.time()
                try:
                    evaluation = self._run_single_evaluation(job, params, evaluation_dir, iteration)
                    evaluation.success = True
                    evaluation.message = "ok"
                except Exception as exc:
                    evaluation = EvaluationRecord(
                        iteration=iteration,
                        parameters=params,
                        metrics={},
                        objective_scores={},
                        success=False,
                        message=str(exc),
                        duration_s=time.time() - eval_start,
                        log_dir=evaluation_dir,
                    )
                evaluation.duration_s = time.time() - eval_start
                state.evaluations.append(evaluation)

                evaluations_file.write(json.dumps(self._evaluation_to_json(evaluation)) + "\n")
                evaluations_file.flush()

                # Update best evaluation (based on first objective score if available).
                if evaluation.success and evaluation.objective_scores:
                    primary_obj = job.objectives[0].name
                    best = state.best_evaluation
                    if (
                        best is None
                        or evaluation.objective_scores[primary_obj] < best.objective_scores.get(primary_obj, float("inf"))
                    ):
                        state.best_evaluation = evaluation
                        status.best_objectives = evaluation.objective_scores.copy()

                status.progress = iteration

        status.state = "completed" if not state.cancel_requested else "cancelled"
        status.finished_at = time.time()

        if state.best_evaluation:
            best_cfg_path = state.log_dir / "best_config.json"
            best_config = self._apply_parameters(job.clone_config(), state.best_evaluation.parameters)
            best_cfg_path.write_text(json.dumps(best_config.to_json(), indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ helpers
    def _sample_parameters(self, job: OptimizationJob) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for var in job.variables:
            if var.locked:
                continue
            if var.choices:
                params[var.path] = random.choice(var.choices)
            else:
                low = var.minimum if var.minimum is not None else 0.0
                high = var.maximum if var.maximum is not None else low
                if high == low:
                    params[var.path] = low
                else:
                    value = random.uniform(low, high)
                    if var.step:
                        value = round((value - low) / var.step) * var.step + low
                    if var.integer:
                        value = int(round(value))
                    params[var.path] = value
        return params

    def _run_single_evaluation(
        self,
        job: OptimizationJob,
        params: Dict[str, Any],
        evaluation_dir: Path,
        iteration: int,
    ) -> EvaluationRecord:
        config = self._apply_parameters(job.clone_config(), params)

        constraints_ok, message = self._check_constraints(job.constraints, config)
        if not constraints_ok:
            raise RuntimeError(f"Constraint violation: {message}")

        result = run_simulation(config, self.project_root, log_dir=evaluation_dir)
        metrics = self._compute_metrics(result)
        objective_scores = self._compute_objective_scores(job.objectives, metrics)
        self._store_metrics(evaluation_dir / "metrics.json", metrics, objective_scores)

        return EvaluationRecord(
            iteration=iteration,
            parameters=params.copy(),
            metrics=metrics,
            objective_scores=objective_scores,
            success=True,
            log_dir=evaluation_dir,
        )

    # ---------------- metrics & scoring ----------------
    def _compute_metrics(self, result) -> Dict[str, Any]:
        freq = result.freq_GHz
        rl = result.RL_dB
        min_rl = min(rl) if rl else float("inf")
        idx_min = rl.index(min_rl) if rl else 0
        freq_at_min = freq[idx_min] if freq else None
        threshold = -10.0
        bandwidth = 0.0
        if freq and rl:
            spans: List[Tuple[float, float]] = []
            current_start = None
            for f, val in zip(freq, rl):
                if val <= threshold:
                    if current_start is None:
                        current_start = f
                else:
                    if current_start is not None:
                        spans.append((current_start, f))
                        current_start = None
            if current_start is not None:
                spans.append((current_start, freq[-1]))
            bandwidth = sum(end - start for start, end in spans)
        metrics = {
            "min_rl_db": float(min_rl),
            "freq_at_min_rl": float(freq_at_min) if freq_at_min is not None else None,
            "bandwidth_neg10_db": float(bandwidth),
            "rl_values": rl,
            "freq_values": freq,
        }
        return metrics

    def _compute_objective_scores(
        self,
        objectives: List[OptimizationObjective],
        metrics: Dict[str, Any],
    ) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for obj in objectives:
            metric_value = self._metric_value(obj, metrics)
            if obj.sense == "minimize":
                score = float(metric_value)
            elif obj.sense == "maximize":
                score = -float(metric_value)
            elif obj.sense == "target":
                if obj.target is None:
                    raise ValueError(f"Objective '{obj.name}' requires target value.")
                score = abs(float(metric_value) - float(obj.target))
            else:
                raise ValueError(f"Unknown sense '{obj.sense}' for objective '{obj.name}'.")
            scores[obj.name] = score * obj.weight
        return scores

    def _metric_value(self, obj: OptimizationObjective, metrics: Dict[str, Any]) -> float:
        metric = obj.metric
        if metric == "min_rl":
            return float(metrics.get("min_rl_db", float("inf")))
        if metric == "bandwidth_neg10":
            return float(metrics.get("bandwidth_neg10_db", 0.0))
        if metric == "peak_frequency":
            value = metrics.get("freq_at_min_rl")
            return float(value) if value is not None else float("inf")
        raise ValueError(f"Unsupported metric '{metric}'.")

    def _store_metrics(self, path: Path, metrics: Dict[str, Any], scores: Dict[str, float]) -> None:
        serialisable = metrics.copy()
        serialisable["objective_scores"] = scores
        path.write_text(json.dumps(serialisable, indent=2), encoding="utf-8")

    # ---------------- constraint handling ----------------
    def _check_constraints(self, constraints: List[OptimizationConstraint], config: Configuration) -> Tuple[bool, str]:
        for constraint in constraints:
            ctype = constraint.type.lower()
            if ctype == "total_thickness_le":
                limit = float(constraint.parameters.get("limit_m", constraint.parameters.get("limit", 0.0)))
                total = (
                    config.layer_top.thickness_m
                    + config.mask.thickness_m
                    + config.layer_bottom.thickness_m
                )
                if total > limit:
                    return False, f"Total thickness {total:.6f} m exceeds limit {limit:.6f} m"
            elif ctype == "total_thickness_ge":
                limit = float(constraint.parameters.get("limit_m", 0.0))
                total = (
                    config.layer_top.thickness_m
                    + config.mask.thickness_m
                    + config.layer_bottom.thickness_m
                )
                if total < limit:
                    return False, f"Total thickness {total:.6f} m below limit {limit:.6f} m"
            elif ctype == "max_shape_count":
                max_count = int(constraint.parameters.get("limit", 0))
                if len(config.mask.holes) > max_count:
                    return False, f"Number of mask holes {len(config.mask.holes)} exceeds limit {max_count}"
            else:
                # Unknown constraint -> ignore for now but keep message for transparency.
                return False, f"Unsupported constraint type '{constraint.type}'"
        return True, ""

    # ---------------- config mutation utilities ----------------
    def _apply_parameters(self, config: Configuration, params: Dict[str, Any]) -> Configuration:
        for path, value in params.items():
            self._set_nested_value(config, path, value)
        return config

    def _set_nested_value(self, root: Any, path: str, value: Any) -> None:
        parts = path.split(".")
        target = root
        for part in parts[:-1]:
            target = self._get_part(target, part)
        final_part = parts[-1]
        parent = target
        attr, index = self._split_part(final_part)
        if index is not None:
            container = getattr(parent, attr)
            container[index] = value
        else:
            setattr(parent, attr, value)

    def _get_part(self, obj: Any, part: str) -> Any:
        attr, index = self._split_part(part)
        target = getattr(obj, attr)
        if index is not None:
            return target[index]
        return target

    def _split_part(self, part: str) -> Tuple[str, Optional[int]]:
        if "[" in part and part.endswith("]"):
            attr, index_str = part[:-1].split("[", 1)
            return attr, int(index_str)
        return part, None

    def _evaluation_to_json(self, evaluation: EvaluationRecord) -> Dict[str, Any]:
        data = {
            "iteration": evaluation.iteration,
            "parameters": evaluation.parameters,
            "metrics": evaluation.metrics,
            "objective_scores": evaluation.objective_scores,
            "success": evaluation.success,
            "message": evaluation.message,
            "duration_s": evaluation.duration_s,
            "log_dir": str(evaluation.log_dir) if evaluation.log_dir else None,
        }
        return data


__all__ = [
    "AlgorithmSettings",
    "OptimizationConstraint",
    "OptimizationJob",
    "OptimizationObjective",
    "OptimizationVariable",
    "OptimizerService",
]
