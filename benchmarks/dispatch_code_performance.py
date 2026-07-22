#!/usr/bin/env python3
"""Benchmark bounded DRYML code-analysis, probe, and dispatch scenarios."""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from importlib import import_module
from importlib import metadata
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "fixtures"
if str(FIXTURE_DIR) not in sys.path:
    sys.path.insert(0, str(FIXTURE_DIR))

import probe_targets  # noqa: E402

import dryml.code as code  # noqa: E402
from dryml import annotations, environments  # noqa: E402
from dryml.core import Definition  # noqa: E402
from dryml.core.repo import Repo  # noqa: E402
from dryml.core.store.dir import DirStore  # noqa: E402
from dryml.dispatch import Dispatcher  # noqa: E402
from dryml.environments import CurrentEnvironmentSpec, PythonExecutableSpec  # noqa: E402
from dryml.formats.canonical import json_ready  # noqa: E402
from dryml.worlds import LocalResourceInventory  # noqa: E402


SCHEMA = "dryml.dispatch_code_performance.v1"
SCHEMA_VERSION = 1
PURE_DEFAULT = 20
PURE_MAX = 1000
MANAGED_DEFAULT = 5
MANAGED_MAX = 100
RESULT_LIMIT_BYTES = 32 * 1024 * 1024
DEPENDENCIES = ("dryml", "dill", "numpy", "packaging")


@dataclass(frozen=True)
class Observation:
    metrics: dict[str, Any] = field(default_factory=dict)
    operations: dict[str, int] = field(default_factory=dict)
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)
    cleanup: str = "complete"


@dataclass(frozen=True)
class Scenario:
    name: str
    sample_class: str
    run: Callable[["BenchmarkContext"], Observation]
    input: dict[str, Any] = field(default_factory=dict)

    @property
    def managed(self) -> bool:
        return self.sample_class == "managed_process"


class BenchmarkContext:
    """Own temporary persistence and deterministic dispatch inputs."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.store = DirStore(root / "store", query_index="none")
        self.repo = Repo(stores=[self.store])
        self.inventory = LocalResourceInventory(
            cpus=(0, 1, 2, 3),
            memory=2 * 1024**3,
            metadata={"source": "benchmark"},
        )
        self.environment = CurrentEnvironmentSpec()
        self.python_environment = PythonExecutableSpec(
            sys.executable,
            pythonpath_policy="explicit",
            extra_pythonpath=(str(FIXTURE_DIR),),
        )
        self.single_world = {
            "roles": {"main": {"replicas": 1, "process": {"resources": {"cpus": 1}}}},
        }
        self.world_one = {
            "worker": {"replicas": 1, "process": {"resources": {"cpus": 1}}},
        }
        self.world_four = {
            "worker": {"replicas": 4, "process": {"resources": {"cpus": 1}}},
        }
        self.dispatcher = Dispatcher(store=self.store, inventory=self.inventory)
        obj = probe_targets.BenchmarkObject(repo=self.repo)
        self.repo.save(obj, store=self.store, record_policy="none")
        self.cdef = obj.definition

    def dispatch_kwargs(self) -> dict[str, Any]:
        return {
            "environment": self.environment,
            "world": self.single_world,
            "inventory": self.inventory,
            "requirement_policy": "strict",
        }


class CallCounters:
    """Temporarily wrap stable seams and restore them after one sample."""

    def __init__(self) -> None:
        self.values: Counter[str] = Counter()
        self._stack = ExitStack()

    def wrap(self, owner: Any, attribute: str, counter: str) -> None:
        original = getattr(owner, attribute)

        @functools.wraps(original)
        def counted(*args, **kwargs):
            self.values[counter] += 1
            return original(*args, **kwargs)

        setattr(owner, attribute, counted)
        self._stack.callback(setattr, owner, attribute, original)

    def __enter__(self) -> "CallCounters":
        return self

    def __exit__(self, *exc_info) -> None:
        self._stack.close()


def positive_sample_count(value: Any, *, maximum: int, name: str) -> int:
    """Validate one programmatic or CLI sample count."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive base-10 integer")
    if isinstance(value, str):
        if not value or value.strip() != value or not value.isascii() or not value.isdecimal():
            raise ValueError(f"{name} must be a positive base-10 integer")
        value = int(value, 10)
    if not isinstance(value, int) or value <= 0 or value > maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")
    return value


def percentile_nearest_rank(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _encoded_size(value: Any) -> int:
    return len(json.dumps(json_ready(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def _analysis_observation(target: Any, algorithms: tuple[str, ...] | None = None) -> Observation:
    source_module = import_module("dryml.code.algorithms.source")
    ast_access_module = import_module("dryml.code.algorithms.ast_access")
    static_calls_module = import_module("dryml.code.algorithms.static_calls")
    static_analysis_module = import_module("dryml.code.algorithms.static_analysis")
    with CallCounters() as counters:
        counters.wrap(source_module, "get_source_info", "source_extractions")
        counters.wrap(ast_access_module, "get_source_info", "source_extractions")
        counters.wrap(static_calls_module, "get_source_info", "source_extractions")
        counters.wrap(static_analysis_module.ast, "parse", "ast_parses")
        result = code.analyze(target, algorithms=algorithms)
    data = result.to_data()
    return Observation(
        metrics={
            "facts": len(result.facts),
            "diagnostics": len(result.diagnostics),
            "serialized_bytes": _encoded_size(data),
        },
        operations={"analysis_calls": 1, **dict(counters.values)},
        stages={"analysis": {"available": True, "seconds": None}},
    )


def _fragment_observation(count: int) -> Observation:
    fragments = tuple(
        annotations.AnnotationFragment(
            "environment",
            "requirement",
            {"requirements": [f"benchmark-fragment-{index}>=1"]},
            annotations.SourceTrace("synthetic", label=f"fragment-{index}"),
        )
        for index in range(count)
    )
    result = annotations.resolve_fragments(fragments)
    return Observation(
        metrics={"fragments": len(result.fragments), "diagnostics": len(result.diagnostics)},
        operations={"fragment_resolutions": 1},
    )


def _probe_metrics(result: Any) -> dict[str, Any]:
    data = result.to_data()
    analysis = result.analysis
    return {
        "ok": result.ok,
        "facts": len(analysis.facts) if analysis is not None else 0,
        "diagnostics": len(result.diagnostics),
        "serialized_bytes": _encoded_size(data),
        "environment_record": result.environment_record is not None,
    }


def _inline_probe(_context: BenchmarkContext) -> Observation:
    result = code.probe_target(probe_targets.plain_function, include_environment_record=False)
    return Observation(metrics=_probe_metrics(result), operations={"code_probe_calls": 1, "managed_process_launches": 0})


def _managed_code_probe(context: BenchmarkContext, *, include_environment: bool, explicit: bool) -> Observation:
    probe_module = import_module("dryml.code.probe")

    with CallCounters() as counters:
        counters.wrap(probe_module, "_run_bounded_command", "managed_process_launches")
        target = (
            "probe_targets:plain_function"
            if explicit else "dryml.code.algorithms.static_calls:analyze_target"
        )
        result = code.probe_target(
            target,
            environment=context.python_environment if explicit else None,
            include_environment_record=include_environment,
            timeout=30.0,
        )
    if not result.ok:
        raise RuntimeError("managed code-probe benchmark did not complete successfully")
    return Observation(
        metrics=_probe_metrics(result),
        operations={"code_probe_calls": 1, **dict(counters.values)},
        stages={
            "managed_child_analysis": {"available": False, "seconds": None},
            "environment_introspection": {"available": False, "seconds": None},
        },
    )


def _environment_probe(context: BenchmarkContext) -> Observation:
    probe_module = import_module("dryml.environments.probe")

    with CallCounters() as counters:
        counters.wrap(probe_module, "_run_bounded_command", "managed_process_launches")
        result = environments.probe(context.environment, timeout=30.0)
    if not result.ok:
        raise RuntimeError("environment-probe benchmark did not complete successfully")
    data = result.to_data()
    return Observation(
        metrics={
            "ok": result.ok,
            "diagnostics": len(result.report.issues) if result.report is not None else 0,
            "serialized_bytes": _encoded_size(data),
        },
        operations={"environment_probe_calls": 1, **dict(counters.values)},
        stages={"environment_introspection": {"available": False, "seconds": None}},
    )


@contextmanager
def _dispatch_counters(*, managed: bool = False):
    import dryml.dispatch.requirements as requirements_module
    from dryml.dispatch.requirements import DispatchPlanningResolution

    code_probe_module = import_module("dryml.code.probe")
    environment_probe_module = import_module("dryml.environments.probe")

    with CallCounters() as counters:
        counters.wrap(requirements_module, "probe_target", "code_probe_calls")
        counters.wrap(requirements_module.environments, "probe", "environment_probe_calls")
        counters.wrap(DispatchPlanningResolution, "metadata", "planning_metadata_snapshots")
        if managed:
            counters.wrap(code_probe_module, "_run_bounded_command", "managed_process_launches")
            counters.wrap(environment_probe_module, "_run_bounded_command", "managed_process_launches")
        yield counters


def _dispatch_observation(context: BenchmarkContext, *, method: bool, explain: bool) -> Observation:
    target = context.cdef if method else probe_targets.plain_function
    args = (target, "ping") if method else (target,)
    with _dispatch_counters() as counters:
        if explain:
            result = context.dispatcher.explain(*args, **context.dispatch_kwargs())
            payload = result.to_data()
        else:
            result = context.dispatcher.plan(*args, record_policy="none", **context.dispatch_kwargs())
            payload = result.envelope.to_json()
    return Observation(
        metrics={"serialized_bytes": _encoded_size(payload), "launchable": True},
        operations=dict(counters.values),
    )


def _managed_dispatch_probe(context: BenchmarkContext, *, environment_resolution: bool) -> Observation:
    if environment_resolution:
        target = probe_targets.current_python_required_function
        kwargs = {
            "environment_candidates": (context.environment,),
            "world": context.single_world,
            "inventory": context.inventory,
            "requirement_policy": "strict",
        }
    else:
        target = probe_targets.plain_function
        kwargs = {
            "environment": context.python_environment,
            "world": context.single_world,
            "inventory": context.inventory,
            "requirement_policy": "strict",
        }
    with environments.use(None), _dispatch_counters(managed=True) as counters:
        result = context.dispatcher.plan(target, record_policy="none", **kwargs)
    return Observation(
        metrics={"serialized_bytes": _encoded_size(result.envelope.to_json()), "launchable": True},
        operations=dict(counters.values),
        stages={"managed_probe": {"available": False, "seconds": None}},
    )


def _trace_observation(target: Callable[..., Any], calls: int) -> Observation:
    context = code.CodeAnalysisContext(
        algorithms=("dynamic_trace",),
        allow_dynamic_execution=True,
        include_annotations=False,
        include_method_contracts=False,
    )
    args = () if calls == 0 else (Definition(probe_targets.BenchmarkTraceModel),)
    result = code.trace(target, args=args, context=context, policy=code.DynamicTracePolicy(max_calls=64))
    observed = len(result.facts_of_kind("dynamic_call"))
    if observed != calls:
        raise RuntimeError(f"trace scenario expected {calls} calls but observed {observed}")
    return Observation(
        metrics={"observed_calls": observed, "serialized_bytes": _encoded_size(result.to_data())},
        operations={"analysis_calls": 1, "dynamic_calls": observed},
    )


def _world_plan(context: BenchmarkContext, replicas: int) -> Observation:
    world = context.world_one if replicas == 1 else context.world_four
    with _dispatch_counters() as counters:
        plan = context.dispatcher.plan_world(
            platform.python_version,
            world=world,
            environment=context.environment,
            inventory=context.inventory,
            record_policy="none",
        )
    if len(plan.worker_plans) != replicas:
        raise RuntimeError("local-world benchmark planned the wrong worker count")
    payload = {
        "dispatch_spec": plan.dispatch_spec,
        "execution_recipe": plan.execution_recipe,
        "world_spec": plan.world_spec,
        "world_allocation_spec": plan.world_allocation_spec,
        "worker_envelopes": [worker.envelope.to_json() for worker in plan.worker_plans],
    }
    return Observation(
        metrics={"workers": len(plan.worker_plans), "serialized_bytes": _encoded_size(payload)},
        operations={"local_allocations": 1, **dict(counters.values)},
    )


def _worker_observation(context: BenchmarkContext) -> Observation:
    plan = context.dispatcher.plan(
        platform.python_version,
        environment=context.environment,
        world=context.single_world,
        inventory=context.inventory,
        record_policy="none",
    )
    started = time.perf_counter()
    future = context.dispatcher.submit(plan)
    handshake_seconds = time.perf_counter() - started
    work_dir = Path(future.work_dir)
    started = time.perf_counter()
    response = future.result(timeout=30.0)
    completion_seconds = time.perf_counter() - started
    if response.status != "ok":
        raise RuntimeError(f"worker benchmark failed with status {response.status}")
    cleanup = "complete" if not work_dir.exists() else "incomplete"
    if cleanup != "complete":
        raise RuntimeError("worker benchmark did not clean its temporary work directory")
    return Observation(
        metrics={"status": response.status},
        operations={"worker_launches": 1, "managed_process_launches": 1},
        stages={
            "submit_to_handshake": {"available": True, "seconds": handshake_seconds},
            "handshake_to_completion": {"available": True, "seconds": completion_seconds},
        },
        cleanup=cleanup,
    )


def scenarios() -> tuple[Scenario, ...]:
    pure = [
        Scenario("analysis.importable_default", "warm_in_process", lambda _c: _analysis_observation("probe_targets:plain_function"), {"target": "importable_function"}),
        Scenario("analysis.live_default", "warm_in_process", lambda _c: _analysis_observation(probe_targets.plain_function), {"target": "live_function"}),
        Scenario("analysis.bounded_large_default", "warm_in_process", lambda _c: _analysis_observation("dryml.code.algorithms.static_calls:analyze_target"), {"target": "bounded_large"}),
        Scenario("analysis.source_ast_static_calls", "warm_in_process", lambda _c: _analysis_observation("dryml.code.algorithms.static_calls:analyze_target", ("source", "ast_access", "static_calls")), {"algorithms": ["source", "ast_access", "static_calls"]}),
        Scenario("analysis.direct_annotations", "warm_in_process", lambda _c: _analysis_observation(probe_targets.decorated_function, ("direct_annotations",)), {"algorithms": ["direct_annotations"]}),
    ]
    pure.extend(
        Scenario(f"annotations.fragments_{count}", "warm_in_process", lambda _c, count=count: _fragment_observation(count), {"fragments": count})
        for count in (0, 1, 16, 64)
    )
    pure.extend([
        Scenario("probe.inline_no_environment", "warm_in_process", _inline_probe),
        Scenario("dispatch.function_explain_no_probe", "warm_in_process", lambda c: _dispatch_observation(c, method=False, explain=True)),
        Scenario("dispatch.function_plan_no_probe", "warm_in_process", lambda c: _dispatch_observation(c, method=False, explain=False)),
        Scenario("dispatch.cdef_explain_no_probe", "warm_in_process", lambda c: _dispatch_observation(c, method=True, explain=True)),
        Scenario("dispatch.cdef_plan_no_probe", "warm_in_process", lambda c: _dispatch_observation(c, method=True, explain=False)),
        Scenario("trace.zero", "warm_in_process", lambda _c: _trace_observation(probe_targets.trace_zero, 0)),
        Scenario("trace.one", "warm_in_process", lambda _c: _trace_observation(probe_targets.trace_one, 1)),
        Scenario("trace.repeated_16", "warm_in_process", lambda _c: _trace_observation(probe_targets.trace_repeated, 16)),
        Scenario("world.plan_one_worker", "warm_in_process", lambda c: _world_plan(c, 1)),
        Scenario("world.plan_four_workers", "warm_in_process", lambda c: _world_plan(c, 4)),
    ])
    managed = [
        Scenario("probe.current_python_no_environment", "managed_process", lambda c: _managed_code_probe(c, include_environment=False, explicit=False)),
        Scenario("probe.current_python_with_environment", "managed_process", lambda c: _managed_code_probe(c, include_environment=True, explicit=False)),
        Scenario("probe.python_executable", "managed_process", lambda c: _managed_code_probe(c, include_environment=False, explicit=True)),
        Scenario("probe.environment_only_current", "managed_process", _environment_probe),
        Scenario("dispatch.final_code_probe_plan", "managed_process", lambda c: _managed_dispatch_probe(c, environment_resolution=False)),
        Scenario("dispatch.environment_resolution_plan", "managed_process", lambda c: _managed_dispatch_probe(c, environment_resolution=True)),
        Scenario("worker.local_subprocess_trivial", "managed_process", _worker_observation),
    ]
    return tuple([*pure, *managed])


def _measure_scenario(scenario: Scenario, context: BenchmarkContext, samples: int) -> dict[str, Any]:
    warmup = None
    if not scenario.managed:
        started = time.perf_counter()
        scenario.run(context)
        warmup = {"wall_seconds": time.perf_counter() - started}
    measured = []
    for index in range(samples):
        started = time.perf_counter()
        observation = scenario.run(context)
        wall = time.perf_counter() - started
        measured.append({
            "index": index,
            "wall_seconds": wall,
            "stages": observation.stages,
            "operations": observation.operations,
            "metrics": observation.metrics,
            "cleanup": observation.cleanup,
        })
    durations = [sample["wall_seconds"] for sample in measured]
    return {
        "name": scenario.name,
        "sample_class": scenario.sample_class,
        "input": scenario.input,
        "warmup": warmup,
        "samples": measured,
        "summary": {
            "count": len(measured),
            "median_seconds": statistics.median(durations),
            "p95_seconds": percentile_nearest_rank(durations, 0.95),
            "percentile_method": "nearest_rank",
        },
    }


def _git_sha(path: Path) -> str | None:
    try:
        top_level = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=path, text=True, stderr=subprocess.DEVNULL,
        ).strip()
        resolved_top = os.path.normcase(str(Path(top_level).resolve()))
        if resolved_top != os.path.normcase(str(path.resolve())):
            return None
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _dependency_versions() -> dict[str, str | None]:
    versions = {}
    for package in DEPENDENCIES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def run_benchmark(
    *, mode: str = "pure", pure_samples: Any = PURE_DEFAULT,
    managed_samples: Any = MANAGED_DEFAULT, scenario_names: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Run selected scenarios and return one bounded JSON-ready result."""
    if mode not in {"pure", "managed", "all"}:
        raise ValueError("mode must be pure, managed, or all")
    pure_count = positive_sample_count(pure_samples, maximum=PURE_MAX, name="pure samples")
    managed_count = positive_sample_count(managed_samples, maximum=MANAGED_MAX, name="managed samples")
    registry = scenarios()
    by_name = {scenario.name: scenario for scenario in registry}
    unknown = sorted(set(scenario_names) - set(by_name))
    if unknown:
        raise ValueError(f"unknown benchmark scenario: {unknown[0]}")
    selected = [
        scenario for scenario in registry
        if (not scenario_names or scenario.name in scenario_names)
        and (mode == "all" or scenario.managed == (mode == "managed"))
    ]
    if not selected:
        raise ValueError("no benchmark scenarios selected")
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="dryml-dispatch-code-benchmark-") as temp_root:
        context = BenchmarkContext(Path(temp_root))
        results = [
            _measure_scenario(
                scenario,
                context,
                managed_count if scenario.managed else pure_count,
            )
            for scenario in selected
        ]
    result = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "success",
        "candidate": {"parent": _git_sha(ROOT.parent), "nested": _git_sha(ROOT)},
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.system().lower(),
            "machine": platform.machine(),
            "dependencies": _dependency_versions(),
        },
        "configuration": {
            "mode": mode,
            "pure_samples": pure_count,
            "managed_samples": managed_count,
            "pure_max": PURE_MAX,
            "managed_max": MANAGED_MAX,
        },
        "wall_seconds": time.perf_counter() - started,
        "scenarios": results,
    }
    encoded = json.dumps(result, sort_keys=True, allow_nan=False).encode()
    if len(encoded) > RESULT_LIMIT_BYTES:
        raise RuntimeError("benchmark result exceeds the bounded output limit")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pure", "managed", "all"), default="pure")
    parser.add_argument("--pure-samples", default=str(PURE_DEFAULT))
    parser.add_argument("--managed-samples", default=str(MANAGED_DEFAULT))
    parser.add_argument("--scenario", action="append", default=[])
    args = parser.parse_args(argv)
    try:
        result = run_benchmark(
            mode=args.mode,
            pure_samples=args.pure_samples,
            managed_samples=args.managed_samples,
            scenario_names=tuple(args.scenario),
        )
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
