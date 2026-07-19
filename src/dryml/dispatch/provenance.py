"""Persistence-safe projections for launch configuration provenance."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dryml import worlds
from dryml.worlds.resources import canonical_byte_size


PERSISTENCE_PROJECTION_SCHEMA = "dryml.dispatch.persistence_projection.v1"


def redaction_marker() -> dict[str, str]:
    """Return the stable marker used in place of launch-only values."""

    return {"__dryml_redacted__": "launch_only"}


def project_environment_config(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Project an environment candidate without locators or environment values."""

    if value is None:
        return None
    data = dict(value)
    result = {
        key: data[key]
        for key in ("schema_version", "kind", "launch_mode", "pythonpath_policy")
        if key in data
    }
    redacted = []
    for key in ("executable", "prefix", "name", "conda_executable", "extra_pythonpath", "image", "runtime", "env"):
        if key in data and data[key] not in (None, {}, [], ()):
            result[key] = redaction_marker()
            redacted.append(key)
        elif key in data:
            result[key] = data[key]
    unknown = set(data) - set(result)
    if unknown:
        result["unrecognized_fields"] = redaction_marker()
        redacted.append("unrecognized_fields")
    result["persistence_projection"] = _projection_info(redacted)
    return result


def project_runtime_config(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Project runtime configuration while retaining stable worker policy facts."""

    if value is None:
        return None
    data = dict(value)
    result: dict[str, Any] = {}
    if "mode" in data:
        result["mode"] = data["mode"]
    visibility = data.get("device_visibility")
    redacted = []
    if isinstance(visibility, Mapping):
        result["device_visibility"] = {}
        policy = visibility.get("policy")
        if policy in {"none", "assigned", "inherit", "explicit"}:
            result["device_visibility"]["policy"] = policy
        elif "policy" in visibility:
            result["device_visibility"]["policy"] = redaction_marker()
            redacted.append("device_visibility.policy")
        accelerators = visibility.get("accelerators")
        if accelerators not in (None, {}, [], ()):
            if isinstance(accelerators, Mapping):
                result["device_visibility"]["accelerator_counts"] = {
                    str(key): _resource_count(item)
                    for key, item in sorted(accelerators.items(), key=lambda pair: str(pair[0]))
                }
            else:
                result["device_visibility"]["accelerator_count"] = _resource_count(accelerators)
            result["device_visibility"]["accelerators"] = redaction_marker()
            redacted.append("device_visibility.accelerators")
        hidden = set(visibility) - {"policy", "accelerators"}
        if hidden:
            result["device_visibility"]["launch_details"] = redaction_marker()
            redacted.append("device_visibility.launch_details")
    elif "device_visibility" in data:
        result["device_visibility"] = redaction_marker()
        redacted.append("device_visibility")
    for key in ("frameworks", "limits", "env", "metadata", "world_allocation_id"):
        if key in data and data[key] not in (None, {}, [], ()):
            result[key] = redaction_marker()
            redacted.append(key)
        elif key in data:
            result[key] = data[key]
    known = {"mode", "device_visibility", "frameworks", "limits", "env", "metadata", "world_allocation_id"}
    unknown = set(data) - known
    if unknown:
        result["unrecognized_fields"] = redaction_marker()
        redacted.append("unrecognized_fields")
    result["persistence_projection"] = _projection_info(redacted)
    return result


def project_world_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return a canonical requested-world spec with launch details removed."""

    if spec.get("schema") == "dryml.world.v1":
        payload = spec.get("payload")
    elif "roles" in spec:
        payload = spec
    else:
        # Match the established local-world shorthand normalization only for
        # provenance paths that do not yet have the allocator's canonical spec.
        payload = {
            "roles": spec,
            "backend": {"kind": "local_world", "parameters": {}},
        }
    world = worlds.WorldSpec.from_data(payload)
    roles: dict[str, Any] = {}
    redacted = []
    for role_name in sorted(world.roles):
        role = world.roles[role_name]
        resources = role.process.resources.to_data()
        process_redacted = []
        for key in ("devices", "named"):
            if resources.pop(key, None):
                process_redacted.append(f"resources.{key}")
        if role.process.environment is not None:
            process_redacted.append("environment")
        if role.process.runtime is not None:
            process_redacted.append("runtime")
        if role.process.env:
            process_redacted.append("env")
        if role.process.metadata:
            process_redacted.append("metadata")
        process: dict[str, Any] = {
            "resources": resources,
            "environment": None,
            "runtime": None,
            "env": {},
            "metadata": {},
        }
        if process_redacted:
            process["metadata"]["dryml.persistence_projection"] = _projection_info(process_redacted)
            redacted.extend(f"roles.{role_name}.process.{path}" for path in process_redacted)
        roles[role_name] = {"replicas": role.replicas, "process": process}
    backend = {"kind": world.backend.get("kind")}
    if world.backend.get("parameters"):
        backend["parameters"] = redaction_marker()
        redacted.append("backend.parameters")
    else:
        backend["parameters"] = {}
    extra_backend = set(world.backend) - {"kind", "parameters"}
    if extra_backend:
        backend["launch_details"] = redaction_marker()
        redacted.append("backend.launch_details")
    if spec.get("metadata"):
        redacted.append("metadata")
    backend["persistence_projection"] = _projection_info(redacted)
    return worlds.attach_world_id(
        worlds.make_world_spec(
            roles,
            backend=backend,
        )
    )


def project_world_allocation_spec(spec: Mapping[str, Any], *, world_id: str) -> dict[str, Any]:
    """Return a canonical allocation spec without host or process environment."""

    allocation = worlds.WorldAllocation.from_data(spec["payload"])
    roles: dict[str, list[dict[str, Any]]] = {}
    redacted = []
    for role_name in sorted(allocation.roles):
        roles[role_name] = []
        for item in allocation.roles[role_name]:
            resources: dict[str, Any] = {
                "cpus": list(range(len(item.cpus))),
                "accelerators": {
                    key: list(range(len(item.accelerators[key])))
                    for key in sorted(item.accelerators)
                },
            }
            if item.memory is not None:
                resources["memory"] = canonical_byte_size(item.memory)
            item_redacted = []
            if item.cpus:
                item_redacted.append("resources.cpus")
            item_redacted.extend(
                f"resources.accelerators.{key}"
                for key, values in sorted(item.accelerators.items())
                if values
            )
            if item.devices:
                item_redacted.append("resources.devices")
            if item.environment is not None:
                item_redacted.append("environment")
            if item.env:
                item_redacted.append("env")
            safe_metadata = {
                key: item.metadata[key]
                for key in ("allocation_policy", "world_size", "role_size")
                if key in item.metadata
            }
            if set(item.metadata) - set(safe_metadata):
                item_redacted.append("metadata")
            if item_redacted:
                safe_metadata["dryml.persistence_projection"] = _projection_info(item_redacted)
                redacted.extend(f"roles.{role_name}.{item.replica}.{path}" for path in item_redacted)
            roles[role_name].append(
                {
                    "replica": item.replica,
                    "rank": item.rank,
                    "local_rank": item.local_rank,
                    "resources": resources,
                    "environment": None,
                    "env": {},
                    "metadata": safe_metadata,
                }
            )
    backend = {key: allocation.backend[key] for key in ("name", "kind", "version") if key in allocation.backend}
    extra_backend = set(allocation.backend) - set(backend)
    if extra_backend:
        backend["launch_details"] = redaction_marker()
        redacted.append("backend.launch_details")
    if spec.get("metadata"):
        redacted.append("metadata")
    backend["requested_world_id"] = world_id
    backend["persistence_projection"] = _projection_info(redacted)
    return worlds.attach_world_allocation_id(
        worlds.make_world_allocation_spec(
            roles,
            backend=backend,
            kind=spec.get("kind", "local_allocation"),
        )
    )


def project_inventory_summary(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Retain inventory capacity without discovery metadata or identifiers."""

    if value is None:
        return None
    result = {
        "cpu_count": value.get("cpu_count"),
        "accelerator_counts": dict(value.get("accelerator_counts") or {}),
        "memory": value.get("memory"),
        "metadata": {},
    }
    if value.get("metadata"):
        result["metadata"] = redaction_marker()
        result["persistence_projection"] = _projection_info(["metadata"])
    return result


def project_allocation_summary(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Retain allocation shape and capacity without concrete resource IDs."""

    if value is None:
        return None
    workers = []
    redacted = []
    for index, worker in enumerate(value.get("workers") or ()):
        cpus = worker.get("cpus") or ()
        accelerators = worker.get("accelerators") or {}
        workers.append(
            {
                "role": worker.get("role"),
                "replica": worker.get("replica"),
                "cpu_count": len(cpus),
                "memory": worker.get("memory"),
                "accelerator_counts": {
                    str(key): _resource_count(item)
                    for key, item in sorted(accelerators.items(), key=lambda pair: str(pair[0]))
                },
            }
        )
        if cpus:
            redacted.append(f"workers.{index}.cpus")
        if any(_resource_count(item) for item in accelerators.values()):
            redacted.append(f"workers.{index}.accelerators")
    return {
        "backend": value.get("backend"),
        "allocation_policy": value.get("allocation_policy"),
        "workers": workers,
        "persistence_projection": _projection_info(redacted),
    }


def project_world_synthesis(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Remove private inventory metadata from world-synthesis provenance."""

    if value is None:
        return None
    result = dict(value)
    if isinstance(result.get("inventory"), Mapping):
        result["inventory"] = project_inventory_summary(result["inventory"])
    return result


def project_requirement_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    """Project resolved annotation defaults and source traces for persistence."""

    data = dict(value)
    if isinstance(data.get("environment_default"), Mapping):
        data["environment_default"] = project_environment_config(data["environment_default"])
    if isinstance(data.get("world_default"), Mapping):
        data["world_default"] = project_world_spec(data["world_default"])["payload"]
    if isinstance(data.get("runtime_default"), Mapping):
        data["runtime_default"] = project_runtime_config(data["runtime_default"])
    if isinstance(data.get("runtime_requirement"), Mapping):
        data["runtime_requirement"] = project_runtime_config(data["runtime_requirement"])
    data["fragments"] = [_project_fragment(item) for item in data.get("fragments", ())]
    data["source_traces"] = [_project_source_trace(item) for item in data.get("source_traces", ())]
    data["diagnostics"] = [
        {key: item.get(key) for key in ("level", "code")}
        for item in data.get("diagnostics", ())
        if isinstance(item, Mapping)
    ]
    if data.get("merge_report") is not None:
        data["merge_report"] = redaction_marker()
    return data


def _project_fragment(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"fragment": redaction_marker()}
    result = {key: value.get(key) for key in ("namespace", "kind", "priority", "merge_policy", "schema_version")}
    fragment = value.get("fragment")
    if isinstance(fragment, Mapping) and value.get("namespace") == "runtime":
        result["fragment"] = project_runtime_config(fragment)
    elif value.get("kind") == "default" and isinstance(fragment, Mapping):
        if value.get("namespace") == "environment":
            result["fragment"] = project_environment_config(fragment)
        elif value.get("namespace") == "world":
            result["fragment"] = project_world_spec(fragment)["payload"]
        else:
            result["fragment"] = redaction_marker()
    else:
        result["fragment"] = fragment
    result["source"] = _project_source_trace(value.get("source"))
    return result


def _project_source_trace(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"details": redaction_marker()}
    target = value.get("target")
    safe_target = None
    if isinstance(target, Mapping):
        safe_target = {
            key: target.get(key)
            for key in ("kind", "module", "qualname", "owner_module", "owner_qualname")
        }
        if target.get("metadata"):
            safe_target["metadata"] = redaction_marker()
    result = {
        key: value.get(key)
        for key in ("kind", "label", "namespace", "priority", "merge_policy", "fragment_index")
        if key in value
    }
    result["target"] = safe_target
    if value.get("path") is not None:
        result["path"] = redaction_marker()
    if value.get("metadata") or value.get("data"):
        result["details"] = redaction_marker()
    return result


def _projection_info(redacted_fields: list[str]) -> dict[str, Any]:
    return {
        "schema": PERSISTENCE_PROJECTION_SCHEMA,
        "redacted": bool(redacted_fields),
        "redacted_fields": sorted(set(redacted_fields)),
    }


def _resource_count(value: Any) -> int | None:
    if isinstance(value, Mapping):
        return len(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return len(value)
    return None


__all__ = [
    "PERSISTENCE_PROJECTION_SCHEMA",
    "project_environment_config",
    "project_allocation_summary",
    "project_inventory_summary",
    "project_requirement_provenance",
    "project_runtime_config",
    "project_world_synthesis",
    "project_world_allocation_spec",
    "project_world_spec",
    "redaction_marker",
]
