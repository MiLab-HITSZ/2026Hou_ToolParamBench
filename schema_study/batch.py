"""Dataset-independent experiment grids and stable result identities."""
from __future__ import annotations
import copy
import itertools
import json
import random
import re
from pathlib import Path
from . import experiment as workspace
from .paths import PROJECT_ROOT

SYSTEM_PROMPT = (
    "Complete the user's task using the available tools. "
    "In single-agent mode, submit the task result using the downstream submission tool."
)
INSTRUCTIONS_A = INSTRUCTIONS_B = None
MAX_ROUNDTRIPS = 4
MAX_SUBMISSION_REVISIONS = 2


def _choices(values, label):
    if isinstance(values, (str, bytes)) or values is None:
        raise ValueError(f"{label} must be a nonempty sequence, e.g. [None]")
    values = list(values)
    if not values or len(set(values)) != len(values):
        raise ValueError(f"{label} must be nonempty and contain no duplicates")
    return values


def build_plan(*, profile="structured", data_root=None, cases=None, modes=None, conditions=None,
               policies=None, repeats=1, shuffle_seed=None,
               model=None, model_a=None, model_b=None, memory_modes=None, **overrides):
    """预先构建并校验全部组合；policies=[None] 表示不追加 policy。

    overrides 透传 make_experiment，例如 system_prompt、source_tools、data_root。
    不跨条件复用自定义 downstream_tool，除非其 schema 同时满足所选条件。
    repeats 供重复实验脚本构建多轮计划；调用方需按 repetition 拆分并使用独立输出目录。
    """
    if data_root is None or model is None:
        raise ValueError("Explicit data_root and model are required")
    if profile not in ("files", "structured"):
        raise ValueError("profile must be files or structured")
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if shuffle_seed is not None and type(shuffle_seed) is not int:
        raise ValueError("shuffle_seed must be an integer or None")
    root = Path(data_root).expanduser()
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    root = root.resolve()
    if not (root / "scenario.json").is_file():
        raise ValueError(f"Dataset scenario.json not found: {root}")
    cases = _choices(sorted(p.stem for p in (root / "tasks").glob("*.json"))
                     if cases is None else cases, "cases")
    modes = _choices(["single_agent"] if modes is None else modes, "modes")
    modes = _choices(["agent_as_tool" if m == "multi_agent" else m for m in modes], "normalized modes")
    conditions = _choices(["absent", "optional", "required"] if conditions is None else conditions, "conditions")
    policies = _choices(["P0_none"] if policies is None else policies, "policies")
    options = dict(system_prompt=SYSTEM_PROMPT, instructions_a=INSTRUCTIONS_A,
                   instructions_b=INSTRUCTIONS_B, max_roundtrips=MAX_ROUNDTRIPS,
                   max_submission_revisions=MAX_SUBMISSION_REVISIONS, data_root=root)
    options.update(overrides)
    if "tool_profile" in options:
        raise ValueError("Use profile to select the tool profile")
    trials = []
    memories = _choices([None] if memory_modes is None else memory_modes, 'memory_modes')
    memories = _choices([None if m == 'none' else 'direct' if m == 'direct_value' else m for m in memories], 'normalized memory modes')
    for case, mode, policy, condition, memory_mode in itertools.product(cases, modes, policies, conditions, memories):
        base = workspace.make_experiment(mode, case=case, condition=condition, policy=policy,
            tool_profile=profile, model=model, model_a=model_a, model_b=model_b, memory_mode=memory_mode, **options)
        for repetition in range(repeats):
            e = copy.deepcopy(base)
            e["batch"] = {"repetition": repetition, "plan_index": len(trials),
                          "shuffle_seed": shuffle_seed}
            trials.append(e)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(trials)
    for index, e in enumerate(trials):
        e["batch"]["index"] = index
    return trials


def build_multi_plan(datasets, *, shuffle_seed=None, **shared):
    """合并多个数据集，统一编号和打乱；只构建计划，不调用模型或写结果。"""
    if isinstance(datasets, (str, bytes, dict)) or datasets is None:
        raise ValueError("datasets must be a nonempty sequence of dataset settings")
    if shuffle_seed is not None and type(shuffle_seed) is not int:
        raise ValueError("shuffle_seed must be an integer or None")
    if "data_root" in shared or "cases" in shared:
        raise ValueError("Set data_root and cases separately in each dataset")
    trials, names = [], set()
    for config in datasets:
        if not isinstance(config, dict) or "data_root" not in config or set(config) - {"data_root", "cases"}:
            raise ValueError("Each dataset must contain data_root and optionally cases")
        root = Path(config["data_root"]).expanduser()
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        root = root.resolve()
        # 输出目录使用数据集文件夹名，禁止重名导致相互覆盖。
        name = root.name.casefold()
        if name in names:
            raise ValueError(f"Duplicate dataset directory name: {root.name}")
        names.add(name)
        cases = config.get("cases")
        if cases is None:
            cases = sorted(p.stem for p in (root / "tasks").glob("*.json"))
        trials.extend(build_plan(data_root=root, cases=cases, shuffle_seed=None, **shared))
    if not trials:
        raise ValueError("Empty dataset plan")
    for index, e in enumerate(trials):
        e["batch"].update(plan_index=index, shuffle_seed=shuffle_seed)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(trials)
    for index, e in enumerate(trials):
        e["batch"]["index"] = index
    return trials


def result_directory(experiment, output_root):
    """Stable results/memory/policy/condition/mode/dataset/task path."""
    parts = [experiment['policy'] if experiment['policy'] is not None else 'None',
             experiment['condition'], experiment['mode'], Path(experiment['case_path']).name,
             experiment['case_id']]
    parts.insert(0, (experiment.get('memory') or {}).get('mode', 'none'))
    reserved = {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1,10)), *(f'LPT{i}' for i in range(1,10))}
    for part in parts:
        if (not isinstance(part, str) or not re.fullmatch(r'[\w.-]+', part)
                or part in ('.', '..') or part.endswith('.') or part.split('.')[0].upper() in reserved):
            raise ValueError(f'Invalid result directory component: {part!r}')
    root = Path(output_root).resolve()
    directory = root.joinpath(*parts)
    if not directory.resolve().is_relative_to(root):
        raise ValueError('Result directory resolves outside output_root')
    return directory


def _resume_config(experiment):
    """Ignore execution order and the runner's live flag, retain experimental inputs."""
    result = json.loads(json.dumps(experiment, default=str))
    result.pop("batch", None)
    result.get("config", {}).pop("live", None)
    result.get("config", {}).pop("request_timeout", None)
    result.get("config", {}).setdefault("api_type", "responses")
    if result.get('config', {}).get('openrouter_options') is None:
        result.get('config', {}).pop('openrouter_options', None)
    for config in result.get("agent_configs", {}).values():
        config.pop("live", None)
        config.pop("request_timeout", None)
        config.setdefault("api_type", "responses")
        if config.get('openrouter_options') is None:
            config.pop('openrouter_options', None)
    return result
