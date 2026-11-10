"""One configuration-driven entry point for all datasets and model providers."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import copy
import json
import os
from pathlib import Path
import re
from urllib.parse import urlparse

from .paths import PROJECT_ROOT, OUTPUT_ROOT


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def model_config(path, name):
    from .experiment import MODEL_FIELDS
    document = read_json(path)
    inline_key = None
    if isinstance(document, dict) and 'models' in document:
        if not isinstance(document['models'], dict) or not isinstance(document.get('defaults', {}), dict):
            raise ValueError('models and defaults must be JSON objects')
        if name not in document['models']:
            raise ValueError(f'Unknown model preset {name!r}; available: {list(document["models"])}')
        if not isinstance(document['models'][name], dict):
            raise ValueError('Each model preset must be a JSON object')
        model = {**document.get('defaults', {}), **document['models'][name]}
        inline_key = model.pop('api_key', None)
    else:
        entries = document if isinstance(document, list) else [document]
        if not entries or any(not isinstance(e, dict) or not isinstance(e.get('model'), str)
                              or not e['model'].strip() for e in entries):
            raise ValueError('Expected a models mapping or a nonempty list of model configurations')
        # Prefer exact names/IDs; a backend alias must identify exactly one entry.
        matches = [e for e in entries if name in (e.get('name'), e['model'])]
        if not matches:
            matches = [e for e in entries if name == e.get('backend')]
        if len(matches) != 1:
            available = [e['model'] for e in entries]
            raise ValueError(f'Model selector {name!r} matched {len(matches)} entries; select a unique model ID: {available}')
        entry = dict(matches[0])
        inline_key = entry.pop('api_key', None)
        if inline_key is not None and not isinstance(inline_key, str):
            raise ValueError('api_key must be a string')
        entry.pop('backend', None)
        entry.pop('name', None)
        host = urlparse(entry.get('base_url', '')).hostname
        provider = entry.get('provider') or ('openrouter' if host == 'openrouter.ai' else
                                            'openai' if host == 'api.openai.com' else 'compatible')
        # Legacy OpenAI-compatible configs omit API type; use Chat Completions.
        # Explicit fields always override these defaults.
        model = dict(max_turns=16, trial_timeout=180, request_timeout=60, tool_choice='auto',
                     reasoning_effort=None, temperature=None, max_tokens=None,
                     provider=provider, api_type='chat_completions',
                     api_key_env='SCHEMA_STUDY_LOCAL_API_KEY')
        model.update(entry)
    if inline_key is not None and not isinstance(inline_key, str):
        raise ValueError('api_key must be a string')
    if inline_key and inline_key.strip():
        model.setdefault('api_key_env', 'SCHEMA_STUDY_LOCAL_API_KEY')
    unknown = model.keys() - set(MODEL_FIELDS)
    if unknown:
        raise ValueError(f'Unsupported model keys: {sorted(unknown)}. Use api_key or api_key_env for credentials.')
    for field in ('model', 'api_type', 'base_url', 'api_key_env', 'provider'):
        if not isinstance(model.get(field), str) or not model[field].strip():
            raise ValueError(f'Model configuration requires {field}')
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', model['api_key_env']):
        raise ValueError('api_key_env must be an environment variable name, such as MODEL_API_KEY; '
                         'put a literal API key in api_key instead')
    if inline_key and inline_key.strip():
        # Never put credentials in returned configurations or saved plans.
        os.environ[model['api_key_env']] = inline_key
    return model


def build_experiments(config, model, *, data_root=None):
    """Build the exact configured conditions, without files or model requests."""
    from .batch import build_multi_plan, result_directory
    allowed = {'profile', 'repeats', 'start_repeat', 'datasets', 'conditions', 'overrides'}
    if config.keys() - allowed:
        raise ValueError(f'Unknown experiment settings: {sorted(config.keys() - allowed)}')
    repeats = positive(config.get('repeats', 1), 'repeats')
    start = positive(config.get('start_repeat', 1), 'start_repeat')
    datasets = copy.deepcopy(config['datasets'])
    if not isinstance(datasets, list) or not datasets:
        raise ValueError('datasets must be a nonempty list')
    for d in datasets:
        if set(d) - {'data_root', 'cases'}:
            raise ValueError('Dataset entries accept only data_root and cases')
        path = Path(d['data_root'])
        if data_root is not None:
            # Replace the test_data prefix, not the names/contents of datasets.
            if path.is_absolute() or not path.parts or path.parts[0] != 'test_data':
                raise ValueError('--data-root requires configured paths beginning with test_data/')
            path = Path(data_root).resolve().joinpath(*path.parts[1:])
        elif not path.is_absolute():
            path = PROJECT_ROOT / path
        d['data_root'] = str(path.resolve())
    conditions = config['conditions']
    if not isinstance(conditions, list) or not conditions:
        raise ValueError('conditions must be a nonempty list of explicit combinations')
    overrides = config.get('overrides', {})
    allowed_overrides = {'system_prompt', 'instructions_a', 'instructions_b', 'model_a', 'model_b',
                         'max_roundtrips', 'max_submission_revisions', 'delegation_description'}
    if set(overrides) - allowed_overrides:
        raise ValueError(f'Unsupported overrides: {sorted(set(overrides) - allowed_overrides)}')
    from .experiment import MODEL_FIELDS
    for field in ('model_a', 'model_b'):
        if overrides.get(field) is not None and set(overrides[field]) - set(MODEL_FIELDS):
            raise ValueError(f'Unsupported {field} fields; credentials must use api_key_env')
    trials = []
    for condition in conditions:
        if set(condition) != {'mode', 'condition', 'policy', 'memory_mode'}:
            raise ValueError('Each condition requires mode, condition, policy and memory_mode')
        trials.extend(build_multi_plan(datasets, profile=config.get('profile', 'structured'),
            modes=[condition['mode']], conditions=[condition['condition']],
            policies=[condition['policy']], memory_modes=[condition['memory_mode']],
            model=model, repeats=repeats, shuffle_seed=None, **overrides))
    # Keep the legacy execution order within each scenario and repetition.
    scenario_order = {read_json(Path(d['data_root']) / 'scenario.json')['scenario_id']: i
                      for i, d in enumerate(datasets)}
    case_order = {str(Path(d['data_root'])): {case: i for i, case in enumerate(
        d.get('cases') if d.get('cases') is not None else
        sorted(p.stem for p in (Path(d['data_root']) / 'tasks').glob('*.json')))} for d in datasets}
    def ordered(field):
        values = [c[field] for c in conditions]
        return list(dict.fromkeys(values))
    modes, policies, schemas, memories = [ordered(k) for k in ('mode', 'policy', 'condition', 'memory_mode')]
    modes = list(dict.fromkeys('agent_as_tool' if m == 'multi_agent' else m for m in modes))
    memories = list(dict.fromkeys(None if m in (None, 'none') else 'direct' if m == 'direct_value' else m for m in memories))
    trials.sort(key=lambda e: (e['batch']['repetition'], scenario_order[e['dataset']['scenario_id']],
        modes.index(e['mode']), policies.index(e['policy']), schemas.index(e['condition']),
        case_order[e['case_path']][e['case_id']], memories.index((e.get('memory') or {}).get('mode'))))
    from .lifecycle import ORDER
    identities = set()
    for i, e in enumerate(trials):
        e['batch'].update(index=i, plan_index=i, execution_order=ORDER, shuffle_seed=None)
        e['batch']['repetition'] += start - 1
        identity = str(result_directory(e, OUTPUT_ROOT / f"repeat_{e['batch']['repetition'] + 1:03}"))
        if identity in identities:
            raise ValueError(f'Duplicate configured experiment: {e["case_id"]}')
        identities.add(identity)
    return trials


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'configs/main90.json')
    parser.add_argument('--models', type=Path, default=PROJECT_ROOT / 'configs/models.example.json')
    parser.add_argument('--model', required=True, help='Preset name in the model configuration')
    parser.add_argument('--data-root', type=Path, help='Optional replacement for the test_data directory')
    parser.add_argument('--out-dir', type=Path, help='Default: results/<config>/<model>')
    parser.add_argument('--repeats', type=int)
    parser.add_argument('--start-repeat', type=int)
    parser.add_argument('--limit', type=int, default=0, help='Run at most N eligible trials; 0 means all')
    parser.add_argument('--retry-failed', action='store_true', help='Archive and retry only failed/partial runs')
    parser.add_argument('--live', action='store_true', help='Actually call the model; default is plan preview')
    args = parser.parse_args(argv)
    from . import lifecycle
    try:
        if args.limit < 0:
            raise ValueError('--limit must be nonnegative')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.model) or args.model.endswith('.'):
            raise ValueError('Model preset names must be simple directory names')
        config = read_json(args.config)
        if args.repeats is not None:
            config['repeats'] = args.repeats
        if args.start_repeat is not None:
            config['start_repeat'] = args.start_repeat
        model = model_config(args.models, args.model)
        plan = build_experiments(config, model, data_root=args.data_root)
        root = (args.out_dir or OUTPUT_ROOT / args.config.stem / args.model).resolve()
        if args.live:
            for e in plan:
                for actor in ('A',) if e['mode'] == 'single_agent' else ('A', 'B'):
                    m = e['agent_configs'][actor]
                    if 'YOUR-ENDPOINT' in (m.get('base_url') or ''):
                        raise ValueError('Replace the example base_url before a live run')
                    if not os.environ.get(m['api_key_env'], '').strip():
                        raise ValueError(f'Missing environment variable: {m["api_key_env"]}')
        settings = dict(experiment=config, model=model, data_root=str(args.data_root) if args.data_root else None)
        rows, folder = lifecycle.prepare(root, plan, settings, args.config)
        selected = lifecycle.select_runs(plan, rows, args.retry_failed, args.limit)
        print(json.dumps(dict(planned=len(plan), selected=len(selected), live=args.live,
                              statuses=dict(Counter(r['status'] for r in rows)), plan_dir=str(folder)), indent=2))
        if args.live:
            report = asyncio.run(lifecycle.execute(root, selected, folder, args.retry_failed))
            if any(r['status'] == 'error' for r in report):
                return 1
        return 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    raise SystemExit(main())
