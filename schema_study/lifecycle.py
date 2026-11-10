"""Preserve completed/failed records; retry only when explicitly selected."""
import copy
import csv
import hashlib
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from . import batch, runner
from .paths import PROJECT_ROOT
HERE = PROJECT_ROOT
ORDER = ['repetition', 'scenario', 'mode', 'policy', 'condition', 'task', 'memory']
TRANSPORT = {'base_url', 'api_key_env', 'api_type', 'provider', 'request_timeout', 'live', 'openrouter_options'}
def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def semantic_config(experiment):
    value = batch._resume_config(experiment)
    for config in [value.get('config', {}), *value.get('agent_configs', {}).values()]:
        for key in TRANSPORT:
            config.pop(key, None)
    return value


def existing_status(experiment, directory):
    path = directory/'record.json'
    resolved = directory/'resolved_experiment.json'
    old = None
    if resolved.exists():
        try:
            old = read(resolved)
            if not isinstance(old, dict):
                old = None
        except (OSError, ValueError):
            pass
    if old is not None and semantic_config(old) != semantic_config(experiment):
        return 'conflict', 'Experimental inputs differ; use a new output directory/repeat'
    if not path.exists():
        if directory.exists() and any(directory.iterdir()):
            return 'preserved_partial', 'Existing partial output retained'
        return 'pending', 'No previous record'
    try:
        record = read(path)
        if not isinstance(record, dict):
            raise ValueError('Invalid record')
    except (OSError, ValueError):
        return 'preserved_partial', 'Unreadable record retained; no implicit retry'
    if record.get('status') != 'completed' or record.get('error'):
        return 'preserved_error', 'Existing unsuccessful record retained; no implicit retry'
    if record.get('execution_mode') != 'live':
        return 'preserved_nonlive', 'Completed non-live record retained'
    if old is None:
        return 'conflict', 'Completed record has no readable resolved configuration; inspect manually'
    if batch._resume_config(old) != batch._resume_config(experiment):
        return 'skipped_transport_change', 'Completed original retained with original endpoint/API metadata'
    return 'skipped_completed', 'Completed original retained'


def directory(root, e):
    return batch.result_directory(e, root/f"repeat_{e['batch']['repetition']+1:03}")


def prepare(root, plan, settings, config_path):
    rows=[]
    for i,e in enumerate(plan,1):
        dest=directory(root,e)
        status, reason=existing_status(e,dest)
        rows.append(dict(execution_index=i, repetition=e['batch']['repetition'], scenario=e['dataset']['scenario_id'],
                         mode=e['mode'], policy=e['policy'], condition=e['condition'], task=e['case_id'],
                         memory=(e.get('memory')or{}).get('mode','none'), status=status, reason=reason, run_dir=str(dest)))
    if any(r['status']=='conflict' for r in rows):
        raise ValueError('Existing runs have different model/task/experimental settings; use another output directory or repeat. Nothing overwritten.')
    folder=root/'_plans'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder.mkdir(parents=True, exist_ok=False)
    safe=copy.deepcopy(settings); safe.pop('api_key',None)
    files = sorted((HERE / 'schema_study').glob('*.py')) + [HERE / 'run.py', HERE / 'evaluate.py']
    hashes = {}
    for path in files:
        data = path.read_bytes()
        name = path.relative_to(HERE).as_posix()
        hashes[name] = hashlib.sha256(data).hexdigest()
        dest = folder / 'source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    manifest=dict(order=ORDER, planned=len(plan), config_source=str(config_path.resolve()), configuration=safe,
                  source_hashes=hashes, plan_sha256=hashlib.sha256(json.dumps(plan,sort_keys=True,default=str).encode()).hexdigest(),
                  note='Source/order/transport changes never overwrite completed records; original provenance retained.')
    (folder/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    with (folder/'plan.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return rows,folder


def select_runs(plan, rows, retry_failed=False, limit=0):
    statuses = {'preserved_error', 'preserved_partial'} if retry_failed else {'pending'}
    candidates = [e for e,r in zip(plan,rows) if r['status'] in statuses]
    return candidates[:limit] if limit else candidates


def archive_failed(root, experiment, folder):
    root = root.resolve()
    dest = directory(root, experiment).resolve()
    archive = (root/'_retry_history'/folder.name/dest.relative_to(root)).resolve()
    if not dest.is_relative_to(root) or not archive.is_relative_to(root) or archive == dest:
        raise ValueError('Retry paths must remain inside the selected result root')
    status, _ = existing_status(experiment, dest)
    if status not in {'preserved_error', 'preserved_partial'}:
        raise ValueError(f'Retry target is no longer failed/partial: {dest}')
    if archive.exists():
        raise ValueError(f'Retry archive already exists: {archive}')
    archive.parent.mkdir(parents=True, exist_ok=True)
    dest.rename(archive)  # Preserve every original file; never delete/overwrite it.
    return archive


async def execute(root, selected, folder, retry_failed=False):
    # Give each invocation separate aggregate files; never rebuild result_all.json
    # from a limited subset and accidentally discard earlier aggregate entries.
    report=[]
    for index,e in enumerate(selected,1):
        dest=directory(root,e)
        print(f'[{index}/{len(selected)}] {e["case_id"]} / {e["mode"]} / {e["condition"]} / {e["policy"]} / {(e.get("memory") or {}).get("mode", "none")}',flush=True)
        archived = None
        if retry_failed:
            archived = archive_failed(root, e, folder)
        elif dest.exists() and any(dest.iterdir()):
            raise ValueError(f'Run directory appeared after preparation: {dest}')
        e=copy.deepcopy(e)
        e['batch'].update(batch_id=folder.name, result_key=dest.relative_to(root).as_posix())
        if archived is not None:
            e['batch']['retry_of'] = str(archived)
        entry = {'run_dir':str(dest), 'status':'running', 'retry_of':str(archived) if archived else None}
        report.append(entry)
        (folder/'execution.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        # 这里才开始执行
        try:
            await runner.run_and_record(e,live=True,output=folder/'result_all.json',run_dir=dest)
            status='completed'
        except Exception as exc:
            status='error'
            traceback.print_exc()
            print(f'Run failed: {type(exc).__name__}; inspect record.json',flush=True)
        entry['status'] = status
        (folder/'execution.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report
