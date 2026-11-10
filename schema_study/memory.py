"""Load synthetic pre-existing memory; only content is exposed to the querying agent."""
import json


def load_memory(root, mode, scenario_id, target, targets):
    if mode in (None, 'none'):
        return None, []
    mode = 'direct' if mode == 'direct_value' else mode
    if mode not in ('pointer', 'direct'):
        raise ValueError('memory_mode must be None, pointer or direct')
    folder = root / 'memory'
    mapping_path = folder / 'memory_mapping.json'
    mapping = json.loads(mapping_path.read_text(encoding='utf-8-sig'))
    scenario = 'workspace' if scenario_id == 'workspace_structured_v0_2' else scenario_id
    if mapping.get('scenario_id') != scenario:
        raise ValueError('Memory mapping scenario mismatch')
    condition = 'direct_value' if mode == 'direct' else mode
    name = mapping.get('conditions', {}).get(condition, 'memory_direct.json' if mode == 'direct' else 'memory_pointer.json')
    path = (folder / name).resolve()
    if path.parent != folder.resolve():
        raise ValueError('Memory file must be directly inside memory/')
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if (data.get('scenario_id') != scenario or data.get('memory_condition') != condition
            or data.get('schema_version', 'memory-v0.1') != 'memory-v0.1'):
        raise ValueError('Memory metadata mismatch')
    records = data['memories']
    if len({m['memory_id'] for m in records}) != len(records):
        raise ValueError('Duplicate memory IDs')
    if mode == 'direct':
        links = [m for m in mapping.get('target_to_direct_memory', mapping.get('task_memory_mapping', []))
                 if m['target_id'] == target['target_id']]
        if len(links) != 1:
            raise ValueError('Expected exactly one direct memory mapping for target')
        records = [m for m in records if m['memory_id'] == links[0]['direct_memory_id']]
        if len(records) != 1 or records[0].get('target_id') != target['target_id']:
            raise ValueError('Direct memory target mismatch')
    if not records or any(m.get('memory_type', condition) != condition or not isinstance(m.get('content'), str)
                          or not m['content'].strip() for m in records):
        raise ValueError('Invalid memory content/type')
    text = '\n'.join('- ' + m['content'] for m in records)
    if mode == 'pointer' and any(t['value'] in text for t in targets):
        raise ValueError('Pointer memory must not contain target values')
    if mode == 'direct' and target['value'] not in text:
        raise ValueError('Direct memory must contain the selected target value')
    return {'mode': mode, 'recipient': 'A', 'injection': 'preloaded_context',
            'context_format': 'plain_content_v1',
            'ids': [m['memory_id'] for m in records], 'content': text,
            'target_present': target['value'] in text}, [mapping_path, path]


def context(memory):
    # Keep only the selected dataset content. The format marker above is metadata
    # for resume checks; it is never included in the model's instructions.
    return '\n\n' + memory['content']
