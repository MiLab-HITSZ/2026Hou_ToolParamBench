"""Shared structured queries for five domains; original authorization rules preserved."""
import json

def load_collections(root):
    directory = root / 'resources'
    if not directory.is_dir():
        raise ValueError(f'Missing resources directory: {directory}; supply the original test_data directory')
    data = {}
    for path in sorted(directory.glob('*.json')):
        value = json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(value, dict) or set(value) != {path.stem} or not isinstance(value[path.stem], list):
            raise ValueError(f'Expected {{"{path.stem}": [...]}} in {path}')
        data[path.stem] = value[path.stem]
    return data


SCENARIOS = {
    'finance_admin': {
        'get_budget_status': ('budgets', ('project',), 'project'),
        'get_expense_status': ('expenses', ('expense_id',), 'expense_id'),
        'get_invoice_status': ('invoices', ('invoice_id',), 'invoice_id'),
        'get_payment_request_status': ('payment_requests', ('request_id',), 'request_id'),
        'get_finance_profile': ('finance_profiles', ('subject_id',), 'subject_id'),
        'get_employee_finance_record': ('employee_finance_records', ('employee_name',), 'employee_name'),
    },
    'it_operations': {
        'get_asset_status': ('assets', ('asset_id',), 'asset_id'),
        'get_ticket_status': ('tickets', ('asset_id', 'ticket_id'), 'asset_id_or_ticket_id'),
        'get_patch_status': ('patches', ('asset_id',), 'asset_id'),
        'get_incident_status': ('incidents', ('incident_id',), 'incident_id'),
        'get_admin_contact': ('admin_contacts', ('employee_name',), 'employee_name'),
        'get_employee_record': ('employee_records', ('employee_name',), 'employee_name'),
    },
    'power_grid': {
        'get_substation_status': ('substations', ('substation_id',), 'substation_id'),
        'get_alarm_summary': ('alarms', ('asset_id',), 'asset_id'),
        'get_maintenance_schedule': ('maintenance', ('asset_id',), 'asset_id'),
        'get_contingency_result': ('contingencies', ('case_id',), 'case_id'),
        'get_operator_contact': ('operator_contacts', ('operator_name',), 'operator_name'),
        'get_operator_profile': ('operator_profiles', ('operator_name',), 'operator_name'),
    },
    'travel': {
        'get_flight_status': ('flights', ('trip_id',), 'trip_id'),
        'get_hotel_booking': ('hotels', ('trip_id',), 'trip_id'),
        'get_itinerary': ('itineraries', ('trip_id',), 'trip_id'),
        'get_expense_estimate': ('estimates', ('trip_id',), 'trip_id'),
        'get_traveler_contact': ('traveler_contacts', ('traveler_id',), 'traveler_id'),
        'get_traveler_profile': ('traveler_profiles', ('traveler_id',), 'traveler_id'),
    },
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def load_scenario(root, task, targets, scenario):
    specs = SCENARIOS[scenario['scenario_id']]
    data, contracts = load_collections(root), read(root / 'tool_contracts.json')
    declared = scenario['normal_query_tools'] + scenario['sensitive_query_tools']
    if len(declared) != len(set(declared)) or set(declared) != set(specs):
        raise ValueError('Scenario tool list does not match supported query contracts')
    for name, (table, columns, arg) in specs.items():
        if table in data and (not isinstance(data[table], list) or any(not isinstance(row, dict) or
                any(column not in row for column in columns) for row in data[table])):
            raise ValueError(f'Invalid collection for {name}: {table}')
        if 'parameters' in contracts[name] and contracts[name]['parameters'] != {arg: 'string'}:
            raise ValueError(f'Contract parameters conflict with adapter: {name}')
    authorized = task['authorized_tools']
    if not isinstance(authorized, list) or not authorized or len(set(authorized)) != len(authorized) or set(authorized) - set(specs):
        raise ValueError('Invalid authorized_tools')
    calls = []
    arguments = task.get('authorized_arguments')
    for name in authorized:
        rule = {'tool': name}
        if arguments is not None:
            args = arguments[name]
            if set(args) != {specs[name][2]} or not all(isinstance(v, str) for v in args.values()):
                raise ValueError(f'Invalid authorized arguments: {name}')
            rule['arguments'] = args
        calls.append(rule)
    task = {**task, 'authorized_resources': list(dict.fromkeys(specs[name][0] for name in authorized)),
            'authorization_granularity': 'tool_and_arguments' if arguments is not None else 'tool'}
    normalized_targets = []
    for target in targets:
        if target['source_tool'] not in scenario['sensitive_query_tools']:
            raise ValueError(f"Invalid target source: {target['target_id']}")
        table = specs[target['source_tool']][0]
        if not isinstance(target['value'], str) or not target['value']:
            raise ValueError('Target must be a nonempty string')
        # Target is validated host-side; its row key is never added to tools or prompts.
        matches = [row for row in data.get(table, []) if row.get(target['target_type']) == target['value']]
        if table in data and len(matches) != 1:
            raise ValueError(f"Target value/field must match exactly one row: {target['target_id']}")
        normalized_targets.append({**target, 'resource': table, 'detected_type': target['target_type']})
    resources = {table: {'group': 'resources', 'text': json.dumps(rows, ensure_ascii=False)}
                 for table, rows in data.items()}
    definitions = [{'handler': name, 'name': name, 'description': contracts[name]['description'],
                    'parameters': {'type': 'object', 'properties': {arg: {'type': 'string'}},
                                   'required': [arg], 'additionalProperties': False}}
                   for name, (_, _, arg) in specs.items()]
    return task, normalized_targets, resources, calls, definitions, contracts['submit_task_result']['description']


ECOMMERCE_SPECS = {
    'get_order_status': ('orders', None, 'order_id'),
    'get_delivery_status': ('deliveries', None, 'order_id'),
    'get_return_status': ('returns', None, 'return_id_or_order_id'),
    'get_payment_status': ('payments', None, 'order_id'),
    'get_customer_contact': ('customer_contacts', None, 'customer_id'),
    'get_customer_profile': ('customer_profiles', None, 'customer_id'),
}


def load_ecommerce(root, task, targets, scenario):
    data = load_collections(root)
    contracts = read(root / 'tool_contracts.json')
    declared = scenario['normal_query_tools'] + scenario['sensitive_query_tools']
    if len(declared) != len(set(declared)) or set(declared) != set(ECOMMERCE_SPECS):
        raise ValueError('Unsupported ecommerce query tools')
    for name, (collection, _, arg) in ECOMMERCE_SPECS.items():
        if contracts[name]['parameters'] != {arg: 'string'}:
            raise ValueError(f'Unsupported ecommerce contract: {name}')
        if collection in data and not isinstance(data[collection], list):
            raise ValueError(f'Invalid ecommerce collection: {collection}')
    calls = []
    for name in task['authorized_tools']:
        if name not in ECOMMERCE_SPECS:
            raise ValueError(f'Unknown authorized tool: {name}')
        args = task['authorized_arguments'][name]
        if set(args) != {ECOMMERCE_SPECS[name][2]} or not all(isinstance(v, str) for v in args.values()):
            raise ValueError(f'Invalid authorized arguments: {name}')
        calls.append({'tool': name, 'arguments': args})
    task = {**task, 'authorized_resources': list(dict.fromkeys(ECOMMERCE_SPECS[c['tool']][0] for c in calls))}
    targets = [{**t, 'resource': ECOMMERCE_SPECS[t['source_tool']][0], 'detected_type': t['target_type']} for t in targets]
    for target in targets:
        if target['resource'] not in data:
            continue
        rows = data[target['resource']]
        matches = [r for r in rows if r.get('customer_id') == target['customer_id']
                   and r.get(target['detected_type']) == target['value']]
        if len(matches) != 1:
            raise ValueError(f"Target does not match its customer/field: {target['target_id']}")
    resources = {name: {'group': 'resources', 'text': json.dumps(rows, ensure_ascii=False)}
                 for name, rows in data.items()}
    definitions = [{'handler': name, 'name': name, 'description': contracts[name]['description'],
                    'parameters': {'type': 'object', 'properties': {arg: {'type': 'string'}},
                                   'required': [arg], 'additionalProperties': False}}
                   for name, (_, _, arg) in ECOMMERCE_SPECS.items()]
    return task, targets, resources, calls, definitions, contracts['submit_task_result']['description']


def tools(backend, actor):
    ecommerce = backend.e['query_adapter'] == 'ecommerce'
    specs = ECOMMERCE_SPECS if ecommerce else SCENARIOS[backend.e['query_adapter']]

    def handler(name):
        def invoke(args):
            table, columns, arg = specs[name]
            allowed = any(rule['tool'] == name and ('arguments' not in rule or rule['arguments'] == args)
                          for rule in backend.e['case_config']['authorization']['authorized_calls'])
            backend.log.add('structured_query', agent=actor, query=name, arguments=args, authorized=allowed)
            if table not in backend.e['resources']:
                raise ValueError(f'Data source unavailable: {table}')
            rows = json.loads(backend.e['resources'][table]['text'])
            if ecommerce:
                if name == 'get_return_status':
                    selected = [r for r in rows if r['return_id'] == args[arg] or r['order_id'] == args[arg]]
                else:
                    selected = [r for r in rows if r.get(arg) == args[arg]]
            else:
                selected = [row for row in rows if any(row[column] == args[arg] for column in columns)]
            result = {'records': selected, 'found': bool(selected)}
            if selected:
                backend.log.add('resource_read', agent=actor, resource=table, query=name, arguments=args,
                                authorized=allowed, content=json.dumps(result, ensure_ascii=False))
            return result
        return invoke

    return [backend.tool(t['name'], t['description'], t['parameters'], actor, handler(t['handler']))
            for t in backend.e['source_tools']]
