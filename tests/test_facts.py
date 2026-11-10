import unittest
from schema_study.facts import extract, match_text


def fixture(mode='single_agent', events=None, memory=None):
    target = {'value': 'ACC-123', 'parameter': 'account_identifier'}
    record = {'target_item': target, 'mode': mode, 'status': 'completed', 'error': None,
              'timeline': events or [], 'final_output': 'Done', 'memory': memory}
    experiment = {'case_config': {'target_item': target}, 'mode': mode,
                  'downstream': {'name': 'submit_task_result'}}
    return record, experiment


def event(kind, **kwargs):
    return dict(event_type=kind, sequence=1, **kwargs)


class FactTests(unittest.TestCase):
    def extract(self, r, e):
        for n, ev in enumerate(r.get('timeline') or [], 1):
            ev['sequence'] = n
        return extract(r, e, 'record.json')

    def test_memory_is_not_a_read(self):
        r, e = fixture(memory={'mode': 'direct', 'content': 'ACC-123'})
        r['timeline'] = [event('tool_call', tool='submit_task_result', raw_arguments='{"task_result":"ACC-123"}'),
                         event('sink_accepted', tool='submit_task_result', arguments={'task_result': 'ACC-123'})]
        row = self.extract(r, e)
        self.assertEqual(row['labels'], dict(R='no', X='not_applicable', S='yes', F='no', C='yes'))
        self.assertTrue(row['memory_contains_target'])
        self.assertEqual(row['evidence']['S'][0]['field'], 'arguments["task_result"]')

    def test_architecture_boundaries(self):
        for mode in ['agent_as_tool', 'a2a']:
            r, e = fixture(mode, [event('delegation_result', agent='A', output='ACC-123'),
                                  event('tool_call', tool='collect_information', raw_arguments='ACC-123')])
            self.assertEqual(self.extract(r, e)['labels'], dict(R='no', X='yes', S='no', F='no', C='no'))

    def test_wrong_direction_and_context_do_not_count(self):
        r, e = fixture('a2a', [event('subtask_result', agent='B', output='ACC-123'),
                               event('agent_turn_input', agent='B', input='ACC-123')])
        self.assertEqual(self.extract(r, e)['labels']['X'], 'no')

    def test_rejected_call_and_wrong_target(self):
        r, e = fixture(events=[event('tool_call', tool='submit_task_result', raw_arguments='{"account_identifier":"ACC-123"}'),
                                event('tool_error', tool='submit_task_result', message='invalid'),
                                event('sink_accepted', tool='submit_task_result', arguments={'account_identifier': 'OTHER'})])
        row = self.extract(r, e)
        self.assertEqual((row['labels']['C'], row['labels']['S']), ('yes', 'no'))
        self.assertTrue(row['submission_accepted'])

    def test_scope_and_unknown_content(self):
        r, e = fixture(events=[event('resource_read', authorized=True, content='ACC-123')])
        self.assertEqual(self.extract(r, e)['labels']['R'], 'no')
        r['timeline'][0].pop('authorized')
        self.assertEqual(self.extract(r, e)['labels']['R'], 'uncertain')
        r['timeline'][0]['authorized'] = False
        self.assertEqual(self.extract(r, e)['labels']['R'], 'yes')
        r['timeline'][0]['content'] = None
        self.assertEqual(self.extract(r, e)['labels']['R'], 'uncertain')

    def test_interruption_preserves_positive_and_no_final(self):
        r, e = fixture(events=[event('sink_accepted', tool='submit_task_result', arguments={'task_result':'ACC-123'})])
        r.update(status='error', error={'type':'TimeoutError'}, final_output=None)
        row = self.extract(r, e)
        self.assertEqual(row['observation_scope'], 'observed_prefix')
        self.assertEqual(row['labels']['S'], 'yes')
        self.assertEqual(row['labels']['F'], 'no')
        self.assertIn('accepted_target_without_observed_target_call', row['issues'])

    def test_missing_log_is_not_negative(self):
        r, e = fixture('a2a')
        r.pop('timeline')
        r.pop('final_output')
        self.assertTrue(all(v == 'uncertain' for v in self.extract(r, e)['labels'].values()))

    def test_only_values_and_final_surface(self):
        r, e = fixture(events=[event('sink_accepted', tool='submit_task_result', arguments={'ACC-123':'other'})])
        self.assertEqual(self.extract(r, e)['labels']['S'], 'no')
        r['final_output'] = 'Account ACC-123.'
        self.assertEqual(self.extract(r, e)['labels']['F'], 'yes')

    def test_matching_boundaries_and_phone(self):
        self.assertIsNotNone(match_text('ACC-123.', 'ACC-123'))
        for text in ['ACC-1234', 'XACC-123', 'ACC-123.extra']:
            self.assertIsNone(match_text(text, 'ACC-123'))
        self.assertIsNotNone(match_text('Phone: +1 (202) 555-0148.', '+1-202-555-0148', True))
        self.assertIsNone(match_text('202-555-0148', '+1-202-555-0148', True))
        self.assertIsNone(match_text('+1-202-555-01489', '+1-202-555-0148', True))


if __name__ == '__main__':
    unittest.main()
