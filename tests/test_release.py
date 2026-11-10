"""Offline checks for experiment identity, memory, retry preservation and metrics."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

from schema_study import cli, lifecycle, runner
from schema_study.evaluation import extract, ordered_path, summarize
from schema_study.paths import PROJECT_ROOT
from tests.test_ecommerce_entry import plan, call, script
from agents.testing import assistant_message


class ReleaseTests(unittest.IsolatedAsyncioTestCase):
    def test_exact_experiment_sets_and_memory(self):
        model = cli.model_config(PROJECT_ROOT / 'configs/models.example.json', 'deepseek')
        main = cli.build_experiments(cli.read_json(PROJECT_ROOT / 'configs/main90.json'), model)
        extended = cli.build_experiments(cli.read_json(PROJECT_ROOT / 'configs/extended45.json'), model)
        self.assertEqual(len(main), 270)
        self.assertEqual(len({(e['dataset']['scenario_id'], e['case_id']) for e in main}), 90)
        self.assertEqual(len(extended), 270)
        self.assertEqual(len({(e['dataset']['scenario_id'], e['case_id']) for e in extended}), 45)
        for e in extended:
            memory = e.get('memory')
            if memory:
                self.assertEqual(memory['mode'] == 'direct', e['case_config']['target_item']['value'] in memory['content'])
                prompts = runner.effective_instructions(e)
                self.assertIn(memory['content'], prompts['A'])
                if 'B' in prompts:
                    self.assertNotIn(memory['content'], prompts['B'])
        by_task = {}
        for e in main:
            by_task.setdefault(e['case_id'], {})[e['condition']] = e
        for variants in by_task.values():
            optional = copy.deepcopy(variants['optional']['downstream'])
            required = copy.deepcopy(variants['required']['downstream'])
            parameter = variants['required']['case_config']['target_item']['parameter']
            required['parameters']['required'].remove(parameter)
            self.assertEqual(optional, required)
            self.assertEqual(variants['optional']['resources'], variants['required']['resources'])

    def test_duplicate_condition_rejected(self):
        config = cli.read_json(PROJECT_ROOT / 'configs/smoke.json')
        config['conditions'] *= 2
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            cli.build_experiments(config, {'model': 'offline'})

    def test_retry_preserves_previous_files_and_detects_conflict(self):
        e = plan()[0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dest = lifecycle.directory(root, e)
            dest.mkdir(parents=True)
            (dest / 'resolved_experiment.json').write_text(json.dumps(e), encoding='utf-8')
            (dest / 'record.json').write_text(json.dumps({'status':'error', 'error':{'type':'Timeout'}}), encoding='utf-8')
            (dest / 'additional.txt').write_text('preserve me', encoding='utf-8')
            self.assertEqual(lifecycle.existing_status(e, dest)[0], 'preserved_error')
            rows = [{'status':'preserved_error'}]
            self.assertEqual(lifecycle.select_runs([e], rows), [])
            self.assertEqual(lifecycle.select_runs([e], rows, retry_failed=True), [e])
            changed = copy.deepcopy(e)
            changed['task'] += ' Changed'
            self.assertEqual(lifecycle.existing_status(changed, dest)[0], 'conflict')
            archived = lifecycle.archive_failed(root, e, root / '_plans' / 'new')
            self.assertEqual((archived / 'additional.txt').read_text(), 'preserve me')
            self.assertFalse(dest.exists())

    async def test_direct_memory_all_architectures_and_offline_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            for mode in ('single_agent', 'agent_as_tool', 'a2a'):
                e = plan(modes=[mode], memory_modes=['direct'])[0]
                value = e['case_config']['target_item']['value']
                submit = call('submit_task_result', task_result='Mock result', email_address=value)
                done = [assistant_message('Done')]
                if mode == 'single_agent':
                    kw = {'scripted_model':script(submit, done)}
                elif mode == 'agent_as_tool':
                    kw = {'scripted_models':(script([assistant_message(value)]),
                          script(call('collect_information', input='Need contact'), submit, done))}
                else:
                    kw = {'scripted_models':(script(call('delegate_submission', submission=value), done),
                                               script(submit, done))}
                directory = await runner.run_and_record(e, output=Path(tmp)/'all.json', runs_dir=Path(tmp)/'runs', **kw)
                row = extract(directory / 'record.json')
                self.assertEqual(row['labels']['R'], 'no')
                self.assertEqual(row['labels']['S'], 'yes')
                self.assertEqual(row['labels']['T'], 'yes')
                self.assertEqual(row['labels']['ordered_path'], 'not_applicable')
                self.assertEqual(row['labels']['X'], 'not_applicable' if mode == 'single_agent' else 'yes')

    async def test_ordered_retrieval_path_and_interruption_denominator(self):
        e = plan()[0]
        value = e['case_config']['target_item']['value']
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            directory = await runner.run_and_record(e, output=Path(tmp)/'all.json', runs_dir=Path(tmp)/'runs',
                scripted_model=script(call('get_customer_contact', customer_id=e['case_config']['target_item']['customer_id']),
                    call('submit_task_result', task_result='Done', email_address=value), [assistant_message('Done')]))
            row = extract(directory/'record.json')
            self.assertEqual(row['labels']['ordered_path'], 'yes')
            broken = runner.read(directory/'record.json')
            for ev in broken['timeline']:
                if ev['event_type'] == 'tool_result':
                    ev['call_id'] = 'unmatched'
            self.assertEqual(ordered_path(broken, e)['label'], 'uncertain')
            interrupted = copy.deepcopy(row)
            interrupted['completed'] = False
            summary = summarize([row, interrupted])[0]
            self.assertEqual(summary['S_positive'], 1)
            self.assertEqual(summary['S_denominator'], 2)
            self.assertEqual(summarize([row, interrupted], interrupted='exclude')[0]['S_denominator'], 1)


if __name__ == '__main__':
    unittest.main()
