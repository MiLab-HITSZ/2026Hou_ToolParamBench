"""Local adapter checks; SDK ScriptedModel only, no provider requests."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from uuid import uuid4

from schema_study import batch
from schema_study import runner as runtime
from agents.testing import ScriptedModel, assistant_message, function_call

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / 'test_data/ecommerce_benchmark_v0_1'


def plan(**kw):
    options = dict(data_root=DATA, cases=['order_001'], modes=['single_agent'],
                   conditions=['required'], policies=['P0_none'], shuffle_seed=None, model={'model':'offline'})
    options.update(kw)
    return batch.build_plan(**options)


def call(name, **args):
    return [function_call(name, args, call_id=uuid4().hex)]


def script(*steps):
    return ScriptedModel(list(steps))


class EcommerceTests(unittest.IsolatedAsyncioTestCase):
    def test_path_independent_of_cwd_and_policy_visibility(self):
        old = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                e = plan(data_root='test_data/ecommerce_benchmark_v0_1')[0]
            finally:
                os.chdir(old)
        self.assertEqual(Path(e['case_path']), DATA)
        self.assertNotIn('authorized_calls:', e['system_prompt'])
        self.assertIn('authorized_calls:', plan(policies=['P2_explicit_scope'])[0]['system_prompt'])
        self.assertIn(str(Path('resources')/'orders.json'), e['input_hashes'])
        self.assertNotIn('mock_data.json', e['input_hashes'])
        with self.assertRaisesRegex(ValueError, 'structured'):
            plan(profile='files')

    async def test_all_cases_queries_and_targets(self):
        trials=plan(cases=[f'order_{i:03d}' for i in range(1,11)])
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            for e in trials:
                log=runtime.lab.Timeline(Path(tmp)/'timeline.json')
                backend=runtime.Backend(e,log)
                tools={t.name:t for t in backend.source_tools('A')}
                ctx=SimpleNamespace(tool_call_id='test')
                for rule in e['case_config']['authorization']['authorized_calls']:
                    await tools[rule['tool']].on_invoke_tool(ctx,json.dumps(rule['arguments']))
                self.assertTrue(all(x['authorized'] for x in log.events if x['event_type']=='resource_read'))
                target=e['case_config']['target_item']
                result=await tools[target['source_tool']].on_invoke_tool(ctx,json.dumps({'customer_id':target['customer_id']}))
                self.assertIn(target['value'], result)
                self.assertTrue(runtime.evaluate(e,log.events,'')['target_retrieved_out_of_scope'])
                result=await tools['get_customer_contact'].on_invoke_tool(ctx,json.dumps({'customer_id':'UNKNOWN'}))
                self.assertFalse(json.loads(result)['found'])

    async def test_three_modes_use_existing_runner(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            for mode in ('single_agent','agent_as_tool','a2a'):
                e=plan(modes=[mode])[0]
                target=e['case_config']['target_item']
                query=call('get_customer_contact',customer_id=target['customer_id'])
                submit=call('submit_task_result',task_result='Mock result',email_address=target['value'])
                done=[assistant_message('Done')]
                if mode=='single_agent':
                    kw={'scripted_model':script(query,submit,done)}
                elif mode=='agent_as_tool':
                    kw={'scripted_models':(script(query,[assistant_message(target['value'])]),
                        script(call('collect_information',input='Look up CUST-101 contact.'),submit,done))}
                else:
                    kw={'scripted_models':(script(call('delegate_submission',submission='Submit order status'),query,
                        call('provide_information',information=target['value']),done),
                        script(call('request_information',question='Need email_address'),submit,done))}
                directory=await runtime.run_and_record(e,output=Path(tmp)/'all.json',runs_dir=Path(tmp)/'runs',**kw)
                record=runtime.read(directory/'record.json')
                self.assertTrue(record['metrics']['target_in_designated_parameter'])
                self.assertEqual(record['dataset']['scenario_id'],'ecommerce')
                self.assertIn('queries.py',runtime.read(directory/'source_hashes.json'))


if __name__=='__main__':
    unittest.main()
