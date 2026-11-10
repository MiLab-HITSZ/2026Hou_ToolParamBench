"""Four scenario adapters: local plans, scope checks and scripted SDK execution."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from schema_study import batch
from schema_study import runner as runtime
from schema_study.queries import SCENARIOS
from tests.test_ecommerce_entry import call, script
from agents.testing import assistant_message

PREFIXES = dict(finance_admin='finance', it_operations='it', power_grid='grid', travel='travel')
HERE = Path(__file__).resolve().parents[1]


def plan(scenario, **kw):
    defaults = dict(data_root=HERE/'test_data'/f'{scenario}_benchmark_v0_1',
                    cases=[f'{PREFIXES[scenario]}_001'], modes=['single_agent'],
                    conditions=['required'], policies=['P0_none'], shuffle_seed=None, model={'model':'offline'})
    defaults.update(kw)
    return batch.build_plan(**defaults)


def target_query(e):
    target = e['case_config']['target_item']
    table, columns, arg = SCENARIOS[e['query_adapter']][target['source_tool']]
    rows = json.loads(e['resources'][table]['text'])
    row = next(r for r in rows if r.get(target['parameter']) == target['value'])
    return target['source_tool'], {arg: row[columns[0]]}


class ScenarioTests(unittest.IsolatedAsyncioTestCase):
    def test_full_plans_and_policy_scope(self):
        for scenario, prefix in PREFIXES.items():
            with self.subTest(scenario=scenario):
                trials=plan(scenario,cases=[f'{prefix}_{i:03d}' for i in range(1,11)],
                            modes=['single_agent','agent_as_tool','a2a'],
                            conditions=['absent','optional','required'],policies=list(batch.workspace.POLICIES))
                self.assertEqual(len(trials),450)
                for e in trials:
                    self.assertEqual(e['case_config']['authorization_granularity'],'tool')
                    self.assertEqual(len(e['source_tools']),6)
                    self.assertTrue(e['resource_availability']['target_value_available'])
                    explicit=e['policy'] in batch.workspace.POLICIES[2:]
                    self.assertEqual('authorized_tools:' in e['system_prompt'],explicit)
                    self.assertNotIn('or query other records',e['system_prompt'])
                    if not explicit:
                        self.assertNotIn('Authorization is defined at tool level',e['system_prompt'])

    async def test_all_targets_and_tool_level_scope(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            for scenario, prefix in PREFIXES.items():
                for e in plan(scenario,cases=[f'{prefix}_{i:03d}' for i in range(1,11)]):
                    log=runtime.lab.Timeline(Path(tmp)/'timeline.json')
                    backend=runtime.Backend(e,log)
                    tools={t.name:t for t in backend.source_tools('A')}
                    ctx=SimpleNamespace(tool_call_id='test')
                    normal=e['case_config']['authorized_tools'][0]
                    table,columns,arg=SCENARIOS[scenario][normal]
                    # All records queried with an authorized tool are authorized, including other objects.
                    for row in json.loads(e['resources'][table]['text']):
                        await tools[normal].on_invoke_tool(ctx,json.dumps({arg:row[columns[0]]}))
                    self.assertTrue(all(x['authorized'] for x in log.events if x['event_type']=='resource_read'))
                    name,args=target_query(e)
                    result=await tools[name].on_invoke_tool(ctx,json.dumps(args))
                    self.assertIn(e['case_config']['target_item']['value'],result)
                    self.assertTrue(runtime.evaluate(e,log.events,'')['target_retrieved_out_of_scope'])
                    result=await tools[name].on_invoke_tool(ctx,json.dumps({next(iter(args)):'UNKNOWN'}))
                    self.assertFalse(json.loads(result)['found'])

    async def test_three_modes_for_each_scenario(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            for scenario in PREFIXES:
                for mode in ('single_agent','agent_as_tool','a2a'):
                    e=plan(scenario,modes=[mode])[0]
                    target=e['case_config']['target_item']
                    name,args=target_query(e)
                    query=call(name,**args)
                    submit=call('submit_task_result',task_result='Mock result',**{target['parameter']:target['value']})
                    done=[assistant_message('Done')]
                    if mode=='single_agent':
                        kw={'scripted_model':script(query,submit,done)}
                    elif mode=='agent_as_tool':
                        kw={'scripted_models':(script(query,[assistant_message(target['value'])]),
                            script(call('collect_information',input='Retrieve information'),submit,done))}
                    else:
                        kw={'scripted_models':(script(call('delegate_submission',submission='Submit result'),query,
                            call('provide_information',information=target['value']),done),
                            script(call('request_information',question='Need target field'),submit,done))}
                    directory=await runtime.run_and_record(e,output=Path(tmp)/'all.json',runs_dir=Path(tmp)/'runs',**kw)
                    record=runtime.read(directory/'record.json')
                    self.assertTrue(record['metrics']['target_in_designated_parameter'])
                    self.assertEqual(record['dataset']['scenario_id'],scenario)
                    self.assertIn('queries.py',runtime.read(directory/'source_hashes.json'))


if __name__=='__main__':
    unittest.main()
