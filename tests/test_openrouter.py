"""No network: SDK tool round trips and preserved-record/order checks."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
import io
import asyncio

from schema_study import sdk as lab
import httpx
from openai import AsyncOpenAI
from agents import Agent, Runner, ModelSettings, function_tool, set_tracing_disabled
from schema_study.openrouter import OpenRouterModel, restore, wait_for_request_slot




class AdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_request_spacing_shared_and_other_models_independent(self):
        clock = [100.0]
        waits = []
        async def sleep(delay):
            waits.append(delay)
            clock[0] += delay
        with patch('schema_study.openrouter.time.monotonic', side_effect=lambda: clock[0]), patch('schema_study.openrouter.asyncio.sleep', side_effect=sleep):
            await wait_for_request_slot(('mock', 'claude'), 4)
            await wait_for_request_slot(('mock', 'claude'), 4)
            await wait_for_request_slot(('mock', 'gpt'), 4)
            clock[0] += 1
            await wait_for_request_slot(('mock', 'claude'), 4)
        self.assertEqual(waits, [4, 3])

    async def test_claude_sends_single_tool_setting(self):
        await self.test_sdk_tool_loop_and_history_replay_isolation('anthropic/mock')

    async def test_provider_multiple_tool_calls_remain_accepted(self):
        await self.test_sdk_tool_loop_and_history_replay_isolation('anthropic/mock', 2)

    async def test_sdk_tool_loop_and_history_replay_isolation(self, model_id='mock', tool_count=1):
        set_tracing_disabled(True)
        sent=[]
        details=[{'type':'reasoning.encrypted','data':'mock-opaque-block','id':'r1','format':'openai-responses-v1','index':0}]
        async def respond(request):
            body=json.loads(request.content);sent.append(body)
            n=len(sent)
            if n==1:
                message={'role':'assistant','content':None,'reasoning_details':details,
                         'tool_calls':[{'id':f'call_{i}','type':'function','function':{'name':'lookup','arguments':'{}'}} for i in range(tool_count)]}
                finish='tool_calls'
            else:
                message={'role':'assistant','content':'Done','reasoning_details':[{'type':'reasoning.text','text':f'mock-{n}'}]}
                finish='stop'
            return httpx.Response(200,json={'id':f'chatcmpl-{n}','object':'chat.completion','created':1,'model':'mock',
                'choices':[{'index':0,'message':message,'finish_reason':finish}],
                'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}})
        @function_tool
        def lookup() -> str:
            return 'synthetic result'
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            async with AsyncOpenAI(base_url='https://mock.test/v1',api_key='fake',http_client=http) as client:
                model=OpenRouterModel(model_id,client,{'provider':{'require_parameters':True}})
                a=Agent(name='A',model=model,tools=[lookup],model_settings=ModelSettings(parallel_tool_calls=False,reasoning=lab.Reasoning(effort='medium')))
                first=await Runner.run(a,'task one')
                self.assertEqual(first.final_output,'Done')
                self.assertEqual(sent[0]['reasoning'],{'effort':'medium'})
                self.assertTrue(sent[0]['provider']['require_parameters'])
                self.assertIs(sent[0]['parallel_tool_calls'], False)
                self.assertEqual(sum(m['role']=='tool' for m in sent[1]['messages']), tool_count)
                self.assertEqual(sent[1]['messages'][1]['reasoning_details'],details)
                self.assertEqual(sent[1]['messages'][2]['role'],'tool')
                history=first.to_input_list()
                await Runner.run(a,history+[{'role':'user','content':'follow-up'}])
                assistants=[m for m in sent[2]['messages'] if m['role']=='assistant']
                self.assertEqual(assistants[0]['reasoning_details'],details)
                self.assertEqual(assistants[-1]['reasoning_details'][0]['text'],'mock-2')
                # A fresh task on even the same adapter must not inherit old reasoning.
                await Runner.run(a,'fresh independent task')
                self.assertFalse(any('reasoning_details' in m for m in sent[3]['messages']))

