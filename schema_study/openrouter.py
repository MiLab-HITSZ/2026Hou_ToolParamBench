"""Project-local SDK adapter: preserve OpenRouter reasoning_details in own history.

No global monkeypatch, response cache, shared conversation state, or SDK modification.
Only request timing is shared between agents in the same process/event loop.
Only non-streaming Runner.run is supported by this experiment adapter.
"""
from collections import defaultdict, deque
from contextvars import ContextVar
from copy import deepcopy
from types import SimpleNamespace
import asyncio
import time
from weakref import WeakKeyDictionary
import uuid

from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

FIELD = 'openrouter_replay_v1'
_LIMITERS = WeakKeyDictionary()


async def wait_for_request_slot(key, interval):
    """Share request spacing across agents and trials in one event loop/process."""
    if interval <= 0:
        return
    states = _LIMITERS.setdefault(asyncio.get_running_loop(), {})
    state = states.setdefault(key, {'lock': asyncio.Lock(), 'last': None})
    async with state['lock']:
        if state['last'] is not None:
            delay = interval - (time.monotonic() - state['last'])
            if delay > 0:
                await asyncio.sleep(delay)
        state['last'] = time.monotonic()


def signature(message):
    calls = message.get('tool_calls') or []
    if calls:
        return ('calls', tuple(c['id'] for c in calls))
    content = message.get('content')
    if isinstance(content, list):
        content = ''.join(x.get('text', '') for x in content if x.get('type') == 'text')
    return ('text', content or '', message.get('refusal') or '')


def restore(messages, history):
    queues, seen = defaultdict(deque), set()
    for item in history if isinstance(history, list) else []:
        data = (item.get('provider_data') or {}).get(FIELD)
        if data and data['id'] not in seen:
            queues[signature(data['message'])].append(data['message'].get('reasoning_details'))
            seen.add(data['id'])
    restored = deepcopy(messages)
    for message in restored:
        if message.get('role') == 'assistant':
            queue = queues.get(signature(message))
            if queue:
                details = queue.popleft()
                if details is None:
                    continue
                # Remove SDK-normalized reasoning; replay the original structured blocks.
                message.pop('reasoning', None)
                message.pop('reasoning_content', None)
                message['reasoning_details'] = deepcopy(details)
    if any(queues.values()):
        raise ValueError('OpenRouter reasoning history could not be matched; refusing to drop blocks')
    return restored


class _ClientProxy:
    def __init__(self, client, owner):
        self.client, self.owner = client, owner
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def __getattr__(self, name):
        return getattr(self.client, name)

    async def create(self, **kwargs):
        state = self.owner._state.get()
        if kwargs.get('stream') is True:
            raise ValueError('OpenRouter experiment adapter requires non-streaming calls')
        kwargs['messages'] = restore(kwargs['messages'], state['history'])
        extra = deepcopy(kwargs.get('extra_body') or {})
        extra.update(deepcopy(self.owner.options))
        effort = kwargs.pop('reasoning_effort', None)
        extra['reasoning'] = {'effort': effort} if isinstance(effort, str) else {'enabled': True}
        kwargs['extra_body'] = extra
        await wait_for_request_slot(self.owner._rate_key, self.owner.min_request_interval)
        response = await self.client.chat.completions.create(**kwargs)
        if response.choices:
            message = response.choices[0].message.model_dump(exclude_none=True)
            state['replay'] = {'id': uuid.uuid4().hex, 'message': message}
        return response


class OpenRouterModel(OpenAIChatCompletionsModel):
    def __init__(self, model, openai_client, options=None):
        self.options = deepcopy(options or {})
        self.min_request_interval = float(self.options.pop('min_request_interval', 0))
        if not 0 <= self.min_request_interval < float('inf'):
            raise ValueError('min_request_interval must be finite and nonnegative')
        self._rate_key = (str(openai_client.base_url), model)
        if set(self.options) - {'provider'}:
            raise ValueError('Only provider routing is supported in openrouter_options')
        self._state = ContextVar('openrouter_request_state')
        super().__init__(model=model, openai_client=_ClientProxy(openai_client, self))

    async def get_response(self, *args, **kwargs):
        history = kwargs.get('input', args[1] if len(args) > 1 else None)
        state = {'history': history, 'replay': None}
        token = self._state.set(state)
        try:
            response = await super().get_response(*args, **kwargs)
            if state['replay']:
                for item in response.output:
                    item.provider_data = {**(getattr(item, 'provider_data', None) or {}),
                                          FIELD: deepcopy(state['replay'])}
            return response
        finally:
            self._state.reset(token)

    async def stream_response(self, *args, **kwargs):
        raise ValueError('Use Runner.run, not streaming, with OpenRouterModel')
        yield  # async generator interface
