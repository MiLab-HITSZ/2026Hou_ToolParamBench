"""SDK bootstrap, model construction, and request/event recording."""
from __future__ import annotations
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from .paths import PROJECT_ROOT

# The SDK is supplied separately by the repository owner. Prefer a local checkout.
explicit = os.environ.get('OPENAI_AGENTS_SDK_PATH')
candidates = ([Path(explicit).expanduser()] if explicit else
              [PROJECT_ROOT / 'openai-agents-python', PROJECT_ROOT.parent / 'openai-agents-python'])
for candidate in candidates:
    source = candidate / 'src'
    if (source / 'agents' / '__init__.py').is_file():
        sys.path.insert(0, str(source.resolve()))
        break
else:
    if explicit:
        raise FileNotFoundError('OPENAI_AGENTS_SDK_PATH must point to the SDK checkout containing src/agents')

import httpx
from openai import AsyncOpenAI
from openai.types.shared.reasoning import Reasoning
try:
    from agents import Agent, Runner, ModelSettings, set_tracing_disabled
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError('Provide openai-agents-python beside this project or install its dependencies; see README.md') from exc
from agents.models.openai_responses import OpenAIResponsesModel
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

@dataclass
class Config:
    provider: str = 'deepseek'
    model: str = 'deepseek-v4-flash'
    api_type: str = 'responses'
    base_url: str | None = 'https://api.deepseek.com'
    api_key_env: str = 'DEEPSEEK_API_KEY'
    tool_choice: str = 'auto'
    reasoning_effort: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    max_turns: int = 8
    trial_timeout: float = 180
    request_timeout: float = 60
    live: bool = False
    openrouter_options: dict | None = None

    def __post_init__(self):
        if self.api_type not in ('responses', 'chat_completions'):
            raise ValueError('api_type must be responses or chat_completions')

def run_folder_name(c):
    """用实际配置命名；unset 区分未发送设置与字符串 none。"""
    effort = "unset" if c.reasoning_effort is None else str(c.reasoning_effort)
    choice = str(c.tool_choice)
    safe = lambda value: "".join(ch if ch.isascii() and (ch.isalnum() or ch in "-_") else "_" for ch in value)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-reasoning_{safe(effort)}-tool_choice_{safe(choice)}-{uuid.uuid4().hex[:8]}"


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def make_sdk_model(config, client):
    """Select the SDK adapter per agent, without changing SDK global defaults."""
    if config.provider == 'openrouter':
        if config.api_type != 'chat_completions':
            raise ValueError('OpenRouter experiment adapter requires chat_completions')
        from .openrouter import OpenRouterModel
        return OpenRouterModel(config.model, client, config.openrouter_options)
    # Keep the experiment's canonical identity; official API uses an unprefixed ID.
    api_model = config.model.removeprefix('openai/') if config.provider == 'openai' else config.model
    if config.api_type == "chat_completions":
        return OpenAIChatCompletionsModel(model=api_model, openai_client=client)
    if config.api_type == "responses":
        return OpenAIResponsesModel(model=api_model, openai_client=client)
    raise ValueError("api_type must be responses or chat_completions")


class Timeline:
    def __init__(self, path):
        self.path, self.events, self.round = path, [], 0

    def add(self, kind, **details):
        event = {"sequence": len(self.events) + 1, "event_type": kind,
                 "request_attempt": self.round, "monotonic_ns": time.monotonic_ns(), **details}
        self.events.append(event)
        # 每个事件后保存完整 JSON 数组，便于展开查看；异常前的事件也已落盘。
        dump(self.path, self.events)
        print(f"  [{kind}] request={self.round} {details.get('server', '')} {details.get('tool', '')}", flush=True)

    async def request(self, request):
        if request.method == "POST" and request.url.path.rstrip("/").endswith(("/responses", "/chat/completions")):
            self.round += 1
            self.add("model_request", body=json.loads(await request.aread()))

    async def response(self, response):
        if response.request.method == "POST" and response.request.url.path.rstrip("/").endswith(("/responses", "/chat/completions")):
            raw = await response.aread()
            try:
                body = json.loads(raw)
            except ValueError:
                body = raw.decode(errors="replace")
            self.add("model_response", status=response.status_code, body=body)
