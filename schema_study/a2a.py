"""Host-mediated, stateful A2A JSON-RPC over two real loopback HTTP endpoints.

Adapted from v1 a2a_lab/host.py and agent_server.py; uses installed a2a-sdk 1.x.
Servers share a process, but each owns its agent, context history and task store.
"""
from __future__ import annotations

import asyncio
import json
import socket
import uuid
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager

import httpx
import uvicorn
from google.protobuf.json_format import MessageToDict
from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.helpers import get_message_text, new_task_from_user_message, new_text_message
from a2a.server.agent_execution import AgentExecutor
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCard, AgentCapabilities, AgentInterface, AgentSkill, Role, SendMessageRequest, TaskState
from starlette.applications import Starlette

from . import runner as runtime


class NotebookServer(uvicorn.Server):
    @contextmanager
    def capture_signals(self):
        yield


class Executor(AgentExecutor):
    def __init__(self, actor, agent, config, pending, root, timeline):
        self.actor, self.agent, self.config = actor, agent, config
        self.pending, self.root, self.timeline = pending, root, timeline
        self.histories = {}
        self.lock = asyncio.Lock()

    async def execute(self, context, event_queue):
        async with self.lock:
            task = context.current_task
            if task is None:
                task = new_task_from_user_message(context.message)
                await event_queue.enqueue_event(task)
            updater = TaskUpdater(event_queue, task.id, task.context_id)
            self.timeline.add("a2a_received", agent=self.actor, task_id=task.id,
                              context_id=task.context_id, message=MessageToDict(context.message))
            await updater.update_status(TaskState.TASK_STATE_WORKING)
            history = list(self.histories.get(task.context_id, []))
            history.append({"role": "user", "content": get_message_text(context.message)})
            self.timeline.add("agent_turn_input", agent=self.actor, input=history)
            self.pending.clear()
            try:
                result = await runtime.lab.Runner.run(self.agent, history, max_turns=self.config.max_turns)
                self.histories[task.context_id] = result.to_input_list()
                runtime.atomic_json(self.root / f"conversation_{self.actor}.json", result.to_input_list())
                if len(self.pending) > 1:
                    raise ValueError("Only one peer request per A2A turn is supported")
                payload = self.pending[0] if self.pending else {"kind": "final", "content": str(result.final_output)}
                state = (TaskState.TASK_STATE_INPUT_REQUIRED if self.pending
                         else TaskState.TASK_STATE_COMPLETED)
                message = new_text_message(json.dumps(payload, ensure_ascii=False))
                await updater.update_status(state, message=message)
                self.timeline.add("a2a_reply", agent=self.actor, task_id=task.id, context_id=task.context_id,
                                  state=TaskState.Name(state), message=MessageToDict(message))
            except BaseException as exc:
                self.timeline.add("a2a_execution_error", agent=self.actor, error_type=type(exc).__name__, error=str(exc))
                await updater.update_status(TaskState.TASK_STATE_FAILED,
                    message=new_text_message(f"Agent execution failed: {type(exc).__name__}"))
                raise

    async def cancel(self, context, event_queue):
        if context.current_task:
            task = context.current_task
            await TaskUpdater(event_queue, task.id, task.context_id).update_status(TaskState.TASK_STATE_CANCELED)


@asynccontextmanager
async def serve(actor, agent, config, pending, root, timeline):
    sock = socket.socket()
    server = worker = handler = None
    try:
        sock.bind(("127.0.0.1", 0))
        sock.listen(128)
        sock.setblocking(False)
        url = f"http://127.0.0.1:{sock.getsockname()[1]}/"
        card = AgentCard(name=f"WorkspaceAgent{actor}", description="Local synthetic workspace agent",
            version="0.1", default_input_modes=["text/plain"], default_output_modes=["text/plain"],
            capabilities=AgentCapabilities(streaming=False),
            supported_interfaces=[AgentInterface(protocol_binding="JSONRPC", protocol_version="1.0", url=url)],
            skills=[AgentSkill(id=f"workspace_{actor}", name=f"Workspace {actor}",
                              description="Handle the user task and delegate submission" if actor == "A" else "Execute submission subtask", tags=["synthetic"])])
        runtime.atomic_json(root / f"agent_card_{actor}.json", MessageToDict(card))
        executor = Executor(actor, agent, config, pending, root, timeline)
        handler = DefaultRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore(), agent_card=card)
        app = Starlette(routes=[*create_agent_card_routes(card), *create_jsonrpc_routes(handler, "/")])
        server = NotebookServer(uvicorn.Config(app, host="127.0.0.1", log_level="critical",
            access_log=False, lifespan="off", timeout_graceful_shutdown=1))
        # Do not install process-wide signal handlers in the notebook kernel.
        worker = asyncio.create_task(server.serve(sockets=[sock]))
        async with asyncio.timeout(10):
            while not server.started:
                if worker.done():
                    await worker
                    raise RuntimeError(f"Agent {actor} server exited during startup")
                await asyncio.sleep(.01)
        timeline.add("a2a_server_started", agent=actor, url=url)
        yield url
    finally:
        if handler is not None:
            await handler.aclose()
        if server is not None:
            server.should_exit = True
        if worker is not None:
            try:
                await asyncio.wait_for(asyncio.shield(worker), 3)
            except (TimeoutError, asyncio.CancelledError):
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
        sock.close()
        timeline.add("a2a_server_stopped", agent=actor)


async def exchange(e, root, timeline, backend, make_agent, configs):
    if e.get("a2a_flow") != "a_delegates_submission_v1":
        raise ValueError("Rebuild the A2A experiment for the A-led submission flow")
    prompts = runtime.effective_instructions(e)
    pending = {"A": [], "B": []}
    agents = {}
    for actor in ("A", "B"):
        definitions = ([
            ("delegate_submission", "submission", "Delegate the submission subtask to B, including your collected information."),
            ("provide_information", "information", "Return the additional information or inability explanation requested by B."),
        ] if actor == "A" else [("request_information", "question", e["delegation_description"])])
        peers = []
        for name, argument, description in definitions:
            def queue(args, actor=actor, argument=argument, name=name):
                pending[actor].append({"kind": name, "content": args[argument]})
                return "Queued for delivery through A2A."
            peers.append(backend.tool(name, description,
                runtime.object_schema(**{argument: "Message to the peer agent."}), actor, queue))
        tools = [*backend.source_tools(actor), *peers] if actor == "A" else [*peers, backend.sink_tool(actor)]
        agents[actor] = await make_agent(actor, prompts[actor], tools)
        agents[actor].tool_use_behavior = {"stop_at_tool_names": [t.name for t in peers]}
        timeline.add("agent_initial_input", agent=actor, instructions=prompts[actor],
                     input=[{"role": "user", "content": e["task"]}] if actor == "A" else [])

    async with AsyncExitStack() as stack:
        http = await stack.enter_async_context(httpx.AsyncClient(timeout=e["config"]["trial_timeout"], trust_env=False))
        clients = {}
        for actor in ("A", "B"):
            url = await stack.enter_async_context(serve(actor, agents[actor], configs[actor], pending[actor], root, timeline))
            card = await A2ACardResolver(httpx_client=http, base_url=url).get_agent_card()
            timeline.add("a2a_card_resolved", agent=actor, card=MessageToDict(card))
            clients[actor] = await create_client(agent=card, client_config=ClientConfig(streaming=False, httpx_client=http))
            stack.push_async_callback(clients[actor].close)

        async def send(actor, text, context_id, task_id=None):
            message = new_text_message(text, role=Role.ROLE_USER)
            message.context_id = context_id
            if task_id:
                message.task_id = task_id
            timeline.add("a2a_send", recipient=actor, message=MessageToDict(message))
            task = None
            async for response in clients[actor].send_message(SendMessageRequest(message=message)):
                timeline.add("a2a_response", sender=actor, response=MessageToDict(response))
                if response.HasField("task"):
                    task = response.task
            if task is None or task.status.state not in (TaskState.TASK_STATE_COMPLETED, TaskState.TASK_STATE_INPUT_REQUIRED):
                raise RuntimeError(f"Agent {actor} did not return a successful A2A task")
            payload = json.loads(get_message_text(task.status.message))
            if not isinstance(payload, dict) or not isinstance(payload.get("content"), str):
                raise RuntimeError("Invalid A2A peer response envelope")
            return task, payload

        contexts = {actor: uuid.uuid4().hex for actor in ("A", "B")}
        task_a, payload = await send("A", e["task"], contexts["A"])
        if payload["kind"] == "final":
            timeline.add("a2a_completed", roundtrips=0, submission_delegated=False)
            runtime.atomic_json(root / "conversation.json", runtime.read(root / "conversation_A.json"))
            return payload["content"]
        if payload["kind"] != "delegate_submission":
            raise RuntimeError("A must delegate_submission before providing supplementary information")
        timeline.add("subtask_delegated", agent="A", recipient="B", output=payload["content"])
        timeline.add("delegation_result", agent="A", output=payload["content"], transfer_kind="subtask", roundtrip=0)
        task_b, payload = await send("B", json.dumps({"source": "Agent A", "kind": "submission_subtask",
            "content": payload["content"]}, ensure_ascii=False), contexts["B"])
        rounds, revisions = 0, 0
        while True:
            if task_b.status.state == TaskState.TASK_STATE_COMPLETED:
                if payload["kind"] != "final":
                    raise RuntimeError("Expected B's final submission outcome")
                timeline.add("subtask_result", agent="B", recipient="A", output=payload["content"], revision=revisions)
                task_a, final = await send("A", json.dumps({"source": "Agent B", "kind": "submission_outcome",
                    "content": payload["content"]}, ensure_ascii=False), contexts["A"],
                    task_a.id if task_a.status.state == TaskState.TASK_STATE_INPUT_REQUIRED else None)
                if final["kind"] == "final":
                    break
                if final["kind"] not in ("provide_information", "delegate_submission"):
                    raise RuntimeError("Unexpected response to B's submission outcome")
                if revisions >= e.get("max_submission_revisions", 2):
                    timeline.add("a2a_revision_limit", limit=e.get("max_submission_revisions", 2),
                                 pending_correction=final["content"])
                    raise RuntimeError("A2A max_submission_revisions exceeded")
                revisions += 1
                timeline.add("submission_revision", agent="A", recipient="B", output=final["content"], revision=revisions)
                timeline.add("delegation_result", agent="A", output=final["content"],
                             transfer_kind="submission_revision", revision=revisions)
                # A completed A2A task cannot be resumed; keep B's context/history but start a new task.
                task_b, payload = await send("B", json.dumps({"source": "Agent A", "kind": "submission_revision",
                    "content": final["content"]}, ensure_ascii=False), contexts["B"])
                continue
            if payload["kind"] != "request_information":
                raise RuntimeError("Unexpected request from Agent B")
            if rounds >= e["max_roundtrips"]:
                raise RuntimeError("A2A max_roundtrips exceeded")
            rounds += 1
            timeline.add("delegation_request", agent="B", arguments={"question": payload["content"]}, roundtrip=rounds)
            task_a, reply = await send("A", json.dumps({"source": "Agent B", "kind": "supplement_request",
                "content": payload["content"]}, ensure_ascii=False), contexts["A"],
                task_a.id if task_a.status.state == TaskState.TASK_STATE_INPUT_REQUIRED else None)
            if reply["kind"] not in ("provide_information", "final"):
                raise RuntimeError("A must answer B's request, not delegate another subtask")
            information = reply["content"]
            # Record the actual decoded A2A response, not an expected synthetic value.
            timeline.add("delegation_result", agent="A", output=information, roundtrip=rounds)
            task_b, payload = await send("B", json.dumps({"source": "Agent A", "kind": "supplement",
                "content": information}, ensure_ascii=False), contexts["B"], task_b.id)
        timeline.add("a2a_completed", roundtrips=rounds, submission_delegated=True, submission_revisions=revisions)
        runtime.atomic_json(root / "conversation.json", runtime.read(root / "conversation_A.json"))
        return final["content"]
