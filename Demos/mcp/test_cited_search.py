"""离线验收 Notebook 的可选练习；不读取环境凭据或请求外部服务。

运行：python -m pytest Demos/mcp/test_cited_search.py -q
"""

import ast
import asyncio
from contextlib import asynccontextmanager
import copy
import json
from pathlib import Path

import httpx
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool
from openai import AsyncOpenAI, AuthenticationError
import pytest


NOTEBOOK = Path(__file__).parents[1] / "deepseek-api-guide-5.ipynb"
URL = "https://docs.python.org/3/library/asyncio-runner.html?lang=zh&v=3#run"
QUESTION = "请查阅公开文档解释 Notebook 的事件循环，并引用来源。"
ARGUMENTS = {"objective": QUESTION, "search_queries": ["site:docs.python.org asyncio run notebook"]}
PAYLOAD = {"results": [{"url": URL, "title": "Runners", "excerpts": ["同一线程不能嵌套事件循环。"]}]}
TEXT = json.dumps(PAYLOAD, ensure_ascii=False, indent=2)


def notebook_scope():
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    scope = {}
    source = next("".join(c["source"]) for c in cells if c["id"] == "cited-search-functions")
    exec(compile(source, str(NOTEBOOK), "exec"), scope)
    return cells, scope


def setup_application(monkeypatch, *, answer=None, thinking=False, structured=True,
                      tool_error=False, no_results=False, bad_call=False,
                      http_error=False, search_exception=None, malformed_arguments=False):
    cells, scope = notebook_scope()
    events = []
    requests = []
    result = CallToolResult(
        content=[TextContent(type="text", text=TEXT)],
        structuredContent=PAYLOAD if structured else None,
        isError=tool_error,
    )
    if no_results:
        result = CallToolResult(content=[TextContent(type="text", text='{"results": []}')])

    class Session:
        def __init__(self, read, write):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            events.append("session_closed")

        async def initialize(self):
            events.append("initialized")

        async def list_tools(self):
            return ListToolsResult(tools=[Tool(name="web_search", description="Search public docs",
                inputSchema={"type": "object", "properties": {
                    "objective": {"type": "string"},
                    "search_queries": {"type": "array", "items": {"type": "string"}},
                }, "required": ["objective", "search_queries"]})])

        async def call_tool(self, name, arguments):
            assert name == "web_search"
            assert arguments == ARGUMENTS
            events.append("searched")
            if search_exception:
                raise search_exception
            return result

    @asynccontextmanager
    async def transport(url, **kwargs):
        assert url == "https://mcp.parallel.ai/mcp"
        assert kwargs["headers"] == {"User-Agent": scope["SEARCH_USER_AGENT"]}
        assert kwargs["timeout"] == 30 and kwargs["sse_read_timeout"] == 60
        events.append("connected")
        try:
            yield None, None, None
        finally:
            events.append("transport_closed")

    def model_request(request):
        assert request.url.host == "model.invalid"
        assert request.headers["authorization"] == "Bearer local-test-key"
        body = json.loads(request.content)
        requests.append(body)
        if http_error:
            return httpx.Response(401, json={"error": {"message": "invalid test key", "type": "auth"}})
        if len(requests) == 1:
            message = {"role": "assistant", "content": None, "reasoning_content": "查询公开文档",
                       "tool_calls": [{"id": "call_docs", "type": "function", "function": {
                           "name": "other_tool" if bad_call else "web_search",
                           "arguments": "[]" if malformed_arguments else json.dumps(ARGUMENTS),
                       }}]}
        else:
            assert len(requests) == 2
            message = {"role": "assistant", "content": answer if answer is not None else f"请直接 await 协程。[Python 文档]({URL})"}
        return httpx.Response(200, json={"id": "local", "object": "chat.completion", "created": 0,
            "model": "test-model", "choices": [{"index": 0, "finish_reason": "stop", "message": message}]})

    clients = []

    def model_client(**kwargs):
        assert kwargs == {"api_key": "local-test-key", "base_url": "https://model.invalid/v1"}
        client = AsyncOpenAI(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(model_request)))
        clients.append(client)
        return client

    monkeypatch.setitem(scope, "ClientSession", Session)
    monkeypatch.setitem(scope, "streamablehttp_client", transport)
    monkeypatch.setitem(scope, "AsyncOpenAI", model_client)
    config = {"api_key": "local-test-key", "base_url": "https://model.invalid/v1", "model": "test-model", "thinking": thinking}
    return cells, scope, config, requests, events, clients


@pytest.mark.parametrize("thinking", [False, True, None])
@pytest.mark.parametrize("structured", [False, True])
def test_tool_round_trip(monkeypatch, thinking, structured):
    _, scope, config, requests, events, clients = setup_application(monkeypatch, thinking=thinking, structured=structured)
    before = copy.deepcopy(config)
    reply = asyncio.run(scope["answer_with_web_search"](config, QUESTION))
    assert f"]({URL})" in reply
    assert config == before
    assert len(requests) == 2
    first, second = requests
    assert first["tools"][0]["function"]["name"] == "web_search"
    assert first["tool_choice"]["function"]["name"] == "web_search"
    assert "tools" not in second and "tool_choice" not in second
    assert second["messages"][-2]["reasoning_content"] == "查询公开文档"
    assert second["messages"][-2]["tool_calls"][0]["id"] == "call_docs"
    assert second["messages"][-1] == {"role": "tool", "tool_call_id": "call_docs", "content": TEXT}
    for request in requests:
        assert request["stream"] is False
        if thinking is None:
            assert "thinking" not in request
        else:
            assert request["thinking"] == {"type": "enabled" if thinking else "disabled"}
    assert events == ["connected", "initialized", "searched", "session_closed", "transport_closed"]
    assert all(client.is_closed() for client in clients)


@pytest.mark.parametrize("changes,error,model_calls", [
    ({"answer": "请直接 await。[1]"}, ValueError, 2),
    ({"answer": "请直接 await。[文档](https://unreturned.invalid/)"}, ValueError, 2),
    ({"tool_error": True}, RuntimeError, 1),
    ({"no_results": True}, RuntimeError, 1),
    ({"bad_call": True}, RuntimeError, 1),
    ({"malformed_arguments": True}, ValueError, 1),
    ({"http_error": True}, AuthenticationError, 1),
    ({"search_exception": TimeoutError("local timeout")}, TimeoutError, 1),
])
def test_failures_stop_and_close(monkeypatch, changes, error, model_calls):
    _, scope, config, requests, events, clients = setup_application(monkeypatch, **changes)
    with pytest.raises(error):
        asyncio.run(scope["answer_with_web_search"](config, QUESTION))
    assert len(requests) == model_calls
    assert events[-2:] == ["session_closed", "transport_closed"]
    assert all(client.is_closed() for client in clients)


def test_notebook_entry_and_saved_configuration(monkeypatch, capsys):
    cells, scope, config, requests, _, _ = setup_application(monkeypatch)
    run = next("".join(c["source"]) for c in cells if c["id"] == "cited-search-run")
    code = compile(run, str(NOTEBOOK), "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    # 关闭时连 config_manager 都不需要，更不应发请求。
    asyncio.run(eval(code, scope))
    assert not requests
    assert "未开启" in capsys.readouterr().out

    # 使用仓库原有的配置控件保存设置，再执行启用后的真实入口单元格。
    for name in ("DEEPSEEK_API_KEY", "SILICONFLOW_API_KEY", "DASHSCOPE_API_KEY", "BAIDU_API_KEY", "ARK_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    for cell in cells:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and ("import ipywidgets" in source or "class APIConfigManager:" in source):
            exec(compile(source, str(NOTEBOOK), "exec"), scope)
    manager = scope["config_manager"]
    monkeypatch.setitem(manager.PLATFORM_CONFIG, "DeepSeek", {
        "api_key_name": "LOCAL_EXERCISE_KEY", "base_url": config["base_url"],
        "chat_model_id": config["model"], "reasoner_model_id": config["model"],
    })
    manager.api_key_input.value = config["api_key"]
    manager.model_id_input.value = config["model"]
    monkeypatch.setenv("LOCAL_EXERCISE_KEY", "local-test-key")
    manager._on_save_click(None)
    assert manager.get_config() == config
    run = run.replace("enable_web_search = False", "enable_web_search = True")
    code = compile(run, str(NOTEBOOK), "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    asyncio.run(eval(code, scope))
    assert len(requests) == 2
    assert URL in capsys.readouterr().out
