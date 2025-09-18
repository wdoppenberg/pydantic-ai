from __future__ import annotations as _annotations

from typing import Any

import pytest
from dirty_equals import IsListOrTuple
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.messages import (
    FinalResultEvent,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage

from ..conftest import IsDatetime, IsStr, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.deepseek import DeepSeekProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


async def test_deepseek_model_thinking_part(allow_model_requests: None, deepseek_api_key: str):
    deepseek_model = OpenAIChatModel('deepseek-reasoner', provider=DeepSeekProvider(api_key=deepseek_api_key))
    agent = Agent(model=deepseek_model)
    result = await agent.run('How do I cross the street?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='How do I cross the street?', timestamp=IsDatetime())]),
            ModelResponse(
                parts=[
                    ThinkingPart(content=IsStr(), id='reasoning_content', provider_name='deepseek'),
                    TextPart(content=IsStr()),
                ],
                usage=RequestUsage(
                    input_tokens=12,
                    output_tokens=789,
                    details={
                        'prompt_cache_hit_tokens': 0,
                        'prompt_cache_miss_tokens': 12,
                        'reasoning_tokens': 415,
                    },
                ),
                model_name='deepseek-reasoner',
                timestamp=IsDatetime(),
                provider_name='deepseek',
                provider_details={'finish_reason': 'stop'},
                provider_response_id='181d9669-2b3a-445e-bd13-2ebff2c378f6',
                finish_reason='stop',
            ),
        ]
    )


async def test_deepseek_model_thinking_stream(allow_model_requests: None, deepseek_api_key: str):
    deepseek_model = OpenAIChatModel('deepseek-reasoner', provider=DeepSeekProvider(api_key=deepseek_api_key))
    agent = Agent(model=deepseek_model)

    event_parts: list[Any] = []
    async with agent.iter(user_prompt='Hello') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    assert event_parts == IsListOrTuple(
        positions={
            0: snapshot(
                PartStartEvent(
                    index=0, part=ThinkingPart(content='H', id='reasoning_content', provider_name='deepseek')
                )
            ),
            1: snapshot(PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='mm', provider_name='deepseek'))),
            2: snapshot(PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',', provider_name='deepseek'))),
            198: snapshot(PartStartEvent(index=1, part=TextPart(content='Hello'))),
            199: snapshot(FinalResultEvent(tool_name=None, tool_call_id=None)),
            200: snapshot(PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' there'))),
            201: snapshot(PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='!'))),
        },
        length=211,
    )
