from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator
from datetime import timezone
from typing import Any

import pytest
from dirty_equals import IsJson
from inline_snapshot import snapshot
from pydantic_core import to_json

from pydantic_ai import Agent, ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage

from ..conftest import IsNow, try_import

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup as ExceptionGroup  # pragma: lax no cover
else:
    ExceptionGroup = ExceptionGroup  # pragma: lax no cover

with try_import() as logfire_imports_successful:
    from logfire.testing import CaptureLogfire


pytestmark = pytest.mark.anyio


def success_response(_model_messages: list[ModelMessage], _agent_info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('success')])


def failure_response(_model_messages: list[ModelMessage], _agent_info: AgentInfo) -> ModelResponse:
    raise ModelHTTPError(status_code=500, model_name='test-function-model', body={'error': 'test error'})


success_model = FunctionModel(success_response)
failure_model = FunctionModel(failure_response)


def test_init() -> None:
    fallback_model = FallbackModel(failure_model, success_model)
    assert fallback_model.model_name == snapshot('fallback:function:failure_response:,function:success_response:')
    assert fallback_model.system == 'fallback:function,function'
    assert fallback_model.base_url is None


def test_first_successful() -> None:
    fallback_model = FallbackModel(success_model, failure_model)
    agent = Agent(model=fallback_model)
    result = agent.run_sync('hello')
    assert result.output == snapshot('success')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc)),
                ]
            ),
            ModelResponse(
                parts=[TextPart(content='success')],
                usage=RequestUsage(input_tokens=51, output_tokens=1),
                model_name='function:success_response:',
                timestamp=IsNow(tz=timezone.utc),
            ),
        ]
    )


def test_first_failed() -> None:
    fallback_model = FallbackModel(failure_model, success_model)
    agent = Agent(model=fallback_model)
    result = agent.run_sync('hello')
    assert result.output == snapshot('success')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='hello',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ]
            ),
            ModelResponse(
                parts=[TextPart(content='success')],
                usage=RequestUsage(input_tokens=51, output_tokens=1),
                model_name='function:success_response:',
                timestamp=IsNow(tz=timezone.utc),
            ),
        ]
    )


@pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
def test_first_failed_instrumented(capfire: CaptureLogfire) -> None:
    fallback_model = FallbackModel(failure_model, success_model)
    agent = Agent(model=fallback_model, instrument=True)
    result = agent.run_sync('hello')
    assert result.output == snapshot('success')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='hello',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ]
            ),
            ModelResponse(
                parts=[TextPart(content='success')],
                usage=RequestUsage(input_tokens=51, output_tokens=1),
                model_name='function:success_response:',
                timestamp=IsNow(tz=timezone.utc),
            ),
        ]
    )
    assert capfire.exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'chat function:success_response:',
                'context': {'trace_id': 1, 'span_id': 3, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'start_time': 2000000000,
                'end_time': 3000000000,
                'attributes': {
                    'gen_ai.operation.name': 'chat',
                    'model_request_parameters': {
                        'function_tools': [],
                        'builtin_tools': [],
                        'output_mode': 'text',
                        'output_object': None,
                        'output_tools': [],
                        'allow_text_output': True,
                    },
                    'logfire.span_type': 'span',
                    'logfire.msg': 'chat fallback:function:failure_response:,function:success_response:',
                    'gen_ai.system': 'function',
                    'gen_ai.request.model': 'function:success_response:',
                    'gen_ai.input.messages': [{'role': 'user', 'parts': [{'type': 'text', 'content': 'hello'}]}],
                    'gen_ai.output.messages': [
                        {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'success'}]}
                    ],
                    'gen_ai.usage.input_tokens': 51,
                    'gen_ai.usage.output_tokens': 1,
                    'gen_ai.response.model': 'function:success_response:',
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'gen_ai.input.messages': {'type': 'array'},
                            'gen_ai.output.messages': {'type': 'array'},
                            'model_request_parameters': {'type': 'object'},
                        },
                    },
                },
            },
            {
                'name': 'agent run',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 4000000000,
                'attributes': {
                    'model_name': 'fallback:function:failure_response:,function:success_response:',
                    'agent_name': 'agent',
                    'logfire.msg': 'agent run',
                    'logfire.span_type': 'span',
                    'gen_ai.usage.input_tokens': 51,
                    'gen_ai.usage.output_tokens': 1,
                    'pydantic_ai.all_messages': [
                        {'role': 'user', 'parts': [{'type': 'text', 'content': 'hello'}]},
                        {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'success'}]},
                    ],
                    'final_result': 'success',
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'pydantic_ai.all_messages': {'type': 'array'},
                            'final_result': {'type': 'object'},
                        },
                    },
                },
            },
        ]
    )


@pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
async def test_first_failed_instrumented_stream(capfire: CaptureLogfire) -> None:
    fallback_model = FallbackModel(failure_model_stream, success_model_stream)
    agent = Agent(model=fallback_model, instrument=True)
    async with agent.run_stream('input') as result:
        assert [c async for c, _is_last in result.stream_responses(debounce_by=None)] == snapshot(
            [
                ModelResponse(
                    parts=[TextPart(content='hello ')],
                    usage=RequestUsage(input_tokens=50, output_tokens=1),
                    model_name='function::success_response_stream',
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ModelResponse(
                    parts=[TextPart(content='hello world')],
                    usage=RequestUsage(input_tokens=50, output_tokens=2),
                    model_name='function::success_response_stream',
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ModelResponse(
                    parts=[TextPart(content='hello world')],
                    usage=RequestUsage(input_tokens=50, output_tokens=2),
                    model_name='function::success_response_stream',
                    timestamp=IsNow(tz=timezone.utc),
                ),
            ]
        )
        assert result.is_complete

    assert capfire.exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'chat function::success_response_stream',
                'context': {'trace_id': 1, 'span_id': 3, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'start_time': 2000000000,
                'end_time': 3000000000,
                'attributes': {
                    'gen_ai.operation.name': 'chat',
                    'model_request_parameters': {
                        'function_tools': [],
                        'builtin_tools': [],
                        'output_mode': 'text',
                        'output_object': None,
                        'output_tools': [],
                        'allow_text_output': True,
                    },
                    'logfire.span_type': 'span',
                    'logfire.msg': 'chat fallback:function::failure_response_stream,function::success_response_stream',
                    'gen_ai.system': 'function',
                    'gen_ai.request.model': 'function::success_response_stream',
                    'gen_ai.input.messages': [{'role': 'user', 'parts': [{'type': 'text', 'content': 'input'}]}],
                    'gen_ai.output.messages': [
                        {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'hello world'}]}
                    ],
                    'gen_ai.usage.input_tokens': 50,
                    'gen_ai.usage.output_tokens': 2,
                    'gen_ai.response.model': 'function::success_response_stream',
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'gen_ai.input.messages': {'type': 'array'},
                            'gen_ai.output.messages': {'type': 'array'},
                            'model_request_parameters': {'type': 'object'},
                        },
                    },
                },
            },
            {
                'name': 'agent run',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 4000000000,
                'attributes': {
                    'model_name': 'fallback:function::failure_response_stream,function::success_response_stream',
                    'agent_name': 'agent',
                    'logfire.msg': 'agent run',
                    'logfire.span_type': 'span',
                    'gen_ai.usage.input_tokens': 50,
                    'gen_ai.usage.output_tokens': 2,
                    'pydantic_ai.all_messages': [
                        {'role': 'user', 'parts': [{'type': 'text', 'content': 'input'}]},
                        {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'hello world'}]},
                    ],
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'pydantic_ai.all_messages': {'type': 'array'},
                            'final_result': {'type': 'object'},
                        },
                    },
                },
            },
        ]
    )


def test_all_failed() -> None:
    fallback_model = FallbackModel(failure_model, failure_model)
    agent = Agent(model=fallback_model)
    with pytest.raises(ExceptionGroup) as exc_info:
        agent.run_sync('hello')
    assert 'All models from FallbackModel failed' in exc_info.value.args[0]
    exceptions = exc_info.value.exceptions
    assert len(exceptions) == 2
    assert isinstance(exceptions[0], ModelHTTPError)
    assert exceptions[0].status_code == 500
    assert exceptions[0].model_name == 'test-function-model'
    assert exceptions[0].body == {'error': 'test error'}


def add_missing_response_model(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for span in spans:
        attrs = span.setdefault('attributes', {})
        if 'gen_ai.request.model' in attrs:
            attrs.setdefault('gen_ai.response.model', attrs['gen_ai.request.model'])
    return spans


@pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
def test_all_failed_instrumented(capfire: CaptureLogfire) -> None:
    fallback_model = FallbackModel(failure_model, failure_model)
    agent = Agent(model=fallback_model, instrument=True)
    with pytest.raises(ExceptionGroup) as exc_info:
        agent.run_sync('hello')
    assert 'All models from FallbackModel failed' in exc_info.value.args[0]
    exceptions = exc_info.value.exceptions
    assert len(exceptions) == 2
    assert isinstance(exceptions[0], ModelHTTPError)
    assert exceptions[0].status_code == 500
    assert exceptions[0].model_name == 'test-function-model'
    assert exceptions[0].body == {'error': 'test error'}
    assert add_missing_response_model(capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)) == snapshot(
        [
            {
                'name': 'chat fallback:function:failure_response:,function:failure_response:',
                'context': {'trace_id': 1, 'span_id': 3, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'start_time': 2000000000,
                'end_time': 4000000000,
                'attributes': {
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.system': 'fallback:function,function',
                    'gen_ai.request.model': 'fallback:function:failure_response:,function:failure_response:',
                    'model_request_parameters': {
                        'function_tools': [],
                        'builtin_tools': [],
                        'output_mode': 'text',
                        'output_object': None,
                        'output_tools': [],
                        'allow_text_output': True,
                    },
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {'model_request_parameters': {'type': 'object'}},
                    },
                    'logfire.span_type': 'span',
                    'logfire.msg': 'chat fallback:function:failure_response:,function:failure_response:',
                    'logfire.level_num': 17,
                    'gen_ai.response.model': 'fallback:function:failure_response:,function:failure_response:',
                },
                'events': [
                    {
                        'name': 'exception',
                        'timestamp': 3000000000,
                        'attributes': {
                            'exception.type': 'pydantic_ai.exceptions.FallbackExceptionGroup',
                            'exception.message': 'All models from FallbackModel failed (2 sub-exceptions)',
                            'exception.stacktrace': '+------------------------------------',
                            'exception.escaped': 'False',
                        },
                    }
                ],
            },
            {
                'name': 'agent run',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 6000000000,
                'attributes': {
                    'model_name': 'fallback:function:failure_response:,function:failure_response:',
                    'agent_name': 'agent',
                    'logfire.msg': 'agent run',
                    'logfire.span_type': 'span',
                    'pydantic_ai.all_messages': [{'role': 'user', 'parts': [{'type': 'text', 'content': 'hello'}]}],
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'pydantic_ai.all_messages': {'type': 'array'},
                            'final_result': {'type': 'object'},
                        },
                    },
                    'logfire.level_num': 17,
                },
                'events': [
                    {
                        'name': 'exception',
                        'timestamp': 5000000000,
                        'attributes': {
                            'exception.type': 'pydantic_ai.exceptions.FallbackExceptionGroup',
                            'exception.message': 'All models from FallbackModel failed (2 sub-exceptions)',
                            'exception.stacktrace': '+------------------------------------',
                            'exception.escaped': 'False',
                        },
                    }
                ],
            },
        ]
    )


async def success_response_stream(_model_messages: list[ModelMessage], _agent_info: AgentInfo) -> AsyncIterator[str]:
    yield 'hello '
    yield 'world'


async def failure_response_stream(_model_messages: list[ModelMessage], _agent_info: AgentInfo) -> AsyncIterator[str]:
    # Note: today we can only handle errors that are raised before the streaming begins
    raise ModelHTTPError(status_code=500, model_name='test-function-model', body={'error': 'test error'})
    yield 'uh oh... '


success_model_stream = FunctionModel(stream_function=success_response_stream)
failure_model_stream = FunctionModel(stream_function=failure_response_stream)


async def test_first_success_streaming() -> None:
    fallback_model = FallbackModel(success_model_stream, failure_model_stream)
    agent = Agent(model=fallback_model)
    async with agent.run_stream('input') as result:
        assert [c async for c, _is_last in result.stream_responses(debounce_by=None)] == snapshot(
            [
                ModelResponse(
                    parts=[TextPart(content='hello ')],
                    usage=RequestUsage(input_tokens=50, output_tokens=1),
                    model_name='function::success_response_stream',
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ModelResponse(
                    parts=[TextPart(content='hello world')],
                    usage=RequestUsage(input_tokens=50, output_tokens=2),
                    model_name='function::success_response_stream',
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ModelResponse(
                    parts=[TextPart(content='hello world')],
                    usage=RequestUsage(input_tokens=50, output_tokens=2),
                    model_name='function::success_response_stream',
                    timestamp=IsNow(tz=timezone.utc),
                ),
            ]
        )
        assert result.is_complete


async def test_first_failed_streaming() -> None:
    fallback_model = FallbackModel(failure_model_stream, success_model_stream)
    agent = Agent(model=fallback_model)
    async with agent.run_stream('input') as result:
        assert [c async for c, _is_last in result.stream_responses(debounce_by=None)] == snapshot(
            [
                ModelResponse(
                    parts=[TextPart(content='hello ')],
                    usage=RequestUsage(input_tokens=50, output_tokens=1),
                    model_name='function::success_response_stream',
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ModelResponse(
                    parts=[TextPart(content='hello world')],
                    usage=RequestUsage(input_tokens=50, output_tokens=2),
                    model_name='function::success_response_stream',
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ModelResponse(
                    parts=[TextPart(content='hello world')],
                    usage=RequestUsage(input_tokens=50, output_tokens=2),
                    model_name='function::success_response_stream',
                    timestamp=IsNow(tz=timezone.utc),
                ),
            ]
        )
        assert result.is_complete


async def test_all_failed_streaming() -> None:
    fallback_model = FallbackModel(failure_model_stream, failure_model_stream)
    agent = Agent(model=fallback_model)
    with pytest.raises(ExceptionGroup) as exc_info:
        async with agent.run_stream('hello') as result:
            [c async for c, _is_last in result.stream_responses(debounce_by=None)]  # pragma: lax no cover
    assert 'All models from FallbackModel failed' in exc_info.value.args[0]
    exceptions = exc_info.value.exceptions
    assert len(exceptions) == 2
    assert isinstance(exceptions[0], ModelHTTPError)
    assert exceptions[0].status_code == 500
    assert exceptions[0].model_name == 'test-function-model'
    assert exceptions[0].body == {'error': 'test error'}


async def test_fallback_condition_override() -> None:
    def should_fallback(exc: Exception) -> bool:
        return False

    fallback_model = FallbackModel(failure_model, success_model, fallback_on=should_fallback)
    agent = Agent(model=fallback_model)
    with pytest.raises(ModelHTTPError):
        await agent.run('hello')


class PotatoException(Exception): ...


def potato_exception_response(_model_messages: list[ModelMessage], _agent_info: AgentInfo) -> ModelResponse:
    raise PotatoException()


async def test_fallback_condition_tuple() -> None:
    potato_model = FunctionModel(potato_exception_response)
    fallback_model = FallbackModel(potato_model, success_model, fallback_on=(PotatoException, ModelHTTPError))
    agent = Agent(model=fallback_model)

    response = await agent.run('hello')
    assert response.output == 'success'


async def test_fallback_model_settings_merge():
    """Test that FallbackModel properly merges model settings from wrapped model and runtime settings."""

    def return_settings(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(to_json(info.model_settings).decode())])

    base_model = FunctionModel(return_settings, settings=ModelSettings(temperature=0.1, max_tokens=1024))
    fallback_model = FallbackModel(base_model)

    # Test that base model settings are preserved when no additional settings are provided
    agent = Agent(fallback_model)
    result = await agent.run('Hello')
    assert result.output == IsJson({'max_tokens': 1024, 'temperature': 0.1})

    # Test that runtime model_settings are merged with base settings
    agent_with_settings = Agent(fallback_model, model_settings=ModelSettings(temperature=0.5, parallel_tool_calls=True))
    result = await agent_with_settings.run('Hello')
    expected = {'max_tokens': 1024, 'temperature': 0.5, 'parallel_tool_calls': True}
    assert result.output == IsJson(expected)

    # Test that run-time model_settings override both base and agent settings
    result = await agent_with_settings.run(
        'Hello', model_settings=ModelSettings(temperature=0.9, extra_headers={'runtime_setting': 'runtime_value'})
    )
    expected = {
        'max_tokens': 1024,
        'temperature': 0.9,
        'parallel_tool_calls': True,
        'extra_headers': {
            'runtime_setting': 'runtime_value',
        },
    }
    assert result.output == IsJson(expected)


async def test_fallback_model_settings_merge_streaming():
    """Test that FallbackModel properly merges model settings in streaming mode."""

    async def return_settings_stream(_: list[ModelMessage], info: AgentInfo):
        # Yield the merged settings as JSON to verify they were properly combined
        yield to_json(info.model_settings).decode()

    base_model = FunctionModel(
        stream_function=return_settings_stream,
        settings=ModelSettings(temperature=0.1, extra_headers={'anthropic-beta': 'context-1m-2025-08-07'}),
    )
    fallback_model = FallbackModel(base_model)

    # Test that base model settings are preserved in streaming mode
    agent = Agent(fallback_model)
    async with agent.run_stream('Hello') as result:
        output = await result.get_output()

    assert json.loads(output) == {'extra_headers': {'anthropic-beta': 'context-1m-2025-08-07'}, 'temperature': 0.1}

    # Test that runtime model_settings are merged with base settings in streaming mode
    agent_with_settings = Agent(fallback_model, model_settings=ModelSettings(temperature=0.5))
    async with agent_with_settings.run_stream('Hello') as result:
        output = await result.get_output()

    expected = {'extra_headers': {'anthropic-beta': 'context-1m-2025-08-07'}, 'temperature': 0.5}
    assert json.loads(output) == expected
