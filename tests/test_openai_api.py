"""Tests for OpenAI API compatibility implementation."""

from __future__ import annotations

import json
from http import HTTPStatus
from typing import Any

import httpx
import pytest
from asgi_lifespan import LifespanManager
from openai import AsyncOpenAI
from pydantic import BaseModel

from pydantic_ai import models
from pydantic_ai.agent import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.openai_api import OpenAIApp

models.ALLOW_MODEL_REQUESTS = False

pytestmark = [
    pytest.mark.anyio,
]


class WeatherResult(BaseModel):
    """Weather result model for testing."""

    location: str
    temperature: int


def get_weather(location: str) -> WeatherResult:
    """Get the weather for a location.

    Args:
        location: The location to get weather for.

    Returns:
        Weather information for the location.
    """
    return WeatherResult(location=location, temperature=20)


async def test_chat_completions_basic() -> None:
    """Test basic chat completions endpoint."""
    agent = Agent(TestModel())
    app = OpenAIApp(agent)

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as client:
            client.base_url = 'http://test'
            response = await client.post(
                '/v1/chat/completions',
                json={'model': 'test', 'messages': [{'role': 'user', 'content': 'Hello, how are you?'}]},
            )

            assert response.status_code == HTTPStatus.OK
            data = response.json()

            # Verify structure and key fields
            assert data['object'] == 'chat.completion'
            assert data['model'] == 'test'
            assert isinstance(data['id'], str)
            assert isinstance(data['created'], int)
            assert len(data['choices']) == 1
            assert data['choices'][0]['finish_reason'] == 'stop'
            assert data['choices'][0]['message']['role'] == 'assistant'
            assert isinstance(data['choices'][0]['message']['content'], str)
            assert 'usage' in data
            assert isinstance(data['usage']['completion_tokens'], int)
            assert isinstance(data['usage']['prompt_tokens'], int)
            assert isinstance(data['usage']['total_tokens'], int)


async def test_chat_completions_streaming() -> None:
    """Test streaming chat completions endpoint."""
    agent = Agent(TestModel())
    app = OpenAIApp(agent)

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as client:
            client.base_url = 'http://test'
            response = await client.post(
                '/v1/chat/completions',
                json={'model': 'test', 'messages': [{'role': 'user', 'content': 'Hello'}], 'stream': True},
            )

            assert response.status_code == HTTPStatus.OK
            assert response.headers['content-type'] == 'text/event-stream; charset=utf-8'

            chunks: list[dict[str, Any]] = []
            async for line in response.aiter_lines():
                if line.startswith('data: ') and line != 'data: [DONE]':
                    chunk_data = json.loads(line[6:])  # Remove "data: " prefix
                    chunks.append(chunk_data)

            assert len(chunks) > 0
            # First chunk should have role
            assert chunks[0]['choices'][0]['delta'].get('role') == 'assistant'
            # Last chunk should have finish_reason
            assert chunks[-1]['choices'][0]['finish_reason'] in ['stop', 'tool_calls']
            # All chunks should have correct structure
            for chunk in chunks:
                assert chunk['object'] == 'chat.completion.chunk'
                assert chunk['model'] == 'test'


async def test_chat_completions_with_tools() -> None:
    """Test chat completions with tools."""
    agent = Agent(TestModel(), tools=[get_weather])
    app = OpenAIApp(agent)

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as client:
            client.base_url = 'http://test'
            response = await client.post(
                '/v1/chat/completions',
                json={'model': 'test', 'messages': [{'role': 'user', 'content': "What's the weather in London?"}]},
            )

            assert response.status_code == HTTPStatus.OK
            data = response.json()

            assert data['object'] == 'chat.completion'
            assert data['model'] == 'test'
            # The response should include tool calls or stop
            assert data['choices'][0]['finish_reason'] in ['stop', 'tool_calls']
            assert 'usage' in data


async def test_chat_completions_multimodal() -> None:
    """Test chat completions with multimodal content."""
    agent = Agent(TestModel())
    app = OpenAIApp(agent)

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as client:
            client.base_url = 'http://test'
            response = await client.post(
                '/v1/chat/completions',
                json={
                    'model': 'test',
                    'messages': [
                        {
                            'role': 'user',
                            'content': [
                                {'type': 'text', 'text': "What's in this image?"},
                                {'type': 'image_url', 'image_url': {'url': 'https://example.com/image.jpg'}},
                            ],
                        }
                    ],
                },
            )

            assert response.status_code == HTTPStatus.OK
            data = response.json()
            assert data['object'] == 'chat.completion'
            assert data['model'] == 'test'
            assert data['choices'][0]['message']['role'] == 'assistant'
            assert data['choices'][0]['finish_reason'] == 'stop'


async def test_chat_completions_with_history() -> None:
    """Test chat completions with conversation history."""
    agent = Agent(TestModel())
    app = OpenAIApp(agent)

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as client:
            client.base_url = 'http://test'
            response = await client.post(
                '/v1/chat/completions',
                json={
                    'model': 'test',
                    'messages': [
                        {'role': 'system', 'content': 'You are a helpful assistant.'},
                        {'role': 'user', 'content': 'Hello'},
                        {'role': 'assistant', 'content': 'Hi there!'},
                        {'role': 'user', 'content': 'How are you?'},
                    ],
                },
            )

            assert response.status_code == HTTPStatus.OK
            data = response.json()
            assert data['object'] == 'chat.completion'
            assert data['model'] == 'test'
            assert data['choices'][0]['message']['role'] == 'assistant'
            assert data['choices'][0]['finish_reason'] == 'stop'


async def test_chat_completions_invalid_request() -> None:
    """Test chat completions with invalid request."""
    agent = Agent(TestModel())
    app = OpenAIApp(agent)

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as client:
            client.base_url = 'http://test'
            response = await client.post('/v1/chat/completions', json={'invalid': 'request'})

            assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
            # Response is JSON string containing error list
            error_data = response.text
            assert 'missing' in error_data or 'required' in error_data.lower()


async def test_responses_endpoint() -> None:
    """Test the responses endpoint."""
    agent = Agent(TestModel())
    app = OpenAIApp(agent)

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as client:
            client.base_url = 'http://test'
            response = await client.post('/v1/responses', json={'model': 'test', 'input': 'Hello'})

            assert response.status_code == HTTPStatus.OK
            data = response.json()

            # Verify structure and key fields
            assert data['object'] == 'response'
            assert data['model'] == 'test'
            assert isinstance(data['id'], str)
            assert isinstance(data['created_at'], float)
            assert data['status'] == 'completed'
            assert len(data['output']) > 0
            assert data['output'][0]['type'] == 'message'
            assert data['output'][0]['role'] == 'assistant'
            assert data['output'][0]['status'] == 'completed'
            assert len(data['output'][0]['content']) > 0
            assert data['output'][0]['content'][0]['type'] == 'output_text'
            assert isinstance(data['output'][0]['content'][0]['text'], str)
            assert 'usage' in data
            assert isinstance(data['usage']['input_tokens'], int)
            assert isinstance(data['usage']['output_tokens'], int)
            assert isinstance(data['usage']['total_tokens'], int)
            assert data['instructions'] is None
            assert data['parallel_tool_calls'] is True
            assert data['tool_choice'] == 'auto'
            assert data['tools'] == []


async def test_to_openai_api_method() -> None:
    """Test the to_openai_api method on Agent."""
    agent = Agent(TestModel())
    app = agent.to_openai_api()

    assert isinstance(app, OpenAIApp)

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as client:
            client.base_url = 'http://test'
            response = await client.post(
                '/v1/chat/completions',
                json={'model': 'test', 'messages': [{'role': 'user', 'content': 'Test message'}]},
            )

            assert response.status_code == HTTPStatus.OK
            data = response.json()
            assert data['object'] == 'chat.completion'
            assert data['model'] == 'test'


async def test_custom_parameters() -> None:
    """Test OpenAI app with custom parameters."""
    agent = Agent(TestModel())
    app = agent.to_openai_api(debug=True, model='custom-model')

    assert isinstance(app, OpenAIApp)
    assert app.debug is True


async def test_agent_parameters_override() -> None:
    """Test that request parameters can override agent defaults."""
    agent = Agent(TestModel())
    app = OpenAIApp(agent, model='test')

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as client:
            client.base_url = 'http://test'
            response = await client.post(
                '/v1/chat/completions',
                json={'model': 'override-model', 'messages': [{'role': 'user', 'content': 'Test'}]},
            )

            assert response.status_code == HTTPStatus.OK
            data = response.json()
            # The request model should override the app default
            assert data['model'] == 'override-model'
            assert data['object'] == 'chat.completion'


async def test_openai_client_chat_completions() -> None:
    """Test chat completions with OpenAI client."""
    agent = Agent(TestModel())
    app = OpenAIApp(agent)

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as client:
            client.base_url = 'http://test'
            openai_client = AsyncOpenAI(api_key='sk_test', http_client=client)

            chat_completion = await openai_client.chat.completions.create(
                model='test',
                messages=[{'role': 'user', 'content': 'Hello, how are you?'}],
                stream=False,
            )

            assert chat_completion.model == 'test'
            assert chat_completion.choices[0].message.role == 'assistant'
            assert chat_completion.choices[0].message.content is not None
            assert chat_completion.choices[0].finish_reason == 'stop'


async def test_openai_client_responses() -> None:
    """Test responses with OpenAI client."""
    agent = Agent(TestModel())
    app = OpenAIApp(agent)

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as client:
            client.base_url = 'http://test'
            openai_client = AsyncOpenAI(api_key='sk_test', http_client=client)

            response = await openai_client.responses.create(
                input=[{'role': 'user', 'content': 'Hello, how are you?'}],
                model='test',
                stream=False,
            )

            assert response.model == 'test'
            assert response.object == 'response'
            assert len(response.output) > 0

            for item in response.output:
                if item.type == 'message':
                    assert item.role == 'assistant'
                    assert item.content is not None


async def test_responses_with_instructions() -> None:
    """Test responses endpoint with instructions parameter."""
    agent = Agent(TestModel())
    app = OpenAIApp(agent)

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as client:
            client.base_url = 'http://test'
            response = await client.post(
                '/v1/responses',
                json={
                    'model': 'test',
                    'input': 'Hello',
                    'instructions': 'You are a helpful assistant.',
                },
            )

            assert response.status_code == HTTPStatus.OK
            data = response.json()
            assert data['object'] == 'response'
            assert data['instructions'] == 'You are a helpful assistant.'


async def test_responses_streaming() -> None:
    """Test that responses endpoint streaming works correctly."""
    agent = Agent(TestModel())
    app = OpenAIApp(agent)

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as client:
            client.base_url = 'http://test'
            response = await client.post(
                '/v1/responses',
                json={
                    'model': 'test',
                    'input': 'Hello',
                    'stream': True,
                },
            )

            assert response.status_code == HTTPStatus.OK
            assert response.headers['content-type'] == 'text/event-stream; charset=utf-8'

            events: list[dict[str, Any]] = []
            async for line in response.aiter_lines():
                if line.startswith('data: ') and line != 'data: [DONE]':
                    data_content = line[6:]  # Remove "data: " prefix
                    try:
                        event_data = json.loads(data_content)
                        events.append(event_data)
                    except json.JSONDecodeError:
                        pass  # Skip malformed lines

            assert len(events) > 0
            # First event should be response.created
            assert events[0]['type'] == 'response.created'
            assert events[0]['response']['status'] == 'in_progress'
            # Last event should be response.completed
            assert events[-1]['type'] == 'response.completed'
            assert events[-1]['response']['status'] == 'completed'


async def test_chat_completions_empty_message() -> None:
    """Test chat completions with empty messages list."""
    agent = Agent(TestModel())
    app = OpenAIApp(agent)

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as client:
            client.base_url = 'http://test'
            response = await client.post(
                '/v1/chat/completions',
                json={'model': 'test', 'messages': []},
            )

            # Empty messages should still work - the agent will handle it
            assert response.status_code in [HTTPStatus.OK, HTTPStatus.UNPROCESSABLE_ENTITY]


async def test_chat_completions_with_system_message() -> None:
    """Test chat completions with system message only."""
    agent = Agent(TestModel())
    app = OpenAIApp(agent)

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as client:
            client.base_url = 'http://test'
            response = await client.post(
                '/v1/chat/completions',
                json={
                    'model': 'test',
                    'messages': [
                        {'role': 'system', 'content': 'You are a helpful assistant.'},
                        {'role': 'user', 'content': 'Hello'},
                    ],
                },
            )

            assert response.status_code == HTTPStatus.OK
            data = response.json()
            assert data['object'] == 'chat.completion'


async def test_openai_client_streaming() -> None:
    """Test streaming with OpenAI client."""
    agent = Agent(TestModel())
    app = OpenAIApp(agent)

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as client:
            client.base_url = 'http://test'
            openai_client = AsyncOpenAI(api_key='sk_test', http_client=client)

            stream = await openai_client.chat.completions.create(
                model='test',
                messages=[{'role': 'user', 'content': 'Hello'}],
                stream=True,
            )

            chunks: list[Any] = []
            async for chunk in stream:
                chunks.append(chunk)

            assert len(chunks) > 0
            # Last chunk should have finish_reason
            assert chunks[-1].choices[0].finish_reason in ['stop', 'tool_calls']


async def test_openai_client_responses_streaming() -> None:
    """Test streaming Responses API with OpenAI client."""
    agent = Agent(TestModel())
    app = OpenAIApp(agent)

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as client:
            client.base_url = 'http://test'
            openai_client = AsyncOpenAI(api_key='sk_test', http_client=client)

            stream = await openai_client.responses.create(
                input=[{'role': 'user', 'content': 'Hello'}],
                model='test',
                stream=True,
            )

            events: list[Any] = []
            async for event in stream:
                events.append(event)

            assert len(events) > 0
            # First event should be response.created
            assert events[0].type == 'response.created'
            # Last event should be response.completed
            assert events[-1].type == 'response.completed'
