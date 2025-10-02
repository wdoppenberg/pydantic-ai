from __future__ import annotations

import json
from http import HTTPStatus

import httpx
import pytest
from asgi_lifespan import LifespanManager
from openai import AsyncOpenAI
from pydantic import BaseModel
from starlette.applications import Starlette

from pydantic_ai import models
from pydantic_ai.agent import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.openai_api import OpenAIApp

models.ALLOW_MODEL_REQUESTS = False


class WeatherResult(BaseModel):
    location: str
    temperature: int


def get_weather(location: str) -> WeatherResult:
    """Get the weather for a location."""
    return WeatherResult(location=location, temperature=20)


@pytest.fixture
def simple_agent():
    """Create a simple agent for testing."""
    return Agent(TestModel())


@pytest.fixture
def weather_agent():
    """Create an agent with weather tool for testing."""
    return Agent(TestModel(), tools=[get_weather])


@pytest.fixture
def openai_app(simple_agent):
    """Create OpenAI app for testing."""
    return OpenAIApp(simple_agent)


@pytest.fixture
def multi_agent_openai_app(simple_agent, weather_agent):
    routes = [*simple_agent.to_openai_api().routes, *weather_agent.to_openai_api().routes]
    app = Starlette(routes=routes)
    return app


@pytest.fixture
async def async_openai_client(openai_app):
    async with LifespanManager(openai_app):
        transport = httpx.ASGITransport(openai_app)
        async with httpx.AsyncClient(transport=transport) as client:
            client.base_url = 'http://test'

            yield AsyncOpenAI(api_key='sk_12341234', http_client=client)


async def test_chat_completions_basic(openai_app):
    """Test basic chat completions endpoint."""
    async with LifespanManager(openai_app):
        transport = httpx.ASGITransport(openai_app)
        async with httpx.AsyncClient(transport=transport) as client:
            client.base_url = 'http://test'
            response = await client.post(
                '/v1/chat/completions',
                json={'model': 'test', 'messages': [{'role': 'user', 'content': 'Hello, how are you?'}]},
            )

            assert response.status_code == HTTPStatus.OK
            data = response.json()

            assert data['object'] == 'chat.completion'
            assert data['model'] == 'test'
            assert len(data['choices']) == 1
            assert data['choices'][0]['message']['role'] == 'assistant'
            assert data['choices'][0]['message']['content'] is not None
            assert data['choices'][0]['finish_reason'] == 'stop'
            assert 'usage' in data


async def test_chat_completions_streaming(openai_app):
    """Test streaming chat completions endpoint."""
    async with LifespanManager(openai_app):
        transport = httpx.ASGITransport(openai_app)
        async with httpx.AsyncClient(transport=transport) as client:
            client.base_url = 'http://test'
            response = await client.post(
                '/v1/chat/completions',
                json={'model': 'test', 'messages': [{'role': 'user', 'content': 'Hello'}], 'stream': True},
            )

            assert response.status_code == HTTPStatus.OK
            assert response.headers['content-type'] == 'text/event-stream; charset=utf-8'

            chunks = []
            async for line in response.aiter_lines():
                if line.startswith('data: ') and line != 'data: [DONE]':
                    chunk_data = json.loads(line[6:])  # Remove "data: " prefix
                    chunks.append(chunk_data)

            assert len(chunks) > 0
            # Last chunk should have finish_reason
            assert chunks[-1]['choices'][0]['finish_reason'] in ['stop', 'tool_calls']


async def test_chat_completions_with_tools():
    """Test chat completions with tools."""
    weather_agent = Agent(TestModel(), tools=[get_weather])
    app = OpenAIApp(weather_agent)

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
            # The response should include tool calls or stop
            assert data['choices'][0]['finish_reason'] in ['stop', 'tool_calls']


async def test_chat_completions_multimodal(openai_app):
    """Test chat completions with multimodal content."""
    async with LifespanManager(openai_app):
        transport = httpx.ASGITransport(openai_app)
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


async def test_chat_completions_with_history(openai_app):
    """Test chat completions with conversation history."""
    async with LifespanManager(openai_app):
        transport = httpx.ASGITransport(openai_app)
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


async def test_chat_completions_invalid_request(openai_app):
    """Test chat completions with invalid request."""
    async with LifespanManager(openai_app):
        transport = httpx.ASGITransport(openai_app)
        async with httpx.AsyncClient(transport=transport) as client:
            client.base_url = 'http://test'
            response = await client.post('/v1/chat/completions', json={'invalid': 'request'})

            assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


async def test_responses_endpoint(openai_app):
    """Test the responses endpoint."""
    async with LifespanManager(openai_app):
        transport = httpx.ASGITransport(openai_app)
        async with httpx.AsyncClient(transport=transport) as client:
            client.base_url = 'http://test'
            response = await client.post('/v1/responses', json={'model': 'test', 'input': 'Hello'})

            # Should return a Response object, not ChatCompletion
            assert response.status_code == HTTPStatus.OK
            data = response.json()
            assert data['object'] == 'response'
            assert 'output' in data
            assert isinstance(data['output'], list)


async def test_to_openai_api_method():
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


async def test_custom_parameters():
    """Test OpenAI app with custom parameters."""
    agent = Agent(TestModel())
    app = agent.to_openai_api(debug=True, model='custom-model')

    assert isinstance(app, OpenAIApp)
    assert app.debug is True


async def test_agent_parameters_override():
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


async def test_openai_client_chat_completions(async_openai_client):
    """Test chat completions with OpenAI client."""
    chat_completion = await async_openai_client.chat.completions.create(
        model='test',
        messages=[{'role': 'user', 'content': 'Hello, how are you?'}],
        stream=False,
    )
    assert chat_completion.model == 'test'
    assert chat_completion.choices[0].message.role == 'assistant'
    assert chat_completion.choices[0].message.content is not None
    assert chat_completion.choices[0].finish_reason == 'stop'


async def test_openai_client_responses(async_openai_client):
    """Test responses with OpenAI client."""
    response = await async_openai_client.responses.create(
        input=[{'role': 'user', 'content': 'Hello, how are you?'}], model='test', stream=False
    )
    assert response.model == 'test'
    for item in response.output:
        if item.type == 'message':
            assert item.role == 'assistant'
            assert item.content is not None
        else:
            raise ValueError('Unexpected output type')
