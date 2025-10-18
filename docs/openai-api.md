# OpenAI API Compatibility

Pydantic AI agents can be exposed through OpenAI-compatible API endpoints, allowing them to be used as drop-in replacements for OpenAI's Chat Completions API and Responses API. This enables integration with any client or tool that supports the OpenAI API specification.

## Installation

The only dependencies are:

- [openai](https://github.com/openai/openai-python): to provide the OpenAI types and schemas.
- [starlette](https://www.starlette.io): to handle [ASGI](https://asgi.readthedocs.io/en/latest/) requests from a framework like FastAPI.

You can install Pydantic AI with the `openai` extra to ensure you have all the required dependencies:

```bash
pip/uv-add 'pydantic-ai-slim[openai,starlette]'
```

To run the examples you'll also need:

- [uvicorn](https://www.uvicorn.org/) or another ASGI compatible server

```bash
pip/uv-add uvicorn
```

## Usage

There are three ways to expose a Pydantic AI agent through OpenAI-compatible endpoints, from most to least flexible. If you're using a Starlette-based web framework like FastAPI, you'll typically want to use the second method.

1. [`OpenAIApp`][pydantic_ai.openai_api.OpenAIApp] creates an ASGI application that exposes your agent through both `/v1/chat/completions` and `/v1/responses` endpoints. It takes optional [`Agent.iter()`][pydantic_ai.Agent.iter] arguments including `deps`, but these will be the same for each request. This ASGI app can be used standalone or [mounted](https://fastapi.tiangolo.com/advanced/sub-applications/) at a given path in an existing FastAPI app.
2. [`handle_chat_completions_request()`][pydantic_ai.openai_api.handle_chat_completions_request] and [`handle_responses_request()`][pydantic_ai.openai_api.handle_responses_request] take an agent and a Starlette request (e.g. from FastAPI), and return a streaming or non-streaming Starlette response that you can return directly from your endpoint. They also take optional [`Agent.iter()`][pydantic_ai.Agent.iter] arguments including `deps`, that you can vary for each request (e.g. based on the authenticated user).
3. Direct use of the conversion functions if you need full control over request parsing and response generation. This can be modified to work with any web framework.

### Stand-alone ASGI app

This example uses [`OpenAIApp`][pydantic_ai.openai_api.OpenAIApp] to turn the agent into a stand-alone ASGI application that exposes both OpenAI-compatible endpoints:

```py {title="openai_app.py" hl_lines="4"}
from pydantic_ai import Agent
from pydantic_ai.openai_api import OpenAIApp

agent = Agent('openai:gpt-4o', instructions='Be helpful!')
app = OpenAIApp(agent)
```

Since `app` is an ASGI application, it can be used with any ASGI server:

```shell
uvicorn openai_app:app
```

This will expose the agent with two OpenAI-compatible endpoints:
- `POST /v1/chat/completions` - Chat Completions API endpoint
- `POST /v1/responses` - Responses API endpoint

Your client applications can now connect to this server using any OpenAI-compatible client library:

```py {title="client.py"}
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # API key not required for local agent
)

response = client.chat.completions.create(
    model="not-used",  # Model specified in agent definition
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

### Handle Starlette requests

This example uses [`handle_chat_completions_request()`][pydantic_ai.openai_api.handle_chat_completions_request] and [`handle_responses_request()`][pydantic_ai.openai_api.handle_responses_request] to directly handle FastAPI requests and return responses. Something analogous to this will work with any Starlette-based web framework.

```py {title="handle_openai_request.py"}
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import Response

from pydantic_ai import Agent
from pydantic_ai.openai_api import (
    handle_chat_completions_request,
    handle_responses_request,
)

agent = Agent('openai:gpt-4o', instructions='Be helpful!')

app = FastAPI()

@app.post('/v1/chat/completions')
async def chat_completions(request: Request) -> Response:
    return await handle_chat_completions_request(agent, request)

@app.post('/v1/responses')
async def responses(request: Request) -> Response:
    return await handle_responses_request(agent, request)
```

Since `app` is an ASGI application, it can be used with any ASGI server:

```shell
uvicorn handle_openai_request:app
```

This will expose the agent as an OpenAI-compatible server, and your client applications can start sending requests to it.

### Mount in existing FastAPI app

You can also mount an `OpenAIApp` at a specific path in an existing FastAPI application:

```py {title="mounted_openai.py"}
from fastapi import FastAPI
from pydantic_ai import Agent
from pydantic_ai.openai_api import OpenAIApp

app = FastAPI()

agent = Agent('openai:gpt-4o', instructions='Be helpful!')
openai_app = OpenAIApp(agent)

# Mount the OpenAI-compatible endpoints at /api/ai
app.mount('/api/ai', openai_app)

# Your other FastAPI routes
@app.get('/')
async def root():
    return {"message": "Hello World"}
```

Now the OpenAI endpoints are available at:
- `POST /api/ai/v1/chat/completions`
- `POST /api/ai/v1/responses`

## Design

The Pydantic AI OpenAI API integration provides compatibility with two OpenAI API endpoints:

### Chat Completions API

The [Chat Completions API](https://platform.openai.com/docs/api-reference/chat/create) (`/v1/chat/completions`) is OpenAI's standard conversational interface. It uses:

- **Request format**: `messages` array with roles (system, user, assistant, tool)
- **Response format**: `choices` array containing the assistant's message
- **Streaming**: Server-Sent Events with `ChatCompletionChunk` objects

The integration converts OpenAI message formats to Pydantic AI's internal types and streams responses back as OpenAI-compatible chunks.

### Responses API

The [Responses API](https://platform.openai.com/docs/api-reference/responses/create) (`/v1/responses`) is a newer OpenAI endpoint format. It uses:

- **Request format**: `input` (string or array of items) and optional `instructions`
- **Response format**: `output` array containing messages and other items
- **Streaming**: Server-Sent Events with `ResponseStreamEvent` objects

The integration converts the Responses API input format to Pydantic AI messages and formats results according to the Responses API specification.

## Features

### Message History

Both endpoints support full conversation history:

```py {title="conversation.py"}
from pydantic_ai import Agent
from pydantic_ai.openai_api import OpenAIApp

agent = Agent('openai:gpt-4o', instructions='You are a helpful assistant.')
app = OpenAIApp(agent)
```

Client code with conversation history:

```py
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# Multi-turn conversation
messages = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 equals 4."},
    {"role": "user", "content": "What about 3+3?"}
]

response = client.chat.completions.create(
    model="not-used",
    messages=messages
)

print(response.choices[0].message.content)
```

### Streaming

The Chat Completions endpoint supports streaming responses:

```py
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

stream = client.chat.completions.create(
    model="not-used",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

Both Chat Completions and Responses API endpoints support streaming responses.

### Tool Calls

Both endpoints support tool calls, enabling function calling capabilities:

```py {title="tools_example.py"}
from pydantic_ai import Agent, RunContext
from pydantic_ai.openai_api import OpenAIApp

agent = Agent('openai:gpt-4o', instructions='You are a helpful calculator.')

@agent.tool
def add_numbers(ctx: RunContext, a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

app = OpenAIApp(agent)
```

Client code:

```py
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="not-used",
    messages=[{"role": "user", "content": "What is 15 + 27?"}]
)

print(response.choices[0].message.content)
```

The agent will automatically call the tool and return the result in OpenAI's function calling format.

### Multimodal Inputs

The Chat Completions endpoint supports multimodal inputs including images and audio:

```py
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="not-used",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.jpg"}
                }
            ]
        }
    ]
)

print(response.choices[0].message.content)
```

### Per-Request Dependencies

When using `handle_chat_completions_request()` or `handle_responses_request()`, you can provide different dependencies for each request:

```py {title="per_request_deps.py"}
from dataclasses import dataclass
from fastapi import FastAPI, Depends
from starlette.requests import Request
from starlette.responses import Response

from pydantic_ai import Agent, RunContext
from pydantic_ai.openai_api import handle_chat_completions_request

@dataclass
class UserDeps:
    user_id: str
    username: str

agent = Agent('openai:gpt-4o', deps_type=UserDeps)

@agent.system_prompt
def system_prompt(ctx: RunContext[UserDeps]) -> str:
    return f"You are assisting {ctx.deps.username}."

app = FastAPI()

def get_current_user() -> UserDeps:
    # Your authentication logic here
    return UserDeps(user_id="123", username="Alice")

@app.post('/v1/chat/completions')
async def chat_completions(
    request: Request, 
    user: UserDeps = Depends(get_current_user)
) -> Response:
    return await handle_chat_completions_request(agent, request, deps=user)
```

### Usage Tracking

Both endpoints return token usage information in their responses:

```py
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="not-used",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(f"Prompt tokens: {response.usage.prompt_tokens}")
print(f"Completion tokens: {response.usage.completion_tokens}")
print(f"Total tokens: {response.usage.total_tokens}")
```

## Known Limitations

1. **Model Parameter**: The `model` parameter in requests is ignored. The model specified when creating the agent is used instead. However, the model name from the request is echoed back in responses for client compatibility.

2. **Some Response Fields**: Certain OpenAI-specific response fields like `logprobs` are always returned as `None` since they don't have equivalents in Pydantic AI.

## Examples

For more examples of how to use the OpenAI API compatibility features, see the test suite in `tests/test_openai_api.py`.
