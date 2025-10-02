"""OpenAI API compatibility module for Pydantic AI agents.

This module provides OpenAI API-compatible endpoints for Pydantic AI agents, allowing them to be used
as drop-in replacements for OpenAI's Chat Completions API and Responses API. The module includes
functionality for handling both streaming and non-streaming requests, tool calls, multimodal inputs,
and conversation history.

Example:
    ```python
    from pydantic_ai import Agent
    from pydantic_ai.openai_api import OpenAIApp

    agent = Agent('openai:gpt-4o')
    app = OpenAIApp(agent)
    ```

    The `app` is an ASGI application that can be used with any ASGI server:

    ```bash
    uvicorn app:app --host 0.0.0.0 --port 8000
    ```
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from dataclasses import dataclass
from http import HTTPStatus
from typing import (
    Any,
    Callable,
    Generic,
    Literal, TypeAlias,
)

from pydantic import TypeAdapter, ValidationError

from pydantic_graph import End

from ._agent_graph import AgentNode
from .agent import AbstractAgent
from .messages import (
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponseStreamEvent,
    PartDeltaEvent,
    PartStartEvent,
    SystemPromptPart,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
    UserPromptPart,
    ModelRequestPart,
    UserContent,
    BinaryContent,
)
from .models import KnownModelName, Model
from .output import OutputDataT, OutputSpec
from .run import AgentRunResult
from .settings import ModelSettings
from .tools import AgentDepsT
from .toolsets import AbstractToolset
from .usage import RunUsage, UsageLimits

try:
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.requests import Request
    from starlette.responses import Response, StreamingResponse
    from starlette.routing import BaseRoute
    from starlette.types import ExceptionHandler, Lifespan
except ImportError as e:  # pragma: no cover
    raise ImportError(
        'Please install the `starlette` package to use `Agent.to_openai_api()` method, '
        'you can use the `openai` & `starlette` optional group — `pip install "pydantic-ai-slim[openai,starlette]"`'
    ) from e

try:
    from openai.types import AllModels, CompletionUsage, chat, responses
    from openai.types.chat import (
        ChatCompletion,
        ChatCompletionChunk,
        ChatCompletionContentPartImageParam,
        ChatCompletionContentPartInputAudioParam,
        ChatCompletionContentPartParam,
        ChatCompletionContentPartTextParam,
        ChatCompletionMessage,
        ChatCompletionMessageParam,
        CompletionCreateParams as CompletionCreateParamsT,
    )
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice, ChoiceDelta
    from openai.types.chat.chat_completion_content_part_image_param import ImageURL
    from openai.types.chat.chat_completion_content_part_input_audio_param import InputAudio
    from openai.types.chat.chat_completion_content_part_param import File, FileFile
    from openai.types.chat.chat_completion_message_custom_tool_call import ChatCompletionMessageCustomToolCall
    from openai.types.chat.chat_completion_message_function_tool_call import (
        ChatCompletionMessageFunctionToolCall,
        Function,
    )
    from openai.types.chat.chat_completion_message_function_tool_call_param import (
        ChatCompletionMessageFunctionToolCallParam,
    )
    from openai.types.chat.chat_completion_prediction_content_param import ChatCompletionPredictionContentParam
    from openai.types.chat.completion_create_params import (
        CompletionCreateParamsNonStreaming as CompletionCreateParamsNonStreamingT,
        CompletionCreateParamsStreaming as CompletionCreateParamsStreamingT,
        WebSearchOptions,
        WebSearchOptionsUserLocation,
        WebSearchOptionsUserLocationApproximate,
    )
    from openai.types.responses import (
        ComputerToolParam,
        FileSearchToolParam,
        Response as ResponseObject,
        ResponseCreateParams as ResponseCreateParamsT,
        ResponseOutputMessage,
        ResponseOutputText,
        ResponseStreamEvent,
        ResponseUsage,
        WebSearchToolParam,
    )
    from openai.types.responses.response_create_params import (
        ResponseCreateParamsNonStreaming as ResponseCreateParamsNonStreamingT,
        ResponseCreateParamsStreaming as ResponseCreateParamsStreamingT,
    )
    from openai.types.responses.response_input_param import FunctionCallOutput, Message
    from openai.types.shared import ReasoningEffort
    from openai.types.shared_params import Reasoning
except ImportError as _import_error:
    raise ImportError(
        'Please install `openai` to use the OpenAI model, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error


class OpenAIApp(Generic[AgentDepsT, OutputDataT], Starlette):
    """ASGI application for running Pydantic AI agents with OpenAI API-compatible endpoints.

    This class provides a Starlette-based ASGI application that exposes Pydantic AI agents through
    OpenAI-compatible API endpoints, specifically `/v1/chat/completions` and `/v1/responses`.
    The application handles both streaming and non-streaming requests, tool calls, multimodal inputs,
    and conversation history, making it a drop-in replacement for OpenAI's API.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai.openai_api import OpenAIApp

        agent = Agent('openai:gpt-4o')
        app = OpenAIApp(agent)
        ```

        To run the application:

        ```bash
        uvicorn app:app --host 0.0.0.0 --port 8000
        ```

    The application automatically sets up the following routes:
    - `POST /v1/chat/completions` - OpenAI Chat Completions API compatible endpoint
    - `POST /v1/responses` - OpenAI Responses API compatible endpoint
    """

    def __init__(
        self,
        agent: AbstractAgent[AgentDepsT, OutputDataT],
        *,
        # Agent.iter parameters.
        output_type: OutputSpec[Any] | None = None,
        model: Model | KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: UsageLimits | None = None,
        usage: RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        # Starlette parameters.
        debug: bool = False,
        routes: Sequence[BaseRoute] | None = None,
        middleware: Sequence[Middleware] | None = None,
        exception_handlers: Mapping[Any, ExceptionHandler] | None = None,
        on_startup: Sequence[Callable[[], Any]] | None = None,
        on_shutdown: Sequence[Callable[[], Any]] | None = None,
        lifespan: Lifespan[OpenAIApp[AgentDepsT, OutputDataT]] | None = None,
    ):
        """Initialize the OpenAI API-compatible ASGI application.

        Args:
            agent: The Pydantic AI agent to expose via the OpenAI API endpoints.

            output_type: Custom output type to use for this run, `output_type` may only be used if the agent has
                no output validators since output validators would expect an argument that matches the agent's
                output type.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.

            debug: Boolean indicating if debug tracebacks should be returned on errors.
            routes: A list of routes to serve incoming HTTP and WebSocket requests.
            middleware: A list of middleware to run for every request. A starlette application will always
                automatically include two middleware classes. `ServerErrorMiddleware` is added as the very
                outermost middleware, to handle any uncaught errors occurring anywhere in the entire stack.
                `ExceptionMiddleware` is added as the very innermost middleware, to deal with handled
                exception cases occurring in the routing or endpoints.
            exception_handlers: A mapping of either integer status codes, or exception class types onto
                callables which handle the exceptions. Exception handler callables should be of the form
                `handler(request, exc) -> response` and may be either standard functions, or async functions.
            on_startup: A list of callables to run on application startup. Startup handler callables do not
                take any arguments, and may be either standard functions, or async functions.
            on_shutdown: A list of callables to run on application shutdown. Shutdown handler callables do
                not take any arguments, and may be either standard functions, or async functions.
            lifespan: A lifespan context function, which can be used to perform startup and shutdown tasks.
                This is a newer style that replaces the `on_startup` and `on_shutdown` handlers. Use one or
                the other, not both.
        """
        super().__init__(
            debug=debug,
            routes=routes,
            middleware=middleware,
            exception_handlers=exception_handlers,
            on_startup=on_startup,
            on_shutdown=on_shutdown,
            lifespan=lifespan,
        )

        async def chat_completions_endpoint(request: Request) -> Response:
            return await handle_chat_completions_request(
                agent,
                request,
                output_type=output_type,
                model=model,
                deps=deps,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                infer_name=infer_name,
                toolsets=toolsets,
            )

        async def responses_endpoint(request: Request) -> Response:
            return await handle_responses_request(
                agent,
                request,
                output_type=output_type,
                model=model,
                deps=deps,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                infer_name=infer_name,
                toolsets=toolsets,
            )

        self.router.add_route(
            '/v1/chat/completions', chat_completions_endpoint, methods=['POST'], name='chat_completions'
        )
        self.router.add_route('/v1/responses', responses_endpoint, methods=['POST'], name='chat_completions')


CompletionCreateParams = TypeAdapter(CompletionCreateParamsT)
CompletionCreateParamsNonStreaming = TypeAdapter(CompletionCreateParamsNonStreamingT)
CompletionCreateParamsStreaming = TypeAdapter(CompletionCreateParamsStreamingT)

ResponseCreateParams = TypeAdapter(ResponseCreateParamsT)
ResponseCreateParamsNonStreaming = TypeAdapter(ResponseCreateParamsNonStreamingT)
ResponseCreateParamsStreaming = TypeAdapter(ResponseCreateParamsStreamingT)

OpenAIRole: TypeAlias = Literal["developer", "system", "assistant", "user", "tool", "function"]

@dataclass
class _StreamContext:
    """Internal context for tracking streaming chat completion state.

    This dataclass maintains state information during streaming responses to ensure
    proper OpenAI-compatible chunk generation, particularly for managing role
    information and tool call streaming.

    Attributes:
        role_sent: Whether the assistant role has been sent in the stream yet.
        tool_call_part_started: The current tool call part being streamed, if any.
        tool_call_index: Index counter for tool calls in the current response.
        got_tool_calls: Whether any tool calls have been encountered in this response.
    """

    role_sent: bool = False
    tool_call_part_started: ToolCallPart | None = None
    tool_call_index: int = 0
    got_tool_calls: bool = False


def _from_openai_messages(messages: Iterable[ChatCompletionMessageParam]) -> list[ModelMessage]:
    """Converts OpenAI chat completion messages to Pydantic AI's internal message format.

    This function transforms OpenAI API message formats into Pydantic AI's structured
    message types, handling different roles (system, user, assistant, tool), content types
    (text, images), and tool calls. It maintains conversation flow by grouping related
    messages into ModelRequest and ModelResponse objects.

    Args:
        messages: An iterable of OpenAI chat completion message parameters.

    Returns:
        A list of ModelMessage objects (ModelRequest and ModelResponse) that represent
        the conversation history in Pydantic AI's internal format.

    Example:
        ```python
        openai_messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        pydantic_messages = _from_openai_messages(openai_messages)
        ```
    """
    history: list[ModelMessage] = []
    request_parts: list[ModelRequestPart] = []

    for message in messages:
        role: OpenAIRole | None = message.get('role')

        if role is None:
            raise RuntimeError("No role found in message.")

        if role == 'assistant':
            if request_parts:
                history.append(ModelRequest(parts=request_parts))
                request_parts = []
            response_parts: list[TextPart | ToolCallPart] = []
            if content := message.get('content'):
                # TODO: Handle 'Iterable[ContentArrayOfContentPart]'
                #  This is now simply stringified
                response_parts.append(TextPart(content=str(content)))

            if tool_calls := message.get('tool_calls'):
                for tc in tool_calls:
                    response_parts.append(
                        # All fields are Required[] so indexing is safe
                        ToolCallPart(
                            tool_call_id=tc['id'],
                            tool_name=tc['function']['name'],
                            # Store as json string
                            args=tc['function']['arguments'],
                        )
                    )
            history.append(ModelResponse(parts=response_parts))
        else:
            if role == 'system':
                if content := message.get('content'):
                    request_parts.append(SystemPromptPart(content=str(content)))

            elif role == 'user':
                content = message.get('content')
                if isinstance(content, str):
                    request_parts.append(UserPromptPart(content=content))
                elif content:  # should be a list
                    parts: list[UserContent] = []
                    for part in content:
                        part_type = part.get('type')

                        if part_type == 'text':
                            parts.append(part.get('text', ''))

                        elif part_type == 'image_url':
                            image_url = part.get('image_url')
                            url = image_url.get('url', '')
                            parts.append(ImageUrl(url=url))

                        elif part_type == 'input_audio':
                            input_audio = part['input_audio']
                            data = input_audio['data']
                            format = input_audio['format']

                            # Only wav and mp3 are supported currently
                            media_type = 'audio/wav' if format == 'wav' else 'audio/mpeg'

                            parts.append(BinaryContent(data=data, media_type=media_type))

                        # TODO: Handle 'File' type

                    if parts:
                        request_parts.append(UserPromptPart(content=parts))
            elif role == 'tool':
                request_parts.append(
                    ToolReturnPart(tool_call_id=message.get('tool_call_id', ''), content=str(message.get('content')))
                )

            elif role == 'function':
                # 'function' role is deprecated in OpenAI API but still supported for backward compatibility
                # It should be treated similarly to 'tool' role
                request_parts.append(
                    ToolReturnPart(
                        tool_call_id=message.get('name', ''),  # function role uses 'name' field
                        content=str(message.get('content'))
                    )
                )

    if request_parts:
        history.append(ModelRequest(parts=request_parts))

    return history


def _from_responses_input(input_data: Any, instructions: str | None = None) -> list[ModelMessage]:
    """Converts OpenAI Responses API input to Pydantic AI's internal message format.

    The Responses API uses a different input format than Chat Completions:
    - `input` can be a string (simple user message) or a list of various item types
    - `instructions` (if provided) acts as a system prompt

    Args:
        input_data: The input field from ResponseCreateParams - can be str or list of items.
        instructions: Optional instructions to prepend as a system message.

    Returns:
        A list of ModelMessage objects representing the conversation in Pydantic AI format.

    Example:
        ```python
        messages = _from_responses_input("Hello!", "You are helpful")
        # Results in system prompt + user message
        ```
    """
    history: list[ModelMessage] = []
    request_parts: list[ModelRequestPart] = []

    # Handle instructions as system prompt
    if instructions:
        request_parts.append(SystemPromptPart(content=instructions))

    # Handle input - can be string or list
    if isinstance(input_data, str):
        # Simple string input becomes a user message
        request_parts.append(UserPromptPart(content=input_data))
    elif isinstance(input_data, list):
        # List of items - need to parse each item by type
        # For now, we'll do a simplified conversion that handles the most common cases
        for item in input_data:
            if isinstance(item, dict):
                item_type = item.get('type')
                
                # Handle message items (user/system/assistant)
                if item_type == 'message':
                    role = item.get('role', 'user')
                    content = item.get('content', '')
                    
                    if role == 'system':
                        request_parts.append(SystemPromptPart(content=str(content)))
                    elif role == 'user':
                        request_parts.append(UserPromptPart(content=str(content)))
                    elif role == 'assistant':
                        # Finish current request if any, start response
                        if request_parts:
                            history.append(ModelRequest(parts=request_parts))
                            request_parts = []
                        history.append(ModelResponse(parts=[TextPart(content=str(content))]))
                        
                # Handle function call outputs (tool returns)
                elif item_type == 'function_call_output':
                    call_id = item.get('call_id', '')
                    output = item.get('output', '')
                    request_parts.append(ToolReturnPart(tool_call_id=call_id, content=str(output)))

    # Add any remaining request parts
    if request_parts:
        history.append(ModelRequest(parts=request_parts))

    return history


def _to_openai_chat_completion(run: AgentRunResult[OutputDataT], model: str) -> ChatCompletion:
    """Converts a Pydantic AI agent run result to an OpenAI ChatCompletion object.

    This function transforms the result of a Pydantic AI agent run into an OpenAI-compatible
    ChatCompletion response format, including message content, tool calls, usage statistics,
    and appropriate finish reasons.

    Args:
        run: The completed agent run result containing the conversation and response data.
        model: The model name to include in the response metadata.

    Returns:
        An OpenAI ChatCompletion object with the agent's response formatted according to
        the OpenAI API specification.

    Example:
        ```python
        completion = _to_openai_chat_completion(agent_run, "gpt-4o")
        print(completion.choices[0].message.content)
        ```
    """
    last_response = next((m for m in reversed(run.all_messages()) if isinstance(m, ModelResponse)), None)

    content_parts: list[str] = []
    tool_calls = []
    if last_response:
        for part in last_response.parts:
            if isinstance(part, TextPart):
                content_parts.append(part.content)
            elif isinstance(part, ToolCallPart):
                tool_calls.append(
                    ChatCompletionMessageFunctionToolCall(
                        id=part.tool_call_id,
                        function=Function(name=part.tool_name, arguments=json.dumps(part.args)),
                        type='function',
                    )
                )

    content = ''.join(content_parts) if content_parts else None
    finish_reason: Literal['tool_calls', 'stop'] = 'tool_calls' if tool_calls else 'stop'

    run_usage = run.usage()
    completion_usage = CompletionUsage(
        completion_tokens=run_usage.output_tokens,
        prompt_tokens=run_usage.input_tokens,
        total_tokens=run_usage.total_tokens,
    )

    return ChatCompletion(
        id=str(uuid.uuid4()),
        choices=[
            Choice(
                finish_reason=finish_reason,
                index=0,
                logprobs=None,
                message=ChatCompletionMessage(
                    content=content,
                    role='assistant',
                    tool_calls=tool_calls if tool_calls else None,
                ),
            )
        ],
        created=int(time.time()),
        model=model,
        object='chat.completion',
        usage=completion_usage,
    )


def _to_openai_response(run: AgentRunResult[OutputDataT], model: str, input_data: Any, instructions: str | None = None) -> ResponseObject:
    """Converts a Pydantic AI agent run result to an OpenAI Response object.

    This function transforms the result of a Pydantic AI agent run into a Responses API-compatible
    Response format. Unlike ChatCompletion which uses `choices`, Response uses an `output` list
    containing messages and other items.

    Args:
        run: The completed agent run result containing the conversation and response data.
        model: The model name to include in the response metadata.
        input_data: The original input data from the request.
        instructions: Optional instructions from the request.

    Returns:
        An OpenAI Response object with the agent's response formatted according to
        the Responses API specification.

    Example:
        ```python
        response = _to_openai_response(agent_run, "gpt-4o", "Hello", None)
        print(response.output[0])
        ```
    """
    from openai.types.responses import ResponseFunctionToolCall
    
    last_response = next((m for m in reversed(run.all_messages()) if isinstance(m, ModelResponse)), None)

    output: list[Any] = []
    
    if last_response:
        # Build output message with text and tool calls
        message_content: list[ResponseOutputText] = []
        
        for part in last_response.parts:
            if isinstance(part, TextPart):
                message_content.append(
                    ResponseOutputText(
                        type='output_text',
                        text=part.content,
                        annotations=[],
                    )
                )
            elif isinstance(part, ToolCallPart):
                # Tool calls are separate items in the output list
                output.append(
                    ResponseFunctionToolCall(
                        type='function_tool_call',
                        call_id=part.tool_call_id,
                        name=part.tool_name,
                        arguments=part.args_as_json_str(),
                    )
                )
        
        # Add message to output if there's any text content
        if message_content:
            output.insert(0, ResponseOutputMessage(
                type='message',
                id=str(uuid.uuid4()),
                role='assistant',
                status='completed',
                content=message_content,
            ))

    from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails
    
    run_usage = run.usage()
    response_usage = ResponseUsage(
        input_tokens=run_usage.input_tokens,
        input_tokens_details=InputTokensDetails(cached_tokens=0),
        output_tokens=run_usage.output_tokens,
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        total_tokens=run_usage.total_tokens,
    )

    return ResponseObject(
        id=str(uuid.uuid4()),
        object='response',
        created_at=time.time(),
        model=model,
        output=output,
        status='completed',
        usage=response_usage,
        instructions=instructions,
        parallel_tool_calls=True,
        tool_choice='auto',
        tools=[],
    )


def _to_openai_chat_completion_chunk(
    event: ModelResponseStreamEvent, model: str, run_id: str, context: _StreamContext
) -> ChatCompletionChunk | None:
    r"""Converts a Pydantic AI agent stream event to an OpenAI ChatCompletionChunk object.

    Args:
        event: Agent stream event (PartStartEvent or PartDeltaEvent)
        model: The model name to include in the chunk metadata.
        run_id: Unique identifier for the streaming run.
        context: Stream context for maintaining state across chunks.

    Returns:
        An OpenAI ChatCompletionChunk object if the event contains streamable content,
        or None if the event should be skipped in the stream.

    Example:
        ```python
        context = _StreamContext()
        chunk = _to_openai_chat_completion_chunk(event, "gpt-4o", run_id, context)
        if chunk:
            yield f'data: {chunk.model_dump_json()}\n\n'
        ```
    """
    delta = ChoiceDelta()

    if isinstance(event, PartStartEvent):
        if isinstance(event.part, ToolCallPart):
            context.tool_call_part_started = event.part
            context.got_tool_calls = True

    elif isinstance(event, PartDeltaEvent):
        if not context.role_sent:
            delta.role = 'assistant'
            context.role_sent = True

        if isinstance(event.delta, TextPartDelta):
            delta.content = event.delta.content_delta
        elif isinstance(event.delta, ToolCallPartDelta):
            if context.tool_call_part_started:
                # First delta for a new tool call
                delta.tool_calls = [
                    ChoiceDelta.ToolCall(
                        index=context.tool_call_index,
                        id=context.tool_call_part_started.tool_call_id,
                        type='function',
                        function=ChoiceDelta.ToolCall.Function(
                            name=context.tool_call_part_started.tool_name,
                            arguments=event.delta.args_delta if isinstance(event.delta.args_delta, str) else json.dumps(event.delta.args_delta),
                        ),
                    )
                ]
                context.tool_call_part_started = None  # Consume it
            else:
                # Subsequent delta for the same tool call
                delta.tool_calls = [
                    ChoiceDelta.ToolCall(
                        index=context.tool_call_index,
                        function=ChoiceDelta.ToolCall.Function(
                            arguments=event.delta.args_delta if isinstance(event.delta.args_delta, str) else json.dumps(event.delta.args_delta)
                        ),
                    )
                ]

    if not delta.role and not delta.content and not delta.tool_calls:
        return None

    return ChatCompletionChunk(
        id=run_id,
        choices=[ChunkChoice(delta=delta, index=0, finish_reason=None, logprobs=None)],
        created=int(time.time()),
        model=model,
        object='chat.completion.chunk',
    )


async def handle_chat_completions_request(
    agent: AbstractAgent[AgentDepsT, Any],
    request: Request,
    *,
    output_type: OutputSpec[Any] | None = None,
    model: Model | KnownModelName | str | None = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    infer_name: bool = True,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
) -> Response:
    """Handles OpenAI Chat Completions API requests.

    This function processes HTTP requests to the `/v1/chat/completions` endpoint,
    providing OpenAI API compatibility for Pydantic AI agents. It supports both
    streaming and non-streaming responses, tool calls, and multimodal inputs.

    Args:
        agent: The Pydantic AI agent to run for generating responses.
        request: The HTTP request containing the chat completion parameters.

        output_type: Custom output type to use for this run, `output_type` may only be used if the agent has
            no output validators since output validators would expect an argument that matches the agent's
            output type.
        model: Optional model to use for this run, required if `model` was not set when creating the agent.
        deps: Optional dependencies to use for this run.
        model_settings: Optional settings to use for this model's request.
        usage_limits: Optional limits on model request count or token usage.
        usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
        infer_name: Whether to try to infer the agent name from the call frame if it's not set.
        toolsets: Optional additional toolsets for this run.

    Returns:
        A Starlette Response object containing either:
        - A StreamingResponse with server-sent events for streaming requests
        - A JSON Response with the complete chat completion for non-streaming requests
        - An error response for invalid requests

    Example:
        ```python
        response = await handle_chat_completions_request(agent, request)
        ```
    """
    try:
        params: CompletionCreateParamsT = CompletionCreateParams.validate_python(await request.json())
    except ValidationError as e:  # pragma: no cover
        return Response(
            content=json.dumps(e.json()),
            media_type='application/json',
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
        )

    messages = _from_openai_messages(params.get('messages', []))
    agent_kwargs = dict(
        output_type=output_type,
        model=model or params.get('model'),
        deps=deps,
        model_settings=model_settings,
        usage_limits=usage_limits,
        usage=usage,
        infer_name=infer_name,
        toolsets=toolsets,
    )

    if params.get('stream'):
        run_id = str(uuid.uuid4())
        context = _StreamContext()

        async def stream_generator() -> AsyncIterator[str]:
            from ._agent_graph import ModelRequestNode
            from pydantic_graph import End
            
            async with agent.iter(message_history=messages, **agent_kwargs) as run:
                async for node in run:
                    if isinstance(node, End):
                        finish_reason: Literal['tool_calls', 'stop'] = (
                            'tool_calls' if context.got_tool_calls else 'stop'
                        )
                        final_chunk = ChatCompletionChunk(
                            id=run_id,
                            choices=[
                                ChunkChoice(delta=ChoiceDelta(), index=0, finish_reason=finish_reason, logprobs=None)
                            ],
                            created=int(time.time()),
                            model=params.get('model'),
                            object='chat.completion.chunk',
                        )
                        yield f'data: {final_chunk.model_dump_json(exclude_unset=True)}\n\n'
                        yield 'data: [DONE]\n\n'
                    elif isinstance(node, ModelRequestNode):
                        async with node.stream(run.ctx) as request_stream:
                            async for agent_event in request_stream:
                                chunk = _to_openai_chat_completion_chunk(agent_event, params.get('model'), run_id, context)
                                if chunk:
                                    yield f'data: {chunk.model_dump_json(exclude_unset=True)}\n\n'

        return StreamingResponse(
            stream_generator(),
            media_type='text/event-stream',
        )
    else:
        run_result = await agent.run(message_history=messages, **agent_kwargs)
        completion = _to_openai_chat_completion(run_result, model=params.get('model'))
        return Response(
            content=completion.model_dump_json(exclude_unset=True),
            media_type='application/json',
        )


async def handle_responses_request(
    agent: AbstractAgent[AgentDepsT, Any],
    request: Request,
    *,
    output_type: OutputSpec[Any] | None = None,
    model: Model | KnownModelName | str | None = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    infer_name: bool = True,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
) -> Response:
    """Handles OpenAI Responses API requests.

    This function processes HTTP requests to the `/v1/responses` endpoint,
    providing OpenAI Responses API compatibility for Pydantic AI agents. Unlike
    Chat Completions, the Responses API uses different request/response formats
    with `input` instead of `messages` and `output` instead of `choices`.

    Args:
        agent: The Pydantic AI agent to run for generating responses.
        request: The HTTP request containing the response parameters.

        output_type: Custom output type to use for this run, `output_type` may only be used if the agent has
            no output validators since output validators would expect an argument that matches the agent's
            output type.
        model: Optional model to use for this run, required if `model` was not set when creating the agent.
        deps: Optional dependencies to use for this run.
        model_settings: Optional settings to use for this model's request.
        usage_limits: Optional limits on model request count or token usage.
        usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
        infer_name: Whether to try to infer the agent name from the call frame if it's not set.
        toolsets: Optional additional toolsets for this run.

    Returns:
        A Starlette Response object containing either:
        - A StreamingResponse with server-sent events for streaming requests (not yet implemented)
        - A JSON Response with the complete Response object for non-streaming requests
        - An error response for invalid requests

    Example:
        ```python
        response = await handle_responses_request(agent, request)
        ```
    """
    try:
        params: ResponseCreateParamsT = ResponseCreateParams.validate_python(await request.json())
    except ValidationError as e:  # pragma: no cover
        return Response(
            content=json.dumps(e.json()),
            media_type='application/json',
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
        )

    # Extract input and instructions from Responses API format
    input_data = params.get('input')
    instructions = params.get('instructions')
    
    # Convert Responses API input to internal message format
    messages = _from_responses_input(input_data, instructions)
    
    agent_kwargs = dict(
        output_type=output_type,
        model=model or params.get('model'),
        deps=deps,
        model_settings=model_settings,
        usage_limits=usage_limits,
        usage=usage,
        infer_name=infer_name,
        toolsets=toolsets,
    )

    # For now, only support non-streaming responses
    # Streaming could be added in the future using ResponseStreamEvent
    if params.get('stream'):
        # TODO: Implement streaming support for Responses API
        return Response(
            content=json.dumps({'error': 'Streaming not yet supported for Responses API'}),
            media_type='application/json',
            status_code=HTTPStatus.NOT_IMPLEMENTED,
        )
    
    # Run the agent and convert to Response format
    run_result = await agent.run(message_history=messages, **agent_kwargs)
    response_obj = _to_openai_response(run_result, model=params.get('model'), input_data=input_data, instructions=instructions)
    
    return Response(
        content=response_obj.model_dump_json(exclude_unset=True),
        media_type='application/json',
    )
