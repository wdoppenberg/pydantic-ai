from __future__ import annotations as _annotations

import io
from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, cast, overload

from typing_extensions import assert_never

from pydantic_ai.builtin_tools import CodeExecutionTool, WebSearchTool

from .. import ModelHTTPError, UnexpectedModelBehavior, _utils, usage
from .._run_context import RunContext
from .._utils import guard_tool_call_id as _guard_tool_call_id
from ..exceptions import UserError
from ..messages import (
    BinaryContent,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    DocumentUrl,
    FinishReason,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    ModelResponseStreamEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from ..profiles import ModelProfileSpec
from ..providers import Provider, infer_provider
from ..providers.anthropic import AsyncAnthropicClient
from ..settings import ModelSettings
from ..tools import ToolDefinition
from . import Model, ModelRequestParameters, StreamedResponse, check_allow_model_requests, download_item, get_user_agent

_FINISH_REASON_MAP: dict[BetaStopReason, FinishReason] = {
    'end_turn': 'stop',
    'max_tokens': 'length',
    'stop_sequence': 'stop',
    'tool_use': 'tool_call',
    'pause_turn': 'stop',
    'refusal': 'content_filter',
}


try:
    from anthropic import NOT_GIVEN, APIStatusError, AsyncStream
    from anthropic.types.beta import (
        BetaBase64PDFBlockParam,
        BetaBase64PDFSourceParam,
        BetaCitationsDelta,
        BetaCodeExecutionTool20250522Param,
        BetaCodeExecutionToolResultBlock,
        BetaCodeExecutionToolResultBlockParam,
        BetaContentBlock,
        BetaContentBlockParam,
        BetaImageBlockParam,
        BetaInputJSONDelta,
        BetaMessage,
        BetaMessageParam,
        BetaMetadataParam,
        BetaPlainTextSourceParam,
        BetaRawContentBlockDeltaEvent,
        BetaRawContentBlockStartEvent,
        BetaRawContentBlockStopEvent,
        BetaRawMessageDeltaEvent,
        BetaRawMessageStartEvent,
        BetaRawMessageStopEvent,
        BetaRawMessageStreamEvent,
        BetaRedactedThinkingBlock,
        BetaRedactedThinkingBlockParam,
        BetaServerToolUseBlock,
        BetaServerToolUseBlockParam,
        BetaSignatureDelta,
        BetaStopReason,
        BetaTextBlock,
        BetaTextBlockParam,
        BetaTextDelta,
        BetaThinkingBlock,
        BetaThinkingBlockParam,
        BetaThinkingConfigParam,
        BetaThinkingDelta,
        BetaToolChoiceParam,
        BetaToolParam,
        BetaToolResultBlockParam,
        BetaToolUnionParam,
        BetaToolUseBlock,
        BetaToolUseBlockParam,
        BetaWebSearchTool20250305Param,
        BetaWebSearchToolResultBlock,
        BetaWebSearchToolResultBlockParam,
    )
    from anthropic.types.beta.beta_web_search_tool_20250305_param import UserLocation
    from anthropic.types.model_param import ModelParam

except ImportError as _import_error:
    raise ImportError(
        'Please install `anthropic` to use the Anthropic model, '
        'you can use the `anthropic` optional group — `pip install "pydantic-ai-slim[anthropic]"`'
    ) from _import_error

LatestAnthropicModelNames = ModelParam
"""Latest Anthropic models."""

AnthropicModelName = str | LatestAnthropicModelNames
"""Possible Anthropic model names.

Since Anthropic supports a variety of date-stamped models, we explicitly list the latest models but
allow any name in the type hints.
See [the Anthropic docs](https://docs.anthropic.com/en/docs/about-claude/models) for a full list.
"""


class AnthropicModelSettings(ModelSettings, total=False):
    """Settings used for an Anthropic model request."""

    # ALL FIELDS MUST BE `anthropic_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.

    anthropic_metadata: BetaMetadataParam
    """An object describing metadata about the request.

    Contains `user_id`, an external identifier for the user who is associated with the request.
    """

    anthropic_thinking: BetaThinkingConfigParam
    """Determine whether the model should generate a thinking block.

    See [the Anthropic docs](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking) for more information.
    """


@dataclass(init=False)
class AnthropicModel(Model):
    """A model that uses the Anthropic API.

    Internally, this uses the [Anthropic Python client](https://github.com/anthropics/anthropic-sdk-python) to interact with the API.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    client: AsyncAnthropicClient = field(repr=False)

    _model_name: AnthropicModelName = field(repr=False)
    _provider: Provider[AsyncAnthropicClient] = field(repr=False)

    def __init__(
        self,
        model_name: AnthropicModelName,
        *,
        provider: Literal['anthropic'] | Provider[AsyncAnthropicClient] = 'anthropic',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize an Anthropic model.

        Args:
            model_name: The name of the Anthropic model to use. List of model names available
                [here](https://docs.anthropic.com/en/docs/about-claude/models).
            provider: The provider to use for the Anthropic API. Can be either the string 'anthropic' or an
                instance of `Provider[AsyncAnthropicClient]`. If not provided, the other parameters will be used.
            profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
            settings: Default model settings for this model instance.
        """
        self._model_name = model_name

        if isinstance(provider, str):
            provider = infer_provider(provider)
        self._provider = provider
        self.client = provider.client

        super().__init__(settings=settings, profile=profile or provider.model_profile)

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    @property
    def model_name(self) -> AnthropicModelName:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The model provider."""
        return self._provider.name

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        check_allow_model_requests()
        response = await self._messages_create(
            messages, False, cast(AnthropicModelSettings, model_settings or {}), model_request_parameters
        )
        model_response = self._process_response(response)
        return model_response

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        check_allow_model_requests()
        response = await self._messages_create(
            messages, True, cast(AnthropicModelSettings, model_settings or {}), model_request_parameters
        )
        async with response:
            yield await self._process_streamed_response(response, model_request_parameters)

    @overload
    async def _messages_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[True],
        model_settings: AnthropicModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncStream[BetaRawMessageStreamEvent]:
        pass

    @overload
    async def _messages_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[False],
        model_settings: AnthropicModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> BetaMessage:
        pass

    async def _messages_create(
        self,
        messages: list[ModelMessage],
        stream: bool,
        model_settings: AnthropicModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> BetaMessage | AsyncStream[BetaRawMessageStreamEvent]:
        # standalone function to make it easier to override
        tools = self._get_tools(model_request_parameters)
        builtin_tools, tool_headers = self._get_builtin_tools(model_request_parameters)
        tools += builtin_tools

        tool_choice: BetaToolChoiceParam | None

        if not tools:
            tool_choice = None
        else:
            if not model_request_parameters.allow_text_output:
                tool_choice = {'type': 'any'}
            else:
                tool_choice = {'type': 'auto'}

            if (allow_parallel_tool_calls := model_settings.get('parallel_tool_calls')) is not None:
                tool_choice['disable_parallel_tool_use'] = not allow_parallel_tool_calls

        system_prompt, anthropic_messages = await self._map_message(messages)

        try:
            extra_headers = model_settings.get('extra_headers', {})
            for k, v in tool_headers.items():
                extra_headers.setdefault(k, v)
            extra_headers.setdefault('User-Agent', get_user_agent())

            return await self.client.beta.messages.create(
                max_tokens=model_settings.get('max_tokens', 4096),
                system=system_prompt or NOT_GIVEN,
                messages=anthropic_messages,
                model=self._model_name,
                tools=tools or NOT_GIVEN,
                tool_choice=tool_choice or NOT_GIVEN,
                stream=stream,
                thinking=model_settings.get('anthropic_thinking', NOT_GIVEN),
                stop_sequences=model_settings.get('stop_sequences', NOT_GIVEN),
                temperature=model_settings.get('temperature', NOT_GIVEN),
                top_p=model_settings.get('top_p', NOT_GIVEN),
                timeout=model_settings.get('timeout', NOT_GIVEN),
                metadata=model_settings.get('anthropic_metadata', NOT_GIVEN),
                extra_headers=extra_headers,
                extra_body=model_settings.get('extra_body'),
            )
        except APIStatusError as e:
            if (status_code := e.status_code) >= 400:
                raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.body) from e
            raise  # pragma: lax no cover

    def _process_response(self, response: BetaMessage) -> ModelResponse:
        """Process a non-streamed response, and prepare a message to return."""
        items: list[ModelResponsePart] = []
        for item in response.content:
            if isinstance(item, BetaTextBlock):
                items.append(TextPart(content=item.text))
            elif isinstance(item, BetaWebSearchToolResultBlock | BetaCodeExecutionToolResultBlock):
                items.append(
                    BuiltinToolReturnPart(
                        provider_name=self.system,
                        tool_name=item.type,
                        content=item.content,
                        tool_call_id=item.tool_use_id,
                    )
                )
            elif isinstance(item, BetaServerToolUseBlock):
                items.append(
                    BuiltinToolCallPart(
                        provider_name=self.system,
                        tool_name=item.name,
                        args=cast(dict[str, Any], item.input),
                        tool_call_id=item.id,
                    )
                )
            elif isinstance(item, BetaRedactedThinkingBlock):
                items.append(
                    ThinkingPart(id='redacted_thinking', content='', signature=item.data, provider_name=self.system)
                )
            elif isinstance(item, BetaThinkingBlock):
                items.append(ThinkingPart(content=item.thinking, signature=item.signature, provider_name=self.system))
            else:
                assert isinstance(item, BetaToolUseBlock), f'unexpected item type {type(item)}'
                items.append(
                    ToolCallPart(
                        tool_name=item.name,
                        args=cast(dict[str, Any], item.input),
                        tool_call_id=item.id,
                    )
                )

        finish_reason: FinishReason | None = None
        provider_details: dict[str, Any] | None = None
        if raw_finish_reason := response.stop_reason:  # pragma: no branch
            provider_details = {'finish_reason': raw_finish_reason}
            finish_reason = _FINISH_REASON_MAP.get(raw_finish_reason)

        return ModelResponse(
            parts=items,
            usage=_map_usage(response),
            model_name=response.model,
            provider_response_id=response.id,
            provider_name=self._provider.name,
            finish_reason=finish_reason,
            provider_details=provider_details,
        )

    async def _process_streamed_response(
        self, response: AsyncStream[BetaRawMessageStreamEvent], model_request_parameters: ModelRequestParameters
    ) -> StreamedResponse:
        peekable_response = _utils.PeekableAsyncStream(response)
        first_chunk = await peekable_response.peek()
        if isinstance(first_chunk, _utils.Unset):
            raise UnexpectedModelBehavior('Streamed response ended without content or tool calls')  # pragma: no cover

        assert isinstance(first_chunk, BetaRawMessageStartEvent)

        return AnthropicStreamedResponse(
            model_request_parameters=model_request_parameters,
            _model_name=first_chunk.message.model,
            _response=peekable_response,
            _timestamp=_utils.now_utc(),
            _provider_name=self._provider.name,
        )

    def _get_tools(self, model_request_parameters: ModelRequestParameters) -> list[BetaToolParam]:
        return [self._map_tool_definition(r) for r in model_request_parameters.tool_defs.values()]

    def _get_builtin_tools(
        self, model_request_parameters: ModelRequestParameters
    ) -> tuple[list[BetaToolUnionParam], dict[str, str]]:
        tools: list[BetaToolUnionParam] = []
        extra_headers: dict[str, str] = {}
        for tool in model_request_parameters.builtin_tools:
            if isinstance(tool, WebSearchTool):
                user_location = UserLocation(type='approximate', **tool.user_location) if tool.user_location else None
                tools.append(
                    BetaWebSearchTool20250305Param(
                        name='web_search',
                        type='web_search_20250305',
                        max_uses=tool.max_uses,
                        allowed_domains=tool.allowed_domains,
                        blocked_domains=tool.blocked_domains,
                        user_location=user_location,
                    )
                )
            elif isinstance(tool, CodeExecutionTool):  # pragma: no branch
                extra_headers['anthropic-beta'] = 'code-execution-2025-05-22'
                tools.append(BetaCodeExecutionTool20250522Param(name='code_execution', type='code_execution_20250522'))
            else:  # pragma: no cover
                raise UserError(
                    f'`{tool.__class__.__name__}` is not supported by `AnthropicModel`. If it should be, please file an issue.'
                )
        return tools, extra_headers

    async def _map_message(self, messages: list[ModelMessage]) -> tuple[str, list[BetaMessageParam]]:  # noqa: C901
        """Just maps a `pydantic_ai.Message` to a `anthropic.types.MessageParam`."""
        system_prompt_parts: list[str] = []
        anthropic_messages: list[BetaMessageParam] = []
        for m in messages:
            if isinstance(m, ModelRequest):
                user_content_params: list[BetaContentBlockParam] = []
                for request_part in m.parts:
                    if isinstance(request_part, SystemPromptPart):
                        system_prompt_parts.append(request_part.content)
                    elif isinstance(request_part, UserPromptPart):
                        async for content in self._map_user_prompt(request_part):
                            user_content_params.append(content)
                    elif isinstance(request_part, ToolReturnPart):
                        tool_result_block_param = BetaToolResultBlockParam(
                            tool_use_id=_guard_tool_call_id(t=request_part),
                            type='tool_result',
                            content=request_part.model_response_str(),
                            is_error=False,
                        )
                        user_content_params.append(tool_result_block_param)
                    elif isinstance(request_part, RetryPromptPart):  # pragma: no branch
                        if request_part.tool_name is None:
                            text = request_part.model_response()  # pragma: no cover
                            retry_param = BetaTextBlockParam(type='text', text=text)  # pragma: no cover
                        else:
                            retry_param = BetaToolResultBlockParam(
                                tool_use_id=_guard_tool_call_id(t=request_part),
                                type='tool_result',
                                content=request_part.model_response(),
                                is_error=True,
                            )
                        user_content_params.append(retry_param)
                if len(user_content_params) > 0:
                    anthropic_messages.append(BetaMessageParam(role='user', content=user_content_params))
            elif isinstance(m, ModelResponse):
                assistant_content_params: list[
                    BetaTextBlockParam
                    | BetaToolUseBlockParam
                    | BetaServerToolUseBlockParam
                    | BetaWebSearchToolResultBlockParam
                    | BetaCodeExecutionToolResultBlockParam
                    | BetaThinkingBlockParam
                    | BetaRedactedThinkingBlockParam
                ] = []
                for response_part in m.parts:
                    if isinstance(response_part, TextPart):
                        if response_part.content:
                            assistant_content_params.append(BetaTextBlockParam(text=response_part.content, type='text'))
                    elif isinstance(response_part, ToolCallPart):
                        tool_use_block_param = BetaToolUseBlockParam(
                            id=_guard_tool_call_id(t=response_part),
                            type='tool_use',
                            name=response_part.tool_name,
                            input=response_part.args_as_dict(),
                        )
                        assistant_content_params.append(tool_use_block_param)
                    elif isinstance(response_part, ThinkingPart):
                        if (
                            response_part.provider_name == self.system and response_part.signature is not None
                        ):  # pragma: no branch
                            if response_part.id == 'redacted_thinking':
                                assistant_content_params.append(
                                    BetaRedactedThinkingBlockParam(
                                        data=response_part.signature,
                                        type='redacted_thinking',
                                    )
                                )
                            else:
                                assistant_content_params.append(
                                    BetaThinkingBlockParam(
                                        thinking=response_part.content,
                                        signature=response_part.signature,
                                        type='thinking',
                                    )
                                )
                        elif response_part.content:  # pragma: no branch
                            start_tag, end_tag = self.profile.thinking_tags
                            assistant_content_params.append(
                                BetaTextBlockParam(
                                    text='\n'.join([start_tag, response_part.content, end_tag]), type='text'
                                )
                            )
                    elif isinstance(response_part, BuiltinToolCallPart):
                        if response_part.provider_name == self.system:
                            server_tool_use_block_param = BetaServerToolUseBlockParam(
                                id=_guard_tool_call_id(t=response_part),
                                type='server_tool_use',
                                name=cast(Literal['web_search', 'code_execution'], response_part.tool_name),
                                input=response_part.args_as_dict(),
                            )
                            assistant_content_params.append(server_tool_use_block_param)
                    elif isinstance(response_part, BuiltinToolReturnPart):
                        if response_part.provider_name == self.system:
                            tool_use_id = _guard_tool_call_id(t=response_part)
                            if response_part.tool_name == 'web_search_tool_result':
                                server_tool_result_block_param = BetaWebSearchToolResultBlockParam(
                                    tool_use_id=tool_use_id, type=response_part.tool_name, content=response_part.content
                                )
                            elif response_part.tool_name == 'code_execution_tool_result':
                                server_tool_result_block_param = BetaCodeExecutionToolResultBlockParam(
                                    tool_use_id=tool_use_id, type=response_part.tool_name, content=response_part.content
                                )
                            else:
                                raise ValueError(f'Unsupported tool name: {response_part.tool_name}')
                            assistant_content_params.append(server_tool_result_block_param)
                    else:
                        assert_never(response_part)
                if len(assistant_content_params) > 0:
                    anthropic_messages.append(BetaMessageParam(role='assistant', content=assistant_content_params))
            else:
                assert_never(m)
        if instructions := self._get_instructions(messages):
            system_prompt_parts.insert(0, instructions)
        system_prompt = '\n\n'.join(system_prompt_parts)
        return system_prompt, anthropic_messages

    @staticmethod
    async def _map_user_prompt(
        part: UserPromptPart,
    ) -> AsyncGenerator[BetaContentBlockParam]:
        if isinstance(part.content, str):
            if part.content:  # Only yield non-empty text
                yield BetaTextBlockParam(text=part.content, type='text')
        else:
            for item in part.content:
                if isinstance(item, str):
                    if item:  # Only yield non-empty text
                        yield BetaTextBlockParam(text=item, type='text')
                elif isinstance(item, BinaryContent):
                    if item.is_image:
                        yield BetaImageBlockParam(
                            source={'data': io.BytesIO(item.data), 'media_type': item.media_type, 'type': 'base64'},  # type: ignore
                            type='image',
                        )
                    elif item.media_type == 'application/pdf':
                        yield BetaBase64PDFBlockParam(
                            source=BetaBase64PDFSourceParam(
                                data=io.BytesIO(item.data),
                                media_type='application/pdf',
                                type='base64',
                            ),
                            type='document',
                        )
                    else:
                        raise RuntimeError('Only images and PDFs are supported for binary content')
                elif isinstance(item, ImageUrl):
                    yield BetaImageBlockParam(source={'type': 'url', 'url': item.url}, type='image')
                elif isinstance(item, DocumentUrl):
                    if item.media_type == 'application/pdf':
                        yield BetaBase64PDFBlockParam(source={'url': item.url, 'type': 'url'}, type='document')
                    elif item.media_type == 'text/plain':
                        downloaded_item = await download_item(item, data_format='text')
                        yield BetaBase64PDFBlockParam(
                            source=BetaPlainTextSourceParam(
                                data=downloaded_item['data'], media_type=item.media_type, type='text'
                            ),
                            type='document',
                        )
                    else:  # pragma: no cover
                        raise RuntimeError(f'Unsupported media type: {item.media_type}')
                else:
                    raise RuntimeError(f'Unsupported content type: {type(item)}')  # pragma: no cover

    @staticmethod
    def _map_tool_definition(f: ToolDefinition) -> BetaToolParam:
        return {
            'name': f.name,
            'description': f.description or '',
            'input_schema': f.parameters_json_schema,
        }


def _map_usage(message: BetaMessage | BetaRawMessageStartEvent | BetaRawMessageDeltaEvent) -> usage.RequestUsage:
    if isinstance(message, BetaMessage):
        response_usage = message.usage
    elif isinstance(message, BetaRawMessageStartEvent):
        response_usage = message.message.usage
    elif isinstance(message, BetaRawMessageDeltaEvent):
        response_usage = message.usage
    else:
        assert_never(message)

    # Store all integer-typed usage values in the details, except 'output_tokens' which is represented exactly by
    # `response_tokens`
    details: dict[str, int] = {
        key: value for key, value in response_usage.model_dump().items() if isinstance(value, int)
    }

    # Usage coming from the RawMessageDeltaEvent doesn't have input token data, hence using `get`
    # Tokens are only counted once between input_tokens, cache_creation_input_tokens, and cache_read_input_tokens
    # This approach maintains request_tokens as the count of all input tokens, with cached counts as details
    cache_write_tokens = details.get('cache_creation_input_tokens', 0)
    cache_read_tokens = details.get('cache_read_input_tokens', 0)
    request_tokens = details.get('input_tokens', 0) + cache_write_tokens + cache_read_tokens

    return usage.RequestUsage(
        input_tokens=request_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        output_tokens=response_usage.output_tokens,
        details=details,
    )


@dataclass
class AnthropicStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for Anthropic models."""

    _model_name: AnthropicModelName
    _response: AsyncIterable[BetaRawMessageStreamEvent]
    _timestamp: datetime
    _provider_name: str

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:  # noqa: C901
        current_block: BetaContentBlock | None = None

        async for event in self._response:
            if isinstance(event, BetaRawMessageStartEvent):
                self._usage = _map_usage(event)
                self.provider_response_id = event.message.id

            elif isinstance(event, BetaRawContentBlockStartEvent):
                current_block = event.content_block
                if isinstance(current_block, BetaTextBlock) and current_block.text:
                    maybe_event = self._parts_manager.handle_text_delta(
                        vendor_part_id=event.index, content=current_block.text
                    )
                    if maybe_event is not None:  # pragma: no branch
                        yield maybe_event
                elif isinstance(current_block, BetaThinkingBlock):
                    yield self._parts_manager.handle_thinking_delta(
                        vendor_part_id=event.index,
                        content=current_block.thinking,
                        signature=current_block.signature,
                        provider_name=self.provider_name,
                    )
                elif isinstance(current_block, BetaRedactedThinkingBlock):
                    yield self._parts_manager.handle_thinking_delta(
                        vendor_part_id=event.index,
                        id='redacted_thinking',
                        signature=current_block.data,
                        provider_name=self.provider_name,
                    )
                elif isinstance(current_block, BetaToolUseBlock):
                    maybe_event = self._parts_manager.handle_tool_call_delta(
                        vendor_part_id=current_block.id,
                        tool_name=current_block.name,
                        args=cast(dict[str, Any], current_block.input) or None,
                        tool_call_id=current_block.id,
                    )
                    if maybe_event is not None:  # pragma: no branch
                        yield maybe_event
                elif isinstance(current_block, BetaServerToolUseBlock):
                    pass

            elif isinstance(event, BetaRawContentBlockDeltaEvent):
                if isinstance(event.delta, BetaTextDelta):
                    maybe_event = self._parts_manager.handle_text_delta(
                        vendor_part_id=event.index, content=event.delta.text
                    )
                    if maybe_event is not None:  # pragma: no branch
                        yield maybe_event
                elif isinstance(event.delta, BetaThinkingDelta):
                    yield self._parts_manager.handle_thinking_delta(
                        vendor_part_id=event.index,
                        content=event.delta.thinking,
                        provider_name=self.provider_name,
                    )
                elif isinstance(event.delta, BetaSignatureDelta):
                    yield self._parts_manager.handle_thinking_delta(
                        vendor_part_id=event.index,
                        signature=event.delta.signature,
                        provider_name=self.provider_name,
                    )
                elif (
                    current_block
                    and event.delta.type == 'input_json_delta'
                    and isinstance(current_block, BetaToolUseBlock)
                ):  # pragma: no branch
                    maybe_event = self._parts_manager.handle_tool_call_delta(
                        vendor_part_id=current_block.id,
                        tool_name='',
                        args=event.delta.partial_json,
                        tool_call_id=current_block.id,
                    )
                    if maybe_event is not None:  # pragma: no branch
                        yield maybe_event
                elif isinstance(event.delta, BetaInputJSONDelta):
                    pass
                # TODO(Marcelo): We need to handle citations.
                elif isinstance(event.delta, BetaCitationsDelta):
                    pass

            elif isinstance(event, BetaRawMessageDeltaEvent):
                self._usage = _map_usage(event)
                if raw_finish_reason := event.delta.stop_reason:  # pragma: no branch
                    self.provider_details = {'finish_reason': raw_finish_reason}
                    self.finish_reason = _FINISH_REASON_MAP.get(raw_finish_reason)

            elif isinstance(event, BetaRawContentBlockStopEvent | BetaRawMessageStopEvent):  # pragma: no branch
                current_block = None

    @property
    def model_name(self) -> AnthropicModelName:
        """Get the model name of the response."""
        return self._model_name

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return self._provider_name

    @property
    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        return self._timestamp
