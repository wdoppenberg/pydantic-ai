from __future__ import annotations as _annotations

import base64
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, cast, overload
from uuid import uuid4

from typing_extensions import assert_never

from .. import UnexpectedModelBehavior, _utils, usage
from .._output import OutputObjectDefinition
from .._run_context import RunContext
from ..builtin_tools import CodeExecutionTool, UrlContextTool, WebSearchTool
from ..exceptions import UserError
from ..messages import (
    BinaryContent,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    FileUrl,
    FinishReason,
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
    VideoUrl,
)
from ..profiles import ModelProfileSpec
from ..providers import Provider
from ..settings import ModelSettings
from ..tools import ToolDefinition
from . import (
    Model,
    ModelRequestParameters,
    StreamedResponse,
    check_allow_model_requests,
    download_item,
    get_user_agent,
)

try:
    from google.genai import Client
    from google.genai.types import (
        ContentDict,
        ContentUnionDict,
        CountTokensConfigDict,
        ExecutableCodeDict,
        FinishReason as GoogleFinishReason,
        FunctionCallDict,
        FunctionCallingConfigDict,
        FunctionCallingConfigMode,
        FunctionDeclarationDict,
        GenerateContentConfigDict,
        GenerateContentResponse,
        GenerationConfigDict,
        GoogleSearchDict,
        HttpOptionsDict,
        MediaResolution,
        Part,
        PartDict,
        SafetySettingDict,
        ThinkingConfigDict,
        ToolCodeExecutionDict,
        ToolConfigDict,
        ToolDict,
        ToolListUnionDict,
        UrlContextDict,
    )

    from ..providers.google import GoogleProvider
except ImportError as _import_error:
    raise ImportError(
        'Please install `google-genai` to use the Google model, '
        'you can use the `google` optional group — `pip install "pydantic-ai-slim[google]"`'
    ) from _import_error

LatestGoogleModelNames = Literal[
    'gemini-2.0-flash',
    'gemini-2.0-flash-lite',
    'gemini-2.5-flash',
    'gemini-2.5-flash-lite',
    'gemini-2.5-pro',
]
"""Latest Gemini models."""

GoogleModelName = str | LatestGoogleModelNames
"""Possible Gemini model names.

Since Gemini supports a variety of date-stamped models, we explicitly list the latest models but
allow any name in the type hints.
See [the Gemini API docs](https://ai.google.dev/gemini-api/docs/models/gemini#model-variations) for a full list.
"""

_FINISH_REASON_MAP: dict[GoogleFinishReason, FinishReason | None] = {
    GoogleFinishReason.FINISH_REASON_UNSPECIFIED: None,
    GoogleFinishReason.STOP: 'stop',
    GoogleFinishReason.MAX_TOKENS: 'length',
    GoogleFinishReason.SAFETY: 'content_filter',
    GoogleFinishReason.RECITATION: 'content_filter',
    GoogleFinishReason.LANGUAGE: 'error',
    GoogleFinishReason.OTHER: None,
    GoogleFinishReason.BLOCKLIST: 'content_filter',
    GoogleFinishReason.PROHIBITED_CONTENT: 'content_filter',
    GoogleFinishReason.SPII: 'content_filter',
    GoogleFinishReason.MALFORMED_FUNCTION_CALL: 'error',
    GoogleFinishReason.IMAGE_SAFETY: 'content_filter',
    GoogleFinishReason.UNEXPECTED_TOOL_CALL: 'error',
}


class GoogleModelSettings(ModelSettings, total=False):
    """Settings used for a Gemini model request."""

    # ALL FIELDS MUST BE `gemini_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.

    google_safety_settings: list[SafetySettingDict]
    """The safety settings to use for the model.

    See <https://ai.google.dev/gemini-api/docs/safety-settings> for more information.
    """

    google_thinking_config: ThinkingConfigDict
    """The thinking configuration to use for the model.

    See <https://ai.google.dev/gemini-api/docs/thinking> for more information.
    """

    google_labels: dict[str, str]
    """User-defined metadata to break down billed charges. Only supported by the Vertex AI API.

    See the [Gemini API docs](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/add-labels-to-api-calls) for use cases and limitations.
    """

    google_video_resolution: MediaResolution
    """The video resolution to use for the model.

    See <https://ai.google.dev/api/generate-content#MediaResolution> for more information.
    """

    google_cached_content: str
    """The name of the cached content to use for the model.

    See <https://ai.google.dev/gemini-api/docs/caching> for more information.
    """


@dataclass(init=False)
class GoogleModel(Model):
    """A model that uses Gemini via `generativelanguage.googleapis.com` API.

    This is implemented from scratch rather than using a dedicated SDK, good API documentation is
    available [here](https://ai.google.dev/api).

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    client: Client = field(repr=False)

    _model_name: GoogleModelName = field(repr=False)
    _provider: Provider[Client] = field(repr=False)
    _url: str | None = field(repr=False)

    def __init__(
        self,
        model_name: GoogleModelName,
        *,
        provider: Literal['google-gla', 'google-vertex'] | Provider[Client] = 'google-gla',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize a Gemini model.

        Args:
            model_name: The name of the model to use.
            provider: The provider to use for authentication and API access. Can be either the string
                'google-gla' or 'google-vertex' or an instance of `Provider[httpx.AsyncClient]`.
                If not provided, a new provider will be created using the other parameters.
            profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
            settings: The model settings to use. Defaults to None.
        """
        self._model_name = model_name

        if isinstance(provider, str):
            provider = GoogleProvider(vertexai=provider == 'google-vertex')
        self._provider = provider
        self.client = provider.client

        super().__init__(settings=settings, profile=profile or provider.model_profile)

    @property
    def base_url(self) -> str:
        return self._provider.base_url

    @property
    def model_name(self) -> GoogleModelName:
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
        model_settings = cast(GoogleModelSettings, model_settings or {})
        response = await self._generate_content(messages, False, model_settings, model_request_parameters)
        return self._process_response(response)

    async def count_tokens(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> usage.RequestUsage:
        check_allow_model_requests()
        model_settings = cast(GoogleModelSettings, model_settings or {})
        contents, generation_config = await self._build_content_and_config(
            messages, model_settings, model_request_parameters
        )

        # Annoyingly, the type of `GenerateContentConfigDict.get` is "partially `Unknown`" because `response_schema` includes `typing._UnionGenericAlias`,
        # so without this we'd need `pyright: ignore[reportUnknownMemberType]` on every line and wouldn't get type checking anyway.
        generation_config = cast(dict[str, Any], generation_config)

        config = CountTokensConfigDict(
            http_options=generation_config.get('http_options'),
        )
        if self._provider.name != 'google-gla':
            # The fields are not supported by the Gemini API per https://github.com/googleapis/python-genai/blob/7e4ec284dc6e521949626f3ed54028163ef9121d/google/genai/models.py#L1195-L1214
            config.update(  # pragma: lax no cover
                system_instruction=generation_config.get('system_instruction'),
                tools=cast(list[ToolDict], generation_config.get('tools')),
                # Annoyingly, GenerationConfigDict has fewer fields than GenerateContentConfigDict, and no extra fields are allowed.
                generation_config=GenerationConfigDict(
                    temperature=generation_config.get('temperature'),
                    top_p=generation_config.get('top_p'),
                    max_output_tokens=generation_config.get('max_output_tokens'),
                    stop_sequences=generation_config.get('stop_sequences'),
                    presence_penalty=generation_config.get('presence_penalty'),
                    frequency_penalty=generation_config.get('frequency_penalty'),
                    seed=generation_config.get('seed'),
                    thinking_config=generation_config.get('thinking_config'),
                    media_resolution=generation_config.get('media_resolution'),
                    response_mime_type=generation_config.get('response_mime_type'),
                    response_schema=generation_config.get('response_schema'),
                ),
            )

        response = await self.client.aio.models.count_tokens(
            model=self._model_name,
            contents=contents,
            config=config,
        )
        if response.total_tokens is None:
            raise UnexpectedModelBehavior(  # pragma: no cover
                'Total tokens missing from Gemini response', str(response)
            )
        return usage.RequestUsage(
            input_tokens=response.total_tokens,
        )

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        check_allow_model_requests()
        model_settings = cast(GoogleModelSettings, model_settings or {})
        response = await self._generate_content(messages, True, model_settings, model_request_parameters)
        yield await self._process_streamed_response(response, model_request_parameters)  # type: ignore

    def _get_tools(self, model_request_parameters: ModelRequestParameters) -> list[ToolDict] | None:
        if model_request_parameters.builtin_tools:
            if model_request_parameters.output_tools:
                raise UserError(
                    'Gemini does not support output tools and built-in tools at the same time. Use `output_type=PromptedOutput(...)` instead.'
                )
            if model_request_parameters.function_tools:
                raise UserError('Gemini does not support user tools and built-in tools at the same time.')

        tools: list[ToolDict] = [
            ToolDict(function_declarations=[_function_declaration_from_tool(t)])
            for t in model_request_parameters.tool_defs.values()
        ]
        for tool in model_request_parameters.builtin_tools:
            if isinstance(tool, WebSearchTool):
                tools.append(ToolDict(google_search=GoogleSearchDict()))
            elif isinstance(tool, UrlContextTool):
                tools.append(ToolDict(url_context=UrlContextDict()))
            elif isinstance(tool, CodeExecutionTool):  # pragma: no branch
                tools.append(ToolDict(code_execution=ToolCodeExecutionDict()))
            else:  # pragma: no cover
                raise UserError(
                    f'`{tool.__class__.__name__}` is not supported by `GoogleModel`. If it should be, please file an issue.'
                )
        return tools or None

    def _get_tool_config(
        self, model_request_parameters: ModelRequestParameters, tools: list[ToolDict] | None
    ) -> ToolConfigDict | None:
        if not model_request_parameters.allow_text_output and tools:
            names: list[str] = []
            for tool in tools:
                for function_declaration in tool.get('function_declarations') or []:
                    if name := function_declaration.get('name'):  # pragma: no branch
                        names.append(name)
            return _tool_config(names)
        else:
            return None

    @overload
    async def _generate_content(
        self,
        messages: list[ModelMessage],
        stream: Literal[False],
        model_settings: GoogleModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> GenerateContentResponse: ...

    @overload
    async def _generate_content(
        self,
        messages: list[ModelMessage],
        stream: Literal[True],
        model_settings: GoogleModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> Awaitable[AsyncIterator[GenerateContentResponse]]: ...

    async def _generate_content(
        self,
        messages: list[ModelMessage],
        stream: bool,
        model_settings: GoogleModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> GenerateContentResponse | Awaitable[AsyncIterator[GenerateContentResponse]]:
        contents, config = await self._build_content_and_config(messages, model_settings, model_request_parameters)
        func = self.client.aio.models.generate_content_stream if stream else self.client.aio.models.generate_content
        return await func(model=self._model_name, contents=contents, config=config)  # type: ignore

    async def _build_content_and_config(
        self,
        messages: list[ModelMessage],
        model_settings: GoogleModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[list[ContentUnionDict], GenerateContentConfigDict]:
        tools = self._get_tools(model_request_parameters)
        response_mime_type = None
        response_schema = None
        if model_request_parameters.output_mode == 'native':
            if tools:
                raise UserError(
                    'Gemini does not support `NativeOutput` and tools at the same time. Use `output_type=ToolOutput(...)` instead.'
                )
            response_mime_type = 'application/json'
            output_object = model_request_parameters.output_object
            assert output_object is not None
            response_schema = self._map_response_schema(output_object)
        elif model_request_parameters.output_mode == 'prompted' and not tools:
            response_mime_type = 'application/json'

        tool_config = self._get_tool_config(model_request_parameters, tools)
        system_instruction, contents = await self._map_messages(messages)

        http_options: HttpOptionsDict = {
            'headers': {'Content-Type': 'application/json', 'User-Agent': get_user_agent()}
        }
        if timeout := model_settings.get('timeout'):
            if isinstance(timeout, int | float):
                http_options['timeout'] = int(1000 * timeout)
            else:
                raise UserError('Google does not support setting ModelSettings.timeout to a httpx.Timeout')

        config = GenerateContentConfigDict(
            http_options=http_options,
            system_instruction=system_instruction,
            temperature=model_settings.get('temperature'),
            top_p=model_settings.get('top_p'),
            max_output_tokens=model_settings.get('max_tokens'),
            stop_sequences=model_settings.get('stop_sequences'),
            presence_penalty=model_settings.get('presence_penalty'),
            frequency_penalty=model_settings.get('frequency_penalty'),
            seed=model_settings.get('seed'),
            safety_settings=model_settings.get('google_safety_settings'),
            thinking_config=model_settings.get('google_thinking_config'),
            labels=model_settings.get('google_labels'),
            media_resolution=model_settings.get('google_video_resolution'),
            cached_content=model_settings.get('google_cached_content'),
            tools=cast(ToolListUnionDict, tools),
            tool_config=tool_config,
            response_mime_type=response_mime_type,
            response_schema=response_schema,
        )
        return contents, config

    def _process_response(self, response: GenerateContentResponse) -> ModelResponse:
        if not response.candidates or len(response.candidates) != 1:
            raise UnexpectedModelBehavior('Expected exactly one candidate in Gemini response')  # pragma: no cover
        candidate = response.candidates[0]
        if candidate.content is None or candidate.content.parts is None:
            if candidate.finish_reason == 'SAFETY':
                raise UnexpectedModelBehavior('Safety settings triggered', str(response))
            else:
                raise UnexpectedModelBehavior(
                    'Content field missing from Gemini response', str(response)
                )  # pragma: no cover
        parts = candidate.content.parts or []

        vendor_id = response.response_id
        vendor_details: dict[str, Any] | None = None
        finish_reason: FinishReason | None = None
        if raw_finish_reason := candidate.finish_reason:  # pragma: no branch
            vendor_details = {'finish_reason': raw_finish_reason.value}
            finish_reason = _FINISH_REASON_MAP.get(raw_finish_reason)

        usage = _metadata_as_usage(response)
        return _process_response_from_parts(
            parts,
            response.model_version or self._model_name,
            self._provider.name,
            usage,
            vendor_id=vendor_id,
            vendor_details=vendor_details,
            finish_reason=finish_reason,
        )

    async def _process_streamed_response(
        self, response: AsyncIterator[GenerateContentResponse], model_request_parameters: ModelRequestParameters
    ) -> StreamedResponse:
        """Process a streamed response, and prepare a streaming response to return."""
        peekable_response = _utils.PeekableAsyncStream(response)
        first_chunk = await peekable_response.peek()
        if isinstance(first_chunk, _utils.Unset):
            raise UnexpectedModelBehavior('Streamed response ended without content or tool calls')  # pragma: no cover

        return GeminiStreamedResponse(
            model_request_parameters=model_request_parameters,
            _model_name=first_chunk.model_version or self._model_name,
            _response=peekable_response,
            _timestamp=first_chunk.create_time or _utils.now_utc(),
            _provider_name=self._provider.name,
        )

    async def _map_messages(self, messages: list[ModelMessage]) -> tuple[ContentDict | None, list[ContentUnionDict]]:
        contents: list[ContentUnionDict] = []
        system_parts: list[PartDict] = []

        for m in messages:
            if isinstance(m, ModelRequest):
                message_parts: list[PartDict] = []

                for part in m.parts:
                    if isinstance(part, SystemPromptPart):
                        system_parts.append({'text': part.content})
                    elif isinstance(part, UserPromptPart):
                        message_parts.extend(await self._map_user_prompt(part))
                    elif isinstance(part, ToolReturnPart):
                        message_parts.append(
                            {
                                'function_response': {
                                    'name': part.tool_name,
                                    'response': part.model_response_object(),
                                    'id': part.tool_call_id,
                                }
                            }
                        )
                    elif isinstance(part, RetryPromptPart):
                        if part.tool_name is None:
                            message_parts.append({'text': part.model_response()})  # pragma: no cover
                        else:
                            message_parts.append(
                                {
                                    'function_response': {
                                        'name': part.tool_name,
                                        'response': {'call_error': part.model_response()},
                                        'id': part.tool_call_id,
                                    }
                                }
                            )
                    else:
                        assert_never(part)

                # Google GenAI requires at least one part in the message.
                if not message_parts:
                    message_parts = [{'text': ''}]
                contents.append({'role': 'user', 'parts': message_parts})
            elif isinstance(m, ModelResponse):
                contents.append(_content_model_response(m, self.system))
            else:
                assert_never(m)
        if instructions := self._get_instructions(messages):
            system_parts.insert(0, {'text': instructions})
        system_instruction = ContentDict(role='user', parts=system_parts) if system_parts else None
        return system_instruction, contents

    async def _map_user_prompt(self, part: UserPromptPart) -> list[PartDict]:
        if isinstance(part.content, str):
            return [{'text': part.content}]
        else:
            content: list[PartDict] = []
            for item in part.content:
                if isinstance(item, str):
                    content.append({'text': item})
                elif isinstance(item, BinaryContent):
                    # NOTE: The type from Google GenAI is incorrect, it should be `str`, not `bytes`.
                    base64_encoded = base64.b64encode(item.data).decode('utf-8')
                    inline_data_dict = {'inline_data': {'data': base64_encoded, 'mime_type': item.media_type}}
                    if item.vendor_metadata:
                        inline_data_dict['video_metadata'] = item.vendor_metadata
                    content.append(inline_data_dict)  # type: ignore
                elif isinstance(item, VideoUrl) and item.is_youtube:
                    file_data_dict = {'file_data': {'file_uri': item.url, 'mime_type': item.media_type}}
                    if item.vendor_metadata:  # pragma: no branch
                        file_data_dict['video_metadata'] = item.vendor_metadata
                    content.append(file_data_dict)  # type: ignore
                elif isinstance(item, FileUrl):
                    if item.force_download or (
                        # google-gla does not support passing file urls directly, except for youtube videos
                        # (see above) and files uploaded to the file API (which cannot be downloaded anyway)
                        self.system == 'google-gla'
                        and not item.url.startswith(r'https://generativelanguage.googleapis.com/v1beta/files')
                    ):
                        downloaded_item = await download_item(item, data_format='base64')
                        inline_data = {'data': downloaded_item['data'], 'mime_type': downloaded_item['data_type']}
                        content.append({'inline_data': inline_data})  # type: ignore
                    else:
                        content.append(
                            {'file_data': {'file_uri': item.url, 'mime_type': item.media_type}}
                        )  # pragma: lax no cover
                else:
                    assert_never(item)
        return content

    def _map_response_schema(self, o: OutputObjectDefinition) -> dict[str, Any]:
        response_schema = o.json_schema.copy()
        if o.name:
            response_schema['title'] = o.name
        if o.description:
            response_schema['description'] = o.description

        return response_schema


@dataclass
class GeminiStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for the Gemini model."""

    _model_name: GoogleModelName
    _response: AsyncIterator[GenerateContentResponse]
    _timestamp: datetime
    _provider_name: str

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:  # noqa: C901
        async for chunk in self._response:
            self._usage = _metadata_as_usage(chunk)

            assert chunk.candidates is not None
            candidate = chunk.candidates[0]

            if chunk.response_id:  # pragma: no branch
                self.provider_response_id = chunk.response_id

            if raw_finish_reason := candidate.finish_reason:
                self.provider_details = {'finish_reason': raw_finish_reason.value}
                self.finish_reason = _FINISH_REASON_MAP.get(raw_finish_reason)

            if candidate.content is None or candidate.content.parts is None:
                if candidate.finish_reason == 'STOP':  # pragma: no cover
                    # Normal completion - skip this chunk
                    continue
                elif candidate.finish_reason == 'SAFETY':  # pragma: no cover
                    raise UnexpectedModelBehavior('Safety settings triggered', str(chunk))
                else:  # pragma: no cover
                    raise UnexpectedModelBehavior('Content field missing from streaming Gemini response', str(chunk))
            parts = candidate.content.parts or []
            for part in parts:
                if part.thought_signature:
                    signature = base64.b64encode(part.thought_signature).decode('utf-8')
                    yield self._parts_manager.handle_thinking_delta(
                        vendor_part_id='thinking',
                        signature=signature,
                        provider_name=self.provider_name,
                    )

                if part.text is not None:
                    if part.thought:
                        yield self._parts_manager.handle_thinking_delta(vendor_part_id='thinking', content=part.text)
                    else:
                        maybe_event = self._parts_manager.handle_text_delta(vendor_part_id='content', content=part.text)
                        if maybe_event is not None:  # pragma: no branch
                            yield maybe_event
                elif part.function_call:
                    maybe_event = self._parts_manager.handle_tool_call_delta(
                        vendor_part_id=uuid4(),
                        tool_name=part.function_call.name,
                        args=part.function_call.args,
                        tool_call_id=part.function_call.id,
                    )
                    if maybe_event is not None:  # pragma: no branch
                        yield maybe_event
                elif part.executable_code is not None:
                    pass
                elif part.code_execution_result is not None:
                    pass
                else:
                    assert part.function_response is not None, f'Unexpected part: {part}'  # pragma: no cover

    @property
    def model_name(self) -> GoogleModelName:
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


def _content_model_response(m: ModelResponse, provider_name: str) -> ContentDict:
    parts: list[PartDict] = []
    thought_signature: bytes | None = None
    for item in m.parts:
        part: PartDict = {}
        if thought_signature:
            part['thought_signature'] = thought_signature
            thought_signature = None

        if isinstance(item, ToolCallPart):
            function_call = FunctionCallDict(name=item.tool_name, args=item.args_as_dict(), id=item.tool_call_id)
            part['function_call'] = function_call
        elif isinstance(item, TextPart):
            part['text'] = item.content
        elif isinstance(item, ThinkingPart):
            if item.provider_name == provider_name and item.signature:
                # The thought signature is to be included on the _next_ part, not the thought part itself
                thought_signature = base64.b64decode(item.signature)

            if item.content:
                part['text'] = item.content
                part['thought'] = True
        elif isinstance(item, BuiltinToolCallPart):
            if item.provider_name == provider_name:
                if item.tool_name == 'code_execution':  # pragma: no branch
                    part['executable_code'] = cast(ExecutableCodeDict, item.args)
        elif isinstance(item, BuiltinToolReturnPart):
            if item.provider_name == provider_name:
                if item.tool_name == 'code_execution':  # pragma: no branch
                    part['code_execution_result'] = item.content
        else:
            assert_never(item)

        if part:
            parts.append(part)
    return ContentDict(role='model', parts=parts)


def _process_response_from_parts(
    parts: list[Part],
    model_name: GoogleModelName,
    provider_name: str,
    usage: usage.RequestUsage,
    vendor_id: str | None,
    vendor_details: dict[str, Any] | None = None,
    finish_reason: FinishReason | None = None,
) -> ModelResponse:
    items: list[ModelResponsePart] = []
    item: ModelResponsePart | None = None
    for part in parts:
        if part.thought_signature:
            signature = base64.b64encode(part.thought_signature).decode('utf-8')
            if not isinstance(item, ThinkingPart):
                item = ThinkingPart(content='')
                items.append(item)
            item.signature = signature
            item.provider_name = provider_name

        if part.executable_code is not None:
            item = BuiltinToolCallPart(
                provider_name=provider_name, args=part.executable_code.model_dump(), tool_name='code_execution'
            )
        elif part.code_execution_result is not None:
            item = BuiltinToolReturnPart(
                provider_name=provider_name,
                tool_name='code_execution',
                content=part.code_execution_result,
                tool_call_id='not_provided',
            )
        elif part.text is not None:
            if part.thought:
                item = ThinkingPart(content=part.text)
            else:
                item = TextPart(content=part.text)
        elif part.function_call:
            assert part.function_call.name is not None
            item = ToolCallPart(tool_name=part.function_call.name, args=part.function_call.args)
            if part.function_call.id is not None:
                item.tool_call_id = part.function_call.id  # pragma: no cover
        else:  # pragma: no cover
            raise UnexpectedModelBehavior(
                f'Unsupported response from Gemini, expected all parts to be function calls, text, or thoughts, got: {part!r}'
            )

        items.append(item)
    return ModelResponse(
        parts=items,
        model_name=model_name,
        usage=usage,
        provider_response_id=vendor_id,
        provider_details=vendor_details,
        provider_name=provider_name,
        finish_reason=finish_reason,
    )


def _function_declaration_from_tool(tool: ToolDefinition) -> FunctionDeclarationDict:
    json_schema = tool.parameters_json_schema
    f = FunctionDeclarationDict(
        name=tool.name,
        description=tool.description or '',
        parameters=json_schema,  # type: ignore
    )
    return f


def _tool_config(function_names: list[str]) -> ToolConfigDict:
    mode = FunctionCallingConfigMode.ANY
    function_calling_config = FunctionCallingConfigDict(mode=mode, allowed_function_names=function_names)
    return ToolConfigDict(function_calling_config=function_calling_config)


def _metadata_as_usage(response: GenerateContentResponse) -> usage.RequestUsage:
    metadata = response.usage_metadata
    if metadata is None:
        return usage.RequestUsage()
    details: dict[str, int] = {}
    if cached_content_token_count := metadata.cached_content_token_count:
        details['cached_content_tokens'] = cached_content_token_count

    if thoughts_token_count := (metadata.thoughts_token_count or 0):
        details['thoughts_tokens'] = thoughts_token_count

    if tool_use_prompt_token_count := metadata.tool_use_prompt_token_count:
        details['tool_use_prompt_tokens'] = tool_use_prompt_token_count

    input_audio_tokens = 0
    output_audio_tokens = 0
    cache_audio_read_tokens = 0
    for prefix, metadata_details in [
        ('prompt', metadata.prompt_tokens_details),
        ('cache', metadata.cache_tokens_details),
        ('candidates', metadata.candidates_tokens_details),
        ('tool_use_prompt', metadata.tool_use_prompt_tokens_details),
    ]:
        assert getattr(metadata, f'{prefix}_tokens_details') is metadata_details
        if not metadata_details:
            continue
        for detail in metadata_details:
            if not detail.modality or not detail.token_count:  # pragma: no cover
                continue
            details[f'{detail.modality.lower()}_{prefix}_tokens'] = detail.token_count
            if detail.modality != 'AUDIO':
                continue
            if metadata_details is metadata.prompt_tokens_details:
                input_audio_tokens = detail.token_count
            elif metadata_details is metadata.candidates_tokens_details:
                output_audio_tokens = detail.token_count
            elif metadata_details is metadata.cache_tokens_details:  # pragma: no branch
                cache_audio_read_tokens = detail.token_count

    return usage.RequestUsage(
        input_tokens=metadata.prompt_token_count or 0,
        output_tokens=(metadata.candidates_token_count or 0) + thoughts_token_count,
        cache_read_tokens=cached_content_token_count or 0,
        input_audio_tokens=input_audio_tokens,
        output_audio_tokens=output_audio_tokens,
        cache_audio_read_tokens=cache_audio_read_tokens,
        details=details,
    )
