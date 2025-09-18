from __future__ import annotations as _annotations

import datetime
import os
import re
from typing import Any

import pytest
from httpx import Timeout
from inline_snapshot import Is, snapshot
from pydantic import BaseModel
from typing_extensions import TypedDict

from pydantic_ai import UsageLimitExceeded
from pydantic_ai.agent import Agent
from pydantic_ai.builtin_tools import CodeExecutionTool, UrlContextTool, WebSearchTool
from pydantic_ai.exceptions import ModelRetry, UnexpectedModelBehavior, UserError
from pydantic_ai.messages import (
    AudioUrl,
    BinaryContent,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    DocumentUrl,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ImageUrl,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    VideoUrl,
)
from pydantic_ai.output import NativeOutput, PromptedOutput, TextOutput, ToolOutput
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage, RunUsage, UsageLimits

from ..conftest import IsDatetime, IsInstance, IsStr, try_import
from ..parts_from_messages import part_types_from_messages

with try_import() as imports_successful:
    from google.genai.types import (
        CodeExecutionResult,
        GenerateContentResponse,
        GenerateContentResponseUsageMetadata,
        HarmBlockThreshold,
        HarmCategory,
        Language,
        MediaModality,
        ModalityTokenCount,
        Outcome,
    )

    from pydantic_ai.models.google import GoogleModel, GoogleModelSettings, _metadata_as_usage  # type: ignore
    from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
    from pydantic_ai.providers.google import GoogleProvider
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='google-genai not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


@pytest.fixture()
def google_provider(gemini_api_key: str) -> GoogleProvider:
    return GoogleProvider(api_key=gemini_api_key)


async def test_google_model(allow_model_requests: None, google_provider: GoogleProvider):
    model = GoogleModel('gemini-1.5-flash', provider=google_provider)
    assert model.base_url == 'https://generativelanguage.googleapis.com/'
    assert model.system == 'google-gla'
    agent = Agent(model=model, system_prompt='You are a chatbot.')

    result = await agent.run('Hello!')
    assert result.output == snapshot('Hello there! How can I help you today?\n')
    assert result.usage() == snapshot(
        RunUsage(
            requests=1,
            input_tokens=7,
            output_tokens=11,
            details={'text_prompt_tokens': 7, 'text_candidates_tokens': 11},
        )
    )
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(
                        content='You are a chatbot.',
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content='Hello!',
                        timestamp=IsDatetime(),
                    ),
                ]
            ),
            ModelResponse(
                parts=[TextPart(content='Hello there! How can I help you today?\n')],
                usage=RequestUsage(
                    input_tokens=7, output_tokens=11, details={'text_candidates_tokens': 11, 'text_prompt_tokens': 7}
                ),
                model_name='gemini-1.5-flash',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
            ),
        ]
    )


async def test_google_model_structured_output(allow_model_requests: None, google_provider: GoogleProvider):
    model = GoogleModel('gemini-1.5-flash', provider=google_provider)
    agent = Agent(model=model, system_prompt='You are a helpful chatbot.', retries=5)

    class Response(TypedDict):
        temperature: str
        date: datetime.date
        city: str

    @agent.tool_plain
    async def temperature(city: str, date: datetime.date) -> str:
        """Get the temperature in a city on a specific date.

        Args:
            city: The city name.
            date: The date.

        Returns:
            The temperature in degrees Celsius.
        """
        return '30°C'

    result = await agent.run('What was the temperature in London 1st January 2022?', output_type=Response)
    assert result.output == snapshot({'temperature': '30°C', 'date': datetime.date(2022, 1, 1), 'city': 'London'})
    assert result.usage() == snapshot(
        RunUsage(
            requests=2,
            input_tokens=224,
            output_tokens=35,
            tool_calls=1,
            details={'text_prompt_tokens': 224, 'text_candidates_tokens': 35},
        )
    )
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(
                        content='You are a helpful chatbot.',
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content='What was the temperature in London 1st January 2022?',
                        timestamp=IsDatetime(),
                    ),
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='temperature', args={'date': '2022-01-01', 'city': 'London'}, tool_call_id=IsStr()
                    )
                ],
                usage=RequestUsage(
                    input_tokens=101,
                    output_tokens=14,
                    details={'text_candidates_tokens': 14, 'text_prompt_tokens': 101},
                ),
                model_name='gemini-1.5-flash',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='temperature', content='30°C', tool_call_id=IsStr(), timestamp=IsDatetime()
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args={'temperature': '30°C', 'date': '2022-01-01', 'city': 'London'},
                        tool_call_id=IsStr(),
                    )
                ],
                usage=RequestUsage(
                    input_tokens=123,
                    output_tokens=21,
                    details={'text_candidates_tokens': 21, 'text_prompt_tokens': 123},
                ),
                model_name='gemini-1.5-flash',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ]
            ),
        ]
    )


async def test_google_model_stream(allow_model_requests: None, google_provider: GoogleProvider):
    model = GoogleModel('gemini-2.0-flash-exp', provider=google_provider)
    agent = Agent(model=model, system_prompt='You are a helpful chatbot.', model_settings={'temperature': 0.0})
    async with agent.run_stream('What is the capital of France?') as result:
        data = await result.get_output()
        async for response, is_last in result.stream_responses(debounce_by=None):
            if is_last:
                assert response == snapshot(
                    ModelResponse(
                        parts=[TextPart(content='The capital of France is Paris.\n')],
                        usage=RequestUsage(
                            input_tokens=13,
                            output_tokens=8,
                            details={'text_prompt_tokens': 13, 'text_candidates_tokens': 8},
                        ),
                        model_name='gemini-2.0-flash-exp',
                        timestamp=IsDatetime(),
                        provider_name='google-gla',
                        provider_details={'finish_reason': 'STOP'},
                        provider_response_id='w1peaMz6INOvnvgPgYfPiQY',
                        finish_reason='stop',
                    )
                )
    assert data == snapshot('The capital of France is Paris.\n')


async def test_google_model_builtin_code_execution_stream(
    allow_model_requests: None,
    google_provider: GoogleProvider,
):
    """Test Gemini streaming only code execution result or executable_code."""
    model = GoogleModel('gemini-2.0-flash', provider=google_provider)
    agent = Agent(
        model=model,
        system_prompt='Be concise and always use Python to do calculations no matter how small.',
        builtin_tools=[CodeExecutionTool()],
    )
    event_parts: list[str] = []
    async with agent.run_stream(user_prompt='what is 65465-6544 * 65464-6+1.02255') as result:
        async for chunk in result.stream_text():
            event_parts.append(chunk)

    assert event_parts == snapshot(['The answer is -428330955.97745.\n'])


async def test_google_model_retry(allow_model_requests: None, google_provider: GoogleProvider):
    model = GoogleModel('gemini-2.5-pro-preview-03-25', provider=google_provider)
    agent = Agent(
        model=model, system_prompt='You are a helpful chatbot.', model_settings={'temperature': 0.0}, retries=2
    )

    @agent.tool_plain
    async def get_capital(country: str) -> str:
        """Get the capital of a country.

        Args:
            country: The country name.
        """
        raise ModelRetry('The country is not supported.')

    result = await agent.run('What is the capital of France?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='You are a helpful chatbot.', timestamp=IsDatetime()),
                    UserPromptPart(content='What is the capital of France?', timestamp=IsDatetime()),
                ]
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='get_capital', args={'country': 'France'}, tool_call_id=IsStr())],
                usage=RequestUsage(
                    input_tokens=57, output_tokens=170, details={'thoughts_tokens': 155, 'text_prompt_tokens': 57}
                ),
                model_name='models/gemini-2.5-pro',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='The country is not supported.',
                        tool_name='get_capital',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='I am sorry, I cannot fulfill this request. The country "France" is not supported by my system.'
                    )
                ],
                usage=RequestUsage(
                    input_tokens=104, output_tokens=200, details={'thoughts_tokens': 178, 'text_prompt_tokens': 104}
                ),
                model_name='models/gemini-2.5-pro',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
            ),
        ]
    )


async def test_google_model_max_tokens(allow_model_requests: None, google_provider: GoogleProvider):
    model = GoogleModel('gemini-1.5-flash', provider=google_provider)
    agent = Agent(model=model, system_prompt='You are a helpful chatbot.', model_settings={'max_tokens': 5})
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('The capital of France is')


async def test_google_model_top_p(allow_model_requests: None, google_provider: GoogleProvider):
    model = GoogleModel('gemini-1.5-flash', provider=google_provider)
    agent = Agent(model=model, system_prompt='You are a helpful chatbot.', model_settings={'top_p': 0.5})
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('The capital of France is Paris.\n')


async def test_google_model_thinking_config(allow_model_requests: None, google_provider: GoogleProvider):
    model = GoogleModel('gemini-2.5-pro-preview-03-25', provider=google_provider)
    settings = GoogleModelSettings(google_thinking_config={'include_thoughts': False})
    agent = Agent(model=model, system_prompt='You are a helpful chatbot.', model_settings=settings)
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('The capital of France is **Paris**.')


async def test_google_model_gla_labels_raises_value_error(allow_model_requests: None, google_provider: GoogleProvider):
    model = GoogleModel('gemini-2.0-flash', provider=google_provider)
    settings = GoogleModelSettings(google_labels={'environment': 'test', 'team': 'analytics'})
    agent = Agent(model=model, system_prompt='You are a helpful chatbot.', model_settings=settings)

    # Raises before any request is made.
    with pytest.raises(ValueError, match='labels parameter is not supported in Gemini API.'):
        await agent.run('What is the capital of France?')


async def test_google_model_vertex_provider(
    allow_model_requests: None, vertex_provider: GoogleProvider
):  # pragma: lax no cover
    model = GoogleModel('gemini-2.0-flash', provider=vertex_provider)
    agent = Agent(model=model, system_prompt='You are a helpful chatbot.')
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('The capital of France is Paris.\n')


async def test_google_model_vertex_labels(
    allow_model_requests: None, vertex_provider: GoogleProvider
):  # pragma: lax no cover
    model = GoogleModel('gemini-2.0-flash', provider=vertex_provider)
    settings = GoogleModelSettings(google_labels={'environment': 'test', 'team': 'analytics'})
    agent = Agent(model=model, system_prompt='You are a helpful chatbot.', model_settings=settings)
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('The capital of France is Paris.\n')


async def test_google_model_iter_stream(allow_model_requests: None, google_provider: GoogleProvider):
    model = GoogleModel('gemini-2.0-flash', provider=google_provider)
    agent = Agent(model=model, system_prompt='You are a helpful chatbot.')

    @agent.tool_plain
    async def get_capital(country: str) -> str:
        """Get the capital of a country.

        Args:
            country: The country name.
        """
        return 'Paris'  # pragma: lax no cover

    @agent.tool_plain
    async def get_temperature(city: str) -> str:
        """Get the temperature in a city.

        Args:
            city: The city name.
        """
        return '30°C'

    event_parts: list[Any] = []
    async with agent.iter(user_prompt='What is the temperature of the capital of France?') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    assert event_parts == snapshot(
        [
            PartStartEvent(
                index=0,
                part=ToolCallPart(tool_name='get_capital', args={'country': 'France'}, tool_call_id=IsStr()),
            ),
            IsInstance(FunctionToolCallEvent),
            FunctionToolResultEvent(
                result=ToolReturnPart(
                    tool_name='get_capital', content='Paris', tool_call_id=IsStr(), timestamp=IsDatetime()
                )
            ),
            PartStartEvent(
                index=0,
                part=ToolCallPart(tool_name='get_temperature', args={'city': 'Paris'}, tool_call_id=IsStr()),
            ),
            IsInstance(FunctionToolCallEvent),
            FunctionToolResultEvent(
                result=ToolReturnPart(
                    tool_name='get_temperature', content='30°C', tool_call_id=IsStr(), timestamp=IsDatetime()
                )
            ),
            PartStartEvent(index=0, part=TextPart(content='The temperature in Paris')),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' is 30°C.\n')),
        ]
    )


async def test_google_model_image_as_binary_content_input(
    allow_model_requests: None, image_content: BinaryContent, google_provider: GoogleProvider
):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)
    agent = Agent(m, system_prompt='You are a helpful chatbot.')

    result = await agent.run(['What fruit is in the image?', image_content])
    assert result.output == snapshot('The fruit in the image is a kiwi.')


async def test_google_model_video_as_binary_content_input(
    allow_model_requests: None, video_content: BinaryContent, google_provider: GoogleProvider
):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)
    agent = Agent(m, system_prompt='You are a helpful chatbot.')

    result = await agent.run(['Explain me this video', video_content])
    assert result.output == snapshot("""\
Okay, I can describe what is visible in the image.

The image shows a camera setup in an outdoor setting. The camera is mounted on a tripod and has an external monitor attached to it. The monitor is displaying a scene that appears to be a desert landscape with rocky formations and mountains in the background. The foreground and background of the overall image, outside of the camera monitor, is also a blurry, desert landscape. The colors in the background are warm and suggest either sunrise, sunset, or reflected light off the rock formations.

It looks like someone is either reviewing footage on the monitor, or using it as an aid for framing the shot.\
""")


async def test_google_model_video_as_binary_content_input_with_vendor_metadata(
    allow_model_requests: None, video_content: BinaryContent, google_provider: GoogleProvider
):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)
    agent = Agent(m, system_prompt='You are a helpful chatbot.')
    video_content.vendor_metadata = {'start_offset': '2s', 'end_offset': '10s'}

    result = await agent.run(['Explain me this video', video_content])
    assert result.output == snapshot("""\
Okay, I can describe what is visible in the image.

The image shows a camera setup in an outdoor setting. The camera is mounted on a tripod and has an external monitor attached to it. The monitor is displaying a scene that appears to be a desert landscape with rocky formations and mountains in the background. The foreground and background of the overall image, outside of the camera monitor, is also a blurry, desert landscape. The colors in the background are warm and suggest either sunrise, sunset, or reflected light off the rock formations.

It looks like someone is either reviewing footage on the monitor, or using it as an aid for framing the shot.\
""")


async def test_google_model_image_url_input(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)
    agent = Agent(m, system_prompt='You are a helpful chatbot.')

    result = await agent.run(
        [
            'What is this vegetable?',
            ImageUrl(url='https://t3.ftcdn.net/jpg/00/85/79/92/360_F_85799278_0BBGV9OAdQDTLnKwAPBCcg1J7QtiieJY.jpg'),
        ]
    )
    assert result.output == snapshot('That is a potato.\n')


async def test_google_model_video_url_input(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)
    agent = Agent(m, system_prompt='You are a helpful chatbot.')

    result = await agent.run(
        [
            'Explain me this video',
            VideoUrl(url='https://github.com/pydantic/pydantic-ai/raw/refs/heads/main/tests/assets/small_video.mp4'),
        ]
    )
    assert result.output == snapshot("""\
Okay, based on the image, here's what I can infer:

*   **A camera monitor is mounted on top of a camera.**
*   **The monitor's screen is on, displaying a view of the rocky mountains.**
*   **This setting suggests a professional video shoot.**

If you'd like a more detailed explanation, please provide additional information about the video.\
""")


async def test_google_model_youtube_video_url_input_with_vendor_metadata(
    allow_model_requests: None, google_provider: GoogleProvider
):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)
    agent = Agent(m, system_prompt='You are a helpful chatbot.')

    result = await agent.run(
        [
            'Explain me this video',
            VideoUrl(
                url='https://youtu.be/lCdaVNyHtjU',
                vendor_metadata={'fps': 0.2},
            ),
        ]
    )
    assert result.output == snapshot("""\
Okay, based on the image, here's what I can infer:

*   **A camera monitor is mounted on top of a camera.**
*   **The monitor's screen is on, displaying a view of the rocky mountains.**
*   **This setting suggests a professional video shoot.**

If you'd like a more detailed explanation, please provide additional information about the video.\
""")


async def test_google_model_document_url_input(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)
    agent = Agent(m, system_prompt='You are a helpful chatbot.')

    document_url = DocumentUrl(url='https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf')

    result = await agent.run(['What is the main content on this document?', document_url])
    assert result.output == snapshot('The document appears to be a "Dummy PDF file".\n')


async def test_google_model_text_document_url_input(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)
    agent = Agent(m, system_prompt='You are a helpful chatbot.')

    text_document_url = DocumentUrl(url='https://example-files.online-convert.com/document/txt/example.txt')

    result = await agent.run(['What is the main content on this document?', text_document_url])
    assert result.output == snapshot(
        'The main content of the document is an example of a TXT file, specifically providing information about the placeholder names "John Doe" (and related variations) used for unidentified or anonymous individuals, particularly in legal contexts in the United States and Canada. It also explains alternative names used in other countries and some additional context and examples of when "John Doe" might be used. The document also includes attribution to Wikipedia for the example content and a link to the license under which it is shared.\n'
    )


async def test_google_model_text_as_binary_content_input(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)
    agent = Agent(m, system_prompt='You are a helpful chatbot.')

    text_content = BinaryContent(data=b'This is a test document.', media_type='text/plain')

    result = await agent.run(['What is the main content on this document?', text_content])
    assert result.output == snapshot('The main content of the document is that it is a test document.\n')


async def test_google_model_instructions(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)

    def instructions() -> str:
        return 'You are a helpful assistant.'

    agent = Agent(m, instructions=instructions)

    result = await agent.run('What is the capital of France?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is the capital of France?', timestamp=IsDatetime())],
                instructions='You are a helpful assistant.',
            ),
            ModelResponse(
                parts=[TextPart(content='The capital of France is Paris.\n')],
                usage=RequestUsage(
                    input_tokens=13, output_tokens=8, details={'text_candidates_tokens': 8, 'text_prompt_tokens': 13}
                ),
                model_name='gemini-2.0-flash',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
            ),
        ]
    )


async def test_google_model_multiple_documents_in_history(
    allow_model_requests: None, google_provider: GoogleProvider, document_content: BinaryContent
):
    m = GoogleModel(model_name='gemini-2.0-flash', provider=google_provider)
    agent = Agent(model=m)

    result = await agent.run(
        'What is in the documents?',
        message_history=[
            ModelRequest(parts=[UserPromptPart(content=['Here is a PDF document: ', document_content])]),
            ModelResponse(parts=[TextPart(content='foo bar')]),
            ModelRequest(parts=[UserPromptPart(content=['Here is another PDF document: ', document_content])]),
            ModelResponse(parts=[TextPart(content='foo bar 2')]),
        ],
    )

    assert result.output == snapshot('Both documents contain the text "Dummy PDF file" at the top of the page.')


async def test_google_model_safety_settings(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-1.5-flash', provider=google_provider)
    settings = GoogleModelSettings(
        google_safety_settings=[
            {
                'category': HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                'threshold': HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            }
        ]
    )
    agent = Agent(m, instructions='You hate the world!', model_settings=settings)

    with pytest.raises(UnexpectedModelBehavior, match='Safety settings triggered'):
        await agent.run('Tell me a joke about a Brazilians.')


async def test_google_model_web_search_tool(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)
    agent = Agent(m, system_prompt='You are a helpful chatbot.', builtin_tools=[WebSearchTool()])

    result = await agent.run('What day is today in Utrecht?')
    assert result.output == snapshot('Today is Wednesday, May 28, 2025, in Utrecht.\n')


async def test_google_model_url_context_tool(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.5-flash', provider=google_provider)
    agent = Agent(m, system_prompt='You are a helpful chatbot.', builtin_tools=[UrlContextTool()])

    result = await agent.run(
        'What is the first sentence on the page https://ai.pydantic.dev? Reply with only the sentence.'
    )

    assert result.output == snapshot(
        'Pydantic AI is a Python agent framework designed to make it less painful to build production grade applications with Generative AI.'
    )


async def test_google_model_code_execution_tool(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)
    agent = Agent(m, system_prompt='You are a helpful chatbot.', builtin_tools=[CodeExecutionTool()])

    result = await agent.run('What day is today in Utrecht?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='You are a helpful chatbot.', timestamp=IsDatetime()),
                    UserPromptPart(content='What day is today in Utrecht?', timestamp=IsDatetime()),
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="""\
To determine the day of the week in Utrecht, I need to know the current date. I will use the python tool to get the current date and time, and then extract the day of the week.

"""
                    ),
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args={
                            'code': """\
import datetime

now = datetime.datetime.now()
day_of_week = now.strftime("%A")
print(f'{day_of_week=}')
""",
                            'language': Language.PYTHON,
                        },
                        tool_call_id=IsStr(),
                        provider_name='google-gla',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content=CodeExecutionResult(outcome=Outcome.OUTCOME_OK, output="day_of_week='Thursday'\n"),
                        tool_call_id='not_provided',
                        timestamp=IsDatetime(),
                        provider_name='google-gla',
                    ),
                    TextPart(content='Today is Thursday in Utrecht.\n'),
                ],
                usage=RequestUsage(
                    input_tokens=13,
                    output_tokens=95,
                    details={
                        'tool_use_prompt_tokens': 101,
                        'text_candidates_tokens': 95,
                        'text_prompt_tokens': 13,
                        'text_tool_use_prompt_tokens': 101,
                    },
                ),
                model_name='gemini-2.0-flash',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
            ),
        ]
    )

    result = await agent.run('What day is tomorrow?', message_history=result.all_messages())
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='You are a helpful chatbot.', timestamp=IsDatetime()),
                    UserPromptPart(content='What day is today in Utrecht?', timestamp=IsDatetime()),
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="""\
To determine the day of the week in Utrecht, I need to know the current date. I will use the python tool to get the current date and time, and then extract the day of the week.

"""
                    ),
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args={
                            'code': """\
import datetime

now = datetime.datetime.now()
day_of_week = now.strftime("%A")
print(f'{day_of_week=}')
""",
                            'language': Language.PYTHON,
                        },
                        tool_call_id=IsStr(),
                        provider_name='google-gla',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content=CodeExecutionResult(outcome=Outcome.OUTCOME_OK, output="day_of_week='Thursday'\n"),
                        tool_call_id='not_provided',
                        timestamp=IsDatetime(),
                        provider_name='google-gla',
                    ),
                    TextPart(content='Today is Thursday in Utrecht.\n'),
                ],
                usage=RequestUsage(
                    input_tokens=13,
                    output_tokens=95,
                    details={
                        'tool_use_prompt_tokens': 101,
                        'text_candidates_tokens': 95,
                        'text_prompt_tokens': 13,
                        'text_tool_use_prompt_tokens': 101,
                    },
                ),
                model_name='gemini-2.0-flash',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
            ),
            ModelRequest(parts=[UserPromptPart(content='What day is tomorrow?', timestamp=IsDatetime())]),
            ModelResponse(
                parts=[
                    TextPart(
                        content="""\
To determine what day is tomorrow, I'll use the python tool to calculate tomorrow's date and then find the corresponding day of the week.

"""
                    ),
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args={
                            'code': """\
import datetime

today = datetime.date.today()
tomorrow = today + datetime.timedelta(days=1)
day_of_week = tomorrow.strftime("%A")
print(f'{day_of_week=}')
""",
                            'language': Language.PYTHON,
                        },
                        tool_call_id=IsStr(),
                        provider_name='google-gla',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content=CodeExecutionResult(outcome=Outcome.OUTCOME_OK, output="day_of_week='Friday'\n"),
                        tool_call_id='not_provided',
                        timestamp=IsDatetime(),
                        provider_name='google-gla',
                    ),
                    TextPart(content='Tomorrow is Friday.\n'),
                ],
                usage=RequestUsage(
                    input_tokens=113,
                    output_tokens=95,
                    details={
                        'tool_use_prompt_tokens': 203,
                        'text_candidates_tokens': 95,
                        'text_prompt_tokens': 113,
                        'text_tool_use_prompt_tokens': 203,
                    },
                ),
                model_name='gemini-2.0-flash',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
            ),
        ]
    )


async def test_google_model_server_tool_receive_history_from_another_provider(
    allow_model_requests: None, anthropic_api_key: str, gemini_api_key: str
):
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    anthropic_model = AnthropicModel('claude-sonnet-4-0', provider=AnthropicProvider(api_key=anthropic_api_key))
    google_model = GoogleModel('gemini-2.0-flash', provider=GoogleProvider(api_key=gemini_api_key))
    agent = Agent(builtin_tools=[CodeExecutionTool()])

    result = await agent.run('How much is 3 * 12390?', model=anthropic_model)
    assert part_types_from_messages(result.all_messages()) == snapshot(
        [[UserPromptPart], [TextPart, BuiltinToolCallPart, BuiltinToolReturnPart, TextPart]]
    )

    result = await agent.run('Multiplied by 12390', model=google_model, message_history=result.all_messages())
    assert part_types_from_messages(result.all_messages()) == snapshot(
        [
            [UserPromptPart],
            [TextPart, BuiltinToolCallPart, BuiltinToolReturnPart, TextPart],
            [UserPromptPart],
            [TextPart, BuiltinToolCallPart, BuiltinToolReturnPart, TextPart],
        ]
    )


async def test_google_model_receive_web_search_history_from_another_provider(
    allow_model_requests: None, anthropic_api_key: str, gemini_api_key: str
):
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    anthropic_model = AnthropicModel('claude-sonnet-4-0', provider=AnthropicProvider(api_key=anthropic_api_key))
    anthropic_agent = Agent(model=anthropic_model, builtin_tools=[WebSearchTool()])

    result = await anthropic_agent.run('What are the latest news in the Netherlands?')
    assert part_types_from_messages(result.all_messages()) == snapshot(
        [
            [UserPromptPart],
            [
                BuiltinToolCallPart,
                BuiltinToolReturnPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
            ],
        ]
    )

    google_model = GoogleModel('gemini-2.0-flash', provider=GoogleProvider(api_key=gemini_api_key))
    google_agent = Agent(model=google_model)
    result = await google_agent.run('What day is tomorrow?', message_history=result.all_messages())
    assert part_types_from_messages(result.all_messages()) == snapshot(
        [
            [UserPromptPart],
            [
                BuiltinToolCallPart,
                BuiltinToolReturnPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
            ],
            [UserPromptPart],
            [TextPart],
        ]
    )


async def test_google_model_empty_user_prompt(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-1.5-flash', provider=google_provider)
    agent = Agent(m, instructions='You are a helpful assistant.')

    result = await agent.run()
    assert result.output == snapshot("I'm ready to assist you.  Please tell me what you need.\n")


async def test_google_model_empty_assistant_response(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-1.5-flash', provider=google_provider)
    agent = Agent(m)

    result = await agent.run(
        'Empty?',
        message_history=[
            ModelRequest(parts=[UserPromptPart(content='Hi')]),
            ModelResponse(parts=[TextPart(content='')]),
        ],
    )

    assert result.output == snapshot('Yes, your previous message was empty.  Is there anything I can help you with?\n')


async def test_google_model_thinking_part(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.5-pro', provider=google_provider)
    settings = GoogleModelSettings(google_thinking_config={'include_thoughts': True})
    agent = Agent(m, system_prompt='You are a helpful assistant.', model_settings=settings)

    # Google only emits thought signatures when there are tools: https://ai.google.dev/gemini-api/docs/thinking#signatures
    @agent.tool_plain
    def dummy() -> None: ...  # pragma: no cover

    result = await agent.run('How do I cross the street?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(
                        content='You are a helpful assistant.',
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content='How do I cross the street?',
                        timestamp=IsDatetime(),
                    ),
                ]
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        signature=IsStr(),
                        provider_name='google-gla',
                    ),
                    TextPart(content=IsStr()),
                ],
                usage=RequestUsage(
                    input_tokens=34, output_tokens=1246, details={'thoughts_tokens': 817, 'text_prompt_tokens': 34}
                ),
                model_name='gemini-2.5-pro',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id='sebBaN7rGrSsqtsPhf3J0Q4',
                finish_reason='stop',
            ),
        ]
    )

    result = await agent.run(
        'Considering the way to cross the street, analogously, how do I cross the river?',
        message_history=result.all_messages(),
    )
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Considering the way to cross the street, analogously, how do I cross the river?',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        signature=IsStr(),
                        provider_name='google-gla',
                    ),
                    TextPart(content=IsStr()),
                ],
                usage=RequestUsage(
                    input_tokens=978, output_tokens=2417, details={'thoughts_tokens': 1476, 'text_prompt_tokens': 978}
                ),
                model_name='gemini-2.5-pro',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id='zObBaKreOqSIqtsP7uur4A0',
                finish_reason='stop',
            ),
        ]
    )


async def test_google_model_thinking_part_from_other_model(
    allow_model_requests: None, google_provider: GoogleProvider, openai_api_key: str
):
    provider = OpenAIProvider(api_key=openai_api_key)
    m = OpenAIResponsesModel('gpt-5', provider=provider)
    settings = OpenAIResponsesModelSettings(openai_reasoning_effort='high', openai_reasoning_summary='detailed')
    agent = Agent(m, system_prompt='You are a helpful assistant.', model_settings=settings)

    # Google only emits thought signatures when there are tools: https://ai.google.dev/gemini-api/docs/thinking#signatures
    @agent.tool_plain
    def dummy() -> None: ...  # pragma: no cover

    result = await agent.run('How do I cross the street?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(
                        content='You are a helpful assistant.',
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content='How do I cross the street?',
                        timestamp=IsDatetime(),
                    ),
                ]
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c1fb6c15c48196b964881266a03c8e0c14a8a9087e8689',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c1fb6c15c48196b964881266a03c8e0c14a8a9087e8689',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c1fb6c15c48196b964881266a03c8e0c14a8a9087e8689',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c1fb6c15c48196b964881266a03c8e0c14a8a9087e8689',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c1fb6c15c48196b964881266a03c8e0c14a8a9087e8689',
                    ),
                    TextPart(content=IsStr(), id='msg_68c1fb814fdc8196aec1a46164ddf7680c14a8a9087e8689'),
                ],
                usage=RequestUsage(input_tokens=45, output_tokens=1719, details={'reasoning_tokens': 1408}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68c1fb6b6a248196a6216e80fc2ace380c14a8a9087e8689',
                finish_reason='stop',
            ),
        ]
    )

    result = await agent.run(
        'Considering the way to cross the street, analogously, how do I cross the river?',
        model=GoogleModel(
            'gemini-2.5-pro',
            provider=google_provider,
            settings=GoogleModelSettings(google_thinking_config={'include_thoughts': True}),
        ),
        message_history=result.all_messages(),
    )
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Considering the way to cross the street, analogously, how do I cross the river?',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        signature=IsStr(),
                        provider_name='google-gla',
                    ),
                    TextPart(content=IsStr()),
                ],
                usage=RequestUsage(
                    input_tokens=1106, output_tokens=1867, details={'thoughts_tokens': 1089, 'text_prompt_tokens': 1106}
                ),
                model_name='gemini-2.5-pro',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id='mPvBaJmNOMywqtsPsb_l2A4',
                finish_reason='stop',
            ),
        ]
    )


async def test_google_model_thinking_part_iter(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.5-pro', provider=google_provider)
    settings = GoogleModelSettings(google_thinking_config={'include_thoughts': True})
    agent = Agent(m, system_prompt='You are a helpful assistant.', model_settings=settings)

    # Google only emits thought signatures when there are tools: https://ai.google.dev/gemini-api/docs/thinking#signatures
    @agent.tool_plain
    def dummy() -> None: ...  # pragma: no cover

    event_parts: list[Any] = []
    async with agent.iter(user_prompt='How do I cross the street?') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    assert agent_run.result is not None
    assert agent_run.result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(
                        content='You are a helpful assistant.',
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content='How do I cross the street?',
                        timestamp=IsDatetime(),
                    ),
                ]
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        signature=IsStr(),
                        provider_name='google-gla',
                    ),
                    TextPart(content=IsStr()),
                ],
                usage=RequestUsage(
                    input_tokens=34, output_tokens=1256, details={'thoughts_tokens': 787, 'text_prompt_tokens': 34}
                ),
                model_name='gemini-2.5-pro',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id='beHBaJfEMIi-qtsP3769-Q8',
                finish_reason='stop',
            ),
        ]
    )

    assert event_parts == snapshot(
        [
            PartStartEvent(index=0, part=ThinkingPart(content=IsStr())),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(signature_delta=IsStr(), provider_name='google-gla')),
            PartStartEvent(index=1, part=TextPart(content=IsStr())),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
        ]
    )


@pytest.mark.parametrize(
    'url,expected_output',
    [
        pytest.param(
            AudioUrl(url='https://cdn.openai.com/API/docs/audio/alloy.wav'),
            'The URL discusses the sunrise in the east and sunset in the west, a phenomenon known to humans for millennia.',
            id='AudioUrl',
        ),
        pytest.param(
            DocumentUrl(url='https://storage.googleapis.com/cloud-samples-data/generative-ai/pdf/2403.05530.pdf'),
            "The URL points to a technical report from Google DeepMind introducing Gemini 1.5 Pro, a multimodal AI model designed for understanding and reasoning over extremely large contexts (millions of tokens). It details the model's architecture, training, performance across a range of tasks, and responsible deployment considerations. Key highlights include near-perfect recall on long-context retrieval tasks, state-of-the-art performance in areas like long-document question answering, and surprising new capabilities like in-context learning of new languages.",
            id='DocumentUrl',
        ),
        pytest.param(
            ImageUrl(url='https://upload.wikimedia.org/wikipedia/commons/6/6a/Www.wikipedia_screenshot_%282021%29.png'),
            "The URL's main content is the landing page of Wikipedia, showcasing the available language editions with article counts, a search bar, and links to other Wikimedia projects.",
            id='ImageUrl',
        ),
        pytest.param(
            VideoUrl(url='https://upload.wikimedia.org/wikipedia/commons/8/8f/Panda_at_Smithsonian_zoo.webm'),
            """The main content of the image is a panda eating bamboo in a zoo enclosure. The enclosure is designed to mimic the panda's natural habitat, with rocks, bamboo, and a painted backdrop of mountains. There is also a large, smooth, tan-colored ball-shaped object in the enclosure.""",
            id='VideoUrl',
        ),
        pytest.param(
            VideoUrl(url='https://youtu.be/lCdaVNyHtjU'),
            'The main content of the URL is an analysis of recent 404 HTTP responses. The analysis identifies several patterns including the most common endpoints with 404 errors, request patterns, timeline-related issues, organization/project access, and configuration and authentication. The analysis also provides some recommendations.',
            id='VideoUrl (YouTube)',
        ),
        pytest.param(
            AudioUrl(url='gs://pydantic-ai-dev/openai-alloy.wav'),
            'The content describes the basic concept of the sun rising in the east and setting in the west.',
            id='AudioUrl (gs)',
        ),
        pytest.param(
            DocumentUrl(url='gs://pydantic-ai-dev/Gemini_1_5_Pro_Technical_Report_Arxiv_1805.pdf'),
            "The URL leads to a research paper titled \"Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context\".  \n\nThe paper introduces Gemini 1.5 Pro, a new model in the Gemini family. It's described as a highly compute-efficient multimodal mixture-of-experts model.  A key feature is its ability to recall and reason over fine-grained information from millions of tokens of context, including long documents and hours of video and audio.  The paper presents experimental results showcasing the model's capabilities on long-context retrieval tasks, QA, ASR, and its performance compared to Gemini 1.0 models. It covers the model's architecture, training data, and evaluations on both synthetic and real-world tasks.  A notable highlight is its ability to learn to translate from English to Kalamang, a low-resource language, from just a grammar manual and dictionary provided in context.  The paper also discusses responsible deployment considerations, including impact assessments and mitigation efforts.\n",
            id='DocumentUrl (gs)',
        ),
        pytest.param(
            ImageUrl(url='gs://pydantic-ai-dev/wikipedia_screenshot.png'),
            "The main content of the URL is the Wikipedia homepage, featuring options to access Wikipedia in different languages and information about the number of articles in each language. It also includes links to other Wikimedia projects and information about Wikipedia's host, the Wikimedia Foundation.\n",
            id='ImageUrl (gs)',
        ),
        pytest.param(
            VideoUrl(url='gs://pydantic-ai-dev/grepit-tiny-video.mp4'),
            'The image shows a charming outdoor cafe in a Greek coastal town. The cafe is nestled between traditional whitewashed buildings, with tables and chairs set along a narrow cobblestone pathway. The sea is visible in the distance, adding to the picturesque and relaxing atmosphere.',
            id='VideoUrl (gs)',
        ),
    ],
)
async def test_google_url_input(
    url: AudioUrl | DocumentUrl | ImageUrl | VideoUrl,
    expected_output: str,
    allow_model_requests: None,
    vertex_provider: GoogleProvider,
) -> None:  # pragma: lax no cover
    m = GoogleModel('gemini-2.0-flash', provider=vertex_provider)
    agent = Agent(m)
    result = await agent.run(['What is the main content of this URL?', url])

    assert result.output == snapshot(Is(expected_output))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=['What is the main content of this URL?', Is(url)],
                        timestamp=IsDatetime(),
                    ),
                ]
            ),
            ModelResponse(
                parts=[TextPart(content=Is(expected_output))],
                usage=IsInstance(RequestUsage),
                model_name='gemini-2.0-flash',
                timestamp=IsDatetime(),
                provider_name='google-vertex',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
            ),
        ]
    )


@pytest.mark.skipif(
    not os.getenv('CI', False), reason='Requires properly configured local google vertex config to pass'
)
@pytest.mark.vcr()
async def test_google_url_input_force_download(
    allow_model_requests: None, vertex_provider: GoogleProvider
) -> None:  # pragma: lax no cover
    m = GoogleModel('gemini-2.0-flash', provider=vertex_provider)
    agent = Agent(m)

    video_url = VideoUrl(url='https://data.grepit.app/assets/tiny_video.mp4', force_download=True)
    result = await agent.run(['What is the main content of this URL?', video_url])

    output = 'The image shows a picturesque scene in what appears to be a Greek island town. The focus is on an outdoor dining area with tables and chairs, situated in a narrow alleyway between whitewashed buildings. The ocean is visible at the end of the alley, creating a beautiful and inviting atmosphere.'

    assert result.output == output
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=['What is the main content of this URL?', Is(video_url)],
                        timestamp=IsDatetime(),
                    ),
                ]
            ),
            ModelResponse(
                parts=[TextPart(content=Is(output))],
                usage=IsInstance(RequestUsage),
                model_name='gemini-2.0-flash',
                timestamp=IsDatetime(),
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                provider_name='google-vertex',
                finish_reason='stop',
            ),
        ]
    )


async def test_google_gs_url_force_download_raises_user_error(allow_model_requests: None) -> None:
    provider = GoogleProvider(project='pydantic-ai', location='us-central1')
    m = GoogleModel('gemini-2.0-flash', provider=provider)
    agent = Agent(m)

    url = ImageUrl(url='gs://pydantic-ai-dev/wikipedia_screenshot.png', force_download=True)
    with pytest.raises(UserError, match='Downloading from protocol "gs://" is not supported.'):
        _ = await agent.run(['What is the main content of this URL?', url])


async def test_google_tool_config_any_with_tool_without_args(
    allow_model_requests: None, google_provider: GoogleProvider
):
    class Foo(TypedDict):
        bar: str

    m = GoogleModel('gemini-2.0-flash', provider=google_provider)
    agent = Agent(m, output_type=Foo)

    @agent.tool_plain
    async def bar() -> str:
        return 'hello'

    result = await agent.run('run bar for me please')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='run bar for me please',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='bar', args={}, tool_call_id=IsStr())],
                usage=RequestUsage(
                    input_tokens=21, output_tokens=1, details={'text_candidates_tokens': 1, 'text_prompt_tokens': 21}
                ),
                model_name='gemini-2.0-flash',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='bar',
                        content='hello',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args={'bar': 'hello'},
                        tool_call_id=IsStr(),
                    )
                ],
                usage=RequestUsage(
                    input_tokens=27, output_tokens=5, details={'text_candidates_tokens': 5, 'text_prompt_tokens': 27}
                ),
                model_name='gemini-2.0-flash',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ]
            ),
        ]
    )


async def test_google_timeout(allow_model_requests: None, google_provider: GoogleProvider):
    model = GoogleModel('gemini-1.5-flash', provider=google_provider)
    agent = Agent(model=model)

    result = await agent.run('Hello!', model_settings={'timeout': 10})
    assert result.output == snapshot('Hello there! How can I help you today?\n')

    with pytest.raises(UserError, match='Google does not support setting ModelSettings.timeout to a httpx.Timeout'):
        await agent.run('Hello!', model_settings={'timeout': Timeout(10)})


async def test_google_tool_output(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)

    class CityLocation(BaseModel):
        city: str
        country: str

    agent = Agent(m, output_type=ToolOutput(CityLocation))

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'

    result = await agent.run('What is the largest city in the user country?')
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in the user country?',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='get_user_country', args={}, tool_call_id=IsStr())],
                usage=RequestUsage(
                    input_tokens=33, output_tokens=5, details={'text_candidates_tokens': 5, 'text_prompt_tokens': 33}
                ),
                model_name='gemini-2.0-flash',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args={'city': 'Mexico City', 'country': 'Mexico'},
                        tool_call_id=IsStr(),
                    )
                ],
                usage=RequestUsage(
                    input_tokens=47, output_tokens=8, details={'text_candidates_tokens': 8, 'text_prompt_tokens': 47}
                ),
                model_name='gemini-2.0-flash',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ]
            ),
        ]
    )


async def test_google_text_output_function(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.5-pro-preview-05-06', provider=google_provider)

    def upcase(text: str) -> str:
        return text.upper()

    agent = Agent(m, output_type=TextOutput(upcase))

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'

    result = await agent.run(
        'What is the largest city in the user country? Use the get_user_country tool and then your own world knowledge.'
    )
    assert result.output == snapshot('THE LARGEST CITY IN MEXICO IS MEXICO CITY.')

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in the user country? Use the get_user_country tool and then your own world knowledge.',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='get_user_country', args={}, tool_call_id=IsStr())],
                usage=RequestUsage(
                    input_tokens=49, output_tokens=276, details={'thoughts_tokens': 264, 'text_prompt_tokens': 49}
                ),
                model_name='models/gemini-2.5-pro',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[TextPart(content='The largest city in Mexico is Mexico City.')],
                usage=RequestUsage(
                    input_tokens=80, output_tokens=159, details={'thoughts_tokens': 150, 'text_prompt_tokens': 80}
                ),
                model_name='models/gemini-2.5-pro',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
            ),
        ]
    )


async def test_google_native_output_with_tools(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)

    class CityLocation(BaseModel):
        city: str
        country: str

    agent = Agent(m, output_type=NativeOutput(CityLocation))

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'  # pragma: no cover

    with pytest.raises(
        UserError,
        match=re.escape(
            'Gemini does not support `NativeOutput` and tools at the same time. Use `output_type=ToolOutput(...)` instead.'
        ),
    ):
        await agent.run('What is the largest city in the user country?')


async def test_google_native_output(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)

    class CityLocation(BaseModel):
        """A city and its country."""

        city: str
        country: str

    agent = Agent(m, output_type=NativeOutput(CityLocation))

    result = await agent.run('What is the largest city in Mexico?')
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in Mexico?',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="""\
{
  "city": "Mexico City",
  "country": "Mexico"
}\
"""
                    )
                ],
                usage=RequestUsage(
                    input_tokens=25, output_tokens=20, details={'text_candidates_tokens': 20, 'text_prompt_tokens': 25}
                ),
                model_name='gemini-2.0-flash',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
            ),
        ]
    )


async def test_google_native_output_multiple(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)

    class CityLocation(BaseModel):
        city: str
        country: str

    class CountryLanguage(BaseModel):
        country: str
        language: str

    agent = Agent(m, output_type=NativeOutput([CityLocation, CountryLanguage]))

    result = await agent.run('What is the primarily language spoken in Mexico?')
    assert result.output == snapshot(CountryLanguage(country='Mexico', language='Spanish'))

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the primarily language spoken in Mexico?',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="""\
{
  "result": {
    "kind": "CountryLanguage",
    "data": {
      "country": "Mexico",
      "language": "Spanish"
    }
  }
}\
"""
                    )
                ],
                usage=RequestUsage(
                    input_tokens=50, output_tokens=46, details={'text_candidates_tokens': 46, 'text_prompt_tokens': 50}
                ),
                model_name='gemini-2.0-flash',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
            ),
        ]
    )


async def test_google_prompted_output(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)

    class CityLocation(BaseModel):
        city: str
        country: str

    agent = Agent(m, output_type=PromptedOutput(CityLocation))

    result = await agent.run('What is the largest city in Mexico?')
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in Mexico?',
                        timestamp=IsDatetime(),
                    )
                ],
                instructions="""\
Always respond with a JSON object that's compatible with this schema:

{"properties": {"city": {"type": "string"}, "country": {"type": "string"}}, "required": ["city", "country"], "title": "CityLocation", "type": "object"}

Don't include any text or Markdown fencing before or after.\
""",
            ),
            ModelResponse(
                parts=[TextPart(content='{"city": "Mexico City", "country": "Mexico"}')],
                usage=RequestUsage(
                    input_tokens=80, output_tokens=13, details={'text_candidates_tokens': 13, 'text_prompt_tokens': 80}
                ),
                model_name='gemini-2.0-flash',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
            ),
        ]
    )


async def test_google_prompted_output_with_tools(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.5-pro-preview-05-06', provider=google_provider)

    class CityLocation(BaseModel):
        city: str
        country: str

    agent = Agent(m, output_type=PromptedOutput(CityLocation))

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'

    result = await agent.run(
        'What is the largest city in the user country? Use the get_user_country tool and then your own world knowledge.'
    )
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in the user country? Use the get_user_country tool and then your own world knowledge.',
                        timestamp=IsDatetime(),
                    )
                ],
                instructions="""\
Always respond with a JSON object that's compatible with this schema:

{"properties": {"city": {"type": "string"}, "country": {"type": "string"}}, "required": ["city", "country"], "title": "CityLocation", "type": "object"}

Don't include any text or Markdown fencing before or after.\
""",
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='get_user_country', args={}, tool_call_id=IsStr())],
                usage=RequestUsage(
                    input_tokens=123, output_tokens=144, details={'thoughts_tokens': 132, 'text_prompt_tokens': 123}
                ),
                model_name='models/gemini-2.5-pro',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                instructions="""\
Always respond with a JSON object that's compatible with this schema:

{"properties": {"city": {"type": "string"}, "country": {"type": "string"}}, "required": ["city", "country"], "title": "CityLocation", "type": "object"}

Don't include any text or Markdown fencing before or after.\
""",
            ),
            ModelResponse(
                parts=[TextPart(content='{"city": "Mexico City", "country": "Mexico"}')],
                usage=RequestUsage(
                    input_tokens=154, output_tokens=166, details={'thoughts_tokens': 153, 'text_prompt_tokens': 154}
                ),
                model_name='models/gemini-2.5-pro',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
            ),
        ]
    )


async def test_google_prompted_output_multiple(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)

    class CityLocation(BaseModel):
        city: str
        country: str

    class CountryLanguage(BaseModel):
        country: str
        language: str

    agent = Agent(m, output_type=PromptedOutput([CityLocation, CountryLanguage]))

    result = await agent.run('What is the largest city in Mexico?')
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in Mexico?',
                        timestamp=IsDatetime(),
                    )
                ],
                instructions="""\
Always respond with a JSON object that's compatible with this schema:

{"type": "object", "properties": {"result": {"anyOf": [{"type": "object", "properties": {"kind": {"type": "string", "const": "CityLocation"}, "data": {"properties": {"city": {"type": "string"}, "country": {"type": "string"}}, "required": ["city", "country"], "type": "object"}}, "required": ["kind", "data"], "additionalProperties": false, "title": "CityLocation"}, {"type": "object", "properties": {"kind": {"type": "string", "const": "CountryLanguage"}, "data": {"properties": {"country": {"type": "string"}, "language": {"type": "string"}}, "required": ["country", "language"], "type": "object"}}, "required": ["kind", "data"], "additionalProperties": false, "title": "CountryLanguage"}]}}, "required": ["result"], "additionalProperties": false}

Don't include any text or Markdown fencing before or after.\
""",
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='{"result": {"kind": "CityLocation", "data": {"city": "Mexico City", "country": "Mexico"}}}'
                    )
                ],
                usage=RequestUsage(
                    input_tokens=240,
                    output_tokens=27,
                    details={'text_candidates_tokens': 27, 'text_prompt_tokens': 240},
                ),
                model_name='gemini-2.0-flash',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
            ),
        ]
    )


async def test_google_model_usage_limit_exceeded(allow_model_requests: None, google_provider: GoogleProvider):
    model = GoogleModel('gemini-2.5-flash', provider=google_provider)
    agent = Agent(model=model)

    with pytest.raises(
        UsageLimitExceeded,
        match='The next request would exceed the input_tokens_limit of 9 \\(input_tokens=12\\)',
    ):
        await agent.run(
            'The quick brown fox jumps over the lazydog.',
            usage_limits=UsageLimits(input_tokens_limit=9, count_tokens_before_request=True),
        )


async def test_google_model_usage_limit_not_exceeded(allow_model_requests: None, google_provider: GoogleProvider):
    model = GoogleModel('gemini-2.5-flash', provider=google_provider)
    agent = Agent(model=model)

    result = await agent.run(
        'The quick brown fox jumps over the lazydog.',
        usage_limits=UsageLimits(input_tokens_limit=15, count_tokens_before_request=True),
    )
    assert result.output == snapshot("""\
That's a classic! It's famously known as a **pangram**, which means it's a sentence that contains every letter of the alphabet.

It's often used for:
*   **Typing practice:** To ensure all keys are hit.
*   **Displaying font samples:** Because it showcases every character.

Just a small note, it's typically written as "lazy dog" (two words) and usually ends with a period:

**The quick brown fox jumps over the lazy dog.**\
""")


async def test_google_vertexai_model_usage_limit_exceeded(
    allow_model_requests: None, vertex_provider: GoogleProvider
):  # pragma: lax no cover
    model = GoogleModel('gemini-2.0-flash', provider=vertex_provider, settings=ModelSettings(max_tokens=100))

    agent = Agent(model, system_prompt='You are a chatbot.')

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'  # pragma: no cover

    with pytest.raises(
        UsageLimitExceeded, match='The next request would exceed the total_tokens_limit of 9 \\(total_tokens=36\\)'
    ):
        await agent.run(
            'What is the largest city in the user country? Use the get_user_country tool and then your own world knowledge.',
            usage_limits=UsageLimits(total_tokens_limit=9, count_tokens_before_request=True),
        )


def test_map_usage():
    assert _metadata_as_usage(GenerateContentResponse()) == RequestUsage()

    response = GenerateContentResponse(
        usage_metadata=GenerateContentResponseUsageMetadata(
            prompt_token_count=1,
            candidates_token_count=2,
            cached_content_token_count=9100,
            thoughts_token_count=9500,
            prompt_tokens_details=[ModalityTokenCount(modality=MediaModality.AUDIO, token_count=9200)],
            cache_tokens_details=[ModalityTokenCount(modality=MediaModality.AUDIO, token_count=9300)],
            candidates_tokens_details=[ModalityTokenCount(modality=MediaModality.AUDIO, token_count=9400)],
        )
    )
    assert _metadata_as_usage(response) == snapshot(
        RequestUsage(
            input_tokens=1,
            cache_read_tokens=9100,
            output_tokens=9502,
            input_audio_tokens=9200,
            cache_audio_read_tokens=9300,
            output_audio_tokens=9400,
            details={
                'cached_content_tokens': 9100,
                'thoughts_tokens': 9500,
                'audio_prompt_tokens': 9200,
                'audio_cache_tokens': 9300,
                'audio_candidates_tokens': 9400,
            },
        )
    )


async def test_google_builtin_tools_with_other_tools(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.5-flash', provider=google_provider)

    agent = Agent(m, builtin_tools=[UrlContextTool()])

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'  # pragma: no cover

    with pytest.raises(
        UserError,
        match=re.escape('Gemini does not support user tools and built-in tools at the same time.'),
    ):
        await agent.run('What is the largest city in the user country?')

    class CityLocation(BaseModel):
        city: str
        country: str

    agent = Agent(m, output_type=ToolOutput(CityLocation), builtin_tools=[UrlContextTool()])

    with pytest.raises(
        UserError,
        match=re.escape(
            'Gemini does not support output tools and built-in tools at the same time. Use `output_type=PromptedOutput(...)` instead.'
        ),
    ):
        await agent.run('What is the largest city in Mexico?')

    agent = Agent(m, output_type=PromptedOutput(CityLocation), builtin_tools=[UrlContextTool()])
    result = await agent.run('What is the largest city in Mexico?')
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))
