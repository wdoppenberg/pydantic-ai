import json
from dataclasses import replace
from typing import Any, cast

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel
from typing_extensions import TypedDict

from pydantic_ai import (
    BinaryContent,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    DocumentUrl,
    FinalResultEvent,
    ImageUrl,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.agent import Agent
from pydantic_ai.builtin_tools import CodeExecutionTool, WebSearchTool
from pydantic_ai.exceptions import ModelHTTPError, ModelRetry
from pydantic_ai.messages import (
    BuiltinToolCallEvent,  # pyright: ignore[reportDeprecated]
    BuiltinToolResultEvent,  # pyright: ignore[reportDeprecated]
)
from pydantic_ai.output import NativeOutput, PromptedOutput, TextOutput, ToolOutput
from pydantic_ai.profiles.openai import openai_model_profile
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage, RunUsage

from ..conftest import IsDatetime, IsStr, TestEnv, try_import
from .mock_openai import MockOpenAIResponses, get_mock_responses_kwargs, response_message

with try_import() as imports_successful:
    from openai.types.responses.response_output_message import Content, ResponseOutputMessage, ResponseOutputText
    from openai.types.responses.response_reasoning_item import ResponseReasoningItem, Summary
    from openai.types.responses.response_usage import ResponseUsage

    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
    pytest.mark.filterwarnings(
        'ignore:`BuiltinToolCallEvent` is deprecated, look for `PartStartEvent` and `PartDeltaEvent` with `BuiltinToolCallPart` instead.:DeprecationWarning'
    ),
    pytest.mark.filterwarnings(
        'ignore:`BuiltinToolResultEvent` is deprecated, look for `PartStartEvent` and `PartDeltaEvent` with `BuiltinToolReturnPart` instead.:DeprecationWarning'
    ),
]


def test_openai_responses_model(env: TestEnv):
    env.set('OPENAI_API_KEY', 'test')
    model = OpenAIResponsesModel('gpt-4o')
    assert model.model_name == 'gpt-4o'
    assert model.system == 'openai'


async def test_openai_responses_model_simple_response(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model)
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('The capital of France is Paris.')


async def test_openai_responses_image_detail_vendor_metadata(allow_model_requests: None):
    c = response_message(
        [
            ResponseOutputMessage(
                id='output-1',
                content=cast(list[Content], [ResponseOutputText(text='done', type='output_text', annotations=[])]),
                role='assistant',
                status='completed',
                type='message',
            )
        ]
    )
    mock_client = MockOpenAIResponses.create_mock(c)
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(model=model)

    image_url = ImageUrl('https://example.com/image.png', vendor_metadata={'detail': 'high'})
    binary_image = BinaryContent(b'\x89PNG', media_type='image/png', vendor_metadata={'detail': 'high'})

    result = await agent.run(['Describe these inputs.', image_url, binary_image])
    assert result.output == 'done'

    response_kwargs = get_mock_responses_kwargs(mock_client)
    image_parts = [
        item
        for message in response_kwargs[0]['input']
        if message.get('role') == 'user'
        for item in message['content']
        if item['type'] == 'input_image'
    ]
    assert image_parts
    assert all(part['detail'] == 'high' for part in image_parts)


async def test_openai_responses_model_simple_response_with_tool_call(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))

    agent = Agent(model=model)

    @agent.tool_plain
    async def get_capital(country: str) -> str:
        return 'Potato City'

    result = await agent.run('What is the capital of PotatoLand?')
    assert result.output == snapshot('The capital of PotatoLand is Potato City.')


async def test_openai_responses_output_type(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))

    class MyOutput(TypedDict):
        name: str
        age: int

    agent = Agent(model=model, output_type=MyOutput)
    result = await agent.run('Give me the name and age of Brazil, Argentina, and Chile.')
    assert result.output == snapshot({'name': 'Brazil', 'age': 2023})


async def test_openai_responses_reasoning_effort(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('o3-mini', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model, model_settings=OpenAIResponsesModelSettings(openai_reasoning_effort='low'))
    result = await agent.run(
        'Explain me how to cook uruguayan alfajor. Do not send whitespaces at the end of the lines.'
    )
    assert [line.strip() for line in result.output.splitlines()] == snapshot(
        [
            'Ingredients for the dough:',
            '• 300 g cornstarch',
            '• 200 g flour',
            '• 150 g powdered sugar',
            '• 200 g unsalted butter',
            '• 3 egg yolks',
            '• Zest of 1 lemon',
            '• 1 teaspoon vanilla extract',
            '• A pinch of salt',
            '',
            'Ingredients for the filling (dulce de leche):',
            '• 400 g dulce de leche',
            '',
            'Optional coating:',
            '• Powdered sugar for dusting',
            '• Grated coconut',
            '• Crushed peanuts or walnuts',
            '• Melted chocolate',
            '',
            'Steps:',
            '1. In a bowl, mix together the cornstarch, flour, powdered sugar, and salt.',
            '2. Add the unsalted butter cut into small pieces. Work it into the dry ingredients until the mixture resembles coarse breadcrumbs.',
            '3. Incorporate the egg yolks, lemon zest, and vanilla extract. Mix until you obtain a smooth and homogeneous dough.',
            '4. Wrap the dough in plastic wrap and let it rest in the refrigerator for at least one hour.',
            '5. Meanwhile, prepare a clean workspace by lightly dusting it with flour.',
            '6. Roll out the dough on the working surface until it is about 0.5 cm thick.',
            '7. Use a round cutter (approximately 3-4 cm in diameter) to cut out circles. Re-roll any scraps to maximize the number of cookies.',
            '8. Arrange the circles on a baking sheet lined with parchment paper.',
            '9. Preheat the oven to 180°C (350°F) and bake the cookies for about 10-12 minutes until they are lightly golden at the edges. They should remain soft.',
            '10. Remove the cookies from the oven and allow them to cool completely on a rack.',
            '11. Once the cookies are cool, spread dulce de leche on the flat side of one cookie and sandwich it with another.',
            '12. If desired, roll the edges of the alfajores in powdered sugar, grated coconut, crushed nuts, or dip them in melted chocolate.',
            '13. Allow any coatings to set before serving.',
            '',
            'Enjoy your homemade Uruguayan alfajores!',
        ]
    )


async def test_openai_responses_reasoning_generate_summary(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('computer-use-preview', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(
        model=model,
        model_settings=OpenAIResponsesModelSettings(
            openai_reasoning_summary='concise',
            openai_truncation='auto',
        ),
    )
    result = await agent.run('What should I do to cross the street?')
    assert result.output == snapshot("""\
To cross the street safely, follow these steps:

1. **Use a Crosswalk**: Always use a designated crosswalk or pedestrian crossing whenever available.
2. **Press the Button**: If there is a pedestrian signal button, press it and wait for the signal.
3. **Look Both Ways**: Look left, right, and left again before stepping off the curb.
4. **Wait for the Signal**: Cross only when the pedestrian signal indicates it is safe to do so or when there is a clear gap in traffic.
5. **Stay Alert**: Be mindful of turning vehicles and stay attentive while crossing.
6. **Walk, Don't Run**: Walk across the street; running can increase the risk of falling or not noticing an oncoming vehicle.

Always follow local traffic rules and be cautious, even when crossing at a crosswalk. Safety is the priority.\
""")


async def test_openai_responses_system_prompt(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model, system_prompt='You are a helpful assistant.')
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('The capital of France is Paris.')


async def test_openai_responses_model_retry(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model)

    @agent.tool_plain
    async def get_location(loc_name: str) -> str:
        if loc_name == 'London':
            return json.dumps({'lat': 51, 'lng': 0})
        else:
            raise ModelRetry('Wrong location, I only know about "London".')

    result = await agent.run('What is the location of Londos and London?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the location of Londos and London?',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_location',
                        args='{"loc_name":"Londos"}',
                        tool_call_id=IsStr(),
                    ),
                    ToolCallPart(
                        tool_name='get_location',
                        args='{"loc_name":"London"}',
                        tool_call_id=IsStr(),
                    ),
                ],
                usage=RequestUsage(details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_67e547c48c9481918c5c4394464ce0c60ae6111e84dd5c08',
                finish_reason='stop',
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Wrong location, I only know about "London".',
                        tool_name='get_location',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    ),
                    ToolReturnPart(
                        tool_name='get_location',
                        content='{"lat": 51, "lng": 0}',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    ),
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="""\
It seems "Londos" might be incorrect or unknown. If you meant something else, please clarify.

For **London**, it's located at approximately latitude 51° N and longitude 0° W.\
""",
                        id='msg_67e547c615ec81918d6671a184f82a1803a2086afed73b47',
                    )
                ],
                usage=RequestUsage(input_tokens=335, output_tokens=44, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_67e547c5a2f08191802a1f43620f348503a2086afed73b47',
                finish_reason='stop',
            ),
        ]
    )


@pytest.mark.vcr()
async def test_image_as_binary_content_tool_response(
    allow_model_requests: None, image_content: BinaryContent, openai_api_key: str
):
    m = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m)

    @agent.tool_plain
    async def get_image() -> BinaryContent:
        return image_content

    result = await agent.run(['What fruit is in the image you can get from the get_image tool?'])
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=['What fruit is in the image you can get from the get_image tool?'],
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='get_image', args='{}', tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=40, output_tokens=11, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_681134d3aa3481919ca581a267db1e510fe7a5a4e2123dc3',
                finish_reason='stop',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_image',
                        content='See file 1c8566',
                        tool_call_id='call_FLm3B1f8QAan0KpbUXhNY8bA|fc_681134d47cf48191b3f62e4d28b6c3820fe7a5a4e2123dc3',
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content=[
                            'This is file 1c8566:',
                            image_content,
                        ],
                        timestamp=IsDatetime(),
                    ),
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='The fruit in the image is a kiwi.',
                        id='msg_681134d770d881919f3a3148badde27802cbfeaababb040c',
                    )
                ],
                usage=RequestUsage(input_tokens=1185, output_tokens=11, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_681134d53c48819198ce7b89db78dffd02cbfeaababb040c',
                finish_reason='stop',
            ),
        ]
    )


async def test_image_as_binary_content_input(
    allow_model_requests: None, image_content: BinaryContent, openai_api_key: str
):
    m = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m)

    result = await agent.run(['What fruit is in the image?', image_content])
    assert result.output == snapshot('The fruit in the image is a kiwi.')


async def test_openai_responses_audio_as_binary_content_input(
    allow_model_requests: None, audio_content: BinaryContent, openai_api_key: str
):
    m = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m)

    with pytest.raises(NotImplementedError):
        await agent.run(['Whose name is mentioned in the audio?', audio_content])


async def test_openai_responses_document_as_binary_content_input(
    allow_model_requests: None, document_content: BinaryContent, openai_api_key: str
):
    m = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m)

    result = await agent.run(['What is in the document?', document_content])
    assert result.output == snapshot('The document contains the text "Dummy PDF file."')


async def test_openai_responses_document_url_input(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m)

    document_url = DocumentUrl(url='https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf')

    result = await agent.run(['What is the main content on this document?', document_url])
    assert result.output == snapshot(
        'The main content of this document is a simple text placeholder: "Dummy PDF file."'
    )


async def test_openai_responses_text_document_url_input(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m)

    text_document_url = DocumentUrl(url='https://example-files.online-convert.com/document/txt/example.txt')

    result = await agent.run(['What is the main content on this document?', text_document_url])
    assert result.output == snapshot(
        'The main content of this document is an example of a TXT file type, with an explanation of the use of placeholder names like "John Doe" and "Jane Doe" in legal, medical, and other contexts. It discusses the practice in the U.S. and Canada, mentions equivalent practices in other English-speaking countries, and touches on cultural references. The document also notes that it\'s an example file created by an online conversion tool, with content sourced from Wikipedia under a Creative Commons license.'
    )


async def test_openai_responses_image_url_input(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m)

    result = await agent.run(
        [
            'hello',
            ImageUrl(url='https://t3.ftcdn.net/jpg/00/85/79/92/360_F_85799278_0BBGV9OAdQDTLnKwAPBCcg1J7QtiieJY.jpg'),
        ]
    )
    assert result.output == snapshot("Hello! I see you've shared an image of a potato. How can I assist you today?")


async def test_openai_responses_stream(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model)

    @agent.tool_plain
    async def get_capital(country: str) -> str:
        return 'Paris'

    output_text: list[str] = []
    async with agent.run_stream('What is the capital of France?') as result:
        async for output in result.stream_text():
            output_text.append(output)
        async for response, is_last in result.stream_responses(debounce_by=None):
            if is_last:
                assert response == snapshot(
                    ModelResponse(
                        parts=[
                            TextPart(
                                content='The capital of France is Paris.',
                                id='msg_67e554a28bec8191b56d3e2331eff88006c52f0e511c76ed',
                            )
                        ],
                        usage=RequestUsage(input_tokens=278, output_tokens=9, details={'reasoning_tokens': 0}),
                        model_name='gpt-4o-2024-08-06',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                        provider_details={'finish_reason': 'completed'},
                        provider_response_id='resp_67e554a21aa88191b65876ac5e5bbe0406c52f0e511c76ed',
                        finish_reason='stop',
                    )
                )

    assert output_text == snapshot(['The capital of France is Paris.'])


async def test_openai_responses_model_http_error(allow_model_requests: None, openai_api_key: str):
    """Set temperature to -1 to trigger an error, given only values between 0 and 1 are allowed."""
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model, model_settings=OpenAIResponsesModelSettings(temperature=-1))

    with pytest.raises(ModelHTTPError):
        async with agent.run_stream('What is the capital of France?'):
            ...  # pragma: lax no cover


async def test_openai_responses_model_builtin_tools_web_search(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    settings = OpenAIResponsesModelSettings(openai_builtin_tools=[{'type': 'web_search'}])
    agent = Agent(model=model, model_settings=settings)
    result = await agent.run('Give me the top 3 news in the world today')

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Give me the top 3 news in the world today',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_0e3d55e9502941380068c4aaa4efb081958605d7b31e838366',
                        signature='gAAAAABoxKrgd0uCWxLjgCiIWj3ei9eYp9sdRdHLVNWOpZvOS6TS_8hF6IEgz5acjqUiaGnXfLl3kn78UERavEItdZ-6PupaB2V7M8btQ2v76ZJCPXR5DGvXe3K2y_zrSLC-qbX4ui3hPfGG01qGiftAM7m04zuCdJ33SVDyOasB8uzV7vSqFzM4CkcAeN0jueQtuGDJ9U5Qq9blCXo6Vxx4BVOVPYnCONMQvwJXlbZ7i_s3VmUFFDf2GlNYtkT07Z1Uc5ESVUVDYfVC2qlOWWp2MLh20tbsUMqHPYzO0R7Y1lmwAqNxaT4HIhhlQ0xVer1qBRgUfLn1fGXX0vBb4rN0N_w7c2w-iwY-4XAvhAr-Y3pejueHfepmv76G67cJVQjzgM37wlQFdl_UmDfkVDIxmAE62QjOjPs8TweVPEXUXAK4itTDQiS7M42dS6QzxivPVvzoMkNOjJ58vUy83DCr-Obw8SMfFGB5sd1hGg9enLYiGxN_Qzs9IGegBU4cH1wpCvARmuVP10-CJe0jzSFy0OI76JUgGMVido_cEgrAF5eEOS-3vkel6L07Q9Sl_f8C-ZW04zF40ZIvCZ4RJfRAKr2bfXH6IVNhu528-ilQTCoCeFy_CG6UYlUY2jws_DRuTsAVb6691hPRI8mG28NCPXNGV5h8sVgypbeqWyBNZEnSgqFcNVplAPTxDNqlcFps5bEND4Q0SLSNTZv9vFbRvfyrf-4s3UWqn-SI4QAmGzKRRuTumEpldsTuZgv69Nu2qA7px1ZNu-hN7S0E7ONGDs2fCaUG4X-Xp3j2fizfaTkZpOC_sdTK5e10lIG019zKGngXSrBy_sOWyTIsjiRGdr0Va-RjDw2ruFr3ewQcH5vZ8LgUwTzijfqLqbkF1zgZopHTnz1Gpt42AbZiyP30S9BQuDODD8RmtZQ5oB1NKmISeGkLCJRd6dZKGibFskFFMFr53YvUfVZx4mRpxSjuadceNKPhTVkbGPYE6XrZbChCxDL9aJJ37ctRxf91r9QAXMqeFZR-4HR13_Pp0AyN_H7gqBR2yVuGbXkhs1QwkEhl-6_keNsJYUaRSSf5QN9gRjsuWchWEsTr8AqTbIApGO24a5Rr4GDnZ_6ICYBr-IhUesv0VJKQF3DcNFaOQCLtLTKCC4G4SqURt60V0zkQKWBdUdUGFkxDUN5gtcKrR0F4J5hvZ6OMV3XaP6kpgx62TL_gd9g_QyV8QDFwXuDDrGyXi6l68veZXOElkZ4lpVAjfeXnysK401DRt3vF0z99wUc-QVMjZG0wVZUr5rYHjKKaB2vG85n_onMrddThz2_a1NG_THQZ3L1rprThcQY7FdPtw1JXWfXWeS7ZuOOZCZvjyCrVhevaxTl5UKNbkguqYhNJQfx5X8IkwJWVRObA3QxFD0ZEgW9OKt-v-g_EAsjtftPbeeqaDfPBwqVguYJUEZqPPwcsG2cv8Xu5sCc6h7J8fvwTK-MY847JS5Q5CSDe4GDFvJn4Tk4aIOeGlr-VlrgwOS_yaKd1GogBIDzjh8pXIXXSDP2UkEOd2T0zSoa0u8oewPf8Pwmd7pmVb10Y9tHPgEo44ZQRiyVCe9S36BVjf1iZgTYetfBfq9JJom1Ksz-WUf74sHYfLkUY96lOlSvziyFFmTXxFgssLFgtBuWNaehKeuJ0QiQm2r4jEvX3n7dvUj09tWw_boLWGUJqL5YkxVadlw8wF1KRFJjGIAvEvO7YNoEoyolmS9616ZBvWNlBg54A5DITXEfIMloXVYNmYomoBloM74USiV7AjQE5hPIIqO97dW4btd2zMx9Nbr8G-nZsLgCqrqzDVz0UorAHTgaThtp9BW6VJZJ9q3Ew_z_494P7GNv9ehuK6m3fT-MXIq-t0Bo28YGgGhiFjoYSSYUd1adlHQdPHZCxZojt4-DxgD3iFoWQGc7BBRU3f9rRVRzbDvlHpaLRUQUFXiaB6rQ=',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'query': 'top world news September 12, 2025 Reuters', 'type': 'search'},
                        tool_call_id='ws_0e3d55e9502941380068c4aaab56508195a1effa9583720d20',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content={'status': 'completed'},
                        tool_call_id='ws_0e3d55e9502941380068c4aaab56508195a1effa9583720d20',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_0e3d55e9502941380068c4aaaef4b481959dfd7d8031662152',
                        signature='gAAAAABoxKrgnD1AQrb0I2TSTLa6DiSpAwMbkF6gxz_b8tkkns4MZ4Rr6a8ejwmX7aGMoXEgOO2nkuLFeKoQBzQBrfNZIhmCy68QZMQQKZfKBUv1k8OAKzz4A1dO-xNH6xLMS-3cG4ev4zqjQEOBSGoZNKcZMU9L3B0VCvZsBU7S50g7zCcVwEk6H0wx4HO6IuUEOzgqqx8NYHmOkudSv3ikiHn1xhLc1JEzXkupTyRxyw1O81jJEpNzLlEUIFeu0vkAJrlwQzAHeEzxFMMQMoru3pKwnzujgljefGG8RY34jsAc6XcbJSstAa5GnKn24ehA_CQu80ICcibs7LBKsa3oO8wWWHXgDhMCPJn0N322MZcHfH77PhgEr-T1YSIRrSMPXcxoPaptN0O4ceK9BYN4FDRddaR1jXzWdZ3VhYBNbRrQEuO6z0TOWsPmzIlDql1a20jiOteGNQgIX94Af4PB5g_DYWzJW8YVffnhKXJEmU7BmYuctQgyewLj_CoQYfQ9HtGcae6ZElUEP96lo1ID3AW2iMa3iP4C2xULWDVh-8rWf0D2fgS1toexXXCtWbXn8XlYMGWVjq3WX5q16Kq0KyInuCZleABTeFRuzh0MTx1GaYhDTwHxG8BRPYUxz0bHHESz-h_UGmhGu8-a49YdBpLe36_Z1wprXJ82Yg7KvJy68VwKnLeH1Zm56aMHviJl143iZYgiZaVmRBIRExMvnI9LVAT5pv0Y3CdCCSq8Bs2jSbhU0xe26HAqfZZnAsE0LpPAfW1tMCiKzqhtzoKR6yauAYCXP5YtnX6BqFr-J8px6owPJhepjyrSVCObyya7v7_rV81BkYOtLQSwCUUhOjbawgI6XDQ_FK0hye5lFVKckFNM3cVpgRcZymeqx-XoQeoFOR8uLtcXv2DIoo0TfP7RxgBvAvdohv8vZx7xJSXlrYKqLEK1ASQDcc36gIfNQuNXM24WuXForXTO2l_sTeos58eX5FGxWJFDghhrNa_ia1dL7towjcegQzf9LtLjLlnqUGpEte-o23DKKQQEiFfMpLlvGu2cOVwYUuoeOpEBe7QpDbJGdBjq0hOKdakHGl6KwBw6vCkRp_wtW4R7QBuncdYyRT6AJ1_Z_byBP7kH1A2-P6QMVycBVcXlUgc0BzuGlkt51l__O3CM4z-PmI8zR5cL6ZCXoQzG2Yp-OhQ-n-3hgMaCfBGca6J3wP1vgQpR2AF0',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'query': 'Nepal protests September 12 2025 Reuters', 'type': 'search'},
                        tool_call_id='ws_0e3d55e9502941380068c4aab0c534819593df0190332e7aa3',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content={'status': 'completed'},
                        tool_call_id='ws_0e3d55e9502941380068c4aab0c534819593df0190332e7aa3',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_0e3d55e9502941380068c4aab3df148195bdec4fe186f23877',
                        signature='gAAAAABoxKrgZN3V5pGaqoIM8EEiFso41O_kxOTWpzAh5Nlj3pqqDjIGrFH2zcmDyURpUmXdExY8L9K6KcwGOAlF6okEgQojeTxysBi4-gDVFNCVfp6c6K4tAtCBrvq5wC2g22Ny1pU2OMyxVU2GCxIIehCZiPQio_7IS8WY_VWkwLOag7bT4FBGn-aVFyoEfDDpIPF-4Zpcal6bAvdjD2hYGl6_-8alwh36ttUkJroo2qG-Mn0LsAWJ7YEzfrHgoPTDF7TB3Mfvvc5M_eP3pzY8O4WhZKMLBSnM92iIt5J3nSJYhRoiwEjaCamIM4vK0cnJR0oX87u_XtGvnNBX93ttrIrXDKK-mh-LIoe_sK1dViFINxk6rJHZvkFK12J6UXMK4me-C3uQ_qGygpw4uYvWhYk7LDR9Zgxfv1OoDg13DCYWWrHX7Oa1ALXPotk1Uw_Tof-Wc_wDqE16Elm1a5TP-ISH45v9W_Xl1IXo7J_jwOlAjkXvrh2a8YNljWQqBFCca-M2hSWvKuX8JuNF_tkI2q2E7jIDNt77jGd2yavqb1W2WoB_s7jqyAWomT91E2gZQtGJa4X2ydeTPQ_oWv2hgdTUynV0nbOKWA6suZixvxVDLLedhYHRnKY6EOtyso9MZav1qhr_DpHExn1_woquJXtS7c3Fe3Rs_YrU6PpRx5_DEVjVKme-3XjLJNclx6NF-rbXYqhXXExqPk-od7n-YMyrYhpfVP8lmLCewwyzVRb1koOEcCqnuhqM9DWyazKAcdvejM7VEM1AEk8ugT02cTiF7CfLefYFsLSYVBM0Ox47Ceh4BOA82jdlf1pZNvGqgHi8kKm9HLVh-yM_DAhD8O5Ub-SCd3bNi8735XPDWVIm6sKMdg1bcgVehz_R4iEBr_pguKfZUJLcckUTI6fitAQ6YSLpLAfRA0nMDBfM6p43jqsSCP8Ovjx58TwAPElgpme4ENBCozS_VaxmqawpfUfvnD60xia57wtSBYr5s1j-FUUjBsFTInjHdKcp0EBd3Pv-mpVE-Yj0MYExbn1upi3RxWN6jwVeYc603HQBjsjqsb-op9Tb0GZxf5Z4DpZ_eeb4IBTWNf3FTLIbsVg18Oyl128Std9CkMGak8iI_dFCvm1ZQQ6u3CyLEwxGsMZnkZl6OhSKDlnHDvRsF0F0OcRtFV5i7j92kMs9_qJ2JLdb5LzdqOBnFfKOcUCXBOflL58PYIav',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={
                            'query': 'UN Security Council condemns attack in Doha September 12 2025 Reuters',
                            'type': 'search',
                        },
                        tool_call_id='ws_0e3d55e9502941380068c4aab597f48195ac7021b00e057308',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content={'status': 'completed'},
                        tool_call_id='ws_0e3d55e9502941380068c4aab597f48195ac7021b00e057308',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_0e3d55e9502941380068c4aab85e7c81958c4b4ae526792e49',
                        signature='gAAAAABoxKrghWofTkZCzljg86Akl6ch3bNwR70Wz37t6mBpLah1wZ-7U6isPixPCn0t66fA6xKxGX75bmjRu8Gts4cIaYpm78c8n6R44UULYfQoDC9ZEyGgImbQUKoHFU63nSbsjuPTTFtLdLHhEebDE_t6AfqIWBlyZKRqYlXS_8mTZ_NwM5_JgJun1Xz-I3Pb0X5ZgX8RTP_Kh7Kk79PStvg0-qcVoxMtFsK4ZN3fzQBOSUwvkMhglIweiS3s9CpTbtOs0PYqFCOIjKEYZ2-Rt_7SKhOGaWEMuvuWggMLeO_Wkl8HyIHre5JolVFR9M-43XByZXQxrvBxFzzwHubiyCs-WHFicgMyZcAF8e2KR9KdUJxAwQ3acCi3zBc7e5q1jgc8-Csm-vZQJMTyABDu4yuLena6rF777C8jq-naUe5M-bBpiimK1nbpg5YDiwx7-TbZz5eiTpptHL3P6izhgEOXuEvLhlrhxPBKTezDkiwu-wjs0tHguRYbOIMf-3NZGHuYnOcGfC2wJKkE9DmRvbicnChrLqzHmiXWblYhPwsH9wt-QDvrz3tgCH4B3ri9APreQjBmxtZEVGQAtfdm1qpgiDcWqEijrj05rvr4HxbSReCFszZJDYAufNhJSPhuJXl4e7EHRLyVd2uJA264ONj-MxT2WRr4MGzubSXtPd1QJn7IEkCCuPZxbLf9q27DTSpAvS1oZVs1Ad0J4lbRV5tS_sG54JLvpXf4jtYHD-R2CG0vkL1i0273IJroXScLaPELp0iJMn-WzAkbEjjMsX8gmZlV2X06XuvSjry-dh2sU9Yldqw4NHMLM8rpZIfKbsm6w0ub5Icmu19E856R57JM3K3Pjm3fdO3HR-adVsJTAaIusyUVX3SOiTY53-X6UbqBJh5H3WOORqkwW2nGbNur6B_tyRjlegD3CGJzC-A9rNxMWrecALmCEJBwnXxOuvpsGkSgjP8vjnY9JJNj53hxAirHFIxknDMrKt5qlsRHxGlCdN9H7YuTGdTSgPWH_L9C4BtZrr2Qk41osiDCpacMwBeUDwo1YwYWd1SO0DEzm2qGlXSYeuAQ6Fvyc7sZHCkOsl-bINhCuY1aEBOLzXS7kcu0YAIuEZGVp5wUrr2L6YssdrzpzQ_KENFI7LiB2v5CrF1wZN85H2dkwaGciOXznAa0Su1fWD3BUdpyR0h_mVIcHUxmeoCywWbWO-Do3LFu70MMxKmfSzVfL9hlU2B2jo1aqJ5HesWsWbsbslW5FfREayeUzK7hxkrjliDePhN6gkfy0HOYQijPN6dko4TNEeKFO6Q-aw7c4X5IF3WBCYd_IszlLBK-vTX4EX2J5QtaLRfwFgRwz_K2fkOTT64eknQ6R3fFJpgeyLBZ5ut7j2o7xhEuHeE4KPm2T_AJi8yRScMU-ZsDcUZ8IVYAduy2TGov51AM7K2WojgvqWi62AwSLd16eEnd7SUD8fiCwtRN3zTdmh3MenUogxtKG2YL4hUvSN6Ia1STXpfU4ToLvBnPS5FoY2GuOG-EdEAHdKfYsSUZmSauAlQy7sT43STLkDE42lOKWqtSNHOygkGUodv1GNR0sA6CIg_gVAOyUG-o20rMsfANynNokpoKxJBPJScf1Mbivm-7wJFRipf2-Ay4HzXhXZ4RTkpoq2MMC7cZkHkEprUlLshEhCIHF_6sb1Uhqg4E3UPCCNZ-X0epbQ2GmhtaaIt6BCnWz4SccN5qTks5XpQarlyTW1HubLoLjjXmwJ5DImdUGZkitiJw6ermiOFAFhLfhug-XVKBcTBZOG_CHjrR_2j5TPn6FNLHbYpLYS5hkrUWCJy4U_1xebGl3F6VdQDy3LHZehxuKPowPtdYFenqdJ-naK_A2ygjDUdGBoB2-QFaq8ZPTAti5_Ca6LgiZPvzZdGZ712BED-Opges0mwyAhhsgKRvjjztcsiZ21QpfUaSGLS0vO7J-NcRVvCDyBisMRKfRcWk0PFa4LKcqx9_FNU9nqXH1RXYh_WNAJRVLJDR3WzpNzDv7xMcPOYUUx0wuAYAWcGbc3i5mkVRlzRW_WymBibPF_Y9Yf5yt7plmai5dzlg6aoRdrzSwT9Lphrf79QI3LfYzOV4sXmRGEnN1ud0FyfVB4aLHSsc59_eiPswLL-xg8XT0L27IU_Gja0VuE3zBlErtlQB4uPq778Ojs8hucNTD0rjxs2qqA==',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={
                            'query': 'Israel airstrikes Yemen Sanaa September 10 2025 Reuters death toll',
                            'type': 'search',
                        },
                        tool_call_id='ws_0e3d55e9502941380068c4aabe83a88195b7a5ec62ec10a26e',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content={'status': 'completed'},
                        tool_call_id='ws_0e3d55e9502941380068c4aabe83a88195b7a5ec62ec10a26e',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_0e3d55e9502941380068c4aac1c2e081959628b721b679ddfb',
                        signature='gAAAAABoxKrgINXOfVTRxYfQpc8ZZGXsBHdv43DhHkpUjfExhAS41ACM9vHyRgDNfC9E62QVMWRCWPuz5QX9ks0NtD76PYS8n5bessBYeBtYgbiMtl0piW1gE5dlw-BeKLiijMhIVwytWhF3JTzoxoA60FjPK_sA8mFk6wDCNKDXlaLWsLaECxUwCtdktN9SQnQFgxKNemRKQTyRTNKsurCZSSt0tHyd4lxO0Ei3F2mO3WB4Oq28BeVG7RKlcZ9BmLRdBhFQX5eoLxTBHwC_qgSIGzoVCiyClW1OzFzXzmaCUCm3oUDQjooYIZtQqK1b8FBArzN9seOJ4vuxu2qqdtF-JC1vAi-_9J61EwELhN5gYvld83zGCSPg_asjeKeoA6qnA5RFtYwh5kmMSFo9VzGp9MlCmb4_-L-iux3JKc7Kz-jvF1sXSH7YfKgBvcn8HcOdXGjU-aBJTmdP3hCZSL9ko-NNsUO31667QwMZsQTlVoTCAfWS_xDEI0QgmV2kFReKhKanzMmOToUECPPQHQfofCGxwxjbGllSyhpSZHIdyjXpHBmwFALBflPAfeM8wUbqQbNyWbWTdx4Uz62Z4j0OGfcMpgMlDb6BON8vvpIjmlV-fOqRlzkP97klPBygPKeRyT-UezEN5Vj5t00nmB-cV2kNj1WYmL8-eBuJPs3LOU_4Q3ysb90AxYxRJGOsl74lEBqfUKb6b4JWff9JFv11EVJ-puIpE7MA3DPM4NcgGfDZYyDvLS589wbTVxSngBqEOIOEcAZF5Tae93Drajy_x8fXm9uWc8daMf5kqUeq_vwr-ZqEz5ZBUvhvGPL7xkYfTfn-RrQXBx2JfyDRakf4X4D1W6jaO_LXfExH922e9hQ1vH8VA_GPdOIqL5BTiIeO3qFjDSRxMi94XWPPRm87yStxEjx8bse00Bzi3grZ1c6M5dEUXNaHrnvEdJZECT6lz365_Qbl73_Ma_2CLYZhLhtqZRZ6Tycfpprg7rWxqTftOKq4twUgCzzv7kg0e1f_JM_om5loPP6r4MOeAL9O1p49tWmj1kQt_nmYcX1WFTQOgRuB_h3t6ZeOsDb3-VYjIjK0pvj_X_VArrT2suBVitTBXumnG2dXg_z2k5t4KTbWVe-aaGhije0VNxgPWCcu1RlIxOaz',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'query': 'typhoon September 12 2025', 'type': 'search'},
                        tool_call_id='ws_0e3d55e9502941380068c4aac378308195aca61a302c5ebae6',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content={'status': 'completed'},
                        tool_call_id='ws_0e3d55e9502941380068c4aac378308195aca61a302c5ebae6',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_0e3d55e9502941380068c4aac5d2bc8195b79e407fee2673cc',
                        signature='gAAAAABoxKrgkfoWE9D7tW3LtG9Hb8kBR9vHgjhSKvDrW0_FUU34LIByJhhBiwZOr5RfbqX9mBwahQKrIVAev3WGtBfgtJF0kP67CIXXRjA1-RHuY-4QXL_w-t9gttak5Dje2NU5hNyp-LyW0plO7DZwZDkFUgeW5plxMzcAFNTdflBSC_-zYqBFl9p-11YKOslzKYxkfQrDiodarFGFhDOJr97qwo-l4BhSg8jywQwgFOTSrOjJMlZRSrTkHd8CUaSF5rUaLKpY4AZWtpiR71otchA9N-d0AaVwnnzJbe53PXJpe4fGUkmkcZt-ZOcNTQlIpifirDsXln2Sc3jxSM05fteSPKoUeUFIIqbCaZwBPau45DKq54PvkVQ4Fpv8JtfqKEuQtJ6EVlNJALuDlskdxM2H3Z7XJsXkcNCVAKmpA80yYwh3eApMr_cERl2bLS9jJpGt8QN3z1yRe5oCPCNWj2_NTgtzjknxcFy8HdT-pcTzLDOhLJPYyl66psc0Hn8V_GFIFkRBa8tWb7CTLt77a3pW3Ifnxov5ANAaaJLM9gGiH_DgkkuNZMR3dz2sVnHzAG5TxmSQteu-uYQgIYanBH_D2BN24JfBFxckpT0z-kGHbJnL5q_wBeyy7o2puohaH3MNIluzWARcDWaFa1tGkzeZg59woqrrddAdWLRNULpnX9fzr7aAWXr1U5-XkSjyfWa4nmIFtchwPSC-12wHRNFDzdZiUvQDdJ2ENGoIXeYpob_O4Wa5zx4zZj_qHXoQWXLELyEMJZCVADjAjO8uy2gXDxZKcUxyDgi17hIyFtC9Z_4rxDbV_S_JJ68s1qHBZljuH0mrkLU0KXmYi5ZgB_z1CEaz9KkL32FGBt0YXuFoR0LjnrdpOTa9ifWC82ZhDfjz1E4y9FUoGPVl-QYQ5ihDY0LswB1x_FJfvwRLvLRtMeeGqNYEwnkX-XAcVa72acijnRJVxd5WjV5nolIrtq55l941oeun2ThZJZWujP7eMDuQ8SycBOx_6Bz6wECDbnCrfyxypwpVhKSPGuI1IoP_8fCeFDWzZZhD2bTbH2Uw6nzm9SLODQ47GqYlZ6ZtTIgNBlGpiSUrqXhtj9_1hkGZuGv6AE9UAjFNqAWX25db2I2uH1MXdsYRPLZFhYan9G60cozj6N0ekasNkbaAod39JQ7zL6Np2O_qz85s3bcJSS1_aIxW4YFSEv5IYFlztQrhnlyE_gloA8eRntHAinUaGbL9IKTmuj4w74Al1sN7ELITivL6aZ-EM-F7vvFM6Rt4gL0NvlfTYsafoUL99EfBTh3Rfl7pIwOQWXxg_p-51s13BQ1-HWOQxu1lyxbZdJHmhi-tIzk9iyQh1tbkCZJeh_qF-eGH6voxUlcz07gvTckVKR147UPjIrfSm6EO5zXBgva0Zk3nvGFCZshZSau3tLQrAnB7hQ3AAyQT8_6eFBHtsscuApVGtRYIw3vi9decgXmFdvSEg4Iq6JNObTilSq6a3zmUt8fop_M5qYzq-0ctNsXN5lkqi9iB19lLw9EyHNDgClaTAviXWh6aDdbWP-atkQQ82PXBnKJAiP7luW1qf-YVHtKkwNadbMy82CT-dMNu9c-chRSx3g0tdwTex6tgwKMdBRbPWa8NVZreuTy8x2yarHskXhHM21jrexM0pMbk',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'query': 'Nepal protests September 12 2025 BBC', 'type': 'search'},
                        tool_call_id='ws_0e3d55e9502941380068c4aac9b92081958054d2ec8fabe63f',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content={'status': 'completed'},
                        tool_call_id='ws_0e3d55e9502941380068c4aac9b92081958054d2ec8fabe63f',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_0e3d55e9502941380068c4aaccad0c8195bc667f0e38c64c2b',
                        signature='gAAAAABoxKrgxBU1Y3g_B0Eo5nVBHYxLC3Lgh2vNx7AcpSm-o7XiHfQvzLqaLvkI-Cc3F15mQexU0OTvx9FePdIKbwkMNm_X_s_K7YazPjZUTQ0TEod2VereH-Ebh6Xjq3bHm7mh5PWWGnY2SqVMCdKGtrXkoMzBraxedlv2-Tz8o0p6SYuyzM8yHecIfkG6Zd40AdZSiDzsnRNg7gA0zCddrDrRcOpeMTzSPw1z74UZtng-_pPeiv-TGCgwdlmBv8RRr2cuQTYE-yhcp6doCMKqemL8ShuIyfJz0KhQPwYE1zM1CB8sFc_TuArJJD3V2U-Bl3o8anIA8X7YclTlzz_N7HROtVI5qFQjSNhSrbxZKUBFDfAayrpQBEOyIRu7J42uAiBmoyms1WG1E2UtO69nx2ELSJs5yheEuVy4cTXyndBJr2sCs8VkVvcX7xvYkfKeChvkAbUfCotc991qAiyVNzhncM2Z31IEXDEDypeo2IFSwAcKuuXgePFFPBiJxmNQAQmErqbSoB3Woe1j5XjAzJ2eY5YEBZ-68GI3B5wmiZOLsPla_L4iBrczHI1iwGASgtMsuHPj5KVzwef093kg9QBlt-7pZHM3yoU1l5DFSJ5C168MdMdNGF3hn0T2Q3teUmJ5khgcKMKz4_ZVUjEDq8bPwp8DiaWlFgTv-Y-I8etik4o35EFmmmZbIZ7tk69xlBrGizm_KlcYWHBQ5BfuNyZDXZ13MKDyn4uyYxRvkHq4z4jPFEiZ3xX79mlNP3-B0T9g8CsqX1G1prKI7lde6oAHcWPFSWqZmM_JxvYXDBbck2DpEpx4xTuE_iJfGnKiNzanqV4EdOXiCTBVLZhMvXj9rAbwnhttvz5WhIeYAdsKEE0M1MUHuSWuWFVtClp6lPKSLtHQCBtE6mpPDyzUuaw6S1DoixZ6f33Sr8DB-EwF_deHRa95kEN9w4i_LqNbl5QQPF_1je6spo-yQTDpHc5wUidI0fBEQzM57rr9XH0F2afZtrQv9HcLfWKVufBTdd7ScpyOaKj70zgqTAq08Te-Yrj9eo3tbDt698U1fKEYW_uqP48ZKmnSNtFzKOoBzkPpKcwA5AQUiFOYH4-iDPDTOH23SYx8vlymoRiK1imCdPwWYI3miMURxPr9-zCHoM7AiB8cnJlD--zk-j1vQqcf3AntIKPwqycSEuJ7MWb9iN5Ybd1YE25_ZiXKJNVg8wnmTueelRdeM-2JVzAQwth1_3gnsemXn5v0uDVNpxvXoRtR1w8L_zQzKzag8kZMvfESnLCAEwYsCcrP-ngO97iKVvUQnII4RUtG_mSPV4V6Ses_cMUVqyHiM_W_frIosY-7dXnlox89-SPWrRwyC1jlGRA_LE1fpPZ2cZU7Gcyzrxp6yBuTCx8BHr9FJvqgbqtAUeYDpr_Sv-RsG8-w4IulSNZLH5Bh8TyvBGDhi8_lUbDCFTS3KI1ZJ8KJwbNLxF4YUI156zkWIN5yU0WDVlwoxpJD0naMPZzR0sQadMuaXEvLXTFm9Gtb667B2cjdzJqbb8z6NkAx3txRRD6EoezoYADq_ZR_LYha0iwv3bHvg4HIblhU_GVhnU-a-lQGQhTJ5Mh4OmrnTGUVD2Is1OVI0EmNscUuaVc7M1_ga5KbOgyff6bYS0ARh3Io5ekKQKkPVyBLgjjKlej4tB-vSEgitDhEJ-PD__ouuFaogm6twZy7hWVn9cgJmt-RHDZ6gOZm4QP8dWqRpuyEAtTpWR2TLTQVgM05hWpDqDL5AvBjAQ_GWkHCvdCvUINyyl5TsyXUcL207shrLUDCpBe_kESpF5dpAVng8_Zfu1dt3c04cCG1eg40e9JcO5iA9-upTrEPIPrXnAKy4vw-vbhQyL1r2jZWRVga9Do2idmzVf-c7yQ_AHGmf62SHGm-qqbljw0sXJe1rdPt2IHxzYXkhxpqqoaUueQk-pXLUvpMFeMcH97sK3toeCO3oiWQPG-nev0B0b__U8ntgI5m9df6n4IA97iS2zSylSY-F-XEJmLM2TKuSEdgAx1EBL_jyRQKB_8PW-0hSQGJLT70SQqDUJexwyrKABkApv3FuSH4FO0rXZ9TGN3GsnJSkIrTrzE2NG4OXK4syrmtBCb8DjsiicvjAvQhcouOM1xMZ89aSG9Psx5HRnViy6M73TIhYmWO71BRNEayMJaOMgUlgpl5alvV1YFBsChL6mxLVAJWUFuv2YPNaaDRqZEXYHWljhwSn24ASetweLc5GhnehdiT4JVJ_nfT3bygPIjEzvvIa7bbJSeL_bcY-qGAgsuR5m70BdjIH6xLmuqn3lEqulh9n6IPaDciryWqRr1OwxZJQ0-x3u6-G1wrbtrhVMK2Z6cyNUX6MvIMz39B_782X4JcLMrVm9Jgt6qzmfbJPnGA_NK3e9dlz6hP_AYoY-Je-IZEtpv4wyXAYE8v7QXsZbf6DetAM2LzGmxkEI647-pwVPQua-L-84L56GoAw9yDeoXxgyxyf40sbaPIiVLgl_3A4Nghl7uOnOX_1VnZL2X85zCkOZbmm5pZbuSeKesBYbX002PN-_P-P5xRv5b8dZzD0utGv4GUuZJXKJPhbpv8cuBUR0BYHKBQkmOzOBxgCFCDtX84VkZcrFwmQHcS7zmjgqEl39UNrqq6NZXW6HZDyi_SSvEYV7eJfJfxnUUF7RJ49RtSbC9n0AkzorBi0mSMnCC_A1zhamNLjT1-tj4E2a1zI9YsBZ8lPv3t7a6U85iMYjl3kCPiAXkRIDVBihBK4ki_OEa4v6kNBEgXNMuFmd1l8O3WTqZRSTLek4yH95V_uE5DQ9NH52pkgrN7QOe0QXxZ0aErqjkSQRbbhFVVRYp2VN7QpvMGZIAtu_mGssA5Id3X1ZsLEU9zGNibIzAmJdBjS98fVj2MsD-4qZmzlWiCGcC5ko2bbpTrFGtr4r3-SNc4UMOa3dsdyrRlnK3o_tbXbPN7c1H44oneAsqWuekfUVFGvCRm3yA0X7njFB2l8tSXkAuophgRUlWnzp4mEMcpFRwEX3WEnK9hPqXEhdirLtC18yupkKYBtIpCIT98zgJNb5TRbfwRplInEG1E8dk4gCbwyXCNu67QEI2NM2yqCHc4P5rWhwTGAl30tmDQ064ba920L9ZV8d6PgpBHZmUxpJ-JUZuYMzXfCFdlBQANdjtuxCy3-Pi0-cO7UEA84WN-keYB-kHck3aPpeTG7-lv3je0N-407H_A1TKUqkSknjlmwVdL3h41bbGmqxFGizNXfq-uCGUD2tWaZ-cdmZZtGXxgEQ2z7_tLur28eS1tlx43y9CKtKPPJruJm_7BljMOCMPnSmOJDI0JnoGpjNRqzKbSuZFTihaQSBo_Vc-NxRpFwM4xJgq3z5eShb_WamKw9uYrjCBEEwYFTW2QjmiQJtM9eVHBuLkfOVa66YZowcCvL8aCccsuPbe7KBMCD21IGzH4nlhfgUKa1cTAUiWjRSgn6SO5Wqahxs7dEf44F5HvPG6XUy9HFOe-d61ZE-tJQsHZgssQWqV1UfPsccqgyWIc2yv9aK4pPpu2lcrlGu8aDZDz7pBD-dPUG_B9XWt5c0CQj4CCnURDATNWqH8J8VvKap6Zn7pBHW_PxNSJ3f0z_l-GjBlx7U4w6XmOMBtJK8lE_Y8CuuQY9dNVnTGMPibCeJt7M_Q9-IYcqhriUh7Q5WkCvDVu8157gIRwwUAvgqsWcD2msXtO9svRkXKxNxYFdW7KolF-y8oxXRPwVJy1bf89pAOa8djb21ovJuJmbvrRzplFGYNj8rGZ2hXenxDoYiKv71LGALVU63mS9q-Y1zfTHCPpA-Rw7oR6T5G_Q35H-elaA_u-vkgh64mQNP5sgc_kpwbVlM0wSl79RcExnmBTpA-kn7B4w_QPwt185WD9jQRjhh3LMQa_crf4nCWLlsYcDCyB07TU0vXQiQ3nynqsX2MstUc2DaiseVG1SO0UEv8oobwLhnSvl3n8zWMWq93NSuISAsaWmqriNhM74aSHw4CVPoO68RSSdNrpxaKGf8kuO9Xy6iLr3VPE_vyMJDq65q42AEvKqP0TCoFUzXA28Tkrg0tsMLsXIhuT5MGtO3O8RpLnthF9vT0lM64jMp9_QSH2BuWYtwgok7xk3gRX5yBQeksAos3c7Jn2bLM9VNrV9dLi7MH_mRl5C64b0Lgj6Zi1USCyyPhL95ZJIvdxLWHSII2RFbL9ToCThKp_cgPZklLAVJXBeIOqG09pIQ==',
                        provider_name='openai',
                    ),
                    TextPart(
                        content=IsStr(),
                        id='msg_0e3d55e9502941380068c4aada6d8c8195b8b6f92edbb53b4f',
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=115886,
                    cache_read_tokens=92160,
                    output_tokens=1720,
                    details={'reasoning_tokens': 1472},
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_0e3d55e9502941380068c4aa9a62f48195a373978ed720ac63',
                finish_reason='stop',
            ),
        ]
    )


@pytest.mark.vcr()
async def test_openai_responses_model_instructions(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m, instructions='You are a helpful assistant.')

    result = await agent.run('What is the capital of France?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is the capital of France?', timestamp=IsDatetime())],
                instructions='You are a helpful assistant.',
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='The capital of France is Paris.',
                        id='msg_67f3fdfe15b881918d7b865e6a5f4fb1003bc73febb56d77',
                    )
                ],
                usage=RequestUsage(input_tokens=24, output_tokens=8, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_67f3fdfd9fa08191a3d5825db81b8df6003bc73febb56d77',
                finish_reason='stop',
            ),
        ]
    )


async def test_openai_responses_model_web_search_tool(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m, instructions='You are a helpful assistant.', builtin_tools=[WebSearchTool()])

    result = await agent.run('What is the weather in San Francisco today?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the weather in San Francisco today?',
                        timestamp=IsDatetime(),
                    )
                ],
                instructions='You are a helpful assistant.',
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_028829e50fbcad090068c9c82f1b148195863375312bf5dc00',
                        signature='gAAAAABoycg20IcGnesrSxQLRh2yjCjaCx-O9xA4RVYLpo0E7n_6m0T1IUyes5d6U4gDzUNRWbxasFx_3NEhFIuRx4ymcqI_K-nZ6QNsq3V4CgwBbWBXRcBEDVzXSZZ4IoFASBzHpQGbs80RvZkgqmJkk8UzBw0ikt1q9jlUrwMKf1iGdH-S0fIgZn_uEbli1yGWRDryyS2YQWDKNTYuaER_WHVg8DadL6_ltUTwJ9dMzaXyFEenPfuLdDgmba8DP_-WYFMbggATUfdMNfM0O4YqnTmjR5ZnSA6kAbXvnp9sBoC-t8e2mWiCXzvy8iIJozNPo_NE_O1IcMdj1lsaY3__yWzoyLOFCgkrZEnB-_WQNCSx-sVcWWLZO_Tqxw2Afw9sWAvFR6CvTTKdigzDpbmRlvlAJCiOkFQCMrQeEiyGEu0SSfqmx6ptOukfJn4HtQguvigLDWUctpjmNPutwP880S1YwAcd7A-3xp611erVJtYFf6oxGDXKKb63QAff_nZ57-7LdlzSSUr6VaJa5dneGwCgKl-9J3H0Mo-cOns-8ahZOL8Qlpj8Z2vZLS5_JQrNgtmDaaoze13ONE5R84e6fcgHK8eRhBNTULgSD13F59Xx7ww3chlqWeiYfHFwmOkNZp0iNO7RJ-s7crs79n2l6Ppxx5kd4abA0c58k1AZj7avFrexN_t7snuYqCNPsUHMUK_1fSq1toGa7hTVX5b8A56WFSdMlFD51AuzeIzgaEqBtGvq51murGbghqUmOy9g-6_vHz-WOPZeE1M2p13VB1n5fIh3-V7nd9PAXLX1kLLKiS2ox5tODYvkxf6oqjgR56n5KCuWtF9WzCwikaSMN8pwC3ewW6nkkSCPhTBASEJ7BK9a7lDlV60T6gikDbZGHcAfSKDZ5mBBwSBRpDfH3F0MI0Uo4oQ83J63J8a5r3JKy4KVa-5eNsNZsCgxO-7xx_fan1MH9zT85SLwocpvryGSbIDD9itBHK7Yo7REFRV6_U_cdi5RhDpEc13QETSsFT6CaeoL4GAwvJDCrcKjW5u64StH8l-Z4XDAtChG-znHeme6WlJNElY5unp9L-IolqqypTS6lybk7bfUtGPBDeuZp6CD80qFkyd46M16vP1mudv8rMC_ZEdFvCoHDmUg6_KxBxdVbYi-jaXtXYY9D8G6SlfVkeBcNiDCWjsDXSlhE1ibI2pHHN2E-kJLRaHA_Pse0Gknu6ZecQLaUCKWr_mKh3axV9d-pkvxpCcVVakOF08By0bUe8h5ORELsRe5zzMpfbYGaUVhB360OxwqzizyISXmqhW3Q7FHcgZQOCZQVfpuk6ccAYpZwgZbft2YZWqw7_1MyK6TitpdyIwdLFnt2t81JNoJ8zWLveZGpuKABxW6krhjQ0_qJCnLHm03o_D-9BximrLUCs0PbleK5mu4Le8lCCs4eoVjeDHQs4xMm-VtJk_3KMT6EVe4nrb41ddSKX8hH9rh9l2NlPpmPh5UTledwhbtQYdJdQBNFkGei5gpAQ1oHaLkSOYRqrRmy-VIBobxAVBaQWNKcv8CrGx8RIMxrAiU8JoyRsU7Vsobwt1Jboo=',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'query': 'weather: San Francisco, CA', 'type': 'search'},
                        tool_call_id='ws_028829e50fbcad090068c9c8306aec8195ae9451d32175ed69',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content={'status': 'completed'},
                        tool_call_id='ws_028829e50fbcad090068c9c8306aec8195ae9451d32175ed69',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_028829e50fbcad090068c9c83115608195bd323499ece67477',
                        signature='gAAAAABoycg2MBei1jlOMd9YfezZ45PArjJAExhzJt4YG36vuQT_e4K6W78Awn6mrJEueCnEAbciBRoPBd8n0YMXbqTiKdgeceqAoZu_UJAVWxgY7tVDlkg4e8BgJ_SrAumbi0yL4Ttwy5yZNU8g1aICCSdjGqfI0cmVbJpXEyCU8Wt4UKV_912jaG62vA6Tlqii0ikc8UItcrgk94TEGpOEQXlG1HXsWyAryCvOMSM2F785Q4Jx2XOrNv4klRPEZGUeIbp4ReTVXVi0JT-cjc6O3gKNxN6vxzUbvPhmcyTa9UogLuCTHjv3KpcIvBOw-_pF3Z02oQE0GaJKBpP4SJLE2yZsIII4uMls7Lw07EuHZjsZoCQRg12dRle6rwba7IeRw0RJWYEp9aavT0Ttrj69dO0e20NpispmeAXLh0xxrRCKcjxAn6c5XtEbJP54_ka1FUSVY4x8IaU_pCKI85fGmHIx-HarXBtWzZO9B5O1K4Pqr3BE7LELTXaMwWQ2SU-RGsvgmDpmUZjwifQ2YgamjIJPt0UcuGWb8BTwssP81XT5mQ2Tsq1YjQmgfzeF28yeb7XhkEaBUNejSou3SuEXZ9aEuSaMz62gzPSpsSrr51QoBJpMBF9Jd7LXuFJwaQV7jP9NJawF9GT-CMWj2IOXgVca7cL_d99IMSR94vNyg8yPzDsncJZ9Dw3HXFsPfdGHtO2FaFUB3RRZAVKoHy7S1NTNfLxdtB-p0eDuu1JbcsgtULWC71E6TbPxg8OguiEgAPTXJviUAed6udruUrSMlZQv-AgRYfxYPPMXLeUIWTTUo6PKICy_PO3U5CF6VBkaNUvCLf317L47FCeEAJNTb9Uj_S67ZqoAnEG0tQG7tVPuN13cy12xO2-8xFQSpO7gg0DzF8vCD1cAcKAvo0FUEnIeXOVHVQxThLHDiXOmB_ZpoT-qJYb88RTLNoAq5oI0ZuZYvPHJ63EhVjaANKwNe4DrfAvoPpf0qWiBOH2vHxnlIJc84pRh33ixB-azK7arhetqwIuLhDo4u9REcD2avxew8rDEOTqb5Tk02hhCKX9drLYCriNdkQh3mrC3KYzOWZ9aebwOR1c-s54KbvGDHAjTNPCLlROf30MmTON3jb-NW15YyzQrVFfV1c-egUiWRwMVE3KeWi4wmicK_QGMZkdyEqZMSzNcgOZMFfUWxdUKxACHY5J_7lUZltrz9JnhsfuM7KMuEW3GMASIP8f8WmR03nleJTi7k21oLtX-xz1gjble9WzSzd5pTz9GrFw4KWatCyrLXtKWw9fAqm_k5HpIJdya9KK3jNve6MirP6jdetIUNIbN3MGkMJ8lfavyTaa6-t4hsQSmyTQn6OKwhK_PA8-KTluNMW-dpqZU2YPFYk_QHYW6EJe_Kw5aOq-zpKR3hGgoHm75Ossr23QERsVgP0LChljPzR4OQlce1GMDtRNqLX0wGu1RO7OdM9R_lqJWMlIaAa5wfvdH5LznaQV1vuGPrfpzGL4mlocKDv8ASvrxA4bm5fWBoqsfzcLu-H8uz069vLDyHgrPNse6W4Ex1BVY6By0K_f7sidbmc1FxwP3ypVv4nX_lncg6RiZzaQTHTxXJFmvVO8_L9XBHJcGkQGpEuEjx2aMTWZGJNxfaO2fKJ8U3XflYVXJkSg5b5ixTHuvDYjCOELs3fTVAy50CuMXMoCEgyZlqZNg_EJXEmz5niLNQnwQPRWUbe3kicaLzJqvZrtrvPOPcTM31Ph2-_dfEOeKNOIE2B0pvMgTaFRck_xOc7s5J2tWAEYszDz6aMXvnvzm1WH9cXYLbgZPyJmMUxeGZ70DdnueVbrNr8VA5bzvjkgjEkhks_BQprXEAZL1lSL2s0O9G8ekgFnt75JBJmSFGT0twl-t1ia1BFkRtMGXLIj91xWJb2GsF6ZN9Uknfm0Akfk1STtRbxFIeBRlwQsix5rQ7EstyhfsBXiBILky2rSfj0UJwH1NjDskXjFxxpy-FEE7KRYwMws9rKKuMQMyURUK-DbLvMmQoxekYvqu7bJfWqxj3lndGwD1sQL78cpVVPVfJeqnlAw7k_xd6QdHg9DwSlGNb4OCYdFWT4xaaltFIJfo6g1Pay7HD8gWTrrgUzHgEWfbJxcKIXs1etHx1lxYVTmm9TFkXshmsbKptL7kAaxBy9JknSsGsh9gZXf3YFkocEj1xa8f8Xcuf3zatefAeFFh1Q629b0Sc-GzfXnu-KfuSyJzAZulrP1IQ0jlOiGP5hKnvzePVL_JZGTNJrJxmtWXejLodY-JzLzUjIeALKtyUsu1ELFtwDxyadPSsFW8qvMeolLcVDysGm8NkmRgLzQTBDGR4AcipdozZmElDRTm5P6JArLlqdZCxXpiOH2x4juPIYUfRrrTT2g6emTXHz_AurjFgYn55G6xv1YGSuM5tNBXc_WP5ya9cdpBIEYj1i05DIMsvUPsNAkt0MIeTiVSPPDMgpT4lLsR1ezwBMx2kQBJI6E7rmH9f3Abn5H6yeKQLZckAAru1SLkVwoDxcTTJZqD3sZt6RhBDuuMWX5ZoB21K-zkE3Tde6caBupWLK-W2eGJSJ_oOaG2YGQxL56irxU6DIVxLuMWUTOVH5vpqeo2RlrGpXu-lJkg3tC69gXlNd55233uIkchhihakwSIxFF1Ka-hcBlKtn0Kz7CXrXam4B0sSWjc9xGRfSOaQ6LiameoozXfhj8r_GSOwoV8EMa2vIBFggFGrPEzaczNkOKBiA-xTQtdEPqmfQNznuZ-B-VX-s0E0Ew2EopP4ljZ4QMW8k6pbNX1aegBBxbxkNc5ugJhBBoSVJeEAC2Lw3iCZUnX_leWUJBp2up09oJtRWlnGG4mLAu7nYsI7blues0ZLZE4C49v2eYBmfkeyq1DBAGXu0RC1qMz5729tzLPUEPYpKS1H7w2iGHQ9P1jBBWAAfFoqgn1lYtBF1ioxL7ry6YMrvCgTlqvVRXB7zmAUlsJdPq-CTWpF79YSco4fAhrDVCmxdS6Y4arD7p26YWk8PioCDt9ranaUi7--wlyh2OTdJPHAUHW2-o5NaXXfhqaIVfCqH1sbVmNwP0BRiAmUlwK7GB_m7dtEztYz1sHl5sXmXEDcFjJtr6uozFDjEA42F48AVuZMlQfQ3eJNSRqHEThYeyzbtCdYZ6J6ntg2XS0uDHISgM4zi1mDeur6-ZCw4rGwUXvB1BWXifFeh2miEGtvRzw3sa1zBKBCGtYtRsl4Iz5Plo9RNN8eQ_vvwmfDk2F-5YWsDZbpJuSXQXy1hjDvyM7TVGj4uL9gxFQ-ZCxFl9cufUeqfEGgHX38mZoJAT2emXbe4A4byFYvWfM-NxjpbNA67ZkOWgcDPtY853Y6dKoBihh49ZAzvmEjmPixKp2rBuNX26jJzhW2OJH91GpsncHGwJ3ajWht88XbKBp4Lb8sNVxYD3hK4c-mB95WYYaUKe5_ugc-PhC4FGu-FYNLYTX2ZxLKpk_T4uEG64zBQ0NbS9y8WWiTojeQ7b4-MBG_j3VJr5Pi0T0meC623J2ldwud3DRBZXB5q5rKgofFF6WqvwhIDi8YLL7CVUJ9aOE57SkUKVrYYD48Cv8Wv9piI2hbTgXwWkCpg_tVROBjl4RYfYVlOBV4pM1G5AK73PXfDGsPdiCxhmxHlvzanAm30eVKIctRaS1xlcBqLp8CUPkgnPDlPVclMagd1CjIlN4igMnFN9gDPOUckrA0-VBlg-EKsHG3o_jNMbsvgfXg8BuApc=',
                        provider_name='openai',
                    ),
                    TextPart(
                        content=IsStr(),
                        id='msg_028829e50fbcad090068c9c8362ef08195a8a69090feef1ac8',
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=9299, cache_read_tokens=8448, output_tokens=577, details={'reasoning_tokens': 512}
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_028829e50fbcad090068c9c82e1e0081958ddc581008b39428',
                finish_reason='stop',
            ),
        ]
    )

    messages = result.all_messages()
    result = await agent.run(user_prompt='how about Mexico City?', message_history=messages)
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='how about Mexico City?',
                        timestamp=IsDatetime(),
                    )
                ],
                instructions='You are a helpful assistant.',
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_028829e50fbcad090068c9c83c3bc88195a040e3b9fa2b3ee4',
                        signature='gAAAAABoychCJ5ngp_Es2S-gVyH01AJS0uQK5F4HdAkVFJcXPiZanGRjmEPXNGx45nUyTonk62hht1dZ8siaE7v0SCE-oBFoP3du4UqNqtCmJ_0_EmkXG7sHh3pR_zuS_iEDGae9S_qM-vcVXyqFYbEtEVD9ZimiQGtLEU7WFyQq4UeLuD-U4vRhpFreMCAfen1DkV9txJijEPRL_2cTUGT47rpi2HYyuN1CzYKzRrn2qbHsgDjnPtZ8cY-QGTm5Mm0LHV9GeDh4MmRY5Lgxt0slssKI7vy3OqTWR3OCESp-5VmMR3fbyVNxkeogT9XqPfnl_9maf5jYLv57tVGVRJUEx50QvMJ9V20qbUzIAuMw5d11s8q627IyyFu-bD8QmjGsaBj_wsjdMe6adDF8hzOau3svjuouGf066I73I2euw2NpokdNA8fbI3bAHfqyXpFDADKXg7WL_zYB0eyREbWe3n2mo3KL2sLW2908ScYEvsv9VlAo6q1vByI0wfGmnkqkgBvh04Fe15ljjSkvLy7iRnOFL_CCPakpDcViIOD-yRSDk-MSHpQsK1sP2GgxHHy8jGO2g_ef2bOH4FkcYZK1oJLIUGqhLJI0LurXFnLZ3zcUML01aV0rMFyweQwbdIjpivIGaAg1BUPU1Tc8nCNmZC5aRcbixMzzu21HtW1SWnMziebhKHyN66b5skUXl_RHrCoKhFyJxSJJjxHeuUKHQ5VxvJDJSylZjHvMkX0KQ-Vn78pv-Be5ETRxR2G3Agp-a-iX0zM4HbwVyoF5l5t7g07pTrfEMP0WFJu4_OG_tsy4u53JGMQwQLB_RNYcd2n1yXPCpZYHuq8Vkt6-A7kYHW3wvUmI2cSyZGBNpwt-pL7kqdPaGyqnfhMTDzTS_CTXBBrCjjQg-RsWGu9hYon5iKgHFv-w_qGykzyPtEzZt_VWUrVm0WFOinLqLXTQgiKm0sypDdGRht69Rbfe9WqP3fhFychLwcP22IvDQsh_OenHiF5ytB1XTI90VB4e890QUI2CzsnH-8fFkQT9Bj7ou-MstjIeOQrCwDGAPRnxP8PWoCg3uYk0DuAWuJY0lYq6isqGKc57Lz1bLaGRG3oYpWH0MC6b-D2y7c4cAgOYMhOzYq2ufblZDinvBLrr9TV5jtog21xrBy22o7dbVEgIJ2T2HI2XOmjG-l7qrchcAykaosXQkW3ASIv0OpfG-SSd9UU1_1dOUFzOXGej5UMxZidzQa_dW3XPLCqVqgiDW9HCu_XCmSZo36DY95I2hofXq5mXUHT4qxdZ48y7KGiM6mllFudcdyXu1w8ZGFlU0BfzKDOfbhEJz7MRLuXL6GO0bCHqgFo5WHJrsTNrXuHNNTe2LxPPIpejVl6kvE_1LtHy2jKffOR_BcBCS1c_KLIIbl7U10__OWglq3KpDXuupMa9-fXXSn0Ko8rRybTLQpXIn1D6phbi8hhS93EkaVE-9zZZGBvgcYhPP2fa0XniiexQcX-VDQ==',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'query': 'weather: Mexico City, Mexico', 'type': 'search'},
                        tool_call_id='ws_028829e50fbcad090068c9c83e3a648195a241c1a97eddfee8',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content={'status': 'completed'},
                        tool_call_id='ws_028829e50fbcad090068c9c83e3a648195a241c1a97eddfee8',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_028829e50fbcad090068c9c83eeb7c8195ad29c9e8eaf82127',
                        signature='gAAAAABoychC16oV3bX-2fpfsHcxFWoRnoEx69lgG3ff6IIJemlYvbM8OVbf4e41oydklk1kkBbyWg2spj4puPrSV9w19NOknK00NJ170UxqM5jqvtZHcvAAdeBjA9XSyRObTamXE16W3KtXvRvyRBmBpqpC6pQGX15fxxdAESZUV6uUexSQIYZEfCT2q2aRj7YV4kCGXUQoMRvjzFE9YLE4LDNrQykcrIytjZ95hz6czjxsd95qmYtGdjMU-s4BlOvs34pE-d-H7cR3a3cQHI8SkpaQrL7bCOxZk2fYws-t0YBXEsOIRNCpX3uEany3iGgq_8jn-ggeZtwvnA6oRFtIkzpscLaU4kwhlZbYHNI_RinezdR5ByRjwSdc2UHvqoLb4a2rYmHSLLpSmvr1f9UesAz2M5AexJYlk4sDmGhMD5DoiLy05lbnbo86osBDmRpwXhb4F0pSVgPxUEadMvvr_l69Mv_VAhTJdr_iLFn3E15HCLPCFND9TcROgxPzhW7aeDrt8fJPwEZZ4fZ3BAphxP5sOzzmd3-6uwCHLZxB-51ILHGMkBVmGxFSXB3u5mr7TtaDafh7bxWQv2bpLoV3Y5QD1lRvBj6sx95B6J-CWgw0WeOd7jSgHR2Y6nDzD6XAGgg-aEK5Jk3CDGLsSqv6SxYMoY9MvT16syFsNuEki6XDx3cF252VeOHIPNPQiqBB5NRgf0Vx1zAMgAn8EYWarg8bWsJrazh_nSKWmM4gCFFAUK3Tqi2rfbx6eCPlPBYHxX73GdiHrypeAA50pqVySFxXzXgeRKghzGEQetBPzNMPykyUmiDuq3oPc_bliFQu_15-rDhEfmJcfS65DpL-_tLdtTFV4-BeAjVNsdPjX-7I1bTHdZzyuBiMr5sltxKzmHd4fLWLKv_ZsAustyfUmQnO5_reR0T3SwlY2Ytg4wJo96dtx-XUqJxWgZ9tAW8_rhwgejaH2H8zTM2wczgWVXJZxlsIl_U1xY4pSgxosqBq8a5EPrAqJFnpcZqj9ctCImVN5oElb8o4474pOhSeY0qFQgL5iol5d6QB1gNTKugU_rCgAPbHwBAvnONLJ0v3hQXncgcuIJgQw8BjpOgS6KTXLmf-5uH6CyXum-oE3JJy8EMBjvyerecMMQl6dpeJxYHlB6B0RUUzTI5bHFaoJeSGetoKH7t-L2lUwgcL7F84Wf1ZU3EUkCPWl6DdUq99aLfYLWPqd3bQ2JCvWiMVrlwuHZr_8l_N3gCWuy2t43N2nAKBBc3HWoWRJPgHCmkj0MIMdnZBiUD7IXz-b9jO_1ASYT0NhOPc3gqipzP_9lFE0EojjvqUXV1P_OiAX-Cl2cFpn7ACDQpxAGyW0yr-lgffzLI0GA6dP47DMYs0P6dQBD6XJFbvlxigcl_9GURApvAb66ITpFWMeQAJOCGdMMPZF2CahK8Riq9b0RtkSmgmmEL9SUNaMpEJBlk6j41_IdZnxnO4Qm0Fqos6RFKFbwqfxEopy9rVWvkbjFzRS_B7gAc0kH9AbFx0CZ61NZYNVnQcN1qpr0iuJtSGG-DW9EjT56IFtnt_clgrjfFuFj3cwX5ZcKMrN_RTQNgY5QhAPShSXUB4MDstvHgFhBObn-4rDl3TIFJiIgNY9lBz5egE8YZZXg8XxW7nFZpP0fmQD9a0CPdA1BhafzNcvCbReTjddrVeJcHhflTNjy0YiXrXUyJmlmjO1y0opcXkS8R3E-Md73KKEW9wJUOuEFDDr9PAaocHUsvqWPTNb_Lu90knDMKEi_NnlB8SHf2Agg6FkyMo4Z_k-T_51IGYfFJHPuGRZ8-CqK-qI8-6BRIDpnei_UIi2K9ALXGOuYrcG9A4YexW_vPg2qmoVgishgzr-ddFGOuWr_j05j7AKffDc6wqK0PNBTEqpnMKSVICOdOEBcilXsncLhjFm_JmS3JfxaM0Ly83tKhZqjP83hxrL_JvBjBQRuW7LwyYuFbE_8dAysUMI5jYwqPd40mGPALADFca0U1rolFD41tdX6LijA7Wz9JjYpfuphLiXNH5cGqTe4T_ReZAN29DffISVS08dRiQUEnw2-OMBYz_nY2qe1vyEItwYmUe2fjOgec4ClJPdRDXBW0HWVS6ei1sgOOD6FvA0moRFpSJypcEC2R1PiRqN_FEoTXzRsSAPF6pXoQIlgXxudLwitpW5xSZS4v_DZTlGa7GgHnq_dhDRdSw5GzCvqPU3CSlP7GmvxZKA_9WoiHNd6JdOSVJg6x8BGpxDjvJy9T-XB8SIKyNx2ymCVKaEhnNTh9UefBGcEXR32oYiRa6GOLtVLt_7OJ_YOqSU4XB9OEjoWlWisBxCrvnAI6URp-wxVLLkLzAPhX-O1sbjcOkCillvnJWyDbnL12JkI0NsvenYonUdprMbVKcX68KdkkpgmKyMICY7eUKpZfWy32E5stRQFUE1GMZ6wYKGOBFa8a5QiIwIx_4IAU44BZCqBDaV57H9KAlsHhqY0K9PJa2fetDVGb2MKohfcEmF4lAzmHKiu22OINYHBYX1LZulsVrcQUj6zSA7r3GEEP6K6wBmk6i1SuLgf4ze9WC2pyb9zemaZ7dHbb3btZw_xAk5a-RVoNb2hIXfiX9clN3BkMw5V2vbpDHaNM80N8z_3VC5uXkQ_v1543ZFWvxbdvEVHlR8P9JyG_Asts0VrwDnFAo6rTGmPj52GJcmhLVAgZ0KPDrujpGHu9HTV7sO-3KvqxOMHYuKG34GvpjfZzlgV8GzbXtpsRk2E-GJPKLfLN9KIHYMxdfkaWBurYvea7iMYe954Gcwehfvlk83foG1ez6FtysZ2V4eLjg9IcVJVAWucdnUWyIIgYMocgpS6ESkO2wRs6pUz4mg8MT8q-h03BJXmWiJIi-4_3TOhz0owLKMza_1IljVaMAUIHp6Kd9yEPohWQo3uyGulXU-vEsSeSkId_sVxLphe9yuimK3CtzU7FBjewoGhaj9vnTdv5_abDRZ13Glp_b4vpfUrr37CBAX_RwJ_mTqGhbv-mPuFRVD6ESjlg-JrJDCUY605dcyU_0hyvjSFepiHQ4FCEHzL6GNSfR',
                        provider_name='openai',
                    ),
                    TextPart(
                        content='Today (Tuesday, September 16, 2025) in Mexico City: mostly cloudy, around 73°F (23°C) now. High near 74°F (23°C), low around 57°F (14°C) tonight.',
                        id='msg_028829e50fbcad090068c9c8422f108195b9836a498cc32b98',
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=9506, cache_read_tokens=8576, output_tokens=439, details={'reasoning_tokens': 384}
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_028829e50fbcad090068c9c83b9fb88195b6b84a32e1fc83c0',
                finish_reason='stop',
            ),
        ]
    )


async def test_openai_responses_model_web_search_tool_with_user_location(
    allow_model_requests: None, openai_api_key: str
):
    m = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(
        m,
        instructions='You are a helpful assistant.',
        builtin_tools=[WebSearchTool(user_location={'city': 'Utrecht', 'region': 'NL'})],
    )

    result = await agent.run('What is the weather?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the weather?',
                        timestamp=IsDatetime(),
                    )
                ],
                instructions='You are a helpful assistant.',
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_0b385a0fdc82fd920068c4aaf48ec08197be1398090bcf313b',
                        signature='gAAAAABoxKsLzYlqCMk-hnKu6ctqpIydzdi403okaNJo_-bPYc4bxS_ICvnX_bpkqwPT_LEd0bZq6X-3L2XLY0WNvH_0SrcNCc-tByTcFHNFkQnE_K7tiXA9H5-IA_93M-2CqB2GVOASFmCMBMu5KwzsYIYlAb-swkkEzltsf5JEmn1Fx9Dqh5V0hxkZI6cz35VsK0LEJSYpkJjAMcfoax1mXlnTMo7tDQ_eBtoCa_O2IQqdxwPnzAalvnO_F4tlycUBM5JQkGfocppxsh9CQWpr7bZysWq0zGfuUvtuOi1YJkHVlrqdeWJGDZN7bgBuTAHMiuzx68N-ZrNgQ2fvot0aoVYBnBDxJFbr82VJexB-Kuk_Zf3_FVm-MGcQfiMxvwHgEYsnaJBvMA56__KLlc3G4nL91fibIXbh3AZ24p3j1Dl1V3D03LaEdU3x6RF7fF47y5eyaFWyWkmPl1RwiEaYy9Pi7WHuh-6n69ADGYWbv0m4mgvECbmvbBIIkZWr4y0UK0B8hbC-Oqz776Taww73OmchIzgkg09rIz9CfoKcGMXgvzbpIBa4sME5BQ3mQtfIdPLY7uUIwya4o_g5wVy583MQva75jNsR4A6sRVW9SgVEWusMJPHv6NLzHCdWehp6SBcKuovxZayoM4KQrIvUMNlUkrSR-euoBaa_WNc1HeY8ikKolX6emm2LhRzXH5HssCgH0g8GUvWilYx7U-UFSB0r6yoy44_DzsyH85pXN1ivsSU5dGIBQgG7WiN3bfk6oBGSrz4XkBLiHJiBX9ZUe270TeDNfpgjmKO34_k35zviIUd7-kVY4EsJGGijEhjbkInFwhilyH08EdKvYDzrzpKJIHT235drt3eLTKXKEA-g3iW-qOMqH15KPk-slzPNkE8yahWEkLrYsqGsjwdHVXiKF77-i8rwvDWOf-pOs9d3bBxily3t-22D6RsOL6wFYQS6BsuroKdlO3b_0Ju5E2Kq4P3jxtZ8jnG9D2--XEcEB5x9yX_brfdFuFHrF3C4mYVWTrNN3_S9V8zUp4CdIh3EqAuSs_QJPJuN-RNlorK3bwYqOymgNlcezKIqxhWnqtS1vxuxC7msRlJRmzTN_Lg6XuLRNS1uIp8jmx7TcCnDx62ynYn2oGCOCLSspK_T_LVTG6js4Oiw9ZB5A_I3TfDLrtnLRh7pGJnAv9nVnfYd4Y1czSjhPui5LF-FvLOlzWxSu_1Mo56QA1BIerB9lCQsDjPOkLF_XHOFLWGLQANx5nQ2wlbgBNyMcPacQowRyn3NncjfzlSLyaPijEZ0HROyL_Hff5JXCMu5-6muvxQz1TirmbyjBbLjtv93JpXrVvby14mdXdNs97dMATIiqpwF2r0873_dijDKRxIDMZxqFB2ZBaHJc80khjG_NaA_jxv1GEqVWmllBXBz-wUDbUJKtNtI86YmcZboZIA71V416UW94-TXbtyQpGlB8tj_764sn9fKitg3vCqC42mr5Kj_aTzAN34BXLykkFWYl_AfVL5PRbJXc0Uh0GW0xTH8eD0hvqd2Xsr9eCoP0nGM6TBNMCl4T82wOhRy7jelWMpt8LBxAYkw3nAlVVOi2puCoYRaRFWNQnLcO5iYBF8_rg9oX-cUsBFepGGDmoOfwUmWLlYqNZDho3AJ_SL3azAVJz7lqa3vcFubrRMFiGcee6sHj0HJI_2N2mZqBO77kEbXrJ6SiUV0EXX5vrjZGzpU_wZ9G8AUz9Tdgistq8XLVsMC0uZWlbRdqD6-UjmnsJW7XINzH6MnkQwPvbduRKF4ywViUUbKVs5XRVFUQF5gTdVMTK8mIIppJx6fQRfZBju1NuNrdTDjd-5P9_QNBQj89_Y_N1fow_676bSvYrhlrIXVuLGy0-RuWezuqEwenIZ_U5wSTp9remqWzeuolwKnF7xG_QlcxGOgCivkRvqAyDxWiqlBhUtC-oPEQtychFa_W9uLHyBhm4bcSUz9KvOlUTt9fNYgvDWFciGCE7B5iPz2s-lCS-Onq0ZvUiZY5nB63htK1bIMzB5lc4N7XVh6COcSIArGBnXKARHdIenJ9vYBSmB4XBrKOIU6SmNNM4fq3ZFoWIc4gsS8L5LZyhTX_qlmY2L6znek3XT0Z7kjEHs5qQ87_sw9ho2KaqNSjMalbUEp7L0JlU73szrtdpMkmBk3BK0of4Nl_v_CCbmYWW9z_rsNpTpPQgUHNVn1s38DX3cesMqlzlBOky-rpLAj2-sS-Xj6WZWBs_8n0lLFS7FL3IpKzveOXE9eV4zjJSZ0y74b_g7u5US3dT8EgSEeHa_pGOMn3t3J37oz1pZcSufD8vjyG7wtGxYUGn8L9U3zJHN1VdOR9id5VYOo3OLtMjCrSqPO',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'query': 'weather: Utrecht, Netherlands', 'type': 'search'},
                        tool_call_id='ws_0b385a0fdc82fd920068c4aaf7037081978e951ac15bf07978',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content={'status': 'completed'},
                        tool_call_id='ws_0b385a0fdc82fd920068c4aaf7037081978e951ac15bf07978',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_0b385a0fdc82fd920068c4aaf877808197aaba6a27b43b39aa',
                        signature='gAAAAABoxKsLhc5YCXvcJidIAJvFyzs2T3IwW0fie9oMN3Nk5fAcAP3apWArzw8DdWjWjR0tn8Fpw_H_xATFGktsCeA5nzkcKvdc0Bbu2bwMo2QUkQZfFcLHqlcNnAcvrw49XpolGFl-mu7hAyP38LGGtjbTBNRh4dHkd-hYZzy3nYd56JQi5GLS_KuxdU78xUW3gNOtAvrseTx1fcY2eseUcNLm8uDi8a_qDw16nFvuY31ZkrmuVCESawkppmxrhGFVg0Y99dgyufnSVXXMKyE89tmXMc60yZiaB1i5cIJQcZMkwupod7yZNGqmr1GtFru5uJq-bJfGx7nAEs50jUcu-rP-_ZbvptkuADDC-bfzFjaeq13wCih8wCXqDWqnGjqIHlFkBM6agn6VKOcuDC18L3caqcH3KEYT4f3TGwg_ZZjsiRDdBC-saqIduaAjjMDqMKx9XpmreRq5BLfC7fPjRykpUcWQQYbQ07J9pe0EW2VhZwoGtd1u96fmz55MzryX4VOWIwDsUTEZAoCzULvVrEBnzFqnfvQwejBxJX2XU4fIlOtT_XpOcI2afolh8KgitzHHpJ8Dr9ELI-Be2KEd6enxmdaPhgYUif2D8ZCVfOoXZEmrFBMQTRyuxtp9H0U3zGamEYuUxRavxkQD77HhmqWOSr1Agm8pWzAN97jxJSxxY4BEnjtrgp1mavtv4G7VHjrpNWrL-smZEWmnCPGKVxP9afrdSZYL-HXKY9yO6__0PR6DdX1o0JvUq1KFPx2dzag4eXDxb56HI5MKNr6J5P8Smmxxwoelx6UXEKw_hyFWMmPUHYD5Yw5dxrXeYmAiomYKFpG0bxVbuAb4_iAVliHkdIsOBcWoix0KLxmS-4RJnikZPMvDwLDWfENZ2sh9_RrQbuMBAgjHwlfWM_tww0ufm_aVdDZ1CULJ5Ki3ZxH_0oIRRyyB-a25q3DARnVzutgo32H9X6qjMb06ExMn--ndCinBglTTGvj1QOIJews6UMrcKj5ZPTc7GyPbHXvdPmPdIrtJ0wCqFj4cgNRuxjiaZDSCqmEQERYyX9Fxu8tY4f7-Fxje6A_zflqrIyhLfzo1iMaoNbba4HNkzRMWba1L1fC8St8MO4ZuZTGs_60FwzSUmBDW4Gl0CcRAdY39BE65uEpKGZeRqDfxvLUelG9YlJTowqN8hzAYShzcPPkgWk_s1AtY0RT_roregPuQ8PQayvHcJzKqnijOIhRA9k6LjF6cnHj90d6fSzTYn8F27rhufLySe56n9SA2WDWhVcjsFEFAcsL461tjiQ5U0mjaFdBQ5H__s09dhp0NzhE35I4q0pzM2KI1YWgLnwlyPFnnfce9bbL81jvbXw8DDC2KfZVOGU-ZDdqIqF0UmwNyBaMYb4SonrG8vrj5bFmCMPSFsEeuDPv_bmD8HRx8536b30RmYD0K38Wf6-UoatMxzgMpgmwsBP6Wh0HCpFeIhjRsJLxYXeoafypcKJPQgKXJwuXVLi4iejXkrbjBdc2Sq2dqIVzzUhULLJSPBYouyjeyVSbYYp9WPoBNWj67uQsX7OUbQN1_qxopsPJdqqQynJIAtULNHjKrDA0GKpyZ3OUV660OkogPAWoxTVevRemwkIJZbr2hXyy0Nx6Xc1Vf9xC0nPclJ6VXapdnjK69bIDHxDUZGCh8UZt6DbcA7azBrugcXlbaMJzoHWkzmusJoTh_2UXRjrS3B33jsxf6LQnUl0s1ETo3Tif868zLvkTEtfo6btbND0FPDFFQrdeVlW4mUWEOJhPeOmwnDeLsafTfRCI_V_xTzgkpQxx7pVZt6mkYZ2qDTE--NhqgFfHPlw-nC4zU6klRdbaO8284QGlbJvHmdsmHi4AtMSWAf-_jegocmaneM1wUquNKoy6hnbkZFul9qV2c-_L077uC4nZYNjRay3lT_3giVH6Ra6WnBovt9ocCYIwSeygVAyqBHxo5EJpfyJhNCtak3bl-CIz2TraYqqUCiB0h1fyxIF7M0uENZKALtwqRVHOtEsN5JVotgv-8YzaBRFs3qvtjQn7eEcw-zrIg5fwMP7tDi8O3TXl6qPVWTCHMa1wkfb7OkfuwXREognLvO-3qdRgxinodvKyHn9XbsUcQMQjPPFMLOs4wpEhTJpcIFPqtR6tArjTT3P-T21mc8B56K1wXfEDvpU64XQ0HnfZWaqS1TbDyfL2i12ddhhnxbCV-0f3lUGnZVsfeGEc4FlST7iqUguhwPGb4mBpjBVFu2dv3DMCIPHew1v92gZH1OJqZJJVDUpu0vvFGTqxHz31LSX6lWa4gn2l6hvkT1e4aXkjHg93iy0ZXMpB0JqJbbWseZY0LDYzpH9noHq626Q9H4ZEKPo_MYBWSS_yH-V2_cN6a4HarqhcRwD9oT1QJ4_4AzWeFIrCZlClYbA-84H1CbBfQjgtRh6zTZLDHM2In2M8mKGyFSfeIhMHIcfPBTpG4flLBmTNrwwbuOP-0ss_bb5gxLeDsgU5xjwfaUzOWXudPJOEorz4t6Oc88MiRH42troV2fun6Uf7e7j1OQSGtTQ1kXf0rroz2ykDfVIXCefX_3io_xJ7ev9dH54CNlARSF6cVpTqzbyLWkA0BJeAVYcX2JW_AT-9VYTOo1Vixja7KtMAmMMk1E08japeGnoAd_a_4-bEfklFTChseUDgZhOt5_XtBiuQdPvJDorSQWQl8VCPKdMATr-EdUiZN54GSM46pdBr6p-Dg7LvB-zBAbTlm_6SET0O0k4RkkHxUCtgRMZQ52aC4brcym771djtWC-BbaR5CefibOoSo-i-BP2Zf-RVaS_MuFar0dT03zXdb0XuC2vuhbVPPF-7gsJez2dufEiU9LBhV3__zTDlFc-rGwwf04Fh5KuleNzr1QNyVPH9GZSS8jZkja6EcRfGn0X-oBr2oRLyxuL5vWgOdPadBOJGjIoRnMhCAxGla_gD_5m0qwF9CtWWv7ugW7YpATe62zE0O1icYDPwaXGovzTOeRDRn4BfJzgzwLRkP3-zOgF_09X41umrq0TCnCujXe-JOhFuIcYx8IxOb_cCcfGRqGXeZYP7z',
                        provider_name='openai',
                    ),
                    TextPart(
                        content=IsStr(),
                        id='msg_0b385a0fdc82fd920068c4ab0996c08197a1adfce3593080f0',
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=9463, cache_read_tokens=8320, output_tokens=660, details={'reasoning_tokens': 512}
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_0b385a0fdc82fd920068c4aaf3ced88197a88711e356b032c4',
                finish_reason='stop',
            ),
        ]
    )
    assert result.output == snapshot(IsStr())


async def test_openai_responses_model_web_search_tool_with_invalid_region(
    allow_model_requests: None, openai_api_key: str
):
    m = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(
        m,
        instructions='You are a helpful assistant.',
        builtin_tools=[WebSearchTool(user_location={'city': 'Salvador', 'region': 'BRLO'})],
    )

    result = await agent.run('What is the weather?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the weather?',
                        timestamp=IsDatetime(),
                    )
                ],
                instructions='You are a helpful assistant.',
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_0b4f29854724a3120068c4ab0be6a08191b495f9009b885649',
                        signature='gAAAAABoxKsml4Y3hqqolEa8BSvPr6mIoOyAbWRJz9FeLHoqX03v4b6Kni2j9HxfifAm2cHD_m9-b2nOHcwDPOeJA28LQpl_BfOakn7h4saDElA_yz3WgVfy8ZN_oTLQz2ONqptBxdxCaLMGOADqBJ1tJ93B5s8bsFNZUdGXe382lPpCNX0aKPGxd0e-UBAICRmjGVnKd9cVzB8jhtQWBrITvMOLBvi6bE_TqnpXWf8-rhed78mFVMRweh6zAzukkJPMAjD7QfUAiODvD6oynwU6G04UOJoFTItUsAULPfyAw-YZqRwfcfMxoiLAE0rOOj9V7-eyp_J7DYu2uF16jaOopnrehFDJr-0pIGMFRxMSyFp7Ze7z3gWvcCOB4VwpSFao12nozedMeinybf71wo0750TNXXQ9Uye6qsUxxMamqcNiB02LjCM3nyBQ6FpWa59TD5O5UytT5FPOWSflYEhuiTFknt_JRHbKoeqVTfe_CTeSVlYBtiW8ouhkTHAAVI5lXi_mgvUMHINTYw5MEilzBSPunuMRquopRjt_07YMKuwPDQ8o__s1NlyrDAYKLA0gPzse4tWMkKREcfxuvU948pEJwVN9RuKS-NNXI2KiKKOAtPoXLbflAEtpx9N9PpPdwvz_z3yhF6S1_D_9P8OrSdxd8ldqvnqec75Jwt-a0fuQvRTSC3GsYuhk1Cb1aBvZdBtfcwBd2CXRuDUEdtzbLZ5AUNBy3f0mC3ITHG9aSpuD4GUHQDTjF_10-Qr4Rzygnj4-qubY5ibVxGtHlXkI0QzvGMVf7obhHMNxEQNaJ4k2dKddRJEhrSFWmAVYdWbKiZp-Dwx8veUSlpwMu8kLfGUq64MBQOApf-Srtry0eJAr3cTBqzmUIU5OOPg2C8j9SbAuTLbbcR6XeWizp5fbxdcVipVRqqp_PJptIJhaAUpHaaOB9u1nZbtlKWFJhJbrZzdktth5DNim4ayYBbBX1VAefwCugReld4C6QtB5Q-j_Tt3dug3Jh9TJmkhS8pJE4aHURzbCikFohJHAukZYgMY7wCuLWlahQ8snlIj8kbhPP-l-iH-e0xM2vFDF8rZnfYblnDLZYQBezfiZ4GtvO64SB5apQuRXkxExfZyBd6Kv-WhAxhPGoQdmTXfVEXePJLvbzAJcAXYpmmzt1STxoxR9cnaeLL13fFXZ4DGXe4j68-R7xCC52jfoV-l8JZjI0NDRJ3Mx1R26bp-lnvoertQBs1c18QHVShluHtH5c6V3j4yOMgG6cA2aVM25i6sjhUV3iltijuRv3E19ZlzgVTtrypeCVH7ab0PQ3Qki28mFI9s5M1z1TSuFis1qhHwf3r0kkmjLXIUbXAnfJkcv50tlcweXRTLKs0ZX0nxsxiZptBo95wxqBf4VaqfOY4NUNAWVoZ4AS5oSIgjGfUZtfrLisWmX8NjDWiOiENLmn9fCCq9nxDDsaucnwNhsMZo9jJqJS_99kryMXi0yGX4GManClCTe31Fj5zOrtRIezlEILiTla6fZwvD6vcl8GWO2wuyEY9zsEvfjyuvcU6Ernvw9S5HFPnQ-FnDxNtSTe1A8IHTspfEROnuSNVCMs6j02eFZMbXFKMaVi6LNDD2i7SYn3dMbN7aOfubtjeilMpIZ20U-J3uBUsc0rr8s4b-szDB1lkmiMvRDVY8YKNqH3iJFCToE3OibVwHeaUnMmEHJkIvJvBOX4hSwmAMxjZArusTnlYnLE2raAD707H_Q5JhpWXwtgFPj5ra6HFtOjtbPtDWrDn5_M180klxvF-JxfSxSl6U6y2FYeou36ttPRprWJynfcPSPY_sdrB9ZupHDR5zZy01Uby1J7XXOZt5an91kuHr0qU4bQJsq6AigFQ72C_YxpDNmQXcy5awJDBlXv9SoLiXRcTxpoXgii9alV8MeorRbc23O0fP_O6XKUso-lp-e7Q6bOqzV0c9K3imYUDzM9cqlvEyUGMDLlWzEvVGSwpag1CsLCNQ5bPc31W8hc-2WXrlltP6JZ9gYpcueL5AIud6RUTSJWg4Li6Th4ZGNs5cqh6Nk6oSu07P4Ie2JJ5bt1tAJbE4EupK3NVzUpzYzFdPrQkBY-VQ-klCFq4icnvlpD3pajYv9OoCpo0z8GfsdLeJlefIQ1NejuMg3EwbGRA_OEWn7sJzR2RFCYkt3YIuWRJb2UzIzvWhZsLxr4UpihrsieNKggGBh7nDpOXeAZhS8pGrNSlKjfvWtvmWG9NKXSpx79dNLSkumiD3FsQjk-L1Ov-K5WksY0yJTgc3ipgO2UpN9zolpXhXum9Uy8UeKLlB35cCtte15t_HSogTh2HDkc9SuCq4d3adSdstdXodr9jLbST50cHYn-F9qmkKiqV2nBzxW-9A4BB9WB_tWEoazKWYHtIdmjRm6O9NxvOxYuWIwhMmRf-OE6MHOeH0emhuTFaeuZ4zjbM0T9peRh9shiUw6T1NT0doCgfyRAq1NL1rG7iSc4jxrc5ahP0gN',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'query': 'weather: Brazil, Bahia, Salvador', 'type': 'search'},
                        tool_call_id='ws_0b4f29854724a3120068c4ab0f070c8191b903ff534320cb64',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content={'status': 'completed'},
                        tool_call_id='ws_0b4f29854724a3120068c4ab0f070c8191b903ff534320cb64',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_0b4f29854724a3120068c4ab1022f88191b2e3e726df9f635e',
                        signature='gAAAAABoxKsmZCctfduUbipds6REy8FkoOiADLcLER75WMHyO7PtQt26NIhcGkiXReZWucbDdEBRKk7_g9PUuu9g-zEBe6kIQwm4lwjxCGPy-rQdmfJpueznyPJ14Ood-wazqT9a8ab_BMFS7VLonsOjZR_b1gxcx5yO62oLvv1GnZfkEykIgRbGIBSYDWX6I55Sfwkf0JRaiOFgeHoOvQ6f2mdb5UetdJwbIFgRh9Bk-_l6goC-ONyElqvPxrh8zlLxqEhL-KtTVw6TPNW67QeYxekA4vdXseYT4W2oJMcMKp8aIfxYr3-ZWSy81UqGPD2PAfs1DoOYkWMHxt_VnZjLQs0qkO-JBPsWBFWEofZC1GxOIT6gd_dDvExBXkaFdNH7xf0OxsxSMWfyKSXMlq3kmVsDIN3hKImwfZQ171mkFEwgwIBeo4XiY58YJXmzyNXSs3c82gAeGpS8cOQw5shjC449uJZkixSaXmwOKwtm0z1MOVAp17QLGeD_2YVa-DZUy1z6xTqStuZWnwLDOz8HPL_rW3MXGcmC63kWmcTCsFngwR_IArcTd8lsRAXJghnEdOZDYgrU7uc7bbqO8W_PyzPDDnrAbcwo0InMqJ2BZErMXOmy2dEm1jlJPEn23PL26k8r_sKNCZpg-I-q8epjbF225NJ9S8g_vvqLsyCzo-WnHPHaFDMfUhRxU-ylSReCZO3pcjNJXAfmsNiBs3g272BtvNWn7GpDqlJL9aB9Erc79CpLghKKV9JiVRsr79aW-JSzn9gJET3JteU2MMCvRxv3ePPkmZUvQdKOmzZQMwQ8j0FQHd--4qMkXDdAz-lsUjitCKK0z2ES0oSnWOVVPoR5AVIUCSfg-yGwBWhKv6qIkMTsCYaCaR86j_hGlCSxNqYdbMwy7sr6nwqDmqgmcsiNkAVUAUeU7LLXmVfGDR9InNL3lNCICpmcHMd8YJO5A1wFMPHFgfXt3o4CZP1ZSjQjQuQ-Oh2AfLaAYSNbU4y8JDtKPiini_rWIqH1yykwV0Xt__QvQtj600ksUqij_zxbKnZKy_u3Ud5E04bNgTZ0Mq9ihVtPBlcDCtWSsp5U8Sm6JL0ZXV5XaT3CVG3T7Mj-kKs4yHHOLNoR2rKAGPTA6VRzaJDNO4goMeE7aIqWKhFTYMBcKJEGD-B2J2J36iZ2RNGo9JbxmUw4ZPMVaPPulSfpLvDptYEN3LX0D6L4Xu5iaW900EQ_Ym60siMB257NRxfVPb5Sg8hqxGeKKgg6NGa6y-qyVXvqjy4HA-ODvHLbiT2n75fTD_OE2CX1FpLgmpmKkSopjT5G1vv5qtXqdhigDy-l_b9Qxwvbd7XXD72EUVPzDVwMDBZNeJkylcCecaRVJZRnhmOMkGbV4WFrMxjy7eoYrIBQ6zytutBFXNkAb6a6UXdTrlOlzclPP4P81sp3J6BytVSaLJXCIpZ3pAM9aWVzfavRW22R-rIMbmCWT9hq-1ZDfjdglHN7yowAF_rjVGrgl02wsh8IlLKfJreh7ughi9vSk1WMinlsiZfZynp33IfB3ayv00a_huU4oSKXstf1KaeQ1Z8L-ReCdPRwDYaLbP1ZT7BQAbXKgIjUsLdSiU3MmW8FVBdevLQq8AUUKsXxfQLS4TsjMYTNZ_8LkMcVeuwTDQTBYkBdyTl7jawXy2jujxDJe5mK3ZvvS_70sWokuPXkCApVFkJpNRDdcvBuoLG3g_KZ7dA0oQW9QHkKpd_-FEuUZFnL6-ZhjR7pe-EmR6gqJbuQVs19N2qho2pnNEe21WqAN-anBb4H7QN2V1ODJkW6vDDRH5sV8Ya7YYUScSI3TUASWH3MWapL1_-lRiXtVIM9Q8leFFIO_qkr8DFXoDOHp29HNa3gpQkjOqAFqX0VLg1Ub6X6C-kUbXWMcYIUoKNvQx5-Yhy5Lo0N6izxdE4Zw6U6Lfu90rA2DWeQ5-iae79H9yUy74jZw3bclkJFzGkydXWIP4OkKnDPemIKmsh28ovmfgtz_gJ99SlQDBmI6paH6P8wmHd7QvDQkMBnuACOnTnTud_MqdNUR4-qtcnPoNkFPXoTfYJNDDBkxvaEIXylqKK0wPf9aBsICsvB0N96nPpQTYuV2YHfIr8PagOi8wWC9ceUmDib8fMq3xgClujOcXOPk2Hh4Xuslecn315m-SoLjRg-dIdmTjuIyT9CrSdXMto5Jp7vcPTsRPebw41Tf4iR78BOTuGhbe_B7_WDm5FH10EptF1e3GZ0eO--VdgqLY3T3ivuoxtXIkTvDHvLHqNwFJIvH4ULUAIx3UGqJwE84_OqGwKBRT4UuQRm5wwZUZ0teyzOQx0cp7aKhsOkBzKY8jVFMmTBKin52ioD1inMiyBUYICYwYUngdYRmE5Qx7qzqB6Mg5CSW_7TaXuZFNVuVnitQp5uw2RrOlookLqyKYIQhruNjaUAvvDnhhIrTjh_Bi7f-wv7znhbJDE7YWy_zC_ufQj9VfxJcz6eXKu3fXr4EKlLayk2nwO5BkwaijetPdBNs4SOroEo6WfvFgVtbt-c6kkEfY5abo5zK6OPVHrpBVyew-A53SA0bQNptBVMNkZDiPczaviF3H3fnkMQH59RhIhMV9knjfCbAhP5BTmBFyFIXjX_ErOJgb3RtUObwjnifMNwN2hIE_-eMqk8K-jxMrT7xNoojwqcCgmzcY5w8hbmA77xW4ZnlBuTZORjFhppokfhLPcoVCcbt1AEWLc3oFYhquugqG9WZbS_7p_pI8C_zB4Q4x8MTn7lO9RZFufBeI9iTm6JP95asBuEafpQxP91ZAhfiU93UybWsoaKQb78PvjqwwK2D-LRumK6ftSMU3LNn1MBmiFowwzOLPxrkN4dzqF89rXEXJCuqS3jl9fEwKOdCvhpXyVRN6Kx5VBxSrY8KO9ItwWkrjHF4cWCTRVNePbw92TzRnzgLB4aEZ9T5TkIvdNgOyCQYSaOZ1TMSgO3a-i03avh9KisZcyt-gUbD11-EJmt_KOSeK5o-Jn3GmUKnZJJX9hKCOWCmN00qv8DzYCfIO9Bd6kfOXAqJJ0RFDHn6a4VHv4NrZNyXQWrX12_V3H4oHVZhDurhlhhak-6xoSC6KWeHFFlU39xzKx-2BfggTfghpTj4x8WiObhHvg7I6OY67vzfyRtJoA4muFzqq0c-RJ1QMvOXLGDEMJMSmuXxT0GOux0GvkB6VB4snKw5ZWdzTdm-maT6LBL9POZ8f2psW9CtE9tuzs1EfrBS9SHn9s_B6NHRCahEwwaIRFePU0v9mT3hhQoq_CawOykzNVGAPPAKyA8PNZr5GGmdmV7v0fWppgHUZA_sQPbq0XuxgoQFLJttwnCEf_mkS1zPYMYBv16U9G-kZQ25-rdHBFyZG-Wa6nBCSk7lm6ZNkDKSN7L-lBAVgpPgzDvXlCHaklZmQXwtNnBSPOZ3yO2-MBcDmSyoDbXpdM0zYZhMCyv0vMf2mKhEP91a2xD4tsp-Og6gAo0AXgk6Ge_be4zhMaUxm_NdPGg65mkaSaOZqCuevYVh0En18B7x2erzzUAMuJoo5C8ab1yLVGZSKNda3z8j40JeqcaYLN-yS4RaGaNdva_pmCq0dXYadIjaoivy4TqnHig9uJtboQqBevHPq2xXdsSutQOyEEexxjYbEz1USu25bTvog4tJs5okxNWDnL_0vBXZTpYCGdVo2WcMJgwqNBp-CPoZjMxCQ9IM6iS3KKETc9U46ksBbN95ZSeRUoUUtO_i0AoBsxE9A4NFbK9Uox2RGcJxOlC9HM2n5D6LmOyIO5KaYl16sfmURTRlcNpgTYAvat5HbfDYMFrH9EgSxu0y735-2wvZSuD0credILM3XFTyBmM7-278If-6-QaDX7zV9JxJaXrXx92T-srNH2Z5DLBOJDkl7oo1lVGKcFAmEgHjnkT_rPt8DvU4tlh0eI8HzSe7B35oA02GJE17hiWk-_VOUG2zNaOaesGK437EOzcCcc1dMZAtN206qPtzDZsNPhQNEBUx9Ta_jPG6waGpwihNxVfhwVvrR0zFUy1IspR9B1ONXttsi7nQ0YAtDSJaBuUgwwtYk2KL4QqRAixv_KSma8mOfuxs0th-sTyFGQ5f77q71ZcLUeYqVqrsjcDsh0K9pDvj4-KXcQXgd6EzY8zfh7VvXOHIr2aHBcHk1tw9zjYAR19sP87lo7YdVNrYlB09IkCICT9N1RSWJHUsszCvP0oBSmdNPfelx1CvHlClrc2qNGcyalsF8hc4wnG3mrYIC0rb4sHLc6Xp47g7vWnXH1ud169K4dB5YwnLam08lPwSYJwqculJw5d_L2egSoNIdYGvlvH-4prN6EkkyiqmZCHXYSNoKorU-ce7cRpc6mbxxU6CLCS_1FhlgfG_mZFP-KAZ3b-lQVdimYcudQeCgtjaydeAcUP4raEP_Wa3bhMB-GK90eskPs0cZgeRDvwohATR8ynHvxFCAeoiQcL-3bQgdOhZxY6r8dn6HF3RWWaeA6o4xS0XTlxecl4rOXs4nJAvn3jGZ4VmU9qkYcoVBW44IkLnbx0q07n4rRiurI4596rknVRJwbeb--_d9l9gSqn_ZwIHHyO4tk9np7I8yMTGp0j3ea_GbKrss2_8gU-XDU57ihgCQyOrAcyyfljyHTE6m-upNK0glJ-2m9r0ktOToCN-6ve4H3trSNvRL26rmH_WV8d-gwsF76cPYdlCZu46pC3Ib_R4sHUeBjg39ilY0IxUTOsLz-34NuMeKKnaViX68pZw1XzMLb7ZJOYhe0AKKO4Yrrkwpwlqvbpgd369PENtcqdakdbn44wKOfp49d9czQYQcYlRK3L08MhGsHXuDTlUcqqEYSDpwM_D2__AicfRazviJzdWQQMNJHA_0COIuhQ4c0dbPOOZqCMM9BxQe69fNlTfZEpFL2Axh_6-TqEXdqU8CO2fYScvQfuXZ2AMbmit46qlhUJMj5082R_XYNwIR_b-QMqm0e6aI_vZRVw8MwdJHG73Z_u4whBIR36VHrrK1qUYLxC2pYyLOwHlPEYlyN7HlTs6i_iJ9z4TQuK_mk_b1bc4-1XfgQUU8ZfjYPNoQNII_Dtym-9k7Ukv-pU5Nk1lItlLk07wiCcKMlui8Y-23K9mb03O38x9ZhN051SusVM9ItehAp684sy-kb6MymRW0LsXXIPdRc9LxI85RZ3aANfAtMaHbRov2jpVvZT4OQhTQIJLg3656y_NG32DJvFQoBLEgfFCTKYQgpKWmbxj1gRsVDrdk8EBF3rz1ohyUfxqyrHSYM39YGs2bnk9TkvaOaHOluV_ZoY-qIDysJ_p1eKxJVdpF2VCxZ1ctwuKCbVx6pl6XLuN-g2KaJnpgxVcVbrnxsgLrh5OGeDuXiBFYeLYaF09wFBHTHF0naw63TgB8jy61c5r7_y4DVAiicoSJ3B8SJxEmB5qgXVse_vwmKOxvULXcgU9XLaONbYYIUulkSNOSK_x_xWnVRL7yWHj9xMjWTvBXgVcux1CmehPPQ7dGhooXgzCoipDZ_y_sRl43wYZiaqG7Nl79ciyfdwi6xKUb0CgLQp1D2Q90bHKRUV1Y1IdcIUl-atTUcMGYDyLKmYQQ0BWvqXeaZtHra_yDzoIlB7rR9Hg9agchVJsUA46egTwwvlHdiYPIxJidKAQFgpDospYReegQxCIZHg_PI0FPVfXBfNR2Vc8fIrXiNwzPi4jvj83YmDTvTJ1xBLYDao7QzDQUjkpl09EnP4UoGlvFYlrXH0Ev1sWz_svhFVAduqJzHke7BW5b7gYipmIqQCvPgehCMuD8-NkaEAtE613V6BLPTu51IPtkvFoS_zSRCkLnspDFVTeDToBKQlN0-u1LlMF9f1dQDPxBE8ZLacKFP2F6lezHhikzuoJTyfCzF0xT4nn8alqzDzRV3K0wAl_4NKjhwSHz9i8MRxPo1WEfO8Xpt1aKa6WIbZ2rr5ayhX3H4ASPQ7UDoMNrRZP82lcAerRb_j7wyL57W6oE7VetxnmbexD15h_7LukUqUNSSgg6D0zxX2C23EhpBaQ7Bw4Va_costesVZBuYwEig3VR5Y-9WvmN0CuaeE1oZkXJ5zBCBgO5F_hIESxHP9zx9Z4fs7fswQDJHaick1xpSSZNDbBghUqlswGvI4TTtUWGPc5R1mf9dLQDF6j5wTo1kycMpfXIUF6hVqZRlKHgP4DRetOCsAgb_WMW0b_GCVyK8JyeZsTSXN547g8Q6WMRYikbZDP25hglrI5hU03GLf3m2WLJAd4eKB5e1nlDhIqAGn289gdttwfe8rUzB5BhdSZ6BcaWAEVp64EHYFmtco1aBleXa0RVlSDS6gt7U7ozAp0YxkBW7YlqXxfM8A8y-Dn8LkKewv5p7q7yL5Bkun5Cy7rZ_FPQ_4ktHUr_RzqpQbgSgtXwOSyCfoDKqIPNg4AhjaI33nD93HuRQeV_mhxYwXN5GNTq-7SxkulMwTSgg7b2UhmOSu87pX_FMk5nFaglzYzHKpoZA3QuNxwHzTVInF8Ufu6fAIOPT5fEuhfilDU3uxCkpC-us4yeLwm8e36ICJZFfcqa5dXHkFezEXPKvFbhpVgjTO-TI2EH_vb4QcYNQxtQGWUqFcuQ7IaIgYChVS7ifjkPc65wR9ffjTEEqFAt6e-_mviI4ltyiTLTNTWY68JV64SnjeMQ9qR9gPYmefUp_E_LyOdwfetRYKBJ81jAMz2piWNoJHwHbFjBxeZj8iZ34TnirgvWRltUi20aN09b8TN_IbFNPFjkI1UwshqMwLY9GXT4eq0QaIdvhW9CE90--KNVjGvqyRLodo0gsGTpmTcoTPDgF_AuaeDlaBrbAnW-pFr1HOV5YqUGja5_vkDvi9mdKooFrlSau-Dt1HmZf81izJ8odFR-tHl0u-wT66G0aEkk1DS81IXvSLLNAQlIpj5FoZYx2RPFWyw1WBlY8iSa4r6HyN5YKW9taJ7ljUliA8KClax8VM282lqYL5Fd-wtYu5Iceez8jGGj4cZ7JetWp6X-wjLHeo6SDUGjNO7k7h3ODmCRnIKJZVtbx6qJEVX1u8J9mIAXEjdArqa_7YiUBTuka0W7IxVXZUx9R96h5f',
                        provider_name='openai',
                    ),
                    TextPart(
                        content=IsStr(),
                        id='msg_0b4f29854724a3120068c4ab22122081918f25e06f1368274e',
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=9939, cache_read_tokens=8320, output_tokens=1610, details={'reasoning_tokens': 1344}
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_0b4f29854724a3120068c4ab0b660081919707b95b47552782',
                finish_reason='stop',
            ),
        ]
    )


async def test_openai_responses_model_web_search_tool_stream(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(
        m,
        instructions='You are a helpful assistant.',
        builtin_tools=[WebSearchTool()],
        model_settings=OpenAIResponsesModelSettings(openai_include_web_search_sources=True),
    )

    event_parts: list[Any] = []
    async with agent.iter(user_prompt='What is the weather in San Francisco today?') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    assert agent_run.result is not None
    messages = agent_run.result.all_messages()
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the weather in San Francisco today?',
                        timestamp=IsDatetime(),
                    )
                ],
                instructions='You are a helpful assistant.',
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_00a60507bf41223d0068c9d2fc927081a088e0b920cdfe3866',
                        signature='gAAAAABoydMADQ6HaJB8mYQXlwd-4MrCfzmKqMHXUnSAXWV3huK1UrU1h3Do3pbK4bcD4BAvNiHTH-Pn27MGZDP_53IhKj_vB0egVf6Z_Y2uFPtzmyasYtTzrTkGSfAMR0xfI4wJk99aatk3UyLPNE7EO_vWYzN6CSX5ifJNNcmY3ArW1A7XnmsnMSBys05PsWqLMHZOUFuBvM2W37QUW6QOfBXZy0TamoO5UknNUfZb_TwvSnMEDpa-lXyDn4VuzfxreEGVHGdSyz5oLN0nBr3KwHIfxMRZIf9gi9-hKCnxX7i-ZktNIfTgd_WEmNKlaPO-qjKHPlO_XPKbEfpBdMv5b2P9BIC20ZG3m6qnEc4OqafWZa1iC2szi4eKOEa6neh2ltVLsLS3MlurF4sO-EHQT4O9t-zJ-738mZsOgjsI9rTrLm_aTAJrntSSWRLcP6PI6_ILHyeAl_aN4svtnwQJZhv4_Qf62q70SZQ5fSfqoqfO1YHLcXq6Op99iH3CfAhOjH-NcgThFLpT4-VLYABl8wiWBTsWzdndZoPmvMLEOaEGJOcM6_922FC0Q-fUio3psm_pLcElaG-XIkyn4oNuk6OJQonFE-Bm6WS_1I9sMF0ncSD4gH1Ey-5y2Ayxi3Kb3XWjFvs1RKW17KFXj8sthF3vY5WHUeRKA14WtN-cHsi4lXBFYJmn2FiD3CmV-_4ErzXH8sIMJrDDsqfCoiSbHwih25INTTIj7KAPL2QtIpU6A8zbzQIK-GOKqb0n4wGeOIyf7J4C2-5jhmlF2a6HUApFXZsRcD8e3X1WqSjdTdnRu_0GzDuHhPghRQJ3DHfGwDvoZy6UK55zb2MaxpNyMHT149sMwUWkCVg0BruxnOUfziuURWhT-VJWzv5mr3Z765TFB1PfHJhznKPFiZN0MTStVtqKQlOe8nkwLevCgZY4oT1Mysg7YJhcWtkquKILXe-y6luJBHzUy_aFAgFliUbcrOhkoBk5olAbSz8Y4sSz5vWugYA1kwlIofnRm4sPcvoIXgUD_SGGI3QNsQyRWQEhf7G5mNRrxmLhZZLXAcBAzkw10nEjRfew2Fri7bdvyzJ1OS_af9fHmeqCZG5ievKIX6keUkIYQo_qm4FQFkXZSl9lMHsUSF-di4F6ws31vM0zVLMmH52u12Z3SZhvAFzIV5Vtyt_IfrMV3ANMqVF4SmS4k2qUlv1KuPQVgqGCVHvfeE1oSyYgYF6oFX8ThXNB79wxvi4Oo8fWEZLzZMFH9QEr2c7sOWHYWk-wUMP1auXTQNExEVz22pBxueZGZhRyLdpcA12v8o6vJkVuBj-2eR8GRI7P6InJdQAO9TIBhM7NtJU2NUpeP_84js3RTBVktqBT74nWPaHIddGMSfW2aGmFJovvshhxGMLtN_6XMh4wRKW0IE_-Rfbhk8_-xHKI5McYI048N_TMYOS8KqPPAmGVklRGqPZ5xXMNvQEVweThDTYTo3NoAsS0fN2yMmSwrjRYBHsgYMtil4pd6ddp8dvF_XSJUkW0nF8t6ciI_k47sug3gyw4usqspWxY9Hwbzb4OFzzrgtO_7Ll6lFFFUx2oHy8AO9sJ97Y3Fg6luuew7ZRDzA_4XMrT7mNW6YuT-o2DunaZw-jvQezNHjPN2WhaTS7fkisyhFSFTMBYE-H4psfj_sizutv-LjwbumTcX2mnYE9SZhVr8dL0c7sgwHP1831RxTSSl3ql_obE3ICDooyuM8PYE56Jx0HOOGbEeJd3w91SzNHPG_3SQfXszrZlw4BGWrEUHBbtVY2ZEnsyGNAx6vKO8lz9D-6yZ618foDJSH-Ilk56a5rhr0beWjSd9mYMsr3zpVz6HcpTLYGEgHfPxpT2eaYaC1H_znw7y1eMKamwudYmtz_azX5LrOtwc0p-pXH-kdoNe248pSz9qsmHcXA41fuj2weKQNrmBcghwtfM95B060tnmebJ_B_KkLXL4cNF-hZqi0wAHrHYrZ_WM0Dy90AFH-b7iiWuWz5M1EhZXo179iEdybM-1PgccFJ0zvOqODl7FNxSgWVyNS1k9R42aZx2PzFAfAbBtJ-KVMhUayAvGLNmi35EAT0G6FK65VBEe7A6zPFqzrrAiG8dy3Z0I0253WzIblHPNMpmxI_ca5tIx3u8Za6Nu9rx8mi0CY2jsRSKnqb7RZvLuB78Uj32lb_9jbq5_gL9_y7Bt7U7i7FospyqMFzEYQLvdyrtfNrfY0rB4zr4Mo0tDn_4YOD_d_nP5axUh9_ruqXZ_d3eVdNmlITjQZj8ALe1EfidP8a-Dl62t6STVv8d2y8v9-jy3J7wReLJbJ6gDDnygJllY7NrIVXSjR45FXiCDnpaRonu--I_0b_LRJFOoJUJX0S9YMaXAkKyHSEj-UWjiuk8cIBNcXxwlxnqqNMezvvV113MAOEbfHygDnphzjzZQxteAVbSy0ucGDR2FPi30d6z51NxGnXNS_sM7wnjBMNp4Li0hhttOp6PgvDKPSMAcgUtKLFKE8iWQAvERoUVxw5Et20hNTNXf_0sXOyh0bF0URPGDxSYz9uZI6-nlwVlo1aobdEnn7STSq2_tuTDIrQyfBGZzhv8OB0H3cj9mBs=',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'query': 'weather: San Francisco, CA', 'type': 'search'},
                        tool_call_id='ws_00a60507bf41223d0068c9d30021d081a0962d80d50c12e317',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content={
                            'sources': [{'type': 'api', 'url': None, 'name': 'oai-weather'}],
                            'status': 'completed',
                        },
                        tool_call_id='ws_00a60507bf41223d0068c9d30021d081a0962d80d50c12e317',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_00a60507bf41223d0068c9d300b23481a0b77a03d911213220',
                        signature='gAAAAABoydMLww_4DcIPiCy5zW1t-Dtx57JodsdP9YkyDCvY9r0nBfoD-bwcBl8FfcFxRuq5nK5ndf90J6Xdxmvl9kGVbCUYSFaOd-kWeE4tgwXM8BwwE3jVs6ZMG3BdiWyXS3alnUO5jcE6kuXeun1LufAdZ6rWtJl3zSmqPycnTh9zoQ4cBBxLDq_qcVS1fgU4WjsWgjCZw6ZWRPWwZk8hywmU7ykCLH7SXItM4oH1m_GCEounJAS8YR4ajUh5KAdN6a1wngfGnXhYdzov98yiNLP6Nrkgr--K4vMTTqWXLTcR6fbWgkijanIeKmfSErCjMT6k5TrAbkFx0rgblHbdQii7zj8seV1BWZse92_k4sltxfc5Ocpyho1YSHhgxyGL7g442xMUEibjPCv6kwPMcW9yGu9wPMWsfPYCXpBbG6kQibQPNFJ_bEubwBRaqdSDq93Aqr1YkTYBja7Tewn8UfzZ8YYaGe5y_K4ZD47lfvDp019dOdXmOuZGC1ECRrMqKzSFYVG1CFY1VhjGdPmzobDoMcpZcLn25s1pg6lnNqNQwOk_IA4MvUcCU5HHD5YjmFkEy5-i_iRoDVu5coK0zyEMvPJ_h10y_ByszcfzS9e0ht5CSilckkFdxTBkZ5epp0YIg1e-PrZ790P-I35Ucquam9OXyULV1Y5bn9ohZa93Tv0JZRxUeTDG72_28xRj8tkJaBAZjoCC7VICw39KVmz-ZkuVN6IIX1WdNzyC4d808-2Tz4UZaU42-wxEWDnSDMD7iZu1Bi9fKKwAYBJt_OcEsJwpW63ZaUSG2PVFfm7a3wRcSMxMTUTTJB7L1Keu1hmNepif5tavn3P35nSq28D_IJyAqAgX7ZyROk2bJqjzSE4A0MddqAoBFFqKBi68n49KH09vDtDXIoh8jVWuIgowgVGr8pN3kuhLI9cir4Pr_WES0tPD7yWHPTzrD7OIJCfQbr_4Y4dEza4ixNi0RTADWzMUZBfr7bvwIsgvg6ZNuQlx_d71Go5VDsT2KI8H8AldiRvNWoLyYTFGyK9Kot97YsS5sEmSYgNAH48NU7pgnM0jNDQU1G39nTNFEjL_ziDwjDT5g3jm4S_gbQfwx-XFT3Pv-JYR-E71AqR--Lg71OsASq49rrlULfl5OENfiT-NB6x8MqnfUI6NpcCsOWLp8XfRbgqmZFutLIi43pcnxEe3cXHLWGF77qJXP6dFb-G5Ide7n9tAOoEgfsVu7hCDPEQ_xrIYRdc2DzDPUMCtXBai24E0AnQF8kxsEtlDW_YmAgGNTl9Gx0tFSGdDuUCsNx__c7v-_LOMWycXUKmH3iEr_su83oGIMapNp2PnLccN4iOxspdZQq0C6WBaR6SrdnGzK-0KwRPRoyKDLNWS8zfluR5bIgKlqd3Sbv_7eL-WO4LQXMvdKP3KS-DBt1HbA-gmyFW03iX2smPQbtVmRLWi1vG329R_07-tHMJSO9OQy6_6aiyO8Rgpbl_CHa1Q9BEkI2csonayDJRPvEXBPuk9-NPUP4VLNPB7npWBLlAqes5ZmhagnC7srTL0fFiLGLJiAxWo1f0BBiIlXjwqHdlgBjTw0KryCnEU8Ic8ATzrqEXXhs-FTBCcWInf3Bt5bzUhy20g7cTtYP-VCbsku-lXQ6wceWrfQVFtjKKICD8I4g9QusAIAvgCUm7J2rR3TLkzwOKngdTFPGQrQ1TYzlkA7q_Ew1uZpaPRckMaEioZYC6Sv_B0rgW0nyBJ0GLrB3AUN60hDrOFntyFHp0FM-Zh1SY-GKGBwZwVetOzM0ZAJ-NreFg1XVgyLTYDNjUrYJjRhr_JARsZ5t0pU4_yI6dPqM5jKO5_k4UpZspfQon6d2-NlWX0EDmz6G4CMTx0TScehYHrQZtPzpVnivc8h_pmXV3jO5GLzNeLWoB70SDPTETo1Of4txiEUaC2komu5B7MN9aR4c7VBOTv1NIjoiZcrd1HFACzZ7r1qAE-G38j1f1YhfZ0_TiMmtfR1cqjAKcFkyRM7rZMyMvvnsH7NFq59gFgWZt0dy0aAdw03XWXFNT67lrw58OYC3NcVozH4SKlmleu7TfjHNWSnJVjJ66riLn9DZWVxPeTk4zuISZn0yyaoXcdW8OMn_mJ9vP-8L1wElMyxKbtBRz-0cW7MshmJ3YXmHWDKbnqETSbDMtqcN_QyRJovopwlptJ8VzL7biuURRFw-l63Kc9vKP72Z-QWOUIPLB4q4nX4yb-IV0mkWFxIUlfv5Cze2anf7zDFyGzeU9xG0onfhJE4HFKcoUT8MzfrHZ0dDZtnEYeL5Xem3GuHpwEVGCxRE_J1joTmJfeWxSVnr2Vey9gaPmXCyRrdKS75v9xSXJFfHvcOO8Qp35Dzk-yFqL3dSOJfOEwDZbEf6QnV7VU1EhJvW4XmRS-wsRLMLCYcLrOx96NHEwb2h2l6gNfbCVJoQrMhMg68qBPnoSYLhML2ho7hWkSNZFy61yX5I-oEJV5XdtjFcBkyurmUD6uYTkJSqXyxLexQiPbT-uv49Yp9cAfFBG23sC9lUQ=',
                        provider_name='openai',
                    ),
                    TextPart(
                        content='San Francisco weather today (Tuesday, September 16, 2025): Mostly sunny and pleasant. Current conditions around 71°F; expected high near 73°F and low around 58°F. A light jacket is useful for the cooler evening. ',
                        id='msg_00a60507bf41223d0068c9d30b055481a0b0ee28a021919c94',
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=9463,
                    cache_read_tokens=8320,
                    output_tokens=582,
                    details={'reasoning_tokens': 512},
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_00a60507bf41223d0068c9d2fbf93481a0ba2a7796ae2cab4c',
                finish_reason='stop',
            ),
        ]
    )

    assert event_parts == snapshot(
        [
            PartStartEvent(
                index=0,
                part=ThinkingPart(
                    content='',
                    id='rs_00a60507bf41223d0068c9d2fc927081a088e0b920cdfe3866',
                    signature='gAAAAABoydMADQ6HaJB8mYQXlwd-4MrCfzmKqMHXUnSAXWV3huK1UrU1h3Do3pbK4bcD4BAvNiHTH-Pn27MGZDP_53IhKj_vB0egVf6Z_Y2uFPtzmyasYtTzrTkGSfAMR0xfI4wJk99aatk3UyLPNE7EO_vWYzN6CSX5ifJNNcmY3ArW1A7XnmsnMSBys05PsWqLMHZOUFuBvM2W37QUW6QOfBXZy0TamoO5UknNUfZb_TwvSnMEDpa-lXyDn4VuzfxreEGVHGdSyz5oLN0nBr3KwHIfxMRZIf9gi9-hKCnxX7i-ZktNIfTgd_WEmNKlaPO-qjKHPlO_XPKbEfpBdMv5b2P9BIC20ZG3m6qnEc4OqafWZa1iC2szi4eKOEa6neh2ltVLsLS3MlurF4sO-EHQT4O9t-zJ-738mZsOgjsI9rTrLm_aTAJrntSSWRLcP6PI6_ILHyeAl_aN4svtnwQJZhv4_Qf62q70SZQ5fSfqoqfO1YHLcXq6Op99iH3CfAhOjH-NcgThFLpT4-VLYABl8wiWBTsWzdndZoPmvMLEOaEGJOcM6_922FC0Q-fUio3psm_pLcElaG-XIkyn4oNuk6OJQonFE-Bm6WS_1I9sMF0ncSD4gH1Ey-5y2Ayxi3Kb3XWjFvs1RKW17KFXj8sthF3vY5WHUeRKA14WtN-cHsi4lXBFYJmn2FiD3CmV-_4ErzXH8sIMJrDDsqfCoiSbHwih25INTTIj7KAPL2QtIpU6A8zbzQIK-GOKqb0n4wGeOIyf7J4C2-5jhmlF2a6HUApFXZsRcD8e3X1WqSjdTdnRu_0GzDuHhPghRQJ3DHfGwDvoZy6UK55zb2MaxpNyMHT149sMwUWkCVg0BruxnOUfziuURWhT-VJWzv5mr3Z765TFB1PfHJhznKPFiZN0MTStVtqKQlOe8nkwLevCgZY4oT1Mysg7YJhcWtkquKILXe-y6luJBHzUy_aFAgFliUbcrOhkoBk5olAbSz8Y4sSz5vWugYA1kwlIofnRm4sPcvoIXgUD_SGGI3QNsQyRWQEhf7G5mNRrxmLhZZLXAcBAzkw10nEjRfew2Fri7bdvyzJ1OS_af9fHmeqCZG5ievKIX6keUkIYQo_qm4FQFkXZSl9lMHsUSF-di4F6ws31vM0zVLMmH52u12Z3SZhvAFzIV5Vtyt_IfrMV3ANMqVF4SmS4k2qUlv1KuPQVgqGCVHvfeE1oSyYgYF6oFX8ThXNB79wxvi4Oo8fWEZLzZMFH9QEr2c7sOWHYWk-wUMP1auXTQNExEVz22pBxueZGZhRyLdpcA12v8o6vJkVuBj-2eR8GRI7P6InJdQAO9TIBhM7NtJU2NUpeP_84js3RTBVktqBT74nWPaHIddGMSfW2aGmFJovvshhxGMLtN_6XMh4wRKW0IE_-Rfbhk8_-xHKI5McYI048N_TMYOS8KqPPAmGVklRGqPZ5xXMNvQEVweThDTYTo3NoAsS0fN2yMmSwrjRYBHsgYMtil4pd6ddp8dvF_XSJUkW0nF8t6ciI_k47sug3gyw4usqspWxY9Hwbzb4OFzzrgtO_7Ll6lFFFUx2oHy8AO9sJ97Y3Fg6luuew7ZRDzA_4XMrT7mNW6YuT-o2DunaZw-jvQezNHjPN2WhaTS7fkisyhFSFTMBYE-H4psfj_sizutv-LjwbumTcX2mnYE9SZhVr8dL0c7sgwHP1831RxTSSl3ql_obE3ICDooyuM8PYE56Jx0HOOGbEeJd3w91SzNHPG_3SQfXszrZlw4BGWrEUHBbtVY2ZEnsyGNAx6vKO8lz9D-6yZ618foDJSH-Ilk56a5rhr0beWjSd9mYMsr3zpVz6HcpTLYGEgHfPxpT2eaYaC1H_znw7y1eMKamwudYmtz_azX5LrOtwc0p-pXH-kdoNe248pSz9qsmHcXA41fuj2weKQNrmBcghwtfM95B060tnmebJ_B_KkLXL4cNF-hZqi0wAHrHYrZ_WM0Dy90AFH-b7iiWuWz5M1EhZXo179iEdybM-1PgccFJ0zvOqODl7FNxSgWVyNS1k9R42aZx2PzFAfAbBtJ-KVMhUayAvGLNmi35EAT0G6FK65VBEe7A6zPFqzrrAiG8dy3Z0I0253WzIblHPNMpmxI_ca5tIx3u8Za6Nu9rx8mi0CY2jsRSKnqb7RZvLuB78Uj32lb_9jbq5_gL9_y7Bt7U7i7FospyqMFzEYQLvdyrtfNrfY0rB4zr4Mo0tDn_4YOD_d_nP5axUh9_ruqXZ_d3eVdNmlITjQZj8ALe1EfidP8a-Dl62t6STVv8d2y8v9-jy3J7wReLJbJ6gDDnygJllY7NrIVXSjR45FXiCDnpaRonu--I_0b_LRJFOoJUJX0S9YMaXAkKyHSEj-UWjiuk8cIBNcXxwlxnqqNMezvvV113MAOEbfHygDnphzjzZQxteAVbSy0ucGDR2FPi30d6z51NxGnXNS_sM7wnjBMNp4Li0hhttOp6PgvDKPSMAcgUtKLFKE8iWQAvERoUVxw5Et20hNTNXf_0sXOyh0bF0URPGDxSYz9uZI6-nlwVlo1aobdEnn7STSq2_tuTDIrQyfBGZzhv8OB0H3cj9mBs=',
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=1,
                part=BuiltinToolCallPart(
                    tool_name='web_search',
                    tool_call_id='ws_00a60507bf41223d0068c9d30021d081a0962d80d50c12e317',
                    provider_name='openai',
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta={'query': 'weather: San Francisco, CA', 'type': 'search'},
                    tool_call_id='ws_00a60507bf41223d0068c9d30021d081a0962d80d50c12e317',
                ),
            ),
            PartStartEvent(
                index=2,
                part=BuiltinToolReturnPart(
                    tool_name='web_search',
                    content={'sources': [{'type': 'api', 'url': None, 'name': 'oai-weather'}], 'status': 'completed'},
                    tool_call_id='ws_00a60507bf41223d0068c9d30021d081a0962d80d50c12e317',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=3,
                part=ThinkingPart(
                    content='',
                    id='rs_00a60507bf41223d0068c9d300b23481a0b77a03d911213220',
                    signature='gAAAAABoydMLww_4DcIPiCy5zW1t-Dtx57JodsdP9YkyDCvY9r0nBfoD-bwcBl8FfcFxRuq5nK5ndf90J6Xdxmvl9kGVbCUYSFaOd-kWeE4tgwXM8BwwE3jVs6ZMG3BdiWyXS3alnUO5jcE6kuXeun1LufAdZ6rWtJl3zSmqPycnTh9zoQ4cBBxLDq_qcVS1fgU4WjsWgjCZw6ZWRPWwZk8hywmU7ykCLH7SXItM4oH1m_GCEounJAS8YR4ajUh5KAdN6a1wngfGnXhYdzov98yiNLP6Nrkgr--K4vMTTqWXLTcR6fbWgkijanIeKmfSErCjMT6k5TrAbkFx0rgblHbdQii7zj8seV1BWZse92_k4sltxfc5Ocpyho1YSHhgxyGL7g442xMUEibjPCv6kwPMcW9yGu9wPMWsfPYCXpBbG6kQibQPNFJ_bEubwBRaqdSDq93Aqr1YkTYBja7Tewn8UfzZ8YYaGe5y_K4ZD47lfvDp019dOdXmOuZGC1ECRrMqKzSFYVG1CFY1VhjGdPmzobDoMcpZcLn25s1pg6lnNqNQwOk_IA4MvUcCU5HHD5YjmFkEy5-i_iRoDVu5coK0zyEMvPJ_h10y_ByszcfzS9e0ht5CSilckkFdxTBkZ5epp0YIg1e-PrZ790P-I35Ucquam9OXyULV1Y5bn9ohZa93Tv0JZRxUeTDG72_28xRj8tkJaBAZjoCC7VICw39KVmz-ZkuVN6IIX1WdNzyC4d808-2Tz4UZaU42-wxEWDnSDMD7iZu1Bi9fKKwAYBJt_OcEsJwpW63ZaUSG2PVFfm7a3wRcSMxMTUTTJB7L1Keu1hmNepif5tavn3P35nSq28D_IJyAqAgX7ZyROk2bJqjzSE4A0MddqAoBFFqKBi68n49KH09vDtDXIoh8jVWuIgowgVGr8pN3kuhLI9cir4Pr_WES0tPD7yWHPTzrD7OIJCfQbr_4Y4dEza4ixNi0RTADWzMUZBfr7bvwIsgvg6ZNuQlx_d71Go5VDsT2KI8H8AldiRvNWoLyYTFGyK9Kot97YsS5sEmSYgNAH48NU7pgnM0jNDQU1G39nTNFEjL_ziDwjDT5g3jm4S_gbQfwx-XFT3Pv-JYR-E71AqR--Lg71OsASq49rrlULfl5OENfiT-NB6x8MqnfUI6NpcCsOWLp8XfRbgqmZFutLIi43pcnxEe3cXHLWGF77qJXP6dFb-G5Ide7n9tAOoEgfsVu7hCDPEQ_xrIYRdc2DzDPUMCtXBai24E0AnQF8kxsEtlDW_YmAgGNTl9Gx0tFSGdDuUCsNx__c7v-_LOMWycXUKmH3iEr_su83oGIMapNp2PnLccN4iOxspdZQq0C6WBaR6SrdnGzK-0KwRPRoyKDLNWS8zfluR5bIgKlqd3Sbv_7eL-WO4LQXMvdKP3KS-DBt1HbA-gmyFW03iX2smPQbtVmRLWi1vG329R_07-tHMJSO9OQy6_6aiyO8Rgpbl_CHa1Q9BEkI2csonayDJRPvEXBPuk9-NPUP4VLNPB7npWBLlAqes5ZmhagnC7srTL0fFiLGLJiAxWo1f0BBiIlXjwqHdlgBjTw0KryCnEU8Ic8ATzrqEXXhs-FTBCcWInf3Bt5bzUhy20g7cTtYP-VCbsku-lXQ6wceWrfQVFtjKKICD8I4g9QusAIAvgCUm7J2rR3TLkzwOKngdTFPGQrQ1TYzlkA7q_Ew1uZpaPRckMaEioZYC6Sv_B0rgW0nyBJ0GLrB3AUN60hDrOFntyFHp0FM-Zh1SY-GKGBwZwVetOzM0ZAJ-NreFg1XVgyLTYDNjUrYJjRhr_JARsZ5t0pU4_yI6dPqM5jKO5_k4UpZspfQon6d2-NlWX0EDmz6G4CMTx0TScehYHrQZtPzpVnivc8h_pmXV3jO5GLzNeLWoB70SDPTETo1Of4txiEUaC2komu5B7MN9aR4c7VBOTv1NIjoiZcrd1HFACzZ7r1qAE-G38j1f1YhfZ0_TiMmtfR1cqjAKcFkyRM7rZMyMvvnsH7NFq59gFgWZt0dy0aAdw03XWXFNT67lrw58OYC3NcVozH4SKlmleu7TfjHNWSnJVjJ66riLn9DZWVxPeTk4zuISZn0yyaoXcdW8OMn_mJ9vP-8L1wElMyxKbtBRz-0cW7MshmJ3YXmHWDKbnqETSbDMtqcN_QyRJovopwlptJ8VzL7biuURRFw-l63Kc9vKP72Z-QWOUIPLB4q4nX4yb-IV0mkWFxIUlfv5Cze2anf7zDFyGzeU9xG0onfhJE4HFKcoUT8MzfrHZ0dDZtnEYeL5Xem3GuHpwEVGCxRE_J1joTmJfeWxSVnr2Vey9gaPmXCyRrdKS75v9xSXJFfHvcOO8Qp35Dzk-yFqL3dSOJfOEwDZbEf6QnV7VU1EhJvW4XmRS-wsRLMLCYcLrOx96NHEwb2h2l6gNfbCVJoQrMhMg68qBPnoSYLhML2ho7hWkSNZFy61yX5I-oEJV5XdtjFcBkyurmUD6uYTkJSqXyxLexQiPbT-uv49Yp9cAfFBG23sC9lUQ=',
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=4,
                part=TextPart(content='San Francisco', id='msg_00a60507bf41223d0068c9d30b055481a0b0ee28a021919c94'),
            ),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' weather')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' today')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='Tuesday')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' September')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='16')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='202')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='5')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='):')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' Mostly')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' sunny')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' pleasant')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' Current')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' conditions')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' around')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='71')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='°F')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=';')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' expected')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' high')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' near')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='73')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='°F')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' low')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' around')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='58')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='°F')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' A light jacket')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' is useful')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' for the')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' cooler evening')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='. ')),
            BuiltinToolCallEvent(  # pyright: ignore[reportDeprecated]
                part=BuiltinToolCallPart(
                    tool_name='web_search',
                    args={'query': 'weather: San Francisco, CA', 'type': 'search'},
                    tool_call_id='ws_00a60507bf41223d0068c9d30021d081a0962d80d50c12e317',
                    provider_name='openai',
                )
            ),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='web_search',
                    content={'sources': [{'type': 'api', 'url': None, 'name': 'oai-weather'}], 'status': 'completed'},
                    tool_call_id='ws_00a60507bf41223d0068c9d30021d081a0962d80d50c12e317',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                )
            ),
        ]
    )

    result = await agent.run(user_prompt='how about Mexico City?', message_history=messages)
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='how about Mexico City?',
                        timestamp=IsDatetime(),
                    )
                ],
                instructions='You are a helpful assistant.',
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_00a60507bf41223d0068c9d316accc81a096fd539b77c931cd',
                        signature='gAAAAABoydMovnl5STyQJfKyyT-LV6102tn7M3ppFZHklPnA1LWETYbnDdCSLgeh1OqOXicuil2GTd-peiKj033k_NL0ZF5mCymWY-g5qoovU8OauQyb2uR9zmLe-cjghlOuiIJjiZC1_DCbwY1MHObzuME-Hn5WiSlfTTcdKfZqaQpzIKVKgbx6cSDDyS5j29ClLw-M6GQUDVDsjkclLEcc8pdoAwvuWDoARgMYXwcS-7Ajl46_9oA92RP-64VjrO6Wxzz9HjKcnBTcSDUcyJxsdolHq6G0TjZFwECg4RWvzcpijO53OF58a4_SfgUqbupni7o-tMzITyF1lwE5Xq9fluUFHXmbH0QCrk_7lGRjeiFqY9tTv_VKbNeHSVj5obUnA5HyAYb5jEqgy9M-CgdN1DJeODMTq3Ncu1y81_p7sXqxpbh1c-2eHkGj6yMFjO-dF9LpX_GUZZgAoPXN-J0k3_6VFWc6FjwOGbPU_weslCBpBnS0USfiif9y8nzH2xg0VrHCUEliBOkN-QLqq68edZOBAmYgG8iRDx-yG762TzOBri-0EdFHGWnMij_onb0y4f0UOXD-qSqHvBj8WKasOSRkBpJmIkDViKXYab3nhOtUb4Y3jNhSh6KYEW1QETK9oOMc1zd0Osk-z0QBLQdGtMuFiR00Bs1M_E4T0lMYEsFRqQ8TZmM5-hmrAkBVx3u1f9-ccBZE0ANOiNWH-G75LozwgZhYrOwbuDSnG3wq2M0L7F1mkseg5lOGKgyaxkaifO6WyS6JCHMwDZUF4gZKyHItg3x3PACmTdUy_Wda55J5oIFklWtjFGbU-dY7vr8wvyF0Q0jEeMp8tFvMpGOGTVlydMBq6SCWrZAz8uDoMRxuNLecaHj3bSQHbfeC3hs8uKCLOMr0X_ZCQ8ATXSSjjml3onzNvqChlsspKcwtEKKSwHNTMUJbY6cyy45EQdYhbKg75k-ZL7Y6BXMRjCc5CJd-4uuD8_cXHi4ikmkpHmgZLHcQPOdFflXeDlpYVTF9-Hyblg4SsxvLX9Vp5h4T4J_RcalfwPsIAwIEn8RSutJyMAIm0tYsEzq5i4usmLMxyEBbekCgP5DlHbeWvj3B8h0WoPE7C4cA1m29A_7bRDcJiL06D2T13r9zh17W7UYucDtTcJF7dtKHJTFK_C9m6wW-rHhXi1CgTFU8acDLYGK_VhZhQmTD7tM5JX7IEw_yokWzqyZzWFHmN4mgvAn3imeOXliVLY2YxD7I8-6xAgez6tVyX6plXIpE4KL-GLnFXyqORwIhH4F4EvEm6AcurW8pPWBXXVOY8Ml25-3D1tSu6sQ4PFzgvE5FWiwkBUpLSKwBjZqfg3_aG3NQe4exExztofsCD1l12US7OTx76h7utifDiu_FuzSZHOq0sM0kWfsrzoaPW79T7CT0Ew97HqEJTvYvhkdmzgtA-57zYK-8kc2bUTmTNdl_nUovO-xRhvwamIjMTzgqo3FXjLAtj4QZYWIHInkGj8GIxLluow315yWxARpfTehrpgvwYbd-tJ0UFyCZ1J0RwXQ8QmBu7UV-qPxj88d8cuY9sn8xba3kFCLifxlohEOupJcDDNHjta5eunNYoE127ap0Pv5KdJHWaOUcpScrXz3dIEXBlax12ySZNkghKGgGqYzOyQBKvkAgcV2rHaUQjuAkEbV3uQuE7iG3413fqfRVyAOKHKv3ig0jUM2DqBfhK9Tmxdbh-5VI5H5r5dgw3GmTQtSZVd0Q3mIMCeghrfHeCW4Ms1lRjcwEbn1Uyffs7KylhabOdqmiRTUPavLgKZmSrh7q0Vrkmb3s-nZEcfnVL6o2OpuQrdm83K-aI0Pvnsf9V9U_qoW1HWf61ENQUhnMECD2P70EsSmXLnQ_7f3v4Nyw-MCWCPpdzJvCh0TrpcTpY4WcflgbkNxm9xorCEiTlnEaeGSYj0MDcNm8sJYZbWzNQoNmbj58XS4IgnfCIYcoyu6PTceMcE7o_w50MPC3LcMTzZWKSYnGA7xDrvfeD7boqfj-Xd37SDYSTp9OAifiwiTXZyl7FqVTk1Y-1RCYTvIPPpnhXedT4ehYPRL9_fYmTgVISPLK8IQyNHpme86nG1-0FOJoitzwOa94MICeNKJArYvZ4Kj9WlP5-cTjP6zoDlaYxXXuln6DRmOnqL5CDVqf3f-7Dg-n8ARgNFwaAuvLXhCxuuRdcnNN5gx1z5vnvusq2sMCZx-eRqaGQsRoAoWo1VsrW5bwPGHwZN9Ip97KeORMAV8ExDttxjS4DXO-nB5fVZ2KToAsglOjLfvoXi7ArwK4Du3u7N_kzERB8lVT25jOltMdhOISXCGzY-ORQr6WhS_fgM8s8wHJSAtEl2w5VaFku57kEgWmfmasDNz5O1iMlqKOzVGpd9qNUtWaqYDK9DIxaL-O1pQGbzzuCsq332tez68SMNdbjNaf5RS3MHgAKHmI0I2RaGdBcaXjlap3sEMANG7keCNYSrtU-vfoMfb708dt2Ux2dDktmtSMFwZyzbOnGOshGhxsW5O98Uo-I-PZLsHSj4ZJSD5yIayNiuf8bZ0_REJ-9I-5xdfyUDstO7xj4IRjwwnsF9Td8CUycBKxr4gsttwfOoo04LVLOg7mDbK1GtoLEP2e-nXBHsFsOObaW3bOTx7TZwQf5DLggHsEfqdArl1-MqhRllSJNFtBLV3T8bRIvDl-YCV_LYjvWqRvo0RsR3oxrrPGwHM5ROy0WdfHixv2t5voksrS40VJI-KVXqgvF4ixUTMCjpL_pKpBq3pVZEnsJc4yZgK-C-sz72NZNKFHZviJhcdPDuwd4dX7oiI9X2KbnRfoo67xMqTuQCryLeiF7FpFoBHIjH2OhMzk2HbJR5YK9Q8blsWHpAdy',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'query': 'weather: Mexico City, Mexico', 'type': 'search'},
                        tool_call_id='ws_00a60507bf41223d0068c9d31b6aec81a09d9e568afa7b59aa',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content={
                            'sources': [{'type': 'api', 'url': None, 'name': 'oai-weather'}],
                            'status': 'completed',
                        },
                        tool_call_id='ws_00a60507bf41223d0068c9d31b6aec81a09d9e568afa7b59aa',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_00a60507bf41223d0068c9d31c935881a0b835341209f6ac8b',
                        signature='gAAAAABoydMoKoctyyCO6gsPILkjEnvCX0VL-9Gqk9qAmNEdWKNRPvxRIBVCxX4hGZ4m5fZJmuSIjjrA-nU-cUj_XIsARJsJywo2ka8IDmGRF8m7lm5atgcSJQjytRVpIA6s7sz0Sw3iAKrjtQcbymz2sUViTiOn7OqUStKtW0h98UIubdU6d19hu3iDwNddCuAC4QDy8cg3qJhjq9QTtovoBwFpibBJ12ISJqoPLSs43YvWK26o-evCMfzVbkuqJ7Gqie14gZ0oQChxGj7-bopeml1MCaDAz0EUxD5EDfjSdgjB_JABqF13kTTFdAVJu8gY1WgjFt0m1CONQGlM2oQA7cywjU7NnGWSNOqZp_NSDeTBYsKykAmyJP_lTzIDhhG37GBW7PwvBwuUYbvPcMmsRR9FDXxcMeVcpZPmaDjXhRAkJ-Am48Xz676pYl5Sx732-Pv9w503O66ARt6jwQYB4ZW5GgJAnqoqugbmJoGfOV4TaF0glOfKB5XPNQx--_hARpmXuQX3M_Xg1zLa6n7xGmf9pv__Gnhk3V0OlEnTD5HPZzc13F2hKX1PZ8E4ykq4843ZHDV3vpc5WsNCp6C6Cq8STXq58_QAU8P9vpqEP8khnYt3EJTjzbweiqVrMj6cSoUS9C32z8dFcA0rQrTmt_tEMTaoTN1Q5nTboSm0jX1arXqGh3RhcDkqddBDLfI6PdTVulEPVnBkmZJmCFqdfm_aD9FCSCVJdKE5pktBFqtmGFRJ6RVeGbc_YB6XG9najhjXNhhXIpy176CIPLZbeXkxcgsJQBdDGm4PpUePHZAGKxOpFCNv7kZMyGcsd-Ye-envhfdGhJ5dMOqRq-1KtjopdvNFfmxASkrT8f33YFj6n07fXOOfY02pTl9Dyv7fp0gk_3DR6zKFZRwv-Y3u0sTjQTkk7xTZsuEb0iP_zpqMNcj834fq4FZFvmhJ_siVVOQUPMaP0OFJnYFTteQR8S8JXud4Er1jEZlVojHugyJ3K4yMoj5c16jIQLaFn1_Jk1G97LCO-WZjSxpDD5niEXmYEoC1cw5zweUE7MjkzG1cBU2Wgjw_K0zt0Ko9DxYMDDDS-ZphpCJFPKBiX7pDcpKDpkQnDkEpzIIyDQ3mEKoKvYAXLveKuhOnNnVpUVN28hvW5_QfhD3C1WEBTzz2-dfxLpiS_MHI9NVUZdIue_ThGAM8TFY9MqDrTfAMRMD_mdQHW8XE_QdxighLLuG56AqufuA4CutwifYdbMiAE_mWtApqG4U6dx8cMnmIxnN_lrerv3IQR9_rk6vgPG-MfyJ0drDmSaJGMKyBexYau6sCzyMZYzFO-YgPDa0Yz4DYwhjTnGqtoMSE94ciYiJWZV473WIcyvJ8lE2mQD735nf1OKk7FHsai2mmQzk6NHyyEvvltkTPN8ply0fqmxLksng1bKD43zkHjnP_wUU5uInfAPIGMtIXuwJJXUziMTFRcCawC0KcUUP1J9GK9nrIMeO2B-yM5GXwfvMq3TiI4VFHD9Dav18T5BufMsjIY6uOUuWKNHSOpSQ6VHoql3k7fh2NVGOWqq3juBo2P3BNwXpP6mPr_6diYK4ciukrh4MiUd3pkLZnaW_iv4XYoq0Wix4ENU4zI1kMj5ObFAQOEbeoqdC6u4I5MIOXU6Pep-kaFl6P3yb37Ce95GyPq6xx8q4G29DK6Rx9Qowha8x9BIphuSL01Z6snFTewQW9rqAP7GyEltkso456vXzay08wtzG0dGpxoCIc87mAhx7-ulTj1Wti0qekLhsavem7GPfNKqso4CPsiXMxtTBBoIHk0xAvXcpZcw33pY_71-SHpMafrMrkS-Rp2T6YztbX2u_Nx__O8NAD2V0T0l69gR4S0khT_z-rttSPuCfx0-C4_hz7mCjVPMlLGDzxahOxG25Z9LHst6NPvlfg0xxX5rQ80XAS9GtLJ5uKMEwMxoGCatV3VL2zT2M0SpNiZKLZpH2tHfm0j_2dFcsLWN0a9MAooVZQ1Rlnq_7r0QrAPqcca_Y1Q7Jlzx2dgiEylYfFzNlNU2JTtinZg25gq3A7WayuWE5iBV5dhPijkcgEQbDETKg0eRa584q_cd68Rlm7qYeID3pc8gAbZ4zdqz6SfcQqoZS_EN43Z4Mc-t_HKN-9BwgXFNfvzbLoNekhoCiTrcEUikzXjVKqTbcuczAtH-uie_bfQkwfljFn7J8t7A3SeP961mvpx7iE-yJ4HXTeFhJI2TlBm4JB3OKMCoJSFdEiHjx82bX7TEPvq9g940TgPaooWUD2mEJ_f9ByY84L4EywrGFhtj-DxA1igkbWnCgWlxEquBcvmkRHkbTylkJz6kyz-_-5EPUEJLHqGsDHgotxYWXsxCalzDktH_GivrkeTYqhy1SikEJw93-X5SPMLD7EdQUS_K3XIe9p4T9lpn__zs_tCqssrun7ZQEpY9ULoYiMn2ENU9rK4IYpDoV0beXs4Xa24nj3qgrzbuzbLeKKbm8Y8RxNStogi4E4pK_difBVb_1oTIxfPrLnAJibQ8H-Tb9v20L2Zd3RWXtKi46-XJizKe9r-_JI2HmZ4QM2JOaBhHdybeBrwnu1Z36WhPk4m7YyK8-0K-kIPd-mW_ZF29tHBVhLifqPOq7D3HkJbnBH--KJum-F3v5LLqmeBN-3LWv6bk9-jqQNum9pm2WHtUkOMvH3zw0h8yiBjK3Qov7XHAP9dKHKs3B1eVqiVFGNbuB3Ss07ZzXQrSxgNFP2z64-HtdLJdsSXu3BGc7BqFrnF1tUVeu-KDXKXxJ0SFYaxnLqThuQ4b8CUXYWd8fnhCbhu3OE9Pd2aKWr-4bj73DTDcHLnYmy53mgNKtItsJBfA7m5Dzf6WKREmictNl5nMUWWlEay0nvE6so39zkRlc7wihRthJTEMDbMUdARJw7o1F8JBUPY3cIJchDnq0ZiGkrCA-OyPx-rkxbrQq9usJoTT7XUZNVZ5u7mXH8dY6uY4opcJmV02W2eJms-VtTxgkXuh_HLz_VPmCRMGfACFMwigpShdnr_j3T70ixy80FLcY6ILu1EbuZeLeqo4L8Z5fznYZ1',
                        provider_name='openai',
                    ),
                    TextPart(
                        content='Mexico City weather today (Tuesday, September 16, 2025): Cloudy. Current around 73°F; high near 74°F and low around 56°F. Showers return midweek. ',
                        id='msg_00a60507bf41223d0068c9d326034881a0bb60d6d5d39347bd',
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=9703,
                    cache_read_tokens=8576,
                    output_tokens=638,
                    details={'reasoning_tokens': 576},
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_00a60507bf41223d0068c9d31574d881a090c232646860a771',
                finish_reason='stop',
            ),
        ]
    )


def test_model_profile_strict_not_supported():
    my_tool = ToolDefinition(
        name='my_tool',
        description='This is my tool',
        parameters_json_schema={'type': 'object', 'title': 'Result', 'properties': {'spam': {'type': 'number'}}},
        strict=True,
    )

    m = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='foobar'))
    tool_param = m._map_tool_definition(my_tool)  # type: ignore[reportPrivateUsage]

    assert tool_param == snapshot(
        {
            'name': 'my_tool',
            'parameters': {'type': 'object', 'title': 'Result', 'properties': {'spam': {'type': 'number'}}},
            'type': 'function',
            'description': 'This is my tool',
            'strict': True,
        }
    )

    # Some models don't support strict tool definitions
    m = OpenAIResponsesModel(
        'gpt-4o',
        provider=OpenAIProvider(api_key='foobar'),
        profile=replace(openai_model_profile('gpt-4o'), openai_supports_strict_tool_definition=False),
    )
    tool_param = m._map_tool_definition(my_tool)  # type: ignore[reportPrivateUsage]

    assert tool_param == snapshot(
        {
            'name': 'my_tool',
            'parameters': {'type': 'object', 'title': 'Result', 'properties': {'spam': {'type': 'number'}}},
            'type': 'function',
            'description': 'This is my tool',
            'strict': False,
        }
    )


@pytest.mark.vcr()
async def test_reasoning_model_with_temperature(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel('o3-mini', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m, model_settings=OpenAIResponsesModelSettings(temperature=0.5))
    result = await agent.run('What is the capital of Mexico?')
    assert result.output == snapshot(
        'The capital of Mexico is Mexico City. It serves as the political, cultural, and economic heart of the country and is one of the largest metropolitan areas in the world.'
    )


@pytest.mark.vcr()
async def test_tool_output(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))

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
                parts=[ToolCallPart(tool_name='get_user_country', args='{}', tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=62, output_tokens=12, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68477f0b40a8819cb8d55594bc2c232a001fd29e2d5573f7',
                finish_reason='stop',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id='call_ZWkVhdUjupo528U9dqgFeRkH|fc_68477f0bb8e4819cba6d781e174d77f8001fd29e2d5573f7',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args='{"city":"Mexico City","country":"Mexico"}',
                        tool_call_id='call_iFBd0zULhSZRR908DfH73VwN|fc_68477f0c91cc819e8024e7e633f0f09401dc81d4bc91f560',
                    )
                ],
                usage=RequestUsage(input_tokens=85, output_tokens=20, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68477f0bfda8819ea65458cd7cc389b801dc81d4bc91f560',
                finish_reason='stop',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='call_iFBd0zULhSZRR908DfH73VwN|fc_68477f0c91cc819e8024e7e633f0f09401dc81d4bc91f560',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
        ]
    )


@pytest.mark.vcr()
async def test_text_output_function(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))

    def upcase(text: str) -> str:
        return text.upper()

    agent = Agent(m, output_type=TextOutput(upcase))

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'

    result = await agent.run('What is the largest city in the user country?')
    assert result.output == snapshot('THE LARGEST CITY IN MEXICO IS MEXICO CITY.')

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
                parts=[
                    ToolCallPart(
                        tool_name='get_user_country',
                        args='{}',
                        tool_call_id='call_aTJhYjzmixZaVGqwl5gn2Ncr|fc_68477f0dff5c819ea17a1ffbaea621e00356a60c98816d6a',
                    )
                ],
                usage=RequestUsage(input_tokens=36, output_tokens=12, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68477f0d9494819ea4f123bba707c9ee0356a60c98816d6a',
                finish_reason='stop',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id='call_aTJhYjzmixZaVGqwl5gn2Ncr|fc_68477f0dff5c819ea17a1ffbaea621e00356a60c98816d6a',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='The largest city in Mexico is Mexico City.',
                        id='msg_68477f0ebf54819d88a44fa87aadaff503434b607c02582d',
                    )
                ],
                usage=RequestUsage(input_tokens=59, output_tokens=11, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68477f0e2b28819d9c828ef4ee526d6a03434b607c02582d',
                finish_reason='stop',
            ),
        ]
    )


@pytest.mark.vcr()
async def test_native_output(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))

    class CityLocation(BaseModel):
        """A city and its country."""

        city: str
        country: str

    agent = Agent(m, output_type=NativeOutput(CityLocation))

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
                parts=[ToolCallPart(tool_name='get_user_country', args='{}', tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=66, output_tokens=12, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68477f0f220081a1a621d6bcdc7f31a50b8591d9001d2329',
                finish_reason='stop',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id='call_tTAThu8l2S9hNky2krdwijGP|fc_68477f0fa7c081a19a525f7c6f180f310b8591d9001d2329',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='{"city":"Mexico City","country":"Mexico"}',
                        id='msg_68477f10846c81929f1e833b0785e6f3020197534e39cc1f',
                    )
                ],
                usage=RequestUsage(input_tokens=89, output_tokens=16, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68477f0fde708192989000a62809c6e5020197534e39cc1f',
                finish_reason='stop',
            ),
        ]
    )


@pytest.mark.vcr()
async def test_native_output_multiple(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))

    class CityLocation(BaseModel):
        city: str
        country: str

    class CountryLanguage(BaseModel):
        country: str
        language: str

    agent = Agent(m, output_type=NativeOutput([CityLocation, CountryLanguage]))

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
                parts=[ToolCallPart(tool_name='get_user_country', args='{}', tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=153, output_tokens=12, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68477f10f2d081a39b3438f413b3bafc0dd57d732903c563',
                finish_reason='stop',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id='call_UaLahjOtaM2tTyYZLxTCbOaP|fc_68477f1168a081a3981e847cd94275080dd57d732903c563',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='{"result":{"kind":"CityLocation","data":{"city":"Mexico City","country":"Mexico"}}}',
                        id='msg_68477f1235b8819d898adc64709c7ebf061ad97e2eef7871',
                    )
                ],
                usage=RequestUsage(input_tokens=176, output_tokens=26, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68477f119830819da162aa6e10552035061ad97e2eef7871',
                finish_reason='stop',
            ),
        ]
    )


@pytest.mark.vcr()
async def test_prompted_output(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))

    class CityLocation(BaseModel):
        city: str
        country: str

    agent = Agent(m, output_type=PromptedOutput(CityLocation))

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
                ],
                instructions="""\
Always respond with a JSON object that's compatible with this schema:

{"properties": {"city": {"type": "string"}, "country": {"type": "string"}}, "required": ["city", "country"], "title": "CityLocation", "type": "object"}

Don't include any text or Markdown fencing before or after.\
""",
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='get_user_country', args='{}', tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=107, output_tokens=12, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68482f12d63881a1830201ed101ecfbf02f8ef7f2fb42b50',
                finish_reason='stop',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id='call_FrlL4M0CbAy8Dhv4VqF1Shom|fc_68482f1b0ff081a1b37b9170ee740d1e02f8ef7f2fb42b50',
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
                parts=[
                    TextPart(
                        content='{"city":"Mexico City","country":"Mexico"}',
                        id='msg_68482f1c159081918a2405f458009a6a044fdb7d019d4115',
                    )
                ],
                usage=RequestUsage(input_tokens=130, output_tokens=12, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68482f1b556081918d64c9088a470bf0044fdb7d019d4115',
                finish_reason='stop',
            ),
        ]
    )


@pytest.mark.vcr()
async def test_prompted_output_multiple(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))

    class CityLocation(BaseModel):
        city: str
        country: str

    class CountryLanguage(BaseModel):
        country: str
        language: str

    agent = Agent(m, output_type=PromptedOutput([CityLocation, CountryLanguage]))

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
                ],
                instructions="""\
Always respond with a JSON object that's compatible with this schema:

{"type": "object", "properties": {"result": {"anyOf": [{"type": "object", "properties": {"kind": {"type": "string", "const": "CityLocation"}, "data": {"properties": {"city": {"type": "string"}, "country": {"type": "string"}}, "required": ["city", "country"], "type": "object"}}, "required": ["kind", "data"], "additionalProperties": false, "title": "CityLocation"}, {"type": "object", "properties": {"kind": {"type": "string", "const": "CountryLanguage"}, "data": {"properties": {"country": {"type": "string"}, "language": {"type": "string"}}, "required": ["country", "language"], "type": "object"}}, "required": ["kind", "data"], "additionalProperties": false, "title": "CountryLanguage"}]}}, "required": ["result"], "additionalProperties": false}

Don't include any text or Markdown fencing before or after.\
""",
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='get_user_country', args='{}', tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=283, output_tokens=12, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68482f1d38e081a1ac828acda978aa6b08e79646fe74d5ee',
                finish_reason='stop',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id='call_my4OyoVXRT0m7bLWmsxcaCQI|fc_68482f2889d481a199caa61de7ccb62c08e79646fe74d5ee',
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
                        content='{"result":{"kind":"CityLocation","data":{"city":"Mexico City","country":"Mexico"}}}',
                        id='msg_68482f296bfc81a18665547d4008ab2c06b4ab2d00d03024',
                    )
                ],
                usage=RequestUsage(input_tokens=306, output_tokens=22, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68482f28c1b081a1ae73cbbee012ee4906b4ab2d00d03024',
                finish_reason='stop',
            ),
        ]
    )


@pytest.mark.vcr()
async def test_openai_responses_verbosity(allow_model_requests: None, openai_api_key: str):
    """Test that verbosity setting is properly passed to the OpenAI API"""
    # Following GPT-5 + verbosity documentation pattern
    provider = OpenAIProvider(
        api_key=openai_api_key,
        base_url='https://api.openai.com/v1',  # Explicitly set base URL
    )
    model = OpenAIResponsesModel('gpt-5', provider=provider)
    agent = Agent(model=model, model_settings=OpenAIResponsesModelSettings(openai_text_verbosity='low'))
    result = await agent.run('What is 2+2?')
    assert result.output == snapshot('4')


@pytest.mark.vcr()
async def test_openai_previous_response_id(allow_model_requests: None, openai_api_key: str):
    """Test if previous responses are detected via previous_response_id in settings"""
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model)
    result = await agent.run('The secret key is sesame')
    settings = OpenAIResponsesModelSettings(openai_previous_response_id=result.all_messages()[-1].provider_response_id)  # type: ignore
    result = await agent.run('What is the secret code?', model_settings=settings)
    assert result.output == snapshot('sesame')


@pytest.mark.vcr()
async def test_openai_previous_response_id_auto_mode(allow_model_requests: None, openai_api_key: str):
    """Test if invalid previous response id is ignored when history contains non-OpenAI responses"""
    history = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content='The first secret key is sesame',
                ),
            ],
        ),
        ModelResponse(
            parts=[
                TextPart(content='Open sesame! What would you like to unlock?'),
            ],
            model_name='gpt-5',
            provider_name='openai',
            provider_response_id='resp_68b9bd97025c8195b443af591ca2345c08cb6072affe6099',
        ),
        ModelRequest(
            parts=[
                UserPromptPart(
                    content='The second secret key is olives',
                ),
            ],
        ),
        ModelResponse(
            parts=[
                TextPart(content='Understood'),
            ],
            model_name='gpt-5',
            provider_name='openai',
            provider_response_id='resp_68b9bda81f5c8197a5a51a20a9f4150a000497db2a4c777b',
        ),
    ]

    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model)
    settings = OpenAIResponsesModelSettings(openai_previous_response_id='auto')
    result = await agent.run('what is the first secret key', message_history=history, model_settings=settings)
    assert result.output == snapshot('sesame')


async def test_openai_previous_response_id_mixed_model_history(allow_model_requests: None, openai_api_key: str):
    """Test if invalid previous response id is ignored when history contains non-OpenAI responses"""
    history = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content='The first secret key is sesame',
                ),
            ],
        ),
        ModelResponse(
            parts=[
                TextPart(content='Open sesame! What would you like to unlock?'),
            ],
            model_name='claude-3-5-sonnet-latest',
            provider_name='anthropic',
            provider_response_id='msg_01XUQuedGz9gusk4xZm4gWJj',
        ),
        ModelRequest(
            parts=[
                UserPromptPart(
                    content='what is the first secret key?',
                ),
            ],
        ),
    ]

    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    previous_response_id, messages = model._get_previous_response_id_and_new_messages(history)  # type: ignore
    assert not previous_response_id
    assert messages == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='The first secret key is sesame', timestamp=IsDatetime())]),
            ModelResponse(
                parts=[TextPart(content='Open sesame! What would you like to unlock?')],
                usage=RequestUsage(),
                model_name='claude-3-5-sonnet-latest',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_response_id='msg_01XUQuedGz9gusk4xZm4gWJj',
            ),
            ModelRequest(parts=[UserPromptPart(content='what is the first secret key?', timestamp=IsDatetime())]),
        ]
    )


async def test_openai_previous_response_id_same_model_history(allow_model_requests: None, openai_api_key: str):
    """Test if message history is trimmed when model responses are from same model"""
    history = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content='The first secret key is sesame',
                ),
            ],
        ),
        ModelResponse(
            parts=[
                TextPart(content='Open sesame! What would you like to unlock?'),
            ],
            model_name='gpt-5',
            provider_name='openai',
            provider_response_id='resp_68b9bd97025c8195b443af591ca2345c08cb6072affe6099',
        ),
        ModelRequest(
            parts=[
                UserPromptPart(
                    content='The second secret key is olives',
                ),
            ],
        ),
        ModelResponse(
            parts=[
                TextPart(content='Understood'),
            ],
            model_name='gpt-5',
            provider_name='openai',
            provider_response_id='resp_68b9bda81f5c8197a5a51a20a9f4150a000497db2a4c777b',
        ),
        ModelRequest(
            parts=[
                UserPromptPart(
                    content='what is the first secret key?',
                ),
            ],
        ),
    ]

    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    previous_response_id, messages = model._get_previous_response_id_and_new_messages(history)  # type: ignore
    assert previous_response_id == 'resp_68b9bda81f5c8197a5a51a20a9f4150a000497db2a4c777b'
    assert messages == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='what is the first secret key?', timestamp=IsDatetime())]),
        ]
    )


async def test_openai_responses_usage_without_tokens_details(allow_model_requests: None):
    c = response_message(
        [
            ResponseOutputMessage(
                id='123',
                content=cast(list[Content], [ResponseOutputText(text='4', type='output_text', annotations=[])]),
                role='assistant',
                status='completed',
                type='message',
            )
        ],
        # Intentionally use model_construct so that input_tokens_details and output_tokens_details will not be set.
        usage=ResponseUsage.model_construct(input_tokens=14, output_tokens=1, total_tokens=15),
    )
    mock_client = MockOpenAIResponses.create_mock(c)
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))

    agent = Agent(model=model)
    result = await agent.run('What is 2+2?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is 2+2?',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[TextPart(content='4', id='123')],
                usage=RequestUsage(input_tokens=14, output_tokens=1, details={'reasoning_tokens': 0}),
                model_name='gpt-4o-123',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_response_id='123',
            ),
        ]
    )

    assert result.usage() == snapshot(
        RunUsage(input_tokens=14, output_tokens=1, details={'reasoning_tokens': 0}, requests=1)
    )


async def test_openai_responses_model_thinking_part(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    settings = OpenAIResponsesModelSettings(openai_reasoning_effort='high', openai_reasoning_summary='detailed')
    agent = Agent(m, model_settings=settings)

    result = await agent.run('How do I cross the street?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='How do I cross the street?', timestamp=IsDatetime())]),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c42c90b950819c9e32c46d4f8326ca07460311b0c8d3de',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    ThinkingPart(content=IsStr(), id='rs_68c42c90b950819c9e32c46d4f8326ca07460311b0c8d3de'),
                    ThinkingPart(content=IsStr(), id='rs_68c42c90b950819c9e32c46d4f8326ca07460311b0c8d3de'),
                    ThinkingPart(content=IsStr(), id='rs_68c42c90b950819c9e32c46d4f8326ca07460311b0c8d3de'),
                    ThinkingPart(content=IsStr(), id='rs_68c42c90b950819c9e32c46d4f8326ca07460311b0c8d3de'),
                    ThinkingPart(content=IsStr(), id='rs_68c42c90b950819c9e32c46d4f8326ca07460311b0c8d3de'),
                    TextPart(
                        content=IsStr(),
                        id='msg_68c42cb1aaec819cb992bd92a8c7766007460311b0c8d3de',
                    ),
                ],
                usage=RequestUsage(input_tokens=13, output_tokens=2199, details={'reasoning_tokens': 1920}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68c42c902794819cb9335264c342f65407460311b0c8d3de',
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
                        id='rs_68c42cb43d3c819caf078978cc2514ea07460311b0c8d3de',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    ThinkingPart(content=IsStr(), id='rs_68c42cb43d3c819caf078978cc2514ea07460311b0c8d3de'),
                    ThinkingPart(content=IsStr(), id='rs_68c42cb43d3c819caf078978cc2514ea07460311b0c8d3de'),
                    ThinkingPart(content=IsStr(), id='rs_68c42cb43d3c819caf078978cc2514ea07460311b0c8d3de'),
                    ThinkingPart(content=IsStr(), id='rs_68c42cb43d3c819caf078978cc2514ea07460311b0c8d3de'),
                    TextPart(
                        content=IsStr(),
                        id='msg_68c42cd36134819c800463490961f7df07460311b0c8d3de',
                    ),
                ],
                usage=RequestUsage(input_tokens=314, output_tokens=2737, details={'reasoning_tokens': 2112}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68c42cb3d520819c9d28b07036e9059507460311b0c8d3de',
                finish_reason='stop',
            ),
        ]
    )


async def test_openai_responses_thinking_part_from_other_model(
    allow_model_requests: None, anthropic_api_key: str, openai_api_key: str
):
    m = AnthropicModel(
        'claude-sonnet-4-0',
        provider=AnthropicProvider(api_key=anthropic_api_key),
        settings=AnthropicModelSettings(anthropic_thinking={'type': 'enabled', 'budget_tokens': 1024}),
    )
    agent = Agent(m)

    result = await agent.run('How do I cross the street?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='How do I cross the street?',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        signature=IsStr(),
                        provider_name='anthropic',
                    ),
                    TextPart(content=IsStr()),
                ],
                usage=RequestUsage(
                    input_tokens=42,
                    output_tokens=291,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 42,
                        'output_tokens': 291,
                    },
                ),
                model_name='claude-sonnet-4-20250514',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='msg_0114iHK2ditgTf1N8FWomc4E',
                finish_reason='stop',
            ),
        ]
    )

    result = await agent.run(
        'Considering the way to cross the street, analogously, how do I cross the river?',
        model=OpenAIResponsesModel(
            'gpt-5',
            provider=OpenAIProvider(api_key=openai_api_key),
            settings=OpenAIResponsesModelSettings(openai_reasoning_effort='high', openai_reasoning_summary='detailed'),
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
                        id='rs_68c42ce323d48193bcf88db6278980cf0ad492c7955fc6fc',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    ThinkingPart(content=IsStr(), id='rs_68c42ce323d48193bcf88db6278980cf0ad492c7955fc6fc'),
                    ThinkingPart(content=IsStr(), id='rs_68c42ce323d48193bcf88db6278980cf0ad492c7955fc6fc'),
                    ThinkingPart(content=IsStr(), id='rs_68c42ce323d48193bcf88db6278980cf0ad492c7955fc6fc'),
                    ThinkingPart(content=IsStr(), id='rs_68c42ce323d48193bcf88db6278980cf0ad492c7955fc6fc'),
                    ThinkingPart(content=IsStr(), id='rs_68c42ce323d48193bcf88db6278980cf0ad492c7955fc6fc'),
                    TextPart(content=IsStr(), id='msg_68c42d0b5e5c819385352dde1f447d910ad492c7955fc6fc'),
                ],
                usage=RequestUsage(input_tokens=306, output_tokens=3134, details={'reasoning_tokens': 2496}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68c42ce277ac8193ba08881bcefabaf70ad492c7955fc6fc',
                finish_reason='stop',
            ),
        ]
    )


async def test_openai_responses_thinking_part_iter(allow_model_requests: None, openai_api_key: str):
    provider = OpenAIProvider(api_key=openai_api_key)
    responses_model = OpenAIResponsesModel('o3-mini', provider=provider)
    settings = OpenAIResponsesModelSettings(openai_reasoning_effort='high', openai_reasoning_summary='detailed')
    agent = Agent(responses_model, model_settings=settings)

    async with agent.iter(user_prompt='How do I cross the street?') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for _ in request_stream:
                        pass

    assert agent_run.result is not None
    assert agent_run.result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='How do I cross the street?',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c42d1d0878819d8266007cd3d1402c08fbf9b1584184ff',
                        signature='gAAAAABoxC0m_QWpOlSt8wyPk_gtnjiI4mNLOryYlNXO-6rrVeIqBYDDAyMVg2_ldboZvfhW8baVbpki29gkTAyNygTr7L8gF1XK0hFovoa23ZYJKvuOnyLIJF-rXCsbDG7YdMYhi3bm82pMFVQxNK4r5muWCQcHmyJ2S1YtBoJtF_D1Ah7GpW2ACvJWsGikb3neAOnI-RsmUxCRu-cew7rVWfSj8jFKs8RGNQRvDaUzVniaMXJxVW9T5C7Ytzi852MF1PfVq0U-aNBzZBtAdwQcbn5KZtGkYLYTChmCi2hMrh5-lg9CgS8pqqY9-jv2EQvKHIumdv6oLiW8K59Zvo8zGxYoqT--osfjfS0vPZhTHiSX4qCkK30YNJrWHKJ95Hpe23fnPBL0nEQE5l6XdhsyY7TwMom016P3dgWwgP5AtWmQ30zeXDs=',
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c42d1d0878819d8266007cd3d1402c08fbf9b1584184ff',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c42d1d0878819d8266007cd3d1402c08fbf9b1584184ff',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c42d1d0878819d8266007cd3d1402c08fbf9b1584184ff',
                    ),
                    TextPart(
                        content=IsStr(),
                        id='msg_68c42d26866c819da8d5c606621c911608fbf9b1584184ff',
                    ),
                ],
                usage=RequestUsage(input_tokens=13, output_tokens=1680, details={'reasoning_tokens': 1408}),
                model_name='o3-mini-2025-01-31',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68c42d0fb418819dbfa579f69406b49508fbf9b1584184ff',
                finish_reason='stop',
            ),
        ]
    )


async def test_openai_responses_thinking_with_tool_calls(allow_model_requests: None, openai_api_key: str):
    provider = OpenAIProvider(api_key=openai_api_key)
    m = OpenAIResponsesModel(
        model_name='gpt-5',
        provider=provider,
        settings=OpenAIResponsesModelSettings(openai_reasoning_summary='detailed', openai_reasoning_effort='low'),
    )
    agent = Agent(model=m)

    @agent.instructions
    def system_prompt():
        return (
            'You are a helpful assistant that uses planning. You MUST use the update_plan tool and continually '
            "update it as you make progress against the user's prompt"
        )

    @agent.tool_plain
    def update_plan(plan: str) -> str:
        return 'plan updated'

    prompt = (
        'Compose a 12-line poem where the first letters of the odd-numbered lines form the name "SAMIRA" '
        'and the first letters of the even-numbered lines spell out "DAWOOD." Additionally, the first letter '
        'of each word in every line should create the capital of a country'
    )

    result = await agent.run(prompt)

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Compose a 12-line poem where the first letters of the odd-numbered lines form the name "SAMIRA" and the first letters of the even-numbered lines spell out "DAWOOD." Additionally, the first letter of each word in every line should create the capital of a country',
                        timestamp=IsDatetime(),
                    )
                ],
                instructions="You are a helpful assistant that uses planning. You MUST use the update_plan tool and continually update it as you make progress against the user's prompt",
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c42d29124881968e24c1ca8c1fc7860e8bc41441c948f6',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    ThinkingPart(content=IsStr(), id='rs_68c42d29124881968e24c1ca8c1fc7860e8bc41441c948f6'),
                    ThinkingPart(content=IsStr(), id='rs_68c42d29124881968e24c1ca8c1fc7860e8bc41441c948f6'),
                    ThinkingPart(content=IsStr(), id='rs_68c42d29124881968e24c1ca8c1fc7860e8bc41441c948f6'),
                    ThinkingPart(content=IsStr(), id='rs_68c42d29124881968e24c1ca8c1fc7860e8bc41441c948f6'),
                    ToolCallPart(
                        tool_name='update_plan',
                        args=IsStr(),
                        tool_call_id='call_gL7JE6GDeGGsFubqO2XGytyO|fc_68c42d3e9e4881968b15fbb8253f58540e8bc41441c948f6',
                    ),
                ],
                usage=RequestUsage(input_tokens=124, output_tokens=1926, details={'reasoning_tokens': 1792}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68c42d28772c819684459966ee2201ed0e8bc41441c948f6',
                finish_reason='stop',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='update_plan',
                        content='plan updated',
                        tool_call_id='call_gL7JE6GDeGGsFubqO2XGytyO|fc_68c42d3e9e4881968b15fbb8253f58540e8bc41441c948f6',
                        timestamp=IsDatetime(),
                    )
                ],
                instructions="You are a helpful assistant that uses planning. You MUST use the update_plan tool and continually update it as you make progress against the user's prompt",
            ),
            ModelResponse(
                parts=[TextPart(content=IsStr(), id='msg_68c42d408eec8196ae1c5883e07c093e0e8bc41441c948f6')],
                usage=RequestUsage(
                    input_tokens=2087, cache_read_tokens=2048, output_tokens=124, details={'reasoning_tokens': 0}
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68c42d3fd6a08196bce23d6be960ff8a0e8bc41441c948f6',
                finish_reason='stop',
            ),
        ]
    )


async def test_openai_responses_thinking_without_summary(allow_model_requests: None):
    c = response_message(
        [
            ResponseReasoningItem(
                id='rs_123',
                summary=[],
                type='reasoning',
                encrypted_content='123',
            ),
            ResponseOutputMessage(
                id='msg_123',
                content=cast(list[Content], [ResponseOutputText(text='4', type='output_text', annotations=[])]),
                role='assistant',
                status='completed',
                type='message',
            ),
        ],
    )
    mock_client = MockOpenAIResponses.create_mock(c)
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(openai_client=mock_client))

    agent = Agent(model=model)
    result = await agent.run('What is 2+2?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is 2+2?',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(content='', id='rs_123', signature='123', provider_name='openai'),
                    TextPart(content='4', id='msg_123'),
                ],
                model_name='gpt-4o-123',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_response_id='123',
            ),
        ]
    )

    _, openai_messages = await model._map_messages(result.all_messages(), model_settings=model.settings or {})  # type: ignore[reportPrivateUsage]
    assert openai_messages == snapshot(
        [
            {'role': 'user', 'content': 'What is 2+2?'},
            {'id': 'rs_123', 'summary': [], 'encrypted_content': '123', 'type': 'reasoning'},
            {
                'role': 'assistant',
                'id': 'msg_123',
                'content': [{'text': '4', 'type': 'output_text', 'annotations': []}],
                'type': 'message',
                'status': 'completed',
            },
        ]
    )


async def test_openai_responses_thinking_with_multiple_summaries(allow_model_requests: None):
    c = response_message(
        [
            ResponseReasoningItem(
                id='rs_123',
                summary=[
                    Summary(text='1', type='summary_text'),
                    Summary(text='2', type='summary_text'),
                    Summary(text='3', type='summary_text'),
                    Summary(text='4', type='summary_text'),
                ],
                type='reasoning',
                encrypted_content='123',
            ),
            ResponseOutputMessage(
                id='msg_123',
                content=cast(list[Content], [ResponseOutputText(text='4', type='output_text', annotations=[])]),
                role='assistant',
                status='completed',
                type='message',
            ),
        ],
    )
    mock_client = MockOpenAIResponses.create_mock(c)
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(openai_client=mock_client))

    agent = Agent(model=model)
    result = await agent.run('What is 2+2?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is 2+2?',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(content='1', id='rs_123', signature='123', provider_name='openai'),
                    ThinkingPart(content='2', id='rs_123'),
                    ThinkingPart(content='3', id='rs_123'),
                    ThinkingPart(content='4', id='rs_123'),
                    TextPart(content='4', id='msg_123'),
                ],
                model_name='gpt-4o-123',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_response_id='123',
            ),
        ]
    )

    _, openai_messages = await model._map_messages(result.all_messages(), model_settings=model.settings or {})  # type: ignore[reportPrivateUsage]
    assert openai_messages == snapshot(
        [
            {'role': 'user', 'content': 'What is 2+2?'},
            {
                'id': 'rs_123',
                'summary': [
                    {'text': '1', 'type': 'summary_text'},
                    {'text': '2', 'type': 'summary_text'},
                    {'text': '3', 'type': 'summary_text'},
                    {'text': '4', 'type': 'summary_text'},
                ],
                'encrypted_content': '123',
                'type': 'reasoning',
            },
            {
                'role': 'assistant',
                'id': 'msg_123',
                'content': [{'text': '4', 'type': 'output_text', 'annotations': []}],
                'type': 'message',
                'status': 'completed',
            },
        ]
    )


async def test_openai_responses_thinking_with_modified_history(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    settings = OpenAIResponsesModelSettings(openai_reasoning_effort='low', openai_reasoning_summary='detailed')
    agent = Agent(m, model_settings=settings)

    result = await agent.run('What is the meaning of life?')
    messages = result.all_messages()
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the meaning of life?',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c42de022c881948db7ed1cc2529f2e0202c9ad459e0d23',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    TextPart(content=IsStr(), id='msg_68c42de31d348194a251b43ad913ef140202c9ad459e0d23'),
                ],
                usage=RequestUsage(input_tokens=13, output_tokens=248, details={'reasoning_tokens': 64}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68c42ddf9bbc8194aa7b97304dd909cb0202c9ad459e0d23',
                finish_reason='stop',
            ),
        ]
    )

    response = messages[-1]
    assert isinstance(response, ModelResponse)
    assert isinstance(response.parts, list)
    response.parts[1] = TextPart(content='The meaning of life is 42')

    with pytest.raises(
        ModelHTTPError,
        match=r"Item '.*' of type 'reasoning' was provided without its required following item\.",
    ):
        await agent.run('Anything to add?', message_history=messages)

    result = await agent.run(
        'Anything to add?',
        message_history=messages,
        model_settings=OpenAIResponsesModelSettings(
            openai_reasoning_effort='low',
            openai_reasoning_summary='detailed',
            openai_send_reasoning_ids=False,
        ),
    )
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Anything to add?',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c42de4f63c819fb31b6019a4eaf67c051f82c608a83beb',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    TextPart(content=IsStr(), id='msg_68c42de8a410819faf7a9cbebd2b4bc4051f82c608a83beb'),
                ],
                usage=RequestUsage(input_tokens=142, output_tokens=355, details={'reasoning_tokens': 128}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68c42de4afcc819f995a1c59fe87c9d5051f82c608a83beb',
                finish_reason='stop',
            ),
        ]
    )


async def test_openai_responses_thinking_with_code_execution_tool(allow_model_requests: None, openai_api_key: str):
    provider = OpenAIProvider(api_key=openai_api_key)
    m = OpenAIResponsesModel(
        model_name='gpt-5',
        provider=provider,
        settings=OpenAIResponsesModelSettings(
            openai_reasoning_summary='detailed',
            openai_reasoning_effort='low',
            openai_include_code_execution_outputs=True,
        ),
    )
    agent = Agent(model=m, builtin_tools=[CodeExecutionTool()])

    result = await agent.run(user_prompt="what's 123456 to the power of 123?")
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content="what's 123456 to the power of 123?",
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_0cccbe0ca95f935e0068c4986ab048819f9c0f1478f3a295c2',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args={
                            'code': """\
n = pow(123456, 123)
len_str = len(str(n))
len_str, str(n)[:50], str(n)[-50:]\
""",
                            'container_id': 'cntr_68c4986a2df48191acf927f03b5ab8150130cf654974eafb',
                        },
                        tool_call_id='ci_0cccbe0ca95f935e0068c4986ed690819f9dbd272c4c70a011',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={
                            'outputs': [
                                dict(
                                    logs="""\
(627,
 '18030210630404480750814092786593857280734268863855',
 '29749134489643622579100908331839817426366854332416')\
""",
                                    type='logs',
                                )
                            ],
                            'status': 'completed',
                        },
                        tool_call_id='ci_0cccbe0ca95f935e0068c4986ed690819f9dbd272c4c70a011',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_0cccbe0ca95f935e0068c498746e68819fbcaef179058df3cc',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args={
                            'code': """\
str_n = str(n)
str_n[:200]\
""",
                            'container_id': 'cntr_68c4986a2df48191acf927f03b5ab8150130cf654974eafb',
                        },
                        tool_call_id='ci_0cccbe0ca95f935e0068c49877936c819fbcd8ce718f5bc31a',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={
                            'outputs': [
                                dict(
                                    logs="'18030210630404480750814092786593857280734268863855968048844015985795850236081373250219782696986322573087163043641979475893207435038036769764981462654292660266470727587426920177774391231319751632369022'",
                                    type='logs',
                                )
                            ],
                            'status': 'completed',
                        },
                        tool_call_id='ci_0cccbe0ca95f935e0068c49877936c819fbcd8ce718f5bc31a',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content='',
                        id='rs_0cccbe0ca95f935e0068c49877a260819faa558e81b74d00a3',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    TextPart(
                        content="""\
123456^123 equals:
1803021063040448075081409278659385728073426886385596804884401598579585023608137325021978269698632257308716304364197947589320743503803676976498146265429266026647072758742692017777439123131975163236902290188202654590011462134235078832526827852273018210815142998256983234516628795109978467862737585124291404312560193679040132194219142159564780429384029784135632838235232349153620928650701305446902198201185265537637166663065255873102180259349606640396746581577358565927727487182715643033427374054356948524185042601095118624154879402247254855530736695404690558487305849085262939429771481280865270916688165704128104324472168440830911119701969936\
""",
                        id='msg_0cccbe0ca95f935e0068c49877dd4c819fb0949326ca53a0cb',
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=2629, cache_read_tokens=2304, output_tokens=354, details={'reasoning_tokens': 128}
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_0cccbe0ca95f935e0068c498673f08819f9ca3a5ffea0f8f34',
                finish_reason='stop',
            ),
        ]
    )

    messages = result.all_messages()
    result = await agent.run(user_prompt='how about to the power of 124?', message_history=messages)
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='how about to the power of 124?',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_0cccbe0ca95f935e0068c4987dfd64819fbd3984a1d0b6ae8a',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args={
                            'code': """\
n = pow(123456, 124)
str_n = str(n)
len(str_n)\
""",
                            'container_id': 'cntr_68c4986a2df48191acf927f03b5ab8150130cf654974eafb',
                        },
                        tool_call_id='ci_0cccbe0ca95f935e0068c498806d50819f81b64eebd0a9afb5',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={'outputs': [dict(logs='632', type='logs')], 'status': 'completed'},
                        tool_call_id='ci_0cccbe0ca95f935e0068c498806d50819f81b64eebd0a9afb5',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args={
                            'code': 'str_n[:200]',
                            'container_id': 'cntr_68c4986a2df48191acf927f03b5ab8150130cf654974eafb',
                        },
                        tool_call_id='ci_0cccbe0ca95f935e0068c49880a0fc819f9b7df3346d75008a',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={
                            'outputs': [
                                dict(
                                    logs="'22259376835872155755725046390617312444503298968562023914380868375424124867456620159791334926391434395830488007158642181758718171000558674481055514534483546658574101450333778574673152438538112575257499'",
                                    type='logs',
                                )
                            ],
                            'status': 'completed',
                        },
                        tool_call_id='ci_0cccbe0ca95f935e0068c49880a0fc819f9b7df3346d75008a',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args={
                            'code': 'str_n[-200:]',
                            'container_id': 'cntr_68c4986a2df48191acf927f03b5ab8150130cf654974eafb',
                        },
                        tool_call_id='ci_0cccbe0ca95f935e0068c498829764819fa5d652f285f51fb4',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={
                            'outputs': [
                                dict(
                                    logs="'67296874992296948890272517148532094528812086889630338972525614076265743359820638354575643811422961497469318295177638226345664642985187040350404616587909147553443069125481739015616500189546368462749696'",
                                    type='logs',
                                )
                            ],
                            'status': 'completed',
                        },
                        tool_call_id='ci_0cccbe0ca95f935e0068c498829764819fa5d652f285f51fb4',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args={'code': 'str_n', 'container_id': 'cntr_68c4986a2df48191acf927f03b5ab8150130cf654974eafb'},
                        tool_call_id='ci_0cccbe0ca95f935e0068c498850e30819f925c0f879f69541b',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={
                            'outputs': [
                                dict(
                                    logs="'22259376835872155755725046390617312444503298968562023914380868375424124867456620159791334926391434395830488007158642181758718171000558674481055514534483546658574101450333778574673152438538112575257499957691072558869631827866367698332298176634455912721514439215697282609503998798205923943259906494776122169432383635142770936484489945914800519238574614164957266189867725276688514819945285124592421478579353115389025899375605896015758767296874992296948890272517148532094528812086889630338972525614076265743359820638354575643811422961497469318295177638226345664642985187040350404616587909147553443069125481739015616500189546368462749696'",
                                    type='logs',
                                )
                            ],
                            'status': 'completed',
                        },
                        tool_call_id='ci_0cccbe0ca95f935e0068c498850e30819f925c0f879f69541b',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    TextPart(
                        content="""\
123456^124 equals:
22259376835872155755725046390617312444503298968562023914380868375424124867456620159791334926391434395830488007158642181758718171000558674481055514534483546658574101450333778574673152438538112575257499957691072558869631827866367698332298176634455912721514439215697282609503998798205923943259906494776122169432383635142770936484489945914800519238574614164957266189867725276688514819945285124592421478579353115389025899375605896015758767296874992296948890272517148532094528812086889630338972525614076265743359820638354575643811422961497469318295177638226345664642985187040350404616587909147553443069125481739015616500189546368462749696\
""",
                        id='msg_0cccbe0ca95f935e0068c49886b928819f942185e7cf43c579',
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=5973, cache_read_tokens=5120, output_tokens=221, details={'reasoning_tokens': 0}
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_0cccbe0ca95f935e0068c4987bea4c819fbd0d975202557b61',
                finish_reason='stop',
            ),
        ]
    )


async def test_openai_responses_thinking_with_code_execution_tool_stream(
    allow_model_requests: None, openai_api_key: str
):
    provider = OpenAIProvider(api_key=openai_api_key)
    m = OpenAIResponsesModel(
        model_name='gpt-5',
        provider=provider,
        settings=OpenAIResponsesModelSettings(openai_reasoning_summary='detailed', openai_reasoning_effort='low'),
    )
    agent = Agent(model=m, builtin_tools=[CodeExecutionTool()])

    event_parts: list[Any] = []
    async with agent.iter(user_prompt="what's 123456 to the power of 123?") as agent_run:
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
                    UserPromptPart(
                        content="what's 123456 to the power of 123?",
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c3509b2ee0819eba32735182d275ad0f2d670b80edc507',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args='{"container_id":"cntr_68c3509aa0348191ad0bfefe24878dbb0deaa35a4e39052e","code":"n = pow(123456, 123)\\nlen(str(n))"}',
                        tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={'status': 'completed'},
                        tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args='{"container_id":"cntr_68c3509aa0348191ad0bfefe24878dbb0deaa35a4e39052e","code":"str(n)[:100], str(n)[-100:]"}',
                        tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={'status': 'completed'},
                        tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args='{"container_id":"cntr_68c3509aa0348191ad0bfefe24878dbb0deaa35a4e39052e","code":"n"}',
                        tool_call_id='ci_68c350a5e1f8819eb082eccb870199ec0f2d670b80edc507',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={'status': 'completed'},
                        tool_call_id='ci_68c350a5e1f8819eb082eccb870199ec0f2d670b80edc507',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    TextPart(
                        content=IsStr(),
                        id='msg_68c350a75ddc819ea5406470460be7850f2d670b80edc507',
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=3727, cache_read_tokens=3200, output_tokens=347, details={'reasoning_tokens': 128}
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68c35098e6fc819e80fb94b25b7d031b0f2d670b80edc507',
                finish_reason='stop',
            ),
        ]
    )

    assert event_parts == snapshot(
        [
            PartStartEvent(
                index=0, part=ThinkingPart(content='', id='rs_68c3509b2ee0819eba32735182d275ad0f2d670b80edc507')
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='**Calcul')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ating')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' large')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' integer')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
**

I\
"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' need')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' compute')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' 123')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='456')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' raised')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' power')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' 123')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' That')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' an')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' enormous')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' integer')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' user')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' probably')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' wants')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' exact')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' value')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' can')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' use')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Python')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'s")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ability')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' handle')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' big')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' integers')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' but')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' output')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' will')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' likely')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' be')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' extremely')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' long')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' —')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' potentially')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' hundreds')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' digits')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' should')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' consider')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' that')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' prepare')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' return')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' result')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' as')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' plain')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' text')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' even')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' if')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' it')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ends')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' up')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' being')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' around')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' 627')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' digits')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' So')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' let')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' go')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ahead')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' compute')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' that')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='!')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    signature_delta=IsStr(),
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=1,
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507',
                    provider_name='openai',
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='{"container_id":"cntr_68c3509aa0348191ad0bfefe24878dbb0deaa35a4e39052e","code":"',
                    tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507',
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='n', tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' =', tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' pow', tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='(', tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='123', tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='456', tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=',', tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=' ', tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='123', tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta=')\\n', tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='len', tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='(str', tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='(n', tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='))', tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='"}', tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507'
                ),
            ),
            PartStartEvent(
                index=2,
                part=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={'status': 'completed'},
                    tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=3,
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507',
                    provider_name='openai',
                ),
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(
                    args_delta='{"container_id":"cntr_68c3509aa0348191ad0bfefe24878dbb0deaa35a4e39052e","code":"',
                    tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507',
                ),
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(
                    args_delta='str', tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(
                    args_delta='(n', tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(
                    args_delta=')', tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(
                    args_delta='[:', tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(
                    args_delta='100', tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(
                    args_delta='],', tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(
                    args_delta=' str', tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(
                    args_delta='(n', tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(
                    args_delta=')[', tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(
                    args_delta='-', tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(
                    args_delta='100', tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(
                    args_delta=':]', tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(
                    args_delta='"}', tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507'
                ),
            ),
            PartStartEvent(
                index=4,
                part=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={'status': 'completed'},
                    tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=5,
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    tool_call_id='ci_68c350a5e1f8819eb082eccb870199ec0f2d670b80edc507',
                    provider_name='openai',
                ),
            ),
            PartDeltaEvent(
                index=5,
                delta=ToolCallPartDelta(
                    args_delta='{"container_id":"cntr_68c3509aa0348191ad0bfefe24878dbb0deaa35a4e39052e","code":"',
                    tool_call_id='ci_68c350a5e1f8819eb082eccb870199ec0f2d670b80edc507',
                ),
            ),
            PartDeltaEvent(
                index=5,
                delta=ToolCallPartDelta(
                    args_delta='n', tool_call_id='ci_68c350a5e1f8819eb082eccb870199ec0f2d670b80edc507'
                ),
            ),
            PartDeltaEvent(
                index=5,
                delta=ToolCallPartDelta(
                    args_delta='"}', tool_call_id='ci_68c350a5e1f8819eb082eccb870199ec0f2d670b80edc507'
                ),
            ),
            PartStartEvent(
                index=6,
                part=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={'status': 'completed'},
                    tool_call_id='ci_68c350a5e1f8819eb082eccb870199ec0f2d670b80edc507',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                ),
            ),
            PartStartEvent(
                index=7, part=TextPart(content='123', id='msg_68c350a75ddc819ea5406470460be7850f2d670b80edc507')
            ),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='456')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='^')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='123')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta=' equals')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta=':\n')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='180')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='302')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='106')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='304')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='044')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='807')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='508')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='140')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='927')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='865')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='938')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='572')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='807')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='342')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='688')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='638')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='559')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='680')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='488')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='440')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='159')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='857')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='958')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='502')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='360')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='813')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='732')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='502')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='197')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='826')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='969')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='863')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='225')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='730')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='871')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='630')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='436')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='419')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='794')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='758')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='932')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='074')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='350')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='380')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='367')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='697')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='649')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='814')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='626')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='542')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='926')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='602')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='664')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='707')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='275')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='874')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='269')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='201')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='777')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='743')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='912')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='313')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='197')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='516')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='323')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='690')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='221')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='274')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='713')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='845')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='895')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='457')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='748')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='735')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='309')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='484')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='337')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='191')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='373')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='255')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='527')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='928')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='271')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='785')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='206')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='382')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='967')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='998')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='984')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='330')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='482')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='105')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='350')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='942')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='229')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='970')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='677')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='054')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='940')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='838')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='210')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='936')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='952')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='303')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='939')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='401')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='656')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='756')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='127')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='607')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='778')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='599')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='667')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='243')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='702')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='814')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='072')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='746')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='219')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='431')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='942')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='293')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='005')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='416')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='411')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='635')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='076')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='021')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='296')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='045')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='493')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='305')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='133')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='645')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='615')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='566')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='590')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='735')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='965')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='652')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='587')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='934')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='290')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='425')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='473')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='827')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='719')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='935')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='012')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='870')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='093')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='575')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='987')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='789')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='431')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='818')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='047')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='013')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='404')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='691')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='795')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='773')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='170')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='405')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='764')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='614')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='646')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='054')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='949')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='298')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='846')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='184')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='678')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='296')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='813')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='625')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='595')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='333')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='311')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='611')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='385')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='251')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='735')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='244')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='505')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='448')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='443')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='050')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='050')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='547')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='161')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='779')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='229')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='749')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='134')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='489')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='643')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='622')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='579')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='100')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='908')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='331')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='839')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='817')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='426')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='366')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='854')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='332')),
            PartDeltaEvent(index=7, delta=TextPartDelta(content_delta='416')),
            BuiltinToolCallEvent(  # pyright: ignore[reportDeprecated]
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    args='{"container_id":"cntr_68c3509aa0348191ad0bfefe24878dbb0deaa35a4e39052e","code":"n = pow(123456, 123)\\nlen(str(n))"}',
                    tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507',
                    provider_name='openai',
                )
            ),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={'status': 'completed'},
                    tool_call_id='ci_68c3509faff0819e96f6d45e6faf78490f2d670b80edc507',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                )
            ),
            BuiltinToolCallEvent(  # pyright: ignore[reportDeprecated]
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    args='{"container_id":"cntr_68c3509aa0348191ad0bfefe24878dbb0deaa35a4e39052e","code":"str(n)[:100], str(n)[-100:]"}',
                    tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507',
                    provider_name='openai',
                )
            ),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={'status': 'completed'},
                    tool_call_id='ci_68c350a41d2c819ebb23bdfb9ff322770f2d670b80edc507',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                )
            ),
            BuiltinToolCallEvent(  # pyright: ignore[reportDeprecated]
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    args='{"container_id":"cntr_68c3509aa0348191ad0bfefe24878dbb0deaa35a4e39052e","code":"n"}',
                    tool_call_id='ci_68c350a5e1f8819eb082eccb870199ec0f2d670b80edc507',
                    provider_name='openai',
                )
            ),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={'status': 'completed'},
                    tool_call_id='ci_68c350a5e1f8819eb082eccb870199ec0f2d670b80edc507',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                )
            ),
        ]
    )


async def test_openai_responses_streaming_usage(allow_model_requests: None, openai_api_key: str):
    class Result(BaseModel):
        result: int

    agent = Agent(
        model=OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key)),
        model_settings=OpenAIResponsesModelSettings(
            openai_reasoning_effort='low',
            openai_service_tier='flex',
        ),
        output_type=Result,
    )

    async with agent.iter('Calculate 100 * 200 / 3') as run:
        async for node in run:
            if Agent.is_model_request_node(node):
                async with node.stream(run.ctx) as response_stream:
                    async for _ in response_stream:
                        pass
                    assert response_stream.get().usage == snapshot(
                        RequestUsage(input_tokens=53, output_tokens=469, details={'reasoning_tokens': 448})
                    )
                    assert response_stream.usage() == snapshot(
                        RunUsage(input_tokens=53, output_tokens=469, details={'reasoning_tokens': 448}, requests=1)
                    )
                    assert run.usage() == snapshot(RunUsage(requests=1))
                assert run.usage() == snapshot(
                    RunUsage(input_tokens=53, output_tokens=469, details={'reasoning_tokens': 448}, requests=1)
                )
    assert run.usage() == snapshot(
        RunUsage(input_tokens=53, output_tokens=469, details={'reasoning_tokens': 448}, requests=1)
    )


async def test_openai_responses_non_reasoning_model_no_item_ids(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-4.1', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model)

    @agent.tool_plain
    def get_meaning_of_life() -> int:
        return 42

    result = await agent.run('What is the meaning of life?')
    messages = result.all_messages()
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the meaning of life?',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_meaning_of_life',
                        args='{}',
                        tool_call_id='call_3WCunBU7lCG1HHaLmnnRJn8I|fc_68cc4fa649ac8195b0c6c239cd2c14470548824120ffcf74',
                    )
                ],
                usage=RequestUsage(input_tokens=36, output_tokens=15, details={'reasoning_tokens': 0}),
                model_name='gpt-4.1-2025-04-14',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68cc4fa5603481958e2143685133fe530548824120ffcf74',
                finish_reason='stop',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_meaning_of_life',
                        content=42,
                        tool_call_id='call_3WCunBU7lCG1HHaLmnnRJn8I|fc_68cc4fa649ac8195b0c6c239cd2c14470548824120ffcf74',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="""\
The meaning of life, according to popular culture and famously in Douglas Adams' "The Hitchhiker's Guide to the Galaxy," is 42!

If you're looking for a deeper or philosophical answer, let me know your perspective or context, and I can elaborate further.\
""",
                        id='msg_68cc4fa7693081a184ff6f32e5209ab00307c6d4d2ee5985',
                    )
                ],
                usage=RequestUsage(input_tokens=61, output_tokens=56, details={'reasoning_tokens': 0}),
                model_name='gpt-4.1-2025-04-14',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68cc4fa6a8a881a187b0fe1603057bff0307c6d4d2ee5985',
                finish_reason='stop',
            ),
        ]
    )

    _, openai_messages = await model._map_messages(messages, model_settings=model.settings or {})  # type: ignore[reportPrivateUsage]
    assert openai_messages == snapshot(
        [
            {'role': 'user', 'content': 'What is the meaning of life?'},
            {
                'name': 'get_meaning_of_life',
                'arguments': '{}',
                'call_id': 'call_3WCunBU7lCG1HHaLmnnRJn8I',
                'type': 'function_call',
            },
            {'type': 'function_call_output', 'call_id': 'call_3WCunBU7lCG1HHaLmnnRJn8I', 'output': '42'},
            {
                'role': 'assistant',
                'content': """\
The meaning of life, according to popular culture and famously in Douglas Adams' "The Hitchhiker's Guide to the Galaxy," is 42!

If you're looking for a deeper or philosophical answer, let me know your perspective or context, and I can elaborate further.\
""",
            },
        ]
    )


async def test_openai_responses_code_execution_return_image(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel(
        'gpt-5',
        provider=OpenAIProvider(api_key=openai_api_key),
        settings=OpenAIResponsesModelSettings(openai_include_code_execution_outputs=True),
    )

    ad_hoc_agent = Agent(
        model=model,
        builtin_tools=[CodeExecutionTool()],
    )

    result = await ad_hoc_agent.run('Create a chart of y=x^2 for x=-5 to 5')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Create a chart of y=x^2 for x=-5 to 5',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_68cc849a4420819dbd821ade7e9b5d5a07865e46140d854a',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args={
                            'container_id': 'cntr_68cc8499e0c48191841feeb7f14a9d2e066423ab27e58f8e',
                            'code': """\
# Create and display a chart of y = x^2 for x from -5 to 5\r
import numpy as np\r
import matplotlib.pyplot as plt\r
\r
# Generate data\r
x = np.linspace(-5, 5, 1000)\r
y = x**2\r
\r
# Plot\r
plt.figure(figsize=(6, 4))\r
plt.plot(x, y, color='royalblue', linewidth=2, label='y = x^2')\r
plt.title('y = x^2 for x in [-5, 5]')\r
plt.xlabel('x')\r
plt.ylabel('y')\r
plt.grid(True, linestyle='--', alpha=0.5)\r
plt.legend()\r
\r
# Save and show\r
outfile = '/mnt/data/y_equals_x_squared.png'\r
plt.savefig(outfile, dpi=150, bbox_inches='tight')\r
plt.show()\r
\r
outfile\
""",
                        },
                        tool_call_id='ci_68cc84a053e4819d8645e13ae9f7612a07865e46140d854a',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={
                            'status': 'completed',
                            'outputs': [
                                {
                                    'type': 'image',
                                    'url': IsStr(),
                                },
                                {'logs': "'/mnt/data/y_equals_x_squared.png'", 'type': 'logs'},
                            ],
                        },
                        tool_call_id='ci_68cc84a053e4819d8645e13ae9f7612a07865e46140d854a',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    TextPart(
                        content=IsStr(),
                        id='msg_68cc84a90fec819d8c303b35ee0d948c07865e46140d854a',
                    ),
                ],
                usage=RequestUsage(input_tokens=2966, output_tokens=676, details={'reasoning_tokens': 448}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_details={'finish_reason': 'completed'},
                provider_response_id='resp_68cc84978d4c819dad1933fa64334bcb07865e46140d854a',
                finish_reason='stop',
            ),
        ]
    )
