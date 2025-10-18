# Built-in Tools

Built-in tools are native tools provided by LLM providers that can be used to enhance your agent's capabilities. Unlike [common tools](common-tools.md), which are custom implementations that Pydantic AI executes, built-in tools are executed directly by the model provider.

## Overview

Pydantic AI supports the following built-in tools:

- **[`WebSearchTool`][pydantic_ai.builtin_tools.WebSearchTool]**: Allows agents to search the web
- **[`CodeExecutionTool`][pydantic_ai.builtin_tools.CodeExecutionTool]**: Enables agents to execute code in a secure environment
- **[`ImageGenerationTool`][pydantic_ai.builtin_tools.ImageGenerationTool]**: Enables agents to generate images
- **[`UrlContextTool`][pydantic_ai.builtin_tools.UrlContextTool]**: Enables agents to pull URL contents into their context
- **[`MemoryTool`][pydantic_ai.builtin_tools.MemoryTool]**: Enables agents to use memory

These tools are passed to the agent via the `builtin_tools` parameter and are executed by the model provider's infrastructure.

!!! warning "Provider Support"
    Not all model providers support built-in tools. If you use a built-in tool with an unsupported provider, Pydantic AI will raise a [`UserError`][pydantic_ai.exceptions.UserError] when you try to run the agent.

    If a provider supports a built-in tool that is not currently supported by Pydantic AI, please file an issue.

## Web Search Tool

The [`WebSearchTool`][pydantic_ai.builtin_tools.WebSearchTool] allows your agent to search the web,
making it ideal for queries that require up-to-date data.

### Provider Support

| Provider | Supported | Notes |
|----------|-----------|-------|
| OpenAI Responses | ✅ | Full feature support. To include search results on the [`BuiltinToolReturnPart`][pydantic_ai.messages.BuiltinToolReturnPart] that's available via [`ModelResponse.builtin_tool_calls`][pydantic_ai.messages.ModelResponse.builtin_tool_calls], enable the [`OpenAIResponsesModelSettings.openai_include_web_search_sources`][pydantic_ai.models.openai.OpenAIResponsesModelSettings.openai_include_web_search_sources] [model setting](agents.md#model-run-settings). |
| Anthropic | ✅ | Full feature support |
| Google | ✅ | No parameter support. No [`BuiltinToolCallPart`][pydantic_ai.messages.BuiltinToolCallPart] or [`BuiltinToolReturnPart`][pydantic_ai.messages.BuiltinToolReturnPart] is generated when streaming. Using built-in tools and user tools (including [output tools](output.md#tool-output)) at the same time is not supported; to use structured output, use [`PromptedOutput`](output.md#prompted-output) instead. |
| Groq | ✅ | Limited parameter support. To use web search capabilities with Groq, you need to use the [compound models](https://console.groq.com/docs/compound). |
| OpenAI Chat Completions | ❌ | Not supported |
| Bedrock | ❌ | Not supported |
| Mistral | ❌ | Not supported |
| Cohere | ❌ | Not supported |
| HuggingFace | ❌ | Not supported |

### Usage

```py {title="web_search_anthropic.py"}
from pydantic_ai import Agent, WebSearchTool

agent = Agent('anthropic:claude-sonnet-4-0', builtin_tools=[WebSearchTool()])

result = agent.run_sync('Give me a sentence with the biggest news in AI this week.')
print(result.output)
#> Scientists have developed a universal AI detector that can identify deepfake videos.
```

_(This example is complete, it can be run "as is")_

With OpenAI, you must use their responses API to access the web search tool.

```py {title="web_search_openai.py"}
from pydantic_ai import Agent, WebSearchTool

agent = Agent('openai-responses:gpt-4.1', builtin_tools=[WebSearchTool()])

result = agent.run_sync('Give me a sentence with the biggest news in AI this week.')
print(result.output)
#> Scientists have developed a universal AI detector that can identify deepfake videos.
```

_(This example is complete, it can be run "as is")_

### Configuration Options

The `WebSearchTool` supports several configuration parameters:

```py {title="web_search_configured.py"}
from pydantic_ai import Agent, WebSearchTool, WebSearchUserLocation

agent = Agent(
    'anthropic:claude-sonnet-4-0',
    builtin_tools=[
        WebSearchTool(
            search_context_size='high',
            user_location=WebSearchUserLocation(
                city='San Francisco',
                country='US',
                region='CA',
                timezone='America/Los_Angeles',
            ),
            blocked_domains=['example.com', 'spam-site.net'],
            allowed_domains=None,  # Cannot use both blocked_domains and allowed_domains with Anthropic
            max_uses=5,  # Anthropic only: limit tool usage
        )
    ],
)

result = agent.run_sync('Use the web to get the current time.')
print(result.output)
#> In San Francisco, it's 8:21:41 pm PDT on Wednesday, August 6, 2025.
```

_(This example is complete, it can be run "as is")_

#### Provider Support

| Parameter | OpenAI | Anthropic | Groq |
|-----------|--------|-----------|------|
| `search_context_size` | ✅ | ❌ | ❌ |
| `user_location` | ✅ | ✅ | ❌ |
| `blocked_domains` | ❌ | ✅ | ✅ |
| `allowed_domains` | ❌ | ✅ | ✅ |
| `max_uses` | ❌ | ✅ | ❌ |

!!! note "Anthropic Domain Filtering"
    With Anthropic, you can only use either `blocked_domains` or `allowed_domains`, not both.

## Code Execution Tool

The [`CodeExecutionTool`][pydantic_ai.builtin_tools.CodeExecutionTool] enables your agent to execute code
in a secure environment, making it perfect for computational tasks, data analysis, and mathematical operations.

### Provider Support

| Provider | Supported | Notes |
|----------|-----------|-------|
| OpenAI | ✅ | To include code execution output on the [`BuiltinToolReturnPart`][pydantic_ai.messages.BuiltinToolReturnPart] that's available via [`ModelResponse.builtin_tool_calls`][pydantic_ai.messages.ModelResponse.builtin_tool_calls], enable the [`OpenAIResponsesModelSettings.openai_include_code_execution_outputs`][pydantic_ai.models.openai.OpenAIResponsesModelSettings.openai_include_code_execution_outputs] [model setting](agents.md#model-run-settings). If the code execution generated images, like charts, they will be available on [`ModelResponse.images`][pydantic_ai.messages.ModelResponse.images] as [`BinaryImage`][pydantic_ai.messages.BinaryImage] objects. The generated image can also be used as [image output](output.md#image-output) for the agent run. |
| Google | ✅ | Using built-in tools and user tools (including [output tools](output.md#tool-output)) at the same time is not supported; to use structured output, use [`PromptedOutput`](output.md#prompted-output) instead. |
| Anthropic | ✅ | |
| Groq | ❌ | |
| Bedrock | ❌ | |
| Mistral | ❌ | |
| Cohere | ❌ | |
| HuggingFace | ❌ | |

### Usage

```py {title="code_execution_basic.py"}
from pydantic_ai import Agent, CodeExecutionTool

agent = Agent('anthropic:claude-sonnet-4-0', builtin_tools=[CodeExecutionTool()])

result = agent.run_sync('Calculate the factorial of 15.')
print(result.output)
#> The factorial of 15 is **1,307,674,368,000**.
print(result.response.builtin_tool_calls)
"""
[
    (
        BuiltinToolCallPart(
            tool_name='code_execution',
            args={
                'code': 'import math\n\n# Calculate factorial of 15\nresult = math.factorial(15)\nprint(f"15! = {result}")\n\n# Let\'s also show it in a more readable format with commas\nprint(f"15! = {result:,}")'
            },
            tool_call_id='srvtoolu_017qRH1J3XrhnpjP2XtzPCmJ',
            provider_name='anthropic',
        ),
        BuiltinToolReturnPart(
            tool_name='code_execution',
            content={
                'content': [],
                'return_code': 0,
                'stderr': '',
                'stdout': '15! = 1307674368000\n15! = 1,307,674,368,000',
                'type': 'code_execution_result',
            },
            tool_call_id='srvtoolu_017qRH1J3XrhnpjP2XtzPCmJ',
            timestamp=datetime.datetime(...),
            provider_name='anthropic',
        ),
    )
]
"""
```

_(This example is complete, it can be run "as is")_

In addition to text output, code execution with OpenAI can generate images as part of their response. Accessing this image via [`ModelResponse.images`][pydantic_ai.messages.ModelResponse.images] or [image output](output.md#image-output) requires the [`OpenAIResponsesModelSettings.openai_include_code_execution_outputs`][pydantic_ai.models.openai.OpenAIResponsesModelSettings.openai_include_code_execution_outputs] [model setting](agents.md#model-run-settings) to be enabled.

```py {title="code_execution_openai.py"}
from pydantic_ai import Agent, BinaryImage, CodeExecutionTool
from pydantic_ai.models.openai import OpenAIResponsesModelSettings

agent = Agent(
    'openai-responses:gpt-5',
    builtin_tools=[CodeExecutionTool()],
    output_type=BinaryImage,
    model_settings=OpenAIResponsesModelSettings(openai_include_code_execution_outputs=True),
)

result = agent.run_sync('Generate a chart of y=x^2 for x=-5 to 5.')
assert isinstance(result.output, BinaryImage)
```

_(This example is complete, it can be run "as is")_

## Image Generation Tool

The [`ImageGenerationTool`][pydantic_ai.builtin_tools.ImageGenerationTool] enables your agent to generate images.

### Provider Support

| Provider | Supported | Notes |
|----------|-----------|-------|
| OpenAI Responses | ✅ | Full feature support. Only supported by models newer than `gpt-4o`. Metadata about the generated image, like the [`revised_prompt`](https://platform.openai.com/docs/guides/tools-image-generation#revised-prompt) sent to the underlying image model, is available on the [`BuiltinToolReturnPart`][pydantic_ai.messages.BuiltinToolReturnPart] that's available via [`ModelResponse.builtin_tool_calls`][pydantic_ai.messages.ModelResponse.builtin_tool_calls]. |
| Google | ✅ | No parameter support. Only supported by [image generation models](https://ai.google.dev/gemini-api/docs/image-generation) like `gemini-2.5-flash-image`. These models do not support [structured output](output.md) or [function tools](tools.md). These models will always generate images, even if this built-in tool is not explicitly specified. |
| Anthropic | ❌ | |
| Groq | ❌ | |
| Bedrock | ❌ | |
| Mistral | ❌ | |
| Cohere | ❌ | |
| HuggingFace | ❌ | |

### Usage

Generated images are available on [`ModelResponse.images`][pydantic_ai.messages.ModelResponse.images] as [`BinaryImage`][pydantic_ai.messages.BinaryImage] objects:

```py {title="image_generation_openai.py"}
from pydantic_ai import Agent, BinaryImage, ImageGenerationTool

agent = Agent('openai-responses:gpt-5', builtin_tools=[ImageGenerationTool()])

result = agent.run_sync('Tell me a two-sentence story about an axolotl with an illustration.')
print(result.output)
"""
Once upon a time, in a hidden underwater cave, lived a curious axolotl named Pip who loved to explore. One day, while venturing further than usual, Pip discovered a shimmering, ancient coin that granted wishes!
"""

assert isinstance(result.response.images[0], BinaryImage)
```

_(This example is complete, it can be run "as is")_

Image generation with Google [image generation models](https://ai.google.dev/gemini-api/docs/image-generation) does not require the `ImageGenerationTool` built-in tool to be explicitly specified:

```py {title="image_generation_google.py"}
from pydantic_ai import Agent, BinaryImage

agent = Agent('google-gla:gemini-2.5-flash-image')

result = agent.run_sync('Tell me a two-sentence story about an axolotl with an illustration.')
print(result.output)
"""
Once upon a time, in a hidden underwater cave, lived a curious axolotl named Pip who loved to explore. One day, while venturing further than usual, Pip discovered a shimmering, ancient coin that granted wishes!
"""

assert isinstance(result.response.images[0], BinaryImage)
```

_(This example is complete, it can be run "as is")_

The `ImageGenerationTool` can be used together with `output_type=BinaryImage` to get [image output](output.md#image-output). If the `ImageGenerationTool` built-in tool is not explicitly specified, it will be enabled automatically:

```py {title="image_generation_output.py"}
from pydantic_ai import Agent, BinaryImage

agent = Agent('openai-responses:gpt-5', output_type=BinaryImage)

result = agent.run_sync('Generate an image of an axolotl.')
assert isinstance(result.output, BinaryImage)
```

_(This example is complete, it can be run "as is")_

### Configuration Options

The `ImageGenerationTool` supports several configuration parameters:

```py {title="image_generation_configured.py"}
from pydantic_ai import Agent, BinaryImage, ImageGenerationTool

agent = Agent(
    'openai-responses:gpt-5',
    builtin_tools=[
        ImageGenerationTool(
            background='transparent',
            input_fidelity='high',
            moderation='low',
            output_compression=100,
            output_format='png',
            partial_images=3,
            quality='high',
            size='1024x1024',
        )
    ],
    output_type=BinaryImage,
)

result = agent.run_sync('Generate an image of an axolotl.')
assert isinstance(result.output, BinaryImage)
```

_(This example is complete, it can be run "as is")_

For more details, check the [API documentation][pydantic_ai.builtin_tools.ImageGenerationTool].

#### Provider Support

| Parameter | OpenAI | Google |
|-----------|--------|--------|
| `background` | ✅ | ❌ |
| `input_fidelity` | ✅ | ❌ |
| `moderation` | ✅ | ❌ |
| `output_compression` | ✅ | ❌ |
| `output_format` | ✅ | ❌ |
| `partial_images` | ✅ | ❌ |
| `quality` | ✅ | ❌ |
| `size` | ✅ | ❌ |

## URL Context Tool

The [`UrlContextTool`][pydantic_ai.builtin_tools.UrlContextTool] enables your agent to pull URL contents into its context,
allowing it to pull up-to-date information from the web.

### Provider Support

| Provider | Supported | Notes |
|----------|-----------|-------|
| Google | ✅ | No [`BuiltinToolCallPart`][pydantic_ai.messages.BuiltinToolCallPart] or [`BuiltinToolReturnPart`][pydantic_ai.messages.BuiltinToolReturnPart] is currently generated; please submit an issue if you need this. Using built-in tools and user tools (including [output tools](output.md#tool-output)) at the same time is not supported; to use structured output, use [`PromptedOutput`](output.md#prompted-output) instead. |
| OpenAI | ❌ | |
| Anthropic | ❌ | |
| Groq | ❌ | |
| Bedrock | ❌ | |
| Mistral | ❌ | |
| Cohere | ❌ | |
| HuggingFace | ❌ | |

### Usage

```py {title="url_context_basic.py"}
from pydantic_ai import Agent, UrlContextTool

agent = Agent('google-gla:gemini-2.5-flash', builtin_tools=[UrlContextTool()])

result = agent.run_sync('What is this? https://ai.pydantic.dev')
print(result.output)
#> A Python agent framework for building Generative AI applications.
```

_(This example is complete, it can be run "as is")_

## Memory Tool

The [`MemoryTool`][pydantic_ai.builtin_tools.MemoryTool] enables your agent to use memory.

### Provider Support

| Provider | Supported | Notes |
|----------|-----------|-------|
| Anthropic | ✅ | Requires a tool named `memory` to be defined that implements [specific sub-commands](https://docs.claude.com/en/docs/agents-and-tools/tool-use/memory-tool#tool-commands). You can use a subclass of [`anthropic.lib.tools.BetaAbstractMemoryTool`](https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/lib/tools/_beta_builtin_memory_tool.py) as documented below. |
| Google | ❌ | |
| OpenAI | ❌ | |
| Groq | ❌ | |
| Bedrock | ❌ | |
| Mistral | ❌ | |
| Cohere | ❌ | |
| HuggingFace | ❌ | |

### Usage

The Anthropic SDK provides an abstract [`BetaAbstractMemoryTool`](https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/lib/tools/_beta_builtin_memory_tool.py) class that you can subclass to create your own memory storage solution (e.g., database, cloud storage, encrypted files, etc.). Their [`LocalFilesystemMemoryTool`](https://github.com/anthropics/anthropic-sdk-python/blob/main/examples/memory/basic.py) example can serve as a starting point.

The following example uses a subclass that hard-codes a specific memory. The bits specific to Pydantic AI are the `MemoryTool` built-in tool and the `memory` tool definition that forwards commands to the `call` method of the `BetaAbstractMemoryTool` subclass.

```py {title="anthropic_memory.py"}
from typing import Any

from anthropic.lib.tools import BetaAbstractMemoryTool
from anthropic.types.beta import (
    BetaMemoryTool20250818CreateCommand,
    BetaMemoryTool20250818DeleteCommand,
    BetaMemoryTool20250818InsertCommand,
    BetaMemoryTool20250818RenameCommand,
    BetaMemoryTool20250818StrReplaceCommand,
    BetaMemoryTool20250818ViewCommand,
)

from pydantic_ai import Agent, MemoryTool


class FakeMemoryTool(BetaAbstractMemoryTool):
    def view(self, command: BetaMemoryTool20250818ViewCommand) -> str:
        return 'The user lives in Mexico City.'

    def create(self, command: BetaMemoryTool20250818CreateCommand) -> str:
        return f'File created successfully at {command.path}'

    def str_replace(self, command: BetaMemoryTool20250818StrReplaceCommand) -> str:
        return f'File {command.path} has been edited'

    def insert(self, command: BetaMemoryTool20250818InsertCommand) -> str:
        return f'Text inserted at line {command.insert_line} in {command.path}'

    def delete(self, command: BetaMemoryTool20250818DeleteCommand) -> str:
        return f'File deleted: {command.path}'

    def rename(self, command: BetaMemoryTool20250818RenameCommand) -> str:
        return f'Renamed {command.old_path} to {command.new_path}'

    def clear_all_memory(self) -> str:
        return 'All memory cleared'

fake_memory = FakeMemoryTool()

agent = Agent('anthropic:claude-sonnet-4-5', builtin_tools=[MemoryTool()])


@agent.tool_plain
def memory(**command: Any) -> Any:
    return fake_memory.call(command)


result = agent.run_sync('Remember that I live in Mexico City')
print(result.output)
"""
Got it! I've recorded that you live in Mexico City. I'll remember this for future reference.
"""

result = agent.run_sync('Where do I live?')
print(result.output)
#> You live in Mexico City.
```

_(This example is complete, it can be run "as is")_

## API Reference

For complete API documentation, see the [API Reference](api/builtin_tools.md).
