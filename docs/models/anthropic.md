# Anthropic

## Install

To use `AnthropicModel` models, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `anthropic` optional group:

```bash
pip/uv-add "pydantic-ai-slim[anthropic]"
```

## Configuration

To use [Anthropic](https://anthropic.com) through their API, go to [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys) to generate an API key.

`AnthropicModelName` contains a list of available Anthropic models.

## Environment variable

Once you have the API key, you can set it as an environment variable:

```bash
export ANTHROPIC_API_KEY='your-api-key'
```

You can then use `AnthropicModel` by name:

```python
from pydantic_ai import Agent

agent = Agent('anthropic:claude-sonnet-4-5')
...
```

Or initialise the model directly with just the model name:

```python
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel

model = AnthropicModel('claude-sonnet-4-5')
agent = Agent(model)
...
```

## `provider` argument

You can provide a custom `Provider` via the `provider` argument:

```python
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

model = AnthropicModel(
    'claude-sonnet-4-5', provider=AnthropicProvider(api_key='your-api-key')
)
agent = Agent(model)
...
```

## Custom HTTP Client

You can customize the `AnthropicProvider` with a custom `httpx.AsyncClient`:

```python
from httpx import AsyncClient

from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

custom_http_client = AsyncClient(timeout=30)
model = AnthropicModel(
    'claude-sonnet-4-5',
    provider=AnthropicProvider(api_key='your-api-key', http_client=custom_http_client),
)
agent = Agent(model)
...
```

## Prompt Caching

Anthropic supports [prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) to reduce costs by caching parts of your prompts. Pydantic AI provides three ways to use prompt caching:

### 1. Cache User Messages with `CachePoint`

Insert a [`CachePoint`][pydantic_ai.messages.CachePoint] marker in your user messages to cache everything before it:

```python {test="skip"}
from pydantic_ai import Agent, CachePoint

agent = Agent('anthropic:claude-sonnet-4-5')

async def main():
    # Everything before CachePoint will be cached
    result = await agent.run([
        'Long context that should be cached...',
        CachePoint(),
        'Your question here'
    ])
    print(result.output)
```

### 2. Cache System Instructions

Use `anthropic_cache_instructions=True` to cache your system prompt:

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModelSettings

agent = Agent(
    'anthropic:claude-sonnet-4-5',
    system_prompt='Long detailed instructions...',
    model_settings=AnthropicModelSettings(
        anthropic_cache_instructions=True
    ),
)

async def main():
    result = await agent.run('Your question')
    print(result.output)
```

### 3. Cache Tool Definitions

Use `anthropic_cache_tool_definitions=True` to cache your tool definitions:

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModelSettings

agent = Agent(
    'anthropic:claude-sonnet-4-5',
    model_settings=AnthropicModelSettings(
        anthropic_cache_tool_definitions=True
    ),
)

@agent.tool
def my_tool() -> str:
    """Tool definition will be cached."""
    return 'result'

async def main():
    result = await agent.run('Use the tool')
    print(result.output)
```

### Combining Cache Strategies

You can combine all three caching strategies for maximum savings:

```python {test="skip"}
from pydantic_ai import Agent, CachePoint, RunContext
from pydantic_ai.models.anthropic import AnthropicModelSettings

agent = Agent(
    'anthropic:claude-sonnet-4-5',
    system_prompt='Detailed instructions...',
    model_settings=AnthropicModelSettings(
        anthropic_cache_instructions=True,
        anthropic_cache_tool_definitions=True,
    ),
)

@agent.tool
def search_docs(ctx: RunContext, query: str) -> str:
    """Search documentation."""
    return f'Results for {query}'

async def main():
    # First call - writes to cache
    result1 = await agent.run([
        'Long context from documentation...',
        CachePoint(),
        'First question'
    ])

    # Subsequent calls - read from cache (90% cost reduction)
    result2 = await agent.run([
        'Long context from documentation...',  # Same content
        CachePoint(),
        'Second question'
    ])
    print(f'First: {result1.output}')
    print(f'Second: {result2.output}')
```

Access cache usage statistics via `result.usage()`:

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModelSettings

agent = Agent(
    'anthropic:claude-sonnet-4-5',
    system_prompt='Instructions...',
    model_settings=AnthropicModelSettings(
        anthropic_cache_instructions=True
    ),
)

async def main():
    result = await agent.run('Your question')
    usage = result.usage()
    print(f'Cache write tokens: {usage.cache_write_tokens}')
    print(f'Cache read tokens: {usage.cache_read_tokens}')
```
