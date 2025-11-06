#!/usr/bin/env python3
"""Example demonstrating Anthropic prompt caching.

This example shows how to use CachePoint to reduce costs by caching:
- Long system prompts
- Large context (like documentation)
- Tool definitions

Run with: uv run -m pydantic_ai_examples.anthropic_prompt_caching
"""

from pydantic_ai import Agent, CachePoint

# Sample long context to demonstrate caching
# Need at least 1024 tokens - repeating 10x to be safe
LONG_CONTEXT = (
    """
# Product Documentation

## Overview
Our API provides comprehensive data access with the following features:

### Authentication
All requests require a Bearer token in the Authorization header.
Rate limits: 1000 requests/hour for standard tier.

### Endpoints

#### GET /api/users
Returns a list of users with pagination support.
Parameters:
- page: Page number (default: 1)
- limit: Items per page (default: 20, max: 100)
- filter: Optional filter expression

#### GET /api/products
Returns product catalog with detailed specifications.
Parameters:
- category: Filter by category
- in_stock: Boolean, filter available items
- sort: Sort order (price_asc, price_desc, name)

#### POST /api/orders
Create a new order. Requires authentication.
Request body:
- user_id: Integer, required
- items: Array of {product_id, quantity}
- shipping_address: Object with address details

#### Error Handling
Standard HTTP status codes are used:
- 200: Success
- 400: Bad request
- 401: Unauthorized
- 404: Not found
- 500: Server error

## Best Practices
1. Always handle rate limiting with exponential backoff
2. Cache responses where appropriate
3. Use pagination for large datasets
4. Validate input before submission
5. Monitor API usage through dashboard

## Code Examples
See detailed examples in our GitHub repository.
"""
    * 10
)  # Repeat 10x to ensure we exceed Anthropic's minimum cache size (1024 tokens)


async def main() -> None:
    """Demonstrate prompt caching with Anthropic."""
    print('=== Anthropic Prompt Caching Demo ===\n')

    agent = Agent(
        'anthropic:claude-sonnet-4-5',
        system_prompt='You are a helpful API documentation assistant.',
    )

    # First request with cache point - this will write to cache
    print('First request (will cache context)...')
    result1 = await agent.run(
        [
            LONG_CONTEXT,
            CachePoint(),  # Everything before this will be cached
            'What authentication method does the API use?',
        ]
    )

    print(f'Response: {result1.output}\n')
    usage1 = result1.usage()
    print(f'Usage: {usage1}')
    if usage1.cache_write_tokens:
        print(
            f'  Cache write tokens: {usage1.cache_write_tokens} (tokens written to cache)'
        )
    print()

    # Second request with same cached context - should use cache
    print('Second request (should read from cache)...')
    result2 = await agent.run(
        [
            LONG_CONTEXT,
            CachePoint(),  # Same content, should hit cache
            'What are the available API endpoints?',
        ]
    )

    print(f'Response: {result2.output}\n')
    usage2 = result2.usage()
    print(f'Usage: {usage2}')
    if usage2.cache_read_tokens:
        print(
            f'  Cache read tokens: {usage2.cache_read_tokens} (tokens read from cache)'
        )
        print(
            f'  Cache savings: ~{usage2.cache_read_tokens * 0.9:.0f} token-equivalents (90% discount)'
        )
    print()

    # Third request with different question, same cache
    print('Third request (should also read from cache)...')
    result3 = await agent.run(
        [
            LONG_CONTEXT,
            CachePoint(),
            'How should I handle rate limiting?',
        ]
    )

    print(f'Response: {result3.output}\n')
    usage3 = result3.usage()
    print(f'Usage: {usage3}')
    if usage3.cache_read_tokens:
        print(f'  Cache read tokens: {usage3.cache_read_tokens}')
    print()

    print('=== Summary ===')
    total_usage = usage1 + usage2 + usage3
    print(f'Total input tokens: {total_usage.input_tokens}')
    print(f'Total cache write: {total_usage.cache_write_tokens}')
    print(f'Total cache read: {total_usage.cache_read_tokens}')
    if total_usage.cache_read_tokens:
        savings = total_usage.cache_read_tokens * 0.9
        print(f'Estimated savings: ~{savings:.0f} token-equivalents')


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
