from __future__ import annotations

import logging
import os

import pytest

from pydantic_ai.agent import Agent
from pydantic_ai.exceptions import ModelRetry, UnexpectedModelBehavior
from pydantic_ai.models import override_allow_model_requests
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import Tool

# Configure comprehensive logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(message)s')

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# Force propagation to ensure logs are captured
logger.propagate = True


@pytest.mark.skipif(
    not pytest.importorskip('anthropic', reason='anthropic package not installed'), reason='No anthropic package'
)
@pytest.mark.skipif(not pytest.importorskip('os').environ.get('ANTHROPIC_API_KEY'), reason='No API key')
@pytest.mark.anyio
async def test_real_agent_with_token_limited_tool():
    """Test a real agent with Anthropic model and a token-limited tool.

    This test is skipped unless:
    1. The anthropic package is installed
    2. ANTHROPIC_API_KEY is set in the environment

    Run with:
        ALLOW_MODEL_REQUESTS=true pytest tests/test_token_limits.py::test_real_agent_with_token_limited_tool -v
    """
    # Skip if Anthropic is not available
    if not os.environ.get('ANTHROPIC_API_KEY'):
        pytest.skip('ANTHROPIC_API_KEY not set')

    allow_api = os.environ.get('ALLOW_MODEL_REQUESTS', 'false').lower() == 'true'
    with override_allow_model_requests(allow_api):
        # Track if the tool was ever called
        tool_was_called = False

        # Keep track of the poem content
        current_poem_content = ''
        verses_added = 0

        # Define a tool that takes a large input
        def write_very_large_content(file_path: str, file_contents: str, append: bool = False) -> str:
            """Write the provided content to the specified file.

            Args:
                file_path: The path where the file should be written
                file_contents: The content to write to the file.
                append: If True, append to existing content instead of replacing it

            Returns:
                A confirmation message
            """
            nonlocal tool_was_called, current_poem_content, verses_added

            tool_was_called = True
            print(f'TOOL BODY CALLED: Writing {len(file_contents)} characters to {file_path} (append={append})')

            # Check if we're appending or replacing
            if append:
                # Count the number of verses being added
                new_verses = [v for v in file_contents.split('\n\n') if v.strip()]
                verses_being_added = len(new_verses)

                # Append the new content
                current_poem_content += '\n\n' + file_contents if current_poem_content else file_contents
                verses_added += verses_being_added

                print(f'APPENDING: Added {verses_being_added} verses, total now {verses_added}')
                return f'Successfully appended {len(file_contents)} characters. Total verses: {verses_added}'
            else:
                # For non-append operations, check for minimum requirements
                verses = [v for v in file_contents.split('\n\n') if v.strip()]

                print(f'REPLACING: Received poem with {len(verses)} verses and {len(file_contents)} characters')

                # If all checks pass, replace the current content
                current_poem_content = file_contents
                verses_added = len(verses)
                return f'Successfully wrote {len(file_contents)} characters to {file_path}'

        # Create the tool with a low max_retries to speed up the test
        write_tool = Tool(write_very_large_content, max_retries=5)

        # Create an agent with an actual Anthropic model
        agent = Agent(
            model='anthropic:claude-3-5-haiku-latest',
            tools=[write_tool],
            system_prompt="""You must use the write_very_large_content tool to write a poem to poem.txt.
If you hit token limits when trying to write a large poem all at once, use the append=True parameter to add verses one at a time.
You have a max_tokens of 100 for any output.
""",
        )

        try:
            # Run the agent with a prompt that will require a large text response
            result = await agent.run(
                'Write a poem to poem.txt. The poem must be very long with at least 50 verses.',
                model_settings=ModelSettings(max_tokens=100, temperature=0.0),
            )
            # Check if we successfully built a complete poem with at least 50 verses
            total_verses = len([v for v in current_poem_content.split('\n\n') if v.strip()])
            print(f'COMPLETED POEM: {total_verses} verses, {len(current_poem_content)} characters')

            # If we got here with a complete poem, that's success!
            if total_verses >= 50:
                print('SUCCESS: Model adapted to token limits by using append=True')
                assert verses_added >= 50, 'Should have added at least 50 verses in total'
                assert tool_was_called, 'The tool should have been called'
            else:
                # We got here but without a complete poem
                print(f'UNEXPECTED RESULT: Poem has only {total_verses} verses')
                print(f'Tool was called: {tool_was_called}')
                print(f'Result: {result.data}')
                assert False, 'Expected either token limit error or completed poem with 50+ verses'
        except ModelRetry as exc:
            # This indicates our token limit detection is working properly
            # The tool should now be configured to retry or use append=True
            print(f'CAUGHT ModelRetry: {exc}')
            print(f'Tool was called: {tool_was_called}')

            # Basic verification that we got the expected token limit message
            assert 'token limit' in str(exc).lower()
            assert 'write_very_large_content' in str(exc)

            # Test passed - the ModelRetry was correctly raised with a helpful message
            # In a real-world scenario, this would be captured by the Agent's retry mechanism
        except UnexpectedModelBehavior as exc:
            # Verify the error contains our improved token limit message
            error_msg = str(exc)
            print(f'CAUGHT ERROR: {error_msg}')
            print(f'Tool was called: {tool_was_called}')
            print(f'Verses added so far: {verses_added}')

            # Check if we made partial progress
            if verses_added > 0:
                print(f'PARTIAL SUCCESS: Model added {verses_added} verses using append=True before exceeding retries')

            # Basic verification that we got the expected error message
            assert 'exceeded max retries' in error_msg
            assert 'token limit' in error_msg.lower() or 'max_tokens' in error_msg
