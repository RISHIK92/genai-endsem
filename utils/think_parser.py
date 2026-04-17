"""
Utility to separate <think>...</think> blocks from LLM responses.

The Groq/Llama model sometimes emits chain-of-thought reasoning inside
<think> tags. This module strips those tags from the visible response
and returns the thinking content separately so it can be shown in a
collapsible UI element without polluting the main output or being
forwarded to downstream models.
"""
import re


def separate_thinking(text: str) -> tuple[str, str]:
    """Separate <think>...</think> blocks from the main response.

    Args:
        text: Raw LLM response that may contain <think>...</think> blocks.

    Returns:
        A tuple of (clean_response, thinking_content).
        - clean_response: The response with all <think> blocks removed.
        - thinking_content: The concatenated contents of all <think> blocks
          (empty string if none found).
    """
    if not text:
        return "", ""

    # Capture all <think>...</think> blocks (supports multiline, lazy match)
    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    
    thinking_parts = think_pattern.findall(text)
    thinking_content = "\n\n".join(part.strip() for part in thinking_parts if part.strip())

    # Remove all <think>...</think> blocks from the response
    clean_response = think_pattern.sub("", text).strip()

    # Clean up any leftover double newlines from removal
    clean_response = re.sub(r"\n{3,}", "\n\n", clean_response)

    return clean_response, thinking_content
