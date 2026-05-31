"""TAMU OpenAI-gateway workaround.

TAMU's gateway at chat-api.tamu.ai returns SSE-streamed responses for any
structured-output request (tools=..., response_format=...) even when the
client does not set stream=True. The OpenAI Python client receives the raw
SSE text as a string, which crashes downstream parsers like Instructor with
``'str' object has no attribute 'choices'``.

This module exposes ``wrap_for_tamu(openai_client)`` returning a drop-in
client whose ``chat.completions.create`` forces ``stream=True`` whenever
structured output is requested, then accumulates the chunks into a single
ChatCompletion identical to a non-streaming response. Pass the wrapped
client to ``ragas.llms.llm_factory(..., client=wrapped_client)``.
"""

from __future__ import annotations

from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)


def _accumulate_stream(stream) -> ChatCompletion:
    """Collapse a Stream[ChatCompletionChunk] into a ChatCompletion."""
    chunks = list(stream)
    if not chunks:
        raise RuntimeError("TAMU stream returned no chunks")

    head = chunks[0]
    content_parts: list[str] = []
    # idx → {id, name, arguments}
    tool_call_acc: dict[int, dict] = {}
    finish_reason: str | None = None

    for chunk in chunks:
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        delta = choice.delta
        if delta and delta.content:
            content_parts.append(delta.content)
        if delta and getattr(delta, "tool_calls", None):
            for tc in delta.tool_calls:
                idx = tc.index if tc.index is not None else 0
                acc = tool_call_acc.setdefault(idx, {"id": None, "name": None, "arguments": ""})
                if tc.id:
                    acc["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        acc["name"] = tc.function.name
                    if tc.function.arguments:
                        acc["arguments"] += tc.function.arguments
        if choice.finish_reason:
            finish_reason = choice.finish_reason

    tool_calls: list[ChatCompletionMessageToolCall] | None = None
    if tool_call_acc:
        tool_calls = []
        for idx, acc in sorted(tool_call_acc.items()):
            tool_calls.append(
                ChatCompletionMessageToolCall(
                    id=acc["id"] or f"call_tamu_{idx}",
                    type="function",
                    function=Function(
                        name=acc["name"] or "",
                        arguments=acc["arguments"],
                    ),
                )
            )

    message = ChatCompletionMessage(
        role="assistant",
        content="".join(content_parts) if content_parts else None,
        tool_calls=tool_calls,  # type: ignore[arg-type]
    )

    choice = Choice(
        index=0,
        message=message,
        finish_reason=finish_reason or "stop",  # type: ignore[arg-type]
    )

    return ChatCompletion(
        id=head.id,
        object="chat.completion",
        created=head.created,
        model=head.model,
        choices=[choice],
        usage=None,
    )


def wrap_for_tamu(openai_client: OpenAI) -> OpenAI:
    """Patch an OpenAI client to defuse TAMU's SSE-on-everything bug.

    Returns the same client (mutated in place) so ``isinstance(client, OpenAI)``
    checks in downstream libraries (Instructor, ragas) still pass.
    """
    completions = openai_client.chat.completions
    original_create = completions.create

    def create_with_sse_workaround(**kwargs):
        # Ragas passes max_tokens=1024 explicitly, which truncates JSON
        # tool-call payloads on long syllabus content. Bump any value
        # below 4096 up to 4096 so structured outputs complete.
        mt = kwargs.get("max_tokens")
        if mt is None or mt < 4096:
            kwargs["max_tokens"] = 4096
        # If the caller explicitly asked for a stream, honor it untouched.
        if kwargs.get("stream"):
            return original_create(**kwargs)
        # TAMU returns SSE-as-string for every chat completion. Force
        # stream=True so the OpenAI client parses chunks, then reassemble.
        kwargs["stream"] = True
        stream = original_create(**kwargs)
        return _accumulate_stream(stream)

    completions.create = create_with_sse_workaround  # type: ignore[method-assign]
    return openai_client
