from __future__ import annotations

from langchain.memory import ConversationBufferMemory


def build_memory() -> ConversationBufferMemory:
    return ConversationBufferMemory(
        memory_key="history",
        input_key="question",
        output_key="answer",
        return_messages=False,
    )
