from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TypedDict

from langgraph.graph import END, StateGraph

from llm import ModelLoader
from memory import build_memory
from prompts import REASONING_PROMPT, RESPONSE_FORMAT_PROMPT, SYSTEM_PROMPT
from query_router import QueryRouter
from retriever import UrbanMobilityRetriever
from tools import demand_analysis, pattern_detection, trip_statistics

LOGGER = logging.getLogger(__name__)


class MobilityState(TypedDict):
    question: str
    intent: str
    context: str
    tool_output: str
    answer: str


@dataclass
class UrbanMobilityRAG:
    retriever: UrbanMobilityRetriever = field(default_factory=UrbanMobilityRetriever)
    router: QueryRouter = field(default_factory=QueryRouter)

    def __post_init__(self) -> None:
        self.llm = ModelLoader().load()
        self.memory = build_memory()
        self.graph = self._build_graph()

    def answer(self, question: str) -> str:
        result = self.graph.invoke({"question": question, "intent": "", "context": "", "tool_output": "", "answer": ""})
        self.memory.save_context({"question": question}, {"answer": result["answer"]})
        return result["answer"]

    def _build_graph(self):
        graph = StateGraph(MobilityState)
        graph.add_node("intent_detection", self._intent_detection)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("tools", self._run_tools)
        graph.add_node("generate", self._generate)
        graph.set_entry_point("intent_detection")
        graph.add_edge("intent_detection", "retrieve")
        graph.add_edge("retrieve", "tools")
        graph.add_edge("tools", "generate")
        graph.add_edge("generate", END)
        return graph.compile()

    def _intent_detection(self, state: MobilityState) -> MobilityState:
        state["intent"] = self.router.detect(state["question"])
        return state

    def _retrieve(self, state: MobilityState) -> MobilityState:
        state["context"] = self.retriever.context(state["question"])
        return state

    def _run_tools(self, state: MobilityState) -> MobilityState:
        intent = state["intent"]
        if intent == "demand_analysis":
            state["tool_output"] = demand_analysis.invoke({"group_by": "pickup_hour"})
        elif intent == "pattern_analysis":
            state["tool_output"] = pattern_detection.invoke({"metric": "trip_distance"})
        elif intent == "trip_distance":
            state["tool_output"] = trip_statistics.invoke({})
        else:
            state["tool_output"] = ""
        return state

    def _generate(self, state: MobilityState) -> MobilityState:
        history = self.memory.load_memory_variables({}).get("history", "")
        prompt = REASONING_PROMPT.format(
            system_prompt=SYSTEM_PROMPT,
            history=history,
            context=state["context"],
            tool_output=state["tool_output"],
            question=state["question"],
        )
        prompt = f"{prompt}\n\n{RESPONSE_FORMAT_PROMPT}"
        response = self.llm.invoke(prompt)
        state["answer"] = str(response).strip()
        return state
