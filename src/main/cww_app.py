#%%
from langgraph.graph import END, StateGraph

from typing_extensions import TypedDict
from typing import Annotated
from operator import add

import sys
sys.path.append('../')

from components.RAG.agents import *
from components.RAG.tools import *
from components.RAG.helper import *

#%%

def cww_app(question):
    # Create Graph State Object
    class GraphState(TypedDict):
        """
        Represents the state of our graph.

        Attributes:
            question: question
            generation: LLM generation
            documents: list of documents
        """
        question: str
        generation: str
        documents: Annotated[list[str], add]
        sources: Annotated[list[str], add]
        next_step: str

    # Define the nodes
    workflow = StateGraph(GraphState)
    workflow.add_node("vstore_agent", vstore_selection)
    workflow.add_node("evaluator_agent", evaluator)
    workflow.add_node("cs_evaluator", evaluator)
    workflow.add_node("assistant_agent", router_assistant)
    workflow.add_node("cs_api", api_agent)
    workflow.add_node("tavily_search", web_search)
    workflow.add_node("generate", generate_answer)

    # Defines the edges
    workflow.set_conditional_entry_point(
        entry_agent,
        {"vector store": "vstore_agent", "routing assistant": "assistant_agent"},
    )
    workflow.add_edge("vstore_agent", "evaluator_agent")
    # While at the evaluator agent, generate an answer or go to the router assistant for more information
    workflow.add_conditional_edges("evaluator_agent", vs_evaluation, {'assistant_agent':'assistant_agent',
                                                                   'generate':'generate'}) # do not need nodes for router agents, only the routing functions
    workflow.add_conditional_edges("assistant_agent",choose_secondary_source)
    workflow.add_edge("cs_api","cs_evaluator")
    workflow.add_conditional_edges("cs_evaluator", api_evaluation)

    workflow.add_edge("tavily_search", "generate")

    workflow.add_edge("generate", END)

    # Compile
    graph = workflow.compile()

    response = graph.invoke({"question": question})
    return response

#%%
if __name__ == "__main__":
    question = "How much do graduates of Rice University earn 6 years after graduation?"

    print(cww_app(question))
