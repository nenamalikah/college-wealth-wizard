#%%
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from typing_extensions import TypedDict
from operator import add
from pydantic import BaseModel, Field
from typing import List, Dict, Annotated
from langchain_core.runnables import RunnableParallel

import sys
sys.path.append('../../')

from components.RAG.entry_router_agent import entry_router_agent
from components.RAG.vstore_router_agent import vstore_router_agent
from components.RAG.evaluator_agent import evaluator_agent
from components.RAG.routing_assistant_agent import routing_assistant_agent
from components.RAG.api_parser_agent import api_parser_agent, api_call_tool
#%%
# Import the Vector Stores

embeddings = HuggingFaceEmbeddings(model_name = 'BAAI/bge-large-en-v1.5')
ipeds_store = Chroma(embedding_function=embeddings,persist_directory='../../../data/vector_store/data_store',collection_name='IPEDS_Education_Information')
ipeds_retriever = ipeds_store.as_retriever()

bls_store = Chroma(embedding_function=embeddings,persist_directory='../../../data/vector_store/data_store',collection_name='BLS_Occupational_Information')
bls_retriever = bls_store.as_retriever(search_kwargs={"k": 5})

cip_soc_store = Chroma(embedding_function=embeddings,persist_directory='../../../data/vector_store/data_store',collection_name='CIP_SOC_Associations')
cip_soc_retriever = cip_soc_store.as_retriever()

#%%
# Set up API keys and Additional Tools

os.environ['TAVILY_API_KEY'] = "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"
api_key = 'CeGhuqvlvZmiC0D1ePV6OIBV3XwExBoT6eC3mdYT'
web_search_tool = TavilySearchResults(k=5)

#%%
# Declare the RAG template
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

llm = HuggingFaceEndpoint(
    repo_id=repo_id
)

rag_template = """Answer the question based only on the following contexts:
{context}

Question: {question}
"""

rag_prompt = ChatPromptTemplate.from_template(rag_template)
rag_chain = rag_prompt | llm | StrOutputParser()

#%%
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
    next_step: str


#%%
def vstore_selection(state):
    """
    Route question to corresponding RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    class Vstore(BaseModel):
        datasource: Dict[str, List[str]] = Field(description="the vector stores to use for the user query")


    print("---ROUTE QUESTION---")
    question = state["question"]
    source = vstore_router_agent(repo_id='mistralai/Mistral-7B-Instruct-v0.2',
                          query=question,pydantic_obj=Vstore)

    chosen_vstore = source['datasource']

    if all(item in ['IPEDs vector store', 'BLS vector store', 'CIP-SOC vector store'] for item in chosen_vstore):
        print("---ROUTE QUESTION TO IPEDS, CIP-SOC, & BLS VECTOR STORES---")
        runnable = RunnableParallel(ipeds_documents=ipeds_retriever, bls_documents=bls_retriever, cip_soc_documents=cip_soc_retriever)
        answer = runnable.invoke(state['question'])
        print("---RETRIEVE DOCUMENTS FROM IPEDS, CIP-SOC, & BLS VECTOR STORES---")
        return {"documents": [answer], "question": question}

    elif all(item in ['IPEDs vector store', 'BLS vector store'] for item in chosen_vstore):
        print("---ROUTE QUESTION TO IPEDS & BLS VECTOR STORES---")
        runnable = RunnableParallel(ipeds_documents=ipeds_retriever, bls_documents=bls_retriever)
        answer = runnable.invoke(state['question'])
        print("---RETRIEVE DOCUMENTS FROM IPEDS & BLS VECTOR STORES---")
        return {"documents": [answer], "question": question}

    elif all(item in ['BLS vector store','CIP-SOC vector store'] for item in chosen_vstore):
        print("---ROUTE QUESTION TO CIP-SOC & BLS VECTOR STORES---")
        runnable = RunnableParallel(cip_soc_documents=cip_soc_retriever, bls_documents=bls_retriever)
        answer = runnable.invoke(state['question'])
        print("---RETRIEVE DOCUMENTS FROM CIP-SOC & BLS VECTOR STORES---")
        return {"documents": [answer], "question": question}

    elif chosen_vstore == ["IPEDs vector store"]:
        print("---ROUTE QUESTION TO IPEDS---")
        documents = ipeds_retriever.invoke(question)
        return {"documents": [documents], "question": question}

    elif chosen_vstore == ["BLS vector store"]:
        print("---ROUTE QUESTION TO BLS vector store---")
        documents = bls_retriever.invoke(question)
        return {"documents": [documents], "question": question}

    elif chosen_vstore == ["CIP_SOC vector store"]:
        print("---ROUTE QUESTION TO CIP_SOC vector store---")
        documents = cip_soc_retriever.invoke(question)
        return {"documents": [documents], "question": question}

def entry_agent(state):
    answer = entry_router_agent(query=state["question"],
                       repo_id="mistralai/Mistral-7B-Instruct-v0.2")

    print(f'---DIRECTING USER QUERY TO {answer}---')
    return answer['datasource']

def evaluator(state):
    print(f'---EVALUATING IF DOCUMENTS SUFFICIENT FOR GENERATION---')
    documents = state["documents"]
    question = state["question"]

    answer = evaluator_agent(question=question,
                    context=documents,
                    repo_id="mistralai/Mistral-7B-Instruct-v0.2")

    print(f'---EVALUATOR AGENT SAYS "{answer}"---')

    return {"next_step": answer["relevance"]}

def router_assistant(state):
    print(f'---ASSISTANT ROUTING AGENT ANALYZING QUERY---')
    answer = routing_assistant_agent(query=state["question"],
                            repo_id="mistralai/Mistral-7B-Instruct-v0.2")


    print(f'--- ASSISTANT ROUTING AGENT SENDING QUERY TO "{answer["datasource"]}"---')
    return {"next_step": answer["datasource"]}


def api_agent(state):
    class API(BaseModel):
        filters: Dict[str, str] = Field(description="the filters to use for the API request")
        fields: List[str] = Field(description="list of fields from the college scorecard api to include")

    question = state["question"]
    print(f'---PARSING QUERY TO API PARAMETERS---')
    param_dict = api_parser_agent(query=question,
                     pydantic_obj=API,
                     repo_id="mistralai/Mistral-7B-Instruct-v0.2")
    print(f'---CONTACTING API WITH PARAMETERS---')
    answer = api_call_tool(param_dict=param_dict,
                  API_KEY_HERE=api_key)

    # Ensure 'documents' is initialized in state if it doesn't exist
    documents = state.get("documents", [])

    if documents is not None:
        documents.append(answer)
    else:
        documents = [answer]

    return {"documents": documents, "question": question}

from langchain.schema import Document
def web_search(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    question = state["question"]

    # Ensure 'documents' is initialized in state if it doesn't exist
    documents = state.get("documents", [])
    if not isinstance(documents, list):
        documents = []


    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)
    state["documents"] = documents  # Update state with the modified documents list
    return {"documents": [documents], "question": question}

def generate_answer(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

#%%
from langgraph.graph import START, END, StateGraph
# Define the nodes
workflow = StateGraph(GraphState)
# workflow.add_node("entry_agent", entry_agent)
workflow.add_node("vstore_agent", vstore_selection)
workflow.add_node("evaluator_agent", evaluator)
workflow.add_node("assistant_agent", router_assistant)
workflow.add_node("cs_api", api_agent)
workflow.add_node("tavily_search", web_search) # web search
workflow.add_node("generate", generate_answer)

# Route the user query to the vector stores or the routing assistant
workflow.set_conditional_entry_point(
    entry_agent,
    {"vector store": "vstore_agent", "routing assistant": "assistant_agent"},
)
workflow.add_edge("vstore_agent", "evaluator_agent")

def evaluation(state):
    print(f'---NEXT STEP IS {state["next_step"]}---')
    if state['next_step'] == 'yes':
        return 'generate'
    else:
        return 'assistant_agent'
# While at the evaluator agent, generate an answer or go to the router assistant for more information
workflow.add_conditional_edges("evaluator_agent", evaluation) # do not need nodes for router agents, only the routing functions


def choose_secondary_source(state):
    print(f'---NEXT STEP IS {state["next_step"]}---')
    if state['next_step'] == 'web_search':
        return 'tavily_search'
    else:
        return 'cs_api'
# need to create an assistant_holder node
workflow.add_conditional_edges("assistant_agent",choose_secondary_source)


workflow.add_edge("tavily_search", "generate")
workflow.add_edge("cs_api", "generate")

workflow.add_edge("generate", END)

# Compile
graph = workflow.compile()
#%%

from IPython.display import Image, display
from PIL import Image as PILImage
import io

# Get the image as a byte stream (PNG image)
img_bytes = graph.get_graph().draw_mermaid_png()

# Convert the byte stream to an Image object
img = PILImage.open(io.BytesIO(img_bytes))

# Save the image to a file (for example, "graph_image.png")
img.save('graph_image.png')

# Optionally, display it
display(Image('graph_image.png'))

#%%
question = "What CIP codes are associated with a career in data science?"

response = graph.invoke({"question": question})
print(response)
