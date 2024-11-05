from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

llm = HuggingFaceEndpoint(
    repo_id=repo_id
)

#%%
# Declare the RAG template
rag_template = """Answer the question based only on the following contexts:
{context}

Question: {question}
"""

#%%

router_template = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a user question to a series of vector stores or a web search. For questions on tuition, financial aid, room and board, books, offered fields of study, and other collegiate expenses at colleges and universities, use the IPEDs vector store. For questions related to mean average wages, total employment, and SOC codes of specific occupations, use the BLS vector store. For questions about fields of study (CIP codes/titles) and their associated occupations (SOC codes/titles), use the CIP_SOC vector store. You do not need to be stringent with the keywords in the question related to these topics. Otherwise, use web_search. Give one of the following choices based on the question: 'IPEDs vector store', 'BLS vector store',  'CIP_SOC vector store', or 'web_search'. Return a JSON with a single key 'datasource' and no preamble or explanation. Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question"],
)

#%%
# Set up the RAG, Router and
embeddings = HuggingFaceEmbeddings(model_name = 'BAAI/bge-large-en-v1.5')
ipeds_store = Chroma(embedding_function=embeddings,persist_directory='../../../data/vector_store/data_store',collection_name='IPEDS_Education_Information')
ipeds_retriever = ipeds_store.as_retriever()

bls_store = Chroma(embedding_function=embeddings,persist_directory='../../../data/vector_store/data_store',collection_name='BLS_Occupational_Information')
bls_retriever = bls_store.as_retriever(search_kwargs={"k": 5})

cip_soc_store = Chroma(embedding_function=embeddings,persist_directory='../../../data/vector_store/data_store',collection_name='CIP_SOC_Associations')
cip_soc_retriever = cip_soc_store.as_retriever()

import os
from langchain_community.tools.tavily_search import TavilySearchResults
os.environ['TAVILY_API_KEY'] = "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"
web_search_tool = TavilySearchResults(k=5)

rag_prompt = ChatPromptTemplate.from_template(rag_template)
rag_chain = rag_prompt | llm | StrOutputParser()


question_router = router_template | llm | JsonOutputParser()

#%%
from typing_extensions import TypedDict
from typing import List

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
    web_search: str
    documents: List[str]

#%%
def ipeds_retrieve(state):
    """
    Retrieve documents from the IPEDs vector store

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE IPEDs---")
    question = state["question"]

    # Retrieval
    documents = ipeds_retriever.invoke(question)
    return {"documents": documents, "question": question}

def bls_retrieve(state):
    """
    Retrieve documents from the BLS vector store

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE BLS---")
    question = state["question"]

    # Retrieval
    documents = bls_retriever.invoke(question)
    return {"documents": documents, "question": question}

def cip_soc_retrieve(state):
    """
    Retrieve documents from the CIP-SOC vector store

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE CIP_SOC---")
    question = state["question"]

    # Retrieval
    documents = cip_soc_retriever.invoke(question)
    return {"documents": documents, "question": question}

#%%
def generate(state):
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
    # documents = state["documents"]


    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}


def router(state):
    """
    Route question to corresponding RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})

    if isinstance(source, str):
        holder = {}
        holder['datasource'] = source
        source = holder

    if source['datasource'] == "IPEDs vector store":
        print("---ROUTE QUESTION TO IPEDS---")
        return "IPEDs vector store"
    elif source['datasource']  == "BLS vector store":
        print("---ROUTE QUESTION TO BLS vector store---")
        return "BLS vector store"
    elif source['datasource']  == "CIP_SOC vector store":
        print("---ROUTE QUESTION TO CIP_SOC vector store---")
        return "CIP_SOC vector store"
    elif source['datasource']  == "web_search":
        print("---ROUTE QUESTION TO web_search---")
        return "web_search"

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"
#%%
from langgraph.graph import END, StateGraph
# Define the nodes
workflow = StateGraph(GraphState)

workflow.add_node("tavily_search", web_search) # web search
workflow.add_node("ipeds_retrieve", ipeds_retrieve)
workflow.add_node("bls_retrieve", bls_retrieve)
workflow.add_node("cip_soc_retrieve", cip_soc_retrieve)
workflow.add_node("generate", generate)

workflow.set_conditional_entry_point(
    router,
    {
            "IPEDs vector store": "ipeds_retrieve",
            "BLS vector store": "bls_retrieve",
            "CIP_SOC vector store": "cip_soc_retrieve",
            "web_search": "tavily_search",
        },
)

workflow.add_edge("ipeds_retrieve", "generate")
workflow.add_edge("bls_retrieve", "generate")
workflow.add_edge("cip_soc_retrieve", "generate")
workflow.add_edge("tavily_search", "generate")
workflow.add_edge("generate", END)


# Compile
graph = workflow.compile()

#%%
question = "How many data scientists are employed in the Information industry?"

response = graph.invoke({"question": question})
print(response)