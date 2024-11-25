#%%
import os
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults

from pydantic import BaseModel, Field
from typing import List, Dict
from urllib.request import urlopen
from json import loads

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableParallel

import sys
sys.path.append('../')
from components.RAG.agents import vstore_router_agent
from components.RAG.agents import api_parser_agent
#%%
# Set up API keys and Additional Tools




#%%
# ---------- DEFINE VECTOR STORE RETRIEVAL TOOLS ----------

embeddings = HuggingFaceEmbeddings(model_name = 'BAAI/bge-large-en-v1.5')
ipeds_store = Chroma(embedding_function=embeddings,persist_directory='../../../data/vector_store/data_store',collection_name='IPEDS_Education_Information')
ipeds_retriever = ipeds_store.as_retriever(search_kwargs={"k": 3})

bls_store = Chroma(embedding_function=embeddings,persist_directory='../../../data/vector_store/data_store',collection_name='BLS_Occupational_Information')
bls_retriever = bls_store.as_retriever(search_kwargs={"k": 5})

cip_soc_store = Chroma(embedding_function=embeddings,persist_directory='../../../data/vector_store/data_store',collection_name='CIP_SOC_Associations')
cip_soc_retriever = cip_soc_store.as_retriever(search_kwargs={"k": 5})

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

    print(f'---QUESTION ROUTED TO {chosen_vstore}---')

    if chosen_vstore == ["IPEDs vector store"]:
        print("---RETRIEVE DOCUMENTS FROM IPEDS VECTOR STORE---")
        documents = ipeds_retriever.invoke(question)
        return {"documents": [documents], "question": question, "sources":[chosen_vstore]}

    elif chosen_vstore == ["BLS vector store"]:
        print("---RETRIEVE DOCUMENTS FROM BLS VECTOR STORE---")
        documents = bls_retriever.invoke(question)
        return {"documents": [documents], "question": question, "sources":[chosen_vstore]}

    elif chosen_vstore == ["CIP_SOC vector store"]:
        print("---RETRIEVE DOCUMENTS FROM CIP_SOC VECTOR STORE---")
        documents = cip_soc_retriever.invoke(question)
        return {"documents": [documents], "question": question, "sources":[chosen_vstore]}

    elif all(item in ['IPEDs vector store', 'BLS vector store', 'CIP-SOC vector store'] for item in chosen_vstore):
        # print("---ROUTE QUESTION TO IPEDS, CIP-SOC, & BLS VECTOR STORES---")
        runnable = RunnableParallel(ipeds_documents=ipeds_retriever, bls_documents=bls_retriever, cip_soc_documents=cip_soc_retriever)
        answer = runnable.invoke(state['question'])
        print("---RETRIEVE DOCUMENTS FROM IPEDS, CIP-SOC, & BLS VECTOR STORES---")
        return {"documents": [answer], "question": question, "sources":[chosen_vstore]}

    elif all(item in ['IPEDs vector store', 'BLS vector store'] for item in chosen_vstore):
        # print("---ROUTE QUESTION TO IPEDS & BLS VECTOR STORES---")
        runnable = RunnableParallel(ipeds_documents=ipeds_retriever, bls_documents=bls_retriever)
        answer = runnable.invoke(state['question'])
        print("---RETRIEVE DOCUMENTS FROM IPEDS & BLS VECTOR STORES---")
        return {"documents": [answer], "question": question, "sources":[chosen_vstore]}

    elif all(item in ['BLS vector store','CIP-SOC vector store'] for item in chosen_vstore):
        # print("---ROUTE QUESTION TO CIP-SOC & BLS VECTOR STORES---")
        runnable = RunnableParallel(cip_soc_documents=cip_soc_retriever, bls_documents=bls_retriever)
        answer = runnable.invoke(state['question'])
        print("---RETRIEVE DOCUMENTS FROM CIP-SOC & BLS VECTOR STORES---")
        return {"documents": [answer], "question": question, "sources":[chosen_vstore]}

    elif all(item in ['IPEDs vector store','CIP-SOC vector store'] for item in chosen_vstore):
        # print("---ROUTE QUESTION TO CIP-SOC & BLS VECTOR STORES---")
        runnable = RunnableParallel(cip_soc_documents=cip_soc_retriever, bls_documents=bls_retriever)
        answer = runnable.invoke(state['question'])
        print("---RETRIEVE DOCUMENTS FROM CIP-SOC & BLS VECTOR STORES---")
        return {"documents": [answer], "question": question, "sources":[chosen_vstore]}


#%%
# ---------- DEFINE TAVILY WEB SEARCH TOOL ----------

def web_search(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """
    tavily_key = os.getenv("TAVILY_API_KEY")
    print("---WEB SEARCH---")
    question = state["question"]

    # Ensure 'documents' is initialized in state if it doesn't exist
    documents = state.get("documents", [])
    if not isinstance(documents, list):
        documents = []


    web_search_tool = TavilySearchResults(k=3)

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)
    state["documents"] = documents  # Update state with the modified documents list
    return {"documents": [documents], "question": question, "sources":['Web Search']}

#%%

def api_call_tool(param_dict, API_KEY_HERE):
    filters = ''
    filter_names = []
    try:
        filter_dict = param_dict['filters']
        for key, value in filter_dict.items():
            filter_names.append(key)
            filters += key
            filters += '='
            filters += value.replace(' ', '%20')
            filters += '&'
    except:
        print('There are no filters available.')

    fields = param_dict['fields'] + filter_names
    fields = ','.join(fields)

    api_url = f'https://api.data.gov/ed/collegescorecard/v1/schools?api_key={API_KEY_HERE}&{filters}fields={fields}'

    response = urlopen(api_url)
    data = loads(response.read())

    return data

def api_agent(state):
    # class API(BaseModel):
    #     filters: Dict[str, str] = Field(description="the filters to use for the API request")
    #     fields: List[str] = Field(description="list of fields from the college scorecard api to include")
    cs_key = os.getenv("CS_API_KEY")
    question = state["question"]
    print(f'---PARSING QUERY TO API PARAMETERS---')
    param_dict = api_parser_agent(query=question,
                     repo_id="mistralai/Mistral-7B-Instruct-v0.2")
    print(f'---CONTACTING API WITH PARAMETERS---')
    answer = api_call_tool(param_dict=param_dict,
                  API_KEY_HERE=cs_key)

    # Ensure 'documents' is initialized in state if it doesn't exist
    documents = state.get("documents", [])

    if documents is not None:
        documents.append(answer)
    else:
        documents = [answer]

    return {"documents": documents, "question": question, "sources":['College Scorecard API']}