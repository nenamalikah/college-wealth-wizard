#%%
import os
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults

from pydantic import BaseModel, Field
from typing import List, Dict
#%%
#
# Set up API keys and Additional Tools
tavily_key = os.getenv("TAVILY_API_KEY")
cs_key = os.getenv("CS_API_KEY")

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


    web_search_tool = TavilySearchResults(k=3)

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)
    state["documents"] = documents  # Update state with the modified documents list
    return {"documents": [documents], "question": question, "sources":['Web Search']}

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
                  API_KEY_HERE=cs_key)

    # Ensure 'documents' is initialized in state if it doesn't exist
    documents = state.get("documents", [])

    if documents is not None:
        documents.append(answer)
    else:
        documents = [answer]

    return {"documents": documents, "question": question, "sources":['College Scorecard API']}