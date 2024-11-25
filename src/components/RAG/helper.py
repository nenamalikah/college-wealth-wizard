#%%
import sys

sys.path.append('../')
from components.RAG.agents import *

#%%
def entry_agent(state):
    """
        A helper function for the entry router agent. This function calls the entry router agent and returns where the query should be redirected to.

        Args:
                state (dict): The graph state. The "question" key is utilized from the dictionary.

        Returns:
                str: A string indicating which data source to use to answer the user query.

    """
    answer = entry_router_agent(query=state["question"],
                       repo_id="mistralai/Mistral-7B-Instruct-v0.2")

    print(f'---DIRECTING USER QUERY TO {answer}---')
    return answer['datasource']

#%%
def evaluator(state):
    """
       A helper function for the evaluator agent. This function calls the evaluator agent and returns where the query should be redirected to.

       Args:
               state (dict): The graph state. The "documents" and "question" keys are utilized from the dictionary.

       Returns:
               dict: A dictionary indicating what the next step of the graph should be. The options are to generate a response or to go to the router assistant agent.

    """
    print(f'---EVALUATING IF DOCUMENTS SUFFICIENT FOR GENERATION---')
    documents = state["documents"]
    question = state["question"]

    answer = evaluator_agent(question=question,
                    context=documents,
                    repo_id="mistralai/Mistral-7B-Instruct-v0.2")

    print(f'---EVALUATOR AGENT SAYS "{answer}"---')

    return {"next_step": answer["relevance"]}

#%%
def router_assistant(state):
    """
       A helper function for the router assistant agent. This function calls the router assistant agent and returns where the query should be redirected to.

       Args:
               state (dict): The graph state. The "question" key is utilized from the dictionary.

       Returns:
               dict: A dictionary indicating what the next step of the graph should be. The options are to use the web search tool or to use the College Scorecard API.

    """
    print(f'---ASSISTANT ROUTING AGENT ANALYZING QUERY---')
    question = state["question"]
    answer = routing_assistant_agent(query=question,
                            repo_id="mistralai/Mistral-7B-Instruct-v0.2")

    print(f'--- ASSISTANT ROUTING AGENT SENDING QUERY TO "{answer["datasource"]}"---')
    return {"next_step": answer["datasource"]}

#%%

def vs_evaluation(state):
    print(f'---NEXT STEP IS {state["next_step"]}---')
    if state['next_step'] == 'assistant_agent':
        return 'assistant_agent'
    else:
        return 'generate'

def choose_secondary_source(state):
    print(f'---NEXT STEP IS {state["next_step"]}---')
    if state['next_step'] == 'College Scorecard API':
        return 'cs_api'
    else:
        return 'tavily_search'

def api_evaluation(state):
    print(f'---EVALUATING API DOCUMENTS---')
    if state['next_step'] == 'assistant_agent':
        print(f'---NEXT STEP IS WEB SEARCH---')
        return 'tavily_search'
    else:
        print(f'---NEXT STEP IS GENERATE---')
        return 'generate'