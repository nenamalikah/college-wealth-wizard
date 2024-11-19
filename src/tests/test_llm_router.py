import sys
sys.path.append('../')
from components.RAG.vstore_router_agent import vstore_router_agent

#%%
def test_llm_router_prompt(repo_id):
    query = input('Please enter your query.')
    source = input('Please enter the source(s) for your query, separated by a comma.')
    source = source.strip("'").strip('"')
    source = source.split(",")

    source = [item.lstrip() for item in source]

    print('The LLM will now route your query to the correct data source.')

    answer = vstore_router_agent(repo_id, query)

    if all(item in answer.get('datasource') for item in source):
        print(f'\nTEST PASSED: User input of source "{source}" equals LLM output of "{answer.get("datasource")}".')
    else:
        print(f'\nTEST FAILED: User input of source "{source}" DOES NOT equal LLM output of "{answer.get("datasource")}".')

#%%
if __name__ == '__main__':
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    test_llm_router_prompt(repo_id)
