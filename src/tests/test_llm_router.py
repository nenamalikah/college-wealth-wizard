from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_huggingface import HuggingFaceEndpoint

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

llm = HuggingFaceEndpoint(
    repo_id=repo_id
)

#%%
def build_llm_router(query):
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

    llm = HuggingFaceEndpoint(
        repo_id=repo_id
    )
    router_template = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a user question to a series of vector stores or a web search. For questions on tuition, financial aid, room and board, books, offered fields of study, and other collegiate expenses at colleges and universities, use the IPEDs vector store. 
        
        For questions related to mean average wages, total employment, and SOC codes of specific occupations, use the BLS vector store. 
        
        For questions about fields of study (CIP codes/titles) and their associated occupations (SOC codes/titles), use the CIP_SOC vector store. You do not need to be stringent with the keywords in the question related to these topics. Otherwise, use web_search. 
        
        Based on the question, return a dictionary with a single key 'datasource' and a list as the value containing one or more of the following choices: 'IPEDs vector store', 'BLS vector store', 'CIP_SOC vector store', or 'web_search'. Ensure that the output is in proper JSON format, with **double quotes** for keys and values. 
        
         Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question"],
    )
    # router_template = PromptTemplate(template=prompt,
    #                                  input_variables=["question"])
    question_router = router_template | llm | JsonOutputParser()
    answer = question_router.invoke(query)

    return answer

build_llm_router('What net income can I expect as a data scientist after taking loans to study at Georgetown University?')
#%%
def test_llm_router_prompt():
    query = input('Please enter your query.')
    source = input('Please enter the source(s) for your query, separated by a comma.')
    source = source.strip("'").strip('"')
    source = source.split(",")

    source = [item.lstrip() for item in source]

    print('The LLM will now route your query to the correct data source.')

    answer = build_llm_router(query)

    if all(item in answer.get('datasource') for item in source):
        print(f'\nTEST PASSED: User input of source "{source}" equals LLM output of "{answer.get("datasource")}".')
    else:
        print(f'\nTEST FAILED: User input of source "{source}" DOES NOT equal LLM output of "{answer.get("datasource")}".')

#%%
if __name__ == '__main__':
    test_llm_router_prompt()
