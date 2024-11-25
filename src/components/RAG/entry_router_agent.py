from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_huggingface import HuggingFaceEndpoint
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# %%
def entry_router_agent(repo_id, query):
    """
        The following function utilizes a HuggingFace LLM to route a user query to a data source that can be used to answer the query.

        Args:
            repo_id (str): The repository ID fo the HuggingFace LLM.
            query (str): The user query to route.

        Returns:
            dict: A dictionary indicating which data source to use to answer the user query.

        """

    response_schemas = [
        ResponseSchema(name="datasource",
                       description="whether the provided context should be sent to 'vector stores', which contain information on college expenses, financial aid, average salary and wages, CIP codes, and SOC codes, or if should be sent to the 'routing assistant'."),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    llm = HuggingFaceEndpoint(
        repo_id=repo_id
    )

    router_template = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a user question to a series of vector stores or to a routing assistant. For questions on college tuition, financial aid and grants, collegiate expenses, mean average wages for a given occupation, salary for a given occupation, total employment for a given occupation, fields of study (CIP codes/titles) and their associated occupations (SOC codes/titles), use the vector stores. You do not need to be stringent with the keywords in the question related to these topics. 
        
        For questions about admission rates, post-graduation salary, post-graduation earnings, student loan debt payments, student loan default rates for educational institutions, and other questions, route the query to the routing assistant. 

        Based on the question, return a dictionary with a single key 'datasource' and one of the following choices: 'vector store' or 'routing assistant'. Ensure that the output is in proper JSON format, with **double quotes** for keys and values. 
        
        Here is how your response should be formatted: {format_instructions}\n
         Here is the user query: {question}\n <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question"],
                            partial_variables={"format_instructions": output_parser.get_format_instructions()}
    )

    entry_direction = router_template | llm | JsonOutputParser()
    answer = entry_direction.invoke(query)

    return answer


# %%
if __name__ == '__main__':
    q1 = 'What is the average net price for attendance at Georgetown University?'
    q2 = 'What university in the US has the best data science program?'

    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

    answer1 = entry_router_agent(query=q1, repo_id=repo_id)
    answer2 = entry_router_agent(query=q2, repo_id=repo_id)

    print(f'For query "{q1}", the answer is: {answer1}')
    print(f'For query "{q2}", the answer is: {answer2}')



