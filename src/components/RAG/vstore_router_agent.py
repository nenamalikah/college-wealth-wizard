from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_huggingface import HuggingFaceEndpoint


# %%
def vstore_router_agent(repo_id, query):
    """
        The following function utilizes a HuggingFace LLM to route a user query to a data source that can be used to answer the query.

        Args:
            repo_id (str): The repository ID fo the HuggingFace LLM.
            query (str): The user query to route.

        Returns:
            dict: A dictionary indicating which data source to use to answer the user query.

        """
    llm = HuggingFaceEndpoint(
        repo_id=repo_id
    )

    router_template = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a user question to a series of vector stores. For questions on tuition, grants and scholarships, financial aid packages, room and board, books, offered fields of study, and other collegiate expenses at colleges and universities, use the IPEDs vector store. 
        
        For questions related to mean average wages, total employment, and SOC codes of specific occupations, use the BLS vector store. 
        
        For questions about fields of study (CIP codes/titles) and their associated occupations (SOC codes/titles), use the CIP_SOC vector store. You do not need to be stringent with the keywords in the question related to these topics. 
        
        Based on the question, return a dictionary with a single key 'datasource' and a list as the value containing one or more of the following choices: 'IPEDs vector store', 'BLS vector store', or 'CIP_SOC vector store'. Ensure that the output is in proper JSON format, with **double quotes** for keys and values. 
        
         Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question"],
    )

    question_router = router_template | llm | JsonOutputParser()
    answer = question_router.invoke(query)

    return answer

# %%
if __name__ == '__main__':
    query = 'What net income can I expect as a data scientist after taking loans to study at Georgetown University?'
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    vstore_router_agent(repo_id, query)



