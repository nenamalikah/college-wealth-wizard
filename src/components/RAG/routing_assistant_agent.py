from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_huggingface import HuggingFaceEndpoint


# %%
def routing_assistant_agent(query, repo_id):
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
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant whose task is to route a user question to a web search or the College Scorecard API. 

        For questions about admission rates, post-graduation salary, post-graduation earnings, student loan debt payments, and student loan default rates for educational institutions, use the College Scorecard API. You do not need to be stringent with the keywords in the question related to these topics. Otherwise, use the web search. 

        Based on the question, return a JSON dictionary with a single key 'datasource' and your selected choice: 'web_search' or 'College Scorecard API'. Ensure that the output is in proper JSON format, with **double quotes** for keys and values. 

         Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question"],
    )

    question_router = router_template | llm | JsonOutputParser()
    answer = question_router.invoke(query)
    return answer

#%%
if __name__ == "__main__":
    q1 = 'What is the median salary of students at Georgetown University four years after graduation?'
    q2 = 'What university in the US has the best data science program?'

    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

    answer1 = routing_assistant_agent(query=q1, repo_id=repo_id)
    answer2 = routing_assistant_agent(query=q2, repo_id=repo_id)

    print(f'For query "{q1}", the answer is: {answer1}')
    print(f'For query "{q2}", the answer is: {answer2}')