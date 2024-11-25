from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_huggingface import HuggingFaceEndpoint
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
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
                       description="whether the provided context should be sent to 'vector stores', which contain information on college expenses, average salary and wages, CIP codes, and SOC codes, or if should be sent to the 'routing assistant'."),
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

#%%
def vstore_router_agent(repo_id, query, pydantic_obj):
    """
        The following function utilizes a HuggingFace LLM to route a user query to a data source that can be used to answer the query.

        Args:
            repo_id (str): The repository ID fo the HuggingFace LLM.
            query (str): The user query to route.

        Returns:
            dict: A dictionary indicating which data source to use to answer the user query.

        """

    parser = PydanticOutputParser(pydantic_object=pydantic_obj)
    llm = HuggingFaceEndpoint(
        repo_id=repo_id
    )

    router_template = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a user question to a series of vector stores. For questions on tuition, grants and scholarships, financial aid packages, room and board, books, offered fields of study, and other collegiate expenses at colleges and universities, use the IPEDs vector store. 

        For questions related to salary, mean average wages, total employment, and SOC codes of specific occupations, use the BLS vector store. 

        For questions about fields of study (CIP codes/titles) and their associated occupations (SOC codes/titles), use the CIP_SOC vector store. You do not need to be stringent with the keywords in the question related to these topics. 

        Based on the question, return a dictionary with a single key 'datasource' and a list as the value containing one or more of the following choices: 'IPEDs vector store', 'BLS vector store', or 'CIP_SOC vector store'. Ensure that the output is in proper JSON format, with **double quotes** for keys and values. 

        Here are the formatting instructions: {format_instructions}
         Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    question_router = router_template | llm | JsonOutputParser()
    answer = question_router.invoke(query)

    return answer

#%%
def evaluator_agent(question, context, repo_id):
    response_schemas = [
        ResponseSchema(name="relevance",
                       description="whether or not the provided context can answer the user query. if it can, the value is generate. otherwise, it is assistant_agent."),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    evaluator_template = PromptTemplate(
        template="""You are an expert at evaluating whether provided context can be used to answer a user query. If the context is relevant, you generate an answer, Otherwise, you pass the query to the routing assistant.

         Based on the user query, return a SINGLE JSON object with a key called 'relevance'. If the provided context provides sufficient information to answer the user question, choose 'generate' as the dictionary value. Otherwise, choose 'assistant_agent'.\n

         Here is how your response should be formatted: {format_instructions}\n
         Here is the context: {context}\n
         Here is the user query: {question}\n
    """,
        input_variables=["question", "context"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()})

    llm = HuggingFaceEndpoint(
        repo_id=repo_id
    )
    evaluator_agent = evaluator_template | llm | JsonOutputParser()

    answer = evaluator_agent.invoke({'question': question, 'context': context})
    return answer

#%%
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
def api_parser_agent(query, repo_id):
    """Parses a user query into API parameters using an LLM."""

    llm = HuggingFaceEndpoint(
        repo_id=repo_id
    )

    response_schemas = [
        ResponseSchema(name="filters",
                       description="the filters to use for the API request",type='dict'),
        ResponseSchema(name="fields",
                       description="list of fields from the college scorecard api to include",type='list'),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    # parser = PydanticOutputParser(pydantic_object=pydantic_obj)
    prompt = PromptTemplate(template="""
    You are an API expert for the College Scorecard API.

    The API has the following filters:
    - 'school.name': The name of the city.
    - 'school.state': The name of the state (optional).
    - 'school.city': The name of the country (optional).

    The API has the following fields: 
    - 'latest.admissions.admission_rate_suppressed.overall': The percentage of students who are admitted to the university after applying
    - 'latest.earnings.1_yr_after_completion.median': The median salary of students 1 year after graduating university
    - 'latest.earnings.4_yrs_after_completion.median': The median salary of students 4 years after graduating university
    - 'latest.earnings.6_yrs_after_entry.median': The median salary of students 6 years after entering university
    - 'latest.earnings.8_yrs_after_entry.median': The median salary of students 8 years after entering university
    - 'latest.earnings.10_yrs_after_entry.median': The median salary of students 10 years after entering university
    - 'aid.median_debt_suppressed.overall': The median debt of all students overall
    - 'aid.median_debt_suppressed.completers.overall': The median debt of students who graduated from the university
    - 'aid.median_debt_suppressed.completers.monthly_payments': The median debt of students who graduated from the university expressed in monthly payments

    Given a user query, you need to extract the relevant filters and fields to call the College Scorecard API. 

    Here are the formatting instructions: {format_instructions}
    Here is the user query: {query}
    """, input_variables=["query"],
                            partial_variables={"format_instructions": output_parser.get_format_instructions()})
    query_router = prompt | llm | JsonOutputParser()

    answer = query_router.invoke(query)

    return answer

#%%
def generate_answer(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    # Declare the RAG template
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

    llm = HuggingFaceEndpoint(
        repo_id=repo_id)

    rag_template = """Answer the question based only on the following contexts:
    {context}

    Question: {question}
    """

    rag_prompt = ChatPromptTemplate.from_template(rag_template)
    rag_chain = rag_prompt | llm | StrOutputParser()

    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}
