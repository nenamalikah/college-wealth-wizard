#%%
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os

import sys
sys.path.append('../')
from main.app import app

#%%
def query_key_parser(query, repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1"):
    """
        The following function utilizes a HuggingFace LLM to parse the key aspects of a user query.

        Args:
            repo_id (str): The repository ID fo the HuggingFace LLM.
            query (str): The user query to route.

        Returns:
            dict: A dictionary indicating the key aspects of the user query.

        """

    response_schemas = [ResponseSchema(name="keys",
                       description="A single list of key aspects extracted from the user query. The list must ONLY include key aspects as strings."),]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    llm = HuggingFaceEndpoint(
        repo_id=repo_id
    )

    router_template = PromptTemplate(
        template="""
You are an expert at identifying the keywords, key phrases, and aspects of user queries, especially in the context of education, employment, or occupations.

Your task is to extract and return **only** a single list of key aspects from the user query. 
- Do NOT include any additional text, explanations, or formatting other than the specified JSON format.
- The output must strictly follow this structure:
  {{
    "keys": ["aspect_1", "aspect_2", "aspect_3"]
  }}

Here is the query: {query}
{format_instructions}
""",
        input_variables=["query"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()}
    )

    entry_direction = router_template | llm | JsonOutputParser()
    answer = entry_direction.invoke(query)

    return answer

def test_query_answer(questions, repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1"):

    data = {'Query':[],
            'Key Aspects': [],
            'Sources':[],
            'Documents':[],
            'Generation':[]}
    tavily_key = os.getenv("TAVILY_API_KEY")
    cs_key = os.getenv("CS_API_KEY")
    for query in questions:
            try:
                print('\n\n------GENERATING KEY ASPECTS OF QUERY------\n\n')

                key_aspects = query_key_parser(repo_id = repo_id, query = query)
                print('\n\n------GENERATING LLM RESPONSE------\n\n')

                response = app(query)
                generation = response['generation']
                data['Generation'].append(generation)
                data['Key Aspects'].append(key_aspects['keys'])
                data['Query'].append(query)
                data['Sources'].append(response['sources'])
                data['Documents'].append(response['documents'])

            except:
                print(f'ERROR PROCESSING QUERY: {query} \n\n')
                continue

    df = pd.DataFrame(data)
    print(df.head())

    return df



#%%
if __name__ == '__main__':

    bls_questions = [
    "What is the mean wage for a software engineer?",
    "What is the SOC code for a teacher?",
    "What is the total employment of carpenters in the US?",
    "What is the mean annual salary for a graphic designer in the US?",
    "What is the mean wage for a mechanical engineer?",
    "What is the SOC code for a nurse?",
    "What is the total employment of architects in the US?",
    "What is the mean annual salary for a chemist in the US?",
    "What is the mean wage for an accountant?",
    "What is the SOC code for a civil engineer?",
    "What is the total employment of welders in the US?",
    "What is the mean annual salary for a software developer in the US?",
    "What is the mean wage for a biologist?",
    "What is the SOC code for a physician?",
    "What is the total employment of bartenders in the US?",
    "What is the mean annual salary for a web developer in the US?",
    "What is the mean wage for a psychologist?",
    "What is the SOC code for a marketing manager?",
    "What is the total employment of lawyers in the US?",
    "What is the mean annual salary for a plumber in the US?",
    "What is the mean wage for a pharmacist?",
    "What is the SOC code for an HR specialist?",
    "What is the total employment of truck drivers in the US?",
    "What is the mean annual salary for a veterinarian in the US?",
    "What is the mean wage for an aerospace engineer?",
    'How much do auto mechanics make?',
    'How much is the average salary of web developers?']
    print('\n\n-----BEGINNING BLS TEST------\n\n')

    df_bls = test_query_answer(bls_questions)
    print(df_bls.head())

    df_bls.to_excel('bls_app_test.xlsx')
    print('\n\n-----BLS TEST COMPLETE------\n\n')

    print('\n\n-----BEGINNING IPEDs TEST------\n\n')

    ipeds_questions = [
    "What is the average net price at Georgetown University?",
    "What fields of study are offered at George Mason University?",
    "How much are room and board expenses at Harvard University?",
    "How much is the tuition at Yale University?",
    "What is the average net price at Princeton University?",
    "What fields of study are offered at Columbia University?",
    "How much are room and board expenses at Duke University?",
    "How much is the tuition at Brown University?",
    "What is the average net price at the University of Chicago?",
    "What fields of study are offered at the University of Pennsylvania?",
    "How much are room and board expenses at Northwestern University?",
    "How much is the tuition at Dartmouth College?",
    "What is the average net price at Johns Hopkins University?",
    "What fields of study are offered at the Massachusetts Institute of Technology?",
    "How much are room and board expenses at the California Institute of Technology?",
    "How much is the tuition at the University of California, Berkeley?",
    "What is the average net price at UCLA?",
    "What fields of study are offered at the University of Virginia?",
    "How much are room and board expenses at the University of Michigan?",
    "How much is the tuition at the University of Texas at Austin?",
    "What is the average net price at New York University?",
    "What fields of study are offered at Boston University?",
    "How much are room and board expenses at the University of North Carolina at Chapel Hill?",
    "How much is the tuition at Carnegie Mellon University?",
    "What is the average net price at Purdue University?",
    "What CIP codes are offered at Emory University?",
    'What is the tuition at Georgia Institue of Technology?',
    'What is the average net price at Tufts University?']

    df_ipeds = test_query_answer(ipeds_questions)
    print(df_ipeds.head())

    df_ipeds.to_excel('ipeds_app_test.xlsx')
    print('\n\n-----IPEDS TEST COMPLETE------\n\n')
    print('\n\n-----BEGINNING CIP-SOC TEST------\n\n')

    cip_soc_questions = [
    "What occupations are associated with a CIP Code in 110101?",
    "What field of study should I choose for a career in Software Engineering?",
    "What CIP codes are associated with a career in Mechanical Engineering?",
    "What SOC codes are associated with a field of study in Computer Science?",
    "What occupations are associated with a CIP Code in 260101?",
    "What field of study should I choose for a career in Accounting?",
    "What CIP codes are associated with a career in Electrical Engineering?",
    "What SOC codes are associated with a field of study in Marketing?",
    "What occupations are associated with a CIP Code in 140101?",
    "What field of study should I choose for a career in Nursing?",
    "What CIP codes are associated with a career in Architecture?",
    "What SOC codes are associated with a field of study in Psychology?",
    "What occupations are associated with a CIP Code in 450101?",
    "What field of study should I choose for a career in Environmental Science?",
    "What CIP codes are associated with a career in Biomedical Engineering?",
    "What SOC codes are associated with a field of study in Economics?",
    "What occupations are associated with a CIP Code in 510201?",
    "What field of study should I choose for a career in Public Health?",
    "What CIP codes are associated with a career in Civil Engineering?",
    "What SOC codes are associated with a field of study in Finance?",
    "What occupations are associated with a CIP Code in 270101?",
    "What field of study should I choose for a career in Education?",
    "What CIP codes are associated with a career in Aerospace Engineering?",
    "What SOC codes are associated with a field of study in Data Analytics?",
    "What occupations are associated with a CIP Code in 300101?"]

    df_cip_soc = test_query_answer(cip_soc_questions)
    print(df_cip_soc.head())

    df_cip_soc.to_excel('cipsoc_app_test.xlsx')
    print('\n\n-----CIP-SOC TEST COMPLETE------\n\n')

    print('\n\n-----BEGINNING COLLEGE SCORECARD TEST------\n\n')
    cs_questions = [
    "What is the salary of students at Harvard University 4 years after graduation?",
    "What is the salary of students at Yale University 1 year after graduation?",
    "How much are the monthly debt payments for students who graduated from Stanford University?",
    "How much student loan debt do graduates of Princeton University have?",
    "What is the salary of students at Columbia University 4 years after graduation?",
    "What is the salary of students at Duke University 1 year after graduation?",
    "How much are the monthly debt payments for students who graduated from the University of Chicago?",
    "How much student loan debt do graduates of the University of Pennsylvania have?",
    "What is the salary of students at Dartmouth College 4 years after graduation?",
    "What is the salary of students at Cornell University 1 year after graduation?",
    "How much are the monthly debt payments for students who graduated from Brown University?",
    "How much student loan debt do graduates of Johns Hopkins University have?",
    "What is the salary of students at the University of California, Berkeley 4 years after graduation?",
    "What is the salary of students at UCLA 1 year after graduation?",
    "How much are the monthly debt payments for students who graduated from the University of Michigan?",
    "How much student loan debt do graduates of the University of Virginia have?",
    "What is the salary of students at the University of North Carolina at Chapel Hill 4 years after graduation?",
    "What is the salary of students at the University of Texas at Austin 1 year after graduation?",
    "How much are the monthly debt payments for students who graduated from Purdue University?",
    "How much student loan debt do graduates of Boston University have?",
    "What is the salary of students at New York University 4 years after graduation?",
    "What is the salary of students at Carnegie Mellon University 1 year after graduation?",
    "How much are the monthly debt payments for students who graduated from the University of Southern California?",
    "How much student loan debt do graduates of the University of Washington have?",
    "What is the salary of students at Vanderbilt University 4 years after graduation?",
    "How much do students at the University of Texas Dallas make 6 years after entering school?",
    "What is the admissions rate for Kansas State University?",
    "I am a student at Massachusetts Institute of Technology. What salary can I expect 1 year after graduation?",
    "What is the admissions rate at Duke University?",
    'What is the average salary of graduates of New York University?',
    'What are the monthly student loan debt payments for graduates of Carnegie Mellon University?',
    'How much do graduates of Dartmouth College make a year after graduation?',
    'How much do students at Emory University make 4 years after graduating?',
    '6 years after starting college, how much do graduates of John Hopkins Univesity make?',
    '6 years after entering school, how much can a graduate of Northwestern University expect to make?',
    'What is the admissions rate of Duke University? Is it competitive?',
    'Does Princeton University have a low or high admissions rate?']

    df_cs = test_query_answer(cs_questions)
    print(df_cs.head())

    df_cs.to_excel('csapi_app_test.xlsx')
    print('\n\n-----COLLEGE SCORECARD TEST COMPLETE------\n\n')

