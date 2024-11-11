from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict

from urllib.request import urlopen
from json import loads




# %%
def api_parser_agent(query, pydantic_obj, repo_id):
    """Parses a user query into API parameters using an LLM."""

    llm = HuggingFaceEndpoint(
        repo_id=repo_id
    )

    parser = PydanticOutputParser(pydantic_object=pydantic_obj)
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
                            partial_variables={"format_instructions": parser.get_format_instructions()})
    query_router = prompt | llm | JsonOutputParser()

    answer = query_router.invoke(query)

    return answer

# %%
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

#%%
if __name__ == '__main__':
    class API(BaseModel):
        filters: Dict[str, str] = Field(description="the filters to use for the API request")
        fields: List[str] = Field(description="list of fields from the college scorecard api to include")


    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    param_dictionary = api_parser_agent(query="What is the median salary of students at Georgetown University six years after enrollment?", pydantic_obj=API, repo_id=repo_id)
    api_key = ' '

    api_results = api_call_tool(param_dict=param_dictionary, API_KEY_HERE=api_key)
    print(api_results)
