#%%
import pandas as pd
import sys
sys.path.append('../')
from main.cw_app import cw_app

#%%
def test_graph_sources(expected_sources, test_type):
    """A function to compare the expected sources the langgraph application will use for a query versus the actual sources used to answer the query.

    Args:
        expected_sources (dict[str, list]): A dictionary that contains the user query and expected sources. The keys are the user queries, and the values are the expected sources used to answer the query.

        test_type (str): The test type to run. If it is 'print', then the test results will be printed. If it is 'export' then a dataframe will be returned.

    Returns:
        - DataFrame: If test_type is 'export'
        - None: If test_type is 'print'
        """

    if test_type == 'print':
        for question in expected_sources.keys():
            response = cw_app(question)
            graph_source = response['sources']

            if all(item in graph_source for item in expected_sources[question]):
                print(f'TEST PASSED: Graph used each of the expected sources: {expected_sources[question]}. \nThe sources utilized were: {graph_source}.')
            elif any(item in graph_source for item in expected_sources[question]):
                print(f'TEST PARTIALLY PASSED: Graph used some of the expected sources: {expected_sources[question]}. \nThe sources utilized were: {graph_source}.')
            else:
                print(f'TEST FAILED: Graph did not use any of the expected sources: {expected_sources[question]}. \nThe sources utilized were: {graph_source}.')


    elif test_type == 'export':
        graph_sources = []
        scores = []
        for question in expected_sources.keys():
            response = cw_app(question)
            graph_source = response['sources']
            graph_sources.append(graph_source)

            if all(item in graph_source for item in expected_sources[question]):
                scores.append(1)
            elif any(item in graph_source for item in expected_sources[question]):
                scores.append(0)
            else:
                scores.append(-1)

        df = pd.DataFrame({'Question':expected_sources.keys(), 'Score':scores, 'Expected_Sources':expected_sources.values(), 'Graph_Sources':graph_sources})
        return df

#%%
if __name__ == '__main__':
    questions = {'What is the average wage for Data Scientists?':['BLS vector store']}
    test_graph_sources(questions, 'print')

