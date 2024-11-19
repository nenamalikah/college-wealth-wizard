from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_huggingface import HuggingFaceEndpoint
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

#%%

def evaluator_agent(question, context, repo_id):
    response_schemas = [
        ResponseSchema(name="relevance", description="whether or not the provided context can answer the user query. if it can, the value is generate. otherwise, it is assistant_agent."),
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
if __name__ == '__main__':
    import sys
    sys.path.append('../../')
    from components.preprocess.generate_documents import load_documents

    docs = load_documents(document_obj_fp='../../../data/document_objs/ipeds_doc_obj.pkl')
    test_doc = docs[15]

    question = 'How much are the books and supplies at Aaniiih Nakoda College?'
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    print(evaluator_agent(question=question, context=test_doc, repo_id=repo_id))
