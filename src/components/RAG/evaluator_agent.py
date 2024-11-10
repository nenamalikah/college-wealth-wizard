from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_huggingface import HuggingFaceEndpoint

#%%

def evaluator_agent(question, context, repo_id):
    evaluator_template = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a helpful assistant. Given the following context and the user's query, evaluate whether this document contains the relevant information to answer the query. 

    Give a binary score 'yes' or 'no' score to indicate whether the context provides sufficient information to answer the user question. \n

    Provide the binary score as a JSON with a single key 'score' and no premable or explanation. Ensure that the output is in proper JSON format, with **double quotes** for keys and values. 
     <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the context: \n\n {context} \n\n
    Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
        input_variables=["question", "context"])

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

    evaluator_agent(question=question, context=test_doc, repo_id=repo_id)
