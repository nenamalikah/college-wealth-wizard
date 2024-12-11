import sys

sys.path.append('../')
from components.preprocess.generate_documents import load_documents
from components.RAG.agents import evaluator_agent

#%%
def test_llm_evaluator_agent(question, context, relevance, repo_id):
    test_case = evaluator_agent(question, context, repo_id)

    if test_case['score'] == relevance:
        print(f'TEST PASSED: Evaluator agent correctly correctly labeled provided documents for query as {relevance}.')
    else:
        print(f'TEST FAILED: Evaluator agent incorrectly labeled provided documents for query as {test_case["score"]}.')

#%%
if __name__ == '__main__':
    docs = load_documents(document_obj_fp='../../data/document_objs/ipeds_doc_obj.pkl')

    test_doc = docs[15]
    question = 'How much are the books and supplies at Aaniiih Nakoda College?'
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

    test_llm_evaluator_agent(question=question, context=test_doc, relevance='yes', repo_id=repo_id)
