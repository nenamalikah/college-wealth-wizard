#%%
import pandas as pd
import sy
sys.path.append('../')
from RAG.components.generate_retrieval_metrics import retrieval_accuracy
#%%

def test_langchain_rag(test_cases):
    for idx in test_cases.index:
        if str(test_cases["source_doc_vld"][idx]) in test_cases["source_doc_rtrv"][idx]:
            print(f"Test passed for query: {test_cases['question'][idx]}")
        else:
            print(f"Test failed for query: {test_cases['question'][idx]}. Expected Document: {test_cases['source_doc_vld'][idx]}, got: {test_cases['source_doc_rtrv'][idx]}")

#%%
if __name__ == '__main__':
    comparison_fp = '../../data/retrieval_results/accuracy_retrieval_bls.xlsx'
    # retrieval_accuracy(validation_questions_fp, retrieval_fp, comparison_fp)

    results = pd.read_excel(comparison_fp)
    print(results.columns)
    test_cases = results.iloc[:3, :]
    test_langchain_rag(test_cases=test_cases)