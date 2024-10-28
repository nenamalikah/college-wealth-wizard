#%%
import pandas as pd

#%%
def retrieval_accuracy(validation_questions_fp, retrieval_fp, comparison_fp):
    """
    The following function generates accuracy metrics for vector store retrieval.

    Args:
        validation_questions_fp (str): The filepath of the RAG validation questions.
        retrieval_fp (str): The filepath of the documents retrieved from the vector store.
        comparison_fp (str): The filepath that the metrics should be saved to.

    Returns:
        pd.DataFrame: A dataframe that compares the correct document for each RAG validation question and the document retrieved from the vector store.

    """
    validation_df = pd.read_excel(validation_questions_fp)
    print(f'The columns in the validation dataset are: {validation_df.columns}. \n The shape of the validation dataset is: {validation_df.shape}.')

    retrieval_df = pd.read_excel(retrieval_fp)
    retrieval_df = retrieval_df.groupby(by='question')[['source_doc','kth_document']].agg(list).reset_index()
    print(f'The columns in the retrieval dataset are {retrieval_df.columns}. \n The shape of the retrieval dataset is: {retrieval_df.shape}')

    comparison = retrieval_df.merge(validation_df, how='left', on='question', suffixes=('_rtrv','_vld'))

    print(comparison.head())

    answers = []

    for idx in comparison.index:
        if comparison['source_doc_vld'][idx] in comparison['source_doc_rtrv'][idx]:
            answers.append(1)
        else:
            answers.append(0)


    comparison['Correct_Doc'] = answers

    accuracy = (sum(answers)/len(answers))*100
    print(f'The accuracy of the retrieval dataset is {accuracy}')

    comparison.to_excel(comparison_fp, index=False)
    print(f'The comparison has been saved to {comparison_fp}')

#%%
# if __name__ == '__main__':
#     validation_questions_fp = '../../data/retrieval_results/bls_qa_questions.xlsx'
#     retrieval_fp = '../../data/retrieval_results/test_vstore_retrieval_bls.xlsx'
#
#     retrieval_accuracy(validation_questions_fp=validation_questions_fp, retrieval_fp=retrieval_fp,
#                        comparison_fp= '../../data/retrieval_results/accuracy_retrieval_bls.xlsx')