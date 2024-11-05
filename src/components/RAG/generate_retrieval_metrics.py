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
    # Create 'chunk-source-doc' column by combining 'chunk' and 'source_doc'
    validation_df['chunk_source_doc'] = (
            validation_df['chunk'].astype(str) + '-' + validation_df['source_doc'].astype(str)
    )
    print(f'The columns in the validation dataset are: {validation_df.columns}. \n The shape of the validation dataset is: {validation_df.shape}.')

    retrieval_df = pd.read_excel(retrieval_fp)
    # Create 'chunk-source-doc' column by combining 'chunk' and 'source_doc'
    retrieval_df['chunk_source_doc'] = (
            retrieval_df['chunk'].astype(str) + '-' + retrieval_df['source_doc'].astype(str)
    )
    retrieval_df = retrieval_df.groupby(by='question')[['chunk_source_doc','chunk','source_doc','kth_document', 'context']].agg(list).reset_index()
    print(f'The columns in the retrieval dataset are {retrieval_df.columns}. \n The shape of the retrieval dataset is: {retrieval_df.shape}')

    comparison = retrieval_df.merge(validation_df, how='left', on='question', suffixes=('_rtrv','_vld'))

    print(comparison.head())

    answers = []

    for idx in comparison.index:
        if comparison['source_doc_vld'][idx] in comparison['source_doc_rtrv'][idx]:
            answers.append(1)
        else:
            answers.append(0)

    chunk_source_acc = []
    for idx in comparison.index:
        if comparison['chunk_source_doc_vld'][idx] in comparison['chunk_source_doc_rtrv'][idx]:
            chunk_source_acc.append(1)
        else:
            chunk_source_acc.append(0)

    comparison['Correct_Source_Doc'] = answers
    comparison['Correct_Chunk_Source_Doc'] = chunk_source_acc

    accuracy = (sum(answers)/len(answers))*100
    print(f'The accuracy of the retrieval dataset for documents is {accuracy}')

    chunk_accuracy = (sum(chunk_source_acc) / len(chunk_source_acc)) * 100
    print(f'The accuracy of the retrieval dataset for chunks and documents is {chunk_accuracy}')

    comparison.to_excel(comparison_fp, index=False)
    print(f'The comparison has been saved to {comparison_fp}')

#%%
# if __name__ == '__main__':
#     validation_questions_fp = '../../../data/retrieval_results/bls_qa_questions.xlsx'
#     retrieval_fp = '../../../data/retrieval_results/test_vstore_retrieval_bls.xlsx'
#
#     retrieval_accuracy(validation_questions_fp=validation_questions_fp, retrieval_fp=retrieval_fp,
#                        comparison_fp= '../../../data/retrieval_results/accuracy_retrieval_bls.xlsx')