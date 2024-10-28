#%%
from langchain_chroma import Chroma
import pandas as pd

#%%
def retrieve_documents(questions, embeddings, vector_path, collection_name, results_path, k):
    """
        The following function retrieves documents from a given vector store.

        Args:
            questions (List[str]): A list of questions to query the vector store.
            embeddings (HuggingFaceEmbeddings): An instance of HuggingFaceEmbeddings used for generating embeddings for the documents.
            vector_path (str): The filepath to the Chroma vector store.
            collection_name (str): The name of the collection in the Chroma vector store to query.
            results_path (str): The filepath where the retrieved documents will be saved.
            k (int): The number of documents to retrieve for each question.

        Returns:
            pd.DataFrame: A dataframe that contains the documents retrieved for each question.

        """

    vector_store = Chroma(embedding_function=embeddings,
                          persist_directory=vector_path,
                          collection_name=collection_name)

    data = {
        'question': [],
        'source_doc': [],
        'context': [] ,
        'kth_document': []# Replace with your actual content field
    }

    for query in questions:
        # Retrieves k number of documents for specified query
        answer = vector_store.similarity_search(query, k=k)
        # Prepare data for DataFrame
        i = 1
        for result in answer:
            data['source_doc'].append(result.metadata["row"])  # Assuming the result has an 'id' attribute
            data['question'].append(query)  # Assuming the result has a 'score' attribute
            data['context'].append(result.page_content)
            data['kth_document'].append(i)
            i += 1

    df = pd.DataFrame.from_dict(data)

    print(f'The dataframe shape for the retrieved documents is {df.shape}.')
    df.to_excel(results_path, index=False)

    print(f'Retrieved documents saved to {results_path}.')


#%%
# if __name__ == '__main__':
#     #Retrieve Documents for BLS-Related Questions
#     from langchain_huggingface import HuggingFaceEmbeddings
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
#
#     df = pd.read_excel('../../data/retrieval_results/bls_qa_questions.xlsx')
#     bls_questions = list(df['question'])
#
#     results_fp = '../../data/retrieval_results/test_vstore_retrieval_bls.xlsx'
#     retrieve_documents(questions=bls_questions,
#                        embeddings=embeddings,
#                        vector_path='../../data/vector_store/wizard_store',
#                        collection_name='BLS_Occupational_Information',
#                        results_path=results_fp,
#                        k=30)
#
#     print(f'Sample output saved at: {results_fp}')
