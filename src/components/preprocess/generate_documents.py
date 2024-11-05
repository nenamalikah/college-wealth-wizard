import pickle
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from tqdm import tqdm
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document

#%%
def doc_gen_list(text_list,doc_source,sep='.',splitter='CharacterTextSplitter',chunk_size=150,chunk_overlap=50,output='Generate',document_obj_fp=None):
    """
        The following function generates langchain document objects from a list of python strings.

        Args:
            text_list (List[str]): A list of strings to convert to documents.
            doc_source (str): The name of the source where the text came from.
            splitter (str): The text splitter to use for documents. Options are 'CharacterTextSplitter' and 'RecursiveCharacterTextSplitter'.
            chunk_size (int): The chunk size to split the text into chunks.
            chunk_overlap (int): The chunk overlap size to split the text into chunks.
            output (str): Whether to return a list of document objects or export the documents to a pickle file. Options include 'Generate', which returns the list or 'Save', which exports the documents.
            document_obj_fp (str): The filepath to save the document objects at.

            Returns:
                Union[list, None]:
            If output is 'Generate', returns a list of document objects.
            If output is 'Save', does not return a value. It saves the data to a pickle file.
        """

    docs = []
    i = 0
    for doc in tqdm(text_list):
        docs.append(Document(page_content=doc,metadata={"source": doc_source,
                                                        "row":i}))
        i += 1

    if splitter == 'CharacterTextSplitter':
        chunker = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=sep)
        chunked_docs = chunker.split_documents(docs)

    elif splitter == 'RecursiveCharacterTextSplitter':
        chunker = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunked_docs = chunker.split_documents(docs)

    new_docs = []
    curr_row = chunked_docs[0].metadata.get("row")
    chunk_i = 1
    for doc in chunked_docs:
        row = doc.metadata.get("row")  # get the row of the current chunk
        if row == curr_row:
            new_docs.append(Document(page_content=doc.page_content, metadata={"source": doc_source,
                                                                              "row": row, "chunk": chunk_i}))
            chunk_i += 1
        else:
            curr_row += 1
            chunk_i = 1

    print(f'Len of documents: {len(new_docs)}')

    if output=='Generate':
        return new_docs
    elif output=='Save':
        with open(document_obj_fp, "wb") as file:
            pickle.dump(new_docs, file)
            print(f'Documents have been exported to filepath: {document_obj_fp} \n')


#%%
def doc_gen_csv(csv_fp,sep='.',chunk_size=150,chunk_overlap=50,splitter='CharacterTextSplitter',output='Generate',document_obj_fp=None):
    """
        The following function generates langchain document objects from a csv file.

        Args:
            csv_fp (str): The path of the csv file to generate documents from.
            chunk_size (int): The chunk size to split the text into chunks.
            chunk_overlap (int): The chunk overlap size to split the text into chunks.
            splitter (str): The text splitter to use for documents. Options are 'CharacterTextSplitter' and 'RecursiveCharacterTextSplitter'.
            output (str): Whether to return a list of document objects or export the documents to a pickle file. Options include 'Generate', which returns the list or 'Save', which exports the documents.
            document_obj_fp (str): The filepath to save the document objects at.

            Returns:
                Union[list, None]:
            If output is 'Generate', returns a list of document objects.
            If output is 'Save', does not return a value. It saves the data to a pickle file.
    """
    loader = CSVLoader(file_path=csv_fp,
                       csv_args={
                           'delimiter': ',',
                       })

    docs = []
    docs_lazy = loader.lazy_load()

    for doc in tqdm(docs_lazy):
        docs.append(doc)

    if splitter == 'CharacterTextSplitter':
        chunker = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=sep)
        chunked_docs = chunker.split_documents(docs)

    elif splitter == 'RecursiveCharacterTextSplitter':
        chunker = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap).split_documents(docs)
        chunked_docs = chunker.split_documents(docs)

    new_docs = []
    curr_row = chunked_docs[0].metadata.get("row")
    chunk_i = 1
    for doc in chunked_docs:
        row = doc.metadata.get("row")  # get the row of the current chunk
        if row == curr_row:
            new_docs.append(Document(page_content=doc.page_content, metadata={"source": doc_source,
                                                                              "row": row, "chunk": chunk_i}))
            chunk_i += 1
        else:
            curr_row += 1
            chunk_i = 1

    print(f'\n Here is a sample doc: {new_docs[0]}')
    print(f'\n Here is the next sample doc: {new_docs[1]}')

    print(f'Len of documents: {len(new_docs)}')
    if output=='Generate':
        return new_docs
    elif output=='Save':
        with open(document_obj_fp, "wb") as file:
            pickle.dump(new_docs, file)
            print(f'Documents have been exported to filepath: {document_obj_fp} \n')

#%%
def load_documents(document_obj_fp):
    """
        The following function loads previously generated document objects from a pickle file.

        Args:
            document_obj_fp (str): The filepath to load the documents from.

        Returns:
            documents (List[Document]): A list of document objects.
        """
    with open(document_obj_fp, 'rb') as file:
        documents = pickle.load(file)
    print(f"Documents loaded from {document_obj_fp}.")
    return documents

#%%
if __name__ == '__main__':
    import pandas as pd
    csv_fp = '../../../data/soc_employment_information.csv'
    df = pd.read_csv(csv_fp, usecols=['Occupation_Summary'])
    doc_gen_list(text_list=df['Occupation_Summary'][:5],
                 doc_source='BLS Employment Data',
                 output='Save',
                 document_obj_fp='../../../data/sample_data/bls_sample.csv')
