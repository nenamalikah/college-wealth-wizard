import pickle
from langchain_community.document_loaders import CSVLoader
from tqdm import tqdm
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

#%%
def save_document(csv_fp,document_obj_fp,chunk_size,chunk_overlap,splitter):
    loader = CSVLoader(file_path=csv_fp,
                       csv_args={
                           'delimiter': ',',
                       })

    docs = []
    docs_lazy = loader.lazy_load()

    for doc in tqdm(docs_lazy):
        docs.append(doc)

    if splitter == 'CharacterTextSplitter':
        docs = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap).split_documents(docs)
    elif splitter == 'RecursiveCharacterTextSplitter':
        docs = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap).split_documents(docs)

    with open(document_obj_fp, "wb") as file:
        pickle.dump(docs, file)

    print(f'Documents have been uploaded to filepath: {document_obj_fp} \n')

def load_documents(document_obj_fp):
    """Load Document objects from a file."""
    with open(document_obj_fp, 'rb') as file:
        documents = pickle.load(file)
    print(f"Documents loaded from {document_obj_fp}.")
    return documents

